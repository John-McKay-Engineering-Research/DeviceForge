from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from ..core.field import Field
from ..core.result import SimulationResult
from ..core.simulation import Simulation
from ..linalg import (
    LinearSolverProtocol,
    LinearSystem,
    SparseDirectSolver,
)


VACUUM_PERMITTIVITY = 8.8541878128e-12


@dataclass(frozen=True, slots=True)
class PoissonSolver2D:
    """
    Two-dimensional electrostatic Poisson solver.

    Solves

        -div(epsilon_0 * epsilon_r(x, y) * grad(phi)) = rho(x, y)

    on a uniform two-dimensional Cartesian grid using a conservative
    five-point finite-volume stencil.

    Relative permittivity is derived from the materials assigned to the
    simulation device. Face-centred permittivity is calculated with the
    harmonic mean so that normal electric displacement remains continuous
    across material interfaces.

    The matrix is assembled in sparse CSR format and delegated to a
    configurable linear solver. SparseDirectSolver is used by default.

    The first implementation supports only Dirichlet conditions and
    requires every node on the outer rectangular boundary to be fixed.

    Array convention
    ----------------
    Field values use ``values[i, j]``. Linear-system numbering uses NumPy
    C-order flattening:

        linear_index = i * number_of_axis_1_points + j
    """

    linear_solver: LinearSolverProtocol = field(
        default_factory=SparseDirectSolver
    )
    name: str = "poisson_sparse_2d"

    def __post_init__(self) -> None:
        """Validate and normalise solver configuration."""

        if not isinstance(
            self.linear_solver,
            LinearSolverProtocol,
        ):
            raise TypeError(
                "PoissonSolver2D linear_solver must satisfy "
                "LinearSolverProtocol."
            )

        if not isinstance(self.name, str):
            raise TypeError("Solver name must be a string.")

        normalised_name = self.name.strip()

        if not normalised_name:
            raise ValueError("Solver name must not be empty.")

        object.__setattr__(self, "name", normalised_name)

    @property
    def backend_name(self) -> str:
        """Return the backend used by the configured linear solver."""

        return self.linear_solver.backend_name

    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        """Solve a two-dimensional electrostatic Poisson problem."""

        if not isinstance(simulation, Simulation):
            raise TypeError(
                "PoissonSolver2D requires a Simulation instance."
            )

        self._validate_simulation(simulation)

        start_time = perf_counter()
        linear_system = self._assemble_system(simulation)

        try:
            linear_result = self.linear_solver.solve(linear_system)
        except Exception as exc:
            raise RuntimeError(
                "PoissonSolver2D could not solve the assembled "
                "linear system."
            ) from exc

        potential_vector = linear_result.solution
        residual = linear_result.final_residual

        if residual is None:
            residual = linear_system.residual_norm(potential_vector)

        potential_values = np.asarray(
            potential_vector,
            dtype=np.float64,
        ).reshape(
            simulation.grid.shape,
            order="C",
        )

        runtime_seconds = perf_counter() - start_time
        charge_density = simulation.create_charge_density_field()
        relative_permittivity = (
            simulation.device.relative_permittivity_field()
        )

        potential_field = Field(
            name="electrostatic_potential",
            units="V",
            grid=simulation.grid,
            values=potential_values,
        )

        equation_name = (
            "poisson"
            if simulation.has_charge_density
            else "laplace"
        )

        poisson_converged = (
            linear_result.converged
            and residual <= simulation.tolerance
        )

        grid_spacing = tuple(
            float(value)
            for value in simulation.grid.spacing
        )

        return SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=poisson_converged,
            iterations=linear_result.iterations,
            residual_history=linear_result.residual_history,
            runtime_seconds=runtime_seconds,
            solver_name=self.name,
            backend_name=linear_result.backend_name,
            metadata={
                "equation": equation_name,
                "spatial_dimension": 2,
                "discretisation": (
                    "conservative_five_point_finite_volume"
                ),
                "interface_averaging": "harmonic",
                "linear_index_order": "C",
                "linear_solver": linear_result.solver_name,
                "linear_solver_backend": (
                    linear_result.backend_name
                ),
                "linear_solver_converged": (
                    linear_result.converged
                ),
                "linear_solver_iterations": (
                    linear_result.iterations
                ),
                "linear_solver_final_residual": (
                    linear_result.final_residual
                ),
                "linear_solver_termination_reason": (
                    linear_result.termination_reason
                ),
                "linear_solver_metadata": dict(
                    linear_result.metadata
                ),
                "matrix_storage": "csr",
                "matrix_shape": linear_system.shape,
                "matrix_nonzero_entries": (
                    linear_system.number_of_nonzero_entries
                ),
                "matrix_density": linear_system.density,
                "vacuum_permittivity_f_per_m": (
                    VACUUM_PERMITTIVITY
                ),
                "relative_permittivity_min": (
                    relative_permittivity.minimum
                ),
                "relative_permittivity_max": (
                    relative_permittivity.maximum
                ),
                "charge_density_present": (
                    simulation.has_charge_density
                ),
                "charge_density_min_c_per_m3": (
                    charge_density.minimum
                ),
                "charge_density_max_c_per_m3": (
                    charge_density.maximum
                ),
                "grid_shape": simulation.grid.shape,
                "number_of_grid_points": (
                    simulation.grid.number_of_points
                ),
                "grid_spacing_metres": grid_spacing,
            },
        )

    @staticmethod
    def _validate_simulation(
        simulation: Simulation,
    ) -> None:
        """Validate that the simulation is supported."""

        if simulation.grid.dimension != 2:
            raise ValueError(
                "PoissonSolver2D supports only two-dimensional grids."
            )

        if any(
            number_of_points < 3
            for number_of_points in simulation.grid.shape
        ):
            raise ValueError(
                "PoissonSolver2D requires at least three grid points "
                "along each axis."
            )

        if simulation.neumann_boundaries:
            raise ValueError(
                "PoissonSolver2D currently supports only Dirichlet "
                "boundary conditions."
            )

        if not simulation.dirichlet_boundaries:
            raise ValueError(
                "PoissonSolver2D requires at least one Dirichlet "
                "boundary condition."
            )

        if not simulation.device.require_full_coverage:
            raise ValueError(
                "PoissonSolver2D requires complete material coverage "
                "of the device grid."
            )

        fixed_mask = simulation.create_fixed_potential_mask()

        boundary_mask = np.zeros(
            simulation.grid.shape,
            dtype=np.bool_,
        )
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True

        if not np.all(fixed_mask[boundary_mask]):
            raise ValueError(
                "PoissonSolver2D requires Dirichlet boundary "
                "conditions on every outer-boundary grid point."
            )

        relative_permittivity = (
            simulation.device.relative_permittivity_field().values
        )

        if relative_permittivity.shape != simulation.grid.shape:
            raise ValueError(
                "Relative-permittivity field shape must match the "
                "two-dimensional grid shape."
            )

        if not np.all(np.isfinite(relative_permittivity)):
            raise ValueError(
                "Relative permittivity must be finite at every grid "
                "point."
            )

        if np.any(relative_permittivity <= 0.0):
            raise ValueError(
                "Every grid point must have positive relative "
                "permittivity."
            )

    def _assemble_system(
        self,
        simulation: Simulation,
    ) -> LinearSystem:
        """
        Assemble the conservative sparse two-dimensional system.

        For an unconstrained interior node ``(i, j)``:

            -epsilon_w * (dy / dx) * phi_(i-1,j)
            -epsilon_e * (dy / dx) * phi_(i+1,j)
            -epsilon_s * (dx / dy) * phi_(i,j-1)
            -epsilon_n * (dx / dy) * phi_(i,j+1)

            + [(epsilon_w + epsilon_e) * (dy / dx)
               + (epsilon_s + epsilon_n) * (dx / dy)] * phi_(i,j)

                = rho_(i,j) * dx * dy / epsilon_0

        Dirichlet conditions are imposed using symmetric elimination so
        that the final matrix remains symmetric positive definite.
        """

        number_axis_0, number_axis_1 = simulation.grid.shape
        spacing_axis_0, spacing_axis_1 = simulation.grid.spacing

        number_of_unknowns = number_axis_0 * number_axis_1

        matrix = lil_matrix(
            (number_of_unknowns, number_of_unknowns),
            dtype=np.float64,
        )
        right_hand_side = np.zeros(
            number_of_unknowns,
            dtype=np.float64,
        )

        fixed_mask = simulation.create_fixed_potential_mask()

        boundary_values = np.zeros(
            simulation.grid.shape,
            dtype=np.float64,
        )
        # updated
        # for boundary in simulation.dirichlet_boundaries:
            # boundary_values[boundary.mask] = boundary.value

        for boundary in simulation.dirichlet_boundaries:
            boundary_values[boundary.mask] = (
                boundary.values_on_mask()
            )

        charge_density = (
            simulation.create_charge_density_field().values
        )
        relative_permittivity = (
            simulation.device.relative_permittivity_field().values
        )

        face_axis_0 = self._harmonic_face_values(
            relative_permittivity,
            axis=0,
        )
        face_axis_1 = self._harmonic_face_values(
            relative_permittivity,
            axis=1,
        )

        axis_0_weight = spacing_axis_1 / spacing_axis_0
        axis_1_weight = spacing_axis_0 / spacing_axis_1
        cell_area = spacing_axis_0 * spacing_axis_1

        for index_axis_0 in range(1, number_axis_0 - 1):
            for index_axis_1 in range(1, number_axis_1 - 1):
                centre = self._linear_index(
                    index_axis_0,
                    index_axis_1,
                    number_axis_1,
                )
                west = self._linear_index(
                    index_axis_0 - 1,
                    index_axis_1,
                    number_axis_1,
                )
                east = self._linear_index(
                    index_axis_0 + 1,
                    index_axis_1,
                    number_axis_1,
                )
                south = self._linear_index(
                    index_axis_0,
                    index_axis_1 - 1,
                    number_axis_1,
                )
                north = self._linear_index(
                    index_axis_0,
                    index_axis_1 + 1,
                    number_axis_1,
                )

                west_permittivity = face_axis_0[
                    index_axis_0 - 1,
                    index_axis_1,
                ]
                east_permittivity = face_axis_0[
                    index_axis_0,
                    index_axis_1,
                ]
                south_permittivity = face_axis_1[
                    index_axis_0,
                    index_axis_1 - 1,
                ]
                north_permittivity = face_axis_1[
                    index_axis_0,
                    index_axis_1,
                ]

                west_coefficient = (
                    west_permittivity * axis_0_weight
                )
                east_coefficient = (
                    east_permittivity * axis_0_weight
                )
                south_coefficient = (
                    south_permittivity * axis_1_weight
                )
                north_coefficient = (
                    north_permittivity * axis_1_weight
                )

                matrix[centre, west] = -west_coefficient
                matrix[centre, east] = -east_coefficient
                matrix[centre, south] = -south_coefficient
                matrix[centre, north] = -north_coefficient
                matrix[centre, centre] = (
                    west_coefficient
                    + east_coefficient
                    + south_coefficient
                    + north_coefficient
                )

                right_hand_side[centre] = (
                    charge_density[
                        index_axis_0,
                        index_axis_1,
                    ]
                    * cell_area
                    / VACUUM_PERMITTIVITY
                )

        fixed_indices = np.flatnonzero(
            fixed_mask.ravel(order="C")
        )
        boundary_vector = boundary_values.ravel(order="C")

        for fixed_index in fixed_indices:
            fixed_value = boundary_vector[fixed_index]

            column = (
                matrix[:, fixed_index]
                .toarray()
                .ravel()
            )
            right_hand_side -= column * fixed_value

            matrix[:, fixed_index] = 0.0
            matrix[fixed_index, :] = 0.0
            matrix[fixed_index, fixed_index] = 1.0
            right_hand_side[fixed_index] = fixed_value

        return LinearSystem(
            matrix=matrix.tocsr(),
            right_hand_side=right_hand_side,
            name=(
                f"{simulation.name}_poisson_2d_system"
            ),
        )

    @staticmethod
    def _linear_index(
        index_axis_0: int,
        index_axis_1: int,
        number_axis_1: int,
    ) -> int:
        """Map a 2D grid index to a C-order linear index."""

        return (
            index_axis_0
            * number_axis_1
            + index_axis_1
        )

    @staticmethod
    def _harmonic_face_values(
        node_values: NDArray[np.float64],
        *,
        axis: int,
    ) -> NDArray[np.float64]:
        """Calculate harmonic means between adjacent 2D nodes."""

        values = np.asarray(
            node_values,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "Two-dimensional face-value calculation requires "
                "a two-dimensional array."
            )

        if axis not in (0, 1):
            raise ValueError(
                "Face-value axis must be either 0 or 1."
            )

        if values.shape[axis] < 2:
            raise ValueError(
                "At least two node values are required along the "
                "selected axis."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Permittivity values must be finite."
            )

        if np.any(values <= 0.0):
            raise ValueError(
                "Permittivity values must be positive."
            )

        lower_slices = [slice(None), slice(None)]
        upper_slices = [slice(None), slice(None)]
        lower_slices[axis] = slice(None, -1)
        upper_slices[axis] = slice(1, None)

        lower_values = values[tuple(lower_slices)]
        upper_values = values[tuple(upper_slices)]

        return (
            2.0
            * lower_values
            * upper_values
            / (lower_values + upper_values)
        )