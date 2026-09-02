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
class PoissonSolver:
    """
    One-dimensional electrostatic Poisson solver.

    Solves

        d/dx [
            epsilon_0
            * epsilon_r(x)
            * dphi/dx
        ] = -rho(x)

    on a uniform one-dimensional Cartesian grid using a conservative
    finite-difference discretisation.

    Relative permittivity is derived from the materials assigned to the
    simulation device. Face-centred permittivity is calculated using the
    harmonic mean so that dielectric flux remains continuous across
    material interfaces.

    The assembled coefficient matrix is stored in sparse CSR format and
    delegated to a configurable linear solver. SparseDirectSolver is used
    by default.

    A simulation without a prescribed charge-density field is treated as
    having zero charge density and therefore reduces to Laplace's equation.

    Only Dirichlet boundary conditions are currently supported.

    The class satisfies SolverProtocol structurally and does not inherit
    from or import that protocol.

    Parameters
    ----------
    linear_solver:
        Linear solver satisfying LinearSolverProtocol. The default is
        SparseDirectSolver.

    name:
        Human-readable physics-solver name.
    """

    linear_solver: LinearSolverProtocol = field(
        default_factory=SparseDirectSolver
    )

    name: str = "poisson_sparse_1d"

    def __post_init__(self) -> None:
        """Validate and normalise solver configuration."""

        if not isinstance(
            self.linear_solver,
            LinearSolverProtocol,
        ):
            raise TypeError(
                "PoissonSolver linear_solver must satisfy "
                "LinearSolverProtocol."
            )

        if not isinstance(self.name, str):
            raise TypeError(
                "Solver name must be a string."
            )

        normalised_name = self.name.strip()

        if not normalised_name:
            raise ValueError(
                "Solver name must not be empty."
            )

        object.__setattr__(
            self,
            "name",
            normalised_name,
        )

    @property
    def backend_name(self) -> str:
        """Return the backend used by the configured linear solver."""

        return self.linear_solver.backend_name

    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        """
        Solve a one-dimensional electrostatic Poisson problem.

        Parameters
        ----------
        simulation:
            Immutable DeviceForge simulation definition.

        Returns
        -------
        SimulationResult
            Solved electrostatic-potential field and numerical diagnostics.
        """

        if not isinstance(simulation, Simulation):
            raise TypeError(
                "PoissonSolver requires a Simulation instance."
            )

        self._validate_simulation(
            simulation
        )

        start_time = perf_counter()

        linear_system = self._assemble_system(
            simulation
        )

        try:
            linear_result = self.linear_solver.solve(
                linear_system
            )
        except Exception as exc:
            raise RuntimeError(
                "PoissonSolver could not solve the assembled "
                "linear system."
            ) from exc

        potential_values = linear_result.solution

        residual = linear_result.final_residual

        if residual is None:
            residual = linear_system.residual_norm(
                potential_values
            )

        runtime_seconds = (
            perf_counter()
            - start_time
        )

        charge_density = (
            simulation
            .create_charge_density_field()
        )

        relative_permittivity = (
            simulation
            .device
            .relative_permittivity_field()
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

        return SimulationResult(
            fields={
                "electrostatic_potential": (
                    potential_field
                ),
            },
            converged=poisson_converged,
            iterations=linear_result.iterations,
            residual_history=(
                linear_result.residual_history
            ),
            runtime_seconds=runtime_seconds,
            solver_name=self.name,
            backend_name=(
                linear_result.backend_name
            ),
            metadata={
                "equation": equation_name,
                "spatial_dimension": 1,
                "discretisation": (
                    "conservative_central_"
                    "finite_difference"
                ),
                "interface_averaging": (
                    "harmonic"
                ),
                "linear_solver": (
                    linear_result.solver_name
                ),
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
                "matrix_shape": (
                    linear_system.shape
                ),
                "matrix_nonzero_entries": (
                    linear_system
                    .number_of_nonzero_entries
                ),
                "matrix_density": (
                    linear_system.density
                ),
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
                "number_of_grid_points": (
                    simulation
                    .grid
                    .number_of_points
                ),
                "grid_spacing_metres": (
                    simulation.grid.spacing[0]
                ),
            },
        )

    @staticmethod
    def _validate_simulation(
            simulation: Simulation,
    ) -> None:
        """
        Validate that the simulation is supported.

        Supported one-dimensional endpoint combinations are:

            Dirichlet + Dirichlet
            Dirichlet + Neumann
            Neumann   + Dirichlet

        Pure-Neumann problems are rejected because they do not define a unique
        absolute electrostatic potential without an additional gauge condition.
        """

        if simulation.grid.dimension != 1:
            raise ValueError(
                "PoissonSolver currently supports only "
                "one-dimensional grids."
            )

        if (
                not simulation
                        .device
                        .require_full_coverage
        ):
            raise ValueError(
                "PoissonSolver requires complete material "
                "coverage of the device grid."
            )

        relative_permittivity = (
            simulation
            .device
            .relative_permittivity_field()
            .values
        )

        if np.any(
                relative_permittivity <= 0.0
        ):
            raise ValueError(
                "Every grid point must have positive "
                "relative permittivity."
            )

        dirichlet_boundaries = (
            simulation.dirichlet_boundaries
        )

        neumann_boundaries = (
            simulation.neumann_boundaries
        )

        if (
                not dirichlet_boundaries
                and neumann_boundaries
        ):
            raise ValueError(
                "PoissonSolver does not currently support "
                "pure Neumann problems because the absolute "
                "potential is undefined without a gauge condition."
            )

        if not (
                dirichlet_boundaries
                or neumann_boundaries
        ):
            raise ValueError(
                "PoissonSolver requires endpoint boundary "
                "conditions."
            )

        for boundary in dirichlet_boundaries:
            if boundary.units != "V":
                raise ValueError(
                    "Dirichlet boundary conditions used by "
                    "PoissonSolver must use units of 'V'."
                )

        for boundary in neumann_boundaries:
            if boundary.units != "V/m":
                raise ValueError(
                    "Neumann boundary conditions used by "
                    "PoissonSolver must use units of 'V/m'."
                )

        left_is_dirichlet = any(
            boundary.mask[0]
            for boundary in dirichlet_boundaries
        )

        left_is_neumann = any(
            boundary.mask[0]
            for boundary in neumann_boundaries
        )

        right_is_dirichlet = any(
            boundary.mask[-1]
            for boundary in dirichlet_boundaries
        )

        right_is_neumann = any(
            boundary.mask[-1]
            for boundary in neumann_boundaries
        )

        if not (
                left_is_dirichlet
                or left_is_neumann
        ):
            raise ValueError(
                "The first grid point must have a "
                "Dirichlet or Neumann boundary condition."
            )

        if not (
                right_is_dirichlet
                or right_is_neumann
        ):
            raise ValueError(
                "The final grid point must have a "
                "Dirichlet or Neumann boundary condition."
            )

        if (
                left_is_dirichlet
                and left_is_neumann
        ):
            raise ValueError(
                "The first grid point cannot have both "
                "Dirichlet and Neumann boundary conditions."
            )

        if (
                right_is_dirichlet
                and right_is_neumann
        ):
            raise ValueError(
                "The final grid point cannot have both "
                "Dirichlet and Neumann boundary conditions."
            )

    def _assemble_system(
            self,
            simulation: Simulation,
    ) -> LinearSystem:
        """
        Assemble the conservative sparse finite-difference system.

        The positive-definite interior operator is

            -epsilon_(i-1/2) * phi_(i-1)

            + [
                epsilon_(i-1/2)
                + epsilon_(i+1/2)
              ] * phi_i

            - epsilon_(i+1/2) * phi_(i+1)

                = rho_i * dx^2 / epsilon_0

        Face permittivities are harmonic means of neighbouring
        node-centred relative permittivities.

        Neumann boundary conditions are interpreted as outward-normal
        potential derivatives:

            dphi/dn = g

        which gives, in one dimension,

            left boundary:
                phi_0 - phi_1 = g_left * dx

            right boundary:
                phi_(N-1) - phi_(N-2) = g_right * dx

        The Neumann rows are scaled by the adjacent face permittivity so
        that their off-diagonal coefficients match the neighbouring
        interior equations. This preserves symmetry of the mixed-boundary
        system.

        Dirichlet conditions are imposed afterwards using symmetric
        elimination so that the final coefficient matrix remains symmetric
        positive definite for supported mixed-boundary problems.

        The matrix is assembled in LIL format for efficient incremental
        assignment and converted to CSR before constructing LinearSystem.
        """

        number_of_points = simulation.grid.shape[0]
        grid_spacing = simulation.grid.spacing[0]

        matrix = lil_matrix(
            (number_of_points, number_of_points),
            dtype=np.float64,
        )

        right_hand_side = np.zeros(
            number_of_points,
            dtype=np.float64,
        )

        fixed_mask = (
            simulation.create_fixed_potential_mask()
        )

        boundary_values = np.zeros(
            simulation.grid.shape,
            dtype=np.float64,
        )

        for boundary in simulation.dirichlet_boundaries:
            boundary_values[boundary.mask] = (
                boundary.values_on_mask()
            )

        charge_density = (
            simulation
            .create_charge_density_field()
            .values
        )

        relative_permittivity = (
            simulation
            .device
            .relative_permittivity_field()
            .values
        )

        face_permittivity = (
            self._harmonic_face_values(
                relative_permittivity
            )
        )

        # ------------------------------------------------------------
        # Interior Poisson equations
        # ------------------------------------------------------------

        for index in range(
                1,
                number_of_points - 1,
        ):
            left_permittivity = (
                face_permittivity[index - 1]
            )

            right_permittivity = (
                face_permittivity[index]
            )

            matrix[index, index - 1] = (
                -left_permittivity
            )

            matrix[index, index] = (
                    left_permittivity
                    + right_permittivity
            )

            matrix[index, index + 1] = (
                -right_permittivity
            )

            right_hand_side[index] = (
                    charge_density[index]
                    * grid_spacing ** 2
                    / VACUUM_PERMITTIVITY
            )

        # ------------------------------------------------------------
        # Neumann boundary equations
        # ------------------------------------------------------------

        for boundary in simulation.neumann_boundaries:
            boundary_values_on_mask = (
                boundary.values_on_mask()
            )

            boundary_indices = np.flatnonzero(
                boundary.mask
            )

            for local_index, grid_index in enumerate(
                    boundary_indices
            ):
                neumann_value = (
                    boundary_values_on_mask[
                        local_index
                    ]
                )

                if grid_index == 0:
                    left_face_permittivity = (
                        face_permittivity[0]
                    )

                    matrix[0, 0] = (
                        left_face_permittivity
                    )

                    matrix[0, 1] = (
                        -left_face_permittivity
                    )

                    right_hand_side[0] = (
                            left_face_permittivity
                            * neumann_value
                            * grid_spacing
                    )

                elif grid_index == (
                        number_of_points - 1
                ):
                    right_face_permittivity = (
                        face_permittivity[-1]
                    )

                    matrix[-1, -2] = (
                        -right_face_permittivity
                    )

                    matrix[-1, -1] = (
                        right_face_permittivity
                    )

                    right_hand_side[-1] = (
                            right_face_permittivity
                            * neumann_value
                            * grid_spacing
                    )

                else:
                    raise ValueError(
                        "PoissonSolver supports Neumann "
                        "boundary conditions only at the "
                        "first or final grid point."
                    )

        # ------------------------------------------------------------
        # Symmetric Dirichlet elimination
        # ------------------------------------------------------------

        fixed_indices = np.flatnonzero(
            fixed_mask
        )

        for fixed_index in fixed_indices:
            fixed_value = (
                boundary_values[fixed_index]
            )

            column = (
                matrix[:, fixed_index]
                .toarray()
                .ravel()
            )

            right_hand_side -= (
                    column
                    * fixed_value
            )

            matrix[:, fixed_index] = 0.0
            matrix[fixed_index, :] = 0.0
            matrix[
                fixed_index,
                fixed_index,
            ] = 1.0

            right_hand_side[
                fixed_index
            ] = fixed_value

        return LinearSystem(
            matrix=matrix.tocsr(),
            right_hand_side=right_hand_side,
            name=(
                f"{simulation.name}_"
                "poisson_system"
            ),
        )

    @staticmethod
    def _harmonic_face_values(
        node_values: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Calculate harmonic means between adjacent grid nodes.

        For adjacent values a and b:

            harmonic_mean = 2ab / (a + b)
        """

        values = np.asarray(
            node_values,
            dtype=np.float64,
        )

        if values.ndim != 1:
            raise ValueError(
                "Face-value calculation requires a "
                "one-dimensional array."
            )

        if values.size < 2:
            raise ValueError(
                "At least two node values are required."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Permittivity values must be finite."
            )

        if np.any(values <= 0.0):
            raise ValueError(
                "Permittivity values must be positive."
            )

        return (
            2.0
            * values[:-1]
            * values[1:]
            / (
                values[:-1]
                + values[1:]
            )
        )