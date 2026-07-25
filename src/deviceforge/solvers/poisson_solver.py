from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from ..core.field import Field
from ..core.result import SimulationResult
from ..core.simulation import Simulation


VACUUM_PERMITTIVITY = 8.8541878128e-12


@dataclass(frozen=True, slots=True)
class PoissonSolver:
    """
    One-dimensional electrostatic Poisson solver.

    Solves

        d/dx [epsilon_0 * epsilon_r(x) * dphi/dx] = -rho(x)

    on a uniform one-dimensional Cartesian grid using a conservative
    finite-difference discretisation and a direct dense NumPy solve.

    Relative permittivity is derived from the materials assigned to the
    simulation device. Face-centred permittivity is calculated using the
    harmonic mean so that dielectric flux remains continuous across
    material interfaces.

    A simulation without a prescribed charge-density field is treated as
    having zero charge density and therefore reduces to Laplace's equation.

    Only Dirichlet boundary conditions are currently supported.

    The class satisfies SolverProtocol structurally and does not inherit
    from or import the protocol.
    """

    name: str = "poisson_direct_1d"
    backend_name: str = "numpy"

    def __post_init__(self) -> None:
        """Validate solver configuration."""

        if not isinstance(self.name, str):
            raise TypeError(
                "Solver name must be a string."
            )

        normalised_name = self.name.strip()

        if not normalised_name:
            raise ValueError(
                "Solver name must not be empty."
            )

        if not isinstance(self.backend_name, str):
            raise TypeError(
                "Backend name must be a string."
            )

        normalised_backend_name = self.backend_name.strip()

        if not normalised_backend_name:
            raise ValueError(
                "Backend name must not be empty."
            )

        object.__setattr__(
            self,
            "name",
            normalised_name,
        )

        object.__setattr__(
            self,
            "backend_name",
            normalised_backend_name,
        )

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

        self._validate_simulation(simulation)

        start_time = perf_counter()

        matrix, right_hand_side = self._assemble_system(
            simulation
        )

        try:
            potential_values = np.linalg.solve(
                matrix,
                right_hand_side,
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "PoissonSolver could not solve the assembled "
                "linear system."
            ) from exc

        residual = self._calculate_residual(
            matrix=matrix,
            solution=potential_values,
            right_hand_side=right_hand_side,
        )

        runtime_seconds = perf_counter() - start_time

        charge_density = (
            simulation.create_charge_density_field()
        )

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

        return SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=residual <= simulation.tolerance,
            iterations=1,
            residual_history=np.asarray(
                [residual],
                dtype=np.float64,
            ),
            runtime_seconds=runtime_seconds,
            solver_name=self.name,
            backend_name=self.backend_name,
            metadata={
                "equation": equation_name,
                "spatial_dimension": 1,
                "discretisation": (
                    "conservative_central_finite_difference"
                ),
                "interface_averaging": "harmonic",
                "linear_solver": "numpy.linalg.solve",
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
                    simulation.grid.number_of_points
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
        """Validate that the simulation is supported."""

        if simulation.grid.dimension != 1:
            raise ValueError(
                "PoissonSolver currently supports only "
                "one-dimensional grids."
            )

        if simulation.neumann_boundaries:
            raise ValueError(
                "PoissonSolver currently supports only "
                "Dirichlet boundary conditions."
            )

        if not simulation.dirichlet_boundaries:
            raise ValueError(
                "PoissonSolver requires at least one "
                "Dirichlet boundary condition."
            )

        if not simulation.device.require_full_coverage:
            raise ValueError(
                "PoissonSolver requires complete material coverage "
                "of the device grid."
            )

        fixed_mask = simulation.create_fixed_potential_mask()

        if not fixed_mask[0]:
            raise ValueError(
                "The first grid point must have a Dirichlet "
                "boundary condition."
            )

        if not fixed_mask[-1]:
            raise ValueError(
                "The final grid point must have a Dirichlet "
                "boundary condition."
            )

        relative_permittivity = (
            simulation.device
            .relative_permittivity_field()
            .values
        )

        if np.any(relative_permittivity <= 0.0):
            raise ValueError(
                "Every grid point must have positive relative "
                "permittivity."
            )

    def _assemble_system(
        self,
        simulation: Simulation,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Assemble the conservative finite-difference system.

        At an unconstrained interior point:

            epsilon_(i-1/2) * phi_(i-1)
            - [epsilon_(i-1/2) + epsilon_(i+1/2)] * phi_i
            + epsilon_(i+1/2) * phi_(i+1)

                = -rho_i * dx^2 / epsilon_0

        The face permittivities are harmonic means of neighbouring
        node-centred relative permittivities.
        """

        number_of_points = simulation.grid.shape[0]
        grid_spacing = simulation.grid.spacing[0]

        matrix = np.zeros(
            (number_of_points, number_of_points),
            dtype=np.float64,
        )

        right_hand_side = np.zeros(
            number_of_points,
            dtype=np.float64,
        )

        fixed_mask = simulation.create_fixed_potential_mask()

        boundary_values = np.zeros(
            simulation.grid.shape,
            dtype=np.float64,
        )

        for boundary in simulation.dirichlet_boundaries:
            boundary_values[boundary.mask] = boundary.value

        charge_density = (
            simulation
            .create_charge_density_field()
            .values
        )

        relative_permittivity = (
            simulation.device
            .relative_permittivity_field()
            .values
        )

        face_permittivity = self._harmonic_face_values(
            relative_permittivity
        )

        for index in range(number_of_points):
            if fixed_mask[index]:
                matrix[index, index] = 1.0
                right_hand_side[index] = boundary_values[index]
                continue

            left_permittivity = face_permittivity[index - 1]
            right_permittivity = face_permittivity[index]

            matrix[index, index - 1] = left_permittivity

            matrix[index, index] = -(
                left_permittivity
                + right_permittivity
            )

            matrix[index, index + 1] = right_permittivity

            right_hand_side[index] = (
                -charge_density[index]
                * grid_spacing**2
                / VACUUM_PERMITTIVITY
            )

        return matrix, right_hand_side

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

        if not np.all(np.isfinite(values)):
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

    @staticmethod
    def _calculate_residual(
        *,
        matrix: NDArray[np.float64],
        solution: NDArray[np.float64],
        right_hand_side: NDArray[np.float64],
    ) -> float:
        """Return the infinity norm of the algebraic residual."""

        residual_vector = (
            matrix @ solution
            - right_hand_side
        )

        return float(
            np.linalg.norm(
                residual_vector,
                ord=np.inf,
            )
        )