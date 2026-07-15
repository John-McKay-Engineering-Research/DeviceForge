from __future__ import annotations

from time import perf_counter

import numpy as np

from deviceforge.core import (
    BoundaryCondition,
    Simulation,
    SimulationResult,
)
from deviceforge.physics.electrostatics import (
    compute_absolute_permittivity,
    compute_electrostatic_source_term,
    compute_fixed_charge_density,
)

from .base import BaseSolver, SolverConfiguration


class PoissonJacobiSolver(BaseSolver):
    """
    Two-dimensional Jacobi solver for the electrostatic Poisson equation.

    The solver currently evaluates:

        laplacian(phi) = -rho / epsilon

    It supports:

    - Uniform structured two-dimensional grids
    - Equal spacing in both spatial dimensions
    - Spatially constant permittivity
    - Fixed charge from ionised donor and acceptor concentrations
    - Dirichlet boundary conditions
    - Homogeneous Neumann boundary conditions
    - NumPy execution

    Notes
    -----
    This initial implementation assumes constant permittivity throughout the
    device. A variable-permittivity device requires discretising:

        div(epsilon * grad(phi)) = -rho

    which uses a different interface-aware finite-difference formulation.
    """

    def __init__(
        self,
        configuration: SolverConfiguration | None = None,
    ) -> None:
        super().__init__(configuration)

    @property
    def name(self) -> str:
        """Return the public solver name."""

        return "poisson_jacobi"

    def validate_simulation(
        self,
        simulation: Simulation,
    ) -> None:
        """Validate that the simulation is supported by this solver."""

        super().validate_simulation(simulation)

        spacing_x, spacing_y = simulation.grid.spacing

        if not np.isclose(
            spacing_x,
            spacing_y,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                "PoissonJacobiSolver currently requires equal grid spacing "
                "in both spatial dimensions."
            )

        for boundary in simulation.neumann_boundaries:
            if not np.isclose(
                boundary.value,
                0.0,
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise ValueError(
                    "PoissonJacobiSolver currently supports only homogeneous "
                    "Neumann boundary conditions with value 0.0."
                )

        permittivity = compute_absolute_permittivity(
            simulation.device
        )

        reference_permittivity = permittivity.values.flat[0]

        if not np.allclose(
            permittivity.values,
            reference_permittivity,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                "PoissonJacobiSolver currently requires spatially constant "
                "permittivity."
            )

    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        """
        Solve the two-dimensional electrostatic Poisson equation.

        Parameters
        ----------
        simulation:
            Validated DeviceForge simulation.

        Returns
        -------
        SimulationResult
            Potential, charge density, permittivity, source field and
            convergence diagnostics.
        """

        self.validate_simulation(simulation)

        tolerance = self.effective_tolerance(simulation)
        max_iterations = self.effective_max_iterations(
            simulation
        )

        charge_density = compute_fixed_charge_density(
            simulation.device
        )

        permittivity = compute_absolute_permittivity(
            simulation.device
        )

        source_term = compute_electrostatic_source_term(
            charge_density,
            permittivity,
        )

        initial_field = (
            simulation.create_initial_potential_field()
        )

        potential = initial_field.values.copy()

        spacing = simulation.grid.spacing[0]
        source_contribution = (
            spacing**2 * source_term.values
        )

        residual_history: list[float] = []
        converged = False

        start_time = perf_counter()

        for _ in range(max_iterations):
            previous = potential.copy()

            potential[1:-1, 1:-1] = 0.25 * (
                previous[2:, 1:-1]
                + previous[:-2, 1:-1]
                + previous[1:-1, 2:]
                + previous[1:-1, :-2]
                + source_contribution[1:-1, 1:-1]
            )

            self._apply_homogeneous_neumann_boundaries(
                potential=potential,
                boundaries=simulation.neumann_boundaries,
            )

            self._apply_dirichlet_boundaries(
                potential=potential,
                boundaries=simulation.dirichlet_boundaries,
            )

            maximum_change = float(
                np.max(
                    np.abs(
                        potential - previous
                    )
                )
            )

            residual_history.append(maximum_change)

            if maximum_change <= tolerance:
                converged = True
                break

        runtime_seconds = perf_counter() - start_time

        solved_potential = initial_field.copy_with_values(
            potential
        )

        return SimulationResult(
            fields={
                "electrostatic_potential": solved_potential,
                "fixed_charge_density": charge_density,
                "absolute_permittivity": permittivity,
                "electrostatic_source_term": source_term,
            },
            converged=converged,
            iterations=len(residual_history),
            residual_history=np.asarray(
                residual_history,
                dtype=np.float64,
            ),
            runtime_seconds=runtime_seconds,
            solver_name=self.name,
            backend_name=self.backend_name,
            metadata={
                "equation": "poisson",
                "charge_model": "fully_ionised_fixed_dopants",
                "tolerance": tolerance,
                "maximum_iterations": max_iterations,
                "grid_shape": simulation.grid.shape,
                "grid_spacing": simulation.grid.spacing,
                "permittivity_model": "spatially_constant",
            },
        )

    @staticmethod
    def _apply_dirichlet_boundaries(
        *,
        potential: np.ndarray,
        boundaries: tuple[BoundaryCondition, ...],
    ) -> None:
        """Reapply fixed-potential boundary values."""

        for boundary in boundaries:
            potential[boundary.mask] = boundary.value

    @staticmethod
    def _apply_homogeneous_neumann_boundaries(
        *,
        potential: np.ndarray,
        boundaries: tuple[BoundaryCondition, ...],
    ) -> None:
        """Apply zero-normal-gradient Neumann conditions."""

        rows, columns = potential.shape

        for boundary in boundaries:
            indices = np.argwhere(boundary.mask)

            for row, column in indices:
                if row == 0:
                    potential[row, column] = potential[
                        row + 1,
                        column,
                    ]

                elif row == rows - 1:
                    potential[row, column] = potential[
                        row - 1,
                        column,
                    ]

                elif column == 0:
                    potential[row, column] = potential[
                        row,
                        column + 1,
                    ]

                elif column == columns - 1:
                    potential[row, column] = potential[
                        row,
                        column - 1,
                    ]