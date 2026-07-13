from __future__ import annotations

from time import perf_counter

import numpy as np

from deviceforge.core import (
    BoundaryCondition,
    Simulation,
    SimulationResult,
)

from .base import BaseSolver, SolverConfiguration


class JacobiSolver(BaseSolver):
    """
    Two-dimensional Jacobi iterative solver for the Laplace equation.

    The solver currently supports:

    - Uniform structured two-dimensional grids
    - Dirichlet boundary conditions
    - Homogeneous Neumann boundary conditions
    - NumPy execution

    Convergence is measured using the maximum absolute change between
    consecutive potential fields.
    """

    def __init__(
        self,
        configuration: SolverConfiguration | None = None,
    ) -> None:
        super().__init__(configuration)

    @property
    def name(self) -> str:
        """Return the public solver name."""

        return "jacobi"

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
        ): # made edit here, incorrect numpy spacing
            raise ValueError(
                "JacobiSolver currently requires equal grid spacing "
                "in both spatial dimensions."
            )

        for boundary in simulation.neumann_boundaries:
            if not np.isclose(boundary.value, 0.0):
                raise ValueError(
                    "JacobiSolver currently supports only homogeneous "
                    "Neumann boundary conditions with value 0.0."
                )

    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        """
        Solve the two-dimensional Laplace equation.

        Parameters
        ----------
        simulation:
            Validated DeviceForge simulation definition.

        Returns
        -------
        SimulationResult
            Solved electrostatic potential and convergence diagnostics.
        """

        self.validate_simulation(simulation)

        tolerance = self.effective_tolerance(simulation)
        max_iterations = self.effective_max_iterations(simulation)

        potential_field = simulation.create_initial_potential_field()
        potential = potential_field.values.copy()

        # fixed_mask = simulation.create_fixed_potential_mask()
        # updated, do not need this.
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
            )

            self._apply_homogeneous_neumann_boundaries(
                potential=potential,
                boundaries=simulation.neumann_boundaries,
            )

            self._apply_dirichlet_boundaries(
                potential=potential,
                boundaries=simulation.dirichlet_boundaries,
            )

            # potential[fixed_mask] = potential[fixed_mask]
            # don't need this actually
            residual = float(
                np.max(np.abs(potential - previous))
            )

            residual_history.append(residual)

            if residual <= tolerance:
                converged = True
                break

        runtime_seconds = perf_counter() - start_time

        solved_potential = potential_field.copy_with_values(potential)

        return SimulationResult(
            fields={
                "electrostatic_potential": solved_potential,
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
                "equation": "laplace",
                "tolerance": tolerance,
                "maximum_iterations": max_iterations,
                "grid_shape": simulation.grid.shape,
                "grid_spacing": simulation.grid.spacing,
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
        """
        Apply zero-normal-gradient Neumann boundary conditions.

        Boundary values are copied from their nearest interior neighbours.
        """

        rows, columns = potential.shape

        for boundary in boundaries:
            indices = np.argwhere(boundary.mask)

            for row, column in indices:
                if row == 0:
                    potential[row, column] = potential[row + 1, column]

                elif row == rows - 1:
                    potential[row, column] = potential[row - 1, column]

                elif column == 0:
                    potential[row, column] = potential[row, column + 1]

                elif column == columns - 1:
                    potential[row, column] = potential[row, column - 1]