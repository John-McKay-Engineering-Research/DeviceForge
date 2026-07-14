from __future__ import annotations

from time import perf_counter

import numpy as np

from deviceforge.core import (
    BoundaryCondition,
    Simulation,
    SimulationResult,
)

from .base import BaseSolver, SolverConfiguration


class GaussSeidelSolver(BaseSolver):
    """
    Two-dimensional Gauss-Seidel solver for the Laplace equation.

    The solver currently supports:

    - Uniform structured two-dimensional grids
    - Dirichlet boundary conditions
    - Homogeneous Neumann boundary conditions
    - NumPy/Python execution

    Unlike Jacobi iteration, Gauss-Seidel updates the potential field in
    place and immediately reuses newly calculated values.
    """

    def __init__(
        self,
        configuration: SolverConfiguration | None = None,
    ) -> None:
        super().__init__(configuration)

    @property
    def name(self) -> str:
        """Return the public solver name."""

        return "gauss_seidel"

    def validate_simulation(
        self,
        simulation: Simulation,
    ) -> None:
        """Validate that the simulation is supported."""

        super().validate_simulation(simulation)

        spacing_x, spacing_y = simulation.grid.spacing

        if not np.isclose(
            spacing_x,
            spacing_y,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                "GaussSeidelSolver currently requires equal grid spacing "
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
                    "GaussSeidelSolver currently supports only homogeneous "
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
            Validated DeviceForge simulation.

        Returns
        -------
        SimulationResult
            Solved potential and numerical diagnostics.
        """

        self.validate_simulation(simulation)

        tolerance = self.effective_tolerance(simulation)
        max_iterations = self.effective_max_iterations(simulation)

        initial_field = simulation.create_initial_potential_field()
        potential = initial_field.values.copy()

        residual_history: list[float] = []
        converged = False

        rows, columns = potential.shape

        start_time = perf_counter()

        for _ in range(max_iterations):
            maximum_change = 0.0

            for row in range(1, rows - 1):
                for column in range(1, columns - 1):
                    previous_value = potential[row, column]

                    updated_value = 0.25 * (
                        potential[row + 1, column]
                        + potential[row - 1, column]
                        + potential[row, column + 1]
                        + potential[row, column - 1]
                    )

                    potential[row, column] = updated_value

                    change = abs(updated_value - previous_value)

                    if change > maximum_change:
                        maximum_change = change

            self._apply_homogeneous_neumann_boundaries(
                potential=potential,
                boundaries=simulation.neumann_boundaries,
            )

            self._apply_dirichlet_boundaries(
                potential=potential,
                boundaries=simulation.dirichlet_boundaries,
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