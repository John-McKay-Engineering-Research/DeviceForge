from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from deviceforge.core import Simulation, SimulationResult


@dataclass(frozen=True, slots=True)
class SolverConfiguration:
    """
    Common numerical settings shared by iterative solvers.

    Parameters
    ----------
    tolerance:
        Convergence threshold.

    max_iterations:
        Maximum number of iterations permitted.

    backend_name:
        Name of the compute backend used by the solver.
    """

    tolerance: float = 1.0e-8
    max_iterations: int = 10_000
    backend_name: str = "numpy"

    def __post_init__(self) -> None:
        """Validate the common solver settings."""

        if not np.isfinite(self.tolerance):
            raise ValueError("Solver tolerance must be finite.")

        if self.tolerance <= 0.0:
            raise ValueError("Solver tolerance must be positive.")

        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations,
            int,
        ):
            raise TypeError("Maximum iteration count must be an integer.")

        if self.max_iterations <= 0:
            raise ValueError(
                "Maximum iteration count must be greater than zero."
            )

        if not self.backend_name.strip():
            raise ValueError("Backend name must not be empty.")


class BaseSolver(ABC):
    """
    Abstract interface implemented by all DeviceForge numerical solvers.

    Solvers consume a validated ``Simulation`` and return a
    ``SimulationResult`` containing fields and numerical diagnostics.
    """

    def __init__(
        self,
        configuration: SolverConfiguration | None = None,
    ) -> None:
        self._configuration = (
            SolverConfiguration()
            if configuration is None
            else configuration
        )

    @property
    def configuration(self) -> SolverConfiguration:
        """Return the immutable solver configuration."""

        return self._configuration

    @property
    def tolerance(self) -> float:
        """Return the solver convergence tolerance."""

        return self.configuration.tolerance

    @property
    def max_iterations(self) -> int:
        """Return the maximum permitted iteration count."""

        return self.configuration.max_iterations

    @property
    def backend_name(self) -> str:
        """Return the selected compute backend name."""

        return self.configuration.backend_name

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the public solver name."""

    @abstractmethod
    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        """
        Solve a validated DeviceForge simulation.

        Parameters
        ----------
        simulation:
            Simulation definition containing the device, boundary conditions
            and initial field settings.

        Returns
        -------
        SimulationResult
            Solved fields and numerical diagnostics.
        """

    def validate_simulation(
        self,
        simulation: Simulation,
    ) -> None:
        """
        Perform common validation before a solver begins.

        Subclasses may override this method, but should normally call
        ``super().validate_simulation(simulation)`` first.
        """

        if not isinstance(simulation, Simulation):
            raise TypeError(
                "Solver input must be a Simulation instance."
            )

        if simulation.grid.dimension != 2:
            raise ValueError(
                f"{self.name} currently supports only two-dimensional grids."
            )

        if not simulation.dirichlet_boundaries:
            raise ValueError(
                f"{self.name} requires at least one Dirichlet boundary."
            )

    def effective_tolerance(
        self,
        simulation: Simulation,
    ) -> float:
        """
        Return the stricter of solver and simulation tolerances.

        This prevents a solver configuration from silently weakening the
        accuracy requested by the simulation.
        """

        return min(
            self.configuration.tolerance,
            simulation.tolerance,
        )

    def effective_max_iterations(
        self,
        simulation: Simulation,
    ) -> int:
        """
        Return the lower iteration limit from the solver and simulation.

        This prevents either configuration from silently exceeding the other
        object's requested limit.
        """

        return min(
            self.configuration.max_iterations,
            simulation.max_iterations,
        )