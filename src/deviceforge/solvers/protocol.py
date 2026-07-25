from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..core.result import SimulationResult
from ..core.simulation import Simulation


@runtime_checkable
class SolverProtocol(Protocol):
    """
    Structural interface implemented by DeviceForge numerical solvers.

    A solver receives an immutable Simulation definition, performs its
    numerical algorithm independently, and returns a SimulationResult.

    Solvers must not depend on runtime orchestration, analyses, voltage
    sweeps, optimisation workflows, visualisation, or user-interface state.
    """

    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        """
        Solve the supplied numerical simulation.

        Parameters
        ----------
        simulation:
            Immutable definition of the numerical problem.

        Returns
        -------
        SimulationResult
            Numerical fields, convergence information, execution statistics,
            and solver-specific metadata.
        """
        ...