from __future__ import annotations

from typing import Protocol, runtime_checkable

from .linear_system import LinearSystem
from .result import LinearSolveResult


@runtime_checkable
class LinearSolverProtocol(Protocol):
    """
    Structural interface for DeviceForge linear solvers.

    A compatible linear solver exposes identifying information and solves a
    validated LinearSystem, returning a LinearSolveResult.

    Concrete implementations do not need to inherit from this protocol.
    """

    @property
    def name(self) -> str:
        """Return the linear-solver name."""

        ...

    @property
    def backend_name(self) -> str:
        """Return the numerical backend name."""

        ...

    def solve(
        self,
        system: LinearSystem,
    ) -> LinearSolveResult:
        """
        Solve a validated linear system.

        Parameters
        ----------
        system:
            Validated DeviceForge linear system.

        Returns
        -------
        LinearSolveResult
            Numerical solution and solver diagnostics.
        """

        ...