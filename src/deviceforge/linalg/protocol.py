from __future__ import annotations

from typing import (
    Protocol,
    runtime_checkable,
)

import numpy as np
from numpy.typing import NDArray

from .linear_system import LinearSystem
from .result import LinearSolveResult


@runtime_checkable
class LinearSolverProtocol(Protocol):
    """
    Structural interface for DeviceForge linear solvers.

    A compatible linear solver must provide

        solve(system: LinearSystem) -> ndarray

    Concrete implementations do not need to inherit from this protocol.
    """
    # updated ***
    """Structural interface for DeviceForge linear solvers."""

    @property
    def name(self) -> str:
        ...

    @property
    def backend_name(self) -> str:
        ...

    def solve(
        self,
        system: LinearSystem,
    ) -> LinearSolveResult:
        """Solve a validated linear system."""

        ...