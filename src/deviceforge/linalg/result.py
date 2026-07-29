from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class LinearSolveResult:
    """
    Immutable result returned by a DeviceForge linear solver.

    Parameters
    ----------
    solution:
        One-dimensional solution vector.

    converged:
        Whether the solver satisfied its convergence criterion.

    iterations:
        Number of solver iterations or direct solution steps.

    residual_history:
        Algebraic residual norm recorded after each iteration or solution
        step.

    solver_name:
        Name of the linear solver.

    backend_name:
        Numerical backend used by the solver.

    termination_reason:
        Human-readable explanation of why the solver stopped.

    metadata:
        Additional solver-specific diagnostics.
    """

    solution: ArrayLike
    converged: bool
    iterations: int
    residual_history: ArrayLike
    solver_name: str
    backend_name: str
    termination_reason: str
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate and normalise the solve result."""

        solution = np.asarray(
            self.solution,
            dtype=np.float64,
        )

        if solution.ndim != 1:
            raise ValueError(
                "Linear-solver solution must be one-dimensional."
            )

        if solution.size == 0:
            raise ValueError(
                "Linear-solver solution must contain at least one value."
            )

        if not np.all(np.isfinite(solution)):
            raise ValueError(
                "Linear-solver solution must not contain "
                "NaN or infinite values."
            )

        if not isinstance(self.converged, bool):
            raise TypeError(
                "Linear-solver converged flag must be Boolean."
            )

        if isinstance(self.iterations, bool) or not isinstance(
            self.iterations,
            int,
        ):
            raise TypeError(
                "Linear-solver iteration count must be an integer."
            )

        if self.iterations < 0:
            raise ValueError(
                "Linear-solver iteration count must not be negative."
            )

        residual_history = np.asarray(
            self.residual_history,
            dtype=np.float64,
        )

        if residual_history.ndim != 1:
            raise ValueError(
                "Linear-solver residual history must be "
                "one-dimensional."
            )

        if residual_history.size != self.iterations:
            raise ValueError(
                "Residual-history length must match the iteration count. "
                f"Received {residual_history.size} residuals for "
                f"{self.iterations} iterations."
            )

        if not np.all(np.isfinite(residual_history)):
            raise ValueError(
                "Linear-solver residual history must not contain "
                "NaN or infinite values."
            )

        if np.any(residual_history < 0.0):
            raise ValueError(
                "Linear-solver residual values must not be negative."
            )

        solver_name = self._normalise_text(
            self.solver_name,
            "Linear-solver name",
        )

        backend_name = self._normalise_text(
            self.backend_name,
            "Linear-solver backend name",
        )

        termination_reason = self._normalise_text(
            self.termination_reason,
            "Linear-solver termination reason",
        )

        immutable_solution = solution.copy()
        immutable_solution.setflags(write=False)

        immutable_residual_history = (
            residual_history.copy()
        )
        immutable_residual_history.setflags(
            write=False
        )

        if not isinstance(self.metadata, Mapping):
            raise TypeError(
                "Linear-solver metadata must be a mapping."
            )

        immutable_metadata = MappingProxyType(
            dict(self.metadata)
        )

        object.__setattr__(
            self,
            "solution",
            immutable_solution,
        )
        object.__setattr__(
            self,
            "residual_history",
            immutable_residual_history,
        )
        object.__setattr__(
            self,
            "solver_name",
            solver_name,
        )
        object.__setattr__(
            self,
            "backend_name",
            backend_name,
        )
        object.__setattr__(
            self,
            "termination_reason",
            termination_reason,
        )
        object.__setattr__(
            self,
            "metadata",
            immutable_metadata,
        )

    @staticmethod
    def _normalise_text(
        value: str,
        label: str,
    ) -> str:
        """Validate and normalise a required text value."""

        if not isinstance(value, str):
            raise TypeError(
                f"{label} must be a string."
            )

        normalised_value = value.strip()

        if not normalised_value:
            raise ValueError(
                f"{label} must not be empty."
            )

        return normalised_value

    @property
    def final_residual(self) -> float | None:
        """Return the final residual, when one was recorded."""

        if self.residual_history.size == 0:
            return None

        return float(
            self.residual_history[-1]
        )

    @property
    def solution_size(self) -> int:
        """Return the number of solution values."""

        return int(self.solution.size)