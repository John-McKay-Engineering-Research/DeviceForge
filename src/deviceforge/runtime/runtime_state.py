from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from deviceforge.core.result import SimulationResult


@dataclass(slots=True)
class RuntimeState:
    """
    Mutable execution state for a DeviceForge simulation runtime.

    RuntimeState stores quantities that change while a simulation is being
    prepared or executed. It deliberately contains no numerical solution
    logic; solvers and SimulationRuntime are responsible for updating it.

    Parameters
    ----------
    current_solution:
        Most recently generated solver solution or field collection.

    previous_solution:
        Solution retained from the previous solve, typically for warm starts.

    last_result:
        Complete SimulationResult returned by the most recent successful
        solver execution.

    iteration_count:
        Number of iterations completed during the most recent solve.

    converged:
        Whether the most recent solve converged. ``None`` indicates that no
        completed solver result is currently stored.

    residual_history:
        Residual values recorded during the most recent solve.

    elapsed_time:
        Execution time of the most recent solve in seconds. ``None`` indicates
        that no completed solver result is currently stored.

    metadata:
        Additional runtime-specific information.
    """

    current_solution: Any | None = None
    previous_solution: Any | None = None
    last_result: SimulationResult | None = None

    iteration_count: int = 0
    converged: bool | None = None
    residual_history: list[float] = field(default_factory=list)
    elapsed_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalise the runtime state."""

        self._validate_result(self.last_result)
        self._validate_iteration_count(self.iteration_count)
        self._validate_converged(self.converged)
        self._validate_elapsed_time(self.elapsed_time)

        self.iteration_count = int(self.iteration_count)

        self.residual_history = self._normalise_residual_history(
            self.residual_history
        )

        self.metadata = dict(self.metadata)

        if self.elapsed_time is not None:
            self.elapsed_time = float(self.elapsed_time)

    @staticmethod
    def _validate_result(
        value: SimulationResult | None,
    ) -> None:
        """Validate a stored simulation result."""

        if value is not None and not isinstance(
            value,
            SimulationResult,
        ):
            raise TypeError(
                "last_result must be a SimulationResult instance or None."
            )

    @staticmethod
    def _validate_iteration_count(value: int) -> None:
        """Validate an iteration count."""

        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "Iteration count must be an integer."
            )

        if value < 0:
            raise ValueError(
                "Iteration count must not be negative."
            )

    @staticmethod
    def _validate_converged(value: bool | None) -> None:
        """Validate the convergence state."""

        if value is not None and not isinstance(value, bool):
            raise TypeError(
                "Converged must be a Boolean value or None."
            )

    @staticmethod
    def _validate_elapsed_time(
        value: float | None,
    ) -> None:
        """Validate elapsed execution time."""

        if value is None:
            return

        if isinstance(value, bool) or not np.isscalar(value):
            raise TypeError(
                "Elapsed time must be a scalar value or None."
            )

        if not np.isfinite(value):
            raise ValueError(
                "Elapsed time must be finite."
            )

        if value < 0.0:
            raise ValueError(
                "Elapsed time must not be negative."
            )

    @staticmethod
    def _normalise_residual_history(
        values: list[float] | tuple[float, ...] | np.ndarray,
    ) -> list[float]:
        """Validate and copy a sequence of residual values."""

        if isinstance(values, np.ndarray):
            values = values.tolist()

        if not isinstance(values, (list, tuple)):
            raise TypeError(
                "Residual history must be a list, tuple, or NumPy array "
                "of scalar values."
            )

        residuals: list[float] = []

        for residual in values:
            if isinstance(residual, bool) or not np.isscalar(
                residual
            ):
                raise TypeError(
                    "Residual history must contain scalar values."
                )

            if not np.isfinite(residual):
                raise ValueError(
                    "Residual history must contain only finite values."
                )

            if residual < 0.0:
                raise ValueError(
                    "Residual values must not be negative."
                )

            residuals.append(float(residual))

        return residuals

    @property
    def has_solution(self) -> bool:
        """Return whether a current solution is available."""

        return self.current_solution is not None

    @property
    def has_previous_solution(self) -> bool:
        """Return whether a previous solution is available."""

        return self.previous_solution is not None

    @property
    def has_result(self) -> bool:
        """Return whether a completed SimulationResult is available."""

        return self.last_result is not None

    @property
    def has_run(self) -> bool:
        """Return whether a completed solve has been recorded."""

        return self.has_result

    @property
    def final_residual(self) -> float | None:
        """Return the final recorded residual."""

        if not self.residual_history:
            return None

        return self.residual_history[-1]

    def record_result(
        self,
        result: SimulationResult,
    ) -> None:
        """
        Record a completed solver result.

        Existing solution data is retained as the previous solution before
        the new result is installed. Runtime summary values are synchronised
        from the authoritative SimulationResult.
        """

        if not isinstance(result, SimulationResult):
            raise TypeError(
                "result must be a SimulationResult instance."
            )

        self.previous_solution = self.current_solution
        self.current_solution = result.fields
        self.last_result = result

        self.iteration_count = int(result.iterations)
        self.converged = bool(result.converged)

        self.residual_history = self._normalise_residual_history(
            result.residual_history
        )

        self.elapsed_time = float(result.runtime_seconds)
        self.metadata = dict(result.metadata)

    def reset(self) -> None:
        """
        Reset all mutable execution state.

        Metadata is cleared because it may describe the previous execution.
        """

        self.current_solution = None
        self.previous_solution = None
        self.last_result = None

        self.iteration_count = 0
        self.converged = None
        self.residual_history.clear()
        self.elapsed_time = None
        self.metadata.clear()

    def copy(self) -> RuntimeState:
        """
        Return an independent shallow copy of the runtime state.

        State containers are copied, but solution and result objects are
        retained by reference. Solver-specific deep copying can be introduced
        later when concrete mutable solution types exist.
        """

        return RuntimeState(
            current_solution=self.current_solution,
            previous_solution=self.previous_solution,
            last_result=self.last_result,
            iteration_count=self.iteration_count,
            converged=self.converged,
            residual_history=self.residual_history.copy(),
            elapsed_time=self.elapsed_time,
            metadata=self.metadata.copy(),
        )