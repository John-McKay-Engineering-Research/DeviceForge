from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .field import Field
from .grid import Grid


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """
    Results and diagnostics returned by a numerical simulation.

    Parameters
    ----------
    fields:
        Mapping of field names to solved or derived physical fields.

    converged:
        Whether the numerical solver satisfied its convergence criterion.

    iterations:
        Number of solver iterations completed.

    residual_history:
        Residual value recorded after each solver iteration.

    runtime_seconds:
        Total solver runtime in seconds.

    solver_name:
        Name of the numerical solver.

    backend_name:
        Name of the compute backend, such as ``"numpy"`` or ``"cuda"``.

    metadata:
        Optional additional information describing the simulation run.
    """

    fields: Mapping[str, Field]
    converged: bool
    iterations: int
    residual_history: NDArray[np.floating]
    runtime_seconds: float
    solver_name: str
    backend_name: str = "numpy"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalise the simulation result."""

        if not self.fields:
            raise ValueError(
                "Simulation result must contain at least one field."
            )

        normalised_fields = dict(self.fields)

        if any(not name.strip() for name in normalised_fields):
            raise ValueError("Simulation field names must not be empty.")

        if any(
            not isinstance(field_value, Field)
            for field_value in normalised_fields.values()
        ):
            raise TypeError(
                "Every value in fields must be a Field instance."
            )

        grids = {
            field_value.grid
            for field_value in normalised_fields.values()
        }

        if len(grids) != 1:
            raise ValueError(
                "All result fields must use the same computational grid."
            )

        if isinstance(self.iterations, bool) or not isinstance(
            self.iterations,
            int,
        ):
            raise TypeError("Iteration count must be an integer.")

        if self.iterations < 0:
            raise ValueError("Iteration count must not be negative.")

        residual_history = np.asarray(
            self.residual_history,
            dtype=np.float64,
        )

        if residual_history.ndim != 1:
            raise ValueError(
                "Residual history must be a one-dimensional array."
            )

        if not np.all(np.isfinite(residual_history)):
            raise ValueError(
                "Residual history must not contain NaN or infinite values."
            )

        if np.any(residual_history < 0.0):
            raise ValueError(
                "Residual values must not be negative."
            )

        if residual_history.size != self.iterations:
            raise ValueError(
                "Residual-history length must match the iteration count. "
                f"Received {residual_history.size} residuals for "
                f"{self.iterations} iterations."
            )

        if not np.isfinite(self.runtime_seconds):
            raise ValueError("Runtime must be finite.")

        if self.runtime_seconds < 0.0:
            raise ValueError("Runtime must not be negative.")

        if not self.solver_name.strip():
            raise ValueError("Solver name must not be empty.")

        if not self.backend_name.strip():
            raise ValueError("Backend name must not be empty.")

        metadata = dict(self.metadata)

        residual_history.setflags(write=False)

        object.__setattr__(
            self,
            "fields",
            MappingProxyType(normalised_fields),
        )
        object.__setattr__(
            self,
            "residual_history",
            residual_history,
        )
        object.__setattr__(
            self,
            "runtime_seconds",
            float(self.runtime_seconds),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(metadata),
        )

    @property
    def grid(self) -> Grid:
        """Return the computational grid shared by all result fields."""

        return next(iter(self.fields.values())).grid

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return the available result-field names."""

        return tuple(self.fields.keys())

    @property
    def final_residual(self) -> float | None:
        """
        Return the final residual.

        Returns ``None`` if the solver completed zero iterations.
        """

        if self.residual_history.size == 0:
            return None

        return float(self.residual_history[-1])

    @property
    def initial_residual(self) -> float | None:
        """
        Return the first recorded residual.

        Returns ``None`` if the solver completed zero iterations.
        """

        if self.residual_history.size == 0:
            return None

        return float(self.residual_history[0])

    @property
    def residual_reduction(self) -> float | None:
        """
        Return the ratio of initial residual to final residual.

        Larger values indicate greater residual reduction. Returns ``None``
        if no residuals exist or if the final residual is zero.
        """

        initial = self.initial_residual
        final = self.final_residual

        if initial is None or final is None or final == 0.0:
            return None

        return initial / final

    def get_field(self, name: str) -> Field:
        """
        Return a result field by name.

        Raises
        ------
        KeyError
            If the requested field does not exist.
        """

        try:
            return self.fields[name]
        except KeyError as exc:
            raise KeyError(
                f"Simulation result has no field named '{name}'."
            ) from exc

    @property
    def potential(self) -> Field:
        """
        Return the electrostatic-potential field.

        Raises
        ------
        KeyError
            If the result does not contain ``electrostatic_potential``.
        """

        return self.get_field("electrostatic_potential")

    def summary(self) -> dict[str, Any]:
        """Return a serialisable summary of the solver result."""

        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "runtime_seconds": self.runtime_seconds,
            "solver_name": self.solver_name,
            "backend_name": self.backend_name,
            "initial_residual": self.initial_residual,
            "final_residual": self.final_residual,
            "residual_reduction": self.residual_reduction,
            "field_names": self.field_names,
        }