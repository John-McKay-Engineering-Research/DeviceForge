"""Data model for a single voltage-sweep operating point.

This module defines :class:`VoltageSweepResult`, which stores the numerical
solution and solver diagnostics associated with one applied voltage during a
DeviceForge voltage sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class VoltageSweepResult:
    """Store the solution obtained at one voltage-sweep operating point.

    Parameters
    ----------
    voltage:
        Applied terminal voltage in volts.

    current:
        Extracted terminal current in amperes.

    potential:
        Electrostatic potential field in volts.

    electron_density:
        Electron concentration field.

    hole_density:
        Hole concentration field.

    electric_field:
        Electric-field values corresponding to the solved operating point.

    electron_current_density:
        Electron current-density field.

    hole_current_density:
        Hole current-density field.

    total_current_density:
        Total current-density field.

    iterations:
        Number of nonlinear solver iterations performed.

    residual:
        Final nonlinear residual or convergence metric.

    solve_time:
        Wall-clock solution time in seconds.

    converged:
        Whether the solver satisfied its convergence criterion.

    metadata:
        Optional additional information associated with the operating point.

    Notes
    -----
    Array inputs are converted to independent ``float64`` NumPy arrays during
    initialisation. This prevents later modification of the original solver
    arrays from silently changing stored sweep results.
    """

    voltage: float
    current: float

    potential: FloatArray
    electron_density: FloatArray
    hole_density: FloatArray
    electric_field: FloatArray

    electron_current_density: FloatArray
    hole_current_density: FloatArray
    total_current_density: FloatArray

    iterations: int
    residual: float
    solve_time: float
    converged: bool

    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate scalar values and normalise stored arrays."""

        self.voltage = self._validate_finite_float(
            self.voltage,
            name="voltage",
        )
        self.current = self._validate_finite_float(
            self.current,
            name="current",
        )
        self.residual = self._validate_non_negative_float(
            self.residual,
            name="residual",
        )
        self.solve_time = self._validate_non_negative_float(
            self.solve_time,
            name="solve_time",
        )

        if isinstance(self.iterations, bool) or not isinstance(
            self.iterations,
            (int, np.integer),
        ):
            raise TypeError("iterations must be an integer.")

        self.iterations = int(self.iterations)

        if self.iterations < 0:
            raise ValueError("iterations must be greater than or equal to zero.")

        if not isinstance(self.converged, (bool, np.bool_)):
            raise TypeError("converged must be a boolean value.")

        self.converged = bool(self.converged)

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        self.metadata = dict(self.metadata)

        self.potential = self._normalise_array(
            self.potential,
            name="potential",
        )
        self.electron_density = self._normalise_array(
            self.electron_density,
            name="electron_density",
        )
        self.hole_density = self._normalise_array(
            self.hole_density,
            name="hole_density",
        )
        self.electric_field = self._normalise_array(
            self.electric_field,
            name="electric_field",
        )
        self.electron_current_density = self._normalise_array(
            self.electron_current_density,
            name="electron_current_density",
        )
        self.hole_current_density = self._normalise_array(
            self.hole_current_density,
            name="hole_current_density",
        )
        self.total_current_density = self._normalise_array(
            self.total_current_density,
            name="total_current_density",
        )

        self._validate_field_shapes()

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the stored scalar solution fields."""

        return self.potential.shape

    @property
    def number_of_points(self) -> int:
        """Return the total number of stored scalar grid points."""

        return int(self.potential.size)

    @property
    def absolute_current(self) -> float:
        """Return the magnitude of the extracted terminal current."""

        return abs(self.current)

    @property
    def current_density_mismatch(self) -> FloatArray:
        """Return the difference between stored total and component currents.

        The returned field is calculated as

        ``J_total - (J_n + J_p)``.

        For a fully consistent result this quantity should be close to zero,
        subject to floating-point and discretisation error.
        """

        return self.total_current_density - (
            self.electron_current_density + self.hole_current_density
        )

    def copy(self) -> VoltageSweepResult:
        """Return a deep copy of the voltage-sweep result."""

        return VoltageSweepResult(
            voltage=self.voltage,
            current=self.current,
            potential=self.potential.copy(),
            electron_density=self.electron_density.copy(),
            hole_density=self.hole_density.copy(),
            electric_field=self.electric_field.copy(),
            electron_current_density=self.electron_current_density.copy(),
            hole_current_density=self.hole_current_density.copy(),
            total_current_density=self.total_current_density.copy(),
            iterations=self.iterations,
            residual=self.residual,
            solve_time=self.solve_time,
            converged=self.converged,
            metadata=self.metadata.copy(),
        )

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary of the operating point."""

        return {
            "voltage": self.voltage,
            "current": self.current,
            "absolute_current": self.absolute_current,
            "iterations": self.iterations,
            "residual": self.residual,
            "solve_time": self.solve_time,
            "converged": self.converged,
            "shape": self.shape,
            "number_of_points": self.number_of_points,
        }

    @staticmethod
    def _normalise_array(
        values: NDArray[np.floating] | list[float] | tuple[float, ...],
        *,
        name: str,
    ) -> FloatArray:
        """Convert an array-like input to an independent finite float array."""

        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name} must be convertible to a NumPy floating-point array."
            ) from exc

        if array.ndim == 0:
            raise ValueError(f"{name} must contain at least one dimension.")

        if array.size == 0:
            raise ValueError(f"{name} must not be empty.")

        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")

        return np.array(array, dtype=np.float64, copy=True)

    @staticmethod
    def _validate_finite_float(
        value: float,
        *,
        name: str,
    ) -> float:
        """Validate and return a finite floating-point value."""

        if isinstance(value, bool):
            raise TypeError(f"{name} must be a real number.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real number.") from exc

        if not np.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite.")

        return numeric_value

    @classmethod
    def _validate_non_negative_float(
        cls,
        value: float,
        *,
        name: str,
    ) -> float:
        """Validate and return a finite non-negative floating-point value."""

        numeric_value = cls._validate_finite_float(value, name=name)

        if numeric_value < 0.0:
            raise ValueError(f"{name} must be greater than or equal to zero.")

        return numeric_value

    def _validate_field_shapes(self) -> None:
        """Ensure compatible fields use compatible array shapes."""

        scalar_shape = self.potential.shape

        scalar_fields = {
            "electron_density": self.electron_density,
            "hole_density": self.hole_density,
            "electron_current_density": self.electron_current_density,
            "hole_current_density": self.hole_current_density,
            "total_current_density": self.total_current_density,
        }

        for field_name, field_values in scalar_fields.items():
            if field_values.shape != scalar_shape:
                raise ValueError(
                    f"{field_name} must have shape {scalar_shape}, "
                    f"but received {field_values.shape}."
                )

        valid_electric_field_shapes = {
            scalar_shape,
            scalar_shape + (len(scalar_shape),),
        }

        if self.electric_field.shape not in valid_electric_field_shapes:
            expected_shapes = ", ".join(
                str(shape) for shape in sorted(
                    valid_electric_field_shapes,
                    key=len,
                )
            )

            raise ValueError(
                "electric_field must either match the scalar field shape or "
                "contain one vector component per spatial dimension. "
                f"Expected one of {expected_shapes}, but received "
                f"{self.electric_field.shape}."
            )