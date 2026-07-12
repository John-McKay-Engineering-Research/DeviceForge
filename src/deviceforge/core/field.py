from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .grid import Grid


@dataclass(frozen=True, slots=True)
class Field:
    """
    Scalar physical field defined on a structured computational grid.

    Parameters
    ----------
    name:
        Human-readable name of the physical quantity.

    units:
        Physical units of the field values, for example ``"V"``,
        ``"C/m^3"``, or ``"1/m^3"``.

    grid:
        Computational grid on which the field is defined.

    values:
        Numerical values at every grid point. The array shape must match
        ``grid.shape``.
    """

    name: str
    units: str
    grid: Grid
    values: NDArray[np.floating]

    def __post_init__(self) -> None:
        """Validate and normalise the field definition."""

        if not self.name.strip():
            raise ValueError("Field name must not be empty.")

        if not self.units.strip():
            raise ValueError("Field units must not be empty.")

        values = np.asarray(self.values, dtype=np.float64)

        if values.shape != self.grid.shape:
            raise ValueError(
                "Field values must have the same shape as the associated grid. "
                f"Expected {self.grid.shape}, received {values.shape}."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Field values must not contain NaN or infinite values."
            )

        object.__setattr__(self, "values", values)

    @classmethod
    def zeros(
        cls,
        *,
        name: str,
        units: str,
        grid: Grid,
    ) -> Field:
        """
        Create a field containing zeros at every grid point.

        Parameters
        ----------
        name:
            Human-readable field name.

        units:
            Physical units.

        grid:
            Grid on which the field is defined.
        """

        return cls(
            name=name,
            units=units,
            grid=grid,
            values=np.zeros(grid.shape, dtype=np.float64),
        )

    @classmethod
    def full(
        cls,
        *,
        name: str,
        units: str,
        grid: Grid,
        fill_value: float,
    ) -> Field:
        """
        Create a field containing one constant value throughout the grid.
        """

        if not np.isfinite(fill_value):
            raise ValueError("Field fill value must be finite.")

        return cls(
            name=name,
            units=units,
            grid=grid,
            values=np.full(
                grid.shape,
                fill_value,
                dtype=np.float64,
            ),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the field."""

        return self.values.shape

    @property
    def minimum(self) -> float:
        """Return the minimum field value."""

        return float(np.min(self.values))

    @property
    def maximum(self) -> float:
        """Return the maximum field value."""

        return float(np.max(self.values))

    @property
    def mean(self) -> float:
        """Return the arithmetic mean of the field values."""

        return float(np.mean(self.values))

    def copy_with_values(
        self,
        values: NDArray[np.floating],
        *,
        name: str | None = None,
        units: str | None = None,
    ) -> Field:
        """
        Create a new field using the same grid and new numerical values.

        This method preserves the immutability of the original field.
        """

        return Field(
            name=self.name if name is None else name,
            units=self.units if units is None else units,
            grid=self.grid,
            values=values,
        )