from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grid import Grid


@dataclass(frozen=True, slots=True)
class FaceField:
    """
    Scalar field defined on faces between adjacent grid nodes.

    A FaceField is associated with one grid axis. The field values lie on
    faces between neighbouring nodes along that axis.

    For a grid with shape

        (N0, N1, ..., Nk),

    a FaceField associated with axis ``a`` has shape

        (N0, ..., Na - 1, ..., Nk).

    For backward compatibility with the original one-dimensional
    implementation, ``axis`` defaults to zero.

    Parameters
    ----------
    name:
        Human-readable field name.

    units:
        Physical units associated with the field.

    grid:
        Node-centred grid from which the face locations are derived.

    values:
        Scalar values defined on faces normal to ``axis``.

    axis:
        Grid axis normal to the faces. Defaults to zero.
    """

    name: str
    units: str
    grid: Grid
    values: ArrayLike
    axis: int = 0

    def __post_init__(self) -> None:
        """Validate and normalise the face-centred field."""

        if not isinstance(self.name, str):
            raise TypeError(
                "Face-field name must be a string."
            )

        normalised_name = self.name.strip()

        if not normalised_name:
            raise ValueError(
                "Face-field name must not be empty."
            )

        if not isinstance(self.units, str):
            raise TypeError(
                "Face-field units must be a string."
            )

        normalised_units = self.units.strip()

        if not normalised_units:
            raise ValueError(
                "Face-field units must not be empty."
            )

        if not isinstance(self.grid, Grid):
            raise TypeError(
                "Face-field grid must be a Grid instance."
            )

        if isinstance(self.axis, bool) or not isinstance(
            self.axis,
            int,
        ):
            raise TypeError(
                "Face-field axis must be an integer."
            )

        if self.axis < 0 or self.axis >= self.grid.dimension:
            raise ValueError(
                f"Face-field axis {self.axis} is invalid for "
                f"a {self.grid.dimension}D grid."
            )

        normalised_values = np.asarray(
            self.values,
            dtype=np.float64,
        )

        expected_shape_values = list(
            self.grid.shape
        )

        expected_shape_values[self.axis] -= 1

        expected_shape = tuple(
            expected_shape_values
        )

        if normalised_values.shape != expected_shape:
            raise ValueError(
                "Face-field values must have shape "
                f"{expected_shape} for axis {self.axis}. "
                f"Received {normalised_values.shape}."
            )

        if not np.all(np.isfinite(normalised_values)):
            raise ValueError(
                "Face-field values must not contain "
                "NaN or infinite values."
            )

        immutable_values = normalised_values.copy()
        immutable_values.setflags(write=False)

        object.__setattr__(
            self,
            "name",
            normalised_name,
        )

        object.__setattr__(
            self,
            "units",
            normalised_units,
        )

        object.__setattr__(
            self,
            "values",
            immutable_values,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the face-centred values."""

        return self.values.shape

    @property
    def number_of_faces(self) -> int:
        """Return the number of faces represented by the field."""

        return self.values.size

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
        """Return the mean field value."""

        return float(np.mean(self.values))

    def coordinates(
        self,
    ) -> NDArray[np.float64]:
        """
        Return coordinates of the faces along the associated grid axis.

        The returned one-dimensional array contains the midpoint coordinate
        between each pair of neighbouring nodes along ``axis``.
        """

        node_coordinates = self.grid.coordinates(
            self.axis
        )

        coordinates = 0.5 * (
            node_coordinates[:-1]
            + node_coordinates[1:]
        )

        coordinates.setflags(write=False)

        return coordinates