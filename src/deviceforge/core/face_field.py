from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .grid import Grid


@dataclass(frozen=True, slots=True)
class FaceField:
    """
    Scalar field defined on faces between adjacent grid nodes.

    For a one-dimensional grid containing N nodes, a FaceField contains
    N - 1 values. Face coordinates lie halfway between neighbouring
    node coordinates.

    Parameters
    ----------
    name:
        Human-readable field name.

    units:
        Physical units associated with the field.

    grid:
        Node-centred grid from which the face locations are derived.

    values:
        One-dimensional values defined between neighbouring grid nodes.
    """

    name: str
    units: str
    grid: Grid
    values: ArrayLike

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

        if self.grid.dimension != 1:
            raise ValueError(
                "FaceField currently supports only "
                "one-dimensional grids."
            )

        normalised_values = np.asarray(
            self.values,
            dtype=np.float64,
        )

        expected_shape = (
            self.grid.shape[0] - 1,
        )

        if normalised_values.shape != expected_shape:
            raise ValueError(
                "Face-field values must have shape "
                f"{expected_shape}. Received "
                f"{normalised_values.shape}."
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
        """Return the number of grid faces."""

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
        """Return the physical coordinates of the grid faces."""

        node_coordinates = self.grid.coordinates(0)

        coordinates = 0.5 * (
            node_coordinates[:-1]
            + node_coordinates[1:]
        )

        coordinates.setflags(write=False)

        return coordinates