from __future__ import annotations

from dataclasses import dataclass
from math import prod

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Grid:
    """
    Structured Cartesian computational grid.

    Parameters
    ----------
    shape:
        Number of grid points along each spatial dimension.

        Examples:
        - ``(200,)`` for a one-dimensional grid
        - ``(200, 100)`` for a two-dimensional grid
        - ``(200, 100, 50)`` for a three-dimensional grid

    spacing:
        Distance between neighbouring grid points along each dimension,
        expressed in metres.

    origin:
        Physical coordinate of the first grid point, expressed in metres.
        If omitted, the origin is set to zero in every dimension.
    """

    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate and initialise the grid definition."""

        if len(self.shape) not in (1, 2, 3):
            raise ValueError(
                "Grid dimensionality must be one, two, or three."
            )

        if len(self.shape) != len(self.spacing):
            raise ValueError(
                "Shape and spacing must contain the same number of dimensions."
            )

        if any(
            isinstance(number_of_points, bool)
            or not isinstance(number_of_points, int)
            for number_of_points in self.shape
        ):
            raise TypeError("Grid shape values must be integers.")

        if any(number_of_points < 2 for number_of_points in self.shape):
            raise ValueError(
                "Each grid dimension must contain at least two points."
            )

        if any(step <= 0.0 for step in self.spacing):
            raise ValueError("Grid spacing values must be positive.")

        resolved_origin = self.origin

        if resolved_origin is None:
            resolved_origin = tuple(0.0 for _ in self.shape)

        if len(resolved_origin) != len(self.shape):
            raise ValueError(
                "Origin must contain the same number of dimensions as shape."
            )

        object.__setattr__(self, "origin", resolved_origin)

    @property
    def dimension(self) -> int:
        """Return the number of spatial dimensions."""

        return len(self.shape)

    @property
    def number_of_points(self) -> int:
        """Return the total number of grid points."""

        return prod(self.shape)

    @property
    def physical_size(self) -> tuple[float, ...]:
        """
        Return the physical extent of the grid along each dimension.

        The physical extent is the distance between the first and last
        grid points, not the number of points multiplied by the spacing.
        """

        return tuple(
            (number_of_points - 1) * step
            for number_of_points, step in zip(
                self.shape,
                self.spacing,
                strict=True,
            )
        )

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """
        Return the minimum and maximum coordinate in each dimension.

        Returns
        -------
        tuple of tuple
            One ``(minimum, maximum)`` pair per spatial dimension.
        """

        return tuple(
            (start, start + extent)
            for start, extent in zip(
                self.origin,
                self.physical_size,
                strict=True,
            )
        )

    def coordinates(self, axis: int) -> NDArray[np.float64]:
        """
        Return coordinate values along one grid axis.

        Parameters
        ----------
        axis:
            Zero-based axis index.

        Returns
        -------
        numpy.ndarray
            One-dimensional coordinate array.

        Raises
        ------
        IndexError
            If the requested axis does not exist.
        """

        if axis < 0 or axis >= self.dimension:
            raise IndexError(
                f"Axis {axis} is invalid for a {self.dimension}D grid."
            )

        start = self.origin[axis]
        number_of_points = self.shape[axis]
        step = self.spacing[axis]

        return start + np.arange(number_of_points, dtype=np.float64) * step

    def mesh(self) -> tuple[NDArray[np.float64], ...]:
        """
        Return coordinate arrays spanning the complete computational domain.

        Returns
        -------
        tuple of numpy.ndarray
            Coordinate arrays using matrix indexing.

        Notes
        -----
        For a two-dimensional grid this returns ``(x, y)`` arrays with the
        same shape as the grid. For a three-dimensional grid it returns
        ``(x, y, z)`` arrays.
        """

        coordinate_axes = tuple(
            self.coordinates(axis)
            for axis in range(self.dimension)
        )

        return tuple(
            np.asarray(values, dtype=np.float64)
            for values in np.meshgrid(
                *coordinate_axes,
                indexing="ij",
            )
        )