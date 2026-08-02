from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .grid import Grid


BoundaryValue: TypeAlias = float | NDArray[np.float64]


class BoundaryConditionType(StrEnum):
    """Supported boundary-condition categories."""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


@dataclass(frozen=True, slots=True)
class BoundaryCondition:
    """
    Boundary condition applied to selected outer-grid points.

    ``value`` may be either a finite scalar or a finite NumPy array with
    the same shape as ``grid.shape``. Array values allow spatially varying
    boundary conditions; only entries selected by ``mask`` are applied.
    """

    name: str
    grid: Grid
    mask: NDArray[np.bool_]
    condition_type: BoundaryConditionType | str
    value: BoundaryValue
    units: str

    def __post_init__(self) -> None:
        """Validate and normalise the boundary-condition definition."""

        if not isinstance(self.name, str):
            raise TypeError(
                "Boundary-condition name must be a string."
            )

        normalised_name = self.name.strip()

        if not normalised_name:
            raise ValueError(
                "Boundary-condition name must not be empty."
            )

        if not isinstance(self.units, str):
            raise TypeError(
                "Boundary-condition units must be a string."
            )

        normalised_units = self.units.strip()

        if not normalised_units:
            raise ValueError(
                "Boundary-condition units must not be empty."
            )

        if not isinstance(self.grid, Grid):
            raise TypeError(
                "Boundary condition requires a Grid instance."
            )

        mask = np.asarray(
            self.mask,
            dtype=np.bool_,
        )

        if mask.shape != self.grid.shape:
            raise ValueError(
                "Boundary mask must have the same shape as the associated "
                f"grid. Expected {self.grid.shape}, received {mask.shape}."
            )

        if not np.any(mask):
            raise ValueError(
                "Boundary mask must contain at least one grid point."
            )

        outer_boundary_mask = (
            self._create_outer_boundary_mask()
        )

        if np.any(mask & ~outer_boundary_mask):
            raise ValueError(
                "Boundary mask contains points that are not on the outer "
                "boundary of the grid."
            )

        try:
            condition_type = BoundaryConditionType(
                str(self.condition_type).strip().lower()
            )
        except ValueError as exc:
            raise ValueError(
                "Boundary-condition type must be either "
                "'dirichlet' or 'neumann'."
            ) from exc

        normalised_value = self._normalise_value(
            self.value
        )

        immutable_mask = mask.copy()
        immutable_mask.setflags(
            write=False
        )

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
            "mask",
            immutable_mask,
        )
        object.__setattr__(
            self,
            "condition_type",
            condition_type,
        )
        object.__setattr__(
            self,
            "value",
            normalised_value,
        )

    def _normalise_value(
        self,
        value: BoundaryValue,
    ) -> BoundaryValue:
        """Validate and normalise a scalar or spatial value."""

        if isinstance(
            value,
            (bool, np.bool_),
        ):
            raise TypeError(
                "Boundary-condition value must be a real scalar "
                "or a NumPy array."
            )

        if np.isscalar(value):
            try:
                scalar_value = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Boundary-condition value must be a real scalar "
                    "or a NumPy array."
                ) from exc

            if not np.isfinite(scalar_value):
                raise ValueError(
                    "Boundary-condition value must be finite."
                )

            return scalar_value

        values = np.asarray(
            value,
            dtype=np.float64,
        )

        if values.shape != self.grid.shape:
            raise ValueError(
                "Spatial boundary values must have the same shape as the "
                f"associated grid. Expected {self.grid.shape}, "
                f"received {values.shape}."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                "Spatial boundary values must not contain NaN or "
                "infinite values."
            )

        immutable_values = values.copy()
        immutable_values.setflags(
            write=False
        )

        return immutable_values

    def _create_outer_boundary_mask(
        self,
    ) -> NDArray[np.bool_]:
        """Return a mask selecting every outer-grid point."""

        outer = np.zeros(
            self.grid.shape,
            dtype=np.bool_,
        )

        for axis in range(
            self.grid.dimension
        ):
            lower_face = (
                [slice(None)]
                * self.grid.dimension
            )
            upper_face = (
                [slice(None)]
                * self.grid.dimension
            )

            lower_face[axis] = 0
            upper_face[axis] = -1

            outer[
                tuple(lower_face)
            ] = True
            outer[
                tuple(upper_face)
            ] = True

        return outer

    @property
    def number_of_points(self) -> int:
        """Return the number of selected boundary points."""

        return int(
            np.count_nonzero(
                self.mask
            )
        )

    @property
    def is_dirichlet(self) -> bool:
        """Return whether this is a Dirichlet condition."""

        return (
            self.condition_type
            is BoundaryConditionType.DIRICHLET
        )

    @property
    def is_neumann(self) -> bool:
        """Return whether this is a Neumann condition."""

        return (
            self.condition_type
            is BoundaryConditionType.NEUMANN
        )

    @property
    def is_spatially_varying(self) -> bool:
        """Return whether the boundary stores a value field."""

        return isinstance(
            self.value,
            np.ndarray,
        )

    def full_value_array(
        self,
    ) -> NDArray[np.float64]:
        """Return values represented over the complete grid."""

        if self.is_spatially_varying:
            return np.array(
                self.value,
                dtype=np.float64,
                copy=True,
            )

        return np.full(
            self.grid.shape,
            float(self.value),
            dtype=np.float64,
        )

    def values_on_mask(
        self,
    ) -> NDArray[np.float64]:
        """Return values for the points selected by ``mask``."""

        if self.is_spatially_varying:
            return np.asarray(
                self.value[
                    self.mask
                ],
                dtype=np.float64,
            ).copy()

        return np.full(
            self.number_of_points,
            float(self.value),
            dtype=np.float64,
        )

    def values_at(
        self,
        selection: NDArray[np.bool_],
    ) -> NDArray[np.float64]:
        """Return values at a Boolean subset of this boundary."""

        selection_mask = np.asarray(
            selection,
            dtype=np.bool_,
        )

        if selection_mask.shape != (
            self.grid.shape
        ):
            raise ValueError(
                "Boundary-value selection must have the same shape as "
                f"the grid. Expected {self.grid.shape}, "
                f"received {selection_mask.shape}."
            )

        if np.any(
            selection_mask
            & ~self.mask
        ):
            raise ValueError(
                "Boundary-value selection contains points outside this "
                "boundary condition."
            )

        if self.is_spatially_varying:
            return np.asarray(
                self.value[
                    selection_mask
                ],
                dtype=np.float64,
            ).copy()

        return np.full(
            int(
                np.count_nonzero(
                    selection_mask
                )
            ),
            float(self.value),
            dtype=np.float64,
        )

    def contains_index(
        self,
        index: tuple[int, ...],
    ) -> bool:
        """Return whether a grid index belongs to the boundary."""

        if len(index) != (
            self.grid.dimension
        ):
            raise ValueError(
                "Index dimensionality must match the grid dimensionality."
            )

        for axis, coordinate in enumerate(
            index
        ):
            if (
                coordinate < 0
                or coordinate
                >= self.grid.shape[axis]
            ):
                raise IndexError(
                    f"Index {index} lies outside grid shape "
                    f"{self.grid.shape}."
                )

        return bool(
            self.mask[index]
        )
