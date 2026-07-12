from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from .grid import Grid


class BoundaryConditionType(StrEnum):
    """Supported boundary-condition categories."""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


@dataclass(frozen=True, slots=True)
class BoundaryCondition:
    """
    Boundary condition applied to selected outer-grid points.

    Parameters
    ----------
    name:
        Human-readable boundary name, such as ``"left_contact"``.

    grid:
        Computational grid on which the boundary is defined.

    mask:
        Boolean array selecting boundary points. Its shape must match
        ``grid.shape``.

    condition_type:
        Either ``"dirichlet"`` or ``"neumann"``.

        A Dirichlet condition fixes the field value.

        A Neumann condition fixes the normal derivative or flux.

    value:
        Numerical boundary-condition value.

    units:
        Units associated with the boundary value, for example ``"V"``
        for electrostatic potential or ``"V/m"`` for a potential gradient.
    """

    name: str
    grid: Grid
    mask: NDArray[np.bool_]
    condition_type: BoundaryConditionType | str
    value: float
    units: str

    def __post_init__(self) -> None:
        """Validate and normalise the boundary-condition definition."""

        if not self.name.strip():
            raise ValueError("Boundary-condition name must not be empty.")

        if not self.units.strip():
            raise ValueError("Boundary-condition units must not be empty.")

        mask = np.asarray(self.mask, dtype=np.bool_)

        if mask.shape != self.grid.shape:
            raise ValueError(
                "Boundary mask must have the same shape as the associated grid. "
                f"Expected {self.grid.shape}, received {mask.shape}."
            )

        if not np.any(mask):
            raise ValueError(
                "Boundary mask must contain at least one grid point."
            )

        if not np.isfinite(self.value):
            raise ValueError("Boundary-condition value must be finite.")

        try:
            condition_type = BoundaryConditionType(
                str(self.condition_type).strip().lower()
            )
        except ValueError as exc:
            raise ValueError(
                "Boundary-condition type must be either "
                "'dirichlet' or 'neumann'."
            ) from exc

        outer_boundary_mask = self._create_outer_boundary_mask()

        if np.any(mask & ~outer_boundary_mask):
            raise ValueError(
                "Boundary mask contains points that are not on the outer "
                "boundary of the grid."
            )

        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "condition_type", condition_type)
        object.__setattr__(self, "value", float(self.value))

    def _create_outer_boundary_mask(self) -> NDArray[np.bool_]:
        """Return a mask selecting every outer-grid point."""

        outer = np.zeros(self.grid.shape, dtype=np.bool_)

        for axis in range(self.grid.dimension):
            lower_face = [slice(None)] * self.grid.dimension
            upper_face = [slice(None)] * self.grid.dimension

            lower_face[axis] = 0
            upper_face[axis] = -1

            outer[tuple(lower_face)] = True
            outer[tuple(upper_face)] = True

        return outer

    @property
    def number_of_points(self) -> int:
        """Return the number of grid points selected by the boundary."""

        return int(np.count_nonzero(self.mask))

    @property
    def is_dirichlet(self) -> bool:
        """Return whether this is a Dirichlet boundary condition."""

        return self.condition_type is BoundaryConditionType.DIRICHLET

    @property
    def is_neumann(self) -> bool:
        """Return whether this is a Neumann boundary condition."""

        return self.condition_type is BoundaryConditionType.NEUMANN

    def contains_index(self, index: tuple[int, ...]) -> bool:
        """
        Return whether a grid index belongs to the boundary condition.

        Parameters
        ----------
        index:
            Grid index with one entry per spatial dimension.
        """

        if len(index) != self.grid.dimension:
            raise ValueError(
                "Index dimensionality must match the grid dimensionality."
            )

        for axis, coordinate in enumerate(index):
            if coordinate < 0 or coordinate >= self.grid.shape[axis]:
                raise IndexError(
                    f"Index {index} lies outside grid shape {self.grid.shape}."
                )

        return bool(self.mask[index])