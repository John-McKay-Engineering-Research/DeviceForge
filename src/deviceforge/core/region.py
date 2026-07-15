from __future__ import annotations

from dataclasses import dataclass
# fix for pytest
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

# NOTE***
# These fixes are to stop circular imports

# from deviceforge.physics import Material
# a fix to stop python pytest failure
# from deviceforge.physics.materials import Material

from .grid import Grid

if TYPE_CHECKING:
    from deviceforge.physics.materials import Material

@dataclass(frozen=True, slots=True)
class Region:
    """
    Named material region defined on a computational grid.

    Parameters
    ----------
    name:
        Human-readable region name, such as ``"source"``, ``"channel"``
        or ``"oxide"``.

    grid:
        Computational grid on which the region is defined.

    material:
        Material assigned to the region.

    mask:
        Boolean array identifying the grid points belonging to the region.
        The mask shape must match ``grid.shape``.

    donor_density:
        Donor doping concentration in inverse cubic metres.

    acceptor_density:
        Acceptor doping concentration in inverse cubic metres.

    region_type:
        Optional semantic category such as ``"semiconductor"``,
        ``"dielectric"``, ``"contact"`` or ``"substrate"``.
    """

    name: str
    grid: Grid
    material: Material
    mask: NDArray[np.bool_]
    donor_density: float = 0.0
    acceptor_density: float = 0.0
    region_type: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the region definition."""

        if not self.name.strip():
            raise ValueError("Region name must not be empty.")

        mask = np.asarray(self.mask, dtype=np.bool_)

        if mask.shape != self.grid.shape:
            raise ValueError(
                "Region mask must have the same shape as the associated grid. "
                f"Expected {self.grid.shape}, received {mask.shape}."
            )

        if not np.any(mask):
            raise ValueError("Region mask must contain at least one grid point.")

        donor_density = float(self.donor_density)
        acceptor_density = float(self.acceptor_density)

        if not np.isfinite(donor_density):
            raise ValueError("Donor density must be finite.")

        if not np.isfinite(acceptor_density):
            raise ValueError("Acceptor density must be finite.")

        if donor_density < 0.0:
            raise ValueError("Donor density must not be negative.")

        if acceptor_density < 0.0:
            raise ValueError("Acceptor density must not be negative.")

        normalised_region_type = self.region_type

        if normalised_region_type is not None:
            normalised_region_type = normalised_region_type.strip().lower()

            if not normalised_region_type:
                raise ValueError(
                    "Region type must not be empty when provided."
                )

        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "donor_density", donor_density)
        object.__setattr__(self, "acceptor_density", acceptor_density)
        object.__setattr__(
            self,
            "region_type",
            normalised_region_type,
        )

    @property
    def number_of_points(self) -> int:
        """Return the number of grid points contained in the region."""

        return int(np.count_nonzero(self.mask))

    @property
    def fraction_of_grid(self) -> float:
        """Return the fraction of the grid occupied by the region."""

        return self.number_of_points / self.grid.number_of_points

    @property
    def net_doping_density(self) -> float:
        """
        Return donor density minus acceptor density.

        Positive values represent net n-type doping.
        Negative values represent net p-type doping.
        """

        return self.donor_density - self.acceptor_density

    @property
    def is_n_type(self) -> bool:
        """Return whether the region has net n-type doping."""

        return self.net_doping_density > 0.0

    @property
    def is_p_type(self) -> bool:
        """Return whether the region has net p-type doping."""

        return self.net_doping_density < 0.0

    @property
    def is_intrinsic(self) -> bool:
        """Return whether the donor and acceptor densities are equal."""

        return self.net_doping_density == 0.0

    def contains_index(self, index: tuple[int, ...]) -> bool:
        """
        Return whether a grid index lies inside the region.

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