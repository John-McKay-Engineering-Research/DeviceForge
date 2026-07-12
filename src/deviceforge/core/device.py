from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import Field
from .grid import Grid
from .region import Region


@dataclass(frozen=True, slots=True)
class Device:
    """
    Semiconductor device assembled from grid-aligned regions.

    Parameters
    ----------
    name:
        Human-readable device name.

    grid:
        Computational grid shared by every region.

    regions:
        Regions defining the device geometry, materials and doping.

    require_full_coverage:
        If True, every grid point must belong to exactly one region.
    """

    name: str
    grid: Grid
    regions: tuple[Region, ...]
    require_full_coverage: bool = True

    def __post_init__(self) -> None:
        """Validate the complete device definition."""

        if not self.name.strip():
            raise ValueError("Device name must not be empty.")

        if not self.regions:
            raise ValueError("Device must contain at least one region.")

        region_names = [region.name for region in self.regions]

        if len(region_names) != len(set(region_names)):
            raise ValueError("Device region names must be unique.")

        for region in self.regions:
            if region.grid != self.grid:
                raise ValueError(
                    f"Region '{region.name}' does not use the device grid."
                )

        coverage_count = np.zeros(self.grid.shape, dtype=np.int32)

        for region in self.regions:
            coverage_count += region.mask.astype(np.int32)

        if np.any(coverage_count > 1):
            overlap_count = int(np.count_nonzero(coverage_count > 1))
            raise ValueError(
                f"Device regions overlap at {overlap_count} grid points."
            )

        if self.require_full_coverage and np.any(coverage_count == 0):
            uncovered_count = int(np.count_nonzero(coverage_count == 0))
            raise ValueError(
                f"Device contains {uncovered_count} uncovered grid points."
            )

    @property
    def number_of_regions(self) -> int:
        """Return the number of regions in the device."""

        return len(self.regions)

    @property
    def covered_fraction(self) -> float:
        """Return the fraction of grid points covered by regions."""

        covered = np.zeros(self.grid.shape, dtype=bool)

        for region in self.regions:
            covered |= region.mask

        return float(np.count_nonzero(covered) / self.grid.number_of_points)

    def get_region(self, name: str) -> Region:
        """
        Return a region by name.

        Raises
        ------
        KeyError
            If no region with the requested name exists.
        """

        for region in self.regions:
            if region.name == name:
                return region

        raise KeyError(f"Device has no region named '{name}'.")

    def material_id_field(self) -> Field:
        """
        Return an integer-like field identifying each device region.

        Region IDs follow the order of ``self.regions``.
        """

        values = np.full(self.grid.shape, -1.0, dtype=np.float64)

        for region_id, region in enumerate(self.regions):
            values[region.mask] = float(region_id)

        return Field(
            name="material_region_id",
            units="dimensionless",
            grid=self.grid,
            values=values,
        )

    def relative_permittivity_field(self) -> Field:
        """Return relative permittivity across the complete device."""

        values = np.zeros(self.grid.shape, dtype=np.float64)

        for region in self.regions:
            values[region.mask] = region.material.relative_permittivity

        return Field(
            name="relative_permittivity",
            units="dimensionless",
            grid=self.grid,
            values=values,
        )

    def donor_density_field(self) -> Field:
        """Return donor density across the complete device."""

        values = np.zeros(self.grid.shape, dtype=np.float64)

        for region in self.regions:
            values[region.mask] = region.donor_density

        return Field(
            name="donor_density",
            units="1/m^3",
            grid=self.grid,
            values=values,
        )

    def acceptor_density_field(self) -> Field:
        """Return acceptor density across the complete device."""

        values = np.zeros(self.grid.shape, dtype=np.float64)

        for region in self.regions:
            values[region.mask] = region.acceptor_density

        return Field(
            name="acceptor_density",
            units="1/m^3",
            grid=self.grid,
            values=values,
        )

    def net_doping_field(self) -> Field:
        """Return donor density minus acceptor density."""

        donor = self.donor_density_field()
        acceptor = self.acceptor_density_field()

        return Field(
            name="net_doping_density",
            units="1/m^3",
            grid=self.grid,
            values=donor.values - acceptor.values,
        )