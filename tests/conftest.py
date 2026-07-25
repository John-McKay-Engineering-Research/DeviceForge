from __future__ import annotations

import numpy as np
import pytest

from deviceforge import (
    BoundaryCondition,
    Device,
    Grid,
    Region,
    Simulation,
)
from deviceforge.physics import SILICON


@pytest.fixture
def grid() -> Grid:
    """
    Return a small one-dimensional grid for shared unit tests.

    Eleven nodes are sufficient for testing object relationships while
    keeping fixture construction inexpensive.
    """

    return Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )


@pytest.fixture
def silicon():
    """Return DeviceForge's standard silicon material."""

    return SILICON


@pytest.fixture
def region_mask(grid: Grid) -> np.ndarray:
    """Return a mask covering the entire test grid."""

    return np.ones(
        grid.shape,
        dtype=bool,
    )


@pytest.fixture
def region(
    grid: Grid,
    silicon,
    region_mask: np.ndarray,
) -> Region:
    """Return a uniform intrinsic-silicon region."""

    return Region(
        name="silicon_region",
        grid=grid,
        material=silicon,
        mask=region_mask,
        region_type="semiconductor",
    )


@pytest.fixture
def device(
    grid: Grid,
    region: Region,
) -> Device:
    """Return a simple one-region semiconductor device."""

    return Device(
        name="test_device",
        grid=grid,
        regions=(region,),
    )


@pytest.fixture
def left_boundary_mask(grid: Grid) -> np.ndarray:
    """Return a mask selecting the first grid node."""

    mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    mask[0] = True

    return mask


@pytest.fixture
def right_boundary_mask(grid: Grid) -> np.ndarray:
    """Return a mask selecting the final grid node."""

    mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    mask[-1] = True

    return mask


@pytest.fixture
def left_boundary_condition(
    grid: Grid,
    left_boundary_mask: np.ndarray,
) -> BoundaryCondition:
    """Return a zero-volt Dirichlet condition at the left boundary."""

    return BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_boundary_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )


@pytest.fixture
def right_boundary_condition(
    grid: Grid,
    right_boundary_mask: np.ndarray,
) -> BoundaryCondition:
    """Return a zero-volt Dirichlet condition at the right boundary."""

    return BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_boundary_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )


@pytest.fixture
def boundary_conditions(
    left_boundary_condition: BoundaryCondition,
    right_boundary_condition: BoundaryCondition,
) -> tuple[BoundaryCondition, ...]:
    """Return the complete boundary-condition collection."""

    return (
        left_boundary_condition,
        right_boundary_condition,
    )


@pytest.fixture
def simulation(
    device: Device,
    boundary_conditions: tuple[BoundaryCondition, ...],
) -> Simulation:
    """Return a valid immutable simulation definition."""

    return Simulation(
        name="test_simulation",
        device=device,
        boundary_conditions=boundary_conditions,
        tolerance=1.0e-8,
        max_iterations=500,
        initial_potential=0.0,
    )