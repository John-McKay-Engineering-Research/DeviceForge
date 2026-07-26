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
from deviceforge.physics import SILICON, SILICON_DIOXIDE


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

# one-dimensional silicon-dioxide/silicon dielectric stack

@pytest.fixture
def dielectric_stack_simulation() -> Simulation:
    """
    Return a one-dimensional silicon-dioxide/silicon dielectric stack.

    The left endpoint is fixed at 0 V and the right endpoint at 1 V.
    The device is charge-free and contains one material interface.
    """

    number_of_points = 101

    grid = Grid(
        shape=(number_of_points,),
        spacing=(1.0e-9,),
    )

    interface_index = number_of_points // 2

    oxide_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    oxide_mask[:interface_index] = True

    silicon_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    silicon_mask[interface_index:] = True

    oxide_region = Region(
        name="silicon_dioxide",
        grid=grid,
        material=SILICON_DIOXIDE,
        mask=oxide_mask,
        region_type="dielectric",
    )

    silicon_region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=silicon_mask,
        region_type="semiconductor",
    )

    device = Device(
        name="dielectric_stack_device",
        grid=grid,
        regions=(
            oxide_region,
            silicon_region,
        ),
    )

    left_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    left_mask[0] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    right_mask[-1] = True

    left_boundary = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type="dirichlet",
        value=1.0,
        units="V",
    )

    return Simulation(
        name="dielectric_stack_simulation",
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-10,
        max_iterations=500,
        initial_potential=0.0,
    )