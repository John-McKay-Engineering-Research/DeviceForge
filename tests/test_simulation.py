import numpy as np
import pytest

from deviceforge import (
    BoundaryCondition,
    BoundaryConditionType,
    Device,
    Grid,
    Region,
    Simulation,
)
from deviceforge.physics import SILICON


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(6, 4),
        spacing=(1.0e-9, 1.0e-9),
    )


@pytest.fixture
def device_2d(grid_2d: Grid) -> Device:
    mask = np.ones(grid_2d.shape, dtype=bool)

    region = Region(
        name="silicon_domain",
        grid=grid_2d,
        material=SILICON,
        mask=mask,
    )

    return Device(
        name="rectangular_device",
        grid=grid_2d,
        regions=(region,),
    )


@pytest.fixture
def left_boundary(
    grid_2d: Grid,
) -> BoundaryCondition:
    mask = np.zeros(grid_2d.shape, dtype=bool)
    mask[0, :] = True

    return BoundaryCondition(
        name="left_contact",
        grid=grid_2d,
        mask=mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )


@pytest.fixture
def right_boundary(
    grid_2d: Grid,
) -> BoundaryCondition:
    mask = np.zeros(grid_2d.shape, dtype=bool)
    mask[-1, :] = True

    return BoundaryCondition(
        name="right_contact",
        grid=grid_2d,
        mask=mask,
        condition_type="dirichlet",
        value=1.0,
        units="V",
    )


def test_create_simulation(
    device_2d: Device,
    left_boundary: BoundaryCondition,
    right_boundary: BoundaryCondition,
) -> None:
    simulation = Simulation(
        name="laplace_rectangle",
        device=device_2d,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-10,
        max_iterations=5_000,
        initial_potential=0.25,
    )

    assert simulation.name == "laplace_rectangle"
    assert simulation.grid is device_2d.grid
    assert simulation.tolerance == pytest.approx(1.0e-10)
    assert simulation.max_iterations == 5_000
    assert simulation.initial_potential == pytest.approx(0.25)
    assert simulation.number_of_boundary_conditions == 2


def test_boundary_classification(
    device_2d: Device,
    left_boundary: BoundaryCondition,
    grid_2d: Grid,
) -> None:
    top_mask = np.zeros(grid_2d.shape, dtype=bool)
    top_mask[1:, -1] = True

    top_boundary = BoundaryCondition(
        name="top_insulated",
        grid=grid_2d,
        mask=top_mask,
        condition_type="neumann",
        value=0.0,
        units="V/m",
    )

    simulation = Simulation(
        device=device_2d,
        boundary_conditions=(
            left_boundary,
            top_boundary,
        ),
    )

    assert len(simulation.dirichlet_boundaries) == 1
    assert len(simulation.neumann_boundaries) == 1

    assert (
        simulation.dirichlet_boundaries[0].condition_type
        is BoundaryConditionType.DIRICHLET
    )

    assert (
        simulation.neumann_boundaries[0].condition_type
        is BoundaryConditionType.NEUMANN
    )


def test_get_boundary_condition(
    device_2d: Device,
    left_boundary: BoundaryCondition,
) -> None:
    simulation = Simulation(
        device=device_2d,
        boundary_conditions=(left_boundary,),
    )

    selected = simulation.get_boundary_condition("left_contact")

    assert selected is left_boundary


def test_missing_boundary_condition_raises_key_error(
    device_2d: Device,
    left_boundary: BoundaryCondition,
) -> None:
    simulation = Simulation(
        device=device_2d,
        boundary_conditions=(left_boundary,),
    )

    with pytest.raises(KeyError, match="missing"):
        simulation.get_boundary_condition("missing")


def test_initial_potential_field(
    device_2d: Device,
    left_boundary: BoundaryCondition,
    right_boundary: BoundaryCondition,
) -> None:
    simulation = Simulation(
        device=device_2d,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        initial_potential=0.5,
    )

    potential = simulation.create_initial_potential_field()

    assert potential.name == "electrostatic_potential"
    assert potential.units == "V"
    assert potential.grid is device_2d.grid

    np.testing.assert_allclose(
        potential.values[0, :],
        0.0,
    )

    np.testing.assert_allclose(
        potential.values[-1, :],
        1.0,
    )

    np.testing.assert_allclose(
        potential.values[1:-1, :],
        0.5,
    )


def test_fixed_potential_mask(
    device_2d: Device,
    left_boundary: BoundaryCondition,
    right_boundary: BoundaryCondition,
) -> None:
    simulation = Simulation(
        device=device_2d,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
    )

    fixed_mask = simulation.create_fixed_potential_mask()

    assert fixed_mask.dtype == np.bool_
    assert np.all(fixed_mask[0, :])
    assert np.all(fixed_mask[-1, :])
    assert not np.any(fixed_mask[1:-1, :])


def test_duplicate_boundary_names_raise_value_error(
    device_2d: Device,
    left_boundary: BoundaryCondition,
    grid_2d: Grid,
) -> None:
    second_mask = np.zeros(grid_2d.shape, dtype=bool)
    second_mask[-1, :] = True

    duplicate = BoundaryCondition(
        name="left_contact",
        grid=grid_2d,
        mask=second_mask,
        condition_type="dirichlet",
        value=1.0,
        units="V",
    )

    with pytest.raises(ValueError, match="unique"):
        Simulation(
            device=device_2d,
            boundary_conditions=(
                left_boundary,
                duplicate,
            ),
        )


def test_conflicting_overlapping_boundaries_raise_value_error(
    device_2d: Device,
    grid_2d: Grid,
) -> None:
    left_mask = np.zeros(grid_2d.shape, dtype=bool)
    left_mask[0, :] = True

    bottom_mask = np.zeros(grid_2d.shape, dtype=bool)
    bottom_mask[:, 0] = True

    left = BoundaryCondition(
        name="left",
        grid=grid_2d,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    bottom = BoundaryCondition(
        name="bottom",
        grid=grid_2d,
        mask=bottom_mask,
        condition_type="dirichlet",
        value=1.0,
        units="V",
    )

    with pytest.raises(ValueError, match="conflict"):
        Simulation(
            device=device_2d,
            boundary_conditions=(left, bottom),
        )


def test_identical_overlapping_boundaries_are_allowed(
    device_2d: Device,
    grid_2d: Grid,
) -> None:
    left_mask = np.zeros(grid_2d.shape, dtype=bool)
    left_mask[0, :] = True

    bottom_mask = np.zeros(grid_2d.shape, dtype=bool)
    bottom_mask[:, 0] = True

    left = BoundaryCondition(
        name="left",
        grid=grid_2d,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    bottom = BoundaryCondition(
        name="bottom",
        grid=grid_2d,
        mask=bottom_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    simulation = Simulation(
        device=device_2d,
        boundary_conditions=(left, bottom),
    )

    assert simulation.number_of_boundary_conditions == 2


def test_boundary_from_different_grid_raises_value_error(
    device_2d: Device,
) -> None:
    other_grid = Grid(
        shape=(8, 4),
        spacing=(1.0e-9, 1.0e-9),
    )

    mask = np.zeros(other_grid.shape, dtype=bool)
    mask[0, :] = True

    boundary = BoundaryCondition(
        name="wrong_grid",
        grid=other_grid,
        mask=mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    with pytest.raises(ValueError, match="device grid"):
        Simulation(
            device=device_2d,
            boundary_conditions=(boundary,),
        )


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_empty_name_raises_value_error(
    device_2d: Device,
    left_boundary: BoundaryCondition,
    name: str,
) -> None:
    with pytest.raises(ValueError, match="name"):
        Simulation(
            name=name,
            device=device_2d,
            boundary_conditions=(left_boundary,),
        )


@pytest.mark.parametrize(
    "tolerance",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_tolerance_raises_value_error(
    device_2d: Device,
    left_boundary: BoundaryCondition,
    tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        Simulation(
            device=device_2d,
            boundary_conditions=(left_boundary,),
            tolerance=tolerance,
        )


@pytest.mark.parametrize(
    "max_iterations",
    [
        0,
        -1,
        -100,
    ],
)
def test_invalid_max_iterations_raises_value_error(
    device_2d: Device,
    left_boundary: BoundaryCondition,
    max_iterations: int,
) -> None:
    with pytest.raises(ValueError, match="iteration"):
        Simulation(
            device=device_2d,
            boundary_conditions=(left_boundary,),
            max_iterations=max_iterations,
        )


def test_boolean_max_iterations_raises_type_error(
    device_2d: Device,
    left_boundary: BoundaryCondition,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        Simulation(
            device=device_2d,
            boundary_conditions=(left_boundary,),
            max_iterations=True,
        )


@pytest.mark.parametrize(
    "initial_potential",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_initial_potential_raises_value_error(
    device_2d: Device,
    left_boundary: BoundaryCondition,
    initial_potential: float,
) -> None:
    with pytest.raises(ValueError, match="Initial potential"):
        Simulation(
            device=device_2d,
            boundary_conditions=(left_boundary,),
            initial_potential=initial_potential,
        )


def test_empty_boundary_collection_raises_value_error(
    device_2d: Device,
) -> None:
    with pytest.raises(ValueError, match="at least one"):
        Simulation(
            device=device_2d,
            boundary_conditions=(),
        )