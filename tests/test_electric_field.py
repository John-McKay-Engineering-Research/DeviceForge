import numpy as np
import pytest

from deviceforge import Field, Grid
from deviceforge.postprocessing import (
    ElectricField,
    compute_electric_field,
)


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(11, 7),
        spacing=(0.5, 0.25),
    )


def test_compute_linear_electric_field(
    grid_2d: Grid,
) -> None:
    x_coordinates, y_coordinates = grid_2d.mesh()

    potential_values = (
        2.0 * x_coordinates
        + 3.0 * y_coordinates
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
        values=potential_values,
    )

    electric_field = compute_electric_field(potential)

    np.testing.assert_allclose(
        electric_field.x_component.values,
        -2.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        electric_field.y_component.values,
        -3.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        electric_field.magnitude.values,
        np.sqrt(13.0),
        atol=1.0e-12,
    )


def test_electric_field_metadata(
    grid_2d: Grid,
) -> None:
    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
    )

    electric_field = compute_electric_field(potential)

    assert isinstance(electric_field, ElectricField)
    assert electric_field.grid is grid_2d

    assert electric_field.x_component.name == (
        "electric_field_x"
    )

    assert electric_field.y_component.name == (
        "electric_field_y"
    )

    assert electric_field.magnitude.name == (
        "electric_field_magnitude"
    )

    assert electric_field.x_component.units == "V/m"
    assert electric_field.y_component.units == "V/m"
    assert electric_field.magnitude.units == "V/m"


def test_zero_potential_produces_zero_field(
    grid_2d: Grid,
) -> None:
    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
    )

    electric_field = compute_electric_field(potential)

    np.testing.assert_allclose(
        electric_field.x_component.values,
        0.0,
    )

    np.testing.assert_allclose(
        electric_field.y_component.values,
        0.0,
    )

    np.testing.assert_allclose(
        electric_field.magnitude.values,
        0.0,
    )


def test_components_property(
    grid_2d: Grid,
) -> None:
    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
    )

    electric_field = compute_electric_field(potential)

    assert electric_field.components == (
        electric_field.x_component,
        electric_field.y_component,
    )


def test_one_dimensional_field_is_rejected() -> None:
    grid = Grid(
        shape=(10,),
        spacing=(1.0,),
    )

    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid,
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        compute_electric_field(potential)


def test_non_voltage_field_is_rejected(
    grid_2d: Grid,
) -> None:
    invalid_field = Field.zeros(
        name="temperature",
        units="K",
        grid=grid_2d,
    )

    with pytest.raises(
        ValueError,
        match="units of V",
    ):
        compute_electric_field(invalid_field)


def test_small_grid_uses_first_order_edges() -> None:
    grid = Grid(
        shape=(2, 2),
        spacing=(1.0, 1.0),
    )

    x_coordinates, _ = grid.mesh()

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=x_coordinates,
    )

    electric_field = compute_electric_field(potential)

    np.testing.assert_allclose(
        electric_field.x_component.values,
        -1.0,
    )