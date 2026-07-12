import numpy as np
import pytest

from deviceforge import Field, Grid


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(4, 3),
        spacing=(1.0e-9, 2.0e-9),
    )


def test_create_field(grid_2d: Grid) -> None:
    values = np.arange(12, dtype=np.float64).reshape(4, 3)

    field = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
        values=values,
    )

    assert field.name == "electrostatic_potential"
    assert field.units == "V"
    assert field.grid is grid_2d
    assert field.shape == (4, 3)

    np.testing.assert_allclose(field.values, values)


def test_values_are_converted_to_float64(grid_2d: Grid) -> None:
    field = Field(
        name="doping",
        units="1/m^3",
        grid=grid_2d,
        values=np.ones(grid_2d.shape, dtype=np.int32),
    )

    assert field.values.dtype == np.float64


def test_zero_field_factory(grid_2d: Grid) -> None:
    field = Field.zeros(
        name="potential",
        units="V",
        grid=grid_2d,
    )

    assert field.shape == grid_2d.shape
    assert field.minimum == pytest.approx(0.0)
    assert field.maximum == pytest.approx(0.0)
    assert field.mean == pytest.approx(0.0)


def test_full_field_factory(grid_2d: Grid) -> None:
    field = Field.full(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid_2d,
        fill_value=11.7,
    )

    assert field.minimum == pytest.approx(11.7)
    assert field.maximum == pytest.approx(11.7)
    assert field.mean == pytest.approx(11.7)


def test_field_statistics(grid_2d: Grid) -> None:
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )

    field = Field(
        name="test_field",
        units="arbitrary",
        grid=grid_2d,
        values=values,
    )

    assert field.minimum == pytest.approx(1.0)
    assert field.maximum == pytest.approx(12.0)
    assert field.mean == pytest.approx(6.5)


def test_copy_with_new_values(grid_2d: Grid) -> None:
    original = Field.zeros(
        name="potential",
        units="V",
        grid=grid_2d,
    )

    updated_values = np.full(grid_2d.shape, 2.5)

    updated = original.copy_with_values(updated_values)

    assert updated is not original
    assert updated.grid is original.grid
    assert updated.name == original.name
    assert updated.units == original.units
    assert updated.mean == pytest.approx(2.5)
    assert original.mean == pytest.approx(0.0)


def test_copy_can_change_metadata(grid_2d: Grid) -> None:
    original = Field.zeros(
        name="potential",
        units="V",
        grid=grid_2d,
    )

    updated = original.copy_with_values(
        np.ones(grid_2d.shape),
        name="normalised_potential",
        units="dimensionless",
    )

    assert updated.name == "normalised_potential"
    assert updated.units == "dimensionless"


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((4, 4)),
        np.zeros((3, 4)),
        np.zeros((12,)),
    ],
)
def test_shape_mismatch_raises_value_error(
    grid_2d: Grid,
    values: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="same shape"):
        Field(
            name="invalid",
            units="V",
            grid=grid_2d,
            values=values,
        )


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        "\t",
    ],
)
def test_empty_name_raises_value_error(
    grid_2d: Grid,
    name: str,
) -> None:
    with pytest.raises(ValueError, match="name"):
        Field.zeros(
            name=name,
            units="V",
            grid=grid_2d,
        )


@pytest.mark.parametrize(
    "units",
    [
        "",
        "   ",
        "\t",
    ],
)
def test_empty_units_raise_value_error(
    grid_2d: Grid,
    units: str,
) -> None:
    with pytest.raises(ValueError, match="units"):
        Field.zeros(
            name="potential",
            units=units,
            grid=grid_2d,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_values_raise_value_error(
    grid_2d: Grid,
    invalid_value: float,
) -> None:
    values = np.zeros(grid_2d.shape)
    values[0, 0] = invalid_value

    with pytest.raises(ValueError, match="NaN or infinite"):
        Field(
            name="invalid",
            units="V",
            grid=grid_2d,
            values=values,
        )


@pytest.mark.parametrize(
    "fill_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_fill_value_raises_value_error(
    grid_2d: Grid,
    fill_value: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        Field.full(
            name="invalid",
            units="V",
            grid=grid_2d,
            fill_value=fill_value,
        )