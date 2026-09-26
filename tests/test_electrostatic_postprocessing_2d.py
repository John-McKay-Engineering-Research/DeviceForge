from __future__ import annotations

import numpy as np
import pytest

from deviceforge import (
    Field,
    Grid,
)

from deviceforge.core.face_field import (
    FaceField,
)

from deviceforge.postprocessing import (
    calculate_electric_field_components_2d,
    calculate_electric_field_magnitude_2d,
    calculate_electrostatic_fields_2d,
    calculate_face_electric_field_components_2d,
    calculate_face_relative_permittivity_components_2d,
    calculate_face_electric_displacement_components_2d,
    calculate_face_electrostatic_fields_2d,
)



def create_linear_potential_field_2d() -> tuple[
    Field,
    float,
    float,
]:
    """
    Create

        phi(x, y) = 2x - 3y

    whose exact electric field is

        E_axis_0 = -2 V/m
        E_axis_1 =  3 V/m.
    """

    grid = Grid(
        shape=(11, 9),
        spacing=(0.2, 0.35),
        origin=(0.0, 0.0),
    )

    coordinates_axis_0 = (
        grid.coordinates(0)
    )

    coordinates_axis_1 = (
        grid.coordinates(1)
    )

    potential_values = (
        2.0
        * coordinates_axis_0[:, None]
        - 3.0
        * coordinates_axis_1[None, :]
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=potential_values,
    )

    return potential, -2.0, 3.0


def test_components_match_linear_analytical_field() -> None:
    (
        potential,
        expected_axis_0,
        expected_axis_1,
    ) = create_linear_potential_field_2d()

    (
        electric_field_axis_0,
        electric_field_axis_1,
    ) = calculate_electric_field_components_2d(
        potential
    )

    np.testing.assert_allclose(
        electric_field_axis_0.values,
        expected_axis_0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        electric_field_axis_1.values,
        expected_axis_1,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_magnitude_matches_analytical_value() -> None:
    potential, _, _ = (
        create_linear_potential_field_2d()
    )

    (
        _,
        _,
        electric_field_magnitude,
    ) = calculate_electrostatic_fields_2d(
        potential
    )

    np.testing.assert_allclose(
        electric_field_magnitude.values,
        np.sqrt(13.0),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_postprocessing_preserves_grid_and_shapes() -> None:
    potential, _, _ = (
        create_linear_potential_field_2d()
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
        electric_field_magnitude,
    ) = calculate_electrostatic_fields_2d(
        potential
    )

    for field in (
        electric_field_axis_0,
        electric_field_axis_1,
        electric_field_magnitude,
    ):
        assert field.grid is potential.grid
        assert field.values.shape == (
            potential.grid.shape
        )


def test_component_names_and_units() -> None:
    potential, _, _ = (
        create_linear_potential_field_2d()
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
        electric_field_magnitude,
    ) = calculate_electrostatic_fields_2d(
        potential
    )

    assert electric_field_axis_0.name == (
        "electric_field_axis_0"
    )
    assert electric_field_axis_1.name == (
        "electric_field_axis_1"
    )
    assert electric_field_magnitude.name == (
        "electric_field_magnitude"
    )

    assert electric_field_axis_0.units == "V/m"
    assert electric_field_axis_1.units == "V/m"
    assert electric_field_magnitude.units == "V/m"


def test_components_reject_one_dimensional_field() -> None:
    grid = Grid(
        shape=(11,),
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
        calculate_electric_field_components_2d(
            potential
        )


def test_components_reject_wrong_units() -> None:
    grid = Grid(
        shape=(5, 5),
        spacing=(1.0, 1.0),
    )

    invalid = Field.zeros(
        name="invalid",
        units="A",
        grid=grid,
    )

    with pytest.raises(
        ValueError,
        match="units must be 'V'",
    ):
        calculate_electric_field_components_2d(
            invalid
        )


def test_magnitude_rejects_mismatched_grids() -> None:
    first_grid = Grid(
        shape=(5, 5),
        spacing=(1.0, 1.0),
    )

    second_grid = Grid(
        shape=(5, 5),
        spacing=(2.0, 1.0),
    )

    first = Field.zeros(
        name="electric_field_axis_0",
        units="V/m",
        grid=first_grid,
    )

    second = Field.zeros(
        name="electric_field_axis_1",
        units="V/m",
        grid=second_grid,
    )

    with pytest.raises(
        ValueError,
        match="same grid",
    ):
        calculate_electric_field_magnitude_2d(
            first,
            second,
        )

def test_components_match_quadratic_analytical_field() -> None:
    """
    Verify second-order 2D electric-field differentiation using a
    quadratic potential with variation and coupling along both axes.
    """

    grid = Grid(
        shape=(13, 11),
        spacing=(0.2, 0.35),
        origin=(0.0, 0.0),
    )

    coordinates_axis_0 = grid.coordinates(0)
    coordinates_axis_1 = grid.coordinates(1)

    x = coordinates_axis_0[:, None]
    y = coordinates_axis_1[None, :]

    coefficient_x_squared = 1.5
    coefficient_y_squared = -0.75
    coefficient_xy = 0.8
    coefficient_x = 2.0
    coefficient_y = -3.0
    offset = 0.4

    potential_values = (
        coefficient_x_squared * x**2
        + coefficient_y_squared * y**2
        + coefficient_xy * x * y
        + coefficient_x * x
        + coefficient_y * y
        + offset
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=potential_values,
    )

    expected_axis_0 = -(
        2.0
        * coefficient_x_squared
        * x
        + coefficient_xy * y
        + coefficient_x
    )

    expected_axis_1 = -(
        2.0
        * coefficient_y_squared
        * y
        + coefficient_xy * x
        + coefficient_y
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
    ) = calculate_electric_field_components_2d(
        potential
    )

    np.testing.assert_allclose(
        electric_field_axis_0.values,
        expected_axis_0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        electric_field_axis_1.values,
        expected_axis_1,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

def test_face_electric_field_components_match_linear_analytical_field() -> None:
    """
    Verify face-centred electric-field components for a linear
    two-dimensional potential.
    """

    grid = Grid(
        shape=(11, 9),
        spacing=(0.2, 0.35),
        origin=(0.0, 0.0),
    )

    coordinates_axis_0 = grid.coordinates(0)
    coordinates_axis_1 = grid.coordinates(1)

    x = coordinates_axis_0[:, None]
    y = coordinates_axis_1[None, :]

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=2.0 * x - 3.0 * y,
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
    ) = calculate_face_electric_field_components_2d(
        potential
    )

    np.testing.assert_allclose(
        electric_field_axis_0.values,
        -2.0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        electric_field_axis_1.values,
        3.0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_face_electric_field_components_have_expected_shapes() -> None:
    grid = Grid(
        shape=(11, 9),
        spacing=(0.2, 0.35),
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=np.zeros(grid.shape),
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
    ) = calculate_face_electric_field_components_2d(
        potential
    )

    assert electric_field_axis_0.shape == (10, 9)
    assert electric_field_axis_1.shape == (11, 8)

    assert electric_field_axis_0.axis == 0
    assert electric_field_axis_1.axis == 1


def test_face_electric_field_components_preserve_grid() -> None:
    grid = Grid(
        shape=(11, 9),
        spacing=(0.2, 0.35),
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=np.zeros(grid.shape),
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
    ) = calculate_face_electric_field_components_2d(
        potential
    )

    assert electric_field_axis_0.grid is grid
    assert electric_field_axis_1.grid is grid


def test_face_electric_field_components_have_expected_names_and_units() -> None:
    grid = Grid(
        shape=(11, 9),
        spacing=(0.2, 0.35),
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=np.zeros(grid.shape),
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
    ) = calculate_face_electric_field_components_2d(
        potential
    )

    assert (
        electric_field_axis_0.name
        == "electric_field_axis_0_face"
    )
    assert (
        electric_field_axis_1.name
        == "electric_field_axis_1_face"
    )

    assert electric_field_axis_0.units == "V/m"
    assert electric_field_axis_1.units == "V/m"


def test_face_electric_field_components_reject_one_dimensional_field() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(0.2,),
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=np.zeros(grid.shape),
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        calculate_face_electric_field_components_2d(
            potential
        )


def test_face_electric_field_components_reject_wrong_units() -> None:
    grid = Grid(
        shape=(11, 9),
        spacing=(0.2, 0.35),
    )

    potential = Field(
        name="electrostatic_potential",
        units="K",
        grid=grid,
        values=np.zeros(grid.shape),
    )

    with pytest.raises(
        ValueError,
        match="volts",
    ):
        calculate_face_electric_field_components_2d(
            potential
        )

def test_face_relative_permittivity_components_constant_material() -> None:
    """
    Verify that constant nodal relative permittivity remains constant
    on faces along both grid axes.
    """

    grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=np.full(
            grid.shape,
            11.7,
        ),
    )

    (
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
    ) = calculate_face_relative_permittivity_components_2d(
        relative_permittivity
    )

    np.testing.assert_allclose(
        relative_permittivity_axis_0.values,
        11.7,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        relative_permittivity_axis_1.values,
        11.7,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_face_relative_permittivity_components_use_harmonic_mean() -> None:
    """
    Verify harmonic averaging across material interfaces along both axes.
    """

    grid = Grid(
        shape=(4, 4),
        spacing=(1.0, 1.0),
    )

    values = np.array(
        [
            [2.0, 2.0, 8.0, 8.0],
            [2.0, 2.0, 8.0, 8.0],
            [18.0, 18.0, 72.0, 72.0],
            [18.0, 18.0, 72.0, 72.0],
        ],
        dtype=np.float64,
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=values,
    )

    (
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
    ) = calculate_face_relative_permittivity_components_2d(
        relative_permittivity
    )

    expected_axis_0 = (
        2.0
        * values[:-1, :]
        * values[1:, :]
        / (
            values[:-1, :]
            + values[1:, :]
        )
    )

    expected_axis_1 = (
        2.0
        * values[:, :-1]
        * values[:, 1:]
        / (
            values[:, :-1]
            + values[:, 1:]
        )
    )

    np.testing.assert_allclose(
        relative_permittivity_axis_0.values,
        expected_axis_0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        relative_permittivity_axis_1.values,
        expected_axis_1,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    # Explicitly verify representative material-interface values.
    assert (
        relative_permittivity_axis_0.values[1, 0]
        == pytest.approx(3.6)
    )

    assert (
        relative_permittivity_axis_1.values[0, 1]
        == pytest.approx(3.2)
    )


def test_face_relative_permittivity_components_have_expected_shapes_and_axes() -> None:
    grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=np.full(
            grid.shape,
            11.7,
        ),
    )

    (
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
    ) = calculate_face_relative_permittivity_components_2d(
        relative_permittivity
    )

    assert relative_permittivity_axis_0.shape == (5, 5)
    assert relative_permittivity_axis_1.shape == (6, 4)

    assert relative_permittivity_axis_0.axis == 0
    assert relative_permittivity_axis_1.axis == 1

    assert (
        relative_permittivity_axis_0.units
        == "dimensionless"
    )
    assert (
        relative_permittivity_axis_1.units
        == "dimensionless"
    )


def test_face_relative_permittivity_components_reject_one_dimensional_field() -> None:
    grid = Grid(
        shape=(6,),
        spacing=(0.2,),
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=np.full(
            grid.shape,
            11.7,
        ),
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        calculate_face_relative_permittivity_components_2d(
            relative_permittivity
        )


def test_face_relative_permittivity_components_reject_wrong_units() -> None:
    grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="F/m",
        grid=grid,
        values=np.full(
            grid.shape,
            11.7,
        ),
    )

    with pytest.raises(
        ValueError,
        match="dimensionless",
    ):
        calculate_face_relative_permittivity_components_2d(
            relative_permittivity
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        0.0,
        -1.0,
    ],
)
def test_face_relative_permittivity_components_reject_non_positive_values(
    invalid_value: float,
) -> None:
    grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    values = np.full(
        grid.shape,
        11.7,
    )
    values[2, 2] = invalid_value

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=values,
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        calculate_face_relative_permittivity_components_2d(
            relative_permittivity
        )

def test_face_electric_displacement_components_2d() -> None:
    grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    electric_field_axis_0 = FaceField(
        name="electric_field_axis_0_face",
        units="V/m",
        grid=grid,
        values=np.full(
            (5, 5),
            -2.0,
        ),
        axis=0,
    )

    electric_field_axis_1 = FaceField(
        name="electric_field_axis_1_face",
        units="V/m",
        grid=grid,
        values=np.full(
            (6, 4),
            3.0,
        ),
        axis=1,
    )

    relative_permittivity_axis_0 = FaceField(
        name="relative_permittivity_axis_0_face",
        units="dimensionless",
        grid=grid,
        values=np.full(
            (5, 5),
            11.7,
        ),
        axis=0,
    )

    relative_permittivity_axis_1 = FaceField(
        name="relative_permittivity_axis_1_face",
        units="dimensionless",
        grid=grid,
        values=np.full(
            (6, 4),
            11.7,
        ),
        axis=1,
    )

    (
        displacement_axis_0,
        displacement_axis_1,
    ) = calculate_face_electric_displacement_components_2d(
        electric_field_axis_0,
        electric_field_axis_1,
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
    )

    vacuum_permittivity = 8.8541878128e-12

    np.testing.assert_allclose(
        displacement_axis_0.values,
        vacuum_permittivity * 11.7 * -2.0,
    )

    np.testing.assert_allclose(
        displacement_axis_1.values,
        vacuum_permittivity * 11.7 * 3.0,
    )


def test_face_electric_displacement_components_2d_metadata() -> None:
    grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    electric_field_axis_0 = FaceField(
        name="electric_field_axis_0_face",
        units="V/m",
        grid=grid,
        values=np.ones((5, 5)),
        axis=0,
    )

    electric_field_axis_1 = FaceField(
        name="electric_field_axis_1_face",
        units="V/m",
        grid=grid,
        values=np.ones((6, 4)),
        axis=1,
    )

    relative_permittivity_axis_0 = FaceField(
        name="relative_permittivity_axis_0_face",
        units="dimensionless",
        grid=grid,
        values=np.ones((5, 5)),
        axis=0,
    )

    relative_permittivity_axis_1 = FaceField(
        name="relative_permittivity_axis_1_face",
        units="dimensionless",
        grid=grid,
        values=np.ones((6, 4)),
        axis=1,
    )

    (
        displacement_axis_0,
        displacement_axis_1,
    ) = calculate_face_electric_displacement_components_2d(
        electric_field_axis_0,
        electric_field_axis_1,
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
    )

    assert displacement_axis_0.grid == grid
    assert displacement_axis_1.grid == grid

    assert displacement_axis_0.axis == 0
    assert displacement_axis_1.axis == 1

    assert displacement_axis_0.shape == (5, 5)
    assert displacement_axis_1.shape == (6, 4)

    assert displacement_axis_0.units == "C/m^2"
    assert displacement_axis_1.units == "C/m^2"

    assert (
        displacement_axis_0.name
        == "electric_displacement_axis_0_face"
    )
    assert (
        displacement_axis_1.name
        == "electric_displacement_axis_1_face"
    )

def test_face_electric_displacement_components_2d_reject_wrong_electric_field_units() -> None:
    grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    electric_field_axis_0 = FaceField(
        name="electric_field_axis_0_face",
        units="V",
        grid=grid,
        values=np.ones((5, 5)),
        axis=0,
    )

    electric_field_axis_1 = FaceField(
        name="electric_field_axis_1_face",
        units="V/m",
        grid=grid,
        values=np.ones((6, 4)),
        axis=1,
    )

    relative_permittivity_axis_0 = FaceField(
        name="relative_permittivity_axis_0_face",
        units="dimensionless",
        grid=grid,
        values=np.ones((5, 5)),
        axis=0,
    )

    relative_permittivity_axis_1 = FaceField(
        name="relative_permittivity_axis_1_face",
        units="dimensionless",
        grid=grid,
        values=np.ones((6, 4)),
        axis=1,
    )

    with pytest.raises(
        ValueError,
        match="V/m",
    ):
        calculate_face_electric_displacement_components_2d(
            electric_field_axis_0,
            electric_field_axis_1,
            relative_permittivity_axis_0,
            relative_permittivity_axis_1,
        )


def test_face_electric_displacement_components_2d_reject_wrong_permittivity_units() -> None:
    grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    electric_field_axis_0 = FaceField(
        name="electric_field_axis_0_face",
        units="V/m",
        grid=grid,
        values=np.ones((5, 5)),
        axis=0,
    )

    electric_field_axis_1 = FaceField(
        name="electric_field_axis_1_face",
        units="V/m",
        grid=grid,
        values=np.ones((6, 4)),
        axis=1,
    )

    relative_permittivity_axis_0 = FaceField(
        name="relative_permittivity_axis_0_face",
        units="F/m",
        grid=grid,
        values=np.ones((5, 5)),
        axis=0,
    )

    relative_permittivity_axis_1 = FaceField(
        name="relative_permittivity_axis_1_face",
        units="dimensionless",
        grid=grid,
        values=np.ones((6, 4)),
        axis=1,
    )

    with pytest.raises(
        ValueError,
        match="dimensionless",
    ):
        calculate_face_electric_displacement_components_2d(
            electric_field_axis_0,
            electric_field_axis_1,
            relative_permittivity_axis_0,
            relative_permittivity_axis_1,
        )


def test_face_electric_displacement_components_2d_reject_mismatched_grids() -> None:
    grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    other_grid = Grid(
        shape=(6, 5),
        spacing=(0.25, 0.35),
    )

    electric_field_axis_0 = FaceField(
        name="electric_field_axis_0_face",
        units="V/m",
        grid=grid,
        values=np.ones((5, 5)),
        axis=0,
    )

    electric_field_axis_1 = FaceField(
        name="electric_field_axis_1_face",
        units="V/m",
        grid=grid,
        values=np.ones((6, 4)),
        axis=1,
    )

    relative_permittivity_axis_0 = FaceField(
        name="relative_permittivity_axis_0_face",
        units="dimensionless",
        grid=other_grid,
        values=np.ones((5, 5)),
        axis=0,
    )

    relative_permittivity_axis_1 = FaceField(
        name="relative_permittivity_axis_1_face",
        units="dimensionless",
        grid=grid,
        values=np.ones((6, 4)),
        axis=1,
    )

    with pytest.raises(
        ValueError,
        match="same grid",
    ):
        calculate_face_electric_displacement_components_2d(
            electric_field_axis_0,
            electric_field_axis_1,
            relative_permittivity_axis_0,
            relative_permittivity_axis_1,
        )

def test_face_electric_displacement_components_2d_reject_incorrect_axes() -> None:
    grid = Grid(
        shape=(5, 5),
        spacing=(0.2, 0.35),
    )

    electric_field_axis_0 = FaceField(
        name="electric_field_axis_0_face",
        units="V/m",
        grid=grid,
        values=np.ones((5, 4)),
        axis=1,
    )

    electric_field_axis_1 = FaceField(
        name="electric_field_axis_1_face",
        units="V/m",
        grid=grid,
        values=np.ones((5, 4)),
        axis=1,
    )

    relative_permittivity_axis_0 = FaceField(
        name="relative_permittivity_axis_0_face",
        units="dimensionless",
        grid=grid,
        values=np.ones((4, 5)),
        axis=0,
    )

    relative_permittivity_axis_1 = FaceField(
        name="relative_permittivity_axis_1_face",
        units="dimensionless",
        grid=grid,
        values=np.ones((5, 4)),
        axis=1,
    )

    with pytest.raises(
        ValueError,
        match="axis",
    ):
        calculate_face_electric_displacement_components_2d(
            electric_field_axis_0,
            electric_field_axis_1,
            relative_permittivity_axis_0,
            relative_permittivity_axis_1,
        )

def test_face_electric_displacement_components_2d_reject_non_positive_permittivity() -> None:
    grid = Grid(
        shape=(5, 5),
        spacing=(0.2, 0.35),
    )

    electric_field_axis_0 = FaceField(
        name="electric_field_axis_0_face",
        units="V/m",
        grid=grid,
        values=np.ones((4, 5)),
        axis=0,
    )

    electric_field_axis_1 = FaceField(
        name="electric_field_axis_1_face",
        units="V/m",
        grid=grid,
        values=np.ones((5, 4)),
        axis=1,
    )

    permittivity_axis_0_values = np.ones((4, 5))
    permittivity_axis_0_values[2, 2] = 0.0

    relative_permittivity_axis_0 = FaceField(
        name="relative_permittivity_axis_0_face",
        units="dimensionless",
        grid=grid,
        values=permittivity_axis_0_values,
        axis=0,
    )

    relative_permittivity_axis_1 = FaceField(
        name="relative_permittivity_axis_1_face",
        units="dimensionless",
        grid=grid,
        values=np.ones((5, 4)),
        axis=1,
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        calculate_face_electric_displacement_components_2d(
            electric_field_axis_0,
            electric_field_axis_1,
            relative_permittivity_axis_0,
            relative_permittivity_axis_1,
        )

def test_face_electrostatic_fields_2d_matches_individual_calculations() -> None:
    """
    Verify that the 2D face-electrostatic convenience pipeline
    reproduces the independently calculated face fields.
    """

    grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    axis_0_coordinates, axis_1_coordinates = grid.mesh()

    potential = Field(
        name="potential",
        units="V",
        grid=grid,
        values=(
            2.0 * axis_0_coordinates
            - 3.0 * axis_1_coordinates
        ),
    )

    relative_permittivity_values = np.where(
        axis_0_coordinates < 0.5,
        4.0,
        12.0,
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=relative_permittivity_values,
    )

    expected_electric_field_axis_0, expected_electric_field_axis_1 = (
        calculate_face_electric_field_components_2d(
            potential
        )
    )

    (
        expected_relative_permittivity_axis_0,
        expected_relative_permittivity_axis_1,
    ) = calculate_face_relative_permittivity_components_2d(
        relative_permittivity
    )

    (
        expected_displacement_axis_0,
        expected_displacement_axis_1,
    ) = calculate_face_electric_displacement_components_2d(
        expected_electric_field_axis_0,
        expected_electric_field_axis_1,
        expected_relative_permittivity_axis_0,
        expected_relative_permittivity_axis_1,
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
        displacement_axis_0,
        displacement_axis_1,
    ) = calculate_face_electrostatic_fields_2d(
        potential,
        relative_permittivity,
    )

    np.testing.assert_allclose(
        electric_field_axis_0.values,
        expected_electric_field_axis_0.values,
    )

    np.testing.assert_allclose(
        electric_field_axis_1.values,
        expected_electric_field_axis_1.values,
    )

    np.testing.assert_allclose(
        relative_permittivity_axis_0.values,
        expected_relative_permittivity_axis_0.values,
    )

    np.testing.assert_allclose(
        relative_permittivity_axis_1.values,
        expected_relative_permittivity_axis_1.values,
    )

    np.testing.assert_allclose(
        displacement_axis_0.values,
        expected_displacement_axis_0.values,
    )

    np.testing.assert_allclose(
        displacement_axis_1.values,
        expected_displacement_axis_1.values,
    )

def test_face_electrostatic_fields_2d_rejects_mismatched_grids() -> None:
    potential_grid = Grid(
        shape=(6, 5),
        spacing=(0.2, 0.35),
    )

    permittivity_grid = Grid(
        shape=(6, 5),
        spacing=(0.25, 0.35),
    )

    potential = Field(
        name="potential",
        units="V",
        grid=potential_grid,
        values=np.zeros(
            potential_grid.shape,
        ),
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=permittivity_grid,
        values=np.full(
            permittivity_grid.shape,
            11.7,
        ),
    )

    with pytest.raises(
        ValueError,
        match="same grid",
    ):
        calculate_face_electrostatic_fields_2d(
            potential,
            relative_permittivity,
        )