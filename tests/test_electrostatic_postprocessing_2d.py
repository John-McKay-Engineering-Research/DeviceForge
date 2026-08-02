from __future__ import annotations

import numpy as np
import pytest

from deviceforge import (
    Field,
    Grid,
)
from deviceforge.postprocessing import (
    calculate_electric_field_components_2d,
    calculate_electric_field_magnitude_2d,
    calculate_electrostatic_fields_2d,
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