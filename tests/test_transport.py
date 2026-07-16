import numpy as np
import pytest

from deviceforge import Field, Grid
from deviceforge.physics import (
    ELEMENTARY_CHARGE,
    compute_electron_current_density,
    compute_hole_current_density,
    compute_total_current_density,
    diffusion_coefficient,
    thermal_voltage,
)


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(11, 5),
        spacing=(1.0e-9, 1.0e-9),
    )


def test_einstein_relation() -> None:
    mobility = 0.135

    expected = mobility * thermal_voltage(300.0)

    assert diffusion_coefficient(
        mobility,
        temperature=300.0,
    ) == pytest.approx(expected)


def test_constant_electron_drift_current(
    grid_2d: Grid,
) -> None:
    concentration = Field.full(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=1.0e21,
    )

    electric_field = Field.full(
        name="electric_field_x",
        units="V/m",
        grid=grid_2d,
        fill_value=2.0e5,
    )

    mobility = 0.135

    current = compute_electron_current_density(
        electron_concentration=concentration,
        electric_field_x=electric_field,
        mobility=mobility,
    )

    expected = (
        ELEMENTARY_CHARGE
        * mobility
        * 1.0e21
        * 2.0e5
    )

    np.testing.assert_allclose(
        current.values,
        expected,
    )


def test_constant_hole_drift_current(
    grid_2d: Grid,
) -> None:
    concentration = Field.full(
        name="hole_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=2.0e21,
    )

    electric_field = Field.full(
        name="electric_field_x",
        units="V/m",
        grid=grid_2d,
        fill_value=-1.0e5,
    )

    mobility = 0.048

    current = compute_hole_current_density(
        hole_concentration=concentration,
        electric_field_x=electric_field,
        mobility=mobility,
    )

    expected = (
        ELEMENTARY_CHARGE
        * mobility
        * 2.0e21
        * -1.0e5
    )

    np.testing.assert_allclose(
        current.values,
        expected,
    )


def test_electron_diffusion_current(
    grid_2d: Grid,
) -> None:
    x, _ = grid_2d.mesh()

    gradient = 2.0e28

    concentration = Field(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        values=1.0e20 + gradient * x,
    )

    electric_field = Field.zeros(
        name="electric_field_x",
        units="V/m",
        grid=grid_2d,
    )

    mobility = 0.135

    current = compute_electron_current_density(
        electron_concentration=concentration,
        electric_field_x=electric_field,
        mobility=mobility,
    )

    expected = (
        ELEMENTARY_CHARGE
        * diffusion_coefficient(mobility)
        * gradient
    )

    np.testing.assert_allclose(
        current.values,
        expected,
        rtol=1.0e-12,
    )


def test_hole_diffusion_has_opposite_gradient_sign(
    grid_2d: Grid,
) -> None:
    x, _ = grid_2d.mesh()

    gradient = 2.0e28

    concentration = Field(
        name="hole_concentration",
        units="1/m^3",
        grid=grid_2d,
        values=1.0e20 + gradient * x,
    )

    electric_field = Field.zeros(
        name="electric_field_x",
        units="V/m",
        grid=grid_2d,
    )

    mobility = 0.048

    current = compute_hole_current_density(
        hole_concentration=concentration,
        electric_field_x=electric_field,
        mobility=mobility,
    )

    expected = (
        -ELEMENTARY_CHARGE
        * diffusion_coefficient(mobility)
        * gradient
    )

    np.testing.assert_allclose(
        current.values,
        expected,
        rtol=1.0e-12,
    )


def test_total_current_density(
    grid_2d: Grid,
) -> None:
    electron_current = Field.full(
        name="electron_current_density_x",
        units="A/m^2",
        grid=grid_2d,
        fill_value=3.0,
    )

    hole_current = Field.full(
        name="hole_current_density_x",
        units="A/m^2",
        grid=grid_2d,
        fill_value=2.0,
    )

    total = compute_total_current_density(
        electron_current_density=electron_current,
        hole_current_density=hole_current,
    )

    np.testing.assert_allclose(
        total.values,
        5.0,
    )


@pytest.mark.parametrize(
    "mobility",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_mobility_is_rejected(
    mobility: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Mobility",
    ):
        diffusion_coefficient(mobility)