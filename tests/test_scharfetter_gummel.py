import numpy as np
import pytest

from deviceforge import Field, Grid
from deviceforge.physics import (
    ELEMENTARY_CHARGE,
    bernoulli_function,
    compute_electron_scharfetter_gummel_current_x,
    compute_hole_scharfetter_gummel_current_x,
    compute_total_scharfetter_gummel_current_x,
    diffusion_coefficient,
    thermal_voltage,
)


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(11, 5),
        spacing=(1.0e-9, 1.0e-9),
    )


def test_bernoulli_at_zero() -> None:
    assert bernoulli_function(0.0) == pytest.approx(
        1.0
    )


@pytest.mark.parametrize(
    "value",
    [
        1.0e-12,
        -1.0e-12,
        1.0e-8,
        -1.0e-8,
    ],
)
def test_bernoulli_is_stable_near_zero(
    value: float,
) -> None:
    expected = (
        1.0
        - value / 2.0
        + value**2 / 12.0
    )

    assert bernoulli_function(value) == pytest.approx(
        expected,
        rel=1.0e-14,
        abs=1.0e-14,
    )


@pytest.mark.parametrize(
    "value",
    [
        0.1,
        1.0,
        10.0,
        -0.1,
        -1.0,
        -10.0,
    ],
)
def test_bernoulli_symmetry_identity(
    value: float,
) -> None:
    """
    The Bernoulli function satisfies:

        B(-x) = exp(x) B(x)
    """

    assert bernoulli_function(
        -value
    ) == pytest.approx(
        np.exp(value)
        * bernoulli_function(value),
        rel=1.0e-12,
    )


def test_bernoulli_large_arguments_remain_finite() -> None:
    values = bernoulli_function(
        np.array(
            [
                -1.0e3,
                -100.0,
                100.0,
                1.0e3,
            ]
        )
    )

    assert np.all(np.isfinite(values))
    assert np.all(values >= 0.0)


def test_electron_zero_field_diffusion(
    grid_2d: Grid,
) -> None:
    x, _ = grid_2d.mesh()

    concentration_gradient = 2.0e28

    electrons = Field(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        values=(
            1.0e20
            + concentration_gradient * x
        ),
    )

    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
    )

    mobility = 0.135

    current = (
        compute_electron_scharfetter_gummel_current_x(
            potential=potential,
            electron_concentration=electrons,
            mobility=mobility,
        )
    )

    expected = (
        ELEMENTARY_CHARGE
        * diffusion_coefficient(mobility)
        * concentration_gradient
    )

    np.testing.assert_allclose(
        current.values,
        expected,
        rtol=1.0e-12,
    )


def test_hole_zero_field_diffusion(
    grid_2d: Grid,
) -> None:
    x, _ = grid_2d.mesh()

    concentration_gradient = 2.0e28

    holes = Field(
        name="hole_concentration",
        units="1/m^3",
        grid=grid_2d,
        values=(
            1.0e20
            + concentration_gradient * x
        ),
    )

    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
    )

    mobility = 0.048

    current = (
        compute_hole_scharfetter_gummel_current_x(
            potential=potential,
            hole_concentration=holes,
            mobility=mobility,
        )
    )

    expected = (
        -ELEMENTARY_CHARGE
        * diffusion_coefficient(mobility)
        * concentration_gradient
    )

    np.testing.assert_allclose(
        current.values,
        expected,
        rtol=1.0e-12,
    )


def test_equilibrium_electron_flux_cancels(
    grid_2d: Grid,
) -> None:
    voltage = thermal_voltage(300.0)

    x_coordinates = grid_2d.coordinates(
        axis=0
    )

    potential_profile = np.linspace(
        -0.2,
        0.2,
        grid_2d.shape[0],
    )

    potential_values = np.repeat(
        potential_profile[:, np.newaxis],
        grid_2d.shape[1],
        axis=1,
    )

    intrinsic_concentration = 1.0e16

    electron_values = (
        intrinsic_concentration
        * np.exp(
            potential_values / voltage
        )
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
        values=potential_values,
    )

    electrons = Field(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        values=electron_values,
    )

    current = (
        compute_electron_scharfetter_gummel_current_x(
            potential=potential,
            electron_concentration=electrons,
            mobility=0.135,
            temperature=300.0,
        )
    )

    np.testing.assert_allclose(
        current.values,
        0.0,
        atol=1.0e-6,
    )


def test_equilibrium_hole_flux_cancels(
    grid_2d: Grid,
) -> None:
    voltage = thermal_voltage(300.0)

    potential_profile = np.linspace(
        -0.2,
        0.2,
        grid_2d.shape[0],
    )

    potential_values = np.repeat(
        potential_profile[:, np.newaxis],
        grid_2d.shape[1],
        axis=1,
    )

    intrinsic_concentration = 1.0e16

    hole_values = (
        intrinsic_concentration
        * np.exp(
            -potential_values / voltage
        )
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
        values=potential_values,
    )

    holes = Field(
        name="hole_concentration",
        units="1/m^3",
        grid=grid_2d,
        values=hole_values,
    )

    current = (
        compute_hole_scharfetter_gummel_current_x(
            potential=potential,
            hole_concentration=holes,
            mobility=0.048,
            temperature=300.0,
        )
    )

    np.testing.assert_allclose(
        current.values,
        0.0,
        atol=1.0e-6,
    )


def test_edge_grid_geometry(
    grid_2d: Grid,
) -> None:
    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
    )

    electrons = Field.full(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=1.0e20,
    )

    current = (
        compute_electron_scharfetter_gummel_current_x(
            potential=potential,
            electron_concentration=electrons,
            mobility=0.135,
        )
    )

    assert current.grid.shape == (
        grid_2d.shape[0] - 1,
        grid_2d.shape[1],
    )

    assert current.grid.spacing == grid_2d.spacing

    assert current.grid.origin[0] == pytest.approx(
        grid_2d.origin[0]
        + 0.5 * grid_2d.spacing[0]
    )


def test_total_edge_current(
    grid_2d: Grid,
) -> None:
    edge_grid = Grid(
        shape=(
            grid_2d.shape[0] - 1,
            grid_2d.shape[1],
        ),
        spacing=grid_2d.spacing,
        origin=(
            grid_2d.origin[0]
            + 0.5 * grid_2d.spacing[0],
            grid_2d.origin[1],
        ),
    )

    electron_current = Field.full(
        name="electron_current_density_x_edges",
        units="A/m^2",
        grid=edge_grid,
        fill_value=3.0,
    )

    hole_current = Field.full(
        name="hole_current_density_x_edges",
        units="A/m^2",
        grid=edge_grid,
        fill_value=2.0,
    )

    total = compute_total_scharfetter_gummel_current_x(
        electron_current_density=electron_current,
        hole_current_density=hole_current,
    )

    np.testing.assert_allclose(
        total.values,
        5.0,
    )

# added 1 dimensional test for electrons
def test_one_dimensional_electron_current() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )

    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid,
    )

    electrons = Field.full(
        name="electron_concentration",
        units="1/m^3",
        grid=grid,
        fill_value=1.0e20,
    )

    current = (
        compute_electron_scharfetter_gummel_current_x(
            potential=potential,
            electron_concentration=electrons,
            mobility=0.135,
        )
    )

    assert current.grid.shape == (10,)

    np.testing.assert_allclose(
        current.values,
        0.0,
    )

# added 1 dimensional test for holes
def test_one_dimensional_hole_current() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )

    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid,
    )

    holes = Field.full(
        name="hole_concentration",
        units="1/m^3",
        grid=grid,
        fill_value=1.0e20,
    )

    current = (
        compute_hole_scharfetter_gummel_current_x(
            potential=potential,
            hole_concentration=holes,
            mobility=0.048,
        )
    )

    assert current.grid.shape == (10,)

    np.testing.assert_allclose(
        current.values,
        0.0,
    )