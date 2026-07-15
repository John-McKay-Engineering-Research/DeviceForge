import numpy as np
import pytest

from deviceforge import Field, Grid
from deviceforge.physics import (
    DEFAULT_SILICON_INTRINSIC_CONCENTRATION,
    charge_neutral_potential,
    compute_equilibrium_charge_density,
    equilibrium_carrier_concentrations,
    thermal_voltage,
)


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(5, 3),
        spacing=(1.0e-9, 1.0e-9),
    )


def test_thermal_voltage_at_room_temperature() -> None:
    voltage = thermal_voltage(300.0)

    assert voltage == pytest.approx(
        0.025852,
        rel=1.0e-4,
    )


def test_intrinsic_potential_produces_equal_carriers(
    grid_2d: Grid,
) -> None:
    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
    )

    electrons, holes = (
        equilibrium_carrier_concentrations(
            potential
        )
    )

    expected = (
        DEFAULT_SILICON_INTRINSIC_CONCENTRATION
    )

    np.testing.assert_allclose(
        electrons.values,
        expected,
    )

    np.testing.assert_allclose(
        holes.values,
        expected,
    )


def test_carriers_satisfy_mass_action_law(
    grid_2d: Grid,
) -> None:
    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
        values=np.linspace(
            -0.2,
            0.2,
            grid_2d.number_of_points,
        ).reshape(grid_2d.shape),
    )

    intrinsic = 1.0e16

    electrons, holes = (
        equilibrium_carrier_concentrations(
            potential,
            intrinsic_concentration=intrinsic,
        )
    )

    np.testing.assert_allclose(
        electrons.values * holes.values,
        intrinsic**2,
        rtol=1.0e-12,
    )


def test_positive_potential_increases_electrons(
    grid_2d: Grid,
) -> None:
    potential = Field.full(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
        fill_value=0.1,
    )

    electrons, holes = (
        equilibrium_carrier_concentrations(
            potential
        )
    )

    assert np.all(
        electrons.values
        > DEFAULT_SILICON_INTRINSIC_CONCENTRATION
    )

    assert np.all(
        holes.values
        < DEFAULT_SILICON_INTRINSIC_CONCENTRATION
    )


def test_negative_potential_increases_holes(
    grid_2d: Grid,
) -> None:
    potential = Field.full(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
        fill_value=-0.1,
    )

    electrons, holes = (
        equilibrium_carrier_concentrations(
            potential
        )
    )

    assert np.all(
        holes.values
        > DEFAULT_SILICON_INTRINSIC_CONCENTRATION
    )

    assert np.all(
        electrons.values
        < DEFAULT_SILICON_INTRINSIC_CONCENTRATION
    )


def test_equilibrium_charge_density(
    grid_2d: Grid,
) -> None:
    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
    )

    donor = Field.full(
        name="donor_density",
        units="1/m^3",
        grid=grid_2d,
        fill_value=2.0e21,
    )

    acceptor = Field.full(
        name="acceptor_density",
        units="1/m^3",
        grid=grid_2d,
        fill_value=1.0e21,
    )

    charge, electrons, holes = (
        compute_equilibrium_charge_density(
            potential=potential,
            donor_density=donor,
            acceptor_density=acceptor,
        )
    )

    assert charge.units == "C/m^3"
    assert electrons.units == "1/m^3"
    assert holes.units == "1/m^3"

    assert np.all(charge.values > 0.0)


@pytest.mark.parametrize(
    "net_doping",
    [
        -1.0e21,
        0.0,
        1.0e21,
    ],
)
def test_charge_neutral_potential_removes_local_charge(
    grid_2d: Grid,
    net_doping: float,
) -> None:
    neutral_potential = float(
        charge_neutral_potential(net_doping)
    )

    potential = Field.full(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
        fill_value=neutral_potential,
    )

    donor_value = max(net_doping, 0.0)
    acceptor_value = max(-net_doping, 0.0)

    donor = Field.full(
        name="donor_density",
        units="1/m^3",
        grid=grid_2d,
        fill_value=donor_value,
    )

    acceptor = Field.full(
        name="acceptor_density",
        units="1/m^3",
        grid=grid_2d,
        fill_value=acceptor_value,
    )

    charge, _, _ = compute_equilibrium_charge_density(
        potential=potential,
        donor_density=donor,
        acceptor_density=acceptor,
    )

    np.testing.assert_allclose(
        charge.values,
        0.0,
        atol=1.0e-10,
    )


def test_neutral_potential_has_expected_sign() -> None:
    p_type = float(
        charge_neutral_potential(-1.0e21)
    )

    intrinsic = float(
        charge_neutral_potential(0.0)
    )

    n_type = float(
        charge_neutral_potential(1.0e21)
    )

    assert p_type < 0.0
    assert intrinsic == pytest.approx(0.0)
    assert n_type > 0.0
    assert n_type == pytest.approx(-p_type)


@pytest.mark.parametrize(
    "temperature",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_temperature_raises_value_error(
    temperature: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Temperature",
    ):
        thermal_voltage(temperature)


@pytest.mark.parametrize(
    "intrinsic_concentration",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_intrinsic_concentration_is_rejected(
    grid_2d: Grid,
    intrinsic_concentration: float,
) -> None:
    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
    )

    with pytest.raises(
        ValueError,
        match="Intrinsic concentration",
    ):
        equilibrium_carrier_concentrations(
            potential,
            intrinsic_concentration=(
                intrinsic_concentration
            ),
        )