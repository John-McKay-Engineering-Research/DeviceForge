import numpy as np
import pytest

from deviceforge import Field, Grid
from deviceforge.physics import (
    BOLTZMANN_CONSTANT,
    SRHParameters,
    compute_net_generation_rate,
    compute_shockley_read_hall_rate,
)


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(6, 4),
        spacing=(1.0e-9, 1.0e-9),
    )


def test_default_srh_parameters() -> None:
    parameters = SRHParameters()

    assert parameters.electron_lifetime == pytest.approx(
        1.0e-6
    )

    assert parameters.hole_lifetime == pytest.approx(
        1.0e-6
    )

    assert (
        parameters.trap_energy_relative_to_intrinsic
        == pytest.approx(0.0)
    )


def test_midgap_trap_reference_concentrations() -> None:
    parameters = SRHParameters(
        intrinsic_concentration=1.0e16,
        trap_energy_relative_to_intrinsic=0.0,
    )

    assert (
        parameters.electron_reference_concentration
        == pytest.approx(1.0e16)
    )

    assert (
        parameters.hole_reference_concentration
        == pytest.approx(1.0e16)
    )


def test_nonzero_trap_energy_reference_concentrations() -> None:
    intrinsic = 1.0e16
    temperature = 300.0
    trap_energy = (
        0.1
        * 1.602_176_634e-19
    )

    parameters = SRHParameters(
        intrinsic_concentration=intrinsic,
        temperature=temperature,
        trap_energy_relative_to_intrinsic=trap_energy,
    )

    expected_exponent = (
        trap_energy
        / (
            BOLTZMANN_CONSTANT
            * temperature
        )
    )

    assert (
        parameters.electron_reference_concentration
        == pytest.approx(
            intrinsic
            * np.exp(expected_exponent)
        )
    )

    assert (
        parameters.hole_reference_concentration
        == pytest.approx(
            intrinsic
            * np.exp(-expected_exponent)
        )
    )


def test_equilibrium_mass_action_has_zero_recombination(
    grid_2d: Grid,
) -> None:
    intrinsic = 1.0e16

    electrons = Field.full(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=1.0e20,
    )

    holes = Field.full(
        name="hole_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=(
            intrinsic**2 / 1.0e20
        ),
    )

    parameters = SRHParameters(
        intrinsic_concentration=intrinsic,
    )

    rate = compute_shockley_read_hall_rate(
        electron_concentration=electrons,
        hole_concentration=holes,
        parameters=parameters,
    )

    np.testing.assert_allclose(
        rate.values,
        0.0,
        atol=1.0e-12,
    )


def test_excess_carriers_produce_positive_recombination(
    grid_2d: Grid,
) -> None:
    electrons = Field.full(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=2.0e16,
    )

    holes = Field.full(
        name="hole_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=2.0e16,
    )

    rate = compute_shockley_read_hall_rate(
        electron_concentration=electrons,
        hole_concentration=holes,
        parameters=SRHParameters(
            intrinsic_concentration=1.0e16,
        ),
    )

    assert np.all(rate.values > 0.0)


def test_carrier_deficit_produces_net_generation(
    grid_2d: Grid,
) -> None:
    electrons = Field.full(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=0.5e16,
    )

    holes = Field.full(
        name="hole_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=0.5e16,
    )

    rate = compute_shockley_read_hall_rate(
        electron_concentration=electrons,
        hole_concentration=holes,
        parameters=SRHParameters(
            intrinsic_concentration=1.0e16,
        ),
    )

    assert np.all(rate.values < 0.0)


def test_symmetric_midgap_analytical_rate(
    grid_2d: Grid,
) -> None:
    intrinsic = 1.0e16
    carrier_density = 2.0e16
    lifetime = 1.0e-6

    electrons = Field.full(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=carrier_density,
    )

    holes = Field.full(
        name="hole_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=carrier_density,
    )

    parameters = SRHParameters(
        electron_lifetime=lifetime,
        hole_lifetime=lifetime,
        intrinsic_concentration=intrinsic,
        trap_energy_relative_to_intrinsic=0.0,
    )

    result = compute_shockley_read_hall_rate(
        electron_concentration=electrons,
        hole_concentration=holes,
        parameters=parameters,
    )

    expected = (
        carrier_density**2
        - intrinsic**2
    ) / (
        lifetime
        * (carrier_density + intrinsic)
        + lifetime
        * (carrier_density + intrinsic)
    )

    np.testing.assert_allclose(
        result.values,
        expected,
    )


def test_recombination_metadata(
    grid_2d: Grid,
) -> None:
    electrons = Field.full(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=2.0e16,
    )

    holes = Field.full(
        name="hole_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=2.0e16,
    )

    result = compute_shockley_read_hall_rate(
        electron_concentration=electrons,
        hole_concentration=holes,
    )

    assert result.name == (
        "shockley_read_hall_recombination_rate"
    )

    assert result.units == "1/(m^3 s)"
    assert result.grid is grid_2d


def test_net_generation_is_negative_recombination(
    grid_2d: Grid,
) -> None:
    electrons = Field.full(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=2.0e16,
    )

    holes = Field.full(
        name="hole_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=2.0e16,
    )

    recombination = compute_shockley_read_hall_rate(
        electron_concentration=electrons,
        hole_concentration=holes,
    )

    generation = compute_net_generation_rate(
        recombination
    )

    np.testing.assert_allclose(
        generation.values,
        -recombination.values,
    )


@pytest.mark.parametrize(
    "electron_lifetime",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_electron_lifetime_is_rejected(
    electron_lifetime: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Electron lifetime",
    ):
        SRHParameters(
            electron_lifetime=electron_lifetime
        )


@pytest.mark.parametrize(
    "hole_lifetime",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_hole_lifetime_is_rejected(
    hole_lifetime: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Hole lifetime",
    ):
        SRHParameters(
            hole_lifetime=hole_lifetime
        )


def test_different_carrier_grids_are_rejected(
    grid_2d: Grid,
) -> None:
    other_grid = Grid(
        shape=(5, 4),
        spacing=(1.0e-9, 1.0e-9),
    )

    electrons = Field.full(
        name="electron_concentration",
        units="1/m^3",
        grid=grid_2d,
        fill_value=1.0e16,
    )

    holes = Field.full(
        name="hole_concentration",
        units="1/m^3",
        grid=other_grid,
        fill_value=1.0e16,
    )

    with pytest.raises(
        ValueError,
        match="same grid",
    ):
        compute_shockley_read_hall_rate(
            electron_concentration=electrons,
            hole_concentration=holes,
        )