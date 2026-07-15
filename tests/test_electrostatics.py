import numpy as np
import pytest

from deviceforge import Device, Field, Grid, Region
from deviceforge.physics import (
    ELEMENTARY_CHARGE,
    SILICON,
    SILICON_DIOXIDE,
    VACUUM_PERMITTIVITY,
    compute_absolute_permittivity,
    compute_electrostatic_source_term,
    compute_fixed_charge_density,
)


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(6, 4),
        spacing=(1.0e-9, 1.0e-9),
    )


@pytest.fixture
def pn_device(grid_2d: Grid) -> Device:
    p_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    p_mask[:3, :] = True

    n_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    n_mask[3:, :] = True

    p_region = Region(
        name="p_region",
        grid=grid_2d,
        material=SILICON,
        mask=p_mask,
        acceptor_density=1.0e23,
    )

    n_region = Region(
        name="n_region",
        grid=grid_2d,
        material=SILICON,
        mask=n_mask,
        donor_density=2.0e23,
    )

    return Device(
        name="pn_junction",
        grid=grid_2d,
        regions=(
            p_region,
            n_region,
        ),
    )


def test_fixed_charge_density_metadata(
    pn_device: Device,
) -> None:
    charge_density = compute_fixed_charge_density(
        pn_device
    )

    assert charge_density.name == (
        "fixed_charge_density"
    )
    assert charge_density.units == "C/m^3"
    assert charge_density.grid is pn_device.grid


def test_fixed_charge_density_values(
    pn_device: Device,
) -> None:
    charge_density = compute_fixed_charge_density(
        pn_device
    )

    expected_p_charge = (
        -ELEMENTARY_CHARGE * 1.0e23
    )

    expected_n_charge = (
        ELEMENTARY_CHARGE * 2.0e23
    )

    np.testing.assert_allclose(
        charge_density.values[:3, :],
        expected_p_charge,
    )

    np.testing.assert_allclose(
        charge_density.values[3:, :],
        expected_n_charge,
    )


def test_intrinsic_device_has_zero_charge(
    grid_2d: Grid,
) -> None:
    region = Region(
        name="intrinsic_silicon",
        grid=grid_2d,
        material=SILICON,
        mask=np.ones(
            grid_2d.shape,
            dtype=bool,
        ),
    )

    device = Device(
        name="intrinsic_device",
        grid=grid_2d,
        regions=(region,),
    )

    charge_density = compute_fixed_charge_density(
        device
    )

    np.testing.assert_allclose(
        charge_density.values,
        0.0,
    )


def test_absolute_permittivity_for_silicon(
    pn_device: Device,
) -> None:
    permittivity = compute_absolute_permittivity(
        pn_device
    )

    expected = (
        VACUUM_PERMITTIVITY
        * SILICON.relative_permittivity
    )

    assert permittivity.name == (
        "absolute_permittivity"
    )
    assert permittivity.units == "F/m"

    np.testing.assert_allclose(
        permittivity.values,
        expected,
    )


def test_multiple_material_permittivity(
    grid_2d: Grid,
) -> None:
    silicon_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    silicon_mask[:3, :] = True

    oxide_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    oxide_mask[3:, :] = True

    silicon_region = Region(
        name="silicon",
        grid=grid_2d,
        material=SILICON,
        mask=silicon_mask,
    )

    oxide_region = Region(
        name="oxide",
        grid=grid_2d,
        material=SILICON_DIOXIDE,
        mask=oxide_mask,
    )

    device = Device(
        name="silicon_oxide_device",
        grid=grid_2d,
        regions=(
            silicon_region,
            oxide_region,
        ),
    )

    permittivity = compute_absolute_permittivity(
        device
    )

    np.testing.assert_allclose(
        permittivity.values[:3, :],
        VACUUM_PERMITTIVITY
        * SILICON.relative_permittivity,
    )

    np.testing.assert_allclose(
        permittivity.values[3:, :],
        VACUUM_PERMITTIVITY
        * SILICON_DIOXIDE.relative_permittivity,
    )


def test_electrostatic_source_term(
    pn_device: Device,
) -> None:
    charge_density = compute_fixed_charge_density(
        pn_device
    )

    permittivity = compute_absolute_permittivity(
        pn_device
    )

    source = compute_electrostatic_source_term(
        charge_density,
        permittivity,
    )

    expected = (
        charge_density.values
        / permittivity.values
    )

    assert source.name == (
        "electrostatic_source_term"
    )
    assert source.units == "V/m^2"

    np.testing.assert_allclose(
        source.values,
        expected,
    )


def test_source_term_rejects_different_grids(
    grid_2d: Grid,
) -> None:
    other_grid = Grid(
        shape=(5, 4),
        spacing=(1.0e-9, 1.0e-9),
    )

    charge_density = Field.zeros(
        name="charge_density",
        units="C/m^3",
        grid=grid_2d,
    )

    permittivity = Field.full(
        name="permittivity",
        units="F/m",
        grid=other_grid,
        fill_value=1.0,
    )

    with pytest.raises(
        ValueError,
        match="same grid",
    ):
        compute_electrostatic_source_term(
            charge_density,
            permittivity,
        )


def test_source_term_rejects_charge_units(
    grid_2d: Grid,
) -> None:
    charge_density = Field.zeros(
        name="invalid_charge",
        units="V",
        grid=grid_2d,
    )

    permittivity = Field.full(
        name="permittivity",
        units="F/m",
        grid=grid_2d,
        fill_value=1.0,
    )

    with pytest.raises(
        ValueError,
        match="C/m",
    ):
        compute_electrostatic_source_term(
            charge_density,
            permittivity,
        )


def test_source_term_rejects_permittivity_units(
    grid_2d: Grid,
) -> None:
    charge_density = Field.zeros(
        name="charge_density",
        units="C/m^3",
        grid=grid_2d,
    )

    permittivity = Field.full(
        name="invalid_permittivity",
        units="dimensionless",
        grid=grid_2d,
        fill_value=11.7,
    )

    with pytest.raises(
        ValueError,
        match="F/m",
    ):
        compute_electrostatic_source_term(
            charge_density,
            permittivity,
        )


def test_source_term_rejects_nonpositive_permittivity(
    grid_2d: Grid,
) -> None:
    charge_density = Field.zeros(
        name="charge_density",
        units="C/m^3",
        grid=grid_2d,
    )

    permittivity = Field.zeros(
        name="permittivity",
        units="F/m",
        grid=grid_2d,
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        compute_electrostatic_source_term(
            charge_density,
            permittivity,
        )