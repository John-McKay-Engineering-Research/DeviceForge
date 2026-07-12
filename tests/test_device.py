import numpy as np
import pytest

from deviceforge import Device, Grid, Region
from deviceforge.physics import SILICON, SILICON_DIOXIDE


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(6, 4),
        spacing=(1.0e-9, 1.0e-9),
    )


@pytest.fixture
def two_region_device(grid_2d: Grid) -> Device:
    left_mask = np.zeros(grid_2d.shape, dtype=bool)
    left_mask[:3, :] = True

    right_mask = np.zeros(grid_2d.shape, dtype=bool)
    right_mask[3:, :] = True

    p_region = Region(
        name="p_region",
        grid=grid_2d,
        material=SILICON,
        mask=left_mask,
        acceptor_density=1.0e23,
    )

    n_region = Region(
        name="n_region",
        grid=grid_2d,
        material=SILICON,
        mask=right_mask,
        donor_density=2.0e23,
    )

    return Device(
        name="pn_junction",
        grid=grid_2d,
        regions=(p_region, n_region),
    )


def test_create_device(two_region_device: Device) -> None:
    assert two_region_device.name == "pn_junction"
    assert two_region_device.number_of_regions == 2
    assert two_region_device.covered_fraction == pytest.approx(1.0)


def test_get_region(two_region_device: Device) -> None:
    region = two_region_device.get_region("p_region")

    assert region.name == "p_region"
    assert region.is_p_type


def test_missing_region_raises_key_error(
    two_region_device: Device,
) -> None:
    with pytest.raises(KeyError, match="missing"):
        two_region_device.get_region("missing")


def test_permittivity_field(two_region_device: Device) -> None:
    field = two_region_device.relative_permittivity_field()

    assert field.shape == two_region_device.grid.shape
    assert field.minimum == pytest.approx(11.7)
    assert field.maximum == pytest.approx(11.7)


def test_net_doping_field(two_region_device: Device) -> None:
    field = two_region_device.net_doping_field()

    np.testing.assert_allclose(field.values[:3, :], -1.0e23)
    np.testing.assert_allclose(field.values[3:, :], 2.0e23)


def test_material_id_field(two_region_device: Device) -> None:
    field = two_region_device.material_id_field()

    assert np.all(field.values[:3, :] == 0.0)
    assert np.all(field.values[3:, :] == 1.0)


def test_duplicate_region_names_raise_value_error(
    grid_2d: Grid,
) -> None:
    left_mask = np.zeros(grid_2d.shape, dtype=bool)
    left_mask[:3, :] = True

    right_mask = np.zeros(grid_2d.shape, dtype=bool)
    right_mask[3:, :] = True

    region_a = Region(
        name="duplicate",
        grid=grid_2d,
        material=SILICON,
        mask=left_mask,
    )

    region_b = Region(
        name="duplicate",
        grid=grid_2d,
        material=SILICON_DIOXIDE,
        mask=right_mask,
    )

    with pytest.raises(ValueError, match="unique"):
        Device(
            name="invalid",
            grid=grid_2d,
            regions=(region_a, region_b),
        )


def test_overlapping_regions_raise_value_error(
    grid_2d: Grid,
) -> None:
    mask_a = np.zeros(grid_2d.shape, dtype=bool)
    mask_a[:4, :] = True

    mask_b = np.zeros(grid_2d.shape, dtype=bool)
    mask_b[2:, :] = True

    region_a = Region(
        name="region_a",
        grid=grid_2d,
        material=SILICON,
        mask=mask_a,
    )

    region_b = Region(
        name="region_b",
        grid=grid_2d,
        material=SILICON_DIOXIDE,
        mask=mask_b,
    )

    with pytest.raises(ValueError, match="overlap"):
        Device(
            name="invalid",
            grid=grid_2d,
            regions=(region_a, region_b),
        )


def test_uncovered_points_raise_value_error(
    grid_2d: Grid,
) -> None:
    partial_mask = np.zeros(grid_2d.shape, dtype=bool)
    partial_mask[:3, :] = True

    region = Region(
        name="partial",
        grid=grid_2d,
        material=SILICON,
        mask=partial_mask,
    )

    with pytest.raises(ValueError, match="uncovered"):
        Device(
            name="invalid",
            grid=grid_2d,
            regions=(region,),
        )


def test_partial_coverage_can_be_allowed(
    grid_2d: Grid,
) -> None:
    partial_mask = np.zeros(grid_2d.shape, dtype=bool)
    partial_mask[:3, :] = True

    region = Region(
        name="partial",
        grid=grid_2d,
        material=SILICON,
        mask=partial_mask,
    )

    device = Device(
        name="partial_device",
        grid=grid_2d,
        regions=(region,),
        require_full_coverage=False,
    )

    assert device.covered_fraction == pytest.approx(0.5)