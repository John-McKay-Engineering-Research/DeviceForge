import numpy as np
import pytest

from deviceforge import Grid, Region
from deviceforge.physics import SILICON, SILICON_DIOXIDE


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(6, 4),
        spacing=(1.0e-9, 1.0e-9),
    )


@pytest.fixture
def left_half_mask(grid_2d: Grid) -> np.ndarray:
    mask = np.zeros(grid_2d.shape, dtype=bool)
    mask[:3, :] = True
    return mask


def test_create_region(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
) -> None:
    region = Region(
        name="source",
        grid=grid_2d,
        material=SILICON,
        mask=left_half_mask,
        donor_density=1.0e23,
        region_type="semiconductor",
    )

    assert region.name == "source"
    assert region.grid is grid_2d
    assert region.material is SILICON
    assert region.region_type == "semiconductor"
    assert region.donor_density == pytest.approx(1.0e23)
    assert region.acceptor_density == pytest.approx(0.0)


def test_region_type_is_normalised(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
) -> None:
    region = Region(
        name="oxide",
        grid=grid_2d,
        material=SILICON_DIOXIDE,
        mask=left_half_mask,
        region_type="  DIELECTRIC  ",
    )

    assert region.region_type == "dielectric"


def test_region_point_count(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
) -> None:
    region = Region(
        name="left",
        grid=grid_2d,
        material=SILICON,
        mask=left_half_mask,
    )

    assert region.number_of_points == 12
    assert region.fraction_of_grid == pytest.approx(0.5)


def test_n_type_region(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
) -> None:
    region = Region(
        name="n_region",
        grid=grid_2d,
        material=SILICON,
        mask=left_half_mask,
        donor_density=2.0e23,
        acceptor_density=1.0e22,
    )

    assert region.net_doping_density == pytest.approx(1.9e23)
    assert region.is_n_type
    assert not region.is_p_type
    assert not region.is_intrinsic


def test_p_type_region(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
) -> None:
    region = Region(
        name="p_region",
        grid=grid_2d,
        material=SILICON,
        mask=left_half_mask,
        donor_density=1.0e22,
        acceptor_density=2.0e23,
    )

    assert region.net_doping_density == pytest.approx(-1.9e23)
    assert region.is_p_type
    assert not region.is_n_type
    assert not region.is_intrinsic


def test_intrinsic_region(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
) -> None:
    region = Region(
        name="intrinsic_region",
        grid=grid_2d,
        material=SILICON,
        mask=left_half_mask,
    )

    assert region.net_doping_density == pytest.approx(0.0)
    assert region.is_intrinsic


def test_contains_index(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
) -> None:
    region = Region(
        name="left",
        grid=grid_2d,
        material=SILICON,
        mask=left_half_mask,
    )

    assert region.contains_index((0, 0))
    assert region.contains_index((2, 3))
    assert not region.contains_index((3, 0))
    assert not region.contains_index((5, 3))


def test_integer_mask_is_converted_to_boolean(
    grid_2d: Grid,
) -> None:
    mask = np.zeros(grid_2d.shape, dtype=np.int32)
    mask[:2, :] = 1

    region = Region(
        name="converted_mask",
        grid=grid_2d,
        material=SILICON,
        mask=mask,
    )

    assert region.mask.dtype == np.bool_


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_empty_name_raises_value_error(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
    name: str,
) -> None:
    with pytest.raises(ValueError, match="name"):
        Region(
            name=name,
            grid=grid_2d,
            material=SILICON,
            mask=left_half_mask,
        )


def test_mask_shape_mismatch_raises_value_error(
    grid_2d: Grid,
) -> None:
    invalid_mask = np.zeros((5, 4), dtype=bool)

    with pytest.raises(ValueError, match="same shape"):
        Region(
            name="invalid",
            grid=grid_2d,
            material=SILICON,
            mask=invalid_mask,
        )


def test_empty_mask_raises_value_error(
    grid_2d: Grid,
) -> None:
    empty_mask = np.zeros(grid_2d.shape, dtype=bool)

    with pytest.raises(ValueError, match="at least one"):
        Region(
            name="empty",
            grid=grid_2d,
            material=SILICON,
            mask=empty_mask,
        )


@pytest.mark.parametrize(
    "donor_density",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_donor_density_raises_value_error(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
    donor_density: float,
) -> None:
    with pytest.raises(ValueError, match="Donor density"):
        Region(
            name="invalid",
            grid=grid_2d,
            material=SILICON,
            mask=left_half_mask,
            donor_density=donor_density,
        )


@pytest.mark.parametrize(
    "acceptor_density",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_acceptor_density_raises_value_error(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
    acceptor_density: float,
) -> None:
    with pytest.raises(ValueError, match="Acceptor density"):
        Region(
            name="invalid",
            grid=grid_2d,
            material=SILICON,
            mask=left_half_mask,
            acceptor_density=acceptor_density,
        )


def test_empty_region_type_raises_value_error(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="Region type"):
        Region(
            name="invalid",
            grid=grid_2d,
            material=SILICON,
            mask=left_half_mask,
            region_type="   ",
        )


def test_wrong_index_dimension_raises_value_error(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
) -> None:
    region = Region(
        name="left",
        grid=grid_2d,
        material=SILICON,
        mask=left_half_mask,
    )

    with pytest.raises(ValueError, match="dimensionality"):
        region.contains_index((0,))


@pytest.mark.parametrize(
    "index",
    [
        (-1, 0),
        (6, 0),
        (0, -1),
        (0, 4),
    ],
)
def test_out_of_bounds_index_raises_index_error(
    grid_2d: Grid,
    left_half_mask: np.ndarray,
    index: tuple[int, int],
) -> None:
    region = Region(
        name="left",
        grid=grid_2d,
        material=SILICON,
        mask=left_half_mask,
    )

    with pytest.raises(IndexError, match="outside"):
        region.contains_index(index)