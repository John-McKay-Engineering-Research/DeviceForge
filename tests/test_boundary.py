import numpy as np
import pytest

from deviceforge import (
    BoundaryCondition,
    BoundaryConditionType,
    Grid,
)


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(6, 4),
        spacing=(1.0e-9, 1.0e-9),
    )


@pytest.fixture
def left_boundary_mask(grid_2d: Grid) -> np.ndarray:
    mask = np.zeros(grid_2d.shape, dtype=bool)
    mask[0, :] = True
    return mask


def test_create_dirichlet_boundary(
    grid_2d: Grid,
    left_boundary_mask: np.ndarray,
) -> None:
    boundary = BoundaryCondition(
        name="left_contact",
        grid=grid_2d,
        mask=left_boundary_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    assert boundary.name == "left_contact"
    assert boundary.grid is grid_2d
    assert boundary.condition_type is BoundaryConditionType.DIRICHLET
    assert boundary.is_dirichlet
    assert not boundary.is_neumann
    assert boundary.value == pytest.approx(0.0)
    assert boundary.units == "V"


def test_create_neumann_boundary(
    grid_2d: Grid,
) -> None:
    mask = np.zeros(grid_2d.shape, dtype=bool)
    mask[:, -1] = True

    boundary = BoundaryCondition(
        name="top_insulated",
        grid=grid_2d,
        mask=mask,
        condition_type=BoundaryConditionType.NEUMANN,
        value=0.0,
        units="V/m",
    )

    assert boundary.condition_type is BoundaryConditionType.NEUMANN
    assert boundary.is_neumann
    assert not boundary.is_dirichlet


def test_condition_type_is_normalised(
    grid_2d: Grid,
    left_boundary_mask: np.ndarray,
) -> None:
    boundary = BoundaryCondition(
        name="left_contact",
        grid=grid_2d,
        mask=left_boundary_mask,
        condition_type="  DIRICHLET  ",
        value=1.0,
        units="V",
    )

    assert boundary.condition_type is BoundaryConditionType.DIRICHLET


def test_boundary_point_count(
    grid_2d: Grid,
    left_boundary_mask: np.ndarray,
) -> None:
    boundary = BoundaryCondition(
        name="left_contact",
        grid=grid_2d,
        mask=left_boundary_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    assert boundary.number_of_points == 4


def test_contains_index(
    grid_2d: Grid,
    left_boundary_mask: np.ndarray,
) -> None:
    boundary = BoundaryCondition(
        name="left_contact",
        grid=grid_2d,
        mask=left_boundary_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    assert boundary.contains_index((0, 0))
    assert boundary.contains_index((0, 3))
    assert not boundary.contains_index((1, 0))


def test_integer_mask_is_converted_to_boolean(
    grid_2d: Grid,
) -> None:
    mask = np.zeros(grid_2d.shape, dtype=np.int32)
    mask[-1, :] = 1

    boundary = BoundaryCondition(
        name="right_contact",
        grid=grid_2d,
        mask=mask,
        condition_type="dirichlet",
        value=1.0,
        units="V",
    )

    assert boundary.mask.dtype == np.bool_


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
    left_boundary_mask: np.ndarray,
    name: str,
) -> None:
    with pytest.raises(ValueError, match="name"):
        BoundaryCondition(
            name=name,
            grid=grid_2d,
            mask=left_boundary_mask,
            condition_type="dirichlet",
            value=0.0,
            units="V",
        )


@pytest.mark.parametrize(
    "units",
    [
        "",
        " ",
        "\t",
    ],
)
def test_empty_units_raise_value_error(
    grid_2d: Grid,
    left_boundary_mask: np.ndarray,
    units: str,
) -> None:
    with pytest.raises(ValueError, match="units"):
        BoundaryCondition(
            name="left_contact",
            grid=grid_2d,
            mask=left_boundary_mask,
            condition_type="dirichlet",
            value=0.0,
            units=units,
        )


@pytest.mark.parametrize(
    "condition_type",
    [
        "",
        "fixed",
        "periodic",
        "unknown",
    ],
)
def test_invalid_condition_type_raises_value_error(
    grid_2d: Grid,
    left_boundary_mask: np.ndarray,
    condition_type: str,
) -> None:
    with pytest.raises(ValueError, match="type"):
        BoundaryCondition(
            name="invalid",
            grid=grid_2d,
            mask=left_boundary_mask,
            condition_type=condition_type,
            value=0.0,
            units="V",
        )


@pytest.mark.parametrize(
    "value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_value_raises_value_error(
    grid_2d: Grid,
    left_boundary_mask: np.ndarray,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="finite"):
        BoundaryCondition(
            name="invalid",
            grid=grid_2d,
            mask=left_boundary_mask,
            condition_type="dirichlet",
            value=value,
            units="V",
        )


def test_mask_shape_mismatch_raises_value_error(
    grid_2d: Grid,
) -> None:
    invalid_mask = np.zeros((5, 4), dtype=bool)

    with pytest.raises(ValueError, match="same shape"):
        BoundaryCondition(
            name="invalid",
            grid=grid_2d,
            mask=invalid_mask,
            condition_type="dirichlet",
            value=0.0,
            units="V",
        )


def test_empty_mask_raises_value_error(
    grid_2d: Grid,
) -> None:
    empty_mask = np.zeros(grid_2d.shape, dtype=bool)

    with pytest.raises(ValueError, match="at least one"):
        BoundaryCondition(
            name="invalid",
            grid=grid_2d,
            mask=empty_mask,
            condition_type="dirichlet",
            value=0.0,
            units="V",
        )


def test_interior_point_raises_value_error(
    grid_2d: Grid,
) -> None:
    mask = np.zeros(grid_2d.shape, dtype=bool)
    mask[2, 2] = True

    with pytest.raises(ValueError, match="outer boundary"):
        BoundaryCondition(
            name="invalid",
            grid=grid_2d,
            mask=mask,
            condition_type="dirichlet",
            value=0.0,
            units="V",
        )


def test_mixed_outer_and_interior_points_raise_value_error(
    grid_2d: Grid,
) -> None:
    mask = np.zeros(grid_2d.shape, dtype=bool)
    mask[0, 0] = True
    mask[2, 2] = True

    with pytest.raises(ValueError, match="outer boundary"):
        BoundaryCondition(
            name="invalid",
            grid=grid_2d,
            mask=mask,
            condition_type="dirichlet",
            value=0.0,
            units="V",
        )


def test_wrong_index_dimension_raises_value_error(
    grid_2d: Grid,
    left_boundary_mask: np.ndarray,
) -> None:
    boundary = BoundaryCondition(
        name="left_contact",
        grid=grid_2d,
        mask=left_boundary_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    with pytest.raises(ValueError, match="dimensionality"):
        boundary.contains_index((0,))


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
    left_boundary_mask: np.ndarray,
    index: tuple[int, int],
) -> None:
    boundary = BoundaryCondition(
        name="left_contact",
        grid=grid_2d,
        mask=left_boundary_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    with pytest.raises(IndexError, match="outside"):
        boundary.contains_index(index)


# additional boundary tests

def test_boundary_accepts_spatial_value_array(
    grid_2d,
) -> None:
    mask = np.zeros(
        grid_2d.shape,
        dtype=np.bool_,
    )
    mask[:, -1] = True

    values = np.zeros(
        grid_2d.shape,
        dtype=np.float64,
    )
    values[:, -1] = np.linspace(
        0.0,
        1.0,
        grid_2d.shape[0],
    )

    boundary = BoundaryCondition(
        name="top_profile",
        grid=grid_2d,
        mask=mask,
        condition_type="dirichlet",
        value=values,
        units="V",
    )

    assert boundary.is_spatially_varying

    np.testing.assert_allclose(
        boundary.values_on_mask(),
        values[mask],
    )

def test_boundary_rejects_spatial_value_shape_mismatch(
    grid_2d,
) -> None:
    mask = np.zeros(
        grid_2d.shape,
        dtype=np.bool_,
    )
    mask[:, -1] = True

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        BoundaryCondition(
            name="invalid_profile",
            grid=grid_2d,
            mask=mask,
            condition_type="dirichlet",
            value=np.zeros((3, 3)),
            units="V",
        )


def test_boundary_rejects_non_finite_spatial_values(
    grid_2d,
) -> None:
    mask = np.zeros(
        grid_2d.shape,
        dtype=np.bool_,
    )
    mask[:, -1] = True

    values = np.zeros(
        grid_2d.shape,
        dtype=np.float64,
    )
    values[0, -1] = np.nan

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        BoundaryCondition(
            name="invalid_profile",
            grid=grid_2d,
            mask=mask,
            condition_type="dirichlet",
            value=values,
            units="V",
        )

def test_spatial_boundary_values_are_immutable(
    grid_2d,
) -> None:
    mask = np.zeros(
        grid_2d.shape,
        dtype=np.bool_,
    )
    mask[:, -1] = True

    values = np.zeros(
        grid_2d.shape,
        dtype=np.float64,
    )

    boundary = BoundaryCondition(
        name="top_profile",
        grid=grid_2d,
        mask=mask,
        condition_type="dirichlet",
        value=values,
        units="V",
    )

    with pytest.raises(ValueError):
        boundary.value[0, -1] = 1.0

def test_scalar_values_on_mask_repeat_scalar(
    grid_2d,
) -> None:
    mask = np.zeros(
        grid_2d.shape,
        dtype=np.bool_,
    )
    mask[0, :] = True

    boundary = BoundaryCondition(
        name="left_contact",
        grid=grid_2d,
        mask=mask,
        condition_type="dirichlet",
        value=0.25,
        units="V",
    )

    np.testing.assert_allclose(
        boundary.values_on_mask(),
        0.25,
    )

    assert boundary.values_on_mask().shape == (
        boundary.number_of_points,
    )