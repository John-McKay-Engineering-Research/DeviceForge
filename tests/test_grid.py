import numpy as np
import pytest

from deviceforge import Grid


def test_create_two_dimensional_grid() -> None:
    grid = Grid(
        shape=(200, 100),
        spacing=(1.0e-9, 2.0e-9),
    )

    assert grid.dimension == 2
    assert grid.number_of_points == 20_000
    assert grid.origin == (0.0, 0.0)
    assert grid.physical_size == pytest.approx(
        (199.0e-9, 198.0e-9)
    )


def test_create_three_dimensional_grid() -> None:
    grid = Grid(
        shape=(20, 10, 5),
        spacing=(1.0e-9, 2.0e-9, 3.0e-9),
    )

    assert grid.dimension == 3
    assert grid.number_of_points == 1_000
    assert grid.physical_size == pytest.approx(
        (19.0e-9, 18.0e-9, 12.0e-9)
    )


def test_custom_origin() -> None:
    grid = Grid(
        shape=(3, 4),
        spacing=(0.5, 0.25),
        origin=(1.0, -1.0),
    )

    assert grid.origin == (1.0, -1.0)
    assert grid.bounds[0] == pytest.approx((1.0, 2.0))
    assert grid.bounds[1] == pytest.approx((-1.0, -0.25))


def test_coordinates() -> None:
    grid = Grid(
        shape=(4, 3),
        spacing=(0.5, 2.0),
        origin=(1.0, -2.0),
    )

    x_coordinates = grid.coordinates(axis=0)
    y_coordinates = grid.coordinates(axis=1)

    np.testing.assert_allclose(
        x_coordinates,
        np.array([1.0, 1.5, 2.0, 2.5]),
    )

    np.testing.assert_allclose(
        y_coordinates,
        np.array([-2.0, 0.0, 2.0]),
    )


def test_mesh_matches_grid_shape() -> None:
    grid = Grid(
        shape=(4, 3),
        spacing=(0.5, 2.0),
    )

    x_coordinates, y_coordinates = grid.mesh()

    assert x_coordinates.shape == grid.shape
    assert y_coordinates.shape == grid.shape

    assert x_coordinates[0, 0] == pytest.approx(0.0)
    assert x_coordinates[-1, 0] == pytest.approx(1.5)

    assert y_coordinates[0, 0] == pytest.approx(0.0)
    assert y_coordinates[0, -1] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("shape", "spacing"),
    [
        ((100,), ()),
        ((100, 100), (1.0e-9,)),
        ((100, 100, 100, 100), (1.0e-9,) * 4),
    ],
)
def test_dimension_mismatch_raises_value_error(
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        Grid(
            shape=shape,
            spacing=spacing,
        )


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (100, 1),
        (0, 100),
        (-10, 100),
    ],
)
def test_invalid_shape_size_raises_value_error(
    shape: tuple[int, ...],
) -> None:
    spacing = tuple(1.0 for _ in shape)

    with pytest.raises(ValueError):
        Grid(
            shape=shape,
            spacing=spacing,
        )


@pytest.mark.parametrize(
    "spacing",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, -0.5),
    ],
)
def test_invalid_spacing_raises_value_error(
    spacing: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        Grid(
            shape=(100, 100),
            spacing=spacing,
        )


def test_invalid_axis_raises_index_error() -> None:
    grid = Grid(
        shape=(100, 100),
        spacing=(1.0, 1.0),
    )

    with pytest.raises(IndexError):
        grid.coordinates(axis=2)