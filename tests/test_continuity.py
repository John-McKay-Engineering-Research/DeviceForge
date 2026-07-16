import numpy as np
import pytest

from deviceforge import Field, Grid
from deviceforge.physics import (
    ELEMENTARY_CHARGE,
    compute_current_divergence_x,
    compute_electron_continuity_residual,
    compute_hole_continuity_residual,
    maximum_absolute_continuity_residual,
)


@pytest.fixture
def edge_grid() -> Grid:
    """
    Represent x-directed edge centres for an original 6 x 4 node grid.
    """

    return Grid(
        shape=(5, 4),
        spacing=(1.0e-9, 1.0e-9),
        origin=(0.5e-9, 0.0),
    )


def test_constant_current_has_zero_divergence(
    edge_grid: Grid,
) -> None:
    current = Field.full(
        name="constant_edge_current",
        units="A/m^2",
        grid=edge_grid,
        fill_value=5.0,
    )

    divergence = compute_current_divergence_x(
        current
    )

    np.testing.assert_allclose(
        divergence.values,
        0.0,
    )


def test_linear_current_has_constant_divergence(
    edge_grid: Grid,
) -> None:
    x_coordinates = edge_grid.coordinates(
        axis=0
    )

    gradient = 2.0e9

    current_profile = (
        3.0
        + gradient * x_coordinates
    )

    values = np.repeat(
        current_profile[:, np.newaxis],
        edge_grid.shape[1],
        axis=1,
    )

    current = Field(
        name="linear_edge_current",
        units="A/m^2",
        grid=edge_grid,
        values=values,
    )

    divergence = compute_current_divergence_x(
        current
    )

    np.testing.assert_allclose(
        divergence.values,
        gradient,
        rtol=1.0e-12,
    )


def test_divergence_metadata(
    edge_grid: Grid,
) -> None:
    current = Field.full(
        name="edge_current",
        units="A/m^2",
        grid=edge_grid,
        fill_value=1.0,
    )

    divergence = compute_current_divergence_x(
        current
    )

    assert divergence.name == (
        "current_density_divergence_x"
    )

    assert divergence.units == "A/m^3"

    assert divergence.grid.shape == (
        edge_grid.shape[0] - 1,
        edge_grid.shape[1],
    )

    assert divergence.grid.origin[0] == pytest.approx(
        1.0e-9
    )


def test_electron_continuity_residual_sign(
    edge_grid: Grid,
) -> None:
    x_coordinates = edge_grid.coordinates(
        axis=0
    )

    gradient = 4.0e9

    values = np.repeat(
        (
            gradient
            * x_coordinates
        )[:, np.newaxis],
        edge_grid.shape[1],
        axis=1,
    )

    current = Field(
        name="electron_current_density_x_edges",
        units="A/m^2",
        grid=edge_grid,
        values=values,
    )

    residual = compute_electron_continuity_residual(
        current
    )

    expected = gradient / ELEMENTARY_CHARGE

    np.testing.assert_allclose(
        residual.values,
        expected,
        rtol=1.0e-12,
    )


def test_hole_continuity_residual_has_opposite_sign(
    edge_grid: Grid,
) -> None:
    x_coordinates = edge_grid.coordinates(
        axis=0
    )

    gradient = 4.0e9

    values = np.repeat(
        (
            gradient
            * x_coordinates
        )[:, np.newaxis],
        edge_grid.shape[1],
        axis=1,
    )

    current = Field(
        name="hole_current_density_x_edges",
        units="A/m^2",
        grid=edge_grid,
        values=values,
    )

    residual = compute_hole_continuity_residual(
        current
    )

    expected = (
        -gradient
        / ELEMENTARY_CHARGE
    )

    np.testing.assert_allclose(
        residual.values,
        expected,
        rtol=1.0e-12,
    )


def test_constant_electron_current_satisfies_continuity(
    edge_grid: Grid,
) -> None:
    current = Field.full(
        name="electron_current_density_x_edges",
        units="A/m^2",
        grid=edge_grid,
        fill_value=1.0e5,
    )

    residual = compute_electron_continuity_residual(
        current
    )

    np.testing.assert_allclose(
        residual.values,
        0.0,
    )


def test_constant_hole_current_satisfies_continuity(
    edge_grid: Grid,
) -> None:
    current = Field.full(
        name="hole_current_density_x_edges",
        units="A/m^2",
        grid=edge_grid,
        fill_value=-2.0e5,
    )

    residual = compute_hole_continuity_residual(
        current
    )

    np.testing.assert_allclose(
        residual.values,
        0.0,
    )


def test_maximum_absolute_residual(
    edge_grid: Grid,
) -> None:
    interior_grid = Grid(
        shape=(
            edge_grid.shape[0] - 1,
            edge_grid.shape[1],
        ),
        spacing=edge_grid.spacing,
        origin=(
            edge_grid.origin[0]
            + 0.5 * edge_grid.spacing[0],
            edge_grid.origin[1],
        ),
    )

    residual = Field(
        name="electron_continuity_residual",
        units="1/(m^3 s)",
        grid=interior_grid,
        values=np.array(
            [
                [1.0, -2.0, 3.0, -4.0],
                [5.0, -6.0, 7.0, -8.0],
                [9.0, -10.0, 11.0, -12.0],
                [13.0, -14.0, 15.0, -16.0],
            ]
        ),
    )

    assert maximum_absolute_continuity_residual(
        residual
    ) == pytest.approx(16.0)


def test_invalid_current_units_are_rejected(
    edge_grid: Grid,
) -> None:
    current = Field.full(
        name="invalid_current",
        units="V/m",
        grid=edge_grid,
        fill_value=1.0,
    )

    with pytest.raises(
        ValueError,
        match="A/m",
    ):
        compute_current_divergence_x(current)

# removed as field already gaurantees that every field contains finite values
# throws errors due to NAN values
"""
def test_non_finite_current_is_rejected(
    edge_grid: Grid,
) -> None:
    values = np.ones(edge_grid.shape)
    values[1, 1] = np.nan

    current = Field(
        name="invalid_current",
        units="A/m^2",
        grid=edge_grid,
        values=values,
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        compute_current_divergence_x(current)
"""