from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deviceforge import Field, Grid
from deviceforge.postprocessing import (
    calculate_electrostatic_fields_2d,
)
from deviceforge.visualisation import (
    plot_electric_field_magnitude_2d,
    plot_electric_field_vectors_2d,
    plot_electrostatic_potential_2d,
    plot_electrostatic_solution_2d,
    plot_equipotential_contours_2d,
)


@pytest.fixture
def potential_2d() -> Field:
    """Return a simple linear two-dimensional potential field."""

    grid = Grid(
        shape=(11, 9),
        spacing=(
            1.0e-9,
            2.0e-9,
        ),
        origin=(
            0.0,
            0.0,
        ),
    )

    coordinate_axis_0 = (
        grid.coordinates(0)
    )

    coordinate_axis_1 = (
        grid.coordinates(1)
    )

    values = (
        coordinate_axis_0[:, None]
        + 0.5
        * coordinate_axis_1[None, :]
    )

    return Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=values,
    )


@pytest.fixture
def electric_fields_2d(
    potential_2d: Field,
) -> tuple[Field, Field, Field]:
    """Return derived electric-field quantities."""

    return calculate_electrostatic_fields_2d(
        potential_2d
    )


def test_plot_potential_returns_figure_and_axis(
    potential_2d: Field,
) -> None:
    figure, axis = (
        plot_electrostatic_potential_2d(
            potential_2d
        )
    )

    assert isinstance(
        figure,
        Figure,
    )

    assert isinstance(
        axis,
        Axes,
    )

    assert axis.get_title() == (
        "Electrostatic Potential"
    )

    plt.close(figure)


def test_plot_potential_uses_supplied_axis(
    potential_2d: Field,
) -> None:
    figure, supplied_axis = (
        plt.subplots()
    )

    returned_figure, returned_axis = (
        plot_electrostatic_potential_2d(
            potential_2d,
            axis=supplied_axis,
        )
    )

    assert returned_figure is figure
    assert returned_axis is supplied_axis

    plt.close(figure)


def test_plot_equipotential_contours(
    potential_2d: Field,
) -> None:
    figure, axis = (
        plot_equipotential_contours_2d(
            potential_2d,
            number_of_levels=8,
        )
    )

    assert isinstance(
        figure,
        Figure,
    )

    assert axis.get_title() == (
        "Electrostatic Potential "
        "and Equipotentials"
    )

    plt.close(figure)


@pytest.mark.parametrize(
    "number_of_levels",
    [
        0,
        1,
        -5,
    ],
)
def test_equipotential_plot_rejects_too_few_levels(
    potential_2d: Field,
    number_of_levels: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        plot_equipotential_contours_2d(
            potential_2d,
            number_of_levels=number_of_levels,
        )


def test_plot_field_magnitude(
    electric_fields_2d: tuple[
        Field,
        Field,
        Field,
    ],
) -> None:
    _, _, magnitude = (
        electric_fields_2d
    )

    figure, axis = (
        plot_electric_field_magnitude_2d(
            magnitude
        )
    )

    assert isinstance(
        figure,
        Figure,
    )

    assert axis.get_title() == (
        "Electric-Field Magnitude"
    )

    plt.close(figure)


def test_plot_field_vectors(
    electric_fields_2d: tuple[
        Field,
        Field,
        Field,
    ],
) -> None:
    axis_0, axis_1, _ = (
        electric_fields_2d
    )

    figure, axis = (
        plot_electric_field_vectors_2d(
            axis_0,
            axis_1,
            stride=2,
        )
    )

    assert isinstance(
        figure,
        Figure,
    )

    assert axis.get_title() == (
        "Electric-Field Vectors"
    )

    plt.close(figure)


def test_plot_normalised_field_vectors(
    electric_fields_2d: tuple[
        Field,
        Field,
        Field,
    ],
) -> None:
    axis_0, axis_1, _ = (
        electric_fields_2d
    )

    figure, _ = (
        plot_electric_field_vectors_2d(
            axis_0,
            axis_1,
            normalise=True,
        )
    )

    plt.close(figure)


@pytest.mark.parametrize(
    "stride",
    [
        0,
        -1,
    ],
)
def test_vector_plot_rejects_invalid_stride(
    electric_fields_2d: tuple[
        Field,
        Field,
        Field,
    ],
    stride: int,
) -> None:
    axis_0, axis_1, _ = (
        electric_fields_2d
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        plot_electric_field_vectors_2d(
            axis_0,
            axis_1,
            stride=stride,
        )


def test_plot_summary_returns_four_axes(
    potential_2d: Field,
    electric_fields_2d: tuple[
        Field,
        Field,
        Field,
    ],
) -> None:
    axis_0, axis_1, magnitude = (
        electric_fields_2d
    )

    figure, axes = (
        plot_electrostatic_solution_2d(
            potential_2d,
            axis_0,
            axis_1,
            magnitude,
            vector_stride=2,
        )
    )

    assert isinstance(
        figure,
        Figure,
    )

    assert len(axes) == 4

    for axis in axes:
        assert isinstance(
            axis,
            Axes,
        )

    plt.close(figure)


def test_visualisation_rejects_one_dimensional_potential() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )

    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid,
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        plot_electrostatic_potential_2d(
            potential
        )


def test_visualisation_rejects_wrong_potential_units() -> None:
    grid = Grid(
        shape=(5, 5),
        spacing=(
            1.0e-9,
            1.0e-9,
        ),
    )

    invalid = Field.zeros(
        name="invalid",
        units="A",
        grid=grid,
    )

    with pytest.raises(
        ValueError,
        match="units must be 'V'",
    ):
        plot_electrostatic_potential_2d(
            invalid
        )