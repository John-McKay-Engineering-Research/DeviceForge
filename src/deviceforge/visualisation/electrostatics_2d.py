from __future__ import annotations

from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.field import Field
from matplotlib.ticker import ScalarFormatter

PlotResult: TypeAlias = tuple[Figure, Axes]


def _validate_field_2d(
    field: Field,
    *,
    expected_units: str | None = None,
    label: str = "Field",
) -> None:
    """
    Validate a field used by a 2D visualisation function.
    """

    if not isinstance(field, Field):
        raise TypeError(
            f"{label} must be a Field instance."
        )

    if field.grid.dimension != 2:
        raise ValueError(
            f"{label} must use a two-dimensional grid."
        )

    if field.values.shape != field.grid.shape:
        raise ValueError(
            f"{label} values must match the grid shape."
        )

    if expected_units is not None:
        if field.units != expected_units:
            raise ValueError(
                f"{label} units must be "
                f"'{expected_units}'."
            )


def _coordinates_in_nanometres(
    field: Field,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the two grid-coordinate axes in nanometres.
    """

    axis_0_coordinates = (
        field.grid.coordinates(0)
        * 1.0e9
    )

    axis_1_coordinates = (
        field.grid.coordinates(1)
        * 1.0e9
    )

    return (
        axis_0_coordinates,
        axis_1_coordinates,
    )


def _create_coordinate_mesh(
    field: Field,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return coordinate meshes matching ``field.values``.

    ``indexing="ij"`` preserves DeviceForge's array convention:

        values[i, j]

    where axis 0 corresponds to the first coordinate array and axis 1
    corresponds to the second coordinate array.
    """

    (
        axis_0_coordinates,
        axis_1_coordinates,
    ) = _coordinates_in_nanometres(
        field
    )

    return np.meshgrid(
        axis_0_coordinates,
        axis_1_coordinates,
        indexing="ij",
    )


def plot_electrostatic_potential_2d(
    potential: Field,
    *,
    axis: Axes | None = None,
) -> PlotResult:
    """
    Plot a two-dimensional electrostatic-potential colour map.

    Parameters
    ----------
    potential:
        Two-dimensional electrostatic-potential field in volts.

    axis:
        Optional Matplotlib axis. A new figure and axis are created when
        omitted.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axis containing the plot.
    """

    _validate_field_2d(
        potential,
        expected_units="V",
        label="Potential",
    )

    if axis is None:
        figure, axis = plt.subplots()
    else:
        figure = axis.figure

    coordinate_axis_0, coordinate_axis_1 = (
        _create_coordinate_mesh(
            potential
        )
    )

    colour_map = axis.pcolormesh(
        coordinate_axis_0,
        coordinate_axis_1,
        potential.values,
        shading="auto",
    )

    figure.colorbar(
        colour_map,
        ax=axis,
        label="Potential (V)",
    )

    axis.set_xlabel(
        "Axis 0 position (nm)"
    )
    axis.set_ylabel(
        "Axis 1 position (nm)"
    )
    axis.set_title(
        "Electrostatic Potential"
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    figure.tight_layout()

    return figure, axis


def plot_equipotential_contours_2d(
    potential: Field,
    *,
    number_of_levels: int = 15,
    axis: Axes | None = None,
) -> PlotResult:
    """
    Plot potential as a colour map with equipotential contours.

    Parameters
    ----------
    potential:
        Two-dimensional electrostatic-potential field in volts.

    number_of_levels:
        Number of contour levels.

    axis:
        Optional Matplotlib axis.
    """

    _validate_field_2d(
        potential,
        expected_units="V",
        label="Potential",
    )

    if (
        isinstance(number_of_levels, bool)
        or not isinstance(
            number_of_levels,
            int,
        )
    ):
        raise TypeError(
            "Number of contour levels must be an integer."
        )

    if number_of_levels < 2:
        raise ValueError(
            "At least two contour levels are required."
        )

    if axis is None:
        figure, axis = plt.subplots()
    else:
        figure = axis.figure

    coordinate_axis_0, coordinate_axis_1 = (
        _create_coordinate_mesh(
            potential
        )
    )

    filled_contours = axis.contourf(
        coordinate_axis_0,
        coordinate_axis_1,
        potential.values,
        levels=number_of_levels,
    )

    contour_lines = axis.contour(
        coordinate_axis_0,
        coordinate_axis_1,
        potential.values,
        levels=number_of_levels,
    )

    axis.clabel(
        contour_lines,
        inline=True,
        fontsize=8,
        fmt="%.3g",
    )

    figure.colorbar(
        filled_contours,
        ax=axis,
        label="Potential (V)",
    )

    axis.set_xlabel(
        "Axis 0 position (nm)"
    )
    axis.set_ylabel(
        "Axis 1 position (nm)"
    )
    axis.set_title(
        "Electrostatic Potential and Equipotentials"
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    figure.tight_layout()

    return figure, axis

# updated this plot function due to matplotlib applying an offset
def plot_electric_field_magnitude_2d(
    electric_field_magnitude: Field,
    *,
    axis: Axes | None = None,
    uniform_relative_tolerance: float = 1.0e-10,
) -> PlotResult:
    """
    Plot a two-dimensional electric-field-magnitude colour map.

    Nearly uniform fields are displayed using a controlled colour range
    centred on the mean value. This prevents floating-point round-off from
    being visually exaggerated by Matplotlib's automatic normalisation.

    Parameters
    ----------
    electric_field_magnitude:
        Two-dimensional electric-field-magnitude field in V/m.

    axis:
        Optional Matplotlib axis. A new figure and axis are created when
        omitted.

    uniform_relative_tolerance:
        Maximum relative field variation for the field to be treated as
        effectively uniform.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axis containing the plot.
    """

    _validate_field_2d(
        electric_field_magnitude,
        expected_units="V/m",
        label="Electric-field magnitude",
    )

    if isinstance(
        uniform_relative_tolerance,
        bool,
    ) or not isinstance(
        uniform_relative_tolerance,
        (int, float),
    ):
        raise TypeError(
            "Uniform relative tolerance must be a real number."
        )

    uniform_relative_tolerance = float(
        uniform_relative_tolerance
    )

    if (
        not np.isfinite(
            uniform_relative_tolerance
        )
        or uniform_relative_tolerance < 0.0
    ):
        raise ValueError(
            "Uniform relative tolerance must be finite "
            "and non-negative."
        )

    if axis is None:
        figure, axis = plt.subplots()
    else:
        figure = axis.figure

    (
        coordinate_axis_0,
        coordinate_axis_1,
    ) = _create_coordinate_mesh(
        electric_field_magnitude
    )

    values = np.asarray(
        electric_field_magnitude.values,
        dtype=np.float64,
    )

    minimum_value = float(
        np.min(values)
    )

    maximum_value = float(
        np.max(values)
    )

    mean_value = float(
        np.mean(values)
    )

    absolute_variation = (
        maximum_value
        - minimum_value
    )

    reference_scale = max(
        abs(mean_value),
        np.finfo(np.float64).tiny,
    )

    relative_variation = (
        absolute_variation
        / reference_scale
    )

    effectively_uniform = (
        relative_variation
        <= uniform_relative_tolerance
    )

    if effectively_uniform:
        colour_half_range = max(
            abs(mean_value) * 1.0e-6,
            np.finfo(np.float64).eps,
        )

        colour_map = axis.pcolormesh(
            coordinate_axis_0,
            coordinate_axis_1,
            values,
            shading="auto",
            vmin=(
                mean_value
                - colour_half_range
            ),
            vmax=(
                mean_value
                + colour_half_range
            ),
        )

        axis.text(
            0.5,
            0.5,
            (
                "Effectively uniform field\n"
                f"|E| = {mean_value:.6e} V/m"
            ),
            transform=axis.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
            bbox={
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "none",
            },
        )
    else:
        colour_map = axis.pcolormesh(
            coordinate_axis_0,
            coordinate_axis_1,
            values,
            shading="auto",
        )

    colour_bar = figure.colorbar(
        colour_map,
        ax=axis,
        label=(
            "Electric-field magnitude (V/m)"
        ),
    )

    colour_bar.formatter.set_scientific(
        True
    )

    colour_bar.formatter.set_powerlimits(
        (0, 0)
    )

    colour_bar.update_ticks()

    axis.set_xlabel(
        "Axis 0 position (nm)"
    )

    axis.set_ylabel(
        "Axis 1 position (nm)"
    )

    axis.set_title(
        "Electric-Field Magnitude"
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    figure.tight_layout()

    return figure, axis


def plot_electric_field_vectors_2d(
    electric_field_axis_0: Field,
    electric_field_axis_1: Field,
    *,
    stride: int = 1,
    normalise: bool = False,
    axis: Axes | None = None,
) -> PlotResult:
    """
    Plot a two-dimensional electric-field vector map.

    Parameters
    ----------
    electric_field_axis_0:
        Electric-field component along grid axis 0.

    electric_field_axis_1:
        Electric-field component along grid axis 1.

    stride:
        Plot every ``stride`` grid points along both axes.

    normalise:
        If true, display vector direction with unit-length arrows.

    axis:
        Optional Matplotlib axis.
    """

    _validate_field_2d(
        electric_field_axis_0,
        expected_units="V/m",
        label="Axis-0 electric field",
    )

    _validate_field_2d(
        electric_field_axis_1,
        expected_units="V/m",
        label="Axis-1 electric field",
    )

    if (
        electric_field_axis_0.grid
        != electric_field_axis_1.grid
    ):
        raise ValueError(
            "Electric-field components must use the same grid."
        )

    if (
        isinstance(stride, bool)
        or not isinstance(stride, int)
    ):
        raise TypeError(
            "Vector stride must be an integer."
        )

    if stride <= 0:
        raise ValueError(
            "Vector stride must be positive."
        )

    if not isinstance(normalise, bool):
        raise TypeError(
            "Normalise must be a Boolean value."
        )

    if axis is None:
        figure, axis = plt.subplots()
    else:
        figure = axis.figure

    coordinate_axis_0, coordinate_axis_1 = (
        _create_coordinate_mesh(
            electric_field_axis_0
        )
    )

    component_axis_0 = np.asarray(
        electric_field_axis_0.values,
        dtype=np.float64,
    ).copy()

    component_axis_1 = np.asarray(
        electric_field_axis_1.values,
        dtype=np.float64,
    ).copy()

    if normalise:
        magnitude = np.hypot(
            component_axis_0,
            component_axis_1,
        )

        nonzero_mask = magnitude > 0.0

        component_axis_0[
            nonzero_mask
        ] /= magnitude[nonzero_mask]

        component_axis_1[
            nonzero_mask
        ] /= magnitude[nonzero_mask]

    index = (
        slice(None, None, stride),
        slice(None, None, stride),
    )

    axis.quiver(
        coordinate_axis_0[index],
        coordinate_axis_1[index],
        component_axis_0[index],
        component_axis_1[index],
    )

    axis.set_xlabel(
        "Axis 0 position (nm)"
    )
    axis.set_ylabel(
        "Axis 1 position (nm)"
    )
    axis.set_title(
        "Electric-Field Vectors"
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    figure.tight_layout()

    return figure, axis


def plot_electrostatic_solution_2d(
    potential: Field,
    electric_field_axis_0: Field,
    electric_field_axis_1: Field,
    electric_field_magnitude: Field,
    *,
    vector_stride: int = 1,
) -> tuple[Figure, tuple[Axes, ...]]:
    """
    Create a four-panel summary of a 2D electrostatic solution.

    The panels contain:

    1. Potential colour map
    2. Equipotential contours
    3. Electric-field magnitude
    4. Electric-field vectors
    """

    figure, axes_array = plt.subplots(
        2,
        2,
        figsize=(11, 9),
    )

    axes = tuple(
        axes_array.ravel()
    )

    plot_electrostatic_potential_2d(
        potential,
        axis=axes[0],
    )

    plot_equipotential_contours_2d(
        potential,
        axis=axes[1],
    )

    plot_electric_field_magnitude_2d(
        electric_field_magnitude,
        axis=axes[2],
    )

    plot_electric_field_vectors_2d(
        electric_field_axis_0,
        electric_field_axis_1,
        stride=vector_stride,
        axis=axes[3],
    )

    figure.tight_layout()

    return figure, axes