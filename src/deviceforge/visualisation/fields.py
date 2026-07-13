from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deviceforge import Field, SimulationResult


def plot_scalar_field(
    field: Field,
    *,
    title: str | None = None,
    colourbar_label: str | None = None,
    contour_levels: int = 15,
) -> tuple[Figure, Axes]:
    """
    Plot a two-dimensional scalar field as a filled contour map.

    Parameters
    ----------
    field:
        Two-dimensional scalar field to visualise.

    title:
        Optional figure title.

    colourbar_label:
        Optional colour-bar label. Defaults to the field name and units.

    contour_levels:
        Number of filled contour levels.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """

    if field.grid.dimension != 2:
        raise ValueError(
            "plot_scalar_field currently supports only two-dimensional fields."
        )

    x_coordinates = field.grid.coordinates(axis=0)
    y_coordinates = field.grid.coordinates(axis=1)

    figure, axes = plt.subplots()

    contour = axes.contourf(
        x_coordinates * 1.0e9,
        y_coordinates * 1.0e9,
        field.values.T,
        levels=contour_levels,
    )

    colourbar = figure.colorbar(contour, ax=axes)

    if colourbar_label is None:
        colourbar_label = f"{field.name} [{field.units}]"

    colourbar.set_label(colourbar_label)

    axes.set_xlabel("x [nm]")
    axes.set_ylabel("y [nm]")
    axes.set_title(title or field.name.replace("_", " ").title())

    figure.tight_layout()

    return figure, axes


def plot_convergence(
    result: SimulationResult,
) -> tuple[Figure, Axes]:
    """
    Plot solver residual against iteration number.
    """

    if result.iterations == 0:
        raise ValueError(
            "Cannot plot convergence for a zero-iteration result."
        )

    iterations = np.arange(
        1,
        result.iterations + 1,
    )

    figure, axes = plt.subplots()

    axes.semilogy(
        iterations,
        result.residual_history,
    )

    axes.set_xlabel("Iteration")
    axes.set_ylabel("Maximum potential change [V]")
    axes.set_title(
        f"{result.solver_name.title()} convergence history"
    )
    axes.grid(True)

    figure.tight_layout()

    return figure, axes


def save_figure(
    figure: Figure,
    path: str | Path,
    *,
    dpi: int = 300,
) -> None:
    """
    Save a Matplotlib figure and create parent folders if required.
    """

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )