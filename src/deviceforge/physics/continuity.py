from __future__ import annotations

import numpy as np

from deviceforge.core.field import Field
from deviceforge.core.grid import Grid

from .constants import ELEMENTARY_CHARGE


def compute_current_divergence_x(
    edge_current_density: Field,
) -> Field:
    """
    Calculate the x-directed divergence of an edge-current field.

    The input field is defined at x-edge centres with shape:

        (number_of_x_nodes - 1, number_of_y_nodes)

    The divergence is calculated at interior node locations:

        div(J)_i = (J_(i+1/2) - J_(i-1/2)) / dx

    The returned field therefore has shape:

        (number_of_x_nodes - 2, number_of_y_nodes)

    Parameters
    ----------
    edge_current_density:
        X-directed current density defined on x-edge centres in A/m^2.

    Returns
    -------
    Field
        Current-density divergence in A/m^3.
    """

    _validate_edge_current(edge_current_density)

    spacing_x = edge_current_density.grid.spacing[0]

    divergence_values = (
        edge_current_density.values[1:, :]
        - edge_current_density.values[:-1, :]
    ) / spacing_x

    return Field(
        name="current_density_divergence_x",
        units="A/m^3",
        grid=_create_interior_node_grid(
            edge_current_density.grid
        ),
        values=divergence_values,
    )


def compute_electron_continuity_residual(
    electron_edge_current_density: Field,
    *,
    recombination_rate: Field | None = None,
) -> Field:
    """
    Calculate the steady-state electron continuity residual.

    The residual is:

        R_n = div(J_n) / q - U

    where positive U represents net recombination.
    """

    divergence = compute_current_divergence_x(
        electron_edge_current_density
    )

    values = (
        divergence.values
        / ELEMENTARY_CHARGE
    )

    if recombination_rate is not None:
        _validate_recombination_for_interior_grid(
            recombination_rate=recombination_rate,
            interior_grid=divergence.grid,
        )

        values = (
            values
            - recombination_rate.values
        )

    return Field(
        name="electron_continuity_residual",
        units="1/(m^3 s)",
        grid=divergence.grid,
        values=values,
    )


def compute_hole_continuity_residual(
    hole_edge_current_density: Field,
    *,
    recombination_rate: Field | None = None,
) -> Field:
    """
    Calculate the steady-state hole continuity residual.

    The residual is:

        R_p = -div(J_p) / q - U

    where positive U represents net recombination.
    """

    divergence = compute_current_divergence_x(
        hole_edge_current_density
    )

    values = (
        -divergence.values
        / ELEMENTARY_CHARGE
    )

    if recombination_rate is not None:
        _validate_recombination_for_interior_grid(
            recombination_rate=recombination_rate,
            interior_grid=divergence.grid,
        )

        values = (
            values
            - recombination_rate.values
        )

    return Field(
        name="hole_continuity_residual",
        units="1/(m^3 s)",
        grid=divergence.grid,
        values=values,
    )


def maximum_absolute_continuity_residual(
    residual: Field,
) -> float:
    """
    Return the maximum absolute carrier continuity residual.
    """

    if residual.units != "1/(m^3 s)":
        raise ValueError(
            "Continuity residual must use units of 1/(m^3 s)."
        )

    return float(
        np.max(
            np.abs(residual.values)
        )
    )


def _create_interior_node_grid(
    edge_grid: Grid,
) -> Grid:
    """
    Reconstruct the interior-node grid from an x-edge grid.

    If the original node grid has shape ``(nx, ny)``, the x-edge grid has
    shape ``(nx - 1, ny)`` and the interior-node grid has shape
    ``(nx - 2, ny)``.
    """

    if edge_grid.dimension != 2:
        raise ValueError(
            "Continuity calculations currently support "
            "two-dimensional grids."
        )

    if edge_grid.shape[0] < 2:
        raise ValueError(
            "At least two x-directed edges are required "
            "to calculate current divergence."
        )

    return Grid(
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


def _validate_edge_current(
    edge_current_density: Field,
) -> None:
    """Validate a current-density field defined on x edges."""

    if edge_current_density.grid.dimension != 2:
        raise ValueError(
            "Continuity calculations currently support "
            "two-dimensional grids."
        )

    if edge_current_density.grid.shape[0] < 2:
        raise ValueError(
            "At least two x-directed edges are required "
            "to calculate current divergence."
        )

    if edge_current_density.units != "A/m^2":
        raise ValueError(
            "Edge current density must use units of A/m^2."
        )

    if not np.all(
        np.isfinite(edge_current_density.values)
    ):
        raise ValueError(
            "Edge current density must contain only finite values."
        )

# helper function
def _validate_recombination_for_interior_grid(
    *,
    recombination_rate: Field,
    interior_grid: Grid,
) -> None:
    """Validate recombination supplied to a continuity residual."""

    if recombination_rate.grid != interior_grid:
        raise ValueError(
            "Recombination rate must use the interior-node grid."
        )

    if recombination_rate.units != "1/(m^3 s)":
        raise ValueError(
            "Recombination rate must use units of 1/(m^3 s)."
        )