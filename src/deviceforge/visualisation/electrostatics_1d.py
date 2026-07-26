from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core.field import Field
from ..workflows import ElectrostaticWorkflowResult
from ..core.face_field import FaceField



def _validate_one_dimensional_field(
    field: Field,
) -> None:
    """Validate a field before one-dimensional plotting."""

    if not isinstance(field, Field):
        raise TypeError(
            "One-dimensional plotting requires a Field instance."
        )

    if field.grid.dimension != 1:
        raise ValueError(
            "One-dimensional plotting requires a one-dimensional field."
        )


def _coordinates_in_nanometres(
    field: Field,
) -> np.ndarray:
    """Return the field coordinates in nanometres."""

    return field.grid.coordinates(0) * 1.0e9


def _create_field_plot(
    field: Field,
    *,
    title: str,
    y_label: str,
    transform: Callable[
        [np.ndarray],
        np.ndarray,
    ] | None = None,
) -> tuple[Figure, Axes]:
    """Create a standard one-dimensional field plot."""

    _validate_one_dimensional_field(field)

    coordinates = _coordinates_in_nanometres(field)

    values = np.asarray(
        field.values,
        dtype=np.float64,
    )

    if transform is not None:
        values = transform(values)

    figure, axis = plt.subplots()

    axis.plot(
        coordinates,
        values,
    )

    axis.set_xlabel("Position (nm)")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True)

    figure.tight_layout()

    return figure, axis


def plot_electrostatic_potential(
    output: ElectrostaticWorkflowResult,
) -> tuple[Figure, Axes]:
    """Plot electrostatic potential against position."""

    _validate_workflow_output(output)

    return _create_field_plot(
        output.potential,
        title="Electrostatic Potential",
        y_label="Potential (V)",
    )


def plot_electric_field(
    output: ElectrostaticWorkflowResult,
) -> tuple[Figure, Axes]:
    """Plot electric field against position."""

    _validate_workflow_output(output)

    return _create_field_plot(
        output.electric_field,
        title="Electric Field",
        y_label="Electric field (V/m)",
    )


def plot_electric_displacement(
    output: ElectrostaticWorkflowResult,
) -> tuple[Figure, Axes]:
    """Plot electric displacement against position."""

    _validate_workflow_output(output)

    return _create_field_plot(
        output.electric_displacement,
        title="Electric Displacement",
        y_label="Electric displacement (C/m²)",
    )


def plot_electrostatic_energy_density(
    output: ElectrostaticWorkflowResult,
) -> tuple[Figure, Axes]:
    """Plot electrostatic energy density against position."""

    _validate_workflow_output(output)

    return _create_field_plot(
        output.energy_density,
        title="Electrostatic Energy Density",
        y_label="Energy density (J/m³)",
    )


def plot_relative_permittivity(
    output: ElectrostaticWorkflowResult,
    relative_permittivity: Field,
) -> tuple[Figure, Axes]:
    """Plot the relative-permittivity material profile."""

    _validate_workflow_output(output)
    _validate_one_dimensional_field(
        relative_permittivity
    )

    if relative_permittivity.units != "dimensionless":
        raise ValueError(
            "Relative-permittivity field units must be "
            "'dimensionless'."
        )

    if relative_permittivity.grid != output.potential.grid:
        raise ValueError(
            "Relative permittivity must use the workflow-result grid."
        )

    return _create_field_plot(
        relative_permittivity,
        title="Relative Permittivity Profile",
        y_label="Relative permittivity",
    )


def plot_residual_history(
    output: ElectrostaticWorkflowResult,
) -> tuple[Figure, Axes]:
    """Plot solver residual against iteration number."""

    _validate_workflow_output(output)

    residual_history = np.asarray(
        output.residual_history,
        dtype=np.float64,
    )

    iterations = np.arange(
        1,
        residual_history.size + 1,
        dtype=np.int64,
    )

    figure, axis = plt.subplots()

    if residual_history.size == 0:
        axis.text(
            0.5,
            0.5,
            "No residual history",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
        )
    elif np.all(residual_history > 0.0):
        axis.semilogy(
            iterations,
            residual_history,
            marker="o",
        )
    else:
        axis.plot(
            iterations,
            residual_history,
            marker="o",
        )

    axis.set_xlabel("Iteration")
    axis.set_ylabel("Residual")
    axis.set_title("Solver Residual History")
    axis.grid(True)

    figure.tight_layout()

    return figure, axis


def _validate_workflow_output(
    output: ElectrostaticWorkflowResult,
) -> None:
    """Validate a workflow result supplied for plotting."""

    if not isinstance(
        output,
        ElectrostaticWorkflowResult,
    ):
        raise TypeError(
            "Electrostatic plotting requires an "
            "ElectrostaticWorkflowResult instance."
        )

# update displacement visualisation
# updated to return a cleaner figure.
def plot_face_electric_displacement(
    face_displacement: FaceField,
) -> tuple[Figure, Axes]:
    """
    Plot conservative face-centred electric displacement.

    Nearly constant fields are displayed with a sensible vertical range
    rather than allowing Matplotlib to magnify floating-point roundoff.
    """

    if not isinstance(face_displacement, FaceField):
        raise TypeError(
            "Face-displacement plotting requires a FaceField."
        )

    if face_displacement.units != "C/m^2":
        raise ValueError(
            "Face displacement units must be 'C/m^2'."
        )

    coordinates_nm = (
        face_displacement.coordinates()
        * 1.0e9
    )

    values = np.asarray(
        face_displacement.values,
        dtype=np.float64,
    )

    mean_value = float(
        np.mean(values)
    )

    value_spread = float(
        np.ptp(values)
    )

    relative_spread = (
        value_spread
        / max(
            abs(mean_value),
            np.finfo(np.float64).tiny,
        )
    )

    figure, axis = plt.subplots()

    axis.plot(
        coordinates_nm,
        values,
    )

    axis.set_xlabel("Position (nm)")
    axis.set_ylabel(
        "Electric displacement (C/m²)"
    )
    axis.set_title(
        "Face-Centred Electric Displacement"
    )
    axis.grid(True)

    # Avoid magnifying machine-precision variation in an effectively
    # constant conservative flux.
    if relative_spread < 1.0e-10:
        display_margin = max(
            abs(mean_value) * 1.0e-3,
            1.0e-12,
        )

        axis.set_ylim(
            mean_value - display_margin,
            mean_value + display_margin,
        )

        axis.ticklabel_format(
            axis="y",
            style="scientific",
            scilimits=(-3, 3),
            useOffset=False,
        )

        axis.text(
            0.02,
            0.95,
            (
                f"Mean D = {mean_value:.6e} C/m²\n"
                f"Relative variation = {relative_spread:.3e}"
            ),
            transform=axis.transAxes,
            verticalalignment="top",
        )

    figure.tight_layout()

    return figure, axis