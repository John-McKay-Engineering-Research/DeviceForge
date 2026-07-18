from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from deviceforge.core.field import Field
from deviceforge.core.grid import Grid

from .constants import ELEMENTARY_CHARGE, ROOM_TEMPERATURE
from .equilibrium import thermal_voltage
from .transport import diffusion_coefficient


def bernoulli_function(
    value: ArrayLike,
) -> NDArray[np.float64] | float:
    """
    Evaluate the Bernoulli function used by Scharfetter-Gummel fluxes.

    The Bernoulli function is:

        B(x) = x / (exp(x) - 1)

    Direct evaluation is numerically unstable near zero and may overflow
    for large positive values. This implementation uses:

    - a Taylor expansion near zero
    - an exponential form for positive values
    - a stable denominator for negative values

    Parameters
    ----------
    value:
        Scalar or array-like dimensionless argument.

    Returns
    -------
    numpy.ndarray or float
        Evaluated Bernoulli function.
    """

    values = np.asarray(
        value,
        dtype=np.float64,
    )

    result = np.empty_like(values)

    small = np.abs(values) < 1.0e-6
    positive = values >= 1.0e-6
    negative = values <= -1.0e-6

    small_values = values[small]

    result[small] = (
        1.0
        - small_values / 2.0
        + small_values**2 / 12.0
        - small_values**4 / 720.0
    )

    positive_values = values[positive]
    positive_exponential = np.exp(
        -positive_values
    )

    result[positive] = (
        positive_values
        * positive_exponential
        / (1.0 - positive_exponential)
    )

    negative_values = values[negative]

    result[negative] = (
        -negative_values
        / (1.0 - np.exp(negative_values))
    )

    if result.ndim == 0:
        return float(result)

    return result


def compute_electron_scharfetter_gummel_current_x(
    *,
    potential: Field,
    electron_concentration: Field,
    mobility: float,
    temperature: float = ROOM_TEMPERATURE,
) -> Field:
    """
    Calculate electron current density on x-directed grid edges.

    The edge current is:

        J_n = q D_n / dx
              * [n_R B(delta_psi) - n_L B(-delta_psi)]

    where:

        delta_psi = (phi_R - phi_L) / V_T

    Parameters
    ----------
    potential:
        Node-centred electrostatic potential in volts.

    electron_concentration:
        Node-centred electron concentration in inverse cubic metres.

    mobility:
        Electron mobility in square metres per volt-second.

    temperature:
        Absolute temperature in kelvin.

    Returns
    -------
    Field
        X-directed electron current density on edge centres in A/m^2.
    """

    _validate_scharfetter_gummel_inputs(
        potential=potential,
        concentration=electron_concentration,
        carrier_name="Electron",
    )

    diffusivity = diffusion_coefficient(
        mobility,
        temperature=temperature,
    )

    voltage = thermal_voltage(temperature)
    spacing_x = potential.grid.spacing[0]
    # updated to support 1 and 2 dimensions
    potential_left = potential.values[:-1]
    potential_right = potential.values[1:]

    concentration_left = (
        electron_concentration.values[:-1]
    )
    concentration_right = (
        electron_concentration.values[1:]
    )

    normalised_potential_difference = (
        potential_right - potential_left
    ) / voltage

    values = (
        ELEMENTARY_CHARGE
        * diffusivity
        / spacing_x
        * (
            concentration_right
            * bernoulli_function(
                normalised_potential_difference
            )
            - concentration_left
            * bernoulli_function(
                -normalised_potential_difference
            )
        )
    )

    return Field(
        name="electron_current_density_x_edges",
        units="A/m^2",
        grid=_create_x_edge_grid(potential.grid),
        values=values,
    )


def compute_hole_scharfetter_gummel_current_x(
    *,
    potential: Field,
    hole_concentration: Field,
    mobility: float,
    temperature: float = ROOM_TEMPERATURE,
) -> Field:
    """
    Calculate hole current density on x-directed grid edges.

    The edge current is:

        J_p = q D_p / dx
              * [p_L B(delta_psi) - p_R B(-delta_psi)]

    where:

        delta_psi = (phi_R - phi_L) / V_T

    Parameters
    ----------
    potential:
        Node-centred electrostatic potential in volts.

    hole_concentration:
        Node-centred hole concentration in inverse cubic metres.

    mobility:
        Hole mobility in square metres per volt-second.

    temperature:
        Absolute temperature in kelvin.

    Returns
    -------
    Field
        X-directed hole current density on edge centres in A/m^2.
    """

    _validate_scharfetter_gummel_inputs(
        potential=potential,
        concentration=hole_concentration,
        carrier_name="Hole",
    )

    diffusivity = diffusion_coefficient(
        mobility,
        temperature=temperature,
    )

    voltage = thermal_voltage(temperature)
    spacing_x = potential.grid.spacing[0]
    # updated to support 1 and 2 dimensions
    potential_left = potential.values[:-1]
    potential_right = potential.values[1:]

    concentration_left = (
        hole_concentration.values[:-1]
    )
    concentration_right = (
        hole_concentration.values[1:]
    )

    normalised_potential_difference = (
        potential_right - potential_left
    ) / voltage

    values = (
        ELEMENTARY_CHARGE
        * diffusivity
        / spacing_x
        * (
            concentration_left
            * bernoulli_function(
                normalised_potential_difference
            )
            - concentration_right
            * bernoulli_function(
                -normalised_potential_difference
            )
        )
    )

    return Field(
        name="hole_current_density_x_edges",
        units="A/m^2",
        grid=_create_x_edge_grid(potential.grid),
        values=values,
    )


def compute_total_scharfetter_gummel_current_x(
    *,
    electron_current_density: Field,
    hole_current_density: Field,
) -> Field:
    """
    Calculate total Scharfetter-Gummel current density.

    Both input fields must already be defined on the same x-edge grid.
    """

    if (
        electron_current_density.grid
        != hole_current_density.grid
    ):
        raise ValueError(
            "Electron and hole edge-current fields must use "
            "the same grid."
        )

    if electron_current_density.units != "A/m^2":
        raise ValueError(
            "Electron edge current must use units of A/m^2."
        )

    if hole_current_density.units != "A/m^2":
        raise ValueError(
            "Hole edge current must use units of A/m^2."
        )

    return Field(
        name="total_current_density_x_edges",
        units="A/m^2",
        grid=electron_current_density.grid,
        values=(
            electron_current_density.values
            + hole_current_density.values
        ),
    )

# updated to support 1 and 2 dimensions
def _create_x_edge_grid(
    node_grid: Grid,
) -> Grid:
    """
    Create a grid representing centres of x-directed node edges.

    For a one-dimensional node grid with shape ``(nx,)``, the edge grid
    has shape ``(nx - 1,)``.

    For a two-dimensional node grid with shape ``(nx, ny)``, the edge grid
    has shape ``(nx - 1, ny)``.
    """

    if node_grid.dimension not in (1, 2):
        raise ValueError(
            "Scharfetter-Gummel transport currently supports "
            "one-dimensional and two-dimensional grids."
        )

    if node_grid.shape[0] < 2:
        raise ValueError(
            "At least two x-direction grid points are required."
        )

    if node_grid.dimension == 1:
        return Grid(
            shape=(node_grid.shape[0] - 1,),
            spacing=node_grid.spacing,
            origin=(
                node_grid.origin[0]
                + 0.5 * node_grid.spacing[0],
            ),
        )

    return Grid(
        shape=(
            node_grid.shape[0] - 1,
            node_grid.shape[1],
        ),
        spacing=node_grid.spacing,
        origin=(
            node_grid.origin[0]
            + 0.5 * node_grid.spacing[0],
            node_grid.origin[1],
        ),
    )


def _validate_scharfetter_gummel_inputs(
    *,
    potential: Field,
    concentration: Field,
    carrier_name: str,
) -> None:
    """Validate node-centred potential and concentration fields."""

    if potential.grid != concentration.grid:
        raise ValueError(
            "Potential and carrier concentration must use "
            "the same node grid."
        )
    # updated to support 1 and 2 dimensions
    if potential.grid.dimension not in (1, 2):
        raise ValueError(
            "Scharfetter-Gummel transport currently supports "
            "one-dimensional and two-dimensional grids."
        )

    if potential.grid.shape[0] < 2:
        raise ValueError(
            "At least two x-direction grid points are required."
        )

    if potential.units != "V":
        raise ValueError(
            "Electrostatic potential must use units of V."
        )

    if concentration.units != "1/m^3":
        raise ValueError(
            f"{carrier_name} concentration must use units of 1/m^3."
        )

    if np.any(concentration.values < 0.0):
        raise ValueError(
            f"{carrier_name} concentration must not be negative."
        )