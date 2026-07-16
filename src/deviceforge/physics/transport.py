from __future__ import annotations

import numpy as np

from deviceforge.core.field import Field

from .constants import ELEMENTARY_CHARGE, ROOM_TEMPERATURE
from .equilibrium import thermal_voltage


def diffusion_coefficient(
    mobility: float,
    *,
    temperature: float = ROOM_TEMPERATURE,
) -> float:
    """
    Calculate a carrier diffusion coefficient using the Einstein relation.

    The Einstein relation is:

        D = mu * V_T

    where:

    - D is diffusion coefficient in m^2/s
    - mu is carrier mobility in m^2/(V s)
    - V_T is thermal voltage in V
    """

    mobility = float(mobility)

    if not np.isfinite(mobility):
        raise ValueError("Mobility must be finite.")

    if mobility < 0.0:
        raise ValueError("Mobility must not be negative.")

    return mobility * thermal_voltage(temperature)


def compute_electron_current_density(
    *,
    electron_concentration: Field,
    electric_field_x: Field,
    mobility: float,
    temperature: float = ROOM_TEMPERATURE,
) -> Field:
    """
    Calculate one-dimensional electron current density.

    The electron drift-diffusion equation is:

        J_n = q * mu_n * n * E + q * D_n * dn/dx

    Parameters
    ----------
    electron_concentration:
        Electron concentration in 1/m^3.

    electric_field_x:
        Electric-field x-component in V/m.

    mobility:
        Electron mobility in m^2/(V s).

    temperature:
        Absolute temperature in kelvin.

    Returns
    -------
    Field
        Electron current density in A/m^2.
    """

    _validate_transport_fields(
        concentration=electron_concentration,
        electric_field=electric_field_x,
        carrier_name="Electron",
    )

    diffusivity = diffusion_coefficient(
        mobility,
        temperature=temperature,
    )

    spacing_x = electron_concentration.grid.spacing[0]

    concentration_gradient = np.gradient(
        electron_concentration.values,
        spacing_x,
        axis=0,
        edge_order=_gradient_edge_order(
            electron_concentration.grid.shape[0]
        ),
    )

    values = ELEMENTARY_CHARGE * (
        mobility
        * electron_concentration.values
        * electric_field_x.values
        + diffusivity
        * concentration_gradient
    )

    return Field(
        name="electron_current_density_x",
        units="A/m^2",
        grid=electron_concentration.grid,
        values=values,
    )


def compute_hole_current_density(
    *,
    hole_concentration: Field,
    electric_field_x: Field,
    mobility: float,
    temperature: float = ROOM_TEMPERATURE,
) -> Field:
    """
    Calculate one-dimensional hole current density.

    The hole drift-diffusion equation is:

        J_p = q * mu_p * p * E - q * D_p * dp/dx

    Parameters
    ----------
    hole_concentration:
        Hole concentration in 1/m^3.

    electric_field_x:
        Electric-field x-component in V/m.

    mobility:
        Hole mobility in m^2/(V s).

    temperature:
        Absolute temperature in kelvin.

    Returns
    -------
    Field
        Hole current density in A/m^2.
    """

    _validate_transport_fields(
        concentration=hole_concentration,
        electric_field=electric_field_x,
        carrier_name="Hole",
    )

    diffusivity = diffusion_coefficient(
        mobility,
        temperature=temperature,
    )

    spacing_x = hole_concentration.grid.spacing[0]

    concentration_gradient = np.gradient(
        hole_concentration.values,
        spacing_x,
        axis=0,
        edge_order=_gradient_edge_order(
            hole_concentration.grid.shape[0]
        ),
    )

    values = ELEMENTARY_CHARGE * (
        mobility
        * hole_concentration.values
        * electric_field_x.values
        - diffusivity
        * concentration_gradient
    )

    return Field(
        name="hole_current_density_x",
        units="A/m^2",
        grid=hole_concentration.grid,
        values=values,
    )


def compute_total_current_density(
    *,
    electron_current_density: Field,
    hole_current_density: Field,
) -> Field:
    """Calculate total conduction current density."""

    if (
        electron_current_density.grid
        != hole_current_density.grid
    ):
        raise ValueError(
            "Electron and hole current-density fields must use "
            "the same grid."
        )

    if electron_current_density.units != "A/m^2":
        raise ValueError(
            "Electron current density must use units of A/m^2."
        )

    if hole_current_density.units != "A/m^2":
        raise ValueError(
            "Hole current density must use units of A/m^2."
        )

    return Field(
        name="total_current_density_x",
        units="A/m^2",
        grid=electron_current_density.grid,
        values=(
            electron_current_density.values
            + hole_current_density.values
        ),
    )


def _validate_transport_fields(
    *,
    concentration: Field,
    electric_field: Field,
    carrier_name: str,
) -> None:
    """Validate fields used in a drift-diffusion calculation."""

    if concentration.grid != electric_field.grid:
        raise ValueError(
            "Carrier concentration and electric field must use "
            "the same grid."
        )

    if concentration.grid.dimension != 2:
        raise ValueError(
            "Transport post-processing currently supports "
            "two-dimensional grids."
        )

    if concentration.units != "1/m^3":
        raise ValueError(
            f"{carrier_name} concentration must use units of 1/m^3."
        )

    if electric_field.units != "V/m":
        raise ValueError(
            "Electric field must use units of V/m."
        )

    if np.any(concentration.values < 0.0):
        raise ValueError(
            f"{carrier_name} concentration must not be negative."
        )


def _gradient_edge_order(number_of_points: int) -> int:
    """Choose a valid NumPy gradient edge order."""

    return 2 if number_of_points >= 3 else 1