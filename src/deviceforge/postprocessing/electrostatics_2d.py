from __future__ import annotations

import numpy as np

from ..core.field import Field


def calculate_electric_field_components_2d(
    potential: Field,
) -> tuple[Field, Field]:
    """
    Calculate two-dimensional electric-field components.

    The electric field is

        E = -grad(phi)

    with

        E_axis_0 = -dphi / d(axis_0)
        E_axis_1 = -dphi / d(axis_1)

    NumPy's second-order central difference is used at interior points,
    with second-order one-sided differences at the domain boundaries.

    Parameters
    ----------
    potential:
        Two-dimensional electrostatic-potential field in volts.

    Returns
    -------
    tuple[Field, Field]
        Electric-field components along grid axes 0 and 1.
    """

    if not isinstance(
        potential,
        Field,
    ):
        raise TypeError(
            "Potential must be a Field instance."
        )

    if potential.grid.dimension != 2:
        raise ValueError(
            "Two-dimensional electric-field calculation "
            "requires a two-dimensional grid."
        )

    if potential.units != "V":
        raise ValueError(
            "Potential field units must be 'V'."
        )

    values = np.asarray(
        potential.values,
        dtype=np.float64,
    )

    if values.shape != potential.grid.shape:
        raise ValueError(
            "Potential values must match the grid shape."
        )

    spacing_axis_0, spacing_axis_1 = (
        potential.grid.spacing
    )

    (
        derivative_axis_0,
        derivative_axis_1,
    ) = np.gradient(
        values,
        spacing_axis_0,
        spacing_axis_1,
        edge_order=2,
    )

    electric_field_axis_0 = Field(
        name="electric_field_axis_0",
        units="V/m",
        grid=potential.grid,
        values=-derivative_axis_0,
    )

    electric_field_axis_1 = Field(
        name="electric_field_axis_1",
        units="V/m",
        grid=potential.grid,
        values=-derivative_axis_1,
    )

    return (
        electric_field_axis_0,
        electric_field_axis_1,
    )


def calculate_electric_field_magnitude_2d(
    electric_field_axis_0: Field,
    electric_field_axis_1: Field,
) -> Field:
    """
    Calculate the magnitude of a two-dimensional electric field.

    The magnitude is

        |E| = sqrt(E_axis_0^2 + E_axis_1^2).
    """

    if not isinstance(
        electric_field_axis_0,
        Field,
    ) or not isinstance(
        electric_field_axis_1,
        Field,
    ):
        raise TypeError(
            "Electric-field components must be Field instances."
        )

    if (
        electric_field_axis_0.grid
        != electric_field_axis_1.grid
    ):
        raise ValueError(
            "Electric-field components must use the same grid."
        )

    if (
        electric_field_axis_0.grid.dimension
        != 2
    ):
        raise ValueError(
            "Electric-field magnitude calculation requires "
            "a two-dimensional grid."
        )

    if (
        electric_field_axis_0.units
        != "V/m"
        or electric_field_axis_1.units
        != "V/m"
    ):
        raise ValueError(
            "Electric-field component units must be 'V/m'."
        )

    magnitude = np.hypot(
        electric_field_axis_0.values,
        electric_field_axis_1.values,
    )

    return Field(
        name="electric_field_magnitude",
        units="V/m",
        grid=electric_field_axis_0.grid,
        values=magnitude,
    )


def calculate_electrostatic_fields_2d(
    potential: Field,
) -> tuple[Field, Field, Field]:
    """
    Calculate both 2D electric-field components and their magnitude.
    """

    (
        electric_field_axis_0,
        electric_field_axis_1,
    ) = calculate_electric_field_components_2d(
        potential
    )

    electric_field_magnitude = (
        calculate_electric_field_magnitude_2d(
            electric_field_axis_0,
            electric_field_axis_1,
        )
    )

    return (
        electric_field_axis_0,
        electric_field_axis_1,
        electric_field_magnitude,
    )