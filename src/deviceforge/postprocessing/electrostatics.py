from __future__ import annotations

import numpy as np

from ..core.field import Field

VACUUM_PERMITTIVITY = 8.8541878128e-12

def calculate_electric_field(
    potential: Field,
) -> Field:
    """
    Calculate a one-dimensional electric field from electrostatic potential.

    The electric field is defined by

        E = -d(phi)/dx

    A centred finite difference is used at interior grid points. First-order
    one-sided differences are used at the two domain endpoints.

    Parameters
    ----------
    potential:
        Electrostatic-potential field in volts on a one-dimensional grid.

    Returns
    -------
    Field
        Electric-field values in volts per metre.

    Raises
    ------
    TypeError
        If potential is not a Field instance.

    ValueError
        If the field is not defined on a one-dimensional grid or does not
        use volts as its units.
    """

    if not isinstance(potential, Field):
        raise TypeError(
            "Electric-field calculation requires a Field instance."
        )

    if potential.grid.dimension != 1:
        raise ValueError(
            "Electric-field calculation currently supports only "
            "one-dimensional fields."
        )

    if potential.units != "V":
        raise ValueError(
            "Potential field units must be 'V'."
        )

    spacing = potential.grid.spacing[0]
    values = potential.values

    electric_field_values = np.empty_like(
        values,
        dtype=np.float64,
    )

    electric_field_values[0] = -(
        values[1] - values[0]
    ) / spacing

    electric_field_values[-1] = -(
        values[-1] - values[-2]
    ) / spacing

    if values.size > 2:
        electric_field_values[1:-1] = -(
            values[2:] - values[:-2]
        ) / (2.0 * spacing)

    return Field(
        name="electric_field",
        units="V/m",
        grid=potential.grid,
        values=electric_field_values,
    )

# extend electrostatics

def calculate_electric_displacement_field(
    electric_field: Field,
    relative_permittivity: Field,
) -> Field:
    """
    Calculate electric displacement from electric field and permittivity.

    The electric displacement field is

        D = epsilon_0 * epsilon_r * E

    Parameters
    ----------
    electric_field:
        Electric-field values in volts per metre.

    relative_permittivity:
        Dimensionless relative-permittivity field defined on the same grid.

    Returns
    -------
    Field
        Electric displacement in coulombs per square metre.

    Raises
    ------
    TypeError
        If either argument is not a Field.

    ValueError
        If the fields have incompatible grids or units, or if the relative
        permittivity contains non-positive values.
    """

    if not isinstance(electric_field, Field):
        raise TypeError(
            "Electric displacement calculation requires an "
            "electric-field Field."
        )

    if not isinstance(relative_permittivity, Field):
        raise TypeError(
            "Electric displacement calculation requires a "
            "relative-permittivity Field."
        )

    if electric_field.units != "V/m":
        raise ValueError(
            "Electric field units must be 'V/m'."
        )

    if relative_permittivity.units != "dimensionless":
        raise ValueError(
            "Relative permittivity units must be 'dimensionless'."
        )

    if electric_field.grid != relative_permittivity.grid:
        raise ValueError(
            "Electric field and relative permittivity must use "
            "the same grid."
        )

    if np.any(relative_permittivity.values <= 0.0):
        raise ValueError(
            "Relative permittivity values must be positive."
        )

    displacement_values = (
        VACUUM_PERMITTIVITY
        * relative_permittivity.values
        * electric_field.values
    )

    return Field(
        name="electric_displacement",
        units="C/m^2",
        grid=electric_field.grid,
        values=displacement_values,
    )

# electrostatic density

def calculate_electrostatic_energy_density(
    electric_field: Field,
    relative_permittivity: Field,
) -> Field:
    """
    Calculate electrostatic energy density.

    The electrostatic energy density is

        u = 0.5 * epsilon_0 * epsilon_r * E^2

    Parameters
    ----------
    electric_field:
        Electric-field values in volts per metre.

    relative_permittivity:
        Dimensionless relative-permittivity field defined on the same grid.

    Returns
    -------
    Field
        Electrostatic energy density in joules per cubic metre.

    Raises
    ------
    TypeError
        If either argument is not a Field.

    ValueError
        If the fields use incompatible grids or units, or if relative
        permittivity contains non-positive values.
    """

    if not isinstance(electric_field, Field):
        raise TypeError(
            "Electrostatic energy-density calculation requires "
            "an electric-field Field."
        )

    if not isinstance(relative_permittivity, Field):
        raise TypeError(
            "Electrostatic energy-density calculation requires "
            "a relative-permittivity Field."
        )

    if electric_field.units != "V/m":
        raise ValueError(
            "Electric field units must be 'V/m'."
        )

    if relative_permittivity.units != "dimensionless":
        raise ValueError(
            "Relative permittivity units must be 'dimensionless'."
        )

    if electric_field.grid != relative_permittivity.grid:
        raise ValueError(
            "Electric field and relative permittivity must use "
            "the same grid."
        )

    if np.any(relative_permittivity.values <= 0.0):
        raise ValueError(
            "Relative permittivity values must be positive."
        )

    energy_density_values = (
        0.5
        * VACUUM_PERMITTIVITY
        * relative_permittivity.values
        * electric_field.values**2
    )

    return Field(
        name="electrostatic_energy_density",
        units="J/m^3",
        grid=electric_field.grid,
        values=energy_density_values,
    )