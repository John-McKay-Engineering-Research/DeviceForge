from __future__ import annotations

import numpy as np

from deviceforge.core import FaceField, Field


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

def calculate_face_electric_field_components_2d(
    potential: Field,
) -> tuple[FaceField, FaceField]:
    """
    Calculate face-centred electric-field components from a 2D potential.

    The electric field is defined by

        E = -grad(phi).

    The axis-0 component is evaluated on faces between neighbouring
    nodes along grid axis 0:

        E_0[i + 1/2, j]
            = -(phi[i + 1, j] - phi[i, j]) / dx_0.

    The axis-1 component is evaluated on faces between neighbouring
    nodes along grid axis 1:

        E_1[i, j + 1/2]
            = -(phi[i, j + 1] - phi[i, j]) / dx_1.

    Parameters
    ----------
    potential:
        Two-dimensional node-centred electrostatic potential field
        with units of volts.

    Returns
    -------
    tuple[FaceField, FaceField]
        Face-centred electric-field components along grid axes 0 and 1.

        The axis-0 field has shape

            (n0 - 1, n1),

        and the axis-1 field has shape

            (n0, n1 - 1).
    """

    if not isinstance(potential, Field):
        raise TypeError(
            "Potential must be a Field instance."
        )

    if potential.grid.dimension != 2:
        raise ValueError(
            "Face-centred 2D electric-field calculation requires "
            "a two-dimensional potential field."
        )

    if potential.units != "V":
        raise ValueError(
            "Potential field must have units of volts ('V')."
        )

    spacing_axis_0 = potential.grid.spacing[0]
    spacing_axis_1 = potential.grid.spacing[1]

    electric_field_axis_0_values = -(
        np.diff(
            potential.values,
            axis=0,
        )
        / spacing_axis_0
    )

    electric_field_axis_1_values = -(
        np.diff(
            potential.values,
            axis=1,
        )
        / spacing_axis_1
    )

    electric_field_axis_0 = FaceField(
        name="electric_field_axis_0_face",
        units="V/m",
        grid=potential.grid,
        values=electric_field_axis_0_values,
        axis=0,
    )

    electric_field_axis_1 = FaceField(
        name="electric_field_axis_1_face",
        units="V/m",
        grid=potential.grid,
        values=electric_field_axis_1_values,
        axis=1,
    )

    return (
        electric_field_axis_0,
        electric_field_axis_1,
    )

def calculate_face_relative_permittivity_components_2d(
    relative_permittivity: Field,
) -> tuple[FaceField, FaceField]:
    """
    Calculate harmonic face-centred relative permittivity in 2D.

    For neighbouring node values along grid axis 0,

        epsilon_r[i + 1/2, j]
            = 2 * epsilon_r[i, j] * epsilon_r[i + 1, j]
              / (
                  epsilon_r[i, j]
                  + epsilon_r[i + 1, j]
              ).

    For neighbouring node values along grid axis 1,

        epsilon_r[i, j + 1/2]
            = 2 * epsilon_r[i, j] * epsilon_r[i, j + 1]
              / (
                  epsilon_r[i, j]
                  + epsilon_r[i, j + 1]
              ).

    Parameters
    ----------
    relative_permittivity:
        Two-dimensional node-centred relative-permittivity field.
        Values must be positive and units must be "dimensionless".

    Returns
    -------
    tuple[FaceField, FaceField]
        Harmonic face-centred relative permittivity along grid
        axes 0 and 1.

        The axis-0 field has shape

            (n0 - 1, n1),

        and the axis-1 field has shape

            (n0, n1 - 1).
    """

    if not isinstance(
        relative_permittivity,
        Field,
    ):
        raise TypeError(
            "Face-permittivity calculation requires "
            "a Field instance."
        )

    if relative_permittivity.grid.dimension != 2:
        raise ValueError(
            "Face-permittivity calculation requires "
            "a two-dimensional field."
        )

    if (
        relative_permittivity.units
        != "dimensionless"
    ):
        raise ValueError(
            "Relative permittivity units must be "
            "'dimensionless'."
        )

    values = relative_permittivity.values

    if np.any(values <= 0.0):
        raise ValueError(
            "Relative permittivity values must be positive."
        )

    values_axis_0_lower = values[:-1, :]
    values_axis_0_upper = values[1:, :]

    face_values_axis_0 = (
        2.0
        * values_axis_0_lower
        * values_axis_0_upper
        / (
            values_axis_0_lower
            + values_axis_0_upper
        )
    )

    values_axis_1_lower = values[:, :-1]
    values_axis_1_upper = values[:, 1:]

    face_values_axis_1 = (
        2.0
        * values_axis_1_lower
        * values_axis_1_upper
        / (
            values_axis_1_lower
            + values_axis_1_upper
        )
    )

    relative_permittivity_axis_0 = FaceField(
        name="relative_permittivity_axis_0_face",
        units="dimensionless",
        grid=relative_permittivity.grid,
        values=face_values_axis_0,
        axis=0,
    )

    relative_permittivity_axis_1 = FaceField(
        name="relative_permittivity_axis_1_face",
        units="dimensionless",
        grid=relative_permittivity.grid,
        values=face_values_axis_1,
        axis=1,
    )

    return (
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
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