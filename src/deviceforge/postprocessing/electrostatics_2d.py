from __future__ import annotations

import numpy as np

from deviceforge.core import FaceField, Field

from .electrostatics import VACUUM_PERMITTIVITY

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

def calculate_face_electric_displacement_components_2d(
    face_electric_field_axis_0: FaceField,
    face_electric_field_axis_1: FaceField,
    face_relative_permittivity_axis_0: FaceField,
    face_relative_permittivity_axis_1: FaceField,
) -> tuple[FaceField, FaceField]:
    """
    Calculate face-centred electric-displacement components in 2D.

    The electric displacement on each face is

        D = epsilon_0 * epsilon_r * E.

    Parameters
    ----------
    face_electric_field_axis_0:
        Face-centred electric field along grid axis 0.

    face_electric_field_axis_1:
        Face-centred electric field along grid axis 1.

    face_relative_permittivity_axis_0:
        Face-centred relative permittivity along grid axis 0.

    face_relative_permittivity_axis_1:
        Face-centred relative permittivity along grid axis 1.

    Returns
    -------
    tuple[FaceField, FaceField]
        Electric-displacement components along grid axes 0 and 1,
        in coulombs per square metre.
    """

    face_fields = (
        face_electric_field_axis_0,
        face_electric_field_axis_1,
        face_relative_permittivity_axis_0,
        face_relative_permittivity_axis_1,
    )

    if not all(
        isinstance(face_field, FaceField)
        for face_field in face_fields
    ):
        raise TypeError(
            "Face electric-displacement calculation requires "
            "FaceField instances."
        )

    if face_electric_field_axis_0.axis != 0:
        raise ValueError(
            "Axis-0 electric field must be defined on axis 0."
        )

    if face_electric_field_axis_1.axis != 1:
        raise ValueError(
            "Axis-1 electric field must be defined on axis 1."
        )

    if face_relative_permittivity_axis_0.axis != 0:
        raise ValueError(
            "Axis-0 relative permittivity must be defined on axis 0."
        )

    if face_relative_permittivity_axis_1.axis != 1:
        raise ValueError(
            "Axis-1 relative permittivity must be defined on axis 1."
        )

    if (
        face_electric_field_axis_0.units != "V/m"
        or face_electric_field_axis_1.units != "V/m"
    ):
        raise ValueError(
            "Face electric-field units must be 'V/m'."
        )

    if (
        face_relative_permittivity_axis_0.units
        != "dimensionless"
        or face_relative_permittivity_axis_1.units
        != "dimensionless"
    ):
        raise ValueError(
            "Face relative-permittivity units must be "
            "'dimensionless'."
        )

    reference_grid = face_electric_field_axis_0.grid

    if any(
        face_field.grid != reference_grid
        for face_field in face_fields[1:]
    ):
        raise ValueError(
            "Face electric fields and face relative permittivities "
            "must use the same grid."
        )

    if (
        np.any(
            face_relative_permittivity_axis_0.values <= 0.0
        )
        or np.any(
            face_relative_permittivity_axis_1.values <= 0.0
        )
    ):
        raise ValueError(
            "Face relative-permittivity values must be positive."
        )

    displacement_axis_0_values = (
        VACUUM_PERMITTIVITY
        * face_relative_permittivity_axis_0.values
        * face_electric_field_axis_0.values
    )

    displacement_axis_1_values = (
        VACUUM_PERMITTIVITY
        * face_relative_permittivity_axis_1.values
        * face_electric_field_axis_1.values
    )

    displacement_axis_0 = FaceField(
        name="electric_displacement_axis_0_face",
        units="C/m^2",
        grid=reference_grid,
        values=displacement_axis_0_values,
        axis=0,
    )

    displacement_axis_1 = FaceField(
        name="electric_displacement_axis_1_face",
        units="C/m^2",
        grid=reference_grid,
        values=displacement_axis_1_values,
        axis=1,
    )

    return (
        displacement_axis_0,
        displacement_axis_1,
    )

def calculate_face_electrostatic_fields_2d(
    potential: Field,
    relative_permittivity: Field,
) -> tuple[
    FaceField,
    FaceField,
    FaceField,
    FaceField,
    FaceField,
    FaceField,
]:
    """
    Calculate the standard face-centred electrostatic fields in 2D.

    The returned fields are, in order:

        electric_field_axis_0
        electric_field_axis_1
        relative_permittivity_axis_0
        relative_permittivity_axis_1
        electric_displacement_axis_0
        electric_displacement_axis_1

    Parameters
    ----------
    potential:
        Two-dimensional electrostatic-potential field in volts.

    relative_permittivity:
        Two-dimensional dimensionless relative-permittivity field
        defined on the same grid as the potential.

    Returns
    -------
    tuple[FaceField, FaceField, FaceField, FaceField, FaceField, FaceField]
        Face-centred electric-field, relative-permittivity, and
        electric-displacement components along grid axes 0 and 1.

    Raises
    ------
    ValueError
        If potential and relative permittivity do not use the same grid.
    """

    if potential.grid != relative_permittivity.grid:
        raise ValueError(
            "Potential and relative permittivity must use "
            "the same grid."
        )

    (
        electric_field_axis_0,
        electric_field_axis_1,
    ) = calculate_face_electric_field_components_2d(
        potential
    )

    (
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
    ) = calculate_face_relative_permittivity_components_2d(
        relative_permittivity
    )

    (
        electric_displacement_axis_0,
        electric_displacement_axis_1,
    ) = calculate_face_electric_displacement_components_2d(
        electric_field_axis_0,
        electric_field_axis_1,
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
    )

    return (
        electric_field_axis_0,
        electric_field_axis_1,
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
        electric_displacement_axis_0,
        electric_displacement_axis_1,
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

def calculate_face_electrostatic_energy_density_components_2d(
    electric_field_axis_0: FaceField,
    electric_field_axis_1: FaceField,
    electric_displacement_axis_0: FaceField,
    electric_displacement_axis_1: FaceField,
) -> tuple[
    FaceField,
    FaceField,
]:
    """
    Calculate face-centred electrostatic energy-density components
    on a two-dimensional grid.

    For linear electrostatics,

        u = 0.5 * E dot D.

    The two Cartesian contributions are retained on their natural
    face locations:

        u_0 = 0.5 * E_0 * D_0
        u_1 = 0.5 * E_1 * D_1

    The axis-0 and axis-1 quantities therefore remain separate
    FaceField objects because they have different face-centred
    shapes.
    """

    face_fields = (
        electric_field_axis_0,
        electric_field_axis_1,
        electric_displacement_axis_0,
        electric_displacement_axis_1,
    )

    if not all(
        isinstance(
            face_field,
            FaceField,
        )
        for face_field in face_fields
    ):
        raise TypeError(
            "Electric-field and electric-displacement "
            "components must all be FaceField objects."
        )

    if electric_field_axis_0.axis != 0:
        raise ValueError(
            "electric_field_axis_0 must be defined "
            "on axis-0 faces."
        )

    if electric_field_axis_1.axis != 1:
        raise ValueError(
            "electric_field_axis_1 must be defined "
            "on axis-1 faces."
        )

    if electric_displacement_axis_0.axis != 0:
        raise ValueError(
            "electric_displacement_axis_0 must be "
            "defined on axis-0 faces."
        )

    if electric_displacement_axis_1.axis != 1:
        raise ValueError(
            "electric_displacement_axis_1 must be "
            "defined on axis-1 faces."
        )

    if electric_field_axis_0.units != "V/m":
        raise ValueError(
            "electric_field_axis_0 must have units 'V/m'."
        )

    if electric_field_axis_1.units != "V/m":
        raise ValueError(
            "electric_field_axis_1 must have units 'V/m'."
        )

    if electric_displacement_axis_0.units != "C/m^2":
        raise ValueError(
            "electric_displacement_axis_0 must have "
            "units 'C/m^2'."
        )

    if electric_displacement_axis_1.units != "C/m^2":
        raise ValueError(
            "electric_displacement_axis_1 must have "
            "units 'C/m^2'."
        )

    reference_grid = electric_field_axis_0.grid

    if any(
        face_field.grid != reference_grid
        for face_field in face_fields[1:]
    ):
        raise ValueError(
            "Electric-field and electric-displacement "
            "components must use the same grid."
        )

    if (
        electric_field_axis_0.values.shape
        != electric_displacement_axis_0.values.shape
    ):
        raise ValueError(
            "Axis-0 electric-field and "
            "electric-displacement face shapes must match."
        )

    if (
        electric_field_axis_1.values.shape
        != electric_displacement_axis_1.values.shape
    ):
        raise ValueError(
            "Axis-1 electric-field and "
            "electric-displacement face shapes must match."
        )

    energy_density_axis_0_values = (
        0.5
        * electric_field_axis_0.values
        * electric_displacement_axis_0.values
    )

    energy_density_axis_1_values = (
        0.5
        * electric_field_axis_1.values
        * electric_displacement_axis_1.values
    )

    energy_density_axis_0 = FaceField(
        name="electrostatic_energy_density_axis_0_face",
        units="J/m^3",
        grid=reference_grid,
        values=energy_density_axis_0_values,
        axis=0,
    )

    energy_density_axis_1 = FaceField(
        name="electrostatic_energy_density_axis_1_face",
        units="J/m^3",
        grid=reference_grid,
        values=energy_density_axis_1_values,
        axis=1,
    )

    return (
        energy_density_axis_0,
        energy_density_axis_1,
    )

def calculate_total_electrostatic_energy_2d(
    energy_density_axis_0: FaceField,
    energy_density_axis_1: FaceField,
) -> float:
    """
    Integrate face-centred electrostatic energy density over a 2D domain.

    The returned value is electrostatic energy per unit out-of-plane
    depth, with units of J/m.

    The electrostatic energy density is decomposed into its Cartesian
    contributions,

        u = u_0 + u_1,

    where u_0 is defined on axis-0 faces and u_1 is defined on
    axis-1 faces.

    Each component is integrated on its natural staggered grid.
    Along the direction normal to a face family, the faces span the
    complete physical intervals between neighbouring nodes. Along
    the transverse direction, trapezoidal weights are used so that
    boundary nodes contribute half of the corresponding grid spacing.
    """

    if not isinstance(
        energy_density_axis_0,
        FaceField,
    ) or not isinstance(
        energy_density_axis_1,
        FaceField,
    ):
        raise TypeError(
            "Electrostatic energy-density components must "
            "be FaceField objects."
        )

    if energy_density_axis_0.axis != 0:
        raise ValueError(
            "energy_density_axis_0 must be defined "
            "on axis-0 faces."
        )

    if energy_density_axis_1.axis != 1:
        raise ValueError(
            "energy_density_axis_1 must be defined "
            "on axis-1 faces."
        )

    if energy_density_axis_0.units != "J/m^3":
        raise ValueError(
            "energy_density_axis_0 must have units 'J/m^3'."
        )

    if energy_density_axis_1.units != "J/m^3":
        raise ValueError(
            "energy_density_axis_1 must have units 'J/m^3'."
        )

    if (
        energy_density_axis_0.grid
        != energy_density_axis_1.grid
    ):
        raise ValueError(
            "Electrostatic energy-density components "
            "must use the same grid."
        )

    grid = energy_density_axis_0.grid

    if grid.dimension != 2:
        raise ValueError(
            "Two-dimensional electrostatic-energy integration "
            "requires a two-dimensional grid."
        )

    spacing_axis_0, spacing_axis_1 = grid.spacing
    number_axis_0, number_axis_1 = grid.shape

    transverse_weights_axis_1 = np.full(
        number_axis_1,
        spacing_axis_1,
        dtype=np.float64,
    )

    transverse_weights_axis_1[0] *= 0.5
    transverse_weights_axis_1[-1] *= 0.5

    transverse_weights_axis_0 = np.full(
        number_axis_0,
        spacing_axis_0,
        dtype=np.float64,
    )

    transverse_weights_axis_0[0] *= 0.5
    transverse_weights_axis_0[-1] *= 0.5

    total_energy_axis_0 = (
        spacing_axis_0
        * np.sum(
            energy_density_axis_0.values
            * transverse_weights_axis_1[
                np.newaxis,
                :
            ]
        )
    )

    total_energy_axis_1 = (
        spacing_axis_1
        * np.sum(
            energy_density_axis_1.values
            * transverse_weights_axis_0[
                :,
                np.newaxis,
            ]
        )
    )

    return float(
        total_energy_axis_0
        + total_energy_axis_1
    )