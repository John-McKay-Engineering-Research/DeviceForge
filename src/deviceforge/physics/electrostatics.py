from __future__ import annotations

import numpy as np

# from deviceforge.core import Device, Field
# fix to stop python pytest from failing
from deviceforge.core.device import Device
from deviceforge.core.field import Field

from .constants import ELEMENTARY_CHARGE, VACUUM_PERMITTIVITY


def compute_fixed_charge_density(
    device: Device,
) -> Field:
    """
    Calculate fixed semiconductor charge density from device doping.

    The initial DeviceForge electrostatic model assumes fully ionised dopants
    and neglects mobile electron and hole concentrations.

    The fixed charge density is:

        rho = q * (N_D - N_A)

    where:

    - rho is charge density in C/m^3
    - q is the elementary charge in C
    - N_D is donor concentration in 1/m^3
    - N_A is acceptor concentration in 1/m^3

    Positive charge density corresponds to net ionised donor concentration.
    Negative charge density corresponds to net ionised acceptor concentration.

    Parameters
    ----------
    device:
        Device containing donor and acceptor doping distributions.

    Returns
    -------
    Field
        Fixed charge-density field in C/m^3.
    """

    net_doping = device.net_doping_field()

    charge_density_values = (
        ELEMENTARY_CHARGE * net_doping.values
    )

    return Field(
        name="fixed_charge_density",
        units="C/m^3",
        grid=device.grid,
        values=charge_density_values,
    )


def compute_absolute_permittivity(
    device: Device,
) -> Field:
    """
    Calculate absolute permittivity throughout a device.

    Absolute permittivity is calculated using:

        epsilon = epsilon_0 * epsilon_r

    where:

    - epsilon_0 is vacuum permittivity
    - epsilon_r is relative permittivity

    Parameters
    ----------
    device:
        Device containing material assignments.

    Returns
    -------
    Field
        Absolute permittivity in F/m.
    """

    relative_permittivity = (
        device.relative_permittivity_field()
    )

    absolute_values = (
        VACUUM_PERMITTIVITY
        * relative_permittivity.values
    )

    return Field(
        name="absolute_permittivity",
        units="F/m",
        grid=device.grid,
        values=absolute_values,
    )


def compute_electrostatic_source_term(
    charge_density: Field,
    permittivity: Field,
) -> Field:
    """
    Calculate the electrostatic source term rho / epsilon.

    For constant permittivity, the Poisson equation may be written:

        laplacian(phi) = -rho / epsilon

    This helper prepares the source quantity used by the first Poisson solver.

    Parameters
    ----------
    charge_density:
        Charge-density field in C/m^3.

    permittivity:
        Absolute-permittivity field in F/m.

    Returns
    -------
    Field
        Electrostatic source term in V/m^2.
    """

    if charge_density.grid != permittivity.grid:
        raise ValueError(
            "Charge density and permittivity must use the same grid."
        )

    if charge_density.units != "C/m^3":
        raise ValueError(
            "Charge-density field must use units of C/m^3."
        )

    if permittivity.units != "F/m":
        raise ValueError(
            "Permittivity field must use units of F/m."
        )

    if np.any(permittivity.values <= 0.0):
        raise ValueError(
            "Permittivity values must be positive."
        )

    source_values = (
        charge_density.values
        / permittivity.values
    )

    return Field(
        name="electrostatic_source_term",
        units="V/m^2",
        grid=charge_density.grid,
        values=source_values,
    )