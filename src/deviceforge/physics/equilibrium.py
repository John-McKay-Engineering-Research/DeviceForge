from __future__ import annotations

import numpy as np

from deviceforge.core.field import Field

from .constants import (
    BOLTZMANN_CONSTANT,
    ELEMENTARY_CHARGE,
    ROOM_TEMPERATURE,
)


DEFAULT_SILICON_INTRINSIC_CONCENTRATION: float = 1.0e16
"""
Initial silicon intrinsic carrier concentration in 1/m^3.

This corresponds to approximately 1e10 cm^-3. The value is currently
treated as a configurable modelling parameter rather than a complete
temperature-dependent material model.
"""


def thermal_voltage(
    temperature: float = ROOM_TEMPERATURE,
) -> float:
    """
    Calculate the thermal voltage.

    The thermal voltage is:

        V_T = k_B T / q

    Parameters
    ----------
    temperature:
        Absolute temperature in kelvin.

    Returns
    -------
    float
        Thermal voltage in volts.
    """

    temperature = float(temperature)

    if not np.isfinite(temperature):
        raise ValueError("Temperature must be finite.")

    if temperature <= 0.0:
        raise ValueError(
            "Temperature must be greater than zero kelvin."
        )

    return (
        BOLTZMANN_CONSTANT
        * temperature
        / ELEMENTARY_CHARGE
    )


def equilibrium_carrier_concentrations(
    potential: Field,
    *,
    intrinsic_concentration: float = (
        DEFAULT_SILICON_INTRINSIC_CONCENTRATION
    ),
    temperature: float = ROOM_TEMPERATURE,
    maximum_normalised_potential: float = 100.0,
) -> tuple[Field, Field]:
    """
    Calculate equilibrium electron and hole concentrations.

    Under the non-degenerate Boltzmann approximation:

        n = n_i exp(phi / V_T)
        p = n_i exp(-phi / V_T)

    The potential is referenced to the intrinsic electrostatic state.

    Parameters
    ----------
    potential:
        Electrostatic-potential field in volts.

    intrinsic_concentration:
        Intrinsic carrier concentration in inverse cubic metres.

    temperature:
        Absolute temperature in kelvin.

    maximum_normalised_potential:
        Numerical clipping limit applied to phi / V_T to avoid floating-point
        overflow during nonlinear iterations.

    Returns
    -------
    tuple[Field, Field]
        Electron and hole concentration fields.
    """

    if potential.units != "V":
        raise ValueError(
            "Electrostatic potential must use units of V."
        )

    intrinsic_concentration = float(
        intrinsic_concentration
    )

    if not np.isfinite(intrinsic_concentration):
        raise ValueError(
            "Intrinsic concentration must be finite."
        )

    if intrinsic_concentration <= 0.0:
        raise ValueError(
            "Intrinsic concentration must be positive."
        )

    maximum_normalised_potential = float(
        maximum_normalised_potential
    )

    if not np.isfinite(maximum_normalised_potential):
        raise ValueError(
            "Maximum normalised potential must be finite."
        )

    if maximum_normalised_potential <= 0.0:
        raise ValueError(
            "Maximum normalised potential must be positive."
        )

    voltage = thermal_voltage(temperature)

    normalised_potential = np.clip(
        potential.values / voltage,
        -maximum_normalised_potential,
        maximum_normalised_potential,
    )

    electron_values = (
        intrinsic_concentration
        * np.exp(normalised_potential)
    )

    hole_values = (
        intrinsic_concentration
        * np.exp(-normalised_potential)
    )

    electrons = Field(
        name="electron_concentration",
        units="1/m^3",
        grid=potential.grid,
        values=electron_values,
    )

    holes = Field(
        name="hole_concentration",
        units="1/m^3",
        grid=potential.grid,
        values=hole_values,
    )

    return electrons, holes


def compute_equilibrium_charge_density(
    *,
    potential: Field,
    donor_density: Field,
    acceptor_density: Field,
    intrinsic_concentration: float = (
        DEFAULT_SILICON_INTRINSIC_CONCENTRATION
    ),
    temperature: float = ROOM_TEMPERATURE,
) -> tuple[Field, Field, Field]:
    """
    Calculate nonlinear equilibrium semiconductor charge density.

    The total charge density is:

        rho = q * (p - n + N_D - N_A)

    Parameters
    ----------
    potential:
        Electrostatic potential in volts.

    donor_density:
        Donor concentration in inverse cubic metres.

    acceptor_density:
        Acceptor concentration in inverse cubic metres.

    intrinsic_concentration:
        Intrinsic carrier concentration in inverse cubic metres.

    temperature:
        Absolute temperature in kelvin.

    Returns
    -------
    tuple[Field, Field, Field]
        Charge density, electron concentration and hole concentration.
    """

    fields = (
        potential,
        donor_density,
        acceptor_density,
    )

    if any(
        field.grid != potential.grid
        for field in fields
    ):
        raise ValueError(
            "Potential and doping fields must use the same grid."
        )

    if donor_density.units != "1/m^3":
        raise ValueError(
            "Donor-density field must use units of 1/m^3."
        )

    if acceptor_density.units != "1/m^3":
        raise ValueError(
            "Acceptor-density field must use units of 1/m^3."
        )

    if np.any(donor_density.values < 0.0):
        raise ValueError(
            "Donor concentration must not be negative."
        )

    if np.any(acceptor_density.values < 0.0):
        raise ValueError(
            "Acceptor concentration must not be negative."
        )

    electrons, holes = (
        equilibrium_carrier_concentrations(
            potential,
            intrinsic_concentration=(
                intrinsic_concentration
            ),
            temperature=temperature,
        )
    )

    charge_values = ELEMENTARY_CHARGE * (
        holes.values
        - electrons.values
        + donor_density.values
        - acceptor_density.values
    )

    charge_density = Field(
        name="equilibrium_charge_density",
        units="C/m^3",
        grid=potential.grid,
        values=charge_values,
    )

    return charge_density, electrons, holes


def charge_neutral_potential(
    net_doping_density: np.ndarray | float,
    *,
    intrinsic_concentration: float = (
        DEFAULT_SILICON_INTRINSIC_CONCENTRATION
    ),
    temperature: float = ROOM_TEMPERATURE,
) -> np.ndarray:
    """
    Calculate the equilibrium potential required for local charge neutrality.

    For net doping:

        C = N_D - N_A

    neutrality requires:

        p - n + C = 0

    Together with:

        n p = n_i^2

    this gives the dimensionless potential:

        phi / V_T = asinh(C / (2 n_i))

    Parameters
    ----------
    net_doping_density:
        Signed net doping in inverse cubic metres.

    intrinsic_concentration:
        Intrinsic carrier concentration in inverse cubic metres.

    temperature:
        Absolute temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        Charge-neutral potential in volts.
    """

    net_doping = np.asarray(
        net_doping_density,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(net_doping)):
        raise ValueError(
            "Net doping density must contain only finite values."
        )

    intrinsic_concentration = float(
        intrinsic_concentration
    )

    if not np.isfinite(intrinsic_concentration):
        raise ValueError(
            "Intrinsic concentration must be finite."
        )

    if intrinsic_concentration <= 0.0:
        raise ValueError(
            "Intrinsic concentration must be positive."
        )

    return thermal_voltage(temperature) * np.arcsinh(
        net_doping
        / (2.0 * intrinsic_concentration)
    )