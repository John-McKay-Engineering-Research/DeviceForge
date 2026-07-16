from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deviceforge.core.field import Field

from .constants import (
    BOLTZMANN_CONSTANT,
    ROOM_TEMPERATURE,
)
from .equilibrium import (
    DEFAULT_SILICON_INTRINSIC_CONCENTRATION,
)


@dataclass(frozen=True, slots=True)
class SRHParameters:
    """
    Parameters for Shockley-Read-Hall recombination.

    Parameters
    ----------
    electron_lifetime:
        Electron capture lifetime, tau_n, in seconds.

    hole_lifetime:
        Hole capture lifetime, tau_p, in seconds.

    trap_energy_relative_to_intrinsic:
        Trap energy E_t - E_i in joules.

        A value of zero represents a mid-gap trap referenced to the
        intrinsic energy level.

    intrinsic_concentration:
        Intrinsic carrier concentration in 1/m^3.

    temperature:
        Absolute temperature in kelvin.
    """

    electron_lifetime: float = 1.0e-6
    hole_lifetime: float = 1.0e-6
    trap_energy_relative_to_intrinsic: float = 0.0
    intrinsic_concentration: float = (
        DEFAULT_SILICON_INTRINSIC_CONCENTRATION
    )
    temperature: float = ROOM_TEMPERATURE

    def __post_init__(self) -> None:
        """Validate and normalise the SRH parameters."""

        _validate_positive_finite(
            self.electron_lifetime,
            "Electron lifetime",
        )

        _validate_positive_finite(
            self.hole_lifetime,
            "Hole lifetime",
        )

        _validate_positive_finite(
            self.intrinsic_concentration,
            "Intrinsic concentration",
        )

        _validate_positive_finite(
            self.temperature,
            "Temperature",
        )

        if not np.isfinite(
            self.trap_energy_relative_to_intrinsic
        ):
            raise ValueError(
                "Trap energy must be finite."
            )

        object.__setattr__(
            self,
            "electron_lifetime",
            float(self.electron_lifetime),
        )

        object.__setattr__(
            self,
            "hole_lifetime",
            float(self.hole_lifetime),
        )

        object.__setattr__(
            self,
            "trap_energy_relative_to_intrinsic",
            float(
                self.trap_energy_relative_to_intrinsic
            ),
        )

        object.__setattr__(
            self,
            "intrinsic_concentration",
            float(self.intrinsic_concentration),
        )

        object.__setattr__(
            self,
            "temperature",
            float(self.temperature),
        )

    @property
    def electron_reference_concentration(self) -> float:
        """
        Return the SRH electron reference concentration n1.

        n1 = ni * exp((Et - Ei) / (k_B T))
        """

        exponent = (
            self.trap_energy_relative_to_intrinsic
            / (
                BOLTZMANN_CONSTANT
                * self.temperature
            )
        )

        return (
            self.intrinsic_concentration
            * float(np.exp(exponent))
        )

    @property
    def hole_reference_concentration(self) -> float:
        """
        Return the SRH hole reference concentration p1.

        p1 = ni * exp(-(Et - Ei) / (k_B T))
        """

        exponent = (
            -self.trap_energy_relative_to_intrinsic
            / (
                BOLTZMANN_CONSTANT
                * self.temperature
            )
        )

        return (
            self.intrinsic_concentration
            * float(np.exp(exponent))
        )


def compute_shockley_read_hall_rate(
    *,
    electron_concentration: Field,
    hole_concentration: Field,
    parameters: SRHParameters | None = None,
) -> Field:
    """
    Calculate net Shockley-Read-Hall recombination.

    The net SRH rate is:

        U = (n * p - ni^2)
            /
            (
                tau_p * (n + n1)
                + tau_n * (p + p1)
            )

    Sign convention
    ---------------
    Positive values represent net recombination.

    Negative values represent net carrier generation.

    Parameters
    ----------
    electron_concentration:
        Electron concentration in 1/m^3.

    hole_concentration:
        Hole concentration in 1/m^3.

    parameters:
        SRH lifetime, trap and temperature parameters.

    Returns
    -------
    Field
        Net recombination rate in 1/(m^3 s).
    """

    _validate_carrier_fields(
        electron_concentration=electron_concentration,
        hole_concentration=hole_concentration,
    )

    if parameters is None:
        parameters = SRHParameters()

    electrons = electron_concentration.values
    holes = hole_concentration.values

    intrinsic_squared = (
        parameters.intrinsic_concentration**2
    )

    numerator = (
        electrons * holes
        - intrinsic_squared
    )

    denominator = (
        parameters.hole_lifetime
        * (
            electrons
            + parameters.electron_reference_concentration
        )
        + parameters.electron_lifetime
        * (
            holes
            + parameters.hole_reference_concentration
        )
    )

    if np.any(denominator <= 0.0):
        raise ValueError(
            "The SRH recombination denominator must be positive."
        )

    recombination_values = (
        numerator / denominator
    )

    return Field(
        name="shockley_read_hall_recombination_rate",
        units="1/(m^3 s)",
        grid=electron_concentration.grid,
        values=recombination_values,
    )


def compute_net_generation_rate(
    recombination_rate: Field,
) -> Field:
    """
    Convert net recombination into the equivalent net generation rate.

    The DeviceForge SRH convention is:

        U > 0  -> net recombination
        U < 0  -> net generation

    Therefore:

        G_net = -U
    """

    if recombination_rate.units != "1/(m^3 s)":
        raise ValueError(
            "Recombination rate must use units of 1/(m^3 s)."
        )

    return Field(
        name="net_generation_rate",
        units="1/(m^3 s)",
        grid=recombination_rate.grid,
        values=-recombination_rate.values,
    )


def _validate_carrier_fields(
    *,
    electron_concentration: Field,
    hole_concentration: Field,
) -> None:
    """Validate electron and hole concentration fields."""

    if (
        electron_concentration.grid
        != hole_concentration.grid
    ):
        raise ValueError(
            "Electron and hole concentrations must use "
            "the same grid."
        )

    if electron_concentration.units != "1/m^3":
        raise ValueError(
            "Electron concentration must use units of 1/m^3."
        )

    if hole_concentration.units != "1/m^3":
        raise ValueError(
            "Hole concentration must use units of 1/m^3."
        )

    if np.any(
        electron_concentration.values < 0.0
    ):
        raise ValueError(
            "Electron concentration must not be negative."
        )

    if np.any(
        hole_concentration.values < 0.0
    ):
        raise ValueError(
            "Hole concentration must not be negative."
        )


def _validate_positive_finite(
    value: float,
    parameter_name: str,
) -> None:
    """Validate a positive finite scalar parameter."""

    if not np.isfinite(value):
        raise ValueError(
            f"{parameter_name} must be finite."
        )

    if value <= 0.0:
        raise ValueError(
            f"{parameter_name} must be positive."
        )