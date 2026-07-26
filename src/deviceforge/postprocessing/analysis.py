from __future__ import annotations

from dataclasses import dataclass

from ..core.field import Field
from ..core.result import SimulationResult
from ..core.simulation import Simulation
from .electrostatics import (
    calculate_electric_displacement_field,
    calculate_electric_field,
    calculate_electrostatic_energy_density,
)


@dataclass(frozen=True, slots=True)
class ElectrostaticAnalysis:
    """
    Complete electrostatic field dataset derived from a simulation result.

    Parameters
    ----------
    potential:
        Solved electrostatic potential in volts.

    electric_field:
        Electric field in volts per metre.

    electric_displacement:
        Electric displacement in coulombs per square metre.

    energy_density:
        Electrostatic energy density in joules per cubic metre.
    """

    potential: Field
    electric_field: Field
    electric_displacement: Field
    energy_density: Field

    def __post_init__(self) -> None:
        """Validate the electrostatic analysis fields."""

        fields = (
            self.potential,
            self.electric_field,
            self.electric_displacement,
            self.energy_density,
        )

        if any(
            not isinstance(field_value, Field)
            for field_value in fields
        ):
            raise TypeError(
                "Every electrostatic analysis value must be a Field."
            )

        reference_grid = self.potential.grid

        if any(
            field_value.grid != reference_grid
            for field_value in fields[1:]
        ):
            raise ValueError(
                "All electrostatic analysis fields must use "
                "the same grid."
            )

        expected_units = {
            "potential": "V",
            "electric_field": "V/m",
            "electric_displacement": "C/m^2",
            "energy_density": "J/m^3",
        }

        actual_units = {
            "potential": self.potential.units,
            "electric_field": self.electric_field.units,
            "electric_displacement": (
                self.electric_displacement.units
            ),
            "energy_density": self.energy_density.units,
        }

        for field_name, expected_unit in expected_units.items():
            actual_unit = actual_units[field_name]

            if actual_unit != expected_unit:
                raise ValueError(
                    f"{field_name.replace('_', ' ').title()} "
                    f"units must be '{expected_unit}'."
                )

    @property
    def grid(self):
        """Return the grid shared by all analysis fields."""

        return self.potential.grid

    @property
    def fields(self) -> tuple[Field, ...]:
        """Return all analysis fields in physical dependency order."""

        return (
            self.potential,
            self.electric_field,
            self.electric_displacement,
            self.energy_density,
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return the names of all available analysis fields."""

        return tuple(
            field_value.name
            for field_value in self.fields
        )

    def get_field(
        self,
        name: str,
    ) -> Field:
        """
        Return an analysis field by its physical field name.

        Raises
        ------
        KeyError
            If no analysis field has the requested name.
        """

        for field_value in self.fields:
            if field_value.name == name:
                return field_value

        raise KeyError(
            f"Electrostatic analysis has no field named '{name}'."
        )

    def as_dict(self) -> dict[str, Field]:
        """Return the analysis fields as a new dictionary."""

        return {
            field_value.name: field_value
            for field_value in self.fields
        }


def analyse_electrostatics(
    simulation: Simulation,
    result: SimulationResult,
) -> ElectrostaticAnalysis:
    """
    Derive the complete electrostatic field dataset.

    The analysis pipeline is

        potential
            -> electric field
            -> electric displacement
            -> electrostatic energy density

    Parameters
    ----------
    simulation:
        Simulation definition used to obtain material properties.

    result:
        Numerical simulation result containing electrostatic potential.

    Returns
    -------
    ElectrostaticAnalysis
        Immutable collection of solved and derived electrostatic fields.

    Raises
    ------
    TypeError
        If the supplied objects are not Simulation and SimulationResult
        instances.

    ValueError
        If the result grid does not match the simulation grid, the result
        potential has invalid units, or the device does not provide complete
        material coverage.

    KeyError
        If the result does not contain an electrostatic-potential field.
    """

    if not isinstance(simulation, Simulation):
        raise TypeError(
            "Electrostatic analysis requires a Simulation instance."
        )

    if not isinstance(result, SimulationResult):
        raise TypeError(
            "Electrostatic analysis requires a SimulationResult instance."
        )

    if result.grid != simulation.grid:
        raise ValueError(
            "Simulation result does not use the simulation grid."
        )

    if not simulation.device.require_full_coverage:
        raise ValueError(
            "Electrostatic analysis requires complete material "
            "coverage of the device grid."
        )

    potential = result.potential

    if potential.units != "V":
        raise ValueError(
            "Electrostatic potential units must be 'V'."
        )

    relative_permittivity = (
        simulation.device.relative_permittivity_field()
    )

    electric_field = calculate_electric_field(
        potential
    )

    electric_displacement = (
        calculate_electric_displacement_field(
            electric_field,
            relative_permittivity,
        )
    )

    energy_density = (
        calculate_electrostatic_energy_density(
            electric_field,
            relative_permittivity,
        )
    )

    return ElectrostaticAnalysis(
        potential=potential,
        electric_field=electric_field,
        electric_displacement=electric_displacement,
        energy_density=energy_density,
    )