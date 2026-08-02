from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import BoundaryCondition, BoundaryConditionType
from .device import Device
from .field import Field


@dataclass(frozen=True, slots=True)
class Simulation:
    """
    Validated electrostatic simulation definition.

    The Simulation class describes a numerical experiment. It combines a
    semiconductor device, its boundary conditions, convergence controls,
    an initial potential estimate, and an optional prescribed charge-density
    field.

    It does not implement a numerical solution algorithm. Solvers consume
    this object and return a SimulationResult.

    Parameters
    ----------
    device:
        Semiconductor device to simulate.

    boundary_conditions:
        Boundary conditions applied to the device grid.

    charge_density:
        Optional prescribed volumetric charge-density field in C/m^3.
        If omitted, the charge density is treated as zero everywhere.

    tolerance:
        Residual or solution-change threshold used to determine convergence.

    max_iterations:
        Maximum number of solver iterations.

    initial_potential:
        Initial electrostatic potential applied to all unconstrained points,
        in volts.

    name:
        Optional human-readable simulation name.
    """

    device: Device
    boundary_conditions: tuple[BoundaryCondition, ...]
    charge_density: Field | None = None
    tolerance: float = 1.0e-8
    max_iterations: int = 10_000
    initial_potential: float = 0.0
    name: str = "electrostatic_simulation"

    def __post_init__(self) -> None:
        """Validate the simulation definition."""

        if not isinstance(self.device, Device):
            raise TypeError(
                "Simulation device must be a Device instance."
            )

        if not self.name.strip():
            raise ValueError(
                "Simulation name must not be empty."
            )

        if not self.boundary_conditions:
            raise ValueError(
                "Simulation must contain at least one boundary condition."
            )

        if not np.isfinite(self.tolerance):
            raise ValueError(
                "Simulation tolerance must be finite."
            )

        if self.tolerance <= 0.0:
            raise ValueError(
                "Simulation tolerance must be positive."
            )

        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations,
            int,
        ):
            raise TypeError(
                "Maximum iteration count must be an integer."
            )

        if self.max_iterations <= 0:
            raise ValueError(
                "Maximum iteration count must be greater than zero."
            )

        if not np.isfinite(self.initial_potential):
            raise ValueError(
                "Initial potential must be finite."
            )

        boundary_names = [
            boundary.name
            for boundary in self.boundary_conditions
        ]

        if len(boundary_names) != len(set(boundary_names)):
            raise ValueError(
                "Boundary-condition names must be unique within a simulation."
            )

        for boundary in self.boundary_conditions:
            if boundary.grid != self.device.grid:
                raise ValueError(
                    f"Boundary condition '{boundary.name}' does not use "
                    "the device grid."
                )

        self._validate_charge_density()
        self._validate_overlapping_boundary_conditions()

        object.__setattr__(
            self,
            "initial_potential",
            float(self.initial_potential),
        )

    def _validate_charge_density(self) -> None:
        """Validate the optional prescribed charge-density field."""

        if self.charge_density is None:
            return

        if not isinstance(self.charge_density, Field):
            raise TypeError(
                "Charge density must be a Field instance or None."
            )

        if self.charge_density.grid != self.device.grid:
            raise ValueError(
                "Charge-density field does not use the device grid."
            )

        if self.charge_density.units != "C/m^3":
            raise ValueError(
                "Charge-density field units must be 'C/m^3'."
            )

    def _validate_overlapping_boundary_conditions(self) -> None:
        """
        Reject incompatible conditions applied to the same grid points.

        Identical conditions are permitted at shared corner points. Conflicting
        values or conflicting condition types are rejected.
        """

        boundaries = self.boundary_conditions

        for first_index, first in enumerate(boundaries):
            for second in boundaries[first_index + 1 :]:
                overlap = first.mask & second.mask

                if not np.any(overlap):
                    continue

                same_type = (
                    first.condition_type
                    is second.condition_type
                )
                same_units = (
                    first.units
                    == second.units
                )

                first_values = first.values_at(
                    overlap
                )
                second_values = second.values_at(
                    overlap
                )

                same_values = np.allclose(
                    first_values,
                    second_values,
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )

                if not (
                    same_type
                    and same_units
                    and same_values
                ):
                    overlap_count = int(
                        np.count_nonzero(overlap)
                    )

                    raise ValueError(
                        f"Boundary conditions '{first.name}' and "
                        f"'{second.name}' conflict at {overlap_count} "
                        "grid points."
                    )

    @property
    def grid(self):
        """Return the simulation grid."""

        return self.device.grid

    @property
    def number_of_boundary_conditions(self) -> int:
        """Return the number of boundary-condition definitions."""

        return len(self.boundary_conditions)

    @property
    def has_charge_density(self) -> bool:
        """Return whether a charge-density field was supplied."""

        return self.charge_density is not None

    @property
    def dirichlet_boundaries(
        self,
    ) -> tuple[BoundaryCondition, ...]:
        """Return all Dirichlet boundary conditions."""

        return tuple(
            boundary
            for boundary in self.boundary_conditions
            if boundary.condition_type
            is BoundaryConditionType.DIRICHLET
        )

    @property
    def neumann_boundaries(
        self,
    ) -> tuple[BoundaryCondition, ...]:
        """Return all Neumann boundary conditions."""

        return tuple(
            boundary
            for boundary in self.boundary_conditions
            if boundary.condition_type
            is BoundaryConditionType.NEUMANN
        )

    def get_boundary_condition(
        self,
        name: str,
    ) -> BoundaryCondition:
        """
        Return a boundary condition by name.

        Raises
        ------
        KeyError
            If no boundary condition has the requested name.
        """

        for boundary in self.boundary_conditions:
            if boundary.name == name:
                return boundary

        raise KeyError(
            f"Simulation has no boundary condition named '{name}'."
        )

    def create_initial_potential_field(self) -> Field:
        """
        Create the initial electrostatic potential field.

        The field is filled with ``initial_potential`` and then all Dirichlet
        boundary values are applied.
        """

        values = np.full(
            self.grid.shape,
            self.initial_potential,
            dtype=np.float64,
        )
        for boundary in self.dirichlet_boundaries:
            values[boundary.mask] = (
                boundary.values_on_mask()
            )

        return Field(
            name="electrostatic_potential",
            units="V",
            grid=self.grid,
            values=values,
        )

    def create_charge_density_field(self) -> Field:
        """
        Return the prescribed charge-density field.

        If no field was supplied, return an immutable zero-valued field.
        """

        if self.charge_density is not None:
            return self.charge_density

        return Field.zeros(
            name="charge_density",
            units="C/m^3",
            grid=self.grid,
        )

    def create_fixed_potential_mask(self) -> np.ndarray:
        """Return a Boolean mask selecting all Dirichlet points."""

        fixed_mask = np.zeros(
            self.grid.shape,
            dtype=np.bool_,
        )

        for boundary in self.dirichlet_boundaries:
            fixed_mask |= boundary.mask

        return fixed_mask