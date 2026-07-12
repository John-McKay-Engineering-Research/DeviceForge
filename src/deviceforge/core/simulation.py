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
    semiconductor device, its boundary conditions, convergence controls and
    an initial potential estimate.

    It does not yet implement a numerical solution algorithm. Solvers will
    consume this object and return a SimulationResult.

    Parameters
    ----------
    device:
        Semiconductor device to simulate.

    boundary_conditions:
        Boundary conditions applied to the device grid.

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
    tolerance: float = 1.0e-8
    max_iterations: int = 10_000
    initial_potential: float = 0.0
    name: str = "electrostatic_simulation"

    def __post_init__(self) -> None:
        """Validate the simulation definition."""

        if not self.name.strip():
            raise ValueError("Simulation name must not be empty.")

        if not self.boundary_conditions:
            raise ValueError(
                "Simulation must contain at least one boundary condition."
            )

        if not np.isfinite(self.tolerance):
            raise ValueError("Simulation tolerance must be finite.")

        if self.tolerance <= 0.0:
            raise ValueError("Simulation tolerance must be positive.")

        if isinstance(self.max_iterations, bool) or not isinstance(
            self.max_iterations,
            int,
        ):
            raise TypeError("Maximum iteration count must be an integer.")

        if self.max_iterations <= 0:
            raise ValueError(
                "Maximum iteration count must be greater than zero."
            )

        if not np.isfinite(self.initial_potential):
            raise ValueError("Initial potential must be finite.")

        boundary_names = [
            boundary.name for boundary in self.boundary_conditions
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

        self._validate_overlapping_boundary_conditions()

        object.__setattr__(
            self,
            "initial_potential",
            float(self.initial_potential),
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

                same_type = first.condition_type is second.condition_type
                same_units = first.units == second.units
                same_value = np.isclose(
                    first.value,
                    second.value,
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )

                if not (same_type and same_units and same_value):
                    overlap_count = int(np.count_nonzero(overlap))

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
            values[boundary.mask] = boundary.value

        return Field(
            name="electrostatic_potential",
            units="V",
            grid=self.grid,
            values=values,
        )

    def create_fixed_potential_mask(self) -> np.ndarray:
        """
        Return a Boolean mask selecting all Dirichlet boundary points.
        """

        fixed_mask = np.zeros(
            self.grid.shape,
            dtype=np.bool_,
        )

        for boundary in self.dirichlet_boundaries:
            fixed_mask |= boundary.mask

        return fixed_mask