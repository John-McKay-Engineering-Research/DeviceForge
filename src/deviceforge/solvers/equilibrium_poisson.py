from __future__ import annotations

from time import perf_counter

import numpy as np

from deviceforge.core import (
    BoundaryCondition,
    Field,
    Simulation,
    SimulationResult,
)
from deviceforge.physics.constants import (
    ELEMENTARY_CHARGE,
    ROOM_TEMPERATURE,
)
from deviceforge.physics.electrostatics import (
    compute_absolute_permittivity,
)
from deviceforge.physics.equilibrium import (
    DEFAULT_SILICON_INTRINSIC_CONCENTRATION,
    compute_equilibrium_charge_density,
    thermal_voltage,
)

from .base import BaseSolver, SolverConfiguration


class EquilibriumPoissonSolver(BaseSolver):
    """
    Nonlinear equilibrium semiconductor Poisson solver.

    The solver uses damped pointwise Newton updates with Gauss-Seidel
    ordering. Mobile electron and hole concentrations are calculated using
    non-degenerate Boltzmann statistics.

    The nonlinear equation is:

        laplacian(phi) + rho(phi) / epsilon = 0

    where:

        rho = q * (p - n + N_D - N_A)

        n = n_i * exp(phi / V_T)

        p = n_i * exp(-phi / V_T)

    Current limitations:

    - two-dimensional structured grids
    - equal grid spacing
    - spatially constant permittivity
    - fully ionised dopants
    - Boltzmann carrier statistics
    - homogeneous Neumann boundaries
    - NumPy/Python execution
    """

    def __init__(
        self,
        *,
        damping_factor: float = 0.5,
        maximum_potential_step: float = 0.1,
        intrinsic_concentration: float = (
            DEFAULT_SILICON_INTRINSIC_CONCENTRATION
        ),
        temperature: float = ROOM_TEMPERATURE,
        maximum_normalised_potential: float = 100.0,
        configuration: SolverConfiguration | None = None,
    ) -> None:
        super().__init__(configuration)

        if not np.isfinite(damping_factor):
            raise ValueError("Damping factor must be finite.")

        if damping_factor <= 0.0 or damping_factor > 1.0:
            raise ValueError(
                "Damping factor must be greater than 0 and at most 1."
            )

        if not np.isfinite(maximum_potential_step):
            raise ValueError(
                "Maximum potential step must be finite."
            )

        if maximum_potential_step <= 0.0:
            raise ValueError(
                "Maximum potential step must be positive."
            )

        if not np.isfinite(intrinsic_concentration):
            raise ValueError(
                "Intrinsic concentration must be finite."
            )

        if intrinsic_concentration <= 0.0:
            raise ValueError(
                "Intrinsic concentration must be positive."
            )

        if not np.isfinite(temperature):
            raise ValueError("Temperature must be finite.")

        if temperature <= 0.0:
            raise ValueError(
                "Temperature must be greater than zero kelvin."
            )

        if not np.isfinite(maximum_normalised_potential):
            raise ValueError(
                "Maximum normalised potential must be finite."
            )

        if maximum_normalised_potential <= 0.0:
            raise ValueError(
                "Maximum normalised potential must be positive."
            )

        self._damping_factor = float(damping_factor)
        self._maximum_potential_step = float(
            maximum_potential_step
        )
        self._intrinsic_concentration = float(
            intrinsic_concentration
        )
        self._temperature = float(temperature)
        self._maximum_normalised_potential = float(
            maximum_normalised_potential
        )

    @property
    def name(self) -> str:
        """Return the public solver name."""

        return "equilibrium_poisson"

    @property
    def damping_factor(self) -> float:
        """Return the nonlinear damping factor."""

        return self._damping_factor

    @property
    def maximum_potential_step(self) -> float:
        """Return the maximum undamped Newton step in volts."""

        return self._maximum_potential_step

    @property
    def intrinsic_concentration(self) -> float:
        """Return the intrinsic carrier concentration."""

        return self._intrinsic_concentration

    @property
    def temperature(self) -> float:
        """Return the simulation temperature."""

        return self._temperature

    def validate_simulation(
        self,
        simulation: Simulation,
    ) -> None:
        """Validate that the simulation is supported."""

        super().validate_simulation(simulation)

        spacing_x, spacing_y = simulation.grid.spacing

        if not np.isclose(
            spacing_x,
            spacing_y,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                "EquilibriumPoissonSolver currently requires equal "
                "grid spacing in both spatial dimensions."
            )

        for boundary in simulation.neumann_boundaries:
            if not np.isclose(
                boundary.value,
                0.0,
                rtol=0.0,
                atol=1.0e-15,
            ):
                raise ValueError(
                    "EquilibriumPoissonSolver currently supports only "
                    "homogeneous Neumann boundary conditions."
                )

        permittivity = compute_absolute_permittivity(
            simulation.device
        )

        reference_permittivity = float(
            permittivity.values.flat[0]
        )

        if not np.allclose(
            permittivity.values,
            reference_permittivity,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                "EquilibriumPoissonSolver currently requires "
                "spatially constant permittivity."
            )

    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        """
        Solve the nonlinear equilibrium Poisson equation.

        Convergence is currently based on the maximum applied potential
        update during one nonlinear sweep.
        """

        self.validate_simulation(simulation)

        tolerance = self.effective_tolerance(simulation)
        max_iterations = self.effective_max_iterations(
            simulation
        )

        initial_field = (
            simulation.create_initial_potential_field()
        )

        potential = initial_field.values.copy()

        donor_density = (
            simulation.device.donor_density_field()
        )

        acceptor_density = (
            simulation.device.acceptor_density_field()
        )

        permittivity = compute_absolute_permittivity(
            simulation.device
        )

        absolute_permittivity = float(
            permittivity.values.flat[0]
        )

        voltage = thermal_voltage(self.temperature)

        spacing = simulation.grid.spacing[0]
        spacing_squared = spacing**2

        rows, columns = potential.shape

        residual_history: list[float] = []
        converged = False

        start_time = perf_counter()

        for _ in range(max_iterations):
            maximum_update = 0.0

            for row in range(1, rows - 1):
                for column in range(1, columns - 1):
                    current_potential = potential[
                        row,
                        column,
                    ]

                    normalised_potential = np.clip(
                        current_potential / voltage,
                        -self._maximum_normalised_potential,
                        self._maximum_normalised_potential,
                    )

                    electron_concentration = (
                        self.intrinsic_concentration
                        * np.exp(normalised_potential)
                    )

                    hole_concentration = (
                        self.intrinsic_concentration
                        * np.exp(-normalised_potential)
                    )

                    charge_density = ELEMENTARY_CHARGE * (
                        hole_concentration
                        - electron_concentration
                        + donor_density.values[row, column]
                        - acceptor_density.values[row, column]
                    )

                    neighbour_sum = (
                        potential[row + 1, column]
                        + potential[row - 1, column]
                        + potential[row, column + 1]
                        + potential[row, column - 1]
                    )

                    laplacian = (
                        neighbour_sum
                        - 4.0 * current_potential
                    ) / spacing_squared

                    equation_residual = (
                        laplacian
                        + charge_density
                        / absolute_permittivity
                    )

                    charge_derivative = (
                        -ELEMENTARY_CHARGE
                        * (
                            electron_concentration
                            + hole_concentration
                        )
                        / voltage
                    )

                    diagonal_derivative = (
                        -4.0 / spacing_squared
                        + charge_derivative
                        / absolute_permittivity
                    )

                    newton_step = (
                        -equation_residual
                        / diagonal_derivative
                    )

                    limited_step = float(
                        np.clip(
                            newton_step,
                            -self.maximum_potential_step,
                            self.maximum_potential_step,
                        )
                    )

                    applied_update = (
                        self.damping_factor
                        * limited_step
                    )

                    potential[row, column] = (
                        current_potential
                        + applied_update
                    )

                    maximum_update = max(
                        maximum_update,
                        abs(applied_update),
                    )

            self._apply_homogeneous_neumann_boundaries(
                potential=potential,
                boundaries=simulation.neumann_boundaries,
            )

            self._apply_dirichlet_boundaries(
                potential=potential,
                boundaries=simulation.dirichlet_boundaries,
            )

            residual_history.append(maximum_update)

            if maximum_update <= tolerance:
                converged = True
                break

        runtime_seconds = perf_counter() - start_time

        solved_potential = Field(
            name="electrostatic_potential",
            units="V",
            grid=simulation.grid,
            values=potential,
        )

        (
            equilibrium_charge_density,
            electron_density,
            hole_density,
        ) = compute_equilibrium_charge_density(
            potential=solved_potential,
            donor_density=donor_density,
            acceptor_density=acceptor_density,
            intrinsic_concentration=(
                self.intrinsic_concentration
            ),
            temperature=self.temperature,
        )

        return SimulationResult(
            fields={
                "electrostatic_potential": solved_potential,
                "equilibrium_charge_density": (
                    equilibrium_charge_density
                ),
                "electron_concentration": electron_density,
                "hole_concentration": hole_density,
                "donor_density": donor_density,
                "acceptor_density": acceptor_density,
                "absolute_permittivity": permittivity,
            },
            converged=converged,
            iterations=len(residual_history),
            residual_history=np.asarray(
                residual_history,
                dtype=np.float64,
            ),
            runtime_seconds=runtime_seconds,
            solver_name=self.name,
            backend_name=self.backend_name,
            metadata={
                "equation": "nonlinear_equilibrium_poisson",
                "nonlinear_method": (
                    "damped_pointwise_newton_gauss_seidel"
                ),
                "carrier_statistics": "boltzmann",
                "temperature_kelvin": self.temperature,
                "thermal_voltage": voltage,
                "intrinsic_concentration": (
                    self.intrinsic_concentration
                ),
                "damping_factor": self.damping_factor,
                "maximum_potential_step": (
                    self.maximum_potential_step
                ),
                "tolerance": tolerance,
                "maximum_iterations": max_iterations,
                "grid_shape": simulation.grid.shape,
                "grid_spacing": simulation.grid.spacing,
            },
        )

    @staticmethod
    def _apply_dirichlet_boundaries(
        *,
        potential: np.ndarray,
        boundaries: tuple[BoundaryCondition, ...],
    ) -> None:
        """Reapply fixed-potential boundary values."""

        for boundary in boundaries:
            potential[boundary.mask] = boundary.value

    @staticmethod
    def _apply_homogeneous_neumann_boundaries(
        *,
        potential: np.ndarray,
        boundaries: tuple[BoundaryCondition, ...],
    ) -> None:
        """Apply zero-normal-gradient Neumann boundaries."""

        rows, columns = potential.shape

        for boundary in boundaries:
            indices = np.argwhere(boundary.mask)

            for row, column in indices:
                if row == 0:
                    potential[row, column] = potential[
                        row + 1,
                        column,
                    ]

                elif row == rows - 1:
                    potential[row, column] = potential[
                        row - 1,
                        column,
                    ]

                elif column == 0:
                    potential[row, column] = potential[
                        row,
                        column + 1,
                    ]

                elif column == columns - 1:
                    potential[row, column] = potential[
                        row,
                        column - 1,
                    ]