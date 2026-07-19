from __future__ import annotations

from time import perf_counter

import numpy as np

from deviceforge.core import (
    Field,
    Simulation,
    SimulationResult,
)
from deviceforge.physics import (
    compute_electron_scharfetter_gummel_current_x,
    compute_hole_scharfetter_gummel_current_x,
    compute_total_scharfetter_gummel_current_x,
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
    charge_neutral_potential,
    thermal_voltage,
)
from deviceforge.physics.recombination import (
    SRHParameters,
    compute_shockley_read_hall_rate,
)
from deviceforge.physics.scharfetter_gummel import (
    bernoulli_function,
)
from deviceforge.physics.transport import (
    diffusion_coefficient,
)

from .base import BaseSolver, SolverConfiguration
from .tridiagonal import solve_tridiagonal


class GummelDriftDiffusionSolver1D(BaseSolver):
    """
    One-dimensional stationary drift-diffusion solver.

    The solver alternates between:

    1. Poisson equation
    2. Electron continuity equation
    3. Hole continuity equation
    4. SRH recombination update

    Carrier transport uses Scharfetter-Gummel discretisation.

    The primary stopping rule remains the original update-based residual.
    Additional physical residuals are measured and stored as diagnostics,
    but are not yet enforced as convergence conditions. This allows their
    numerical scales to be characterised before suitable tolerances are set.
    """

    def __init__(
        self,
        *,
        applied_voltage: float = 0.0,
        damping_factor: float = 0.5,
        electron_mobility: float = 0.135,
        hole_mobility: float = 0.048,
        intrinsic_concentration: float = (
            DEFAULT_SILICON_INTRINSIC_CONCENTRATION
        ),
        temperature: float = ROOM_TEMPERATURE,
        minimum_concentration: float = 1.0,
        srh_parameters: SRHParameters | None = None,
        current_conservation_tolerance: float = 1.0e-2,
        enforce_current_conservation: bool | None = None,
        configuration: SolverConfiguration | None = None,
    ) -> None:
        super().__init__(configuration)

        if not np.isfinite(applied_voltage):
            raise ValueError("Applied voltage must be finite.")

        if (
            not np.isfinite(damping_factor)
            or damping_factor <= 0.0
            or damping_factor > 1.0
        ):
            raise ValueError(
                "Damping factor must be greater than 0 and at most 1."
            )

        for mobility, name in (
            (electron_mobility, "Electron mobility"),
            (hole_mobility, "Hole mobility"),
        ):
            if not np.isfinite(mobility) or mobility <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")

        if (
            not np.isfinite(intrinsic_concentration)
            or intrinsic_concentration <= 0.0
        ):
            raise ValueError(
                "Intrinsic concentration must be positive and finite."
            )

        if (
            not np.isfinite(minimum_concentration)
            or minimum_concentration <= 0.0
        ):
            raise ValueError(
                "Minimum concentration must be positive and finite."
            )

        if (
            not np.isfinite(current_conservation_tolerance)
            or current_conservation_tolerance <= 0.0
        ):
            raise ValueError(
                "Current-conservation tolerance must be positive and finite."
            )

        if (
            enforce_current_conservation is not None
            and not isinstance(enforce_current_conservation, (bool, np.bool_))
        ):
            raise TypeError(
                "enforce_current_conservation must be a bool or None."
            )

        self._applied_voltage = float(applied_voltage)
        self._damping_factor = float(damping_factor)
        self._electron_mobility = float(electron_mobility)
        self._hole_mobility = float(hole_mobility)
        self._intrinsic_concentration = float(
            intrinsic_concentration
        )
        self._temperature = float(temperature)
        self._minimum_concentration = float(
            minimum_concentration
        )

        self._current_conservation_tolerance = float(
            current_conservation_tolerance
        )
        self._enforce_current_conservation = (
            not np.isclose(applied_voltage, 0.0)
            if enforce_current_conservation is None
            else bool(enforce_current_conservation)
        )

        self._srh_parameters = (
            SRHParameters(
                intrinsic_concentration=intrinsic_concentration,
                temperature=temperature,
            )
            if srh_parameters is None
            else srh_parameters
        )

    @property
    def name(self) -> str:
        return "gummel_drift_diffusion_1d"

    @property
    def applied_voltage(self) -> float:
        return self._applied_voltage

    @property
    def current_conservation_tolerance(self) -> float:
        return self._current_conservation_tolerance

    @property
    def enforce_current_conservation(self) -> bool:
        return self._enforce_current_conservation

    def validate_simulation(
        self,
        simulation: Simulation,
    ) -> None:
        """
        Validate the restricted one-dimensional device model.

        BaseSolver.validate_simulation is not called because it currently
        requires a two-dimensional grid.
        """

        if not isinstance(simulation, Simulation):
            raise TypeError(
                "Solver input must be a Simulation instance."
            )

        if simulation.grid.dimension != 1:
            raise ValueError(
                "GummelDriftDiffusionSolver1D requires a "
                "one-dimensional grid."
            )

        if simulation.grid.shape[0] < 3:
            raise ValueError(
                "At least three grid nodes are required."
            )

        permittivity = compute_absolute_permittivity(
            simulation.device
        )

        if not np.allclose(
            permittivity.values,
            permittivity.values[0],
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ValueError(
                "The initial Gummel solver requires constant permittivity."
            )

    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        """Solve the stationary one-dimensional drift-diffusion system."""

        self.validate_simulation(simulation)

        tolerance = self.effective_tolerance(simulation)
        maximum_iterations = self.effective_max_iterations(
            simulation
        )

        grid = simulation.grid
        number_of_nodes = grid.shape[0]
        spacing = grid.spacing[0]

        donor_field = simulation.device.donor_density_field()
        acceptor_field = simulation.device.acceptor_density_field()

        donors = donor_field.values
        acceptors = acceptor_field.values
        net_doping = donors - acceptors

        permittivity_field = compute_absolute_permittivity(
            simulation.device
        )
        permittivity = float(permittivity_field.values[0])

        thermal = thermal_voltage(self._temperature)

        p_neutral_potential = float(
            charge_neutral_potential(
                net_doping[0],
                intrinsic_concentration=(
                    self._intrinsic_concentration
                ),
                temperature=self._temperature,
            )
        )

        n_neutral_potential = float(
            charge_neutral_potential(
                net_doping[-1],
                intrinsic_concentration=(
                    self._intrinsic_concentration
                ),
                temperature=self._temperature,
            )
        )

        left_potential = p_neutral_potential + self.applied_voltage
        right_potential = n_neutral_potential

        potential = np.linspace(
            left_potential,
            right_potential,
            number_of_nodes,
        )

        electron_density = (
            self._intrinsic_concentration
            * np.exp(
                np.clip(
                    potential / thermal,
                    -100.0,
                    100.0,
                )
            )
        )

        hole_density = (
            self._intrinsic_concentration
            * np.exp(
                np.clip(
                    -potential / thermal,
                    -100.0,
                    100.0,
                )
            )
        )

        left_electron, left_hole = self._neutral_carriers(
            net_doping[0]
        )
        right_electron, right_hole = self._neutral_carriers(
            net_doping[-1]
        )

        electron_density[0] = left_electron
        electron_density[-1] = right_electron
        hole_density[0] = left_hole
        hole_density[-1] = right_hole

        residual_history: list[float] = []
        poisson_residual_history: list[float] = []
        electron_continuity_residual_history: list[float] = []
        hole_continuity_residual_history: list[float] = []
        current_nonuniformity_history: list[float] = []
        relative_current_nonuniformity_history: list[float] = []
        maximum_electron_current_history: list[float] = []
        maximum_hole_current_history: list[float] = []
        maximum_total_current_history: list[float] = []
        electron_quasi_fermi_nonuniformity_history: list[float] = []
        hole_quasi_fermi_nonuniformity_history: list[float] = []

        converged = False
        start_time = perf_counter()

        for _ in range(maximum_iterations):
            previous_potential = potential.copy()
            previous_electrons = electron_density.copy()
            previous_holes = hole_density.copy()

            source_recombination = self._compute_recombination(
                grid=grid,
                electron_density=electron_density,
                hole_density=hole_density,
            )

            solved_potential = self._solve_poisson(
                spacing=spacing,
                permittivity=permittivity,
                donors=donors,
                acceptors=acceptors,
                electrons=electron_density,
                holes=hole_density,
                left_value=left_potential,
                right_value=right_potential,
            )

            potential = self._damped_update(
                previous_potential,
                solved_potential,
            )

            electron_solution = self._solve_electron_continuity(
                potential=potential,
                recombination=source_recombination,
                spacing=spacing,
                left_value=left_electron,
                right_value=right_electron,
            )

            electron_density = np.maximum(
                self._damped_update(
                    previous_electrons,
                    electron_solution,
                ),
                self._minimum_concentration,
            )

            hole_solution = self._solve_hole_continuity(
                potential=potential,
                recombination=source_recombination,
                spacing=spacing,
                left_value=left_hole,
                right_value=right_hole,
            )

            hole_density = np.maximum(
                self._damped_update(
                    previous_holes,
                    hole_solution,
                ),
                self._minimum_concentration,
            )

            final_recombination = self._compute_recombination(
                grid=grid,
                electron_density=electron_density,
                hole_density=hole_density,
            )

            (
                electron_current_field,
                hole_current_field,
                total_current_field,
            ) = self._compute_current_fields(
                grid=grid,
                potential=potential,
                electron_density=electron_density,
                hole_density=hole_density,
            )

            potential_change = float(
                np.max(
                    np.abs(
                        potential - previous_potential
                    )
                )
            )

            electron_change = self._relative_change(
                electron_density,
                previous_electrons,
            )

            hole_change = self._relative_change(
                hole_density,
                previous_holes,
            )

            update_residual = max(
                potential_change,
                electron_change,
                hole_change,
            )

            poisson_residual = self._poisson_residual(
                potential=potential,
                electrons=electron_density,
                holes=hole_density,
                donors=donors,
                acceptors=acceptors,
                permittivity=permittivity,
                spacing=spacing,
            )

            (
                electron_continuity_residual,
                hole_continuity_residual,
            ) = self._continuity_residuals(
                electron_current=(
                    electron_current_field.values
                ),
                hole_current=hole_current_field.values,
                recombination=final_recombination,
                spacing=spacing,
            )

            (
                current_nonuniformity,
                relative_current_nonuniformity,
            ) = self._current_nonuniformity(
                total_current_field.values
            )

            (
                electron_quasi_fermi_nonuniformity,
                hole_quasi_fermi_nonuniformity,
            ) = self._quasi_fermi_nonuniformity(
                potential=potential,
                electron_density=electron_density,
                hole_density=hole_density,
            )

            maximum_electron_current = float(
                np.max(
                    np.abs(electron_current_field.values)
                )
            )
            maximum_hole_current = float(
                np.max(
                    np.abs(hole_current_field.values)
                )
            )
            maximum_total_current = float(
                np.max(
                    np.abs(total_current_field.values)
                )
            )

            residual_history.append(update_residual)
            poisson_residual_history.append(poisson_residual)
            electron_continuity_residual_history.append(
                electron_continuity_residual
            )
            hole_continuity_residual_history.append(
                hole_continuity_residual
            )
            current_nonuniformity_history.append(
                current_nonuniformity
            )
            relative_current_nonuniformity_history.append(
                relative_current_nonuniformity
            )
            maximum_electron_current_history.append(
                maximum_electron_current
            )
            maximum_hole_current_history.append(
                maximum_hole_current
            )
            maximum_total_current_history.append(
                maximum_total_current
            )
            electron_quasi_fermi_nonuniformity_history.append(
                electron_quasi_fermi_nonuniformity
            )
            hole_quasi_fermi_nonuniformity_history.append(
                hole_quasi_fermi_nonuniformity
            )

            update_converged = update_residual <= tolerance
            current_converged = (
                relative_current_nonuniformity
                <= self._current_conservation_tolerance
            )

            if (
                update_converged
                and (
                    not self._enforce_current_conservation
                    or current_converged
                )
            ):
                converged = True
                break

        runtime_seconds = perf_counter() - start_time

        potential_field = Field(
            name="electrostatic_potential",
            units="V",
            grid=grid,
            values=potential,
        )

        electron_field = Field(
            name="electron_concentration",
            units="1/m^3",
            grid=grid,
            values=electron_density,
        )

        hole_field = Field(
            name="hole_concentration",
            units="1/m^3",
            grid=grid,
            values=hole_density,
        )

        recombination_field = compute_shockley_read_hall_rate(
            electron_concentration=electron_field,
            hole_concentration=hole_field,
            parameters=self._srh_parameters,
        )

        (
            electron_current_field,
            hole_current_field,
            total_current_field,
        ) = self._compute_current_fields(
            grid=grid,
            potential=potential,
            electron_density=electron_density,
            hole_density=hole_density,
        )

        final_poisson_residual = self._poisson_residual(
            potential=potential,
            electrons=electron_density,
            holes=hole_density,
            donors=donors,
            acceptors=acceptors,
            permittivity=permittivity,
            spacing=spacing,
        )

        (
            final_electron_continuity_residual,
            final_hole_continuity_residual,
        ) = self._continuity_residuals(
            electron_current=electron_current_field.values,
            hole_current=hole_current_field.values,
            recombination=recombination_field.values,
            spacing=spacing,
        )

        (
            current_density_nonuniformity,
            relative_current_density_nonuniformity,
        ) = self._current_nonuniformity(
            total_current_field.values
        )

        (
            electron_quasi_fermi_nonuniformity,
            hole_quasi_fermi_nonuniformity,
        ) = self._quasi_fermi_nonuniformity(
            potential=potential,
            electron_density=electron_density,
            hole_density=hole_density,
        )

        left_terminal_current_density = float(
            total_current_field.values[0]
        )
        right_terminal_current_density = float(
            total_current_field.values[-1]
        )
        average_terminal_current_density = 0.5 * (
            left_terminal_current_density
            + right_terminal_current_density
        )

        maximum_electron_current_density = float(
            np.max(np.abs(electron_current_field.values))
        )
        maximum_hole_current_density = float(
            np.max(np.abs(hole_current_field.values))
        )
        maximum_total_current_density = float(
            np.max(np.abs(total_current_field.values))
        )

        final_update_residual = (
            float(residual_history[-1])
            if residual_history
            else float("nan")
        )


        update_convergence_achieved = (
            np.isfinite(final_update_residual)
            and final_update_residual <= tolerance
        )
        current_conservation_achieved = (
            np.isfinite(relative_current_density_nonuniformity)
            and relative_current_density_nonuniformity
            <= self._current_conservation_tolerance
        )

        return SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
                "electron_concentration": electron_field,
                "hole_concentration": hole_field,
                "donor_density": donor_field,
                "acceptor_density": acceptor_field,
                "absolute_permittivity": permittivity_field,
                "shockley_read_hall_recombination_rate": (
                    recombination_field
                ),
                "electron_current_density_x_edges": (
                    electron_current_field
                ),
                "hole_current_density_x_edges": (
                    hole_current_field
                ),
                "total_current_density_x_edges": (
                    total_current_field
                ),
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
                "equation": "poisson_drift_diffusion",
                "coupling_method": "gummel_iteration",
                "transport_discretisation": "scharfetter_gummel",
                "recombination_model": "shockley_read_hall",
                "applied_voltage": self.applied_voltage,
                "temperature_kelvin": self._temperature,
                "tolerance": tolerance,
                "maximum_iterations": maximum_iterations,
                "physics_convergence_enforced": (
                    self._enforce_current_conservation
                ),
                "current_conservation_enforced": (
                    self._enforce_current_conservation
                ),
                "current_conservation_tolerance": (
                    self._current_conservation_tolerance
                ),
                "update_convergence_achieved": (
                    bool(update_convergence_achieved)
                ),
                "current_conservation_achieved": (
                    bool(current_conservation_achieved)
                ),
                "left_terminal_current_density": (
                    left_terminal_current_density
                ),
                "right_terminal_current_density": (
                    right_terminal_current_density
                ),
                "average_terminal_current_density": (
                    average_terminal_current_density
                ),
                "maximum_electron_current_density": (
                    maximum_electron_current_density
                ),
                "maximum_hole_current_density": (
                    maximum_hole_current_density
                ),
                "maximum_total_current_density": (
                    maximum_total_current_density
                ),
                "current_density_nonuniformity": (
                    current_density_nonuniformity
                ),
                "relative_current_density_nonuniformity": (
                    relative_current_density_nonuniformity
                ),
                "terminal_current_density_units": "A/m^2",
                "final_update_residual": final_update_residual,
                "final_poisson_residual": final_poisson_residual,
                "final_electron_continuity_residual": (
                    final_electron_continuity_residual
                ),
                "final_hole_continuity_residual": (
                    final_hole_continuity_residual
                ),
                "final_electron_quasi_fermi_nonuniformity": (
                    electron_quasi_fermi_nonuniformity
                ),
                "final_hole_quasi_fermi_nonuniformity": (
                    hole_quasi_fermi_nonuniformity
                ),
                "quasi_fermi_nonuniformity_units": "dimensionless",
                "poisson_residual_history": (
                    poisson_residual_history
                ),
                "electron_continuity_residual_history": (
                    electron_continuity_residual_history
                ),
                "hole_continuity_residual_history": (
                    hole_continuity_residual_history
                ),
                "current_nonuniformity_history": (
                    current_nonuniformity_history
                ),
                "relative_current_nonuniformity_history": (
                    relative_current_nonuniformity_history
                ),
                "maximum_electron_current_history": (
                    maximum_electron_current_history
                ),
                "maximum_hole_current_history": (
                    maximum_hole_current_history
                ),
                "maximum_total_current_history": (
                    maximum_total_current_history
                ),
                "electron_quasi_fermi_nonuniformity_history": (
                    electron_quasi_fermi_nonuniformity_history
                ),
                "hole_quasi_fermi_nonuniformity_history": (
                    hole_quasi_fermi_nonuniformity_history
                ),
            },
        )

    def _solve_poisson(
        self,
        *,
        spacing: float,
        permittivity: float,
        donors: np.ndarray,
        acceptors: np.ndarray,
        electrons: np.ndarray,
        holes: np.ndarray,
        left_value: float,
        right_value: float,
    ) -> np.ndarray:
        interior_size = donors.size - 2

        charge_density = ELEMENTARY_CHARGE * (
            holes
            - electrons
            + donors
            - acceptors
        )

        lower = np.ones(interior_size - 1)
        diagonal = -2.0 * np.ones(interior_size)
        upper = np.ones(interior_size - 1)

        rhs = (
            -(spacing**2)
            * charge_density[1:-1]
            / permittivity
        )

        rhs[0] -= left_value
        rhs[-1] -= right_value

        interior = solve_tridiagonal(
            lower,
            diagonal,
            upper,
            rhs,
        )

        solution = np.empty(donors.size)
        solution[0] = left_value
        solution[-1] = right_value
        solution[1:-1] = interior

        return solution

    def _solve_electron_continuity(
        self,
        *,
        potential: np.ndarray,
        recombination: np.ndarray,
        spacing: float,
        left_value: float,
        right_value: float,
    ) -> np.ndarray:
        diffusivity = diffusion_coefficient(
            self._electron_mobility,
            temperature=self._temperature,
        )

        delta = np.diff(potential) / thermal_voltage(
            self._temperature
        )

        interior_size = potential.size - 2

        lower = bernoulli_function(-delta[:-1])

        diagonal = -(
            bernoulli_function(delta[:-1])
            + bernoulli_function(-delta[1:])
        )

        upper = bernoulli_function(delta[1:])

        rhs = (
            spacing**2
            * recombination[1:-1]
            / diffusivity
        )

        rhs[0] -= lower[0] * left_value
        rhs[-1] -= upper[-1] * right_value

        interior = solve_tridiagonal(
            lower[1:],
            diagonal,
            upper[:-1],
            rhs,
        )

        solution = np.empty(potential.size)
        solution[0] = left_value
        solution[-1] = right_value
        solution[1:-1] = interior

        return solution

    def _solve_hole_continuity(
        self,
        *,
        potential: np.ndarray,
        recombination: np.ndarray,
        spacing: float,
        left_value: float,
        right_value: float,
    ) -> np.ndarray:
        diffusivity = diffusion_coefficient(
            self._hole_mobility,
            temperature=self._temperature,
        )

        delta = np.diff(potential) / thermal_voltage(
            self._temperature
        )

        lower = bernoulli_function(delta[:-1])

        diagonal = -(
            bernoulli_function(-delta[:-1])
            + bernoulli_function(delta[1:])
        )

        upper = bernoulli_function(-delta[1:])

        rhs = (
            spacing**2
            * recombination[1:-1]
            / diffusivity
        )

        rhs[0] -= lower[0] * left_value
        rhs[-1] -= upper[-1] * right_value

        interior = solve_tridiagonal(
            lower[1:],
            diagonal,
            upper[:-1],
            rhs,
        )

        solution = np.empty(potential.size)
        solution[0] = left_value
        solution[-1] = right_value
        solution[1:-1] = interior

        return solution

    def _compute_recombination(
        self,
        *,
        grid,
        electron_density: np.ndarray,
        hole_density: np.ndarray,
    ) -> np.ndarray:
        electron_field = Field(
            name="electron_concentration",
            units="1/m^3",
            grid=grid,
            values=electron_density,
        )

        hole_field = Field(
            name="hole_concentration",
            units="1/m^3",
            grid=grid,
            values=hole_density,
        )

        return compute_shockley_read_hall_rate(
            electron_concentration=electron_field,
            hole_concentration=hole_field,
            parameters=self._srh_parameters,
        ).values

    def _compute_current_fields(
        self,
        *,
        grid,
        potential: np.ndarray,
        electron_density: np.ndarray,
        hole_density: np.ndarray,
    ) -> tuple[Field, Field, Field]:
        """Create the electron, hole and total edge-current fields."""

        potential_field = Field(
            name="electrostatic_potential",
            units="V",
            grid=grid,
            values=potential,
        )

        electron_field = Field(
            name="electron_concentration",
            units="1/m^3",
            grid=grid,
            values=electron_density,
        )

        hole_field = Field(
            name="hole_concentration",
            units="1/m^3",
            grid=grid,
            values=hole_density,
        )

        electron_current_field = (
            compute_electron_scharfetter_gummel_current_x(
                potential=potential_field,
                electron_concentration=electron_field,
                mobility=self._electron_mobility,
                temperature=self._temperature,
            )
        )

        hole_current_field = (
            compute_hole_scharfetter_gummel_current_x(
                potential=potential_field,
                hole_concentration=hole_field,
                mobility=self._hole_mobility,
                temperature=self._temperature,
            )
        )

        total_current_field = (
            compute_total_scharfetter_gummel_current_x(
                electron_current_density=electron_current_field,
                hole_current_density=hole_current_field,
            )
        )

        return (
            electron_current_field,
            hole_current_field,
            total_current_field,
        )

    @staticmethod
    def _poisson_residual(
        *,
        potential: np.ndarray,
        electrons: np.ndarray,
        holes: np.ndarray,
        donors: np.ndarray,
        acceptors: np.ndarray,
        permittivity: float,
        spacing: float,
    ) -> float:
        """Return a dimensionless interior Poisson-equation residual."""

        charge_density = ELEMENTARY_CHARGE * (
            holes
            - electrons
            + donors
            - acceptors
        )

        laplacian = (
            potential[:-2]
            - 2.0 * potential[1:-1]
            + potential[2:]
        ) / spacing**2

        electrostatic_term = permittivity * laplacian
        interior_charge = charge_density[1:-1]
        defect = electrostatic_term + interior_charge

        scale = max(
            float(np.max(np.abs(electrostatic_term))),
            float(np.max(np.abs(interior_charge))),
            np.finfo(np.float64).tiny,
        )

        return float(np.max(np.abs(defect)) / scale)

    @staticmethod
    def _continuity_residuals(
        *,
        electron_current: np.ndarray,
        hole_current: np.ndarray,
        recombination: np.ndarray,
        spacing: float,
    ) -> tuple[float, float]:
        """Return dimensionless electron and hole continuity residuals."""

        electron_divergence = (
            electron_current[1:] - electron_current[:-1]
        ) / spacing

        hole_divergence = (
            hole_current[1:] - hole_current[:-1]
        ) / spacing

        recombination_current_source = (
            ELEMENTARY_CHARGE * recombination[1:-1]
        )

        electron_defect = (
            electron_divergence
            - recombination_current_source
        )
        hole_defect = (
            hole_divergence
            + recombination_current_source
        )

        tiny = np.finfo(np.float64).tiny

        electron_scale = max(
            float(np.max(np.abs(electron_divergence))),
            float(
                np.max(
                    np.abs(recombination_current_source)
                )
            ),
            tiny,
        )

        hole_scale = max(
            float(np.max(np.abs(hole_divergence))),
            float(
                np.max(
                    np.abs(recombination_current_source)
                )
            ),
            tiny,
        )

        return (
            float(
                np.max(np.abs(electron_defect))
                / electron_scale
            ),
            float(
                np.max(np.abs(hole_defect))
                / hole_scale
            ),
        )

    @staticmethod
    def _current_nonuniformity(
        total_current: np.ndarray,
    ) -> tuple[float, float]:
        """Return absolute and relative total-current nonuniformity.

        Absolute nonuniformity is the full range of the total-current field:

            max(J_total) - min(J_total)

        Relative nonuniformity is this range divided by the largest absolute
        total-current value. This preserves the established DeviceForge
        metadata definition and the corresponding unit-test contract.
        """

        maximum_current = float(np.max(total_current))
        minimum_current = float(np.min(total_current))

        absolute_nonuniformity = (
            maximum_current - minimum_current
        )

        current_scale = max(
            float(np.max(np.abs(total_current))),
            np.finfo(np.float64).tiny,
        )

        relative_nonuniformity = (
            absolute_nonuniformity / current_scale
        )

        return (
            float(absolute_nonuniformity),
            float(relative_nonuniformity),
        )

    def _quasi_fermi_nonuniformity(
        self,
        *,
        potential: np.ndarray,
        electron_density: np.ndarray,
        hole_density: np.ndarray,
    ) -> tuple[float, float]:
        """
        Return dimensionless quasi-Fermi nonuniformities.

        At thermal equilibrium both quantities should be spatially constant.
        They are diagnostics only because quasi-Fermi levels are generally
        expected to vary under applied bias.
        """

        voltage = thermal_voltage(self._temperature)

        safe_electrons = np.maximum(
            electron_density,
            self._minimum_concentration,
        )
        safe_holes = np.maximum(
            hole_density,
            self._minimum_concentration,
        )

        electron_quasi_fermi = (
            np.log(
                safe_electrons
                / self._intrinsic_concentration
            )
            - potential / voltage
        )

        hole_quasi_fermi = (
            np.log(
                safe_holes
                / self._intrinsic_concentration
            )
            + potential / voltage
        )

        electron_nonuniformity = float(
            np.max(electron_quasi_fermi)
            - np.min(electron_quasi_fermi)
        )
        hole_nonuniformity = float(
            np.max(hole_quasi_fermi)
            - np.min(hole_quasi_fermi)
        )

        return (
            electron_nonuniformity,
            hole_nonuniformity,
        )

    def _neutral_carriers(
        self,
        net_doping: float,
    ) -> tuple[float, float]:
        neutral_potential = float(
            charge_neutral_potential(
                net_doping,
                intrinsic_concentration=(
                    self._intrinsic_concentration
                ),
                temperature=self._temperature,
            )
        )

        voltage = thermal_voltage(self._temperature)

        electrons = (
            self._intrinsic_concentration
            * np.exp(neutral_potential / voltage)
        )

        holes = (
            self._intrinsic_concentration
            * np.exp(-neutral_potential / voltage)
        )

        return float(electrons), float(holes)

    def _damped_update(
        self,
        previous: np.ndarray,
        solved: np.ndarray,
    ) -> np.ndarray:
        return (
            previous
            + self._damping_factor
            * (solved - previous)
        )

    @staticmethod
    def _relative_change(
        current: np.ndarray,
        previous: np.ndarray,
    ) -> float:
        scale = np.maximum(
            np.abs(previous),
            1.0,
        )

        return float(
            np.max(
                np.abs(current - previous)
                / scale
            )
        )