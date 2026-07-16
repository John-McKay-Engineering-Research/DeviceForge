from __future__ import annotations

from time import perf_counter

import numpy as np

from deviceforge.core import (
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
    Initial one-dimensional stationary drift-diffusion solver.

    The solver alternates between:

    1. Poisson equation
    2. Electron continuity equation
    3. Hole continuity equation
    4. SRH recombination update

    Carrier transport uses Scharfetter-Gummel discretisation.
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
        configuration: SolverConfiguration | None = None,
    ) -> None:
        super().__init__(configuration)

        if not np.isfinite(applied_voltage):
            raise ValueError(
                "Applied voltage must be finite."
            )

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
                raise ValueError(
                    f"{name} must be positive and finite."
                )

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

        donor_field = (
            simulation.device.donor_density_field()
        )
        acceptor_field = (
            simulation.device.acceptor_density_field()
        )

        donors = donor_field.values
        acceptors = acceptor_field.values
        net_doping = donors - acceptors

        permittivity_field = (
            compute_absolute_permittivity(
                simulation.device
            )
        )
        permittivity = float(
            permittivity_field.values[0]
        )

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

        left_potential = (
            p_neutral_potential
            + self.applied_voltage
        )
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

        left_electron, left_hole = (
            self._neutral_carriers(net_doping[0])
        )
        right_electron, right_hole = (
            self._neutral_carriers(net_doping[-1])
        )

        electron_density[0] = left_electron
        electron_density[-1] = right_electron
        hole_density[0] = left_hole
        hole_density[-1] = right_hole

        residual_history: list[float] = []
        converged = False

        start_time = perf_counter()

        for _ in range(maximum_iterations):
            previous_potential = potential.copy()
            previous_electrons = electron_density.copy()
            previous_holes = hole_density.copy()

            recombination = self._compute_recombination(
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
                recombination=recombination,
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
                recombination=recombination,
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

            potential_change = float(
                np.max(
                    np.abs(
                        potential
                        - previous_potential
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

            residual = max(
                potential_change,
                electron_change,
                hole_change,
            )

            residual_history.append(residual)

            if residual <= tolerance:
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

        recombination_field = (
            compute_shockley_read_hall_rate(
                electron_concentration=electron_field,
                hole_concentration=hole_field,
                parameters=self._srh_parameters,
            )
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
                "transport_discretisation": (
                    "scharfetter_gummel"
                ),
                "recombination_model": "shockley_read_hall",
                "applied_voltage": self.applied_voltage,
                "temperature_kelvin": self._temperature,
                "tolerance": tolerance,
                "maximum_iterations": maximum_iterations,
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

        lower = bernoulli_function(
            -delta[:-1]
        )

        diagonal = -(
            bernoulli_function(delta[:-1])
            + bernoulli_function(-delta[1:])
        )

        upper = bernoulli_function(
            delta[1:]
        )

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

        lower = bernoulli_function(
            delta[:-1]
        )

        diagonal = -(
            bernoulli_function(-delta[:-1])
            + bernoulli_function(delta[1:])
        )

        upper = bernoulli_function(
            -delta[1:]
        )

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