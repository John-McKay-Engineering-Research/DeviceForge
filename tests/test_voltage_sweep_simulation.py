"""Tests for the DeviceForge voltage-sweep simulation adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from deviceforge.analysis.sweep_result import VoltageSweepResult
from deviceforge.analysis.voltage_sweep import VoltageSweep
from deviceforge.analysis.voltage_sweep_simulation import (
    VoltageSweepSimulation,
    VoltageSweepSimulationAdapter,
)


@dataclass
class AdapterSolution:
    """Minimal solver-specific solution used by adapter tests."""

    voltage: float
    initial_guess: object | None
    converged: bool = True


class ExistingSimulation:
    """Example simulation with an interface incompatible with VoltageSweep."""

    def __init__(self) -> None:
        self.contact_values: dict[str, float] = {}
        self.solve_inputs: list[object | None] = []
        self.solutions: list[AdapterSolution] = []

    def update_terminal(
        self,
        terminal_name: str,
        terminal_voltage: float,
    ) -> None:
        """Set a terminal voltage using a non-standard method name."""

        self.contact_values[terminal_name] = terminal_voltage

    def run_solver(
        self,
        previous_solution: object | None = None,
    ) -> AdapterSolution:
        """Solve using a non-standard method name and argument."""

        if not self.contact_values:
            raise RuntimeError("No terminal voltage has been configured.")

        voltage = next(reversed(self.contact_values.values()))

        self.solve_inputs.append(previous_solution)

        solution = AdapterSolution(
            voltage=voltage,
            initial_guess=previous_solution,
        )

        self.solutions.append(solution)

        return solution


def create_voltage_sweep_result(
    simulation: ExistingSimulation,
    voltage: float,
    solution: AdapterSolution,
) -> VoltageSweepResult:
    """Convert an adapter solution into a standard sweep result."""

    potential = np.array(
        [
            voltage,
            voltage + 0.1,
            voltage + 0.2,
        ],
        dtype=float,
    )

    electron_density = np.array(
        [1.0e16, 1.1e16, 1.2e16],
        dtype=float,
    )

    hole_density = np.array(
        [1.2e16, 1.1e16, 1.0e16],
        dtype=float,
    )

    electric_field = np.array(
        [-1.0, -1.0, -1.0],
        dtype=float,
    )

    current = voltage * 1.0e-6

    electron_current_density = np.full(
        3,
        0.8 * current,
        dtype=float,
    )

    hole_current_density = np.full(
        3,
        0.2 * current,
        dtype=float,
    )

    total_current_density = (
        electron_current_density
        + hole_current_density
    )

    return VoltageSweepResult(
        voltage=voltage,
        current=current,
        potential=potential,
        electron_density=electron_density,
        hole_density=hole_density,
        electric_field=electric_field,
        electron_current_density=electron_current_density,
        hole_current_density=hole_current_density,
        total_current_density=total_current_density,
        iterations=5,
        residual=1.0e-10,
        solve_time=0.01,
        converged=solution.converged,
        metadata={
            "adapter": True,
            "simulation_type": type(simulation).__name__,
        },
    )


@pytest.fixture
def existing_simulation() -> ExistingSimulation:
    """Return a simulation requiring adaptation."""

    return ExistingSimulation()


@pytest.fixture
def adapter(
    existing_simulation: ExistingSimulation,
) -> VoltageSweepSimulationAdapter[
    ExistingSimulation,
    AdapterSolution
]:
    """Return a configured voltage-sweep adapter."""

    return VoltageSweepSimulationAdapter(
        simulation=existing_simulation,
        set_contact_voltage=(
            lambda simulation, contact, voltage:
            simulation.update_terminal(contact, voltage)
        ),
        solve=(
            lambda simulation, initial_guess:
            simulation.run_solver(initial_guess)
        ),
        build_result=create_voltage_sweep_result,
    )


def test_adapter_satisfies_voltage_sweep_protocol(
    adapter: VoltageSweepSimulationAdapter[
        ExistingSimulation,
        AdapterSolution
    ],
) -> None:
    """The adapter should satisfy the runtime sweep protocol."""

    assert isinstance(
        adapter,
        VoltageSweepSimulation,
    )


def test_adapter_exposes_wrapped_simulation(
    adapter: VoltageSweepSimulationAdapter[
        ExistingSimulation,
        AdapterSolution
    ],
    existing_simulation: ExistingSimulation,
) -> None:
    """The original simulation should remain accessible."""

    assert adapter.simulation is existing_simulation


def test_adapter_sets_contact_voltage(
    adapter: VoltageSweepSimulationAdapter[
        ExistingSimulation,
        AdapterSolution
    ],
    existing_simulation: ExistingSimulation,
) -> None:
    """Contact updates should be forwarded to the callback."""

    adapter.set_contact_voltage(
        "anode",
        0.5,
    )

    assert existing_simulation.contact_values == {
        "anode": pytest.approx(0.5),
    }


def test_adapter_solves_operating_point(
    adapter: VoltageSweepSimulationAdapter[
        ExistingSimulation,
        AdapterSolution
    ],
) -> None:
    """The solve callback should return its solver-specific solution."""

    adapter.set_contact_voltage(
        "anode",
        0.25,
    )

    solution = adapter.solve()

    assert isinstance(solution, AdapterSolution)
    assert solution.voltage == pytest.approx(0.25)
    assert solution.initial_guess is None


def test_adapter_forwards_initial_guess(
    adapter: VoltageSweepSimulationAdapter[
        ExistingSimulation,
        AdapterSolution
    ],
) -> None:
    """Warm-start data should pass through the adapter unchanged."""

    adapter.set_contact_voltage(
        "anode",
        0.0,
    )

    previous_solution = adapter.solve()

    adapter.set_contact_voltage(
        "anode",
        0.1,
    )

    next_solution = adapter.solve(
        initial_guess=previous_solution,
    )

    assert next_solution.initial_guess is previous_solution


def test_adapter_builds_voltage_sweep_result(
    adapter: VoltageSweepSimulationAdapter[
        ExistingSimulation,
        AdapterSolution
    ],
) -> None:
    """The result callback should produce a standard result."""

    adapter.set_contact_voltage(
        "anode",
        0.5,
    )

    solution = adapter.solve()

    result = adapter.build_voltage_sweep_result(
        voltage=0.5,
        solution=solution,
    )

    assert isinstance(result, VoltageSweepResult)
    assert result.voltage == pytest.approx(0.5)
    assert result.current == pytest.approx(0.5e-6)
    assert result.metadata["adapter"] is True


def test_adapter_runs_complete_voltage_sweep(
    adapter: VoltageSweepSimulationAdapter[
        ExistingSimulation,
        AdapterSolution
    ],
    existing_simulation: ExistingSimulation,
) -> None:
    """The adapter should integrate with the complete workflow."""

    sweep = VoltageSweep(
        simulation=adapter,
        contact="anode",
        voltages=[0.0, 0.1, 0.2],
        warm_start=True,
    )

    results = sweep.run()

    assert len(results) == 3

    np.testing.assert_allclose(
        results.voltages,
        np.array([0.0, 0.1, 0.2]),
    )

    np.testing.assert_allclose(
        results.currents,
        np.array([0.0, 0.1e-6, 0.2e-6]),
    )

    assert existing_simulation.solve_inputs[0] is None
    assert (
        existing_simulation.solve_inputs[1]
        is existing_simulation.solutions[0]
    )
    assert (
        existing_simulation.solve_inputs[2]
        is existing_simulation.solutions[1]
    )


@pytest.mark.parametrize(
    ("argument_name", "bad_value"),
    [
        ("set_contact_voltage", None),
        ("solve", 42),
        ("build_result", "invalid"),
    ],
)
def test_adapter_rejects_non_callable_callbacks(
    argument_name: str,
    bad_value: object,
) -> None:
    """Every adapter operation must be configured with a callable."""

    arguments: dict[str, object] = {
        "simulation": ExistingSimulation(),
        "set_contact_voltage": lambda simulation, contact, voltage: None,
        "solve": lambda simulation, initial_guess: AdapterSolution(
            voltage=0.0,
            initial_guess=initial_guess,
        ),
        "build_result": create_voltage_sweep_result,
    }

    arguments[argument_name] = bad_value

    with pytest.raises(
        TypeError,
        match=f"{argument_name} must be callable",
    ):
        VoltageSweepSimulationAdapter(
            **arguments,  # type: ignore[arg-type]
        )


def test_adapter_rejects_none_simulation() -> None:
    """An adapter requires a wrapped simulation object."""

    with pytest.raises(
        TypeError,
        match="simulation must not be None",
    ):
        VoltageSweepSimulationAdapter(
            simulation=None,
            set_contact_voltage=lambda simulation, contact, voltage: None,
            solve=lambda simulation, initial_guess: AdapterSolution(
                voltage=0.0,
                initial_guess=initial_guess,
            ),
            build_result=lambda simulation, voltage, solution: (
                create_voltage_sweep_result(
                    ExistingSimulation(),
                    voltage,
                    solution,
                )
            ),
        )


def test_adapter_rejects_invalid_result_type(
    existing_simulation: ExistingSimulation,
) -> None:
    """The result callback must return VoltageSweepResult."""

    adapter = VoltageSweepSimulationAdapter(
        simulation=existing_simulation,
        set_contact_voltage=(
            lambda simulation, contact, voltage:
            simulation.update_terminal(contact, voltage)
        ),
        solve=(
            lambda simulation, initial_guess:
            simulation.run_solver(initial_guess)
        ),
        build_result=(
            lambda simulation, voltage, solution:
            {"voltage": voltage}
        ),
    )

    adapter.set_contact_voltage(
        "anode",
        0.0,
    )

    solution = adapter.solve()

    with pytest.raises(
        TypeError,
        match="must return a VoltageSweepResult",
    ):
        adapter.build_voltage_sweep_result(
            voltage=0.0,
            solution=solution,
        )


def test_adapter_repr_contains_simulation_type(
    adapter: VoltageSweepSimulationAdapter[
        ExistingSimulation,
        AdapterSolution
    ],
) -> None:
    """The representation should identify the adapted simulation."""

    representation = repr(adapter)

    assert "VoltageSweepSimulationAdapter" in representation
    assert "ExistingSimulation" in representation