"""Tests for the solver-agnostic DeviceForge voltage-sweep workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from deviceforge.analysis.sweep_result import VoltageSweepResult
from deviceforge.analysis.sweep_results import SweepResults
"""
from deviceforge.analysis.voltage_sweep import (
    VoltageSweep,
    VoltageSweepSimulation,
)
"""
from deviceforge.analysis.voltage_sweep import VoltageSweep
from deviceforge.analysis.voltage_sweep_simulation import (
    VoltageSweepSimulation,
)

@dataclass
class FakeSolution:
    """Minimal solver-specific solution used by the test simulation."""

    voltage: float
    solve_number: int
    converged: bool = True


class FakeSimulation(VoltageSweepSimulation):
    """Simple simulation implementing the VoltageSweepSimulation protocol."""

    def __init__(
        self,
        *,
        failed_voltages: set[float] | None = None,
    ) -> None:
        self.failed_voltages = (
            set()
            if failed_voltages is None
            else set(failed_voltages)
        )

        self.current_contact: str | None = None
        self.current_voltage: float | None = None

        self.applied_voltages: list[tuple[str, float]] = []
        self.initial_guesses: list[object | None] = []
        self.solutions: list[FakeSolution] = []

    def set_contact_voltage(
        self,
        contact: str,
        voltage: float,
    ) -> None:
        """Record the applied contact voltage."""

        self.current_contact = contact
        self.current_voltage = voltage
        self.applied_voltages.append((contact, voltage))

    def solve(
        self,
        *,
        initial_guess: object | None = None,
    ) -> FakeSolution:
        """Return a deterministic fake operating-point solution."""

        if self.current_voltage is None:
            raise RuntimeError("No voltage has been applied.")

        self.initial_guesses.append(initial_guess)

        converged = not any(
            np.isclose(
                self.current_voltage,
                failed_voltage,
            )
            for failed_voltage in self.failed_voltages
        )

        solution = FakeSolution(
            voltage=self.current_voltage,
            solve_number=len(self.solutions) + 1,
            converged=converged,
        )

        self.solutions.append(solution)

        return solution

    def build_voltage_sweep_result(
        self,
        *,
        voltage: float,
        solution: object,
    ) -> VoltageSweepResult:
        """Convert the fake solution into a standard sweep result."""

        if not isinstance(solution, FakeSolution):
            raise TypeError("Expected FakeSolution.")

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
            iterations=solution.solve_number,
            residual=(
                1.0e-10
                if solution.converged
                else 1.0e-2
            ),
            solve_time=0.01,
            converged=solution.converged,
            metadata={
                "contact": self.current_contact,
                "solver": "fake",
            },
        )


def test_fake_simulation_satisfies_runtime_protocol() -> None:
    """A compatible simulation should satisfy the runtime protocol."""

    simulation = FakeSimulation()

    assert isinstance(
        simulation,
        VoltageSweepSimulation,
    )


def test_constructor_stores_configuration() -> None:
    """The constructor should preserve valid sweep configuration."""

    simulation = FakeSimulation()

    sweep = VoltageSweep(
        simulation=simulation,
        contact="anode",
        voltages=[-1.0, 0.0, 1.0],
        warm_start=True,
        stop_on_failure=False,
        metadata={"device": "pn_junction"},
    )

    assert sweep.simulation is simulation
    assert sweep.contact == "anode"
    assert sweep.number_of_points == 3
    assert sweep.warm_start is True
    assert sweep.stop_on_failure is False
    assert sweep.metadata == {"device": "pn_junction"}

    np.testing.assert_allclose(
        sweep.voltages,
        np.array([-1.0, 0.0, 1.0]),
    )


def test_constructor_strips_contact_whitespace() -> None:
    """Leading and trailing contact whitespace should be removed."""

    sweep = VoltageSweep(
        simulation=FakeSimulation(),
        contact="  gate  ",
        voltages=[0.0],
    )

    assert sweep.contact == "gate"


def test_constructor_copies_voltage_input() -> None:
    """External voltage-array changes should not modify the sweep."""

    voltages = np.array([-1.0, 0.0, 1.0])

    sweep = VoltageSweep(
        simulation=FakeSimulation(),
        contact="anode",
        voltages=voltages,
    )

    voltages[0] = 100.0

    np.testing.assert_allclose(
        sweep.voltages,
        np.array([-1.0, 0.0, 1.0]),
    )


def test_voltages_property_returns_independent_copy() -> None:
    """The voltage property should not expose internal storage."""

    sweep = VoltageSweep(
        simulation=FakeSimulation(),
        contact="anode",
        voltages=[-1.0, 0.0, 1.0],
    )

    returned_voltages = sweep.voltages
    returned_voltages[0] = 100.0

    assert sweep.voltages[0] == pytest.approx(-1.0)


def test_constructor_accepts_voltage_generator() -> None:
    """Arbitrary voltage iterables should be accepted."""

    sweep = VoltageSweep(
        simulation=FakeSimulation(),
        contact="anode",
        voltages=(
            voltage
            for voltage in [-0.5, 0.0, 0.5]
        ),
    )

    np.testing.assert_allclose(
        sweep.voltages,
        np.array([-0.5, 0.0, 0.5]),
    )


def test_constructor_copies_metadata() -> None:
    """External metadata changes should not modify sweep metadata."""

    metadata = {"device": "pn_junction"}

    sweep = VoltageSweep(
        simulation=FakeSimulation(),
        contact="anode",
        voltages=[0.0],
        metadata=metadata,
    )

    metadata["device"] = "mosfet"

    assert sweep.metadata["device"] == "pn_junction"


class MissingSimulationMethods:
    """Object that does not implement the voltage-sweep interface."""

    def solve(
        self,
        *,
        initial_guess: object | None = None,
    ) -> object:
        """Incomplete fake solve method."""

        return object()


def test_constructor_rejects_incompatible_simulation() -> None:
    """The simulation must expose the complete required interface."""

    with pytest.raises(
        TypeError,
        match="Missing callable methods",
    ):
        VoltageSweep(
            simulation=MissingSimulationMethods(),  # type: ignore[arg-type]
            contact="anode",
            voltages=[0.0],
        )


@pytest.mark.parametrize(
    "bad_contact",
    [
        1,
        None,
        ["anode"],
    ],
)
def test_constructor_rejects_non_string_contact(
    bad_contact: object,
) -> None:
    """Contact names must be strings."""

    with pytest.raises(
        TypeError,
        match="contact must be a string",
    ):
        VoltageSweep(
            simulation=FakeSimulation(),
            contact=bad_contact,  # type: ignore[arg-type]
            voltages=[0.0],
        )


@pytest.mark.parametrize(
    "bad_contact",
    [
        "",
        " ",
        "\t",
    ],
)
def test_constructor_rejects_empty_contact(
    bad_contact: str,
) -> None:
    """Contact names must contain visible characters."""

    with pytest.raises(
        ValueError,
        match="contact must not be empty",
    ):
        VoltageSweep(
            simulation=FakeSimulation(),
            contact=bad_contact,
            voltages=[0.0],
        )


def test_constructor_rejects_empty_voltages() -> None:
    """A sweep must contain at least one operating point."""

    with pytest.raises(
        ValueError,
        match="at least one value",
    ):
        VoltageSweep(
            simulation=FakeSimulation(),
            contact="anode",
            voltages=[],
        )


@pytest.mark.parametrize(
    "bad_voltages",
    [
        1.0,
        None,
        "0.0, 1.0",
    ],
)
def test_constructor_rejects_invalid_voltage_container(
    bad_voltages: object,
) -> None:
    """Voltage input must be a non-string iterable."""

    with pytest.raises(
        TypeError,
        match="voltages must be an iterable",
    ):
        VoltageSweep(
            simulation=FakeSimulation(),
            contact="anode",
            voltages=bad_voltages,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "bad_voltage",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_constructor_rejects_non_finite_voltage(
    bad_voltage: float,
) -> None:
    """Every requested voltage must be finite."""

    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        VoltageSweep(
            simulation=FakeSimulation(),
            contact="anode",
            voltages=[0.0, bad_voltage],
        )


@pytest.mark.parametrize(
    "bad_voltage",
    [
        True,
        "invalid",
        object(),
    ],
)
def test_constructor_rejects_non_numeric_voltage(
    bad_voltage: object,
) -> None:
    """Every requested voltage must be a real number."""

    with pytest.raises(
        TypeError,
        match="must be a real number",
    ):
        VoltageSweep(
            simulation=FakeSimulation(),
            contact="anode",
            voltages=[bad_voltage],  # type: ignore[list-item]
        )


@pytest.mark.parametrize(
    ("argument_name", "bad_value"),
    [
        ("warm_start", 1),
        ("warm_start", "yes"),
        ("stop_on_failure", 0),
        ("stop_on_failure", None),
    ],
)
def test_constructor_rejects_non_boolean_options(
    argument_name: str,
    bad_value: object,
) -> None:
    """Workflow switches should use strict Boolean values."""

    arguments: dict[str, object] = {
        "simulation": FakeSimulation(),
        "contact": "anode",
        "voltages": [0.0],
        "warm_start": True,
        "stop_on_failure": False,
    }

    arguments[argument_name] = bad_value

    with pytest.raises(
        TypeError,
        match=f"{argument_name} must be a boolean",
    ):
        VoltageSweep(**arguments)  # type: ignore[arg-type]


def test_constructor_rejects_invalid_metadata() -> None:
    """Sweep metadata must be a dictionary or None."""

    with pytest.raises(
        TypeError,
        match="metadata must be a dictionary or None",
    ):
        VoltageSweep(
            simulation=FakeSimulation(),
            contact="anode",
            voltages=[0.0],
            metadata=["invalid"],  # type: ignore[arg-type]
        )


def test_run_returns_sweep_results() -> None:
    """Executing a sweep should return a SweepResults container."""

    sweep = VoltageSweep(
        simulation=FakeSimulation(),
        contact="anode",
        voltages=[-0.5, 0.0, 0.5],
    )

    results = sweep.run()

    assert isinstance(results, SweepResults)
    assert len(results) == 3


def test_run_applies_voltages_in_supplied_order() -> None:
    """The workflow must preserve arbitrary voltage ordering."""

    simulation = FakeSimulation()

    sweep = VoltageSweep(
        simulation=simulation,
        contact="gate",
        voltages=[0.5, -0.5, 0.0, 0.25],
    )

    sweep.run()

    assert simulation.applied_voltages == [
        ("gate", 0.5),
        ("gate", -0.5),
        ("gate", 0.0),
        ("gate", 0.25),
    ]


def test_run_builds_expected_result_values() -> None:
    """Generated results should correspond to each applied voltage."""

    sweep = VoltageSweep(
        simulation=FakeSimulation(),
        contact="anode",
        voltages=[-1.0, 0.0, 1.0],
    )

    results = sweep.run()

    np.testing.assert_allclose(
        results.voltages,
        np.array([-1.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(
        results.currents,
        np.array([-1.0e-6, 0.0, 1.0e-6]),
    )

    assert results.all_converged is True


def test_run_populates_standard_metadata() -> None:
    """Result metadata should describe the completed workflow."""

    sweep = VoltageSweep(
        simulation=FakeSimulation(),
        contact="anode",
        voltages=[-0.5, 0.0, 0.5],
        warm_start=True,
        stop_on_failure=False,
        metadata={"device": "pn_junction"},
    )

    results = sweep.run()

    assert results.metadata == {
        "device": "pn_junction",
        "analysis_type": "voltage_sweep",
        "contact": "anode",
        "requested_voltages": [-0.5, 0.0, 0.5],
        "number_requested": 3,
        "warm_start": True,
        "stop_on_failure": False,
        "number_completed": 3,
        "terminated_early": False,
    }


def test_warm_start_passes_previous_converged_solution() -> None:
    """Each converged solution should initialise the next point."""

    simulation = FakeSimulation()

    sweep = VoltageSweep(
        simulation=simulation,
        contact="anode",
        voltages=[0.0, 0.1, 0.2],
        warm_start=True,
    )

    sweep.run()

    assert simulation.initial_guesses[0] is None
    assert simulation.initial_guesses[1] is simulation.solutions[0]
    assert simulation.initial_guesses[2] is simulation.solutions[1]


def test_disabled_warm_start_always_passes_none() -> None:
    """Independent solves should receive no previous solution."""

    simulation = FakeSimulation()

    sweep = VoltageSweep(
        simulation=simulation,
        contact="anode",
        voltages=[0.0, 0.1, 0.2],
        warm_start=False,
    )

    sweep.run()

    assert simulation.initial_guesses == [
        None,
        None,
        None,
    ]


def test_non_converged_solution_is_not_used_as_warm_start() -> None:
    """Failed solutions should not initialise subsequent points."""

    simulation = FakeSimulation(
        failed_voltages={0.1},
    )

    sweep = VoltageSweep(
        simulation=simulation,
        contact="anode",
        voltages=[0.0, 0.1, 0.2],
        warm_start=True,
        stop_on_failure=False,
    )

    results = sweep.run()

    assert len(results) == 3

    assert simulation.initial_guesses[0] is None
    assert simulation.initial_guesses[1] is simulation.solutions[0]
    assert simulation.initial_guesses[2] is None


def test_stop_on_failure_terminates_after_failed_result() -> None:
    """The failed point should be retained before early termination."""

    simulation = FakeSimulation(
        failed_voltages={0.1},
    )

    sweep = VoltageSweep(
        simulation=simulation,
        contact="anode",
        voltages=[0.0, 0.1, 0.2, 0.3],
        stop_on_failure=True,
    )

    results = sweep.run()

    assert len(results) == 2

    np.testing.assert_allclose(
        results.voltages,
        np.array([0.0, 0.1]),
    )

    assert results.number_converged == 1
    assert results.number_failed == 1
    assert results.metadata["number_completed"] == 2
    assert results.metadata["terminated_early"] is True


def test_continue_after_failure_completes_requested_sweep() -> None:
    """A disabled stop option should retain later operating points."""

    simulation = FakeSimulation(
        failed_voltages={0.1},
    )

    sweep = VoltageSweep(
        simulation=simulation,
        contact="anode",
        voltages=[0.0, 0.1, 0.2],
        stop_on_failure=False,
    )

    results = sweep.run()

    assert len(results) == 3
    assert results.number_failed == 1
    assert results.metadata["terminated_early"] is False


def test_repeated_run_creates_fresh_results() -> None:
    """Each run should execute a new independent sweep."""

    simulation = FakeSimulation()

    sweep = VoltageSweep(
        simulation=simulation,
        contact="anode",
        voltages=[0.0, 0.1],
    )

    first_results = sweep.run()
    second_results = sweep.run()

    assert first_results is not second_results
    assert len(first_results) == 2
    assert len(second_results) == 2
    assert len(simulation.solutions) == 4


class VoltageApplicationFailureSimulation(FakeSimulation):
    """Simulation that fails while applying a voltage."""

    def set_contact_voltage(
        self,
        contact: str,
        voltage: float,
    ) -> None:
        """Raise at a chosen voltage."""

        if np.isclose(voltage, 0.1):
            raise ValueError("Contact update failed.")

        super().set_contact_voltage(contact, voltage)


def test_voltage_application_error_is_wrapped_with_context() -> None:
    """Contact-update failures should identify the operating point."""

    sweep = VoltageSweep(
        simulation=VoltageApplicationFailureSimulation(),
        contact="anode",
        voltages=[0.0, 0.1],
    )

    with pytest.raises(
        RuntimeError,
        match=r"apply contact voltage.*point 1.*0.1 V",
    ):
        sweep.run()


class SolveFailureSimulation(FakeSimulation):
    """Simulation that raises during solving."""

    def solve(
        self,
        *,
        initial_guess: object | None = None,
    ) -> FakeSolution:
        """Raise at a chosen applied voltage."""

        if self.current_voltage is not None and np.isclose(
            self.current_voltage,
            0.1,
        ):
            raise RuntimeError("Numerical solver failed.")

        return super().solve(initial_guess=initial_guess)


def test_solver_error_is_wrapped_with_context() -> None:
    """Solver exceptions should identify the failing operating point."""

    sweep = VoltageSweep(
        simulation=SolveFailureSimulation(),
        contact="anode",
        voltages=[0.0, 0.1],
    )

    with pytest.raises(
        RuntimeError,
        match=r"solve operating point.*point 1.*0.1 V",
    ):
        sweep.run()


class ResultConstructionFailureSimulation(FakeSimulation):
    """Simulation that raises while translating a solution."""

    def build_voltage_sweep_result(
        self,
        *,
        voltage: float,
        solution: object,
    ) -> VoltageSweepResult:
        """Raise during result construction."""

        raise ValueError("Unable to extract result fields.")


def test_result_construction_error_is_wrapped_with_context() -> None:
    """Result-construction exceptions should include sweep context."""

    sweep = VoltageSweep(
        simulation=ResultConstructionFailureSimulation(),
        contact="anode",
        voltages=[0.0],
    )

    with pytest.raises(
        RuntimeError,
        match=r"build voltage-sweep result.*point 0.*0.0 V",
    ):
        sweep.run()


class InvalidResultTypeSimulation(FakeSimulation):
    """Simulation returning an invalid result type."""

    def build_voltage_sweep_result(
        self,
        *,
        voltage: float,
        solution: object,
    ) -> object:
        """Return an incompatible value."""

        return {"voltage": voltage}


def test_run_rejects_invalid_result_type() -> None:
    """The simulation must produce a VoltageSweepResult."""

    sweep = VoltageSweep(
        simulation=InvalidResultTypeSimulation(),  # type: ignore[arg-type]
        contact="anode",
        voltages=[0.0],
    )

    with pytest.raises(
        TypeError,
        match="must return a VoltageSweepResult",
    ):
        sweep.run()


class MismatchedVoltageSimulation(FakeSimulation):
    """Simulation returning a result with the wrong voltage."""

    def build_voltage_sweep_result(
        self,
        *,
        voltage: float,
        solution: object,
    ) -> VoltageSweepResult:
        """Build a valid result associated with another voltage."""

        return super().build_voltage_sweep_result(
            voltage=voltage + 1.0,
            solution=solution,
        )


def test_run_rejects_result_with_mismatched_voltage() -> None:
    """Result and applied voltages must agree."""

    sweep = VoltageSweep(
        simulation=MismatchedVoltageSimulation(),
        contact="anode",
        voltages=[0.0],
    )

    with pytest.raises(
        ValueError,
        match="does not match the applied voltage",
    ):
        sweep.run()


def test_from_range_generates_ascending_voltage_sequence() -> None:
    """Ascending ranges should include both endpoints."""

    sweep = VoltageSweep.from_range(
        simulation=FakeSimulation(),
        contact="anode",
        start=-0.2,
        stop=0.2,
        step=0.1,
    )

    np.testing.assert_allclose(
        sweep.voltages,
        np.array([-0.2, -0.1, 0.0, 0.1, 0.2]),
        atol=1.0e-15,
    )


def test_from_range_generates_descending_voltage_sequence() -> None:
    """Descending ranges should support negative increments."""

    sweep = VoltageSweep.from_range(
        simulation=FakeSimulation(),
        contact="gate",
        start=0.2,
        stop=-0.2,
        step=-0.1,
    )

    np.testing.assert_allclose(
        sweep.voltages,
        np.array([0.2, 0.1, 0.0, -0.1, -0.2]),
        atol=1.0e-15,
    )


def test_from_range_includes_stop_with_partial_final_step() -> None:
    """A final shortened interval should reach the requested stop."""

    sweep = VoltageSweep.from_range(
        simulation=FakeSimulation(),
        contact="anode",
        start=0.0,
        stop=1.0,
        step=0.3,
    )

    np.testing.assert_allclose(
        sweep.voltages,
        np.array([0.0, 0.3, 0.6, 0.9, 1.0]),
    )


def test_from_range_equal_start_and_stop_creates_one_point() -> None:
    """A zero-width range should create one operating point."""

    sweep = VoltageSweep.from_range(
        simulation=FakeSimulation(),
        contact="anode",
        start=0.5,
        stop=0.5,
        step=0.1,
    )

    np.testing.assert_allclose(
        sweep.voltages,
        np.array([0.5]),
    )


def test_from_range_preserves_options_and_metadata() -> None:
    """Factory options should be passed to the constructed sweep."""

    sweep = VoltageSweep.from_range(
        simulation=FakeSimulation(),
        contact="gate",
        start=0.0,
        stop=1.0,
        step=0.5,
        warm_start=False,
        stop_on_failure=True,
        metadata={"device": "mosfet"},
    )

    assert sweep.contact == "gate"
    assert sweep.warm_start is False
    assert sweep.stop_on_failure is True
    assert sweep.metadata == {"device": "mosfet"}


def test_from_range_rejects_zero_step() -> None:
    """A range cannot be generated with a zero increment."""

    with pytest.raises(
        ValueError,
        match="step must not be zero",
    ):
        VoltageSweep.from_range(
            simulation=FakeSimulation(),
            contact="anode",
            start=0.0,
            stop=1.0,
            step=0.0,
        )


def test_from_range_rejects_negative_ascending_step() -> None:
    """Ascending ranges require a positive increment."""

    with pytest.raises(
        ValueError,
        match="positive for an ascending",
    ):
        VoltageSweep.from_range(
            simulation=FakeSimulation(),
            contact="anode",
            start=0.0,
            stop=1.0,
            step=-0.1,
        )


def test_from_range_rejects_positive_descending_step() -> None:
    """Descending ranges require a negative increment."""

    with pytest.raises(
        ValueError,
        match="negative for a descending",
    ):
        VoltageSweep.from_range(
            simulation=FakeSimulation(),
            contact="anode",
            start=1.0,
            stop=0.0,
            step=0.1,
        )


@pytest.mark.parametrize(
    ("argument_name", "bad_value", "expected_exception"),
    [
        ("start", np.nan, ValueError),
        ("stop", np.inf, ValueError),
        ("step", -np.inf, ValueError),
        ("start", True, TypeError),
        ("stop", "invalid", TypeError),
    ],
)
def test_from_range_rejects_invalid_numeric_arguments(
    argument_name: str,
    bad_value: object,
    expected_exception: type[Exception],
) -> None:
    """Range parameters should be finite real numbers."""

    arguments: dict[str, object] = {
        "simulation": FakeSimulation(),
        "contact": "anode",
        "start": 0.0,
        "stop": 1.0,
        "step": 0.1,
    }

    arguments[argument_name] = bad_value

    with pytest.raises(expected_exception):
        VoltageSweep.from_range(**arguments)  # type: ignore[arg-type]


def test_repr_contains_useful_configuration() -> None:
    """The representation should summarise the requested workflow."""

    sweep = VoltageSweep(
        simulation=FakeSimulation(),
        contact="anode",
        voltages=[-1.0, 0.0, 1.0],
        warm_start=True,
        stop_on_failure=False,
    )

    representation = repr(sweep)

    assert "VoltageSweep" in representation
    assert "contact='anode'" in representation
    assert "number_of_points=3" in representation
    assert "voltage_range=(-1.0, 1.0)" in representation
    assert "warm_start=True" in representation
    assert "stop_on_failure=False" in representation