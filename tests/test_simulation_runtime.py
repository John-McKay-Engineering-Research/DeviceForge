from __future__ import annotations

import numpy as np
import pytest

from deviceforge import (
    Field,
    Simulation,
    SimulationResult,
)
from deviceforge.runtime import (
    RuntimeState,
    SimulationRuntime,
)


def create_result(
    simulation: Simulation,
    *,
    converged: bool = True,
    iterations: int = 4,
) -> SimulationResult:
    """Create a valid SimulationResult for runtime tests."""

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=simulation.grid,
        values=np.zeros(
            simulation.grid.shape,
            dtype=np.float64,
        ),
    )

    residual_history = np.logspace(
        0,
        -9,
        num=iterations,
        dtype=np.float64,
    )

    return SimulationResult(
        fields={
            "electrostatic_potential": potential,
        },
        converged=converged,
        iterations=iterations,
        residual_history=residual_history,
        runtime_seconds=0.01,
        solver_name="recording_solver",
        backend_name="cpu",
        metadata={
            "test_value": 42,
        },
    )


class RecordingSolver:
    """Test solver that records each received simulation."""

    def __init__(
        self,
        result: SimulationResult,
    ) -> None:
        self.result = result
        self.call_count = 0
        self.received_simulation: Simulation | None = None

    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        self.call_count += 1
        self.received_simulation = simulation

        return self.result


class InvalidResultSolver:
    """Solver-shaped object returning an invalid result."""

    def solve(
        self,
        simulation: Simulation,
    ) -> str:
        return "invalid result"


class MissingSolveMethod:
    """Object that does not implement SolverProtocol."""

    pass


def test_runtime_stores_simulation(
    simulation,
) -> None:
    runtime = SimulationRuntime(
        simulation=simulation,
    )

    assert runtime.simulation is simulation
    assert runtime.device is simulation.device
    assert runtime.grid is simulation.grid


def test_runtime_creates_default_state(
    simulation,
) -> None:
    runtime = SimulationRuntime(
        simulation=simulation,
    )

    assert isinstance(runtime.state, RuntimeState)
    assert not runtime.has_run
    assert not runtime.has_result
    assert not runtime.has_solution


def test_runtime_creates_default_name(
    simulation,
) -> None:
    runtime = SimulationRuntime(
        simulation=simulation,
    )

    assert runtime.name == (
        f"{simulation.name}_runtime"
    )


def test_runtime_normalises_custom_name(
    simulation,
) -> None:
    runtime = SimulationRuntime(
        simulation=simulation,
        name="  custom runtime  ",
    )

    assert runtime.name == "custom runtime"


def test_runtime_rejects_empty_name(
    simulation,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        SimulationRuntime(
            simulation=simulation,
            name="   ",
        )


def test_runtime_starts_without_solver_or_result(
    simulation,
) -> None:
    runtime = SimulationRuntime(
        simulation=simulation,
    )

    assert runtime.solver is None

    assert not runtime.has_solver
    assert not runtime.is_solver_configured

    assert not runtime.has_result
    assert not runtime.has_run
    assert not runtime.has_solution

    assert runtime.result is None
    assert runtime.current_result() is None
    assert runtime.converged is None
    assert runtime.iterations is None


def test_runtime_accepts_solver_during_construction(
    simulation,
) -> None:
    expected_result = create_result(simulation)
    solver = RecordingSolver(expected_result)

    runtime = SimulationRuntime(
        simulation=simulation,
        solver=solver,
    )

    assert runtime.solver is solver
    assert runtime.has_solver
    assert runtime.is_solver_configured


def test_runtime_rejects_invalid_solver_during_construction(
    simulation,
) -> None:
    with pytest.raises(
        TypeError,
        match="implement SolverProtocol",
    ):
        SimulationRuntime(
            simulation=simulation,
            solver=MissingSolveMethod(),
        )


def test_set_solver(
    simulation,
) -> None:
    expected_result = create_result(simulation)
    solver = RecordingSolver(expected_result)

    runtime = SimulationRuntime(
        simulation=simulation,
    )

    runtime.set_solver(solver)

    assert runtime.solver is solver
    assert runtime.has_solver


def test_set_solver_rejects_invalid_solver(
    simulation,
) -> None:
    runtime = SimulationRuntime(
        simulation=simulation,
    )

    with pytest.raises(
        TypeError,
        match="implement SolverProtocol",
    ):
        runtime.set_solver(
            MissingSolveMethod()
        )


def test_clear_solver(
    simulation,
) -> None:
    expected_result = create_result(simulation)
    solver = RecordingSolver(expected_result)

    runtime = SimulationRuntime(
        simulation=simulation,
        solver=solver,
    )

    runtime.clear_solver()

    assert runtime.solver is None
    assert not runtime.has_solver


def test_solve_without_solver_raises_runtime_error(
    simulation,
) -> None:
    runtime = SimulationRuntime(
        simulation=simulation,
    )

    with pytest.raises(
        RuntimeError,
        match="no solver is configured",
    ):
        runtime.solve()


def test_solve_calls_solver_once(
    simulation,
) -> None:
    expected_result = create_result(simulation)
    solver = RecordingSolver(expected_result)

    runtime = SimulationRuntime(
        simulation=simulation,
        solver=solver,
    )

    runtime.solve()

    assert solver.call_count == 1


def test_solve_passes_simulation_to_solver(
    simulation,
) -> None:
    expected_result = create_result(simulation)
    solver = RecordingSolver(expected_result)

    runtime = SimulationRuntime(
        simulation=simulation,
        solver=solver,
    )

    runtime.solve()

    assert solver.received_simulation is simulation


def test_solve_returns_solver_result(
    simulation,
) -> None:
    expected_result = create_result(simulation)
    solver = RecordingSolver(expected_result)

    runtime = SimulationRuntime(
        simulation=simulation,
        solver=solver,
    )

    returned_result = runtime.solve()

    assert returned_result is expected_result


def test_solve_records_result_in_runtime_state(
    simulation,
) -> None:
    expected_result = create_result(
        simulation,
        converged=True,
        iterations=4,
    )
    solver = RecordingSolver(expected_result)

    runtime = SimulationRuntime(
        simulation=simulation,
        solver=solver,
    )

    runtime.solve()

    assert runtime.result is expected_result
    assert runtime.current_result() is expected_result
    assert runtime.state.last_result is expected_result

    assert runtime.has_result
    assert runtime.has_run
    assert runtime.has_solution

    assert runtime.converged is True
    assert runtime.iterations == 4

    assert runtime.state.iteration_count == 4
    assert runtime.state.converged is True

    assert runtime.state.elapsed_time == pytest.approx(
        expected_result.runtime_seconds
    )

    assert runtime.state.residual_history == pytest.approx(
        expected_result.residual_history.tolist()
    )

    assert runtime.state.metadata == expected_result.metadata


def test_solve_rejects_invalid_solver_result(
    simulation,
) -> None:
    runtime = SimulationRuntime(
        simulation=simulation,
        solver=InvalidResultSolver(),
    )

    with pytest.raises(
        TypeError,
        match="must return a SimulationResult",
    ):
        runtime.solve()

    assert runtime.result is None
    assert not runtime.has_result
    assert not runtime.has_run


def test_second_solve_retains_previous_solution(
    simulation,
) -> None:
    first_result = create_result(
        simulation,
        iterations=2,
    )
    second_result = create_result(
        simulation,
        iterations=4,
    )

    solver = RecordingSolver(first_result)

    runtime = SimulationRuntime(
        simulation=simulation,
        solver=solver,
    )

    runtime.solve()
    first_solution = runtime.state.current_solution

    solver.result = second_result
    runtime.solve()

    assert solver.call_count == 2
    assert runtime.state.previous_solution is first_solution
    assert runtime.state.current_solution is second_result.fields
    assert runtime.result is second_result


def test_reset_clears_state_but_preserves_configuration(
    simulation,
) -> None:
    expected_result = create_result(simulation)
    solver = RecordingSolver(expected_result)

    runtime = SimulationRuntime(
        simulation=simulation,
        solver=solver,
    )

    runtime.solve()
    runtime.reset()

    assert runtime.result is None
    assert runtime.current_result() is None

    assert not runtime.has_result
    assert not runtime.has_run
    assert not runtime.has_solution

    assert runtime.converged is None
    assert runtime.iterations is None

    assert runtime.simulation is simulation
    assert runtime.solver is solver
    assert runtime.has_solver


def test_snapshot_state_returns_independent_containers(
    simulation,
) -> None:
    expected_result = create_result(simulation)
    solver = RecordingSolver(expected_result)

    runtime = SimulationRuntime(
        simulation=simulation,
        solver=solver,
    )

    runtime.solve()

    snapshot = runtime.snapshot_state()

    assert snapshot is not runtime.state
    assert snapshot.last_result is runtime.state.last_result
    assert snapshot.current_solution is runtime.state.current_solution

    assert snapshot.residual_history == (
        runtime.state.residual_history
    )
    assert snapshot.residual_history is not (
        runtime.state.residual_history
    )

    assert snapshot.metadata == runtime.state.metadata
    assert snapshot.metadata is not runtime.state.metadata


def test_runtime_repr_contains_execution_status(
    simulation,
) -> None:
    expected_result = create_result(simulation)
    solver = RecordingSolver(expected_result)

    runtime = SimulationRuntime(
        simulation=simulation,
        solver=solver,
    )

    representation = repr(runtime)

    assert "SimulationRuntime" in representation
    assert runtime.name in representation
    assert "RecordingSolver" in representation
    assert "has_run=False" in representation
    assert "has_result=False" in representation
    assert "has_solution=False" in representation