from __future__ import annotations

import numpy as np
import pytest

from deviceforge import Field, SimulationResult
from deviceforge.runtime import RuntimeState


def create_result(
    simulation,
    *,
    converged: bool = True,
    iterations: int = 4,
) -> SimulationResult:
    """Create a valid SimulationResult for runtime-state tests."""

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
        solver_name="test_solver",
        backend_name="cpu",
        metadata={
            "test_value": 42,
        },
    )


def test_runtime_state_starts_empty() -> None:
    state = RuntimeState()

    assert state.current_solution is None
    assert state.previous_solution is None
    assert state.last_result is None

    assert state.iteration_count == 0
    assert state.converged is None
    assert state.residual_history == []
    assert state.elapsed_time is None
    assert state.metadata == {}

    assert not state.has_solution
    assert not state.has_previous_solution
    assert not state.has_result
    assert not state.has_run
    assert state.final_residual is None


def test_runtime_state_rejects_invalid_last_result() -> None:
    with pytest.raises(
        TypeError,
        match="last_result must be a SimulationResult",
    ):
        RuntimeState(
            last_result="invalid",
        )


def test_runtime_state_rejects_boolean_iteration_count() -> None:
    with pytest.raises(
        TypeError,
        match="Iteration count must be an integer",
    ):
        RuntimeState(
            iteration_count=True,
        )


def test_runtime_state_rejects_negative_iteration_count() -> None:
    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        RuntimeState(
            iteration_count=-1,
        )


def test_runtime_state_rejects_invalid_converged_value() -> None:
    with pytest.raises(
        TypeError,
        match="Boolean value or None",
    ):
        RuntimeState(
            converged=1,
        )


def test_runtime_state_rejects_negative_elapsed_time() -> None:
    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        RuntimeState(
            elapsed_time=-0.1,
        )


def test_runtime_state_rejects_non_finite_elapsed_time() -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        RuntimeState(
            elapsed_time=np.inf,
        )


def test_runtime_state_normalises_residual_history() -> None:
    state = RuntimeState(
        residual_history=np.asarray(
            [1.0, 0.1, 0.01],
            dtype=np.float64,
        ),
    )

    assert state.residual_history == [
        1.0,
        0.1,
        0.01,
    ]

    assert all(
        isinstance(value, float)
        for value in state.residual_history
    )


def test_runtime_state_rejects_negative_residual() -> None:
    with pytest.raises(
        ValueError,
        match="must not be negative",
    ):
        RuntimeState(
            residual_history=[
                1.0,
                -0.1,
            ],
        )


def test_runtime_state_final_residual() -> None:
    state = RuntimeState(
        residual_history=[
            1.0,
            0.1,
            0.01,
        ],
    )

    assert state.final_residual == pytest.approx(0.01)


def test_record_result_updates_runtime_state(
    simulation,
) -> None:
    state = RuntimeState()
    result = create_result(
        simulation,
        converged=True,
        iterations=4,
    )

    state.record_result(result)

    assert state.last_result is result
    assert state.current_solution is result.fields
    assert state.previous_solution is None

    assert state.iteration_count == result.iterations
    assert state.converged is result.converged

    assert state.residual_history == pytest.approx(
        result.residual_history.tolist()
    )

    assert state.elapsed_time == pytest.approx(
        result.runtime_seconds
    )

    assert state.metadata == result.metadata

    assert state.has_solution
    assert state.has_result
    assert state.has_run

    assert state.final_residual == pytest.approx(
        result.residual_history[-1]
    )


def test_record_result_retains_previous_solution(
    simulation,
) -> None:
    state = RuntimeState()

    first_result = create_result(
        simulation,
        iterations=2,
    )
    second_result = create_result(
        simulation,
        iterations=4,
    )

    state.record_result(first_result)
    first_solution = state.current_solution

    state.record_result(second_result)

    assert state.previous_solution is first_solution
    assert state.current_solution is second_result.fields
    assert state.last_result is second_result
    assert state.has_previous_solution


def test_record_result_rejects_invalid_result() -> None:
    state = RuntimeState()

    with pytest.raises(
        TypeError,
        match="result must be a SimulationResult",
    ):
        state.record_result("invalid")


def test_reset_clears_all_execution_state(
    simulation,
) -> None:
    state = RuntimeState()
    result = create_result(simulation)

    state.record_result(result)
    state.reset()

    assert state.current_solution is None
    assert state.previous_solution is None
    assert state.last_result is None

    assert state.iteration_count == 0
    assert state.converged is None
    assert state.residual_history == []
    assert state.elapsed_time is None
    assert state.metadata == {}

    assert not state.has_solution
    assert not state.has_previous_solution
    assert not state.has_result
    assert not state.has_run
    assert state.final_residual is None


def test_copy_returns_independent_state_containers(
    simulation,
) -> None:
    state = RuntimeState()
    result = create_result(simulation)

    state.record_result(result)

    copied_state = state.copy()

    assert copied_state is not state

    assert copied_state.current_solution is state.current_solution
    assert copied_state.previous_solution is state.previous_solution
    assert copied_state.last_result is state.last_result

    assert copied_state.iteration_count == state.iteration_count
    assert copied_state.converged is state.converged
    assert copied_state.elapsed_time == state.elapsed_time

    assert copied_state.residual_history == state.residual_history
    assert copied_state.residual_history is not state.residual_history

    assert copied_state.metadata == state.metadata
    assert copied_state.metadata is not state.metadata