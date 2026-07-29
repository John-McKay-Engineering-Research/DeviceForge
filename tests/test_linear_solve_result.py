from __future__ import annotations

import numpy as np
import pytest

from deviceforge.linalg import LinearSolveResult


def create_result() -> LinearSolveResult:
    return LinearSolveResult(
        solution=[1.0, 2.0],
        converged=True,
        iterations=1,
        residual_history=[1.0e-14],
        solver_name="test_solver",
        backend_name="test_backend",
        termination_reason="completed",
        metadata={
            "test": True,
        },
    )


def test_create_linear_solve_result() -> None:
    result = create_result()

    np.testing.assert_allclose(
        result.solution,
        [1.0, 2.0],
    )

    assert result.converged
    assert result.iterations == 1
    assert result.final_residual == pytest.approx(
        1.0e-14
    )
    assert result.solution_size == 2


def test_result_normalises_text() -> None:
    result = LinearSolveResult(
        solution=[1.0],
        converged=True,
        iterations=1,
        residual_history=[0.0],
        solver_name="  solver  ",
        backend_name="  backend  ",
        termination_reason="  completed  ",
    )

    assert result.solver_name == "solver"
    assert result.backend_name == "backend"
    assert result.termination_reason == "completed"


def test_result_rejects_residual_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="length must match",
    ):
        LinearSolveResult(
            solution=[1.0],
            converged=True,
            iterations=2,
            residual_history=[0.0],
            solver_name="solver",
            backend_name="backend",
            termination_reason="completed",
        )


def test_result_rejects_non_finite_solution() -> None:
    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        LinearSolveResult(
            solution=[np.nan],
            converged=False,
            iterations=0,
            residual_history=[],
            solver_name="solver",
            backend_name="backend",
            termination_reason="failed",
        )


def test_result_arrays_are_immutable() -> None:
    result = create_result()

    with pytest.raises(ValueError):
        result.solution[0] = 5.0

    with pytest.raises(ValueError):
        result.residual_history[0] = 1.0


def test_result_metadata_is_read_only() -> None:
    result = create_result()

    with pytest.raises(TypeError):
        result.metadata["new"] = 1