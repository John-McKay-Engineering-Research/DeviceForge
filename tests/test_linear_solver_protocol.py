from __future__ import annotations

import numpy as np

from deviceforge.linalg import (
    LinearSolveResult,
    LinearSolverProtocol,
    LinearSystem,
)


class RecordingLinearSolver:
    """Small structurally compatible test solver."""

    name = "recording_linear_solver"
    backend_name = "numpy"

    def __init__(self) -> None:
        self.call_count = 0
        self.last_system: LinearSystem | None = None

    def solve(
        self,
        system: LinearSystem,
    ) -> LinearSolveResult:
        self.call_count += 1
        self.last_system = system

        solution = np.linalg.solve(
            system.matrix,
            system.right_hand_side,
        )

        residual = system.residual_norm(
            solution
        )

        return LinearSolveResult(
            solution=solution,
            converged=True,
            iterations=1,
            residual_history=np.asarray(
                [residual],
                dtype=np.float64,
            ),
            solver_name=self.name,
            backend_name=self.backend_name,
            termination_reason="direct_solve_completed",
        )


class InvalidLinearSolver:
    """Class that deliberately does not satisfy the protocol."""

    def execute(self) -> None:
        pass


def test_recording_solver_satisfies_protocol() -> None:
    solver = RecordingLinearSolver()

    assert isinstance(
        solver,
        LinearSolverProtocol,
    )


def test_invalid_solver_does_not_satisfy_protocol() -> None:
    solver = InvalidLinearSolver()

    assert not isinstance(
        solver,
        LinearSolverProtocol,
    )


def test_structural_solver_solves_linear_system() -> None:
    system = LinearSystem(
        matrix=np.asarray(
            [
                [2.0, 0.0],
                [0.0, 4.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.asarray(
            [6.0, 8.0],
            dtype=np.float64,
        ),
    )

    solver = RecordingLinearSolver()
    result = solver.solve(system)

    np.testing.assert_allclose(
        result.solution,
        [3.0, 2.0],
    )

    assert result.converged
    assert result.iterations == 1
    assert result.final_residual is not None
    assert result.solver_name == solver.name
    assert result.backend_name == solver.backend_name
    assert result.termination_reason == (
        "direct_solve_completed"
    )

    assert solver.call_count == 1
    assert solver.last_system is system


def test_solution_has_expected_shape() -> None:
    system = LinearSystem(
        matrix=np.eye(3),
        right_hand_side=np.asarray(
            [1.0, 2.0, 3.0],
            dtype=np.float64,
        ),
    )

    result = RecordingLinearSolver().solve(
        system
    )

    assert result.solution.shape == (3,)
    assert result.solution_size == 3


def test_solution_residual_is_zero() -> None:
    system = LinearSystem(
        matrix=np.asarray(
            [
                [3.0, 1.0],
                [1.0, 2.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.asarray(
            [9.0, 8.0],
            dtype=np.float64,
        ),
    )

    result = RecordingLinearSolver().solve(
        system
    )

    assert system.residual_norm(
        result.solution
    ) <= 1.0e-12

    assert result.final_residual is not None
    assert result.final_residual <= 1.0e-12