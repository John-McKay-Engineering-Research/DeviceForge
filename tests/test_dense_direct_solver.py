from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from deviceforge.linalg import (
    DenseDirectSolver,
    LinearSolveResult,
    LinearSolverProtocol,
    LinearSystem,
)


def create_dense_system() -> LinearSystem:
    """Return a small nonsingular dense linear system."""

    matrix = np.asarray(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 3.0],
        ],
        dtype=np.float64,
    )

    right_hand_side = np.asarray(
        [15.0, 10.0, 10.0],
        dtype=np.float64,
    )

    return LinearSystem(
        matrix=matrix,
        right_hand_side=right_hand_side,
        name="dense_test_system",
    )


def test_dense_direct_solver_satisfies_protocol() -> None:
    solver = DenseDirectSolver()

    assert isinstance(
        solver,
        LinearSolverProtocol,
    )


def test_dense_direct_solver_defaults() -> None:
    solver = DenseDirectSolver()

    assert solver.name == "dense_direct"
    assert solver.backend_name == "numpy"


def test_dense_direct_solver_normalises_names() -> None:
    solver = DenseDirectSolver(
        name="  reference solver  ",
        backend_name="  numpy backend  ",
    )

    assert solver.name == "reference solver"
    assert solver.backend_name == "numpy backend"


@pytest.mark.parametrize(
    "name",
    ["", " ", "\t"],
)
def test_dense_direct_solver_rejects_empty_name(
    name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        DenseDirectSolver(name=name)


@pytest.mark.parametrize(
    "backend_name",
    ["", " ", "\t"],
)
def test_dense_direct_solver_rejects_empty_backend_name(
    backend_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="backend name must not be empty",
    ):
        DenseDirectSolver(
            backend_name=backend_name
        )


def test_dense_direct_solver_rejects_non_system() -> None:
    with pytest.raises(
        TypeError,
        match="LinearSystem",
    ):
        DenseDirectSolver().solve("invalid")


def test_dense_direct_solver_returns_result() -> None:
    result = DenseDirectSolver().solve(
        create_dense_system()
    )

    assert isinstance(
        result,
        LinearSolveResult,
    )
    assert result.converged
    assert result.iterations == 1
    assert result.solver_name == "dense_direct"
    assert result.backend_name == "numpy"
    assert result.termination_reason == (
        "direct_solve_completed"
    )


def test_dense_direct_solver_returns_expected_solution() -> None:
    system = create_dense_system()

    result = DenseDirectSolver().solve(
        system
    )

    expected = np.linalg.solve(
        system.matrix,
        system.right_hand_side,
    )

    np.testing.assert_allclose(
        result.solution,
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_dense_direct_solver_accepts_sparse_input() -> None:
    matrix = csr_matrix(
        [
            [2.0, 0.0],
            [0.0, 4.0],
        ],
        dtype=np.float64,
    )

    system = LinearSystem(
        matrix=matrix,
        right_hand_side=np.asarray(
            [6.0, 8.0],
            dtype=np.float64,
        ),
    )

    result = DenseDirectSolver().solve(
        system
    )

    np.testing.assert_allclose(
        result.solution,
        [3.0, 2.0],
    )

    assert result.metadata[
        "matrix_input_storage"
    ] == "sparse"
    assert result.metadata[
        "matrix_conversion"
    ] == "sparse_to_dense"


def test_dense_direct_solver_solution_shape() -> None:
    system = create_dense_system()

    result = DenseDirectSolver().solve(
        system
    )

    assert result.solution.shape == (
        system.number_of_equations,
    )


def test_dense_direct_solver_has_small_residual() -> None:
    result = DenseDirectSolver().solve(
        create_dense_system()
    )

    assert result.final_residual is not None
    assert result.final_residual <= 1.0e-12


def test_dense_direct_solver_rejects_singular_system() -> None:
    system = LinearSystem(
        matrix=np.asarray(
            [
                [1.0, 2.0],
                [2.0, 4.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.asarray(
            [3.0, 6.0],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="singular",
    ):
        DenseDirectSolver().solve(system)


def test_dense_solution_is_independent_array() -> None:
    system = create_dense_system()
    solver = DenseDirectSolver()

    first_result = solver.solve(system)
    second_result = solver.solve(system)

    assert first_result is not second_result
    assert first_result.solution is not (
        second_result.solution
    )

    np.testing.assert_allclose(
        first_result.solution,
        second_result.solution,
    )