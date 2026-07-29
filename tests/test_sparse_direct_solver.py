from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import (
    csc_matrix,
    csr_matrix,
)

from deviceforge.linalg import (
    LinearSolveResult,
    LinearSolverProtocol,
    LinearSystem,
    SparseDirectSolver,
)


def create_sparse_system() -> LinearSystem:
    """Return a small nonsingular sparse linear system."""

    matrix = csr_matrix(
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
        name="sparse_test_system",
    )


def test_sparse_direct_solver_satisfies_protocol() -> None:
    solver = SparseDirectSolver()

    assert isinstance(
        solver,
        LinearSolverProtocol,
    )


def test_sparse_direct_solver_defaults() -> None:
    solver = SparseDirectSolver()

    assert solver.name == "sparse_direct"
    assert solver.backend_name == "scipy"


def test_sparse_direct_solver_normalises_names() -> None:
    solver = SparseDirectSolver(
        name="  direct solver  ",
        backend_name="  scipy backend  ",
    )

    assert solver.name == "direct solver"
    assert solver.backend_name == "scipy backend"


@pytest.mark.parametrize(
    "name",
    ["", " ", "\t"],
)
def test_sparse_direct_solver_rejects_empty_name(
    name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        SparseDirectSolver(name=name)


@pytest.mark.parametrize(
    "backend_name",
    ["", " ", "\t"],
)
def test_sparse_direct_solver_rejects_empty_backend_name(
    backend_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="backend name must not be empty",
    ):
        SparseDirectSolver(
            backend_name=backend_name
        )


def test_sparse_direct_solver_rejects_non_system() -> None:
    with pytest.raises(
        TypeError,
        match="LinearSystem",
    ):
        SparseDirectSolver().solve("invalid")


def test_sparse_direct_solver_rejects_dense_matrix() -> None:
    system = LinearSystem(
        matrix=np.eye(2),
        right_hand_side=np.asarray(
            [1.0, 2.0],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="sparse coefficient matrix",
    ):
        SparseDirectSolver().solve(system)


def test_sparse_direct_solver_returns_result() -> None:
    result = SparseDirectSolver().solve(
        create_sparse_system()
    )

    assert isinstance(
        result,
        LinearSolveResult,
    )
    assert result.converged
    assert result.iterations == 1
    assert result.solver_name == "sparse_direct"
    assert result.backend_name == "scipy"
    assert result.termination_reason == (
        "direct_solve_completed"
    )


def test_sparse_direct_solver_returns_expected_solution() -> None:
    system = create_sparse_system()

    result = SparseDirectSolver().solve(
        system
    )

    expected = np.linalg.solve(
        system.matrix.toarray(),
        system.right_hand_side,
    )

    np.testing.assert_allclose(
        result.solution,
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_sparse_direct_solver_solution_shape() -> None:
    system = create_sparse_system()

    result = SparseDirectSolver().solve(
        system
    )

    assert result.solution.shape == (
        system.number_of_equations,
    )


def test_sparse_direct_solver_has_small_residual() -> None:
    result = SparseDirectSolver().solve(
        create_sparse_system()
    )

    assert result.final_residual is not None
    assert result.final_residual <= 1.0e-12


def test_sparse_direct_solver_accepts_csc_matrix() -> None:
    matrix = csc_matrix(
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

    result = SparseDirectSolver().solve(
        system
    )

    np.testing.assert_allclose(
        result.solution,
        [3.0, 2.0],
    )


def test_sparse_direct_solver_rejects_singular_system() -> None:
    matrix = csr_matrix(
        [
            [1.0, 2.0],
            [2.0, 4.0],
        ],
        dtype=np.float64,
    )

    system = LinearSystem(
        matrix=matrix,
        right_hand_side=np.asarray(
            [3.0, 6.0],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="singular",
    ):
        SparseDirectSolver().solve(system)


def test_sparse_solution_is_independent_array() -> None:
    system = create_sparse_system()
    solver = SparseDirectSolver()

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