import numpy as np
import pytest

from deviceforge.solvers import solve_tridiagonal


def test_tridiagonal_solver_matches_numpy() -> None:
    lower = np.array([-1.0, -1.0, -1.0])
    diagonal = np.array([2.0, 2.0, 2.0, 2.0])
    upper = np.array([-1.0, -1.0, -1.0])
    rhs = np.array([1.0, 0.0, 0.0, 1.0])

    matrix = np.diag(diagonal)
    matrix += np.diag(lower, k=-1)
    matrix += np.diag(upper, k=1)

    expected = np.linalg.solve(matrix, rhs)

    result = solve_tridiagonal(
        lower,
        diagonal,
        upper,
        rhs,
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_single_unknown() -> None:
    result = solve_tridiagonal(
        np.array([]),
        np.array([2.0]),
        np.array([]),
        np.array([6.0]),
    )

    np.testing.assert_allclose(result, [3.0])


def test_invalid_diagonal_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="Lower diagonal"):
        solve_tridiagonal(
            lower=[1.0, 1.0],
            diagonal=[2.0, 2.0],
            upper=[1.0],
            right_hand_side=[1.0, 1.0],
        )


def test_zero_pivot_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero leading pivot"):
        solve_tridiagonal(
            lower=[1.0],
            diagonal=[0.0, 2.0],
            upper=[1.0],
            right_hand_side=[1.0, 1.0],
        )