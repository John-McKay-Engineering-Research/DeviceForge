from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

from deviceforge.linalg import (
    JacobiPreconditioner,
    LinearSystem,
    PreconditionerProtocol,
)


def create_diagonal_system() -> LinearSystem:
    """Return a small diagonal linear system."""

    return LinearSystem(
        matrix=csr_matrix(
            [
                [2.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 8.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.ones(
            3,
            dtype=np.float64,
        ),
    )


def test_jacobi_preconditioner_satisfies_protocol() -> None:
    preconditioner = JacobiPreconditioner()

    assert isinstance(
        preconditioner,
        PreconditionerProtocol,
    )


def test_jacobi_preconditioner_defaults() -> None:
    preconditioner = JacobiPreconditioner()

    assert preconditioner.name == "jacobi"
    assert preconditioner.backend_name == "scipy"

    assert preconditioner.diagonal_tolerance == pytest.approx(
        1.0e-15
    )


@pytest.mark.parametrize(
    "diagonal_tolerance",
    [
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_jacobi_rejects_invalid_diagonal_tolerance(
    diagonal_tolerance: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite and non-negative",
    ):
        JacobiPreconditioner(
            diagonal_tolerance=diagonal_tolerance
        )


def test_jacobi_rejects_boolean_diagonal_tolerance() -> None:
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        JacobiPreconditioner(
            diagonal_tolerance=True
        )


def test_jacobi_rejects_non_system() -> None:
    with pytest.raises(
        TypeError,
        match="LinearSystem",
    ):
        JacobiPreconditioner().build(
            "invalid"
        )


def test_jacobi_returns_linear_operator() -> None:
    operator = JacobiPreconditioner().build(
        create_diagonal_system()
    )

    assert isinstance(
        operator,
        LinearOperator,
    )

    assert operator.shape == (3, 3)


def test_jacobi_applies_inverse_diagonal() -> None:
    operator = JacobiPreconditioner().build(
        create_diagonal_system()
    )

    vector = np.asarray(
        [2.0, 8.0, 24.0],
        dtype=np.float64,
    )

    result = operator @ vector

    np.testing.assert_allclose(
        result,
        [1.0, 2.0, 3.0],
    )


def test_jacobi_uses_only_matrix_diagonal() -> None:
    system = LinearSystem(
        matrix=csr_matrix(
            [
                [2.0, 100.0],
                [100.0, 4.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.ones(2),
    )

    operator = JacobiPreconditioner().build(
        system
    )

    result = operator @ np.asarray(
        [2.0, 8.0],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result,
        [1.0, 2.0],
    )


def test_jacobi_rejects_zero_diagonal() -> None:
    system = LinearSystem(
        matrix=csr_matrix(
            [
                [2.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.ones(2),
    )

    with pytest.raises(
        ValueError,
        match="diagonal value",
    ):
        JacobiPreconditioner().build(
            system
        )


def test_jacobi_rejects_diagonal_below_tolerance() -> None:
    system = LinearSystem(
        matrix=csr_matrix(
            [
                [1.0e-16, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.ones(2),
    )

    with pytest.raises(
        ValueError,
        match="diagonal value",
    ):
        JacobiPreconditioner(
            diagonal_tolerance=1.0e-15,
        ).build(system)