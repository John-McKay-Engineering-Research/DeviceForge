from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from deviceforge.linalg import LinearSystem


def test_create_dense_linear_system() -> None:
    matrix = np.asarray(
        [
            [2.0, -1.0],
            [-1.0, 2.0],
        ],
        dtype=np.float64,
    )

    right_hand_side = np.asarray(
        [1.0, 0.0],
        dtype=np.float64,
    )

    system = LinearSystem(
        matrix=matrix,
        right_hand_side=right_hand_side,
    )

    assert system.shape == (2, 2)
    assert system.number_of_equations == 2
    assert system.is_dense
    assert not system.is_sparse


def test_create_sparse_linear_system() -> None:
    matrix = csr_matrix(
        [
            [2.0, -1.0],
            [-1.0, 2.0],
        ],
        dtype=np.float64,
    )

    system = LinearSystem(
        matrix=matrix,
        right_hand_side=np.asarray(
            [1.0, 0.0],
            dtype=np.float64,
        ),
    )

    assert system.shape == (2, 2)
    assert system.is_sparse
    assert not system.is_dense
    assert system.number_of_nonzero_entries == 4


def test_linear_system_normalises_name() -> None:
    system = LinearSystem(
        matrix=np.eye(2),
        right_hand_side=np.zeros(2),
        name="  test system  ",
    )

    assert system.name == "test system"


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_linear_system_rejects_empty_name(
    name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        LinearSystem(
            matrix=np.eye(2),
            right_hand_side=np.zeros(2),
            name=name,
        )


def test_linear_system_rejects_non_square_matrix() -> None:
    with pytest.raises(
        ValueError,
        match="must be square",
    ):
        LinearSystem(
            matrix=np.zeros((2, 3)),
            right_hand_side=np.zeros(2),
        )


def test_linear_system_rejects_non_matrix() -> None:
    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        LinearSystem(
            matrix=np.zeros(3),
            right_hand_side=np.zeros(3),
        )


def test_linear_system_rejects_empty_matrix() -> None:
    with pytest.raises(
        ValueError,
        match="at least one equation",
    ):
        LinearSystem(
            matrix=np.zeros((0, 0)),
            right_hand_side=np.zeros(0),
        )


def test_linear_system_rejects_rhs_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="length must match",
    ):
        LinearSystem(
            matrix=np.eye(3),
            right_hand_side=np.zeros(2),
        )


def test_linear_system_rejects_multidimensional_rhs() -> None:
    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        LinearSystem(
            matrix=np.eye(2),
            right_hand_side=np.zeros((2, 1)),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_dense_matrix_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    matrix = np.eye(2)
    matrix[0, 0] = invalid_value

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        LinearSystem(
            matrix=matrix,
            right_hand_side=np.zeros(2),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_sparse_matrix_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    matrix = csr_matrix(
        [
            [invalid_value, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        LinearSystem(
            matrix=matrix,
            right_hand_side=np.zeros(2),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_rhs_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    right_hand_side = np.zeros(2)
    right_hand_side[0] = invalid_value

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        LinearSystem(
            matrix=np.eye(2),
            right_hand_side=right_hand_side,
        )


def test_dense_matrix_is_immutable() -> None:
    system = LinearSystem(
        matrix=np.eye(2),
        right_hand_side=np.zeros(2),
    )

    with pytest.raises(ValueError):
        system.matrix[0, 0] = 2.0


def test_rhs_is_immutable() -> None:
    system = LinearSystem(
        matrix=np.eye(2),
        right_hand_side=np.zeros(2),
    )

    with pytest.raises(ValueError):
        system.right_hand_side[0] = 2.0


def test_number_of_nonzero_entries() -> None:
    system = LinearSystem(
        matrix=np.asarray(
            [
                [1.0, 0.0],
                [2.0, 3.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.zeros(2),
    )

    assert system.number_of_nonzero_entries == 3
    assert system.density == pytest.approx(0.75)


def test_matrix_vector_product() -> None:
    system = LinearSystem(
        matrix=np.asarray(
            [
                [2.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.zeros(2),
    )

    result = system.matrix_vector_product(
        np.asarray(
            [4.0, 5.0],
            dtype=np.float64,
        )
    )

    np.testing.assert_allclose(
        result,
        [8.0, 15.0],
    )


def test_residual() -> None:
    system = LinearSystem(
        matrix=np.asarray(
            [
                [2.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.asarray(
            [8.0, 15.0],
            dtype=np.float64,
        ),
    )

    residual = system.residual(
        np.asarray(
            [4.0, 5.0],
            dtype=np.float64,
        )
    )

    np.testing.assert_allclose(
        residual,
        [0.0, 0.0],
    )


def test_residual_norm() -> None:
    system = LinearSystem(
        matrix=np.eye(2),
        right_hand_side=np.asarray(
            [1.0, 2.0],
            dtype=np.float64,
        ),
    )

    residual_norm = system.residual_norm(
        np.asarray(
            [1.0, 3.0],
            dtype=np.float64,
        )
    )

    assert residual_norm == pytest.approx(1.0)


def test_copy_returns_independent_dense_system() -> None:
    system = LinearSystem(
        matrix=np.eye(2),
        right_hand_side=np.zeros(2),
        name="copy_test",
    )

    copied_system = system.copy()

    assert copied_system is not system
    assert copied_system.name == system.name

    assert copied_system.matrix is not (
        system.matrix
    )

    assert copied_system.right_hand_side is not (
        system.right_hand_side
    )

    np.testing.assert_allclose(
        copied_system.matrix,
        system.matrix,
    )


def test_copy_returns_independent_sparse_system() -> None:
    system = LinearSystem(
        matrix=csr_matrix(np.eye(2)),
        right_hand_side=np.zeros(2),
    )

    copied_system = system.copy()

    assert copied_system is not system
    assert copied_system.matrix is not system.matrix

    np.testing.assert_allclose(
        copied_system.matrix.toarray(),
        system.matrix.toarray(),
    )