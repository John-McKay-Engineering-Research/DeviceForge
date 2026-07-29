from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

from deviceforge.linalg import (
    IdentityPreconditioner,
    LinearSystem,
    PreconditionerProtocol,
)


def create_test_system() -> LinearSystem:
    """Return a small SPD linear system."""

    return LinearSystem(
        matrix=csr_matrix(
            [
                [4.0, -1.0],
                [-1.0, 3.0],
            ],
            dtype=np.float64,
        ),
        right_hand_side=np.asarray(
            [1.0, 2.0],
            dtype=np.float64,
        ),
    )


def test_identity_preconditioner_satisfies_protocol() -> None:
    preconditioner = IdentityPreconditioner()

    assert isinstance(
        preconditioner,
        PreconditionerProtocol,
    )


def test_identity_preconditioner_defaults() -> None:
    preconditioner = IdentityPreconditioner()

    assert preconditioner.name == "identity"
    assert preconditioner.backend_name == "scipy"


def test_identity_preconditioner_normalises_names() -> None:
    preconditioner = IdentityPreconditioner(
        name="  no preconditioning  ",
        backend_name="  scipy backend  ",
    )

    assert preconditioner.name == (
        "no preconditioning"
    )

    assert preconditioner.backend_name == (
        "scipy backend"
    )


@pytest.mark.parametrize(
    "name",
    ["", " ", "\t"],
)
def test_identity_preconditioner_rejects_empty_name(
    name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        IdentityPreconditioner(
            name=name
        )


def test_identity_preconditioner_rejects_non_system() -> None:
    with pytest.raises(
        TypeError,
        match="LinearSystem",
    ):
        IdentityPreconditioner().build(
            "invalid"
        )


def test_identity_preconditioner_returns_linear_operator() -> None:
    operator = IdentityPreconditioner().build(
        create_test_system()
    )

    assert isinstance(
        operator,
        LinearOperator,
    )

    assert operator.shape == (2, 2)


def test_identity_preconditioner_leaves_vector_unchanged() -> None:
    operator = IdentityPreconditioner().build(
        create_test_system()
    )

    vector = np.asarray(
        [2.0, -3.0],
        dtype=np.float64,
    )

    result = operator @ vector

    np.testing.assert_allclose(
        result,
        vector,
    )

    assert result is not vector