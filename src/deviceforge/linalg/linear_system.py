from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import (
    issparse,
    spmatrix,
)


DenseMatrix: TypeAlias = NDArray[np.float64]
MatrixType: TypeAlias = DenseMatrix | spmatrix


@dataclass(frozen=True, slots=True)
class LinearSystem:
    """
    Validated representation of a linear system.

    A linear system has the mathematical form

        A x = b

    where

        A:
            Square dense or sparse coefficient matrix.

        x:
            Unknown solution vector.

        b:
            Right-hand-side vector.

    The class stores no physical interpretation and performs no numerical
    solution. Physics solvers assemble LinearSystem instances, while linear
    solver implementations consume them.

    Parameters
    ----------
    matrix:
        Square NumPy dense matrix or SciPy sparse matrix.

    right_hand_side:
        One-dimensional vector whose length matches the matrix dimension.

    name:
        Optional human-readable system name.
    """

    matrix: MatrixType
    right_hand_side: NDArray[np.float64]
    name: str = "linear_system"

    def __post_init__(self) -> None:
        """Validate and normalise the linear system."""

        if not isinstance(self.name, str):
            raise TypeError(
                "Linear-system name must be a string."
            )

        normalised_name = self.name.strip()

        if not normalised_name:
            raise ValueError(
                "Linear-system name must not be empty."
            )

        object.__setattr__(
            self,
            "name",
            normalised_name,
        )

        matrix = self.matrix

        if issparse(matrix):
            normalised_matrix = matrix.astype(
                np.float64,
                copy=True,
            )

            self._validate_sparse_matrix(
                normalised_matrix
            )

        else:
            normalised_matrix = np.asarray(
                matrix,
                dtype=np.float64,
            )

            self._validate_dense_matrix(
                normalised_matrix
            )

            normalised_matrix = (
                normalised_matrix.copy()
            )

            normalised_matrix.setflags(
                write=False
            )

        right_hand_side = np.asarray(
            self.right_hand_side,
            dtype=np.float64,
        )

        if right_hand_side.ndim != 1:
            raise ValueError(
                "Linear-system right-hand side must be "
                "one-dimensional."
            )

        if right_hand_side.shape[0] != (
            normalised_matrix.shape[0]
        ):
            raise ValueError(
                "Right-hand-side length must match the "
                "number of matrix rows. "
                f"Received {right_hand_side.shape[0]} values "
                f"for a matrix with "
                f"{normalised_matrix.shape[0]} rows."
            )

        if not np.all(
            np.isfinite(right_hand_side)
        ):
            raise ValueError(
                "Linear-system right-hand side must not "
                "contain NaN or infinite values."
            )

        immutable_right_hand_side = (
            right_hand_side.copy()
        )

        immutable_right_hand_side.setflags(
            write=False
        )

        object.__setattr__(
            self,
            "matrix",
            normalised_matrix,
        )

        object.__setattr__(
            self,
            "right_hand_side",
            immutable_right_hand_side,
        )

    @staticmethod
    def _validate_dense_matrix(
        matrix: NDArray[np.float64],
    ) -> None:
        """Validate a dense coefficient matrix."""

        if matrix.ndim != 2:
            raise ValueError(
                "Linear-system matrix must be "
                "two-dimensional."
            )

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Linear-system matrix must be square. "
                f"Received shape {matrix.shape}."
            )

        if matrix.shape[0] == 0:
            raise ValueError(
                "Linear-system matrix must contain at "
                "least one equation."
            )

        if not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError(
                "Linear-system matrix must not contain "
                "NaN or infinite values."
            )

    @staticmethod
    def _validate_sparse_matrix(
        matrix: spmatrix,
    ) -> None:
        """Validate a sparse coefficient matrix."""

        if matrix.ndim != 2:
            raise ValueError(
                "Linear-system matrix must be "
                "two-dimensional."
            )

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                "Linear-system matrix must be square. "
                f"Received shape {matrix.shape}."
            )

        if matrix.shape[0] == 0:
            raise ValueError(
                "Linear-system matrix must contain at "
                "least one equation."
            )

        if not np.all(
            np.isfinite(matrix.data)
        ):
            raise ValueError(
                "Linear-system matrix must not contain "
                "NaN or infinite values."
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the coefficient-matrix shape."""

        return self.matrix.shape

    @property
    def number_of_equations(self) -> int:
        """Return the number of equations and unknowns."""

        return self.matrix.shape[0]

    @property
    def is_sparse(self) -> bool:
        """Return whether the coefficient matrix is sparse."""

        return bool(
            issparse(self.matrix)
        )

    @property
    def is_dense(self) -> bool:
        """Return whether the coefficient matrix is dense."""

        return not self.is_sparse

    @property
    def number_of_nonzero_entries(self) -> int:
        """Return the number of nonzero matrix entries."""

        if self.is_sparse:
            return int(self.matrix.nnz)

        return int(
            np.count_nonzero(self.matrix)
        )

    @property
    def density(self) -> float:
        """
        Return the fraction of matrix entries that are nonzero.

        A completely dense nonzero matrix has density 1.0.
        """

        total_entries = (
            self.matrix.shape[0]
            * self.matrix.shape[1]
        )

        return (
            self.number_of_nonzero_entries
            / total_entries
        )

    def matrix_vector_product(
        self,
        vector: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Calculate the matrix-vector product A x.

        Parameters
        ----------
        vector:
            One-dimensional vector with one value per unknown.
        """

        normalised_vector = np.asarray(
            vector,
            dtype=np.float64,
        )

        if normalised_vector.ndim != 1:
            raise ValueError(
                "Matrix-vector product requires a "
                "one-dimensional vector."
            )

        if normalised_vector.shape[0] != (
            self.number_of_equations
        ):
            raise ValueError(
                "Vector length must match the number "
                "of equations."
            )

        if not np.all(
            np.isfinite(normalised_vector)
        ):
            raise ValueError(
                "Vector must not contain NaN or "
                "infinite values."
            )

        result = (
            self.matrix
            @ normalised_vector
        )

        return np.asarray(
            result,
            dtype=np.float64,
        )

    def residual(
        self,
        solution: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Calculate the algebraic residual.

        The residual is

            r = A x - b.
        """

        return (
            self.matrix_vector_product(solution)
            - self.right_hand_side
        )

    def residual_norm(
        self,
        solution: NDArray[np.float64],
    ) -> float:
        """
        Return the infinity norm of the residual.

        The infinity norm is

            ||A x - b||_infinity.
        """

        return float(
            np.linalg.norm(
                self.residual(solution),
                ord=np.inf,
            )
        )

    def copy(self) -> LinearSystem:
        """Return an independent copy of the linear system."""

        if self.is_sparse:
            matrix_copy = self.matrix.copy()
        else:
            matrix_copy = np.array(
                self.matrix,
                copy=True,
            )

        return LinearSystem(
            matrix=matrix_copy,
            right_hand_side=np.array(
                self.right_hand_side,
                copy=True,
            ),
            name=self.name,
        )