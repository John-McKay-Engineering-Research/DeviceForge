from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import (
    MatrixRankWarning,
    spsolve,
)

from .linear_system import LinearSystem
from .result import LinearSolveResult


@dataclass(frozen=True, slots=True)
class SparseDirectSolver:
    """
    Direct solver for sparse linear systems.

    The solver uses SciPy's sparse direct solution routine to solve

        A x = b

    where ``A`` is a SciPy sparse matrix.

    The class satisfies LinearSolverProtocol structurally. It does not
    inherit from or import the protocol.

    Parameters
    ----------
    name:
        Human-readable solver name.

    backend_name:
        Name of the numerical backend.
    """

    name: str = "sparse_direct"
    backend_name: str = "scipy"

    def __post_init__(self) -> None:
        """Validate and normalise solver configuration."""

        if not isinstance(self.name, str):
            raise TypeError(
                "Sparse-direct solver name must be a string."
            )

        normalised_name = self.name.strip()

        if not normalised_name:
            raise ValueError(
                "Sparse-direct solver name must not be empty."
            )

        if not isinstance(self.backend_name, str):
            raise TypeError(
                "Sparse-direct backend name must be a string."
            )

        normalised_backend_name = (
            self.backend_name.strip()
        )

        if not normalised_backend_name:
            raise ValueError(
                "Sparse-direct backend name must not be empty."
            )

        object.__setattr__(
            self,
            "name",
            normalised_name,
        )

        object.__setattr__(
            self,
            "backend_name",
            normalised_backend_name,
        )

    def solve(
            self,
            system: LinearSystem,
    ) -> LinearSolveResult:
        """
        Solve a sparse linear system.

        Parameters
        ----------
        system:
            Validated sparse linear system representing ``A x = b``.

        Returns
        -------
        NDArray[np.float64]
            One-dimensional finite solution vector.

        Raises
        ------
        TypeError
            If ``system`` is not a LinearSystem.

        ValueError
            If the system matrix is dense.

        RuntimeError
            If the sparse matrix is singular or the numerical solution
            fails validation.
        """

        if not isinstance(system, LinearSystem):
            raise TypeError(
                "SparseDirectSolver requires a LinearSystem instance."
            )

        if not system.is_sparse:
            raise ValueError(
                "SparseDirectSolver requires a sparse coefficient matrix."
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error",
                    MatrixRankWarning,
                )

                solution = spsolve(
                    system.matrix,
                    system.right_hand_side,
                )

        except MatrixRankWarning as exc:
            raise RuntimeError(
                "Sparse direct solve failed because the coefficient "
                "matrix is singular."
            ) from exc

        except Exception as exc:
            raise RuntimeError(
                "Sparse direct solve failed."
            ) from exc

        normalised_solution = np.asarray(
            solution,
            dtype=np.float64,
        )

        if normalised_solution.ndim != 1:
            raise RuntimeError(
                "Sparse direct solver returned a solution that is "
                "not one-dimensional."
            )

        expected_shape = (
            system.number_of_equations,
        )

        if normalised_solution.shape != expected_shape:
            raise RuntimeError(
                "Sparse direct solver returned an unexpected solution "
                f"shape. Expected {expected_shape}, received "
                f"{normalised_solution.shape}."
            )

        if not np.all(
            np.isfinite(normalised_solution)
        ):
            raise RuntimeError(
                "Sparse direct solver returned NaN or infinite values."
            )

        residual = system.residual_norm(
            normalised_solution
        )

        return LinearSolveResult(
            solution=normalised_solution,
            converged=True,
            iterations=1,
            residual_history=np.asarray(
                [residual],
                dtype=np.float64,
            ),
            solver_name=self.name,
            backend_name=self.backend_name,
            termination_reason="direct_solve_completed",
            metadata={
                "matrix_input_storage": "sparse",
                "scipy_solver": "spsolve",
            },
        )