from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .linear_system import LinearSystem
from .result import LinearSolveResult


@dataclass(frozen=True, slots=True)
class DenseDirectSolver:
    """
    Direct linear solver using NumPy dense linear algebra.

    The solver evaluates

        A x = b

    using ``numpy.linalg.solve``.

    Sparse input matrices are converted to dense NumPy arrays before
    solution. This makes DenseDirectSolver useful as a reference backend
    for validating sparse and iterative implementations on small systems.

    It should not be used for large sparse systems because dense storage
    scales quadratically with the number of equations.

    The class satisfies LinearSolverProtocol structurally.

    Parameters
    ----------
    name:
        Human-readable linear-solver name.

    backend_name:
        Name of the numerical backend.
    """

    name: str = "dense_direct"
    backend_name: str = "numpy"

    def __post_init__(self) -> None:
        """Validate and normalise solver configuration."""

        if not isinstance(self.name, str):
            raise TypeError(
                "Dense-direct solver name must be a string."
            )

        normalised_name = self.name.strip()

        if not normalised_name:
            raise ValueError(
                "Dense-direct solver name must not be empty."
            )

        if not isinstance(self.backend_name, str):
            raise TypeError(
                "Dense-direct backend name must be a string."
            )

        normalised_backend_name = (
            self.backend_name.strip()
        )

        if not normalised_backend_name:
            raise ValueError(
                "Dense-direct backend name must not be empty."
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
        Solve a linear system using NumPy's dense direct solver.

        Parameters
        ----------
        system:
            Validated DeviceForge linear system.

        Returns
        -------
        NDArray[np.float64]
            One-dimensional finite solution vector.

        Raises
        ------
        TypeError
            If ``system`` is not a LinearSystem.

        RuntimeError
            If the matrix is singular or NumPy cannot solve the system.
        """

        if not isinstance(system, LinearSystem):
            raise TypeError(
                "DenseDirectSolver requires a LinearSystem instance."
            )

        if system.is_sparse:
            dense_matrix = system.matrix.toarray()
        else:
            dense_matrix = np.asarray(
                system.matrix,
                dtype=np.float64,
            )

        try:
            solution = np.linalg.solve(
                dense_matrix,
                system.right_hand_side,
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Dense direct solve failed because the coefficient "
                "matrix is singular or numerically invalid."
            ) from exc

        normalised_solution = np.asarray(
            solution,
            dtype=np.float64,
        )

        expected_shape = (
            system.number_of_equations,
        )

        if normalised_solution.ndim != 1:
            raise RuntimeError(
                "Dense direct solver returned a solution that is "
                "not one-dimensional."
            )

        if normalised_solution.shape != expected_shape:
            raise RuntimeError(
                "Dense direct solver returned an unexpected solution "
                f"shape. Expected {expected_shape}, received "
                f"{normalised_solution.shape}."
            )

        if not np.all(
            np.isfinite(normalised_solution)
        ):
            raise RuntimeError(
                "Dense direct solver returned NaN or infinite values."
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
                "matrix_input_storage": (
                    "sparse"
                    if system.is_sparse
                    else "dense"
                ),
                "matrix_conversion": (
                    "sparse_to_dense"
                    if system.is_sparse
                    else "none"
                ),
            },
        )