from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import cg

from .linear_system import LinearSystem
from .result import LinearSolveResult


@dataclass(frozen=True, slots=True)
class ConjugateGradientSolver:
    """
    Iterative conjugate-gradient linear solver.

    Solves

        A x = b

    using SciPy's conjugate-gradient implementation.

    Conjugate Gradient requires the coefficient matrix to be symmetric
    positive definite. DeviceForge's Poisson assembly is designed to
    satisfy these requirements after symmetric Dirichlet elimination.

    Parameters
    ----------
    relative_tolerance:
        Relative convergence tolerance passed to SciPy.

    absolute_tolerance:
        Absolute convergence tolerance passed to SciPy.

    max_iterations:
        Maximum number of CG iterations. ``None`` allows SciPy to choose
        its default limit.

    initial_guess:
        Optional initial solution vector. Its shape is validated when a
        linear system is solved.

    name:
        Human-readable linear-solver name.

    backend_name:
        Numerical backend name.
    """

    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 0.0
    max_iterations: int | None = None
    initial_guess: ArrayLike | None = None
    name: str = "conjugate_gradient"
    backend_name: str = "scipy"

    def __post_init__(self) -> None:
        """Validate and normalise solver configuration."""

        if isinstance(
            self.relative_tolerance,
            bool,
        ) or not isinstance(
            self.relative_tolerance,
            (int, float),
        ):
            raise TypeError(
                "Relative tolerance must be a real number."
            )

        relative_tolerance = float(
            self.relative_tolerance
        )

        if (
            not np.isfinite(relative_tolerance)
            or relative_tolerance <= 0.0
        ):
            raise ValueError(
                "Relative tolerance must be finite and positive."
            )

        if isinstance(
            self.absolute_tolerance,
            bool,
        ) or not isinstance(
            self.absolute_tolerance,
            (int, float),
        ):
            raise TypeError(
                "Absolute tolerance must be a real number."
            )

        absolute_tolerance = float(
            self.absolute_tolerance
        )

        if (
            not np.isfinite(absolute_tolerance)
            or absolute_tolerance < 0.0
        ):
            raise ValueError(
                "Absolute tolerance must be finite and "
                "non-negative."
            )

        if self.max_iterations is not None:
            if (
                isinstance(self.max_iterations, bool)
                or not isinstance(
                    self.max_iterations,
                    int,
                )
            ):
                raise TypeError(
                    "Maximum iteration count must be an "
                    "integer or None."
                )

            if self.max_iterations <= 0:
                raise ValueError(
                    "Maximum iteration count must be positive."
                )

        name = self._normalise_text(
            self.name,
            "Conjugate-gradient solver name",
        )

        backend_name = self._normalise_text(
            self.backend_name,
            "Conjugate-gradient backend name",
        )

        object.__setattr__(
            self,
            "relative_tolerance",
            relative_tolerance,
        )

        object.__setattr__(
            self,
            "absolute_tolerance",
            absolute_tolerance,
        )

        object.__setattr__(
            self,
            "name",
            name,
        )

        object.__setattr__(
            self,
            "backend_name",
            backend_name,
        )

    @staticmethod
    def _normalise_text(
        value: str,
        label: str,
    ) -> str:
        """Validate and normalise a required string."""

        if not isinstance(value, str):
            raise TypeError(
                f"{label} must be a string."
            )

        normalised_value = value.strip()

        if not normalised_value:
            raise ValueError(
                f"{label} must not be empty."
            )

        return normalised_value

    def solve(
        self,
        system: LinearSystem,
    ) -> LinearSolveResult:
        """
        Solve a linear system using Conjugate Gradient.

        Parameters
        ----------
        system:
            Validated DeviceForge linear system.

        Returns
        -------
        LinearSolveResult
            Solution and iterative convergence diagnostics.

        Raises
        ------
        TypeError
            If ``system`` is not a LinearSystem.

        ValueError
            If the initial guess is invalid.

        RuntimeError
            If SciPy reports illegal input, numerical breakdown, or an
            invalid returned solution.
        """

        if not isinstance(system, LinearSystem):
            raise TypeError(
                "ConjugateGradientSolver requires a "
                "LinearSystem instance."
            )

        initial_guess = self._create_initial_guess(
            system
        )

        residual_history: list[float] = []

        def record_iteration(
            current_solution: NDArray[np.float64],
        ) -> None:
            """
            Record the infinity norm of the algebraic residual.

            SciPy supplies the current solution vector to the callback,
            rather than supplying the residual directly.
            """

            residual_history.append(
                system.residual_norm(
                    current_solution
                )
            )

        try:
            solution, information = cg(
                system.matrix,
                system.right_hand_side,
                x0=initial_guess,
                rtol=self.relative_tolerance,
                atol=self.absolute_tolerance,
                maxiter=self.max_iterations,
                callback=record_iteration,
            )
        except Exception as exc:
            raise RuntimeError(
                "Conjugate-gradient solve failed."
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
                "Conjugate Gradient returned a solution "
                "that is not one-dimensional."
            )

        if normalised_solution.shape != expected_shape:
            raise RuntimeError(
                "Conjugate Gradient returned an unexpected "
                f"solution shape. Expected {expected_shape}, "
                f"received {normalised_solution.shape}."
            )

        if not np.all(
            np.isfinite(normalised_solution)
        ):
            raise RuntimeError(
                "Conjugate Gradient returned NaN or "
                "infinite values."
            )

        final_residual = system.residual_norm(
            normalised_solution
        )

        # Ensure the last recorded residual corresponds exactly to the
        # solution returned by SciPy, without adding a false iteration.
        if residual_history:
            residual_history[-1] = final_residual

        if information == 0:
            converged = True
            termination_reason = (
                "convergence_tolerance_satisfied"
            )

        elif information > 0:
            converged = False
            termination_reason = (
                "maximum_iterations_reached"
            )

        else:
            raise RuntimeError(
                "Conjugate Gradient terminated because of "
                "illegal input or numerical breakdown. "
                f"SciPy information code: {information}."
            )

        iteration_count = len(
            residual_history
        )

        metadata: dict[str, Any] = {
            "scipy_solver": "cg",
            "scipy_information_code": int(
                information
            ),
            "relative_tolerance": (
                self.relative_tolerance
            ),
            "absolute_tolerance": (
                self.absolute_tolerance
            ),
            "maximum_iterations": (
                self.max_iterations
            ),
            "initial_guess_supplied": (
                self.initial_guess is not None
            ),
            "matrix_input_storage": (
                "sparse"
                if system.is_sparse
                else "dense"
            ),
            "residual_norm": "infinity",
            "scipy_convergence_norm": "euclidean",
        }

        return LinearSolveResult(
            solution=normalised_solution,
            converged=converged,
            iterations=iteration_count,
            residual_history=np.asarray(
                residual_history,
                dtype=np.float64,
            ),
            solver_name=self.name,
            backend_name=self.backend_name,
            termination_reason=termination_reason,
            metadata=metadata,
        )

    def _create_initial_guess(
        self,
        system: LinearSystem,
    ) -> NDArray[np.float64] | None:
        """Validate and copy the configured initial guess."""

        if self.initial_guess is None:
            return None

        initial_guess = np.asarray(
            self.initial_guess,
            dtype=np.float64,
        )

        if initial_guess.ndim != 1:
            raise ValueError(
                "Conjugate-gradient initial guess must be "
                "one-dimensional."
            )

        expected_shape = (
            system.number_of_equations,
        )

        if initial_guess.shape != expected_shape:
            raise ValueError(
                "Conjugate-gradient initial-guess shape must "
                "match the number of equations. "
                f"Expected {expected_shape}, received "
                f"{initial_guess.shape}."
            )

        if not np.all(
            np.isfinite(initial_guess)
        ):
            raise ValueError(
                "Conjugate-gradient initial guess must not "
                "contain NaN or infinite values."
            )

        return initial_guess.copy()