from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from ..linear_system import LinearSystem


@dataclass(frozen=True, slots=True)
class JacobiPreconditioner:
    """
    Jacobi diagonal preconditioner.

    For the linear system

        A x = b

    the Jacobi preconditioner uses

        M = diag(A)

    and applies

        M^-1 r

    through elementwise division:

        z_i = r_i / A_ii.

    No dense inverse matrix is constructed.
    """

    diagonal_tolerance: float = 1.0e-15
    name: str = "jacobi"
    backend_name: str = "scipy"

    def __post_init__(self) -> None:
        """Validate and normalise configuration."""

        if isinstance(
            self.diagonal_tolerance,
            bool,
        ) or not isinstance(
            self.diagonal_tolerance,
            (int, float),
        ):
            raise TypeError(
                "Jacobi diagonal tolerance must be "
                "a real number."
            )

        diagonal_tolerance = float(
            self.diagonal_tolerance
        )

        if (
            not np.isfinite(diagonal_tolerance)
            or diagonal_tolerance < 0.0
        ):
            raise ValueError(
                "Jacobi diagonal tolerance must be "
                "finite and non-negative."
            )

        normalised_name = self._normalise_text(
            self.name,
            "Jacobi-preconditioner name",
        )

        normalised_backend_name = self._normalise_text(
            self.backend_name,
            "Jacobi-preconditioner backend name",
        )

        object.__setattr__(
            self,
            "diagonal_tolerance",
            diagonal_tolerance,
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

    @staticmethod
    def _normalise_text(
        value: str,
        label: str,
    ) -> str:
        """Validate and normalise required text."""

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

    def build(
        self,
        system: LinearSystem,
    ) -> LinearOperator:
        """
        Build the inverse-diagonal Jacobi operator.

        Parameters
        ----------
        system:
            Validated DeviceForge linear system.

        Returns
        -------
        LinearOperator
            Operator applying inverse-diagonal scaling.

        Raises
        ------
        TypeError
            If ``system`` is not a LinearSystem.

        ValueError
            If a diagonal value is zero or numerically too small.
        """

        if not isinstance(system, LinearSystem):
            raise TypeError(
                "JacobiPreconditioner requires a "
                "LinearSystem instance."
            )

        diagonal = np.asarray(
            system.matrix.diagonal(),
            dtype=np.float64,
        )

        expected_shape = (
            system.number_of_equations,
        )

        if diagonal.shape != expected_shape:
            raise ValueError(
                "Linear-system diagonal has an unexpected "
                f"shape. Expected {expected_shape}, "
                f"received {diagonal.shape}."
            )

        if not np.all(
            np.isfinite(diagonal)
        ):
            raise ValueError(
                "Linear-system diagonal must not contain "
                "NaN or infinite values."
            )

        if np.any(
            np.abs(diagonal)
            <= self.diagonal_tolerance
        ):
            raise ValueError(
                "Jacobi preconditioning requires every "
                "diagonal value to be greater than the "
                "configured diagonal tolerance."
            )

        inverse_diagonal = (
            1.0 / diagonal
        )

        inverse_diagonal.setflags(
            write=False
        )

        def apply_inverse_diagonal(
            vector: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """Apply inverse diagonal scaling."""

            values = np.asarray(
                vector,
                dtype=np.float64,
            )

            return (
                inverse_diagonal
                * values
            )

        return LinearOperator(
            shape=system.shape,
            matvec=apply_inverse_diagonal,
            rmatvec=apply_inverse_diagonal,
            dtype=np.float64,
        )