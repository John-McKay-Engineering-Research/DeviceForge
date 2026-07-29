from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from ..linear_system import LinearSystem


@dataclass(frozen=True, slots=True)
class IdentityPreconditioner:
    """
    Identity preconditioner.

    The identity preconditioner applies

        M^-1 r = r

    and therefore leaves the original linear system unchanged.

    It provides a common preconditioner interface for unpreconditioned
    iterative solves and acts as a useful reference implementation.
    """

    name: str = "identity"
    backend_name: str = "scipy"

    def __post_init__(self) -> None:
        """Validate and normalise configuration."""

        normalised_name = self._normalise_text(
            self.name,
            "Identity-preconditioner name",
        )

        normalised_backend_name = self._normalise_text(
            self.backend_name,
            "Identity-preconditioner backend name",
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
        Build the identity preconditioning operator.

        Parameters
        ----------
        system:
            Validated DeviceForge linear system.
        """

        if not isinstance(system, LinearSystem):
            raise TypeError(
                "IdentityPreconditioner requires a "
                "LinearSystem instance."
            )

        number_of_equations = (
            system.number_of_equations
        )

        def apply_identity(
            vector: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            """Return an independent copy of the input vector."""

            values = np.asarray(
                vector,
                dtype=np.float64,
            )

            return values.copy()

        return LinearOperator(
            shape=(
                number_of_equations,
                number_of_equations,
            ),
            matvec=apply_identity,
            rmatvec=apply_identity,
            dtype=np.float64,
        )