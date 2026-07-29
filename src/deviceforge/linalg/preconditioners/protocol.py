from __future__ import annotations

from typing import Protocol, runtime_checkable

from scipy.sparse.linalg import LinearOperator

from ..linear_system import LinearSystem


@runtime_checkable
class PreconditionerProtocol(Protocol):
    """
    Structural interface for DeviceForge preconditioners.

    A compatible preconditioner exposes identifying information and builds
    an operator approximating the inverse of a LinearSystem matrix.

    Concrete implementations do not need to inherit from this protocol.
    """

    @property
    def name(self) -> str:
        """Return the preconditioner name."""

        ...

    @property
    def backend_name(self) -> str:
        """Return the numerical backend name."""

        ...

    def build(
        self,
        system: LinearSystem,
    ) -> LinearOperator:
        """
        Build a preconditioning operator for a linear system.

        Parameters
        ----------
        system:
            Validated DeviceForge linear system.

        Returns
        -------
        LinearOperator
            Operator that applies the approximate inverse preconditioner.
        """

        ...