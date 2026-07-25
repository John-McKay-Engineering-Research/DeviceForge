from __future__ import annotations

from dataclasses import dataclass, field

from deviceforge.core.result import SimulationResult
from deviceforge.core.simulation import Simulation
from deviceforge.solvers.protocol import SolverProtocol

from .runtime_state import RuntimeState


@dataclass(slots=True)
class SimulationRuntime:
    """
    Mutable execution environment for a DeviceForge simulation.

    SimulationRuntime coordinates an immutable Simulation definition with a
    numerical solver and its mutable execution state. It contains no numerical
    solution mathematics.

    Parameters
    ----------
    simulation:
        Immutable simulation definition describing the numerical problem.

    solver:
        Optional numerical solver implementing SolverProtocol.

    state:
        Mutable runtime state. A new empty RuntimeState is created by default.

    name:
        Optional human-readable name for the runtime.
    """

    simulation: Simulation
    solver: SolverProtocol | None = None
    state: RuntimeState = field(default_factory=RuntimeState)
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalise the runtime."""

        if not isinstance(self.simulation, Simulation):
            raise TypeError(
                "SimulationRuntime requires a Simulation instance."
            )

        if not isinstance(self.state, RuntimeState):
            raise TypeError(
                "SimulationRuntime state must be a RuntimeState instance."
            )

        if (
            self.solver is not None
            and not isinstance(self.solver, SolverProtocol)
        ):
            raise TypeError(
                "SimulationRuntime solver must implement SolverProtocol."
            )

        if self.name is None:
            self.name = f"{self.simulation.name}_runtime"
        else:
            if not isinstance(self.name, str):
                raise TypeError(
                    "Runtime name must be a string or None."
                )

            self.name = self.name.strip()

            if not self.name:
                raise ValueError(
                    "Runtime name must not be empty."
                )

    @property
    def device(self):
        """Return the simulation device."""

        return self.simulation.device

    @property
    def grid(self):
        """Return the simulation grid."""

        return self.simulation.grid

    @property
    def has_solver(self) -> bool:
        """Return whether a solver has been assigned."""

        return self.solver is not None

    @property
    def is_solver_configured(self) -> bool:
        """
        Return whether a solver has been assigned.

        This property is retained as a compatibility alias for has_solver.
        """

        return self.has_solver

    @property
    def has_solution(self) -> bool:
        """Return whether the runtime currently holds solution fields."""

        return self.state.has_solution

    @property
    def has_result(self) -> bool:
        """Return whether the runtime holds a completed result."""

        return self.state.has_result

    @property
    def has_run(self) -> bool:
        """Return whether a completed solve has been recorded."""

        return self.state.has_run

    @property
    def result(self) -> SimulationResult | None:
        """Return the most recent completed simulation result."""

        return self.state.last_result

    @property
    def converged(self) -> bool | None:
        """
        Return the convergence status of the most recent result.

        None is returned when no completed result is available.
        """

        if self.result is None:
            return None

        return self.result.converged

    @property
    def iterations(self) -> int | None:
        """
        Return the iteration count of the most recent result.

        None is returned when no completed result is available.
        """

        if self.result is None:
            return None

        return self.result.iterations

    def set_solver(
        self,
        solver: SolverProtocol,
    ) -> None:
        """Assign a numerical solver to the runtime."""

        if not isinstance(solver, SolverProtocol):
            raise TypeError(
                "solver must implement SolverProtocol."
            )

        self.solver = solver

    def clear_solver(self) -> None:
        """Remove the currently assigned solver."""

        self.solver = None

    def solve(self) -> SimulationResult:
        """
        Execute the configured numerical solver.

        The solver receives only the immutable Simulation definition. The
        returned SimulationResult is validated and recorded in RuntimeState.

        Returns
        -------
        SimulationResult
            The result returned by the configured solver.

        Raises
        ------
        RuntimeError
            If no solver is configured.

        TypeError
            If the solver returns an object other than SimulationResult.
        """

        if self.solver is None:
            raise RuntimeError(
                "Cannot solve because no solver is configured."
            )

        result = self.solver.solve(
            self.simulation
        )

        if not isinstance(result, SimulationResult):
            raise TypeError(
                "Solver.solve() must return a SimulationResult instance."
            )

        self.state.record_result(result)

        return result

    def current_result(self) -> SimulationResult | None:
        """Return the most recent completed result, if available."""

        return self.state.last_result

    def reset(self) -> None:
        """
        Reset mutable execution state.

        The immutable Simulation definition and configured solver are
        preserved. Use clear_solver() when the solver should also be removed.
        """

        self.state.reset()

    def snapshot_state(self) -> RuntimeState:
        """Return an independent snapshot of the current runtime state."""

        return self.state.copy()

    def __repr__(self) -> str:
        """Return a concise representation of the runtime."""

        solver_name = (
            type(self.solver).__name__
            if self.solver is not None
            else "None"
        )

        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"simulation={self.simulation.name!r}, "
            f"solver={solver_name}, "
            f"has_run={self.has_run}, "
            f"has_result={self.has_result}, "
            f"has_solution={self.has_solution}"
            f")"
        )