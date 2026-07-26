from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..core.field import Field
from ..core.result import SimulationResult
from ..core.simulation import Simulation
from ..postprocessing.analysis import (
    ElectrostaticAnalysis,
    analyse_electrostatics,
)
from ..runtime.runtime_state import RuntimeState
from ..runtime.simulation_runtime import SimulationRuntime
from ..solvers.protocol import SolverProtocol


@dataclass(frozen=True, slots=True)
class ElectrostaticWorkflowResult:
    """
    Complete output returned by an electrostatic workflow.

    The workflow result combines the raw numerical solver result with the
    derived electrostatic analysis fields.

    Parameters
    ----------
    simulation_result:
        Immutable raw output returned by the numerical solver.

    analysis:
        Immutable collection of solved and derived electrostatic fields.
    """

    simulation_result: SimulationResult
    analysis: ElectrostaticAnalysis

    def __post_init__(self) -> None:
        """Validate the workflow output."""

        if not isinstance(
            self.simulation_result,
            SimulationResult,
        ):
            raise TypeError(
                "simulation_result must be a SimulationResult instance."
            )

        if not isinstance(
            self.analysis,
            ElectrostaticAnalysis,
        ):
            raise TypeError(
                "analysis must be an ElectrostaticAnalysis instance."
            )

        if self.simulation_result.grid != self.analysis.grid:
            raise ValueError(
                "Simulation result and electrostatic analysis must "
                "use the same grid."
            )

        if (
            self.simulation_result.potential
            is not self.analysis.potential
        ):
            raise ValueError(
                "Electrostatic analysis must contain the potential "
                "from the simulation result."
            )

    @property
    def potential(self) -> Field:
        """Return the solved electrostatic potential."""

        return self.analysis.potential

    @property
    def electric_field(self) -> Field:
        """Return the derived electric field."""

        return self.analysis.electric_field

    @property
    def electric_displacement(self) -> Field:
        """Return the derived electric displacement."""

        return self.analysis.electric_displacement

    @property
    def energy_density(self) -> Field:
        """Return the derived electrostatic energy density."""

        return self.analysis.energy_density

    @property
    def converged(self) -> bool:
        """Return whether the numerical solver converged."""

        return self.simulation_result.converged

    @property
    def iterations(self) -> int:
        """Return the number of completed solver iterations."""

        return self.simulation_result.iterations

    @property
    def residual_history(self):
        """Return the solver residual history."""

        return self.simulation_result.residual_history

    @property
    def final_residual(self) -> float | None:
        """Return the final solver residual."""

        return self.simulation_result.final_residual

    @property
    def runtime_seconds(self) -> float:
        """Return the numerical solver runtime."""

        return self.simulation_result.runtime_seconds

    @property
    def solver_name(self) -> str:
        """Return the numerical solver name."""

        return self.simulation_result.solver_name

    @property
    def backend_name(self) -> str:
        """Return the compute backend name."""

        return self.simulation_result.backend_name

    @property
    def fields(self) -> Mapping[str, Field]:
        """Return all workflow fields as a read-only mapping."""

        return MappingProxyType(
            self.analysis.as_dict()
        )

    def get_field(
        self,
        name: str,
    ) -> Field:
        """Return a workflow field by name."""

        return self.analysis.get_field(name)


class ElectrostaticWorkflow:
    """
    High-level orchestration of an electrostatic simulation.

    The workflow performs the standard DeviceForge electrostatic pipeline:

        Simulation
            -> SimulationRuntime
            -> numerical solver
            -> SimulationResult
            -> electrostatic postprocessing
            -> ElectrostaticWorkflowResult

    The workflow contains no numerical solution algorithm. It delegates
    execution to SimulationRuntime and physical postprocessing to
    analyse_electrostatics().

    Parameters
    ----------
    simulation:
        Immutable electrostatic simulation definition.

    solver:
        Numerical solver satisfying SolverProtocol.

    name:
        Optional human-readable workflow name.
    """

    __slots__ = (
        "_runtime",
        "_last_output",
        "name",
    )

    def __init__(
        self,
        *,
        simulation: Simulation,
        solver: SolverProtocol,
        name: str | None = None,
    ) -> None:
        """Create an electrostatic workflow."""

        if not isinstance(simulation, Simulation):
            raise TypeError(
                "ElectrostaticWorkflow requires a Simulation instance."
            )

        if not isinstance(solver, SolverProtocol):
            raise TypeError(
                "ElectrostaticWorkflow solver must satisfy SolverProtocol."
            )

        resolved_name = self._resolve_name(
            name=name,
            default=f"{simulation.name}_workflow",
        )

        self._runtime = SimulationRuntime(
            simulation=simulation,
            solver=solver,
            name=f"{resolved_name}_runtime",
        )

        self._last_output: (
            ElectrostaticWorkflowResult | None
        ) = None

        self.name = resolved_name

    @classmethod
    def from_runtime(
        cls,
        runtime: SimulationRuntime,
        *,
        name: str | None = None,
    ) -> ElectrostaticWorkflow:
        """
        Create a workflow around an existing runtime.

        This entry point is intended for advanced users who have already
        configured a SimulationRuntime manually.
        """

        if not isinstance(runtime, SimulationRuntime):
            raise TypeError(
                "ElectrostaticWorkflow.from_runtime requires a "
                "SimulationRuntime instance."
            )

        if not runtime.has_solver:
            raise ValueError(
                "Electrostatic workflow runtime must have a configured solver."
            )

        resolved_name = cls._resolve_name(
            name=name,
            default=f"{runtime.simulation.name}_workflow",
        )

        workflow = cls.__new__(cls)

        workflow._runtime = runtime
        workflow._last_output = None
        workflow.name = resolved_name

        return workflow

    @staticmethod
    def _resolve_name(
        *,
        name: str | None,
        default: str,
    ) -> str:
        """Validate and normalise a workflow name."""

        if name is None:
            return default

        if not isinstance(name, str):
            raise TypeError(
                "Workflow name must be a string or None."
            )

        resolved_name = name.strip()

        if not resolved_name:
            raise ValueError(
                "Workflow name must not be empty."
            )

        return resolved_name

    @property
    def runtime(self) -> SimulationRuntime:
        """Return the underlying simulation runtime."""

        return self._runtime

    @property
    def simulation(self) -> Simulation:
        """Return the immutable simulation definition."""

        return self._runtime.simulation

    @property
    def solver(self) -> SolverProtocol | None:
        """Return the configured numerical solver."""

        return self._runtime.solver

    @property
    def state(self) -> RuntimeState:
        """Return the mutable runtime execution state."""

        return self._runtime.state

    @property
    def simulation_result(
        self,
    ) -> SimulationResult | None:
        """Return the most recent raw simulation result."""

        return self._runtime.result

    @property
    def output(
        self,
    ) -> ElectrostaticWorkflowResult | None:
        """Return the most recent complete workflow output."""

        return self._last_output

    @property
    def has_run(self) -> bool:
        """Return whether the runtime has recorded a solve."""

        return self._runtime.has_run

    @property
    def has_result(self) -> bool:
        """Return whether a raw simulation result exists."""

        return self._runtime.has_result

    @property
    def has_output(self) -> bool:
        """Return whether a complete workflow output exists."""

        return self._last_output is not None

    @property
    def converged(self) -> bool:
        """Return whether the most recent solve converged."""

        return self._runtime.converged

    @property
    def iterations(self) -> int:
        """Return the most recent solver iteration count."""

        return self._runtime.iterations

    def run(self) -> ElectrostaticWorkflowResult:
        """
        Execute the complete electrostatic workflow.

        The runtime performs the numerical solve and records execution state.
        The resulting potential is then passed through the electrostatic
        analysis pipeline.
        """

        self._last_output = None

        simulation_result = self._runtime.solve()

        analysis = analyse_electrostatics(
            self.simulation,
            simulation_result,
        )

        output = ElectrostaticWorkflowResult(
            simulation_result=simulation_result,
            analysis=analysis,
        )

        self._last_output = output

        return output

    def reset(self) -> None:
        """
        Reset runtime state and remove the cached workflow output.

        The configured simulation and solver are preserved.
        """

        self._runtime.reset()
        self._last_output = None

    def set_solver(
        self,
        solver: SolverProtocol,
    ) -> None:
        """
        Replace the numerical solver and clear previous execution state.
        """

        if not isinstance(solver, SolverProtocol):
            raise TypeError(
                "Workflow solver must satisfy SolverProtocol."
            )

        self._runtime.reset()
        self._runtime.set_solver(solver)
        self._last_output = None

    def snapshot_state(self) -> RuntimeState:
        """Return an independent snapshot of runtime state."""

        return self._runtime.snapshot_state()

    def __repr__(self) -> str:
        """Return a concise workflow representation."""

        solver_name = (
            type(self.solver).__name__
            if self.solver is not None
            else "None"
        )

        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"simulation={self.simulation.name!r}, "
            f"solver={solver_name!r}, "
            f"has_output={self.has_output}, "
            f"converged={self.converged}"
            ")"
        )