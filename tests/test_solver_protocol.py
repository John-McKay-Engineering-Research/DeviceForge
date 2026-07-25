from __future__ import annotations

from typing import get_type_hints

from deviceforge import Simulation, SimulationResult
from deviceforge.solvers import SolverProtocol


class ConformingSolver:
    """Minimal solver satisfying SolverProtocol."""

    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        raise NotImplementedError


class MissingSolveMethod:
    """Object that does not satisfy SolverProtocol."""

    pass


class CallableWithDifferentMethod:
    """Object with solver-like behaviour but no solve method."""

    def run(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        raise NotImplementedError


def test_conforming_solver_satisfies_protocol() -> None:
    solver = ConformingSolver()

    assert isinstance(solver, SolverProtocol)


def test_object_without_solve_does_not_satisfy_protocol() -> None:
    solver = MissingSolveMethod()

    assert not isinstance(solver, SolverProtocol)


def test_differently_named_method_does_not_satisfy_protocol() -> None:
    solver = CallableWithDifferentMethod()

    assert not isinstance(solver, SolverProtocol)


def test_protocol_solve_type_annotations() -> None:
    hints = get_type_hints(SolverProtocol.solve)

    assert hints["simulation"] is Simulation
    assert hints["return"] is SimulationResult