from __future__ import annotations

import numpy as np
import pytest

from deviceforge.core.boundary import (
    BoundaryCondition,
    BoundaryConditionType,
)
from deviceforge.core import Field
from deviceforge.core.simulation import Simulation
from deviceforge.runtime import SimulationRuntime
from deviceforge.solvers import PoissonSolver



def create_runtime_simulation(
    simulation: Simulation,
) -> Simulation:
    """Create a 0 V to 1 V one-dimensional test problem."""

    grid = simulation.grid

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1] = True

    left_boundary = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=1.0,
        units="V",
    )

    return Simulation(
        device=simulation.device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-8,
        max_iterations=500,
        initial_potential=0.0,
        name="poisson_runtime_integration",
    )


def test_runtime_executes_poisson_solver(
    simulation,
) -> None:
    poisson_simulation = create_runtime_simulation(
        simulation
    )

    solver = PoissonSolver()

    runtime = SimulationRuntime(
        simulation=poisson_simulation,
        solver=solver,
    )

    result = runtime.solve()

    assert result.converged
    assert runtime.has_run
    assert runtime.has_result
    assert runtime.has_solution

    assert runtime.result is result
    assert runtime.current_result() is result
    assert runtime.converged is True
    assert runtime.iterations == 1


def test_runtime_poisson_solution_is_linear(
    simulation,
) -> None:
    poisson_simulation = create_runtime_simulation(
        simulation
    )

    runtime = SimulationRuntime(
        simulation=poisson_simulation,
        solver=PoissonSolver(),
    )

    result = runtime.solve()

    expected = np.linspace(
        0.0,
        1.0,
        poisson_simulation.grid.shape[0],
        dtype=np.float64,
    )

    assert result.potential.values == pytest.approx(
        expected,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_runtime_reset_preserves_poisson_solver(
    simulation,
) -> None:
    poisson_simulation = create_runtime_simulation(
        simulation
    )

    solver = PoissonSolver()

    runtime = SimulationRuntime(
        simulation=poisson_simulation,
        solver=solver,
    )

    runtime.solve()
    runtime.reset()

    assert runtime.solver is solver
    assert runtime.has_solver

    assert not runtime.has_run
    assert not runtime.has_result
    assert not runtime.has_solution

    assert runtime.result is None


def test_runtime_can_repeat_poisson_solve(
    simulation,
) -> None:
    poisson_simulation = create_runtime_simulation(
        simulation
    )

    runtime = SimulationRuntime(
        simulation=poisson_simulation,
        solver=PoissonSolver(),
    )

    first_result = runtime.solve()
    second_result = runtime.solve()

    assert first_result.converged
    assert second_result.converged

    assert runtime.result is second_result
    assert runtime.state.previous_solution is (
        first_result.fields
    )

    assert second_result.potential.values == pytest.approx(
        first_result.potential.values
    )

# add additional charged runtime integration test

def test_runtime_executes_charged_poisson_problem(
    simulation,
) -> None:
    grid = simulation.grid

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1] = True

    left_boundary = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=0.0,
        units="V",
    )

    charge_density = Field.full(
        name="charge_density",
        units="C/m^3",
        grid=grid,
        fill_value=1.0e5,
    )

    charged_simulation = Simulation(
        device=simulation.device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        charge_density=charge_density,
        tolerance=1.0e-8,
        max_iterations=500,
        name="charged_runtime_integration",
    )

    runtime = SimulationRuntime(
        simulation=charged_simulation,
        solver=PoissonSolver(),
    )

    result = runtime.solve()

    assert result.converged
    assert runtime.has_result
    assert runtime.result is result

    assert result.metadata["equation"] == "poisson"
    assert np.max(result.potential.values) > 0.0