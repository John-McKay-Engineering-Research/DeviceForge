from __future__ import annotations

import pytest

from deviceforge.postprocessing import ElectrostaticAnalysis
from deviceforge.runtime import SimulationRuntime
from deviceforge.solvers import PoissonSolver
from deviceforge.workflows import (
    ElectrostaticWorkflow,
    ElectrostaticWorkflowResult,
)

from deviceforge.linalg import (
    ConjugateGradientSolver,
)


def test_workflow_stores_simulation_and_solver(
    simulation,
) -> None:
    solver = PoissonSolver()

    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=solver,
    )

    assert workflow.simulation is simulation
    assert workflow.solver is solver

    assert isinstance(
        workflow.runtime,
        SimulationRuntime,
    )


def test_workflow_creates_default_name(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    assert workflow.name == (
        f"{simulation.name}_workflow"
    )


def test_workflow_normalises_custom_name(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
        name="  electrostatic study  ",
    )

    assert workflow.name == "electrostatic study"


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_workflow_rejects_empty_name(
    simulation,
    name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        ElectrostaticWorkflow(
            simulation=simulation,
            solver=PoissonSolver(),
            name=name,
        )


def test_workflow_rejects_invalid_simulation() -> None:
    with pytest.raises(
        TypeError,
        match="Simulation instance",
    ):
        ElectrostaticWorkflow(
            simulation="invalid",
            solver=PoissonSolver(),
        )


def test_workflow_rejects_invalid_solver(
    simulation,
) -> None:
    with pytest.raises(
        TypeError,
        match="SolverProtocol",
    ):
        ElectrostaticWorkflow(
            simulation=simulation,
            solver=object(),
        )


def test_workflow_starts_without_output(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    assert not workflow.has_run
    assert not workflow.has_result
    assert not workflow.has_output

    assert workflow.simulation_result is None
    assert workflow.output is None


def test_workflow_run_returns_complete_output(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    output = workflow.run()

    assert isinstance(
        output,
        ElectrostaticWorkflowResult,
    )

    assert isinstance(
        output.analysis,
        ElectrostaticAnalysis,
    )

    assert output.simulation_result is (
        workflow.simulation_result
    )

    assert workflow.output is output
    assert workflow.has_output


def test_workflow_records_runtime_execution(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    output = workflow.run()

    assert workflow.has_run
    assert workflow.has_result
    assert workflow.has_output

    assert workflow.converged
    assert workflow.iterations == (
        output.iterations
    )

    assert workflow.state.last_result is (
        output.simulation_result
    )


def test_workflow_result_exposes_analysis_fields(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    output = workflow.run()

    assert output.potential is (
        output.analysis.potential
    )

    assert output.electric_field is (
        output.analysis.electric_field
    )

    assert output.electric_displacement is (
        output.analysis.electric_displacement
    )

    assert output.energy_density is (
        output.analysis.energy_density
    )


def test_workflow_result_exposes_solver_diagnostics(
    simulation,
) -> None:
    output = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    ).run()

    assert output.converged
    assert output.iterations == 1
    assert output.final_residual is not None
    assert output.runtime_seconds >= 0.0

    assert output.solver_name == (
        output.simulation_result.solver_name
    )

    assert output.backend_name == (
        output.simulation_result.backend_name
    )


def test_workflow_result_get_field(
    simulation,
) -> None:
    output = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    ).run()

    selected = output.get_field(
        "electric_field"
    )

    assert selected is output.electric_field


def test_workflow_result_fields_are_read_only(
    simulation,
) -> None:
    output = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    ).run()

    with pytest.raises(TypeError):
        output.fields["new_field"] = (
            output.potential
        )


def test_workflow_reset_clears_execution_state(
    simulation,
) -> None:
    solver = PoissonSolver()

    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=solver,
    )

    workflow.run()
    workflow.reset()

    assert workflow.solver is solver

    assert not workflow.has_run
    assert not workflow.has_result
    assert not workflow.has_output

    assert workflow.simulation_result is None
    assert workflow.output is None


def test_workflow_can_run_again_after_reset(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    first_output = workflow.run()

    workflow.reset()

    second_output = workflow.run()

    assert first_output is not second_output
    assert second_output.converged
    assert workflow.output is second_output


def test_repeated_run_retains_previous_solution(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    first_output = workflow.run()
    second_output = workflow.run()

    assert workflow.output is second_output

    assert workflow.state.previous_solution is (
        first_output.simulation_result.fields
    )

    assert second_output.potential.values == pytest.approx(
        first_output.potential.values
    )


def test_workflow_from_runtime_uses_existing_runtime(
    simulation,
) -> None:
    runtime = SimulationRuntime(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    workflow = ElectrostaticWorkflow.from_runtime(
        runtime
    )

    assert workflow.runtime is runtime
    assert workflow.simulation is simulation
    assert workflow.solver is runtime.solver


def test_workflow_from_runtime_rejects_invalid_runtime() -> None:
    with pytest.raises(
        TypeError,
        match="SimulationRuntime",
    ):
        ElectrostaticWorkflow.from_runtime(
            "invalid"
        )


def test_workflow_from_runtime_requires_solver(
    simulation,
) -> None:
    runtime = SimulationRuntime(
        simulation=simulation,
    )

    with pytest.raises(
        ValueError,
        match="configured solver",
    ):
        ElectrostaticWorkflow.from_runtime(
            runtime
        )


def test_set_solver_clears_previous_output(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    workflow.run()

    replacement_solver = PoissonSolver(
        name="replacement_poisson_solver"
    )

    workflow.set_solver(
        replacement_solver
    )

    assert workflow.solver is replacement_solver

    assert not workflow.has_run
    assert not workflow.has_result
    assert not workflow.has_output


def test_snapshot_state_is_independent(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    workflow.run()

    snapshot = workflow.snapshot_state()

    assert snapshot is not workflow.state
    assert snapshot.last_result is (
        workflow.state.last_result
    )

    assert snapshot.residual_history is not (
        workflow.state.residual_history
    )


def test_workflow_repr_contains_status(
    simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
        name="test_workflow",
    )

    representation = repr(workflow)

    assert "ElectrostaticWorkflow" in representation
    assert "test_workflow" in representation
    assert "PoissonSolver" in representation
    assert "has_output=False" in representation

# Conjugate tests

def test_workflow_runs_with_conjugate_gradient(
    dielectric_stack_simulation,
) -> None:
    workflow = ElectrostaticWorkflow(
        simulation=dielectric_stack_simulation,
        solver=PoissonSolver(
            linear_solver=ConjugateGradientSolver(
                relative_tolerance=1.0e-12,
                absolute_tolerance=1.0e-14,
                max_iterations=10_000,
            ),
            name="poisson_cg_1d",
        ),
    )

    output = workflow.run()

    assert output.converged
    assert output.iterations > 0
    assert output.residual_history.size == (
        output.iterations
    )

    assert output.simulation_result.metadata[
        "linear_solver"
    ] == "conjugate_gradient"