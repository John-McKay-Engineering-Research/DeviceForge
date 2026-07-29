from __future__ import annotations

import numpy as np
import pytest

from deviceforge.core.field import Field
from deviceforge.core.simulation import Simulation
from deviceforge.linalg import (
    DenseDirectSolver,
    SparseDirectSolver,
)
from deviceforge.postprocessing import (
    calculate_face_electrostatic_fields,
)
from deviceforge.solvers import PoissonSolver


def solve_with_both_backends(
    simulation: Simulation,
):
    """Solve one Poisson problem with dense and sparse backends."""

    dense_result = PoissonSolver(
        linear_solver=DenseDirectSolver(),
        name="poisson_dense_reference_1d",
    ).solve(
        simulation
    )

    sparse_result = PoissonSolver(
        linear_solver=SparseDirectSolver(),
        name="poisson_sparse_1d",
    ).solve(
        simulation
    )

    return dense_result, sparse_result


def test_dense_and_sparse_uniform_problem_match(
    simulation: Simulation,
) -> None:
    dense_result, sparse_result = (
        solve_with_both_backends(
            simulation
        )
    )

    np.testing.assert_allclose(
        dense_result.potential.values,
        sparse_result.potential.values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_dense_and_sparse_dielectric_stack_match(
    dielectric_stack_simulation: Simulation,
) -> None:
    dense_result, sparse_result = (
        solve_with_both_backends(
            dielectric_stack_simulation
        )
    )

    np.testing.assert_allclose(
        dense_result.potential.values,
        sparse_result.potential.values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_dense_and_sparse_charged_problem_match(
    simulation: Simulation,
) -> None:
    charge_density = Field.full(
        name="charge_density",
        units="C/m^3",
        grid=simulation.grid,
        fill_value=1.0e5,
    )

    charged_simulation = Simulation(
        name="dense_sparse_charged_parity",
        device=simulation.device,
        boundary_conditions=(
            simulation.boundary_conditions
        ),
        charge_density=charge_density,
        tolerance=simulation.tolerance,
        max_iterations=simulation.max_iterations,
        initial_potential=(
            simulation.initial_potential
        ),
    )

    dense_result, sparse_result = (
        solve_with_both_backends(
            charged_simulation
        )
    )

    np.testing.assert_allclose(
        dense_result.potential.values,
        sparse_result.potential.values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_dense_and_sparse_results_record_backends(
    dielectric_stack_simulation: Simulation,
) -> None:
    dense_result, sparse_result = (
        solve_with_both_backends(
            dielectric_stack_simulation
        )
    )

    assert dense_result.backend_name == "numpy"
    assert sparse_result.backend_name == "scipy"

    assert dense_result.metadata[
        "linear_solver"
    ] == "dense_direct"

    assert sparse_result.metadata[
        "linear_solver"
    ] == "sparse_direct"


    assert dense_result.metadata[
        "linear_solver_converged"
    ] is True
    assert sparse_result.metadata[
        "linear_solver_converged"
    ] is True

    assert dense_result.metadata[
        "linear_solver_iterations"
    ] == 1
    assert sparse_result.metadata[
        "linear_solver_iterations"
    ] == 1

    assert dense_result.metadata[
        "linear_solver_termination_reason"
    ] == "direct_solve_completed"
    assert sparse_result.metadata[
        "linear_solver_termination_reason"
    ] == "direct_solve_completed"


def test_dense_and_sparse_residuals_are_small(
    dielectric_stack_simulation: Simulation,
) -> None:
    dense_result, sparse_result = (
        solve_with_both_backends(
            dielectric_stack_simulation
        )
    )

    assert dense_result.final_residual is not None
    assert sparse_result.final_residual is not None

    assert dense_result.final_residual <= (
        dielectric_stack_simulation.tolerance
    )

    assert sparse_result.final_residual <= (
        dielectric_stack_simulation.tolerance
    )


@pytest.mark.parametrize(
    "linear_solver",
    [
        DenseDirectSolver(),
        SparseDirectSolver(),
    ],
)
def test_dielectric_flux_is_continuous_for_each_backend(
    dielectric_stack_simulation: Simulation,
    linear_solver,
) -> None:
    result = PoissonSolver(
        linear_solver=linear_solver,
    ).solve(
        dielectric_stack_simulation
    )

    relative_permittivity = (
        dielectric_stack_simulation
        .device
        .relative_permittivity_field()
    )

    (
        _,
        _,
        face_displacement,
    ) = calculate_face_electrostatic_fields(
        result.potential,
        relative_permittivity,
    )

    np.testing.assert_allclose(
        face_displacement.values,
        face_displacement.values[0],
        rtol=1.0e-11,
        atol=1.0e-15,
    )