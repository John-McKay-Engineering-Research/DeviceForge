from __future__ import annotations

import numpy as np
import pytest

from deviceforge.core.field import Field
from deviceforge.core.simulation import Simulation

from deviceforge.postprocessing import (
    calculate_face_electrostatic_fields,
)
from deviceforge.solvers import PoissonSolver

from deviceforge.linalg import (
    ConjugateGradientSolver,
    JacobiPreconditioner,
    SparseDirectSolver,
)


def solve_with_direct_and_cg(
    simulation: Simulation,
):
    """Solve one Poisson problem with sparse direct and CG backends."""

    direct_result = PoissonSolver(
        linear_solver=SparseDirectSolver(),
        name="poisson_sparse_direct_1d",
    ).solve(
        simulation
    )

    cg_result = PoissonSolver(
        linear_solver=ConjugateGradientSolver(
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
            max_iterations=10_000,
        ),
        name="poisson_cg_1d",
    ).solve(
        simulation
    )

    return direct_result, cg_result


def test_cg_matches_direct_for_uniform_laplace(
    simulation: Simulation,
) -> None:
    direct_result, cg_result = (
        solve_with_direct_and_cg(
            simulation
        )
    )

    np.testing.assert_allclose(
        cg_result.potential.values,
        direct_result.potential.values,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    assert cg_result.converged


def test_cg_matches_direct_for_dielectric_stack(
    dielectric_stack_simulation: Simulation,
) -> None:
    direct_result, cg_result = (
        solve_with_direct_and_cg(
            dielectric_stack_simulation
        )
    )

    np.testing.assert_allclose(
        cg_result.potential.values,
        direct_result.potential.values,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    assert cg_result.converged


def test_cg_matches_direct_for_uniform_charge(
    simulation: Simulation,
) -> None:
    charge_density = Field.full(
        name="charge_density",
        units="C/m^3",
        grid=simulation.grid,
        fill_value=1.0e5,
    )

    charged_simulation = Simulation(
        name="cg_uniform_charge_parity",
        device=simulation.device,
        boundary_conditions=(
            simulation.boundary_conditions
        ),
        charge_density=charge_density,
        tolerance=1.0e-8,
        max_iterations=10_000,
        initial_potential=0.0,
    )

    direct_result, cg_result = (
        solve_with_direct_and_cg(
            charged_simulation
        )
    )

    np.testing.assert_allclose(
        cg_result.potential.values,
        direct_result.potential.values,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    assert cg_result.converged


def test_cg_records_iterative_diagnostics(
    dielectric_stack_simulation: Simulation,
) -> None:
    _, cg_result = solve_with_direct_and_cg(
        dielectric_stack_simulation
    )

    assert cg_result.backend_name == "scipy"

    assert cg_result.metadata[
        "linear_solver"
    ] == "conjugate_gradient"

    assert cg_result.metadata[
        "linear_solver_backend"
    ] == "scipy"

    assert cg_result.metadata[
        "linear_solver_converged"
    ] is True

    assert cg_result.metadata[
        "linear_solver_iterations"
    ] == cg_result.iterations

    assert cg_result.metadata[
        "linear_solver_termination_reason"
    ] == "convergence_tolerance_satisfied"

    assert cg_result.iterations > 0

    assert cg_result.residual_history.size == (
        cg_result.iterations
    )

    assert cg_result.final_residual is not None


def test_cg_residual_meets_simulation_tolerance(
    dielectric_stack_simulation: Simulation,
) -> None:
    _, cg_result = solve_with_direct_and_cg(
        dielectric_stack_simulation
    )

    assert cg_result.final_residual is not None

    assert cg_result.final_residual <= (
        dielectric_stack_simulation.tolerance
    )


def test_cg_preserves_dielectric_flux(
    dielectric_stack_simulation: Simulation,
) -> None:
    _, cg_result = solve_with_direct_and_cg(
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
        cg_result.potential,
        relative_permittivity,
    )

    np.testing.assert_allclose(
        face_displacement.values,
        face_displacement.values[0],
        rtol=1.0e-9,
        atol=1.0e-13,
    )


def test_cg_and_direct_endpoint_values_match(
    dielectric_stack_simulation: Simulation,
) -> None:
    direct_result, cg_result = (
        solve_with_direct_and_cg(
            dielectric_stack_simulation
        )
    )

    assert cg_result.potential.values[0] == pytest.approx(
        direct_result.potential.values[0],
        abs=1.0e-12,
    )

    assert cg_result.potential.values[-1] == pytest.approx(
        direct_result.potential.values[-1],
        abs=1.0e-12,
    )

# Poisson-level Jacobi Integration

def test_jacobi_cg_matches_direct_for_dielectric_stack(
    dielectric_stack_simulation: Simulation,
) -> None:
    direct_result = PoissonSolver(
        linear_solver=SparseDirectSolver(),
    ).solve(
        dielectric_stack_simulation
    )

    jacobi_result = PoissonSolver(
        linear_solver=ConjugateGradientSolver(
            preconditioner=JacobiPreconditioner(),
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
            max_iterations=10_000,
        ),
        name="poisson_jacobi_cg_1d",
    ).solve(
        dielectric_stack_simulation
    )

    np.testing.assert_allclose(
        jacobi_result.potential.values,
        direct_result.potential.values,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    assert jacobi_result.converged

    assert jacobi_result.metadata[
        "linear_solver_metadata"
    ][
        "preconditioner"
    ] == "jacobi"