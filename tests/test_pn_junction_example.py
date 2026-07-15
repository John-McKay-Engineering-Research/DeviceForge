import numpy as np

from examples.pn_junction_fixed_charge import (
    ACCEPTOR_DENSITY,
    DONOR_DENSITY,
    build_simulation,
    create_signed_doping_field,
)
from deviceforge.physics import ELEMENTARY_CHARGE
# pytest fix
from deviceforge.solvers import PoissonJacobiSolver, SolverConfiguration


def test_pn_junction_regions_have_opposite_doping() -> None:
    simulation = build_simulation()

    doping = create_signed_doping_field(
        simulation
    )

    junction_index = (
        simulation.grid.shape[0] // 2
    )

    np.testing.assert_allclose(
        doping.values[:junction_index, :],
        -ACCEPTOR_DENSITY,
    )

    np.testing.assert_allclose(
        doping.values[junction_index:, :],
        DONOR_DENSITY,
    )


def test_pn_junction_charge_changes_sign() -> None:
    simulation = build_simulation()
    # pytest edit below, jacobi terminates after 10,000 pn junction does not
    solver = PoissonJacobiSolver(
        SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
            backend_name="numpy",
        )
    )

    result = solver.solve(simulation)

    charge_density = result.get_field(
        "fixed_charge_density"
    )

    junction_index = (
        simulation.grid.shape[0] // 2
    )

    np.testing.assert_allclose(
        charge_density.values[
            :junction_index,
            :,
        ],
        -ELEMENTARY_CHARGE
        * ACCEPTOR_DENSITY,
    )

    np.testing.assert_allclose(
        charge_density.values[
            junction_index:,
            :,
        ],
        ELEMENTARY_CHARGE
        * DONOR_DENSITY,
    )

# edit below for pytest failure
def test_pn_junction_solver_converges() -> None:
    simulation = build_simulation()

    solver = PoissonJacobiSolver(
        SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
            backend_name="numpy",
        )
    )

    result = solver.solve(simulation)

    assert result.converged
    assert result.final_residual is not None
    assert result.final_residual <= simulation.tolerance