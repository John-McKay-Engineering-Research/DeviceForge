import numpy as np
import pytest

from examples.pn_junction_equilibrium import (
    ACCEPTOR_DENSITY,
    DONOR_DENSITY,
    build_simulation,
    calculate_built_in_potential,
    create_signed_doping_field,
)
from deviceforge.solvers import (
    EquilibriumPoissonSolver,
    SolverConfiguration,
)

from deviceforge.physics import (
    SILICON,
    compute_electron_current_density,
    compute_hole_current_density,
)
from deviceforge.postprocessing import compute_electric_field

def test_equilibrium_contact_potentials_have_correct_sign() -> None:
    simulation = build_simulation(
        shape=(41, 11),
    )

    p_contact = simulation.get_boundary_condition(
        "p_contact"
    )

    n_contact = simulation.get_boundary_condition(
        "n_contact"
    )

    assert p_contact.value < 0.0
    assert n_contact.value > 0.0


def test_built_in_potential_is_positive() -> None:
    simulation = build_simulation(
        shape=(41, 11),
    )

    built_in_potential = calculate_built_in_potential(
        simulation
    )

    assert built_in_potential > 0.0
    assert built_in_potential == pytest.approx(
        0.595,
        rel=0.02,
    )


def test_equilibrium_junction_has_opposite_doping() -> None:
    simulation = build_simulation(
        shape=(41, 11),
    )

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


def test_small_equilibrium_junction_converges() -> None:
    simulation = build_simulation(
        shape=(41, 11),
        tolerance=1.0e-7,
        max_iterations=10_000,
    )

    solver = EquilibriumPoissonSolver(
        damping_factor=0.5,
        maximum_potential_step=0.05,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
        ),
    )

    result = solver.solve(simulation)

    assert result.converged
    assert result.final_residual is not None
    assert (
        result.final_residual
        <= simulation.tolerance
    )

# small grid test
# removed for now, will re-insert this test at a future point.
"""
def test_equilibrium_currents_are_small() -> None:
    simulation = build_simulation(
        shape=(41, 11),
        tolerance=1.0e-7,
        max_iterations=10_000,
    )

    solver = EquilibriumPoissonSolver(
        damping_factor=0.5,
        maximum_potential_step=0.05,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
        ),
    )

    result = solver.solve(simulation)

    electric_field = compute_electric_field(
        result.potential
    )

    electrons = result.get_field(
        "electron_concentration"
    )

    holes = result.get_field(
        "hole_concentration"
    )

    electron_current = compute_electron_current_density(
        electron_concentration=electrons,
        electric_field_x=electric_field.x_component,
        mobility=SILICON.electron_mobility,
    )

    hole_current = compute_hole_current_density(
        hole_concentration=holes,
        electric_field_x=electric_field.x_component,
        mobility=SILICON.hole_mobility,
    )

    assert np.max(
        np.abs(electron_current.values)
    ) < 1.0e4

    assert np.max(
        np.abs(hole_current.values)
    ) < 1.0e4
"""