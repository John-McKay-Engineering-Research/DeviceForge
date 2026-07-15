import numpy as np
import pytest

from deviceforge import (
    BoundaryCondition,
    Device,
    Grid,
    Region,
    Simulation,
    SimulationResult,
)
from deviceforge.physics import (
    SILICON,
    charge_neutral_potential,
)
from deviceforge.solvers import (
    EquilibriumPoissonSolver,
    SolverConfiguration,
)


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(21, 9),
        spacing=(1.0e-9, 1.0e-9),
    )


def build_uniform_simulation(
    grid: Grid,
    *,
    donor_density: float = 0.0,
    acceptor_density: float = 0.0,
) -> Simulation:
    """Create a uniform charge-neutral semiconductor problem."""

    region = Region(
        name="uniform_silicon",
        grid=grid,
        material=SILICON,
        mask=np.ones(grid.shape, dtype=bool),
        donor_density=donor_density,
        acceptor_density=acceptor_density,
    )

    device = Device(
        name="uniform_equilibrium_device",
        grid=grid,
        regions=(region,),
    )

    net_doping = donor_density - acceptor_density

    neutral_potential = float(
        charge_neutral_potential(net_doping)
    )

    left_mask = np.zeros(grid.shape, dtype=bool)
    left_mask[0, :] = True

    right_mask = np.zeros(grid.shape, dtype=bool)
    right_mask[-1, :] = True

    bottom_mask = np.zeros(grid.shape, dtype=bool)
    bottom_mask[1:-1, 0] = True

    top_mask = np.zeros(grid.shape, dtype=bool)
    top_mask[1:-1, -1] = True

    boundaries = (
        BoundaryCondition(
            name="left_contact",
            grid=grid,
            mask=left_mask,
            condition_type="dirichlet",
            value=neutral_potential,
            units="V",
        ),
        BoundaryCondition(
            name="right_contact",
            grid=grid,
            mask=right_mask,
            condition_type="dirichlet",
            value=neutral_potential,
            units="V",
        ),
        BoundaryCondition(
            name="bottom_insulated",
            grid=grid,
            mask=bottom_mask,
            condition_type="neumann",
            value=0.0,
            units="V/m",
        ),
        BoundaryCondition(
            name="top_insulated",
            grid=grid,
            mask=top_mask,
            condition_type="neumann",
            value=0.0,
            units="V/m",
        ),
    )

    return Simulation(
        name="uniform_equilibrium_validation",
        device=device,
        boundary_conditions=boundaries,
        tolerance=1.0e-10,
        max_iterations=5_000,
        initial_potential=neutral_potential,
    )


def test_default_solver_properties() -> None:
    solver = EquilibriumPoissonSolver()

    assert solver.name == "equilibrium_poisson"
    assert solver.damping_factor == pytest.approx(0.5)
    assert solver.maximum_potential_step == pytest.approx(
        0.1
    )


@pytest.mark.parametrize(
    ("donor_density", "acceptor_density"),
    [
        (0.0, 0.0),
        (1.0e21, 0.0),
        (0.0, 1.0e21),
    ],
)
def test_uniform_charge_neutral_region_converges(
    grid_2d: Grid,
    donor_density: float,
    acceptor_density: float,
) -> None:
    simulation = build_uniform_simulation(
        grid_2d,
        donor_density=donor_density,
        acceptor_density=acceptor_density,
    )

    result = EquilibriumPoissonSolver().solve(
        simulation
    )

    assert result.converged
    assert result.final_residual is not None
    assert result.final_residual <= simulation.tolerance


def test_uniform_solution_remains_constant(
    grid_2d: Grid,
) -> None:
    simulation = build_uniform_simulation(
        grid_2d,
        donor_density=1.0e21,
    )

    expected = simulation.initial_potential

    result = EquilibriumPoissonSolver().solve(
        simulation
    )

    np.testing.assert_allclose(
        result.potential.values,
        expected,
        atol=1.0e-10,
    )


def test_uniform_solution_is_charge_neutral(
    grid_2d: Grid,
) -> None:
    simulation = build_uniform_simulation(
        grid_2d,
        acceptor_density=1.0e21,
    )

    result = EquilibriumPoissonSolver().solve(
        simulation
    )

    charge_density = result.get_field(
        "equilibrium_charge_density"
    )

    np.testing.assert_allclose(
        charge_density.values,
        0.0,
        atol=1.0e-9,
    )


def test_result_contains_carrier_fields(
    grid_2d: Grid,
) -> None:
    simulation = build_uniform_simulation(
        grid_2d,
        donor_density=1.0e21,
    )

    result = EquilibriumPoissonSolver().solve(
        simulation
    )

    assert isinstance(result, SimulationResult)

    assert "electron_concentration" in result.field_names
    assert "hole_concentration" in result.field_names
    assert "equilibrium_charge_density" in result.field_names
    assert "donor_density" in result.field_names
    assert "acceptor_density" in result.field_names


def test_dirichlet_boundaries_are_preserved(
    grid_2d: Grid,
) -> None:
    simulation = build_uniform_simulation(
        grid_2d,
        donor_density=1.0e21,
    )

    result = EquilibriumPoissonSolver().solve(
        simulation
    )

    expected = simulation.initial_potential

    np.testing.assert_allclose(
        result.potential.values[0, :],
        expected,
    )

    np.testing.assert_allclose(
        result.potential.values[-1, :],
        expected,
    )


def test_solver_stops_at_iteration_limit(
    grid_2d: Grid,
) -> None:
    simulation = build_uniform_simulation(
        grid_2d,
        donor_density=1.0e21,
    )

    perturbed_values = simulation.initial_potential + 0.2

    simulation = Simulation(
        name=simulation.name,
        device=simulation.device,
        boundary_conditions=simulation.boundary_conditions,
        tolerance=1.0e-20,
        max_iterations=2,
        initial_potential=perturbed_values,
    )

    solver = EquilibriumPoissonSolver(
        configuration=SolverConfiguration(
            tolerance=1.0e-20,
            max_iterations=2,
        )
    )

    result = solver.solve(simulation)

    assert not result.converged
    assert result.iterations == 2


@pytest.mark.parametrize(
    "damping_factor",
    [
        0.0,
        -0.1,
        1.1,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_damping_factor_is_rejected(
    damping_factor: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Damping factor",
    ):
        EquilibriumPoissonSolver(
            damping_factor=damping_factor
        )


@pytest.mark.parametrize(
    "maximum_potential_step",
    [
        0.0,
        -0.1,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_maximum_step_is_rejected(
    maximum_potential_step: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Maximum potential step",
    ):
        EquilibriumPoissonSolver(
            maximum_potential_step=(
                maximum_potential_step
            )
        )