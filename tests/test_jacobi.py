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
from deviceforge.physics import SILICON
from deviceforge.solvers import (
    JacobiSolver,
    SolverConfiguration,
)


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(21, 11),
        spacing=(1.0e-9, 1.0e-9),
    )


@pytest.fixture
def device_2d(grid_2d: Grid) -> Device:
    region = Region(
        name="silicon_domain",
        grid=grid_2d,
        material=SILICON,
        mask=np.ones(grid_2d.shape, dtype=bool),
    )

    return Device(
        name="laplace_rectangle",
        grid=grid_2d,
        regions=(region,),
    )


@pytest.fixture
def simulation_2d(
    grid_2d: Grid,
    device_2d: Device,
) -> Simulation:
    left_mask = np.zeros(grid_2d.shape, dtype=bool)
    left_mask[0, :] = True

    right_mask = np.zeros(grid_2d.shape, dtype=bool)
    right_mask[-1, :] = True

    top_mask = np.zeros(grid_2d.shape, dtype=bool)
    top_mask[1:-1, -1] = True

    bottom_mask = np.zeros(grid_2d.shape, dtype=bool)
    bottom_mask[1:-1, 0] = True

    left = BoundaryCondition(
        name="left_contact",
        grid=grid_2d,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    right = BoundaryCondition(
        name="right_contact",
        grid=grid_2d,
        mask=right_mask,
        condition_type="dirichlet",
        value=1.0,
        units="V",
    )

    top = BoundaryCondition(
        name="top_insulated",
        grid=grid_2d,
        mask=top_mask,
        condition_type="neumann",
        value=0.0,
        units="V/m",
    )

    bottom = BoundaryCondition(
        name="bottom_insulated",
        grid=grid_2d,
        mask=bottom_mask,
        condition_type="neumann",
        value=0.0,
        units="V/m",
    )

    return Simulation(
        name="linear_laplace_problem",
        device=device_2d,
        boundary_conditions=(
            left,
            right,
            top,
            bottom,
        ),
        tolerance=1.0e-8,
        max_iterations=20_000,
        initial_potential=0.5,
    )


def test_jacobi_solver_returns_simulation_result(
    simulation_2d: Simulation,
) -> None:
    solver = JacobiSolver()

    result = solver.solve(simulation_2d)

    assert isinstance(result, SimulationResult)
    assert result.solver_name == "jacobi"
    assert result.backend_name == "numpy"
    assert result.iterations > 0
    assert result.runtime_seconds >= 0.0


def test_jacobi_solver_converges(
    simulation_2d: Simulation,
) -> None:
    solver = JacobiSolver()

    result = solver.solve(simulation_2d)

    assert result.converged
    assert result.final_residual is not None
    assert result.final_residual <= 1.0e-8


def test_dirichlet_boundaries_are_preserved(
    simulation_2d: Simulation,
) -> None:
    solver = JacobiSolver()

    result = solver.solve(simulation_2d)
    potential = result.potential.values

    np.testing.assert_allclose(
        potential[0, :],
        0.0,
    )

    np.testing.assert_allclose(
        potential[-1, :],
        1.0,
    )


def test_solution_is_approximately_linear(
    simulation_2d: Simulation,
) -> None:
    solver = JacobiSolver()

    result = solver.solve(simulation_2d)
    potential = result.potential.values

    expected_profile = np.linspace(
        0.0,
        1.0,
        simulation_2d.grid.shape[0],
    )

    centre_column = potential[
        :,
        simulation_2d.grid.shape[1] // 2,
    ]

    np.testing.assert_allclose(
        centre_column,
        expected_profile,
        atol=1.0e-4,
    )


def test_homogeneous_neumann_boundaries(
    simulation_2d: Simulation,
) -> None:
    solver = JacobiSolver()

    result = solver.solve(simulation_2d)
    potential = result.potential.values

    np.testing.assert_allclose(
        potential[1:-1, 0],
        potential[1:-1, 1],
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        potential[1:-1, -1],
        potential[1:-1, -2],
        atol=1.0e-10,
    )


def test_residual_history_matches_iteration_count(
    simulation_2d: Simulation,
) -> None:
    solver = JacobiSolver()

    result = solver.solve(simulation_2d)

    assert result.residual_history.size == result.iterations


def test_residual_generally_decreases(
    simulation_2d: Simulation,
) -> None:
    solver = JacobiSolver()

    result = solver.solve(simulation_2d)

    assert result.final_residual is not None
    assert result.initial_residual is not None

    assert result.final_residual < result.initial_residual


def test_custom_solver_configuration(
    simulation_2d: Simulation,
) -> None:
    configuration = SolverConfiguration(
        tolerance=1.0e-6,
        max_iterations=5_000,
        backend_name="numpy",
    )

    solver = JacobiSolver(configuration)

    result = solver.solve(simulation_2d)

    assert result.metadata["maximum_iterations"] == 5_000
    assert result.metadata["tolerance"] == pytest.approx(1.0e-8)


def test_solver_stops_at_iteration_limit(
    simulation_2d: Simulation,
) -> None:
    solver = JacobiSolver(
        SolverConfiguration(
            tolerance=1.0e-15,
            max_iterations=2,
        )
    )

    result = solver.solve(simulation_2d)

    assert not result.converged
    assert result.iterations == 2


def test_unequal_spacing_is_rejected(
    device_2d: Device,
) -> None:
    grid = Grid(
        shape=(21, 11),
        spacing=(1.0e-9, 2.0e-9),
    )

    region = Region(
        name="silicon_domain",
        grid=grid,
        material=SILICON,
        mask=np.ones(grid.shape, dtype=bool),
    )

    device = Device(
        name="unequal_spacing_device",
        grid=grid,
        regions=(region,),
    )

    left_mask = np.zeros(grid.shape, dtype=bool)
    left_mask[0, :] = True

    boundary = BoundaryCondition(
        name="left",
        grid=grid,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    simulation = Simulation(
        device=device,
        boundary_conditions=(boundary,),
    )

    solver = JacobiSolver()

    with pytest.raises(ValueError, match="equal grid spacing"):
        solver.solve(simulation)


def test_nonzero_neumann_condition_is_rejected(
    grid_2d: Grid,
    device_2d: Device,
) -> None:
    left_mask = np.zeros(grid_2d.shape, dtype=bool)
    left_mask[0, :] = True

    right_mask = np.zeros(grid_2d.shape, dtype=bool)
    right_mask[-1, :] = True

    left = BoundaryCondition(
        name="left",
        grid=grid_2d,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    right_gradient = BoundaryCondition(
        name="right_gradient",
        grid=grid_2d,
        mask=right_mask,
        condition_type="neumann",
        value=1.0,
        units="V/m",
    )

    simulation = Simulation(
        device=device_2d,
        boundary_conditions=(
            left,
            right_gradient,
        ),
    )

    solver = JacobiSolver()

    with pytest.raises(ValueError, match="homogeneous Neumann"):
        solver.solve(simulation)