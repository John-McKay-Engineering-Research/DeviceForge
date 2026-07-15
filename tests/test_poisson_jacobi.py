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
    ELEMENTARY_CHARGE,
    SILICON,
    SILICON_DIOXIDE,
    VACUUM_PERMITTIVITY,
)
from deviceforge.solvers import (
    PoissonJacobiSolver,
    SolverConfiguration,
)


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(21, 9),
        spacing=(1.0e-9, 1.0e-9),
    )


@pytest.fixture
def uniformly_doped_device(
    grid_2d: Grid,
) -> Device:
    region = Region(
        name="uniform_n_type_silicon",
        grid=grid_2d,
        material=SILICON,
        mask=np.ones(
            grid_2d.shape,
            dtype=bool,
        ),
        donor_density=1.0e21,
    )

    return Device(
        name="uniform_poisson_device",
        grid=grid_2d,
        regions=(region,),
    )


@pytest.fixture
def poisson_simulation(
    grid_2d: Grid,
    uniformly_doped_device: Device,
) -> Simulation:
    left_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    left_mask[0, :] = True

    right_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    right_mask[-1, :] = True

    bottom_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    bottom_mask[1:-1, 0] = True

    top_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    top_mask[1:-1, -1] = True

    boundaries = (
        BoundaryCondition(
            name="left_contact",
            grid=grid_2d,
            mask=left_mask,
            condition_type="dirichlet",
            value=0.0,
            units="V",
        ),
        BoundaryCondition(
            name="right_contact",
            grid=grid_2d,
            mask=right_mask,
            condition_type="dirichlet",
            value=0.0,
            units="V",
        ),
        BoundaryCondition(
            name="bottom_insulated",
            grid=grid_2d,
            mask=bottom_mask,
            condition_type="neumann",
            value=0.0,
            units="V/m",
        ),
        BoundaryCondition(
            name="top_insulated",
            grid=grid_2d,
            mask=top_mask,
            condition_type="neumann",
            value=0.0,
            units="V/m",
        ),
    )

    return Simulation(
        name="constant_source_poisson_validation",
        device=uniformly_doped_device,
        boundary_conditions=boundaries,
        tolerance=1.0e-12,
        max_iterations=50_000,
        initial_potential=0.0,
    )


def analytical_constant_source_solution(
    simulation: Simulation,
) -> np.ndarray:
    """
    Return the analytical solution for constant positive charge.

    For:

        d²phi/dx² = -source

    with:

        phi(0) = 0
        phi(L) = 0

    the solution is:

        phi(x) = source * x * (L - x) / 2
    """

    x_coordinates = simulation.grid.coordinates(
        axis=0
    )

    domain_length = simulation.grid.physical_size[0]

    charge_density = (
        ELEMENTARY_CHARGE * 1.0e21
    )

    absolute_permittivity = (
        VACUUM_PERMITTIVITY
        * SILICON.relative_permittivity
    )

    source = (
        charge_density
        / absolute_permittivity
    )

    x_profile = (
        0.5
        * source
        * x_coordinates
        * (domain_length - x_coordinates)
    )

    return np.repeat(
        x_profile[:, np.newaxis],
        simulation.grid.shape[1],
        axis=1,
    )


def test_solver_returns_simulation_result(
    poisson_simulation: Simulation,
) -> None:
    result = PoissonJacobiSolver().solve(
        poisson_simulation
    )

    assert isinstance(result, SimulationResult)
    assert result.solver_name == "poisson_jacobi"
    assert result.backend_name == "numpy"
    assert result.iterations > 0


def test_solver_converges(
    poisson_simulation: Simulation,
) -> None:
    result = PoissonJacobiSolver().solve(
        poisson_simulation
    )

    assert result.converged
    assert result.final_residual is not None
    assert result.final_residual <= 1.0e-12


def test_result_contains_electrostatic_fields(
    poisson_simulation: Simulation,
) -> None:
    result = PoissonJacobiSolver().solve(
        poisson_simulation
    )

    assert result.field_names == (
        "electrostatic_potential",
        "fixed_charge_density",
        "absolute_permittivity",
        "electrostatic_source_term",
    )

    assert (
        result.get_field(
            "fixed_charge_density"
        ).units
        == "C/m^3"
    )

    assert (
        result.get_field(
            "absolute_permittivity"
        ).units
        == "F/m"
    )

    assert (
        result.get_field(
            "electrostatic_source_term"
        ).units
        == "V/m^2"
    )


def test_dirichlet_boundaries_are_preserved(
    poisson_simulation: Simulation,
) -> None:
    result = PoissonJacobiSolver().solve(
        poisson_simulation
    )

    potential = result.potential.values

    np.testing.assert_allclose(
        potential[0, :],
        0.0,
    )

    np.testing.assert_allclose(
        potential[-1, :],
        0.0,
    )


def test_homogeneous_neumann_boundaries(
    poisson_simulation: Simulation,
) -> None:
    result = PoissonJacobiSolver().solve(
        poisson_simulation
    )

    potential = result.potential.values

    np.testing.assert_allclose(
        potential[1:-1, 0],
        potential[1:-1, 1],
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        potential[1:-1, -1],
        potential[1:-1, -2],
        atol=1.0e-12,
    )


def test_solution_matches_constant_source_analytical_solution(
    poisson_simulation: Simulation,
) -> None:
    result = PoissonJacobiSolver().solve(
        poisson_simulation
    )

    expected = analytical_constant_source_solution(
        poisson_simulation
    )

    np.testing.assert_allclose(
        result.potential.values,
        expected,
        atol=2.0e-9,
        rtol=1.0e-4,
    )


def test_solution_is_independent_of_y(
    poisson_simulation: Simulation,
) -> None:
    result = PoissonJacobiSolver().solve(
        poisson_simulation
    )

    potential = result.potential.values

    centre_column = potential[
        :,
        potential.shape[1] // 2,
    ]

    for column in range(potential.shape[1]):
        np.testing.assert_allclose(
            potential[:, column],
            centre_column,
            atol=1.0e-12,
        )


def test_zero_charge_produces_zero_potential(
    grid_2d: Grid,
) -> None:
    intrinsic_region = Region(
        name="intrinsic_silicon",
        grid=grid_2d,
        material=SILICON,
        mask=np.ones(
            grid_2d.shape,
            dtype=bool,
        ),
    )

    device = Device(
        name="intrinsic_device",
        grid=grid_2d,
        regions=(intrinsic_region,),
    )

    left_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    left_mask[0, :] = True

    right_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    right_mask[-1, :] = True

    boundaries = (
        BoundaryCondition(
            name="left",
            grid=grid_2d,
            mask=left_mask,
            condition_type="dirichlet",
            value=0.0,
            units="V",
        ),
        BoundaryCondition(
            name="right",
            grid=grid_2d,
            mask=right_mask,
            condition_type="dirichlet",
            value=0.0,
            units="V",
        ),
    )

    simulation = Simulation(
        device=device,
        boundary_conditions=boundaries,
        tolerance=1.0e-12,
    )

    result = PoissonJacobiSolver().solve(
        simulation
    )

    np.testing.assert_allclose(
        result.potential.values,
        0.0,
    )


def test_solver_stops_at_iteration_limit(
    poisson_simulation: Simulation,
) -> None:
    solver = PoissonJacobiSolver(
        SolverConfiguration(
            tolerance=1.0e-20,
            max_iterations=2,
        )
    )

    result = solver.solve(
        poisson_simulation
    )

    assert not result.converged
    assert result.iterations == 2


def test_unequal_spacing_is_rejected(
    uniformly_doped_device: Device,
) -> None:
    grid = Grid(
        shape=(21, 9),
        spacing=(1.0e-9, 2.0e-9),
    )

    region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=np.ones(grid.shape, dtype=bool),
        donor_density=1.0e21,
    )

    device = Device(
        name="unequal_spacing_device",
        grid=grid,
        regions=(region,),
    )

    mask = np.zeros(grid.shape, dtype=bool)
    mask[0, :] = True

    boundary = BoundaryCondition(
        name="left",
        grid=grid,
        mask=mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    simulation = Simulation(
        device=device,
        boundary_conditions=(boundary,),
    )

    with pytest.raises(
        ValueError,
        match="equal grid spacing",
    ):
        PoissonJacobiSolver().solve(simulation)


def test_variable_permittivity_is_rejected(
    grid_2d: Grid,
) -> None:
    silicon_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    silicon_mask[:10, :] = True

    oxide_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    oxide_mask[10:, :] = True

    silicon_region = Region(
        name="silicon",
        grid=grid_2d,
        material=SILICON,
        mask=silicon_mask,
    )

    oxide_region = Region(
        name="oxide",
        grid=grid_2d,
        material=SILICON_DIOXIDE,
        mask=oxide_mask,
    )

    device = Device(
        name="variable_permittivity_device",
        grid=grid_2d,
        regions=(
            silicon_region,
            oxide_region,
        ),
    )

    left_mask = np.zeros(
        grid_2d.shape,
        dtype=bool,
    )
    left_mask[0, :] = True

    boundary = BoundaryCondition(
        name="left",
        grid=grid_2d,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    simulation = Simulation(
        device=device,
        boundary_conditions=(boundary,),
    )

    with pytest.raises(
        ValueError,
        match="constant permittivity",
    ):
        PoissonJacobiSolver().solve(simulation)