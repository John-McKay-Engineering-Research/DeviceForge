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
from deviceforge.solvers import BaseSolver, SolverConfiguration


class DummySolver(BaseSolver):
    """Minimal concrete solver used to test the abstract interface."""

    @property
    def name(self) -> str:
        return "dummy"

    def solve(
        self,
        simulation: Simulation,
    ) -> SimulationResult:
        self.validate_simulation(simulation)

        potential = simulation.create_initial_potential_field()

        return SimulationResult(
            fields={
                "electrostatic_potential": potential,
            },
            converged=True,
            iterations=0,
            residual_history=np.array([], dtype=np.float64),
            runtime_seconds=0.0,
            solver_name=self.name,
            backend_name=self.backend_name,
        )


@pytest.fixture
def simulation_2d() -> Simulation:
    grid = Grid(
        shape=(6, 4),
        spacing=(1.0e-9, 1.0e-9),
    )

    region = Region(
        name="silicon_domain",
        grid=grid,
        material=SILICON,
        mask=np.ones(grid.shape, dtype=bool),
    )

    device = Device(
        name="rectangular_device",
        grid=grid,
        regions=(region,),
    )

    left_mask = np.zeros(grid.shape, dtype=bool)
    left_mask[0, :] = True

    left_boundary = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    return Simulation(
        device=device,
        boundary_conditions=(left_boundary,),
        tolerance=1.0e-10,
        max_iterations=2_000,
    )


def test_default_configuration() -> None:
    configuration = SolverConfiguration()

    assert configuration.tolerance == pytest.approx(1.0e-8)
    assert configuration.max_iterations == 10_000
    assert configuration.backend_name == "numpy"


def test_solver_uses_default_configuration() -> None:
    solver = DummySolver()

    assert solver.name == "dummy"
    assert solver.tolerance == pytest.approx(1.0e-8)
    assert solver.max_iterations == 10_000
    assert solver.backend_name == "numpy"


def test_solver_uses_custom_configuration() -> None:
    configuration = SolverConfiguration(
        tolerance=1.0e-12,
        max_iterations=5_000,
        backend_name="test_backend",
    )

    solver = DummySolver(configuration)

    assert solver.configuration is configuration
    assert solver.tolerance == pytest.approx(1.0e-12)
    assert solver.max_iterations == 5_000
    assert solver.backend_name == "test_backend"


def test_dummy_solver_returns_result(
    simulation_2d: Simulation,
) -> None:
    solver = DummySolver()

    result = solver.solve(simulation_2d)

    assert isinstance(result, SimulationResult)
    assert result.converged
    assert result.solver_name == "dummy"
    assert result.backend_name == "numpy"
    assert result.potential.grid is simulation_2d.grid


def test_effective_tolerance_uses_stricter_value(
    simulation_2d: Simulation,
) -> None:
    solver = DummySolver(
        SolverConfiguration(
            tolerance=1.0e-8,
            max_iterations=10_000,
        )
    )

    assert solver.effective_tolerance(
        simulation_2d
    ) == pytest.approx(1.0e-10)


def test_effective_iteration_limit_uses_lower_value(
    simulation_2d: Simulation,
) -> None:
    solver = DummySolver(
        SolverConfiguration(
            tolerance=1.0e-8,
            max_iterations=10_000,
        )
    )

    assert solver.effective_max_iterations(simulation_2d) == 2_000


def test_invalid_solver_input_raises_type_error() -> None:
    solver = DummySolver()

    with pytest.raises(TypeError, match="Simulation"):
        solver.validate_simulation("not a simulation")  # type: ignore[arg-type]


def test_abstract_solver_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseSolver()  # type: ignore[abstract]


@pytest.mark.parametrize(
    "tolerance",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_tolerance_raises_value_error(
    tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        SolverConfiguration(tolerance=tolerance)


@pytest.mark.parametrize(
    "max_iterations",
    [
        0,
        -1,
        -100,
    ],
)
def test_invalid_iteration_limit_raises_value_error(
    max_iterations: int,
) -> None:
    with pytest.raises(ValueError, match="iteration"):
        SolverConfiguration(max_iterations=max_iterations)


def test_boolean_iteration_limit_raises_type_error() -> None:
    with pytest.raises(TypeError, match="integer"):
        SolverConfiguration(max_iterations=True)


@pytest.mark.parametrize(
    "backend_name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_empty_backend_name_raises_value_error(
    backend_name: str,
) -> None:
    with pytest.raises(ValueError, match="Backend name"):
        SolverConfiguration(backend_name=backend_name)