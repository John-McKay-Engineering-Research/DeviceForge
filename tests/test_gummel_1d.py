import numpy as np

from deviceforge import (
    Device,
    Grid,
    Region,
    Simulation,
)
from deviceforge.physics import SILICON
from deviceforge.solvers import (
    GummelDriftDiffusionSolver1D,
    SolverConfiguration,
)

from deviceforge import (
    BoundaryCondition,
    Device,
    Grid,
    Region,
    Simulation,
)
# updated helper function due to pytest failure
def build_uniform_intrinsic_simulation() -> Simulation:
    grid = Grid(
        shape=(21,),
        spacing=(1.0e-9,),
    )

    region = Region(
        name="intrinsic_silicon",
        grid=grid,
        material=SILICON,
        mask=np.ones(grid.shape, dtype=bool),
    )

    device = Device(
        name="intrinsic_1d_device",
        grid=grid,
        regions=(region,),
    )

    left_mask = np.zeros(grid.shape, dtype=bool)
    left_mask[0] = True

    right_mask = np.zeros(grid.shape, dtype=bool)
    right_mask[-1] = True

    left_contact = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    right_contact = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    return Simulation(
        name="intrinsic_gummel_validation",
        device=device,
        boundary_conditions=(
            left_contact,
            right_contact,
        ),
        tolerance=1.0e-8,
        max_iterations=500,
        initial_potential=0.0,
    )


def test_intrinsic_zero_bias_converges() -> None:
    simulation = build_uniform_intrinsic_simulation()

    solver = GummelDriftDiffusionSolver1D(
        applied_voltage=0.0,
        damping_factor=0.5,
        configuration=SolverConfiguration(
            tolerance=1.0e-8,
            max_iterations=500,
        ),
    )

    result = solver.solve(simulation)

    assert result.converged


def test_intrinsic_zero_bias_potential_is_zero() -> None:
    simulation = build_uniform_intrinsic_simulation()

    result = GummelDriftDiffusionSolver1D(
        configuration=SolverConfiguration(
            tolerance=1.0e-8,
            max_iterations=500,
        )
    ).solve(simulation)

    np.testing.assert_allclose(
        result.potential.values,
        0.0,
        atol=1.0e-10,
    )


def test_intrinsic_carriers_remain_equal() -> None:
    simulation = build_uniform_intrinsic_simulation()

    result = GummelDriftDiffusionSolver1D(
        configuration=SolverConfiguration(
            tolerance=1.0e-8,
            max_iterations=500,
        )
    ).solve(simulation)

    electrons = result.get_field(
        "electron_concentration"
    )

    holes = result.get_field(
        "hole_concentration"
    )

    np.testing.assert_allclose(
        electrons.values,
        holes.values,
        rtol=1.0e-10,
    )