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

from deviceforge import (
    BoundaryCondition,
    Device,
    Grid,
    Region,
    Simulation,
)
from deviceforge.physics import (
    SILICON,
    compute_electron_scharfetter_gummel_current_x,
    compute_hole_scharfetter_gummel_current_x,
    compute_total_scharfetter_gummel_current_x,
    diffusion_coefficient,
)
from deviceforge.physics import ELEMENTARY_CHARGE
import pytest

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

# PN-junction test helper
# boundary values are currently placeholders
# because GummelDriftDiffusionSolver1D calculates charge neutral contacts internally
def build_pn_junction_simulation(
    *,
    shape: tuple[int] = (81,),
    device_length: float = 80.0e-9,
    tolerance: float = 1.0e-7,
    max_iterations: int = 2_000,
) -> Simulation:
    """Create a symmetric one-dimensional abrupt PN junction."""

    if len(shape) != 1:
        raise ValueError("The PN-junction grid must be one-dimensional.")

    number_of_nodes = shape[0]

    if number_of_nodes < 3:
        raise ValueError("At least three grid nodes are required.")

    spacing = device_length / (number_of_nodes - 1)

    grid = Grid(
        shape=shape,
        spacing=(spacing,),
    )

    junction_index = grid.shape[0] // 2

    p_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    p_mask[:junction_index] = True

    n_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    n_mask[junction_index:] = True

    p_region = Region(
        name="p_type_silicon",
        grid=grid,
        material=SILICON,
        mask=p_mask,
        acceptor_density=1.0e21,
        region_type="semiconductor",
    )

    n_region = Region(
        name="n_type_silicon",
        grid=grid,
        material=SILICON,
        mask=n_mask,
        donor_density=1.0e21,
        region_type="semiconductor",
    )

    device = Device(
        name="one_dimensional_pn_junction",
        grid=grid,
        regions=(
            p_region,
            n_region,
        ),
    )

    left_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    left_mask[0] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    right_mask[-1] = True

    left_contact = BoundaryCondition(
        name="p_contact",
        grid=grid,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    right_contact = BoundaryCondition(
        name="n_contact",
        grid=grid,
        mask=right_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    return Simulation(
        name="one_dimensional_equilibrium_pn_junction",
        device=device,
        boundary_conditions=(
            left_contact,
            right_contact,
        ),
        tolerance=tolerance,
        max_iterations=max_iterations,
        initial_potential=0.0,
    )

# Add basic convergence test
# uses damping factor=0.2.
def test_zero_bias_pn_junction_converges() -> None:
    simulation = build_pn_junction_simulation()

    solver = GummelDriftDiffusionSolver1D(
        applied_voltage=0.0,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
        ),
    )

    result = solver.solve(simulation)

    assert result.converged
    assert result.final_residual is not None
    assert result.final_residual <= simulation.tolerance

# added new test
def test_gummel_result_contains_current_density_fields() -> None:
    simulation = build_pn_junction_simulation()

    solver = GummelDriftDiffusionSolver1D(
        applied_voltage=0.0,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
        ),
    )

    result = solver.solve(simulation)

    electron_current = result.get_field(
        "electron_current_density_x_edges"
    )

    hole_current = result.get_field(
        "hole_current_density_x_edges"
    )

    total_current = result.get_field(
        "total_current_density_x_edges"
    )

    expected_edge_shape = (
        simulation.grid.shape[0] - 1,
    )

    assert electron_current.grid.shape == expected_edge_shape
    assert hole_current.grid.shape == expected_edge_shape
    assert total_current.grid.shape == expected_edge_shape

    assert electron_current.units == "A/m^2"
    assert hole_current.units == "A/m^2"
    assert total_current.units == "A/m^2"

# added new test
def test_total_current_is_sum_of_electron_and_hole_currents() -> None:
    simulation = build_pn_junction_simulation()

    result = GummelDriftDiffusionSolver1D(
        applied_voltage=0.0,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
        ),
    ).solve(simulation)

    electron_current = result.get_field(
        "electron_current_density_x_edges"
    )

    hole_current = result.get_field(
        "hole_current_density_x_edges"
    )

    total_current = result.get_field(
        "total_current_density_x_edges"
    )

    np.testing.assert_allclose(
        total_current.values,
        electron_current.values + hole_current.values,
    )

def test_terminal_current_metadata_matches_current_field() -> None:
    simulation = build_pn_junction_simulation()

    result = GummelDriftDiffusionSolver1D(
        applied_voltage=0.0,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
        ),
    ).solve(simulation)

    total_current = result.get_field(
        "total_current_density_x_edges"
    )

    expected_left_current = float(
        total_current.values[0]
    )

    expected_right_current = float(
        total_current.values[-1]
    )

    expected_average_current = 0.5 * (
        expected_left_current
        + expected_right_current
    )

    assert result.metadata[
        "left_terminal_current_density"
    ] == pytest.approx(expected_left_current)

    assert result.metadata[
        "right_terminal_current_density"
    ] == pytest.approx(expected_right_current)

    assert result.metadata[
        "average_terminal_current_density"
    ] == pytest.approx(expected_average_current)

# added new test
def test_current_nonuniformity_metadata_matches_current_field() -> None:
    simulation = build_pn_junction_simulation()

    result = GummelDriftDiffusionSolver1D(
        applied_voltage=0.0,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
        ),
    ).solve(simulation)

    total_current = result.get_field(
        "total_current_density_x_edges"
    )

    expected_nonuniformity = float(
        np.max(total_current.values)
        - np.min(total_current.values)
    )

    assert result.metadata[
        "current_density_nonuniformity"
    ] == pytest.approx(expected_nonuniformity)

# verify the potential polarity
def test_zero_bias_pn_potential_has_correct_polarity() -> None:
    simulation = build_pn_junction_simulation()

    result = GummelDriftDiffusionSolver1D(
        applied_voltage=0.0,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
        ),
    ).solve(simulation)

    potential = result.potential.values
    junction_index = simulation.grid.shape[0] // 2

    assert np.mean(
        potential[:junction_index]
    ) < 0.0

    assert np.mean(
        potential[junction_index:]
    ) > 0.0

#  verify the majority carries
def test_zero_bias_pn_majority_carriers() -> None:
    simulation = build_pn_junction_simulation()

    result = GummelDriftDiffusionSolver1D(
        applied_voltage=0.0,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
        ),
    ).solve(simulation)

    electrons = result.get_field(
        "electron_concentration"
    ).values

    holes = result.get_field(
        "hole_concentration"
    ).values

    junction_index = simulation.grid.shape[0] // 2

    assert np.mean(
        holes[:junction_index]
    ) > np.mean(
        electrons[:junction_index]
    )

    assert np.mean(
        electrons[junction_index:]
    ) > np.mean(
        holes[junction_index:]
    )

# extract edge currents
# currents should ideally be constant across the device at steady state
def test_zero_bias_pn_terminal_current_is_small() -> None:
    simulation = build_pn_junction_simulation()

    result = GummelDriftDiffusionSolver1D(
        applied_voltage=0.0,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
        ),
    ).solve(simulation)

    total_current = result.get_field(
        "total_current_density_x_edges"
    )

    terminal_current = float(
        0.5
        * (
            total_current.values[0]
            + total_current.values[-1]
        )
    )

    maximum_carrier_concentration = max(
        float(
            np.max(
                result.get_field(
                    "electron_concentration"
                ).values
            )
        ),
        float(
            np.max(
                result.get_field(
                    "hole_concentration"
                ).values
            )
        ),
    )

    maximum_diffusivity = max(
        diffusion_coefficient(
            SILICON.electron_mobility
        ),
        diffusion_coefficient(
            SILICON.hole_mobility
        ),
    )

    characteristic_current_density = (
        ELEMENTARY_CHARGE
        * maximum_diffusivity
        * maximum_carrier_concentration
        / simulation.grid.spacing[0]
    )

    relative_terminal_current = (
        abs(terminal_current)
        / characteristic_current_density
    )

    assert relative_terminal_current < 1.0e-7


def test_continuity_diagnostics_are_present_and_finite() -> None:
    simulation = build_pn_junction_simulation()

    result = GummelDriftDiffusionSolver1D(
        applied_voltage=0.05,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=1.0e-10,
            max_iterations=simulation.max_iterations,
        ),
    ).solve(simulation)

    diagnostic_keys = (
        "maximum_electron_current_divergence",
        "maximum_hole_current_divergence",
        "maximum_recombination_current_source",
        "maximum_electron_continuity_defect",
        "maximum_hole_continuity_defect",
        "electron_continuity_relative_defect",
        "hole_continuity_relative_defect",
    )

    for key in diagnostic_keys:
        assert key in result.metadata
        assert np.isfinite(result.metadata[key])
        assert result.metadata[key] >= 0.0

    assert (
        result.metadata[
            "continuity_raw_diagnostic_units"
        ]
        == "A/m^3"
    )

    assert (
        result.metadata[
            "continuity_relative_defect_units"
        ]
        == "dimensionless"
    )


def test_continuity_diagnostics_align_edges_with_interior_nodes() -> None:
    electron_current = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float64,
    )

    hole_current = np.asarray(
        [4.0, 3.0, 2.0, 1.0],
        dtype=np.float64,
    )

    recombination = np.zeros(
        5,
        dtype=np.float64,
    )

    diagnostics = (
        GummelDriftDiffusionSolver1D
        ._continuity_diagnostics(
            electron_current=electron_current,
            hole_current=hole_current,
            recombination=recombination,
            spacing=1.0,
        )
    )

    assert diagnostics[
        "maximum_electron_current_divergence"
    ] == pytest.approx(1.0)

    assert diagnostics[
        "maximum_hole_current_divergence"
    ] == pytest.approx(1.0)

    assert diagnostics[
        "maximum_electron_continuity_defect"
    ] == pytest.approx(1.0)

    assert diagnostics[
        "maximum_hole_continuity_defect"
    ] == pytest.approx(1.0)

    assert diagnostics[
        "electron_continuity_relative_defect"
    ] == pytest.approx(1.0)

    assert diagnostics[
        "hole_continuity_relative_defect"
    ] == pytest.approx(1.0)

def test_algebraic_continuity_diagnostics_are_present_and_finite() -> None:
    simulation = build_pn_junction_simulation()

    result = GummelDriftDiffusionSolver1D(
        applied_voltage=0.05,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=1.0e-10,
            max_iterations=simulation.max_iterations,
        ),
    ).solve(simulation)

    diagnostic_keys = (
        "maximum_electron_linear_solve_algebraic_residual",
        "electron_linear_solve_relative_algebraic_residual",
        "maximum_hole_linear_solve_algebraic_residual",
        "hole_linear_solve_relative_algebraic_residual",
        "maximum_electron_damped_state_algebraic_residual",
        "electron_damped_state_relative_algebraic_residual",
        "maximum_hole_damped_state_algebraic_residual",
        "hole_damped_state_relative_algebraic_residual",
        "maximum_recombination_lag",
        "relative_recombination_lag",
    )

    for key in diagnostic_keys:
        assert key in result.metadata
        assert np.isfinite(result.metadata[key])
        assert result.metadata[key] >= 0.0

    assert (
        result.metadata[
            "electron_linear_solve_relative_algebraic_residual"
        ]
        < 1.0e-10
    )

    assert (
        result.metadata[
            "hole_linear_solve_relative_algebraic_residual"
        ]
        < 1.0e-10
    )

    assert (
        result.metadata[
            "continuity_algebraic_residual_units"
        ]
        == "1/m^3"
    )

    assert (
        result.metadata["recombination_lag_units"]
        == "1/(m^3 s)"
    )


def test_algebraic_continuity_residual_is_zero_for_linear_profile() -> None:
    solver = GummelDriftDiffusionSolver1D()

    potential = np.zeros(5, dtype=np.float64)
    recombination = np.zeros(5, dtype=np.float64)
    candidate = np.linspace(1.0, 3.0, 5)

    for carrier in ("electron", "hole"):
        diagnostics = solver._continuity_algebraic_diagnostics(
            carrier=carrier,
            potential=potential,
            recombination=recombination,
            spacing=1.0,
            left_value=1.0,
            right_value=3.0,
            candidate=candidate,
        )

        assert diagnostics[
            "maximum_algebraic_residual"
        ] == pytest.approx(0.0, abs=1.0e-14)

        assert diagnostics[
            "relative_algebraic_residual"
        ] == pytest.approx(0.0, abs=1.0e-14)
