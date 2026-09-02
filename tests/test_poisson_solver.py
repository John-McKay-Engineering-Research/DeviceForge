from __future__ import annotations

import numpy as np
import pytest

from deviceforge.core.boundary import (
    BoundaryCondition,
    BoundaryConditionType,
)
from deviceforge.core.simulation import Simulation
from deviceforge.solvers import (
    PoissonSolver,
    SolverProtocol,
)

from deviceforge.core import Field

from deviceforge import Device, Grid, Region
from deviceforge.physics import SILICON, SILICON_DIOXIDE

from scipy.sparse import csr_matrix
from deviceforge.linalg import LinearSystem
from deviceforge.linalg import ConjugateGradientSolver

def create_linear_simulation(
    simulation: Simulation,
    *,
    left_voltage: float = 0.0,
    right_voltage: float = 1.0,
) -> Simulation:
    """
    Create a one-dimensional Laplace problem with fixed endpoint voltages.
    """

    grid = simulation.grid

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1] = True

    left_boundary = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=left_voltage,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=right_voltage,
        units="V",
    )

    return Simulation(
        device=simulation.device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-8,
        max_iterations=500,
        initial_potential=0.0,
        name="linear_poisson_test",
    )

# added additional help function below

def create_charged_simulation(
    simulation: Simulation,
    *,
    charge_density_value: float,
) -> Simulation:
    """
    Create a uniformly charged one-dimensional Poisson problem.

    Both endpoint potentials are fixed to zero.
    """

    grid = simulation.grid

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1] = True

    left_boundary = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=0.0,
        units="V",
    )

    charge_density = Field.full(
        name="charge_density",
        units="C/m^3",
        grid=grid,
        fill_value=charge_density_value,
    )

    return Simulation(
        device=simulation.device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        charge_density=charge_density,
        tolerance=1.0e-8,
        max_iterations=500,
        initial_potential=0.0,
        name="uniform_charge_poisson_test",
    )

def expected_linear_potential_from_slope(
    simulation: Simulation,
    *,
    reference_position: float,
    reference_potential: float,
    slope: float,
) -> np.ndarray:
    """
    Return the analytical linear potential

        phi(x) = phi_ref + slope * (x - x_ref).
    """

    coordinates = (
        simulation.grid.coordinates(0)
    )

    return (
        reference_potential
        + slope
        * (
            coordinates
            - reference_position
        )
    )

def create_mixed_boundary_simulation(
    simulation: Simulation,
    *,
    left_condition_type: BoundaryConditionType,
    left_value: float,
    left_units: str,
    right_condition_type: BoundaryConditionType,
    right_value: float,
    right_units: str,
) -> Simulation:
    """
    Create a one-dimensional Poisson problem with independently
    configured endpoint boundary conditions.
    """

    grid = simulation.grid

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1] = True

    left_boundary = BoundaryCondition(
        name="left_boundary",
        grid=grid,
        mask=left_mask,
        condition_type=left_condition_type,
        value=left_value,
        units=left_units,
    )

    right_boundary = BoundaryCondition(
        name="right_boundary",
        grid=grid,
        mask=right_mask,
        condition_type=right_condition_type,
        value=right_value,
        units=right_units,
    )

    return Simulation(
        device=simulation.device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-8,
        max_iterations=500,
        initial_potential=0.0,
        name="mixed_boundary_poisson_test",
    )

def test_poisson_solver_satisfies_solver_protocol() -> None:
    solver = PoissonSolver()

    assert isinstance(
        solver,
        SolverProtocol,
    )


def test_poisson_solver_rejects_non_simulation() -> None:
    solver = PoissonSolver()

    with pytest.raises(
        TypeError,
        match="requires a Simulation",
    ):
        solver.solve("invalid")


def test_poisson_solver_returns_converged_result(
    simulation,
) -> None:
    linear_simulation = create_linear_simulation(
        simulation
    )

    solver = PoissonSolver()

    result = solver.solve(
        linear_simulation
    )

    assert result.converged
    assert result.iterations == 1

    assert result.final_residual is not None
    assert result.final_residual <= (
        linear_simulation.tolerance
    )


def test_poisson_solver_returns_potential_field(
    simulation,
) -> None:
    linear_simulation = create_linear_simulation(
        simulation
    )

    solver = PoissonSolver()

    result = solver.solve(
        linear_simulation
    )

    assert (
        "electrostatic_potential"
        in result.fields
    )

    potential = result.potential

    assert potential.name == "electrostatic_potential"
    assert potential.units == "V"
    assert potential.grid is linear_simulation.grid
    assert potential.values.shape == (
        linear_simulation.grid.shape
    )


def test_poisson_solver_matches_linear_solution(
    simulation,
) -> None:
    left_voltage = 0.0
    right_voltage = 1.0

    linear_simulation = create_linear_simulation(
        simulation,
        left_voltage=left_voltage,
        right_voltage=right_voltage,
    )

    solver = PoissonSolver()

    result = solver.solve(
        linear_simulation
    )

    coordinates = (
        linear_simulation.grid.coordinates(0)
    )

    minimum_coordinate = coordinates[0]
    maximum_coordinate = coordinates[-1]

    expected_potential = (
        left_voltage
        + (
            right_voltage
            - left_voltage
        )
        * (
            coordinates
            - minimum_coordinate
        )
        / (
            maximum_coordinate
            - minimum_coordinate
        )
    )

    assert result.potential.values == pytest.approx(
        expected_potential,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_poisson_solver_applies_endpoint_voltages(
    simulation,
) -> None:
    linear_simulation = create_linear_simulation(
        simulation,
        left_voltage=-0.5,
        right_voltage=1.5,
    )

    solver = PoissonSolver()

    result = solver.solve(
        linear_simulation
    )

    assert result.potential.values[0] == pytest.approx(
        -0.5
    )

    assert result.potential.values[-1] == pytest.approx(
        1.5
    )


def test_poisson_solver_records_metadata(
    simulation,
) -> None:
    linear_simulation = create_linear_simulation(
        simulation
    )

    solver = PoissonSolver()

    result = solver.solve(
        linear_simulation
    )

    # assert result.solver_name == "poisson_direct_1d"
    # assert result.backend_name == "numpy"#

    # sparse matrix addtions
    assert result.solver_name == "poisson_sparse_1d"
    assert result.backend_name == "scipy"

    assert result.metadata["equation"] == "laplace"
    assert result.metadata["spatial_dimension"] == 1

    assert result.metadata["discretisation"] == (
        "conservative_central_finite_difference"
    )

    assert result.metadata["interface_averaging"] == "harmonic"

    assert result.metadata[
               "relative_permittivity_min"
           ] == pytest.approx(11.7)

    assert result.metadata[
               "relative_permittivity_max"
           ] == pytest.approx(11.7)

    # assert result.metadata["linear_solver"] == (
        # "numpy.linalg.solve"
    # )

    # sparse matrix tests ***
    assert result.metadata["linear_solver"] == (
        "sparse_direct"
    )

    assert result.metadata[
               "linear_solver_backend"
           ] == "scipy"

    assert result.metadata[
        "matrix_storage"
    ] == "csr"

    assert result.metadata[
        "linear_solver_converged"
    ] is True

    assert result.metadata[
        "linear_solver_iterations"
    ] == 1

    assert result.metadata[
        "linear_solver_final_residual"
    ] is not None

    assert result.metadata[
        "linear_solver_termination_reason"
    ] == "direct_solve_completed"

    assert isinstance(
        result.metadata["linear_solver_metadata"],
        dict,
    )
    assert result.metadata["charge_density_present"] is False
    assert (
            result.metadata["charge_density_min_c_per_m3"]
            == pytest.approx(0.0)
    )
    assert (
            result.metadata["charge_density_max_c_per_m3"]
            == pytest.approx(0.0)
    )

    assert result.metadata["number_of_grid_points"] == (
        linear_simulation.grid.number_of_points
    )

# added additional test
# updated permittivity
def test_poisson_solver_matches_uniform_charge_solution(
    simulation,
) -> None:
    charge_density_value = 1.0e5
    vacuum_permittivity = 8.8541878128e-12

    charged_simulation = create_charged_simulation(
        simulation,
        charge_density_value=charge_density_value,
    )

    relative_permittivity = (
        charged_simulation
        .device
        .relative_permittivity_field()
        .values[0]
    )

    permittivity = (
        vacuum_permittivity
        * relative_permittivity
    )

    result = PoissonSolver().solve(
        charged_simulation
    )

    coordinates = charged_simulation.grid.coordinates(0)

    x = coordinates - coordinates[0]
    length = coordinates[-1] - coordinates[0]

    expected_potential = (
        charge_density_value
        / (2.0 * permittivity)
        * x
        * (length - x)
    )

    np.testing.assert_allclose(
        result.potential.values,
        expected_potential,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

    assert result.converged
    assert result.metadata["equation"] == "poisson"
    assert result.metadata["charge_density_present"] is True


def test_positive_charge_produces_positive_internal_potential(
    simulation,
) -> None:
    charged_simulation = create_charged_simulation(
        simulation,
        charge_density_value=1.0e5,
    )

    result = PoissonSolver().solve(
        charged_simulation
    )

    assert result.potential.values[0] == pytest.approx(0.0)
    assert result.potential.values[-1] == pytest.approx(0.0)
    assert np.all(result.potential.values[1:-1] > 0.0)

"""
def test_poisson_solver_rejects_neumann_boundary(
    simulation,
) -> None:
    grid = simulation.grid

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1] = True

    left_boundary = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_flux",
        grid=grid,
        mask=right_mask,
        condition_type=BoundaryConditionType.NEUMANN,
        value=0.0,
        units="V/m",
    )

    neumann_simulation = Simulation(
        device=simulation.device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-8,
        max_iterations=500,
        initial_potential=0.0,
        name="unsupported_neumann_test",
    )

    solver = PoissonSolver()

    with pytest.raises(
        ValueError,
        match="only Dirichlet",
    ):
        solver.solve(
            neumann_simulation
        )
"""

def test_poisson_solver_supports_left_dirichlet_right_neumann(
    simulation,
) -> None:
    left_potential = 0.25
    slope = 2.0e6

    mixed_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            left_value=left_potential,
            left_units="V",
            right_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            right_value=slope,
            right_units="V/m",
        )
    )

    result = PoissonSolver().solve(
        mixed_simulation
    )

    coordinates = (
        mixed_simulation
        .grid
        .coordinates(0)
    )

    expected = (
        expected_linear_potential_from_slope(
            mixed_simulation,
            reference_position=coordinates[0],
            reference_potential=left_potential,
            slope=slope,
        )
    )

    assert result.converged

    np.testing.assert_allclose(
        result.potential.values,
        expected,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

def test_poisson_solver_requires_left_endpoint_condition(
    simulation,
) -> None:
    grid = simulation.grid

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1] = True

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=1.0,
        units="V",
    )

    incomplete_simulation = Simulation(
        device=simulation.device,
        boundary_conditions=(
            right_boundary,
        ),
        tolerance=1.0e-8,
        max_iterations=500,
        initial_potential=0.0,
        name="missing_left_boundary_test",
    )

    solver = PoissonSolver()

    with pytest.raises(
        ValueError,
        match="first grid point",
    ):
        solver.solve(
            incomplete_simulation
        )


def test_poisson_solver_requires_right_endpoint_condition(
    simulation,
) -> None:
    grid = simulation.grid

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0] = True

    left_boundary = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=0.0,
        units="V",
    )

    incomplete_simulation = Simulation(
        device=simulation.device,
        boundary_conditions=(
            left_boundary,
        ),
        tolerance=1.0e-8,
        max_iterations=500,
        initial_potential=0.0,
        name="missing_right_boundary_test",
    )

    solver = PoissonSolver()

    with pytest.raises(
        ValueError,
        match="final grid point",
    ):
        solver.solve(
            incomplete_simulation
        )

# added additional harmonic-mean unit tests

def test_harmonic_face_values_for_uniform_material() -> None:
    node_values = np.asarray(
        [11.7, 11.7, 11.7],
        dtype=np.float64,
    )

    face_values = PoissonSolver._harmonic_face_values(
        node_values
    )

    np.testing.assert_allclose(
        face_values,
        [11.7, 11.7],
    )


def test_harmonic_face_values_at_material_interface() -> None:
    node_values = np.asarray(
        [3.9, 11.7],
        dtype=np.float64,
    )

    face_values = PoissonSolver._harmonic_face_values(
        node_values
    )

    expected = (
        2.0
        * 3.9
        * 11.7
        / (3.9 + 11.7)
    )

    assert face_values[0] == pytest.approx(
        expected
    )

# added two-material interface test

def test_poisson_solver_preserves_dielectric_flux_at_interface() -> None:
    grid = Grid(
        shape=(21,),
        spacing=(1.0e-9,),
    )

    oxide_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    oxide_mask[:10] = True

    silicon_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    silicon_mask[10:] = True

    oxide_region = Region(
        name="oxide",
        grid=grid,
        material=SILICON_DIOXIDE,
        mask=oxide_mask,
    )

    silicon_region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=silicon_mask,
    )

    device = Device(
        name="oxide_silicon_stack",
        grid=grid,
        regions=(
            oxide_region,
            silicon_region,
        ),
    )

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1] = True

    left_boundary = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=1.0,
        units="V",
    )

    simulation = Simulation(
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-10,
        name="dielectric_interface_test",
    )

    result = PoissonSolver().solve(
        simulation
    )

    relative_permittivity = (
        device.relative_permittivity_field().values
    )

    face_permittivity = (
        PoissonSolver._harmonic_face_values(
            relative_permittivity
        )
    )

    potential_difference = np.diff(
        result.potential.values
    )

    relative_displacement_flux = (
        face_permittivity
        * potential_difference
        / grid.spacing[0]
    )

    np.testing.assert_allclose(
        relative_displacement_flux,
        np.full_like(
            relative_displacement_flux,
            relative_displacement_flux[0],
        ),
        rtol=1.0e-11,
        atol=1.0e-6,
    )

    oxide_field = np.mean(
        potential_difference[1:8]
    ) / grid.spacing[0]

    silicon_field = np.mean(
        potential_difference[11:18]
    ) / grid.spacing[0]

    expected_field_ratio = (
        SILICON.relative_permittivity
        / SILICON_DIOXIDE.relative_permittivity
    )

    assert (
        oxide_field / silicon_field
        == pytest.approx(
            expected_field_ratio,
            rel=1.0e-11,
        )
    )

# sparse matrix tests

def test_poisson_solver_assembles_sparse_system(
    simulation,
) -> None:
    solver = PoissonSolver()

    system = solver._assemble_system(
        simulation
    )

    assert isinstance(
        system,
        LinearSystem,
    )

    assert system.is_sparse
    assert isinstance(
        system.matrix,
        csr_matrix,
    )

    assert system.shape == (
        simulation.grid.number_of_points,
        simulation.grid.number_of_points,
    )


def test_sparse_poisson_system_is_tridiagonal(
    simulation,
) -> None:
    system = PoissonSolver()._assemble_system(
        simulation
    )

    rows, columns = system.matrix.nonzero()

    assert np.all(
        np.abs(rows - columns) <= 1
    )

# additional tests

def test_poisson_matrix_is_symmetric(
    dielectric_stack_simulation,
) -> None:
    system = PoissonSolver()._assemble_system(
        dielectric_stack_simulation
    )

    difference = (
        system.matrix
        - system.matrix.T
    )

    np.testing.assert_allclose(
        difference.toarray(),
        0.0,
        atol=1.0e-14,
    )

def test_poisson_matrix_has_positive_diagonal(
    dielectric_stack_simulation,
) -> None:
    system = PoissonSolver()._assemble_system(
        dielectric_stack_simulation
    )

    assert np.all(
        system.matrix.diagonal() > 0.0
    )

def test_poisson_matrix_is_positive_definite(
    dielectric_stack_simulation,
) -> None:
    system = PoissonSolver()._assemble_system(
        dielectric_stack_simulation
    )

    eigenvalues = np.linalg.eigvalsh(
        system.matrix.toarray()
    )

    assert np.all(eigenvalues > 0.0)


def test_poisson_solver_supports_left_neumann_right_dirichlet(
    simulation,
) -> None:
    right_potential = 1.25
    slope = 2.0e6

    mixed_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            left_value=-slope,
            left_units="V/m",
            right_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            right_value=right_potential,
            right_units="V",
        )
    )

    coordinates = (
        mixed_simulation
        .grid
        .coordinates(0)
    )

    expected = (
        expected_linear_potential_from_slope(
            mixed_simulation,
            reference_position=coordinates[-1],
            reference_potential=right_potential,
            slope=slope,
        )
    )

    result = PoissonSolver().solve(
        mixed_simulation
    )

    assert result.converged

    np.testing.assert_allclose(
        result.potential.values,
        expected,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

def test_left_neumann_uses_outward_normal_sign(
    simulation,
) -> None:
    outward_derivative = 3.0e6

    mixed_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            left_value=outward_derivative,
            left_units="V/m",
            right_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            right_value=0.0,
            right_units="V",
        )
    )

    result = PoissonSolver().solve(
        mixed_simulation
    )

    spacing = (
        mixed_simulation.grid.spacing[0]
    )

    numerical_slope = (
        result.potential.values[1]
        - result.potential.values[0]
    ) / spacing

    assert numerical_slope == pytest.approx(
        -outward_derivative,
        rel=1.0e-11,
        abs=1.0e-6,
    )

def test_right_neumann_uses_outward_normal_sign(
    simulation,
) -> None:
    outward_derivative = 3.0e6

    mixed_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            left_value=0.0,
            left_units="V",
            right_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            right_value=outward_derivative,
            right_units="V/m",
        )
    )

    result = PoissonSolver().solve(
        mixed_simulation
    )

    spacing = (
        mixed_simulation.grid.spacing[0]
    )

    numerical_slope = (
        result.potential.values[-1]
        - result.potential.values[-2]
    ) / spacing

    assert numerical_slope == pytest.approx(
        outward_derivative,
        rel=1.0e-11,
        abs=1.0e-6,
    )

def test_poisson_solver_rejects_pure_neumann_problem(
    simulation,
) -> None:
    pure_neumann_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            left_value=0.0,
            left_units="V/m",
            right_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            right_value=0.0,
            right_units="V/m",
        )
    )

    with pytest.raises(
        ValueError,
        match="pure Neumann",
    ):
        PoissonSolver().solve(
            pure_neumann_simulation
        )


def test_poisson_solver_rejects_invalid_neumann_units(
    simulation,
) -> None:
    invalid_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            left_value=0.0,
            left_units="V",
            right_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            right_value=0.0,
            right_units="V",
        )
    )

    with pytest.raises(
        ValueError,
        match="Neumann.*V/m",
    ):
        PoissonSolver().solve(
            invalid_simulation
        )

def test_poisson_solver_rejects_invalid_dirichlet_units(
    simulation,
) -> None:
    invalid_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            left_value=0.0,
            left_units="V/m",
            right_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            right_value=0.0,
            right_units="V/m",
        )
    )

    with pytest.raises(
        ValueError,
        match="Dirichlet.*V",
    ):
        PoissonSolver().solve(
            invalid_simulation
        )

def test_mixed_left_dirichlet_right_neumann_matrix_is_symmetric(
    simulation,
) -> None:
    mixed_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            left_value=0.0,
            left_units="V",
            right_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            right_value=2.0e6,
            right_units="V/m",
        )
    )

    solver = PoissonSolver()

    system = solver._assemble_system(
        mixed_simulation
    )

    matrix = system.matrix.toarray()

    np.testing.assert_allclose(
        matrix,
        matrix.T,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_mixed_left_dirichlet_right_neumann_matrix_is_positive_definite(
    simulation,
) -> None:
    mixed_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            left_value=0.0,
            left_units="V",
            right_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            right_value=2.0e6,
            right_units="V/m",
        )
    )

    solver = PoissonSolver()

    system = solver._assemble_system(
        mixed_simulation
    )

    matrix = system.matrix.toarray()

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    assert np.all(
        eigenvalues > 0.0
    )


def test_mixed_left_neumann_right_dirichlet_matrix_is_symmetric(
    simulation,
) -> None:
    mixed_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            left_value=-2.0e6,
            left_units="V/m",
            right_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            right_value=1.0,
            right_units="V",
        )
    )

    solver = PoissonSolver()

    system = solver._assemble_system(
        mixed_simulation
    )

    matrix = system.matrix.toarray()

    np.testing.assert_allclose(
        matrix,
        matrix.T,
        rtol=0.0,
        atol=1.0e-14,
    )


def test_mixed_left_neumann_right_dirichlet_matrix_is_positive_definite(
    simulation,
) -> None:
    mixed_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            left_value=-2.0e6,
            left_units="V/m",
            right_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            right_value=1.0,
            right_units="V",
        )
    )

    solver = PoissonSolver()

    system = solver._assemble_system(
        mixed_simulation
    )

    matrix = system.matrix.toarray()

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    assert np.all(
        eigenvalues > 0.0
    )

def test_poisson_solver_supports_left_dirichlet_right_neumann_with_cg(
    simulation,
) -> None:
    left_potential = 0.25
    slope = 2.0e6

    mixed_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            left_value=left_potential,
            left_units="V",
            right_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            right_value=slope,
            right_units="V/m",
        )
    )

    coordinates = (
        mixed_simulation
        .grid
        .coordinates(0)
    )

    expected = (
        expected_linear_potential_from_slope(
            mixed_simulation,
            reference_position=coordinates[0],
            reference_potential=left_potential,
            slope=slope,
        )
    )

    solver = PoissonSolver(
        linear_solver=ConjugateGradientSolver()
    )

    result = solver.solve(
        mixed_simulation
    )

    assert result.converged

    np.testing.assert_allclose(
        result.potential.values,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_poisson_solver_supports_left_neumann_right_dirichlet_with_cg(
    simulation,
) -> None:
    right_potential = 1.25
    slope = 2.0e6

    mixed_simulation = (
        create_mixed_boundary_simulation(
            simulation,
            left_condition_type=(
                BoundaryConditionType.NEUMANN
            ),
            left_value=-slope,
            left_units="V/m",
            right_condition_type=(
                BoundaryConditionType.DIRICHLET
            ),
            right_value=right_potential,
            right_units="V",
        )
    )

    coordinates = (
        mixed_simulation
        .grid
        .coordinates(0)
    )

    expected = (
        expected_linear_potential_from_slope(
            mixed_simulation,
            reference_position=coordinates[-1],
            reference_potential=right_potential,
            slope=slope,
        )
    )

    solver = PoissonSolver(
        linear_solver=ConjugateGradientSolver()
    )

    result = solver.solve(
        mixed_simulation
    )

    assert result.converged

    np.testing.assert_allclose(
        result.potential.values,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
