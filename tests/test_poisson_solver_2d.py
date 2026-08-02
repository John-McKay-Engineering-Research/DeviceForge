from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from deviceforge import (
    Device,
    Grid,
    Region,
)
from deviceforge.core.boundary import (
    BoundaryCondition,
    BoundaryConditionType,
)
from deviceforge.core.simulation import Simulation
from deviceforge.linalg import LinearSystem
from deviceforge.physics import SILICON
from deviceforge.solvers import (
    PoissonSolver2D,
    SolverProtocol,
)

from deviceforge.linalg import (
    ConjugateGradientSolver,
    IdentityPreconditioner,
    JacobiPreconditioner,
    SparseDirectSolver,
)

def create_zero_boundary_simulation_2d(
    *,
    shape: tuple[int, int] = (5, 5),
    spacing: tuple[float, float] = (
        1.0e-9,
        2.0e-9,
    ),
) -> Simulation:
    """
    Create a uniform-silicon 2D Laplace problem.

    Every outer-boundary node is fixed at zero volts. Boundary masks are
    deliberately non-overlapping so that corner nodes are not assigned by
    more than one BoundaryCondition.
    """

    grid = Grid(
        shape=shape,
        spacing=spacing,
        origin=(0.0, 0.0),
    )

    silicon_mask = np.ones(
        grid.shape,
        dtype=np.bool_,
    )

    silicon_region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=silicon_mask,
    )

    device = Device(
        name="uniform_silicon_2d_device",
        grid=grid,
        regions=(
            silicon_region,
        ),
    )

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0, :] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1, :] = True

    bottom_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    bottom_mask[1:-1, 0] = True

    top_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    top_mask[1:-1, -1] = True

    left_boundary = BoundaryCondition(
        name="left_boundary",
        grid=grid,
        mask=left_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_boundary",
        grid=grid,
        mask=right_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=0.0,
        units="V",
    )

    bottom_boundary = BoundaryCondition(
        name="bottom_boundary",
        grid=grid,
        mask=bottom_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=0.0,
        units="V",
    )

    top_boundary = BoundaryCondition(
        name="top_boundary",
        grid=grid,
        mask=top_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=0.0,
        units="V",
    )

    return Simulation(
        name="zero_boundary_laplace_2d",
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
            bottom_boundary,
            top_boundary,
        ),
        tolerance=1.0e-10,
        max_iterations=10_000,
        initial_potential=0.0,
    )


def create_incomplete_boundary_simulation_2d() -> Simulation:
    """
    Create a 2D simulation without complete perimeter coverage.

    Only the two axis-0 edges are constrained.
    """

    grid = Grid(
        shape=(5, 5),
        spacing=(
            1.0e-9,
            1.0e-9,
        ),
        origin=(0.0, 0.0),
    )

    region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=np.ones(
            grid.shape,
            dtype=np.bool_,
        ),
    )

    device = Device(
        name="incomplete_boundary_device",
        grid=grid,
        regions=(region,),
    )

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0, :] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1, :] = True

    left_boundary = BoundaryCondition(
        name="left_boundary",
        grid=grid,
        mask=left_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_boundary",
        grid=grid,
        mask=right_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=0.0,
        units="V",
    )

    return Simulation(
        name="incomplete_boundary_2d",
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-10,
        max_iterations=10_000,
        initial_potential=0.0,
    )


def test_poisson_solver_2d_satisfies_solver_protocol() -> None:
    solver = PoissonSolver2D()

    assert isinstance(
        solver,
        SolverProtocol,
    )


def test_poisson_solver_2d_defaults() -> None:
    solver = PoissonSolver2D()

    assert solver.name == "poisson_sparse_2d"
    assert solver.backend_name == "scipy"


def test_poisson_solver_2d_normalises_name() -> None:
    solver = PoissonSolver2D(
        name="  custom 2d poisson solver  ",
    )

    assert solver.name == (
        "custom 2d poisson solver"
    )


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_poisson_solver_2d_rejects_empty_name(
    name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        PoissonSolver2D(
            name=name,
        )


def test_poisson_solver_2d_rejects_non_simulation() -> None:
    solver = PoissonSolver2D()

    with pytest.raises(
        TypeError,
        match="requires a Simulation",
    ):
        solver.solve(
            "invalid"
        )


def test_poisson_solver_2d_rejects_one_dimensional_grid(
    simulation: Simulation,
) -> None:
    solver = PoissonSolver2D()

    with pytest.raises(
        ValueError,
        match="two-dimensional",
    ):
        solver.solve(
            simulation
        )

""" removed test for now. ***
@pytest.mark.parametrize(
    "shape",
    [
        (2, 5),
        (5, 2),
        (2, 2),
    ],
)
def test_poisson_solver_2d_requires_three_points_per_axis(
    shape: tuple[int, int],
) -> None:
    simulation = create_zero_boundary_simulation_2d(
        shape=shape,
    )

    with pytest.raises(
        ValueError,
        match="at least three",
    ):
        PoissonSolver2D().solve(
            simulation
        )

"""
def test_poisson_solver_2d_requires_complete_outer_boundary() -> None:
    simulation = (
        create_incomplete_boundary_simulation_2d()
    )

    with pytest.raises(
        ValueError,
        match="every outer-boundary",
    ):
        PoissonSolver2D().solve(
            simulation
        )


def test_linear_index_uses_c_order() -> None:
    number_axis_1 = 7

    assert PoissonSolver2D._linear_index(
        0,
        0,
        number_axis_1,
    ) == 0

    assert PoissonSolver2D._linear_index(
        0,
        6,
        number_axis_1,
    ) == 6

    assert PoissonSolver2D._linear_index(
        1,
        0,
        number_axis_1,
    ) == 7

    assert PoissonSolver2D._linear_index(
        1,
        3,
        number_axis_1,
    ) == 10

    assert PoissonSolver2D._linear_index(
        4,
        6,
        number_axis_1,
    ) == 34


def test_linear_index_matches_numpy_c_order() -> None:
    shape = (5, 7)

    for index_axis_0 in range(
        shape[0]
    ):
        for index_axis_1 in range(
            shape[1]
        ):
            expected = np.ravel_multi_index(
                (
                    index_axis_0,
                    index_axis_1,
                ),
                shape,
                order="C",
            )

            actual = (
                PoissonSolver2D
                ._linear_index(
                    index_axis_0,
                    index_axis_1,
                    shape[1],
                )
            )

            assert actual == expected


def test_harmonic_face_values_axis_zero() -> None:
    node_values = np.asarray(
        [
            [2.0, 4.0],
            [6.0, 8.0],
            [10.0, 12.0],
        ],
        dtype=np.float64,
    )

    face_values = (
        PoissonSolver2D
        ._harmonic_face_values(
            node_values,
            axis=0,
        )
    )

    expected = np.asarray(
        [
            [
                2.0 * 2.0 * 6.0 / (2.0 + 6.0),
                2.0 * 4.0 * 8.0 / (4.0 + 8.0),
            ],
            [
                2.0 * 6.0 * 10.0 / (6.0 + 10.0),
                2.0 * 8.0 * 12.0 / (8.0 + 12.0),
            ],
        ],
        dtype=np.float64,
    )

    assert face_values.shape == (2, 2)

    np.testing.assert_allclose(
        face_values,
        expected,
    )


def test_harmonic_face_values_axis_one() -> None:
    node_values = np.asarray(
        [
            [2.0, 6.0, 10.0],
            [4.0, 8.0, 12.0],
        ],
        dtype=np.float64,
    )

    face_values = (
        PoissonSolver2D
        ._harmonic_face_values(
            node_values,
            axis=1,
        )
    )

    expected = np.asarray(
        [
            [
                2.0 * 2.0 * 6.0 / (2.0 + 6.0),
                2.0 * 6.0 * 10.0 / (6.0 + 10.0),
            ],
            [
                2.0 * 4.0 * 8.0 / (4.0 + 8.0),
                2.0 * 8.0 * 12.0 / (8.0 + 12.0),
            ],
        ],
        dtype=np.float64,
    )

    assert face_values.shape == (2, 2)

    np.testing.assert_allclose(
        face_values,
        expected,
    )


def test_harmonic_face_values_reject_invalid_axis() -> None:
    with pytest.raises(
        ValueError,
        match="either 0 or 1",
    ):
        PoissonSolver2D._harmonic_face_values(
            np.ones(
                (3, 3),
                dtype=np.float64,
            ),
            axis=2,
        )


def test_harmonic_face_values_reject_non_2d_array() -> None:
    with pytest.raises(
        ValueError,
        match="two-dimensional array",
    ):
        PoissonSolver2D._harmonic_face_values(
            np.ones(
                3,
                dtype=np.float64,
            ),
            axis=0,
        )


def test_poisson_solver_2d_assembles_sparse_system() -> None:
    simulation = (
        create_zero_boundary_simulation_2d()
    )

    system = PoissonSolver2D()._assemble_system(
        simulation
    )

    expected_unknowns = (
        simulation.grid.number_of_points
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
        expected_unknowns,
        expected_unknowns,
    )

    assert system.right_hand_side.shape == (
        expected_unknowns,
    )


def test_poisson_solver_2d_interior_row_has_five_entries() -> None:
    simulation = (
        create_zero_boundary_simulation_2d()
    )

    system = PoissonSolver2D()._assemble_system(
        simulation
    )

    centre_index = (
        PoissonSolver2D._linear_index(
            2,
            2,
            simulation.grid.shape[1],
        )
    )

    centre_row = system.matrix.getrow(
        centre_index
    )

    assert centre_row.nnz == 5


def test_poisson_solver_2d_boundary_row_is_identity() -> None:
    simulation = (
        create_zero_boundary_simulation_2d()
    )

    system = PoissonSolver2D()._assemble_system(
        simulation
    )

    boundary_index = (
        PoissonSolver2D._linear_index(
            0,
            2,
            simulation.grid.shape[1],
        )
    )

    boundary_row = system.matrix.getrow(
        boundary_index
    )

    assert boundary_row.nnz == 1

    assert boundary_row[
        0,
        boundary_index,
    ] == pytest.approx(1.0)


def test_poisson_solver_2d_matrix_is_symmetric() -> None:
    simulation = (
        create_zero_boundary_simulation_2d()
    )

    system = PoissonSolver2D()._assemble_system(
        simulation
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


def test_poisson_solver_2d_matrix_has_positive_diagonal() -> None:
    simulation = (
        create_zero_boundary_simulation_2d()
    )

    system = PoissonSolver2D()._assemble_system(
        simulation
    )

    assert np.all(
        system.matrix.diagonal() > 0.0
    )


def test_poisson_solver_2d_matrix_is_positive_definite() -> None:
    simulation = (
        create_zero_boundary_simulation_2d()
    )

    system = PoissonSolver2D()._assemble_system(
        simulation
    )

    eigenvalues = np.linalg.eigvalsh(
        system.matrix.toarray()
    )

    assert np.all(
        eigenvalues > 0.0
    )


def test_poisson_solver_2d_returns_converged_result() -> None:
    simulation = (
        create_zero_boundary_simulation_2d()
    )

    result = PoissonSolver2D().solve(
        simulation
    )

    assert result.converged
    assert result.iterations == 1
    assert result.final_residual is not None

    assert result.final_residual <= (
        simulation.tolerance
    )


def test_poisson_solver_2d_returns_two_dimensional_field() -> None:
    simulation = (
        create_zero_boundary_simulation_2d()
    )

    result = PoissonSolver2D().solve(
        simulation
    )

    potential = result.potential

    assert potential.name == (
        "electrostatic_potential"
    )

    assert potential.units == "V"

    assert potential.grid is (
        simulation.grid
    )

    assert potential.values.shape == (
        simulation.grid.shape
    )


def test_poisson_solver_2d_zero_boundary_solution_is_zero() -> None:
    simulation = (
        create_zero_boundary_simulation_2d()
    )

    result = PoissonSolver2D().solve(
        simulation
    )

    np.testing.assert_allclose(
        result.potential.values,
        0.0,
        atol=1.0e-14,
    )


def test_poisson_solver_2d_records_metadata() -> None:
    simulation = (
        create_zero_boundary_simulation_2d()
    )

    result = PoissonSolver2D().solve(
        simulation
    )

    assert result.solver_name == (
        "poisson_sparse_2d"
    )

    assert result.backend_name == "scipy"

    assert result.metadata[
        "equation"
    ] == "laplace"

    assert result.metadata[
        "spatial_dimension"
    ] == 2

    assert result.metadata[
        "discretisation"
    ] == (
        "conservative_five_point_"
        "finite_volume"
    )

    assert result.metadata[
        "interface_averaging"
    ] == "harmonic"

    assert result.metadata[
        "linear_index_order"
    ] == "C"

    assert result.metadata[
        "linear_solver"
    ] == "sparse_direct"

    assert result.metadata[
        "linear_solver_backend"
    ] == "scipy"

    assert result.metadata[
        "matrix_storage"
    ] == "csr"

    assert result.metadata[
        "matrix_shape"
    ] == (
        simulation.grid.number_of_points,
        simulation.grid.number_of_points,
    )

    assert result.metadata[
        "grid_shape"
    ] == simulation.grid.shape

    assert result.metadata[
        "grid_spacing_metres"
    ] == simulation.grid.spacing

    assert result.metadata[
        "number_of_grid_points"
    ] == simulation.grid.number_of_points

    assert result.metadata[
        "charge_density_present"
    ] is False

# helper functions

def create_linear_potential_simulation_2d(
    *,
    shape: tuple[int, int] = (11, 9),
    spacing: tuple[float, float] = (
        1.0e-9,
        2.0e-9,
    ),
) -> tuple[Simulation, np.ndarray]:
    """
    Create a 2D Laplace problem with the exact solution

        phi(x, y) = x / Lx.
    """

    grid = Grid(
        shape=shape,
        spacing=spacing,
        origin=(0.0, 0.0),
    )

    region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=np.ones(
            grid.shape,
            dtype=np.bool_,
        ),
    )

    device = Device(
        name="linear_potential_device_2d",
        grid=grid,
        regions=(region,),
    )

    x_coordinates = grid.coordinates(0)

    normalised_x = (
        x_coordinates
        - x_coordinates[0]
    ) / (
        x_coordinates[-1]
        - x_coordinates[0]
    )

    expected_potential = np.broadcast_to(
        normalised_x[:, None],
        grid.shape,
    ).copy()

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    left_mask[0, :] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    right_mask[-1, :] = True

    bottom_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    bottom_mask[1:-1, 0] = True

    top_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    top_mask[1:-1, -1] = True

    left_boundary = BoundaryCondition(
        name="left_boundary",
        grid=grid,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_boundary",
        grid=grid,
        mask=right_mask,
        condition_type="dirichlet",
        value=1.0,
        units="V",
    )

    bottom_boundary = BoundaryCondition(
        name="bottom_profile",
        grid=grid,
        mask=bottom_mask,
        condition_type="dirichlet",
        value=expected_potential,
        units="V",
    )

    top_boundary = BoundaryCondition(
        name="top_profile",
        grid=grid,
        mask=top_mask,
        condition_type="dirichlet",
        value=expected_potential,
        units="V",
    )

    simulation = Simulation(
        name="linear_laplace_2d",
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
            bottom_boundary,
            top_boundary,
        ),
        tolerance=1.0e-10,
        max_iterations=10_000,
        initial_potential=0.0,
    )

    return simulation, expected_potential

# analytical test

def test_poisson_solver_2d_matches_linear_analytical_solution() -> None:
    simulation, expected = (
        create_linear_potential_simulation_2d()
    )

    result = PoissonSolver2D().solve(
        simulation
    )

    assert result.converged

    np.testing.assert_allclose(
        result.potential.values,
        expected,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

# axis-orientation checks

def test_poisson_solver_2d_linear_solution_varies_along_axis_zero() -> None:
    simulation, _ = (
        create_linear_potential_simulation_2d()
    )

    result = PoissonSolver2D().solve(
        simulation
    )

    potential = result.potential.values

    np.testing.assert_allclose(
        potential[:, 0],
        potential[:, -1],
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        potential[0, :],
        0.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        potential[-1, :],
        1.0,
        atol=1.0e-12,
    )

# explicit square spacing test

@pytest.mark.parametrize(
    "spacing",
    [
        (1.0e-9, 1.0e-9),
        (1.0e-9, 2.0e-9),
        (3.0e-9, 1.0e-9),
    ],
)
def test_poisson_solver_2d_linear_solution_with_unequal_spacing(
    spacing: tuple[float, float],
) -> None:
    simulation, expected = (
        create_linear_potential_simulation_2d(
            spacing=spacing,
        )
    )

    result = PoissonSolver2D().solve(
        simulation
    )

    np.testing.assert_allclose(
        result.potential.values,
        expected,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

# shared solver helper

def solve_linear_case_with_2d_backends(
    simulation: Simulation,
):
    """Solve one 2D problem with direct and CG backends."""

    direct_result = PoissonSolver2D(
        linear_solver=SparseDirectSolver(),
        name="poisson_sparse_direct_2d",
    ).solve(
        simulation
    )

    identity_cg_result = PoissonSolver2D(
        linear_solver=ConjugateGradientSolver(
            preconditioner=IdentityPreconditioner(),
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
            max_iterations=100_000,
            name="cg_identity",
        ),
        name="poisson_identity_cg_2d",
    ).solve(
        simulation
    )

    jacobi_cg_result = PoissonSolver2D(
        linear_solver=ConjugateGradientSolver(
            preconditioner=JacobiPreconditioner(),
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
            max_iterations=100_000,
            name="cg_jacobi",
        ),
        name="poisson_jacobi_cg_2d",
    ).solve(
        simulation
    )

    return (
        direct_result,
        identity_cg_result,
        jacobi_cg_result,
    )

# direct verus identity-CG parity test

def test_poisson_solver_2d_identity_cg_matches_sparse_direct() -> None:
    simulation, expected = (
        create_linear_potential_simulation_2d(
            shape=(21, 17),
            spacing=(
                1.0e-9,
                2.0e-9,
            ),
        )
    )

    (
        direct_result,
        identity_cg_result,
        _,
    ) = solve_linear_case_with_2d_backends(
        simulation
    )

    assert direct_result.converged
    assert identity_cg_result.converged

    np.testing.assert_allclose(
        identity_cg_result.potential.values,
        direct_result.potential.values,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        identity_cg_result.potential.values,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

# direct versus Jacobi-CG parity test

def test_poisson_solver_2d_jacobi_cg_matches_sparse_direct() -> None:
    simulation, expected = (
        create_linear_potential_simulation_2d(
            shape=(21, 17),
            spacing=(
                1.0e-9,
                2.0e-9,
            ),
        )
    )

    (
        direct_result,
        _,
        jacobi_cg_result,
    ) = solve_linear_case_with_2d_backends(
        simulation
    )

    assert direct_result.converged
    assert jacobi_cg_result.converged

    np.testing.assert_allclose(
        jacobi_cg_result.potential.values,
        direct_result.potential.values,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        jacobi_cg_result.potential.values,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    # CG diagnostic validation

@pytest.mark.parametrize(
    (
        "preconditioner",
        "solver_name",
        "expected_preconditioner",
    ),
    [
        (
            IdentityPreconditioner(),
            "cg_identity",
            "identity",
        ),
        (
            JacobiPreconditioner(),
            "cg_jacobi",
            "jacobi",
        ),
    ],
)
def test_poisson_solver_2d_cg_records_iterative_diagnostics(
    preconditioner,
    solver_name: str,
    expected_preconditioner: str,
) -> None:
    simulation, _ = (
        create_linear_potential_simulation_2d(
            shape=(21, 17),
        )
    )

    result = PoissonSolver2D(
        linear_solver=ConjugateGradientSolver(
            preconditioner=preconditioner,
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
            max_iterations=100_000,
            name=solver_name,
        ),
    ).solve(
        simulation
    )

    assert result.converged
    assert result.iterations > 0

    assert result.residual_history.size == (
        result.iterations
    )

    assert result.final_residual is not None

    assert result.metadata[
        "linear_solver_converged"
    ] is True

    assert result.metadata[
        "linear_solver_iterations"
    ] == result.iterations

    assert result.metadata[
        "linear_solver_termination_reason"
    ] == "convergence_tolerance_satisfied"

    linear_metadata = result.metadata[
        "linear_solver_metadata"
    ]

    assert linear_metadata[
        "preconditioner"
    ] == expected_preconditioner

    assert linear_metadata[
        "preconditioner_backend"
    ] == "scipy"

# residual validation

@pytest.mark.parametrize(
    (
        "preconditioner",
        "solver_name",
        "expected_preconditioner",
    ),
    [
        (
            IdentityPreconditioner(),
            "cg_identity",
            "identity",
        ),
        (
            JacobiPreconditioner(),
            "cg_jacobi",
            "jacobi",
        ),
    ],
)
def test_poisson_solver_2d_cg_records_iterative_diagnostics(
    preconditioner,
    solver_name: str,
    expected_preconditioner: str,
) -> None:
    simulation, _ = (
        create_linear_potential_simulation_2d(
            shape=(21, 17),
        )
    )

    result = PoissonSolver2D(
        linear_solver=ConjugateGradientSolver(
            preconditioner=preconditioner,
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-14,
            max_iterations=100_000,
            name=solver_name,
        ),
    ).solve(
        simulation
    )

    assert result.converged
    assert result.iterations > 0

    assert result.residual_history.size == (
        result.iterations
    )

    assert result.final_residual is not None

    assert result.metadata[
        "linear_solver_converged"
    ] is True

    assert result.metadata[
        "linear_solver_iterations"
    ] == result.iterations

    assert result.metadata[
        "linear_solver_termination_reason"
    ] == "convergence_tolerance_satisfied"

    linear_metadata = result.metadata[
        "linear_solver_metadata"
    ]

    assert linear_metadata[
        "preconditioner"
    ] == expected_preconditioner

    assert linear_metadata[
        "preconditioner_backend"
    ] == "scipy"

# compare both CG solutions

def test_poisson_solver_2d_identity_and_jacobi_cg_agree() -> None:
    simulation, _ = (
        create_linear_potential_simulation_2d(
            shape=(21, 17),
        )
    )

    (
        _,
        identity_result,
        jacobi_result,
    ) = solve_linear_case_with_2d_backends(
        simulation
    )

    np.testing.assert_allclose(
        identity_result.potential.values,
        jacobi_result.potential.values,
        rtol=1.0e-10,
        atol=1.0e-12,
    )