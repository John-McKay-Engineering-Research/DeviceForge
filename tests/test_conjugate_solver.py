from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from deviceforge.linalg import (
    ConjugateGradientSolver,
    LinearSolveResult,
    LinearSolverProtocol,
    LinearSystem,
    SparseDirectSolver,
)


def create_spd_system() -> LinearSystem:
    """Return a small symmetric positive-definite system."""

    matrix = csr_matrix(
        [
            [4.0, -1.0, 0.0],
            [-1.0, 4.0, -1.0],
            [0.0, -1.0, 3.0],
        ],
        dtype=np.float64,
    )

    return LinearSystem(
        matrix=matrix,
        right_hand_side=np.asarray(
            [15.0, 10.0, 10.0],
            dtype=np.float64,
        ),
        name="cg_test_system",
    )


def test_conjugate_gradient_satisfies_protocol() -> None:
    solver = ConjugateGradientSolver()

    assert isinstance(
        solver,
        LinearSolverProtocol,
    )


def test_conjugate_gradient_defaults() -> None:
    solver = ConjugateGradientSolver()

    assert solver.name == "conjugate_gradient"
    assert solver.backend_name == "scipy"
    assert solver.relative_tolerance == pytest.approx(
        1.0e-10
    )
    assert solver.absolute_tolerance == pytest.approx(
        0.0
    )
    assert solver.max_iterations is None


@pytest.mark.parametrize(
    "relative_tolerance",
    [
        0.0,
        -1.0,
        np.inf,
        np.nan,
    ],
)
def test_conjugate_gradient_rejects_invalid_relative_tolerance(
    relative_tolerance: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite and positive",
    ):
        ConjugateGradientSolver(
            relative_tolerance=relative_tolerance
        )


@pytest.mark.parametrize(
    "absolute_tolerance",
    [
        -1.0,
        np.inf,
        np.nan,
    ],
)
def test_conjugate_gradient_rejects_invalid_absolute_tolerance(
    absolute_tolerance: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="finite and non-negative",
    ):
        ConjugateGradientSolver(
            absolute_tolerance=absolute_tolerance
        )


@pytest.mark.parametrize(
    "max_iterations",
    [
        0,
        -1,
    ],
)
def test_conjugate_gradient_rejects_invalid_max_iterations(
    max_iterations: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        ConjugateGradientSolver(
            max_iterations=max_iterations
        )


def test_conjugate_gradient_rejects_non_system() -> None:
    with pytest.raises(
        TypeError,
        match="LinearSystem",
    ):
        ConjugateGradientSolver().solve(
            "invalid"
        )


def test_conjugate_gradient_returns_result() -> None:
    result = ConjugateGradientSolver().solve(
        create_spd_system()
    )

    assert isinstance(
        result,
        LinearSolveResult,
    )

    assert result.converged
    assert result.iterations > 0
    assert result.solver_name == (
        "conjugate_gradient"
    )
    assert result.backend_name == "scipy"
    assert result.termination_reason == (
        "convergence_tolerance_satisfied"
    )


def test_conjugate_gradient_matches_sparse_direct() -> None:
    system = create_spd_system()

    cg_result = ConjugateGradientSolver(
        relative_tolerance=1.0e-12,
        absolute_tolerance=1.0e-14,
    ).solve(system)

    direct_result = SparseDirectSolver().solve(
        system
    )

    np.testing.assert_allclose(
        cg_result.solution,
        direct_result.solution,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_conjugate_gradient_has_small_residual() -> None:
    result = ConjugateGradientSolver(
        relative_tolerance=1.0e-12,
        absolute_tolerance=1.0e-14,
    ).solve(
        create_spd_system()
    )

    assert result.final_residual is not None
    assert result.final_residual <= 1.0e-10


def test_residual_history_matches_iteration_count() -> None:
    result = ConjugateGradientSolver().solve(
        create_spd_system()
    )

    assert result.residual_history.size == (
        result.iterations
    )


def test_initial_guess_is_supported() -> None:
    system = create_spd_system()

    result = ConjugateGradientSolver(
        initial_guess=np.zeros(
            system.number_of_equations,
            dtype=np.float64,
        ),
    ).solve(system)

    assert result.converged
    assert result.metadata[
        "initial_guess_supplied"
    ] is True


def test_invalid_initial_guess_shape_is_rejected() -> None:
    system = create_spd_system()

    solver = ConjugateGradientSolver(
        initial_guess=np.zeros(2)
    )

    with pytest.raises(
        ValueError,
        match="shape must match",
    ):
        solver.solve(system)


def test_exact_initial_guess_requires_no_iterations() -> None:
    system = create_spd_system()

    exact_solution = (
        SparseDirectSolver()
        .solve(system)
        .solution
    )

    result = ConjugateGradientSolver(
        initial_guess=exact_solution,
    ).solve(system)

    assert result.converged
    assert result.iterations == 0
    assert result.residual_history.size == 0
    assert result.final_residual is None


def test_maximum_iteration_termination() -> None:
    matrix = csr_matrix(
        np.diag(
            np.linspace(
                1.0,
                100.0,
                num=20,
            )
        ),
        dtype=np.float64,
    )

    system = LinearSystem(
        matrix=matrix,
        right_hand_side=np.ones(
            20,
            dtype=np.float64,
        ),
    )

    result = ConjugateGradientSolver(
        relative_tolerance=1.0e-15,
        absolute_tolerance=0.0,
        max_iterations=1,
    ).solve(system)

    assert not result.converged
    assert result.iterations == 1
    assert result.termination_reason == (
        "maximum_iterations_reached"
    )


def test_cg_metadata_records_configuration() -> None:
    result = ConjugateGradientSolver(
        relative_tolerance=1.0e-8,
        absolute_tolerance=1.0e-12,
        max_iterations=50,
    ).solve(
        create_spd_system()
    )

    assert result.metadata[
        "scipy_solver"
    ] == "cg"

    assert result.metadata[
        "relative_tolerance"
    ] == pytest.approx(1.0e-8)

    assert result.metadata[
        "absolute_tolerance"
    ] == pytest.approx(1.0e-12)

    assert result.metadata[
        "maximum_iterations"
    ] == 50