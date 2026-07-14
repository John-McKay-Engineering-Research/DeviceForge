import numpy as np
import pytest

from benchmarks.compare_iterative_solvers import (
    analytical_solution,
    build_simulation,
    maximum_analytical_error,
)
from deviceforge import Field, SimulationResult


def test_analytical_solution_shape() -> None:
    simulation = build_simulation(
        shape=(11, 7),
    )

    expected = analytical_solution(simulation)

    assert expected.shape == simulation.grid.shape

    np.testing.assert_allclose(
        expected[0, :],
        0.0,
    )

    np.testing.assert_allclose(
        expected[-1, :],
        1.0,
    )


def test_analytical_solution_is_linear() -> None:
    simulation = build_simulation(
        shape=(5, 3),
    )

    expected = analytical_solution(simulation)

    np.testing.assert_allclose(
        expected[:, 1],
        np.array(
            [
                0.0,
                0.25,
                0.5,
                0.75,
                1.0,
            ]
        ),
    )


def test_maximum_analytical_error() -> None:
    simulation = build_simulation(
        shape=(5, 3),
    )

    expected = analytical_solution(simulation)

    values = expected.copy()
    values[2, 1] += 0.05

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=simulation.grid,
        values=values,
    )

    result = SimulationResult(
        fields={
            "electrostatic_potential": potential,
        },
        converged=True,
        iterations=1,
        residual_history=np.array([0.01]),
        runtime_seconds=0.001,
        solver_name="test_solver",
    )

    error = maximum_analytical_error(
        result,
        expected,
    )

    assert error == pytest.approx(0.05)