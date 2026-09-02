from __future__ import annotations

import numpy as np
import pytest

from deviceforge import (
    Device,
    Grid,
    Region,
)
from deviceforge.core import Field
from deviceforge.core.boundary import (
    BoundaryCondition,
    BoundaryConditionType,
)
from deviceforge.core.simulation import Simulation
from deviceforge.physics import SILICON
from deviceforge.solvers import PoissonSolver


VACUUM_PERMITTIVITY = 8.8541878128e-12


def create_manufactured_sine_simulation(
    number_of_points: int,
    *,
    domain_length: float = 1.0e-8,
) -> tuple[
    Simulation,
    np.ndarray,
]:
    """
    Create a one-dimensional manufactured Poisson problem.

    The analytical potential is

        phi(x) = sin(pi * x / L)

    with homogeneous Dirichlet boundary conditions

        phi(0) = 0
        phi(L) = 0.

    For constant relative permittivity, the corresponding charge
    density is

        rho(x)
            = epsilon_0
            * epsilon_r
            * (pi / L)^2
            * sin(pi * x / L).

    This manufactured solution is used to measure the observed
    spatial convergence order of the one-dimensional Poisson solver.
    """

    if number_of_points < 3:
        raise ValueError(
            "Manufactured convergence problem requires "
            "at least three grid points."
        )

    spacing = (
        domain_length
        / (number_of_points - 1)
    )

    grid = Grid(
        shape=(number_of_points,),
        spacing=(spacing,),
    )

    region_mask = np.ones(
        grid.shape,
        dtype=np.bool_,
    )

    silicon_region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=region_mask,
    )

    device = Device(
        name="manufactured_sine_device",
        grid=grid,
        regions=(
            silicon_region,
        ),
    )

    coordinates = grid.coordinates(0)

    local_coordinates = (
        coordinates
        - coordinates[0]
    )

    analytical_potential = np.sin(
        np.pi
        * local_coordinates
        / domain_length
    )

    relative_permittivity = (
        SILICON.relative_permittivity
    )

    charge_density_values = (
        VACUUM_PERMITTIVITY
        * relative_permittivity
        * (
            np.pi
            / domain_length
        )**2
        * analytical_potential
    )

    charge_density = Field(
        name="charge_density",
        units="C/m^3",
        grid=grid,
        values=charge_density_values,
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
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=0.0,
        units="V",
    )

    simulation = Simulation(
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        charge_density=charge_density,
        tolerance=1.0e-10,
        max_iterations=500,
        initial_potential=0.0,
        name=(
            f"manufactured_sine_"
            f"{number_of_points}_points"
        ),
    )

    return (
        simulation,
        analytical_potential,
    )


def calculate_rms_error(
    numerical: np.ndarray,
    analytical: np.ndarray,
) -> float:
    """
    Calculate the root-mean-square error between numerical and
    analytical solutions.
    """

    difference = (
        numerical
        - analytical
    )

    return float(
        np.sqrt(
            np.mean(
                difference**2
            )
        )
    )


def calculate_observed_order(
    coarse_error: float,
    fine_error: float,
    *,
    refinement_ratio: float = 2.0,
) -> float:
    """
    Calculate the observed convergence order.

    For errors e_h and e_(h/r),

        p = log(e_h / e_(h/r)) / log(r).
    """

    if coarse_error <= 0.0:
        raise ValueError(
            "Coarse-grid error must be positive."
        )

    if fine_error <= 0.0:
        raise ValueError(
            "Fine-grid error must be positive."
        )

    if refinement_ratio <= 1.0:
        raise ValueError(
            "Refinement ratio must be greater than one."
        )

    return float(
        np.log(
            coarse_error
            / fine_error
        )
        / np.log(
            refinement_ratio
        )
    )


@pytest.mark.parametrize(
    "number_of_points",
    (
        21,
        41,
        81,
        161,
    ),
)
def test_manufactured_sine_solution_converges(
    number_of_points: int,
) -> None:
    """
    Verify that every manufactured-solution problem converges and
    produces a finite numerical solution.
    """

    simulation, analytical = (
        create_manufactured_sine_simulation(
            number_of_points
        )
    )

    result = PoissonSolver().solve(
        simulation
    )

    assert result.converged

    assert np.all(
        np.isfinite(
            result.potential.values
        )
    )

    error = calculate_rms_error(
        result.potential.values,
        analytical,
    )

    assert np.isfinite(error)
    assert error > 0.0


def test_poisson_solver_has_second_order_grid_convergence() -> None:
    """
    Verify second-order spatial convergence for the one-dimensional
    Poisson solver using a sinusoidal manufactured solution.
    """

    grid_sizes = (
        21,
        41,
        81,
        161,
    )

    errors: list[float] = []

    for number_of_points in grid_sizes:
        simulation, analytical = (
            create_manufactured_sine_simulation(
                number_of_points
            )
        )

        result = PoissonSolver().solve(
            simulation
        )

        assert result.converged

        error = calculate_rms_error(
            result.potential.values,
            analytical,
        )

        errors.append(error)

    assert (
        errors[1] < errors[0]
    )

    assert (
        errors[2] < errors[1]
    )

    assert (
        errors[3] < errors[2]
    )

    observed_orders = [
        calculate_observed_order(
            errors[index],
            errors[index + 1],
        )
        for index in range(
            len(errors) - 1
        )
    ]
    """
    print()
    print(
        "1D Poisson manufactured-solution "
        "convergence study"
    )
    print(
        "Grid points | RMS error | "
        "Error ratio | Observed order"
    )
    print(
        "-" * 64
    )

    print(
        f"{grid_sizes[0]:11d} | "
        f"{errors[0]:.12e} | "
        f"{'-':>11} | "
        f"{'-':>14}"
    )

    for index in range(
            1,
            len(grid_sizes),
    ):
        error_ratio = (
                errors[index - 1]
                / errors[index]
        )

        observed_order = (
            observed_orders[index - 1]
        )

        print(
            f"{grid_sizes[index]:11d} | "
            f"{errors[index]:.12e} | "
            f"{error_ratio:11.6f} | "
            f"{observed_order:14.8f}"
        )
    """
    for order in observed_orders:
        assert order == pytest.approx(
            2.0,
            abs=0.05,
        )