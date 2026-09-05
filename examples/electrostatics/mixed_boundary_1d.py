from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

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

DOMAIN_LENGTH = 10.0e-9

LEFT_POTENTIAL = 0.25

RIGHT_OUTWARD_DERIVATIVE = 5.0e6

CHARGE_DENSITY = 1.0e5


def calculate_analytical_solution(
    coordinates: NDArray[np.float64],
    *,
    domain_length: float,
    left_potential: float,
    right_outward_derivative: float,
    charge_density: float,
    relative_permittivity: float,
) -> NDArray[np.float64]:
    """
    Calculate the analytical mixed-boundary Poisson solution.

    DeviceForge solves

        -d/dx(
            epsilon_r * d(phi)/dx
        ) = rho / epsilon_0.

    For constant relative permittivity and uniform charge density,

        d^2(phi)/dx^2
            = -rho
              / (
                  epsilon_0
                  * epsilon_r
              ).

    The boundary conditions are

        phi(0) = phi_left

    and, at the right boundary,

        d(phi)/dn = g_right.

    In one dimension, the outward normal at x = L points in the
    positive x direction, so

        d(phi)/dx |_(x=L) = g_right.

    The analytical solution is therefore

        phi(x)
            = phi_left
              + C*x
              - rho*x^2
                / (
                    2
                    * epsilon_0
                    * epsilon_r
                )

    where

        C
            = g_right
              + rho*L
                / (
                    epsilon_0
                    * epsilon_r
                ).
    """

    shifted_coordinates = (
        coordinates
        - coordinates[0]
    )

    curvature_coefficient = (
        charge_density
        / (
            VACUUM_PERMITTIVITY
            * relative_permittivity
        )
    )

    linear_coefficient = (
        right_outward_derivative
        + curvature_coefficient
        * domain_length
    )

    return (
        left_potential
        + linear_coefficient
        * shifted_coordinates
        - 0.5
        * curvature_coefficient
        * shifted_coordinates**2
    )


def create_simulation(
    *,
    number_of_points: int = 161,
) -> tuple[
    Simulation,
    NDArray[np.float64],
]:
    """
    Create a charged one-dimensional mixed-boundary Poisson problem.
    """

    if number_of_points < 3:
        raise ValueError(
            "Mixed-boundary example requires "
            "at least three grid points."
        )

    spacing = (
        DOMAIN_LENGTH
        / (
            number_of_points
            - 1
        )
    )

    grid = Grid(
        shape=(
            number_of_points,
        ),
        spacing=(
            spacing,
        ),
    )

    silicon_region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=np.ones(
            grid.shape,
            dtype=np.bool_,
        ),
    )

    device = Device(
        name="mixed_boundary_1d_device",
        grid=grid,
        regions=(
            silicon_region,
        ),
    )

    charge_density = Field.full(
        name="charge_density",
        units="C/m^3",
        grid=grid,
        fill_value=CHARGE_DENSITY,
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
        name="left_dirichlet_contact",
        grid=grid,
        mask=left_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=LEFT_POTENTIAL,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_neumann_boundary",
        grid=grid,
        mask=right_mask,
        condition_type=(
            BoundaryConditionType.NEUMANN
        ),
        value=(
            RIGHT_OUTWARD_DERIVATIVE
        ),
        units="V/m",
    )

    simulation = Simulation(
        name="mixed_dirichlet_neumann_1d",
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        charge_density=charge_density,
        tolerance=1.0e-10,
        max_iterations=500,
        initial_potential=LEFT_POTENTIAL,
    )

    coordinates = grid.coordinates(0)

    analytical_potential = (
        calculate_analytical_solution(
            coordinates,
            domain_length=DOMAIN_LENGTH,
            left_potential=LEFT_POTENTIAL,
            right_outward_derivative=(
                RIGHT_OUTWARD_DERIVATIVE
            ),
            charge_density=(
                CHARGE_DENSITY
            ),
            relative_permittivity=(
                SILICON.relative_permittivity
            ),
        )
    )

    return (
        simulation,
        analytical_potential,
    )


def calculate_error_metrics(
    numerical: NDArray[np.float64],
    analytical: NDArray[np.float64],
) -> tuple[
    float,
    float,
]:
    """
    Calculate maximum absolute and RMS potential errors.
    """

    error = (
        numerical
        - analytical
    )

    maximum_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    rms_error = float(
        np.sqrt(
            np.mean(
                error**2
            )
        )
    )

    return (
        maximum_error,
        rms_error,
    )


def create_output_directory() -> Path:
    """
    Create and return the example output directory.
    """

    repository_root = (
        Path(__file__).resolve().parents[2]
    )

    output_directory = (
        repository_root
        / "examples"
        / "figures"
        / "examples"
        / "mixed_boundary_1d"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def plot_potential_comparison(
    *,
    coordinates: NDArray[np.float64],
    numerical: NDArray[np.float64],
    analytical: NDArray[np.float64],
    output_directory: Path,
) -> Path:
    """
    Plot numerical and analytical potential profiles.
    """

    coordinate_nanometres = (
        coordinates
        * 1.0e9
    )

    figure, axis = plt.subplots(
        figsize=(
            7,
            5,
        )
    )

    axis.plot(
        coordinate_nanometres,
        analytical,
        linewidth=2.0,
        label="Analytical solution",
    )

    axis.plot(
        coordinate_nanometres,
        numerical,
        marker="o",
        markersize=3.0,
        markevery=8,
        label="DeviceForge numerical solution",
    )

    axis.set_xlabel(
        "Position (nm)"
    )

    axis.set_ylabel(
        "Electrostatic potential (V)"
    )

    axis.set_title(
        "1D Mixed Dirichlet–Neumann Poisson Problem"
    )

    axis.grid(
        True
    )

    axis.legend()

    figure.tight_layout()

    output_path = (
        output_directory
        / "01_potential_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def plot_potential_error(
    *,
    coordinates: NDArray[np.float64],
    numerical: NDArray[np.float64],
    analytical: NDArray[np.float64],
    output_directory: Path,
) -> Path:
    """
    Plot the numerical potential error.
    """

    coordinate_nanometres = (
        coordinates
        * 1.0e9
    )

    error = (
        numerical
        - analytical
    )

    figure, axis = plt.subplots(
        figsize=(
            7,
            5,
        )
    )

    axis.plot(
        coordinate_nanometres,
        error,
        marker="o",
        markersize=3.0,
        markevery=8,
    )

    axis.axhline(
        0.0,
        linewidth=1.0,
        linestyle="--",
    )

    axis.set_xlabel(
        "Position (nm)"
    )

    axis.set_ylabel(
        "Potential error (V)"
    )

    axis.set_title(
        "1D Mixed-Boundary Potential Error"
    )

    axis.grid(
        True
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / "02_potential_error.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def main() -> None:
    """
    Run the charged 1D mixed Dirichlet–Neumann example.
    """

    print("=" * 72)

    print(
        "DeviceForge — 1D mixed "
        "Dirichlet–Neumann Poisson example"
    )

    print("=" * 72)

    simulation, analytical = (
        create_simulation()
    )

    print()
    print(
        f"Domain length:               "
        f"{DOMAIN_LENGTH * 1.0e9:.3f} nm"
    )

    print(
        f"Grid points:                 "
        f"{simulation.grid.number_of_points}"
    )

    print(
        f"Relative permittivity:       "
        f"{SILICON.relative_permittivity:.6f}"
    )

    print(
        f"Charge density:              "
        f"{CHARGE_DENSITY:.6e} C/m^3"
    )

    print(
        f"Left Dirichlet potential:    "
        f"{LEFT_POTENTIAL:.6f} V"
    )

    print(
        f"Right outward derivative:    "
        f"{RIGHT_OUTWARD_DERIVATIVE:.6e} V/m"
    )

    print()
    print(
        "Solving..."
    )

    result = PoissonSolver().solve(
        simulation
    )

    numerical = np.asarray(
        result.potential.values,
        dtype=np.float64,
    )

    coordinates = (
        simulation.grid.coordinates(0)
    )

    (
        maximum_error,
        rms_error,
    ) = calculate_error_metrics(
        numerical,
        analytical,
    )

    print()
    print(
        "Solution results:"
    )

    print(
        f"  Converged:           "
        f"{result.converged}"
    )

    print(
        f"  Maximum error:       "
        f"{maximum_error:.12e} V"
    )

    print(
        f"  RMS error:           "
        f"{rms_error:.12e} V"
    )

    print(
        f"  Left potential:      "
        f"{numerical[0]:.12e} V"
    )

    print(
        f"  Right potential:     "
        f"{numerical[-1]:.12e} V"
    )

    print(
        f"  Analytical right:    "
        f"{analytical[-1]:.12e} V"
    )

    output_directory = (
        create_output_directory()
    )

    potential_plot_path = (
        plot_potential_comparison(
            coordinates=coordinates,
            numerical=numerical,
            analytical=analytical,
            output_directory=(
                output_directory
            ),
        )
    )

    error_plot_path = (
        plot_potential_error(
            coordinates=coordinates,
            numerical=numerical,
            analytical=analytical,
            output_directory=(
                output_directory
            ),
        )
    )

    print()
    print(
        "Saved outputs:"
    )

    print(
        f"  {potential_plot_path}"
    )

    print(
        f"  {error_plot_path}"
    )


if __name__ == "__main__":
    main()