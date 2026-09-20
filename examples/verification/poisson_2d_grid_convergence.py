from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

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
from deviceforge.solvers import PoissonSolver2D


VACUUM_PERMITTIVITY = 8.8541878128e-12

GRID_SIZES = (
    21,
    41,
    81,
    161,
)

DOMAIN_LENGTH_AXIS_0 = 1.0e-8
DOMAIN_LENGTH_AXIS_1 = 1.0e-8

OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "verification"
    / "poisson_2d_grid_convergence"
)


def create_manufactured_sine_simulation_2d(
    number_of_points: int,
    *,
    domain_length_axis_0: float = DOMAIN_LENGTH_AXIS_0,
    domain_length_axis_1: float = DOMAIN_LENGTH_AXIS_1,
) -> tuple[
    Simulation,
    np.ndarray,
]:
    """
    Create a two-dimensional manufactured Poisson problem.

    The analytical potential is

        phi(x, y)
            = sin(pi * x / Lx)
            * sin(pi * y / Ly)

    with homogeneous Dirichlet boundary conditions on the complete
    outer boundary.

    For constant relative permittivity,

        rho(x, y)
            = epsilon_0
            * epsilon_r
            * [
                (pi / Lx)^2
                + (pi / Ly)^2
              ]
            * phi(x, y).
    """

    if number_of_points < 3:
        raise ValueError(
            "Manufactured convergence problem requires "
            "at least three grid points per axis."
        )

    spacing_axis_0 = (
        domain_length_axis_0
        / (number_of_points - 1)
    )

    spacing_axis_1 = (
        domain_length_axis_1
        / (number_of_points - 1)
    )

    grid = Grid(
        shape=(
            number_of_points,
            number_of_points,
        ),
        spacing=(
            spacing_axis_0,
            spacing_axis_1,
        ),
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
        name="manufactured_sine_device_2d",
        grid=grid,
        regions=(
            silicon_region,
        ),
    )

    coordinates_axis_0 = grid.coordinates(0)
    coordinates_axis_1 = grid.coordinates(1)

    local_axis_0 = (
        coordinates_axis_0
        - coordinates_axis_0[0]
    )

    local_axis_1 = (
        coordinates_axis_1
        - coordinates_axis_1[0]
    )

    analytical_potential = (
        np.sin(
            np.pi
            * local_axis_0[:, None]
            / domain_length_axis_0
        )
        * np.sin(
            np.pi
            * local_axis_1[None, :]
            / domain_length_axis_1
        )
    )

    relative_permittivity = (
        SILICON.relative_permittivity
    )

    eigenvalue = (
        (
            np.pi
            / domain_length_axis_0
        )**2
        + (
            np.pi
            / domain_length_axis_1
        )**2
    )

    charge_density_values = (
        VACUUM_PERMITTIVITY
        * relative_permittivity
        * eigenvalue
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

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )

    bottom_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )

    top_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )

    # Use non-overlapping Dirichlet masks.
    # Left and right boundaries own the corner nodes.
    left_mask[0, :] = True
    right_mask[-1, :] = True

    bottom_mask[1:-1, 0] = True
    top_mask[1:-1, -1] = True

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

    bottom_boundary = BoundaryCondition(
        name="bottom_contact",
        grid=grid,
        mask=bottom_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=0.0,
        units="V",
    )

    top_boundary = BoundaryCondition(
        name="top_contact",
        grid=grid,
        mask=top_mask,
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
            bottom_boundary,
            top_boundary,
        ),
        charge_density=charge_density,
        tolerance=1.0e-10,
        max_iterations=500,
        initial_potential=0.0,
        name=(
            f"manufactured_sine_2d_"
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

    return float(
        np.log(
            coarse_error
            / fine_error
        )
        / np.log(
            refinement_ratio
        )
    )


def run_convergence_study(
) -> tuple[
    list[float],
    list[float],
    list[float],
    np.ndarray,
    np.ndarray,
]:
    """
    Run the manufactured-solution study for all configured grids.
    """

    errors: list[float] = []
    runtimes: list[float] = []

    finest_numerical: np.ndarray | None = None
    finest_analytical: np.ndarray | None = None

    for number_of_points in GRID_SIZES:
        simulation, analytical = (
            create_manufactured_sine_simulation_2d(
                number_of_points
            )
        )

        start_time = perf_counter()

        result = PoissonSolver2D().solve(
            simulation
        )

        elapsed_time = (
            perf_counter()
            - start_time
        )

        if not result.converged:
            raise RuntimeError(
                "PoissonSolver2D did not converge for "
                f"{number_of_points} x "
                f"{number_of_points} grid."
            )

        numerical = result.potential.values

        error = calculate_rms_error(
            numerical,
            analytical,
        )

        errors.append(error)
        runtimes.append(elapsed_time)

        if number_of_points == GRID_SIZES[-1]:
            finest_numerical = numerical.copy()
            finest_analytical = analytical.copy()

    if (
        finest_numerical is None
        or finest_analytical is None
    ):
        raise RuntimeError(
            "Finest-grid solution was not recorded."
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

    return (
        errors,
        observed_orders,
        runtimes,
        finest_numerical,
        finest_analytical,
    )


def print_results(
    errors: list[float],
    observed_orders: list[float],
    runtimes: list[float],
) -> None:
    """
    Print the convergence study as a compact table.
    """

    print()
    print(
        "2D Poisson manufactured-solution "
        "convergence study"
    )

    print(
        "Grid points | RMS error | "
        "Error ratio | Observed order | Runtime (s)"
    )

    print(
        "-" * 82
    )

    print(
        f"{GRID_SIZES[0]:4d} x "
        f"{GRID_SIZES[0]:<4d} | "
        f"{errors[0]:.12e} | "
        f"{'-':>11} | "
        f"{'-':>14} | "
        f"{runtimes[0]:.6f}"
    )

    for index in range(
        1,
        len(GRID_SIZES),
    ):
        error_ratio = (
            errors[index - 1]
            / errors[index]
        )

        observed_order = (
            observed_orders[index - 1]
        )

        print(
            f"{GRID_SIZES[index]:4d} x "
            f"{GRID_SIZES[index]:<4d} | "
            f"{errors[index]:.12e} | "
            f"{error_ratio:11.6f} | "
            f"{observed_order:14.8f} | "
            f"{runtimes[index]:.6f}"
        )


def write_results_csv(
    errors: list[float],
    observed_orders: list[float],
    runtimes: list[float],
) -> None:
    """
    Write the convergence results to CSV.
    """

    output_path = (
        OUTPUT_DIRECTORY
        / "convergence_results.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            (
                "grid_points_axis_0",
                "grid_points_axis_1",
                "rms_error",
                "error_ratio",
                "observed_order",
                "runtime_seconds",
            )
        )

        for index, number_of_points in enumerate(
            GRID_SIZES
        ):
            if index == 0:
                error_ratio = ""
                observed_order = ""
            else:
                error_ratio = (
                    errors[index - 1]
                    / errors[index]
                )

                observed_order = (
                    observed_orders[index - 1]
                )

            writer.writerow(
                (
                    number_of_points,
                    number_of_points,
                    errors[index],
                    error_ratio,
                    observed_order,
                    runtimes[index],
                )
            )


def plot_error_convergence(
    errors: list[float],
) -> None:
    """
    Plot RMS error against grid spacing on logarithmic axes.
    """

    spacings = np.asarray(
        [
            DOMAIN_LENGTH_AXIS_0
            / (number_of_points - 1)
            for number_of_points in GRID_SIZES
        ],
        dtype=np.float64,
    )

    errors_array = np.asarray(
        errors,
        dtype=np.float64,
    )

    reference_second_order = (
        errors_array[0]
        * (
            spacings
            / spacings[0]
        )**2
    )

    figure, axis = plt.subplots()

    axis.loglog(
        spacings,
        errors_array,
        marker="o",
        label="Numerical RMS error",
    )

    axis.loglog(
        spacings,
        reference_second_order,
        linestyle="--",
        label="Second-order reference",
    )

    axis.set_xlabel(
        "Grid spacing (m)"
    )

    axis.set_ylabel(
        "RMS potential error (V)"
    )

    axis.set_title(
        "2D Poisson Grid Convergence"
    )

    axis.grid(
        True,
        which="both",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "error_convergence.png",
        dpi=200,
    )

    plt.close(
        figure
    )


def plot_runtime_scaling(
    runtimes: list[float],
) -> None:
    """
    Plot measured solver runtime against total grid nodes.

    The timings are illustrative only and are not intended to establish
    formal computational-complexity scaling.
    """

    total_nodes = np.asarray(
        [
            number_of_points**2
            for number_of_points in GRID_SIZES
        ],
        dtype=np.int64,
    )

    figure, axis = plt.subplots()

    axis.plot(
        total_nodes,
        runtimes,
        marker="o",
    )

    axis.set_xlabel(
        "Total grid nodes"
    )

    axis.set_ylabel(
        "Runtime (s)"
    )

    axis.set_title(
        "2D Poisson Solver Runtime"
    )

    axis.grid(
        True,
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "runtime_scaling.png",
        dpi=200,
    )

    plt.close(
        figure
    )


def plot_numerical_vs_analytical(
    numerical: np.ndarray,
    analytical: np.ndarray,
) -> None:
    """
    Plot the numerical, analytical, and absolute-error fields on the
    finest grid.
    """

    absolute_error = np.abs(
        numerical
        - analytical
    )

    potential_minimum = min(
        float(np.min(numerical)),
        float(np.min(analytical)),
    )

    potential_maximum = max(
        float(np.max(numerical)),
        float(np.max(analytical)),
    )

    extent_nm = (
        0.0,
        DOMAIN_LENGTH_AXIS_1 * 1.0e9,
        0.0,
        DOMAIN_LENGTH_AXIS_0 * 1.0e9,
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            4.5,
        ),
    )

    numerical_image = axes[0].imshow(
        numerical,
        origin="lower",
        extent=extent_nm,
        aspect="auto",
        vmin=potential_minimum,
        vmax=potential_maximum,
    )

    axes[0].set_title(
        "Numerical potential"
    )

    axes[0].set_xlabel(
        "Axis 1 position (nm)"
    )

    axes[0].set_ylabel(
        "Axis 0 position (nm)"
    )

    figure.colorbar(
        numerical_image,
        ax=axes[0],
        label="Potential (V)",
    )

    analytical_image = axes[1].imshow(
        analytical,
        origin="lower",
        extent=extent_nm,
        aspect="auto",
        vmin=potential_minimum,
        vmax=potential_maximum,
    )

    axes[1].set_title(
        "Analytical potential"
    )

    axes[1].set_xlabel(
        "Axis 1 position (nm)"
    )

    axes[1].set_ylabel(
        "Axis 0 position (nm)"
    )

    figure.colorbar(
        analytical_image,
        ax=axes[1],
        label="Potential (V)",
    )

    error_image = axes[2].imshow(
        absolute_error,
        origin="lower",
        extent=extent_nm,
        aspect="auto",
    )

    axes[2].set_title(
        "Absolute error"
    )

    axes[2].set_xlabel(
        "Axis 1 position (nm)"
    )

    axes[2].set_ylabel(
        "Axis 0 position (nm)"
    )

    figure.colorbar(
        error_image,
        ax=axes[2],
        label="Absolute error (V)",
    )

    figure.suptitle(
        "2D Poisson Manufactured-Solution Verification"
    )

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "numerical_vs_analytical.png",
        dpi=200,
    )

    plt.close(
        figure
    )


def main() -> None:
    """
    Run the complete 2D Poisson grid-convergence verification.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        errors,
        observed_orders,
        runtimes,
        finest_numerical,
        finest_analytical,
    ) = run_convergence_study()

    print_results(
        errors,
        observed_orders,
        runtimes,
    )

    write_results_csv(
        errors,
        observed_orders,
        runtimes,
    )

    plot_error_convergence(
        errors
    )

    plot_runtime_scaling(
        runtimes
    )

    plot_numerical_vs_analytical(
        finest_numerical,
        finest_analytical,
    )

    print()
    print(
        "Verification outputs written to:"
    )

    print(
        OUTPUT_DIRECTORY
    )


if __name__ == "__main__":
    main()