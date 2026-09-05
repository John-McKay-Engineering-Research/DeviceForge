from __future__ import annotations

from pathlib import Path
import csv
import time

import matplotlib.pyplot as plt
import numpy as np

from deviceforge import Device, Grid, Region
from deviceforge.core import Field
from deviceforge.core.boundary import (
    BoundaryCondition,
    BoundaryConditionType,
)
from deviceforge.core.simulation import Simulation
from deviceforge.physics import SILICON
from deviceforge.solvers import PoissonSolver


VACUUM_PERMITTIVITY = 8.8541878128e-12


def create_manufactured_simulation(
    number_of_points: int,
    *,
    domain_length: float = 1.0e-8,
) -> tuple[Simulation, np.ndarray]:
    spacing = domain_length / (number_of_points - 1)

    grid = Grid(
        shape=(number_of_points,),
        spacing=(spacing,),
    )

    region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=np.ones(grid.shape, dtype=np.bool_),
    )

    device = Device(
        name="manufactured_sine_device",
        grid=grid,
        regions=(region,),
    )

    coordinates = grid.coordinates(0)
    x = coordinates - coordinates[0]

    analytical = np.sin(
        np.pi * x / domain_length
    )

    charge_density = Field(
        name="charge_density",
        units="C/m^3",
        grid=grid,
        values=(
            VACUUM_PERMITTIVITY
            * SILICON.relative_permittivity
            * (np.pi / domain_length) ** 2
            * analytical
        ),
    )

    left_mask = np.zeros(grid.shape, dtype=np.bool_)
    left_mask[0] = True

    right_mask = np.zeros(grid.shape, dtype=np.bool_)
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
        name=f"manufactured_sine_{number_of_points}",
    )

    return simulation, analytical


def plot_numerical_vs_analytical(
    *,
    coordinates: np.ndarray,
    numerical: np.ndarray,
    analytical: np.ndarray,
    output_directory: Path,
) -> Path:
    """
    Plot the finest-grid numerical solution against the exact
    manufactured analytical solution.
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
        label="Analytical solution",
        linewidth=2.0,
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
        "1D Poisson analytical comparison"
    )

    axis.grid(
        True
    )

    axis.legend()

    figure.tight_layout()

    output_path = (
        output_directory
        / "numerical_vs_analytical.png"
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
    grid_sizes = (
        21,
        41,
        81,
        161,
    )

    output_directory = (
        Path(__file__).resolve().parents[1]
        / "figures"
        / "verification"
        / "poisson_1d_grid_convergence"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[
        dict[
            str,
            float | int,
        ]
    ] = []

    finest_coordinates: np.ndarray | None = None
    finest_numerical: np.ndarray | None = None
    finest_analytical: np.ndarray | None = None

    print("=" * 72)
    print(
        "DeviceForge — 1D Poisson manufactured-solution "
        "grid-convergence study"
    )
    print("=" * 72)

    for number_of_points in grid_sizes:
        print(
            f"\nRunning grid "
            f"{number_of_points} points..."
        )

        simulation, analytical = (
            create_manufactured_simulation(
                number_of_points
            )
        )

        start = time.perf_counter()

        result = PoissonSolver().solve(
            simulation
        )

        runtime = (
            time.perf_counter()
            - start
        )

        numerical = (
            result.potential.values
        )

        rms_error = float(
            np.sqrt(
                np.mean(
                    (
                        numerical
                        - analytical
                    )**2
                )
            )
        )

        print(
            f"  RMS error: "
            f"{rms_error:.12e}"
        )

        print(
            f"  Runtime:   "
            f"{runtime:.6e} s"
        )

        results.append(
            {
                "grid_points": (
                    number_of_points
                ),
                "grid_spacing": (
                    simulation.grid.spacing[0]
                ),
                "rms_error": (
                    rms_error
                ),
                "runtime": (
                    runtime
                ),
            }
        )

        if (
            number_of_points
            == grid_sizes[-1]
        ):
            finest_coordinates = (
                simulation.grid.coordinates(0).copy()
            )

            finest_numerical = (
                np.asarray(
                    numerical,
                    dtype=np.float64,
                ).copy()
            )

            finest_analytical = (
                np.asarray(
                    analytical,
                    dtype=np.float64,
                ).copy()
            )

    errors = [
        float(
            row["rms_error"]
        )
        for row in results
    ]

    observed_orders: list[
        float | None
    ] = [None]

    for index in range(
        1,
        len(errors),
    ):
        order = (
            np.log(
                errors[index - 1]
                / errors[index]
            )
            / np.log(
                2.0
            )
        )

        observed_orders.append(
            float(order)
        )

    print(
        "\nGrid-convergence results:\n"
    )

    for index, row in enumerate(
        results
    ):
        print(
            f"Grid: "
            f"{int(row['grid_points'])} points"
        )

        print(
            f"  Spacing:        "
            f"{float(row['grid_spacing']):.6e} m"
        )

        print(
            f"  RMS error:      "
            f"{float(row['rms_error']):.12e}"
        )

        if index == 0:
            print(
                "  Observed order: —"
            )

        else:
            print(
                f"  Observed order: "
                f"{observed_orders[index]:.8f}"
            )

        print(
            f"  Runtime:        "
            f"{float(row['runtime']):.6e} s"
        )

        print()

    csv_path = (
        output_directory
        / "convergence_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "grid_points",
                "grid_spacing_m",
                "rms_error_V",
                "runtime_s",
                "observed_order",
            ]
        )

        for index, row in enumerate(
            results
        ):
            writer.writerow(
                [
                    row[
                        "grid_points"
                    ],
                    row[
                        "grid_spacing"
                    ],
                    row[
                        "rms_error"
                    ],
                    row[
                        "runtime"
                    ],
                    observed_orders[
                        index
                    ],
                ]
            )

    grid_points = [
        int(
            row["grid_points"]
        )
        for row in results
    ]

    rms_errors = [
        float(
            row["rms_error"]
        )
        for row in results
    ]

    runtimes = [
        float(
            row["runtime"]
        )
        for row in results
    ]

    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.loglog(
        grid_points,
        rms_errors,
        marker="o",
        label="RMS error",
    )

    plt.loglog(
        grid_points,
        [
            rms_errors[0]
            * (
                grid_points[0]
                / number_of_points
            )**2
            for number_of_points
            in grid_points
        ],
        linestyle="--",
        label=(
            "Second-order reference"
        ),
    )

    plt.xlabel(
        "Grid points"
    )

    plt.ylabel(
        "RMS potential error (V)"
    )

    plt.title(
        "1D Poisson grid convergence"
    )

    plt.grid(
        True,
        which="both",
    )

    plt.legend()

    plt.tight_layout()

    error_plot_path = (
        output_directory
        / "error_convergence.png"
    )

    plt.savefig(
        error_plot_path,
        dpi=200,
    )

    plt.close()

    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    plt.loglog(
        grid_points,
        runtimes,
        marker="o",
    )

    plt.xlabel(
        "Grid points"
    )

    plt.ylabel(
        "Solver runtime (s)"
    )

    plt.title(
        "1D Poisson runtime scaling"
    )

    plt.grid(
        True,
        which="both",
    )

    plt.tight_layout()

    runtime_plot_path = (
        output_directory
        / "runtime_scaling.png"
    )

    plt.savefig(
        runtime_plot_path,
        dpi=200,
    )

    plt.close()

    if (
        finest_coordinates is None
        or finest_numerical is None
        or finest_analytical is None
    ):
        raise RuntimeError(
            "Finest-grid solution was not recorded."
        )

    analytical_plot_path = (
        plot_numerical_vs_analytical(
            coordinates=finest_coordinates,
            numerical=finest_numerical,
            analytical=finest_analytical,
            output_directory=output_directory,
        )
    )

    print(
        "Saved outputs:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {error_plot_path}"
    )

    print(
        f"  {runtime_plot_path}"
    )

    print(
        f"  {analytical_plot_path}"
    )


if __name__ == "__main__":
    main()