from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from deviceforge import (
    BoundaryCondition,
    Device,
    Grid,
    Region,
    Simulation,
)
from deviceforge.physics import SILICON
from deviceforge.postprocessing import (
    calculate_electrostatic_fields_2d,
)
from deviceforge.solvers import PoissonSolver2D


DOMAIN_LENGTH_AXIS_0 = 100.0e-9
DOMAIN_LENGTH_AXIS_1 = 80.0e-9


@dataclass(frozen=True, slots=True)
class ConvergenceRecord:
    """Results from one grid-resolution study case."""

    shape_axis_0: int
    shape_axis_1: int
    grid_points: int
    spacing_axis_0: float
    spacing_axis_1: float
    representative_spacing: float

    solver_runtime_seconds: float
    total_runtime_seconds: float
    matrix_nonzero_entries: int

    potential_maximum_error: float
    potential_rms_error: float

    electric_field_axis_0_maximum_error: float
    electric_field_axis_0_rms_error: float

    electric_field_axis_1_maximum_error: float
    electric_field_axis_1_rms_error: float

    electric_field_magnitude_maximum_error: float
    electric_field_magnitude_rms_error: float

    final_residual: float
    converged: bool


def calculate_analytical_solution(
    grid: Grid,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Calculate the exact sinusoidal two-dimensional Laplace solution.

    The solution is

        phi(x, y)
            = sin(pi*x/Lx)
              * sinh(pi*y/Lx)
              / sinh(pi*Ly/Lx).
    """

    coordinate_axis_0 = (
        grid.coordinates(0)
    )

    coordinate_axis_1 = (
        grid.coordinates(1)
    )

    shifted_axis_0 = (
        coordinate_axis_0[:, None]
        - coordinate_axis_0[0]
    )

    shifted_axis_1 = (
        coordinate_axis_1[None, :]
        - coordinate_axis_1[0]
    )

    domain_length_axis_0 = (
        coordinate_axis_0[-1]
        - coordinate_axis_0[0]
    )

    domain_length_axis_1 = (
        coordinate_axis_1[-1]
        - coordinate_axis_1[0]
    )

    wave_number = (
        np.pi
        / domain_length_axis_0
    )

    denominator = np.sinh(
        wave_number
        * domain_length_axis_1
    )

    sine_term = np.sin(
        wave_number
        * shifted_axis_0
    )

    cosine_term = np.cos(
        wave_number
        * shifted_axis_0
    )

    hyperbolic_sine_term = np.sinh(
        wave_number
        * shifted_axis_1
    )

    hyperbolic_cosine_term = np.cosh(
        wave_number
        * shifted_axis_1
    )

    potential = (
        sine_term
        * hyperbolic_sine_term
        / denominator
    )

    electric_field_axis_0 = -(
        wave_number
        * cosine_term
        * hyperbolic_sine_term
        / denominator
    )

    electric_field_axis_1 = -(
        wave_number
        * sine_term
        * hyperbolic_cosine_term
        / denominator
    )

    expected_shape = grid.shape

    potential = np.broadcast_to(
        potential,
        expected_shape,
    ).copy()

    electric_field_axis_0 = np.broadcast_to(
        electric_field_axis_0,
        expected_shape,
    ).copy()

    electric_field_axis_1 = np.broadcast_to(
        electric_field_axis_1,
        expected_shape,
    ).copy()

    electric_field_magnitude = np.hypot(
        electric_field_axis_0,
        electric_field_axis_1,
    )

    return (
        potential,
        electric_field_axis_0,
        electric_field_axis_1,
        electric_field_magnitude,
    )


def create_simulation(
    *,
    shape: tuple[int, int],
) -> tuple[
    Simulation,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Create one fixed-domain sinusoidal Laplace problem."""

    spacing = (
        DOMAIN_LENGTH_AXIS_0
        / (shape[0] - 1),
        DOMAIN_LENGTH_AXIS_1
        / (shape[1] - 1),
    )

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
        region_type="semiconductor",
    )

    device = Device(
        name=(
            f"sinusoidal_laplace_"
            f"{shape[0]}x{shape[1]}"
        ),
        grid=grid,
        regions=(region,),
    )

    (
        analytical_potential,
        analytical_field_axis_0,
        analytical_field_axis_1,
        analytical_field_magnitude,
    ) = calculate_analytical_solution(
        grid
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
        name="left_zero_potential",
        grid=grid,
        mask=left_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_zero_potential",
        grid=grid,
        mask=right_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    bottom_boundary = BoundaryCondition(
        name="bottom_zero_potential",
        grid=grid,
        mask=bottom_mask,
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    top_boundary = BoundaryCondition(
        name="top_sinusoidal_potential",
        grid=grid,
        mask=top_mask,
        condition_type="dirichlet",
        value=analytical_potential,
        units="V",
    )

    simulation = Simulation(
        name=(
            f"sinusoidal_laplace_2d_"
            f"{shape[0]}x{shape[1]}"
        ),
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
            bottom_boundary,
            top_boundary,
        ),
        tolerance=1.0e-10,
        max_iterations=100_000,
        initial_potential=0.0,
    )

    return (
        simulation,
        analytical_potential,
        analytical_field_axis_0,
        analytical_field_axis_1,
        analytical_field_magnitude,
    )


def calculate_error_metrics(
    numerical_values: NDArray[np.float64],
    analytical_values: NDArray[np.float64],
) -> tuple[float, float]:
    """Return maximum absolute and RMS errors."""

    error = np.asarray(
        numerical_values
        - analytical_values,
        dtype=np.float64,
    )

    maximum_error = float(
        np.max(
            np.abs(error)
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


def run_case(
    shape: tuple[int, int],
) -> ConvergenceRecord:
    """Run one grid resolution and return its results."""

    case_start = perf_counter()

    (
        simulation,
        analytical_potential,
        analytical_field_axis_0,
        analytical_field_axis_1,
        analytical_field_magnitude,
    ) = create_simulation(
        shape=shape
    )

    result = PoissonSolver2D().solve(
        simulation
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
        electric_field_magnitude,
    ) = calculate_electrostatic_fields_2d(
        result.potential
    )

    potential_errors = (
        calculate_error_metrics(
            result.potential.values,
            analytical_potential,
        )
    )

    field_axis_0_errors = (
        calculate_error_metrics(
            electric_field_axis_0.values,
            analytical_field_axis_0,
        )
    )

    field_axis_1_errors = (
        calculate_error_metrics(
            electric_field_axis_1.values,
            analytical_field_axis_1,
        )
    )

    magnitude_errors = (
        calculate_error_metrics(
            electric_field_magnitude.values,
            analytical_field_magnitude,
        )
    )

    total_runtime_seconds = (
        perf_counter()
        - case_start
    )

    spacing_axis_0, spacing_axis_1 = (
        simulation.grid.spacing
    )

    representative_spacing = max(
        spacing_axis_0,
        spacing_axis_1,
    )

    final_residual = result.final_residual

    if final_residual is None:
        raise RuntimeError(
            "Convergence study requires a final residual."
        )

    return ConvergenceRecord(
        shape_axis_0=shape[0],
        shape_axis_1=shape[1],
        grid_points=(
            simulation.grid.number_of_points
        ),
        spacing_axis_0=spacing_axis_0,
        spacing_axis_1=spacing_axis_1,
        representative_spacing=(
            representative_spacing
        ),
        solver_runtime_seconds=(
            result.runtime_seconds
        ),
        total_runtime_seconds=(
            total_runtime_seconds
        ),
        matrix_nonzero_entries=int(
            result.metadata[
                "matrix_nonzero_entries"
            ]
        ),
        potential_maximum_error=(
            potential_errors[0]
        ),
        potential_rms_error=(
            potential_errors[1]
        ),
        electric_field_axis_0_maximum_error=(
            field_axis_0_errors[0]
        ),
        electric_field_axis_0_rms_error=(
            field_axis_0_errors[1]
        ),
        electric_field_axis_1_maximum_error=(
            field_axis_1_errors[0]
        ),
        electric_field_axis_1_rms_error=(
            field_axis_1_errors[1]
        ),
        electric_field_magnitude_maximum_error=(
            magnitude_errors[0]
        ),
        electric_field_magnitude_rms_error=(
            magnitude_errors[1]
        ),
        final_residual=float(
            final_residual
        ),
        converged=result.converged,
    )


def calculate_observed_orders(
    records: tuple[
        ConvergenceRecord,
        ...
    ],
    attribute_name: str,
) -> tuple[float | None, ...]:
    """
    Calculate observed convergence orders between consecutive grids.

    For two consecutive grids,

        p = log(error_coarse / error_fine)
            / log(h_coarse / h_fine).
    """

    orders: list[
        float | None
    ] = [None]

    for coarse, fine in zip(
        records[:-1],
        records[1:],
    ):
        coarse_error = float(
            getattr(
                coarse,
                attribute_name,
            )
        )

        fine_error = float(
            getattr(
                fine,
                attribute_name,
            )
        )

        spacing_ratio = (
            coarse.representative_spacing
            / fine.representative_spacing
        )

        if (
            coarse_error <= 0.0
            or fine_error <= 0.0
            or spacing_ratio <= 1.0
        ):
            orders.append(None)
            continue

        order = (
            np.log(
                coarse_error
                / fine_error
            )
            / np.log(
                spacing_ratio
            )
        )

        orders.append(
            float(order)
        )

    return tuple(orders)


def create_output_directory() -> Path:
    """Create and return the convergence-study output directory."""

    repository_root = (
        Path(__file__).resolve().parents[2]
    )

    output_directory = (
        repository_root
        / "examples"
        / "figures"
        / "verification"
        / "sinusoidal_laplace_2d_grid_convergence"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def save_csv(
    records: tuple[
        ConvergenceRecord,
        ...
    ],
    output_directory: Path,
) -> Path:
    """Save convergence records and observed orders."""

    potential_maximum_orders = (
        calculate_observed_orders(
            records,
            "potential_maximum_error",
        )
    )

    potential_rms_orders = (
        calculate_observed_orders(
            records,
            "potential_rms_error",
        )
    )

    field_magnitude_rms_orders = (
        calculate_observed_orders(
            records,
            (
                "electric_field_"
                "magnitude_rms_error"
            ),
        )
    )

    output_path = (
        output_directory
        / "convergence_results.csv"
    )

    fieldnames = [
        "shape_axis_0",
        "shape_axis_1",
        "grid_points",
        "spacing_axis_0_m",
        "spacing_axis_1_m",
        "representative_spacing_m",
        "solver_runtime_seconds",
        "total_runtime_seconds",
        "matrix_nonzero_entries",
        "potential_maximum_error_v",
        "potential_maximum_order",
        "potential_rms_error_v",
        "potential_rms_order",
        "electric_field_axis_0_maximum_error_v_per_m",
        "electric_field_axis_0_rms_error_v_per_m",
        "electric_field_axis_1_maximum_error_v_per_m",
        "electric_field_axis_1_rms_error_v_per_m",
        "electric_field_magnitude_maximum_error_v_per_m",
        "electric_field_magnitude_rms_error_v_per_m",
        "electric_field_magnitude_rms_order",
        "final_residual",
        "converged",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, record in enumerate(
            records
        ):
            writer.writerow(
                {
                    "shape_axis_0": (
                        record.shape_axis_0
                    ),
                    "shape_axis_1": (
                        record.shape_axis_1
                    ),
                    "grid_points": (
                        record.grid_points
                    ),
                    "spacing_axis_0_m": (
                        record.spacing_axis_0
                    ),
                    "spacing_axis_1_m": (
                        record.spacing_axis_1
                    ),
                    "representative_spacing_m": (
                        record.representative_spacing
                    ),
                    "solver_runtime_seconds": (
                        record.solver_runtime_seconds
                    ),
                    "total_runtime_seconds": (
                        record.total_runtime_seconds
                    ),
                    "matrix_nonzero_entries": (
                        record.matrix_nonzero_entries
                    ),
                    "potential_maximum_error_v": (
                        record.potential_maximum_error
                    ),
                    "potential_maximum_order": (
                        potential_maximum_orders[
                            index
                        ]
                    ),
                    "potential_rms_error_v": (
                        record.potential_rms_error
                    ),
                    "potential_rms_order": (
                        potential_rms_orders[
                            index
                        ]
                    ),
                    (
                        "electric_field_axis_0_"
                        "maximum_error_v_per_m"
                    ): (
                        record
                        .electric_field_axis_0_maximum_error
                    ),
                    (
                        "electric_field_axis_0_"
                        "rms_error_v_per_m"
                    ): (
                        record
                        .electric_field_axis_0_rms_error
                    ),
                    (
                        "electric_field_axis_1_"
                        "maximum_error_v_per_m"
                    ): (
                        record
                        .electric_field_axis_1_maximum_error
                    ),
                    (
                        "electric_field_axis_1_"
                        "rms_error_v_per_m"
                    ): (
                        record
                        .electric_field_axis_1_rms_error
                    ),
                    (
                        "electric_field_magnitude_"
                        "maximum_error_v_per_m"
                    ): (
                        record
                        .electric_field_magnitude_maximum_error
                    ),
                    (
                        "electric_field_magnitude_"
                        "rms_error_v_per_m"
                    ): (
                        record
                        .electric_field_magnitude_rms_error
                    ),
                    (
                        "electric_field_magnitude_"
                        "rms_order"
                    ): (
                        field_magnitude_rms_orders[
                            index
                        ]
                    ),
                    "final_residual": (
                        record.final_residual
                    ),
                    "converged": (
                        record.converged
                    ),
                }
            )

    return output_path


def plot_error_convergence(
    records: tuple[
        ConvergenceRecord,
        ...
    ],
    output_directory: Path,
) -> Path:
    """Plot potential and electric-field RMS errors."""

    spacing_nanometres = np.asarray(
        [
            record.representative_spacing
            * 1.0e9
            for record in records
        ],
        dtype=np.float64,
    )

    potential_rms_errors = np.asarray(
        [
            record.potential_rms_error
            for record in records
        ],
        dtype=np.float64,
    )

    field_magnitude_rms_errors = np.asarray(
        [
            record
            .electric_field_magnitude_rms_error
            for record in records
        ],
        dtype=np.float64,
    )

    figure, axis = plt.subplots()

    axis.loglog(
        spacing_nanometres,
        potential_rms_errors,
        marker="o",
        label="Potential RMS error (V)",
    )

    secondary_axis = axis.twinx()

    secondary_axis.loglog(
        spacing_nanometres,
        field_magnitude_rms_errors,
        marker="s",
        label=(
            "Electric-field magnitude "
            "RMS error (V/m)"
        ),
    )

    axis.set_xlabel(
        "Representative grid spacing (nm)"
    )

    axis.set_ylabel(
        "Potential RMS error (V)"
    )

    secondary_axis.set_ylabel(
        "Electric-field magnitude RMS error (V/m)"
    )

    axis.set_title(
        "Sinusoidal 2D Laplace Grid Convergence"
    )

    axis.grid(
        True,
        which="both",
    )

    axis.invert_xaxis()

    lines = (
        axis.get_lines()
        + secondary_axis.get_lines()
    )

    labels = [
        line.get_label()
        for line in lines
    ]

    axis.legend(
        lines,
        labels,
        loc="best",
    )

    figure.tight_layout()

    output_path = (
        output_directory
        / "error_convergence.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_runtime_scaling(
    records: tuple[
        ConvergenceRecord,
        ...
    ],
    output_directory: Path,
) -> Path:
    """Plot runtime against number of grid points."""

    grid_points = [
        record.grid_points
        for record in records
    ]

    solver_runtime = [
        record.solver_runtime_seconds
        for record in records
    ]

    total_runtime = [
        record.total_runtime_seconds
        for record in records
    ]

    figure, axis = plt.subplots()

    axis.loglog(
        grid_points,
        solver_runtime,
        marker="o",
        label="Solver runtime",
    )

    axis.loglog(
        grid_points,
        total_runtime,
        marker="s",
        label="Total case runtime",
    )

    axis.set_xlabel(
        "Grid points"
    )

    axis.set_ylabel(
        "Runtime (s)"
    )

    axis.set_title(
        "Sinusoidal 2D Laplace Runtime Scaling"
    )

    axis.grid(
        True,
        which="both",
    )

    axis.legend()

    figure.tight_layout()

    output_path = (
        output_directory
        / "runtime_scaling.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def print_results(
    records: tuple[
        ConvergenceRecord,
        ...
    ],
) -> None:
    """Print convergence records and observed orders."""

    potential_maximum_orders = (
        calculate_observed_orders(
            records,
            "potential_maximum_error",
        )
    )

    potential_rms_orders = (
        calculate_observed_orders(
            records,
            "potential_rms_error",
        )
    )

    field_magnitude_rms_orders = (
        calculate_observed_orders(
            records,
            (
                "electric_field_"
                "magnitude_rms_error"
            ),
        )
    )

    print("\nGrid-convergence results:")

    for index, record in enumerate(
        records
    ):
        potential_maximum_order = (
            potential_maximum_orders[index]
        )

        potential_rms_order = (
            potential_rms_orders[index]
        )

        magnitude_rms_order = (
            field_magnitude_rms_orders[index]
        )

        print()
        print(
            f"Grid: {record.shape_axis_0:,}"
            f" × {record.shape_axis_1:,}"
            f" ({record.grid_points:,} points)"
        )

        print(
            "  Spacing:              "
            f"{record.spacing_axis_0:.6e}, "
            f"{record.spacing_axis_1:.6e} m"
        )

        print(
            "  Potential max error:  "
            f"{record.potential_maximum_error:.6e} V"
        )

        print(
            "  Potential max order:  "
            f"{potential_maximum_order:.6f}"
            if potential_maximum_order is not None
            else
            "  Potential max order:  —"
        )

        print(
            "  Potential RMS error:  "
            f"{record.potential_rms_error:.6e} V"
        )

        print(
            "  Potential RMS order:  "
            f"{potential_rms_order:.6f}"
            if potential_rms_order is not None
            else
            "  Potential RMS order:  —"
        )

        print(
            "  Field |E| RMS error:  "
            f"{record.electric_field_magnitude_rms_error:.6e} V/m"
        )

        print(
            "  Field |E| RMS order:  "
            f"{magnitude_rms_order:.6f}"
            if magnitude_rms_order is not None
            else
            "  Field |E| RMS order:  —"
        )

        print(
            "  Solver runtime:       "
            f"{record.solver_runtime_seconds:.6e} s"
        )

        print(
            "  Matrix nonzeros:      "
            f"{record.matrix_nonzero_entries:,}"
        )

        print(
            "  Final residual:       "
            f"{record.final_residual:.6e}"
        )

        print(
            "  Converged:            "
            f"{record.converged}"
        )


def main() -> None:
    """Run the sinusoidal 2D Laplace grid-convergence study."""

    print("=" * 72)
    print(
        "DeviceForge — sinusoidal 2D Laplace grid-convergence study"
    )
    print("=" * 72)

    grid_shapes = (
        (26, 21),
        (51, 41),
        (101, 81),
        (201, 161),
    )

    records: list[
        ConvergenceRecord
    ] = []

    for shape in grid_shapes:
        print()
        print(
            f"Running grid "
            f"{shape[0]} × {shape[1]}..."
        )

        record = run_case(
            shape
        )

        records.append(
            record
        )

        print(
            "  Potential RMS error: "
            f"{record.potential_rms_error:.6e} V"
        )

        print(
            "  Solver runtime:      "
            f"{record.solver_runtime_seconds:.6e} s"
        )

    record_tuple = tuple(
        records
    )

    print_results(
        record_tuple
    )

    output_directory = (
        create_output_directory()
    )

    csv_path = save_csv(
        record_tuple,
        output_directory,
    )

    convergence_plot_path = (
        plot_error_convergence(
            record_tuple,
            output_directory,
        )
    )

    runtime_plot_path = (
        plot_runtime_scaling(
            record_tuple,
            output_directory,
        )
    )

    print("\nSaved outputs:")
    print(f"  {csv_path}")
    print(f"  {convergence_plot_path}")
    print(f"  {runtime_plot_path}")


if __name__ == "__main__":
    main()