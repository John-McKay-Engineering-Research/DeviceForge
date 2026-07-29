from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from deviceforge import (
    BoundaryCondition,
    Device,
    Grid,
    Region,
    Simulation,
)
from deviceforge.linalg import (
    ConjugateGradientSolver,
    DenseDirectSolver,
    SparseDirectSolver,
    IdentityPreconditioner,
    JacobiPreconditioner,

)
from deviceforge.physics import SILICON
from deviceforge.solvers import PoissonSolver


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """Result from one Poisson benchmark execution."""

    grid_points: int
    linear_solver: str
    backend: str
    elapsed_seconds: float
    iterations: int
    final_residual: float
    converged: bool
    maximum_solution_error: float


def create_benchmark_simulation(
    number_of_points: int,
) -> Simulation:
    """
    Create a uniform 1D silicon Laplace problem.

    The analytical solution is linear between 0 V and 1 V.
    """

    grid = Grid(
        shape=(number_of_points,),
        spacing=(1.0e-9,),
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
        name=f"benchmark_device_{number_of_points}",
        grid=grid,
        regions=(region,),
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
        condition_type="dirichlet",
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type="dirichlet",
        value=1.0,
        units="V",
    )

    return Simulation(
        name=f"benchmark_simulation_{number_of_points}",
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-8,
        max_iterations=100_000,
        initial_potential=0.0,
    )


def analytical_solution(
    simulation: Simulation,
) -> np.ndarray:
    """Return the exact linear potential solution."""

    coordinates = simulation.grid.coordinates(0)

    return (
        coordinates
        - coordinates[0]
    ) / (
        coordinates[-1]
        - coordinates[0]
    )


def run_single_benchmark(
    simulation: Simulation,
    linear_solver,
    *,
    repeats: int,
) -> BenchmarkRecord:
    """
    Benchmark one linear solver.

    The best elapsed time is retained to reduce interference from
    temporary operating-system activity.
    """

    elapsed_times: list[float] = []
    latest_result = None

    for _ in range(repeats):
        poisson_solver = PoissonSolver(
            linear_solver=linear_solver,
            name=(
                f"poisson_{linear_solver.name}_1d"
            ),
        )

        start_time = perf_counter()

        latest_result = poisson_solver.solve(
            simulation
        )

        elapsed_times.append(
            perf_counter() - start_time
        )

    if latest_result is None:
        raise RuntimeError(
            "Benchmark did not produce a result."
        )

    final_residual = latest_result.final_residual

    if final_residual is None:
        raise RuntimeError(
            "Benchmark result did not record a residual."
        )

    expected = analytical_solution(
        simulation
    )

    maximum_solution_error = float(
        np.max(
            np.abs(
                latest_result.potential.values
                - expected
            )
        )
    )

    return BenchmarkRecord(
        grid_points=(
            simulation.grid.number_of_points
        ),
        linear_solver=linear_solver.name,
        backend=linear_solver.backend_name,
        elapsed_seconds=min(elapsed_times),
        iterations=latest_result.iterations,
        final_residual=final_residual,
        converged=latest_result.converged,
        maximum_solution_error=(
            maximum_solution_error
        ),
    )


def run_benchmarks() -> tuple[BenchmarkRecord, ...]:
    """Run the complete solver-scaling benchmark."""

    grid_sizes = (
        101,
        501,
        1_001,
        2_001,
        5_001,
        10_001,
    )

    repeats = 3

    records: list[BenchmarkRecord] = []

    for number_of_points in grid_sizes:
        simulation = create_benchmark_simulation(
            number_of_points
        )
        # updated linear solvers
        linear_solvers = [
            SparseDirectSolver(),
            ConjugateGradientSolver(
                preconditioner=IdentityPreconditioner(),
                relative_tolerance=1.0e-10,
                absolute_tolerance=1.0e-12,
                max_iterations=100_000,
                name="cg_identity",
            ),
            ConjugateGradientSolver(
                preconditioner=JacobiPreconditioner(),
                relative_tolerance=1.0e-10,
                absolute_tolerance=1.0e-12,
                max_iterations=100_000,
                name="cg_jacobi",
            ),
        ]

        # Dense conversion becomes expensive quickly, so restrict it
        # to the smaller systems.
        if number_of_points <= 2_001:
            linear_solvers.insert(
                0,
                DenseDirectSolver(),
            )

        print(
            f"\nGrid points: {number_of_points:,}"
        )

        for linear_solver in linear_solvers:
            record = run_single_benchmark(
                simulation,
                linear_solver,
                repeats=repeats,
            )

            records.append(record)

            print(
                f"  {record.linear_solver:20s}"
                f" time={record.elapsed_seconds:.6e} s"
                f" iterations={record.iterations:6d}"
                f" residual={record.final_residual:.6e}"
                f" error={record.maximum_solution_error:.6e}"
                f" converged={record.converged}"
            )

    return tuple(records)


def create_output_directory() -> Path:
    """Create the benchmark-output directory."""

    repository_root = (
        Path(__file__).resolve().parents[1]
    )

    output_directory = (
        repository_root
        / "examples"
        / "figures"
        / "benchmarks"
        / "poisson_linear_solvers"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def plot_runtime_scaling(
    records: tuple[BenchmarkRecord, ...],
    output_directory: Path,
) -> Path:
    """Plot solve time against grid size."""

    figure, axis = plt.subplots()

    solver_names = sorted(
        {
            record.linear_solver
            for record in records
        }
    )

    for solver_name in solver_names:
        selected_records = [
            record
            for record in records
            if record.linear_solver == solver_name
        ]

        axis.loglog(
            [
                record.grid_points
                for record in selected_records
            ],
            [
                record.elapsed_seconds
                for record in selected_records
            ],
            marker="o",
            label=solver_name,
        )

    axis.set_xlabel("Grid points")
    axis.set_ylabel("Elapsed time (s)")
    axis.set_title(
        "1D Poisson Linear-Solver Scaling"
    )
    axis.grid(True)
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

    return output_path

# now plots both conjugate gradients
def plot_cg_iterations(
    records: tuple[BenchmarkRecord, ...],
    output_directory: Path,
) -> Path:
    """Plot CG iteration count against grid size."""

    figure, axis = plt.subplots()

    cg_solver_names = (
        "cg_identity",
        "cg_jacobi",
    )

    for solver_name in cg_solver_names:
        selected_records = [
            record
            for record in records
            if record.linear_solver == solver_name
        ]

        axis.plot(
            [
                record.grid_points
                for record in selected_records
            ],
            [
                record.iterations
                for record in selected_records
            ],
            marker="o",
            label=solver_name,
        )

    axis.set_xlabel("Grid points")
    axis.set_ylabel("CG iterations")
    axis.set_title(
        "Conjugate-Gradient Iteration Scaling"
    )
    axis.grid(True)
    axis.legend()

    figure.tight_layout()

    output_path = (
        output_directory
        / "cg_iteration_scaling.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    return output_path


def save_results(
    records: tuple[BenchmarkRecord, ...],
    output_directory: Path,
) -> Path:
    """Save benchmark records as CSV."""

    output_path = (
        output_directory
        / "benchmark_results.csv"
    )

    header = (
        "grid_points,"
        "linear_solver,"
        "backend,"
        "elapsed_seconds,"
        "iterations,"
        "final_residual,"
        "converged,"
        "maximum_solution_error"
    )

    rows = [header]

    for record in records:
        rows.append(
            ",".join(
                (
                    str(record.grid_points),
                    record.linear_solver,
                    record.backend,
                    f"{record.elapsed_seconds:.16e}",
                    str(record.iterations),
                    f"{record.final_residual:.16e}",
                    str(record.converged),
                    (
                        f"{record.maximum_solution_error:.16e}"
                    ),
                )
            )
        )

    output_path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    """Run, save, and display the benchmark."""

    records = run_benchmarks()

    output_directory = (
        create_output_directory()
    )

    results_path = save_results(
        records,
        output_directory,
    )

    runtime_plot_path = plot_runtime_scaling(
        records,
        output_directory,
    )

    iteration_plot_path = plot_cg_iterations(
        records,
        output_directory,
    )

    print("\nSaved benchmark outputs:")
    print(f"  {results_path}")
    print(f"  {runtime_plot_path}")
    print(f"  {iteration_plot_path}")

    plt.show()


if __name__ == "__main__":
    main()