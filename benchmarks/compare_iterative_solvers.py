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
    SimulationResult,
)
from deviceforge.physics import SILICON
from deviceforge.solvers import (
    BaseSolver,
    GaussSeidelSolver,
    JacobiSolver,
    SORSolver,
    SolverConfiguration,
)


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """Summary of one iterative-solver benchmark run."""

    label: str
    result: SimulationResult
    analytical_max_error: float


def build_simulation(
    *,
    shape: tuple[int, int] = (101, 51),
    spacing: tuple[float, float] = (1.0e-9, 1.0e-9),
    tolerance: float = 1.0e-8,
    max_iterations: int = 50_000,
) -> Simulation:
    """Create the common rectangular Laplace benchmark."""

    grid = Grid(
        shape=shape,
        spacing=spacing,
    )

    region = Region(
        name="silicon_domain",
        grid=grid,
        material=SILICON,
        mask=np.ones(grid.shape, dtype=bool),
    )

    device = Device(
        name="iterative_solver_benchmark",
        grid=grid,
        regions=(region,),
    )

    left_mask = np.zeros(grid.shape, dtype=bool)
    left_mask[0, :] = True

    right_mask = np.zeros(grid.shape, dtype=bool)
    right_mask[-1, :] = True

    bottom_mask = np.zeros(grid.shape, dtype=bool)
    bottom_mask[1:-1, 0] = True

    top_mask = np.zeros(grid.shape, dtype=bool)
    top_mask[1:-1, -1] = True

    boundaries = (
        BoundaryCondition(
            name="left_contact",
            grid=grid,
            mask=left_mask,
            condition_type="dirichlet",
            value=0.0,
            units="V",
        ),
        BoundaryCondition(
            name="right_contact",
            grid=grid,
            mask=right_mask,
            condition_type="dirichlet",
            value=1.0,
            units="V",
        ),
        BoundaryCondition(
            name="bottom_insulated",
            grid=grid,
            mask=bottom_mask,
            condition_type="neumann",
            value=0.0,
            units="V/m",
        ),
        BoundaryCondition(
            name="top_insulated",
            grid=grid,
            mask=top_mask,
            condition_type="neumann",
            value=0.0,
            units="V/m",
        ),
    )

    return Simulation(
        name="iterative_solver_comparison",
        device=device,
        boundary_conditions=boundaries,
        tolerance=tolerance,
        max_iterations=max_iterations,
        initial_potential=0.5,
    )


def analytical_solution(simulation: Simulation) -> np.ndarray:
    """Return the expected linear potential field."""

    number_of_x_points, number_of_y_points = simulation.grid.shape

    x_profile = np.linspace(
        0.0,
        1.0,
        number_of_x_points,
    )

    return np.repeat(
        x_profile[:, np.newaxis],
        number_of_y_points,
        axis=1,
    )


def maximum_analytical_error(
    result: SimulationResult,
    expected: np.ndarray,
) -> float:
    """Return maximum absolute error against the analytical field."""

    return float(
        np.max(
            np.abs(
                result.potential.values
                - expected
            )
        )
    )


def run_solver(
    *,
    label: str,
    solver: BaseSolver,
    simulation: Simulation,
    expected: np.ndarray,
) -> BenchmarkRecord:
    """Run one solver and collect its benchmark metrics."""

    start_time = perf_counter()
    result = solver.solve(simulation)
    measured_runtime = perf_counter() - start_time

    if not np.isclose(
        measured_runtime,
        result.runtime_seconds,
        rtol=0.25,
        atol=1.0e-6,
    ):
        print(
            f"Warning: external runtime and recorded runtime differ "
            f"for {label}."
        )

    return BenchmarkRecord(
        label=label,
        result=result,
        analytical_max_error=maximum_analytical_error(
            result,
            expected,
        ),
    )


def print_summary(
    records: list[BenchmarkRecord],
) -> None:
    """Print a formatted benchmark table."""

    print()
    print("DeviceForge Iterative Solver Benchmark")
    print("=" * 96)

    header = (
        f"{'Solver':<18}"
        f"{'Converged':<12}"
        f"{'Iterations':>12}"
        f"{'Runtime [s]':>16}"
        f"{'Final residual':>20}"
        f"{'Max error [V]':>18}"
    )

    print(header)
    print("-" * 96)

    for record in records:
        result = record.result

        final_residual = (
            float("nan")
            if result.final_residual is None
            else result.final_residual
        )

        print(
            f"{record.label:<18}"
            f"{str(result.converged):<12}"
            f"{result.iterations:>12d}"
            f"{result.runtime_seconds:>16.6f}"
            f"{final_residual:>20.6e}"
            f"{record.analytical_max_error:>18.6e}"
        )

    print("=" * 96)
    print()


def save_convergence_plot(
    records: list[BenchmarkRecord],
    output_path: Path,
) -> None:
    """Save residual history for every solver."""

    figure, axes = plt.subplots()

    for record in records:
        iterations = np.arange(
            1,
            record.result.iterations + 1,
        )

        axes.semilogy(
            iterations,
            record.result.residual_history,
            label=record.label,
        )

    axes.set_xlabel("Iteration")
    axes.set_ylabel("Maximum potential change [V]")
    axes.set_title("Iterative solver convergence comparison")
    axes.grid(True)
    axes.legend()

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_runtime_plot(
    records: list[BenchmarkRecord],
    output_path: Path,
) -> None:
    """Save a runtime comparison bar chart."""

    labels = [record.label for record in records]
    runtimes = [
        record.result.runtime_seconds
        for record in records
    ]

    figure, axes = plt.subplots()

    axes.bar(
        labels,
        runtimes,
    )

    axes.set_ylabel("Runtime [s]")
    axes.set_title("Iterative solver runtime comparison")
    axes.tick_params(
        axis="x",
        rotation=30,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_iteration_plot(
    records: list[BenchmarkRecord],
    output_path: Path,
) -> None:
    """Save an iteration-count comparison bar chart."""

    labels = [record.label for record in records]
    iteration_counts = [
        record.result.iterations
        for record in records
    ]

    figure, axes = plt.subplots()

    axes.bar(
        labels,
        iteration_counts,
    )

    axes.set_ylabel("Iterations")
    axes.set_title("Iterations required for convergence")
    axes.tick_params(
        axis="x",
        rotation=30,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_error_plot(
    records: list[BenchmarkRecord],
    output_path: Path,
) -> None:
    """Save analytical-error comparison."""

    labels = [record.label for record in records]
    errors = [
        record.analytical_max_error
        for record in records
    ]

    figure, axes = plt.subplots()

    axes.bar(
        labels,
        errors,
    )

    axes.set_yscale("log")
    axes.set_ylabel("Maximum absolute error [V]")
    axes.set_title("Error against analytical solution")
    axes.tick_params(
        axis="x",
        rotation=30,
    )

    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    simulation = build_simulation()

    configuration = SolverConfiguration(
        tolerance=1.0e-8,
        max_iterations=50_000,
        backend_name="numpy",
    )

    solver_cases: list[tuple[str, BaseSolver]] = [
        (
            "Jacobi",
            JacobiSolver(configuration),
        ),
        (
            "Gauss-Seidel",
            GaussSeidelSolver(configuration),
        ),
        (
            "SOR omega=1.2",
            SORSolver(
                relaxation_factor=1.2,
                configuration=configuration,
            ),
        ),
        (
            "SOR omega=1.5",
            SORSolver(
                relaxation_factor=1.5,
                configuration=configuration,
            ),
        ),
        (
            "SOR omega=1.8",
            SORSolver(
                relaxation_factor=1.8,
                configuration=configuration,
            ),
        ),
    ]

    expected = analytical_solution(simulation)

    records = [
        run_solver(
            label=label,
            solver=solver,
            simulation=simulation,
            expected=expected,
        )
        for label, solver in solver_cases
    ]

    print_summary(records)

    output_directory = Path(
        "figures/benchmarks"
    )

    save_convergence_plot(
        records,
        output_directory
        / "solver_convergence_comparison.png",
    )

    save_runtime_plot(
        records,
        output_directory
        / "solver_runtime_comparison.png",
    )

    save_iteration_plot(
        records,
        output_directory
        / "solver_iteration_comparison.png",
    )

    save_error_plot(
        records,
        output_directory
        / "solver_error_comparison.png",
    )

    print(
        "Benchmark figures saved to "
        f"{output_directory.resolve()}"
    )


if __name__ == "__main__":
    main()