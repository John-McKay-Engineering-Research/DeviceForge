"""Mesh-convergence study for the DeviceForge 1D PN-junction solver.

Run from the repository root with:

    python examples/verification/mesh_convergence_pn_junction.py
"""

from __future__ import annotations

import csv
import importlib.util
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TEST_GUMMEL_PATH = REPOSITORY_ROOT / "tests" / "test_gummel_1d.py"
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "examples"
    / "figures"
    / "verification"
    / "pn_junction_mesh_convergence"
)

APPLIED_VOLTAGE = 0.05
NODE_COUNTS = (41, 81, 161, 321, 641)

POTENTIAL_FIELD = "electrostatic_potential"
RECOMBINATION_FIELD = "shockley_read_hall_recombination_rate"


def load_gummel_test_module() -> ModuleType:
    if not TEST_GUMMEL_PATH.exists():
        raise FileNotFoundError(TEST_GUMMEL_PATH)

    spec = importlib.util.spec_from_file_location(
        "deviceforge_gummel_test_module",
        TEST_GUMMEL_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {TEST_GUMMEL_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUMMEL_TEST_MODULE = load_gummel_test_module()
build_pn_junction_simulation = GUMMEL_TEST_MODULE.build_pn_junction_simulation
GummelDriftDiffusionSolver1D = GUMMEL_TEST_MODULE.GummelDriftDiffusionSolver1D
SolverConfiguration = GUMMEL_TEST_MODULE.SolverConfiguration


@dataclass(frozen=True)
class MeshResult:
    nodes: int
    spacing_metres: float
    converged: bool
    iterations: int
    runtime_seconds: float
    average_terminal_current_density: float
    relative_current_nonuniformity: float
    poisson_residual: float
    electron_continuity_residual: float
    hole_continuity_residual: float
    peak_electric_field: float
    maximum_absolute_recombination: float
    relative_current_error: float = float("nan")


def metadata_float(result: Any, key: str) -> float:
    if key not in result.metadata:
        raise KeyError(f"Missing result metadata key: {key}")
    return float(result.metadata[key])


def run_mesh(number_of_nodes: int) -> MeshResult:
    simulation = build_pn_junction_simulation(shape=(number_of_nodes,))

    solver = GummelDriftDiffusionSolver1D(
        applied_voltage=APPLIED_VOLTAGE,
        damping_factor=0.2,
        current_conservation_tolerance=1.0e-2,
        enforce_current_conservation=True,
        configuration=SolverConfiguration(
            tolerance=1.0e-10,
            max_iterations=simulation.max_iterations,
        ),
    )

    result = solver.solve(simulation)

    spacing = float(simulation.grid.spacing[0])
    potential = result.get_field(POTENTIAL_FIELD).values
    electric_field = -np.diff(potential) / spacing

    recombination = result.get_field(RECOMBINATION_FIELD).values

    return MeshResult(
        nodes=number_of_nodes,
        spacing_metres=spacing,
        converged=bool(result.converged),
        iterations=int(result.iterations),
        runtime_seconds=float(result.runtime_seconds),
        average_terminal_current_density=metadata_float(
            result, "average_terminal_current_density"
        ),
        relative_current_nonuniformity=metadata_float(
            result, "relative_current_density_nonuniformity"
        ),
        poisson_residual=metadata_float(
            result, "final_poisson_residual"
        ),
        electron_continuity_residual=metadata_float(
            result, "final_electron_continuity_residual"
        ),
        hole_continuity_residual=metadata_float(
            result, "final_hole_continuity_residual"
        ),
        peak_electric_field=float(np.max(np.abs(electric_field))),
        maximum_absolute_recombination=float(
            np.max(np.abs(recombination))
        ),
    )


def add_reference_errors(results: list[MeshResult]) -> list[MeshResult]:
    reference_current = results[-1].average_terminal_current_density
    scale = max(abs(reference_current), np.finfo(np.float64).tiny)

    return [
        replace(
            result,
            relative_current_error=abs(
                result.average_terminal_current_density - reference_current
            )
            / scale,
        )
        for result in results
    ]


def print_summary(results: list[MeshResult]) -> None:
    print()
    print("DeviceForge PN-Junction Mesh-Convergence Study")
    print("=" * 106)
    print(f"Applied voltage: {APPLIED_VOLTAGE:.3f} V")
    print()
    print(
        f"{'Nodes':>6} {'dx (nm)':>10} {'Conv.':>7} {'Iter.':>7} "
        f"{'Javg (A/m^2)':>16} {'J rel. err.':>13} "
        f"{'J nonuniform.':>14} {'Runtime (s)':>12}"
    )
    print("-" * 106)

    for result in results:
        print(
            f"{result.nodes:6d} "
            f"{result.spacing_metres * 1.0e9:10.5f} "
            f"{str(result.converged):>7} "
            f"{result.iterations:7d} "
            f"{result.average_terminal_current_density:16.8e} "
            f"{result.relative_current_error:13.6e} "
            f"{result.relative_current_nonuniformity:14.6e} "
            f"{result.runtime_seconds:12.6f}"
        )
    print()


def save_csv(results: list[MeshResult]) -> Path:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIRECTORY / "mesh_convergence_results.csv"
    field_names = list(MeshResult.__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    return path


def save_plots(results: list[MeshResult]) -> list[Path]:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    nodes = np.asarray([r.nodes for r in results], dtype=float)
    spacing = np.asarray([r.spacing_metres for r in results], dtype=float)
    current = np.asarray(
        [r.average_terminal_current_density for r in results], dtype=float
    )
    current_error = np.asarray(
        [r.relative_current_error for r in results], dtype=float
    )
    runtime = np.asarray([r.runtime_seconds for r in results], dtype=float)
    nonuniformity = np.asarray(
        [r.relative_current_nonuniformity for r in results], dtype=float
    )

    paths: list[Path] = []

    path = OUTPUT_DIRECTORY / "01_current_vs_nodes.png"
    plt.figure(figsize=(8, 5))
    plt.plot(nodes, current, marker="o")
    plt.xlabel("Number of nodes")
    plt.ylabel("Average terminal current density (A/m²)")
    plt.title("PN-Junction Current Mesh Convergence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    paths.append(path)

    path = OUTPUT_DIRECTORY / "02_current_error_vs_spacing.png"
    positive = current_error > 0.0
    plt.figure(figsize=(8, 5))
    plt.loglog(spacing[positive], current_error[positive], marker="o")
    plt.xlabel("Grid spacing (m)")
    plt.ylabel("Relative current error")
    plt.title("Current Error Relative to Finest Mesh")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    paths.append(path)

    path = OUTPUT_DIRECTORY / "03_current_nonuniformity_vs_nodes.png"
    plt.figure(figsize=(8, 5))
    plt.semilogy(nodes, nonuniformity, marker="o")
    plt.xlabel("Number of nodes")
    plt.ylabel("Relative total-current nonuniformity")
    plt.title("Current Conservation Across Meshes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    paths.append(path)

    path = OUTPUT_DIRECTORY / "04_runtime_vs_nodes.png"
    plt.figure(figsize=(8, 5))
    plt.plot(nodes, runtime, marker="o")
    plt.xlabel("Number of nodes")
    plt.ylabel("Runtime (s)")
    plt.title("Solver Runtime Scaling")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    paths.append(path)

    return paths


def main() -> None:
    results: list[MeshResult] = []

    for nodes in NODE_COUNTS:
        print(f"Running {nodes}-node mesh...")
        result = run_mesh(nodes)
        results.append(result)

        if not result.converged:
            print(f"Warning: the {nodes}-node mesh did not converge.")

    results = add_reference_errors(results)
    print_summary(results)

    csv_path = save_csv(results)
    figure_paths = save_plots(results)

    print(f"Saved CSV: {csv_path}")
    for path in figure_paths:
        print(f"Saved figure: {path}")

    print()
    print("Mesh-convergence study completed.")


if __name__ == "__main__":
    main()