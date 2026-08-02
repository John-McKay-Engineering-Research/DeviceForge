from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
from deviceforge.visualisation import (
    plot_electric_field_magnitude_2d,
    plot_electric_field_vectors_2d,
    plot_electrostatic_potential_2d,
    plot_electrostatic_solution_2d,
    plot_equipotential_contours_2d,
)


def create_simulation() -> tuple[
    Simulation,
    np.ndarray,
]:
    """
    Create a rectangular 2D Laplace problem.

    The exact solution is

        phi(x, y) = x / Lx.
    """

    grid = Grid(
        shape=(81, 61),
        spacing=(
            1.0e-9,
            1.5e-9,
        ),
        origin=(
            0.0,
            0.0,
        ),
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
        name="uniform_silicon_rectangle",
        grid=grid,
        regions=(region,),
    )

    coordinate_axis_0 = (
        grid.coordinates(0)
    )

    normalised_axis_0 = (
        coordinate_axis_0
        - coordinate_axis_0[0]
    ) / (
        coordinate_axis_0[-1]
        - coordinate_axis_0[0]
    )

    analytical_potential = np.broadcast_to(
        normalised_axis_0[:, None],
        grid.shape,
    ).copy()

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

    lower_axis_1_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    lower_axis_1_mask[1:-1, 0] = True

    upper_axis_1_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    upper_axis_1_mask[1:-1, -1] = True

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

    lower_boundary = BoundaryCondition(
        name="lower_linear_profile",
        grid=grid,
        mask=lower_axis_1_mask,
        condition_type="dirichlet",
        value=analytical_potential,
        units="V",
    )

    upper_boundary = BoundaryCondition(
        name="upper_linear_profile",
        grid=grid,
        mask=upper_axis_1_mask,
        condition_type="dirichlet",
        value=analytical_potential,
        units="V",
    )

    simulation = Simulation(
        name="analytical_laplace_2d",
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
            lower_boundary,
            upper_boundary,
        ),
        tolerance=1.0e-10,
        max_iterations=100_000,
        initial_potential=0.0,
    )

    return (
        simulation,
        analytical_potential,
    )


def create_output_directory() -> Path:
    """Create and return the example figure directory."""

    repository_root = (
        Path(__file__).resolve().parents[2]
    )

    output_directory = (
        repository_root
        / "examples"
        / "figures"
        / "examples"
        / "laplace_2d"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def save_figure(
    figure,
    output_directory: Path,
    filename: str,
) -> Path:
    """Save and close one Matplotlib figure."""

    output_path = (
        output_directory
        / filename
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def create_figures(
    potential,
    electric_field_axis_0,
    electric_field_axis_1,
    electric_field_magnitude,
) -> tuple[Path, ...]:
    """Create and save all 2D result figures."""

    output_directory = (
        create_output_directory()
    )

    saved_paths: list[Path] = []

    figure, _ = (
        plot_electrostatic_potential_2d(
            potential
        )
    )

    saved_paths.append(
        save_figure(
            figure,
            output_directory,
            "01_electrostatic_potential.png",
        )
    )

    figure, _ = (
        plot_equipotential_contours_2d(
            potential,
            number_of_levels=15,
        )
    )

    saved_paths.append(
        save_figure(
            figure,
            output_directory,
            "02_equipotential_contours.png",
        )
    )

    figure, _ = (
        plot_electric_field_magnitude_2d(
            electric_field_magnitude
        )
    )

    saved_paths.append(
        save_figure(
            figure,
            output_directory,
            "03_electric_field_magnitude.png",
        )
    )

    figure, _ = (
        plot_electric_field_vectors_2d(
            electric_field_axis_0,
            electric_field_axis_1,
            stride=5,
            normalise=True,
        )
    )

    saved_paths.append(
        save_figure(
            figure,
            output_directory,
            "04_electric_field_vectors.png",
        )
    )

    figure, _ = (
        plot_electrostatic_solution_2d(
            potential,
            electric_field_axis_0,
            electric_field_axis_1,
            electric_field_magnitude,
            vector_stride=5,
        )
    )

    saved_paths.append(
        save_figure(
            figure,
            output_directory,
            "05_electrostatic_solution_summary.png",
        )
    )

    return tuple(saved_paths)


def main() -> None:
    """Run the complete analytical 2D Laplace example."""

    print("=" * 72)
    print(
        "DeviceForge — analytical 2D Laplace example"
    )
    print("=" * 72)

    (
        simulation,
        analytical_potential,
    ) = create_simulation()

    solver = PoissonSolver2D()

    result = solver.solve(
        simulation
    )

    potential = result.potential

    (
        electric_field_axis_0,
        electric_field_axis_1,
        electric_field_magnitude,
    ) = calculate_electrostatic_fields_2d(
        potential
    )

    potential_error = np.abs(
        potential.values
        - analytical_potential
    )

    domain_length_axis_0 = (
        simulation.grid.coordinates(0)[-1]
        - simulation.grid.coordinates(0)[0]
    )

    expected_field_axis_0 = (
        -1.0
        / domain_length_axis_0
    )

    field_axis_0_error = np.abs(
        electric_field_axis_0.values
        - expected_field_axis_0
    )

    field_axis_1_error = np.abs(
        electric_field_axis_1.values
    )

    saved_paths = create_figures(
        potential,
        electric_field_axis_0,
        electric_field_axis_1,
        electric_field_magnitude,
    )

    print(
        f"Simulation:        {simulation.name}"
    )
    print(
        f"Device:            "
        f"{simulation.device.name}"
    )
    print(
        f"Grid shape:        "
        f"{simulation.grid.shape}"
    )
    print(
        f"Grid points:       "
        f"{simulation.grid.number_of_points:,}"
    )
    print(
        f"Grid spacing:      "
        f"{simulation.grid.spacing} m"
    )

    print("\nSolver diagnostics:")
    print(
        f"  Solver:          "
        f"{result.solver_name}"
    )
    print(
        f"  Backend:         "
        f"{result.backend_name}"
    )
    print(
        f"  Converged:       "
        f"{result.converged}"
    )
    print(
        f"  Iterations:      "
        f"{result.iterations}"
    )
    print(
        f"  Final residual:  "
        f"{result.final_residual:.6e}"
    )
    print(
        f"  Runtime:         "
        f"{result.runtime_seconds:.6e} s"
    )
    print(
        f"  Matrix shape:    "
        f"{result.metadata['matrix_shape']}"
    )
    print(
        f"  Matrix nonzeros: "
        f"{result.metadata['matrix_nonzero_entries']:,}"
    )

    print("\nAnalytical verification:")
    print(
        f"  Maximum potential error: "
        f"{np.max(potential_error):.6e} V"
    )
    print(
        f"  RMS potential error:     "
        f"{np.sqrt(np.mean(potential_error**2)):.6e} V"
    )
    print(
        f"  Expected axis-0 field:   "
        f"{expected_field_axis_0:.6e} V/m"
    )
    print(
        f"  Maximum axis-0 error:    "
        f"{np.max(field_axis_0_error):.6e} V/m"
    )
    print(
        f"  Maximum axis-1 field:    "
        f"{np.max(field_axis_1_error):.6e} V/m"
    )

    print("\nField ranges:")
    print(
        f"  Potential:       "
        f"{potential.minimum:.6e} to "
        f"{potential.maximum:.6e} V"
    )
    print(
        f"  Axis-0 field:    "
        f"{electric_field_axis_0.minimum:.6e} to "
        f"{electric_field_axis_0.maximum:.6e} V/m"
    )
    print(
        f"  Axis-1 field:    "
        f"{electric_field_axis_1.minimum:.6e} to "
        f"{electric_field_axis_1.maximum:.6e} V/m"
    )
    print(
        f"  Field magnitude: "
        f"{electric_field_magnitude.minimum:.6e} to "
        f"{electric_field_magnitude.maximum:.6e} V/m"
    )

    print("\nSaved figures:")

    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()