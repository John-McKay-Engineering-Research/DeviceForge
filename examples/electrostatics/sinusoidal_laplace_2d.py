from __future__ import annotations

from pathlib import Path

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
from deviceforge.visualisation import (
    plot_electric_field_magnitude_2d,
    plot_electric_field_vectors_2d,
    plot_electrostatic_potential_2d,
    plot_electrostatic_solution_2d,
    plot_equipotential_contours_2d,
)


def calculate_analytical_solution(
    grid: Grid,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Calculate the exact sinusoidal 2D Laplace solution.

    Returns
    -------
    tuple
        Potential, axis-0 electric field, axis-1 electric field,
        and electric-field magnitude.
    """

    coordinate_axis_0 = (
        grid.coordinates(0)
    )

    coordinate_axis_1 = (
        grid.coordinates(1)
    )

    coordinate_mesh_axis_0 = (
        coordinate_axis_0[:, None]
    )

    coordinate_mesh_axis_1 = (
        coordinate_axis_1[None, :]
    )

    domain_length_axis_0 = (
        coordinate_axis_0[-1]
        - coordinate_axis_0[0]
    )

    domain_length_axis_1 = (
        coordinate_axis_1[-1]
        - coordinate_axis_1[0]
    )

    shifted_axis_0 = (
        coordinate_mesh_axis_0
        - coordinate_axis_0[0]
    )

    shifted_axis_1 = (
        coordinate_mesh_axis_1
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

    electric_field_magnitude = np.hypot(
        electric_field_axis_0,
        electric_field_axis_1,
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

    electric_field_magnitude = np.broadcast_to(
        electric_field_magnitude,
        expected_shape,
    ).copy()

    return (
        potential,
        electric_field_axis_0,
        electric_field_axis_1,
        electric_field_magnitude,
    )


def create_simulation() -> tuple[
    Simulation,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Create the sinusoidal rectangular Laplace problem."""

    grid = Grid(
        shape=(101, 81),
        spacing=(
            1.0e-9,
            1.0e-9,
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
        name="sinusoidal_laplace_rectangle",
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
    bottom_mask[:, 0] = True

    top_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    top_mask[:, -1] = True

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
        name="sinusoidal_laplace_2d",
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


def create_output_directory() -> Path:
    """Create the example output directory."""

    repository_root = (
        Path(__file__).resolve().parents[2]
    )

    output_directory = (
        repository_root
        / "examples"
        / "figures"
        / "examples"
        / "sinusoidal_laplace_2d"
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
    """Save and close one figure."""

    output_path = (
        output_directory
        / filename
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


def create_figures(
    potential,
    electric_field_axis_0,
    electric_field_axis_1,
    electric_field_magnitude,
) -> tuple[Path, ...]:
    """Create and save the example figures."""

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
            number_of_levels=20,
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

    return tuple(
        saved_paths
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


def main() -> None:
    """Run the sinusoidal analytical 2D Laplace example."""

    print("=" * 72)
    print(
        "DeviceForge — sinusoidal analytical 2D Laplace example"
    )
    print("=" * 72)

    (
        simulation,
        analytical_potential,
        analytical_field_axis_0,
        analytical_field_axis_1,
        analytical_field_magnitude,
    ) = create_simulation()

    result = PoissonSolver2D().solve(
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

    potential_errors = (
        calculate_error_metrics(
            potential.values,
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
        f"Device:            {simulation.device.name}"
    )
    print(
        f"Grid shape:        {simulation.grid.shape}"
    )
    print(
        f"Grid points:       "
        f"{simulation.grid.number_of_points:,}"
    )

    print("\nSolver diagnostics:")
    print(
        f"  Solver:          {result.solver_name}"
    )
    print(
        f"  Backend:         {result.backend_name}"
    )
    print(
        f"  Converged:       {result.converged}"
    )
    print(
        f"  Iterations:      {result.iterations}"
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
        f"  Matrix nonzeros: "
        f"{result.metadata['matrix_nonzero_entries']:,}"
    )

    print("\nAnalytical verification:")
    print(
        "  Potential maximum error: "
        f"{potential_errors[0]:.6e} V"
    )
    print(
        "  Potential RMS error:     "
        f"{potential_errors[1]:.6e} V"
    )
    print(
        "  Axis-0 field max error:  "
        f"{field_axis_0_errors[0]:.6e} V/m"
    )
    print(
        "  Axis-0 field RMS error:  "
        f"{field_axis_0_errors[1]:.6e} V/m"
    )
    print(
        "  Axis-1 field max error:  "
        f"{field_axis_1_errors[0]:.6e} V/m"
    )
    print(
        "  Axis-1 field RMS error:  "
        f"{field_axis_1_errors[1]:.6e} V/m"
    )
    print(
        "  Magnitude max error:     "
        f"{magnitude_errors[0]:.6e} V/m"
    )
    print(
        "  Magnitude RMS error:     "
        f"{magnitude_errors[1]:.6e} V/m"
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