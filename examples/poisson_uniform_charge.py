from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from deviceforge import (
    BoundaryCondition,
    Device,
    Grid,
    Region,
    Simulation,
)
from deviceforge.physics import (
    ELEMENTARY_CHARGE,
    SILICON,
    VACUUM_PERMITTIVITY,
)
from deviceforge.postprocessing import compute_electric_field
from deviceforge.solvers import (
    PoissonJacobiSolver,
    SolverConfiguration,
)
from deviceforge.visualisation import (
    plot_convergence,
    plot_scalar_field,
    save_figure,
)


def build_simulation() -> Simulation:
    """Create a uniformly doped silicon Poisson problem."""

    grid = Grid(
        shape=(101, 41),
        spacing=(1.0e-9, 1.0e-9),
    )

    silicon_region = Region(
        name="uniform_n_type_silicon",
        grid=grid,
        material=SILICON,
        mask=np.ones(grid.shape, dtype=bool),
        donor_density=1.0e21,
    )

    device = Device(
        name="uniform_charge_device",
        grid=grid,
        regions=(silicon_region,),
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
            value=0.0,
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
        name="uniform_charge_poisson_demo",
        device=device,
        boundary_conditions=boundaries,
        tolerance=1.0e-10,
        max_iterations=100_000,
        initial_potential=0.0,
    )


def analytical_solution(
    simulation: Simulation,
    donor_density: float,
) -> np.ndarray:
    """Return the one-dimensional analytical Poisson solution."""

    x = simulation.grid.coordinates(axis=0)
    length = simulation.grid.physical_size[0]

    charge_density = ELEMENTARY_CHARGE * donor_density

    permittivity = (
        VACUUM_PERMITTIVITY
        * SILICON.relative_permittivity
    )

    source = charge_density / permittivity

    return 0.5 * source * x * (length - x)


def plot_analytical_comparison(
    simulation: Simulation,
    numerical_potential: np.ndarray,
    analytical_potential: np.ndarray,
):
    """Compare the numerical and analytical centre-line profiles."""

    x_nm = (
        simulation.grid.coordinates(axis=0)
        * 1.0e9
    )

    centre_column = numerical_potential[
        :,
        simulation.grid.shape[1] // 2,
    ]

    figure, axes = plt.subplots()

    axes.plot(
        x_nm,
        centre_column,
        label="DeviceForge numerical",
    )

    axes.plot(
        x_nm,
        analytical_potential,
        linestyle="--",
        label="Analytical solution",
    )

    axes.set_xlabel("x [nm]")
    axes.set_ylabel("Potential [V]")
    axes.set_title(
        "Numerical and analytical Poisson solution"
    )
    axes.grid(True)
    axes.legend()

    figure.tight_layout()

    return figure, axes


def plot_electric_field_profile(
    simulation: Simulation,
    electric_field_x: np.ndarray,
):
    """Plot the electric-field x-component through the device centre."""

    x_nm = (
        simulation.grid.coordinates(axis=0)
        * 1.0e9
    )

    centre_column = electric_field_x[
        :,
        simulation.grid.shape[1] // 2,
    ]

    figure, axes = plt.subplots()

    axes.plot(
        x_nm,
        centre_column,
    )

    axes.set_xlabel("x [nm]")
    axes.set_ylabel("Electric field [V/m]")
    axes.set_title("Electric-field profile")
    axes.grid(True)

    figure.tight_layout()

    return figure, axes


def print_summary(
    *,
    result,
    maximum_error: float,
) -> None:
    """Print the solver and validation summary."""

    print()
    print("DeviceForge Uniform-Charge Poisson Demonstration")
    print("=" * 52)
    print(f"Converged:          {result.converged}")
    print(f"Iterations:         {result.iterations}")
    print(f"Final residual:     {result.final_residual:.6e}")
    print(f"Runtime:            {result.runtime_seconds:.6f} s")
    print(f"Maximum error:      {maximum_error:.6e} V")
    print(f"Grid shape:         {result.grid.shape}")
    print(f"Solver:             {result.solver_name}")
    print(f"Backend:            {result.backend_name}")
    print()


def main() -> None:
    donor_density = 1.0e21

    simulation = build_simulation()

    solver = PoissonJacobiSolver(
        SolverConfiguration(
            tolerance=1.0e-10,
            max_iterations=100_000,
            backend_name="numpy",
        )
    )

    result = solver.solve(simulation)

    expected = analytical_solution(
        simulation,
        donor_density,
    )

    centre_column = result.potential.values[
        :,
        simulation.grid.shape[1] // 2,
    ]

    maximum_error = float(
        np.max(
            np.abs(
                centre_column - expected
            )
        )
    )

    electric_field = compute_electric_field(
        result.potential
    )

    print_summary(
        result=result,
        maximum_error=maximum_error,
    )

    potential_figure, _ = plot_scalar_field(
        result.potential,
        title="Uniform fixed-charge potential",
        colourbar_label="Potential [V]",
        contour_levels=25,
    )

    comparison_figure, _ = plot_analytical_comparison(
        simulation,
        result.potential.values,
        expected,
    )

    electric_field_figure, _ = (
        plot_electric_field_profile(
            simulation,
            electric_field.x_component.values,
        )
    )

    convergence_figure, _ = plot_convergence(
        result
    )

    save_figure(
        potential_figure,
        "figures/examples/poisson_uniform_potential.png",
    )

    save_figure(
        comparison_figure,
        "figures/examples/poisson_analytical_comparison.png",
    )

    save_figure(
        electric_field_figure,
        "figures/examples/poisson_electric_field.png",
    )

    save_figure(
        convergence_figure,
        "figures/examples/poisson_convergence.png",
    )

    plt.show()


if __name__ == "__main__":
    main()