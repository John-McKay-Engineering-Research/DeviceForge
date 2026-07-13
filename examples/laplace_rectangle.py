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
from deviceforge.physics import SILICON
from deviceforge.solvers import (
    JacobiSolver,
    SolverConfiguration,
)
from deviceforge.visualisation import (
    plot_convergence,
    plot_scalar_field,
    plot_vector_field,
    save_figure,
)

from deviceforge.postprocessing import compute_electric_field

def build_simulation() -> Simulation:
    """Create a rectangular two-dimensional Laplace problem."""

    grid = Grid(
        shape=(101, 51),
        spacing=(1.0e-9, 1.0e-9),
    )

    region = Region(
        name="silicon_domain",
        grid=grid,
        material=SILICON,
        mask=np.ones(grid.shape, dtype=bool),
    )

    device = Device(
        name="rectangular_laplace_device",
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
        name="two_dimensional_laplace_rectangle",
        device=device,
        boundary_conditions=boundaries,
        tolerance=1.0e-8,
        max_iterations=50_000,
        initial_potential=0.5,
    )


def print_summary(result) -> None:
    """Print a concise numerical summary."""

    print()
    print("DeviceForge Jacobi Demonstration")
    print("=" * 36)
    print(f"Converged:       {result.converged}")
    print(f"Iterations:      {result.iterations}")
    print(f"Final residual:  {result.final_residual:.6e}")
    print(f"Runtime:         {result.runtime_seconds:.6f} s")
    print(f"Solver:          {result.solver_name}")
    print(f"Backend:         {result.backend_name}")
    print(f"Grid shape:      {result.grid.shape}")
    print()


def main() -> None:
    simulation = build_simulation()

    solver = JacobiSolver(
        SolverConfiguration(
            tolerance=1.0e-8,
            max_iterations=50_000,
            backend_name="numpy",
        )
    )

    result = solver.solve(simulation)
    electric_field = compute_electric_field(
        result.potential
    )


    print_summary(result)

    potential_figure, _ = plot_scalar_field(
        result.potential,
        title="Electrostatic potential",
        colourbar_label="Potential [V]",
        contour_levels=25,
    )

    convergence_figure, _ = plot_convergence(result)

    # new lots
    field_magnitude_figure, _ = plot_scalar_field(
        electric_field.magnitude,
        title="Electric-field magnitude",
        colourbar_label="Electric-field magnitude [V/m]",
        contour_levels=25,
    )

    field_vector_figure, _ = plot_vector_field(
        electric_field,
        title="Electric-field vectors",
        stride=5,
    )

    save_figure(
        potential_figure,
        "figures/examples/laplace_potential.png",
    )

    save_figure(
        convergence_figure,
        "figures/examples/jacobi_convergence.png",
    )

    # new figures
    save_figure(
        field_magnitude_figure,
        "figures/examples/electric_field_magnitude.png",
    )

    save_figure(
        field_vector_figure,
        "figures/examples/electric_field_vectors.png",
    )


    plt.show()


if __name__ == "__main__":
    main()