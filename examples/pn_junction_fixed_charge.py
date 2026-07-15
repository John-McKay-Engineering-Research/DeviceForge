from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from deviceforge import (
    BoundaryCondition,
    Device,
    Field,
    Grid,
    Region,
    Simulation,
)
from deviceforge.physics import SILICON
from deviceforge.postprocessing import compute_electric_field
from deviceforge.solvers import (
    PoissonJacobiSolver,
    SolverConfiguration,
)
from deviceforge.visualisation import (
    plot_convergence,
    plot_scalar_field,
    plot_vector_field,
    save_figure,
)


ACCEPTOR_DENSITY = 1.0e21
DONOR_DENSITY = 1.0e21


def build_simulation() -> Simulation:
    """Create a simplified abrupt fixed-charge PN junction."""

    grid = Grid(
        shape=(201, 61),
        spacing=(1.0e-9, 1.0e-9),
    )

    junction_index = grid.shape[0] // 2

    p_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    p_mask[:junction_index, :] = True

    n_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    n_mask[junction_index:, :] = True

    p_region = Region(
        name="p_type_silicon",
        grid=grid,
        material=SILICON,
        mask=p_mask,
        acceptor_density=ACCEPTOR_DENSITY,
        region_type="semiconductor",
    )

    n_region = Region(
        name="n_type_silicon",
        grid=grid,
        material=SILICON,
        mask=n_mask,
        donor_density=DONOR_DENSITY,
        region_type="semiconductor",
    )

    device = Device(
        name="fixed_charge_pn_junction",
        grid=grid,
        regions=(
            p_region,
            n_region,
        ),
    )

    left_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    left_mask[0, :] = True

    right_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    right_mask[-1, :] = True

    bottom_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    bottom_mask[1:-1, 0] = True

    top_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
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
        name="simplified_fixed_charge_pn_junction",
        device=device,
        boundary_conditions=boundaries,
        tolerance=1.0e-10,
        max_iterations=150_000,
        initial_potential=0.0,
    )


def create_signed_doping_field(
    simulation: Simulation,
) -> Field:
    """
    Create signed net doping for visualisation.

    Positive values represent donor-dominated n-type material.
    Negative values represent acceptor-dominated p-type material.
    """

    return simulation.device.net_doping_field()


def plot_centreline_profiles(
    simulation: Simulation,
    *,
    doping: Field,
    charge_density: Field,
    potential: Field,
    electric_field_x: Field,
):
    """Plot centre-line physical quantities across the junction."""

    x_nm = (
        simulation.grid.coordinates(axis=0)
        * 1.0e9
    )

    centre_column = (
        simulation.grid.shape[1] // 2
    )

    junction_position_nm = (
        simulation.grid.coordinates(axis=0)[
            simulation.grid.shape[0] // 2
        ]
        * 1.0e9
    )

    figure, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(8, 10),
    )

    axes[0].plot(
        x_nm,
        doping.values[:, centre_column],
    )
    axes[0].set_ylabel("Net doping\n[1/m³]")
    axes[0].set_title(
        "Simplified fixed-charge PN-junction profiles"
    )
    axes[0].grid(True)

    axes[1].plot(
        x_nm,
        charge_density.values[:, centre_column],
    )
    axes[1].set_ylabel("Charge density\n[C/m³]")
    axes[1].grid(True)

    axes[2].plot(
        x_nm,
        potential.values[:, centre_column],
    )
    axes[2].set_ylabel("Potential\n[V]")
    axes[2].grid(True)

    axes[3].plot(
        x_nm,
        electric_field_x.values[:, centre_column],
    )
    axes[3].set_ylabel("Electric field\n[V/m]")
    axes[3].set_xlabel("x [nm]")
    axes[3].grid(True)

    for axis in axes:
        axis.axvline(
            junction_position_nm,
            linestyle="--",
            label="Junction",
        )

    axes[0].legend()

    figure.tight_layout()

    return figure, axes


def print_summary(
    *,
    result,
    junction_index: int,
) -> None:
    """Print important PN-junction demonstration values."""

    potential = result.potential.values
    charge_density = result.get_field(
        "fixed_charge_density"
    ).values

    centre_column = potential.shape[1] // 2

    print()
    print("DeviceForge Fixed-Charge PN-Junction Demonstration")
    print("=" * 56)
    print(f"Converged:             {result.converged}")
    print(f"Iterations:            {result.iterations}")
    print(
        f"Final residual:        "
        f"{result.final_residual:.6e}"
    )
    print(
        f"Runtime:               "
        f"{result.runtime_seconds:.6f} s"
    )
    print(
        f"Minimum potential:     "
        f"{np.min(potential):.6e} V"
    )
    print(
        f"Maximum potential:     "
        f"{np.max(potential):.6e} V"
    )
    print(
        f"Junction potential:    "
        f"{potential[junction_index, centre_column]:.6e} V"
    )
    print(
        f"P-side charge:         "
        f"{charge_density[junction_index - 1, centre_column]:.6e} C/m³"
    )
    print(
        f"N-side charge:         "
        f"{charge_density[junction_index, centre_column]:.6e} C/m³"
    )
    print(f"Grid shape:            {result.grid.shape}")
    print(f"Solver:                {result.solver_name}")
    print(f"Backend:               {result.backend_name}")
    print()


def main() -> None:
    simulation = build_simulation()

    solver = PoissonJacobiSolver(
        SolverConfiguration(
            tolerance=1.0e-10,
            max_iterations=150_000,
            backend_name="numpy",
        )
    )

    result = solver.solve(simulation)

    electric_field = compute_electric_field(
        result.potential
    )

    doping = create_signed_doping_field(
        simulation
    )

    charge_density = result.get_field(
        "fixed_charge_density"
    )

    junction_index = (
        simulation.grid.shape[0] // 2
    )

    print_summary(
        result=result,
        junction_index=junction_index,
    )

    doping_figure, _ = plot_scalar_field(
        doping,
        title="Signed net doping",
        colourbar_label="Net doping [1/m³]",
        contour_levels=3,
    )

    charge_figure, _ = plot_scalar_field(
        charge_density,
        title="Fixed charge density",
        colourbar_label="Charge density [C/m³]",
        contour_levels=20,
    )

    potential_figure, _ = plot_scalar_field(
        result.potential,
        title="PN-junction electrostatic potential",
        colourbar_label="Potential [V]",
        contour_levels=30,
    )

    field_magnitude_figure, _ = plot_scalar_field(
        electric_field.magnitude,
        title="PN-junction electric-field magnitude",
        colourbar_label="Electric-field magnitude [V/m]",
        contour_levels=30,
    )

    field_vector_figure, _ = plot_vector_field(
        electric_field,
        title="PN-junction electric-field vectors",
        stride=8,
    )

    profile_figure, _ = plot_centreline_profiles(
        simulation,
        doping=doping,
        charge_density=charge_density,
        potential=result.potential,
        electric_field_x=electric_field.x_component,
    )

    convergence_figure, _ = plot_convergence(
        result
    )

    save_figure(
        doping_figure,
        "figures/examples/pn_junction_doping.png",
    )

    save_figure(
        charge_figure,
        "figures/examples/pn_junction_charge_density.png",
    )

    save_figure(
        potential_figure,
        "figures/examples/pn_junction_potential.png",
    )

    save_figure(
        field_magnitude_figure,
        "figures/examples/pn_junction_electric_field_magnitude.png",
    )

    save_figure(
        field_vector_figure,
        "figures/examples/pn_junction_electric_field_vectors.png",
    )

    save_figure(
        profile_figure,
        "figures/examples/pn_junction_profiles.png",
    )

    save_figure(
        convergence_figure,
        "figures/examples/pn_junction_convergence.png",
    )

    plt.show()


if __name__ == "__main__":
    main()