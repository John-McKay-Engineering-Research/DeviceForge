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
from deviceforge.physics import (
    DEFAULT_SILICON_INTRINSIC_CONCENTRATION,
    SILICON,
    charge_neutral_potential,
)
from deviceforge.postprocessing import compute_electric_field
from deviceforge.solvers import (
    EquilibriumPoissonSolver,
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
INTRINSIC_CONCENTRATION = (
    DEFAULT_SILICON_INTRINSIC_CONCENTRATION
)
TEMPERATURE = 300.0


def build_simulation(
    *,
    shape: tuple[int, int] = (101, 31),
    tolerance: float = 1.0e-8,
    max_iterations: int = 30_000,
) -> Simulation:
    """
    Create a self-consistent equilibrium abrupt PN junction.

    The left contact is assigned the charge-neutral p-type potential.
    The right contact is assigned the charge-neutral n-type potential.
    """

    grid = Grid(
        shape=shape,
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
        name="equilibrium_pn_junction",
        grid=grid,
        regions=(
            p_region,
            n_region,
        ),
    )

    p_contact_potential = float(
        charge_neutral_potential(
            -ACCEPTOR_DENSITY,
            intrinsic_concentration=INTRINSIC_CONCENTRATION,
            temperature=TEMPERATURE,
        )
    )

    n_contact_potential = float(
        charge_neutral_potential(
            DONOR_DENSITY,
            intrinsic_concentration=INTRINSIC_CONCENTRATION,
            temperature=TEMPERATURE,
        )
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
            name="p_contact",
            grid=grid,
            mask=left_mask,
            condition_type="dirichlet",
            value=p_contact_potential,
            units="V",
        ),
        BoundaryCondition(
            name="n_contact",
            grid=grid,
            mask=right_mask,
            condition_type="dirichlet",
            value=n_contact_potential,
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

    initial_potential = 0.5 * (
        p_contact_potential
        + n_contact_potential
    )

    return Simulation(
        name="self_consistent_equilibrium_pn_junction",
        device=device,
        boundary_conditions=boundaries,
        tolerance=tolerance,
        max_iterations=max_iterations,
        initial_potential=initial_potential,
    )


def create_signed_doping_field(
    simulation: Simulation,
) -> Field:
    """Return signed donor-minus-acceptor concentration."""

    return simulation.device.net_doping_field()


def calculate_built_in_potential(
    simulation: Simulation,
) -> float:
    """Return the difference between equilibrium contact potentials."""

    p_contact = simulation.get_boundary_condition(
        "p_contact"
    )

    n_contact = simulation.get_boundary_condition(
        "n_contact"
    )

    return n_contact.value - p_contact.value


def plot_equilibrium_profiles(
    simulation: Simulation,
    *,
    potential: Field,
    charge_density: Field,
    electron_density: Field,
    hole_density: Field,
    electric_field_x: Field,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot centre-line equilibrium PN-junction quantities."""

    x_nm = (
        simulation.grid.coordinates(axis=0)
        * 1.0e9
    )

    centre_column = (
        simulation.grid.shape[1] // 2
    )

    junction_index = (
        simulation.grid.shape[0] // 2
    )

    junction_position_nm = (
        simulation.grid.coordinates(axis=0)[
            junction_index
        ]
        * 1.0e9
    )

    figure, axes = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(9, 11),
    )

    axes[0].plot(
        x_nm,
        potential.values[:, centre_column],
    )
    axes[0].set_ylabel("Potential [V]")
    axes[0].set_title(
        "Self-consistent equilibrium PN junction"
    )
    axes[0].grid(True)

    axes[1].plot(
        x_nm,
        charge_density.values[:, centre_column],
    )
    axes[1].set_ylabel(
        "Charge density\n[C/m³]"
    )
    axes[1].grid(True)

    axes[2].semilogy(
        x_nm,
        electron_density.values[:, centre_column],
        label="Electrons",
    )
    axes[2].semilogy(
        x_nm,
        hole_density.values[:, centre_column],
        label="Holes",
    )
    axes[2].set_ylabel(
        "Carrier concentration\n[1/m³]"
    )
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(
        x_nm,
        electric_field_x.values[:, centre_column],
    )
    axes[3].set_ylabel(
        "Electric field\n[V/m]"
    )
    axes[3].set_xlabel("x [nm]")
    axes[3].grid(True)

    for axis in axes:
        axis.axvline(
            junction_position_nm,
            linestyle="--",
        )

    figure.tight_layout()

    return figure, axes


def print_summary(
    *,
    simulation: Simulation,
    result,
    built_in_potential: float,
) -> None:
    """Print equilibrium-junction solver information."""

    potential = result.potential.values

    charge_density = result.get_field(
        "equilibrium_charge_density"
    ).values

    electric_field = compute_electric_field(
        result.potential
    )

    maximum_field = float(
        np.max(
            electric_field.magnitude.values
        )
    )

    print()
    print(
        "DeviceForge Self-Consistent Equilibrium "
        "PN-Junction Demonstration"
    )
    print("=" * 68)
    print(f"Converged:             {result.converged}")
    print(f"Iterations:            {result.iterations}")
    print(
        f"Final update:          "
        f"{result.final_residual:.6e} V"
    )
    print(
        f"Runtime:               "
        f"{result.runtime_seconds:.6f} s"
    )
    print(
        f"Built-in potential:    "
        f"{built_in_potential:.6f} V"
    )
    print(
        f"Minimum potential:     "
        f"{np.min(potential):.6f} V"
    )
    print(
        f"Maximum potential:     "
        f"{np.max(potential):.6f} V"
    )
    print(
        f"Maximum electric field:"
        f" {maximum_field:.6e} V/m"
    )
    print(
        f"Maximum |charge|:      "
        f"{np.max(np.abs(charge_density)):.6e} C/m³"
    )
    print(
        f"Intrinsic concentration:"
        f" {INTRINSIC_CONCENTRATION:.6e} 1/m³"
    )
    print(f"Grid shape:            {simulation.grid.shape}")
    print(f"Solver:                {result.solver_name}")
    print(f"Backend:               {result.backend_name}")
    print()


def main() -> None:
    simulation = build_simulation()

    solver = EquilibriumPoissonSolver(
        damping_factor=0.5,
        maximum_potential_step=0.05,
        intrinsic_concentration=(
            INTRINSIC_CONCENTRATION
        ),
        temperature=TEMPERATURE,
        configuration=SolverConfiguration(
            tolerance=simulation.tolerance,
            max_iterations=simulation.max_iterations,
            backend_name="numpy",
        ),
    )

    result = solver.solve(simulation)

    electric_field = compute_electric_field(
        result.potential
    )

    doping = create_signed_doping_field(
        simulation
    )

    charge_density = result.get_field(
        "equilibrium_charge_density"
    )

    electron_density = result.get_field(
        "electron_concentration"
    )

    hole_density = result.get_field(
        "hole_concentration"
    )

    built_in_potential = calculate_built_in_potential(
        simulation
    )

    print_summary(
        simulation=simulation,
        result=result,
        built_in_potential=built_in_potential,
    )

    doping_figure, _ = plot_scalar_field(
        doping,
        title="Equilibrium PN-junction net doping",
        colourbar_label="Net doping [1/m³]",
        contour_levels=3,
    )

    potential_figure, _ = plot_scalar_field(
        result.potential,
        title="Equilibrium electrostatic potential",
        colourbar_label="Potential [V]",
        contour_levels=30,
    )

    charge_figure, _ = plot_scalar_field(
        charge_density,
        title="Equilibrium space-charge density",
        colourbar_label="Charge density [C/m³]",
        contour_levels=30,
    )

    electric_field_figure, _ = plot_scalar_field(
        electric_field.magnitude,
        title="Equilibrium electric-field magnitude",
        colourbar_label="Electric-field magnitude [V/m]",
        contour_levels=30,
    )

    vector_figure, _ = plot_vector_field(
        electric_field,
        title="Equilibrium electric-field vectors",
        stride=6,
    )

    profile_figure, _ = plot_equilibrium_profiles(
        simulation,
        potential=result.potential,
        charge_density=charge_density,
        electron_density=electron_density,
        hole_density=hole_density,
        electric_field_x=(
            electric_field.x_component
        ),
    )

    convergence_figure, _ = plot_convergence(
        result
    )

    save_figure(
        doping_figure,
        "figures/examples/equilibrium_pn_doping.png",
    )

    save_figure(
        potential_figure,
        "figures/examples/equilibrium_pn_potential.png",
    )

    save_figure(
        charge_figure,
        "figures/examples/equilibrium_pn_charge_density.png",
    )

    save_figure(
        electric_field_figure,
        "figures/examples/equilibrium_pn_electric_field.png",
    )

    save_figure(
        vector_figure,
        "figures/examples/equilibrium_pn_field_vectors.png",
    )

    save_figure(
        profile_figure,
        "figures/examples/equilibrium_pn_profiles.png",
    )

    save_figure(
        convergence_figure,
        "figures/examples/equilibrium_pn_convergence.png",
    )

    plt.show()


if __name__ == "__main__":
    main()