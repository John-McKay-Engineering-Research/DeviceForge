from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from deviceforge import (
    BoundaryCondition,
    BoundaryConditionType,
    Device,
    Grid,
    Region,
    Simulation,
)
from deviceforge.physics import (
    SILICON,
    SILICON_DIOXIDE,
)
from deviceforge.solvers import PoissonSolver
from deviceforge.visualisation import (
    plot_electric_displacement,
    plot_electric_field,
    plot_electrostatic_energy_density,
    plot_electrostatic_potential,
    plot_relative_permittivity,
    plot_residual_history,
)
from deviceforge.workflows import ElectrostaticWorkflow

from pathlib import Path

def create_dielectric_stack_simulation() -> Simulation:
    """
    Create a one-dimensional silicon-dioxide/silicon dielectric stack.

    The model contains:

        silicon dioxide | silicon
        0 V             |       1 V

    The domain is charge-free, so the solver evaluates

        d/dx(epsilon * dphi/dx) = 0

    with material-dependent permittivity.
    """

    number_of_points = 101
    grid_spacing = 1.0e-9

    grid = Grid(
        shape=(number_of_points,),
        spacing=(grid_spacing,),
    )

    interface_index = number_of_points // 2

    oxide_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    oxide_mask[:interface_index] = True

    silicon_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    silicon_mask[interface_index:] = True

    oxide_region = Region(
        name="silicon_dioxide",
        grid=grid,
        material=SILICON_DIOXIDE,
        mask=oxide_mask,
    )

    silicon_region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=silicon_mask,
    )

    device = Device(
        name="silicon_dioxide_silicon_stack",
        grid=grid,
        regions=(
            oxide_region,
            silicon_region,
        ),
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
        condition_type=BoundaryConditionType.DIRICHLET,
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=1.0,
        units="V",
    )

    return Simulation(
        name="dielectric_stack_1d",
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-10,
        max_iterations=1_000,
        initial_potential=0.0,
    )


def print_simulation_summary(
    workflow: ElectrostaticWorkflow,
) -> None:
    """Print the configured simulation and material information."""

    simulation = workflow.simulation
    grid = simulation.grid

    print("=" * 72)
    print("DeviceForge — 1D dielectric-stack electrostatic example")
    print("=" * 72)

    print(f"Workflow:          {workflow.name}")
    print(f"Simulation:        {simulation.name}")
    print(f"Device:            {simulation.device.name}")
    print(f"Grid points:       {grid.number_of_points}")
    print(f"Grid spacing:      {grid.spacing[0]:.3e} m")
    print(f"Domain length:     {grid.physical_size[0]:.3e} m")
    print(f"Number of regions: {simulation.device.number_of_regions}")

    print("\nMaterial regions:")

    for region in simulation.device.regions:
        number_of_region_points = int(
            np.count_nonzero(region.mask)
        )

        print(
            f"  {region.name}: "
            f"{region.material.name}, "
            f"epsilon_r={region.material.relative_permittivity:g}, "
            f"points={number_of_region_points}"
        )


def print_result_summary(
    workflow: ElectrostaticWorkflow,
) -> None:
    """Print workflow and solver diagnostics."""

    output = workflow.output

    if output is None:
        raise RuntimeError(
            "Workflow has not produced an output."
        )

    print("\nSolver diagnostics:")
    print(f"  Solver:          {output.solver_name}")
    print(f"  Backend:         {output.backend_name}")
    print(f"  Converged:       {output.converged}")
    print(f"  Iterations:      {output.iterations}")
    print(f"  Final residual:  {output.final_residual}")
    print(f"  Runtime:         {output.runtime_seconds:.6e} s")

    print("\nField ranges:")
    print(
        "  Potential:       "
        f"{output.potential.minimum:.6e} to "
        f"{output.potential.maximum:.6e} V"
    )
    print(
        "  Electric field:  "
        f"{output.electric_field.minimum:.6e} to "
        f"{output.electric_field.maximum:.6e} V/m"
    )
    print(
        "  Displacement:    "
        f"{output.electric_displacement.minimum:.6e} to "
        f"{output.electric_displacement.maximum:.6e} C/m^2"
    )
    print(
        "  Energy density:  "
        f"{output.energy_density.minimum:.6e} to "
        f"{output.energy_density.maximum:.6e} J/m^3"
    )


def verify_dielectric_interface(
    workflow: ElectrostaticWorkflow,
) -> None:
    """
    Print a simple displacement-continuity check.

    Values close to one indicate that the normal electric displacement is
    approximately continuous across the material interface.
    """

    output = workflow.output

    if output is None:
        raise RuntimeError(
            "Workflow has not produced an output."
        )

    number_of_points = (
        workflow.simulation.grid.shape[0]
    )
    interface_index = number_of_points // 2

    displacement = (
        output.electric_displacement.values
    )

    oxide_displacement = float(
        np.mean(
            displacement[
                max(1, interface_index - 10):
                interface_index - 2
            ]
        )
    )

    silicon_displacement = float(
        np.mean(
            displacement[
                interface_index + 2:
                min(
                    number_of_points - 1,
                    interface_index + 10,
                )
            ]
        )
    )

    if silicon_displacement == 0.0:
        displacement_ratio = np.inf
    else:
        displacement_ratio = (
            oxide_displacement
            / silicon_displacement
        )

    print("\nDielectric-interface check:")
    print(
        "  Mean oxide displacement:   "
        f"{oxide_displacement:.6e} C/m^2"
    )
    print(
        "  Mean silicon displacement: "
        f"{silicon_displacement:.6e} C/m^2"
    )
    print(
        "  Oxide/silicon ratio:       "
        f"{displacement_ratio:.6f}"
    )

# output directory helper function

def create_figure_output_directory() -> Path:
    """
    Create and return the dielectric-stack figure directory.

    The directory is resolved relative to the repository's examples folder:

        examples/figures/examples/dielectric_stack_1d
    """

    examples_directory = Path(__file__).resolve().parents[1]

    output_directory = (
        examples_directory
        / "figures"
        / "examples"
        / "dielectric_stack_1d"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory

# replaced create_figures function to output figures to examples.
# figures are currently 300 dpi

def create_figures(
    workflow: ElectrostaticWorkflow,
) -> tuple[Path, ...]:
    """
    Create and save all electrostatic result figures.

    Returns
    -------
    tuple[Path, ...]
        Paths of all saved figure files.
    """

    output = workflow.output

    if output is None:
        raise RuntimeError(
            "Workflow has not produced an output."
        )

    output_directory = (
        create_figure_output_directory()
    )

    relative_permittivity = (
        workflow
        .simulation
        .device
        .relative_permittivity_field()
    )

    figures = (
        (
            "01_relative_permittivity.png",
            plot_relative_permittivity(
                output,
                relative_permittivity,
            )[0],
        ),
        (
            "02_electrostatic_potential.png",
            plot_electrostatic_potential(
                output
            )[0],
        ),
        (
            "03_electric_field.png",
            plot_electric_field(
                output
            )[0],
        ),
        (
            "04_electric_displacement.png",
            plot_electric_displacement(
                output
            )[0],
        ),
        (
            "05_electrostatic_energy_density.png",
            plot_electrostatic_energy_density(
                output
            )[0],
        ),
        (
            "06_solver_residual_history.png",
            plot_residual_history(
                output
            )[0],
        ),
    )

    saved_paths: list[Path] = []

    for filename, figure in figures:
        output_path = output_directory / filename

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        saved_paths.append(
            output_path
        )

    return tuple(saved_paths)


def main() -> None:
    """Run the complete dielectric-stack demonstration."""

    simulation = (
        create_dielectric_stack_simulation()
    )

    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
        name="dielectric_stack_workflow",
    )

    print_simulation_summary(
        workflow
    )

    workflow.run()

    print_result_summary(
        workflow
    )

    verify_dielectric_interface(
        workflow
    )

    # now shows figures and saves them to path
    saved_figure_paths = create_figures(
        workflow
    )

    print("\nSaved figures:")

    for figure_path in saved_figure_paths:
        print(
            f"  {figure_path}"
        )

    plt.show()


if __name__ == "__main__":
    main()