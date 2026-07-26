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

from deviceforge.postprocessing import (
    calculate_face_electrostatic_fields,
)
from deviceforge.visualisation import (
    plot_face_electric_displacement,
)


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

# updated with more robust checks
def verify_dielectric_interface(
    workflow: ElectrostaticWorkflow,
) -> None:
    """
    Evaluate electric-flux continuity across the dielectric interface.

    Two checks are reported:

    1. Region-averaged node-centred epsilon_r * E, evaluated away from
       the interface.

    2. Conservative face-centred electric displacement,

           D_(i+1/2) = epsilon_0 * epsilon_r_(i+1/2) * E_(i+1/2),

       using harmonic face permittivity and potential differences.

    The face-centred calculation matches the discretisation used by the
    Poisson solver and should remain nearly constant across the stack.
    """

    output = workflow.output

    if output is None:
        raise RuntimeError(
            "Workflow has not produced an output."
        )

    vacuum_permittivity = 8.8541878128e-12

    simulation = workflow.simulation
    grid = simulation.grid

    grid_spacing = grid.spacing[0]

    relative_permittivity = (
        simulation
        .device
        .relative_permittivity_field()
        .values
    )

    potential = output.potential.values
    node_electric_field = output.electric_field.values

    # Locate material transitions automatically.
    interface_faces = np.flatnonzero(
        ~np.isclose(
            relative_permittivity[:-1],
            relative_permittivity[1:],
        )
    )

    if interface_faces.size != 1:
        raise RuntimeError(
            "This example expects exactly one dielectric interface. "
            f"Detected {interface_faces.size}."
        )

    interface_face = int(interface_faces[0])

    # The material on the right begins at this node.
    right_region_start = interface_face + 1

    # Exclude several nodes around the interface so the centred
    # derivative does not sample both material regions.
    exclusion_width = 3

    left_node_slice = slice(
        1,
        max(
            2,
            right_region_start - exclusion_width,
        ),
    )

    right_node_slice = slice(
        min(
            grid.shape[0] - 1,
            right_region_start + exclusion_width,
        ),
        grid.shape[0] - 1,
    )

    # --------------------------------------------------------------
    # Node-centred epsilon_r E check
    # --------------------------------------------------------------

    relative_flux_nodes = (
        relative_permittivity
        * node_electric_field
    )

    left_relative_flux = float(
        np.mean(
            relative_flux_nodes[left_node_slice]
        )
    )

    right_relative_flux = float(
        np.mean(
            relative_flux_nodes[right_node_slice]
        )
    )

    relative_flux_reference = max(
        abs(left_relative_flux),
        abs(right_relative_flux),
        np.finfo(np.float64).tiny,
    )

    relative_flux_mismatch = (
        abs(
            left_relative_flux
            - right_relative_flux
        )
        / relative_flux_reference
    )

    # --------------------------------------------------------------
    # Conservative face-centred displacement check
    # --------------------------------------------------------------

    left_permittivity = relative_permittivity[:-1]
    right_permittivity = relative_permittivity[1:]

    face_relative_permittivity = (
        2.0
        * left_permittivity
        * right_permittivity
        / (
            left_permittivity
            + right_permittivity
        )
    )

    face_electric_field = -(
        potential[1:]
        - potential[:-1]
    ) / grid_spacing

    face_displacement = (
        vacuum_permittivity
        * face_relative_permittivity
        * face_electric_field
    )

    # Faces wholly inside each region. The interface face itself is
    # excluded from these regional averages.
    left_face_slice = slice(
        1,
        interface_face,
    )

    right_face_slice = slice(
        interface_face + 1,
        face_displacement.size - 1,
    )

    left_displacement = float(
        np.mean(
            face_displacement[left_face_slice]
        )
    )

    right_displacement = float(
        np.mean(
            face_displacement[right_face_slice]
        )
    )

    interface_displacement = float(
        face_displacement[interface_face]
    )

    displacement_reference = max(
        abs(left_displacement),
        abs(right_displacement),
        np.finfo(np.float64).tiny,
    )

    displacement_mismatch = (
        abs(
            left_displacement
            - right_displacement
        )
        / displacement_reference
    )

    maximum_face_deviation = float(
        np.max(
            np.abs(
                face_displacement
                - np.mean(face_displacement)
            )
        )
    )

    mean_face_displacement = float(
        np.mean(face_displacement)
    )

    relative_maximum_face_deviation = (
        maximum_face_deviation
        / max(
            abs(mean_face_displacement),
            np.finfo(np.float64).tiny,
        )
    )

    print("\nDielectric-interface flux verification:")

    print("\n  Node-centred epsilon_r E, away from interface:")
    print(
        "    Oxide mean:                "
        f"{left_relative_flux:.12e} V/m"
    )
    print(
        "    Silicon mean:              "
        f"{right_relative_flux:.12e} V/m"
    )
    print(
        "    Relative regional mismatch:"
        f" {relative_flux_mismatch:.12e}"
    )

    print("\n  Face-centred electric displacement:")
    print(
        "    Oxide mean:                "
        f"{left_displacement:.12e} C/m^2"
    )
    print(
        "    Interface face:            "
        f"{interface_displacement:.12e} C/m^2"
    )
    print(
        "    Silicon mean:              "
        f"{right_displacement:.12e} C/m^2"
    )
    print(
        "    Relative regional mismatch:"
        f" {displacement_mismatch:.12e}"
    )
    print(
        "    Maximum relative deviation:"
        f" {relative_maximum_face_deviation:.12e}"
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

    (
        face_electric_field,
        face_relative_permittivity,
        face_displacement,
    ) = calculate_face_electrostatic_fields(
        output.potential,
        relative_permittivity,
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
            "04a_node_centred_electric_displacement.png",
            plot_electric_displacement(
                output
            )[0],
        ),
        (
            # temporary added **
            # ***
            "04b_face_centred_electric_displacement.png",
            plot_face_electric_displacement(
                face_displacement
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