from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deviceforge import (
    Device,
    Grid,
    Region,
)
from deviceforge.core.boundary import (
    BoundaryCondition,
    BoundaryConditionType,
)
from deviceforge.core.simulation import Simulation
from deviceforge.physics.materials import Material
from deviceforge.postprocessing import (
    calculate_face_electrostatic_fields_2d,
)
from deviceforge.solvers import PoissonSolver2D


NUMBER_OF_POINTS_AXIS_0 = 41
NUMBER_OF_POINTS_AXIS_1 = 21

DOMAIN_LENGTH_AXIS_0 = 4.0e-8
DOMAIN_LENGTH_AXIS_1 = 2.0e-8

LOW_RELATIVE_PERMITTIVITY = 4.0
HIGH_RELATIVE_PERMITTIVITY = 12.0

LEFT_POTENTIAL = 0.0
RIGHT_POTENTIAL = 1.0

OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "verification"
    / "poisson_2d_dielectric_interface"
)


def create_dielectric_interface_simulation_2d(
) -> tuple[
    Simulation,
    int,
]:
    """
    Create a charge-free two-material dielectric-interface problem.

    The material interface is normal to axis 0.

    Boundary conditions:

        left:   phi = 0 V
        right:  phi = 1 V
        bottom: d(phi)/dn = 0 V/m
        top:    d(phi)/dn = 0 V/m

    With zero free charge, the normal electric displacement should
    remain constant through the dielectric interface.
    """

    spacing_axis_0 = (
        DOMAIN_LENGTH_AXIS_0
        / (NUMBER_OF_POINTS_AXIS_0 - 1)
    )

    spacing_axis_1 = (
        DOMAIN_LENGTH_AXIS_1
        / (NUMBER_OF_POINTS_AXIS_1 - 1)
    )

    grid = Grid(
        shape=(
            NUMBER_OF_POINTS_AXIS_0,
            NUMBER_OF_POINTS_AXIS_1,
        ),
        spacing=(
            spacing_axis_0,
            spacing_axis_1,
        ),
    )

    low_permittivity_material = Material(
        name="low_permittivity_dielectric",
        relative_permittivity=(
            LOW_RELATIVE_PERMITTIVITY
        ),
        material_type="dielectric",
    )

    high_permittivity_material = Material(
        name="high_permittivity_dielectric",
        relative_permittivity=(
            HIGH_RELATIVE_PERMITTIVITY
        ),
        material_type="dielectric",
    )

    split_index = (
        NUMBER_OF_POINTS_AXIS_0
        // 2
        + 1
    )

    low_permittivity_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )

    high_permittivity_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )

    low_permittivity_mask[
        :split_index,
        :
    ] = True

    high_permittivity_mask[
        split_index:,
        :
    ] = True

    low_permittivity_region = Region(
        name="low_permittivity_region",
        grid=grid,
        material=low_permittivity_material,
        mask=low_permittivity_mask,
    )

    high_permittivity_region = Region(
        name="high_permittivity_region",
        grid=grid,
        material=high_permittivity_material,
        mask=high_permittivity_mask,
    )

    device = Device(
        name="dielectric_interface_device_2d",
        grid=grid,
        regions=(
            low_permittivity_region,
            high_permittivity_region,
        ),
    )

    left_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )

    right_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )

    bottom_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )

    top_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )

    left_mask[0, :] = True
    right_mask[-1, :] = True

    # Dirichlet boundaries own the corner nodes.
    bottom_mask[1:-1, 0] = True
    top_mask[1:-1, -1] = True

    left_boundary = BoundaryCondition(
        name="left_contact",
        grid=grid,
        mask=left_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=LEFT_POTENTIAL,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type=(
            BoundaryConditionType.DIRICHLET
        ),
        value=RIGHT_POTENTIAL,
        units="V",
    )

    bottom_boundary = BoundaryCondition(
        name="bottom_zero_normal_derivative",
        grid=grid,
        mask=bottom_mask,
        condition_type=(
            BoundaryConditionType.NEUMANN
        ),
        value=0.0,
        units="V/m",
    )

    top_boundary = BoundaryCondition(
        name="top_zero_normal_derivative",
        grid=grid,
        mask=top_mask,
        condition_type=(
            BoundaryConditionType.NEUMANN
        ),
        value=0.0,
        units="V/m",
    )

    simulation = Simulation(
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
            bottom_boundary,
            top_boundary,
        ),
        tolerance=1.0e-10,
        max_iterations=500,
        initial_potential=0.0,
        name="dielectric_interface_2d",
    )

    return (
        simulation,
        split_index,
    )


def run_verification(
) -> tuple[
    Simulation,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
]:
    """
    Solve the dielectric-interface problem and calculate diagnostics.
    """

    simulation, split_index = (
        create_dielectric_interface_simulation_2d()
    )

    result = PoissonSolver2D().solve(
        simulation
    )

    if not result.converged:
        raise RuntimeError(
            "PoissonSolver2D did not converge for the "
            "dielectric-interface verification problem."
        )

    relative_permittivity = (
        simulation.device.relative_permittivity_field()
    )

    (
        electric_field_axis_0,
        _,
        face_relative_permittivity_axis_0,
        _,
        displacement_axis_0,
        _,
    ) = calculate_face_electrostatic_fields_2d(
        result.potential,
        relative_permittivity,
    )

    centre_axis_1 = (
        simulation.device.grid.shape[1]
        // 2
    )

    potential_profile = (
        result.potential.values[
            :,
            centre_axis_1,
        ].copy()
    )

    electric_field_profile = (
        electric_field_axis_0.values[
            :,
            centre_axis_1,
        ].copy()
    )

    face_permittivity_profile = (
        face_relative_permittivity_axis_0.values[
            :,
            centre_axis_1,
        ].copy()
    )

    displacement_profile = (
        displacement_axis_0.values[
            :,
            centre_axis_1,
        ].copy()
    )

    mean_displacement = float(
        np.mean(
            displacement_axis_0.values
        )
    )

    maximum_displacement_deviation = float(
        np.max(
            np.abs(
                displacement_axis_0.values
                - mean_displacement
            )
        )
    )

    relative_displacement_deviation = (
        maximum_displacement_deviation
        / abs(mean_displacement)
    )

    low_material_faces = np.isclose(
        face_relative_permittivity_axis_0.values,
        LOW_RELATIVE_PERMITTIVITY,
    )

    high_material_faces = np.isclose(
        face_relative_permittivity_axis_0.values,
        HIGH_RELATIVE_PERMITTIVITY,
    )

    low_material_field = float(
        np.mean(
            np.abs(
                electric_field_axis_0.values[
                    low_material_faces
                ]
            )
        )
    )

    high_material_field = float(
        np.mean(
            np.abs(
                electric_field_axis_0.values[
                    high_material_faces
                ]
            )
        )
    )

    field_ratio = (
        low_material_field
        / high_material_field
    )

    return (
        simulation,
        split_index,
        potential_profile,
        electric_field_profile,
        face_permittivity_profile,
        displacement_profile,
        mean_displacement,
        maximum_displacement_deviation,
        relative_displacement_deviation,
        low_material_field,
        high_material_field,
        field_ratio,
    )


def print_results(
    simulation: Simulation,
    mean_displacement: float,
    maximum_displacement_deviation: float,
    relative_displacement_deviation: float,
    low_material_field: float,
    high_material_field: float,
    field_ratio: float,
) -> None:
    """
    Print the dielectric-interface verification diagnostics.
    """

    expected_field_ratio = (
        HIGH_RELATIVE_PERMITTIVITY
        / LOW_RELATIVE_PERMITTIVITY
    )

    print()
    print(
        "2D Poisson dielectric-interface verification"
    )

    print(
        "-" * 66
    )

    print(
        "Grid shape:                    "
        f"{simulation.device.grid.shape[0]} x "
        f"{simulation.device.grid.shape[1]}"
    )

    print(
        "Relative permittivity left:    "
        f"{LOW_RELATIVE_PERMITTIVITY:.6g}"
    )

    print(
        "Relative permittivity right:   "
        f"{HIGH_RELATIVE_PERMITTIVITY:.6g}"
    )

    print(
        "Applied potential difference:  "
        f"{RIGHT_POTENTIAL - LEFT_POTENTIAL:.6g} V"
    )

    print()

    print(
        "Mean D0:                       "
        f"{mean_displacement:.12e} C/m^2"
    )

    print(
        "Maximum absolute D0 deviation: "
        f"{maximum_displacement_deviation:.12e} C/m^2"
    )

    print(
        "Maximum relative D0 deviation: "
        f"{relative_displacement_deviation:.12e}"
    )

    print()

    print(
        "Mean |E0|, epsilon_r = 4:      "
        f"{low_material_field:.12e} V/m"
    )

    print(
        "Mean |E0|, epsilon_r = 12:     "
        f"{high_material_field:.12e} V/m"
    )

    print(
        "Electric-field ratio:          "
        f"{field_ratio:.12f}"
    )

    print(
        "Expected field ratio:          "
        f"{expected_field_ratio:.12f}"
    )

    print()

    if (
        relative_displacement_deviation
        < 1.0e-10
        and np.isclose(
            field_ratio,
            expected_field_ratio,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    ):
        print(
            "PASS: normal electric displacement "
            "is conserved across the interface."
        )
    else:
        print(
            "FAIL: dielectric-interface verification "
            "did not satisfy the configured tolerances."
        )


def write_profile_csv(
    simulation: Simulation,
    electric_field_profile: np.ndarray,
    face_permittivity_profile: np.ndarray,
    displacement_profile: np.ndarray,
) -> None:
    """
    Write the centre-line axis-0 face quantities to CSV.
    """

    grid = simulation.device.grid

    coordinates_axis_0 = (
        grid.coordinates(0)
    )

    face_coordinates_axis_0 = (
        0.5
        * (
            coordinates_axis_0[:-1]
            + coordinates_axis_0[1:]
        )
    )

    output_path = (
        OUTPUT_DIRECTORY
        / "verification_results.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            (
                "axis_0_face_position_m",
                "relative_permittivity_face",
                "electric_field_axis_0_v_per_m",
                "electric_displacement_axis_0_c_per_m2",
            )
        )

        for index in range(
            face_coordinates_axis_0.size
        ):
            writer.writerow(
                (
                    face_coordinates_axis_0[index],
                    face_permittivity_profile[index],
                    electric_field_profile[index],
                    displacement_profile[index],
                )
            )


def plot_potential_profile(
    simulation: Simulation,
    split_index: int,
    potential_profile: np.ndarray,
) -> None:
    """
    Plot the centre-line electrostatic potential.
    """

    coordinates_nm = (
        simulation.device.grid.coordinates(0)
        * 1.0e9
    )

    interface_position_nm = (
        0.5
        * (
            coordinates_nm[split_index - 1]
            + coordinates_nm[split_index]
        )
    )

    figure, axis = plt.subplots()

    axis.plot(
        coordinates_nm,
        potential_profile,
    )

    axis.axvline(
        interface_position_nm,
        linestyle="--",
        label="Dielectric interface",
    )

    axis.set_xlabel(
        "Axis 0 position (nm)"
    )

    axis.set_ylabel(
        "Potential (V)"
    )

    axis.set_title(
        "2D Dielectric Interface: Potential"
    )

    axis.grid(
        True,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "potential_profile.png",
        dpi=200,
    )

    plt.close(
        figure
    )


def plot_electric_field_profile(
    simulation: Simulation,
    split_index: int,
    electric_field_profile: np.ndarray,
) -> None:
    """
    Plot the centre-line axis-0 face electric field.
    """

    coordinates_axis_0 = (
        simulation.device.grid.coordinates(0)
    )

    face_coordinates_nm = (
        0.5
        * (
            coordinates_axis_0[:-1]
            + coordinates_axis_0[1:]
        )
        * 1.0e9
    )

    interface_position_nm = (
        face_coordinates_nm[
            split_index - 1
        ]
    )

    figure, axis = plt.subplots()

    axis.plot(
        face_coordinates_nm,
        electric_field_profile,
    )

    axis.axvline(
        interface_position_nm,
        linestyle="--",
        label="Dielectric interface",
    )

    axis.set_xlabel(
        "Axis 0 face position (nm)"
    )

    axis.set_ylabel(
        "Electric field E0 (V/m)"
    )

    axis.set_title(
        "2D Dielectric Interface: Electric Field"
    )

    axis.grid(
        True,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "electric_field_profile.png",
        dpi=200,
    )

    plt.close(
        figure
    )


def plot_permittivity_profile(
    simulation: Simulation,
    split_index: int,
    face_permittivity_profile: np.ndarray,
) -> None:
    """
    Plot the harmonic face relative permittivity.
    """

    coordinates_axis_0 = (
        simulation.device.grid.coordinates(0)
    )

    face_coordinates_nm = (
        0.5
        * (
            coordinates_axis_0[:-1]
            + coordinates_axis_0[1:]
        )
        * 1.0e9
    )

    interface_position_nm = (
        face_coordinates_nm[
            split_index - 1
        ]
    )

    figure, axis = plt.subplots()

    axis.plot(
        face_coordinates_nm,
        face_permittivity_profile,
    )

    axis.axvline(
        interface_position_nm,
        linestyle="--",
        label="Dielectric interface",
    )

    axis.set_xlabel(
        "Axis 0 face position (nm)"
    )

    axis.set_ylabel(
        "Face relative permittivity"
    )

    axis.set_title(
        "2D Dielectric Interface: "
        "Harmonic Face Permittivity"
    )

    axis.grid(
        True,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "permittivity_profile.png",
        dpi=200,
    )

    plt.close(
        figure
    )


def plot_displacement_profile(
    simulation: Simulation,
    split_index: int,
    displacement_profile: np.ndarray,
) -> None:
    """
    Plot the relative deviation of the centre-line normal electric
    displacement from its mean value.

    For a charge-free dielectric interface, the normal electric
    displacement should remain constant. Deviations therefore measure
    the numerical conservation error.
    """

    coordinates_axis_0 = (
        simulation.device.grid.coordinates(0)
    )

    face_coordinates_nm = (
        0.5
        * (
            coordinates_axis_0[:-1]
            + coordinates_axis_0[1:]
        )
        * 1.0e9
    )

    interface_position_nm = (
        face_coordinates_nm[
            split_index - 1
        ]
    )

    mean_displacement = float(
        np.mean(
            displacement_profile
        )
    )

    relative_deviation = (
        (
            displacement_profile
            - mean_displacement
        )
        / abs(mean_displacement)
    )

    figure, axis = plt.subplots()

    axis.plot(
        face_coordinates_nm,
        relative_deviation,
    )

    axis.axvline(
        interface_position_nm,
        linestyle="--",
        label="Dielectric interface",
    )

    axis.axhline(
        0.0,
        linestyle=":",
        label="Exact conservation",
    )

    axis.set_xlabel(
        "Axis 0 face position (nm)"
    )

    axis.set_ylabel(
        "Relative D0 deviation"
    )

    axis.set_title(
        "2D Dielectric Interface: "
        "Displacement Conservation Error"
    )

    axis.grid(
        True,
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        OUTPUT_DIRECTORY
        / "displacement_profile.png",
        dpi=200,
    )

    plt.close(
        figure
    )


def main() -> None:
    """
    Run the complete 2D dielectric-interface verification.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        simulation,
        split_index,
        potential_profile,
        electric_field_profile,
        face_permittivity_profile,
        displacement_profile,
        mean_displacement,
        maximum_displacement_deviation,
        relative_displacement_deviation,
        low_material_field,
        high_material_field,
        field_ratio,
    ) = run_verification()

    print_results(
        simulation,
        mean_displacement,
        maximum_displacement_deviation,
        relative_displacement_deviation,
        low_material_field,
        high_material_field,
        field_ratio,
    )

    write_profile_csv(
        simulation,
        electric_field_profile,
        face_permittivity_profile,
        displacement_profile,
    )

    plot_potential_profile(
        simulation,
        split_index,
        potential_profile,
    )

    plot_electric_field_profile(
        simulation,
        split_index,
        electric_field_profile,
    )

    plot_permittivity_profile(
        simulation,
        split_index,
        face_permittivity_profile,
    )

    plot_displacement_profile(
        simulation,
        split_index,
        displacement_profile,
    )

    print()
    print(
        "Verification outputs written to:"
    )

    print(
        OUTPUT_DIRECTORY
    )


if __name__ == "__main__":
    main()