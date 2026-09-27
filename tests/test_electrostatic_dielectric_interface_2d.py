from __future__ import annotations

import numpy as np

from deviceforge import Device, Grid, Region
from deviceforge.core.boundary import (
    BoundaryCondition,
    BoundaryConditionType,
)
from deviceforge.core.simulation import Simulation
from deviceforge.physics.materials import Material

from deviceforge.postprocessing import (
    calculate_face_electrostatic_energy_density_components_2d,
    calculate_face_electrostatic_fields_2d,
    calculate_total_electrostatic_energy_2d,
)

from deviceforge.solvers import PoissonSolver2D

VACUUM_PERMITTIVITY = 8.8541878128e-12


def create_dielectric_interface_simulation_2d() -> Simulation:
    """
    Create a charge-free two-material dielectric interface problem.

    The material interface is normal to axis 0.

    Boundary conditions:
        left:   phi = 0 V
        right:  phi = 1 V
        bottom: d(phi)/dn = 0 V/m
        top:    d(phi)/dn = 0 V/m
    """

    grid = Grid(
        shape=(41, 21),
        spacing=(
            1.0e-9,
            1.0e-9,
        ),
        origin=(0.0, 0.0),
    )

    dielectric_low = Material(
        name="dielectric_low",
        relative_permittivity=4.0,
        material_type="dielectric",
    )

    dielectric_high = Material(
        name="dielectric_high",
        relative_permittivity=12.0,
        material_type="dielectric",
    )

    low_permittivity_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    low_permittivity_mask[:21, :] = True

    high_permittivity_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    high_permittivity_mask[21:, :] = True

    low_permittivity_region = Region(
        name="low_permittivity_region",
        grid=grid,
        material=dielectric_low,
        mask=low_permittivity_mask,
    )

    high_permittivity_region = Region(
        name="high_permittivity_region",
        grid=grid,
        material=dielectric_high,
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
    bottom_mask[1:-1, 0] = True

    top_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    top_mask[1:-1, -1] = True

    left_boundary = BoundaryCondition(
        name="left_boundary",
        grid=grid,
        mask=left_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=0.0,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_boundary",
        grid=grid,
        mask=right_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=1.0,
        units="V",
    )

    bottom_boundary = BoundaryCondition(
        name="bottom_zero_flux",
        grid=grid,
        mask=bottom_mask,
        condition_type=BoundaryConditionType.NEUMANN,
        value=0.0,
        units="V/m",
    )

    top_boundary = BoundaryCondition(
        name="top_zero_flux",
        grid=grid,
        mask=top_mask,
        condition_type=BoundaryConditionType.NEUMANN,
        value=0.0,
        units="V/m",
    )

    return Simulation(
        name="dielectric_interface_2d",
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
            bottom_boundary,
            top_boundary,
        ),
        tolerance=1.0e-10,
        max_iterations=10_000,
        initial_potential=0.0,
    )

def test_dielectric_interface_2d_conserves_normal_displacement() -> None:
    simulation = create_dielectric_interface_simulation_2d()

    result = PoissonSolver2D().solve(
        simulation
    )

    assert result.converged

    relative_permittivity = (
        simulation.device.relative_permittivity_field()
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
        relative_permittivity_axis_0,
        relative_permittivity_axis_1,
        displacement_axis_0,
        displacement_axis_1,
    ) = calculate_face_electrostatic_fields_2d(
        result.potential,
        relative_permittivity,
    )

    displacement_values = (
        displacement_axis_0.values
    )

    reference_displacement = np.mean(
        displacement_values
    )

    np.testing.assert_allclose(
        displacement_values,
        reference_displacement,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

def test_dielectric_interface_2d_has_zero_transverse_field() -> None:
    simulation = create_dielectric_interface_simulation_2d()

    result = PoissonSolver2D().solve(
        simulation
    )

    relative_permittivity = (
        simulation.device.relative_permittivity_field()
    )

    (
        _,
        electric_field_axis_1,
        _,
        _,
        _,
        displacement_axis_1,
    ) = calculate_face_electrostatic_fields_2d(
        result.potential,
        relative_permittivity,
    )

    np.testing.assert_allclose(
        electric_field_axis_1.values,
        0.0,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        displacement_axis_1.values,
        0.0,
        atol=1.0e-12,
    )

def test_dielectric_interface_2d_field_scales_inverse_with_permittivity() -> None:
    simulation = create_dielectric_interface_simulation_2d()

    result = PoissonSolver2D().solve(
        simulation
    )

    relative_permittivity = (
        simulation.device.relative_permittivity_field()
    )

    (
        electric_field_axis_0,
        _,
        face_relative_permittivity_axis_0,
        _,
        _,
        _,
    ) = calculate_face_electrostatic_fields_2d(
        result.potential,
        relative_permittivity,
    )

    face_permittivity = (
        face_relative_permittivity_axis_0.values
    )

    electric_field = (
        electric_field_axis_0.values
    )

    low_material_faces = np.isclose(
        face_permittivity,
        4.0,
    )

    high_material_faces = np.isclose(
        face_permittivity,
        12.0,
    )

    low_material_field = np.mean(
        np.abs(
            electric_field[
                low_material_faces
            ]
        )
    )

    high_material_field = np.mean(
        np.abs(
            electric_field[
                high_material_faces
            ]
        )
    )

    field_ratio = (
        low_material_field
        / high_material_field
    )

    np.testing.assert_allclose(
        field_ratio,
        3.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

def test_dielectric_interface_2d_reports_conservation_diagnostics() -> None:
    simulation = create_dielectric_interface_simulation_2d()

    result = PoissonSolver2D().solve(
        simulation
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

    displacement = displacement_axis_0.values
    face_permittivity = (
        face_relative_permittivity_axis_0.values
    )
    electric_field = electric_field_axis_0.values

    mean_displacement = float(
        np.mean(displacement)
    )

    maximum_displacement_deviation = float(
        np.max(
            np.abs(
                displacement
                - mean_displacement
            )
        )
    )

    relative_displacement_deviation = (
        maximum_displacement_deviation
        / abs(mean_displacement)
    )

    low_material_faces = np.isclose(
        face_permittivity,
        4.0,
    )

    high_material_faces = np.isclose(
        face_permittivity,
        12.0,
    )

    low_material_field = float(
        np.mean(
            np.abs(
                electric_field[
                    low_material_faces
                ]
            )
        )
    )

    high_material_field = float(
        np.mean(
            np.abs(
                electric_field[
                    high_material_faces
                ]
            )
        )
    )

    field_ratio = (
        low_material_field
        / high_material_field
    )

    print(
        "\n2D dielectric-interface conservation diagnostics"
    )
    print(
        f"Mean D0: "
        f"{mean_displacement:.12e} C/m^2"
    )
    print(
        f"Maximum absolute D0 deviation: "
        f"{maximum_displacement_deviation:.12e} C/m^2"
    )
    print(
        f"Maximum relative D0 deviation: "
        f"{relative_displacement_deviation:.12e}"
    )
    print(
        f"Mean |E0| for epsilon_r = 4: "
        f"{low_material_field:.12e} V/m"
    )
    print(
        f"Mean |E0| for epsilon_r = 12: "
        f"{high_material_field:.12e} V/m"
    )
    print(
        f"Electric-field ratio: "
        f"{field_ratio:.12f}"
    )

    assert relative_displacement_deviation < 1.0e-10

    np.testing.assert_allclose(
        field_ratio,
        3.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

def test_dielectric_interface_2d_matches_analytical_stored_energy(
) -> None:
    """
    Verify the complete 2D electrostatic-energy pipeline against the
    analytical discrete series-capacitor energy.

    The analytical reference is constructed directly from the known
    face permittivities, grid spacing, domain width, and applied
    potential difference:

        C' = epsilon_0 * L_transverse
             / sum(delta_x / epsilon_r_face)

        U' = 0.5 * C' * delta_V**2

    where C' is capacitance per unit out-of-plane depth and U' is
    electrostatic energy per unit out-of-plane depth.
    """

    simulation = create_dielectric_interface_simulation_2d()

    result = PoissonSolver2D().solve(
        simulation
    )

    assert result.converged

    relative_permittivity = (
        simulation.device.relative_permittivity_field()
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
        face_relative_permittivity_axis_0,
        _,
        displacement_axis_0,
        displacement_axis_1,
    ) = calculate_face_electrostatic_fields_2d(
        result.potential,
        relative_permittivity,
    )

    (
        energy_density_axis_0,
        energy_density_axis_1,
    ) = (
        calculate_face_electrostatic_energy_density_components_2d(
            electric_field_axis_0,
            electric_field_axis_1,
            displacement_axis_0,
            displacement_axis_1,
        )
    )

    numerical_energy = (
        calculate_total_electrostatic_energy_2d(
            energy_density_axis_0,
            energy_density_axis_1,
        )
    )

    grid = simulation.device.grid

    spacing_axis_0 = grid.spacing[0]
    spacing_axis_1 = grid.spacing[1]

    transverse_length = (
        (grid.shape[1] - 1)
        * spacing_axis_1
    )

    applied_potential_difference = 1.0

    face_relative_permittivity = (
        face_relative_permittivity_axis_0.values[:, 0]
    )

    series_denominator = np.sum(
        spacing_axis_0
        / face_relative_permittivity
    )

    analytical_capacitance = (
        VACUUM_PERMITTIVITY
        * transverse_length
        / series_denominator
    )

    analytical_energy = (
        0.5
        * analytical_capacitance
        * applied_potential_difference**2
    )

    np.testing.assert_allclose(
        numerical_energy,
        analytical_energy,
        rtol=1.0e-10,
        atol=1.0e-20,
    )

def test_dielectric_interface_2d_reports_energy_diagnostics() -> None:
    simulation = create_dielectric_interface_simulation_2d()

    result = PoissonSolver2D().solve(
        simulation
    )

    assert result.converged

    relative_permittivity = (
        simulation.device.relative_permittivity_field()
    )

    (
        electric_field_axis_0,
        electric_field_axis_1,
        face_relative_permittivity_axis_0,
        _,
        displacement_axis_0,
        displacement_axis_1,
    ) = calculate_face_electrostatic_fields_2d(
        result.potential,
        relative_permittivity,
    )

    (
        energy_density_axis_0,
        energy_density_axis_1,
    ) = (
        calculate_face_electrostatic_energy_density_components_2d(
            electric_field_axis_0,
            electric_field_axis_1,
            displacement_axis_0,
            displacement_axis_1,
        )
    )

    numerical_energy = (
        calculate_total_electrostatic_energy_2d(
            energy_density_axis_0,
            energy_density_axis_1,
        )
    )

    grid = simulation.device.grid

    spacing_axis_0 = grid.spacing[0]

    transverse_length = (
        (grid.shape[1] - 1)
        * grid.spacing[1]
    )

    applied_potential_difference = 1.0

    face_relative_permittivity = (
        face_relative_permittivity_axis_0.values[:, 0]
    )

    series_denominator = np.sum(
        spacing_axis_0
        / face_relative_permittivity
    )

    analytical_capacitance = (
        VACUUM_PERMITTIVITY
        * transverse_length
        / series_denominator
    )

    analytical_energy = (
        0.5
        * analytical_capacitance
        * applied_potential_difference**2
    )

    numerical_capacitance = (
        2.0
        * numerical_energy
        / applied_potential_difference**2
    )

    absolute_energy_error = abs(
        numerical_energy
        - analytical_energy
    )

    relative_energy_error = (
        absolute_energy_error
        / abs(analytical_energy)
    )

    print(
        "\n2D dielectric-capacitor energy diagnostics"
    )
    print(
        f"Analytical capacitance per unit depth: "
        f"{analytical_capacitance:.12e} F/m"
    )
    print(
        f"Numerical capacitance per unit depth: "
        f"{numerical_capacitance:.12e} F/m"
    )
    print(
        f"Analytical energy per unit depth: "
        f"{analytical_energy:.12e} J/m"
    )
    print(
        f"Numerical energy per unit depth: "
        f"{numerical_energy:.12e} J/m"
    )
    print(
        f"Absolute energy error: "
        f"{absolute_energy_error:.12e} J/m"
    )
    print(
        f"Relative energy error: "
        f"{relative_energy_error:.12e}"
    )

    np.testing.assert_allclose(
        numerical_energy,
        analytical_energy,
        rtol=1.0e-10,
        atol=1.0e-20,
    )

    np.testing.assert_allclose(
        numerical_capacitance,
        analytical_capacitance,
        rtol=1.0e-10,
        atol=1.0e-20,
    )