from __future__ import annotations

import numpy as np
import pytest

from deviceforge import Grid, Region, Device
from deviceforge.core import Field
from deviceforge.postprocessing import calculate_electric_field

from deviceforge.core.boundary import (
    BoundaryCondition,
    BoundaryConditionType,
)

from deviceforge.core.simulation import Simulation
from deviceforge.physics import SILICON, SILICON_DIOXIDE
from deviceforge.solvers import PoissonSolver

from deviceforge.postprocessing import (
    calculate_electric_displacement_field,
    calculate_electric_field,
    calculate_electrostatic_energy_density,
    calculate_face_electric_field,
    calculate_face_electrostatic_fields,
    calculate_face_relative_permittivity,
)


def test_calculate_electric_field_returns_field() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )

    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid,
    )

    electric_field = calculate_electric_field(
        potential
    )

    assert isinstance(electric_field, Field)
    assert electric_field.name == "electric_field"
    assert electric_field.units == "V/m"
    assert electric_field.grid is grid
    assert electric_field.shape == grid.shape


def test_constant_potential_produces_zero_field() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )

    potential = Field.full(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        fill_value=2.5,
    )

    electric_field = calculate_electric_field(
        potential
    )

    np.testing.assert_allclose(
        electric_field.values,
        0.0,
        atol=0.0,
    )


def test_linear_potential_produces_constant_field() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )

    coordinates = grid.coordinates(0)

    slope = 2.5e6
    offset = 0.4

    potential_values = (
        offset
        + slope * coordinates
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=potential_values,
    )

    electric_field = calculate_electric_field(
        potential
    )

    expected_field = np.full(
        grid.shape,
        -slope,
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        electric_field.values,
        expected_field,
        rtol=1.0e-12,
        atol=1.0e-6,
    )


def test_parabolic_potential_produces_linear_internal_field() -> None:
    grid = Grid(
        shape=(101,),
        spacing=(1.0e-9,),
    )

    coordinates = grid.coordinates(0)

    charge_density = 1.0e5
    permittivity = (
        8.8541878128e-12
        * 11.7
    )

    x = coordinates - coordinates[0]
    length = coordinates[-1] - coordinates[0]

    potential_values = (
        charge_density
        / (2.0 * permittivity)
        * x
        * (length - x)
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=potential_values,
    )

    electric_field = calculate_electric_field(
        potential
    )

    expected_field = -(
        charge_density
        / (2.0 * permittivity)
        * (length - 2.0 * x)
    )

    np.testing.assert_allclose(
        electric_field.values[1:-1],
        expected_field[1:-1],
        rtol=1.0e-12,
        atol=1.0e-6,
    )

# additional validation tests

def test_calculate_electric_field_rejects_non_field() -> None:
    with pytest.raises(
        TypeError,
        match="requires a Field",
    ):
        calculate_electric_field(
            "invalid"
        )


def test_calculate_electric_field_rejects_non_voltage_units() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )

    invalid_field = Field.zeros(
        name="charge_density",
        units="C/m^3",
        grid=grid,
    )

    with pytest.raises(
        ValueError,
        match="units must be 'V'",
    ):
        calculate_electric_field(
            invalid_field
        )


def test_calculate_electric_field_rejects_multidimensional_grid() -> None:
    grid = Grid(
        shape=(11, 7),
        spacing=(1.0e-9, 1.0e-9),
    )

    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid,
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        calculate_electric_field(
            potential
        )


def test_two_point_grid_uses_one_sided_differences() -> None:
    grid = Grid(
        shape=(2,),
        spacing=(0.5,),
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=np.asarray(
            [0.0, 2.0],
            dtype=np.float64,
        ),
    )

    electric_field = calculate_electric_field(
        potential
    )

    np.testing.assert_allclose(
        electric_field.values,
        [-4.0, -4.0],
    )

# add integration test for poisson solver

def test_electric_field_from_poisson_solution_is_constant(
    simulation,
) -> None:
    grid = simulation.grid

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

    electrostatic_simulation = Simulation(
        device=simulation.device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-10,
        name="electric_field_integration_test",
    )

    result = PoissonSolver().solve(
        electrostatic_simulation
    )

    electric_field = calculate_electric_field(
        result.potential
    )

    domain_length = grid.physical_size[0]

    expected_field = -1.0 / domain_length

    np.testing.assert_allclose(
        electric_field.values,
        expected_field,
        rtol=1.0e-11,
        atol=1.0e-6,
    )

# unifrom material test

def test_uniform_electric_displacement_field() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )

    electric_field = Field.full(
        name="electric_field",
        units="V/m",
        grid=grid,
        fill_value=-2.0e6,
    )

    relative_permittivity = Field.full(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        fill_value=11.7,
    )

    displacement = calculate_electric_displacement_field(
        electric_field,
        relative_permittivity,
    )

    expected = (
        8.8541878128e-12
        * 11.7
        * -2.0e6
    )

    np.testing.assert_allclose(
        displacement.values,
        expected,
        rtol=1.0e-12,
        atol=1.0e-18,
    )

    assert displacement.name == "electric_displacement"
    assert displacement.units == "C/m^2"
    assert displacement.grid is grid

# spatially varying permittivity

def test_displacement_uses_spatial_permittivity() -> None:
    grid = Grid(
        shape=(4,),
        spacing=(1.0e-9,),
    )

    electric_field = Field(
        name="electric_field",
        units="V/m",
        grid=grid,
        values=np.asarray(
            [-3.0e6, -3.0e6, -1.0e6, -1.0e6],
            dtype=np.float64,
        ),
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=np.asarray(
            [3.9, 3.9, 11.7, 11.7],
            dtype=np.float64,
        ),
    )

    displacement = calculate_electric_displacement_field(
        electric_field,
        relative_permittivity,
    )

    expected = (
        8.8541878128e-12
        * relative_permittivity.values
        * electric_field.values
    )

    np.testing.assert_allclose(
        displacement.values,
        expected,
        rtol=1.0e-12,
        atol=1.0e-18,
    )

# validation tests

def test_displacement_rejects_invalid_electric_field_units() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    electric_field = Field.zeros(
        name="invalid",
        units="V",
        grid=grid,
    )

    relative_permittivity = Field.full(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        fill_value=11.7,
    )

    with pytest.raises(
        ValueError,
        match="Electric field units",
    ):
        calculate_electric_displacement_field(
            electric_field,
            relative_permittivity,
        )


def test_displacement_rejects_invalid_permittivity_units() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    electric_field = Field.zeros(
        name="electric_field",
        units="V/m",
        grid=grid,
    )

    invalid_permittivity = Field.full(
        name="permittivity",
        units="F/m",
        grid=grid,
        fill_value=1.0,
    )

    with pytest.raises(
        ValueError,
        match="Relative permittivity units",
    ):
        calculate_electric_displacement_field(
            electric_field,
            invalid_permittivity,
        )


def test_displacement_requires_matching_grids() -> None:
    electric_grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    permittivity_grid = Grid(
        shape=(6,),
        spacing=(1.0e-9,),
    )

    electric_field = Field.zeros(
        name="electric_field",
        units="V/m",
        grid=electric_grid,
    )

    relative_permittivity = Field.full(
        name="relative_permittivity",
        units="dimensionless",
        grid=permittivity_grid,
        fill_value=11.7,
    )

    with pytest.raises(
        ValueError,
        match="same grid",
    ):
        calculate_electric_displacement_field(
            electric_field,
            relative_permittivity,
        )


def test_displacement_rejects_non_positive_permittivity() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    electric_field = Field.zeros(
        name="electric_field",
        units="V/m",
        grid=grid,
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=np.asarray(
            [11.7, 11.7, 0.0, 11.7, 11.7],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        calculate_electric_displacement_field(
            electric_field,
            relative_permittivity,
        )

# solver integration tests
# Poisson solve
# → potential
# → electric field
# → displacement field

def test_displacement_from_dielectric_interface_solution_is_continuous() -> None:
    grid = Grid(
        shape=(21,),
        spacing=(1.0e-9,),
    )

    oxide_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    oxide_mask[:10] = True

    silicon_mask = np.zeros(
        grid.shape,
        dtype=np.bool_,
    )
    silicon_mask[10:] = True

    oxide_region = Region(
        name="oxide",
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
        name="dielectric_stack",
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

    simulation = Simulation(
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-10,
        name="displacement_interface_test",
    )

    result = PoissonSolver().solve(
        simulation
    )

    electric_field = calculate_electric_field(
        result.potential
    )

    displacement = calculate_electric_displacement_field(
        electric_field,
        device.relative_permittivity_field(),
    )

    oxide_displacement = np.mean(
        displacement.values[2:8]
    )

    silicon_displacement = np.mean(
        displacement.values[12:18]
    )

    assert oxide_displacement == pytest.approx(
        silicon_displacement,
        rel=1.0e-10,
    )

# electrostaic energy density

def test_uniform_electrostatic_energy_density() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )

    electric_field = Field.full(
        name="electric_field",
        units="V/m",
        grid=grid,
        fill_value=2.0e6,
    )

    relative_permittivity = Field.full(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        fill_value=11.7,
    )

    energy_density = calculate_electrostatic_energy_density(
        electric_field,
        relative_permittivity,
    )

    expected = (
        0.5
        * 8.8541878128e-12
        * 11.7
        * (2.0e6**2)
    )

    np.testing.assert_allclose(
        energy_density.values,
        expected,
        rtol=1.0e-12,
        atol=1.0e-15,
    )

    assert energy_density.name == (
        "electrostatic_energy_density"
    )
    assert energy_density.units == "J/m^3"
    assert energy_density.grid is grid

# symmetry test

def test_energy_density_is_independent_of_field_sign() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    positive_field = Field.full(
        name="electric_field",
        units="V/m",
        grid=grid,
        fill_value=3.0e6,
    )

    negative_field = Field.full(
        name="electric_field",
        units="V/m",
        grid=grid,
        fill_value=-3.0e6,
    )

    relative_permittivity = Field.full(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        fill_value=3.9,
    )

    positive_energy = calculate_electrostatic_energy_density(
        positive_field,
        relative_permittivity,
    )

    negative_energy = calculate_electrostatic_energy_density(
        negative_field,
        relative_permittivity,
    )

    np.testing.assert_allclose(
        positive_energy.values,
        negative_energy.values,
    )

# spatial material test

def test_energy_density_uses_spatial_permittivity() -> None:
    grid = Grid(
        shape=(4,),
        spacing=(1.0e-9,),
    )

    electric_field = Field.full(
        name="electric_field",
        units="V/m",
        grid=grid,
        fill_value=1.0e6,
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=np.asarray(
            [3.9, 3.9, 11.7, 11.7],
            dtype=np.float64,
        ),
    )

    energy_density = calculate_electrostatic_energy_density(
        electric_field,
        relative_permittivity,
    )

    expected = (
        0.5
        * 8.8541878128e-12
        * relative_permittivity.values
        * electric_field.values**2
    )

    np.testing.assert_allclose(
        energy_density.values,
        expected,
        rtol=1.0e-12,
        atol=1.0e-15,
    )

    assert (
        energy_density.values[2]
        / energy_density.values[0]
        == pytest.approx(3.0)
    )

# validation tests

def test_energy_density_rejects_invalid_electric_field_units() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    invalid_field = Field.zeros(
        name="potential",
        units="V",
        grid=grid,
    )

    relative_permittivity = Field.full(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        fill_value=11.7,
    )

    with pytest.raises(
        ValueError,
        match="Electric field units",
    ):
        calculate_electrostatic_energy_density(
            invalid_field,
            relative_permittivity,
        )


def test_energy_density_requires_matching_grids() -> None:
    electric_grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    permittivity_grid = Grid(
        shape=(6,),
        spacing=(1.0e-9,),
    )

    electric_field = Field.zeros(
        name="electric_field",
        units="V/m",
        grid=electric_grid,
    )

    relative_permittivity = Field.full(
        name="relative_permittivity",
        units="dimensionless",
        grid=permittivity_grid,
        fill_value=11.7,
    )

    with pytest.raises(
        ValueError,
        match="same grid",
    ):
        calculate_electrostatic_energy_density(
            electric_field,
            relative_permittivity,
        )


def test_energy_density_rejects_non_positive_permittivity() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    electric_field = Field.zeros(
        name="electric_field",
        units="V/m",
        grid=grid,
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=np.asarray(
            [11.7, 11.7, 0.0, 11.7, 11.7],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        calculate_electrostatic_energy_density(
            electric_field,
            relative_permittivity,
        )

# poisson integration test

def test_energy_density_from_linear_poisson_solution_is_uniform(
    simulation,
) -> None:
    grid = simulation.grid

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

    electrostatic_simulation = Simulation(
        device=simulation.device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-10,
        name="energy_density_integration_test",
    )

    result = PoissonSolver().solve(
        electrostatic_simulation
    )

    electric_field = calculate_electric_field(
        result.potential
    )

    energy_density = calculate_electrostatic_energy_density(
        electric_field,
        simulation.device.relative_permittivity_field(),
    )

    np.testing.assert_allclose(
        energy_density.values,
        energy_density.values[0],
        rtol=1.0e-11,
        atol=1.0e-12,
    )

    assert np.all(
        energy_density.values > 0.0
    )

# additional tests for facefield

def test_face_electric_field_from_linear_potential() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(1.0e-9,),
    )

    coordinates = grid.coordinates(0)
    slope = 2.5e6

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=slope * coordinates,
    )

    face_field = calculate_face_electric_field(
        potential
    )

    np.testing.assert_allclose(
        face_field.values,
        -slope,
        rtol=1.0e-12,
        atol=1.0e-6,
    )


def test_face_permittivity_uses_harmonic_mean() -> None:
    grid = Grid(
        shape=(2,),
        spacing=(1.0e-9,),
    )

    relative_permittivity = Field(
        name="relative_permittivity",
        units="dimensionless",
        grid=grid,
        values=[3.9, 11.7],
    )

    face_permittivity = (
        calculate_face_relative_permittivity(
            relative_permittivity
        )
    )

    expected = (
        2.0
        * 3.9
        * 11.7
        / (3.9 + 11.7)
    )

    assert face_permittivity.values[0] == pytest.approx(
        expected
    )

# updated
def test_face_displacement_is_continuous_for_dielectric_stack(
    dielectric_stack_simulation: Simulation,
) -> None:
    result = PoissonSolver().solve(
        dielectric_stack_simulation
    )

    relative_permittivity = (
        dielectric_stack_simulation
        .device
        .relative_permittivity_field()
    )

    (
        _,
        _,
        face_displacement,
    ) = calculate_face_electrostatic_fields(
        result.potential,
        relative_permittivity,
    )

    np.testing.assert_allclose(
        face_displacement.values,
        face_displacement.values[0],
        rtol=1.0e-11,
        atol=1.0e-15,
    )

def test_quadratic_potential_produces_exact_field_at_endpoints() -> None:
    grid = Grid(
        shape=(11,),
        spacing=(0.1,),
    )

    coordinates = grid.coordinates(0)

    potential_values = (
        2.0 * coordinates**2
        + 3.0 * coordinates
        + 1.0
    )

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=potential_values,
    )

    electric_field = calculate_electric_field(
        potential
    )

    expected = -(
        4.0 * coordinates
        + 3.0
    )

    np.testing.assert_allclose(
        electric_field.values,
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

def test_three_point_grid_uses_second_order_endpoint_differences() -> None:
    grid = Grid(
        shape=(3,),
        spacing=(0.5,),
    )

    coordinates = grid.coordinates(0)

    potential = Field(
        name="electrostatic_potential",
        units="V",
        grid=grid,
        values=coordinates**2,
    )

    electric_field = calculate_electric_field(
        potential
    )

    expected = -2.0 * coordinates

    np.testing.assert_allclose(
        electric_field.values,
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )