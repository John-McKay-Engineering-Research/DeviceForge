from __future__ import annotations

import numpy as np
import pytest

from deviceforge import Device, Grid, Region
from deviceforge.core import Field
from deviceforge.core.boundary import (
    BoundaryCondition,
    BoundaryConditionType,
)
from deviceforge.core.result import SimulationResult
from deviceforge.core.simulation import Simulation
from deviceforge.physics import SILICON

from deviceforge.postprocessing import (
    ElectrostaticAnalysis,
    analyse_electrostatics,
)


from deviceforge.solvers import PoissonSolver


def create_analysis_simulation(
    *,
    number_of_points: int = 11,
    left_voltage: float = 0.0,
    right_voltage: float = 1.0,
) -> Simulation:
    """Create a uniform one-dimensional silicon simulation."""

    grid = Grid(
        shape=(number_of_points,),
        spacing=(1.0e-9,),
    )

    region = Region(
        name="silicon",
        grid=grid,
        material=SILICON,
        mask=np.ones(
            grid.shape,
            dtype=np.bool_,
        ),
    )

    device = Device(
        name="analysis_test_device",
        grid=grid,
        regions=(region,),
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
        value=left_voltage,
        units="V",
    )

    right_boundary = BoundaryCondition(
        name="right_contact",
        grid=grid,
        mask=right_mask,
        condition_type=BoundaryConditionType.DIRICHLET,
        value=right_voltage,
        units="V",
    )

    return Simulation(
        device=device,
        boundary_conditions=(
            left_boundary,
            right_boundary,
        ),
        tolerance=1.0e-10,
        name="electrostatic_analysis_test",
    )


def create_manual_result(
    simulation: Simulation,
    potential: Field,
) -> SimulationResult:
    """Create a valid result containing a supplied potential field."""

    return SimulationResult(
        fields={
            "electrostatic_potential": potential,
        },
        converged=True,
        iterations=1,
        residual_history=np.asarray(
            [0.0],
            dtype=np.float64,
        ),
        runtime_seconds=0.0,
        solver_name="test_solver",
        backend_name="numpy",
    )


def test_analyse_electrostatics_returns_analysis() -> None:
    simulation = create_analysis_simulation()

    result = PoissonSolver().solve(
        simulation
    )

    analysis = analyse_electrostatics(
        simulation,
        result,
    )

    assert isinstance(
        analysis,
        ElectrostaticAnalysis,
    )


def test_analysis_contains_expected_fields() -> None:
    simulation = create_analysis_simulation()

    result = PoissonSolver().solve(
        simulation
    )

    analysis = analyse_electrostatics(
        simulation,
        result,
    )

    assert analysis.potential is result.potential

    assert analysis.electric_field.name == (
        "electric_field"
    )

    assert analysis.electric_displacement.name == (
        "electric_displacement"
    )

    assert analysis.energy_density.name == (
        "electrostatic_energy_density"
    )

    assert analysis.field_names == (
        "electrostatic_potential",
        "electric_field",
        "electric_displacement",
        "electrostatic_energy_density",
    )


def test_analysis_fields_share_simulation_grid() -> None:
    simulation = create_analysis_simulation()

    result = PoissonSolver().solve(
        simulation
    )

    analysis = analyse_electrostatics(
        simulation,
        result,
    )

    assert analysis.grid is simulation.grid

    for field_value in analysis.fields:
        assert field_value.grid is simulation.grid


def test_analysis_of_linear_potential_has_constant_field() -> None:
    simulation = create_analysis_simulation(
        number_of_points=21,
        left_voltage=0.0,
        right_voltage=1.0,
    )

    result = PoissonSolver().solve(
        simulation
    )

    analysis = analyse_electrostatics(
        simulation,
        result,
    )

    expected_field = (
        -1.0
        / simulation.grid.physical_size[0]
    )

    np.testing.assert_allclose(
        analysis.electric_field.values,
        expected_field,
        rtol=1.0e-11,
        atol=1.0e-6,
    )


def test_analysis_of_uniform_material_has_constant_displacement() -> None:
    simulation = create_analysis_simulation(
        number_of_points=21,
    )

    result = PoissonSolver().solve(
        simulation
    )

    analysis = analyse_electrostatics(
        simulation,
        result,
    )

    np.testing.assert_allclose(
        analysis.electric_displacement.values,
        analysis.electric_displacement.values[0],
        rtol=1.0e-11,
        atol=1.0e-18,
    )


def test_analysis_of_linear_potential_has_uniform_energy_density() -> None:
    simulation = create_analysis_simulation(
        number_of_points=21,
    )

    result = PoissonSolver().solve(
        simulation
    )

    analysis = analyse_electrostatics(
        simulation,
        result,
    )

    np.testing.assert_allclose(
        analysis.energy_density.values,
        analysis.energy_density.values[0],
        rtol=1.0e-11,
        atol=1.0e-12,
    )

    assert np.all(
        analysis.energy_density.values > 0.0
    )


def test_analysis_get_field() -> None:
    simulation = create_analysis_simulation()

    result = PoissonSolver().solve(
        simulation
    )

    analysis = analyse_electrostatics(
        simulation,
        result,
    )

    selected = analysis.get_field(
        "electric_field"
    )

    assert selected is analysis.electric_field


def test_analysis_get_field_rejects_unknown_name() -> None:
    simulation = create_analysis_simulation()

    result = PoissonSolver().solve(
        simulation
    )

    analysis = analyse_electrostatics(
        simulation,
        result,
    )

    with pytest.raises(
        KeyError,
        match="missing",
    ):
        analysis.get_field(
            "missing"
        )


def test_analysis_as_dict_returns_all_fields() -> None:
    simulation = create_analysis_simulation()

    result = PoissonSolver().solve(
        simulation
    )

    analysis = analyse_electrostatics(
        simulation,
        result,
    )

    fields = analysis.as_dict()

    assert fields == {
        "electrostatic_potential": analysis.potential,
        "electric_field": analysis.electric_field,
        "electric_displacement": (
            analysis.electric_displacement
        ),
        "electrostatic_energy_density": (
            analysis.energy_density
        ),
    }

    assert fields is not analysis.as_dict()


def test_analyse_electrostatics_rejects_invalid_simulation() -> None:
    simulation = create_analysis_simulation()

    result = PoissonSolver().solve(
        simulation
    )

    with pytest.raises(
        TypeError,
        match="Simulation instance",
    ):
        analyse_electrostatics(
            "invalid",
            result,
        )


def test_analyse_electrostatics_rejects_invalid_result() -> None:
    simulation = create_analysis_simulation()

    with pytest.raises(
        TypeError,
        match="SimulationResult",
    ):
        analyse_electrostatics(
            simulation,
            "invalid",
        )


def test_analyse_electrostatics_rejects_mismatched_grid() -> None:
    simulation = create_analysis_simulation(
        number_of_points=11,
    )

    other_simulation = create_analysis_simulation(
        number_of_points=12,
    )

    result = PoissonSolver().solve(
        other_simulation
    )

    with pytest.raises(
        ValueError,
        match="simulation grid",
    ):
        analyse_electrostatics(
            simulation,
            result,
        )


def test_analyse_electrostatics_rejects_missing_potential() -> None:
    simulation = create_analysis_simulation()

    unrelated_field = Field.zeros(
        name="carrier_density",
        units="1/m^3",
        grid=simulation.grid,
    )

    result = SimulationResult(
        fields={
            "carrier_density": unrelated_field,
        },
        converged=True,
        iterations=1,
        residual_history=np.asarray(
            [0.0],
            dtype=np.float64,
        ),
        runtime_seconds=0.0,
        solver_name="test_solver",
        backend_name="numpy",
    )

    with pytest.raises(
        KeyError,
        match="electrostatic_potential",
    ):
        analyse_electrostatics(
            simulation,
            result,
        )


def test_analyse_electrostatics_rejects_invalid_potential_units() -> None:
    simulation = create_analysis_simulation()

    invalid_potential = Field.zeros(
        name="electrostatic_potential",
        units="mV",
        grid=simulation.grid,
    )

    result = create_manual_result(
        simulation,
        invalid_potential,
    )

    with pytest.raises(
        ValueError,
        match="units must be 'V'",
    ):
        analyse_electrostatics(
            simulation,
            result,
        )

# electrostatic analysis tests

def test_electrostatic_analysis_rejects_mismatched_grids() -> None:
    grid_a = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    grid_b = Grid(
        shape=(6,),
        spacing=(1.0e-9,),
    )

    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid_a,
    )

    electric_field = Field.zeros(
        name="electric_field",
        units="V/m",
        grid=grid_b,
    )

    displacement = Field.zeros(
        name="electric_displacement",
        units="C/m^2",
        grid=grid_a,
    )

    energy_density = Field.zeros(
        name="electrostatic_energy_density",
        units="J/m^3",
        grid=grid_a,
    )

    with pytest.raises(
        ValueError,
        match="same grid",
    ):
        ElectrostaticAnalysis(
            potential=potential,
            electric_field=electric_field,
            electric_displacement=displacement,
            energy_density=energy_density,
        )


def test_electrostatic_analysis_rejects_invalid_units() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    potential = Field.zeros(
        name="electrostatic_potential",
        units="V",
        grid=grid,
    )

    invalid_electric_field = Field.zeros(
        name="electric_field",
        units="V",
        grid=grid,
    )

    displacement = Field.zeros(
        name="electric_displacement",
        units="C/m^2",
        grid=grid,
    )

    energy_density = Field.zeros(
        name="electrostatic_energy_density",
        units="J/m^3",
        grid=grid,
    )

    with pytest.raises(
        ValueError,
        match="Electric Field units",
    ):
        ElectrostaticAnalysis(
            potential=potential,
            electric_field=invalid_electric_field,
            electric_displacement=displacement,
            energy_density=energy_density,
        )