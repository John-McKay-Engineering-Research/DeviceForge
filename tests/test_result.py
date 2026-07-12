import numpy as np
import pytest

from deviceforge import Field, Grid, SimulationResult


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(4, 3),
        spacing=(1.0e-9, 1.0e-9),
    )


@pytest.fixture
def potential_field(grid_2d: Grid) -> Field:
    values = np.linspace(
        0.0,
        1.0,
        grid_2d.number_of_points,
    ).reshape(grid_2d.shape)

    return Field(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
        values=values,
    )


@pytest.fixture
def converged_result(
    potential_field: Field,
) -> SimulationResult:
    return SimulationResult(
        fields={
            "electrostatic_potential": potential_field,
        },
        converged=True,
        iterations=4,
        residual_history=np.array(
            [1.0, 0.1, 0.01, 0.001],
        ),
        runtime_seconds=0.025,
        solver_name="jacobi",
        backend_name="numpy",
        metadata={
            "tolerance": 1.0e-3,
        },
    )


def test_create_simulation_result(
    converged_result: SimulationResult,
) -> None:
    assert converged_result.converged
    assert converged_result.iterations == 4
    assert converged_result.runtime_seconds == pytest.approx(0.025)
    assert converged_result.solver_name == "jacobi"
    assert converged_result.backend_name == "numpy"


def test_grid_is_derived_from_fields(
    converged_result: SimulationResult,
    potential_field: Field,
) -> None:
    assert converged_result.grid is potential_field.grid


def test_field_names(
    converged_result: SimulationResult,
) -> None:
    assert converged_result.field_names == (
        "electrostatic_potential",
    )


def test_get_field(
    converged_result: SimulationResult,
    potential_field: Field,
) -> None:
    assert (
        converged_result.get_field(
            "electrostatic_potential"
        )
        is potential_field
    )


def test_potential_property(
    converged_result: SimulationResult,
    potential_field: Field,
) -> None:
    assert converged_result.potential is potential_field


def test_missing_field_raises_key_error(
    converged_result: SimulationResult,
) -> None:
    with pytest.raises(KeyError, match="charge_density"):
        converged_result.get_field("charge_density")


def test_residual_properties(
    converged_result: SimulationResult,
) -> None:
    assert converged_result.initial_residual == pytest.approx(1.0)
    assert converged_result.final_residual == pytest.approx(0.001)
    assert converged_result.residual_reduction == pytest.approx(
        1_000.0
    )


def test_summary(
    converged_result: SimulationResult,
) -> None:
    summary = converged_result.summary()

    assert summary["converged"] is True
    assert summary["iterations"] == 4
    assert summary["solver_name"] == "jacobi"
    assert summary["final_residual"] == pytest.approx(0.001)
    assert summary["field_names"] == (
        "electrostatic_potential",
    )


def test_zero_iteration_result(
    potential_field: Field,
) -> None:
    result = SimulationResult(
        fields={
            "electrostatic_potential": potential_field,
        },
        converged=False,
        iterations=0,
        residual_history=np.array([]),
        runtime_seconds=0.0,
        solver_name="jacobi",
    )

    assert result.initial_residual is None
    assert result.final_residual is None
    assert result.residual_reduction is None


def test_zero_final_residual_has_no_reduction_ratio(
    potential_field: Field,
) -> None:
    result = SimulationResult(
        fields={
            "electrostatic_potential": potential_field,
        },
        converged=True,
        iterations=2,
        residual_history=np.array([1.0, 0.0]),
        runtime_seconds=0.01,
        solver_name="jacobi",
    )

    assert result.final_residual == pytest.approx(0.0)
    assert result.residual_reduction is None


def test_fields_mapping_is_read_only(
    converged_result: SimulationResult,
) -> None:
    with pytest.raises(TypeError):
        converged_result.fields["new_field"] = (
            converged_result.potential
        )


def test_metadata_mapping_is_read_only(
    converged_result: SimulationResult,
) -> None:
    with pytest.raises(TypeError):
        converged_result.metadata["new_value"] = 10


def test_residual_history_is_read_only(
    converged_result: SimulationResult,
) -> None:
    with pytest.raises(ValueError):
        converged_result.residual_history[0] = 2.0


def test_empty_fields_raise_value_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SimulationResult(
            fields={},
            converged=False,
            iterations=0,
            residual_history=np.array([]),
            runtime_seconds=0.0,
            solver_name="jacobi",
        )


def test_non_field_value_raises_type_error(
    grid_2d: Grid,
) -> None:
    with pytest.raises(TypeError, match="Field"):
        SimulationResult(
            fields={
                "invalid": np.zeros(grid_2d.shape),
            },
            converged=False,
            iterations=0,
            residual_history=np.array([]),
            runtime_seconds=0.0,
            solver_name="jacobi",
        )


def test_fields_from_different_grids_raise_value_error(
    grid_2d: Grid,
    potential_field: Field,
) -> None:
    other_grid = Grid(
        shape=(5, 3),
        spacing=(1.0e-9, 1.0e-9),
    )

    other_field = Field.zeros(
        name="charge_density",
        units="C/m^3",
        grid=other_grid,
    )

    with pytest.raises(ValueError, match="same computational grid"):
        SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
                "charge_density": other_field,
            },
            converged=False,
            iterations=0,
            residual_history=np.array([]),
            runtime_seconds=0.0,
            solver_name="jacobi",
        )


def test_residual_length_must_match_iterations(
    potential_field: Field,
) -> None:
    with pytest.raises(ValueError, match="length"):
        SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=False,
            iterations=3,
            residual_history=np.array([1.0, 0.5]),
            runtime_seconds=0.0,
            solver_name="jacobi",
        )


def test_multidimensional_residual_history_raises_value_error(
    potential_field: Field,
) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=False,
            iterations=2,
            residual_history=np.array(
                [
                    [1.0],
                    [0.5],
                ]
            ),
            runtime_seconds=0.0,
            solver_name="jacobi",
        )


@pytest.mark.parametrize(
    "residual_history",
    [
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
        np.array([1.0, -np.inf]),
    ],
)
def test_non_finite_residual_raises_value_error(
    potential_field: Field,
    residual_history: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=False,
            iterations=2,
            residual_history=residual_history,
            runtime_seconds=0.0,
            solver_name="jacobi",
        )


def test_negative_residual_raises_value_error(
    potential_field: Field,
) -> None:
    with pytest.raises(ValueError, match="negative"):
        SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=False,
            iterations=2,
            residual_history=np.array([1.0, -0.1]),
            runtime_seconds=0.0,
            solver_name="jacobi",
        )


@pytest.mark.parametrize(
    "iterations",
    [
        -1,
        -10,
    ],
)
def test_negative_iterations_raise_value_error(
    potential_field: Field,
    iterations: int,
) -> None:
    with pytest.raises(ValueError, match="negative"):
        SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=False,
            iterations=iterations,
            residual_history=np.array([]),
            runtime_seconds=0.0,
            solver_name="jacobi",
        )


def test_boolean_iterations_raise_type_error(
    potential_field: Field,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=False,
            iterations=True,
            residual_history=np.array([1.0]),
            runtime_seconds=0.0,
            solver_name="jacobi",
        )


@pytest.mark.parametrize(
    "runtime_seconds",
    [
        -1.0,
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_invalid_runtime_raises_value_error(
    potential_field: Field,
    runtime_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="Runtime"):
        SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=False,
            iterations=0,
            residual_history=np.array([]),
            runtime_seconds=runtime_seconds,
            solver_name="jacobi",
        )


@pytest.mark.parametrize(
    "solver_name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_empty_solver_name_raises_value_error(
    potential_field: Field,
    solver_name: str,
) -> None:
    with pytest.raises(ValueError, match="Solver name"):
        SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=False,
            iterations=0,
            residual_history=np.array([]),
            runtime_seconds=0.0,
            solver_name=solver_name,
        )


@pytest.mark.parametrize(
    "backend_name",
    [
        "",
        " ",
        "\t",
    ],
)
def test_empty_backend_name_raises_value_error(
    potential_field: Field,
    backend_name: str,
) -> None:
    with pytest.raises(ValueError, match="Backend name"):
        SimulationResult(
            fields={
                "electrostatic_potential": potential_field,
            },
            converged=False,
            iterations=0,
            residual_history=np.array([]),
            runtime_seconds=0.0,
            solver_name="jacobi",
            backend_name=backend_name,
        )