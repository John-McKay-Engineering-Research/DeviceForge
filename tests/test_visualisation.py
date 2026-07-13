import matplotlib
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deviceforge import Field, Grid, SimulationResult
from deviceforge.postprocessing import compute_electric_field
from deviceforge.visualisation import (
    plot_convergence,
    plot_scalar_field,
    plot_vector_field,
    save_figure,
)


matplotlib.use("Agg")


@pytest.fixture
def grid_2d() -> Grid:
    return Grid(
        shape=(6, 4),
        spacing=(1.0e-9, 1.0e-9),
    )


@pytest.fixture
def potential_field(grid_2d: Grid) -> Field:
    x_coordinates = np.linspace(
        0.0,
        1.0,
        grid_2d.shape[0],
    )

    values = np.repeat(
        x_coordinates[:, np.newaxis],
        grid_2d.shape[1],
        axis=1,
    )

    return Field(
        name="electrostatic_potential",
        units="V",
        grid=grid_2d,
        values=values,
    )


@pytest.fixture
def result(
    potential_field: Field,
) -> SimulationResult:
    return SimulationResult(
        fields={
            "electrostatic_potential": potential_field,
        },
        converged=True,
        iterations=3,
        residual_history=np.array(
            [1.0e-1, 1.0e-2, 1.0e-3],
        ),
        runtime_seconds=0.01,
        solver_name="jacobi",
    )


def test_plot_scalar_field(
    potential_field: Field,
) -> None:
    figure, axes = plot_scalar_field(
        potential_field,
    )

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert axes.get_xlabel() == "x [nm]"
    assert axes.get_ylabel() == "y [nm]"


def test_plot_convergence(
    result: SimulationResult,
) -> None:
    figure, axes = plot_convergence(result)

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert axes.get_yscale() == "log"


def test_save_figure(
    potential_field: Field,
    tmp_path,
) -> None:
    figure, _ = plot_scalar_field(
        potential_field,
    )

    output_path = tmp_path / "figures" / "potential.png"

    save_figure(
        figure,
        output_path,
    )

    assert output_path.exists()


def test_scalar_plot_rejects_non_2d_field() -> None:
    grid = Grid(
        shape=(5,),
        spacing=(1.0e-9,),
    )

    field = Field.zeros(
        name="potential",
        units="V",
        grid=grid,
    )

    with pytest.raises(ValueError, match="two-dimensional"):
        plot_scalar_field(field)


def test_convergence_plot_rejects_zero_iterations(
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

    with pytest.raises(ValueError, match="zero-iteration"):
        plot_convergence(result)


def test_plot_vector_field(
    potential_field: Field,
) -> None:
    electric_field = compute_electric_field(
        potential_field
    )

    figure, axes = plot_vector_field(
        electric_field,
        stride=2,
    )

    assert isinstance(figure, Figure)
    assert isinstance(axes, Axes)
    assert axes.get_xlabel() == "x [nm]"
    assert axes.get_ylabel() == "y [nm]"


def test_invalid_vector_stride_raises_value_error(
    potential_field: Field,
) -> None:
    electric_field = compute_electric_field(
        potential_field
    )

    with pytest.raises(ValueError, match="stride"):
        plot_vector_field(
            electric_field,
            stride=0,
        )