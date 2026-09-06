from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# from deviceforge.core import Field
from deviceforge.core import (
    FaceField,
    Field,
)
from deviceforge.solvers import PoissonSolver

from deviceforge.visualisation import (
    plot_electric_displacement,
    plot_electric_field,
    plot_electrostatic_energy_density,
    plot_electrostatic_potential,
    plot_face_electric_displacement,
    plot_relative_permittivity,
    plot_residual_history,
)

from deviceforge.workflows import ElectrostaticWorkflow


@pytest.fixture
def workflow_output(
    simulation,
):
    """Return a completed electrostatic workflow output."""

    workflow = ElectrostaticWorkflow(
        simulation=simulation,
        solver=PoissonSolver(),
    )

    return workflow.run()


@pytest.mark.parametrize(
    "plot_function",
    [
        plot_electrostatic_potential,
        plot_electric_field,
        plot_electric_displacement,
        plot_electrostatic_energy_density,
        plot_residual_history,
    ],
)
def test_workflow_plot_returns_figure_and_axes(
    workflow_output,
    plot_function,
) -> None:
    figure, axis = plot_function(
        workflow_output
    )

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)

    plt.close(figure)


def test_potential_plot_contains_correct_coordinates(
    workflow_output,
) -> None:
    figure, axis = plot_electrostatic_potential(
        workflow_output
    )

    plotted_line = axis.lines[0]

    expected_coordinates = (
        workflow_output
        .potential
        .grid
        .coordinates(0)
        * 1.0e9
    )

    np.testing.assert_allclose(
        plotted_line.get_xdata(),
        expected_coordinates,
    )

    np.testing.assert_allclose(
        plotted_line.get_ydata(),
        workflow_output.potential.values,
    )

    plt.close(figure)


def test_electric_field_plot_contains_field_values(
    workflow_output,
) -> None:
    figure, axis = plot_electric_field(
        workflow_output
    )

    plotted_line = axis.lines[0]

    np.testing.assert_allclose(
        plotted_line.get_ydata(),
        workflow_output.electric_field.values,
    )

    assert axis.get_ylabel() == (
        "Electric field (V/m)"
    )

    plt.close(figure)


def test_relative_permittivity_plot(
    workflow_output,
) -> None:
    relative_permittivity = Field.full(
        name="relative_permittivity",
        units="dimensionless",
        grid=workflow_output.potential.grid,
        fill_value=11.7,
    )

    figure, axis = plot_relative_permittivity(
        workflow_output,
        relative_permittivity,
    )

    assert isinstance(figure, Figure)
    assert isinstance(axis, Axes)

    np.testing.assert_allclose(
        axis.lines[0].get_ydata(),
        11.7,
    )

    plt.close(figure)


def test_relative_permittivity_plot_rejects_invalid_units(
    workflow_output,
) -> None:
    invalid_field = Field.full(
        name="absolute_permittivity",
        units="F/m",
        grid=workflow_output.potential.grid,
        fill_value=1.0,
    )

    with pytest.raises(
        ValueError,
        match="dimensionless",
    ):
        plot_relative_permittivity(
            workflow_output,
            invalid_field,
        )


def test_plot_rejects_invalid_workflow_output() -> None:
    with pytest.raises(
        TypeError,
        match="ElectrostaticWorkflowResult",
    ):
        plot_electrostatic_potential(
            "invalid"
        )


def test_plot_functions_do_not_call_show(
    workflow_output,
    monkeypatch,
) -> None:
    def fail_if_called() -> None:
        pytest.fail(
            "Library plotting functions must not call plt.show()."
        )

    monkeypatch.setattr(
        plt,
        "show",
        fail_if_called,
    )

    figures = [
        plot_electrostatic_potential(
            workflow_output
        )[0],
        plot_electric_field(
            workflow_output
        )[0],
        plot_electric_displacement(
            workflow_output
        )[0],
        plot_electrostatic_energy_density(
            workflow_output
        )[0],
        plot_residual_history(
            workflow_output
        )[0],
    ]

    for figure in figures:
        plt.close(figure)


def test_face_displacement_plot_returns_figure_and_axes(
    workflow_output,
) -> None:
    grid = workflow_output.potential.grid

    face_displacement = FaceField(
        name="face_electric_displacement",
        units="C/m^2",
        grid=grid,
        values=np.full(
            grid.shape[0] - 1,
            1.0e-3,
            dtype=np.float64,
        ),
    )

    figure, axis = (
        plot_face_electric_displacement(
            face_displacement
        )
    )

    assert isinstance(
        figure,
        Figure,
    )

    assert isinstance(
        axis,
        Axes,
    )

    plt.close(
        figure
    )


def test_face_displacement_plot_uses_face_coordinates(
    workflow_output,
) -> None:
    grid = workflow_output.potential.grid

    values = np.linspace(
        1.0e-3,
        2.0e-3,
        grid.shape[0] - 1,
        dtype=np.float64,
    )

    face_displacement = FaceField(
        name="face_electric_displacement",
        units="C/m^2",
        grid=grid,
        values=values,
    )

    figure, axis = (
        plot_face_electric_displacement(
            face_displacement
        )
    )

    plotted_line = axis.lines[0]

    expected_coordinates = (
        face_displacement.coordinates()
        * 1.0e9
    )

    np.testing.assert_allclose(
        plotted_line.get_xdata(),
        expected_coordinates,
    )

    np.testing.assert_allclose(
        plotted_line.get_ydata(),
        values,
    )

    assert axis.get_xlabel() == (
        "Position (nm)"
    )

    assert axis.get_ylabel() == (
        "Electric displacement (C/m²)"
    )

    assert axis.get_title() == (
        "Face-Centred Electric Displacement"
    )

    plt.close(
        figure
    )


def test_constant_face_displacement_uses_stable_vertical_range(
    workflow_output,
) -> None:
    grid = workflow_output.potential.grid

    constant_value = 2.5e-3

    face_displacement = FaceField(
        name="face_electric_displacement",
        units="C/m^2",
        grid=grid,
        values=np.full(
            grid.shape[0] - 1,
            constant_value,
            dtype=np.float64,
        ),
    )

    figure, axis = (
        plot_face_electric_displacement(
            face_displacement
        )
    )

    lower_limit, upper_limit = (
        axis.get_ylim()
    )

    expected_margin = (
        abs(constant_value)
        * 1.0e-3
    )

    assert lower_limit == pytest.approx(
        constant_value
        - expected_margin
    )

    assert upper_limit == pytest.approx(
        constant_value
        + expected_margin
    )

    assert len(
        axis.texts
    ) == 1

    annotation = (
        axis.texts[0].get_text()
    )

    assert (
        "Mean D"
        in annotation
    )

    assert (
        "Relative variation"
        in annotation
    )

    plt.close(
        figure
    )


def test_face_displacement_plot_rejects_invalid_units(
    workflow_output,
) -> None:
    grid = workflow_output.potential.grid

    invalid_face_field = FaceField(
        name="invalid_face_field",
        units="V/m",
        grid=grid,
        values=np.ones(
            grid.shape[0] - 1,
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="C/m\\^2",
    ):
        plot_face_electric_displacement(
            invalid_face_field
        )


def test_face_displacement_plot_rejects_non_face_field() -> None:
    with pytest.raises(
        TypeError,
        match="FaceField",
    ):
        plot_face_electric_displacement(
            "invalid"
        )