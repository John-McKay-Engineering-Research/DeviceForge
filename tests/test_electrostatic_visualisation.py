from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deviceforge.core import Field
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