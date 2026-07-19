"""DeviceForge one-dimensional equilibrium PN-junction demonstration.

This example:

1. Builds the same PN junction used by the Gummel solver tests.
2. Solves the equilibrium drift-diffusion system.
3. Prints solver and terminal-current information.
4. Produces plots for:
   - doping,
   - electrostatic potential,
   - carrier concentrations,
   - current density,
   - recombination,
   - convergence history.
5. Saves the figures under:

   figures/pn_junction_gummel_1d/

Run from the repository root with:

    python examples/pn_junction_gummel_1d.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from deviceforge.solvers import (
    GummelDriftDiffusionSolver1D,
    SolverConfiguration,
)

from deviceforge.solvers.gummel_1d import (
GummelDriftDiffusionSolver1D,
)

from deviceforge.solvers.base import (
SolverConfiguration,
)
# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

TEST_GUMMEL_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "test_gummel_1d.py"
)

FIGURE_DIRECTORY = (
    REPOSITORY_ROOT
    / "examples"
    / "figures"
    / "examples"
    / "pn_junction_forward_bias_005V"
)


# ---------------------------------------------------------------------------
# Load the already-tested PN-junction construction code
# ---------------------------------------------------------------------------

def load_gummel_test_module() -> ModuleType:
    """Load tests/test_gummel_1d.py as a Python module.

    The demonstration initially reuses the tested PN-junction builder so
    the example and unit tests solve exactly the same physical device.
    """

    if not TEST_GUMMEL_PATH.exists():
        raise FileNotFoundError(
            "Could not find the Gummel test module at:\n"
            f"{TEST_GUMMEL_PATH}"
        )

    module_specification = importlib.util.spec_from_file_location(
        "deviceforge_gummel_test_module",
        TEST_GUMMEL_PATH,
    )

    if module_specification is None:
        raise ImportError(
            "Python could not create an import specification for "
            f"{TEST_GUMMEL_PATH}"
        )

    if module_specification.loader is None:
        raise ImportError(
            "Python could not create an import loader for "
            f"{TEST_GUMMEL_PATH}"
        )

    module = importlib.util.module_from_spec(
        module_specification
    )

    module_specification.loader.exec_module(module)

    return module


GUMMEL_TEST_MODULE = load_gummel_test_module()


def required_module_attribute(
    attribute_name: str,
) -> Any:
    """Return a required object from the loaded Gummel test module."""

    if not hasattr(GUMMEL_TEST_MODULE, attribute_name):
        raise AttributeError(
            "The demonstration expected "
            f"tests/test_gummel_1d.py to define "
            f"'{attribute_name}', but it was not found."
        )

    return getattr(
        GUMMEL_TEST_MODULE,
        attribute_name,
    )


build_pn_junction_simulation = required_module_attribute(
    "build_pn_junction_simulation"
)

# ---------------------------------------------------------------------------
# Result field names
# ---------------------------------------------------------------------------

POTENTIAL_FIELD_NAME = "electrostatic_potential"
ELECTRON_FIELD_NAME = "electron_concentration"
HOLE_FIELD_NAME = "hole_concentration"

ELECTRON_CURRENT_FIELD_NAME = (
    "electron_current_density_x_edges"
)

HOLE_CURRENT_FIELD_NAME = (
    "hole_current_density_x_edges"
)

TOTAL_CURRENT_FIELD_NAME = (
    "total_current_density_x_edges"
)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    applied_voltage: float = 0.0,
) -> tuple[Any, Any]:
    """Build and solve the one-dimensional PN junction.

    Parameters
    ----------
    applied_voltage:
        Applied terminal voltage in volts. Positive values represent
        forward bias according to the solver's voltage convention.
    """

    simulation = build_pn_junction_simulation()

    solver = GummelDriftDiffusionSolver1D(
        applied_voltage=applied_voltage,
        damping_factor=0.2,
        current_conservation_tolerance=1.0e-2,
        enforce_current_conservation=True,
        configuration=SolverConfiguration(
            tolerance=1.0e-10,
            max_iterations=500,
        ),
    )

    result = solver.solve(simulation)

    # temporary verification

    print(
        "Current conservation enforced:",
        result.metadata.get("current_conservation_enforced"),
    )

    print(
        "Current conservation tolerance:",
        result.metadata.get("current_conservation_tolerance"),
    )

    print(
        "Current conservation achieved:",
        result.metadata.get("current_conservation_achieved"),
    )

    print(
        "Update convergence achieved:",
        result.metadata.get("update_convergence_achieved"),
    )
    return simulation, result


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def node_coordinates_metres(
    simulation: Any,
) -> np.ndarray:
    """Return the one-dimensional node coordinates in metres."""

    grid = simulation.grid

    number_of_nodes = grid.shape[0]
    spacing = grid.spacing[0]
    origin = grid.origin[0]

    return (
        origin
        + np.arange(
            number_of_nodes,
            dtype=float,
        )
        * spacing
    )


def edge_coordinates_metres(
    simulation: Any,
) -> np.ndarray:
    """Return the centres of the one-dimensional grid edges."""

    node_coordinates = node_coordinates_metres(
        simulation
    )

    return 0.5 * (
        node_coordinates[:-1]
        + node_coordinates[1:]
    )


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

def get_metadata_float(
    result: Any,
    key: str,
) -> float:
    """Read a numerical metadata value from a simulation result."""

    if key not in result.metadata:
        raise KeyError(
            f"Simulation result metadata does not contain '{key}'."
        )

    return float(result.metadata[key])


def get_optional_metadata_float(
    result: Any,
    key: str,
) -> float:
    """Read optional numerical metadata, returning NaN when absent."""

    value = result.metadata.get(key)

    if value is None:
        return float("nan")

    return float(value)


def find_recombination_field_name(
    result: Any,
) -> str | None:
    """Find the recombination field without assuming an exact key."""

    for field_name in result.field_names:
        lower_name = field_name.lower()

        if "recombination" in lower_name:
            return field_name

    return None


def current_density_units(
    result: Any,
) -> str:
    """Return the current-density units stored by the solver."""

    return str(
        result.metadata.get(
            "terminal_current_density_units",
            "A/m^2",
        )
    )


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_simulation_summary(
    simulation: Any,
    result: Any,
    applied_voltage: float,
) -> None:
    """Print convergence and current-density information."""

    total_current = result.get_field(
        TOTAL_CURRENT_FIELD_NAME
    )

    left_current = get_metadata_float(
        result,
        "left_terminal_current_density",
    )

    right_current = get_metadata_float(
        result,
        "right_terminal_current_density",
    )

    average_current = get_metadata_float(
        result,
        "average_terminal_current_density",
    )

    current_nonuniformity = get_metadata_float(
        result,
        "current_density_nonuniformity",
    )

    relative_current_nonuniformity = get_metadata_float(
        result,
        "relative_current_density_nonuniformity",
    )

    poisson_residual = get_metadata_float(
        result,
        "final_poisson_residual",
    )

    electron_continuity_defect = get_optional_metadata_float(
        result,
        "final_electron_continuity_residual",
    )

    hole_continuity_defect = get_optional_metadata_float(
        result,
        "final_hole_continuity_residual",
    )

    electrons = np.asarray(
        result.get_field(ELECTRON_FIELD_NAME).values,
        dtype=float,
    )

    holes = np.asarray(
        result.get_field(HOLE_FIELD_NAME).values,
        dtype=float,
    )

    units = current_density_units(result)

    print()
    print("DeviceForge 1D PN-Junction Demonstration")
    print("=" * 46)
    print(f"Simulation:               {simulation.name}")
    print(f"Applied voltage:          {applied_voltage:.6f} V")
    print(f"Converged:                {result.converged}")
    print(f"Iterations:               {result.iterations}")
    print(
        "Final residual:           "
        f"{result.final_residual:.6e}"
    )
    print(
        "Runtime:                  "
        f"{result.runtime_seconds:.6f} s"
    )
    print(
        "Left terminal current:    "
        f"{left_current:.6e} {units}"
    )
    print(
        "Right terminal current:   "
        f"{right_current:.6e} {units}"
    )
    print(
        "Average terminal current: "
        f"{average_current:.6e} {units}"
    )
    print(
        "Current nonuniformity:    "
        f"{current_nonuniformity:.6e} {units}"
    )
    print(
        "Relative nonuniformity:   "
        f"{relative_current_nonuniformity:.6e}"
    )
    print(
        "Poisson residual:         "
        f"{poisson_residual:.6e}"
    )
    print(
        "Electron cont. residual:    "
        f"{electron_continuity_defect:.6e}"
    )
    print(
        "Hole cont. residual:        "
        f"{hole_continuity_defect:.6e}"
    )
    print(
        "Minimum electron density: "
        f"{np.min(electrons):.6e} 1/m^3"
    )
    print(
        "Minimum hole density:     "
        f"{np.min(holes):.6e} 1/m^3"
    )
    print(
        "Current edge count:       "
        f"{total_current.values.size}"
    )
    print()
    print("Returned fields:")

    for field_name in result.field_names:
        field = result.get_field(field_name)

        print(
            f"  {field_name}: "
            f"shape={field.values.shape}, "
            f"units={field.units}"
        )

    print()


# ---------------------------------------------------------------------------
# Figure output
# ---------------------------------------------------------------------------

def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    """Save and close a demonstration figure."""

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = FIGURE_DIRECTORY / filename

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# Doping plot
# ---------------------------------------------------------------------------

def plot_doping_profile(
    simulation: Any,
) -> plt.Figure:
    """Plot donor, acceptor and signed net doping."""

    coordinates_nm = (
        node_coordinates_metres(simulation)
        * 1.0e9
    )

    donor_density = np.asarray(
        simulation.device
        .donor_density_field()
        .values,
        dtype=float,
    )

    acceptor_density = np.asarray(
        simulation.device
        .acceptor_density_field()
        .values,
        dtype=float,
    )

    net_doping = donor_density - acceptor_density

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    axis.semilogy(
        coordinates_nm,
        np.maximum(donor_density, 1.0),
        label="Donor density",
    )

    axis.semilogy(
        coordinates_nm,
        np.maximum(acceptor_density, 1.0),
        label="Acceptor density",
    )

    axis.set_xlabel("Position (nm)")
    axis.set_ylabel("Concentration (m$^{-3}$)")
    axis.set_title("PN-Junction Doping Profile")
    axis.grid(True)
    axis.legend()

    # Show the signed net-doping polarity on a secondary axis.
    secondary_axis = axis.twinx()

    secondary_axis.plot(
        coordinates_nm,
        net_doping,
        linestyle="--",
        label="Net doping",
    )

    secondary_axis.set_ylabel(
        "Signed net doping (m$^{-3}$)"
    )

    return figure


# ---------------------------------------------------------------------------
# Electrostatic-potential plot
# ---------------------------------------------------------------------------

def plot_potential(
    simulation: Any,
    result: Any,
) -> plt.Figure:
    """Plot the solved electrostatic potential."""

    coordinates_nm = (
        node_coordinates_metres(simulation)
        * 1.0e9
    )

    potential = result.get_field(
        POTENTIAL_FIELD_NAME
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    axis.plot(
        coordinates_nm,
        potential.values,
    )

    axis.set_xlabel("Position (nm)")
    axis.set_ylabel("Electrostatic potential (V)")
    axis.set_title(
        "PN-Junction Electrostatic Potential"
    )
    axis.grid(True)

    return figure


# ---------------------------------------------------------------------------
# Carrier-concentration plot
# ---------------------------------------------------------------------------

def plot_carrier_concentrations(
    simulation: Any,
    result: Any,
) -> plt.Figure:
    """Plot electron and hole concentrations."""

    coordinates_nm = (
        node_coordinates_metres(simulation)
        * 1.0e9
    )

    electrons = result.get_field(
        ELECTRON_FIELD_NAME
    )

    holes = result.get_field(
        HOLE_FIELD_NAME
    )

    electron_values = np.maximum(
        np.asarray(
            electrons.values,
            dtype=float,
        ),
        1.0,
    )

    hole_values = np.maximum(
        np.asarray(
            holes.values,
            dtype=float,
        ),
        1.0,
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    axis.semilogy(
        coordinates_nm,
        electron_values,
        label="Electron concentration",
    )

    axis.semilogy(
        coordinates_nm,
        hole_values,
        label="Hole concentration",
    )

    axis.set_xlabel("Position (nm)")
    axis.set_ylabel(
        "Carrier concentration (m$^{-3}$)"
    )
    axis.set_title(
        "PN-Junction Carrier Concentrations"
    )
    axis.grid(True)
    axis.legend()

    return figure


# ---------------------------------------------------------------------------
# Current-density plot
# ---------------------------------------------------------------------------

def plot_current_density(
    simulation: Any,
    result: Any,
) -> plt.Figure:
    """Plot electron, hole and total edge current density."""

    coordinates_nm = (
        edge_coordinates_metres(simulation)
        * 1.0e9
    )

    electron_current = result.get_field(
        ELECTRON_CURRENT_FIELD_NAME
    )

    hole_current = result.get_field(
        HOLE_CURRENT_FIELD_NAME
    )

    total_current = result.get_field(
        TOTAL_CURRENT_FIELD_NAME
    )

    if electron_current.values.shape != coordinates_nm.shape:
        raise ValueError(
            "Electron current field shape does not match "
            "the edge-coordinate shape."
        )

    if hole_current.values.shape != coordinates_nm.shape:
        raise ValueError(
            "Hole current field shape does not match "
            "the edge-coordinate shape."
        )

    if total_current.values.shape != coordinates_nm.shape:
        raise ValueError(
            "Total current field shape does not match "
            "the edge-coordinate shape."
        )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    axis.plot(
        coordinates_nm,
        electron_current.values,
        label="Electron current",
    )

    axis.plot(
        coordinates_nm,
        hole_current.values,
        label="Hole current",
    )

    axis.plot(
        coordinates_nm,
        total_current.values,
        linewidth=2.0,
        label="Total current",
    )

    axis.set_xlabel("Position (nm)")
    axis.set_ylabel("Current density (A/m$^2$)")
    axis.set_title(
        "Scharfetter–Gummel Current Density"
    )
    axis.grid(True)
    axis.legend()

    return figure


# ---------------------------------------------------------------------------
# Recombination plot
# ---------------------------------------------------------------------------

def plot_recombination(
    simulation: Any,
    result: Any,
) -> plt.Figure | None:
    """Plot the returned recombination-rate field, when available."""

    recombination_field_name = (
        find_recombination_field_name(result)
    )

    if recombination_field_name is None:
        print(
            "No recombination field was returned; "
            "skipping the recombination plot."
        )

        return None

    recombination = result.get_field(
        recombination_field_name
    )

    coordinates_nm = (
        node_coordinates_metres(simulation)
        * 1.0e9
    )

    if recombination.values.shape != coordinates_nm.shape:
        print(
            "The recombination field is not node-centred; "
            "skipping the recombination plot."
        )

        return None

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    axis.plot(
        coordinates_nm,
        recombination.values,
    )

    axis.set_xlabel("Position (nm)")
    axis.set_ylabel(
        f"Recombination rate ({recombination.units})"
    )
    axis.set_title(
        "Shockley–Read–Hall Recombination"
    )
    axis.grid(True)

    return figure


# ---------------------------------------------------------------------------
# Residual-history plot
# ---------------------------------------------------------------------------

def plot_residual_history(
    result: Any,
) -> plt.Figure:
    """Plot Gummel convergence residual against iteration."""

    residuals = np.asarray(
        result.residual_history,
        dtype=float,
    )

    iterations = np.arange(
        1,
        residuals.size + 1,
        dtype=int,
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    if residuals.size == 0:
        axis.text(
            0.5,
            0.5,
            "No residual history was returned.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
        )
    elif np.all(residuals > 0.0):
        axis.semilogy(
            iterations,
            residuals,
        )
    else:
        axis.plot(
            iterations,
            residuals,
        )

    axis.set_xlabel("Gummel iteration")
    axis.set_ylabel("Residual")
    axis.set_title(
        "Gummel Solver Convergence History"
    )
    axis.grid(True)

    return figure


# ---------------------------------------------------------------------------
# Create all figures
# ---------------------------------------------------------------------------

def create_demonstration_figures(
    simulation: Any,
    result: Any,
) -> None:
    """Create and save every PN-junction demonstration figure."""

    save_figure(
        plot_doping_profile(simulation),
        "01_doping_profile.png",
    )

    save_figure(
        plot_potential(
            simulation,
            result,
        ),
        "02_electrostatic_potential.png",
    )

    save_figure(
        plot_carrier_concentrations(
            simulation,
            result,
        ),
        "03_carrier_concentrations.png",
    )

    save_figure(
        plot_current_density(
            simulation,
            result,
        ),
        "04_current_density.png",
    )

    recombination_figure = plot_recombination(
        simulation,
        result,
    )

    if recombination_figure is not None:
        save_figure(
            recombination_figure,
            "05_recombination.png",
        )

    save_figure(
        plot_residual_history(result),
        "06_residual_history.png",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete PN-junction demonstration."""

    applied_voltage = 0.05

    simulation, result = run_simulation(
        applied_voltage=applied_voltage,
    )

    print_simulation_summary(
        simulation,
        result,
        applied_voltage,
    )

    if not result.converged:
        raise RuntimeError(
            "The PN-junction Gummel solver did not converge. "
            "Figures have not been generated."
        )

    create_demonstration_figures(
        simulation,
        result,
    )

    print()
    print(
        "PN-junction demonstration completed successfully."
    )
    print(
        f"Figures written to: {FIGURE_DIRECTORY}"
    )
    print()


if __name__ == "__main__":
    main()