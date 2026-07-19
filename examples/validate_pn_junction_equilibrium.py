"""Validate the DeviceForge equilibrium 1D PN-junction solution.

This script checks whether the zero-bias Gummel solution satisfies the
principal physical expectations for thermal equilibrium:

1. Mass-action law:
       n * p = n_i^2

2. Built-in potential:
       V_bi = V_T * ln(N_A * N_D / n_i^2)

3. Near-zero total current at equilibrium.

4. Spatially uniform total current.

5. Near-zero net Shockley-Read-Hall recombination.

The script reuses the PN-junction device construction from:

    examples/pn_junction_gummel_1d.py

Run from the repository root with:

    python examples/validate_pn_junction_equilibrium.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Repository and output paths
# ---------------------------------------------------------------------------

EXAMPLES_DIRECTORY = Path(__file__).resolve().parent

PN_JUNCTION_EXAMPLE_PATH = (
    EXAMPLES_DIRECTORY
    / "pn_junction_gummel_1d.py"
)

FIGURE_DIRECTORY = (
    EXAMPLES_DIRECTORY
    / "figures"
    / "examples"
    / "pn_junction_equilibrium_validation"
)


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

ELEMENTARY_CHARGE = 1.602176634e-19
BOLTZMANN_CONSTANT = 1.380649e-23


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
# Load the existing PN-junction example
# ---------------------------------------------------------------------------

def load_python_module(
    module_path: Path,
    module_name: str,
) -> ModuleType:
    """Load a Python file as a module."""

    if not module_path.exists():
        raise FileNotFoundError(
            f"Could not find the required module:\n{module_path}"
        )

    module_specification = (
        importlib.util.spec_from_file_location(
            module_name,
            module_path,
        )
    )

    if module_specification is None:
        raise ImportError(
            "Python could not create a module specification for:\n"
            f"{module_path}"
        )

    if module_specification.loader is None:
        raise ImportError(
            "Python could not create a module loader for:\n"
            f"{module_path}"
        )

    module = importlib.util.module_from_spec(
        module_specification
    )

    module_specification.loader.exec_module(module)

    return module


PN_JUNCTION_EXAMPLE = load_python_module(
    module_path=PN_JUNCTION_EXAMPLE_PATH,
    module_name="deviceforge_pn_junction_example",
)


def required_example_attribute(
    attribute_name: str,
) -> Any:
    """Return a required object from the PN-junction example module."""

    if not hasattr(
        PN_JUNCTION_EXAMPLE,
        attribute_name,
    ):
        raise AttributeError(
            f"{PN_JUNCTION_EXAMPLE_PATH.name} does not define "
            f"'{attribute_name}'."
        )

    return getattr(
        PN_JUNCTION_EXAMPLE,
        attribute_name,
    )


build_pn_junction_simulation = (
    required_example_attribute(
        "build_pn_junction_simulation"
    )
)

GummelDriftDiffusionSolver1D = (
    required_example_attribute(
        "GummelDriftDiffusionSolver1D"
    )
)

SolverConfiguration = required_example_attribute(
    "SolverConfiguration"
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class ValidationMetrics:
    """Store equilibrium-validation measurements."""

    def __init__(
        self,
        intrinsic_concentration: float,
        temperature: float,
        thermal_voltage: float,
        theoretical_built_in_voltage: float,
        simulated_built_in_voltage: float,
        built_in_voltage_relative_error: float,
        maximum_mass_action_relative_error: float,
        mean_mass_action_relative_error: float,
        maximum_total_current_density: float,
        maximum_component_current_density: float,
        current_cancellation_ratio: float,
        total_current_nonuniformity: float,
        maximum_absolute_recombination: float | None,
        integrated_absolute_recombination: float | None,
    ) -> None:
        self.intrinsic_concentration = (
            intrinsic_concentration
        )

        self.temperature = temperature
        self.thermal_voltage = thermal_voltage

        self.theoretical_built_in_voltage = (
            theoretical_built_in_voltage
        )

        self.simulated_built_in_voltage = (
            simulated_built_in_voltage
        )

        self.built_in_voltage_relative_error = (
            built_in_voltage_relative_error
        )

        self.maximum_mass_action_relative_error = (
            maximum_mass_action_relative_error
        )

        self.mean_mass_action_relative_error = (
            mean_mass_action_relative_error
        )

        self.maximum_total_current_density = (
            maximum_total_current_density
        )

        self.maximum_component_current_density = (
            maximum_component_current_density
        )

        self.current_cancellation_ratio = (
            current_cancellation_ratio
        )

        self.total_current_nonuniformity = (
            total_current_nonuniformity
        )

        self.maximum_absolute_recombination = (
            maximum_absolute_recombination
        )

        self.integrated_absolute_recombination = (
            integrated_absolute_recombination
        )


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def node_coordinates_metres(
    simulation: Any,
) -> np.ndarray:
    """Return node-centred coordinates in metres."""

    grid = simulation.grid

    return (
        grid.origin[0]
        + np.arange(
            grid.shape[0],
            dtype=float,
        )
        * grid.spacing[0]
    )


def edge_coordinates_metres(
    simulation: Any,
) -> np.ndarray:
    """Return edge-centred coordinates in metres."""

    nodes = node_coordinates_metres(simulation)

    return 0.5 * (
        nodes[:-1]
        + nodes[1:]
    )


# ---------------------------------------------------------------------------
# Solver execution
# ---------------------------------------------------------------------------

def run_equilibrium_simulation() -> tuple[
    Any,
    Any,
    Any,
]:
    """Build and solve the equilibrium PN junction."""

    simulation = build_pn_junction_simulation()

    solver = GummelDriftDiffusionSolver1D(
        applied_voltage=0.0,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            # Use a strict tolerance for solver verification. Ordinary
            # simulations may continue to use the standard project default.
            tolerance=1.0e-10,
            max_iterations=simulation.max_iterations,
        ),
    )

    result = solver.solve(simulation)

    if not result.converged:
        raise RuntimeError(
            "The equilibrium PN-junction solve did not converge."
        )

    return simulation, solver, result


# ---------------------------------------------------------------------------
# Solver-parameter helpers
# ---------------------------------------------------------------------------

def get_solver_float(
    solver: Any,
    public_name: str,
    private_name: str,
) -> float:
    """Read a numerical solver parameter.

    Public properties are preferred. Private attributes are supported
    temporarily for validation while DeviceForge's public API develops.
    """

    if hasattr(solver, public_name):
        return float(
            getattr(solver, public_name)
        )

    if hasattr(solver, private_name):
        return float(
            getattr(solver, private_name)
        )

    raise AttributeError(
        "The solver does not expose the required parameter "
        f"'{public_name}' or '{private_name}'."
    )


def get_intrinsic_concentration(
    solver: Any,
) -> float:
    """Return the solver intrinsic carrier concentration."""

    return get_solver_float(
        solver=solver,
        public_name="intrinsic_concentration",
        private_name="_intrinsic_concentration",
    )


def get_temperature(
    solver: Any,
) -> float:
    """Return the solver temperature in kelvin."""

    return get_solver_float(
        solver=solver,
        public_name="temperature",
        private_name="_temperature",
    )


# ---------------------------------------------------------------------------
# Recombination-field helper
# ---------------------------------------------------------------------------

def find_recombination_field_name(
    result: Any,
) -> str | None:
    """Return the recombination field name, when present."""

    for field_name in result.field_names:
        if "recombination" in field_name.lower():
            return field_name

    return None


# ---------------------------------------------------------------------------
# Validation calculations
# ---------------------------------------------------------------------------

def calculate_validation_metrics(
    simulation: Any,
    solver: Any,
    result: Any,
) -> tuple[
    ValidationMetrics,
    dict[str, np.ndarray],
]:
    """Calculate equilibrium-validation quantities."""

    potential = np.asarray(
        result.get_field(
            POTENTIAL_FIELD_NAME
        ).values,
        dtype=float,
    )

    electrons = np.asarray(
        result.get_field(
            ELECTRON_FIELD_NAME
        ).values,
        dtype=float,
    )

    holes = np.asarray(
        result.get_field(
            HOLE_FIELD_NAME
        ).values,
        dtype=float,
    )

    electron_current = np.asarray(
        result.get_field(
            ELECTRON_CURRENT_FIELD_NAME
        ).values,
        dtype=float,
    )

    hole_current = np.asarray(
        result.get_field(
            HOLE_CURRENT_FIELD_NAME
        ).values,
        dtype=float,
    )

    total_current = np.asarray(
        result.get_field(
            TOTAL_CURRENT_FIELD_NAME
        ).values,
        dtype=float,
    )

    donors = np.asarray(
        simulation.device
        .donor_density_field()
        .values,
        dtype=float,
    )

    acceptors = np.asarray(
        simulation.device
        .acceptor_density_field()
        .values,
        dtype=float,
    )

    intrinsic_concentration = (
        get_intrinsic_concentration(solver)
    )

    temperature = get_temperature(solver)

    thermal_voltage = (
        BOLTZMANN_CONSTANT
        * temperature
        / ELEMENTARY_CHARGE
    )

    # Use the neutral contact-side doping values.
    acceptor_contact_density = float(
        max(
            acceptors[0] - donors[0],
            acceptors[0],
        )
    )

    donor_contact_density = float(
        max(
            donors[-1] - acceptors[-1],
            donors[-1],
        )
    )

    if acceptor_contact_density <= 0.0:
        raise ValueError(
            "The left contact does not appear to be p-type."
        )

    if donor_contact_density <= 0.0:
        raise ValueError(
            "The right contact does not appear to be n-type."
        )

    theoretical_built_in_voltage = (
        thermal_voltage
        * np.log(
            (
                acceptor_contact_density
                * donor_contact_density
            )
            / intrinsic_concentration**2
        )
    )

    simulated_built_in_voltage = float(
        abs(
            potential[-1]
            - potential[0]
        )
    )

    built_in_voltage_relative_error = float(
        abs(
            simulated_built_in_voltage
            - theoretical_built_in_voltage
        )
        / max(
            abs(theoretical_built_in_voltage),
            np.finfo(float).eps,
        )
    )

    # Thermal equilibrium requires n*p = ni^2.
    mass_action_ratio = (
        electrons
        * holes
        / intrinsic_concentration**2
    )

    mass_action_relative_error = np.abs(
        mass_action_ratio
        - 1.0
    )

    maximum_mass_action_relative_error = float(
        np.max(mass_action_relative_error)
    )

    mean_mass_action_relative_error = float(
        np.mean(mass_action_relative_error)
    )

    maximum_total_current_density = float(
        np.max(
            np.abs(total_current)
        )
    )

    maximum_component_current_density = float(
        max(
            np.max(
                np.abs(electron_current)
            ),
            np.max(
                np.abs(hole_current)
            ),
        )
    )

    current_cancellation_ratio = float(
        maximum_total_current_density
        / max(
            maximum_component_current_density,
            np.finfo(float).eps,
        )
    )

    total_current_nonuniformity = float(
        np.max(total_current)
        - np.min(total_current)
    )

    recombination_field_name = (
        find_recombination_field_name(result)
    )

    recombination: np.ndarray | None = None
    maximum_absolute_recombination: float | None = None
    integrated_absolute_recombination: float | None = None

    if recombination_field_name is not None:
        recombination = np.asarray(
            result.get_field(
                recombination_field_name
            ).values,
            dtype=float,
        )

        maximum_absolute_recombination = float(
            np.max(
                np.abs(recombination)
            )
        )

        integrated_absolute_recombination = float(
            np.trapezoid(
                np.abs(recombination),
                x=node_coordinates_metres(
                    simulation
                ),
            )
        )

    metrics = ValidationMetrics(
        intrinsic_concentration=(
            intrinsic_concentration
        ),
        temperature=temperature,
        thermal_voltage=thermal_voltage,
        theoretical_built_in_voltage=(
            theoretical_built_in_voltage
        ),
        simulated_built_in_voltage=(
            simulated_built_in_voltage
        ),
        built_in_voltage_relative_error=(
            built_in_voltage_relative_error
        ),
        maximum_mass_action_relative_error=(
            maximum_mass_action_relative_error
        ),
        mean_mass_action_relative_error=(
            mean_mass_action_relative_error
        ),
        maximum_total_current_density=(
            maximum_total_current_density
        ),
        maximum_component_current_density=(
            maximum_component_current_density
        ),
        current_cancellation_ratio=(
            current_cancellation_ratio
        ),
        total_current_nonuniformity=(
            total_current_nonuniformity
        ),
        maximum_absolute_recombination=(
            maximum_absolute_recombination
        ),
        integrated_absolute_recombination=(
            integrated_absolute_recombination
        ),
    )

    arrays: dict[str, np.ndarray] = {
        "potential": potential,
        "electrons": electrons,
        "holes": holes,
        "electron_current": electron_current,
        "hole_current": hole_current,
        "total_current": total_current,
        "mass_action_ratio": mass_action_ratio,
        "mass_action_relative_error": (
            mass_action_relative_error
        ),
    }

    if recombination is not None:
        arrays["recombination"] = recombination

    return metrics, arrays


# ---------------------------------------------------------------------------
# Solver-diagnostic helpers
# ---------------------------------------------------------------------------

def metadata_float(
    result: Any,
    key: str,
) -> float | None:
    """Return a finite scalar diagnostic from result metadata."""

    value = result.metadata.get(key)

    if value is None:
        return None

    scalar = float(value)

    if not np.isfinite(scalar):
        return None

    return scalar


def metadata_history(
    result: Any,
    key: str,
) -> np.ndarray | None:
    """Return a non-empty one-dimensional diagnostic history."""

    value = result.metadata.get(key)

    if value is None:
        return None

    history = np.asarray(value, dtype=float)

    if history.ndim != 1 or history.size == 0:
        return None

    return history


def print_optional_diagnostic(
    label: str,
    value: float | None,
    units: str = "",
) -> None:
    """Print an optional scalar solver diagnostic."""

    if value is None:
        print(f"{label:<34} unavailable")
        return

    unit_suffix = f" {units}" if units else ""
    print(f"{label:<34} {value:.6e}{unit_suffix}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def status_label(
    condition: bool,
) -> str:
    """Return a readable validation status."""

    return "PASS" if condition else "CHECK"


def print_validation_report(
    result: Any,
    metrics: ValidationMetrics,
) -> None:
    """Print equilibrium-validation results."""

    mass_action_passed = (
        metrics.maximum_mass_action_relative_error
        <= 1.0e-3
    )

    built_in_voltage_passed = (
        metrics.built_in_voltage_relative_error
        <= 5.0e-2
    )

    current_cancellation_passed = (
        metrics.current_cancellation_ratio
        <= 1.0e-3
    )

    print()
    print("DeviceForge Equilibrium PN-Junction Validation")
    print("=" * 50)
    print(f"Solver converged: {result.converged}")
    print(f"Iterations:       {result.iterations}")
    print()

    print("Physical parameters")
    print("-" * 50)
    print(
        "Temperature:                  "
        f"{metrics.temperature:.6f} K"
    )
    print(
        "Thermal voltage:              "
        f"{metrics.thermal_voltage:.6e} V"
    )
    print(
        "Intrinsic concentration:      "
        f"{metrics.intrinsic_concentration:.6e} m^-3"
    )
    print()

    print("Built-in potential")
    print("-" * 50)
    print(
        "Theoretical built-in voltage: "
        f"{metrics.theoretical_built_in_voltage:.6e} V"
    )
    print(
        "Simulated built-in voltage:   "
        f"{metrics.simulated_built_in_voltage:.6e} V"
    )
    print(
        "Relative error:               "
        f"{metrics.built_in_voltage_relative_error:.6e} "
        f"[{status_label(built_in_voltage_passed)}]"
    )
    print()

    print("Mass-action law: n*p = ni^2")
    print("-" * 50)
    print(
        "Maximum relative error:       "
        f"{metrics.maximum_mass_action_relative_error:.6e} "
        f"[{status_label(mass_action_passed)}]"
    )
    print(
        "Mean relative error:          "
        f"{metrics.mean_mass_action_relative_error:.6e}"
    )
    print()

    print("Equilibrium current")
    print("-" * 50)
    print(
        "Maximum |total current|:      "
        f"{metrics.maximum_total_current_density:.6e} A/m^2"
    )
    print(
        "Maximum component current:    "
        f"{metrics.maximum_component_current_density:.6e} A/m^2"
    )
    print(
        "Current cancellation ratio:   "
        f"{metrics.current_cancellation_ratio:.6e} "
        f"[{status_label(current_cancellation_passed)}]"
    )
    print(
        "Total-current nonuniformity:  "
        f"{metrics.total_current_nonuniformity:.6e} A/m^2"
    )
    print()

    print("Solver-equation diagnostics")
    print("-" * 50)
    print_optional_diagnostic(
        "Final update residual:",
        metadata_float(result, "final_update_residual"),
    )
    print_optional_diagnostic(
        "Final Poisson residual:",
        metadata_float(result, "final_poisson_residual"),
    )
    print_optional_diagnostic(
        "Electron continuity defect:",
        metadata_float(
            result,
            "final_electron_continuity_defect",
        ),
        "A/m^3",
    )
    print_optional_diagnostic(
        "Hole continuity defect:",
        metadata_float(
            result,
            "final_hole_continuity_defect",
        ),
        "A/m^3",
    )
    print_optional_diagnostic(
        "Electron continuity residual:",
        metadata_float(
            result,
            "final_electron_continuity_residual",
        ),
    )
    print_optional_diagnostic(
        "Hole continuity residual:",
        metadata_float(
            result,
            "final_hole_continuity_residual",
        ),
    )
    print_optional_diagnostic(
        "Electron quasi-Fermi variation:",
        metadata_float(
            result,
            "final_electron_quasi_fermi_nonuniformity",
        ),
    )
    print_optional_diagnostic(
        "Hole quasi-Fermi variation:",
        metadata_float(
            result,
            "final_hole_quasi_fermi_nonuniformity",
        ),
    )
    print(
        "Physics convergence enforced:    "
        f"{result.metadata.get('physics_convergence_enforced', False)}"
    )
    print()

    print("SRH recombination")
    print("-" * 50)

    if metrics.maximum_absolute_recombination is None:
        print("No recombination field was returned.")
    else:
        print(
            "Maximum |recombination|:     "
            f"{metrics.maximum_absolute_recombination:.6e} "
            "m^-3 s^-1"
        )

        print(
            "Integrated |recombination|:  "
            f"{metrics.integrated_absolute_recombination:.6e} "
            "m^-2 s^-1"
        )

        if mass_action_passed:
            print(
                "Mass action passes. A substantial SRH rate would "
                "therefore indicate an SRH implementation or "
                "field-storage inconsistency."
            )
        else:
            print(
                "Mass action does not pass. The recombination result "
                "may be responding to a non-equilibrium carrier "
                "solution."
            )

    print()


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    """Save and close a validation figure."""

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        FIGURE_DIRECTORY
        / filename
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved figure: {output_path}")


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_mass_action_ratio(
    simulation: Any,
    arrays: dict[str, np.ndarray],
) -> plt.Figure:
    """Plot n*p/ni^2 across the PN junction."""

    coordinates_nm = (
        node_coordinates_metres(simulation)
        * 1.0e9
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    axis.plot(
        coordinates_nm,
        arrays["mass_action_ratio"],
    )

    axis.axhline(
        1.0,
        linestyle="--",
        label="Thermal equilibrium",
    )

    axis.set_xlabel("Position (nm)")
    axis.set_ylabel("$np/n_i^2$")
    axis.set_title("Equilibrium Mass-Action Ratio")
    axis.grid(True)
    axis.legend()

    return figure


def plot_mass_action_error(
    simulation: Any,
    arrays: dict[str, np.ndarray],
) -> plt.Figure:
    """Plot absolute mass-action relative error."""

    coordinates_nm = (
        node_coordinates_metres(simulation)
        * 1.0e9
    )

    error = np.maximum(
        arrays["mass_action_relative_error"],
        np.finfo(float).tiny,
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    axis.semilogy(
        coordinates_nm,
        error,
    )

    axis.set_xlabel("Position (nm)")
    axis.set_ylabel(
        "Absolute relative error"
    )
    axis.set_title(
        "Mass-Action Law Error"
    )
    axis.grid(True)

    return figure


def plot_equilibrium_currents(
    simulation: Any,
    arrays: dict[str, np.ndarray],
) -> plt.Figure:
    """Plot electron, hole and total currents."""

    coordinates_nm = (
        edge_coordinates_metres(simulation)
        * 1.0e9
    )

    figure, axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    axis.plot(
        coordinates_nm,
        arrays["electron_current"],
        label="Electron current",
    )

    axis.plot(
        coordinates_nm,
        arrays["hole_current"],
        label="Hole current",
    )

    axis.plot(
        coordinates_nm,
        arrays["total_current"],
        linewidth=2.0,
        label="Total current",
    )

    axis.set_xlabel("Position (nm)")
    axis.set_ylabel("Current density (A/m$^2$)")
    axis.set_title(
        "Equilibrium Current Cancellation"
    )
    axis.grid(True)
    axis.legend()

    return figure


def plot_recombination_and_mass_action_error(
    simulation: Any,
    arrays: dict[str, np.ndarray],
) -> plt.Figure | None:
    """Compare recombination with mass-action error."""

    if "recombination" not in arrays:
        return None

    coordinates_nm = (
        node_coordinates_metres(simulation)
        * 1.0e9
    )

    figure, recombination_axis = plt.subplots(
        figsize=(8.0, 5.0)
    )

    recombination_axis.plot(
        coordinates_nm,
        arrays["recombination"],
        label="SRH recombination",
    )

    recombination_axis.set_xlabel(
        "Position (nm)"
    )

    recombination_axis.set_ylabel(
        "Recombination rate (m$^{-3}$ s$^{-1}$)"
    )

    recombination_axis.set_title(
        "SRH Recombination and Mass-Action Error"
    )

    recombination_axis.grid(True)

    error_axis = recombination_axis.twinx()

    error_axis.plot(
        coordinates_nm,
        arrays["mass_action_relative_error"],
        linestyle="--",
        label="Mass-action error",
    )

    error_axis.set_ylabel(
        "Mass-action relative error"
    )

    figure.legend(
        loc="upper right",
        bbox_to_anchor=(0.88, 0.88),
    )

    return figure


def plot_continuity_defect_history(
    result: Any,
) -> plt.Figure | None:
    """Plot raw electron and hole continuity defects."""

    electron_history = metadata_history(
        result,
        "electron_continuity_defect_history",
    )
    hole_history = metadata_history(
        result,
        "hole_continuity_defect_history",
    )

    if electron_history is None or hole_history is None:
        return None

    iterations = np.arange(1, electron_history.size + 1)
    tiny = np.finfo(float).tiny

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.semilogy(
        iterations,
        np.maximum(np.abs(electron_history), tiny),
        label="Electron continuity defect",
    )
    axis.semilogy(
        iterations,
        np.maximum(np.abs(hole_history), tiny),
        label="Hole continuity defect",
    )
    axis.set_xlabel("Gummel iteration")
    axis.set_ylabel("Maximum absolute defect (A/m$^3$)")
    axis.set_title("Raw Continuity-Equation Defects")
    axis.grid(True)
    axis.legend()

    return figure


def plot_continuity_residual_history(
    result: Any,
) -> plt.Figure | None:
    """Plot the existing dimensionless continuity residuals."""

    electron_history = metadata_history(
        result,
        "electron_continuity_residual_history",
    )
    hole_history = metadata_history(
        result,
        "hole_continuity_residual_history",
    )

    if electron_history is None or hole_history is None:
        return None

    iterations = np.arange(1, electron_history.size + 1)
    tiny = np.finfo(float).tiny

    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.semilogy(
        iterations,
        np.maximum(np.abs(electron_history), tiny),
        label="Electron continuity residual",
    )
    axis.semilogy(
        iterations,
        np.maximum(np.abs(hole_history), tiny),
        label="Hole continuity residual",
    )
    axis.set_xlabel("Gummel iteration")
    axis.set_ylabel("Dimensionless residual")
    axis.set_title("Scaled Continuity Diagnostics")
    axis.grid(True)
    axis.legend()

    return figure


def create_validation_figures(
    simulation: Any,
    arrays: dict[str, np.ndarray],
    result: Any,
) -> None:
    """Create all equilibrium-validation figures."""

    save_figure(
        plot_mass_action_ratio(
            simulation,
            arrays,
        ),
        "01_mass_action_ratio.png",
    )

    save_figure(
        plot_mass_action_error(
            simulation,
            arrays,
        ),
        "02_mass_action_error.png",
    )

    save_figure(
        plot_equilibrium_currents(
            simulation,
            arrays,
        ),
        "03_equilibrium_currents.png",
    )

    comparison_figure = (
        plot_recombination_and_mass_action_error(
            simulation,
            arrays,
        )
    )

    if comparison_figure is not None:
        save_figure(
            comparison_figure,
            "04_recombination_and_mass_action_error.png",
        )

    continuity_defect_figure = (
        plot_continuity_defect_history(result)
    )

    if continuity_defect_figure is not None:
        save_figure(
            continuity_defect_figure,
            "05_continuity_defect_history.png",
        )

    continuity_residual_figure = (
        plot_continuity_residual_history(result)
    )

    if continuity_residual_figure is not None:
        save_figure(
            continuity_residual_figure,
            "06_continuity_residual_history.png",
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the equilibrium physics validation."""

    simulation, solver, result = (
        run_equilibrium_simulation()
    )

    metrics, arrays = (
        calculate_validation_metrics(
            simulation=simulation,
            solver=solver,
            result=result,
        )
    )

    print_validation_report(
        result=result,
        metrics=metrics,
    )

    create_validation_figures(
        simulation=simulation,
        arrays=arrays,
        result=result,
    )

    print()
    print(
        "Equilibrium validation completed."
    )
    print(
        f"Figures written to: {FIGURE_DIRECTORY}"
    )
    print()


if __name__ == "__main__":
    main()