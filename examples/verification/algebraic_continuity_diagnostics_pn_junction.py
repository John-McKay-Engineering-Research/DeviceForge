"""Inspect algebraic continuity residuals for a forward-biased PN junction."""

from deviceforge.solvers import (
    GummelDriftDiffusionSolver1D,
    SolverConfiguration,
)

from tests.test_gummel_1d import build_pn_junction_simulation


def print_value(label: str, value: float, units: str = "") -> None:
    suffix = f" {units}" if units else ""
    print(f"{label:<48}: {value:.6e}{suffix}")


def main() -> None:
    simulation = build_pn_junction_simulation()

    result = GummelDriftDiffusionSolver1D(
        applied_voltage=0.05,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=1.0e-10,
            max_iterations=simulation.max_iterations,
        ),
    ).solve(simulation)

    metadata = result.metadata

    print()
    print("=" * 86)
    print("DeviceForge Algebraic Continuity Diagnostics")
    print("=" * 86)
    print(f"Converged                                       : {result.converged}")
    print(f"Iterations                                      : {result.iterations}")
    print()

    print("Exact tridiagonal linear-solve residuals")
    print("-" * 86)
    print_value(
        "Maximum electron linear-solve residual",
        metadata["maximum_electron_linear_solve_algebraic_residual"],
        "1/m^3",
    )
    print_value(
        "Electron relative linear-solve residual",
        metadata["electron_linear_solve_relative_algebraic_residual"],
    )
    print_value(
        "Maximum hole linear-solve residual",
        metadata["maximum_hole_linear_solve_algebraic_residual"],
        "1/m^3",
    )
    print_value(
        "Hole relative linear-solve residual",
        metadata["hole_linear_solve_relative_algebraic_residual"],
    )
    print()

    print("Residuals after Gummel damping")
    print("-" * 86)
    print_value(
        "Maximum electron damped-state residual",
        metadata["maximum_electron_damped_state_algebraic_residual"],
        "1/m^3",
    )
    print_value(
        "Electron relative damped-state residual",
        metadata["electron_damped_state_relative_algebraic_residual"],
    )
    print_value(
        "Maximum hole damped-state residual",
        metadata["maximum_hole_damped_state_algebraic_residual"],
        "1/m^3",
    )
    print_value(
        "Hole relative damped-state residual",
        metadata["hole_damped_state_relative_algebraic_residual"],
    )
    print()

    print("Recombination lag")
    print("-" * 86)
    print_value(
        "Maximum source-to-final recombination change",
        metadata["maximum_recombination_lag"],
        "1/(m^3 s)",
    )
    print_value(
        "Relative source-to-final recombination change",
        metadata["relative_recombination_lag"],
    )
    print()

    print("Existing physical diagnostics")
    print("-" * 86)
    print_value(
        "Relative total-current nonuniformity",
        metadata["relative_current_density_nonuniformity"],
    )
    print_value(
        "Electron current-form continuity defect",
        metadata["electron_continuity_relative_defect"],
    )
    print_value(
        "Hole current-form continuity defect",
        metadata["hole_continuity_relative_defect"],
    )
    print("=" * 86)


if __name__ == "__main__":
    main()