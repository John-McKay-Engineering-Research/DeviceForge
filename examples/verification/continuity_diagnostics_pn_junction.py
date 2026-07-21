"""
DeviceForge Verification Example
================================

Continuity diagnostics for a one-dimensional abrupt silicon PN junction.

This example solves a forward-biased PN junction and reports the raw
continuity-equation quantities used to assess whether

    div(Jn) = qR
    div(Jp) = -qR

are satisfied.

Unlike the mesh-convergence example, this script is intended to diagnose
the numerical behaviour of the Gummel iteration rather than verify mesh
independence.
"""

from deviceforge.solvers import (
    GummelDriftDiffusionSolver1D,
    SolverConfiguration,
)

from tests.test_gummel_1d import (
    build_pn_junction_simulation,
)


def main() -> None:

    simulation = build_pn_junction_simulation()

    solver = GummelDriftDiffusionSolver1D(
        applied_voltage=0.05,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=1.0e-10,
            max_iterations=simulation.max_iterations,
        ),
    )

    result = solver.solve(simulation)

    print()
    print("=" * 72)
    print("DeviceForge Continuity Diagnostics")
    print("=" * 72)

    print(f"Converged                 : {result.converged}")
    print(f"Iterations                : {result.iterations}")
    print()

    print("Solver convergence")
    print("------------------")

    print(
        f"Update residual           : "
        f"{result.metadata['final_update_residual']:.6e}"
    )

    print(
        f"Poisson residual          : "
        f"{result.metadata['final_poisson_residual']:.6e}"
    )

    print(
        f"Current nonuniformity     : "
        f"{result.metadata['relative_current_density_nonuniformity']:.6e}"
    )

    print()

    print("Continuity diagnostics")
    print("----------------------")

    print(
        f"Maximum |div(Jn)|         : "
        f"{result.metadata['maximum_electron_current_divergence']:.6e}"
        " A/m^3"
    )

    print(
        f"Maximum |div(Jp)|         : "
        f"{result.metadata['maximum_hole_current_divergence']:.6e}"
        " A/m^3"
    )

    print(
        f"Maximum |qR|             : "
        f"{result.metadata['maximum_recombination_current_source']:.6e}"
        " A/m^3"
    )

    print(
        f"Maximum electron defect  : "
        f"{result.metadata['maximum_electron_continuity_defect']:.6e}"
        " A/m^3"
    )

    print(
        f"Maximum hole defect      : "
        f"{result.metadata['maximum_hole_continuity_defect']:.6e}"
        " A/m^3"
    )

    print()

    print(
        f"Electron relative defect : "
        f"{result.metadata['electron_continuity_relative_defect']:.6e}"
    )

    print(
        f"Hole relative defect     : "
        f"{result.metadata['hole_continuity_relative_defect']:.6e}"
    )

    print()

    print("Terminal current densities")
    print("--------------------------")

    print(
        f"Left terminal current    : "
        f"{result.metadata['left_terminal_current_density']:.6e}"
        " A/m^2"
    )

    print(
        f"Right terminal current   : "
        f"{result.metadata['right_terminal_current_density']:.6e}"
        " A/m^2"
    )

    print(
        f"Average current          : "
        f"{result.metadata['average_terminal_current_density']:.6e}"
        " A/m^2"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()