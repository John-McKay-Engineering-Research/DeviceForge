"""Compare the verified and algebraic-diagnostics Gummel solvers.

Both implementations solve independently constructed copies of the same
forward-biased PN-junction problem. All common numerical outputs are then
compared to confirm that the added diagnostics do not alter the physical
solution.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from numbers import Number
from collections.abc import Mapping
from typing import Any

import numpy as np

from deviceforge.solvers import (
    GummelDriftDiffusionSolver1D,
    GummelDriftDiffusionSolver1DAlgebraic,
    SolverConfiguration,
)
from tests.test_gummel_1d import build_pn_junction_simulation


RELATIVE_TOLERANCE = 1.0e-12
ABSOLUTE_TOLERANCE = 1.0e-14


def public_attributes(obj: Any) -> dict[str, Any]:
    """Return stored public attributes without depending on one result layout."""

    if hasattr(obj, "__dict__"):
        return {
            name: value
            for name, value in vars(obj).items()
            if not name.startswith("_")
        }

    if is_dataclass(obj):
        return {
            field.name: getattr(obj, field.name)
            for field in fields(obj)
            if not field.name.startswith("_")
        }

    attributes: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except (AttributeError, RuntimeError):
            continue
        if not callable(value):
            attributes[name] = value

    return attributes


def compare_array(
    name: str,
    baseline: Any,
    algebraic: Any,
    *,
    rtol: float = RELATIVE_TOLERANCE,
    atol: float = ABSOLUTE_TOLERANCE,
) -> None:
    """Compare two array-like numerical outputs and print their differences."""

    baseline_array = np.asarray(baseline)
    algebraic_array = np.asarray(algebraic)

    if baseline_array.shape != algebraic_array.shape:
        raise AssertionError(
            f"{name}: shape mismatch "
            f"{baseline_array.shape} != {algebraic_array.shape}"
        )

    np.testing.assert_allclose(
        algebraic_array,
        baseline_array,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        err_msg=f"{name} differs between solver implementations",
    )

    if baseline_array.size == 0:
        maximum_absolute_difference = 0.0
        maximum_relative_difference = 0.0
    else:
        difference = np.abs(algebraic_array - baseline_array)
        maximum_absolute_difference = float(np.max(difference))
        scale = np.maximum(
            np.maximum(np.abs(baseline_array), np.abs(algebraic_array)),
            atol,
        )
        maximum_relative_difference = float(np.max(difference / scale))

    print(
        f"{name:<48}: PASS  "
        f"max abs = {maximum_absolute_difference:.6e}, "
        f"max rel = {maximum_relative_difference:.6e}"
    )


def compare_scalar(
    name: str,
    baseline: Number,
    algebraic: Number,
    *,
    rtol: float = RELATIVE_TOLERANCE,
    atol: float = ABSOLUTE_TOLERANCE,
) -> None:
    """Compare two scalar numerical outputs."""

    np.testing.assert_allclose(
        float(algebraic),
        float(baseline),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        err_msg=f"{name} differs between solver implementations",
    )

    absolute_difference = abs(float(algebraic) - float(baseline))
    scale = max(abs(float(algebraic)), abs(float(baseline)), atol)
    relative_difference = absolute_difference / scale

    print(
        f"{name:<48}: PASS  "
        f"abs diff = {absolute_difference:.6e}, "
        f"rel diff = {relative_difference:.6e}"
    )


def compare_field_like(
    name: str,
    baseline_field: Any,
    algebraic_field: Any,
) -> bool:
    """Compare DeviceForge Field-like objects through their stored values.

    Returns True when both inputs look like compatible Field objects and were
    compared, otherwise False.
    """

    if not (
        hasattr(baseline_field, "values")
        and hasattr(algebraic_field, "values")
    ):
        return False

    baseline_values = getattr(baseline_field, "values")
    algebraic_values = getattr(algebraic_field, "values")

    compare_array(
        f"{name}.values",
        baseline_values,
        algebraic_values,
    )

    # Compare stable descriptive attributes when they are present on both.
    for attribute_name in ("units", "name", "location"):
        if hasattr(baseline_field, attribute_name) and hasattr(
            algebraic_field, attribute_name
        ):
            baseline_attribute = getattr(baseline_field, attribute_name)
            algebraic_attribute = getattr(algebraic_field, attribute_name)

            if baseline_attribute != algebraic_attribute:
                raise AssertionError(
                    f"{name}.{attribute_name}: "
                    f"{baseline_attribute!r} != {algebraic_attribute!r}"
                )

            print(
                f"{name + '.' + attribute_name:<48}: "
                f"PASS  {baseline_attribute!r}"
            )

    return True


def compare_common_mapping(
    name: str,
    baseline_mapping: Mapping[str, Any],
    algebraic_mapping: Mapping[str, Any],
) -> None:
    """Compare common entries in mapping-like result containers."""

    baseline_keys = set(baseline_mapping)
    algebraic_keys = set(algebraic_mapping)

    if baseline_keys != algebraic_keys:
        missing_from_algebraic = sorted(baseline_keys - algebraic_keys)
        extra_in_algebraic = sorted(algebraic_keys - baseline_keys)
        raise AssertionError(
            f"{name}: mapping keys differ. "
            f"Missing from algebraic={missing_from_algebraic}, "
            f"extra in algebraic={extra_in_algebraic}"
        )

    for key in sorted(baseline_keys):
        baseline_value = baseline_mapping[key]
        algebraic_value = algebraic_mapping[key]
        qualified_name = f"{name}.{key}"

        if isinstance(baseline_value, np.ndarray) or isinstance(
            algebraic_value, np.ndarray
        ):
            compare_array(
                qualified_name,
                baseline_value,
                algebraic_value,
            )

        elif isinstance(baseline_value, (Number, np.number)) and isinstance(
            algebraic_value, (Number, np.number)
        ):
            if isinstance(baseline_value, (bool, np.bool_)):
                if bool(baseline_value) != bool(algebraic_value):
                    raise AssertionError(
                        f"{qualified_name}: "
                        f"{baseline_value!r} != {algebraic_value!r}"
                    )
                print(f"{qualified_name:<48}: PASS  {bool(baseline_value)}")
            else:
                compare_scalar(
                    qualified_name,
                    baseline_value,
                    algebraic_value,
                )

        elif isinstance(baseline_value, (str, type(None))) and isinstance(
            algebraic_value, (str, type(None))
        ):
            if baseline_value != algebraic_value:
                raise AssertionError(
                    f"{qualified_name}: "
                    f"{baseline_value!r} != {algebraic_value!r}"
                )
            print(f"{qualified_name:<48}: PASS  {baseline_value!r}")

        elif compare_field_like(
            qualified_name,
            baseline_value,
            algebraic_value,
        ):
            pass

        else:
            print(
                f"{qualified_name:<48}: SKIPPED "
                f"({type(baseline_value).__name__})"
            )


def compare_common_result_attributes(
    baseline_result: Any,
    algebraic_result: Any,
) -> None:
    """Compare every compatible public attribute shared by both results."""

    baseline_attributes = public_attributes(baseline_result)
    algebraic_attributes = public_attributes(algebraic_result)

    excluded_attributes = {"metadata", "runtime_seconds"}
    common_names = sorted(
        (set(baseline_attributes) & set(algebraic_attributes))
        - excluded_attributes
    )

    if not common_names:
        raise AssertionError("No common public result attributes were found.")

    print("Common solver-result attributes")
    print("-" * 90)

    compared_count = 0

    for name in common_names:
        baseline_value = baseline_attributes[name]
        algebraic_value = algebraic_attributes[name]

        if isinstance(baseline_value, Mapping) and isinstance(
            algebraic_value, Mapping
        ):
            compare_common_mapping(
                name,
                baseline_value,
                algebraic_value,
            )
            compared_count += 1

        elif isinstance(baseline_value, np.ndarray) or isinstance(
            algebraic_value, np.ndarray
        ):
            compare_array(name, baseline_value, algebraic_value)
            compared_count += 1

        elif isinstance(baseline_value, (Number, np.number)) and isinstance(
            algebraic_value, (Number, np.number)
        ):
            if isinstance(baseline_value, (bool, np.bool_)):
                if bool(baseline_value) != bool(algebraic_value):
                    raise AssertionError(
                        f"{name}: {baseline_value!r} != {algebraic_value!r}"
                    )
                print(f"{name:<48}: PASS  {bool(baseline_value)}")
            else:
                compare_scalar(name, baseline_value, algebraic_value)
            compared_count += 1

        elif isinstance(baseline_value, (str, type(None))) and isinstance(
            algebraic_value, (str, type(None))
        ):
            if baseline_value != algebraic_value:
                raise AssertionError(
                    f"{name}: {baseline_value!r} != {algebraic_value!r}"
                )
            print(f"{name:<48}: PASS  {baseline_value!r}")
            compared_count += 1

        else:
            print(
                f"{name:<48}: SKIPPED "
                f"({type(baseline_value).__name__})"
            )

    if compared_count == 0:
        raise AssertionError("No comparable numerical result attributes were found.")


def compare_common_metadata(
    baseline_metadata: dict[str, Any],
    algebraic_metadata: dict[str, Any],
) -> None:
    """Compare numerical metadata shared by both solver implementations."""

    common_names = sorted(set(baseline_metadata) & set(algebraic_metadata))

    print()
    print("Common numerical metadata")
    print("-" * 90)

    compared_count = 0

    for name in common_names:
        baseline_value = baseline_metadata[name]
        algebraic_value = algebraic_metadata[name]

        if isinstance(baseline_value, np.ndarray) or isinstance(
            algebraic_value, np.ndarray
        ):
            compare_array(f"metadata.{name}", baseline_value, algebraic_value)
            compared_count += 1

        elif isinstance(baseline_value, (Number, np.number)) and isinstance(
            algebraic_value, (Number, np.number)
        ):
            if isinstance(baseline_value, (bool, np.bool_)):
                if bool(baseline_value) != bool(algebraic_value):
                    raise AssertionError(
                        f"metadata.{name}: "
                        f"{baseline_value!r} != {algebraic_value!r}"
                    )
                print(f"{'metadata.' + name:<48}: PASS  {bool(baseline_value)}")
            else:
                compare_scalar(
                    f"metadata.{name}",
                    baseline_value,
                    algebraic_value,
                )
            compared_count += 1

    if compared_count == 0:
        print("No common numerical metadata fields were found.")


def print_algebraic_diagnostic_summary(metadata: dict[str, Any]) -> None:
    """Print the diagnostic values unique to the algebraic solver."""

    diagnostic_keys = (
        "maximum_electron_linear_solve_algebraic_residual",
        "electron_linear_solve_relative_algebraic_residual",
        "maximum_hole_linear_solve_algebraic_residual",
        "hole_linear_solve_relative_algebraic_residual",
        "maximum_electron_damped_state_algebraic_residual",
        "electron_damped_state_relative_algebraic_residual",
        "maximum_hole_damped_state_algebraic_residual",
        "hole_damped_state_relative_algebraic_residual",
        "maximum_recombination_lag",
        "relative_recombination_lag",
    )

    print()
    print("Algebraic-only diagnostic metadata")
    print("-" * 90)

    missing_keys = [key for key in diagnostic_keys if key not in metadata]
    if missing_keys:
        raise KeyError(
            "The algebraic solver result is missing expected metadata keys: "
            + ", ".join(missing_keys)
        )

    for key in diagnostic_keys:
        print(f"{key:<68}: {float(metadata[key]):.6e}")


def make_solver(solver_class: type, simulation: Any) -> Any:
    """Construct either implementation with identical numerical settings."""

    return solver_class(
        applied_voltage=0.05,
        damping_factor=0.2,
        configuration=SolverConfiguration(
            tolerance=1.0e-10,
            max_iterations=simulation.max_iterations,
        ),
    )


def main() -> None:
    # Build separate problem objects in case solve() mutates its input.
    baseline_simulation = build_pn_junction_simulation()
    algebraic_simulation = build_pn_junction_simulation()

    baseline_solver = make_solver(
        GummelDriftDiffusionSolver1D,
        baseline_simulation,
    )
    algebraic_solver = make_solver(
        GummelDriftDiffusionSolver1DAlgebraic,
        algebraic_simulation,
    )

    baseline_result = baseline_solver.solve(baseline_simulation)
    algebraic_result = algebraic_solver.solve(algebraic_simulation)

    print()
    print("=" * 90)
    print("DeviceForge Gummel Implementation Comparison")
    print("=" * 90)
    print(f"Baseline converged                              : {baseline_result.converged}")
    print(f"Algebraic converged                             : {algebraic_result.converged}")
    print(f"Baseline iterations                             : {baseline_result.iterations}")
    print(f"Algebraic iterations                            : {algebraic_result.iterations}")
    print()

    if baseline_result.converged != algebraic_result.converged:
        raise AssertionError(
            "The two solver implementations disagree on convergence status."
        )

    if baseline_result.iterations != algebraic_result.iterations:
        raise AssertionError(
            "The two solver implementations required different iteration counts: "
            f"{baseline_result.iterations} != {algebraic_result.iterations}"
        )

    compare_common_result_attributes(baseline_result, algebraic_result)
    compare_common_metadata(
        baseline_result.metadata,
        algebraic_result.metadata,
    )
    print_algebraic_diagnostic_summary(algebraic_result.metadata)

    print()
    print("=" * 90)
    print("PASS: Both solvers produced matching common physical outputs.")
    print("The algebraic diagnostics did not alter the numerical solution.")
    print("=" * 90)


if __name__ == "__main__":
    main()