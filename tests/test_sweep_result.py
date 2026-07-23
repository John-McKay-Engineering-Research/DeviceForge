"""Tests for the DeviceForge voltage-sweep result model."""

from __future__ import annotations

import numpy as np
import pytest

from deviceforge.analysis.sweep_result import VoltageSweepResult


@pytest.fixture
def sample_arrays() -> dict[str, np.ndarray]:
    """Return internally consistent one-dimensional result arrays."""

    potential = np.array([0.0, 0.1, 0.2], dtype=float)
    electron_density = np.array([1.0e16, 1.1e16, 1.2e16], dtype=float)
    hole_density = np.array([1.2e16, 1.1e16, 1.0e16], dtype=float)
    electric_field = np.array([-1.0, -1.0, -1.0], dtype=float)

    electron_current_density = np.array([2.0, 2.0, 2.0], dtype=float)
    hole_current_density = np.array([0.5, 0.5, 0.5], dtype=float)
    total_current_density = np.array([2.5, 2.5, 2.5], dtype=float)

    return {
        "potential": potential,
        "electron_density": electron_density,
        "hole_density": hole_density,
        "electric_field": electric_field,
        "electron_current_density": electron_current_density,
        "hole_current_density": hole_current_density,
        "total_current_density": total_current_density,
    }


@pytest.fixture
def sample_result(
    sample_arrays: dict[str, np.ndarray],
) -> VoltageSweepResult:
    """Return a valid voltage-sweep result."""

    return VoltageSweepResult(
        voltage=0.25,
        current=2.5e-6,
        iterations=14,
        residual=1.0e-9,
        solve_time=0.075,
        converged=True,
        metadata={"contact": "anode"},
        **sample_arrays,
    )


def test_result_stores_scalar_values(
    sample_result: VoltageSweepResult,
) -> None:
    """Scalar solver data should be stored without modification."""

    assert sample_result.voltage == pytest.approx(0.25)
    assert sample_result.current == pytest.approx(2.5e-6)
    assert sample_result.iterations == 14
    assert sample_result.residual == pytest.approx(1.0e-9)
    assert sample_result.solve_time == pytest.approx(0.075)
    assert sample_result.converged is True


def test_result_stores_float64_arrays(
    sample_result: VoltageSweepResult,
) -> None:
    """All numerical fields should be stored as float64 arrays."""

    array_fields = (
        sample_result.potential,
        sample_result.electron_density,
        sample_result.hole_density,
        sample_result.electric_field,
        sample_result.electron_current_density,
        sample_result.hole_current_density,
        sample_result.total_current_density,
    )

    for array in array_fields:
        assert isinstance(array, np.ndarray)
        assert array.dtype == np.float64


def test_result_copies_input_arrays(
    sample_arrays: dict[str, np.ndarray],
) -> None:
    """Changing an input array should not modify the stored result."""

    original_potential = sample_arrays["potential"]

    result = VoltageSweepResult(
        voltage=0.0,
        current=0.0,
        iterations=1,
        residual=0.0,
        solve_time=0.0,
        converged=True,
        **sample_arrays,
    )

    original_potential[0] = 999.0

    assert result.potential[0] == pytest.approx(0.0)


def test_shape_property(
    sample_result: VoltageSweepResult,
) -> None:
    """The shape property should reflect the scalar solution fields."""

    assert sample_result.shape == (3,)


def test_number_of_points_property(
    sample_result: VoltageSweepResult,
) -> None:
    """The number-of-points property should return total field size."""

    assert sample_result.number_of_points == 3


def test_absolute_current_property(
    sample_arrays: dict[str, np.ndarray],
) -> None:
    """Absolute current should remove the terminal-current sign."""

    result = VoltageSweepResult(
        voltage=-0.5,
        current=-3.2e-6,
        iterations=10,
        residual=1.0e-8,
        solve_time=0.02,
        converged=True,
        **sample_arrays,
    )

    assert result.absolute_current == pytest.approx(3.2e-6)


def test_current_density_mismatch_is_zero_for_consistent_fields(
    sample_result: VoltageSweepResult,
) -> None:
    """Component currents should reproduce the stored total current."""

    mismatch = sample_result.current_density_mismatch

    np.testing.assert_allclose(
        mismatch,
        np.zeros_like(mismatch),
    )


def test_summary_returns_expected_values(
    sample_result: VoltageSweepResult,
) -> None:
    """The summary should contain compact operating-point information."""

    summary = sample_result.summary()

    assert summary == {
        "voltage": pytest.approx(0.25),
        "current": pytest.approx(2.5e-6),
        "absolute_current": pytest.approx(2.5e-6),
        "iterations": 14,
        "residual": pytest.approx(1.0e-9),
        "solve_time": pytest.approx(0.075),
        "converged": True,
        "shape": (3,),
        "number_of_points": 3,
    }


def test_copy_returns_independent_result(
    sample_result: VoltageSweepResult,
) -> None:
    """The copy method should duplicate arrays and metadata independently."""

    copied_result = sample_result.copy()

    copied_result.potential[0] = 123.0
    copied_result.metadata["contact"] = "cathode"

    assert sample_result.potential[0] == pytest.approx(0.0)
    assert sample_result.metadata["contact"] == "anode"


@pytest.mark.parametrize(
    ("field_name", "bad_values"),
    [
        ("potential", []),
        ("electron_density", []),
        ("hole_density", []),
        ("electric_field", []),
        ("electron_current_density", []),
        ("hole_current_density", []),
        ("total_current_density", []),
    ],
)
def test_empty_arrays_are_rejected(
    sample_arrays: dict[str, np.ndarray],
    field_name: str,
    bad_values: list[float],
) -> None:
    """Stored numerical fields must not be empty."""

    arrays = sample_arrays.copy()
    arrays[field_name] = bad_values

    with pytest.raises(ValueError, match="must not be empty"):
        VoltageSweepResult(
            voltage=0.0,
            current=0.0,
            iterations=1,
            residual=0.0,
            solve_time=0.0,
            converged=True,
            **arrays,
        )


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_non_finite_array_values_are_rejected(
    sample_arrays: dict[str, np.ndarray],
    bad_value: float,
) -> None:
    """Stored arrays must contain finite values."""

    arrays = sample_arrays.copy()
    arrays["potential"] = np.array([0.0, bad_value, 1.0])

    with pytest.raises(
        ValueError,
        match="potential must contain only finite values",
    ):
        VoltageSweepResult(
            voltage=0.0,
            current=0.0,
            iterations=1,
            residual=0.0,
            solve_time=0.0,
            converged=True,
            **arrays,
        )


def test_mismatched_scalar_field_shape_is_rejected(
    sample_arrays: dict[str, np.ndarray],
) -> None:
    """Scalar result fields should share the same grid shape."""

    arrays = sample_arrays.copy()
    arrays["hole_density"] = np.array([1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="hole_density must have shape",
    ):
        VoltageSweepResult(
            voltage=0.0,
            current=0.0,
            iterations=1,
            residual=0.0,
            solve_time=0.0,
            converged=True,
            **arrays,
        )


def test_vector_electric_field_is_accepted_for_two_dimensional_data() -> None:
    """A 2D electric field may store one vector component per dimension."""

    scalar_shape = (2, 3)

    result = VoltageSweepResult(
        voltage=0.0,
        current=0.0,
        potential=np.zeros(scalar_shape),
        electron_density=np.ones(scalar_shape),
        hole_density=np.ones(scalar_shape),
        electric_field=np.zeros(scalar_shape + (2,)),
        electron_current_density=np.zeros(scalar_shape),
        hole_current_density=np.zeros(scalar_shape),
        total_current_density=np.zeros(scalar_shape),
        iterations=1,
        residual=0.0,
        solve_time=0.0,
        converged=True,
    )

    assert result.electric_field.shape == (2, 3, 2)


def test_invalid_electric_field_shape_is_rejected() -> None:
    """The electric field must match the simulation dimensionality."""

    scalar_shape = (2, 3)

    with pytest.raises(
        ValueError,
        match="electric_field must either match",
    ):
        VoltageSweepResult(
            voltage=0.0,
            current=0.0,
            potential=np.zeros(scalar_shape),
            electron_density=np.ones(scalar_shape),
            hole_density=np.ones(scalar_shape),
            electric_field=np.zeros((2, 3, 3)),
            electron_current_density=np.zeros(scalar_shape),
            hole_current_density=np.zeros(scalar_shape),
            total_current_density=np.zeros(scalar_shape),
            iterations=1,
            residual=0.0,
            solve_time=0.0,
            converged=True,
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value", "expected_exception"),
    [
        ("voltage", np.nan, ValueError),
        ("voltage", np.inf, ValueError),
        ("current", np.nan, ValueError),
        ("residual", -1.0, ValueError),
        ("solve_time", -0.1, ValueError),
        ("iterations", -1, ValueError),
        ("iterations", 1.5, TypeError),
        ("converged", 1, TypeError),
    ],
)
def test_invalid_scalar_values_are_rejected(
    sample_arrays: dict[str, np.ndarray],
    field_name: str,
    bad_value: object,
    expected_exception: type[Exception],
) -> None:
    """Invalid scalar diagnostics should be rejected."""

    arguments: dict[str, object] = {
        "voltage": 0.0,
        "current": 0.0,
        "iterations": 1,
        "residual": 0.0,
        "solve_time": 0.0,
        "converged": True,
        **sample_arrays,
    }

    arguments[field_name] = bad_value

    with pytest.raises(expected_exception):
        VoltageSweepResult(**arguments)


def test_metadata_defaults_to_independent_dictionary(
    sample_arrays: dict[str, np.ndarray],
) -> None:
    """Each result should receive its own metadata dictionary."""

    first = VoltageSweepResult(
        voltage=0.0,
        current=0.0,
        iterations=1,
        residual=0.0,
        solve_time=0.0,
        converged=True,
        **sample_arrays,
    )

    second = VoltageSweepResult(
        voltage=0.1,
        current=1.0e-6,
        iterations=2,
        residual=1.0e-8,
        solve_time=0.01,
        converged=True,
        **sample_arrays,
    )

    first.metadata["contact"] = "anode"

    assert second.metadata == {}


def test_non_dictionary_metadata_is_rejected(
    sample_arrays: dict[str, np.ndarray],
) -> None:
    """Metadata should use a dictionary for predictable extension."""

    with pytest.raises(TypeError, match="metadata must be a dictionary"):
        VoltageSweepResult(
            voltage=0.0,
            current=0.0,
            iterations=1,
            residual=0.0,
            solve_time=0.0,
            converged=True,
            metadata=["invalid"],  # type: ignore[arg-type]
            **sample_arrays,
        )