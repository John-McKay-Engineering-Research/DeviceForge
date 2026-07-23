"""Tests for the DeviceForge voltage-sweep results container."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from deviceforge.analysis.sweep_result import VoltageSweepResult
from deviceforge.analysis.sweep_results import SweepResults


def create_result(
    voltage: float,
    current: float,
    *,
    converged: bool = True,
    iterations: int = 10,
    residual: float = 1.0e-9,
    solve_time: float = 0.1,
) -> VoltageSweepResult:
    """Create a compact internally consistent sweep result."""

    potential = np.array(
        [voltage, voltage + 0.1, voltage + 0.2],
        dtype=float,
    )

    electron_density = np.array(
        [1.0e16, 1.1e16, 1.2e16],
        dtype=float,
    )

    hole_density = np.array(
        [1.2e16, 1.1e16, 1.0e16],
        dtype=float,
    )

    electric_field = np.array(
        [-1.0, -1.0, -1.0],
        dtype=float,
    )

    electron_current_density = np.array(
        [0.8 * current, 0.8 * current, 0.8 * current],
        dtype=float,
    )

    hole_current_density = np.array(
        [0.2 * current, 0.2 * current, 0.2 * current],
        dtype=float,
    )

    total_current_density = (
        electron_current_density + hole_current_density
    )

    return VoltageSweepResult(
        voltage=voltage,
        current=current,
        potential=potential,
        electron_density=electron_density,
        hole_density=hole_density,
        electric_field=electric_field,
        electron_current_density=electron_current_density,
        hole_current_density=hole_current_density,
        total_current_density=total_current_density,
        iterations=iterations,
        residual=residual,
        solve_time=solve_time,
        converged=converged,
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_result_list() -> list[VoltageSweepResult]:
    """Return a mixed-convergence voltage-sweep dataset."""

    return [
        create_result(
            -0.5,
            -2.0e-9,
            converged=True,
            iterations=8,
            residual=1.0e-10,
            solve_time=0.08,
        ),
        create_result(
            0.0,
            0.0,
            converged=True,
            iterations=5,
            residual=5.0e-11,
            solve_time=0.05,
        ),
        create_result(
            0.5,
            2.5e-6,
            converged=False,
            iterations=30,
            residual=1.0e-3,
            solve_time=0.30,
        ),
    ]


@pytest.fixture
def sample_results(
    sample_result_list: list[VoltageSweepResult],
) -> SweepResults:
    """Return a populated sweep-results container."""

    return SweepResults(
        sample_result_list,
        metadata={
            "contact": "anode",
            "device": "pn_junction",
        },
    )


def test_empty_container_initialises_successfully() -> None:
    """An empty sweep-results container should be valid."""

    results = SweepResults()

    assert len(results) == 0
    assert results.metadata == {}


def test_container_initialises_from_iterable(
    sample_result_list: list[VoltageSweepResult],
) -> None:
    """The constructor should accept an iterable of valid results."""

    results = SweepResults(sample_result_list)

    assert len(results) == 3


def test_constructor_accepts_generator(
    sample_result_list: list[VoltageSweepResult],
) -> None:
    """The constructor should consume result generators correctly."""

    results = SweepResults(
        result
        for result in sample_result_list
    )

    assert len(results) == 3


def test_constructor_copies_results(
    sample_result_list: list[VoltageSweepResult],
) -> None:
    """Changing an input result should not change stored sweep data."""

    results = SweepResults(sample_result_list)

    sample_result_list[0].potential[0] = 999.0

    assert results[0].potential[0] == pytest.approx(-0.5)


def test_constructor_copies_metadata() -> None:
    """External metadata changes should not affect the container."""

    metadata = {"contact": "anode"}

    results = SweepResults(metadata=metadata)

    metadata["contact"] = "cathode"

    assert results.metadata["contact"] == "anode"


def test_invalid_constructor_metadata_is_rejected() -> None:
    """Container metadata should be a dictionary or None."""

    with pytest.raises(
        TypeError,
        match="metadata must be a dictionary or None",
    ):
        SweepResults(metadata=["invalid"])  # type: ignore[arg-type]


def test_invalid_constructor_result_is_rejected() -> None:
    """Every initial result must be a VoltageSweepResult."""

    with pytest.raises(
        TypeError,
        match="Each sweep result must be",
    ):
        SweepResults([create_result(0.0, 0.0), "invalid"])  # type: ignore[list-item]


def test_voltages_property(
    sample_results: SweepResults,
) -> None:
    """Applied voltages should be returned in sweep order."""

    np.testing.assert_allclose(
        sample_results.voltages,
        np.array([-0.5, 0.0, 0.5]),
    )

    assert sample_results.voltages.dtype == np.float64


def test_currents_property(
    sample_results: SweepResults,
) -> None:
    """Terminal currents should be returned in sweep order."""

    np.testing.assert_allclose(
        sample_results.currents,
        np.array([-2.0e-9, 0.0, 2.5e-6]),
    )


def test_absolute_currents_property(
    sample_results: SweepResults,
) -> None:
    """Absolute currents should contain non-negative magnitudes."""

    np.testing.assert_allclose(
        sample_results.absolute_currents,
        np.array([2.0e-9, 0.0, 2.5e-6]),
    )


def test_iterations_property(
    sample_results: SweepResults,
) -> None:
    """Iteration counts should be returned as integer data."""

    np.testing.assert_array_equal(
        sample_results.iterations,
        np.array([8, 5, 30]),
    )

    assert sample_results.iterations.dtype == np.int64


def test_residuals_property(
    sample_results: SweepResults,
) -> None:
    """Final residuals should be returned in sweep order."""

    np.testing.assert_allclose(
        sample_results.residuals,
        np.array([1.0e-10, 5.0e-11, 1.0e-3]),
    )


def test_solve_times_property(
    sample_results: SweepResults,
) -> None:
    """Solve times should be returned in sweep order."""

    np.testing.assert_allclose(
        sample_results.solve_times,
        np.array([0.08, 0.05, 0.30]),
    )


def test_convergence_flags_property(
    sample_results: SweepResults,
) -> None:
    """Convergence status should be returned as Boolean data."""

    np.testing.assert_array_equal(
        sample_results.convergence_flags,
        np.array([True, True, False]),
    )

    assert sample_results.convergence_flags.dtype == np.bool_


def test_converged_results_property(
    sample_results: SweepResults,
) -> None:
    """Only converged results should be selected."""

    converged = sample_results.converged_results

    assert isinstance(converged, tuple)
    assert len(converged) == 2
    assert all(result.converged for result in converged)


def test_failed_results_property(
    sample_results: SweepResults,
) -> None:
    """Only failed results should be selected."""

    failed = sample_results.failed_results

    assert isinstance(failed, tuple)
    assert len(failed) == 1
    assert failed[0].converged is False
    assert failed[0].voltage == pytest.approx(0.5)


def test_filtered_results_are_independent_copies(
    sample_results: SweepResults,
) -> None:
    """Filtered result objects should not expose internal arrays."""

    converged = sample_results.converged_results
    converged[0].potential[0] = 1000.0

    assert sample_results[0].potential[0] == pytest.approx(-0.5)


def test_convergence_statistics(
    sample_results: SweepResults,
) -> None:
    """Convergence counts and rate should be correct."""

    assert sample_results.number_converged == 2
    assert sample_results.number_failed == 1
    assert sample_results.all_converged is False
    assert sample_results.convergence_rate == pytest.approx(2.0 / 3.0)


def test_empty_convergence_statistics() -> None:
    """Empty containers should have defined convergence statistics."""

    results = SweepResults()

    assert results.number_converged == 0
    assert results.number_failed == 0
    assert results.all_converged is True
    assert results.convergence_rate == pytest.approx(0.0)


def test_solve_time_statistics(
    sample_results: SweepResults,
) -> None:
    """Accumulated and average solve times should be correct."""

    assert sample_results.total_solve_time == pytest.approx(0.43)
    assert sample_results.average_solve_time == pytest.approx(
        0.43 / 3.0
    )


def test_iteration_statistics(
    sample_results: SweepResults,
) -> None:
    """Accumulated and average iteration counts should be correct."""

    assert sample_results.total_iterations == 43
    assert sample_results.average_iterations == pytest.approx(
        43.0 / 3.0
    )


def test_empty_average_statistics_are_zero() -> None:
    """Empty containers should return zero-valued averages."""

    results = SweepResults()

    assert results.total_solve_time == pytest.approx(0.0)
    assert results.average_solve_time == pytest.approx(0.0)
    assert results.total_iterations == 0
    assert results.average_iterations == pytest.approx(0.0)


def test_voltage_range_properties(
    sample_results: SweepResults,
) -> None:
    """Minimum and maximum voltages should be identified."""

    assert sample_results.minimum_voltage == pytest.approx(-0.5)
    assert sample_results.maximum_voltage == pytest.approx(0.5)


def test_empty_voltage_range_is_none() -> None:
    """An empty sweep should not report a voltage range."""

    results = SweepResults()

    assert results.minimum_voltage is None
    assert results.maximum_voltage is None


def test_append_adds_result_copy() -> None:
    """Appending should add an independent operating-point result."""

    source = create_result(0.25, 1.0e-6)
    results = SweepResults()

    results.append(source)
    source.potential[0] = 500.0

    assert len(results) == 1
    assert results[0].potential[0] == pytest.approx(0.25)


def test_append_rejects_invalid_result() -> None:
    """Appending a non-result object should fail."""

    results = SweepResults()

    with pytest.raises(
        TypeError,
        match="Each sweep result must be",
    ):
        results.append("invalid")  # type: ignore[arg-type]


def test_extend_adds_multiple_results(
    sample_result_list: list[VoltageSweepResult],
) -> None:
    """Extending should preserve iterable order."""

    results = SweepResults()

    results.extend(sample_result_list)

    np.testing.assert_allclose(
        results.voltages,
        np.array([-0.5, 0.0, 0.5]),
    )


def test_extend_is_atomic_when_input_contains_invalid_item() -> None:
    """A failed extension should leave the container unchanged."""

    initial_result = create_result(-1.0, -1.0e-10)
    results = SweepResults([initial_result])

    with pytest.raises(TypeError):
        results.extend(
            [
                create_result(0.0, 0.0),
                "invalid",  # type: ignore[list-item]
            ]
        )

    assert len(results) == 1
    assert results[0].voltage == pytest.approx(-1.0)


def test_extend_rejects_non_iterable() -> None:
    """The extend input must be iterable."""

    results = SweepResults()

    with pytest.raises(
        TypeError,
        match="results must be an iterable",
    ):
        results.extend(42)  # type: ignore[arg-type]


def test_insert_places_result_at_requested_index(
    sample_results: SweepResults,
) -> None:
    """Insertion should follow standard list index behaviour."""

    sample_results.insert(
        1,
        create_result(-0.25, -1.0e-9),
    )

    np.testing.assert_allclose(
        sample_results.voltages,
        np.array([-0.5, -0.25, 0.0, 0.5]),
    )


def test_insert_rejects_non_integer_index(
    sample_results: SweepResults,
) -> None:
    """Insertion indices should be integers."""

    with pytest.raises(
        TypeError,
        match="index must be an integer",
    ):
        sample_results.insert(
            1.5,  # type: ignore[arg-type]
            create_result(0.25, 1.0e-6),
        )


def test_clear_removes_all_results(
    sample_results: SweepResults,
) -> None:
    """Clearing should empty the sweep while preserving metadata."""

    sample_results.clear()

    assert len(sample_results) == 0
    assert sample_results.metadata["contact"] == "anode"


def test_indexing_returns_independent_copy(
    sample_results: SweepResults,
) -> None:
    """Indexed result access should not expose internal arrays."""

    selected = sample_results[0]
    selected.potential[0] = 999.0

    assert sample_results[0].potential[0] == pytest.approx(-0.5)


def test_negative_indexing(
    sample_results: SweepResults,
) -> None:
    """Standard negative sequence indices should be supported."""

    assert sample_results[-1].voltage == pytest.approx(0.5)


def test_invalid_index_type_is_rejected(
    sample_results: SweepResults,
) -> None:
    """Only integers and slices should be accepted."""

    with pytest.raises(
        TypeError,
        match="indices must be integers or slices",
    ):
        _ = sample_results[1.5]  # type: ignore[index]


def test_slice_returns_sweep_results(
    sample_results: SweepResults,
) -> None:
    """Slicing should return a new SweepResults container."""

    sliced = sample_results[:2]

    assert isinstance(sliced, SweepResults)
    assert len(sliced) == 2

    np.testing.assert_allclose(
        sliced.voltages,
        np.array([-0.5, 0.0]),
    )


def test_slice_copies_metadata(
    sample_results: SweepResults,
) -> None:
    """A sliced container should have independent metadata."""

    sliced = sample_results[:2]
    sliced.metadata["contact"] = "cathode"

    assert sample_results.metadata["contact"] == "anode"


def test_iteration_returns_results_in_order(
    sample_results: SweepResults,
) -> None:
    """Iteration should preserve sweep order."""

    voltages = [
        result.voltage
        for result in sample_results
    ]

    assert voltages == pytest.approx([-0.5, 0.0, 0.5])


def test_iteration_returns_independent_copies(
    sample_results: SweepResults,
) -> None:
    """Iteration should not expose internal result arrays."""

    first = next(iter(sample_results))
    first.potential[0] = 1234.0

    assert sample_results[0].potential[0] == pytest.approx(-0.5)


def test_reversed_iteration(
    sample_results: SweepResults,
) -> None:
    """Reverse iteration should invert sweep order."""

    voltages = [
        result.voltage
        for result in reversed(sample_results)
    ]

    assert voltages == pytest.approx([0.5, 0.0, -0.5])


def test_membership_detects_equivalent_result(
    sample_result_list: list[VoltageSweepResult],
    sample_results: SweepResults,
) -> None:
    """Membership should compare numerical result content safely."""

    assert sample_result_list[0] in sample_results


def test_membership_rejects_different_or_invalid_values(
    sample_results: SweepResults,
) -> None:
    """Different and unrelated objects should not be members."""

    assert create_result(10.0, 5.0) not in sample_results
    assert "invalid" not in sample_results


def test_copy_returns_independent_container(
    sample_results: SweepResults,
) -> None:
    """Copying should duplicate results and metadata deeply enough."""

    copied = sample_results.copy()

    copied_result = copied[0]
    copied_result.potential[0] = 999.0
    copied.metadata["contact"] = "cathode"
    copied.clear()

    assert len(sample_results) == 3
    assert sample_results[0].potential[0] == pytest.approx(-0.5)
    assert sample_results.metadata["contact"] == "anode"


def test_sorted_by_voltage_ascending() -> None:
    """Voltage sorting should return a new ascending container."""

    results = SweepResults(
        [
            create_result(0.5, 1.0),
            create_result(-0.5, -1.0),
            create_result(0.0, 0.0),
        ]
    )

    sorted_results = results.sorted_by_voltage()

    np.testing.assert_allclose(
        sorted_results.voltages,
        np.array([-0.5, 0.0, 0.5]),
    )

    np.testing.assert_allclose(
        results.voltages,
        np.array([0.5, -0.5, 0.0]),
    )


def test_sorted_by_voltage_descending() -> None:
    """Descending voltage sorting should be supported."""

    results = SweepResults(
        [
            create_result(0.0, 0.0),
            create_result(-0.5, -1.0),
            create_result(0.5, 1.0),
        ]
    )

    sorted_results = results.sorted_by_voltage(reverse=True)

    np.testing.assert_allclose(
        sorted_results.voltages,
        np.array([0.5, 0.0, -0.5]),
    )


def test_sorted_by_voltage_rejects_invalid_reverse_value(
    sample_results: SweepResults,
) -> None:
    """The reverse argument should be strictly Boolean."""

    with pytest.raises(
        TypeError,
        match="reverse must be a boolean",
    ):
        sample_results.sorted_by_voltage(reverse=1)  # type: ignore[arg-type]


def test_result_at_voltage_finds_exact_result(
    sample_results: SweepResults,
) -> None:
    """A stored voltage should be retrievable directly."""

    result = sample_results.result_at_voltage(0.0)

    assert result.voltage == pytest.approx(0.0)
    assert result.current == pytest.approx(0.0)


def test_result_at_voltage_uses_tolerance(
    sample_results: SweepResults,
) -> None:
    """Small floating-point voltage differences should be tolerated."""

    result = sample_results.result_at_voltage(
        0.50000000001,
        atol=1.0e-9,
    )

    assert result.voltage == pytest.approx(0.5)


def test_result_at_voltage_returns_independent_copy(
    sample_results: SweepResults,
) -> None:
    """Voltage lookup should not expose internal arrays."""

    result = sample_results.result_at_voltage(-0.5)
    result.potential[0] = 999.0

    assert sample_results[0].potential[0] == pytest.approx(-0.5)


def test_result_at_voltage_raises_when_missing(
    sample_results: SweepResults,
) -> None:
    """Missing operating points should raise a lookup error."""

    with pytest.raises(
        LookupError,
        match="No sweep result was found",
    ):
        sample_results.result_at_voltage(2.0)


@pytest.mark.parametrize(
    ("argument_name", "bad_value", "expected_exception"),
    [
        ("voltage", np.nan, ValueError),
        ("voltage", np.inf, ValueError),
        ("voltage", True, TypeError),
        ("atol", -1.0, ValueError),
        ("rtol", -1.0, ValueError),
    ],
)
def test_result_at_voltage_rejects_invalid_arguments(
    sample_results: SweepResults,
    argument_name: str,
    bad_value: object,
    expected_exception: type[Exception],
) -> None:
    """Voltage lookup inputs should be finite and valid."""

    arguments: dict[str, object] = {
        "voltage": 0.0,
        "atol": 1.0e-12,
        "rtol": 1.0e-9,
    }

    arguments[argument_name] = bad_value

    with pytest.raises(expected_exception):
        sample_results.result_at_voltage(**arguments)  # type: ignore[arg-type]


def test_summary_returns_expected_statistics(
    sample_results: SweepResults,
) -> None:
    """The summary should contain compact sweep-level statistics."""

    summary = sample_results.summary()

    assert summary == {
        "number_of_results": 3,
        "number_converged": 2,
        "number_failed": 1,
        "all_converged": False,
        "convergence_rate": pytest.approx(2.0 / 3.0),
        "minimum_voltage": pytest.approx(-0.5),
        "maximum_voltage": pytest.approx(0.5),
        "total_iterations": 43,
        "average_iterations": pytest.approx(43.0 / 3.0),
        "total_solve_time": pytest.approx(0.43),
        "average_solve_time": pytest.approx(0.43 / 3.0),
    }


def test_to_records_returns_compact_scalar_data(
    sample_results: SweepResults,
) -> None:
    """Record conversion should exclude full numerical fields."""

    records = sample_results.to_records()

    assert len(records) == 3

    assert records[0] == {
        "voltage": pytest.approx(-0.5),
        "current": pytest.approx(-2.0e-9),
        "absolute_current": pytest.approx(2.0e-9),
        "iterations": 8,
        "residual": pytest.approx(1.0e-10),
        "solve_time": pytest.approx(0.08),
        "converged": True,
    }

    assert "potential" not in records[0]
    assert "electron_density" not in records[0]


def test_export_csv_creates_expected_file(
    sample_results: SweepResults,
    tmp_path: Path,
) -> None:
    """CSV export should write a header and one row per result."""

    output_path = tmp_path / "iv_curve.csv"

    written_path = sample_results.export_csv(output_path)

    assert written_path == output_path.resolve()
    assert output_path.exists()

    with output_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 3

    assert list(rows[0]) == [
        "voltage",
        "current",
        "absolute_current",
        "iterations",
        "residual",
        "solve_time",
        "converged",
    ]

    assert float(rows[0]["voltage"]) == pytest.approx(-0.5)
    assert float(rows[0]["current"]) == pytest.approx(-2.0e-9)
    assert int(rows[0]["iterations"]) == 8
    assert rows[0]["converged"] == "True"

    assert rows[2]["converged"] == "False"


def test_export_csv_adds_csv_suffix(
    sample_results: SweepResults,
    tmp_path: Path,
) -> None:
    """A missing filename extension should be added automatically."""

    written_path = sample_results.export_csv(
        tmp_path / "iv_curve"
    )

    assert written_path.suffix == ".csv"
    assert written_path.exists()


def test_export_csv_creates_parent_directories(
    sample_results: SweepResults,
    tmp_path: Path,
) -> None:
    """Missing parent directories should be created."""

    destination = (
        tmp_path
        / "nested"
        / "results"
        / "iv_curve.csv"
    )

    written_path = sample_results.export_csv(destination)

    assert written_path.exists()
    assert written_path.parent == destination.parent.resolve()


def test_export_csv_rejects_existing_file_without_overwrite(
    sample_results: SweepResults,
    tmp_path: Path,
) -> None:
    """Existing files should be protected by default."""

    destination = tmp_path / "iv_curve.csv"
    destination.write_text("existing data", encoding="utf-8")

    with pytest.raises(
        FileExistsError,
        match="already exists",
    ):
        sample_results.export_csv(destination)


def test_export_csv_overwrites_existing_file_when_enabled(
    sample_results: SweepResults,
    tmp_path: Path,
) -> None:
    """Explicit overwrite permission should replace an existing file."""

    destination = tmp_path / "iv_curve.csv"
    destination.write_text("existing data", encoding="utf-8")

    sample_results.export_csv(
        destination,
        overwrite=True,
    )

    content = destination.read_text(encoding="utf-8")

    assert "voltage,current,absolute_current" in content
    assert "existing data" not in content


def test_export_csv_rejects_directory_path(
    sample_results: SweepResults,
    tmp_path: Path,
) -> None:
    """A directory cannot be used as the CSV destination."""

    with pytest.raises(
        IsADirectoryError,
        match="destination is a directory",
    ):
        sample_results.export_csv(tmp_path)


def test_export_csv_rejects_invalid_path_type(
    sample_results: SweepResults,
) -> None:
    """The output path should be a string or pathlib.Path."""

    with pytest.raises(
        TypeError,
        match="file_path must be",
    ):
        sample_results.export_csv(42)  # type: ignore[arg-type]


def test_export_csv_rejects_invalid_overwrite_value(
    sample_results: SweepResults,
    tmp_path: Path,
) -> None:
    """The overwrite option should be strictly Boolean."""

    with pytest.raises(
        TypeError,
        match="overwrite must be a boolean",
    ):
        sample_results.export_csv(
            tmp_path / "iv_curve.csv",
            overwrite=1,  # type: ignore[arg-type]
        )


def test_empty_container_can_be_exported(
    tmp_path: Path,
) -> None:
    """An empty sweep should still export a valid header-only CSV."""

    results = SweepResults()
    destination = tmp_path / "empty.csv"

    results.export_csv(destination)

    with destination.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        rows = list(csv.reader(csv_file))

    assert rows == [
        [
            "voltage",
            "current",
            "absolute_current",
            "iterations",
            "residual",
            "solve_time",
            "converged",
        ]
    ]


def test_repr_contains_useful_information(
    sample_results: SweepResults,
) -> None:
    """The representation should show size, convergence, and voltage range."""

    representation = repr(sample_results)

    assert "SweepResults" in representation
    assert "number_of_results=3" in representation
    assert "number_converged=2" in representation
    assert "voltage_range=(-0.5, 0.5)" in representation