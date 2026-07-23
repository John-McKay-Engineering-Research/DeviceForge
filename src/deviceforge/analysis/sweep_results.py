"""Container for voltage-sweep operating-point results.

This module defines :class:`SweepResults`, which stores and manages an ordered
collection of :class:`VoltageSweepResult` objects produced during a DeviceForge
voltage sweep.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import overload

import numpy as np
from numpy.typing import NDArray

from deviceforge.analysis.sweep_result import VoltageSweepResult


FloatArray = NDArray[np.float64]


class SweepResults(Sequence[VoltageSweepResult]):
    """Store an ordered collection of voltage-sweep results.

    Parameters
    ----------
    results:
        Optional iterable of :class:`VoltageSweepResult` objects.

    metadata:
        Optional metadata associated with the complete sweep.

    Notes
    -----
    Results are copied when they are added to the container. This prevents
    changes to solver-owned arrays or external result objects from silently
    modifying the stored sweep history.

    The container follows the standard Python sequence protocol and therefore
    supports:

    - ``len(results)``
    - ``results[index]``
    - ``results[start:stop]``
    - iteration
    - reversed iteration
    - membership testing
    """

    def __init__(
        self,
        results: Iterable[VoltageSweepResult] | None = None,
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Initialise an empty or pre-populated sweep-results container."""

        if metadata is None:
            self._metadata: dict[str, object] = {}
        elif isinstance(metadata, dict):
            self._metadata = dict(metadata)
        else:
            raise TypeError("metadata must be a dictionary or None.")

        self._results: list[VoltageSweepResult] = []

        if results is not None:
            self.extend(results)

    @property
    def metadata(self) -> dict[str, object]:
        """Return the mutable metadata dictionary associated with the sweep."""

        return self._metadata

    @property
    def voltages(self) -> FloatArray:
        """Return the applied voltages in sweep order."""

        return np.asarray(
            [result.voltage for result in self._results],
            dtype=np.float64,
        )

    @property
    def currents(self) -> FloatArray:
        """Return the extracted terminal currents in sweep order."""

        return np.asarray(
            [result.current for result in self._results],
            dtype=np.float64,
        )

    @property
    def absolute_currents(self) -> FloatArray:
        """Return the magnitudes of the extracted terminal currents."""

        return np.abs(self.currents)

    @property
    def iterations(self) -> NDArray[np.int64]:
        """Return nonlinear iteration counts for all operating points."""

        return np.asarray(
            [result.iterations for result in self._results],
            dtype=np.int64,
        )

    @property
    def residuals(self) -> FloatArray:
        """Return final residuals for all operating points."""

        return np.asarray(
            [result.residual for result in self._results],
            dtype=np.float64,
        )

    @property
    def solve_times(self) -> FloatArray:
        """Return solve times for all operating points in seconds."""

        return np.asarray(
            [result.solve_time for result in self._results],
            dtype=np.float64,
        )

    @property
    def convergence_flags(self) -> NDArray[np.bool_]:
        """Return convergence flags for all operating points."""

        return np.asarray(
            [result.converged for result in self._results],
            dtype=np.bool_,
        )

    @property
    def converged_results(self) -> tuple[VoltageSweepResult, ...]:
        """Return all converged operating points in sweep order."""

        return tuple(
            result.copy()
            for result in self._results
            if result.converged
        )

    @property
    def failed_results(self) -> tuple[VoltageSweepResult, ...]:
        """Return all non-converged operating points in sweep order."""

        return tuple(
            result.copy()
            for result in self._results
            if not result.converged
        )

    @property
    def number_converged(self) -> int:
        """Return the number of converged operating points."""

        return sum(result.converged for result in self._results)

    @property
    def number_failed(self) -> int:
        """Return the number of non-converged operating points."""

        return len(self._results) - self.number_converged

    @property
    def all_converged(self) -> bool:
        """Return whether every stored operating point converged.

        An empty sweep returns ``True`` because it contains no failed points.
        """

        return all(result.converged for result in self._results)

    @property
    def convergence_rate(self) -> float:
        """Return the fraction of operating points that converged.

        Returns
        -------
        float
            Value between zero and one. An empty container returns ``0.0``.
        """

        if not self._results:
            return 0.0

        return self.number_converged / len(self._results)

    @property
    def total_solve_time(self) -> float:
        """Return the accumulated solve time for all operating points."""

        return float(sum(result.solve_time for result in self._results))

    @property
    def average_solve_time(self) -> float:
        """Return the mean solve time per operating point.

        An empty container returns ``0.0``.
        """

        if not self._results:
            return 0.0

        return self.total_solve_time / len(self._results)

    @property
    def total_iterations(self) -> int:
        """Return the total nonlinear iteration count."""

        return sum(result.iterations for result in self._results)

    @property
    def average_iterations(self) -> float:
        """Return the mean nonlinear iteration count.

        An empty container returns ``0.0``.
        """

        if not self._results:
            return 0.0

        return self.total_iterations / len(self._results)

    @property
    def minimum_voltage(self) -> float | None:
        """Return the minimum stored voltage, or ``None`` when empty."""

        if not self._results:
            return None

        return float(np.min(self.voltages))

    @property
    def maximum_voltage(self) -> float | None:
        """Return the maximum stored voltage, or ``None`` when empty."""

        if not self._results:
            return None

        return float(np.max(self.voltages))

    def append(self, result: VoltageSweepResult) -> None:
        """Append an independent copy of one operating-point result.

        Parameters
        ----------
        result:
            Result to append.

        Raises
        ------
        TypeError
            If ``result`` is not a :class:`VoltageSweepResult`.
        """

        self._validate_result(result)
        self._results.append(result.copy())

    def extend(
        self,
        results: Iterable[VoltageSweepResult],
    ) -> None:
        """Append multiple operating-point results.

        All input values are validated before any result is added. Therefore,
        if one input item is invalid, the container remains unchanged.

        Parameters
        ----------
        results:
            Iterable of voltage-sweep results.

        Raises
        ------
        TypeError
            If ``results`` is not iterable or contains an invalid item.
        """

        try:
            candidate_results = list(results)
        except TypeError as exc:
            raise TypeError(
                "results must be an iterable of VoltageSweepResult objects."
            ) from exc

        for result in candidate_results:
            self._validate_result(result)

        self._results.extend(
            result.copy()
            for result in candidate_results
        )

    def insert(
        self,
        index: int,
        result: VoltageSweepResult,
    ) -> None:
        """Insert an independent result copy at a specified index."""

        if isinstance(index, bool) or not isinstance(
            index,
            (int, np.integer),
        ):
            raise TypeError("index must be an integer.")

        self._validate_result(result)
        self._results.insert(int(index), result.copy())

    def clear(self) -> None:
        """Remove all operating-point results from the container."""

        self._results.clear()

    def copy(self) -> SweepResults:
        """Return a deep copy of the complete sweep-results container."""

        return SweepResults(
            self._results,
            metadata=self._metadata.copy(),
        )

    def sorted_by_voltage(
        self,
        *,
        reverse: bool = False,
    ) -> SweepResults:
        """Return a new container sorted by applied voltage.

        Parameters
        ----------
        reverse:
            Sort from highest to lowest voltage when ``True``.
        """

        if not isinstance(reverse, bool):
            raise TypeError("reverse must be a boolean value.")

        ordered_results = sorted(
            self._results,
            key=lambda result: result.voltage,
            reverse=reverse,
        )

        return SweepResults(
            ordered_results,
            metadata=self._metadata.copy(),
        )

    def result_at_voltage(
        self,
        voltage: float,
        *,
        atol: float = 1.0e-12,
        rtol: float = 1.0e-9,
    ) -> VoltageSweepResult:
        """Return the first result matching a requested voltage.

        Parameters
        ----------
        voltage:
            Requested applied voltage.

        atol:
            Absolute tolerance used by :func:`numpy.isclose`.

        rtol:
            Relative tolerance used by :func:`numpy.isclose`.

        Raises
        ------
        TypeError
            If the inputs cannot be interpreted as real numbers.

        ValueError
            If a value is non-finite or a tolerance is negative.

        LookupError
            If no operating point matches the requested voltage.
        """

        requested_voltage = self._validate_finite_float(
            voltage,
            name="voltage",
        )
        absolute_tolerance = self._validate_non_negative_float(
            atol,
            name="atol",
        )
        relative_tolerance = self._validate_non_negative_float(
            rtol,
            name="rtol",
        )

        for result in self._results:
            if np.isclose(
                result.voltage,
                requested_voltage,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            ):
                return result.copy()

        raise LookupError(
            f"No sweep result was found at voltage {requested_voltage} V."
        )

    def summary(self) -> dict[str, object]:
        """Return a compact serialisable summary of the complete sweep."""

        return {
            "number_of_results": len(self._results),
            "number_converged": self.number_converged,
            "number_failed": self.number_failed,
            "all_converged": self.all_converged,
            "convergence_rate": self.convergence_rate,
            "minimum_voltage": self.minimum_voltage,
            "maximum_voltage": self.maximum_voltage,
            "total_iterations": self.total_iterations,
            "average_iterations": self.average_iterations,
            "total_solve_time": self.total_solve_time,
            "average_solve_time": self.average_solve_time,
        }

    def to_records(self) -> list[dict[str, object]]:
        """Return one compact dictionary per operating point.

        The returned records contain scalar sweep data only. Full numerical
        fields remain stored in each :class:`VoltageSweepResult`.
        """

        return [
            {
                "voltage": result.voltage,
                "current": result.current,
                "absolute_current": result.absolute_current,
                "iterations": result.iterations,
                "residual": result.residual,
                "solve_time": result.solve_time,
                "converged": result.converged,
            }
            for result in self._results
        ]

    def export_csv(
        self,
        file_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        """Export compact voltage-sweep data to a CSV file.

        Parameters
        ----------
        file_path:
            Destination path.

        overwrite:
            Permit replacement of an existing file when ``True``.

        Returns
        -------
        pathlib.Path
            Resolved path to the written CSV file.

        Raises
        ------
        TypeError
            If the path or overwrite flag has an invalid type.

        FileExistsError
            If the destination exists and ``overwrite`` is ``False``.

        IsADirectoryError
            If the destination path refers to a directory.

        Notes
        -----
        Only compact scalar data are exported:

        - voltage
        - current
        - absolute current
        - iterations
        - residual
        - solve time
        - convergence status

        Full solution fields should later be exported using a dedicated
        scientific-data format such as HDF5.
        """

        if not isinstance(file_path, (str, Path)):
            raise TypeError("file_path must be a string or pathlib.Path.")

        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean value.")

        path = Path(file_path).expanduser()

        if path.exists() and path.is_dir():
            raise IsADirectoryError(
                f"The CSV destination is a directory: {path}"
            )

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"The destination file already exists: {path}"
            )

        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")

        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "voltage",
            "current",
            "absolute_current",
            "iterations",
            "residual",
            "solve_time",
            "converged",
        ]

        with path.open(
            mode="w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(self.to_records())

        return path.resolve()

    @overload
    def __getitem__(
        self,
        index: int,
    ) -> VoltageSweepResult:
        ...

    @overload
    def __getitem__(
        self,
        index: slice,
    ) -> SweepResults:
        ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> VoltageSweepResult | SweepResults:
        """Return an independent result copy or sliced container."""

        if isinstance(index, slice):
            return SweepResults(
                self._results[index],
                metadata=self._metadata.copy(),
            )

        if isinstance(index, bool) or not isinstance(
            index,
            (int, np.integer),
        ):
            raise TypeError("sweep result indices must be integers or slices.")

        return self._results[int(index)].copy()

    def __len__(self) -> int:
        """Return the number of stored operating points."""

        return len(self._results)

    def __iter__(self) -> Iterator[VoltageSweepResult]:
        """Iterate over independent copies of stored operating points."""

        for result in self._results:
            yield result.copy()

    def __reversed__(self) -> Iterator[VoltageSweepResult]:
        """Iterate over independent copies in reverse sweep order."""

        for result in reversed(self._results):
            yield result.copy()

    def __contains__(self, item: object) -> bool:
        """Return whether an equal result is present.

        Membership uses object identity-independent scalar and array
        comparisons rather than NumPy's ambiguous default array equality.
        """

        if not isinstance(item, VoltageSweepResult):
            return False

        return any(
            self._results_equal(existing, item)
            for existing in self._results
        )

    def __repr__(self) -> str:
        """Return an informative developer representation."""

        return (
            f"{type(self).__name__}("
            f"number_of_results={len(self)}, "
            f"number_converged={self.number_converged}, "
            f"voltage_range=({self.minimum_voltage}, "
            f"{self.maximum_voltage})"
            ")"
        )

    @staticmethod
    def _validate_result(result: object) -> None:
        """Validate one result before storage."""

        if not isinstance(result, VoltageSweepResult):
            raise TypeError(
                "Each sweep result must be a VoltageSweepResult instance."
            )

    @staticmethod
    def _results_equal(
        first: VoltageSweepResult,
        second: VoltageSweepResult,
    ) -> bool:
        """Return whether two sweep results contain equivalent data."""

        scalar_values_equal = (
            first.voltage == second.voltage
            and first.current == second.current
            and first.iterations == second.iterations
            and first.residual == second.residual
            and first.solve_time == second.solve_time
            and first.converged == second.converged
            and first.metadata == second.metadata
        )

        if not scalar_values_equal:
            return False

        return (
            np.array_equal(first.potential, second.potential)
            and np.array_equal(
                first.electron_density,
                second.electron_density,
            )
            and np.array_equal(
                first.hole_density,
                second.hole_density,
            )
            and np.array_equal(
                first.electric_field,
                second.electric_field,
            )
            and np.array_equal(
                first.electron_current_density,
                second.electron_current_density,
            )
            and np.array_equal(
                first.hole_current_density,
                second.hole_current_density,
            )
            and np.array_equal(
                first.total_current_density,
                second.total_current_density,
            )
        )

    @staticmethod
    def _validate_finite_float(
        value: float,
        *,
        name: str,
    ) -> float:
        """Validate and return a finite floating-point value."""

        if isinstance(value, bool):
            raise TypeError(f"{name} must be a real number.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real number.") from exc

        if not np.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite.")

        return numeric_value

    @classmethod
    def _validate_non_negative_float(
        cls,
        value: float,
        *,
        name: str,
    ) -> float:
        """Validate and return a non-negative finite float."""

        numeric_value = cls._validate_finite_float(
            value,
            name=name,
        )

        if numeric_value < 0.0:
            raise ValueError(
                f"{name} must be greater than or equal to zero."
            )

        return numeric_value