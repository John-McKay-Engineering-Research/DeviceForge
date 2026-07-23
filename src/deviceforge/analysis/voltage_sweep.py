"""Solver-agnostic voltage-sweep workflow for DeviceForge.

This module defines :class:`VoltageSweep`, a high-level analysis workflow that
applies a sequence of contact voltages, executes a simulation at each operating
point, and stores the resulting data in a :class:`SweepResults` container.

The voltage-sweep workflow does not depend on any particular physical model,
numerical solver, spatial dimension, or compute backend. Instead, simulations
participate through the small :class:`VoltageSweepSimulation` protocol.
"""

from __future__ import annotations

from collections.abc import Iterable
# from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from deviceforge.analysis.sweep_result import VoltageSweepResult
from deviceforge.analysis.sweep_results import SweepResults
from deviceforge.analysis.voltage_sweep_simulation import (
    VoltageSweepSimulation,
)

FloatArray = NDArray[np.float64]

# *** removed class ***

class VoltageSweep:
    """Execute a sequence of voltage-biased simulations.

    Parameters
    ----------
    simulation:
        Simulation object implementing :class:`VoltageSweepSimulation`.

    contact:
        Name of the electrical contact whose voltage will be varied.

    voltages:
        Finite sequence of voltage values in volts. Values are executed in
        exactly the order supplied.

    warm_start:
        When ``True``, the raw converged solution from one operating point is
        supplied as the initial guess for the next operating point.

    stop_on_failure:
        When ``True``, the sweep stops after the first result whose
        ``converged`` flag is ``False``. The failed result remains stored.

    metadata:
        Optional user-defined metadata copied into the resulting
        :class:`SweepResults` container.

    Notes
    -----
    ``VoltageSweep`` is solver agnostic. It does not import or inspect a
    Poisson, Gummel, Newton, drift-diffusion, thermal, CPU, or GPU solver.

    Compatibility is determined entirely by the three-method
    :class:`VoltageSweepSimulation` protocol.
    """

    def __init__(
        self,
        simulation: VoltageSweepSimulation,
        contact: str,
        voltages: Iterable[float],
        *,
        warm_start: bool = True,
        stop_on_failure: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Initialise a voltage-sweep workflow."""

        self._validate_simulation(simulation)
        self._simulation = simulation

        self._contact = self._validate_contact(contact)
        self._voltages = self._validate_voltages(voltages)

        if not isinstance(warm_start, bool):
            raise TypeError("warm_start must be a boolean value.")

        if not isinstance(stop_on_failure, bool):
            raise TypeError("stop_on_failure must be a boolean value.")

        self._warm_start = warm_start
        self._stop_on_failure = stop_on_failure

        if metadata is None:
            self._metadata: dict[str, object] = {}
        elif isinstance(metadata, dict):
            self._metadata = dict(metadata)
        else:
            raise TypeError("metadata must be a dictionary or None.")

    @property
    def simulation(self) -> VoltageSweepSimulation:
        """Return the simulation controlled by the sweep."""

        return self._simulation

    @property
    def contact(self) -> str:
        """Return the name of the swept electrical contact."""

        return self._contact

    @property
    def voltages(self) -> FloatArray:
        """Return an independent copy of the requested voltage sequence."""

        return self._voltages.copy()

    @property
    def number_of_points(self) -> int:
        """Return the requested number of operating points."""

        return int(self._voltages.size)

    @property
    def warm_start(self) -> bool:
        """Return whether warm starting is enabled."""

        return self._warm_start

    @property
    def stop_on_failure(self) -> bool:
        """Return whether the sweep stops after non-convergence."""

        return self._stop_on_failure

    @property
    def metadata(self) -> dict[str, object]:
        """Return the mutable user-defined sweep metadata."""

        return self._metadata

    def run(self) -> SweepResults:
        """Execute the complete voltage sweep.

        Returns
        -------
        SweepResults
            Operating-point results in execution order.

        Raises
        ------
        RuntimeError
            If applying a voltage, solving an operating point, or constructing
            a result raises an exception.

        TypeError
            If the simulation does not return a VoltageSweepResult.

        ValueError
            If the result voltage does not match the applied voltage.

        Notes
        -----
        A new :class:`SweepResults` container is created for every call. Calling
        ``run`` twice therefore executes the simulation twice and does not
        append to the previous result collection.

        Warm starting occurs only after a converged operating point. A failed
        solution is not forwarded as the initial guess for the next point.
        """

        results = SweepResults(
            metadata=self._build_results_metadata(),
        )

        initial_guess: object | None = None

        for point_index, voltage_value in enumerate(self._voltages):
            voltage = float(voltage_value)

            try:
                self._simulation.set_contact_voltage(
                    self._contact,
                    voltage,
                )
            except Exception as exc:
                raise RuntimeError(
                    self._format_failure_message(
                        action="apply contact voltage",
                        point_index=point_index,
                        voltage=voltage,
                    )
                ) from exc

            try:
                solution = self._simulation.solve(
                    initial_guess=(
                        initial_guess
                        if self._warm_start
                        else None
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    self._format_failure_message(
                        action="solve operating point",
                        point_index=point_index,
                        voltage=voltage,
                    )
                ) from exc

            try:
                result = self._simulation.build_voltage_sweep_result(
                    voltage=voltage,
                    solution=solution,
                )
            except Exception as exc:
                raise RuntimeError(
                    self._format_failure_message(
                        action="build voltage-sweep result",
                        point_index=point_index,
                        voltage=voltage,
                    )
                ) from exc

            self._validate_generated_result(
                result=result,
                expected_voltage=voltage,
            )

            results.append(result)

            if result.converged:
                if self._warm_start:
                    initial_guess = solution
            else:
                initial_guess = None

                if self._stop_on_failure:
                    break

        results.metadata["number_completed"] = len(results)
        results.metadata["terminated_early"] = (
            len(results) < self.number_of_points
        )

        return results

    @classmethod
    def from_range(
        cls,
        simulation: VoltageSweepSimulation,
        contact: str,
        start: float,
        stop: float,
        step: float,
        *,
        warm_start: bool = True,
        stop_on_failure: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> VoltageSweep:
        """Create a sweep from a start, stop, and nominal step voltage.

        Parameters
        ----------
        simulation:
            Compatible DeviceForge simulation.

        contact:
            Name of the contact to vary.

        start:
            First voltage in volts.

        stop:
            Final voltage in volts.

        step:
            Nominal signed voltage increment.

        warm_start:
            Enable reuse of converged solutions.

        stop_on_failure:
            Stop after the first non-converged result.

        metadata:
            Optional user-defined sweep metadata.

        Returns
        -------
        VoltageSweep
            Configured voltage-sweep workflow.

        Notes
        -----
        The stop voltage is always included. When the requested interval is not
        exactly divisible by ``step``, the final interval may be smaller than
        the nominal step.

        Examples
        --------
        Ascending sweep:

        ``VoltageSweep.from_range(simulation, "anode", -1.0, 1.0, 0.1)``

        Descending sweep:

        ``VoltageSweep.from_range(simulation, "gate", 1.0, -1.0, -0.1)``
        """

        start_voltage = cls._validate_finite_float(
            start,
            name="start",
        )
        stop_voltage = cls._validate_finite_float(
            stop,
            name="stop",
        )
        step_voltage = cls._validate_finite_float(
            step,
            name="step",
        )

        if step_voltage == 0.0:
            raise ValueError("step must not be zero.")

        voltage_difference = stop_voltage - start_voltage

        if voltage_difference > 0.0 and step_voltage < 0.0:
            raise ValueError(
                "step must be positive for an ascending voltage range."
            )

        if voltage_difference < 0.0 and step_voltage > 0.0:
            raise ValueError(
                "step must be negative for a descending voltage range."
            )

        if voltage_difference == 0.0:
            generated_voltages = np.array(
                [start_voltage],
                dtype=np.float64,
            )
        else:
            generated_voltages = cls._generate_voltage_range(
                start=start_voltage,
                stop=stop_voltage,
                step=step_voltage,
            )

        return cls(
            simulation=simulation,
            contact=contact,
            voltages=generated_voltages,
            warm_start=warm_start,
            stop_on_failure=stop_on_failure,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        """Return an informative developer representation."""

        return (
            f"{type(self).__name__}("
            f"contact={self._contact!r}, "
            f"number_of_points={self.number_of_points}, "
            f"voltage_range=("
            f"{float(self._voltages[0])}, "
            f"{float(self._voltages[-1])}"
            f"), "
            f"warm_start={self._warm_start}, "
            f"stop_on_failure={self._stop_on_failure}"
            ")"
        )

    def _build_results_metadata(self) -> dict[str, object]:
        """Construct metadata for one sweep execution."""

        results_metadata = dict(self._metadata)

        results_metadata.update(
            {
                "analysis_type": "voltage_sweep",
                "contact": self._contact,
                "requested_voltages": self._voltages.tolist(),
                "number_requested": self.number_of_points,
                "warm_start": self._warm_start,
                "stop_on_failure": self._stop_on_failure,
            }
        )

        return results_metadata

    def _format_failure_message(
        self,
        *,
        action: str,
        point_index: int,
        voltage: float,
    ) -> str:
        """Return a contextual workflow error message."""

        return (
            f"Failed to {action} at sweep point {point_index} "
            f"for contact {self._contact!r} at {voltage} V."
        )

    @staticmethod
    def _validate_simulation(
        simulation: object,
    ) -> None:
        """Validate the simulation's required interface."""

        required_methods = (
            "set_contact_voltage",
            "solve",
            "build_voltage_sweep_result",
        )

        missing_methods = [
            method_name
            for method_name in required_methods
            if not callable(getattr(simulation, method_name, None))
        ]

        if missing_methods:
            missing_text = ", ".join(missing_methods)

            raise TypeError(
                "simulation must implement the VoltageSweepSimulation "
                f"interface. Missing callable methods: {missing_text}."
            )

    @staticmethod
    def _validate_contact(contact: object) -> str:
        """Validate and normalise a contact name."""

        if not isinstance(contact, str):
            raise TypeError("contact must be a string.")

        normalised_contact = contact.strip()

        if not normalised_contact:
            raise ValueError("contact must not be empty.")

        return normalised_contact

    @classmethod
    def _validate_voltages(
        cls,
        voltages: Iterable[float],
    ) -> FloatArray:
        """Validate and copy the requested voltage sequence."""

        if isinstance(voltages, (str, bytes)):
            raise TypeError(
                "voltages must be an iterable of finite real numbers."
            )

        try:
            voltage_values = list(voltages)
        except TypeError as exc:
            raise TypeError(
                "voltages must be an iterable of finite real numbers."
            ) from exc

        if not voltage_values:
            raise ValueError("voltages must contain at least one value.")

        validated_values = [
            cls._validate_finite_float(
                value,
                name=f"voltages[{index}]",
            )
            for index, value in enumerate(voltage_values)
        ]

        return np.asarray(
            validated_values,
            dtype=np.float64,
        )

    @staticmethod
    def _validate_generated_result(
        *,
        result: object,
        expected_voltage: float,
    ) -> None:
        """Validate a result produced by the simulation."""

        if not isinstance(result, VoltageSweepResult):
            raise TypeError(
                "build_voltage_sweep_result must return a "
                "VoltageSweepResult instance."
            )

        if not np.isclose(
            result.voltage,
            expected_voltage,
            rtol=1.0e-9,
            atol=1.0e-12,
        ):
            raise ValueError(
                "The generated VoltageSweepResult voltage does not match "
                f"the applied voltage. Expected {expected_voltage} V but "
                f"received {result.voltage} V."
            )

    @staticmethod
    def _generate_voltage_range(
        *,
        start: float,
        stop: float,
        step: float,
    ) -> FloatArray:
        """Generate an inclusive ascending or descending voltage range."""

        ratio = (stop - start) / step
        number_of_full_steps = int(np.floor(ratio + 1.0e-12))

        values = start + (
            step
            * np.arange(
                number_of_full_steps + 1,
                dtype=np.float64,
            )
        )

        if np.isclose(
            values[-1],
            stop,
            rtol=1.0e-12,
            atol=1.0e-15,
        ):
            values[-1] = stop
        else:
            values = np.append(
                values,
                np.float64(stop),
            )

        return np.asarray(
            values,
            dtype=np.float64,
        )

    @staticmethod
    def _validate_finite_float(
        value: object,
        *,
        name: str,
    ) -> float:
        """Validate and return a finite real value."""

        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a real number.")

        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name} must be a real number."
            ) from exc

        if not np.isfinite(numeric_value):
            raise ValueError(f"{name} must be finite.")

        return numeric_value