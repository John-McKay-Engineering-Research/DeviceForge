"""Simulation interface and adapter for DeviceForge voltage sweeps.

This module defines the minimal interface required by the solver-agnostic
VoltageSweep workflow and provides an adapter that can expose an existing
simulation object through that interface.

The adapter is intended to bridge existing DeviceForge simulation classes
without requiring those classes to be modified immediately.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol, TypeVar, runtime_checkable

from deviceforge.analysis.sweep_result import VoltageSweepResult


SimulationType = TypeVar("SimulationType")
SolutionType = TypeVar("SolutionType")


@runtime_checkable
class VoltageSweepSimulation(Protocol):
    """Minimal simulation interface required by a voltage sweep.

    A compatible object must provide three operations:

    1. Set the voltage of a named contact.
    2. Solve the current operating point.
    3. Convert the solver-specific solution into a VoltageSweepResult.

    Implementations do not need to inherit from this protocol. Python's
    structural typing allows any object with compatible methods to satisfy it.
    """

    def set_contact_voltage(
        self,
        contact: str,
        voltage: float,
    ) -> None:
        """Apply a voltage to a named electrical contact."""

        ...

    def solve(
        self,
        *,
        initial_guess: object | None = None,
    ) -> object:
        """Solve the current operating point.

        Parameters
        ----------
        initial_guess:
            Optional solver-specific solution from a previous operating point.

        Returns
        -------
        object
            Solver-specific solution object.
        """

        ...

    def build_voltage_sweep_result(
        self,
        *,
        voltage: float,
        solution: object,
    ) -> VoltageSweepResult:
        """Convert a solver solution into a standard voltage-sweep result."""

        ...


SetContactVoltageCallback = Callable[
    [SimulationType, str, float],
    None,
]

SolveCallback = Callable[
    [SimulationType, SolutionType | None],
    SolutionType,
]

BuildResultCallback = Callable[
    [SimulationType, float, SolutionType],
    VoltageSweepResult,
]


class VoltageSweepSimulationAdapter(
    Generic[SimulationType, SolutionType],
):
    """Adapt an arbitrary simulation object for use with VoltageSweep.

    Parameters
    ----------
    simulation:
        Existing simulation object being adapted.

    set_contact_voltage:
        Callback receiving:

        ``simulation, contact, voltage``

        The callback must update the requested contact voltage.

    solve:
        Callback receiving:

        ``simulation, initial_guess``

        The callback must execute one operating-point solve and return its
        solver-specific solution object.

    build_result:
        Callback receiving:

        ``simulation, voltage, solution``

        The callback must convert the solver-specific solution into a
        VoltageSweepResult.

    Notes
    -----
    The adapter allows VoltageSweep to work with existing simulation classes
    before those classes directly implement the VoltageSweepSimulation
    interface.

    Examples
    --------
    A simulation can be adapted using simple callback functions:

    .. code-block:: python

        adapter = VoltageSweepSimulationAdapter(
            simulation=my_simulation,
            set_contact_voltage=lambda sim, contact, voltage: (
                sim.update_contact(contact, voltage)
            ),
            solve=lambda sim, guess: sim.run(initial_state=guess),
            build_result=lambda sim, voltage, solution: (
                create_sweep_result(sim, voltage, solution)
            ),
        )
    """

    def __init__(
        self,
        simulation: SimulationType,
        *,
        set_contact_voltage: SetContactVoltageCallback[SimulationType],
        solve: SolveCallback[SimulationType, SolutionType],
        build_result: BuildResultCallback[
            SimulationType,
            SolutionType,
        ],
    ) -> None:
        """Initialise the adapter and validate its callbacks."""

        if simulation is None:
            raise TypeError("simulation must not be None.")

        self._validate_callback(
            set_contact_voltage,
            name="set_contact_voltage",
        )
        self._validate_callback(
            solve,
            name="solve",
        )
        self._validate_callback(
            build_result,
            name="build_result",
        )

        self._simulation = simulation
        self._set_contact_voltage_callback = set_contact_voltage
        self._solve_callback = solve
        self._build_result_callback = build_result

    @property
    def simulation(self) -> SimulationType:
        """Return the wrapped simulation object."""

        return self._simulation

    def set_contact_voltage(
        self,
        contact: str,
        voltage: float,
    ) -> None:
        """Apply a contact voltage through the configured callback."""

        self._set_contact_voltage_callback(
            self._simulation,
            contact,
            voltage,
        )

    def solve(
        self,
        *,
        initial_guess: object | None = None,
    ) -> SolutionType:
        """Solve one operating point through the configured callback."""

        return self._solve_callback(
            self._simulation,
            initial_guess,  # type: ignore[arg-type]
        )

    def build_voltage_sweep_result(
        self,
        *,
        voltage: float,
        solution: object,
    ) -> VoltageSweepResult:
        """Build a standard sweep result through the configured callback."""

        result = self._build_result_callback(
            self._simulation,
            voltage,
            solution,  # type: ignore[arg-type]
        )

        if not isinstance(result, VoltageSweepResult):
            raise TypeError(
                "The build_result callback must return a "
                "VoltageSweepResult instance."
            )

        return result

    def __repr__(self) -> str:
        """Return an informative developer representation."""

        return (
            f"{type(self).__name__}("
            f"simulation={type(self._simulation).__name__}"
            ")"
        )

    @staticmethod
    def _validate_callback(
        callback: object,
        *,
        name: str,
    ) -> None:
        """Validate that an adapter callback is callable."""

        if not callable(callback):
            raise TypeError(f"{name} must be callable.")