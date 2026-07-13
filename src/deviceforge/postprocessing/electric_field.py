from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deviceforge.core import Field, Grid


@dataclass(frozen=True, slots=True)
class ElectricField:
    """
    Two-dimensional electric-field result.

    Parameters
    ----------
    x_component:
        Electric-field component in the x direction.

    y_component:
        Electric-field component in the y direction.

    magnitude:
        Magnitude of the electric-field vector.
    """

    x_component: Field
    y_component: Field
    magnitude: Field

    def __post_init__(self) -> None:
        """Validate the electric-field components."""

        grid = self.x_component.grid

        if self.y_component.grid != grid:
            raise ValueError(
                "Electric-field components must use the same grid."
            )

        if self.magnitude.grid != grid:
            raise ValueError(
                "Electric-field magnitude must use the same grid "
                "as its components."
            )

        expected_units = "V/m"

        for field in (
            self.x_component,
            self.y_component,
            self.magnitude,
        ):
            if field.units != expected_units:
                raise ValueError(
                    "Electric-field values must use units of V/m."
                )

    @property
    def grid(self) -> Grid:
        """Return the shared computational grid."""

        return self.x_component.grid

    @property
    def components(self) -> tuple[Field, Field]:
        """Return the x and y components."""

        return (
            self.x_component,
            self.y_component,
        )


def compute_electric_field(
    potential: Field,
) -> ElectricField:
    """
    Calculate the electric field from electrostatic potential.

    The electric field is calculated using:

    ``E = -grad(phi)``

    Parameters
    ----------
    potential:
        Two-dimensional electrostatic-potential field in volts.

    Returns
    -------
    ElectricField
        Electric-field components and magnitude.

    Raises
    ------
    ValueError
        If the field is not two-dimensional or does not use volts.
    """

    if potential.grid.dimension != 2:
        raise ValueError(
            "Electric-field post-processing currently supports "
            "only two-dimensional potential fields."
        )

    if potential.units != "V":
        raise ValueError(
            "Electrostatic potential must use units of V."
        )

    spacing_x, spacing_y = potential.grid.spacing

    minimum_axis_size = min(potential.grid.shape)

    edge_order = 2 if minimum_axis_size >= 3 else 1

    potential_gradient_x, potential_gradient_y = np.gradient(
        potential.values,
        spacing_x,
        spacing_y,
        edge_order=edge_order,
    )

    electric_field_x_values = -potential_gradient_x
    electric_field_y_values = -potential_gradient_y

    magnitude_values = np.sqrt(
        electric_field_x_values**2
        + electric_field_y_values**2
    )

    x_component = Field(
        name="electric_field_x",
        units="V/m",
        grid=potential.grid,
        values=electric_field_x_values,
    )

    y_component = Field(
        name="electric_field_y",
        units="V/m",
        grid=potential.grid,
        values=electric_field_y_values,
    )

    magnitude = Field(
        name="electric_field_magnitude",
        units="V/m",
        grid=potential.grid,
        values=magnitude_values,
    )

    return ElectricField(
        x_component=x_component,
        y_component=y_component,
        magnitude=magnitude,
    )