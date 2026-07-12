from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class Material:
    """
    Physical material used in a semiconductor device model.

    All numerical values use SI units unless otherwise stated.

    Parameters
    ----------
    name:
        Human-readable material name.

    relative_permittivity:
        Dimensionless relative permittivity.

    band_gap_ev:
        Electronic band gap in electronvolts.

    electron_affinity_ev:
        Electron affinity in electronvolts.

    electron_mobility:
        Electron mobility in square metres per volt-second.

    hole_mobility:
        Hole mobility in square metres per volt-second.

    thermal_conductivity:
        Thermal conductivity in watts per metre-kelvin.

    material_type:
        Broad material category, such as ``"semiconductor"``,
        ``"dielectric"`` or ``"conductor"``.
    """

    name: str
    relative_permittivity: float
    material_type: str

    band_gap_ev: float | None = None
    electron_affinity_ev: float | None = None
    electron_mobility: float | None = None
    hole_mobility: float | None = None
    thermal_conductivity: float | None = None

    def __post_init__(self) -> None:
        """Validate the material definition."""

        if not self.name.strip():
            raise ValueError("Material name must not be empty.")

        valid_material_types = {
            "semiconductor",
            "dielectric",
            "conductor",
        }

        normalised_type = self.material_type.strip().lower()

        if normalised_type not in valid_material_types:
            raise ValueError(
                "Material type must be one of: "
                "'semiconductor', 'dielectric', or 'conductor'."
            )

        if not isfinite(self.relative_permittivity):
            raise ValueError("Relative permittivity must be finite.")

        if self.relative_permittivity <= 0.0:
            raise ValueError("Relative permittivity must be positive.")

        optional_properties = {
            "band_gap_ev": self.band_gap_ev,
            "electron_affinity_ev": self.electron_affinity_ev,
            "electron_mobility": self.electron_mobility,
            "hole_mobility": self.hole_mobility,
            "thermal_conductivity": self.thermal_conductivity,
        }

        for property_name, value in optional_properties.items():
            if value is None:
                continue

            if not isfinite(value):
                raise ValueError(
                    f"{property_name} must be finite when provided."
                )

            if value < 0.0:
                raise ValueError(
                    f"{property_name} must not be negative."
                )

        object.__setattr__(self, "material_type", normalised_type)

    @property
    def is_semiconductor(self) -> bool:
        """Return whether the material is a semiconductor."""

        return self.material_type == "semiconductor"

    @property
    def is_dielectric(self) -> bool:
        """Return whether the material is a dielectric."""

        return self.material_type == "dielectric"

    @property
    def is_conductor(self) -> bool:
        """Return whether the material is a conductor."""

        return self.material_type == "conductor"

# pre-defined materials ****
SILICON = Material(
    name="Silicon",
    material_type="semiconductor",
    relative_permittivity=11.7,
    band_gap_ev=1.12,
    electron_affinity_ev=4.05,
    electron_mobility=0.135,
    hole_mobility=0.048,
    thermal_conductivity=148.0,
)

SILICON_DIOXIDE = Material(
    name="Silicon dioxide",
    material_type="dielectric",
    relative_permittivity=3.9,
    band_gap_ev=8.9,
    electron_affinity_ev=0.95,
    thermal_conductivity=1.4,
)

GENERIC_CONDUCTOR = Material(
    name="Generic conductor",
    material_type="conductor",
    relative_permittivity=1.0,
)