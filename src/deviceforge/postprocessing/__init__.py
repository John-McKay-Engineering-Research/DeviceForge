from .electric_field import (
    ElectricField,
    compute_electric_field,
)

from .electrostatics import calculate_electric_field

from .electrostatics import (
    calculate_electric_displacement_field,
    calculate_electric_field,
    calculate_electrostatic_energy_density,
    calculate_face_electric_displacement,
    calculate_face_electric_field,
    calculate_face_electrostatic_fields,
    calculate_face_relative_permittivity,
)

from .analysis import (
    ElectrostaticAnalysis,
    analyse_electrostatics,
)

from .electrostatics_2d import (
    calculate_electric_field_components_2d,
    calculate_electric_field_magnitude_2d,
    calculate_electrostatic_fields_2d,
)

__all__ = [
    "ElectricField",
    "compute_electric_field",
    "calculate_electric_field",
    "calculate_electric_displacement_field",
    "calculate_electrostatic_energy_density",
    "ElectrostaticAnalysis",
    "analyse_electrostatics",
    "calculate_face_electric_displacement",
    "calculate_face_electric_field",
    "calculate_face_electrostatic_fields",
    "calculate_face_relative_permittivity",
    "calculate_electric_field_components_2d",
    "calculate_electric_field_magnitude_2d",
    "calculate_electrostatic_fields_2d",
]