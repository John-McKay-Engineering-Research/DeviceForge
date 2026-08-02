from .fields import (
    plot_convergence,
    plot_scalar_field,
    plot_vector_field,
    save_figure,
)

from .electrostatics_1d import (
    plot_electric_displacement,
    plot_electric_field,
    plot_electrostatic_energy_density,
    plot_electrostatic_potential,
    plot_relative_permittivity,
    plot_residual_history,
    plot_face_electric_displacement,
)

from .electrostatics_2d import (
    plot_electric_field_magnitude_2d,
    plot_electric_field_vectors_2d,
    plot_electrostatic_potential_2d,
    plot_electrostatic_solution_2d,
    plot_equipotential_contours_2d,
)

__all__ = [
    "plot_convergence",
    "plot_scalar_field",
    "plot_vector_field",
    "save_figure",
    "plot_electric_displacement",
    "plot_electric_field",
    "plot_electrostatic_energy_density",
    "plot_electrostatic_potential",
    "plot_relative_permittivity",
    "plot_residual_history",
    "plot_face_electric_displacement",
    "plot_electric_field_magnitude_2d",
    "plot_electric_field_vectors_2d",
    "plot_electrostatic_potential_2d",
    "plot_electrostatic_solution_2d",
    "plot_equipotential_contours_2d",
]