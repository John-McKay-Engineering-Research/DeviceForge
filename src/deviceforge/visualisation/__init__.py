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
]