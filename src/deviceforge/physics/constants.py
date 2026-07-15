from __future__ import annotations

"""
Physical constants used by DeviceForge.

All constants use SI units.
"""
# underscores help the readability, python still interprets correctly
# 1.602_176_634e-19 becomes 1.602176634e-19

ELEMENTARY_CHARGE: float = 1.602_176_634e-19
"""Elementary charge in coulombs."""

VACUUM_PERMITTIVITY: float = 8.854_187_812_8e-12
"""Vacuum permittivity in farads per metre."""

BOLTZMANN_CONSTANT: float = 1.380_649e-23
"""Boltzmann constant in joules per kelvin."""

ROOM_TEMPERATURE: float = 300.0
"""Default room temperature in kelvin."""