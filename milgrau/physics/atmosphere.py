"""Atmospheric thermodynamic profile utilities."""

from __future__ import annotations

import numpy as np


def get_standard_atmosphere(altitude_array_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple US Standard Atmosphere-like pressure/temperature profile."""
    altitude_array_m = np.asarray(altitude_array_m, dtype=np.float64)
    z = np.maximum(altitude_array_m, 0.0)

    t0 = 288.15
    p0 = 1013.25
    lapse = 0.0065
    gas_constant = 8.3144598
    gravity = 9.80665
    molar_mass = 0.0289644

    temp = t0 - lapse * z
    temp = np.clip(temp, a_min=216.65, a_max=None)
    base = np.maximum(1.0 - (lapse * z) / t0, 1e-6)
    press = p0 * base ** ((gravity * molar_mass) / (gas_constant * lapse))
    return press.astype(np.float64), temp.astype(np.float64)
