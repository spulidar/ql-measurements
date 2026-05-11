"""Synthetic tests for molecular/Rayleigh physics utilities."""

from __future__ import annotations

import numpy as np

from milgrau.physics.atmosphere import get_standard_atmosphere
from milgrau.physics.molecular import calculate_molecular_profile, find_optimal_reference_altitude


def test_standard_atmosphere_shapes_and_monotonic_pressure() -> None:
    """Fallback atmosphere should produce finite pressure and temperature profiles."""
    altitude_m = np.arange(0.0, 15000.0, 7.5)

    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)

    assert pressure_hpa.shape == altitude_m.shape
    assert temperature_k.shape == altitude_m.shape
    assert np.all(np.isfinite(pressure_hpa))
    assert np.all(np.isfinite(temperature_k))
    assert pressure_hpa[0] > pressure_hpa[-1]
    assert np.all(temperature_k > 0.0)


def test_molecular_profile_is_positive_and_shape_consistent() -> None:
    """Rayleigh molecular profiles should be finite, positive and decreasing with altitude."""
    altitude_m = np.arange(0.0, 15000.0, 7.5)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)

    beta_mol, alpha_mol = calculate_molecular_profile(
        temp_profile=temperature_k,
        press_profile=pressure_hpa,
        wavelength_nm=532.0,
    )

    assert beta_mol.shape == altitude_m.shape
    assert alpha_mol.shape == altitude_m.shape
    assert np.all(np.isfinite(beta_mol))
    assert np.all(np.isfinite(alpha_mol))
    assert np.all(beta_mol > 0.0)
    assert np.all(alpha_mol > 0.0)
    assert beta_mol[0] > beta_mol[-1]
    assert alpha_mol[0] > alpha_mol[-1]


def test_reference_altitude_finds_flat_molecular_ratio_region() -> None:
    """The molecular reference search should find a region inside the requested bounds."""
    altitude_m = np.arange(0.0, 15000.0, 7.5)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    beta_mol, _ = calculate_molecular_profile(temperature_k, pressure_hpa, 532.0)

    # Synthetic perfectly molecular RCS: RCS / beta_mol is nearly constant.
    rcs = beta_mol * 1.0e12

    ref_idx = find_optimal_reference_altitude(
        rcs=rcs,
        beta_mol=beta_mol,
        altitude=altitude_m,
        min_alt=5000.0,
        max_alt=9000.0,
        window_size=50,
        altitude_units="m",
    )

    assert 5000.0 <= altitude_m[ref_idx] <= 9000.0
