"""Synthetic tests for KFS/Fernald Monte Carlo inversion."""

from __future__ import annotations

import numpy as np

from milgrau.physics.atmosphere import get_standard_atmosphere
from milgrau.physics.kfs import kfs_inversion_monte_carlo
from milgrau.physics.molecular import calculate_molecular_profile, find_optimal_reference_altitude


def test_kfs_returns_near_zero_aerosol_for_molecular_signal() -> None:
    """A pure molecular synthetic profile should retrieve near-zero aerosol."""
    altitude_m = np.arange(0.0, 10000.0, 30.0)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    beta_mol, _ = calculate_molecular_profile(temperature_k, pressure_hpa, 532.0)

    rcs = beta_mol * 1.0e12
    rcs_error = np.full_like(rcs, 0.0)
    ref_idx = find_optimal_reference_altitude(
        rcs=rcs,
        beta_mol=beta_mol,
        altitude=altitude_m,
        min_alt=6000.0,
        max_alt=9000.0,
        window_size=20,
        altitude_units="m",
    )

    beta_mean, beta_std, alpha_mean, alpha_std = kfs_inversion_monte_carlo(
        rcs=rcs,
        altitude=altitude_m,
        beta_mol=beta_mol,
        lr_base=60.0,
        lr_std=1.0,
        ref_idx=ref_idx,
        n_iterations=30,
        rcs_error=rcs_error,
        beta_ref_relative_std=0.0,
        aerosol_ref_fraction=0.0,
        altitude_units="m",
        min_lidar_ratio=10.0,
        allow_negative_aerosol=False,
        seed=123,
    )

    valid = np.isfinite(beta_mean[:ref_idx])
    assert valid.sum() > 10
    assert np.nanmedian(beta_mean[:ref_idx]) < np.nanmedian(beta_mol[:ref_idx]) * 0.05
    assert np.all(np.nan_to_num(beta_mean[:ref_idx], nan=0.0) >= 0.0)
    assert beta_std.shape == beta_mean.shape
    assert alpha_mean.shape == beta_mean.shape
    assert alpha_std.shape == beta_mean.shape


def test_kfs_is_reproducible_with_seed() -> None:
    """The Monte Carlo wrapper should be reproducible when a seed is provided."""
    altitude_m = np.arange(0.0, 8000.0, 60.0)
    pressure_hpa, temperature_k = get_standard_atmosphere(altitude_m)
    beta_mol, _ = calculate_molecular_profile(temperature_k, pressure_hpa, 355.0)
    rcs = beta_mol * 1.0e12
    ref_idx = len(altitude_m) - 5

    result_a = kfs_inversion_monte_carlo(
        rcs=rcs,
        altitude=altitude_m,
        beta_mol=beta_mol,
        lr_base=70.0,
        lr_std=5.0,
        ref_idx=ref_idx,
        n_iterations=20,
        beta_ref_relative_std=0.02,
        seed=99,
        altitude_units="m",
    )
    result_b = kfs_inversion_monte_carlo(
        rcs=rcs,
        altitude=altitude_m,
        beta_mol=beta_mol,
        lr_base=70.0,
        lr_std=5.0,
        ref_idx=ref_idx,
        n_iterations=20,
        beta_ref_relative_std=0.02,
        seed=99,
        altitude_units="m",
    )

    for array_a, array_b in zip(result_a, result_b):
        assert np.allclose(array_a, array_b, equal_nan=True)
