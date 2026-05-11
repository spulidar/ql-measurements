"""Synthetic tests for PBL gradient diagnostics."""

from __future__ import annotations

import numpy as np

from milgrau.physics.pbl import calculate_pbl_height_gradient


def test_pbl_gradient_detects_synthetic_aerosol_drop() -> None:
    """A sharp negative RCS gradient should be detected as PBL height."""
    altitude_m = np.arange(0.0, 5000.0, 7.5)
    pbl_true_m = 1500.0

    # Smooth synthetic aerosol-rich boundary layer with a transition near 1.5 km.
    transition = 1.0 / (1.0 + np.exp((altitude_m - pbl_true_m) / 80.0))
    rcs_signal = 0.2 + 5.0 * transition

    pbl_km = calculate_pbl_height_gradient(
        rcs_signal=rcs_signal,
        alt_m=altitude_m,
        min_search_m=500.0,
        max_search_m=3000.0,
        smooth_bins=15,
    )

    assert np.isfinite(pbl_km)
    assert abs((pbl_km * 1000.0) - pbl_true_m) < 150.0


def test_pbl_gradient_returns_nan_for_increasing_profile() -> None:
    """Profiles without a negative drop should not return a false PBL."""
    altitude_m = np.arange(0.0, 5000.0, 7.5)
    rcs_signal = 1.0 + altitude_m / altitude_m.max()

    pbl_km = calculate_pbl_height_gradient(
        rcs_signal=rcs_signal,
        alt_m=altitude_m,
        min_search_m=500.0,
        max_search_m=3000.0,
        smooth_bins=15,
    )

    assert np.isnan(pbl_km)
