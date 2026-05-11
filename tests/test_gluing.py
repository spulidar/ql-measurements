"""Synthetic tests for analog/photon-counting signal gluing."""

from __future__ import annotations

import numpy as np

from milgrau.physics.gluing import slide_glue_signals


def test_slide_glue_signals_recovers_overlap_region() -> None:
    """Highly correlated analog and PC profiles should glue successfully."""
    altitude_m = np.arange(0.0, 12000.0, 7.5)
    base_profile = np.exp(-altitude_m / 2500.0) + 0.05

    analog = base_profile * 2.0
    photon = analog * 3.0 + 0.2

    glued, split_point, slope, intercept, diagnostics = slide_glue_signals(
        analog_sig=analog,
        pc_sig=photon,
        altitude=altitude_m,
        window_size=120,
        min_corr=0.99,
        search_min_idx=200,
        search_max_idx=1000,
        intercept_threshold=0.5,
        gaussian_threshold=0.05,
        minmax_threshold=0.2,
        return_diagnostics=True,
    )

    assert split_point > 0
    assert diagnostics["best_corr"] > 0.99
    assert np.isclose(slope, 3.0, rtol=1.0e-6, atol=1.0e-6)
    assert np.isclose(intercept, 0.2, rtol=1.0e-6, atol=1.0e-6)
    assert glued.shape == analog.shape
    assert np.allclose(glued[:split_point], photon[:split_point])


def test_slide_glue_signals_falls_back_when_uncorrelated() -> None:
    """Uncorrelated profiles should return the PC signal and split_point=-1."""
    rng = np.random.default_rng(42)
    altitude_m = np.arange(0.0, 6000.0, 7.5)
    analog = rng.normal(size=altitude_m.size)
    photon = rng.normal(size=altitude_m.size)

    glued, split_point, _, _ = slide_glue_signals(
        analog_sig=analog,
        pc_sig=photon,
        altitude=altitude_m,
        window_size=80,
        min_corr=0.99,
        search_min_idx=100,
        search_max_idx=500,
    )

    assert split_point == -1
    assert np.allclose(glued, photon)
