"""Tests for preliminary cloud/anomalous-layer screening utilities."""

from __future__ import annotations

import numpy as np

from milgrau.physics.cloud_screening import (
    cloud_screening_config,
    detect_anomalous_layer_mask,
    detect_reference_contamination,
)


def test_detect_anomalous_layer_mask_flags_positive_spike_layer() -> None:
    """A strong multi-bin positive anomaly should be flagged and dilated."""
    altitude = np.arange(0.0, 3000.0, 7.5)
    signal = np.exp(-altitude / 1200.0) + 0.1
    layer = (altitude >= 1200.0) & (altitude <= 1260.0)
    signal[layer] += 20.0

    mask = detect_anomalous_layer_mask(
        signal,
        altitude,
        min_altitude_m=500.0,
        max_altitude_m=2500.0,
        smooth_bins=11,
        robust_z_threshold=5.0,
        min_cloud_bins=3,
        vertical_dilation_bins=2,
    )

    assert mask.dtype == bool
    assert mask[layer].any()
    assert mask.sum() >= layer.sum()


def test_detect_anomalous_layer_mask_ignores_below_min_altitude() -> None:
    """A strong anomaly below the configured search range should not be flagged."""
    altitude = np.arange(0.0, 3000.0, 7.5)
    signal = np.exp(-altitude / 1200.0) + 0.1
    low_layer = (altitude >= 150.0) & (altitude <= 220.0)
    signal[low_layer] += 50.0

    mask = detect_anomalous_layer_mask(signal, altitude, min_altitude_m=500.0, max_altitude_m=2500.0)

    assert not mask[low_layer].any()


def test_detect_reference_contamination_returns_fraction() -> None:
    """Reference contamination should be measured as flagged-bin fraction."""
    altitude = np.arange(0.0, 1000.0, 100.0)
    mask = np.zeros_like(altitude, dtype=bool)
    mask[(altitude >= 300.0) & (altitude <= 500.0)] = True

    fraction = detect_reference_contamination(mask, altitude, 300.0, 700.0)

    assert np.isclose(fraction, 3.0 / 5.0)


def test_cloud_screening_config_uses_defaults_and_yaml_values() -> None:
    """Cloud-screening config helper should expose conservative defaults and overrides."""
    default = cloud_screening_config({})
    assert default["enabled"] is False
    assert default["exclude_clouds_from_reference_fit"] is True

    config = {
        "inversion": {
            "cloud_screening": {
                "enabled": True,
                "robust_z_threshold": 4.5,
                "vertical_dilation_bins": 4,
            }
        }
    }
    extracted = cloud_screening_config(config)
    assert extracted["enabled"] is True
    assert extracted["robust_z_threshold"] == 4.5
    assert extracted["vertical_dilation_bins"] == 4
