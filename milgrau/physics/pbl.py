"""Planetary Boundary Layer diagnostics for lidar RCS profiles."""

from __future__ import annotations

import numpy as np


def calculate_pbl_height_gradient(
    rcs_signal: np.ndarray,
    alt_m: np.ndarray,
    min_search_m: float = 500.0,
    max_search_m: float = 4000.0,
    smooth_bins: int = 15,
) -> float:
    """Estimate PBL height with the strongest negative RCS gradient method."""
    rcs_signal = np.asarray(rcs_signal, dtype=np.float64)
    alt_m = np.asarray(alt_m, dtype=np.float64)

    if rcs_signal.ndim != 1 or alt_m.ndim != 1 or rcs_signal.size != alt_m.size:
        return np.nan

    smooth_bins = max(int(smooth_bins), 3)
    if smooth_bins % 2 == 0:
        smooth_bins += 1

    valid_idx = np.where(
        (alt_m >= float(min_search_m))
        & (alt_m <= float(max_search_m))
        & np.isfinite(alt_m)
        & np.isfinite(rcs_signal)
    )[0]
    if len(valid_idx) < smooth_bins:
        return np.nan

    search_alt = alt_m[valid_idx]
    search_rcs = rcs_signal[valid_idx]
    finite = np.isfinite(search_rcs)
    if finite.sum() < smooth_bins:
        return np.nan

    median_val = np.nanmedian(search_rcs[finite])
    search_rcs = np.where(np.isfinite(search_rcs), search_rcs, median_val)
    window = np.ones(smooth_bins, dtype=np.float64) / smooth_bins
    smoothed_rcs = np.convolve(search_rcs, window, mode="same")
    gradient = np.gradient(smoothed_rcs, search_alt)

    edge_trim = smooth_bins // 2
    if len(gradient) > 2 * edge_trim:
        min_grad_idx = int(np.argmin(gradient[edge_trim:-edge_trim])) + edge_trim
    else:
        min_grad_idx = int(np.argmin(gradient))

    if not np.isfinite(gradient[min_grad_idx]) or gradient[min_grad_idx] >= 0.0:
        return np.nan
    return float(search_alt[min_grad_idx] / 1000.0)
