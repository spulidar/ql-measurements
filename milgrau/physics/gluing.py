"""Analog/photon-counting signal gluing utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit


@njit(fastmath=True)
def _window_stats(values: np.ndarray, start: int, stop: int):
    """Numba helper returning mean, std, min, max and finite-status."""
    n = 0
    s = 0.0
    mn = 1e308
    mx = -1e308
    for k in range(start, stop):
        v = values[k]
        if not np.isfinite(v):
            return 0.0, 0.0, 0.0, 0.0, False
        s += v
        n += 1
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, False
    mean = s / n
    var = 0.0
    for k in range(start, stop):
        d = values[k] - mean
        var += d * d
    var /= n
    return mean, np.sqrt(var), mn, mx, True


@njit(fastmath=True)
def _slide_glue_core(
    an_vals: np.ndarray,
    pc_vals: np.ndarray,
    window_size: int,
    min_corr: float,
    search_min_idx: int,
    search_max_idx: int,
    intercept_threshold: float,
    gaussian_threshold: float,
    minmax_threshold: float,
):
    """Compiled core for analog/photon-counting gluing window selection."""
    n = len(an_vals)
    best_idx = -1
    best_corr = -1.0
    best_slope = 1.0
    best_intercept = 0.0
    if window_size < 3 or n < window_size:
        return best_idx, best_corr, best_slope, best_intercept

    start_search = max(0, search_min_idx)
    end_search = min(n - window_size, search_max_idx)
    if end_search <= start_search:
        start_search = 0
        end_search = n - window_size

    eps = 1e-12
    for i in range(start_search, end_search + 1):
        an_mean, an_std, an_min, an_max, ok_an = _window_stats(an_vals, i, i + window_size)
        pc_mean, pc_std, pc_min, pc_max, ok_pc = _window_stats(pc_vals, i, i + window_size)
        if not ok_an or not ok_pc:
            continue
        if an_std <= eps or pc_std <= eps:
            continue

        cov = 0.0
        for k in range(i, i + window_size):
            cov += (an_vals[k] - an_mean) * (pc_vals[k] - pc_mean)
        cov /= window_size
        corr = cov / (an_std * pc_std)
        if corr < min_corr:
            continue

        slope = cov / (an_std * an_std)
        intercept = pc_mean - slope * an_mean
        rel_intercept = abs(intercept) / (abs(pc_mean) + eps)
        if rel_intercept > intercept_threshold:
            continue

        residual_var = 0.0
        for k in range(i, i + window_size):
            residual = (slope * an_vals[k] + intercept) - pc_vals[k]
            residual_var += residual * residual
        gaussian_score = np.sqrt(residual_var / window_size) / (pc_std + eps)
        if gaussian_score > gaussian_threshold:
            continue

        scaled_min = slope * an_min + intercept
        scaled_max = slope * an_max + intercept
        if scaled_min > scaled_max:
            tmp = scaled_min
            scaled_min = scaled_max
            scaled_max = tmp
        pc_range = abs(pc_max - pc_min) + eps
        minmax_score = max(abs(scaled_min - pc_min), abs(scaled_max - pc_max)) / pc_range
        if minmax_score > minmax_threshold:
            continue

        if corr > best_corr:
            best_corr = corr
            best_idx = i
            best_slope = slope
            best_intercept = intercept

    return best_idx, best_corr, best_slope, best_intercept


def slide_glue_signals(
    analog_sig: np.ndarray,
    pc_sig: np.ndarray,
    altitude: np.ndarray | None = None,
    window_size: int = 150,
    min_corr: float = 0.90,
    search_min_idx: int = 0,
    search_max_idx: int | None = None,
    intercept_threshold: float = 0.5,
    gaussian_threshold: float = 0.1,
    minmax_threshold: float = 0.5,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, int, float, float] | tuple[np.ndarray, int, float, float, dict[str, Any]]:
    """Glue analog and photon-counting signals into one dynamic-range profile."""
    an_vals = np.ascontiguousarray(analog_sig, dtype=np.float64)
    pc_vals = np.ascontiguousarray(pc_sig, dtype=np.float64)
    if an_vals.ndim != 1 or pc_vals.ndim != 1 or an_vals.size != pc_vals.size:
        raise ValueError("analog_sig and pc_sig must be 1D arrays with the same length.")

    n = an_vals.size
    window_size = int(window_size)
    search_min_idx = int(search_min_idx)
    search_max_idx = int(n - window_size if search_max_idx is None else search_max_idx)
    best_idx, best_corr, best_slope, best_intercept = _slide_glue_core(
        an_vals,
        pc_vals,
        window_size,
        float(min_corr),
        search_min_idx,
        search_max_idx,
        float(intercept_threshold),
        float(gaussian_threshold),
        float(minmax_threshold),
    )

    if best_idx == -1:
        glued_sig = pc_vals.copy()
        split_point = -1
    else:
        split_point = int(best_idx + (window_size // 2))
        glued_sig = pc_vals.copy()
        glued_sig[:split_point] = (an_vals[:split_point] * best_slope) + best_intercept

    if return_diagnostics:
        diagnostics = {
            "best_idx": int(best_idx),
            "best_corr": float(best_corr),
            "split_point": int(split_point),
            "search_min_idx": int(search_min_idx),
            "search_max_idx": int(search_max_idx),
            "window_size": int(window_size),
        }
        if altitude is not None and split_point >= 0:
            altitude_arr = np.asarray(altitude, dtype=np.float64)
            if altitude_arr.ndim == 1 and split_point < altitude_arr.size:
                diagnostics["split_altitude"] = float(altitude_arr[split_point])
        return glued_sig, split_point, float(best_slope), float(best_intercept), diagnostics

    return glued_sig, split_point, float(best_slope), float(best_intercept)
