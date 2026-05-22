"""Klett-Fernald-Sasano optical inversion with Monte Carlo uncertainty."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numba import njit, prange

from milgrau.physics.constants import RAYLEIGH_LIDAR_RATIO_SR


def _prepare_altitude_m(altitude: np.ndarray, altitude_units: Literal["auto", "m", "km"]) -> np.ndarray:
    """Return altitude in meters."""
    altitude_arr = np.ascontiguousarray(altitude, dtype=np.float64)
    if altitude_units == "auto":
        return altitude_arr * 1000.0 if np.nanmax(altitude_arr) <= 100.0 else altitude_arr.copy()
    if altitude_units == "km":
        return altitude_arr * 1000.0
    if altitude_units == "m":
        return altitude_arr.copy()
    raise ValueError("altitude_units must be 'auto', 'm', or 'km'.")


def _nanmean_nanstd_no_warning(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return NaN-safe mean and standard deviation without empty-slice warnings.

    ``np.nanmean`` and ``np.nanstd`` intentionally warn when a full reduction
    slice is NaN. In KFS this is an expected scientific outcome for altitude
    bins where the inversion is invalid, so we return NaN quietly for those
    bins and keep finite statistics elsewhere.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    count = finite.sum(axis=axis)
    safe_values = np.where(finite, arr, 0.0)
    summed = safe_values.sum(axis=axis)

    mean = np.full(count.shape, np.nan, dtype=np.float64)
    valid = count > 0
    mean[valid] = summed[valid] / count[valid]

    expanded_mean = np.expand_dims(mean, axis=axis)
    squared = np.where(finite, (arr - expanded_mean) ** 2, 0.0)
    variance_sum = squared.sum(axis=axis)
    std = np.full(count.shape, np.nan, dtype=np.float64)
    std[valid] = np.sqrt(variance_sum[valid] / count[valid])
    return mean, std


@njit
def _fernald_single_profile(
    rcs: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    lr_aer: float,
    beta_total_ref: float,
    ref_idx: int,
    lr_mol: float,
    min_lidar_ratio: float,
    allow_negative_aerosol: bool,
    mode_code: int,
) -> np.ndarray:
    """Single-profile Fernald solution below and optionally above reference."""
    n_bins = rcs.shape[0]
    beta_aer = np.empty(n_bins, dtype=np.float64)
    for j in range(n_bins):
        beta_aer[j] = np.nan

    if not np.isfinite(lr_aer):
        return beta_aer
    if lr_aer < min_lidar_ratio:
        lr_aer = min_lidar_ratio

    beta_mol_ref = beta_mol[ref_idx]
    if (not np.isfinite(beta_total_ref)) or beta_total_ref <= 0.0:
        beta_total_ref = beta_mol_ref
    if beta_total_ref <= 0.0:
        return beta_aer

    x_ref = rcs[ref_idx]
    if (not np.isfinite(x_ref)) or x_ref <= 0.0:
        return beta_aer

    beta_aer_ref = beta_total_ref - beta_mol_ref
    if (not allow_negative_aerosol) and beta_aer_ref < 0.0:
        beta_aer_ref = 0.0
    beta_aer[ref_idx] = beta_aer_ref

    denom0 = x_ref / beta_total_ref

    # Backward branch: reference -> ground.
    mol_int = 0.0
    den_int = 0.0
    y_prev = x_ref
    for j in range(ref_idx - 1, -1, -1):
        dz = altitude_m[j + 1] - altitude_m[j]
        if (not np.isfinite(dz)) or dz <= 0.0:
            break
        x_j = rcs[j]
        if (not np.isfinite(x_j)) or x_j <= 0.0:
            x_j = 1e-30
        bm0 = beta_mol[j]
        bm1 = beta_mol[j + 1]
        if (not np.isfinite(bm0)) or (not np.isfinite(bm1)) or bm0 <= 0.0 or bm1 <= 0.0:
            break
        mol_int += 0.5 * (bm0 + bm1) * dz
        exponent = -2.0 * (lr_aer - lr_mol) * mol_int
        if exponent > 700.0:
            exponent = 700.0
        elif exponent < -700.0:
            exponent = -700.0
        y_j = x_j * np.exp(exponent)
        den_int += 0.5 * (y_j + y_prev) * dz
        denom = denom0 + 2.0 * lr_aer * den_int
        if (not np.isfinite(denom)) or denom <= 0.0:
            break
        beta_total_j = y_j / denom
        beta_aer_j = beta_total_j - bm0
        if (not allow_negative_aerosol) and beta_aer_j < 0.0:
            beta_aer_j = 0.0
        beta_aer[j] = beta_aer_j
        y_prev = y_j

    if mode_code == 0:
        return beta_aer

    # Forward branch: reference -> top. This is useful diagnostically but is
    # more noise-sensitive than the backward branch.
    mol_int = 0.0
    den_int = 0.0
    y_prev = x_ref
    for j in range(ref_idx + 1, n_bins):
        dz = altitude_m[j] - altitude_m[j - 1]
        if (not np.isfinite(dz)) or dz <= 0.0:
            break
        x_j = rcs[j]
        if (not np.isfinite(x_j)) or x_j <= 0.0:
            x_j = 1e-30
        bm0 = beta_mol[j]
        bm1 = beta_mol[j - 1]
        if (not np.isfinite(bm0)) or (not np.isfinite(bm1)) or bm0 <= 0.0 or bm1 <= 0.0:
            break
        mol_int += 0.5 * (bm0 + bm1) * dz
        exponent = -2.0 * (lr_aer - lr_mol) * mol_int
        if exponent > 700.0:
            exponent = 700.0
        elif exponent < -700.0:
            exponent = -700.0
        y_j = x_j * np.exp(exponent)
        den_int += 0.5 * (y_j + y_prev) * dz
        denom = denom0 - 2.0 * lr_aer * den_int
        if (not np.isfinite(denom)) or denom <= 0.0:
            break
        beta_total_j = y_j / denom
        beta_aer_j = beta_total_j - bm0
        if (not allow_negative_aerosol) and beta_aer_j < 0.0:
            beta_aer_j = 0.0
        beta_aer[j] = beta_aer_j
        y_prev = y_j

    return beta_aer


@njit(parallel=True)
def _kfs_fernald_mc_core(
    rcs: np.ndarray,
    rcs_error: np.ndarray,
    altitude_m: np.ndarray,
    beta_mol: np.ndarray,
    lr_samples: np.ndarray,
    beta_total_ref_samples: np.ndarray,
    rcs_noise: np.ndarray,
    ref_idx: int,
    lr_mol: float,
    min_lidar_ratio: float,
    use_rcs_noise: bool,
    allow_negative_aerosol: bool,
    mode_code: int,
):
    """Numba-compiled Monte Carlo Fernald/Klett-Sasano inversion."""
    n_iter = lr_samples.shape[0]
    n_bins = rcs.shape[0]
    beta_aer_sims = np.empty((n_iter, n_bins), dtype=np.float64)
    alpha_aer_sims = np.empty((n_iter, n_bins), dtype=np.float64)

    for i in prange(n_iter):
        rcs_i = np.empty(n_bins, dtype=np.float64)
        for j in range(n_bins):
            beta_aer_sims[i, j] = np.nan
            alpha_aer_sims[i, j] = np.nan
            if use_rcs_noise:
                rcs_i[j] = rcs[j] + rcs_error[j] * rcs_noise[i, j]
            else:
                rcs_i[j] = rcs[j]

        lr_aer = lr_samples[i]
        if not np.isfinite(lr_aer):
            continue
        if lr_aer < min_lidar_ratio:
            lr_aer = min_lidar_ratio

        beta_aer = _fernald_single_profile(
            rcs_i,
            altitude_m,
            beta_mol,
            lr_aer,
            beta_total_ref_samples[i],
            ref_idx,
            lr_mol,
            min_lidar_ratio,
            allow_negative_aerosol,
            mode_code,
        )
        for j in range(n_bins):
            beta_aer_sims[i, j] = beta_aer[j]
            if np.isfinite(beta_aer[j]):
                alpha_aer_sims[i, j] = beta_aer[j] * lr_aer

    return beta_aer_sims, alpha_aer_sims


def kfs_inversion_monte_carlo(
    rcs: np.ndarray,
    altitude: np.ndarray,
    beta_mol: np.ndarray,
    lr_base: float,
    lr_std: float = 10.0,
    ref_idx: int = -1,
    n_iterations: int = 300,
    rcs_error: np.ndarray | None = None,
    beta_ref_relative_std: float = 0.10,
    aerosol_ref_fraction: float = 0.0,
    altitude_units: Literal["auto", "m", "km"] = "auto",
    min_lidar_ratio: float = 10.0,
    allow_negative_aerosol: bool = False,
    seed: int | None = None,
    return_diagnostics: bool = False,
    mode: Literal["backward", "two_sided"] = "backward",
):
    """Run KFS/Fernald-Sasano inversion with Monte Carlo uncertainty.

    ``mode='backward'`` retrieves from the reference altitude downward.
    ``mode='two_sided'`` also performs a forward branch above the reference,
    reproducing the historical full-column behavior but with more sensitivity to
    high-altitude noise.
    """
    rcs_arr = np.ascontiguousarray(rcs, dtype=np.float64)
    beta_mol_arr = np.ascontiguousarray(beta_mol, dtype=np.float64)
    altitude_m = np.ascontiguousarray(_prepare_altitude_m(altitude, altitude_units), dtype=np.float64)

    if rcs_arr.ndim != 1:
        raise ValueError("kfs_inversion_monte_carlo expects a 1D RCS profile.")
    if beta_mol_arr.shape[0] != rcs_arr.shape[0]:
        raise ValueError("beta_mol must have the same length as rcs.")
    if altitude_m.shape[0] != rcs_arr.shape[0]:
        raise ValueError("altitude must have the same length as rcs.")

    n_bins = rcs_arr.shape[0]
    if ref_idx < 0:
        ref_idx = n_bins + int(ref_idx)
    ref_idx = int(ref_idx)
    if ref_idx <= 0 or ref_idx >= n_bins:
        raise ValueError("ref_idx must point to a valid altitude bin above ground.")

    mode_code = 0 if mode == "backward" else 1 if mode == "two_sided" else -1
    if mode_code < 0:
        raise ValueError("mode must be 'backward' or 'two_sided'.")

    use_rcs_noise = rcs_error is not None
    if use_rcs_noise:
        rcs_error_arr = np.ascontiguousarray(rcs_error, dtype=np.float64)
        if rcs_error_arr.shape[0] != n_bins:
            raise ValueError("rcs_error must have the same length as rcs.")
        rcs_error_arr = np.ascontiguousarray(np.where(np.isfinite(rcs_error_arr), rcs_error_arr, 0.0), dtype=np.float64)
    else:
        rcs_error_arr = np.zeros_like(rcs_arr, dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_iterations = int(n_iterations)
    lr_samples = np.ascontiguousarray(rng.normal(float(lr_base), float(lr_std), size=n_iterations), dtype=np.float64)

    beta_total_ref_mean = beta_mol_arr[ref_idx] * (1.0 + float(aerosol_ref_fraction))
    beta_total_ref_samples = np.ascontiguousarray(
        rng.normal(
            beta_total_ref_mean,
            abs(beta_total_ref_mean) * float(beta_ref_relative_std),
            size=n_iterations,
        ),
        dtype=np.float64,
    )

    if use_rcs_noise:
        rcs_noise = np.ascontiguousarray(rng.standard_normal((n_iterations, n_bins)), dtype=np.float64)
    else:
        rcs_noise = np.empty((1, 1), dtype=np.float64)

    beta_sims, alpha_sims = _kfs_fernald_mc_core(
        rcs_arr,
        rcs_error_arr,
        altitude_m,
        beta_mol_arr,
        lr_samples,
        beta_total_ref_samples,
        rcs_noise,
        ref_idx,
        RAYLEIGH_LIDAR_RATIO_SR,
        float(min_lidar_ratio),
        bool(use_rcs_noise),
        bool(allow_negative_aerosol),
        mode_code,
    )

    beta_mean, beta_std = _nanmean_nanstd_no_warning(beta_sims, axis=0)
    alpha_mean, alpha_std = _nanmean_nanstd_no_warning(alpha_sims, axis=0)

    if return_diagnostics:
        diagnostics = {
            "beta_aer_sims": beta_sims,
            "alpha_aer_sims": alpha_sims,
            "lr_samples": lr_samples,
            "beta_total_ref_samples": beta_total_ref_samples,
            "ref_idx": int(ref_idx),
            "altitude_m": altitude_m,
            "used_rcs_noise": bool(use_rcs_noise),
            "mode": mode,
        }
        return beta_mean, beta_std, alpha_mean, alpha_std, diagnostics
    return beta_mean, beta_std, alpha_mean, alpha_std
