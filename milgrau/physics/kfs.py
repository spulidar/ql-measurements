"""Klett-Fernald-Sasano optical inversion with Monte Carlo uncertainty."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numba import njit, prange

from milgrau.physics.constants import RAYLEIGH_LIDAR_RATIO_SR


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
):
    """Numba-compiled Monte Carlo Fernald/Klett-Sasano backward inversion."""
    n_iter = lr_samples.shape[0]
    n_bins = rcs.shape[0]
    beta_aer_sims = np.empty((n_iter, n_bins), dtype=np.float64)
    alpha_aer_sims = np.empty((n_iter, n_bins), dtype=np.float64)

    for i in prange(n_iter):
        for j in range(n_bins):
            beta_aer_sims[i, j] = np.nan
            alpha_aer_sims[i, j] = np.nan

        lr_aer = lr_samples[i]
        if not np.isfinite(lr_aer):
            continue
        if lr_aer < min_lidar_ratio:
            lr_aer = min_lidar_ratio

        beta_total_ref = beta_total_ref_samples[i]
        beta_mol_ref = beta_mol[ref_idx]
        if (not np.isfinite(beta_total_ref)) or beta_total_ref <= 0.0:
            beta_total_ref = beta_mol_ref
        if beta_total_ref <= 0.0:
            continue

        x_ref = rcs[ref_idx]
        if use_rcs_noise:
            x_ref = x_ref + rcs_error[ref_idx] * rcs_noise[i, ref_idx]
        if (not np.isfinite(x_ref)) or x_ref <= 0.0:
            continue

        beta_aer_ref = beta_total_ref - beta_mol_ref
        if (not allow_negative_aerosol) and beta_aer_ref < 0.0:
            beta_aer_ref = 0.0

        beta_aer_sims[i, ref_idx] = beta_aer_ref
        alpha_aer_sims[i, ref_idx] = lr_aer * beta_aer_ref

        denom0 = x_ref / beta_total_ref
        mol_int = 0.0
        den_int = 0.0
        y_prev = x_ref

        for j in range(ref_idx - 1, -1, -1):
            dz = altitude_m[j + 1] - altitude_m[j]
            if (not np.isfinite(dz)) or dz <= 0.0:
                break

            x_j = rcs[j]
            if use_rcs_noise:
                x_j = x_j + rcs_error[j] * rcs_noise[i, j]
            if (not np.isfinite(x_j)) or x_j <= 0.0:
                x_j = 1e-30

            bm0 = beta_mol[j]
            bm1 = beta_mol[j + 1]
            if (
                (not np.isfinite(bm0))
                or (not np.isfinite(bm1))
                or bm0 <= 0.0
                or bm1 <= 0.0
            ):
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

            beta_aer_sims[i, j] = beta_aer_j
            alpha_aer_sims[i, j] = lr_aer * beta_aer_j
            y_prev = y_j

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
):
    """Run KFS/Fernald-Sasano inversion with Monte Carlo uncertainty."""
    rcs_arr = np.ascontiguousarray(rcs, dtype=np.float64)
    beta_mol_arr = np.ascontiguousarray(beta_mol, dtype=np.float64)
    altitude_arr = np.ascontiguousarray(altitude, dtype=np.float64)

    if rcs_arr.ndim != 1:
        raise ValueError("kfs_inversion_monte_carlo expects a 1D RCS profile.")
    if beta_mol_arr.shape[0] != rcs_arr.shape[0]:
        raise ValueError("beta_mol must have the same length as rcs.")
    if altitude_arr.shape[0] != rcs_arr.shape[0]:
        raise ValueError("altitude must have the same length as rcs.")

    if altitude_units == "auto":
        altitude_m = altitude_arr * 1000.0 if np.nanmax(altitude_arr) <= 100.0 else altitude_arr.copy()
    elif altitude_units == "km":
        altitude_m = altitude_arr * 1000.0
    elif altitude_units == "m":
        altitude_m = altitude_arr.copy()
    else:
        raise ValueError("altitude_units must be 'auto', 'm', or 'km'.")

    altitude_m = np.ascontiguousarray(altitude_m, dtype=np.float64)
    n_bins = rcs_arr.shape[0]
    if ref_idx < 0:
        ref_idx = n_bins + int(ref_idx)
    ref_idx = int(ref_idx)
    if ref_idx <= 0 or ref_idx >= n_bins:
        raise ValueError("ref_idx must point to a valid altitude bin above ground.")

    use_rcs_noise = rcs_error is not None
    if use_rcs_noise:
        rcs_error_arr = np.ascontiguousarray(rcs_error, dtype=np.float64)
        if rcs_error_arr.shape[0] != n_bins:
            raise ValueError("rcs_error must have the same length as rcs.")
        rcs_error_arr = np.ascontiguousarray(
            np.where(np.isfinite(rcs_error_arr), rcs_error_arr, 0.0), dtype=np.float64
        )
    else:
        rcs_error_arr = np.zeros_like(rcs_arr, dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_iterations = int(n_iterations)
    lr_samples = np.ascontiguousarray(
        rng.normal(float(lr_base), float(lr_std), size=n_iterations), dtype=np.float64
    )

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
    )

    with np.errstate(all="ignore"):
        beta_mean = np.nanmean(beta_sims, axis=0)
        beta_std = np.nanstd(beta_sims, axis=0)
        alpha_mean = np.nanmean(alpha_sims, axis=0)
        alpha_std = np.nanstd(alpha_sims, axis=0)

    if return_diagnostics:
        diagnostics = {
            "beta_aer_sims": beta_sims,
            "alpha_aer_sims": alpha_sims,
            "lr_samples": lr_samples,
            "beta_total_ref_samples": beta_total_ref_samples,
            "ref_idx": int(ref_idx),
            "altitude_m": altitude_m,
            "used_rcs_noise": bool(use_rcs_noise),
        }
        return beta_mean, beta_std, alpha_mean, alpha_std, diagnostics
    return beta_mean, beta_std, alpha_mean, alpha_std
