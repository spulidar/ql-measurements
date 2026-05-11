"""Level 1 instrumental correction kernels for lidar signals."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _safe_nanmax_xarray(data: xr.DataArray, default: float = 0.0) -> float:
    """Safely compute a finite maximum from an xarray object."""
    try:
        value = float(data.max(skipna=True).values)
        return value if np.isfinite(value) else float(default)
    except Exception:
        return float(default)


def apply_instrumental_corrections(
    sig: xr.DataArray,
    z_da: xr.DataArray,
    shots: float,
    bin_time_us: float,
    deadtime: float,
    shift: int,
    bg_offset: float,
    is_photon: bool,
    bg_mask: xr.DataArray,
    dc_prof: xr.DataArray | None = None,
    dc_err: xr.DataArray | None = None,
    deadtime_min_denominator: float = 0.05,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Apply Level 1 corrections and propagate one-sigma uncertainty."""
    if shots is None or not np.isfinite(float(shots)) or float(shots) <= 0.0:
        raise ValueError(f"Invalid laser shots value: {shots}")
    if bin_time_us is None or not np.isfinite(float(bin_time_us)) or float(bin_time_us) <= 0.0:
        raise ValueError(f"Invalid bin_time_us value: {bin_time_us}")

    shots = float(shots)
    bin_time_us = float(bin_time_us)
    deadtime = float(deadtime)
    shift = int(shift)
    bg_offset = float(bg_offset)
    deadtime_min_denominator = float(deadtime_min_denominator)

    sig_dc = sig.copy()
    err_dc = xr.zeros_like(sig)
    if dc_prof is not None:
        sig_dc = sig_dc - dc_prof
        if dc_err is not None:
            err_dc = dc_err

    if not is_photon:
        sig_dt = sig_dc.copy()
        if _safe_nanmax_xarray(sig_dt) > 1000.0:
            sig_dt = sig_dt / (shots * bin_time_us)
        err_bg = sig_dt.where(bg_mask).std(dim="range", skipna=True)
        err_dt = xr.ones_like(sig_dt) * err_bg
        if dc_prof is not None and dc_err is not None:
            err_dt = np.sqrt(err_dt**2 + err_dc**2)
    else:
        sig_mhz = sig_dc.copy()
        if _safe_nanmax_xarray(sig_mhz) > 150.0:
            sig_mhz = sig_mhz / (shots * bin_time_us)
        n_photons = xr.where(sig_mhz * shots * bin_time_us > 0.0, sig_mhz * shots * bin_time_us, 0.0)
        err_raw = np.sqrt(n_photons) / (shots * bin_time_us)
        if dc_prof is not None and dc_err is not None:
            err_raw = np.sqrt(err_raw**2 + err_dc**2)
        if deadtime > 0.0:
            denom = 1.0 - (sig_mhz * deadtime)
            safe_denom = xr.where(denom < deadtime_min_denominator, deadtime_min_denominator, denom)
            sig_dt = sig_mhz / safe_denom
            err_dt = err_raw / (safe_denom**2)
        else:
            sig_dt, err_dt = sig_mhz, err_raw

    max_sig_val = _safe_nanmax_xarray(sig_dt, default=0.0)
    if shift > 0:
        sig_shift = sig_dt.shift(range=shift, fill_value=max_sig_val)
    elif shift < 0:
        sig_shift = sig_dt.shift(range=shift, fill_value=0.0)
    else:
        sig_shift = sig_dt.copy()
    err_shift = err_dt.shift(range=shift, fill_value=0.0)

    bg_mean = sig_shift.where(bg_mask).mean(dim="range", skipna=True) - bg_offset
    n_bg = int(bg_mask.sum().values) if hasattr(bg_mask.sum(), "values") else int(bg_mask.sum())
    n_bg = max(n_bg, 1)
    err_bg_mean = sig_shift.where(bg_mask).std(dim="range", skipna=True) / np.sqrt(n_bg)

    sig_c = sig_shift - bg_mean
    err_c = np.sqrt(err_shift**2 + err_bg_mean**2)
    rcs = sig_c * (z_da**2)
    err_rcs = err_c * (z_da**2)
    return sig_c, err_c, rcs, err_rcs
