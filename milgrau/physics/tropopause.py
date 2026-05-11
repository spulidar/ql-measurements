"""Tropopause diagnostics based on radiosonde temperature profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_tropopause_heights(df_radiosonde: pd.DataFrame | None) -> tuple[float, float]:
    """Calculate Cold Point and WMO-style Lapse Rate Tropopause heights."""
    if df_radiosonde is None or df_radiosonde.empty:
        return np.nan, np.nan

    required_cols = {"height", "temperature"}
    if not required_cols.issubset(df_radiosonde.columns):
        return np.nan, np.nan

    df = (
        df_radiosonde.dropna(subset=["height", "temperature"])
        .drop_duplicates(subset=["height"], keep="first")
        .sort_values("height")
    )
    if df.empty:
        return np.nan, np.nan

    alt_m = df["height"].to_numpy(dtype=np.float64)
    temp_k = df["temperature"].to_numpy(dtype=np.float64) + 273.15
    valid_idx = np.where((alt_m > 5000.0) & np.isfinite(alt_m) & np.isfinite(temp_k))[0]
    if len(valid_idx) == 0:
        return np.nan, np.nan

    search_alt = alt_m[valid_idx]
    search_temp = temp_k[valid_idx]
    cpt_km = float(search_alt[int(np.argmin(search_temp))] / 1000.0)

    if len(search_alt) < 2 or (search_alt[-1] - search_alt[0]) < 2000.0:
        return cpt_km, np.nan

    z_grid = np.arange(search_alt[0], search_alt[-1], 100.0)
    if z_grid.size < 25:
        return cpt_km, np.nan

    t_grid = np.interp(z_grid, search_alt, search_temp)
    lrt_km = np.nan

    for i in range(len(z_grid) - 1):
        gamma = -(t_grid[i + 1] - t_grid[i]) / 0.1
        if gamma <= 2.0:
            z_i = z_grid[i]
            t_i = t_grid[i]
            window_indices = np.where((z_grid > z_i) & (z_grid <= z_i + 2000.0))[0]
            if len(window_indices) == 0:
                continue

            valid_window = True
            for j in window_indices:
                dz_window_km = (z_grid[j] - z_i) / 1000.0
                if dz_window_km <= 0.0:
                    continue
                gamma_avg = -(t_grid[j] - t_i) / dz_window_km
                if gamma_avg > 2.0:
                    valid_window = False
                    break

            if valid_window:
                lrt_km = float(z_i / 1000.0)
                break

    return cpt_km, lrt_km
