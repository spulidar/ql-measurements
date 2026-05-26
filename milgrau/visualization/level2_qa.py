"""Level 2 QA plotting helpers for LEBEAR products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter

from milgrau.visualization.quicklooks import (
    extract_datetime_strings,
    safe_error_of_mean,
    safe_time_mean,
)
from milgrau.visualization.style import add_footer_and_logos, channel_color, get_output_settings


def altitude_to_km(altitude_values: np.ndarray | xr.DataArray | list[float]) -> np.ndarray:
    """Return altitude in kilometers, accepting coordinates stored in meters or km."""
    alt = np.asarray(altitude_values, dtype=float)
    if alt.size == 0:
        return alt
    if np.nanmax(alt) > 100.0:
        return alt / 1000.0
    return alt


def format_wavelength_label(wavelength_nm: int | float | str) -> str:
    """Return a compact wavelength label such as '532 nm'."""
    return f"{int(float(wavelength_nm))} nm"


def get_wavelength_values(ds_l2: xr.Dataset) -> list[int]:
    """Return wavelength coordinate values from a Level 2 dataset."""
    if "wavelength" not in ds_l2.coords:
        return []
    values: list[int] = []
    for wavelength in ds_l2["wavelength"].values:
        try:
            values.append(int(wavelength))
        except Exception:
            continue
    return values


def _smooth_for_plot(values: np.ndarray | xr.DataArray, bins: int) -> np.ndarray:
    """Return a smoothed copy for visualization without changing saved products."""
    arr = np.asarray(values, dtype=np.float64)
    if bins <= 2 or arr.size < 5:
        return arr.copy()
    window = int(bins)
    if window % 2 == 0:
        window += 1
    window = min(window, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if window < 5:
        return arr.copy()
    finite = np.isfinite(arr)
    if finite.sum() < window:
        return arr.copy()
    fill_x = np.arange(arr.size)
    filled = arr.copy()
    filled[~finite] = np.interp(fill_x[~finite], fill_x[finite], arr[finite])
    smoothed = savgol_filter(filled, window_length=window, polyorder=min(3, window - 2), mode="interp")
    smoothed[~finite] = np.nan
    return smoothed


def _robust_positive_xlim(values: np.ndarray, default_max: float = 6.0) -> tuple[float, float]:
    """Return a positive x-limit that ignores extreme noisy outliers in plots."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, default_max
    high = float(np.nanpercentile(finite, 99.0))
    high = max(1.5, min(max(default_max, high * 1.15), 20.0))
    return 0.0, high


def _visual_scale_to_reference(
    lower_signal: np.ndarray,
    upper_signal: np.ndarray,
    altitude_km: np.ndarray,
    min_alt_km: float = 1.5,
    max_alt_km: float = 12.0,
) -> tuple[float, float, str]:
    """Scale one signal to another for diagnostic display.

    The fit is used only for visualization when operational gluing coefficients
    are unavailable.  A clean gluing region should make the two detector modes
    nearly linearly related after this transformation.
    """
    lower = np.asarray(lower_signal, dtype=np.float64)
    upper = np.asarray(upper_signal, dtype=np.float64)
    alt = np.asarray(altitude_km, dtype=np.float64)
    valid = (
        np.isfinite(lower)
        & np.isfinite(upper)
        & np.isfinite(alt)
        & (alt >= min_alt_km)
        & (alt <= max_alt_km)
        & (lower > 0.0)
        & (upper > 0.0)
    )
    if valid.sum() < 10:
        return 1.0, 0.0, "unscaled AN"
    slope, intercept = np.polyfit(lower[valid], upper[valid], 1)
    return float(slope), float(intercept), "AN scaled for display"


def add_atmospheric_boundaries(ax: Any, ds: xr.Dataset, max_alt_km: float) -> bool:
    """Add PBL and tropopause reference lines to an axis when available."""
    has_legend = False
    if "PBL_Height_km" in ds:
        try:
            pbl_km = float(ds["PBL_Height_km"].mean(skipna=True).values)
            if np.isfinite(pbl_km) and 0 < pbl_km <= max_alt_km:
                ax.axhline(pbl_km, color="crimson", linestyle="--", linewidth=1.4, label=f"Mean PBL ({pbl_km:.1f} km)")
                has_legend = True
        except Exception:
            pass

    for attr_name, label, color, linestyle in (
        ("tropopause_cpt_km", "CPT", "royalblue", ":"),
        ("tropopause_lrt_km", "LRT", "forestgreen", "-."),
    ):
        try:
            value = float(ds.attrs.get(attr_name, -999.0))
            if np.isfinite(value) and 0 < value <= max_alt_km:
                ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.4, label=f"{label} ({value:.1f} km)")
                has_legend = True
        except Exception:
            pass
    return has_legend


def infer_l1_channels_for_wavelength(ds_l1: xr.Dataset | None, wavelength_nm: int | float) -> tuple[str | None, str | None]:
    """Infer Analog and Photon Counting channel names for a wavelength."""
    if ds_l1 is None or "channel" not in ds_l1.coords:
        return None, None
    wavelength = str(int(wavelength_nm))
    channels = [str(channel) for channel in ds_l1["channel"].values]
    analog = next((ch for ch in channels if ch.startswith(f"{wavelength}.") and ch.upper().endswith(".AN")), None)
    photon = next((ch for ch in channels if ch.startswith(f"{wavelength}.") and (ch.upper().endswith(".PC") or ch.upper().endswith(".PH"))), None)
    return analog, photon


def plot_qa_gluing(
    ds_l1: xr.Dataset | None,
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot QA diagnostics for Analog/Photon Counting signal gluing."""
    wavelength = int(wavelength_nm)
    required = {"glued_range_corrected_signal", "gluing_success_flag", "gluing_split_altitude_m"}
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    alt_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(15.0, float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km
    color = channel_color(wavelength)
    smooth_bins = int(config.get("visualization", {}).get("level2_qa", {}).get("smooth_bins", 15))

    glued = ds_l2["glued_range_corrected_signal"].sel(wavelength=wavelength)
    glued_profile = _smooth_for_plot(safe_time_mean(glued).values, smooth_bins)
    split_alt_km = ds_l2["gluing_split_altitude_m"].sel(wavelength=wavelength) / 1000.0
    success = ds_l2["gluing_success_flag"].sel(wavelength=wavelength)
    success_values = np.asarray(success.values, dtype=float)
    success_rate = 100.0 * float(np.nansum(success_values)) / max(success.size, 1)

    fig = plt.figure(figsize=(13.5, 8.0))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 0.95], wspace=0.25)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    scaling_note = "L1 channels unavailable"
    if ds_l1 is not None:
        analog_ch, photon_ch = infer_l1_channels_for_wavelength(ds_l1, wavelength)
        if analog_ch is not None and photon_ch is not None and "range_corrected_signal" in ds_l1:
            analog_profile_raw = safe_time_mean(ds_l1["range_corrected_signal"].sel(channel=analog_ch)).values
            photon_profile = _smooth_for_plot(safe_time_mean(ds_l1["range_corrected_signal"].sel(channel=photon_ch)).values, smooth_bins)

            valid_success = np.isfinite(success_values) & (success_values == 1)
            if bool(valid_success.any()) and {"gluing_slope", "gluing_intercept"}.issubset(set(ds_l2.data_vars)):
                slope_values = ds_l2["gluing_slope"].sel(wavelength=wavelength).values
                intercept_values = ds_l2["gluing_intercept"].sel(wavelength=wavelength).values
                slope = float(np.nanmedian(np.asarray(slope_values)[valid_success]))
                intercept = float(np.nanmedian(np.asarray(intercept_values)[valid_success]))
                scaling_note = "AN scaled with median operational gluing coefficients"
            else:
                slope, intercept, scaling_note = _visual_scale_to_reference(analog_profile_raw, photon_profile, alt_km)

            scaled_analog = _smooth_for_plot(slope * analog_profile_raw + intercept, smooth_bins)
            ax0.plot(scaled_analog[valid_alt], alt_km[valid_alt], linestyle="--", linewidth=1.8, label=f"{analog_ch} scaled")
            ax0.plot(photon_profile[valid_alt], alt_km[valid_alt], linestyle=":", linewidth=2.0, label=f"{photon_ch} mean RCS")

    ax0.plot(glued_profile[valid_alt], alt_km[valid_alt], color=color, linewidth=2.4, label="Glued mean RCS")
    median_split = float(np.nanmedian(split_alt_km.values)) if np.any(np.isfinite(split_alt_km.values)) else np.nan
    if np.isfinite(median_split) and 0 < median_split <= max_alt_km:
        ax0.axhline(median_split, color="black", linestyle="-.", linewidth=1.6, label=f"Median split {median_split:.2f} km")

    if {"gluing_start_altitude_m", "gluing_stop_altitude_m"}.issubset(set(ds_l2.data_vars)):
        start_km = ds_l2["gluing_start_altitude_m"].sel(wavelength=wavelength).values / 1000.0
        stop_km = ds_l2["gluing_stop_altitude_m"].sel(wavelength=wavelength).values / 1000.0
        if np.isfinite(start_km).any() and np.isfinite(stop_km).any():
            ax0.axhspan(float(np.nanmedian(start_km)), float(np.nanmedian(stop_km)), color="gray", alpha=0.14, label="Median fade window")

    ax0.set_title(f"Scaled Gluing QA - {format_wavelength_label(wavelength)}", fontsize=14, fontweight="bold")
    ax0.set_xlabel("RCS on photon-counting scale [a.u.]", fontsize=12, fontweight="bold")
    ax0.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax0.set_ylim(0, max_alt_km)
    ax0.set_xscale("symlog", linthresh=1e-3)
    ax0.grid(True, which="both", alpha=0.45)
    ax0.legend(fontsize=9, loc="best")

    try:
        time_values = pd.to_datetime(ds_l2["time"].values)
        ax1.plot(time_values, split_alt_km.values, marker="o", linestyle="-", linewidth=1.4, markersize=3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.set_xlabel("Time (UTC)", fontsize=12, fontweight="bold")
    except Exception:
        ax1.plot(np.arange(split_alt_km.size), split_alt_km.values, marker="o", linestyle="-", linewidth=1.4, markersize=3)
        ax1.set_xlabel("Profile index", fontsize=12, fontweight="bold")

    ax1.set_title(f"Split Altitude Time Series\nSuccess rate: {success_rate:.1f}%", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Split altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max_alt_km)
    ax1.grid(True, alpha=0.45)
    if np.isfinite(median_split) and 0 < median_split <= max_alt_km:
        ax1.axhline(median_split, color="black", linestyle="-.", linewidth=1.6)
    if success_rate == 0.0:
        ax1.text(0.05, 0.95, "Operational gluing failed for all profiles.\nLeft panel uses diagnostic scaling only.\nKFS should not be interpreted as final.", transform=ax1.transAxes, va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "gray"})
    else:
        ax1.text(0.05, 0.95, scaling_note, transform=ax1.transAxes, va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "gray"})

    fig.suptitle(f"MILGRAU Level 2 QA - Signal Gluing - {format_wavelength_label(wavelength)}\n{date_title}", fontsize=15, fontweight="bold", y=0.97)
    fig.subplots_adjust(top=0.84, bottom=0.14)
    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / f"QA_Gluing_{file_name_prefix}_{wavelength}nm.{output_format}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_qa_molecular_fit(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot QA for Rayleigh molecular calibration."""
    wavelength = int(wavelength_nm)
    required = {"glued_range_corrected_signal_mean", "scaled_molecular_range_corrected_signal", "rayleigh_calibration_factor", "rayleigh_reference_altitude_m"}
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    alt_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(30.0, float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km
    smooth_bins = int(config.get("visualization", {}).get("level2_qa", {}).get("smooth_bins", 15))
    mean_glued = _smooth_for_plot(ds_l2["glued_range_corrected_signal_mean"].sel(wavelength=wavelength).values, smooth_bins)
    rayleigh_rcs = _smooth_for_plot(ds_l2["scaled_molecular_range_corrected_signal"].sel(wavelength=wavelength).values, smooth_bins)
    ref_alt_km = float(ds_l2["rayleigh_reference_altitude_m"].sel(wavelength=wavelength).values) / 1000.0
    calibration_factor = float(ds_l2["rayleigh_calibration_factor"].sel(wavelength=wavelength).values)
    calibration_intercept = float(ds_l2.get("rayleigh_calibration_intercept", xr.zeros_like(ds_l2["rayleigh_calibration_factor"])).sel(wavelength=wavelength).values)

    fig = plt.figure(figsize=(13.5, 8.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.1], wspace=0.25)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    fit_cfg = config.get("inversion", {}).get("molecular_fit", {}) or {}
    ref_min_km = float(fit_cfg.get("ref_alt_min_m", np.nan)) / 1000.0
    ref_max_km = float(fit_cfg.get("ref_alt_max_m", np.nan)) / 1000.0
    ref_window = (alt_km >= ref_min_km) & (alt_km <= ref_max_km) & np.isfinite(mean_glued) & np.isfinite(rayleigh_rcs)

    if np.any(ref_window):
        ax0.plot(rayleigh_rcs[ref_window], mean_glued[ref_window], color="royalblue", linewidth=1.4, label="Reference-region samples")
        x_fit = np.linspace(np.nanmin(rayleigh_rcs[ref_window]), np.nanmax(rayleigh_rcs[ref_window]), 100)
        ax0.plot(x_fit, x_fit + calibration_intercept, color="black", linestyle="--", linewidth=1.8, label="Linear fit diagnostic")
    ax0.set_title("Rayleigh fit region", fontsize=14, fontweight="bold")
    ax0.set_xlabel("Scaled molecular RCS [a.u.]", fontsize=12, fontweight="bold")
    ax0.set_ylabel("Measured glued RCS [a.u.]", fontsize=12, fontweight="bold")
    ax0.grid(True, alpha=0.45)
    ax0.legend(fontsize=9, loc="best")

    ax1.plot(mean_glued[valid_alt], alt_km[valid_alt], color=channel_color(wavelength), linewidth=2.2, label="Mean glued RCS")
    ax1.plot(rayleigh_rcs[valid_alt], alt_km[valid_alt], color="black", linestyle="--", linewidth=2.0, label="Scaled Rayleigh molecular RCS")
    if np.isfinite(ref_min_km) and np.isfinite(ref_max_km):
        ax1.axhspan(ref_min_km, ref_max_km, alpha=0.12, color="gray", label="Molecular-fit search range")
    if np.isfinite(ref_alt_km) and 0 < ref_alt_km <= max_alt_km:
        ax1.axhline(ref_alt_km, color="black", linestyle=":", linewidth=1.8, label=f"Reference {ref_alt_km:.2f} km")

    ax1.set_title(f"Molecular calibration profile\nslope={calibration_factor:.3g}, intercept={calibration_intercept:.3g}", fontsize=14, fontweight="bold")
    ax1.set_xlabel("RCS [a.u.]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max_alt_km)
    ax1.set_xscale("symlog", linthresh=1e-3)
    ax1.grid(True, which="both", alpha=0.45)
    add_atmospheric_boundaries(ax1, ds_l2, max_alt_km)
    ax1.legend(fontsize=9, loc="best")

    fig.suptitle(f"MILGRAU Level 2 QA - Molecular Rayleigh Fit - {format_wavelength_label(wavelength)}\n{date_title}", fontsize=15, fontweight="bold", y=0.97)
    fig.subplots_adjust(top=0.84, bottom=0.14)
    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / f"QA_Molecular_{file_name_prefix}_{wavelength}nm.{output_format}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_qa_scattering_ratio(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Plot mean scattering ratio using smoothing only for visualization."""
    wavelength = int(wavelength_nm)
    if "scattering_ratio_mean" not in ds_l2:
        return None

    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    alt_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(30.0, float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km
    smooth_bins = int(config.get("visualization", {}).get("level2_qa", {}).get("smooth_bins", 15))
    sr = _smooth_for_plot(ds_l2["scattering_ratio_mean"].sel(wavelength=wavelength).values, smooth_bins)
    color = channel_color(wavelength)

    fig, ax = plt.subplots(figsize=(8.6, 9.4))
    fig.subplots_adjust(top=0.86, bottom=0.14)
    ax.plot(sr[valid_alt], alt_km[valid_alt], color=color, linewidth=2.2, label="Scattering ratio")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.4, label="Molecular reference SR=1")
    ax.set_title(f"Scattering Ratio - {format_wavelength_label(wavelength)}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Scattering ratio", fontsize=12, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax.set_xlim(*_robust_positive_xlim(sr[valid_alt], default_max=6.0))
    ax.set_ylim(0, max_alt_km)
    ax.grid(True, alpha=0.45)
    add_atmospheric_boundaries(ax, ds_l2, max_alt_km)

    upper_mask = (alt_km >= 10.0) & valid_alt & np.isfinite(sr)
    if np.any(upper_mask):
        mean_sr = float(np.nanmean(sr[upper_mask]))
        ax.text(0.04, 0.96, f"Mean SR above 10 km = {mean_sr:.2f}\nSavgol plot smoothing = {smooth_bins} bins", transform=ax.transAxes, fontsize=10, va="top", bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "gray"})
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(f"MILGRAU Level 2 QA - Scattering Ratio - {format_wavelength_label(wavelength)}\n{date_title}", fontsize=15, fontweight="bold", y=0.97)
    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / f"QA_ScatteringRatio_{file_name_prefix}_{wavelength}nm.{output_format}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_qa_l2_kfs(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    max_altitude_km: float = 30.0,
) -> Path | None:
    """Render Level 2 KFS QA panel with Monte Carlo uncertainty bands."""
    wavelength = int(wavelength_nm)
    required = {"aerosol_backscatter", "aerosol_backscatter_error", "aerosol_extinction", "aerosol_extinction_error"}
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    alt_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(float(max_altitude_km), float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km
    smooth_bins = int(config.get("visualization", {}).get("level2_qa", {}).get("smooth_bins", 15))

    beta = ds_l2["aerosol_backscatter"].sel(wavelength=wavelength)
    beta_err = ds_l2["aerosol_backscatter_error"].sel(wavelength=wavelength)
    alpha = ds_l2["aerosol_extinction"].sel(wavelength=wavelength)
    alpha_err = ds_l2["aerosol_extinction_error"].sel(wavelength=wavelength)
    beta_mean = _smooth_for_plot(safe_time_mean(beta).values, smooth_bins)
    beta_sigma = _smooth_for_plot(safe_error_of_mean(beta_err).values, smooth_bins)
    alpha_mean = _smooth_for_plot(safe_time_mean(alpha).values, smooth_bins)
    alpha_sigma = _smooth_for_plot(safe_error_of_mean(alpha_err).values, smooth_bins)

    color = channel_color(wavelength)
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharey=ax0)
    ax0.plot(beta_mean[valid_alt] * 1e6, alt_km[valid_alt], color=color, linewidth=2.2, label="Mean beta aer")
    ax0.fill_betweenx(alt_km[valid_alt], (beta_mean[valid_alt] - beta_sigma[valid_alt]) * 1e6, (beta_mean[valid_alt] + beta_sigma[valid_alt]) * 1e6, color=color, alpha=0.25, edgecolor="none", label="MC 1σ")
    ax0.axvline(0.0, color="black", linewidth=0.8)
    ax0.set_title("Aerosol backscatter", fontsize=14, fontweight="bold")
    ax0.set_xlabel(r"$\beta_{aer}$ [Mm$^{-1}$ sr$^{-1}$]", fontsize=12, fontweight="bold")
    ax0.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax0.set_ylim(0, max_alt_km)
    ax0.grid(True, alpha=0.45)

    ax1.plot(alpha_mean[valid_alt] * 1e6, alt_km[valid_alt], color=color, linewidth=2.2, label="Mean alpha aer")
    ax1.fill_betweenx(alt_km[valid_alt], (alpha_mean[valid_alt] - alpha_sigma[valid_alt]) * 1e6, (alpha_mean[valid_alt] + alpha_sigma[valid_alt]) * 1e6, color=color, alpha=0.25, edgecolor="none", label="MC 1σ")
    ax1.axvline(0.0, color="black", linewidth=0.8)
    ax1.set_title("Aerosol extinction", fontsize=14, fontweight="bold")
    ax1.set_xlabel(r"$\alpha_{aer}$ [Mm$^{-1}$]", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.45)
    plt.setp(ax1.get_yticklabels(), visible=False)

    for ax in (ax0, ax1):
        add_atmospheric_boundaries(ax, ds_l2, max_alt_km)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(f"MILGRAU Level 2 QA - KFS Optical Retrieval - {format_wavelength_label(wavelength)}\n{date_title}", fontsize=15, fontweight="bold", y=0.97)
    fig.subplots_adjust(top=0.84, bottom=0.14)
    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / f"QA_L2_KFS_{file_name_prefix}_{wavelength}nm.{output_format}"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_all_level2_qa(
    ds_l2: xr.Dataset,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    ds_l1: xr.Dataset | None = None,
) -> list[Path]:
    """Generate available Level 2 QA plots for each wavelength."""
    generated: list[Path] = []
    qa_cfg = config.get("visualization", {}).get("level2_qa", {}) or {}
    for wavelength_nm in get_wavelength_values(ds_l2):
        if bool(qa_cfg.get("generate_gluing_qa", True)):
            gluing_path = plot_qa_gluing(ds_l1, ds_l2, wavelength_nm, output_folder, file_name_prefix, config, root_dir)
            if gluing_path is not None:
                generated.append(gluing_path)
        if bool(qa_cfg.get("generate_molecular_fit_qa", True)):
            molecular_path = plot_qa_molecular_fit(ds_l2, wavelength_nm, output_folder, file_name_prefix, config, root_dir)
            if molecular_path is not None:
                generated.append(molecular_path)
        if bool(qa_cfg.get("generate_scattering_ratio_qa", True)):
            sr_path = plot_qa_scattering_ratio(ds_l2, wavelength_nm, output_folder, file_name_prefix, config, root_dir)
            if sr_path is not None:
                generated.append(sr_path)
        if bool(qa_cfg.get("generate_kfs_qa", True)):
            kfs_path = plot_qa_l2_kfs(ds_l2, wavelength_nm, output_folder, file_name_prefix, config, root_dir)
            if kfs_path is not None:
                generated.append(kfs_path)
    return generated
