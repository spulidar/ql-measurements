"""
MILGRAU - Visualization Utilities

Plotting helpers for MILGRAU Level 1 and Level 2 products. This module
centralizes visual identity, metadata formatting, institutional footer/logo
placement, Level 1 RCS quicklooks, global mean profiles, and Level 2 QA panels.

@author: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


WAVELENGTH_COLORS: dict[int, str] = {
    355: "rebeccapurple",
    387: "darkblue",
    408: "darkcyan",
    530: "orange",
    532: "forestgreen",
    1064: "crimson",
}

DEFAULT_LOGO_SPECS: tuple[tuple[str, float], ...] = (
    ("CC_BY-NC-ND.png", 0.040),
    ("lalinet_logo2.png", 0.070),
    ("logo_leal2.png", 0.065),
)


@lru_cache(maxsize=16)
def _read_logo_image(path_str: str) -> np.ndarray:
    """Read and cache a logo image from disk."""
    return mpimg.imread(path_str)


def _get_output_settings(config: dict[str, Any]) -> tuple[str, int]:
    """Extract output format and DPI from the visualization configuration."""
    viz_cfg = config.get("visualization", {}) or {}
    output_format = str(viz_cfg.get("output_format", "webp")).lstrip(".").lower()
    dpi = int(viz_cfg.get("dpi", 120))
    return output_format, dpi


def _save_figure(fig: Any, out_path: str | Path, dpi: int) -> Path:
    """Save and close a Matplotlib figure using a single safe helper."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def channel_color(channel_or_wavelength: str | int | float) -> str:
    """Return the MILGRAU display color associated with a wavelength."""
    try:
        if isinstance(channel_or_wavelength, str):
            wavelength = int(channel_or_wavelength.split(".")[0])
        else:
            wavelength = int(channel_or_wavelength)
    except Exception:
        return "black"
    return WAVELENGTH_COLORS.get(wavelength, "black")


def extract_datetime_strings(ds: xr.Dataset) -> tuple[str, str]:
    """Extract human-readable date strings from an xarray Dataset."""
    try:
        dt_in = pd.to_datetime(ds.time.values.min())
        dt_end = pd.to_datetime(ds.time.values.max())
        date_title = f"{dt_in.strftime('%d %b %Y - %H:%M')} to {dt_end.strftime('%H:%M')} UTC"
        date_footer = dt_in.strftime("%d %b %Y")
        return date_title, date_footer
    except Exception:
        return "Unknown date", "Unknown date"


def format_channel_name(raw_name: str) -> str:
    """Convert an internal channel name such as '532.AN' into '532nm AN'."""
    try:
        parts = str(raw_name).split(".")
        wavelength = int(parts[0])
        mode = str(parts[1])
        return f"{wavelength}nm {mode}"
    except Exception:
        return str(raw_name)


def format_wavelength_label(wavelength_nm: int | float | str) -> str:
    """Return a compact wavelength label such as '532 nm'."""
    return f"{int(float(wavelength_nm))} nm"


def altitude_to_km(altitude_values: np.ndarray | xr.DataArray | list[float]) -> np.ndarray:
    """Return altitude in kilometers, accepting coordinates stored in meters or km."""
    alt = np.asarray(altitude_values, dtype=float)
    if alt.size == 0:
        return alt
    if np.nanmax(alt) > 100.0:
        return alt / 1000.0
    return alt


def add_footer_and_logos(fig: Any, root_dir: str | Path) -> None:
    """Add MILGRAU/SPU institutional footer and logos to a figure."""
    fig.text(
        0.08,
        0.04,
        "SPU Lidar Station - São Paulo",
        fontsize=13,
        fontweight="bold",
        color="#333333",
        va="center",
    )

    root_path = Path(root_dir)
    spacing = 0.010
    y_pos = 0.01
    x_right = 0.98

    for logo_name, height in DEFAULT_LOGO_SPECS:
        logo_path = root_path / "img" / logo_name
        if not logo_path.exists():
            continue

        img = _read_logo_image(str(logo_path))
        img_h, img_w = img.shape[:2]
        width = height * (img_w / img_h)
        x_left = x_right - width

        ax_logo = fig.add_axes([x_left, y_pos, width, height], zorder=12)
        ax_logo.imshow(img)
        ax_logo.axis("off")
        x_right = x_left - spacing


def safe_time_mean(da: xr.DataArray) -> xr.DataArray:
    """Return the time mean of a DataArray when a time dimension is present."""
    if "time" in da.dims:
        return da.mean(dim="time", skipna=True)
    return da


def safe_error_of_mean(err_da: xr.DataArray) -> xr.DataArray:
    """
    Combine profile 1-sigma errors over time as uncertainty of the mean.

    This assumes independent profile errors and uses sqrt(sum(sigma^2)) / N.
    """
    if "time" not in err_da.dims:
        return err_da
    n_profiles = max(int(err_da.sizes.get("time", 1)), 1)
    return np.sqrt((err_da**2).sum(dim="time", skipna=True)) / n_profiles


def rolling_altitude(da: xr.DataArray, bins: int = 15) -> xr.DataArray:
    """Apply centered rolling smoothing along altitude when possible."""
    if "altitude" not in da.dims:
        return da
    try:
        return da.rolling(altitude=int(bins), min_periods=1, center=True).mean()
    except Exception:
        return da


def infer_l1_channels_for_wavelength(
    ds_l1: xr.Dataset | None,
    wavelength_nm: int | float,
) -> tuple[str | None, str | None]:
    """Infer analog and photon-counting channel names for one wavelength."""
    if ds_l1 is None or "channel" not in ds_l1.coords:
        return None, None

    wl = str(int(wavelength_nm))
    channels = [str(channel) for channel in ds_l1["channel"].values]
    analog = next(
        (channel for channel in channels if channel.startswith(f"{wl}.") and channel.upper().endswith(".AN")),
        None,
    )
    photon = next(
        (
            channel
            for channel in channels
            if channel.startswith(f"{wl}.")
            and (channel.upper().endswith(".PC") or channel.upper().endswith(".PH"))
        ),
        None,
    )
    return analog, photon


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


def add_atmospheric_boundaries(ax: Any, ds: xr.Dataset, max_alt_km: float) -> bool:
    """Add PBL and tropopause reference lines to an axis when available."""
    has_legend = False

    if "PBL_Height_km" in ds:
        try:
            pbl_km = float(ds["PBL_Height_km"].mean(skipna=True).values)
            if np.isfinite(pbl_km) and 0 < pbl_km <= max_alt_km:
                ax.axhline(
                    pbl_km,
                    color="crimson",
                    linestyle="--",
                    linewidth=1.4,
                    label=f"Mean PBL ({pbl_km:.1f} km)",
                )
                has_legend = True
        except Exception:
            pass

    boundary_specs = (
        ("tropopause_cpt_km", "CPT", "royalblue", ":"),
        ("tropopause_lrt_km", "LRT", "forestgreen", "-."),
    )
    for attr_name, label, color, linestyle in boundary_specs:
        try:
            value = float(ds.attrs.get(attr_name, -999.0))
            if np.isfinite(value) and 0 < value <= max_alt_km:
                ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.4, label=f"{label} ({value:.1f} km)")
                has_legend = True
        except Exception:
            pass
    return has_legend


def plot_quicklook(
    data_slice: xr.DataArray,
    error_slice: xr.DataArray,
    max_altitude: float,
    channel_name: str,
    ds: xr.Dataset,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    pbl_da: xr.DataArray | None = None,
    cpt_km: float = -999.0,
    lrt_km: float = -999.0,
) -> Path:
    """Render one Level 1 RCS quicklook and side mean profile."""
    output_format, dpi = _get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds)
    pretty_channel = format_channel_name(channel_name)
    color = channel_color(channel_name)

    meas_title = (
        f"RCS at {pretty_channel} (0 - {float(max_altitude):g} km)\n"
        f"{date_title}"
    )

    fig = plt.figure(figsize=(15, 7.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.03)

    ax0 = plt.subplot(gs[0])
    plot = data_slice.plot(
        x="time",
        y="altitude",
        cmap="jet",
        robust=True,
        vmin=0,
        add_colorbar=False,
        ax=ax0,
        add_labels=False,
        rasterized=True,
    )

    lower_altitude = 0.16 if "AN" in pretty_channel else 0.5
    ax0.set_title(meas_title, fontsize=15, fontweight="bold", loc="center")
    ax0.set_xlabel("Time (UTC)", fontsize=13, fontweight="bold")
    ax0.set_ylabel("Altitude (km a.g.l.)", fontsize=13, fontweight="bold")
    ax0.set_ylim(lower_altitude, max_altitude)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    if pbl_da is not None:
        ax0.plot(pbl_da.time, pbl_da.values, color="crimson", linestyle=":", linewidth=2.5, alpha=1.0, label="PBL Top")
        ax0.legend(loc="upper right", framealpha=0.7, fontsize=9, facecolor="white", edgecolor="black")

    ax1 = plt.subplot(gs[1], sharey=ax0)
    mean_profile = safe_time_mean(data_slice)
    mean_error = safe_error_of_mean(error_slice)
    smooth_profile = rolling_altitude(mean_profile, bins=20)
    smooth_error = rolling_altitude(mean_error, bins=20)

    ax1.plot(smooth_profile, smooth_profile.altitude, color=color, linewidth=2)
    ax1.fill_betweenx(
        smooth_profile.altitude,
        smooth_profile - smooth_error,
        smooth_profile + smooth_error,
        color=color,
        alpha=0.3,
        edgecolor="none",
    )
    ax1.set_xlabel("Mean RCS", fontsize=12, fontweight="bold")
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.grid(True, linestyle="--", alpha=0.6, which="both")

    values = np.asarray(smooth_profile.values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        p_max = float(np.nanmax(finite_values))
        p_min = float(np.nanmin(finite_values))
        margin = max((p_max - p_min) * 0.15, 1e-12)
        ax1.set_xlim(min(0.0, p_min) - margin, p_max + margin)

    ax1.set_ylim(lower_altitude, max_altitude)
    has_legend = False

    if pbl_da is not None:
        try:
            mean_pbl = float(pbl_da.mean(skipna=True).values)
            if np.isfinite(mean_pbl) and 0 < mean_pbl <= max_altitude:
                ax1.axhline(
                    y=mean_pbl,
                    color="crimson",
                    linestyle="--",
                    linewidth=1.8,
                    zorder=5,
                    label=f"Mean PBL ({mean_pbl:.1f} km)",
                )
                has_legend = True
        except Exception:
            pass

    if np.isfinite(cpt_km) and 0 < cpt_km <= max_altitude:
        ax1.axhline(y=cpt_km, color="royalblue", linestyle=":", linewidth=1.8, zorder=5, label=f"CPT ({cpt_km:.1f} km)")
        has_legend = True
    if np.isfinite(lrt_km) and 0 < lrt_km <= max_altitude:
        ax1.axhline(y=lrt_km, color="forestgreen", linestyle="-.", linewidth=1.8, zorder=5, label=f"LRT ({lrt_km:.1f} km)")
        has_legend = True
    if has_legend:
        ax1.legend(loc="upper right", framealpha=0.9, fontsize=9, facecolor="white", edgecolor="black")

    plt.subplots_adjust(left=0.14, bottom=0.15, right=0.95, top=0.88)
    cb_ax = fig.add_axes([0.06, 0.15, 0.015, 0.73])
    cb = fig.colorbar(plot, cax=cb_ax, orientation="vertical")
    cb.set_label("Intensity [a.u.]", fontsize=12, fontweight="bold")
    cb_ax.yaxis.set_ticks_position("left")
    cb_ax.yaxis.set_label_position("left")

    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / (
        f"Quicklook_{file_name_prefix}_{pretty_channel.replace(' ', '_')}_{float(max_altitude):g}km.{output_format}"
    )
    return _save_figure(fig, out_path, dpi=dpi)


def plot_global_mean_rcs(
    ds: xr.Dataset,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Render a comparative global mean RCS profile for configured channels."""
    output_format, dpi = _get_output_settings(config)
    altitude_ranges = config.get("visualization", {}).get("altitude_ranges_km", [5, 15, 30])
    max_altitude = float(max(altitude_ranges))
    date_title, _ = extract_datetime_strings(ds)

    fig, ax = plt.subplots(figsize=(8, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.15)
    plotted = False

    channels_to_plot = config.get("visualization", {}).get("channels_to_plot", []) or []
    available_channels = {str(channel) for channel in ds.channel.values}
    seen: set[str] = set()

    for channel in channels_to_plot:
        channel_name = str(channel)
        if channel_name in seen:
            continue
        seen.add(channel_name)
        if channel_name not in available_channels:
            continue

        rc_sig = ds["corrected_signal"].sel(channel=channel_name).where(ds["altitude"] <= max_altitude, drop=True)
        rc_err = ds["corrected_signal_error"].sel(channel=channel_name).where(ds["altitude"] <= max_altitude, drop=True)
        if rc_sig.size == 0:
            continue

        mean_prof = rolling_altitude(safe_time_mean(rc_sig), bins=50)
        mean_err = rolling_altitude(safe_error_of_mean(rc_err), bins=50)

        ax.plot(
            mean_prof,
            mean_prof.altitude,
            color=channel_color(channel_name),
            linestyle="-" if "an" in channel_name.lower() else "--",
            label=format_channel_name(channel_name),
            linewidth=2,
        )
        ax.fill_betweenx(
            mean_prof.altitude,
            mean_prof - mean_err,
            mean_prof + mean_err,
            color=channel_color(channel_name),
            alpha=0.2,
            edgecolor="none",
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_title(f"Mean RCS (0 - {max_altitude:g} km)\n{date_title}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean RCS [a.u.]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0, max_altitude)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, which="both", alpha=0.5)

    add_footer_and_logos(fig, root_dir)
    out_path = Path(output_folder) / f"GlobalMeanRCS_{file_name_prefix}.{output_format}"
    return _save_figure(fig, out_path, dpi=dpi)


# =============================================================================
# Level 2 QA plotting engines
# =============================================================================


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
    required = {
        "glued_corrected_signal",
        "molecular_backscatter",
        "rayleigh_calibration_factor",
        "rayleigh_reference_altitude_m",
    }
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = _get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    pretty_wl = format_wavelength_label(wavelength)
    alt_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(15.0, float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km

    mean_glued = rolling_altitude(safe_time_mean(ds_l2["glued_corrected_signal"].sel(wavelength=wavelength)), bins=20)
    beta_mol = ds_l2["molecular_backscatter"].sel(wavelength=wavelength)
    calibration_factor = float(ds_l2["rayleigh_calibration_factor"].sel(wavelength=wavelength).values)
    rayleigh_rcs = rolling_altitude(beta_mol * calibration_factor, bins=20)
    ref_alt_km = float(ds_l2["rayleigh_reference_altitude_m"].sel(wavelength=wavelength).values) / 1000.0

    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    fig.subplots_adjust(top=0.86, bottom=0.14)
    ax.plot(mean_glued.values[valid_alt], alt_km[valid_alt], color=channel_color(wavelength), linewidth=2.2, label="Mean glued RCS")
    ax.plot(rayleigh_rcs.values[valid_alt], alt_km[valid_alt], color="black", linestyle="--", linewidth=2.0, label="Scaled Rayleigh molecular profile")

    if np.isfinite(ref_alt_km) and 0 < ref_alt_km <= max_alt_km:
        ax.axhline(ref_alt_km, color="black", linestyle=":", linewidth=1.8, label=f"Reference {ref_alt_km:.2f} km")

    fit_cfg = config.get("inversion", {}).get("molecular_fit", {}) or {}
    ref_min_m = fit_cfg.get("ref_alt_min_m")
    ref_max_m = fit_cfg.get("ref_alt_max_m")
    if ref_min_m is not None and ref_max_m is not None:
        ax.axhspan(float(ref_min_m) / 1000.0, float(ref_max_m) / 1000.0, alpha=0.12, color="gray", label="Configured molecular-fit search range")

    ax.set_title(f"QA Molecular Fit - {pretty_wl}\n{date_title}", fontsize=14, fontweight="bold")
    ax.set_xlabel("RCS / scaled molecular signal [a.u.]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max_alt_km)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.grid(True, which="both", alpha=0.45)
    add_atmospheric_boundaries(ax, ds_l2, max_alt_km)
    ax.legend(fontsize=9, loc="best")
    add_footer_and_logos(fig, root_dir)

    out_path = Path(output_folder) / f"QA_Molecular_{file_name_prefix}_{wavelength}nm.{output_format}"
    return _save_figure(fig, out_path, dpi=dpi)


def plot_qa_l2_kfs(
    ds_l2: xr.Dataset,
    wavelength_nm: int | float,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
    max_altitude_km: float = 15.0,
) -> Path | None:
    """Render Level 2 KFS QA panel with Monte Carlo uncertainty bands."""
    wavelength = int(wavelength_nm)
    required = {
        "aerosol_backscatter",
        "aerosol_backscatter_error",
        "aerosol_extinction",
        "aerosol_extinction_error",
    }
    if not required.issubset(set(ds_l2.data_vars)):
        return None

    output_format, dpi = _get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds_l2)
    pretty_wl = format_wavelength_label(wavelength)
    alt_km = altitude_to_km(ds_l2["altitude"].values)
    max_alt_km = min(float(max_altitude_km), float(np.nanmax(alt_km)))
    valid_alt = alt_km <= max_alt_km

    beta = ds_l2["aerosol_backscatter"].sel(wavelength=wavelength)
    beta_err = ds_l2["aerosol_backscatter_error"].sel(wavelength=wavelength)
    alpha = ds_l2["aerosol_extinction"].sel(wavelength=wavelength)
    alpha_err = ds_l2["aerosol_extinction_error"].sel(wavelength=wavelength)

    beta_mean = rolling_altitude(safe_time_mean(beta), bins=15)
    beta_sigma = rolling_altitude(safe_error_of_mean(beta_err), bins=15)
    alpha_mean = rolling_altitude(safe_time_mean(alpha), bins=15)
    alpha_sigma = rolling_altitude(safe_error_of_mean(alpha_err), bins=15)

    color = channel_color(wavelength)
    fig = plt.figure(figsize=(13.5, 8.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharey=ax0)

    b = np.asarray(beta_mean.values, dtype=float)
    bs = np.asarray(beta_sigma.values, dtype=float)
    a = np.asarray(alpha_mean.values, dtype=float)
    ase = np.asarray(alpha_sigma.values, dtype=float)

    ax0.plot(b[valid_alt], alt_km[valid_alt], color=color, linewidth=2.2, label="Mean beta aer")
    ax0.fill_betweenx(alt_km[valid_alt], b[valid_alt] - bs[valid_alt], b[valid_alt] + bs[valid_alt], color=color, alpha=0.25, edgecolor="none", label="MC 1σ")
    ax0.axvline(0.0, color="black", linewidth=0.8)
    ax0.set_title("Aerosol backscatter", fontsize=14, fontweight="bold")
    ax0.set_xlabel(r"$\beta_{aer}$ [m$^{-1}$ sr$^{-1}$]", fontsize=12, fontweight="bold")
    ax0.set_ylabel("Altitude (km a.g.l.)", fontsize=12, fontweight="bold")
    ax0.set_ylim(0, max_alt_km)
    ax0.grid(True, alpha=0.45)

    ax1.plot(a[valid_alt], alt_km[valid_alt], color=color, linewidth=2.2, label="Mean alpha aer")
    ax1.fill_betweenx(alt_km[valid_alt], a[valid_alt] - ase[valid_alt], a[valid_alt] + ase[valid_alt], color=color, alpha=0.25, edgecolor="none", label="MC 1σ")
    ax1.axvline(0.0, color="black", linewidth=0.8)
    ax1.set_title("Aerosol extinction", fontsize=14, fontweight="bold")
    ax1.set_xlabel(r"$\alpha_{aer}$ [m$^{-1}$]", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.45)
    plt.setp(ax1.get_yticklabels(), visible=False)

    for ax in (ax0, ax1):
        add_atmospheric_boundaries(ax, ds_l2, max_alt_km)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        f"MILGRAU Level 2 QA - KFS Optical Retrieval - {pretty_wl}\n{date_title}",
        fontsize=15,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(top=0.84, bottom=0.14)
    add_footer_and_logos(fig, root_dir)

    out_path = Path(output_folder) / f"QA_L2_KFS_{file_name_prefix}_{wavelength}nm.{output_format}"
    return _save_figure(fig, out_path, dpi=dpi)


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
    for wavelength_nm in get_wavelength_values(ds_l2):
        molecular_path = plot_qa_molecular_fit(ds_l2, wavelength_nm, output_folder, file_name_prefix, config, root_dir)
        if molecular_path is not None:
            generated.append(molecular_path)
        kfs_path = plot_qa_l2_kfs(ds_l2, wavelength_nm, output_folder, file_name_prefix, config, root_dir)
        if kfs_path is not None:
            generated.append(kfs_path)
    return generated
