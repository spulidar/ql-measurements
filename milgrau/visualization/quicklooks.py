"""Level 1 quicklook and mean-profile plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from milgrau.visualization.style import add_footer_and_logos, channel_color, get_output_settings

RCS_VARIABLE = "range_corrected_signal"
RCS_ERROR_VARIABLE = "range_corrected_signal_error"


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
        return f"{int(parts[0])}nm {parts[1]}"
    except Exception:
        return str(raw_name)


def safe_time_mean(da: xr.DataArray) -> xr.DataArray:
    """Return the time mean of a DataArray when a time dimension is present."""
    if "time" in da.dims:
        return da.mean(dim="time", skipna=True)
    return da


def safe_error_of_mean(err_da: xr.DataArray) -> xr.DataArray:
    """Combine profile one-sigma errors over time as uncertainty of the mean."""
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


def _save_figure(fig: Any, out_path: str | Path, dpi: int) -> Path:
    """Save and close a Matplotlib figure."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _get_gap_threshold_minutes(config: dict[str, Any], data_slice: xr.DataArray) -> float:
    """Return the temporal-gap threshold for drawing missing acquisition gaps."""
    quicklook_cfg = config.get("visualization", {}).get("quicklook", {}) or {}
    configured = quicklook_cfg.get("max_time_gap_minutes")
    if configured is not None:
        return float(configured)

    if "time" not in data_slice.coords or data_slice.sizes.get("time", 0) < 3:
        return 10.0

    times = pd.to_datetime(data_slice["time"].values)
    deltas_min = np.diff(times.values).astype("timedelta64[s]").astype(float) / 60.0
    finite = deltas_min[np.isfinite(deltas_min) & (deltas_min > 0.0)]
    if finite.size == 0:
        return 10.0
    return max(float(np.nanmedian(finite) * 3.0), 5.0)


def _insert_time_gap_markers(data_slice: xr.DataArray, config: dict[str, Any]) -> xr.DataArray:
    """Insert NaN profiles into large temporal gaps so quicklooks show missing data.

    Without this step, pcolormesh-style quicklooks visually stretch the previous
    and next profiles across long acquisition gaps. NaN marker profiles force the
    gap to be rendered with the colormap's ``bad`` color.
    """
    if "time" not in data_slice.dims or "altitude" not in data_slice.dims:
        return data_slice
    if data_slice.sizes.get("time", 0) < 2:
        return data_slice

    da = data_slice.transpose("time", "altitude")
    times = pd.to_datetime(da["time"].values)
    values = np.asarray(da.values)
    threshold = pd.Timedelta(minutes=_get_gap_threshold_minutes(config, da))
    marker_delta = pd.Timedelta(seconds=1)

    new_times: list[pd.Timestamp] = []
    new_profiles: list[np.ndarray] = []
    inserted = False

    for idx in range(len(times)):
        new_times.append(times[idx])
        new_profiles.append(values[idx, :])
        if idx == len(times) - 1:
            continue

        gap = times[idx + 1] - times[idx]
        if gap > threshold:
            left_marker = times[idx] + marker_delta
            right_marker = times[idx + 1] - marker_delta
            if right_marker <= left_marker:
                midpoint = times[idx] + gap / 2
                left_marker = midpoint
                right_marker = midpoint
            nan_profile = np.full(values.shape[1], np.nan, dtype=np.float64)
            new_times.extend([left_marker, right_marker])
            new_profiles.extend([nan_profile, nan_profile.copy()])
            inserted = True

    if not inserted:
        return data_slice

    result = xr.DataArray(
        np.stack(new_profiles, axis=0),
        dims=("time", "altitude"),
        coords={"time": np.asarray(new_times, dtype="datetime64[ns]"), "altitude": da["altitude"].values},
        attrs=da.attrs,
        name=da.name,
    )
    result["altitude"].attrs.update(da["altitude"].attrs)
    return result


def _quicklook_colormap(config: dict[str, Any]):
    """Return the colormap used for quicklooks, including missing-data color."""
    quicklook_cfg = config.get("visualization", {}).get("quicklook", {}) or {}
    cmap_name = str(quicklook_cfg.get("colormap", "jet"))
    missing_color = str(quicklook_cfg.get("missing_data_color", "lightgray"))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=missing_color)
    return cmap


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
    output_format, dpi = get_output_settings(config)
    date_title, _ = extract_datetime_strings(ds)
    pretty_channel = format_channel_name(channel_name)
    color = channel_color(channel_name)
    display_data = _insert_time_gap_markers(data_slice, config)

    fig = plt.figure(figsize=(15, 7.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.03)

    ax0 = plt.subplot(gs[0])
    plot = display_data.plot(
        x="time",
        y="altitude",
        cmap=_quicklook_colormap(config),
        robust=True,
        vmin=0,
        add_colorbar=False,
        ax=ax0,
        add_labels=False,
        rasterized=True,
    )

    lower_altitude = 0.16 if "AN" in pretty_channel else 0.5
    ax0.set_title(
        f"RCS at {pretty_channel} (0 - {float(max_altitude):g} km)\n{date_title}",
        fontsize=15,
        fontweight="bold",
        loc="center",
    )
    ax0.set_xlabel("Time (UTC)", fontsize=13, fontweight="bold")
    ax0.set_ylabel("Altitude (km a.g.l.)", fontsize=13, fontweight="bold")
    ax0.set_ylim(lower_altitude, max_altitude)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax1 = plt.subplot(gs[1], sharey=ax0)
    smooth_profile = rolling_altitude(safe_time_mean(data_slice), bins=20)
    smooth_error = rolling_altitude(safe_error_of_mean(error_slice), bins=20)
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
    ax1.set_ylim(lower_altitude, max_altitude)

    values = np.asarray(smooth_profile.values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size:
        p_max = float(np.nanmax(finite_values))
        p_min = float(np.nanmin(finite_values))
        margin = max((p_max - p_min) * 0.15, 1e-12)
        ax1.set_xlim(min(0.0, p_min) - margin, p_max + margin)

    has_legend = False
    if pbl_da is not None:
        try:
            mean_pbl = float(pbl_da.mean(skipna=True).values)
            if np.isfinite(mean_pbl) and 0 < mean_pbl <= max_altitude:
                ax1.axhline(mean_pbl, color="crimson", linestyle="--", linewidth=1.8, zorder=5, label=f"Mean PBL ({mean_pbl:.1f} km)")
                has_legend = True
        except Exception:
            pass

    if np.isfinite(cpt_km) and 0 < cpt_km <= max_altitude:
        ax1.axhline(cpt_km, color="royalblue", linestyle=":", linewidth=1.8, zorder=5, label=f"CPT ({cpt_km:.1f} km)")
        has_legend = True
    if np.isfinite(lrt_km) and 0 < lrt_km <= max_altitude:
        ax1.axhline(lrt_km, color="forestgreen", linestyle="-.", linewidth=1.8, zorder=5, label=f"LRT ({lrt_km:.1f} km)")
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

    out_path = Path(output_folder) / f"Quicklook_{file_name_prefix}_{pretty_channel.replace(' ', '_')}_{float(max_altitude):g}km.{output_format}"
    return _save_figure(fig, out_path, dpi=dpi)


def plot_global_mean_rcs(
    ds: xr.Dataset,
    output_folder: str | Path,
    file_name_prefix: str,
    config: dict[str, Any],
    root_dir: str | Path,
) -> Path | None:
    """Render a comparative global mean RCS profile for configured channels."""
    output_format, dpi = get_output_settings(config)
    max_altitude = float(max(config.get("visualization", {}).get("altitude_ranges_km", [5, 15, 30])))
    date_title, _ = extract_datetime_strings(ds)

    if RCS_VARIABLE not in ds or RCS_ERROR_VARIABLE not in ds:
        raise KeyError(f"Dataset must contain {RCS_VARIABLE} and {RCS_ERROR_VARIABLE}.")

    fig, ax = plt.subplots(figsize=(8, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.15)
    plotted = False
    available_channels = {str(channel) for channel in ds.channel.values}
    seen: set[str] = set()

    for channel_name in config.get("visualization", {}).get("channels_to_plot", []) or []:
        channel_name = str(channel_name)
        if channel_name in seen:
            continue
        seen.add(channel_name)
        if channel_name not in available_channels:
            continue

        rc_sig = ds[RCS_VARIABLE].sel(channel=channel_name).where(ds["altitude"] <= max_altitude, drop=True)
        rc_err = ds[RCS_ERROR_VARIABLE].sel(channel=channel_name).where(ds["altitude"] <= max_altitude, drop=True)
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
        ax.fill_betweenx(mean_prof.altitude, mean_prof - mean_err, mean_prof + mean_err, color=channel_color(channel_name), alpha=0.2, edgecolor="none")
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
