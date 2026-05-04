"""
MILGRAU - Visualization Utilities
Handles matplotlib configurations, standard Lidar quicklooks,
error band plotting, and aesthetic formatting (logos, footers).
WIP: Phase 2 Interactive QA plots (Gluing, Molecular Fit, KFS).

@author: Luisa Mello, Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import ListedColormap

from datetime import datetime
from pathlib import Path

# ==========================================
# PHASE 1: STRING & METADATA FORMATTING
# ==========================================

def extract_datetime_strings(ds):
    """Dynamically extracts formatted dates from the native time coordinates."""
    try:
        dt_in = pd.to_datetime(ds.time.values.min())
        dt_end = pd.to_datetime(ds.time.values.max())
        
        date_title = f"{dt_in.strftime('%d %b %Y - %H:%M')} to {dt_end.strftime('%H:%M')} UTC"
        date_footer = dt_in.strftime("%d %b %Y")
        return date_title, date_footer
    except Exception:
        return "Unknown date", "Unknown date"

def format_channel_name(raw_name):
    try:
        parts = raw_name.split('.')
        wavelength = int(parts[0])
        mode = str(parts[1])
        return f"{wavelength}nm {mode}"
    except Exception:
        return raw_name

def add_footer_and_logos(fig, root_dir):
    """Adds institutional footer with aligned logos."""
    fig.text(0.08, 0.04, "SPU Lidar Station - São Paulo", fontsize=13, fontweight="bold", color="#333333", va="center")

    root_path = Path(root_dir)
    logos = [
        (root_path / "img" / "CC_BY-NC-ND.png", 0.040),
        (root_path / "img" / "lalinet_logo2.png", 0.070),
        (root_path / "img" / "logo_leal2.png", 0.065),
    ]

    spacing, y_pos, x_right = 0.010, 0.01, 0.98
    for path, height in logos:
        if not path.exists(): continue
        img = mpimg.imread(str(path))
        h, w = img.shape[:2]
        width = height * (w / h) 
        x_left = x_right - width
        ax = fig.add_axes([x_left, y_pos, width, height], zorder=12)
        ax.imshow(img)
        ax.axis("off")
        x_right = x_left - spacing

# ==========================================
# PHASE 1: LEVEL 1 PLOTTING ENGINES
# ==========================================

def plot_quicklook(data_slice, error_slice, max_altitude, channel_name, ds, output_folder, file_name_prefix, config, root_dir, pbl_da=None, cpt_km=-999.0, lrt_km=-999.0):
    date_title, date_footer = extract_datetime_strings(ds)
    pretty_channel = format_channel_name(channel_name)
    meas_title = f"RCS at {pretty_channel} (0 - {max_altitude} km)\n{date_title}\nSPU Lidar Station - São Paulo"

    fig = plt.figure(figsize=[15, 7.5])
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.03)
    
    # ---------------------------------------------------------
    # SUBPLOT 0:  Colormap (RCS)
    # ---------------------------------------------------------
    ax0 = plt.subplot(gs[0])
    
    # Plot standard data
    plot = data_slice.plot(x='time', y='altitude', cmap='jet', robust=True, vmin=0, 
                           add_colorbar=False, ax=ax0, add_labels=False, rasterized=True)
                           
    # Plot the PBL tracking line
    if pbl_da is not None:
        ax0.plot(pbl_da.time, pbl_da.values, color='crimson', linestyle=':', linewidth=2.5, alpha=1.0, label='PBL Top')
        ax0.legend(loc='upper right', framealpha=0.7, fontsize=9, facecolor='white', edgecolor='black')
    
    ax0.set_title(meas_title, fontsize=15, fontweight="bold", loc='center')
    ax0.set_xlabel('Time (UTC)', fontsize=13, fontweight="bold")
    ax0.set_ylabel('Altitude (km a.g.l.)', fontsize=13, fontweight="bold")
    ax0.set_ylim(0.16 if "AN" in pretty_channel else 0.5, max_altitude)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # ---------------------------------------------------------
    # SUBPLOT 1: Mean Profile & Atmospheric Boundaries
    # ---------------------------------------------------------
    ax1 = plt.subplot(gs[1], sharey=ax0)
    mean_profile = data_slice.mean(dim='time')
    mean_error = np.sqrt((error_slice**2).sum(dim='time')) / error_slice.sizes['time']
    smooth_profile = mean_profile.rolling(altitude=20, min_periods=1).mean()
    smooth_error = mean_error.rolling(altitude=20, min_periods=1).mean()

    line_color = {"532": "forestgreen", "355": "rebeccapurple", "1064": "crimson", "387": "darkblue"}.get(next((k for k in ["532", "355", "1064", "387"] if k in channel_name), "black"), "black")

    ax1.plot(smooth_profile, smooth_profile.altitude, color=line_color, linewidth=2)
    ax1.fill_betweenx(smooth_profile.altitude, smooth_profile - smooth_error, smooth_profile + smooth_error, color=line_color, alpha=0.3, edgecolor="none")
    ax1.set_xlabel('Mean RCS', fontsize=12, fontweight="bold")
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.grid(True, linestyle='--', alpha=0.6, which='both')

    p_max = float(np.nanmax(smooth_profile.values))
    p_min = float(np.nanmin(smooth_profile.values))
    margin = (p_max - p_min) * 0.15 
    ax1.set_xlim(min(0, p_min)-margin, p_max + margin)
    ax1.set_ylim(0.16 if "AN" in pretty_channel else 0.5, max_altitude)

    has_legend = False
    # Planetary Boundary Layer 
    if pbl_da is not None:
        mean_pbl = float(pbl_da.mean().values)
        if 0 < mean_pbl <= max_altitude:
            ax1.axhline(y=mean_pbl, color='crimson', linestyle='--', linewidth=1.8, zorder=5, label=f'Mean PBL ({mean_pbl:.1f} km)')
            has_legend = True

    # Cold Point Tropopause (CPT) 
    if 0 < cpt_km <= max_altitude:
        ax1.axhline(y=cpt_km, color='royalblue', linestyle=':', linewidth=1.8, zorder=5, label=f'CPT ({cpt_km:.1f} km)')
        has_legend = True
        
    # Lapse Rate Tropopause (LRT - WMO) 
    if 0 < lrt_km <= max_altitude:
        ax1.axhline(y=lrt_km, color='forestgreen', linestyle='-.', linewidth=1.8, zorder=5, label=f'LRT ({lrt_km:.1f} km)')
        has_legend = True
        
    if has_legend:
        ax1.legend(loc='upper right', framealpha=0.9, fontsize=9, facecolor='white', edgecolor='black')

    # ---------------------------------------------------------
    # FINAL FORMATTING & EXPORT
    # ---------------------------------------------------------
    plt.subplots_adjust(left=0.14, bottom=0.15, right=0.95, top=0.88)
    cb_ax = fig.add_axes([0.06, 0.15, 0.015, 0.73])
    cb = fig.colorbar(plot, cax=cb_ax, orientation='vertical')
    cb.set_label("Intensity [a.u.]", fontsize=12, fontweight="bold")
    cb_ax.yaxis.set_ticks_position('left')
    cb_ax.yaxis.set_label_position('left')

    add_footer_and_logos(fig, root_dir)
    
    out_path = Path(output_folder) / f'Quicklook_{file_name_prefix}_{pretty_channel.replace(" ", "_")}_{max_altitude}km.webp'
    plt.savefig(out_path, dpi=120)
    plt.close(fig)

def plot_global_mean_rcs(ds, output_folder, file_name_prefix, config, root_dir):
    max_altitude = max(config.get("visualization", {}).get("altitude_ranges_km", [5, 15, 30]))
    date_title, date_footer = extract_datetime_strings(ds)
    
    fig, ax = plt.subplots(figsize=(8, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.15)
    base_colors = { 355: "rebeccapurple", 387: "darkblue", 408: "darkcyan", 530: "orange", 532: "forestgreen", 1064: "crimson" }
    plotted = False

    for ch in config.get("visualization", {}).get("channels_to_plot", []):
        if ch in ds.channel.values:
            color = base_colors.get(int(ch.split('.')[0]), "black") if "." in ch else "black"
            
            rc_sig = ds['corrected_signal'].sel(channel=ch).where(ds['altitude'] <= max_altitude, drop=True)
            rc_err = ds['corrected_signal_error'].sel(channel=ch).where(ds['altitude'] <= max_altitude, drop=True)
            
            mean_prof = rc_sig.mean(dim='time').rolling(altitude=50, min_periods=1).mean()
            mean_err = (np.sqrt((rc_err**2).sum(dim='time')) / rc_err.sizes['time']).rolling(altitude=50, min_periods=1).mean()

            ax.plot(mean_prof, mean_prof.altitude, color=color, linestyle="-" if "an" in ch.lower() else "--", label=format_channel_name(ch), linewidth=2)
            ax.fill_betweenx(mean_prof.altitude, mean_prof - mean_err, mean_prof + mean_err, color=color, alpha=0.2, edgecolor="none")
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_title(f"Mean RCS (0 - {max_altitude} km)\n{date_title}\nSPU Lidar Station - São Paulo", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean RCS [a.u.]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0, max_altitude)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, which='both', alpha=0.5)

    add_footer_and_logos(fig, root_dir)
    
    out_path = Path(output_folder) / f'GlobalMeanRCS_{file_name_prefix}.webp'
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
