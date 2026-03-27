"""
LIdar RAnge COrrection Signal - LIRACOS
This script provides tools to handle range corrected signal graphics and RCS maps
(quicklooks). It reads Level 1 NetCDF files, calculates time-averaged profiles,
propagates uncertainties, and plots shaded error bands (1-sigma).

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import warnings

# Suppress expected xarray/numpy warnings for all-NaN slices
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, f"LIRACOS_run_{datetime.now().strftime('%Y%m%d')}.log")

logger = logging.getLogger("LIRACOS")
logger.setLevel(logging.INFO)

# Clear existing handlers to prevent duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# File handler
fh = logging.FileHandler(log_filename)
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

# Prevent log propagation to the root logger
logger.propagate = False

# ==========================================
# SETTINGS
# ==========================================
INCREMENTAL_PROCESSING = False
MAX_WORKERS = 4

rootdir_name = os.getcwd()
files_dir_level1 = "05-data_level1"
base_data_folder = os.path.join(rootdir_name, files_dir_level1)

ALTITUDE_RANGES = [5, 15, 30] # km
VERTICAL_RESOLUTION_M = 7.5

channels_to_plot = [
    "01064.o_an",
    "00532.o_an", "00532.o_ph",
    "00355.o_an", "00355.o_ph",
]

LOGO_LEAL = os.path.join(rootdir_name, "img", "logo_leal.jpeg")
LOGO_LALINET = os.path.join(rootdir_name, "img", "lalinet_logo2.jpeg")
LOGO_LICENSE = os.path.join(rootdir_name, "img", "by-nc-nd.png")

# ==========================================
# FORMATTING
# ==========================================
def extract_datetime_strings(ds):
    """Extract and format start and end times from NetCDF attributes."""
    try:
        dt_in_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Start_Time_UT']).zfill(6)}"
        dt_end_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Stop_Time_UT']).zfill(6)}"

        dt_in = datetime.strptime(dt_in_str, "%Y%m%d%H%M%S")
        dt_end = datetime.strptime(dt_end_str, "%Y%m%d%H%M%S")

        date_title = f"{dt_in.strftime('%d %b %Y - %H:%M')} to {dt_end.strftime('%d %b %Y - %H:%M')} UTC"
        date_footer = dt_in.strftime("%d %b %Y")
        return date_title, date_footer
    except Exception:
        return "Unknown date", "Unknown date"

def format_channel_name(raw_name):
    """Format SCC channel name for plot legends (e.g., '00532.o_ph' -> '532nm PC')."""
    try:
        parts = raw_name.split('.')
        wavelength = int(parts[0])
        mode = parts[1].split('_')[1].upper()
        if mode == 'PH':
            mode = 'PC'
        return f"{wavelength}nm {mode}"
    except Exception:
        return raw_name

def add_footer_and_logos(fig, date_footer):

    # =========================
    # TEXT FOR QUCIKLOOK IMAGES
    # =========================
    fig.text(0.10, 0.03, date_footer,
             fontsize=12, fontweight="bold", va="center")

    fig.text(0.30, 0.03, "SPU-Lidar Station",
             fontsize=12, fontweight="bold",
             color="black", ha="right", va="center")

    # =========================
    # LOGO CONFIGURATION
    # (simple height control)
    # =========================
    logos = [
        (LOGO_LICENSE, 0.040),
        (LOGO_LALINET,  0.070),
        (LOGO_LEAL,    0.065),
    ]

    spacing = 0.006   # Spacing between logos
    y_pos = 0.005     # Vertical position
    x_right = 0.98    # Starts from the right

    # =========================
    # LOOP FOR INSERTING INFORMATION IN QUICKLOOKS
    # =========================
    for path, height in logos:

        if not os.path.exists(path):
            continue

        img = mpimg.imread(path)
        h, w = img.shape[:2]

        # maintains proportion automatically
        width = height * (w / h)

        # CALCULATE POSITION
        x_left = x_right - width

        ax = fig.add_axes([x_left, y_pos, width, height], zorder=12)
        ax.imshow(img)
        ax.axis("off")

        # UPDATING THE IMAGE POSITION
        x_right = x_left - spacing
        
# ==========================================
# QUICKLOOK (COLORMAP + MEAN RCS WITH ERROR BANDS)
# ==========================================
def plot_quicklook(data_slice, error_slice, max_altitude, channel_name, ds, output_folder, file_name_prefix):
    """Generates a 2D colormap quicklook and a 1D mean profile with error bands."""
    date_title, date_footer = extract_datetime_strings(ds)
    pretty_channel = format_channel_name(channel_name)
    meas_title = f"RCS at {pretty_channel} (0 - {max_altitude} km)\n{date_title}\nSPU Lidar Station - São Paulo"

    fig = plt.figure(figsize=[15, 7.5])
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.03)

    # --- Colormap Axis (2D) ---
    ax0 = plt.subplot(gs[0])
    plot = data_slice.plot(
        x='time', y='altitude',
        cmap='jet', robust=True, vmin=0, add_colorbar=False, ax=ax0, add_labels=False
    )
    
    if "AN" in pretty_channel:
        min_altitude = 0.16
    elif "PC" in pretty_channel:
        min_altitude = 0.5
    else:
        min_altitude = 0.0  # valor padrão (opcional)

    ax0.set_title(meas_title, fontsize=15, fontweight="bold", loc='center')
    ax0.set_xlabel('Time (UTC)', fontsize=13, fontweight="bold")
    ax0.set_ylabel('Altitude (km a.g.l.)', fontsize=13, fontweight="bold")
    ax0.set_ylim(min_altitude, max_altitude)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    # --- Mean RCS Axis (1D) with Error Bands ---
    ax1 = plt.subplot(gs[1], sharey=ax0)

    # Calculate time-averaged profile
    mean_profile = data_slice.mean(dim='time')

    # Calculate error of the mean (Quadrature sum / N)
    n_profiles = error_slice.sizes['time']
    mean_error = np.sqrt((error_slice**2).sum(dim='time')) / n_profiles

    # Apply rolling mean to smooth high-frequency noise for visualization
    smooth_profile = mean_profile.rolling(altitude=20, min_periods=1).mean()
    smooth_error = mean_error.rolling(altitude=20, min_periods=1).mean()

    # Channel colors
    line_color = "black"
    if "532" in channel_name: line_color = "forestgreen"
    elif "355" in channel_name: line_color = "rebeccapurple"
    elif "1064" in channel_name: line_color = "crimson"
    elif "387" in channel_name: line_color = "darkblue"
    elif "408" in channel_name: line_color = "darkcyan"
    elif "530" in channel_name: line_color = "orange"

    # Plot main line
    ax1.plot(smooth_profile, smooth_profile.altitude, color=line_color, linewidth=2)

    # Plot 1-sigma uncertainty band
    ax1.fill_betweenx(
        smooth_profile.altitude,
        smooth_profile - smooth_error,
        smooth_profile + smooth_error,
        color=line_color, alpha=0.3, edgecolor="none"
    )

    ax1.set_xlabel('Mean RCS', fontsize=12, fontweight="bold")
    plt.setp(ax1.get_yticklabels(), visible=False)
    ax1.grid(True, linestyle='--', alpha=0.6, which='both')

    # Layout adjustments and colorbar
    plt.subplots_adjust(left=0.14, bottom=0.15, right=0.95, top=0.88)
    cb_ax = fig.add_axes([0.06, 0.15, 0.015, 0.73]) # X, Y, W, H
    cb = fig.colorbar(plot, cax=cb_ax, orientation='vertical')
    cb.set_label("Intensity [a.u.]", fontsize=12, fontweight="bold")
    cb_ax.yaxis.set_ticks_position('left')
    cb_ax.yaxis.set_label_position('left')

    add_footer_and_logos(fig, date_footer)

    safe_channel_name = pretty_channel.replace(" ", "_")
    file_name = f'Quicklook_{file_name_prefix}_{safe_channel_name}_{max_altitude}km.webp'

    plt.savefig(os.path.join(output_folder, file_name), dpi=120)
    plt.close(fig)

# ==========================================
# GLOBAL MEAN RCS (MULTIPLE CHANNELS)
# ==========================================
def plot_global_mean_rcs(ds, output_folder, file_name_prefix):
    """Plots all requested channels on a single mean RCS graph with error bands."""
    max_altitude = max(ALTITUDE_RANGES)
    date_title, date_footer = extract_datetime_strings(ds)
    meas_title = f"Mean RCS (0 - {max_altitude} km)\n{date_title}\nSPU Lidar Station - São Paulo"

    fig, ax = plt.subplots(figsize=(8, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.15)

    base_colors = { 355: "rebeccapurple", 387: "darkblue", 408: "darkcyan", 530: "orange", 532: "forestgreen", 1064: "crimson" }
    plotted_anything = False

    for ch in channels_to_plot:
        if ch in ds.channel.values:
            label = format_channel_name(ch)
            try:
                wavelength = int(ch.split('.')[0])
                color = base_colors.get(wavelength, "black")
            except Exception:
                color = "black"

            line_style = "-" if "an" in ch.lower() else "--"

            # Slice Data and Errors
            rc_signal = ds['Range_Corrected_Signal'].sel(channel=ch)
            rc_error = ds['Range_Corrected_Signal_Error'].sel(channel=ch)

            sig_slice = rc_signal.where(rc_signal['altitude'] <= max_altitude, drop=True)
            err_slice = rc_error.where(rc_error['altitude'] <= max_altitude, drop=True)

            # Calculate Mean and Error
            mean_profile = sig_slice.mean(dim='time')
            n_profiles = err_slice.sizes['time']
            mean_error = np.sqrt((err_slice**2).sum(dim='time')) / n_profiles

            # Smooth
            smooth_profile = mean_profile.rolling(altitude=50, min_periods=1).mean()
            smooth_error = mean_error.rolling(altitude=50, min_periods=1).mean()

            # Plot main line and error band
            ax.plot(smooth_profile, smooth_profile.altitude, color=color, linestyle=line_style, label=label, linewidth=2)
            ax.fill_betweenx(
                smooth_profile.altitude,
                smooth_profile - smooth_error,
                smooth_profile + smooth_error,
                color=color, alpha=0.2, edgecolor="none"
            )
            plotted_anything = True

    if not plotted_anything:
        plt.close(fig)
        return

    ax.set_title(meas_title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean RCS [a.u.]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0, max_altitude)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, which='both', alpha=0.5)

    add_footer_and_logos(fig, date_footer)

    file_name = f'GlobalMeanRCS_{file_name_prefix}.webp'
    plt.savefig(os.path.join(output_folder, file_name), dpi=120)
    plt.close(fig)

# ==========================================
# MAIN PROCESSING
# ==========================================
def process_single_nc(nc_file):
    """Orchestrates the visualization for a single NetCDF Level 1 file."""
    try:
        file_name_prefix = os.path.basename(nc_file).replace('_level1_rcs.nc', '')

        base_folder = os.path.dirname(nc_file)
        output_folder = os.path.join(base_folder, "quicklooks")
        os.makedirs(output_folder, exist_ok=True)

        check_file = os.path.join(output_folder, f'GlobalMeanRCS_{file_name_prefix}.webp')
        if INCREMENTAL_PROCESSING and os.path.exists(check_file):
            return True # Skips if already processed

        logger.debug(f"Plotting quicklooks for {file_name_prefix}")
        ds = xr.open_dataset(nc_file)
        ds.load()

        # Generate altitude array from range resolution
        num_bins = ds.sizes['range']
        new_altitude_km = np.arange(0, num_bins * VERTICAL_RESOLUTION_M, VERTICAL_RESOLUTION_M) / 1000.0
        ds = ds.assign_coords(altitude=("range", new_altitude_km))
        ds = ds.swap_dims({'range': 'altitude'})

        # Build datetime array for the X-axis of the 2D plot
        try:
            dt_in_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Start_Time_UT']).zfill(6)}"
            dt_end_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Stop_Time_UT']).zfill(6)}"
            dt_in = datetime.strptime(dt_in_str, "%Y%m%d%H%M%S")
            dt_end = datetime.strptime(dt_end_str, "%Y%m%d%H%M%S")
            if dt_end < dt_in:
                dt_end += timedelta(days=1)
            time_array = pd.date_range(start=dt_in, end=dt_end, periods=ds.sizes['time'])
            ds = ds.assign_coords(time=time_array)
        except Exception as e:
            logger.warning(f"Could not build precise time array for {file_name_prefix}: {e}")

        # Generate individual channel quicklooks
        for channel_name in channels_to_plot:
            if channel_name in ds.channel.values:
                rc_signal = ds['Range_Corrected_Signal'].sel(channel=channel_name)
                rc_error = ds['Range_Corrected_Signal_Error'].sel(channel=channel_name)

                for max_altitude in ALTITUDE_RANGES:
                    sig_slice = rc_signal.where(rc_signal['altitude'] <= max_altitude, drop=True)
                    err_slice = rc_error.where(rc_error['altitude'] <= max_altitude, drop=True)

                    plot_quicklook(sig_slice, err_slice, max_altitude, channel_name, ds, output_folder, file_name_prefix)

        # Generate the global combined profile
        plot_global_mean_rcs(ds, output_folder, file_name_prefix)

        ds.close()
        logger.info(f"[OK] Plots saved: {file_name_prefix}")
        return True

    except Exception as e:
        logger.error(f"[FAILED] Error plotting {os.path.basename(nc_file)}", exc_info=True)
        return False


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    logger.info("=== Starting LIRACOS rendering ===")
    file_pattern = os.path.join(base_data_folder, '**', '*rcs.nc')
    nc_files = glob.glob(file_pattern, recursive=True)

    if not nc_files:
        logger.warning(f"No Level 1 NetCDF data found in {base_data_folder}. Exiting.")
    else:
        modo = "Incremental" if INCREMENTAL_PROCESSING else "Rewriting"
        logger.info(f"Found {len(nc_files)} Level 1 files. Mode: {modo}. Workers: {MAX_WORKERS}")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(process_single_nc, nc_files))

        if all(results):
            logger.info("=== LIRACOS processing finished successfully for all files! ===")
        elif any(results):
            logger.warning("=== LIRACOS finished, but some plots failed. Check the logs. ===")
        else:
            logger.error("=== LIRACOS failed completely. No plots were generated. ===")
