"""
MILGRAU Suite - Visualization Utilities
Handles matplotlib configurations, standard Lidar quicklooks,
error band plotting, and aesthetic formatting (logos, footers).
Includes Phase 2 Interactive QA plots (Gluing, Molecular Fit, KFS).

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import numpy as np
# Removed matplotlib.use('Agg') to allow Interactive Pop-ups for LEBEAR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter 
from matplotlib.colors import ListedColormap
from datetime import datetime

# Import our physics math for the cloud mask
from functions.physics_utils import calculate_dynamic_cloud_threshold

# ==========================================
# PHASE 1: STRING & METADATA FORMATTING
# ==========================================

def extract_datetime_strings(ds):
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
    try:
        parts = raw_name.split('.')
        wavelength = int(parts[0])
        mode = 'PC' if parts[1].split('_')[1].upper() == 'PH' else 'AN'
        return f"{wavelength}nm {mode}"
    except Exception:
        return raw_name

def add_footer_and_logos(fig, root_dir):
    """Adds a clean, date-free institutional footer with aligned logos."""
    fig.text(0.08, 0.04, "SPU Lidar Station - São Paulo", fontsize=13, fontweight="bold", color="#333333", va="center")

    logos = [
        (os.path.join(root_dir, "img", "CC_BY-NC-ND.png"), 0.040),
        (os.path.join(root_dir, "img", "lalinet_logo2.png"), 0.070),
        (os.path.join(root_dir, "img", "logo_leal2.png"), 0.065),
    ]

    spacing, y_pos, x_right = 0.010, 0.01, 0.98
    for path, height in logos:
        if not os.path.exists(path): continue
        img = mpimg.imread(path)
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

def plot_quicklook(data_slice, error_slice, max_altitude, channel_name, ds, output_folder, file_name_prefix, config, root_dir, pbl_km=-999.0, cpt_km=-999.0, lrt_km=-999.0):
    date_title, date_footer = extract_datetime_strings(ds)
    pretty_channel = format_channel_name(channel_name)
    meas_title = f"RCS at {pretty_channel} (0 - {max_altitude} km)\n{date_title}\nSPU Lidar Station - São Paulo"

    fig = plt.figure(figsize=[15, 7.5])
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.03)
    
    # ---------------------------------------------------------
    # SUBPLOT 0: Spatio-Temporal Colormap (RCS)
    # ---------------------------------------------------------
    ax0 = plt.subplot(gs[0])
    
    apply_cloud_mask = config.get("processing", {}).get("apply_cloud_mask", True)
    
    if apply_cloud_mask:
        multiplier = config.get("processing", {}).get("cloud_mask_multiplier", 10.0)
        dynamic_threshold = calculate_dynamic_cloud_threshold(data_slice, multiplier=multiplier)
        
        raw_vals = data_slice.values
        
        aerosol_mask = np.where(raw_vals < dynamic_threshold, raw_vals, np.nan)
        cloud_mask = np.where(raw_vals >= dynamic_threshold, raw_vals, np.nan)
        
        rcs_aerosol = data_slice.copy(data=aerosol_mask)
        rcs_clouds = data_slice.copy(data=cloud_mask)

        plot = rcs_aerosol.plot(x='time', y='altitude', cmap='jet', robust=True, vmin=0, 
                                add_colorbar=False, ax=ax0, add_labels=False, rasterized=True)
        
        rcs_clouds.plot(x='time', y='altitude', cmap=ListedColormap(['white']), 
                        add_colorbar=False, ax=ax0, add_labels=False, rasterized=True)
                        
        del raw_vals, aerosol_mask, cloud_mask, rcs_aerosol, rcs_clouds
    else:
        # rasterized=True prevents SVG/PDF vector explosion in high-res datasets
        plot = data_slice.plot(x='time', y='altitude', cmap='jet', robust=True, vmin=0, 
                               add_colorbar=False, ax=ax0, add_labels=False, rasterized=True)
    
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

    # --- INJECT VISUAL GUIDES (PBL & TROPOPAUSE) ---
    # Draw horizontal lines only if valid data (-999.0 means missing/failed)
    # Using 'zorder=5' to ensure lines are drawn on top of the shaded error regions
    
    # Planetary Boundary Layer (PBL) - Crimson dashed
    if pbl_km > 0 and pbl_km <= max_altitude:
        ax1.axhline(y=pbl_km, color='crimson', linestyle='--', linewidth=1.8, zorder=5, label=f'PBL ({pbl_km:.1f} km)')
        
    # Cold Point Tropopause (CPT) - Blue dotted
    if cpt_km > 0 and cpt_km <= max_altitude:
        ax1.axhline(y=cpt_km, color='royalblue', linestyle=':', linewidth=1.8, zorder=5, label=f'CPT ({cpt_km:.1f} km)')
        
    # Lapse Rate Tropopause (LRT - WMO) - Green dash-dot
    if lrt_km > 0 and lrt_km <= max_altitude:
        ax1.axhline(y=lrt_km, color='forestgreen', linestyle='-.', linewidth=1.8, zorder=5, label=f'LRT ({lrt_km:.1f} km)')
        
    # Trigger legend if any of the lines were plotted
    if (pbl_km > 0) or (cpt_km > 0) or (lrt_km > 0):
        ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)

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
    plt.savefig(os.path.join(output_folder, f'Quicklook_{file_name_prefix}_{pretty_channel.replace(" ", "_")}_{max_altitude}km.webp'), dpi=120)
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
            rc_sig = ds['Range_Corrected_Signal'].sel(channel=ch).where(ds['altitude'] <= max_altitude, drop=True)
            rc_err = ds['Range_Corrected_Signal_Error'].sel(channel=ch).where(ds['altitude'] <= max_altitude, drop=True)
            
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
    plt.savefig(os.path.join(output_folder, f'GlobalMeanRCS_{file_name_prefix}.webp'), dpi=120)
    plt.close(fig)

# ==========================================
# PHASE 2: LEVEL 2 INTERACTIVE QA PLOTS (LEBEAR)
# ==========================================

def plot_gluing_qa(alt_km, lower_sig, upper_sig, glued_sig, best_idx, window_size, config, channel_base_name, ds, root_dir, save_dir, prefix):
    date_title, date_footer = extract_datetime_strings(ds)
    fig = plt.figure(figsize=[15, 7.5])
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
    
    fig.suptitle(f"Gluing QA Calibration - {channel_base_name}\n{date_title}", fontsize=15, fontweight='bold', y=0.95)
    
    ax1 = plt.subplot(gs[0])
    ax1.plot(lower_sig, alt_km, color='royalblue', label='Analog', alpha=0.8, linewidth=2)
    ax1.plot(upper_sig, alt_km, color='darkorange', label='PC', alpha=0.8, linewidth=2)
    ax1.plot(glued_sig, alt_km, color='black', linestyle='--', label='Glued Signal', linewidth=1.5)
    
    ax1.axhline(alt_km[best_idx], color='crimson', linestyle=':', label='Gluing Core', linewidth=2)
    
    # ax1.set_xscale('log')
    ax1.set_ylim(0, min(15.0, alt_km[-1]))
    ax1.set_xlabel('Range Corrected Signal [a.u.]', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Altitude [km a.g.l.]', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.5, which='both', linestyle='--')
    
    ax2 = plt.subplot(gs[1])
    ax2.plot(lower_sig, alt_km, color='royalblue', label='Analog', marker='.', alpha=0.7)
    ax2.plot(upper_sig, alt_km, color='darkorange', label='PC', marker='.', alpha=0.7)
    ax2.plot(glued_sig, alt_km, color='black', linestyle='--', label='Glued', marker='x', markersize=4)
    
    min_idx = max(0, best_idx - window_size)
    max_idx = min(len(alt_km)-1, best_idx + int(window_size*1.5))
    ax2.set_ylim(alt_km[min_idx], alt_km[max_idx])
    
    if np.nanmax(glued_sig[min_idx:max_idx]) > 0:
        ax2.set_xlim(np.nanmin(glued_sig[min_idx:max_idx])*0.5, np.nanmax(glued_sig[min_idx:max_idx])*2.0)
    
    # ax2.set_xscale('log')
    ax2.set_xlabel('Range Corrected Signal [a.u.]', fontsize=12, fontweight='bold', labelpad=10)
    ax2.set_title(f"Transition Region Window", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.5, which='both', linestyle='--')
    
    add_footer_and_logos(fig, root_dir)
    plt.subplots_adjust(bottom=0.20, top=0.88) # Massive bottom margin for logos
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'QA_Gluing_{prefix}_{channel_base_name.replace(" ", "_")}.webp'), dpi=120)
    
    if config.get('processing', {}).get('interactive_qa', True): plt.show(block=True)
    plt.close(fig)

def plot_molecular_qa(alt_km, rcs, simulated_mol, fit_min_km, fit_max_km, config, channel_name, ds, root_dir, save_dir, prefix):
    date_title, date_footer = extract_datetime_strings(ds)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(rcs, alt_km, color='darkblue', label=f'Glued RCS ({channel_name})', alpha=0.8, linewidth=1.5)
    ax.plot(simulated_mol, alt_km, color='crimson', linestyle='--', label='Rayleigh Fit', linewidth=2.5)
    ax.axhspan(fit_min_km, fit_max_km, color='forestgreen', alpha=0.15, label='Calibration Region')
    
    ax.set_title(f"Molecular Calibration QA - {channel_name}\n{date_title}", fontsize=14, fontweight='bold', y=1.02)
    # ax.set_xscale('log')
    ax.set_ylim(0, min(25.0, alt_km[-1]))
    ax.set_xlabel('Range Corrected Signal [a.u.]', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Altitude [km a.g.l.]', fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.5, which='both', linestyle='--')
    
    add_footer_and_logos(fig, root_dir)
    plt.subplots_adjust(bottom=0.20, top=0.90) # Massive bottom margin
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'QA_Molecular_{prefix}_{channel_name.replace(" ", "_")}.webp'), dpi=120)
    
    if config.get('processing', {}).get('interactive_qa', True): plt.show(block=True)
    plt.close(fig)



def plot_kfs_results(alt_km, beta_mean, beta_std, ext_mean, ext_std, config, channel_name, save_dir, prefix, ds, root_dir):
    date_title, date_footer = extract_datetime_strings(ds)
    fig = plt.figure(figsize=[15, 7.5])
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.15)
    
    fig.suptitle(f"Aerosol Optical Properties (KFS Monte Carlo) - {channel_name}\n{date_title}", fontsize=15, fontweight='bold', y=0.96)
    
    color_map = {"532": "forestgreen", "355": "rebeccapurple", "1064": "crimson"}
    plot_color = color_map.get(next((k for k in color_map.keys() if k in channel_name), "black"), "black")

    # Limit plot to the troposphere/lower stratosphere where aerosols reside
    max_plot_alt = 15.0 
    valid_idx = alt_km <= max_plot_alt
    alt_cut = alt_km[valid_idx]
    scale_beta = 1e6
    
    # AXIS 1: Backscatter (Beta)
    ax1 = plt.subplot(gs[0])
    b_mean = beta_mean[valid_idx] * scale_beta
    b_std = beta_std[valid_idx] * scale_beta
    
    ax1.plot(b_mean, alt_cut, color=plot_color, linewidth=2.5, label=r'Backscatter ($\beta_{aer}$)')
    ax1.fill_betweenx(alt_cut, b_mean - b_std, b_mean + b_std, color=plot_color, alpha=0.25, edgecolor="none")
    ax1.set_xlabel(r'Aerosol Backscatter [$Mm^{-1} sr^{-1}$]', fontsize=13, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Altitude [km a.g.l.]', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6, which='both')
    
    ax1.set_xscale('log')
    # Dynamic scaling: prevents math domain errors on log scale for values <= 0
    b_min_valid = np.nanmin(b_mean[b_mean > 0]) if np.any(b_mean > 0) else 1e-4
    ax1.set_xlim(left=max(1e-4, b_min_valid * 0.1), right=np.nanmax(b_mean) * 2.0)
    ax1.legend(fontsize=12, loc='upper right')

    # AXIS 2: Extinction (Alpha)
    ax2 = plt.subplot(gs[1], sharey=ax1)
    e_mean = ext_mean[valid_idx] * scale_beta
    e_std = ext_std[valid_idx] * scale_beta
    
    ax2.plot(e_mean, alt_cut, color=plot_color, linewidth=2.5, label=r'Extinction ($\alpha_{aer}$)')
    ax2.fill_betweenx(alt_cut, e_mean - e_std, e_mean + e_std, color=plot_color, alpha=0.25, edgecolor="none")
    ax2.set_xlabel(r'Aerosol Extinction [$Mm^{-1}$]', fontsize=13, fontweight='bold', labelpad=10)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.grid(True, linestyle='--', alpha=0.6, which='both')
    
    ax2.set_xscale('log')
    e_min_valid = np.nanmin(e_mean[e_mean > 0]) if np.any(e_mean > 0) else 1e-3
    ax2.set_xlim(left=max(1e-3, e_min_valid * 0.1), right=np.nanmax(e_mean) * 2.0)
    ax2.legend(fontsize=12, loc='upper right')

    # FORMATTING & EXPORT
    add_footer_and_logos(fig, root_dir)
    plt.subplots_adjust(bottom=0.18, top=0.88) 
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'L2_OpticalProps_{prefix}_{channel_name.replace(" ", "_")}.webp'), dpi=120)
    
    if config.get('processing', {}).get('interactive_qa', True): 
        plt.show(block=True)
    plt.close(fig)
