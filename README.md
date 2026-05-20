# MILGRAU 🌩️

**Multi-Indexed LALINET GeneRAlized and Unified algorithm**

MILGRAU is a Python-based atmospheric lidar processing suite developed for the **SPU-Lidar Station** at **IPEN/USP, São Paulo, Brazil**.

The system is designed to process raw Licel lidar measurements into physically traceable atmospheric products, from raw signal standardization to range-corrected signals and, under development, Level 2 aerosol optical-property retrievals.

---

## Scientific purpose

Elastic and Raman lidar systems measure atmospheric backscattered radiation as a function of time and range. These raw signals are affected by instrumental response, photon-counting limitations, analog offsets, background radiation, dark current, incomplete overlap, clouds and atmospheric variability.

MILGRAU organizes the treatment of these signals into processing levels:

1. **Level 0** — raw Licel measurements are parsed, quality-controlled and standardized into NetCDF files compatible with SCC-style processing.
2. **Level 1** — instrumental corrections are applied and Range Corrected Signals are generated with propagated uncertainties.
3. **Visualization** — Level 1 products are rendered as quicklooks and mean profiles for scientific inspection.
4. **Level 2** — aerosol optical properties will be retrieved through molecular calibration, signal gluing and Klett-Fernald-Sasano inversion.

The main scientific objective is to provide a transparent processing framework for the SPU-Lidar Station that can support both operational monitoring and research-grade atmospheric analysis.

---

## Processing modules

| Module | Level | Scientific role |
|---|---:|---|
| **LIBIDS** | Level 0 | Parses raw Licel binary files, builds measurement inventories, associates dark-current acquisitions, applies basic acquisition quality control and writes SCC-compatible Level 0 NetCDF files. |
| **LIPANCORA** | Level 1 | Applies detector and instrumental corrections, propagates uncertainties, computes corrected signal and Range Corrected Signal, estimates PBL height and integrates radiosonde thermodynamic information. |
| **LIRACOS** | Visualization | Generates Range Corrected Signal quicklooks, mean profiles and uncertainty bands for Level 1 scientific inspection. |
| **LEBEAR** | Level 2 | Under development. Intended for Analog/Photon Counting gluing, Rayleigh molecular calibration, cloud screening and Klett-Fernald-Sasano aerosol backscatter/extinction retrieval. |

---


## Level 0 — LIBIDS

LIBIDS converts raw Licel measurements into standardized Level 0 NetCDF files.

Main tasks:

- scan raw measurement folders;
- identify valid measurement and dark-current files;
- parse Licel headers and binary payloads;
- classify acquisitions into `am`, `pm` and `nt`;
- associate dark-current profiles with nearby measurements;
- reject invalid laser-shot acquisitions;
- fetch or fallback to surface meteorological metadata;
- write SCC-compatible Level 0 NetCDF products.

The Level 0 product stores raw lidar data as:

```text
Raw_Lidar_Data(time, channels, points)
```

and includes metadata such as:

```text
Measurement_ID
System
Latitude_degrees_north
Longitude_degrees_east
Accumulated_Shots
RawData_Start_Date
RawData_Start_Time_UT
RawData_Stop_Time_UT
Temperature_C
Pressure_hPa
CloudCover_percent
Source_File_Count
Source_Files
```

When available, dark-current measurements are stored as:

```text
Background_Profile(time_bck, channels, points)
```

---

## Level 1 — LIPANCORA

LIPANCORA transforms Level 0 raw signals into corrected Level 1 lidar products.

The correction sequence includes:

1. dark-current subtraction;
2. photon-counting normalization;
3. non-paralyzable dead-time correction;
4. bin-shift alignment;
5. sky-background subtraction;
6. uncertainty propagation;
7. range correction.

The Level 1 NetCDF explicitly stores both the corrected signal and the Range Corrected Signal:

```text
corrected_signal(time, channel, altitude)
corrected_signal_error(time, channel, altitude)

range_corrected_signal(time, channel, altitude)
range_corrected_signal_error(time, channel, altitude)
```

This distinction is scientifically important.

`corrected_signal` is the lidar signal after instrumental corrections but before multiplication by range squared.

`range_corrected_signal` is defined as:

```text
RCS(z) = corrected_signal(z) · z²
```

and is the primary signal used for Level 1 visualization and future Level 2 inversion.

LIPANCORA also enriches the Level 1 product with atmospheric diagnostics:

```text
PBL_Height_km(time)
Radiosonde_Temperature_K(radiosonde_altitude)
Radiosonde_Pressure_hPa(radiosonde_altitude)
```

when radiosonde data are available.

---

## Atmospheric diagnostics

### Planetary Boundary Layer

The PBL height is estimated from the vertical gradient of a smoothed Range Corrected Signal profile. The method searches for the strongest physically meaningful negative gradient within a configured altitude interval.

Typical configuration keys:

```yaml
physics:
  pbl_min_search_m: 500.0
  pbl_max_search_m: 4000.0
  pbl_smooth_bins: 15
```

### Tropopause

When radiosonde data are available, MILGRAU estimates:

- **Cold Point Tropopause**;
- **Lapse Rate Tropopause**, following a WMO-style lapse-rate criterion.

These diagnostics are stored as global Level 1 attributes and can be overlaid on visual products.

---

## Molecular atmosphere and Rayleigh calculations

The Level 2 development path depends on a molecular reference profile.

MILGRAU includes routines to compute molecular backscatter and extinction from pressure and temperature profiles:

```text
β_mol(z, λ)
α_mol(z, λ)
```

where pressure and temperature may come from radiosonde data or a fallback standard atmosphere.

For Rayleigh scattering, the molecular lidar ratio is treated as:

```text
S_mol = 8π / 3 sr
```

The molecular profile is required for Rayleigh calibration and for the Klett-Fernald-Sasano inversion.

---

## Signal gluing

LEBEAR will use Analog and Photon Counting channels together when both are available for the same wavelength.

The gluing procedure is designed to search for a stable overlap region where the two signals are physically compatible. The selection criteria include:

- high Pearson correlation;
- stable linear mapping between Analog and Photon Counting signals;
- acceptable relative intercept;
- low residual spread;
- compatible min/max envelope after scaling.

The glued signal is intended to preserve near-field analog dynamic range while retaining photon-counting sensitivity at higher altitudes.

---

## Klett-Fernald-Sasano inversion

The planned Level 2 retrieval uses a Klett-Fernald-Sasano-type inversion to estimate aerosol optical properties from calibrated elastic lidar signals.

The expected Level 2 products include:

```text
aerosol_backscatter(time, wavelength, altitude)
aerosol_backscatter_error(time, wavelength, altitude)

aerosol_extinction(time, wavelength, altitude)
aerosol_extinction_error(time, wavelength, altitude)
```

The inversion will use:

- molecular backscatter and extinction;
- calibrated Range Corrected Signal;
- configured aerosol lidar ratios;
- molecular reference altitude selection;
- Monte Carlo perturbations for uncertainty estimation.

---

## Configuration

MILGRAU is configured through `config.yaml`.

The configuration file contains:

- directory paths;
- station metadata;
- instrument constants;
- channel-specific dead-time corrections;
- bin-shift values;
- background offsets;
- PBL search limits;
- radiosonde settings;
- visualization settings;
- Level 2 inversion parameters;
- monthly aerosol lidar ratios;
- cloud-screening parameters.

Physical constants, correction thresholds and processing options should be configured externally and not hardcoded in processing scripts.

Example channel configuration:

```yaml
physics:
  channels:
    "532.PC": [0.0035, -3, 0.0015]
    "532.AN": [0.0000,  6, 0.0000]
```

where the values are:

```text
[deadtime_us, bin_shift, background_offset]
```

---

# MILGRAU Installation

## 1. Install Git and Python

### Debian

```bash
sudo apt update
sudo apt install git python3 python3-pip python3-venv
```

### Arch 

```bash
sudo pacman -Syu
sudo pacman -S git python python-pip
```

### Fedora

```bash
sudo dnf install git python3 python3-pip
```

## 2. Download MILGRAU

Clone the repository:

```bash
git clone https://github.com/spulidar/milgrau.git
cd milgrau
```

## 3. Create a virtual environment

Inside the MILGRAU directory:

```bash
python -m venv .venv
```

Activate the environment:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```cmd
.venv\Scripts\Activate.bat
```
---

## 4. Install MILGRAU

For users who only want to run the program:

```bash
pip install .
```

---

## 5. Run modules

After installation, you can run:

```bash
milgrau-libids
milgrau-lipancora
milgrau-liracos
milgrau-lebear
```

---

### Developer mode

Use this option only if you are going to edit the MILGRAU source code:

```bash
pip install -e .
```
---

## Update MILGRAU

To download new changes from GitHub:

```bash
git pull
```

If the package was not installed in developer mode, reinstall it:

```bash
pip install .
```

---

## Basic usage

Activate the virtual environment whenever you want to use MILGRAU:

```bash
cd milgrau
source .venv/bin/activate
```

Then run:

```bash
milgrau-libids
```

---

## TLDR

```bash
sudo apt install git python3 python3-pip python3-venv
git clone https://github.com/spulidar/milgrau.git
cd milgrau
python -m venv .venv
source .venv/bin/activate
pip install .
milgrau-libids
```

---

## Authors

- Luisa Mello
- Alexandre C. Yoshida
- Alexandre Cacheffo
- Fábio J. S. Lopes
