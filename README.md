# MILGRAU 🌩️
**Multi-Indexed Lalinet GeneRAlized and Unified algorithm**

MILGRAU was developed for the **SPU-Lidar Station** (São Paulo, Brazil), aiming to provide a fully automated, mathematically rigorous, and reproducible workflow. It is a robust, Python-based processing suite designed to handle raw atmospheric Lidar signals, transforming Level 0 binary data into Level 2 inverted optical properties.

The pipeline adheres to the physical and statistical guidelines established by the **Single Calculus Chain (SCC)** of the **LALINET/EARLINET** networks.

---

## Architecture & Modules

The suite is divided into modular processing steps:

| Module | Level | Description |
| :--- | :--- | :--- |
| **`01-LIBIDS`** | Level 0 | Scans raw Licel binary data, sanitizes files, corrects timezones (UTC), flags Dark Current, and exports to NetCDF. |
| **`02-LIPANCORA`** | Level 1 | The physics engine. Dead-time correction, bin-shift alignment, DC/Sky background subtraction, and RCS generation with error tensors. |
| **`03-LIRACOS`** | Viz | Renders high-quality RCS colormaps (quicklooks) and 1D mean profiles with $1\sigma$ shaded error bands. |
| **`04-LIMP`** | Web | Syncs output to Cloudflare R2 buckets and generates static HTML dashboards via GitHub Pages. |
| **`05-LEBEAR`** | Level 2 | *Work in Progress.* Signal gluing (Analog + PC), Rayleigh molecular calculation, and KFS optical property inversion. |

---

## Project Structure

```text
milgrau/
├── atmospheric_lidar/           # Lidar processing backend (Licel, Diva, Generic)
├── atmospheric_lidar_parameters/# System-specific NetCDF parameters
├── functions/                   # Core processing libraries
│   ├── core_io.py               # Input/Output handling
│   ├── physics_utils.py         # Physical constants and algorithms
│   └── viz_utils.py             # Plotting and dashboard logic
├── img/                         # Logos and documentation assets
├── logs/                        # Pipeline execution logs
├── ql-measurements/             # Generated dashboards and calendar
├── 01-LIBIDS.py                 # Level 0 processing script
├── 02-LIPANCORA.py              # Level 1 physics engine
├── 03-LIRACOS.py                # Quicklooks and 1D visualization
├── 04-LIMP.py                   # Cloud publishing and dashboarding
├── config.yaml                  # Master configuration panel
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/spulidar/milgrau.git
   cd milgrau
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate  
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration & Usage

MILGRAU uses a centralized `config.yaml` file to manage paths, thresholds, and hardware constants (like dead-time and bin-shift per channel). **Never hardcode these variables** in the processing scripts.

1. **Edit Configuration:** Adjust `config.yaml` to match your directory structure and Lidar system parameters.
2. **Data Placement:** Ensure your raw data is placed in the configured input directory.
3. **Run the pipeline sequentially:**
   ```bash
   python 01-LIBIDS.py
   python 02-LIPANCORA.py
   python 03-LIRACOS.py
   ```

---

## Security Note

If using the `04-LIMP.py` module to publish to the SPU Lidar dashboard, ensure your `credentials.py` file is correctly configured and strictly added to your `.gitignore`. **Never commit API keys to version control.**

---

## Authors

* **Fábio J. S. Lopes**
* **Alexandre C. Yoshida**
* **Alexandre Cacheffo**
* **Luisa Mello**
