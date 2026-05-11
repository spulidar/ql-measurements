"""Shared visual style helpers for MILGRAU figures."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np

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
def read_logo_image(path_str: str) -> np.ndarray:
    """Read and cache a logo image from disk."""
    return mpimg.imread(path_str)


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


def get_output_settings(config: dict[str, Any]) -> tuple[str, int]:
    """Extract output format and DPI from visualization configuration."""
    viz_cfg = config.get("visualization", {}) or {}
    output_format = str(viz_cfg.get("output_format", "webp")).lstrip(".").lower()
    dpi = int(viz_cfg.get("dpi", 120))
    return output_format, dpi


def add_footer_and_logos(fig: Any, root_dir: str | Path) -> None:
    """Add SPU-Lidar footer and institutional logos to a Matplotlib figure."""
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
        img = read_logo_image(str(logo_path))
        img_h, img_w = img.shape[:2]
        width = height * (img_w / img_h)
        x_left = x_right - width
        ax_logo = fig.add_axes([x_left, y_pos, width, height], zorder=12)
        ax_logo.imshow(img)
        ax_logo.axis("off")
        x_right = x_left - spacing
