"""Time classification helpers for MILGRAU acquisition grouping."""

from __future__ import annotations

from typing import Any

import pandas as pd


def classify_period(local_dt: Any) -> str:
    """Classify a local timestamp into 'am', 'pm' or 'nt'."""
    if 6 <= local_dt.hour < 12:
        return "am"
    if 12 <= local_dt.hour < 18:
        return "pm"
    return "nt"


def get_night_date(local_dt: Any) -> Any:
    """Assign post-midnight night measurements to the previous civil date."""
    if local_dt.hour < 6:
        return local_dt - pd.Timedelta(days=1)
    return local_dt
