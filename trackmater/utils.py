from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def setup_output_path(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def merge_dataframes_for_export(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(x, y, on=["trace", "dataid"], how="outer")


def process_defaults(kwargs: dict) -> dict:
    defaults = {
        "N": 3,
        "short": 8,
        "delta_t": 1,
        "mode": "ECDF",
        "n_pop": 2,
        "time_res": 1,
        "breaks": 100,
        "radius": 1.5,
        "msd_scale": "linlin",
    }
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    return kwargs


def find_log2_y_limits(x: np.ndarray | pd.Series) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    lo = min(0.5, np.min(x)) if len(x) > 0 else 0.5
    hi = max(2.0, np.max(x)) if len(x) > 0 else 2.0
    limit = max(abs(int(np.floor(np.log2(lo)))), int(np.ceil(np.log2(hi))))
    return (2.0 ** (-limit), 2.0**limit)
