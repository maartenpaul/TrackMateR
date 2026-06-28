from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import linregress

from ._types import TrackMateData


def _get_or_create_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is not None:
        return ax.get_figure(), ax
    fig, ax = plt.subplots(figsize=(5, 4))
    return fig, ax


def _extract_df_and_units(
    data: TrackMateData | pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, str]]:
    if isinstance(data, TrackMateData):
        return data.tracks, (data.calibration.spatial_unit, data.calibration.temporal_unit)
    return data, ("um", "s")


def plot_tracks(
    data: TrackMateData | pd.DataFrame,
    summary: bool = False,
    alpha_level: float = 0.5,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    df, units = _extract_df_and_units(data)
    fig, ax = _get_or_create_axes(ax)

    if not summary:
        alpha_level = 0.5

    group_col = "dataid" if summary and "dataid" in df.columns else "trace"
    for _, grp in df.groupby(["trace"] if not summary else ["dataid", "trace"]):
        ax.plot(grp["x"], grp["y"], alpha=alpha_level, linewidth=0.5)

    lim = max(df["x"].max(), df["y"].max())
    ax.set_xlim(0, lim)
    ax.set_ylim(lim, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    if ax.get_subplotspec() is None or len(ax.get_figure().get_axes()) == 1:
        fig.tight_layout()
    return fig, ax


def plot_displacement_over_time(
    data: TrackMateData | pd.DataFrame,
    summary: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    df, units = _extract_df_and_units(data)
    fig, ax = _get_or_create_axes(ax)

    group_cols = ["dataid", "trace"] if summary and "dataid" in df.columns else ["trace"]
    for _, grp in df.groupby(group_cols):
        y_vals = grp["displacement"]
        if len(grp) >= 50:
            y_vals = y_vals.rolling(20, center=True, min_periods=1).mean()
        ax.plot(grp["t"], y_vals, alpha=0.1, linewidth=0.5, color="steelblue")

    sorted_df = df.sort_values("t")
    if len(sorted_df) > 10:
        from scipy.signal import savgol_filter

        window = min(51, len(sorted_df) // 2 * 2 + 1)
        if window >= 5:
            t_mean = sorted_df.groupby("t")["displacement"].mean()
            smooth = savgol_filter(t_mean.values, window, 3)
            ax.plot(t_mean.index, smooth, color="C0", linewidth=2)

    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Time ({units[1]})")
    ax.set_ylabel(f"Displacement ({units[0]})")
    if ax.get_subplotspec() is None or len(ax.get_figure().get_axes()) == 1:
        fig.tight_layout()
    return fig, ax


def plot_cumulative_distance(
    data: TrackMateData | pd.DataFrame,
    summary: bool = False,
    alpha_level: float = 0.1,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    df, units = _extract_df_and_units(data)
    fig, ax = _get_or_create_axes(ax)

    group_cols = ["dataid", "trace"] if summary and "dataid" in df.columns else ["trace"]
    for _, grp in df.groupby(group_cols):
        ax.plot(grp["track_duration"], grp["cumulative_distance"], alpha=alpha_level, linewidth=0.5)

    ax.set_xlabel(f"Time ({units[1]})")
    ax.set_ylabel(f"Cumulative distance ({units[0]})")
    if ax.get_subplotspec() is None or len(ax.get_figure().get_axes()) == 1:
        fig.tight_layout()
    return fig, ax


def _plot_histogram(
    values: pd.Series,
    x_label: str,
    y_label: str = "Frequency",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, float]:
    fig, ax = _get_or_create_axes(ax)
    median_val = float(values.median())
    n_bins = max(int(1 + np.log2(len(values))), 10)

    ax.hist(values, bins=n_bins, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.text(
        0.98, 0.98, f"{median_val:.3f}",
        transform=ax.transAxes, fontsize=8,
        ha="right", va="top",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if ax.get_subplotspec() is None or len(ax.get_figure().get_axes()) == 1:
        fig.tight_layout()
    return fig, ax, median_val


def plot_displacement_hist(
    data: TrackMateData | pd.DataFrame,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, float]:
    df, units = _extract_df_and_units(data)
    return _plot_histogram(df["displacement"], f"Displacement ({units[0]})", ax=ax)


def plot_intensity_hist(
    data: TrackMateData | pd.DataFrame,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, float]:
    df, units = _extract_df_and_units(data)

    intensity_col = None
    if "mean_intensity" in df.columns:
        intensity_col = "mean_intensity"
    else:
        candidates = [c for c in df.columns if c.startswith("mean_intensity_ch")]
        if candidates:
            intensity_col = candidates[0]

    if intensity_col is None:
        fig, ax = _get_or_create_axes(ax)
        return fig, ax, float("nan")

    int_df = df.groupby("trace")[intensity_col].max()
    return _plot_histogram(int_df, "Intensity (AU)", ax=ax)


def plot_duration_hist(
    data: TrackMateData | pd.DataFrame,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, float]:
    df, units = _extract_df_and_units(data)
    dur = df.groupby("trace")["track_duration"].max()
    return _plot_histogram(dur, f"Duration ({units[1]})", ax=ax)


def plot_speed(
    data: TrackMateData | pd.DataFrame,
    summary: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, float]:
    df, units = _extract_df_and_units(data)

    group_cols = ["dataid", "trace"] if summary and "dataid" in df.columns else ["trace"]
    speed_df = df.groupby(group_cols).agg(
        cumdist=("cumulative_distance", "max"),
        cumtime=("track_duration", "max"),
    )
    speed_df["speed"] = speed_df["cumdist"] / speed_df["cumtime"]
    speed_df = speed_df.dropna(subset=["speed"])
    speed_df = speed_df[np.isfinite(speed_df["speed"])]

    return _plot_histogram(speed_df["speed"], f"Speed ({units[0]}/{units[1]})", ax=ax)


def plot_alpha(
    alpha_df: pd.DataFrame,
    median_alpha: float | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    fig, ax = _get_or_create_axes(ax)

    values = alpha_df["alpha"].dropna()
    valid = values[(values >= -4) & (values <= 4)]
    if len(valid) > 0:
        bins = np.arange(valid.min() - 0.05, valid.max() + 0.15, 0.1)
    else:
        bins = 30
    ax.hist(valid, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)

    if median_alpha is not None:
        ax.text(
            0.98, 0.98, f"{median_alpha:.3f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
        )

    ax.set_xlabel("alpha (log2)")
    ax.set_ylabel("Frequency")
    if ax.get_subplotspec() is None or len(ax.get_figure().get_axes()) == 1:
        fig.tight_layout()
    return fig, ax


def plot_dee(
    dee_df: pd.DataFrame,
    median_dee: float | None = None,
    x_label: str = "D",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    fig, ax = _get_or_create_axes(ax)

    values = dee_df["dee"].dropna()
    n_bins = max(int(values.max() / 0.01), 10) if len(values) > 0 and values.max() > 0 else 10
    ax.hist(values, bins=min(n_bins, 100), color="steelblue", edgecolor="white", linewidth=0.5)

    if median_dee is not None:
        ax.text(
            0.98, 0.98, f"{median_dee:.3f}",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
        )

    ax.set_xlim(left=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    if ax.get_subplotspec() is None or len(ax.get_figure().get_axes()) == 1:
        fig.tight_layout()
    return fig, ax


def plot_msd(
    msd_summary: pd.DataFrame,
    units: tuple[str, str] = ("um", "s"),
    error_bars: bool = False,
    xlog: bool = False,
    ylog: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, float]:
    fig, ax = _get_or_create_axes(ax)

    dee = float("nan")
    if msd_summary["mean"].isna().all() or msd_summary["t"].isna().all():
        ax.axhline(0, color="grey")
    else:
        head = msd_summary.head(4).dropna(subset=["mean", "t"])
        if len(head) >= 2:
            res = linregress(head["t"], head["mean"])
            slope, intercept = res.slope, res.intercept
            pred = slope * msd_summary["t"] + intercept
            dee = float(pred.iloc[0] / (4 * msd_summary["t"].iloc[0]))
            ax.plot(msd_summary["t"], pred, "r--", linewidth=1)
        else:
            pred = msd_summary["t"] * 0

    if error_bars:
        ax.errorbar(
            msd_summary["t"], msd_summary["mean"],
            yerr=msd_summary["sd"], fmt="o-", markersize=3, linewidth=1,
        )
    else:
        ax.plot(msd_summary["t"], msd_summary["mean"], linewidth=1)

    ax.text(
        0.02, 0.98, f"D = {dee:.3f}",
        transform=ax.transAxes, fontsize=8, ha="left", va="top",
    )

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    ax.set_xlabel(f"Time ({units[1]})")
    ax.set_ylabel("MSD")
    if ax.get_subplotspec() is None or len(ax.get_figure().get_axes()) == 1:
        fig.tight_layout()
    return fig, ax, dee


def plot_multi_msd(
    msd_df: pd.DataFrame,
    xlog: bool = False,
    ylog: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, pd.DataFrame]:
    fig, ax = _get_or_create_axes(ax)

    if "mean" in msd_df.columns:
        val_col = "value"
        msd_df = msd_df.rename(columns={"mean": "value"})
    else:
        val_col = "value"

    datasets = msd_df["dataid"].unique() if "dataid" in msd_df.columns else []
    if len(datasets) == 0:
        empty = pd.DataFrame(columns=["t", "mean", "sd", "n"])
        return fig, ax, empty

    t1 = msd_df[msd_df["size"] == 1] if "size" in msd_df.columns else msd_df
    min_t = t1["t"].min()
    max_t = msd_df["t"].max()
    steps = int(np.ceil(max_t / min_t)) if min_t > 0 else 1
    t_common = np.arange(1, steps + 1) * min_t

    all_interp = []
    for did in datasets:
        sub = msd_df[msd_df["dataid"] == did].dropna(subset=["t", val_col])
        if len(sub) < 3:
            continue
        interp_vals = np.interp(t_common, sub["t"], sub[val_col], left=np.nan, right=np.nan)
        idf = pd.DataFrame({"t": t_common, "value": interp_vals, "dataid": did})
        all_interp.append(idf)
        ax.plot(sub["t"], sub[val_col], color="blue", alpha=0.5, linewidth=0.5)

    if all_interp:
        all_interp_df = pd.concat(all_interp, ignore_index=True)
        msd_mean = all_interp_df.groupby("t").agg(
            mean=("value", "mean"), sd=("value", "std"), n=("value", "count")
        ).reset_index()

        min_n = max(1, int(np.ceil(msd_mean["n"].max() / 3)))
        cutoff_rows = msd_mean.iloc[1:]
        cutoff_rows = cutoff_rows[cutoff_rows["n"] < min_n]
        max_x = cutoff_rows["t"].min() if len(cutoff_rows) > 0 else None

        ax.fill_between(
            msd_mean["t"], msd_mean["mean"] - msd_mean["sd"],
            msd_mean["mean"] + msd_mean["sd"], alpha=0.2, color="grey",
        )
        ax.plot(msd_mean["t"], msd_mean["mean"], color="black", linewidth=1.5)

        head = msd_mean.head(4).dropna(subset=["mean", "t"])
        if len(head) >= 2:
            res = linregress(head["t"], head["mean"])
            pred = res.slope * msd_mean["t"] + res.intercept
            dee = float(pred.iloc[0] / (4 * msd_mean["t"].iloc[0]))
            ax.plot(msd_mean["t"], pred, "r--", linewidth=1)
            ax.text(0.02, 0.98, f"D = {dee:.3f}", transform=ax.transAxes, fontsize=8, ha="left", va="top")
        else:
            dee = float("nan")

        if max_x is not None and not xlog:
            ax.set_xlim(0, max_x)
    else:
        msd_mean = pd.DataFrame(columns=["t", "mean", "sd", "n"])

    if xlog:
        ax.set_xscale("log")
    else:
        ax.set_ylim(bottom=0)
    if ylog:
        ax.set_yscale("log")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MSD")
    if ax.get_subplotspec() is None or len(ax.get_figure().get_axes()) == 1:
        fig.tight_layout()
    return fig, ax, msd_mean


def plot_neighbours(
    density_df: pd.DataFrame,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, float]:
    return _plot_histogram(density_df["density"], "Track density", ax=ax)


def plot_fd(
    fd_df: pd.DataFrame,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, float]:
    values = fd_df["fd"].dropna()
    fig, ax = _get_or_create_axes(ax)
    median_val = float(values.median())
    n_bins = max(int(1 + np.log2(len(values))), 30) if len(values) > 0 else 30

    if len(values) > 0 and values.std() < 0.001:
        bins = np.linspace(values.mean() - 1, values.mean() + 1, 30)
    else:
        bins = n_bins

    ax.hist(values, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.text(0.98, 0.98, f"{median_val:.3f}", transform=ax.transAxes, fontsize=8, ha="right", va="top")
    ax.set_xlabel("Fractal dimension")
    ax.set_ylabel("Frequency")
    if ax.get_subplotspec() is None or len(ax.get_figure().get_axes()) == 1:
        fig.tight_layout()
    return fig, ax, median_val


def plot_width(
    fd_df: pd.DataFrame,
    units: tuple[str, str] = ("um", "s"),
    ax: Axes | None = None,
) -> tuple[Figure, Axes, float]:
    return _plot_histogram(fd_df["wide"].dropna(), f"Maximum width ({units[0]})", ax=ax)
