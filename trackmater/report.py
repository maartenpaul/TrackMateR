from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ._types import (
    FDResult,
    JDResult,
    MSDResult,
    SummaryStats,
    TrackDensityResult,
    TrackMateData,
)
from .fitting import fit_jd
from .plotting import (
    plot_alpha,
    plot_cumulative_distance,
    plot_dee,
    plot_displacement_hist,
    plot_displacement_over_time,
    plot_duration_hist,
    plot_fd,
    plot_intensity_hist,
    plot_msd,
    plot_multi_msd,
    plot_neighbours,
    plot_speed,
    plot_tracks,
    plot_width,
)

logger = logging.getLogger(__name__)


def make_summary_report(
    data: TrackMateData,
    msd_result: MSDResult,
    jd_result: JDResult | None = None,
    td_result: TrackDensityResult | None = None,
    fd_result: FDResult | None = None,
    title: str = "",
    subtitle: str = "",
    msd_scale: str = "linlin",
    summary_mode: bool = False,
) -> tuple[Figure, SummaryStats | pd.DataFrame]:
    units = (data.calibration.spatial_unit, data.calibration.temporal_unit)

    xlog = msd_scale in ("loglog", "linlog")
    ylog = msd_scale in ("loglog", "loglin")

    fig = plt.figure(figsize=(25 / 2.54, 19 / 2.54))
    gs = GridSpec(4, 5, figure=fig, hspace=0.5, wspace=0.5)

    summary = summary_mode and "dataid" in data.tracks.columns
    alpha_level = 0.5
    if summary:
        n_data = data.tracks["dataid"].nunique()
        alpha_level = 0.5 if n_data < 4 else (0.25 if n_data < 8 else 0.1)

    # Row 0-1, Col 0-1: all tracks
    ax_tracks = fig.add_subplot(gs[0:2, 0:2])
    plot_tracks(data, summary=summary, alpha_level=alpha_level, ax=ax_tracks)

    # Row 0, Col 2: displacement over time
    ax_dot = fig.add_subplot(gs[0, 2])
    plot_displacement_over_time(data, summary=summary, ax=ax_dot)

    # Row 0, Col 3: displacement histogram
    ax_dhist = fig.add_subplot(gs[0, 3])
    _, _, median_disp = plot_displacement_hist(data, ax=ax_dhist)

    # Row 0, Col 4: duration histogram
    ax_durhist = fig.add_subplot(gs[0, 4])
    _, _, median_dur = plot_duration_hist(data, ax=ax_durhist)

    # Row 1, Col 2: cumulative distance
    ax_cumdist = fig.add_subplot(gs[1, 2])
    plot_cumulative_distance(data, summary=summary, alpha_level=alpha_level, ax=ax_cumdist)

    # Row 1, Col 3: speed
    ax_speed = fig.add_subplot(gs[1, 3])
    _, _, median_speed = plot_speed(data, summary=summary, ax=ax_speed)

    # Row 1, Col 4: intensity histogram
    ax_int = fig.add_subplot(gs[1, 4])
    _, _, median_int = plot_intensity_hist(data, ax=ax_int)

    # Row 2, Col 0: MSD
    ax_msd = fig.add_subplot(gs[2, 0])
    dee = float("nan")
    msd_summary_out = None
    if summary and msd_result is not None:
        _, _, msd_summary_out = plot_multi_msd(
            msd_result.summary, xlog=xlog, ylog=ylog, ax=ax_msd
        )
    elif msd_result is not None:
        _, _, dee = plot_msd(msd_result.summary, units=units, xlog=xlog, ylog=ylog, ax=ax_msd)

    # Row 2, Col 1: D histogram
    ax_dee = fig.add_subplot(gs[2, 1])
    alphas = msd_result.alpha.dropna(subset=["alpha"]) if msd_result is not None else pd.DataFrame()
    if not alphas.empty:
        valid = alphas[(alphas["alpha"] <= 4) & (alphas["alpha"] >= -4)]
        median_alpha_val = 2 ** float(valid["alpha"].median()) if not valid.empty else float("nan")
        median_dee = float(valid["dee"].median()) if not valid.empty else float("nan")
        d_label = f"D ({units[0]}²/{units[1]})"
        plot_dee(valid, median_dee=median_dee, x_label=d_label, ax=ax_dee)
    else:
        median_alpha_val = float("nan")
        median_dee = float("nan")

    # Row 2, Col 2: alpha histogram
    ax_alpha = fig.add_subplot(gs[2, 2])
    if not alphas.empty:
        valid_alpha = alphas[(alphas["alpha"] <= 4) & (alphas["alpha"] >= -4)]
        plot_alpha(valid_alpha, median_alpha=median_alpha_val, ax=ax_alpha)

    # Row 2, Col 3: estimator D
    ax_estdee = fig.add_subplot(gs[2, 3])
    cves = msd_result.cve.dropna(subset=["dee"]) if msd_result is not None else pd.DataFrame()
    median_estdee = float("nan")
    if not cves.empty:
        median_estdee = float(cves["dee"].median())
        est_label = f"Estimator D ({units[0]}²/{units[1]})"
        plot_dee(cves, median_dee=median_estdee, x_label=est_label, ax=ax_estdee)

    # Row 2, Col 4 + Row 3, Col 0: jump distance
    ax_jd = fig.add_subplot(gs[2, 4])
    if jd_result is not None:
        try:
            jd_fit = fit_jd(jd_result)
            if jd_fit.fig is not None:
                plt.close(jd_fit.fig)
            ax_jd.clear()
            from .fitting import fit_jd as _fit_jd_inner

            jd_fit2 = _replot_jd_on_ax(jd_result, ax_jd)
        except Exception:
            pass

    # Row 3, Col 0: neighbours
    ax_neigh = fig.add_subplot(gs[3, 0])
    median_density = float("nan")
    if td_result is not None and not td_result.data.empty:
        _, _, median_density = plot_neighbours(td_result.data, ax=ax_neigh)

    # Row 3, Col 1: fractal dimension
    ax_fd = fig.add_subplot(gs[3, 1])
    median_fd = float("nan")
    if fd_result is not None and not fd_result.data.empty:
        _, _, median_fd = plot_fd(fd_result.data, ax=ax_fd)

    # Row 3, Col 2: width
    ax_width = fig.add_subplot(gs[3, 2])
    median_width = float("nan")
    if fd_result is not None and not fd_result.data.empty:
        _, _, median_width = plot_width(fd_result.data, units=units, ax=ax_width)

    # Remove unused axes
    for row_col in [(3, 3), (3, 4)]:
        ax_empty = fig.add_subplot(gs[row_col[0], row_col[1]])
        ax_empty.set_visible(False)

    if title or subtitle:
        fig.suptitle(f"{title}\n{subtitle}" if subtitle else title, fontsize=10, y=1.01)

    if summary_mode:
        return fig, msd_summary_out if msd_summary_out is not None else pd.DataFrame()

    stats = SummaryStats(
        alpha=median_alpha_val,
        speed=median_speed,
        intensity=median_int if not np.isnan(median_int) else 0.0,
        duration=median_dur,
        dee=dee,
        displacement=median_disp,
        neighbours=median_density,
        fd=median_fd,
        width=median_width,
    )
    return fig, stats


def _replot_jd_on_ax(jd_result: JDResult, ax) -> None:
    import warnings

    import numpy as np
    from scipy.optimize import curve_fit

    params = jd_result.params
    jumps = jd_result.jump_distances["jump"].values
    time_res = params.time_res
    breaks = params.breaks
    units = (params.spatial_unit, params.temporal_unit)

    counts, bin_edges = np.histogram(jumps, bins=breaks)
    mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts_cum = np.cumsum(counts) / np.sum(counts)

    if params.mode == "ECDF":
        ax.scatter(mids, counts_cum, s=10, c="black", zorder=3)
        ax.set_xlim(0, mids.max())
        ax.set_ylim(0, 1.4)
        ax.set_ylabel("Frequency")
    else:
        ax.bar(mids, counts, width=np.diff(bin_edges).mean(), color="lightgrey", edgecolor="darkgrey")
        ax.set_ylabel("Counts")

    ax.set_xlabel(f"Displacement ({units[0]})")

    from .fitting import _auto_guess, _fit_1pop, _fit_2pop, _fit_3pop

    init = params.init or _auto_guess(jumps, params.n_pop, params.mode, time_res, breaks)
    x_fit = np.linspace(0, mids.max(), 10 * len(mids))

    try:
        if params.n_pop == 1:
            coeffs, fit_df = _fit_1pop(mids, counts, counts_cum, x_fit, params.mode, init, time_res)
        elif params.n_pop == 2:
            coeffs, fit_df = _fit_2pop(mids, counts, counts_cum, x_fit, params.mode, init, time_res)
        else:
            coeffs, fit_df = _fit_3pop(mids, counts, counts_cum, x_fit, params.mode, init, time_res)

        melted = fit_df.melt(id_vars="x", var_name="variable", value_name="value")
        for _, grp in melted.groupby("variable"):
            ax.plot(grp["x"], grp["value"], linewidth=1)

        from .fitting import _format_coefficients
        ax.text(0.02, 0.98, _format_coefficients(coeffs), transform=ax.transAxes,
                fontsize=6, va="top", ha="left")
    except Exception:
        pass


def report_dataset(
    data: TrackMateData,
    n: int = 3,
    short: int = 8,
    delta_t: int = 1,
    n_pop: int = 2,
    radius: float = 1.5,
    **kwargs,
) -> Figure:
    from .analysis import calculate_fd, calculate_jd, calculate_msd, calculate_track_density

    msd_result = calculate_msd(data.tracks, n=n, short=short)
    jd_result = calculate_jd(data, delta_t=delta_t, n_pop=n_pop, **kwargs)
    td_result = calculate_track_density(data, radius=radius)
    fd_result = calculate_fd(data)

    fig, _ = make_summary_report(
        data, msd_result, jd_result, td_result, fd_result,
        msd_scale=kwargs.get("msd_scale", "linlin"),
    )
    return fig
