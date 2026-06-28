from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import linregress

from ._types import (
    FDResult,
    JDParams,
    JDResult,
    MSDResult,
    TrackDensityResult,
    TrackMateData,
)

logger = logging.getLogger(__name__)


def _build_displacement_matrices(
    df: pd.DataFrame, short: int = 0
) -> tuple[np.ndarray, np.ndarray, list[str], float, int] | None:
    trace_list = list(df["trace"].unique())

    t_offsets: dict[str, np.ndarray] = {}
    for trace_id in trace_list:
        mask = df["trace"] == trace_id
        frames = df.loc[mask, "frame"].values
        frame0 = frames[0]
        t_offsets[trace_id] = frames[1:] - frame0

    if short > 0:
        filtered = [t for t in trace_list if len(t_offsets[t]) > 0 and t_offsets[t].max() >= short]
        if not filtered:
            logger.warning("No traces are long enough. Reverting to short=0.")
        else:
            trace_list = filtered

    if not trace_list:
        return None

    t_list_max = max(t_offsets[t].max() for t in trace_list if len(t_offsets[t]) > 0)
    t_list_max = int(t_list_max)

    n_traces = len(trace_list)
    x_mat = np.full((t_list_max, n_traces), np.nan)
    y_mat = np.full((t_list_max, n_traces), np.nan)

    for col_idx, trace_id in enumerate(trace_list):
        mask = df["trace"] == trace_id
        sub = df.loc[mask, ["x", "y", "frame"]].values
        frame0 = sub[0, 2]
        sub = sub[1:]
        frame_offsets = (sub[:, 2] - frame0).astype(int)
        valid = (frame_offsets >= 1) & (frame_offsets <= t_list_max)
        frame_offsets = frame_offsets[valid]
        x_mat[frame_offsets - 1, col_idx] = sub[valid, 0]
        y_mat[frame_offsets - 1, col_idx] = sub[valid, 1]

    frame_1_mask = df["frame"] == 1
    if frame_1_mask.any():
        t_step = float(df.loc[frame_1_mask, "t"].iloc[0])
    else:
        t_vals = df["t"].unique()
        t_step = float(np.min(np.diff(np.sort(t_vals)))) if len(t_vals) > 1 else 1.0

    return x_mat, y_mat, trace_list, t_step, t_list_max


def calculate_msd(
    df: pd.DataFrame,
    method: str = "timeaveraged",
    n: int = 4,
    short: int = 0,
) -> MSDResult | None:
    if not isinstance(df, pd.DataFrame):
        logger.warning("Function requires a DataFrame.")
        return None

    result = _build_displacement_matrices(df, short=short)
    if result is None:
        return None
    x_mat, y_mat, trace_list, t_step, t_list_max = result

    t_offsets_per_trace = []
    for trace_id in trace_list:
        mask = df["trace"] == trace_id
        frames = df.loc[mask, "frame"].values
        if len(frames) > 1:
            t_offsets_per_trace.append(frames[-1] - frames[0])
    t_list_max_90 = np.percentile(t_offsets_per_trace, 90) if t_offsets_per_trace else t_list_max
    num_delta_t = int(math.floor(t_list_max_90 / n))

    if num_delta_t == 0:
        logger.warning("The number of frames is too few to calculate MSD.")
        return None

    n_traces = len(trace_list)
    msd_arr = np.full((num_delta_t, 5), np.nan)
    track_msd_arr = np.full((num_delta_t, n_traces), np.nan)

    for delta_t in range(1, num_delta_t + 1):
        dx = x_mat[delta_t:t_list_max, :] - x_mat[:t_list_max - delta_t, :]
        dy = y_mat[delta_t:t_list_max, :] - y_mat[:t_list_max - delta_t, :]
        sq_disp = dx**2 + dy**2

        if sq_disp.ndim < 2:
            continue

        each_msd = np.nanmean(sq_disp, axis=0)

        idx = delta_t - 1
        if method == "ensemble":
            msd_arr[idx, 0] = np.nanmean(sq_disp)
            msd_arr[idx, 1] = np.nanstd(sq_disp, ddof=0)
            msd_arr[idx, 2] = np.sum(~np.isnan(sq_disp))
        else:
            msd_arr[idx, 0] = np.nanmean(each_msd)
            msd_arr[idx, 1] = np.nanstd(each_msd, ddof=0)
            msd_arr[idx, 2] = np.sum(~np.isnan(each_msd))

        track_msd_arr[idx, :] = each_msd

    alpha_df = calculate_alpha(track_msd_arr, t_step, trace_list)

    dx1 = x_mat[1:t_list_max, :] - x_mat[:t_list_max - 1, :]
    dy1 = y_mat[1:t_list_max, :] - y_mat[:t_list_max - 1, :]
    cve_df = calculate_cve(dx1, dy1, trace_list, t_step)

    sizes = np.arange(1, num_delta_t + 1)
    msd_summary = pd.DataFrame({
        "mean": msd_arr[:, 0],
        "sd": msd_arr[:, 1],
        "n": msd_arr[:, 2],
        "size": sizes,
        "t": sizes * t_step,
    })

    track_msd_df = pd.DataFrame(track_msd_arr, columns=trace_list)
    track_msd_df["size"] = sizes
    track_msd_long = track_msd_df.melt(
        id_vars="size", var_name="trace", value_name="msd"
    )
    track_msd_long["t"] = track_msd_long["size"] * t_step
    track_msd_long = track_msd_long[["trace", "t", "msd"]].copy()
    track_msd_long["trace"] = track_msd_long["trace"].astype(str)

    return MSDResult(
        summary=msd_summary,
        alpha=alpha_df,
        cve=cve_df,
        track_msd=track_msd_long,
    )


def calculate_alpha(
    msd_matrix: np.ndarray,
    t_step: float,
    trace_names: list[str] | None = None,
) -> pd.DataFrame:
    n_lags, n_tracks = msd_matrix.shape
    if trace_names is None:
        trace_names = [str(i) for i in range(n_tracks)]

    if n_lags < 5 or n_tracks < 2:
        return pd.DataFrame({"trace": trace_names[:1], "alpha": [np.nan], "dee": [np.nan]})

    tee = np.arange(1, n_lags + 1) * t_step
    alpha_vec = np.full(n_tracks, np.nan)
    dee_vec = np.full(n_tracks, np.nan)

    check = np.sum(msd_matrix[:4, :], axis=0)

    for i in range(n_tracks):
        if np.isnan(check[i]):
            continue
        msd_col = msd_matrix[:, i]
        if np.all(np.isnan(msd_col)) or np.all(np.isnan(tee)):
            continue

        t4 = tee[:4]
        m4 = msd_col[:4]
        valid = ~np.isnan(m4)
        if valid.sum() < 2:
            continue

        res = linregress(t4[valid], m4[valid])
        slope, intercept = res.slope, res.intercept

        pred = slope * tee + intercept
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = msd_col / pred
            alpha = np.log2(ratio)

        alpha[:4] = np.nan
        alpha_vec[i] = np.nanmean(alpha)
        dee_vec[i] = pred[0] / (4 * tee[0])

    return pd.DataFrame({"trace": trace_names, "alpha": alpha_vec, "dee": dee_vec})


def calculate_cve(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    trace_list: list[str],
    t_step: float,
) -> pd.DataFrame:
    if x_mat is None or y_mat is None or t_step is None:
        return pd.DataFrame({"trace": [""], "dee": [np.nan], "estsigma": [np.nan]})

    if x_mat.ndim < 2 or x_mat.shape[0] < 4 or x_mat.shape[1] < 2:
        return pd.DataFrame({"trace": [""], "dee": [np.nan], "estsigma": [np.nan]})

    with np.errstate(all="ignore"):
        est_dx = (
            np.nanmean(x_mat**2, axis=0) / (2 * t_step)
            + np.nanmean(x_mat[1:] * x_mat[:-1], axis=0) / t_step
        )
        # Replicating R behavior (uses xMat for y calculation)
        est_dy = (
            np.nanmean(y_mat**2, axis=0) / (2 * t_step)
            + np.nanmean(y_mat[1:] * x_mat[:-1], axis=0) / t_step
        )
        est_d = np.mean(np.column_stack([est_dx, est_dy]), axis=1)

        R = 1 / 6
        est_sigma2_x = (
            R * np.nanmean(x_mat**2, axis=0)
            + (2 * R - 1) * np.nanmean(x_mat[1:] * x_mat[:-1], axis=0)
        )
        est_sigma2_y = (
            R * np.nanmean(y_mat**2, axis=0)
            + (2 * R - 1) * np.nanmean(y_mat[1:] * x_mat[:-1], axis=0)
        )
        est_sigma2 = np.mean(np.column_stack([est_sigma2_x, est_sigma2_y]), axis=1)

        with np.errstate(invalid="ignore"):
            est_sigma = np.sqrt(est_sigma2)

    return pd.DataFrame({"trace": trace_list, "dee": est_d, "estsigma": est_sigma})


def calculate_jd(
    data: TrackMateData,
    delta_t: int = 1,
    n_pop: int = 2,
    mode: str = "ECDF",
    init: dict | None = None,
    time_res: float = 1.0,
    breaks: int = 100,
) -> JDResult | None:
    df = data.tracks
    calibration = data.calibration

    if delta_t < 1 or delta_t != int(delta_t):
        return None

    result = _build_displacement_matrices(df)
    if result is None:
        return None
    x_mat, y_mat, trace_list, t_step, t_list_max = result

    if delta_t > t_list_max:
        return None

    dx = x_mat[delta_t:t_list_max, :] - x_mat[:t_list_max - delta_t, :]
    dy = y_mat[delta_t:t_list_max, :] - y_mat[:t_list_max - delta_t, :]
    jd_mat = np.sqrt(dx**2 + dy**2)

    jd_vec = jd_mat.ravel()
    jd_vec = jd_vec[~np.isnan(jd_vec)]

    jd_df = pd.DataFrame({"jump": jd_vec})

    params = JDParams(
        jumptime=delta_t * calibration.time_interval,
        delta_t=delta_t,
        n_pop=n_pop,
        mode=mode,
        init=init,
        spatial_unit=calibration.spatial_unit,
        temporal_unit=calibration.temporal_unit,
        time_res=time_res,
        breaks=breaks,
    )

    return JDResult(jump_distances=jd_df, params=params)


def calculate_fd(data: TrackMateData) -> FDResult | None:
    df = data.tracks
    trace_list = df["trace"].unique()

    records = []
    for trace_id in trace_list:
        mask = df["trace"] == trace_id
        coords = df.loc[mask, ["x", "y"]].values
        n = len(coords)
        if n < 2:
            records.append({"trace": trace_id, "wide": np.nan, "fd": np.nan})
            continue

        distances = pdist(coords)
        d = np.max(distances) if len(distances) > 0 else 0.0
        cumul = df.loc[mask, "cumulative_distance"].values
        track_length = np.max(cumul)

        if track_length > 0 and d > 0:
            fd = math.log(n) / (math.log(n) + math.log(d / track_length))
        else:
            fd = np.nan

        records.append({"trace": str(trace_id), "wide": d, "fd": fd})

    return FDResult(data=pd.DataFrame(records))


def calculate_track_density(
    data: TrackMateData, radius: float = 1.0
) -> TrackDensityResult | None:
    df = data.tracks
    calibration = data.calibration
    trace_list = df["trace"].unique()

    first_idx = []
    for t in trace_list:
        mask = df["trace"] == t
        first_idx.append(df.index[mask][0])

    frame0s = df.loc[first_idx, "frame"].values
    x0s = df.loc[first_idx, "x"].values
    y0s = df.loc[first_idx, "y"].values

    neighbours = np.zeros(len(trace_list))
    fractions = np.zeros(len(trace_list))

    img_a = (0.0, calibration.width)
    img_b = (0.0, calibration.height)

    for j in range(len(trace_list)):
        frame0 = frame0s[j]
        x0, y0 = x0s[j], y0s[j]

        frame_mask = df["frame"] == frame0
        x_all = df.loc[frame_mask, "x"].values
        y_all = df.loc[frame_mask, "y"].values

        dists = np.sqrt((x_all - x0) ** 2 + (y_all - y0) ** 2)
        neighbours[j] = np.sum(dists <= radius) - 1

        area = _find_td_area(radius, (x0, y0), img_a, img_b)
        fractions[j] = area / (math.pi * radius**2)

    result_df = pd.DataFrame({
        "trace": trace_list.astype(str),
        "neighbours": neighbours,
        "fraction": fractions,
    })
    result_df["density"] = result_df["neighbours"] / result_df["fraction"]

    return TrackDensityResult(data=result_df)


def _find_td_A1(x: float, r: float) -> float:
    if x < r:
        return r**2 * math.acos(x / r) - x * math.sqrt(r**2 - x**2)
    return 0.0


def _find_td_A2(x: float, y: float, r: float) -> float:
    if x**2 + y**2 < r**2:
        return (
            (r**2 / 2) * (math.acos(y / r) + math.acos(x / r) - math.pi / 2)
            - x * math.sqrt(r**2 - x**2) / 2
            - y * math.sqrt(r**2 - y**2) / 2
            + x * y
        )
    return 0.0


def _find_td_area(
    r: float,
    xy: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    x1 = xy[0] - a[0]
    x2 = a[1] - xy[0]
    y1 = xy[1] - b[0]
    y2 = b[1] - xy[1]

    x1 = max(x1, 0.0)
    x2 = max(x2, 0.0)
    y1 = max(y1, 0.0)
    y2 = max(y2, 0.0)

    clipped = (
        _find_td_A1(x1, r)
        + _find_td_A1(x2, r)
        + _find_td_A1(y1, r)
        + _find_td_A1(y2, r)
        - _find_td_A2(x1, y1, r)
        - _find_td_A2(x1, y2, r)
        - _find_td_A2(x2, y1, r)
        - _find_td_A2(x2, y2, r)
    )

    area = math.pi * r**2 - clipped
    return max(area, 0.0)
