from __future__ import annotations

import copy
import logging

import pandas as pd

from ._types import Calibration, TrackMateData

logger = logging.getLogger(__name__)


def correct_data(
    data: TrackMateData,
    xy_scalar: float = 1.0,
    t_scalar: float = 1.0,
    xy_unit: str | None = None,
    t_unit: str | None = None,
) -> TrackMateData:
    if xy_scalar == 1.0 and t_scalar == 1.0 and xy_unit is None and t_unit is None:
        logger.info("No correction applied.")
        return data

    df = data.tracks.copy()
    cal = copy.copy(data.calibration)

    if xy_scalar != 1.0:
        logger.info("Correcting XY scale.")
        df["x"] *= xy_scalar
        df["y"] *= xy_scalar
        df["displacement"] *= xy_scalar
        df["cumulative_distance"] *= xy_scalar
        if "radius" in df.columns:
            df["radius"] *= xy_scalar
        if "area" in df.columns:
            df["area"] *= xy_scalar**2
        if "perimeter" in df.columns:
            df["perimeter"] *= xy_scalar
        if "ellipse_x0" in df.columns:
            for col in ["ellipse_x0", "ellipse_y0", "ellipse_major", "ellipse_minor"]:
                if col in df.columns:
                    df[col] *= xy_scalar

        cal.pixel_size *= xy_scalar
        cal.width *= xy_scalar
        cal.height *= xy_scalar

    if t_scalar != 1.0:
        logger.info("Correcting timescale.")
        df["t"] *= t_scalar
        df["track_duration"] *= t_scalar
        cal.time_interval *= t_scalar

    if xy_unit is not None:
        cal.spatial_unit = xy_unit
    if t_unit is not None:
        cal.temporal_unit = t_unit

    frame_1_mask = df["frame"] == 1
    if frame_1_mask.any():
        tstep = df.loc[frame_1_mask, "t"].iloc[0]
    else:
        tstep = cal.time_interval
    if tstep > 0:
        df["speed"] = df["displacement"] / tstep

    return TrackMateData(tracks=df, calibration=cal)
