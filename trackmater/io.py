from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from ._types import Calibration, TrackMateData

logger = logging.getLogger(__name__)


def read_trackmate_xml(path: str | Path, slim: bool = False) -> TrackMateData | None:
    path = Path(path)
    if not path.exists():
        logger.warning("XML file does not exist: %s", path)
        return None

    tree = ET.parse(path)
    root = tree.getroot()

    track_nodes = root.findall(".//Track")
    if not track_nodes:
        logger.warning("No tracks found in XML file")
        return None

    filtered_ids = {
        node.get("TRACK_ID") for node in root.findall(".//TrackID")
    }
    if not filtered_ids:
        logger.warning("No filtered tracks found in XML file")
        return None

    calibration = _extract_calibration(root)
    if calibration.spatial_unit == "pixel":
        logger.warning("Spatial units are in pixels - consider transforming to real units")
    logger.info(
        "Units are: %s %s and %s %s",
        calibration.pixel_size, calibration.spatial_unit,
        calibration.time_interval, calibration.temporal_unit,
    )

    target_channel = calibration.channel

    feature_nodes = root.findall(".//FeatureDeclarations//SpotFeatures//Feature")
    attribute_names = ["name"] + [n.get("feature") for n in feature_nodes]

    if slim:
        slim_attrs = {
            "name", "POSITION_X", "POSITION_Y", "POSITION_Z", "POSITION_T",
            "FRAME", "MEAN_INTENSITY", f"MEAN_INTENSITY_CH{target_channel}",
        }
        attribute_names = [a for a in attribute_names if a in slim_attrs]

    logger.info("Extracting spot data...")
    spots = _extract_spots(root, attribute_names, slim)

    logger.info("Matching track data...")
    spot_track_map = _build_spot_track_map(track_nodes, filtered_ids)

    result = pd.merge(spot_track_map, spots, on="name", how="inner")
    result.drop(columns=["name"], inplace=True)
    result["trace"] = result["trace"].astype(str)
    result.sort_values(
        by=["trace", "frame"],
        key=lambda col: pd.to_numeric(col, errors="coerce") if col.name == "trace" else col,
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)

    dupe_count = result.duplicated(subset=["trace", "frame"]).sum()
    if dupe_count > 0:
        logger.warning(
            "Detected %d duplicate track-frame combinations. "
            "Subsequent analysis will likely fail!",
            dupe_count,
        )

    logger.info("Calculating distances...")
    result = result.groupby("trace", sort=False).apply(
        _compute_track_metrics, include_groups=False
    ).reset_index(level=0)
    result.reset_index(drop=True, inplace=True)

    min_x = result["x"].min()
    min_y = result["y"].min()
    if min_x < 0:
        result["x"] -= min_x
    if min_y < 0:
        result["y"] -= min_y

    max_x = result["x"].max()
    max_y = result["y"].max()
    calibration.width = max(calibration.width, float(np.ceil(max_x)))
    calibration.height = max(calibration.height, float(np.ceil(max_y)))
    calibration.n_traces = result["trace"].nunique()
    if calibration.time_interval > 0:
        calibration.max_frames = result["track_duration"].max() / calibration.time_interval

    return TrackMateData(tracks=result, calibration=calibration)


def _extract_calibration(root: ET.Element) -> Calibration:
    model_node = root.find(".//Model")
    spatial_unit = model_node.get("spatialunits", "pixel") if model_node is not None else "pixel"
    temporal_unit = model_node.get("timeunits", "frame") if model_node is not None else "frame"
    if temporal_unit == "sec":
        temporal_unit = "s"

    image_node = root.find(".//ImageData")
    pixel_width = float(image_node.get("pixelwidth", "1")) if image_node is not None else 1.0
    time_interval = float(image_node.get("timeinterval", "1")) if image_node is not None else 1.0
    img_w = float(image_node.get("width", "0")) if image_node is not None else 0.0
    img_h = float(image_node.get("height", "0")) if image_node is not None else 0.0

    width = (img_w - 1) * pixel_width
    height = (img_h - 1) * pixel_width

    detector_node = root.find(".//DetectorSettings")
    target_channel = int(detector_node.get("TARGET_CHANNEL", "1")) if detector_node is not None else 1

    return Calibration(
        pixel_size=pixel_width,
        time_interval=time_interval,
        spatial_unit=spatial_unit,
        temporal_unit=temporal_unit,
        width=width,
        height=height,
        n_traces=0,
        max_frames=0,
        channel=target_channel,
    )


def _extract_spots(root: ET.Element, cols: list[str], slim: bool) -> pd.DataFrame:
    spot_nodes = root.findall(".//AllSpots//SpotsInFrame//Spot")
    if not spot_nodes:
        return pd.DataFrame()

    records: dict[str, list] = {c: [] for c in cols}
    for spot in spot_nodes:
        attribs = spot.attrib
        for c in cols:
            records[c].append(attribs.get(c))

    df = pd.DataFrame(records)

    drop_cols = [c for c in df.columns if df[c].isna().all() and c != "name"]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    num_cols = [c for c in df.columns if c != "name"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "FRAME" in df.columns:
        df["FRAME"] = df["FRAME"].astype("Int64")

    rename_map: dict[str, str] = {}
    for c in list(df.columns):
        new_name = c.lower()
        if slim:
            new_name = re.sub(r"_ch\d$", "", new_name, flags=re.IGNORECASE)
        new_name = re.sub(r"^position_?", "", new_name, flags=re.IGNORECASE)
        if new_name != c:
            rename_map[c] = new_name
    df.rename(columns=rename_map, inplace=True)

    return df


def _build_spot_track_map(
    track_nodes: list[ET.Element], filtered_ids: set[str]
) -> pd.DataFrame:
    rows: list[dict] = []
    for track_node in track_nodes:
        track_id = track_node.get("TRACK_ID")
        if track_id not in filtered_ids:
            continue

        edge_nodes = track_node.findall(".//Edge")
        source_ids = set()
        target_data: dict[str, dict] = {}

        for edge in edge_nodes:
            src = edge.get("SPOT_SOURCE_ID")
            tgt = edge.get("SPOT_TARGET_ID")
            source_ids.add(src)
            disp = float(edge.get("DISPLACEMENT", "0") or "0")
            spd = float(edge.get("SPEED", "0") or "0")
            target_data[tgt] = {"displacement": disp, "speed": spd}

        all_ids = source_ids | set(target_data.keys())
        for spot_id in all_ids:
            row = {"name": f"ID{spot_id}", "trace": track_id}
            if spot_id in target_data:
                row["displacement"] = target_data[spot_id]["displacement"]
                row["speed"] = target_data[spot_id]["speed"]
            else:
                row["displacement"] = 0.0
                row["speed"] = 0.0
            rows.append(row)

    return pd.DataFrame(rows)


def _compute_track_metrics(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("frame").copy()
    group["displacement"] = group["displacement"].fillna(0.0)
    group["cumulative_distance"] = group["displacement"].cumsum()
    t_min = group["t"].min()
    group["track_duration"] = group["t"] - t_min
    return group


def read_gt_file(path: str | Path) -> TrackMateData | None:
    path = Path(path)
    if not path.exists():
        logger.warning("Ground truth file does not exist: %s", path)
        return None

    df = pd.read_csv(path)
    expected = {"TrackID", "x", "y", "frame"}
    if not expected.issubset(set(df.columns)):
        logger.warning("Ground truth CSV must have columns: %s", expected)
        return None

    df = df.rename(columns={"TrackID": "trace"})
    df["trace"] = df["trace"].astype(str)
    df["t"] = df["frame"].astype(float)
    df["displacement"] = 0.0
    df["speed"] = 0.0

    df.sort_values(by=["trace", "frame"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    for trace_id, grp in df.groupby("trace"):
        idx = grp.index
        dx = grp["x"].diff().fillna(0.0)
        dy = grp["y"].diff().fillna(0.0)
        df.loc[idx, "displacement"] = np.sqrt(dx**2 + dy**2)

    df["displacement"] = df["displacement"].fillna(0.0)
    df["cumulative_distance"] = df.groupby("trace")["displacement"].cumsum()
    df["track_duration"] = df.groupby("trace")["t"].transform(lambda s: s - s.min())

    min_x, min_y = df["x"].min(), df["y"].min()
    if min_x < 0:
        df["x"] -= min_x
    if min_y < 0:
        df["y"] -= min_y

    calibration = Calibration(
        pixel_size=1.0,
        time_interval=1.0,
        spatial_unit="pixel",
        temporal_unit="frame",
        width=float(np.ceil(df["x"].max())),
        height=float(np.ceil(df["y"].max())),
        n_traces=df["trace"].nunique(),
        max_frames=float(df["frame"].max()),
        channel=1,
    )

    return TrackMateData(tracks=df, calibration=calibration)


def to_csv(
    data: TrackMateData,
    output_path: str | Path | None = None,
    min_points: int = 10,
    columns: list[str] | None = None,
    pixels: bool = False,
) -> pd.DataFrame | None:
    if columns is None:
        columns = ["trace", "frame", "y", "x"]

    df = data.tracks.copy()

    counts = df.groupby("trace")["trace"].transform("count")
    df = df[counts >= min_points].copy()

    if pixels and data.calibration.pixel_size != 0:
        ps = data.calibration.pixel_size
        for col in ["x", "y"]:
            if col in df.columns:
                df[col] = df[col] / ps

    available = [c for c in columns if c in df.columns]
    df = df[available]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return None

    return df
