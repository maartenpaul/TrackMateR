from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import trackmater as tm


class TestReadTrackMateXML:
    def test_returns_trackmate_data(self, trackmate_data):
        assert isinstance(trackmate_data, tm.TrackMateData)

    def test_tracks_is_dataframe(self, trackmate_data):
        assert isinstance(trackmate_data.tracks, pd.DataFrame)

    def test_has_required_columns(self, trackmate_data):
        required = {"trace", "x", "y", "t", "frame", "displacement", "cumulative_distance", "track_duration"}
        assert required.issubset(set(trackmate_data.tracks.columns))

    def test_track_count(self, trackmate_data):
        assert trackmate_data.n_tracks > 0
        assert trackmate_data.n_tracks == trackmate_data.calibration.n_traces

    def test_calibration_units(self, trackmate_data):
        cal = trackmate_data.calibration
        assert cal.spatial_unit != ""
        assert cal.temporal_unit != ""
        assert cal.pixel_size > 0
        assert cal.time_interval > 0

    def test_calibration_dimensions(self, trackmate_data):
        cal = trackmate_data.calibration
        assert cal.width > 0
        assert cal.height > 0

    def test_no_negative_coordinates(self, trackmate_data):
        df = trackmate_data.tracks
        assert df["x"].min() >= 0
        assert df["y"].min() >= 0

    def test_cumulative_distance_non_decreasing(self, trackmate_data):
        df = trackmate_data.tracks
        for _, grp in df.groupby("trace"):
            sorted_grp = grp.sort_values("frame")
            diffs = sorted_grp["cumulative_distance"].diff().dropna()
            assert (diffs >= -1e-10).all()

    def test_track_duration_starts_at_zero(self, trackmate_data):
        df = trackmate_data.tracks
        for _, grp in df.groupby("trace"):
            assert grp["track_duration"].min() == pytest.approx(0.0)

    def test_slim_mode(self, example_xml_path):
        data = tm.read_trackmate_xml(example_xml_path, slim=True)
        assert data is not None
        assert len(data.tracks.columns) < 20

    def test_missing_file_returns_none(self, tmp_path):
        result = tm.read_trackmate_xml(tmp_path / "nonexistent.xml")
        assert result is None

    def test_trace_ids_property(self, trackmate_data):
        ids = trackmate_data.trace_ids
        assert len(ids) == trackmate_data.n_tracks

    def test_repr_html(self, trackmate_data):
        html = trackmate_data._repr_html_()
        assert "<b>TrackMateData</b>" in html

    def test_calibration_repr_html(self, trackmate_data):
        html = trackmate_data.calibration._repr_html_()
        assert "Pixel size" in html


class TestToCsv:
    def test_returns_dataframe_when_no_path(self, trackmate_data):
        result = tm.to_csv(trackmate_data)
        assert isinstance(result, pd.DataFrame)
        assert "trace" in result.columns

    def test_writes_file(self, trackmate_data, tmp_path):
        out = tmp_path / "output.csv"
        result = tm.to_csv(trackmate_data, output_path=out)
        assert result is None
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) > 0

    def test_min_points_filter(self, trackmate_data):
        result_high = tm.to_csv(trackmate_data, min_points=100)
        result_low = tm.to_csv(trackmate_data, min_points=2)
        assert len(result_high) <= len(result_low)

    def test_pixel_mode(self, trackmate_data):
        result_um = tm.to_csv(trackmate_data, pixels=False)
        result_px = tm.to_csv(trackmate_data, pixels=True)
        if trackmate_data.calibration.pixel_size != 1.0:
            assert not np.allclose(result_um["x"].values[:10], result_px["x"].values[:10])


class TestReadGtFile:
    def test_missing_file_returns_none(self, tmp_path):
        result = tm.read_gt_file(tmp_path / "nonexistent.csv")
        assert result is None

    def test_valid_gt_file(self, tmp_path):
        gt = pd.DataFrame({
            "TrackID": [1, 1, 1, 2, 2, 2],
            "x": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
            "y": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "frame": [0, 1, 2, 0, 1, 2],
        })
        path = tmp_path / "gt.csv"
        gt.to_csv(path, index=False)
        data = tm.read_gt_file(path)
        assert data is not None
        assert data.n_tracks == 2
        assert data.calibration.spatial_unit == "pixel"
