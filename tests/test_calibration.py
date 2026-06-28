from __future__ import annotations

import numpy as np
import pytest

import trackmater as tm


class TestCorrectData:
    def test_no_correction_returns_same(self, trackmate_data):
        result = tm.correct_data(trackmate_data)
        assert result is trackmate_data

    def test_xy_scaling(self, trackmate_data):
        result = tm.correct_data(trackmate_data, xy_scalar=2.0)
        assert result is not trackmate_data
        orig_x = trackmate_data.tracks["x"].iloc[0]
        new_x = result.tracks["x"].iloc[0]
        assert new_x == pytest.approx(orig_x * 2.0)

    def test_xy_scaling_calibration(self, trackmate_data):
        result = tm.correct_data(trackmate_data, xy_scalar=0.5)
        assert result.calibration.pixel_size == pytest.approx(
            trackmate_data.calibration.pixel_size * 0.5
        )

    def test_t_scaling(self, trackmate_data):
        result = tm.correct_data(trackmate_data, t_scalar=3.0)
        orig_t = trackmate_data.tracks["t"].iloc[5]
        new_t = result.tracks["t"].iloc[5]
        assert new_t == pytest.approx(orig_t * 3.0)

    def test_t_scaling_calibration(self, trackmate_data):
        result = tm.correct_data(trackmate_data, t_scalar=2.0)
        assert result.calibration.time_interval == pytest.approx(
            trackmate_data.calibration.time_interval * 2.0
        )

    def test_unit_change(self, trackmate_data):
        result = tm.correct_data(trackmate_data, xy_unit="nm", t_unit="ms")
        assert result.calibration.spatial_unit == "nm"
        assert result.calibration.temporal_unit == "ms"

    def test_immutability(self, trackmate_data):
        orig_x = trackmate_data.tracks["x"].copy()
        _ = tm.correct_data(trackmate_data, xy_scalar=10.0)
        assert np.allclose(trackmate_data.tracks["x"].values, orig_x.values)

    def test_displacement_scaled(self, trackmate_data):
        result = tm.correct_data(trackmate_data, xy_scalar=3.0)
        idx = trackmate_data.tracks["displacement"] > 0
        if idx.any():
            first_idx = idx.idxmax()
            orig = trackmate_data.tracks.loc[first_idx, "displacement"]
            new = result.tracks.loc[first_idx, "displacement"]
            assert new == pytest.approx(orig * 3.0)

    def test_method_style(self, trackmate_data):
        result = trackmate_data.correct(xy_scalar=2.0)
        assert isinstance(result, tm.TrackMateData)
        assert result is not trackmate_data
