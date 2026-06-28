from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import trackmater as tm
from trackmater.analysis import _build_displacement_matrices


class TestBuildDisplacementMatrices:
    def test_returns_tuple(self, trackmate_data):
        result = _build_displacement_matrices(trackmate_data.tracks)
        assert result is not None
        x_mat, y_mat, trace_list, t_step, t_list_max = result
        assert isinstance(x_mat, np.ndarray)
        assert isinstance(y_mat, np.ndarray)
        assert x_mat.shape == y_mat.shape
        assert len(trace_list) > 0
        assert t_step > 0
        assert t_list_max > 0

    def test_short_filter(self, trackmate_data):
        result_all = _build_displacement_matrices(trackmate_data.tracks, short=0)
        result_short = _build_displacement_matrices(trackmate_data.tracks, short=20)
        assert result_all is not None and result_short is not None
        assert len(result_short[2]) <= len(result_all[2])


class TestCalculateMSD:
    def test_returns_msd_result(self, msd_result):
        assert isinstance(msd_result, tm.MSDResult)

    def test_summary_columns(self, msd_result):
        assert {"mean", "sd", "n", "size", "t"}.issubset(set(msd_result.summary.columns))

    def test_summary_has_rows(self, msd_result):
        assert len(msd_result.summary) > 0

    def test_mean_positive(self, msd_result):
        assert (msd_result.summary["mean"].dropna() >= 0).all()

    def test_time_increasing(self, msd_result):
        assert msd_result.summary["t"].is_monotonic_increasing

    def test_alpha_dataframe(self, msd_result):
        assert {"trace", "alpha", "dee"}.issubset(set(msd_result.alpha.columns))
        assert len(msd_result.alpha) > 0

    def test_cve_dataframe(self, msd_result):
        assert {"trace", "dee", "estsigma"}.issubset(set(msd_result.cve.columns))
        assert len(msd_result.cve) > 0

    def test_track_msd_dataframe(self, msd_result):
        assert {"trace", "t", "msd"}.issubset(set(msd_result.track_msd.columns))
        assert len(msd_result.track_msd) > 0

    def test_repr_html(self, msd_result):
        html = msd_result._repr_html_()
        assert "MSDResult" in html

    def test_method_style(self, trackmate_data):
        result = trackmate_data.calculate_msd(n=3, short=8)
        assert isinstance(result, tm.MSDResult)

    def test_ensemble_method(self, trackmate_data):
        result = tm.calculate_msd(trackmate_data.tracks, method="ensemble", n=4)
        assert result is not None
        assert len(result.summary) > 0

    def test_invalid_input(self):
        result = tm.calculate_msd("not a dataframe")
        assert result is None


class TestCalculateJD:
    def test_returns_jd_result(self, jd_result):
        assert isinstance(jd_result, tm.JDResult)

    def test_jump_distances_not_empty(self, jd_result):
        assert len(jd_result.jump_distances) > 0

    def test_jumps_non_negative(self, jd_result):
        assert (jd_result.jump_distances["jump"] >= 0).all()

    def test_params_stored(self, jd_result):
        assert jd_result.params.delta_t == 1
        assert jd_result.params.n_pop == 2

    def test_repr_html(self, jd_result):
        html = jd_result._repr_html_()
        assert "JDResult" in html

    def test_method_style(self, trackmate_data):
        result = trackmate_data.calculate_jd(delta_t=1)
        assert isinstance(result, tm.JDResult)

    def test_delta_t_2(self, trackmate_data):
        result = tm.calculate_jd(trackmate_data, delta_t=2)
        assert result is not None
        assert result.params.delta_t == 2


class TestCalculateFD:
    def test_returns_fd_result(self, fd_result):
        assert isinstance(fd_result, tm.FDResult)

    def test_fd_columns(self, fd_result):
        assert {"trace", "fd", "wide"}.issubset(set(fd_result.data.columns))

    def test_fd_values_positive(self, fd_result):
        valid = fd_result.data["fd"].dropna()
        assert (valid > 0).all()
        assert np.all(np.isfinite(valid))

    def test_repr_html(self, fd_result):
        html = fd_result._repr_html_()
        assert "FDResult" in html

    def test_method_style(self, trackmate_data):
        result = trackmate_data.calculate_fd()
        assert isinstance(result, tm.FDResult)


class TestCalculateTrackDensity:
    def test_returns_result(self, td_result):
        assert isinstance(td_result, tm.TrackDensityResult)

    def test_density_columns(self, td_result):
        assert {"trace", "neighbours", "fraction", "density"}.issubset(
            set(td_result.data.columns)
        )

    def test_neighbours_non_negative(self, td_result):
        assert (td_result.data["neighbours"] >= 0).all()

    def test_fraction_range(self, td_result):
        assert (td_result.data["fraction"] > 0).all()
        assert (td_result.data["fraction"] <= 1.0 + 1e-10).all()

    def test_repr_html(self, td_result):
        html = td_result._repr_html_()
        assert "TrackDensityResult" in html

    def test_method_style(self, trackmate_data):
        result = trackmate_data.calculate_track_density(radius=1.5)
        assert isinstance(result, tm.TrackDensityResult)
