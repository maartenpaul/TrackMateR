from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import trackmater as tm
from trackmater.fitting import fit_jd


class TestFitJD:
    def test_ecdf_2pop(self, jd_result):
        result = fit_jd(jd_result)
        assert isinstance(result, tm.JDFitResult)
        assert len(result.coefficients) > 0
        assert "D1" in result.coefficients
        assert result.fig is not None
        plt.close(result.fig)

    def test_ecdf_1pop(self, trackmate_data):
        jd = tm.calculate_jd(trackmate_data, n_pop=1)
        assert jd is not None
        result = fit_jd(jd)
        assert isinstance(result, tm.JDFitResult)
        assert "D" in result.coefficients
        plt.close(result.fig)

    def test_histogram_mode(self, trackmate_data):
        jd = tm.calculate_jd(trackmate_data, n_pop=2, mode="histogram")
        assert jd is not None
        result = fit_jd(jd)
        assert isinstance(result, tm.JDFitResult)
        assert len(result.coefficients) > 0
        plt.close(result.fig)

    def test_fit_curve_has_data(self, jd_result):
        result = fit_jd(jd_result)
        assert len(result.fit_curve) > 0
        assert "x" in result.fit_curve.columns
        plt.close(result.fig)

    def test_invalid_npop_raises(self, trackmate_data):
        jd = tm.calculate_jd(trackmate_data, n_pop=2)
        jd.params.n_pop = 5
        with pytest.raises(ValueError):
            fit_jd(jd)
