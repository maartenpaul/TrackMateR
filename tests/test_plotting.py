from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

import trackmater as tm


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


class TestPlotTracks:
    def test_returns_fig_ax(self, trackmate_data):
        fig, ax = tm.plot_tracks(trackmate_data)
        assert isinstance(fig, Figure)

    def test_equal_aspect(self, trackmate_data):
        _, ax = tm.plot_tracks(trackmate_data)
        assert ax.get_aspect() in ("equal", 1.0)

    def test_custom_ax(self, trackmate_data):
        fig, ax = plt.subplots()
        fig2, ax2 = tm.plot_tracks(trackmate_data, ax=ax)
        assert ax2 is ax


class TestPlotDisplacementOverTime:
    def test_returns_fig_ax(self, trackmate_data):
        fig, ax = tm.plot_displacement_over_time(trackmate_data)
        assert isinstance(fig, Figure)

    def test_ylim_starts_at_zero(self, trackmate_data):
        _, ax = tm.plot_displacement_over_time(trackmate_data)
        assert ax.get_ylim()[0] == pytest.approx(0.0)


class TestPlotCumulativeDistance:
    def test_returns_fig_ax(self, trackmate_data):
        fig, ax = tm.plot_cumulative_distance(trackmate_data)
        assert isinstance(fig, Figure)


class TestPlotDisplacementHist:
    def test_returns_fig_ax_median(self, trackmate_data):
        fig, ax, median_val = tm.plot_displacement_hist(trackmate_data)
        assert isinstance(fig, Figure)
        assert isinstance(median_val, float)
        assert median_val > 0


class TestPlotIntensityHist:
    def test_returns_fig_ax_median(self, trackmate_data):
        fig, ax, median_val = tm.plot_intensity_hist(trackmate_data)
        assert isinstance(fig, Figure)


class TestPlotDurationHist:
    def test_returns_fig_ax_median(self, trackmate_data):
        fig, ax, median_val = tm.plot_duration_hist(trackmate_data)
        assert isinstance(fig, Figure)
        assert isinstance(median_val, float)
        assert median_val > 0


class TestPlotSpeed:
    def test_returns_fig_ax_median(self, trackmate_data):
        fig, ax, median_val = tm.plot_speed(trackmate_data)
        assert isinstance(fig, Figure)
        assert isinstance(median_val, float)
        assert median_val > 0


class TestPlotAlpha:
    def test_returns_fig_ax(self, msd_result):
        alpha_df = msd_result.alpha.dropna(subset=["alpha"])
        fig, ax = tm.plot_alpha(alpha_df)
        assert isinstance(fig, Figure)

    def test_with_median(self, msd_result):
        alpha_df = msd_result.alpha.dropna(subset=["alpha"])
        fig, ax = tm.plot_alpha(alpha_df, median_alpha=1.5)
        assert isinstance(fig, Figure)


class TestPlotDee:
    def test_returns_fig_ax(self, msd_result):
        dee_df = msd_result.alpha.dropna(subset=["dee"])
        fig, ax = tm.plot_dee(dee_df)
        assert isinstance(fig, Figure)


class TestPlotMSD:
    def test_returns_fig_ax_dee(self, msd_result):
        fig, ax, dee = tm.plot_msd(msd_result.summary)
        assert isinstance(fig, Figure)
        assert isinstance(dee, float)

    def test_error_bars(self, msd_result):
        fig, ax, dee = tm.plot_msd(msd_result.summary, error_bars=True)
        assert isinstance(fig, Figure)

    def test_log_scale(self, msd_result):
        fig, ax, dee = tm.plot_msd(msd_result.summary, xlog=True, ylog=True)
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"


class TestPlotFD:
    def test_returns_fig_ax_median(self, fd_result):
        fig, ax, median_val = tm.plot_fd(fd_result.data)
        assert isinstance(fig, Figure)
        assert isinstance(median_val, float)


class TestPlotNeighbours:
    def test_returns_fig_ax_median(self, td_result):
        fig, ax, median_val = tm.plot_neighbours(td_result.data)
        assert isinstance(fig, Figure)
        assert isinstance(median_val, float)


class TestPlotWidth:
    def test_returns_fig_ax_median(self, fd_result):
        fig, ax, median_val = tm.plot_width(fd_result.data)
        assert isinstance(fig, Figure)
        assert isinstance(median_val, float)
