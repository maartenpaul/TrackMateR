from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import trackmater as tm
from trackmater.comparison import make_comparison
from trackmater.utils import find_log2_y_limits, merge_dataframes_for_export, process_defaults


@pytest.fixture(autouse=True)
def close_figs():
    yield
    plt.close("all")


class TestMakeComparison:
    def test_returns_figure(self):
        report_df = pd.DataFrame({
            "condition": ["A", "A", "B", "B"],
            "alpha": [1.0, 1.2, 0.8, 0.9],
            "speed": [0.5, 0.6, 0.3, 0.4],
            "intensity": [100, 110, 90, 95],
            "duration": [10, 12, 8, 9],
            "dee": [0.01, 0.02, 0.005, 0.008],
            "fd": [1.2, 1.3, 1.1, 1.15],
            "width": [0.5, 0.6, 0.4, 0.45],
            "neighbours": [3, 4, 2, 3],
        })
        msd_df = pd.DataFrame({
            "condition": ["A"] * 5 + ["B"] * 5,
            "t": list(range(1, 6)) * 2,
            "mean": [1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "sd": [0.1] * 10,
        })
        fig = make_comparison(report_df, msd_df)
        assert isinstance(fig, Figure)


class TestUtils:
    def test_process_defaults(self):
        result = process_defaults({})
        assert result["N"] == 3
        assert result["short"] == 8
        assert result["delta_t"] == 1
        assert result["mode"] == "ECDF"

    def test_process_defaults_override(self):
        result = process_defaults({"N": 5})
        assert result["N"] == 5
        assert result["short"] == 8

    def test_merge_dataframes(self):
        df1 = pd.DataFrame({"trace": ["1", "2"], "dataid": ["a", "a"], "val1": [1, 2]})
        df2 = pd.DataFrame({"trace": ["1", "2"], "dataid": ["a", "a"], "val2": [3, 4]})
        merged = merge_dataframes_for_export(df1, df2)
        assert "val1" in merged.columns
        assert "val2" in merged.columns
        assert len(merged) == 2

    def test_find_log2_y_limits(self):
        lo, hi = find_log2_y_limits(np.array([0.5, 1.0, 2.0]))
        assert lo > 0
        assert hi > lo
        assert lo <= 0.5
        assert hi >= 2.0
