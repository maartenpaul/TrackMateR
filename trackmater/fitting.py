from __future__ import annotations

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ._types import JDFitResult, JDResult

logger = logging.getLogger(__name__)


# --- ECDF model functions ---

def _ecdf_1pop(r, D1, time_res):
    return 1 - np.exp(-r**2 / (4 * D1 * time_res))


def _ecdf_2pop(r, D1, D2_frac, D3, time_res):
    return 1 - D2_frac * np.exp(-r**2 / (4 * D1 * time_res)) - (1 - D2_frac) * np.exp(
        -r**2 / (4 * D3 * time_res)
    )


def _ecdf_3pop(r, D1, D2, D3, D4, D6, time_res):
    return (
        1
        - D4 * np.exp(-r**2 / (4 * D1 * time_res))
        - (1 - D4 - D6) * np.exp(-r**2 / (4 * D2 * time_res))
        - D6 * np.exp(-r**2 / (4 * D3 * time_res))
    )


# --- Histogram (PDF) model functions ---

def _pdf_1pop(r, D2, D1, time_res):
    return D2 * r / (2 * D1 * time_res) * np.exp(-r**2 / (4 * D1 * time_res))


def _pdf_2pop(r, D1, D2, D3, D4, time_res):
    return (
        D4 * r / (2 * D1 * time_res) * np.exp(-r**2 / (4 * D1 * time_res))
        + D3 * r / (2 * D2 * time_res) * np.exp(-r**2 / (4 * D2 * time_res))
    )


def _pdf_3pop(r, D1, D2, D3, D4, D5, D6, time_res):
    return (
        D4 * r / (2 * D1 * time_res) * np.exp(-r**2 / (4 * D1 * time_res))
        + D5 * r / (2 * D2 * time_res) * np.exp(-r**2 / (4 * D2 * time_res))
        + D6 * r / (2 * D3 * time_res) * np.exp(-r**2 / (4 * D3 * time_res))
    )


def fit_jd(jd_result: JDResult) -> JDFitResult:
    params = jd_result.params
    jumps = jd_result.jump_distances["jump"].values
    n_pop = params.n_pop
    mode = params.mode
    init = params.init
    time_res = params.time_res
    breaks = params.breaks
    units = (params.spatial_unit, params.temporal_unit)

    if n_pop < 1 or n_pop > 3:
        raise ValueError("n_pop must be 1, 2, or 3")

    counts, bin_edges = np.histogram(jumps, bins=breaks)
    mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    counts_cum = np.cumsum(counts) / np.sum(counts)

    fig, ax = plt.subplots(figsize=(5, 4))
    x_label = f"Displacement ({units[0]})"

    if mode == "ECDF":
        ax.scatter(mids, counts_cum, s=10, c="black", zorder=3)
        ax.set_xlim(0, mids.max())
        ax.set_ylim(0, 1.4)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Frequency")
    else:
        ax.bar(mids, counts, width=np.diff(bin_edges).mean(), color="lightgrey", edgecolor="darkgrey")
        ax.set_xlim(0, 1.1 * mids.max())
        ax.set_ylim(0, 1.4 * counts.max())
        ax.set_xlabel(x_label)
        ax.set_ylabel("Counts")

    x_fit = np.linspace(0, mids.max(), 10 * len(mids))
    coefficients: dict[str, float] = {}
    fit_df = pd.DataFrame({"x": x_fit})

    if init is None:
        init = _auto_guess(jumps, n_pop, mode, time_res, breaks)

    try:
        if n_pop == 1:
            coefficients, fit_df = _fit_1pop(mids, counts, counts_cum, x_fit, mode, init, time_res)
        elif n_pop == 2:
            coefficients, fit_df = _fit_2pop(mids, counts, counts_cum, x_fit, mode, init, time_res)
        elif n_pop == 3:
            coefficients, fit_df = _fit_3pop(mids, counts, counts_cum, x_fit, mode, init, time_res)

        melted = fit_df.melt(id_vars="x", var_name="variable", value_name="value")
        for var_name, grp in melted.groupby("variable"):
            ax.plot(grp["x"], grp["value"], label=str(var_name))

        fit_str = _format_coefficients(coefficients)
        ax.text(
            0.02, 0.98, fit_str, transform=ax.transAxes,
            fontsize=7, verticalalignment="top", horizontalalignment="left",
        )
        time_str = f"{time_res} {units[1]}"
        if mode == "ECDF":
            ax.text(
                0.0, 0.98, time_str, transform=ax.transAxes,
                fontsize=7, verticalalignment="bottom", horizontalalignment="right",
            )
        else:
            ax.text(
                0.98, 0.02, time_str, transform=ax.transAxes,
                fontsize=7, verticalalignment="bottom", horizontalalignment="right",
            )
    except RuntimeError:
        logger.warning(
            "Failed to fit jump distances with %d population(s). "
            "Try different parameters for init and/or n_pop.",
            n_pop,
        )

    ax.legend().set_visible(False)
    fig.tight_layout()

    return JDFitResult(coefficients=coefficients, fit_curve=fit_df, fig=fig, ax=ax)


def _auto_guess(
    jumps: np.ndarray, n_pop: int, mode: str, time_res: float, breaks: int
) -> dict:
    if mode == "ECDF":
        guess = ((np.mean(jumps) / time_res) / 4) / breaks
        if n_pop == 1:
            return {"D1": guess}
        if n_pop == 2:
            return {"D1": guess, "D2": 0.4, "D3": guess * 10}
        return {"D1": guess, "D2": guess * 10, "D3": guess * 100, "D4": 0.2, "D6": 0.2}
    else:
        if n_pop == 1:
            return {"D2": 100, "D1": 0.1}
        if n_pop == 2:
            return {"D2": 0.01, "D1": 0.1, "D3": 10, "D4": 100}
        return {"D2": 1, "D1": 0.1, "D3": 0.01, "D4": 10, "D5": 100, "D6": 30}


def _fit_1pop(mids, counts, counts_cum, x_fit, mode, init, time_res):
    coefficients = {}
    if mode == "ECDF":
        p0 = [init.get("D1", 0.05)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lambda r, D1: _ecdf_1pop(r, D1, time_res),
                mids, counts_cum, p0=p0, maxfev=10000,
            )
        D1 = popt[0]
        y = _ecdf_1pop(x_fit, D1, time_res)
        coefficients = {"D": D1}
    else:
        p0 = [init.get("D2", 100), init.get("D1", 0.1)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lambda r, D2, D1: _pdf_1pop(r, D2, D1, time_res),
                mids, counts, p0=p0, maxfev=10000,
            )
        D2, D1 = popt
        y = _pdf_1pop(x_fit, D2, D1, time_res)
        coefficients = {"D": D1, "A": D2}

    fit_df = pd.DataFrame({"x": x_fit, "y": y})
    return coefficients, fit_df


def _fit_2pop(mids, counts, counts_cum, x_fit, mode, init, time_res):
    coefficients = {}
    if mode == "ECDF":
        p0 = [init.get("D1", 0.001), init.get("D2", 0.4), init.get("D3", 0.1)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lambda r, D1, D2, D3: _ecdf_2pop(r, D1, D2, D3, time_res),
                mids, counts_cum, p0=p0, maxfev=10000,
            )
        D1, D2_frac, D3 = popt
        y1 = D2_frac - D2_frac * np.exp(-x_fit**2 / (4 * D1 * time_res))
        y2 = (1 - D2_frac) - (1 - D2_frac) * np.exp(-x_fit**2 / (4 * D3 * time_res))
        coefficients = {"D1": D1, "A1": D2_frac, "D2": D3, "A2": 1 - D2_frac}
    else:
        p0 = [init.get("D1", 0.1), init.get("D2", 0.01), init.get("D3", 10), init.get("D4", 100)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lambda r, D1, D2, D3, D4: _pdf_2pop(r, D1, D2, D3, D4, time_res),
                mids, counts, p0=p0, maxfev=10000,
            )
        D1, D2, D3, D4 = popt
        y1 = D4 * x_fit / (2 * D1 * time_res) * np.exp(-x_fit**2 / (4 * D1 * time_res))
        y2 = D3 * x_fit / (2 * D2 * time_res) * np.exp(-x_fit**2 / (4 * D2 * time_res))
        coefficients = {"D1": D1, "A1": D4, "D2": D2, "A2": D3}

    y = y1 + y2
    fit_df = pd.DataFrame({"x": x_fit, "y": y, "y1": y1, "y2": y2})
    return coefficients, fit_df


def _fit_3pop(mids, counts, counts_cum, x_fit, mode, init, time_res):
    coefficients = {}
    if mode == "ECDF":
        p0 = [
            init.get("D1", 0.1), init.get("D2", 0.8),
            init.get("D3", 0.01), init.get("D4", 0.2), init.get("D6", 0.2),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lambda r, D1, D2, D3, D4, D6: _ecdf_3pop(r, D1, D2, D3, D4, D6, time_res),
                mids, counts_cum, p0=p0, maxfev=10000,
            )
        D1, D2, D3, D4, D6 = popt
        D5 = 1 - D4 - D6
        y1 = D4 - D4 * np.exp(-x_fit**2 / (4 * D1 * time_res))
        y2 = D5 - D5 * np.exp(-x_fit**2 / (4 * D2 * time_res))
        y3 = D6 - D6 * np.exp(-x_fit**2 / (4 * D3 * time_res))
        coefficients = {"D1": D1, "A1": D4, "D2": D2, "A2": D5, "D3": D3, "A3": D6}
    else:
        p0 = [
            init.get("D1", 0.1), init.get("D2", 1), init.get("D3", 0.01),
            init.get("D4", 10), init.get("D5", 100), init.get("D6", 30),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lambda r, D1, D2, D3, D4, D5, D6: _pdf_3pop(r, D1, D2, D3, D4, D5, D6, time_res),
                mids, counts, p0=p0, maxfev=10000,
            )
        D1, D2, D3, D4, D5, D6 = popt
        y1 = D4 * x_fit / (2 * D1 * time_res) * np.exp(-x_fit**2 / (4 * D1 * time_res))
        y2 = D5 * x_fit / (2 * D2 * time_res) * np.exp(-x_fit**2 / (4 * D2 * time_res))
        y3 = D6 * x_fit / (2 * D3 * time_res) * np.exp(-x_fit**2 / (4 * D3 * time_res))
        coefficients = {"D1": D1, "A1": D4, "D2": D2, "A2": D5, "D3": D3, "A3": D6}

    y = y1 + y2 + y3
    fit_df = pd.DataFrame({"x": x_fit, "y": y, "y1": y1, "y2": y2, "y3": y3})
    return coefficients, fit_df


def _format_coefficients(coefficients: dict[str, float]) -> str:
    lines = []
    for k, v in coefficients.items():
        lines.append(f"{k} = {v:.4f}")
    return "\n".join(lines)
