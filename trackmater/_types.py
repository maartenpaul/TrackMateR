from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Calibration:
    pixel_size: float
    time_interval: float
    spatial_unit: str
    temporal_unit: str
    width: float
    height: float
    n_traces: int
    max_frames: float
    channel: int = 1

    def _repr_html_(self) -> str:
        rows = [
            ("Pixel size", f"{self.pixel_size} {self.spatial_unit}"),
            ("Time interval", f"{self.time_interval} {self.temporal_unit}"),
            ("Image size", f"{self.width:.1f} x {self.height:.1f} {self.spatial_unit}"),
            ("Tracks", str(self.n_traces)),
            ("Max frames", f"{self.max_frames:.0f}"),
            ("Channel", str(self.channel)),
        ]
        html = "<table><tr><th>Parameter</th><th>Value</th></tr>"
        for k, v in rows:
            html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        html += "</table>"
        return html


@dataclass
class TrackMateData:
    tracks: pd.DataFrame
    calibration: Calibration

    @property
    def n_tracks(self) -> int:
        return self.tracks["trace"].nunique()

    @property
    def trace_ids(self) -> np.ndarray:
        return self.tracks["trace"].unique()

    def correct(
        self,
        xy_scalar: float = 1.0,
        t_scalar: float = 1.0,
        xy_unit: str | None = None,
        t_unit: str | None = None,
    ) -> TrackMateData:
        from .calibration import correct_data

        return correct_data(self, xy_scalar=xy_scalar, t_scalar=t_scalar, xy_unit=xy_unit, t_unit=t_unit)

    def calculate_msd(
        self, method: str = "timeaveraged", n: int = 4, short: int = 0
    ) -> MSDResult:
        from .analysis import calculate_msd

        return calculate_msd(self.tracks, method=method, n=n, short=short)

    def calculate_jd(
        self,
        delta_t: int = 1,
        n_pop: int = 2,
        mode: str = "ECDF",
        init: dict | None = None,
        time_res: float = 1.0,
        breaks: int = 100,
    ) -> JDResult:
        from .analysis import calculate_jd

        return calculate_jd(
            self, delta_t=delta_t, n_pop=n_pop, mode=mode, init=init,
            time_res=time_res, breaks=breaks,
        )

    def calculate_fd(self) -> FDResult:
        from .analysis import calculate_fd

        return calculate_fd(self)

    def calculate_track_density(self, radius: float = 1.0) -> TrackDensityResult:
        from .analysis import calculate_track_density

        return calculate_track_density(self, radius=radius)

    def report(self, **kwargs: Any) -> Any:
        from .report import report_dataset

        return report_dataset(self, **kwargs)

    def _repr_html_(self) -> str:
        cal = self.calibration
        html = (
            f"<b>TrackMateData</b>: {self.n_tracks} tracks, "
            f"{len(self.tracks)} points<br>"
            f"Units: {cal.pixel_size} {cal.spatial_unit}, "
            f"{cal.time_interval} {cal.temporal_unit}<br>"
        )
        html += self.tracks.head(5).to_html(max_cols=10)
        if len(self.tracks) > 5:
            html += f"<p>... {len(self.tracks) - 5} more rows</p>"
        return html


@dataclass
class MSDResult:
    summary: pd.DataFrame
    alpha: pd.DataFrame
    cve: pd.DataFrame
    track_msd: pd.DataFrame

    def _repr_html_(self) -> str:
        return (
            f"<b>MSDResult</b>: {len(self.summary)} time lags, "
            f"{self.alpha['trace'].nunique()} tracks<br>"
            + self.summary.head(5).to_html()
        )


@dataclass
class JDParams:
    jumptime: float
    delta_t: int
    n_pop: int = 2
    mode: str = "ECDF"
    init: dict | None = None
    spatial_unit: str = "um"
    temporal_unit: str = "s"
    time_res: float = 1.0
    breaks: int = 100


@dataclass
class JDResult:
    jump_distances: pd.DataFrame
    params: JDParams

    def _repr_html_(self) -> str:
        return (
            f"<b>JDResult</b>: {len(self.jump_distances)} jumps, "
            f"deltaT={self.params.delta_t}<br>"
        )


@dataclass
class FDResult:
    data: pd.DataFrame

    def _repr_html_(self) -> str:
        return f"<b>FDResult</b>: {len(self.data)} tracks<br>" + self.data.head(5).to_html()


@dataclass
class TrackDensityResult:
    data: pd.DataFrame

    def _repr_html_(self) -> str:
        return (
            f"<b>TrackDensityResult</b>: {len(self.data)} tracks<br>"
            + self.data.head(5).to_html()
        )


@dataclass
class JDFitResult:
    coefficients: dict[str, float]
    fit_curve: pd.DataFrame
    fig: Any = None
    ax: Any = None


@dataclass
class SummaryStats:
    alpha: float = 0.0
    speed: float = 0.0
    intensity: float = 0.0
    duration: float = 0.0
    dee: float = 0.0
    displacement: float = 0.0
    neighbours: float = 0.0
    fd: float = 0.0
    width: float = 0.0
    condition: str = ""
    dataid: str = ""
