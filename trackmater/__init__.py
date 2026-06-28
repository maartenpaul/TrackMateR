"""trackmater - Analysis of TrackMate (ImageJ/Fiji) single-particle tracking data."""

__version__ = "0.1.0"

from ._types import (
    Calibration,
    FDResult,
    JDFitResult,
    JDParams,
    JDResult,
    MSDResult,
    SummaryStats,
    TrackDensityResult,
    TrackMateData,
)
from .analysis import (
    calculate_alpha,
    calculate_cve,
    calculate_fd,
    calculate_jd,
    calculate_msd,
    calculate_track_density,
)
from .calibration import correct_data
from .comparison import compare_dataset_values, compare_datasets, make_comparison
from .fitting import fit_jd
from .io import read_gt_file, read_trackmate_xml, to_csv
from .plotting import (
    plot_alpha,
    plot_cumulative_distance,
    plot_dee,
    plot_displacement_hist,
    plot_displacement_over_time,
    plot_duration_hist,
    plot_fd,
    plot_intensity_hist,
    plot_msd,
    plot_multi_msd,
    plot_neighbours,
    plot_speed,
    plot_tracks,
    plot_width,
)
from .report import make_summary_report, report_dataset

__all__ = [
    # Types
    "Calibration",
    "TrackMateData",
    "MSDResult",
    "JDParams",
    "JDResult",
    "JDFitResult",
    "FDResult",
    "TrackDensityResult",
    "SummaryStats",
    # I/O
    "read_trackmate_xml",
    "read_gt_file",
    "to_csv",
    # Calibration
    "correct_data",
    # Analysis
    "calculate_msd",
    "calculate_alpha",
    "calculate_cve",
    "calculate_jd",
    "calculate_fd",
    "calculate_track_density",
    # Fitting
    "fit_jd",
    # Plotting
    "plot_tracks",
    "plot_displacement_over_time",
    "plot_cumulative_distance",
    "plot_displacement_hist",
    "plot_intensity_hist",
    "plot_duration_hist",
    "plot_speed",
    "plot_alpha",
    "plot_dee",
    "plot_msd",
    "plot_multi_msd",
    "plot_neighbours",
    "plot_fd",
    "plot_width",
    # Reports
    "make_summary_report",
    "report_dataset",
    # Comparison
    "compare_datasets",
    "compare_dataset_values",
    "make_comparison",
]
