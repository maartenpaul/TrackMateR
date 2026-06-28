from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ._types import SummaryStats, TrackMateData
from .analysis import calculate_fd, calculate_jd, calculate_msd, calculate_track_density
from .calibration import correct_data
from .io import read_trackmate_xml
from .report import make_summary_report
from .utils import find_log2_y_limits, merge_dataframes_for_export, process_defaults, setup_output_path

logger = logging.getLogger(__name__)


def _process_single_file(args: tuple) -> dict | None:
    (
        file_path, cond_name, file_idx, calibrate, calib_df,
        params, output_dir
    ) = args

    data = read_trackmate_xml(file_path, slim=True)
    if data is None:
        logger.warning("Skipping %s - no data found!", file_path)
        return None

    if calibrate and calib_df is not None:
        cal = data.calibration
        xy_scalar = calib_df.iloc[0, 0] / cal.pixel_size if cal.pixel_size != 0 else 1.0
        t_scalar = calib_df.iloc[1, 0] / cal.time_interval if cal.time_interval != 0 else 1.0

        if xy_scalar == 0:
            xy_scalar = 1.0
        if t_scalar == 0:
            t_scalar = 1.0
        if 0.975 < xy_scalar < 1.025:
            xy_scalar = 1.0
        if 0.975 < t_scalar < 1.025:
            t_scalar = 1.0

        xy_unit = str(calib_df.iloc[0, 1]) if len(calib_df.columns) > 1 else None
        t_unit = str(calib_df.iloc[1, 1]) if len(calib_df.columns) > 1 else None

        if xy_scalar != 1.0 or t_scalar != 1.0:
            data = correct_data(data, xy_scalar=xy_scalar, t_scalar=t_scalar,
                                xy_unit=xy_unit, t_unit=t_unit)
        else:
            data = correct_data(data, xy_unit=xy_unit, t_unit=t_unit)

    df = data.tracks
    cal = data.calibration

    if cal.n_traces < 3 and cal.max_frames < 10:
        logger.warning("Skipping %s - too few tracks/frames", file_path)
        return None

    dataid = f"{cond_name}_{file_idx}"
    df["dataid"] = dataid

    msd_result = calculate_msd(df, n=params["N"], short=params["short"])
    jd_result = calculate_jd(
        data, delta_t=params["delta_t"], n_pop=params["n_pop"],
        mode=params["mode"], time_res=params["time_res"], breaks=params["breaks"],
    )
    td_result = calculate_track_density(data, radius=params["radius"])
    fd_result = calculate_fd(data)

    result = {
        "tmDF": df,
        "calibration": cal,
        "dataid": dataid,
        "units": (cal.spatial_unit, cal.temporal_unit),
    }

    if msd_result is not None:
        msd_result.summary["dataid"] = dataid
        msd_result.alpha["dataid"] = dataid
        msd_result.cve["dataid"] = dataid
        result["msdDF"] = msd_result.summary
        result["alphaDF"] = msd_result.alpha
        result["deeDF"] = msd_result.cve

    if jd_result is not None:
        jd_result.jump_distances["dataid"] = dataid
        result["jdDF"] = jd_result.jump_distances
        result["jdParams"] = jd_result.params

    if td_result is not None:
        td_result.data["dataid"] = dataid
        result["tdDF"] = td_result.data

    if fd_result is not None:
        fd_result.data["dataid"] = dataid
        result["fdDF"] = fd_result.data

    both = make_summary_report(
        data, msd_result, jd_result, td_result, fd_result,
        title=cond_name, subtitle=Path(file_path).stem,
        msd_scale=params.get("msd_scale", "linlin"),
    )
    fig, stats = both

    dest = Path(output_dir) / "Plots" / cond_name
    setup_output_path(dest)
    fig.savefig(dest / f"report_{file_idx}.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)

    if isinstance(stats, SummaryStats):
        report_row = {
            "alpha": stats.alpha, "speed": stats.speed,
            "intensity": stats.intensity, "duration": stats.duration,
            "dee": stats.dee, "displacement": stats.displacement,
            "neighbours": stats.neighbours, "fd": stats.fd,
            "width": stats.width, "condition": cond_name,
            "dataid": dataid,
        }
        result["report"] = report_row

    return result


def compare_datasets(
    datadir: str | Path = "Data",
    n_workers: int | None = None,
    output_dir: str | Path = "Output",
    **kwargs,
) -> pd.DataFrame:
    datadir = Path(datadir)
    output_dir = Path(output_dir)
    params = process_defaults(kwargs)

    cond_folders = sorted([d for d in datadir.iterdir() if d.is_dir()])
    if not cond_folders:
        raise FileNotFoundError(
            "No condition folders found. Organise TrackMate XML files into subfolders."
        )

    mega_msd = []
    mega_alpha = []
    mega_dee = []
    mega_td = []
    mega_fd = []
    mega_report = []
    mega_speed = []
    units_vec = ("um", "s")

    for cond_path in cond_folders:
        cond_name = cond_path.name
        xml_files = sorted(cond_path.glob("*.xml"))
        if not xml_files:
            continue

        csv_files = list(cond_path.glob("*.csv"))
        calibrate = bool(csv_files)
        calib_df = pd.read_csv(csv_files[0]) if calibrate else None

        logger.info("Processing %s", cond_name)

        args_list = [
            (str(f), cond_name, idx + 1, calibrate, calib_df, params, str(output_dir))
            for idx, f in enumerate(xml_files)
        ]

        if n_workers is not None and n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_process_single_file, args_list))
        else:
            results = [_process_single_file(a) for a in args_list]

        bigtm_list, bigmsd_list, bigalpha_list, bigdee_list = [], [], [], []
        bigjd_list, bigtd_list, bigfd_list, bigreport_list = [], [], [], []
        jd_params = None

        for res in results:
            if res is None:
                continue
            units_vec = res.get("units", units_vec)
            bigtm_list.append(res["tmDF"])
            if "msdDF" in res:
                bigmsd_list.append(res["msdDF"])
            if "alphaDF" in res:
                bigalpha_list.append(res["alphaDF"])
            if "deeDF" in res:
                bigdee_list.append(res["deeDF"])
            if "jdDF" in res:
                bigjd_list.append(res["jdDF"])
            if "jdParams" in res:
                jd_params = res["jdParams"]
            if "tdDF" in res:
                bigtd_list.append(res["tdDF"])
            if "fdDF" in res:
                bigfd_list.append(res["fdDF"])
            if "report" in res:
                bigreport_list.append(res["report"])

        bigtm = pd.concat(bigtm_list, ignore_index=True) if bigtm_list else pd.DataFrame()
        bigmsd = pd.concat(bigmsd_list, ignore_index=True) if bigmsd_list else pd.DataFrame()
        bigalpha = pd.concat(bigalpha_list, ignore_index=True) if bigalpha_list else pd.DataFrame()
        bigdee = pd.concat(bigdee_list, ignore_index=True) if bigdee_list else pd.DataFrame()
        bigjd = pd.concat(bigjd_list, ignore_index=True) if bigjd_list else pd.DataFrame()
        bigtd = pd.concat(bigtd_list, ignore_index=True) if bigtd_list else pd.DataFrame()
        bigfd = pd.concat(bigfd_list, ignore_index=True) if bigfd_list else pd.DataFrame()
        bigreport = pd.DataFrame(bigreport_list) if bigreport_list else pd.DataFrame()

        from ._types import Calibration, MSDResult, JDResult, JDParams, TrackDensityResult, FDResult

        dummy_cal = Calibration(1.0, 1.0, units_vec[0], units_vec[1], 0, 0, 0, 0)
        bigtm_obj = TrackMateData(tracks=bigtm, calibration=dummy_cal)
        bigmsd_result = MSDResult(summary=bigmsd, alpha=bigalpha, cve=bigdee, track_msd=pd.DataFrame())
        bigjd_result = JDResult(
            jump_distances=bigjd,
            params=jd_params or JDParams(jumptime=0, delta_t=1),
        ) if not bigjd.empty else None
        bigtd_result = TrackDensityResult(data=bigtd) if not bigtd.empty else None
        bigfd_result = FDResult(data=bigfd) if not bigfd.empty else None

        fig, msd_summary = make_summary_report(
            bigtm_obj, bigmsd_result, bigjd_result, bigtd_result, bigfd_result,
            title=cond_name, subtitle="Summary",
            msd_scale=params.get("msd_scale", "linlin"),
            summary_mode=True,
        )
        dest = output_dir / "Plots" / cond_name
        setup_output_path(dest)
        fig.savefig(dest / "combined.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)

        data_dest = output_dir / "Data" / cond_name
        setup_output_path(data_dest)
        bigtm.to_csv(data_dest / "allTM.csv", index=False)
        bigmsd.to_csv(data_dest / "allMSD.csv", index=False)
        bigjd.to_csv(data_dest / "allJD.csv", index=False)
        bigfd.to_csv(data_dest / "allFD.csv", index=False)

        if isinstance(msd_summary, pd.DataFrame) and not msd_summary.empty:
            msd_summary["condition"] = cond_name
            mega_msd.append(msd_summary)

        if not bigalpha.empty:
            mega_alpha.append(bigalpha)
        if not bigdee.empty:
            mega_dee.append(bigdee)
        if not bigtd.empty:
            mega_td.append(bigtd)
        if not bigfd.empty:
            mega_fd.append(bigfd)
        if not bigreport.empty:
            mega_report.append(bigreport)

        if not bigtm.empty and "cumulative_distance" in bigtm.columns:
            speed_df = bigtm.groupby(["dataid", "trace"]).agg(
                cumdist=("cumulative_distance", "max"),
                cumtime=("track_duration", "max"),
            ).reset_index()
            int_col = "mean_intensity"
            if int_col in bigtm.columns:
                int_agg = bigtm.groupby(["dataid", "trace"])[int_col].max().reset_index()
                speed_df = speed_df.merge(int_agg, on=["dataid", "trace"], how="left")
            speed_df["speed"] = speed_df["cumdist"] / speed_df["cumtime"]
            speed_df["condition"] = cond_name
            mega_speed.append(speed_df)

    megamsd_df = pd.concat(mega_msd, ignore_index=True) if mega_msd else pd.DataFrame()
    megareport_df = pd.concat(mega_report, ignore_index=True) if mega_report else pd.DataFrame()

    data_dest = output_dir / "Data"
    setup_output_path(data_dest)
    if not megamsd_df.empty:
        megamsd_df.to_csv(data_dest / "allMSDCurves.csv", index=False)
    if not megareport_df.empty:
        megareport_df.to_csv(data_dest / "allComparison.csv", index=False)

    all_alpha = pd.concat(mega_alpha, ignore_index=True) if mega_alpha else pd.DataFrame()
    all_dee = pd.concat(mega_dee, ignore_index=True) if mega_dee else pd.DataFrame()
    all_td = pd.concat(mega_td, ignore_index=True) if mega_td else pd.DataFrame()
    all_fd = pd.concat(mega_fd, ignore_index=True) if mega_fd else pd.DataFrame()
    all_speed = pd.concat(mega_speed, ignore_index=True) if mega_speed else pd.DataFrame()

    if not all_dee.empty and "dee" in all_dee.columns:
        all_dee = all_dee.rename(columns={"dee": "estdee"})

    dfs_to_merge = [d for d in [all_alpha, all_dee, all_td, all_speed, all_fd] if not d.empty]
    if dfs_to_merge:
        megatrace = reduce(merge_dataframes_for_export, dfs_to_merge)
        megatrace.to_csv(data_dest / "allTraceData.csv", index=False)

    if not megareport_df.empty and not megamsd_df.empty:
        fig = make_comparison(megareport_df, megamsd_df, units=units_vec,
                              msd_scale=params.get("msd_scale", "linlin"))
        plot_dest = output_dir / "Plots"
        setup_output_path(plot_dest)
        n_conds = len(cond_folders)
        w = 19 if n_conds < 3 else (35 if n_conds > 6 else 25)
        fig.savefig(plot_dest / "comparison.pdf", bbox_inches="tight", dpi=150)
        plt.close(fig)

    return megareport_df


def compare_dataset_values(
    datadir: str | Path = "Data",
    output_dir: str | Path = "Output",
    n_workers: int | None = None,
) -> pd.DataFrame:
    datadir = Path(datadir)
    output_dir = Path(output_dir)

    cond_folders = sorted([d for d in datadir.iterdir() if d.is_dir()])
    if not cond_folders:
        raise FileNotFoundError("No condition folders found.")

    all_dfs = []

    for cond_path in cond_folders:
        cond_name = cond_path.name
        xml_files = sorted(cond_path.glob("*.xml"))
        if not xml_files:
            continue

        csv_files = list(cond_path.glob("*.csv"))
        calibrate = bool(csv_files)
        calib_df = pd.read_csv(csv_files[0]) if calibrate else None

        logger.info("Processing %s", cond_name)

        bigtm_list = []
        for idx, f in enumerate(xml_files):
            data = read_trackmate_xml(str(f), slim=True)
            if data is None:
                continue
            if calibrate and calib_df is not None:
                cal = data.calibration
                xy_scalar = calib_df.iloc[0, 0] / cal.pixel_size if cal.pixel_size != 0 else 1.0
                t_scalar = calib_df.iloc[1, 0] / cal.time_interval if cal.time_interval != 0 else 1.0
                if xy_scalar == 0:
                    xy_scalar = 1.0
                if t_scalar == 0:
                    t_scalar = 1.0
                if 0.975 < xy_scalar < 1.025:
                    xy_scalar = 1.0
                if 0.975 < t_scalar < 1.025:
                    t_scalar = 1.0
                xy_unit = str(calib_df.iloc[0, 1]) if len(calib_df.columns) > 1 else None
                t_unit = str(calib_df.iloc[1, 1]) if len(calib_df.columns) > 1 else None
                if xy_scalar != 1.0 or t_scalar != 1.0:
                    data = correct_data(data, xy_scalar=xy_scalar, t_scalar=t_scalar,
                                        xy_unit=xy_unit, t_unit=t_unit)
                else:
                    data = correct_data(data, xy_unit=xy_unit, t_unit=t_unit)

            df = data.tracks
            df["dataid"] = f"{cond_name}_{idx + 1}"
            bigtm_list.append(df)

        if bigtm_list:
            bigtm = pd.concat(bigtm_list, ignore_index=True)
            data_dest = output_dir / "Data" / cond_name
            setup_output_path(data_dest)
            bigtm.to_csv(data_dest / "allTM.csv", index=False)
            bigtm["condition"] = cond_name
            all_dfs.append(bigtm)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    int_col = "mean_intensity"
    agg_dict = {"frame": "count"}
    if int_col in combined.columns:
        agg_dict[int_col] = "mean"

    summary = combined.groupby(["condition", "dataid", "trace"]).agg(
        frame_duration=("frame", "count"),
        speed=("cumulative_distance", lambda s: s.max() / combined.loc[s.index, "track_duration"].max()
               if combined.loc[s.index, "track_duration"].max() > 0 else np.nan),
    ).reset_index()

    if int_col in combined.columns:
        int_agg = combined.groupby(["condition", "dataid", "trace"])[int_col].mean().reset_index()
        summary = summary.merge(int_agg, on=["condition", "dataid", "trace"], how="left")

    data_dest = output_dir / "Data"
    setup_output_path(data_dest)
    summary.to_csv(data_dest / "per_track_summary.csv", index=False)

    super_summary = summary.groupby(["condition", "dataid"]).agg(
        tracks=("trace", "count"),
        avg_frame_duration=("frame_duration", "mean"),
        avg_speed=("speed", "mean"),
    ).reset_index()
    super_summary.to_csv(data_dest / "per_cell_summary.csv", index=False)

    return summary


def make_comparison(
    report_df: pd.DataFrame,
    msd_df: pd.DataFrame,
    units: tuple[str, str] = ("um", "s"),
    msd_scale: str = "linlin",
    title: str = "Comparison",
    subtitle: str | None = None,
) -> Figure:
    fig = plt.figure(figsize=(25 / 2.54, 25 / 2.54))
    gs = GridSpec(3, 3, figure=fig, hspace=0.6, wspace=0.5)

    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False

    def _boxswarm(ax, data, x, y, ylabel):
        if has_seaborn:
            sns.boxplot(data=data, x=x, y=y, ax=ax, color="lightgrey",
                        fliersize=0, width=0.6)
            sns.stripplot(data=data, x=x, y=y, ax=ax, alpha=0.5, size=3, jitter=True)
        else:
            conditions = data[x].unique()
            for i, cond in enumerate(conditions):
                vals = data.loc[data[x] == cond, y].dropna()
                bp = ax.boxplot(vals, positions=[i], widths=0.6, patch_artist=True,
                                flierprops=dict(marker="", markersize=0))
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightgrey")
                jitter = np.random.normal(0, 0.1, size=len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.5, s=10)
            ax.set_xticks(range(len(conditions)))
            ax.set_xticklabels(conditions, rotation=90)

        ax.set_xlabel("")
        ax.set_ylabel(ylabel)

    plots = [
        (0, 0, "alpha", "Mean alpha"),
        (0, 1, "speed", f"Mean speed ({units[0]}/{units[1]})"),
        (0, 2, "intensity", "Median intensity (AU)"),
        (1, 0, "duration", f"Median duration ({units[1]})"),
        (1, 1, "dee", f"Diffusion coefficient ({units[0]}²/{units[1]})"),
        (1, 2, "fd", "Median fractal dimension"),
        (2, 0, "width", "Median track width"),
        (2, 1, "neighbours", "Median track density"),
    ]

    for row, col, metric, ylabel in plots:
        ax = fig.add_subplot(gs[row, col])
        if metric in report_df.columns:
            _boxswarm(ax, report_df, "condition", metric, ylabel)
            if metric == "alpha":
                ax.axhline(1, linestyle="--", color="grey", linewidth=0.5)
                ax.set_yscale("log", base=2)
                symlim = find_log2_y_limits(report_df["alpha"].dropna())
                ax.set_ylim(symlim)

    # MSD comparison
    ax_msd = fig.add_subplot(gs[2, 2])
    if "condition" in msd_df.columns and "mean" in msd_df.columns:
        for cond, grp in msd_df.groupby("condition"):
            ax_msd.fill_between(grp["t"], grp["mean"] - grp["sd"], grp["mean"] + grp["sd"], alpha=0.2)
            ax_msd.plot(grp["t"], grp["mean"], linewidth=1, label=cond)
    xlog = msd_scale in ("loglog", "linlog")
    ylog = msd_scale in ("loglog", "loglin")
    if xlog:
        ax_msd.set_xscale("log")
    else:
        ax_msd.set_xlim(left=0)
    if ylog:
        ax_msd.set_yscale("log")
    else:
        ax_msd.set_ylim(bottom=0)
    ax_msd.set_xlabel(f"Time ({units[1]})")
    ax_msd.set_ylabel("MSD")

    if title:
        fig.suptitle(f"{title}" + (f"\n{subtitle}" if subtitle else ""), fontsize=10)

    return fig
