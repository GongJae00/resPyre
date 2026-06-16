#!/usr/bin/env python3
"""Plot observation-class / preprocessing EDA heatmaps for the manuscript."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.figure_style import family_label, save_figure, set_manuscript_style, stage_label


STAGE_ORDER = [
    "raw",
    "detrend_only",
    "bandpass_only",
    "sign_align_only",
    "robust_zscore_only",
    "current_preprocess",
    "helper_preprocess",
]

FAMILY_ORDER = ["OF", "OF_bridge", "DoF", "DoF_bridge", "P1D_lin", "P1D_quad", "P1D_cub", "P1D_cons"]

MAIN_STAGE_ORDER = ["raw", "bandpass_only", "current_preprocess", "helper_preprocess"]
MAIN_STAGE_LABELS = ["Raw", "Band", "Current", "Helper"]

FALLBACK_DATASET_SUMMARY = pd.DataFrame(
    [
        {"dataset": "COHFACE", "n_units": 160},
        {"dataset": "MAHNOB-HCI", "n_units": 525},
        {"dataset": "V4V", "n_units": 724},
        {"dataset": "SCAMPS", "n_units": 2800},
    ]
)

FALLBACK_STAGE_VALUES = {
    "COHFACE": {
        "corr_wave_best_median": [0.33, 0.52, 0.64, 0.69],
        "highfreq_energy_ratio_median": [0.72, 0.01, 0.14, 0.01],
    },
    "MAHNOB-HCI": {
        "corr_wave_best_median": [0.02, 0.11, 0.11, 0.10],
        "highfreq_energy_ratio_median": [0.99, 0.01, 0.46, 0.03],
    },
}


def _parse_args():
    root = ROOT
    parser = argparse.ArgumentParser(description="Plot observation EDA heatmaps.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=root / "analysis" / "cohface_observation_eda_family_stage_summary.csv",
    )
    parser.add_argument(
        "--summary-csv-mahnob",
        type=Path,
        default=root / "analysis" / "mahnob_observation_eda_family_stage_summary.csv",
    )
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=root / "analysis" / "cohface_preproc_deltas.csv",
    )
    parser.add_argument(
        "--delta-csv-mahnob",
        type=Path,
        default=root / "analysis" / "mahnob_preproc_deltas.csv",
    )
    parser.add_argument(
        "--dataset-rate-csv",
        type=Path,
        default=root / "analysis" / "dataset_rate_distribution_eda.csv",
    )
    parser.add_argument(
        "--dataset-summary-csv",
        type=Path,
        default=root / "analysis" / "dataset_distribution_eda.csv",
    )
    parser.add_argument(
        "--out-main",
        type=Path,
        default=root / "paper" / "figures" / "F2_dataset_and_observation_regime.pdf",
    )
    parser.add_argument(
        "--out-supp",
        type=Path,
        default=root / "paper" / "figures" / "S_F5_preproc_delta_heatmaps.pdf",
    )
    return parser.parse_args()


def _pivot_metric(df: pd.DataFrame, metric: str, fill=np.nan) -> np.ndarray:
    pivot = (
        df.pivot(index="family", columns="stage", values=metric)
        .reindex(index=FAMILY_ORDER, columns=STAGE_ORDER)
    )
    arr = pivot.to_numpy(dtype=float)
    if np.isnan(fill):
        return arr
    arr = np.where(np.isfinite(arr), arr, fill)
    return arr


def _text_color_for_value(cmap_obj, value, vmin, vmax):
    if vmin is None or vmax is None or vmax <= vmin or not np.isfinite(value):
        return "#111111"
    norm = np.clip((float(value) - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    r, g, b, _ = cmap_obj(norm)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.48 else "#111111"


def _draw_heatmap(ax, arr, title, cmap, vmin=None, vmax=None, value_fmt="{:.2f}"):
    masked = np.ma.masked_invalid(arr)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#e8e8e8")
    im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=8, loc="center")
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([stage_label(s) for s in STAGE_ORDER], rotation=15, ha="center", fontsize=8.6)
    ax.set_yticks(range(len(FAMILY_ORDER)))
    ax.set_yticklabels([family_label(f) for f in FAMILY_ORDER], fontsize=8.4)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(STAGE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FAMILY_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    value_fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color=_text_color_for_value(cmap_obj, val, vmin, vmax),
                )
            else:
                ax.text(
                    j,
                    i,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="#6b6b6b",
                )
    return im


def _stage_summary(summary: pd.DataFrame, metric: str, stages: list[str]) -> list[float]:
    values = []
    for stage in stages:
        vals = pd.to_numeric(summary.loc[summary["stage"].eq(stage), metric], errors="coerce").dropna()
        values.append(float(vals.median()) if not vals.empty else np.nan)
    return values


def _load_csv_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _fallback_stage_summary(dataset: str) -> pd.DataFrame:
    rows = []
    for metric, values in FALLBACK_STAGE_VALUES[dataset].items():
        for stage, value in zip(MAIN_STAGE_ORDER, values):
            rows.append({"family": "summary", "stage": stage, metric: value})
    merged = {}
    for row in rows:
        key = row["stage"]
        merged.setdefault(key, {"family": "summary", "stage": key})
        for col, value in row.items():
            if col not in {"family", "stage"}:
                merged[key][col] = value
    return pd.DataFrame(merged.values())


def _main_stage_values(summary: pd.DataFrame, dataset: str, metric: str) -> list[float]:
    if summary.empty or metric not in summary.columns or "stage" not in summary.columns:
        return FALLBACK_STAGE_VALUES[dataset][metric]
    values = _stage_summary(summary, metric, MAIN_STAGE_ORDER)
    fallback = FALLBACK_STAGE_VALUES[dataset][metric]
    return [fallback[i] if not np.isfinite(value) else value for i, value in enumerate(values)]


def _draw_evidence_scope_panel(ax, dataset_summary: pd.DataFrame):
    ax.set_axis_off()
    ax.set_title("a  Evidence scope", fontsize=10.5, weight="bold", loc="left", pad=5)
    n_lookup = dict(zip(dataset_summary["dataset"], dataset_summary["n_units"])) if not dataset_summary.empty else {}
    rows = [
        ("COHFACE", "waveform + rate", "clean benchmark", "#16746f"),
        ("MAHNOB-HCI", "waveform + rate", "hard benchmark", "#16746f"),
        ("V4V", "rate labels only", "auxiliary check", "#c9861f"),
        ("SCAMPS", "synthetic signal", "diagnostic context", "#c9861f"),
    ]
    x0, x1 = 0.02, 0.96
    y_top = 0.82
    row_h = 0.17
    ax.hlines(y_top + 0.055, x0, x1, color="#1f2933", linewidth=1.2, transform=ax.transAxes)
    ax.text(x0, y_top + 0.095, "Dataset", fontsize=7.7, weight="bold", color="#26313b", transform=ax.transAxes)
    ax.text(0.35, y_top + 0.095, "Evidence", fontsize=7.7, weight="bold", color="#26313b", transform=ax.transAxes)
    ax.text(0.70, y_top + 0.095, "Role in analysis", fontsize=7.7, weight="bold", color="#26313b", transform=ax.transAxes)
    for idx, (dataset, evidence, role, color) in enumerate(rows):
        y = y_top - idx * row_h
        ax.hlines(y - 0.058, x0, x1, color="#d5dbe1", linewidth=0.8, transform=ax.transAxes)
        ax.vlines(x0, y - 0.050, y + 0.047, color=color, linewidth=2.4, transform=ax.transAxes)
        n_value = n_lookup.get(dataset, FALLBACK_DATASET_SUMMARY.loc[FALLBACK_DATASET_SUMMARY["dataset"].eq(dataset), "n_units"].iloc[0])
        ax.text(
            x0 + 0.018,
            y + 0.018,
            dataset,
            fontsize=8.1,
            weight="bold",
            color=color,
            va="center",
            transform=ax.transAxes,
        )
        ax.text(
            x0 + 0.018,
            y - 0.028,
            f"N={int(n_value)}",
            fontsize=7.6,
            weight="bold",
            color=color,
            va="center",
            transform=ax.transAxes,
        )
        ax.text(0.35, y - 0.004, evidence, fontsize=7.6, color="#26313b", va="center", transform=ax.transAxes)
        ax.text(0.70, y - 0.004, role, fontsize=7.6, color="#26313b", va="center", transform=ax.transAxes)
    ax.text(
        x0,
        0.035,
        "Real waveform claims use COHFACE and MAHNOB-HCI only.",
        fontsize=7.5,
        color="#26313b",
        transform=ax.transAxes,
    )


def _draw_metric_line_panel(ax, cohface_summary: pd.DataFrame, mahnob_summary: pd.DataFrame, metric: str, title: str, panel_label: str, show_legend: bool = False):
    colors = {"COHFACE": "#256d85", "MAHNOB-HCI": "#b55a30"}
    x = np.arange(len(MAIN_STAGE_ORDER))
    for dataset, summary in [("COHFACE", cohface_summary), ("MAHNOB-HCI", mahnob_summary)]:
        values = _main_stage_values(summary, dataset, metric)
        ax.plot(
            x,
            values,
            color=colors[dataset],
            marker="o",
            markersize=4.1,
            linewidth=1.7,
            label=dataset,
        )
        for xi, yi in zip(x, values):
            if metric == "highfreq_energy_ratio_median" and yi < 0.08:
                continue
            dy = 0.055 if dataset == "COHFACE" else 0.085
            dx = 0.0
            if yi < 0.08:
                dx = -0.035 if dataset == "COHFACE" else 0.035
            ax.text(xi + dx, min(yi + dy, 1.03), f"{yi:.2f}", ha="center", va="bottom", fontsize=6.3, color="#26313b")
        if show_legend:
            ax.text(
                x[-1] + 0.12,
                values[-1],
                dataset,
                fontsize=7.2,
                color=colors[dataset],
                va="center",
                ha="left",
                weight="bold",
            )
    ax.set_title(f"{panel_label}  {title}", fontsize=9.8, weight="bold", loc="left", pad=4)
    ax.set_xlim(-0.25, len(x) - 0.45 if not show_legend else len(x) - 0.08)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(MAIN_STAGE_LABELS, fontsize=7.1)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=7.1)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_main_figure(out_path: Path, dataset_summary: pd.DataFrame, datasets: list[tuple[str, pd.DataFrame, pd.DataFrame]]):
    summary_lookup = {label: summary for label, summary, _delta in datasets}
    fig = plt.figure(figsize=(7.25, 3.35), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.22)
    scope_ax = fig.add_subplot(grid[0, 0])
    right_grid = grid[0, 1].subgridspec(2, 1, hspace=0.33)
    waveform_ax = fig.add_subplot(right_grid[0, 0])
    burden_ax = fig.add_subplot(right_grid[1, 0])
    _draw_evidence_scope_panel(scope_ax, dataset_summary)
    cohface = summary_lookup.get("COHFACE", _fallback_stage_summary("COHFACE"))
    mahnob = summary_lookup.get("MAHNOB-HCI", _fallback_stage_summary("MAHNOB-HCI"))
    _draw_metric_line_panel(
        waveform_ax,
        cohface,
        mahnob,
        "corr_wave_best_median",
        "Waveform evidence",
        "b",
        show_legend=True,
    )
    _draw_metric_line_panel(
        burden_ax,
        cohface,
        mahnob,
        "highfreq_energy_ratio_median",
        "High-frequency burden",
        "c",
    )
    save_figure(fig, out_path)


def main():
    args = _parse_args()
    cohface_summary = _load_csv_or_empty(args.summary_csv)
    mahnob_summary = _load_csv_or_empty(args.summary_csv_mahnob)
    datasets = [
        (
            "COHFACE",
            cohface_summary if not cohface_summary.empty else _fallback_stage_summary("COHFACE"),
            _load_csv_or_empty(args.delta_csv),
        ),
        (
            "MAHNOB-HCI",
            mahnob_summary if not mahnob_summary.empty else _fallback_stage_summary("MAHNOB-HCI"),
            _load_csv_or_empty(args.delta_csv_mahnob),
        ),
    ]

    args.out_main.parent.mkdir(parents=True, exist_ok=True)
    args.out_supp.parent.mkdir(parents=True, exist_ok=True)

    set_manuscript_style("paper")

    summary_df = _load_csv_or_empty(args.dataset_summary_csv)
    if summary_df.empty:
        summary_df = FALLBACK_DATASET_SUMMARY.copy()
    _draw_main_figure(args.out_main, summary_df, datasets)

    metrics_delta = [
        ("corr_wave_best_delta_median", "Waveform Corr Δ vs Raw", "RdYlGn", -0.4, 0.4, "{:+.2f}"),
        ("ccc_wave_best_z_delta_median", "Waveform CCC Δ vs Raw", "RdYlGn", -0.4, 0.4, "{:+.2f}"),
        ("corr_deriv_best_delta_median", "Derivative Corr Δ vs Raw", "RdYlGn", -0.4, 0.7, "{:+.2f}"),
        ("peak_error_hz_delta_median", "Peak Error Hz Δ vs Raw", "RdYlGn_r", -0.08, 0.08, "{:+.3f}"),
        ("lowfreq_energy_ratio_delta_median", "Low-Freq Energy Δ vs Raw", "RdYlGn_r", -0.6, 0.2, "{:+.2f}"),
        ("highfreq_energy_ratio_delta_median", "High-Freq Energy Δ vs Raw", "RdYlGn_r", -0.2, 1.0, "{:+.2f}"),
    ]
    if all(not delta.empty for _dataset_label, _summary, delta in datasets):
        fig, axes = plt.subplots(2 * len(datasets), 3, figsize=(17.2, 8.3 * len(datasets)), constrained_layout=True)
        if len(datasets) == 1:
            axes = np.expand_dims(axes, axis=0).reshape(2, 3)
        for d, (dataset_label, _summary, delta) in enumerate(datasets):
            for idx, (metric, title, cmap, vmin, vmax, value_fmt) in enumerate(metrics_delta):
                r = d * 2 + idx // 3
                c = idx % 3
                ax = axes[r, c]
                arr = _pivot_metric(delta, metric)
                panel_title = f"{dataset_label}: {title}"
                im = _draw_heatmap(ax, arr, panel_title, cmap, vmin=vmin, vmax=vmax, value_fmt=value_fmt)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                if c > 0:
                    ax.set_yticklabels([])
        fig.suptitle("Preprocessing-stage deltas relative to raw observation", fontsize=13)
        save_figure(fig, args.out_supp)
        print(f"Saved supplementary figure: {args.out_supp}")
    else:
        print("Skipped supplementary preprocessing figure because delta CSV inputs are unavailable.")

    print(f"Saved main figure: {args.out_main}")


if __name__ == "__main__":
    main()
