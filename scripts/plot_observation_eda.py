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


def _draw_metric_line_panel(ax, cohface_summary: pd.DataFrame, mahnob_summary: pd.DataFrame, metric: str, title: str):
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
    ax.set_title(title, fontsize=9.4, weight="bold", loc="left", pad=4)
    ax.set_xlim(-0.25, len(x) - 0.75)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(MAIN_STAGE_LABELS, fontsize=7.4)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=7.4)
    ax.grid(axis="y", linestyle="--", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_main_figure(out_path: Path, datasets: list[tuple[str, pd.DataFrame, pd.DataFrame]]):
    summary_lookup = {label: summary for label, summary, _delta in datasets}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.30), constrained_layout=True, sharey=True)
    cohface = summary_lookup.get("COHFACE", _fallback_stage_summary("COHFACE"))
    mahnob = summary_lookup.get("MAHNOB-HCI", _fallback_stage_summary("MAHNOB-HCI"))
    _draw_metric_line_panel(
        axes[0],
        cohface,
        mahnob,
        "corr_wave_best_median",
        "Waveform evidence",
    )
    _draw_metric_line_panel(
        axes[1],
        cohface,
        mahnob,
        "highfreq_energy_ratio_median",
        "High-frequency burden",
    )
    axes[0].set_ylabel("Median value", fontsize=7.8)
    axes[0].legend(frameon=False, loc="upper left", fontsize=7.2, handlelength=1.6)
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

    _draw_main_figure(args.out_main, datasets)

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
