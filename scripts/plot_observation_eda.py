#!/usr/bin/env python3
"""Plot observation-family / preprocessing EDA heatmaps for the manuscript."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STAGE_ORDER = [
    "raw",
    "detrend_only",
    "bandpass_only",
    "sign_align_only",
    "robust_zscore_only",
    "current_preprocess",
    "helper_preprocess",
]

STAGE_LABELS = {
    "raw": "Raw",
    "detrend_only": "Detrend",
    "bandpass_only": "Bandpass",
    "sign_align_only": "Sign",
    "robust_zscore_only": "Robust-z",
    "current_preprocess": "Current",
    "helper_preprocess": "Helper",
}

FAMILY_ORDER = ["OF", "OF_bridge", "DoF", "P1D_lin", "P1D_quad", "P1D_cub"]


def _parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Plot observation EDA heatmaps.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=root / "analysis" / "cohface_preproc_summary.csv",
    )
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=root / "analysis" / "cohface_preproc_deltas.csv",
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


def _draw_heatmap(ax, arr, title, cmap, vmin=None, vmax=None, value_fmt="{:.2f}"):
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(FAMILY_ORDER)))
    ax.set_yticklabels(FAMILY_ORDER, fontsize=9)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(j, i, value_fmt.format(val), ha="center", va="center", fontsize=7, color="black")
    return im


def main():
    args = _parse_args()
    summary = pd.read_csv(args.summary_csv)
    delta = pd.read_csv(args.delta_csv)

    args.out_main.parent.mkdir(parents=True, exist_ok=True)
    args.out_supp.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    metrics_main = [
        ("corr_wave_best_median", "Waveform Corr (median)", "YlGnBu", 0.0, 1.0),
        ("corr_deriv_best_median", "Derivative Corr (median)", "YlGnBu", 0.0, 1.0),
        ("peak_error_hz_median", "Peak Error Hz (median)", "YlOrRd_r", -0.08, 0.08),
        ("highfreq_energy_ratio_median", "High-Freq Energy Ratio (median)", "YlOrRd_r", 0.0, 1.0),
    ]
    for ax, (metric, title, cmap, vmin, vmax) in zip(axes.flat, metrics_main):
        arr = _pivot_metric(summary, metric)
        im = _draw_heatmap(ax, arr, title, cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Observation-family and preprocessing-stage summary", fontsize=13)
    fig.savefig(args.out_main, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    metrics_delta = [
        ("corr_wave_best_delta_median", "Waveform Corr Δ vs Raw", "RdYlGn", -0.4, 0.4),
        ("ccc_wave_best_z_delta_median", "Waveform CCC Δ vs Raw", "RdYlGn", -0.4, 0.4),
        ("corr_deriv_best_delta_median", "Derivative Corr Δ vs Raw", "RdYlGn", -0.4, 0.7),
        ("peak_error_hz_delta_median", "Peak Error Hz Δ vs Raw", "RdYlGn_r", -0.08, 0.08),
        ("lowfreq_energy_ratio_delta_median", "Low-Freq Energy Δ vs Raw", "RdYlGn_r", -0.6, 0.2),
        ("highfreq_energy_ratio_delta_median", "High-Freq Energy Δ vs Raw", "RdYlGn_r", -0.2, 1.0),
    ]
    for ax, (metric, title, cmap, vmin, vmax) in zip(axes.flat, metrics_delta):
        arr = _pivot_metric(delta, metric)
        im = _draw_heatmap(ax, arr, title, cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Preprocessing-stage deltas relative to raw observation", fontsize=13)
    fig.savefig(args.out_supp, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved main figure: {args.out_main}")
    print(f"Saved supplementary figure: {args.out_supp}")


if __name__ == "__main__":
    main()
