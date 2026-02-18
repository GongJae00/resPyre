#!/usr/bin/env python3
"""
Generate compact paper summary plots from run metrics CSVs.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _extract_metric(df: pd.DataFrame, metric: str) -> np.ndarray:
    col = f"{metric}_median" if f"{metric}_median" in df.columns else metric
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="results/<run_name>")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    t_csv = os.path.join(run_dir, "metrics", "metrics_time_domain_summary.csv")
    f_csv = os.path.join(run_dir, "metrics", "metrics_freq_domain_summary.csv")
    if not (os.path.exists(t_csv) and os.path.exists(f_csv)):
        raise FileNotFoundError("Missing time/freq summary CSVs.")

    tdf = pd.read_csv(t_csv)
    fdf = pd.read_csv(f_csv)
    if "Method" not in tdf.columns:
        tdf["Method"] = tdf["method"]
    if "Method" not in fdf.columns:
        fdf["Method"] = fdf["method"]

    methods = tdf["Method"].astype(str).tolist()
    x = np.arange(len(methods))
    width = 0.36

    mae_t = _extract_metric(tdf, "MAE")
    rmse_t = _extract_metric(tdf, "RMSE")
    mae_f = _extract_metric(fdf, "MAE")
    pear_f = _extract_metric(fdf, "PearsonR")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes = axes.ravel()

    axes[0].bar(x - width / 2, mae_t, width, label="Time MAE", color="#2a9d8f")
    axes[0].bar(x + width / 2, rmse_t, width, label="Time RMSE", color="#264653")
    axes[0].set_title("Waveform Fidelity")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar(x - width / 2, mae_f, width, label="Freq MAE", color="#e76f51")
    axes[1].bar(x + width / 2, pear_f, width, label="PearsonR", color="#f4a261")
    axes[1].set_title("Rate Accuracy")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend()

    # Best-method ranking by freq MAE
    order = np.argsort(mae_f)
    rank_methods = [methods[i] for i in order]
    rank_vals = mae_f[order]
    axes[2].barh(np.arange(len(rank_methods)), rank_vals, color="#457b9d")
    axes[2].set_yticks(np.arange(len(rank_methods)))
    axes[2].set_yticklabels(rank_methods)
    axes[2].invert_yaxis()
    axes[2].set_title("Ranking by Freq MAE (lower better)")
    axes[2].grid(axis="x", alpha=0.3)

    # Correlation between time and freq MAE
    axes[3].scatter(mae_t, mae_f, color="#1d3557", alpha=0.8)
    for i, m in enumerate(methods):
        axes[3].annotate(str(i + 1), (mae_t[i], mae_f[i]), fontsize=8, alpha=0.8)
    axes[3].set_xlabel("Time MAE")
    axes[3].set_ylabel("Freq MAE")
    axes[3].set_title("Method Trade-off (Time vs Freq)")
    axes[3].grid(alpha=0.3)

    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")

    fig.suptitle(f"Performance Summary — {os.path.basename(run_dir)}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "paper_performance_summary.png")
    plt.savefig(out_path, dpi=240)
    plt.close(fig)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()

