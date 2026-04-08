#!/usr/bin/env python3
"""Plot same-trial waveform overlay grids from a persistent manifest."""

import argparse
import json
import pickle
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

from core.evaluation.metrics import calculate_cross_corr_alignment


VARIANT_ORDER = ["Base", "KFstd", "PARH"]
VARIANT_COLORS = {
    "Base": "#1f3a5f",
    "KFstd": "#c06c2b",
    "PARH": "#116b4f",
}


def _zscore(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=float).flatten()
    std = np.std(sig)
    if not np.isfinite(std) or std < 1e-9:
        return sig * 0.0
    return (sig - np.mean(sig)) / std


def _resolve_pkl(row: pd.Series, run_dir: Path) -> Path:
    data_file = Path(str(row["data_file"]))
    if data_file.is_absolute():
        return data_file
    return run_dir / data_file


def _load_estimate(payload: dict, method: str) -> dict:
    for item in payload.get("estimates", []):
        if item.get("method") == method:
            return item.get("estimate", {})
    raise KeyError(f"Method {method} not found in payload")


def _series_for_row(row: pd.Series, payload: dict) -> tuple[np.ndarray, np.ndarray, float]:
    est = _load_estimate(payload, str(row["method"]))
    variant = str(row["variant"]) if "variant" in row.index else str(row.name)
    output_type = str(row.get("output_type", "z_full" if variant == "PARH" else "signal_hat"))
    if output_type not in est:
        output_type = "z_full" if "z_full" in est else "signal_hat"
    pred = np.asarray(est[output_type], dtype=float).flatten()
    gt = np.asarray(payload["gt"], dtype=float).flatten()
    fs_est = float(payload.get("fps", 20.0))
    fs_gt = float(payload.get("fs_gt", fs_est))
    aligned_pred, aligned_gt, _ = calculate_cross_corr_alignment(pred, gt, fs_est=fs_est, fs_gt=fs_gt)
    return _zscore(aligned_pred), _zscore(aligned_gt), fs_gt


def _window_signals(pred: np.ndarray, gt: np.ndarray, fs: float, window_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = min(len(pred), len(gt))
    pred = pred[:common]
    gt = gt[:common]
    if common == 0:
        return np.array([]), np.array([]), np.array([])
    win = min(common, max(200, int(round(window_sec * fs))))
    start = max((common - win) // 2, 0)
    end = start + win
    t = np.arange(win, dtype=float) / fs
    return t, pred[start:end], gt[start:end]


def _panel_title(row: pd.Series) -> str:
    variant = str(row["variant"]) if "variant" in row.index else str(row.name)
    return (
        f"{variant}\n"
        f"CCC {float(row['waveform_CCC']):.3f} | "
        f"wMAE {float(row['waveform_MAE']):.3f} | "
        f"DTW {float(row['waveform_DTW']):.3f}"
    )


def plot_manifest(manifest_csv: Path, run_dir: Path, out_path: Path, window_sec: float = 18.0):
    manifest = pd.read_csv(manifest_csv)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_csv}")

    case_order = ["top", "median", "bottom"]
    cases = [c for c in case_order if c in set(manifest["case_rank"])]
    fig, axes = plt.subplots(len(cases), len(VARIANT_ORDER), figsize=(12.6, 8.8), sharex=False, sharey=True, constrained_layout=True)
    if len(cases) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, case_rank in enumerate(cases):
        case_df = manifest[manifest["case_rank"] == case_rank].copy()
        case_df = case_df.set_index("variant")
        video_label = str(case_df.iloc[0]["video"]) if not case_df.empty else ""
        for c, variant in enumerate(VARIANT_ORDER):
            ax = axes[r, c]
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="#d8dde6", linewidth=0.7, alpha=0.6)
            if variant not in case_df.index:
                ax.axis("off")
                continue
            row = case_df.loc[variant]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            payload = pickle.load(open(_resolve_pkl(row, run_dir), "rb"))
            pred, gt, fs = _series_for_row(row, payload)
            t, pred_w, gt_w = _window_signals(pred, gt, fs, window_sec)
            ax.plot(t, gt_w, color="#111111", linewidth=2.0, label="GT")
            ax.plot(t, pred_w, color=VARIANT_COLORS[variant], linewidth=1.8, label=variant)
            ax.set_title(_panel_title(row), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{case_rank.capitalize()}\n{video_label}", fontsize=10)
            if r == len(cases) - 1:
                ax.set_xlabel("Time (s)")
            ax.set_ylim(-3.2, 3.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    family = str(manifest.iloc[0]["family"])
    dataset = str(manifest.iloc[0]["dataset"])
    fig.suptitle(f"{dataset} {family} Same-Trial Overlay: Base vs KFstd vs PARH", fontsize=13, y=1.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    root = ROOT
    parser = argparse.ArgumentParser(description="Plot waveform overlay grid from a persistent manifest.")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory that contains the `data/` folder referenced by manifest data_file paths.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=18.0,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "paper" / "figures" / "F4_waveform_overlay_grid.pdf",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_manifest(
        manifest_csv=args.manifest_csv,
        run_dir=args.run_dir,
        out_path=args.out,
        window_sec=args.window_sec,
    )
    print(json.dumps({"manifest": str(args.manifest_csv), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
