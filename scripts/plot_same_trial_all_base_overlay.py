#!/usr/bin/env python3
"""Plot the main same-trial overlay atlas across all fixed base observations.

The main overlay should answer a different question from the
observation-class supplementary atlases: on the same clip, what topology does each
fixed observation operator produce, and how does the promoted PARH-OSSM readout
compare?  This script therefore uses one COHFACE clip and one MAHNOB-HCI clip
from the final full-dataset set, then plots all fixed Base observations, one pre-locked
OSSM-KF comparator, and the final integrated PARH-OSSM readout.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.figure_style import save_figure, set_manuscript_style, style_axis
from core.evaluation.metrics import waveform_ccc, waveform_dtw, waveform_mae


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    video: str
    base_run_dir: Path
    final_run_dir: Path


@dataclass(frozen=True)
class MethodSpec:
    label: str
    method: str
    output_type: str
    source: str
    color: str


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("OF direct", "of_farneback", "signal_hat", "base", "#24476f"),
    MethodSpec("OF bridge", "of_disp_bridge", "signal_hat", "base", "#24476f"),
    MethodSpec("DoF direct", "DoF", "signal_hat", "base", "#24476f"),
    MethodSpec("DoF bridge", "dof_disp_bridge", "signal_hat", "base", "#24476f"),
    MethodSpec("P1D lin", "profile1D linear", "signal_hat", "base", "#24476f"),
    MethodSpec("P1D quad", "profile1D quadratic", "signal_hat", "base", "#24476f"),
    MethodSpec("P1D cub", "profile1D cubic", "signal_hat", "base", "#24476f"),
    MethodSpec("P1D cons", "profile1d_consensus", "signal_hat", "base", "#24476f"),
    MethodSpec("OSSM-KF\n(P1D quad)", "profile1d_quadratic__kfstd", "signal_hat", "base", "#c06c2b"),
    MethodSpec("PARH-OSSM", "parh_ossm", "z_full", "final", "#116b4f"),
)


def _default_datasets() -> tuple[DatasetSpec, ...]:
    return (
        DatasetSpec(
            dataset="COHFACE",
            video="cohface_17_1",
            base_run_dir=ROOT
            / "results/20260409_cohface_prod_ofbridge_dofbridge_p1dcons_e2e_policy_narrow/"
            / "cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons",
            final_run_dir=ROOT / "results/final_full_validation/cohface",
        ),
        DatasetSpec(
            dataset="MAHNOB-HCI",
            video="mahnob_1042",
            base_run_dir=ROOT
            / "results/20260409_mahnob_prod_ofbridge_dofbridge_p1dcons_e2e/"
            / "mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons",
            final_run_dir=ROOT / "results/final_full_validation/mahnob_tailaligned",
        ),
    )


def _zscore(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=float).flatten()
    std = float(np.nanstd(sig))
    if not np.isfinite(std) or std < 1e-9:
        return np.zeros_like(sig)
    return (sig - float(np.nanmean(sig))) / std


def _resample_to_gt(pred: np.ndarray, fs_est: float, fs_gt: float) -> np.ndarray:
    pred = np.asarray(pred, dtype=float).flatten()
    if len(pred) == 0:
        return pred
    if abs(fs_est - fs_gt) <= 1e-6:
        return pred
    t_orig = np.arange(len(pred), dtype=float) / fs_est
    n_target = int(round(len(pred) * fs_gt / fs_est))
    t_new = np.arange(n_target, dtype=float) / fs_gt
    return np.interp(t_new, t_orig, pred)


def _alignment_lag_samples(pred: np.ndarray, gt: np.ndarray, max_lag_samples: int) -> int:
    """Return pred index offset for a bounded visual alignment to reference time."""
    pred = np.asarray(pred, dtype=float).flatten()
    gt = np.asarray(gt, dtype=float).flatten()
    common = min(len(pred), len(gt))
    if common < 4:
        return 0
    pred_z = _zscore(pred[:common])
    gt_z = _zscore(gt[:common])
    if not (np.isfinite(pred_z).any() and np.isfinite(gt_z).any()):
        return 0
    corr = np.correlate(pred_z, gt_z, mode="full")
    lags = np.arange(-common + 1, common)
    lag = int(lags[int(np.nanargmax(corr))])
    # Keep the visual shift bounded so the same-trial atlas does not collapse
    # into unrelated windows under hard-regime noisy observations.
    max_lag_samples = max(1, int(max_lag_samples))
    return int(np.clip(lag, -max_lag_samples, max_lag_samples))


def _best_reference_window(gt: np.ndarray, fs_gt: float, window_sec: float) -> tuple[int, int]:
    """Select one reference-visible window per dataset for all panels.

    This is a visual readability rule, not a performance selection rule: it uses
    only reference variation to avoid plotting a nearly flat or transition-only
    segment in the same-trial atlas.
    """
    gt = np.asarray(gt, dtype=float).flatten()
    n = len(gt)
    if n == 0:
        return 0, 0
    win = min(n, max(120, int(round(window_sec * fs_gt))))
    if n <= win:
        return 0, n
    step = max(1, win // 12)
    best_start = 0
    best_score = -np.inf
    for start in range(0, n - win + 1, step):
        seg = gt[start : start + win]
        seg_z = _zscore(seg)
        # Prefer windows with sustained reference dynamics over isolated jumps.
        dyn = float(np.nanpercentile(seg_z, 90) - np.nanpercentile(seg_z, 10))
        rough = float(np.nanmedian(np.abs(np.diff(seg_z)))) if len(seg_z) > 1 else 0.0
        score = dyn - 0.25 * rough
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + win


def _take_with_padding(sig: np.ndarray, start: int, end: int) -> np.ndarray:
    sig = np.asarray(sig, dtype=float).flatten()
    if end <= start:
        return np.array([], dtype=float)
    out = np.full(end - start, np.nan, dtype=float)
    src_start = max(start, 0)
    src_end = min(end, len(sig))
    if src_end > src_start:
        dst_start = src_start - start
        out[dst_start : dst_start + (src_end - src_start)] = sig[src_start:src_end]
    return out


def _load_estimate(payload: dict, method: str) -> dict:
    for item in payload.get("estimates", []):
        if item.get("method") == method:
            return item.get("estimate", {})
    available = ", ".join(str(item.get("method")) for item in payload.get("estimates", []))
    raise KeyError(f"Method '{method}' not found. Available: {available}")


def _series(run_dir: Path, video: str, method: str, output_type: str) -> tuple[np.ndarray, np.ndarray, float, dict]:
    pkl_path = run_dir / "data" / f"{video}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    payload = pickle.load(open(pkl_path, "rb"))
    est = _load_estimate(payload, method)
    if output_type not in est:
        output_type = "z_full" if "z_full" in est else "signal_hat"
    pred = np.asarray(est[output_type], dtype=float).flatten()
    gt = np.asarray(payload["gt"], dtype=float).flatten()
    fs_est = float(payload.get("fps", 20.0))
    fs_gt = float(payload.get("fs_gt", fs_est))
    pred_resampled = _resample_to_gt(pred, fs_est=fs_est, fs_gt=fs_gt)
    common = min(len(pred_resampled), len(gt))
    pred_resampled = np.asarray(pred_resampled[:common], dtype=float)
    gt = np.asarray(gt[:common], dtype=float)
    meta = {
        "pkl_path": str(pkl_path),
        "fs_gt": fs_gt,
        "waveform_CCC": waveform_ccc(pred_resampled, gt),
    }
    return pred_resampled, gt, fs_gt, meta


def _visual_window(
    pred: np.ndarray,
    gt: np.ndarray,
    fs_gt: float,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float, float]:
    lag = _alignment_lag_samples(pred, gt, max_lag_samples=round(5.0 * fs_gt))
    pred_w = _take_with_padding(pred, start + lag, end + lag)
    gt_w = _take_with_padding(gt, start, end)
    pred_z = _zscore(pred_w)
    gt_z = _zscore(gt_w)
    finite = np.isfinite(pred_z) & np.isfinite(gt_z)
    if int(np.count_nonzero(finite)) > 3:
        corr_z = float(np.corrcoef(pred_z[finite], gt_z[finite])[0, 1])
    else:
        corr_z = float("nan")
    mae_z = waveform_mae(pred_z[finite], gt_z[finite]) if np.count_nonzero(finite) else float("nan")
    dtw_z = waveform_dtw(pred_z[finite], gt_z[finite]) if np.count_nonzero(finite) else float("nan")
    t = np.arange(end - start, dtype=float) / fs_gt
    return t, pred_z, gt_z, lag, corr_z, mae_z, dtw_z


def plot_atlas(out: Path, manifest_out: Path, window_sec: float = 30.0) -> None:
    set_manuscript_style("paper")
    datasets = _default_datasets()
    ncols = 5
    nrows = len(datasets) * 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(18.4, 12.2),
        sharex=False,
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.992,
        bottom=0.070,
        top=0.855,
        wspace=0.145,
        hspace=0.500,
    )
    manifest_rows: list[dict[str, object]] = []
    for d_idx, ds in enumerate(datasets):
        ref_pred, ref_gt, ref_fs, _ = _series(
            ds.final_run_dir,
            ds.video,
            "parh_ossm",
            "z_full",
        )
        ref_start, ref_end = _best_reference_window(ref_gt, ref_fs, window_sec)
        for m_idx, spec in enumerate(METHODS):
            row = d_idx * 2 + (m_idx // ncols)
            col = m_idx % ncols
            ax = axes[row, col]
            style_axis(ax, grid="y")
            run_dir = ds.final_run_dir if spec.source == "final" else ds.base_run_dir
            pred, gt, fs_gt, meta = _series(run_dir, ds.video, spec.method, spec.output_type)
            tw, pred_w, gt_w, lag, corr_z, mae_z, dtw_z = _visual_window(
                pred,
                gt,
                fs_gt,
                ref_start,
                ref_end,
            )
            ax.plot(tw, pred_w, color=spec.color, linewidth=1.15, alpha=0.86, label=spec.label, zorder=2)
            ax.plot(tw, gt_w, color="#111111", linewidth=1.85, alpha=0.96, label="reference (GT)", zorder=4)
            ax.set_ylim(-3.2, 3.2)
            ax.set_title(f"{spec.label}\nvisual corr {corr_z:.2f}", fontsize=8.2, loc="center", pad=5.0)
            if col == 0:
                second = "fixed Base observation" if m_idx < ncols else "Base/comparator/PARH"
                ax.set_ylabel(f"{ds.dataset}\n{ds.video}\n{second}", fontsize=8.5)
            if row == nrows - 1:
                ax.set_xlabel("Time (s)", fontsize=8.5)
            manifest_rows.append(
                {
                    "dataset": ds.dataset,
                    "video": ds.video,
                    "panel_row": row,
                    "panel_col": col,
                    "label": spec.label.replace("\n", " "),
                    "method": spec.method,
                    "output_type": spec.output_type,
                    "source": spec.source,
                    "run_dir": str(run_dir),
                    "data_file": meta["pkl_path"],
                    "waveform_CCC": meta["waveform_CCC"],
                    "visual_corr_z": corr_z,
                    "visual_lag_samples": lag,
                    "visual_window_start_s": ref_start / fs_gt,
                    "visual_window_end_s": ref_end / fs_gt,
                    "waveform_MAE_z": mae_z,
                    "waveform_DTW_z": dtw_z,
                }
            )

    handles = [
        Line2D([0], [0], color="#111111", linewidth=1.9, label="reference (GT)"),
        Line2D([0], [0], color="#24476f", linewidth=1.6, label="fixed Base observations"),
        Line2D([0], [0], color="#c06c2b", linewidth=1.6, label="OSSM-KF comparator"),
        Line2D([0], [0], color="#116b4f", linewidth=1.6, label="PARH-OSSM"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9.4,
        bbox_to_anchor=(0.5, 0.938),
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.suptitle(
        "Same-trial waveform topology from full-dataset manifest cases",
        fontsize=12.5,
        y=0.985,
    )
    save_figure(fig, out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_rows).to_csv(manifest_out, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot main all-base same-trial waveform overlay atlas.")
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "figures" / "F4_waveform_overlay_grid.pdf")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=ROOT / "paper" / "manifests" / "f4_allbase_same_trial_overlay_manifest.csv",
    )
    parser.add_argument("--window-sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot_atlas(out=args.out, manifest_out=args.manifest_out, window_sec=float(args.window_sec))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
