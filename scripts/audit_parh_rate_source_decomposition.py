#!/usr/bin/env python3
"""Audit where PARH-OSSM's reported rate comes from.

This script is intentionally diagnostic-only. It evaluates native oscillator
tracks, external target-computable readouts, posterior tracks, and the final
reported track with the same windowed-rate protocol used by the main evaluator.
The goal is to prevent hidden readout override from being mistaken for native
state-space improvement.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.evaluation.metrics import getErrors
from core.pipeline.evaluation_step import _instantaneous_freq_hz, _windowed_track_bpm
from core.utils.common import filter_RW


def _load_payloads(data_dir: Path, max_files: int | None) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    files = sorted(data_dir.glob("*.pkl"))
    if max_files is not None and max_files > 0:
        files = files[: int(max_files)]
    for path in files:
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            print(f"[warn] failed to load {path}: {exc}")
            continue
        if isinstance(payload, dict):
            yield path, payload


def _estimate_rows(payload: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    estimates = payload.get("estimates", [])
    if not isinstance(estimates, list):
        return
    for row in estimates:
        if not isinstance(row, dict):
            continue
        method = str(row.get("method", row.get("name", "unknown")))
        est = row.get("estimate", row)
        if isinstance(est, dict):
            yield method, est


def _arr(value: Any) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return np.asarray([], dtype=np.float64)


def _track_sources(est: Dict[str, Any]) -> Dict[str, np.ndarray]:
    diag = est.get("diagnostics", {})
    if not isinstance(diag, dict):
        diag = {}
    sources: Dict[str, np.ndarray] = {
        "final_track_hz": _arr(est.get("track_hz")),
        "native_smoothed_track_hz": _arr(est.get("track_hz_native_smoothed")),
        "native_causal_track_hz": _arr(est.get("track_hz_causal")),
        "external_output_rate_t": _arr(diag.get("external_output_rate_t")),
        "external_rate_posterior_mode_t": _arr(diag.get("external_rate_posterior_mode_t")),
        "external_rate_posterior_mean_t": _arr(diag.get("external_rate_posterior_mean_t")),
        "state_freq_t": _arr(diag.get("freq_t")),
    }
    return {key: val for key, val in sources.items() if val.size > 0 and np.any(np.isfinite(val))}


def _mean_valid(arr: np.ndarray, mask: np.ndarray | None = None) -> float:
    vec = _arr(arr)
    if mask is not None and mask.size == vec.size:
        vec = vec[mask]
    vec = vec[np.isfinite(vec)]
    return float(np.mean(vec)) if vec.size else float("nan")


def _evaluate_track(
    track_hz: np.ndarray,
    gt_inst_hz: np.ndarray,
    *,
    fps: float,
    fs_gt: float,
    win_size: float,
    stride: float,
) -> Dict[str, float]:
    est_bpm, gt_bpm = _windowed_track_bpm(
        np.asarray(track_hz, dtype=np.float64),
        np.asarray(gt_inst_hz, dtype=np.float64),
        fps_est=float(fps),
        fs_gt=float(fs_gt),
        win_size_sec=float(win_size),
        stride_sec=float(stride),
    )
    n = min(est_bpm.size, gt_bpm.size)
    if n <= 5:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "MAPE": float("nan"),
            "PearsonR": float("nan"),
            "n_windows": int(n),
            "est_bpm_avg": float("nan"),
            "gt_bpm_avg": float("nan"),
        }
    est_bpm = est_bpm[:n]
    gt_bpm = gt_bpm[:n]
    errs = getErrors(est_bpm, gt_bpm, None, None, ["MAE", "RMSE", "MAPE", "PearsonR"])
    return {
        "MAE": float(errs[0]),
        "RMSE": float(errs[1]),
        "MAPE": float(errs[2]),
        "PearsonR": float(errs[3]),
        "n_windows": int(n),
        "est_bpm_avg": float(np.nanmean(est_bpm)) if est_bpm.size else float("nan"),
        "gt_bpm_avg": float(np.nanmean(gt_bpm)) if gt_bpm.size else float("nan"),
    }


def _render_markdown(rows: pd.DataFrame, summary: pd.DataFrame, *, run_data: Path) -> str:
    def _fmt_table(df: pd.DataFrame, max_rows: int = 40) -> str:
        if df.empty:
            return "_No rows._"
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.head(max_rows).iterrows():
            vals = []
            for col in cols:
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    vals.append("nan" if not np.isfinite(value) else f"{float(value):.3f}")
                else:
                    vals.append(str(value))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# PARH Rate-Source Decomposition",
        "",
        f"- data: `{run_data}`",
        "- purpose: separate native PARH oscillator rate from external target-computable readout and final reported rate.",
        "- interpretation: if `final_track_hz` is much better than `native_smoothed_track_hz`, the current performance is readout-carried rather than state-carried.",
        "",
        "## Source Summary",
        "",
        _fmt_table(summary),
        "",
        "## Per-Trial Rows",
        "",
        _fmt_table(rows.sort_values(["video", "method", "rate_source"]), max_rows=80),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="Run data directory containing *.pkl files.")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--win-size", type=float, default=30.0)
    parser.add_argument("--stride", type=float, default=1.0)
    parser.add_argument("--min-hz", type=float, default=0.08)
    parser.add_argument("--max-hz", type=float, default=0.50)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        raise SystemExit(f"data-dir does not exist: {data_dir}")

    rows: List[Dict[str, Any]] = []
    for path, payload in _load_payloads(data_dir, args.max_files if args.max_files > 0 else None):
        gt = _arr(payload.get("gt"))
        if gt.size < 10:
            continue
        fps = float(payload.get("fps") or payload.get("fs") or 30.0)
        fs_gt = float(payload.get("fs_gt") or fps)
        gt_filt = filter_RW(gt, fs_gt, lo=float(args.min_hz), hi=float(args.max_hz))
        gt_inst_hz = _instantaneous_freq_hz(gt_filt, fs_gt, float(args.min_hz), float(args.max_hz))
        for method, est in _estimate_rows(payload):
            diag = est.get("diagnostics", {})
            if not isinstance(diag, dict):
                diag = {}
            output_blend = _arr(diag.get("external_output_rate_blend_t"))
            posterior_blend = _arr(diag.get("external_rate_posterior_blend_t"))
            for source, track in _track_sources(est).items():
                metrics = _evaluate_track(
                    track,
                    gt_inst_hz,
                    fps=fps,
                    fs_gt=fs_gt,
                    win_size=float(args.win_size),
                    stride=float(args.stride),
                )
                finite = np.isfinite(track)
                rows.append({
                    "video": path.stem,
                    "data_file": path.name,
                    "method": method,
                    "rate_source": source,
                    **metrics,
                    "track_hz_median": float(np.nanmedian(track)) if np.any(finite) else float("nan"),
                    "track_hz_std": float(np.nanstd(track)) if np.any(finite) else float("nan"),
                    "external_output_blend_mean": _mean_valid(output_blend),
                    "external_posterior_blend_mean": _mean_valid(posterior_blend),
                })

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("no evaluable rate-source rows")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    summary = (
        frame.groupby(["method", "rate_source"], dropna=False)
        .agg(
            n_trials=("video", "nunique"),
            MAE_median=("MAE", "median"),
            RMSE_median=("RMSE", "median"),
            PearsonR_median=("PearsonR", "median"),
            track_hz_median=("track_hz_median", "median"),
            external_output_blend_mean=("external_output_blend_mean", "median"),
            external_posterior_blend_mean=("external_posterior_blend_mean", "median"),
        )
        .reset_index()
        .sort_values(["MAE_median", "rate_source"], na_position="last")
    )
    summary_csv = args.out_csv.with_name(args.out_csv.stem + "_summary.csv")
    summary.to_csv(summary_csv, index=False)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_markdown(frame, summary, run_data=data_dir), encoding="utf-8")
    print(f"wrote {args.out_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {args.out_md}")
    print(json.dumps(summary.head(12).to_dict(orient="records"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
