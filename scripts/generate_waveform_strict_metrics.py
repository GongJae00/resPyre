#!/usr/bin/env python3
"""Generate strict waveform reconstruction metrics directly from saved trial PKLs.

This bypasses a full `main.py` reevaluation and computes the supplementary
strict waveform / cycle metrics from `data/*.pkl` only.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.metrics import waveform_ccc, waveform_mae, waveform_dtw
from core.pipeline.evaluation_step import (
    _cycle_level_metrics,
    _format_scalar,
    _render_table,
    _strict_scale_span,
    _strict_waveform_pair,
)
from core.utils.common import filter_RW, tqdm


STRICT_METRICS = [
    "strict_CCC",
    "strict_MAE",
    "strict_RMSE",
    "strict_DTW",
    "gt_span_p95p05",
    "strict_NMAE_span",
    "strict_NRMSE_span",
    "strict_NDTW_span",
    "peak_time_mae_s",
    "trough_time_mae_s",
    "cycle_ppi_mae_s",
    "cycle_ie_abs_err",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strict waveform metrics from saved PKL trial bundles.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Run data directory containing *.pkl trial bundles.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Metrics output directory. Defaults to sibling 'metrics' next to data-dir.",
    )
    parser.add_argument("--f-min", type=float, default=0.08)
    parser.add_argument("--f-max", type=float, default=0.50)
    return parser.parse_args()


def _method_sort_key(name: str):
    m = str(name)
    mm = m.lower()
    if "__parh_ossm" in mm:
        block = 2
    elif "__kfstd" in mm:
        block = 1
    else:
        block = 0
    return (mm.split("__")[0], block, mm)


def _save_outputs(records, metrics_dir: Path) -> None:
    df = pd.DataFrame(records)
    raw_csv = metrics_dir / "metrics_waveform_strict_raw.csv"
    summary_csv = metrics_dir / "metrics_waveform_strict_summary.csv"
    summary_txt = metrics_dir / "metrics_waveform_strict_summary.txt"
    summary_pkl = metrics_dir / "metrics_waveform_strict.pkl"

    df.to_csv(raw_csv, index=False)

    if df.empty:
        pd.DataFrame(columns=["method"]).to_csv(summary_csv, index=False)
        summary_txt.write_text("# Strict Waveform / Cycle Metrics\n(no records)\n", encoding="utf-8")
        with open(summary_pkl, "wb") as f:
            pickle.dump([STRICT_METRICS, {}], f)
        return

    summary_df = df.copy()
    for m in STRICT_METRICS:
        if m in summary_df.columns:
            summary_df[m] = pd.to_numeric(summary_df[m], errors="coerce")

    summary_stats = summary_df.groupby("method")[STRICT_METRICS].agg(["median", "std"])
    summary_stats.columns = [f"{m}_{s}" for m, s in summary_stats.columns]
    summary_stats = summary_stats.reset_index()
    summary_stats["Method"] = summary_stats["method"]
    summary_stats["sort_key"] = summary_stats["Method"].apply(_method_sort_key)
    summary_stats = summary_stats.sort_values("sort_key").drop(columns=["sort_key"])
    summary_stats.to_csv(summary_csv, index=False)

    table_headers = ["Method"] + [f"{m} (median±std)" for m in STRICT_METRICS]
    table_rows = []
    for _, row in summary_stats.iterrows():
        table_rows.append(
            [row["Method"]]
            + [f"{_format_scalar(row[f'{m}_median'])} (±{_format_scalar(row[f'{m}_std'], decimals=2)})" for m in STRICT_METRICS]
        )
    table_str = _render_table(table_headers, table_rows)
    summary_txt.write_text(
        "# Strict Waveform / Cycle Metrics (zero-lag, unit-preserving)\n" + table_str + "\n",
        encoding="utf-8",
    )

    method_metrics = {}
    for rec in records:
        method_metrics.setdefault(rec["method"], []).append(
            {
                "video": rec["video"],
                "metrics": [rec[k] for k in STRICT_METRICS],
                "source_label": "waveform_strict",
                "data_file": rec.get("data_file"),
                "pair": None,
            }
        )
    with open(summary_pkl, "wb") as f:
        pickle.dump([STRICT_METRICS, method_metrics], f)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    metrics_dir = (args.out_dir or data_dir.parent / "metrics").resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(data_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No PKL files found under {data_dir}")

    all_records = []
    for pkl_path in tqdm(pkl_files, desc="Strict waveform metrics"):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        gt_signal = data.get("gt")
        fps = float(data.get("fps", 30.0))
        fs_gt = float(data.get("fs_gt", fps))
        fname = pkl_path.stem
        data_file_rel = os.path.relpath(pkl_path, data_dir)

        if gt_signal is None or not isinstance(gt_signal, np.ndarray):
            continue
        gt_filt = filter_RW(gt_signal, fs_gt, lo=args.f_min, hi=args.f_max)

        for est in data.get("estimates", []):
            method_name = est.get("method", est.get("name", "unknown"))
            payload = est.get("estimate", est)
            if not isinstance(payload, dict):
                continue

            strict_candidates = {}

            sig_hat = payload.get("signal_hat")
            if sig_hat is not None:
                sig_hat = np.asarray(sig_hat, dtype=np.float64).flatten()
                if sig_hat.size >= 10:
                    strict_candidates[("signal_hat", "smoothed")] = filter_RW(sig_hat, fps, lo=args.f_min, hi=args.f_max)

            zf_smooth = payload.get("z_full_smoothed") if payload.get("z_full_smoothed") is not None else payload.get("z_full")
            zf_causal = payload.get("z_full_causal")

            if zf_smooth is not None:
                zf_smooth = np.asarray(zf_smooth, dtype=np.float64).flatten()
                if zf_smooth.size >= 10:
                    strict_candidates[("z_full", "smoothed")] = filter_RW(zf_smooth, fps, lo=args.f_min, hi=args.f_max)
            if zf_causal is not None:
                zf_causal = np.asarray(zf_causal, dtype=np.float64).flatten()
                if zf_causal.size >= 10:
                    strict_candidates[("z_full", "causal")] = filter_RW(zf_causal, fps, lo=args.f_min, hi=args.f_max)

            for (output_type, variant_label), wf_sig_raw in strict_candidates.items():
                wf_strict, gt_strict = _strict_waveform_pair(wf_sig_raw, gt_filt, fps, fs_gt)
                if len(wf_strict) <= 10:
                    continue
                cycle = _cycle_level_metrics(wf_strict, gt_strict, fps)
                strict_mae = waveform_mae(wf_strict, gt_strict)
                strict_rmse = float(np.sqrt(np.mean((wf_strict - gt_strict) ** 2)))
                strict_dtw = waveform_dtw(wf_strict, gt_strict)
                strict_span = _strict_scale_span(gt_strict)
                if np.isfinite(strict_span):
                    strict_nmae = strict_mae / strict_span
                    strict_nrmse = strict_rmse / strict_span
                    strict_ndtw = strict_dtw / strict_span
                else:
                    strict_nmae = np.nan
                    strict_nrmse = np.nan
                    strict_ndtw = np.nan
                all_records.append(
                    {
                        "video": fname,
                        "method": method_name,
                        "output_type": output_type,
                        "causal_or_smoothed": variant_label,
                        "strict_CCC": waveform_ccc(wf_strict, gt_strict),
                        "strict_MAE": strict_mae,
                        "strict_RMSE": strict_rmse,
                        "strict_DTW": strict_dtw,
                        "gt_span_p95p05": strict_span,
                        "strict_NMAE_span": strict_nmae,
                        "strict_NRMSE_span": strict_nrmse,
                        "strict_NDTW_span": strict_ndtw,
                        "peak_time_mae_s": cycle["peak_time_mae_s"],
                        "trough_time_mae_s": cycle["trough_time_mae_s"],
                        "cycle_ppi_mae_s": cycle["cycle_ppi_mae_s"],
                        "cycle_ie_abs_err": cycle["cycle_ie_abs_err"],
                        "n_peaks_est": cycle["n_peaks_est"],
                        "n_peaks_gt": cycle["n_peaks_gt"],
                        "n_troughs_est": cycle["n_troughs_est"],
                        "n_troughs_gt": cycle["n_troughs_gt"],
                        "data_file": data_file_rel,
                    }
                )

    _save_outputs(all_records, metrics_dir)
    print(f"Saved strict waveform metrics to {metrics_dir}")


if __name__ == "__main__":
    main()
