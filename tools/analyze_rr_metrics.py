#!/usr/bin/env python3
"""
Quick analysis helper for respiration metrics.

This script scans metrics.pkl (track domain) and produces two CSV files:
  1) summary.csv -> per-method medians for R, MAE, SNR, len_valid, etc.
  2) records.csv -> per-trial entries to build scatter plots (e.g., R vs SNR).

Usage:
    python tools/analyze_rr_metrics.py --results results/cohface_motion_oscillator
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RR metrics for tuning guidance.")
    parser.add_argument(
        "--results",
        default="results/cohface_motion_oscillator",
        help="Run directory containing metrics/metrics.pkl (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        help="Optional output directory (default: <results>/analysis)",
    )
    return parser.parse_args()


def load_metrics(metrics_path: Path) -> Tuple[List[str], Dict[str, List[Dict]]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file missing: {metrics_path}")
    with metrics_path.open("rb") as fp:
        names, payload = pickle.load(fp)
    if not isinstance(payload, dict):
        raise ValueError("metrics payload is malformed")
    return names, payload


def extract_values(
    metric_names: List[str], payload: Dict[str, List[Dict]]
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    idx_map = {name: i for i, name in enumerate(metric_names)}
    detailed_rows: List[Dict] = []
    by_method: Dict[str, List[Dict]] = {}
    for method, records in payload.items():
        for rec in records or []:
            metrics_arr = rec.get("metrics") or []
            row = {
                "method": method,
                "trial": rec.get("trial_key") or "",
                "data_file": rec.get("data_file") or "",
                "len_valid": rec.get("len_valid", 0),
                "len_final": rec.get("len_final", 0),
                "len_pred": rec.get("len_pred", 0),
                "degenerate_reason": rec.get("degenerate_reason") or "",
            }
            for key in ("RMSE", "MAE", "MAPE", "R", "SNR"):
                idx = idx_map.get(key)
                if idx is not None and idx < len(metrics_arr):
                    try:
                        row[key] = float(metrics_arr[idx])
                    except (TypeError, ValueError):
                        row[key] = float("nan")
                else:
                    row[key] = float("nan")
            quality = rec.get("quality") or {}
            row["quality_snr"] = float(quality.get("snr_estimate")) if quality.get("snr_estimate") is not None else float("nan")
            row["reliability"] = float(quality.get("reliability_score")) if quality.get("reliability_score") is not None else float("nan")
            detailed_rows.append(row)
            by_method.setdefault(method, []).append(row)
    return detailed_rows, by_method


def summarise(by_method: Dict[str, List[Dict]]) -> List[Dict]:
    summary: List[Dict] = []
    for method, rows in by_method.items():
        def median(key: str) -> float:
            vals = np.asarray([r.get(key, float("nan")) for r in rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            return float(np.nanmedian(vals)) if vals.size else float("nan")

        entry = {
            "method": method,
            "count": len(rows),
            "median_R": median("R"),
            "median_MAE": median("MAE"),
            "median_SNR": median("SNR"),
            "median_quality_SNR": median("quality_snr"),
            "median_len_valid": median("len_valid"),
        }
        summary.append(entry)
    summary.sort(key=lambda item: (item["median_MAE"] if np.isfinite(item["median_MAE"]) else np.inf))
    return summary


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    results_dir = Path(args.results).resolve()
    metrics_dir = results_dir / "metrics"
    metrics_path = metrics_dir / "metrics.pkl"
    metric_names, payload = load_metrics(metrics_path)
    detailed_rows, by_method = extract_values(metric_names, payload)
    summary_rows = summarise(by_method)
    out_dir = Path(args.output).resolve() if args.output else (results_dir / "analysis")
    write_csv(out_dir / "rr_metrics_records.csv", detailed_rows)
    write_csv(out_dir / "rr_metrics_summary.csv", summary_rows)
    print(f"Saved summary to {out_dir / 'rr_metrics_summary.csv'}")
    print(f"Saved detailed records to {out_dir / 'rr_metrics_records.csv'}")
    top5 = summary_rows[:5]
    print("\nTop methods by median MAE:")
    for row in top5:
        print(
            f"  {row['method']}: MAE={row['median_MAE']:.3f}, "
            f"R={row['median_R']:.3f}, SNR={row['median_SNR']:.3f}, "
            f"len_valid={row['median_len_valid']:.1f} (n={row['count']})"
        )


if __name__ == "__main__":
    main()
