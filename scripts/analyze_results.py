#!/usr/bin/env python3
"""
Compare two methods on a chosen metric from raw metrics CSV.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/cohface_robust_ossm")
    ap.add_argument("--domain", choices=["freq", "time"], default="freq")
    ap.add_argument("--method-a", required=True)
    ap.add_argument("--method-b", required=True)
    ap.add_argument("--metric", default="MAE")
    ap.add_argument("--threshold", type=float, default=1.0)
    args = ap.parse_args()

    csv_name = "metrics_freq_domain_raw.csv" if args.domain == "freq" else "metrics_time_domain_raw.csv"
    csv_path = os.path.join(args.run_dir, "metrics", csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not in {csv_name}. Columns: {list(df.columns)}")

    a = df[df["method"] == args.method_a]
    b = df[df["method"] == args.method_b]
    merged = pd.merge(a, b, on="video", suffixes=("_a", "_b"))
    if merged.empty:
        print("No overlapping videos between methods.")
        return

    diff_col = f"{args.metric}_diff"
    merged[diff_col] = merged[f"{args.metric}_a"] - merged[f"{args.metric}_b"]

    print(f"[Domain] {args.domain} | [Metric] {args.metric}")
    print(f"[A] {args.method_a}")
    print(f"[B] {args.method_b}")
    print("\n=== Difference Summary (A - B) ===")
    print(merged[diff_col].describe())

    thr = float(args.threshold)
    worse = merged[merged[diff_col] > thr].sort_values(diff_col, ascending=False)
    better = merged[merged[diff_col] < -thr].sort_values(diff_col, ascending=True)

    print(f"\n=== A significantly worse than B (>{thr}) ===")
    cols = ["video", f"{args.metric}_a", f"{args.metric}_b", diff_col]
    print(worse[cols].head(15).to_string(index=False))

    print(f"\n=== A significantly better than B (<-{thr}) ===")
    print(better[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()

