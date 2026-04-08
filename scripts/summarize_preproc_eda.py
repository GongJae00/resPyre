#!/usr/bin/env python3
"""Summarize observation/preprocessing EDA outputs into stage-gain tables."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "corr_wave_best",
    "ccc_wave_best_z",
    "corr_deriv_best",
    "peak_error_hz",
    "lowfreq_energy_ratio",
    "highfreq_energy_ratio",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize observation/preprocessing EDA stage gains.")
    parser.add_argument(
        "--trial-csv",
        type=Path,
        required=True,
        help="Per-trial CSV created by run_observation_eda.py",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("analysis/preproc_eda_stage_summary.csv"),
        help="Output CSV for stage medians/means",
    )
    parser.add_argument(
        "--delta-out",
        type=Path,
        default=Path("analysis/preproc_eda_stage_deltas.csv"),
        help="Output CSV for deltas versus raw",
    )
    return parser.parse_args()


def agg_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_cols = [
        c for c in df.columns
        if c not in {"dataset", "video", "subject", "trial", "family", "stage", "video_path", "cache_path"}
    ]
    for (dataset, family, stage), g in df.groupby(["dataset", "family", "stage"], dropna=False):
        row = {"dataset": dataset, "family": family, "stage": stage, "trial_count": int(g["video"].nunique())}
        for col in numeric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if not vals.empty else np.nan
            row[f"{col}_median"] = float(vals.median()) if not vals.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "family", "stage"]).reset_index(drop=True)


def delta_vs_raw(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in DEFAULT_METRICS if c in df.columns]
    rows = []
    for (dataset, family), g in df.groupby(["dataset", "family"], dropna=False):
        raw = g[g["stage"] == "raw"][["video"] + metric_cols].copy()
        raw = raw.rename(columns={col: f"{col}__raw" for col in metric_cols})
        for stage, gs in g.groupby("stage", dropna=False):
            if stage == "raw":
                continue
            merged = gs[["video"] + metric_cols].merge(raw, on="video", how="inner")
            row = {"dataset": dataset, "family": family, "stage": stage, "trial_count": int(merged["video"].nunique())}
            for col in metric_cols:
                curr = pd.to_numeric(merged[col], errors="coerce")
                base = pd.to_numeric(merged[f"{col}__raw"], errors="coerce")
                if col == "peak_error_hz":
                    delta = np.abs(base) - np.abs(curr)
                elif col in {"lowfreq_energy_ratio", "highfreq_energy_ratio"}:
                    delta = base - curr
                else:
                    delta = curr - base
                delta = pd.Series(delta).replace([np.inf, -np.inf], np.nan).dropna()
                row[f"{col}_delta_mean"] = float(delta.mean()) if not delta.empty else np.nan
                row[f"{col}_delta_median"] = float(delta.median()) if not delta.empty else np.nan
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "family", "stage"]).reset_index(drop=True)


def main():
    args = parse_args()
    df = pd.read_csv(args.trial_csv)
    summary_df = agg_summary(df)
    delta_df = delta_vs_raw(df)

    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.delta_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_out, index=False)
    delta_df.to_csv(args.delta_out, index=False)

    print(f"Saved summary: {args.summary_out}")
    print(f"Saved deltas: {args.delta_out}")
    if not delta_df.empty:
        print(delta_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
