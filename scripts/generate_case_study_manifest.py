#!/usr/bin/env python3
"""Generate a persistent case-study manifest for overlay figures.

The goal is to prevent cherry-picking. Cases are selected deterministically
from persistent metrics CSVs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def classify_method(method_name: str):
    m = str(method_name).lower().replace(" ", "_")
    if "__parh_ossm" in m:
        variant = "PARH"
        family = m.split("__parh_ossm")[0]
    elif "__kfstd" in m:
        variant = "KFstd"
        family = m.split("__kfstd")[0]
    else:
        variant = "Base"
        family = m
    family_map = {
        "of_farneback": "OF",
        "of_disp_bridge": "OF_bridge",
        "of": "OF",
        "dof": "DoF",
        "profile1d_linear": "P1D_lin",
        "profile1d_quadratic": "P1D_quad",
        "profile1d_cubic": "P1D_cub",
    }
    return family_map.get(family, family), variant


def preferred_output_type(variant: str) -> str:
    return "z_full" if variant == "PARH" else "signal_hat"


def metric_direction(metric_name: str) -> bool:
    """Return True when larger is better."""
    lower_better = {"waveform_mae", "waveform_dtw", "mae", "rmse", "mape"}
    return metric_name.lower() not in lower_better


def select_case_rows(group: pd.DataFrame, metric_col: str, higher_better: bool):
    g = group.copy()
    g[metric_col] = pd.to_numeric(g[metric_col], errors="coerce")
    g = g[np.isfinite(g[metric_col].to_numpy())]
    if g.empty:
        return pd.DataFrame()

    g = g.sort_values(metric_col, ascending=not higher_better).reset_index(drop=True)
    indices = {
        "top": 0,
        "median": len(g) // 2,
        "bottom": len(g) - 1,
    }
    rows = []
    for rank, idx in indices.items():
        row = g.iloc[int(np.clip(idx, 0, len(g) - 1))].copy()
        row["case_rank"] = rank
        row["metric_name"] = metric_col
        row["metric_value"] = row[metric_col]
        rows.append(row)
    return pd.DataFrame(rows)


def build_manifest(waveform_csv: Path, freq_csv: Path, dataset_name: str, metric_col: str):
    wave = pd.read_csv(waveform_csv)
    freq = pd.read_csv(freq_csv) if freq_csv.exists() else pd.DataFrame()

    wave[["family", "variant"]] = wave["method"].apply(lambda m: pd.Series(classify_method(m)))
    wave = wave[wave["causal_or_smoothed"] == "smoothed"].copy()
    wave["preferred_output_type"] = wave["variant"].map(preferred_output_type)
    wave = wave[wave["output_type"] == wave["preferred_output_type"]].copy()

    if not freq.empty:
        freq[["family", "variant"]] = freq["method"].apply(lambda m: pd.Series(classify_method(m)))
        freq_cols = ["video", "method", "data_file", "MAE", "RMSE", "PearsonR", "rate_source"]
        freq_keep = [c for c in freq_cols if c in freq.columns]
        freq = freq[freq_keep].copy()
        freq = freq.rename(
            columns={
                "MAE": "rate_MAE",
                "RMSE": "rate_RMSE",
                "PearsonR": "rate_PearsonR",
            }
        )
        merged = wave.merge(freq, on=["video", "method", "data_file"], how="left")
    else:
        merged = wave.copy()

    higher_better = metric_direction(metric_col)
    rows = []
    for (family, variant), g in merged.groupby(["family", "variant"], dropna=False):
        selected = select_case_rows(g, metric_col, higher_better=higher_better)
        if selected.empty:
            continue
        selected["dataset"] = dataset_name
        selected["panel_id"] = selected.apply(
            lambda r: f"{dataset_name}_{family}_{variant}_{r['case_rank']}", axis=1
        )
        rows.append(selected)

    if not rows:
        return pd.DataFrame()

    manifest = pd.concat(rows, ignore_index=True)
    ordered_cols = [
        "dataset", "family", "variant", "case_rank", "panel_id",
        "video", "method", "data_file", "output_type", "causal_or_smoothed",
        "metric_name", "metric_value", "waveform_CCC", "waveform_MAE",
        "waveform_DTW", "latency_ms", "rate_MAE", "rate_RMSE", "rate_PearsonR",
        "rate_source",
    ]
    existing = [c for c in ordered_cols if c in manifest.columns]
    remaining = [c for c in manifest.columns if c not in existing]
    return manifest[existing + remaining].sort_values(["family", "variant", "case_rank"]).reset_index(drop=True)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate top/median/bottom case-study manifest from metrics CSVs.")
    parser.add_argument(
        "--waveform-csv",
        type=Path,
        required=True,
        help="Path to metrics_waveform_raw.csv",
    )
    parser.add_argument(
        "--freq-csv",
        type=Path,
        default=root / "results" / "cohface_parh_ossm_prod" / "cohface_parh_ossm_prod" / "metrics" / "metrics_freq_domain_raw.csv",
        help="Path to metrics_freq_domain_raw.csv",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="COHFACE",
        help="Dataset label stored in the manifest",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="waveform_CCC",
        help="Primary metric used for ranking cases",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "paper" / "manifests" / "case_study_manifest.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        waveform_csv=args.waveform_csv,
        freq_csv=args.freq_csv,
        dataset_name=args.dataset_name,
        metric_col=args.metric,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(f"Saved manifest: {args.out}")
    print(f"Rows: {len(manifest)}")
    if not manifest.empty:
        print(manifest.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
