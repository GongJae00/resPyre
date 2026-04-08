#!/usr/bin/env python3
"""Generate same-trial overlay manifests for Base/KFstd/PARH waveform figures."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FAMILY_TO_METHODS = {
    "OF": {
        "Base": "of_farneback",
        "KFstd": "of_farneback__kfstd",
        "PARH": "of_farneback__parh_ossm",
    },
    "OF_bridge": {
        "Base": "of_disp_bridge",
        "KFstd": "of_disp_bridge__kfstd",
        "PARH": "of_disp_bridge__parh_ossm",
    },
    "DoF": {
        "Base": "DoF",
        "KFstd": "dof__kfstd",
        "PARH": "dof__parh_ossm",
    },
    "P1D_lin": {
        "Base": "profile1D linear",
        "KFstd": "profile1d_linear__kfstd",
        "PARH": "profile1d_linear__parh_ossm",
    },
    "P1D_quad": {
        "Base": "profile1D quadratic",
        "KFstd": "profile1d_quadratic__kfstd",
        "PARH": "profile1d_quadratic__parh_ossm",
    },
    "P1D_cub": {
        "Base": "profile1D cubic",
        "KFstd": "profile1d_cubic__kfstd",
        "PARH": "profile1d_cubic__parh_ossm",
    },
}


def preferred_output_type(variant: str) -> str:
    return "z_full" if variant == "PARH" else "signal_hat"


def build_manifest(
    waveform_csv: Path,
    freq_csv: Path,
    family: str,
    dataset_name: str,
    metric_col: str,
    out_path: Path,
) -> pd.DataFrame:
    family = str(family)
    if family not in FAMILY_TO_METHODS:
        raise ValueError(f"Unsupported family: {family}")

    wave = pd.read_csv(waveform_csv)
    freq = pd.read_csv(freq_csv)
    wave = wave[wave["causal_or_smoothed"] == "smoothed"].copy()

    rows = []
    for variant, method in FAMILY_TO_METHODS[family].items():
        output_type = preferred_output_type(variant)
        wsub = wave[(wave["method"] == method) & (wave["output_type"] == output_type)].copy()
        if wsub.empty:
            continue
        fsub = freq[freq["method"] == method][["video", "method", "MAE", "RMSE", "PearsonR", "rate_source"]].copy()
        fsub = fsub.rename(
            columns={
                "MAE": "rate_MAE",
                "RMSE": "rate_RMSE",
                "PearsonR": "rate_PearsonR",
            }
        )
        merged = wsub.merge(fsub, on=["video", "method"], how="left")
        merged["variant"] = variant
        rows.append(merged)
    merged = pd.concat(rows, ignore_index=True)

    parh = merged[merged["variant"] == "PARH"].copy()
    parh[metric_col] = pd.to_numeric(parh[metric_col], errors="coerce")
    parh = parh[np.isfinite(parh[metric_col].to_numpy())].copy()
    parh = parh.sort_values(metric_col, ascending=False).reset_index(drop=True)
    if parh.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_path, index=False)
        return pd.DataFrame()

    picks = {
        "top": 0,
        "median": len(parh) // 2,
        "bottom": len(parh) - 1,
    }
    manifest_rows = []
    for case_rank, idx in picks.items():
        ref = parh.iloc[int(np.clip(idx, 0, len(parh) - 1))]
        video = ref["video"]
        for variant in ("Base", "KFstd", "PARH"):
            method = FAMILY_TO_METHODS[family][variant]
            row = merged[(merged["video"] == video) & (merged["method"] == method)]
            if row.empty:
                continue
            row = row.iloc[0].copy()
            row["dataset"] = dataset_name
            row["family"] = family
            row["case_rank"] = case_rank
            row["panel_group"] = f"{dataset_name}_{family}_{case_rank}"
            row["metric_name"] = metric_col
            row["metric_value"] = ref[metric_col]
            manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    ordered = [
        "dataset", "family", "case_rank", "panel_group", "variant",
        "video", "method", "data_file", "output_type", "causal_or_smoothed",
        "metric_name", "metric_value", "waveform_CCC", "waveform_MAE",
        "waveform_DTW", "latency_ms", "rate_MAE", "rate_RMSE",
        "rate_PearsonR", "rate_source",
    ]
    manifest = manifest[[c for c in ordered if c in manifest.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return manifest


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate same-trial waveform overlay manifest.")
    parser.add_argument(
        "--waveform-csv",
        type=Path,
        required=True,
        help="Path to metrics_waveform_raw.csv",
    )
    parser.add_argument(
        "--freq-csv",
        type=Path,
        required=True,
        help="Path to metrics_freq_domain_raw.csv",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="P1D_quad",
        choices=sorted(FAMILY_TO_METHODS),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="COHFACE",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="waveform_CCC",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "paper" / "manifests" / "cohface_waveform_overlay_manifest.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        waveform_csv=args.waveform_csv,
        freq_csv=args.freq_csv,
        family=args.family,
        dataset_name=args.dataset_name,
        metric_col=args.metric,
        out_path=args.out,
    )
    print(f"Saved manifest: {args.out}")
    print(f"Rows: {len(manifest)}")
    if not manifest.empty:
        print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
