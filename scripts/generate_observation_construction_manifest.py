#!/usr/bin/env python3
"""Generate manifest for raw OF vs OF_bridge comparison cases."""

import argparse
from pathlib import Path

import pandas as pd


def build_manifest(waveform_csv: Path, freq_csv: Path, out: Path, dataset_name: str):
    wave = pd.read_csv(waveform_csv)
    freq = pd.read_csv(freq_csv)

    wave = wave[(wave["causal_or_smoothed"] == "smoothed") & (wave["output_type"] == "z_full")].copy()
    freq = freq.copy()

    raw_name = "of_farneback__parh_ossm"
    bridge_name = "of_disp_bridge__parh_ossm"

    raw_wave = wave[wave["method"] == raw_name][["video", "data_file", "waveform_CCC", "waveform_MAE", "waveform_DTW"]].rename(
        columns=lambda c: f"raw_{c}" if c not in {"video", "data_file"} else c
    )
    bridge_wave = wave[wave["method"] == bridge_name][["video", "data_file", "waveform_CCC", "waveform_MAE", "waveform_DTW"]].rename(
        columns=lambda c: f"bridge_{c}" if c not in {"video", "data_file"} else c
    )
    raw_freq = freq[freq["method"] == raw_name][["video", "data_file", "MAE", "RMSE", "PearsonR"]].rename(
        columns={"MAE": "raw_rate_MAE", "RMSE": "raw_rate_RMSE", "PearsonR": "raw_rate_PearsonR"}
    )
    bridge_freq = freq[freq["method"] == bridge_name][["video", "data_file", "MAE", "RMSE", "PearsonR"]].rename(
        columns={"MAE": "bridge_rate_MAE", "RMSE": "bridge_rate_RMSE", "PearsonR": "bridge_rate_PearsonR"}
    )

    merged = raw_wave.merge(bridge_wave, on=["video", "data_file"], how="inner")
    merged = merged.merge(raw_freq, on=["video", "data_file"], how="inner")
    merged = merged.merge(bridge_freq, on=["video", "data_file"], how="inner")

    merged["delta_rate_MAE"] = merged["bridge_rate_MAE"] - merged["raw_rate_MAE"]
    merged["delta_rate_RMSE"] = merged["bridge_rate_RMSE"] - merged["raw_rate_RMSE"]
    merged["delta_rate_PearsonR"] = merged["bridge_rate_PearsonR"] - merged["raw_rate_PearsonR"]
    merged["delta_waveform_CCC"] = merged["bridge_waveform_CCC"] - merged["raw_waveform_CCC"]
    merged["delta_waveform_MAE"] = merged["bridge_waveform_MAE"] - merged["raw_waveform_MAE"]
    merged["delta_waveform_DTW"] = merged["bridge_waveform_DTW"] - merged["raw_waveform_DTW"]
    merged["bridge_score"] = (
        -merged["delta_rate_MAE"] - merged["delta_rate_RMSE"]
        + merged["delta_rate_PearsonR"]
        + 0.5 * merged["delta_waveform_CCC"]
        - 0.5 * merged["delta_waveform_MAE"]
        - 0.5 * merged["delta_waveform_DTW"]
    )

    merged = merged.sort_values("bridge_score", ascending=False).reset_index(drop=True)
    picks = {
        "best_bridge_gain": 0,
        "median_bridge_gain": len(merged) // 2 if len(merged) else 0,
        "worst_bridge_gain": max(len(merged) - 1, 0),
    }
    rows = []
    for label, idx in picks.items():
        if merged.empty:
            break
        row = merged.iloc[idx].copy()
        row["dataset"] = dataset_name
        row["case_rank"] = label
        row["panel_id"] = f"{dataset_name}_OFbridge_{label}"
        rows.append(row)

    manifest = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    return manifest


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate OF raw vs OF_bridge comparison manifest.")
    parser.add_argument(
        "--waveform-csv",
        type=Path,
        default=root / "results" / "20260408_cohface_prod_ofbridge_full" / "cohface_parh_ossm_prod_ofbridge" / "metrics" / "metrics_waveform_raw.csv",
    )
    parser.add_argument(
        "--freq-csv",
        type=Path,
        default=root / "results" / "20260408_cohface_prod_ofbridge_full" / "cohface_parh_ossm_prod_ofbridge" / "metrics" / "metrics_freq_domain_raw.csv",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="COHFACE",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "paper" / "manifests" / "cohface_ofbridge_case_manifest.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(args.waveform_csv, args.freq_csv, args.out, args.dataset_name)
    print(f"Saved manifest: {args.out}")
    print(f"Rows: {len(manifest)}")
    if not manifest.empty:
        print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
