#!/usr/bin/env python3
"""Generate residual-focused case-study manifests for PARH analysis."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.observations.semantics import CANONICAL_OBSERVATION_FAMILY_ORDER, get_observation_family_semantics

FAMILY_ORDER = [
    str(get_observation_family_semantics(name)["display_name"])
    for name in CANONICAL_OBSERVATION_FAMILY_ORDER
]


def ordered_families(families):
    seen = [f for f in FAMILY_ORDER if f in set(families)]
    rest = [f for f in families if f not in seen]
    return seen + rest


def normalize_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.dropna().empty:
        return pd.Series(np.zeros(len(series), dtype=float), index=series.index)
    ranks = vals.rank(method="average", ascending=ascending, pct=True)
    return ranks.fillna(0.0)


def build_manifest(mech_trials: Path, waveform_csv: Path, freq_csv: Path, out: Path, dataset_name: str):
    mech = pd.read_csv(mech_trials)
    wave = pd.read_csv(waveform_csv)
    freq = pd.read_csv(freq_csv)

    mech = mech.copy()
    mech = mech[mech["method"].astype(str).str.contains("__parh_ossm", na=False)].copy()
    if mech.empty:
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out, index=False)
        return pd.DataFrame()

    wave = wave[(wave["output_type"] == "z_full") & (wave["causal_or_smoothed"] == "smoothed")].copy()
    freq = freq.copy()

    merged = mech.merge(
        wave[["video", "method", "waveform_CCC", "waveform_MAE", "waveform_DTW", "data_file"]],
        on=["video", "method"],
        how="left",
    )
    merged = merged.merge(
        freq[["video", "method", "MAE", "RMSE", "PearsonR", "rate_source"]],
        on=["video", "method"],
        how="left",
    ).rename(columns={"MAE": "rate_MAE", "RMSE": "rate_RMSE", "PearsonR": "rate_PearsonR"})

    merged["residual_rank"] = normalize_rank(merged["residual_energy_ratio"], ascending=True)
    merged["nonosc_rank"] = normalize_rank(merged["obs_nonosc_need_mean"], ascending=True)
    merged["waveform_bad_rank"] = normalize_rank(merged["waveform_CCC"], ascending=False)
    merged["rate_bad_rank"] = normalize_rank(merged["rate_MAE"], ascending=True)
    merged["case_score"] = (
        0.45 * merged["residual_rank"]
        + 0.25 * merged["nonosc_rank"]
        + 0.15 * merged["waveform_bad_rank"]
        + 0.15 * merged["rate_bad_rank"]
    )

    rows = []
    for family in ordered_families(list(merged["family"].dropna().unique())):
        g = merged[merged["family"] == family].copy()
        if g.empty:
            continue
        g = g.sort_values(["case_score", "residual_energy_ratio", "obs_nonosc_need_mean"], ascending=[False, False, False]).reset_index(drop=True)
        picks = {
            "high_residual": 0,
            "median_residual": len(g) // 2,
            "low_residual": len(g) - 1,
        }
        for label, idx in picks.items():
            row = g.iloc[int(np.clip(idx, 0, len(g) - 1))].copy()
            row["dataset"] = dataset_name
            row["case_rank"] = label
            row["panel_id"] = f"{dataset_name}_{family}_{label}"
            rows.append(row)

    manifest = pd.DataFrame(rows)
    ordered_cols = [
        "dataset", "family", "case_rank", "panel_id", "video", "method", "data_file",
        "case_score", "residual_energy_ratio", "obs_nonosc_need_mean", "q_osc_mean",
        "obs_osc_support_mean", "waveform_CCC", "waveform_MAE", "waveform_DTW",
        "rate_MAE", "rate_RMSE", "rate_PearsonR", "rate_source",
        "baseline_energy_ratio", "max_abs_zfull_minus_zosc", "max_abs_zosc_causal_minus_smoothed",
    ]
    manifest = manifest[[c for c in ordered_cols if c in manifest.columns]]
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    return manifest


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate residual-focused PARH case-study manifest.")
    parser.add_argument(
        "--mechanism-trials",
        type=Path,
        default=root / "paper" / "manifests" / "cohface_parh_mechanism_trials.csv",
    )
    parser.add_argument(
        "--waveform-csv",
        type=Path,
        default=root / "results" / "20260409_cohface_prod_ofbridge_dofbridge_p1dcons_e2e_policy_narrow" / "cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons" / "metrics" / "metrics_waveform_raw.csv",
    )
    parser.add_argument(
        "--freq-csv",
        type=Path,
        default=root / "results" / "20260409_cohface_prod_ofbridge_dofbridge_p1dcons_e2e_policy_narrow" / "cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons" / "metrics" / "metrics_freq_domain_raw.csv",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="COHFACE",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "paper" / "manifests" / "cohface_residual_case_manifest.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        mech_trials=args.mechanism_trials,
        waveform_csv=args.waveform_csv,
        freq_csv=args.freq_csv,
        out=args.out,
        dataset_name=args.dataset_name,
    )
    print(f"Saved manifest: {args.out}")
    print(f"Rows: {len(manifest)}")
    if not manifest.empty:
        print(manifest.head(18).to_string(index=False))


if __name__ == "__main__":
    main()
