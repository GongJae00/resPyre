#!/usr/bin/env python3
"""Generate median-case observation-class overlay manifests for supplementary figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.observations.semantics import (
    CANONICAL_OBSERVATION_FAMILY_ORDER,
    get_observation_family_semantics,
)
from generate_waveform_overlay_manifest import (
    DEFAULT_VARIANT_ORDER,
    build_variant_methods,
)


def build_observation_class_manifest(
    waveform_csv: Path,
    freq_csv: Path,
    dataset_name: str,
    out_path: Path,
    metric_col: str = "waveform_CCC",
    observation_classes: list[str] | None = None,
) -> pd.DataFrame:
    wave = pd.read_csv(waveform_csv)
    freq = pd.read_csv(freq_csv)
    wave = wave[wave["causal_or_smoothed"] == "smoothed"].copy()

    observation_classes = observation_classes or [
        str(get_observation_family_semantics(name).get("display_name"))
        for name in CANONICAL_OBSERVATION_FAMILY_ORDER
    ]
    manifest_rows: list[pd.Series] = []
    for class_idx, observation_class in enumerate(observation_classes):
        variant_methods = build_variant_methods(observation_class)
        rows = []
        for variant, spec in variant_methods.items():
            method = str(spec["method"])
            output_type = str(spec["output_type"])
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
        if not rows:
            continue

        merged = pd.concat(rows, ignore_index=True)
        parh = merged[merged["variant"] == "PARH"].copy()
        parh[metric_col] = pd.to_numeric(parh[metric_col], errors="coerce")
        parh = parh[np.isfinite(parh[metric_col].to_numpy())].sort_values(metric_col, ascending=False).reset_index(drop=True)
        if parh.empty:
            continue

        ref = parh.iloc[len(parh) // 2]
        video = ref["video"]
        ordered_variants = [v for v in DEFAULT_VARIANT_ORDER if v in variant_methods]
        ordered_variants += [v for v in variant_methods if v not in ordered_variants]
        for display_order, variant in enumerate(ordered_variants):
            method = variant_methods[variant]["method"]
            row = merged[(merged["video"] == video) & (merged["method"] == method)]
            if row.empty:
                continue
            row = row.iloc[0].copy()
            row["dataset"] = dataset_name
            row["observation_class"] = observation_class
            row["case_rank"] = "median"
            row["panel_group"] = f"{dataset_name}_{observation_class}_median"
            row["metric_name"] = metric_col
            row["metric_value"] = ref[metric_col]
            row["variant_display_order"] = display_order
            row["dataset_display_order"] = 0
            row["row_display_order"] = class_idx
            manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    ordered = [
        "dataset", "observation_class", "case_rank", "panel_group", "variant",
        "variant_display_order", "dataset_display_order", "row_display_order",
        "video", "method", "data_file", "output_type", "causal_or_smoothed",
        "metric_name", "metric_value", "waveform_CCC", "waveform_MAE",
        "waveform_DTW", "latency_ms", "rate_MAE", "rate_RMSE",
        "rate_PearsonR", "rate_source",
    ]
    if not manifest.empty:
        manifest = manifest[[c for c in ordered if c in manifest.columns]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate observation-class waveform overlay manifest.")
    parser.add_argument("--waveform-csv", type=Path, required=True)
    parser.add_argument("--freq-csv", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--metric", type=str, default="waveform_CCC")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_observation_class_manifest(
        waveform_csv=args.waveform_csv,
        freq_csv=args.freq_csv,
        dataset_name=args.dataset_name,
        out_path=args.out,
        metric_col=args.metric,
    )
    print(f"Wrote {len(manifest)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
