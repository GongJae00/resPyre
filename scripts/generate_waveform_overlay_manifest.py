#!/usr/bin/env python3
"""Generate same-trial overlay manifests for waveform figures."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FAMILY_TO_METHODS = {
    "OF": {
        "Base": "of_farneback",
        "OSSM-KF": "of_farneback__kfstd",
        "PARH": "of_farneback__parh_ossm",
    },
    "OF_bridge": {
        "Base": "of_disp_bridge",
        "OSSM-KF": "of_disp_bridge__kfstd",
        "PARH": "of_disp_bridge__parh_ossm",
    },
    "DoF": {
        "Base": "DoF",
        "OSSM-KF": "dof__kfstd",
        "PARH": "dof__parh_ossm",
    },
    "DoF_bridge": {
        "Base": "dof_disp_bridge",
        "OSSM-KF": "dof_disp_bridge__kfstd",
        "PARH": "dof_disp_bridge__parh_ossm",
    },
    "P1D_lin": {
        "Base": "profile1D linear",
        "OSSM-KF": "profile1d_linear__kfstd",
        "PARH": "profile1d_linear__parh_ossm",
    },
    "P1D_lin_bridge": {
        "Base": "profile1d_linear_bridge",
        "OSSM-KF": "profile1d_linear_bridge__kfstd",
        "PARH": "profile1d_linear_bridge__parh_ossm",
    },
    "P1D_quad": {
        "Base": "profile1D quadratic",
        "OSSM-KF": "profile1d_quadratic__kfstd",
        "PARH": "profile1d_quadratic__parh_ossm",
    },
    "P1D_quad_bridge": {
        "Base": "profile1d_quadratic_bridge",
        "OSSM-KF": "profile1d_quadratic_bridge__kfstd",
        "PARH": "profile1d_quadratic_bridge__parh_ossm",
    },
    "P1D_cub": {
        "Base": "profile1D cubic",
        "OSSM-KF": "profile1d_cubic__kfstd",
        "PARH": "profile1d_cubic__parh_ossm",
    },
    "P1D_cub_bridge": {
        "Base": "profile1d_cubic_bridge",
        "OSSM-KF": "profile1d_cubic_bridge__kfstd",
        "PARH": "profile1d_cubic_bridge__parh_ossm",
    },
    "P1D_cons": {
        "Base": "profile1d_consensus",
        "OSSM-KF": "profile1d_consensus__kfstd",
        "PARH": "profile1d_consensus__parh_ossm",
    },
}

DEFAULT_VARIANT_ORDER = ["Base", "OSSM-KF", "PARH", "Final"]


def preferred_output_type(variant: str) -> str:
    return "z_full" if variant == "PARH" else "signal_hat"


def parse_extra_variants(specs):
    extras = {}
    for raw in specs or []:
        spec = str(raw).strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(
                f"Invalid --extra-variant '{raw}'. Expected LABEL=method or LABEL=method:output_type."
            )
        label, rhs = spec.split("=", 1)
        label = str(label).strip()
        if not label:
            raise ValueError(f"Invalid --extra-variant '{raw}': empty label.")
        if ":" in rhs:
            method, output_type = rhs.split(":", 1)
        else:
            method, output_type = rhs, "z_full"
        extras[label] = {
            "method": str(method).strip(),
            "output_type": str(output_type).strip() or "z_full",
        }
    return extras


def build_variant_methods(family: str, extra_variants=None):
    family = str(family)
    if family not in FAMILY_TO_METHODS:
        raise ValueError(f"Unsupported family: {family}")
    variant_methods = {
        variant: {"method": method, "output_type": preferred_output_type(variant)}
        for variant, method in FAMILY_TO_METHODS[family].items()
    }
    for label, spec in (extra_variants or {}).items():
        variant_methods[label] = {
            "method": spec["method"],
            "output_type": spec["output_type"],
        }
    return variant_methods


def build_manifest(
    waveform_csv: Path,
    freq_csv: Path,
    family: str,
    dataset_name: str,
    metric_col: str,
    out_path: Path,
    extra_variants=None,
) -> pd.DataFrame:
    family = str(family)
    variant_methods = build_variant_methods(family, extra_variants=extra_variants)

    wave = pd.read_csv(waveform_csv)
    freq = pd.read_csv(freq_csv)
    wave = wave[wave["causal_or_smoothed"] == "smoothed"].copy()

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
        ordered_variants = [v for v in DEFAULT_VARIANT_ORDER if v in variant_methods]
        ordered_variants += [v for v in variant_methods if v not in ordered_variants]
        for display_order, variant in enumerate(ordered_variants):
            method = variant_methods[variant]["method"]
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
            row["variant_display_order"] = display_order
            manifest_rows.append(row)

    manifest = pd.DataFrame(manifest_rows)
    ordered = [
        "dataset", "family", "case_rank", "panel_group", "variant",
        "variant_display_order",
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
        "--extra-variant",
        action="append",
        default=[],
        help="Additional overlay column in the form LABEL=method or LABEL=method:output_type. "
             "Useful for future Final/fused models.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "paper" / "manifests" / "cohface_waveform_overlay_manifest.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extra_variants = parse_extra_variants(args.extra_variant)
    manifest = build_manifest(
        waveform_csv=args.waveform_csv,
        freq_csv=args.freq_csv,
        family=args.family,
        dataset_name=args.dataset_name,
        metric_col=args.metric,
        out_path=args.out,
        extra_variants=extra_variants,
    )
    print(f"Saved manifest: {args.out}")
    print(f"Rows: {len(manifest)}")
    if extra_variants:
        print(json.dumps({"extra_variants": extra_variants}, indent=2))
    if not manifest.empty:
        print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
