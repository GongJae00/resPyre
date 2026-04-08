#!/usr/bin/env python3
"""Generate a compact persistent manifest for main-paper failure cases."""

import argparse
from pathlib import Path

import pandas as pd


def build_manifest(
    residual_manifest: Path,
    bridge_manifest: Path,
    out_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    residual_df = pd.read_csv(residual_manifest)
    bridge_df = pd.read_csv(bridge_manifest)

    rows = []

    residual_row = residual_df[
        (residual_df["family"] == "OF") & (residual_df["case_rank"] == "high_residual")
    ]
    if not residual_row.empty:
        row = residual_row.iloc[0].copy()
        row["dataset"] = dataset_name
        row["panel_kind"] = "residual_heavy_of"
        row["panel_id"] = f"{dataset_name}_residual_heavy_of"
        rows.append(row)

    bridge_row = bridge_df[bridge_df["case_rank"] == "worst_bridge_gain"]
    if not bridge_row.empty:
        row = bridge_row.iloc[0].copy()
        row["dataset"] = dataset_name
        row["panel_kind"] = "of_bridge_failure"
        row["panel_id"] = f"{dataset_name}_of_bridge_failure"
        rows.append(row)

    manifest = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return manifest


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate main-paper failure-case manifest.")
    parser.add_argument(
        "--residual-manifest",
        type=Path,
        default=root / "paper" / "manifests" / "cohface_residual_case_manifest.csv",
    )
    parser.add_argument(
        "--bridge-manifest",
        type=Path,
        default=root / "paper" / "manifests" / "cohface_ofbridge_case_manifest.csv",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="COHFACE",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "paper" / "manifests" / "cohface_failure_case_manifest.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_manifest(
        residual_manifest=args.residual_manifest,
        bridge_manifest=args.bridge_manifest,
        out_path=args.out,
        dataset_name=args.dataset_name,
    )
    print(f"Saved manifest: {args.out}")
    print(f"Rows: {len(manifest)}")
    if not manifest.empty:
        print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
