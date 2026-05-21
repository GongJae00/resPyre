#!/usr/bin/env python3
"""Generate a dual-regime overlay manifest by combining two family-specific overlay manifests."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_waveform_overlay_manifest import build_manifest


def _resolve_data_files(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    out = df.copy()
    out["data_file"] = out["data_file"].map(lambda x: str((run_dir / str(x)).resolve()) if not Path(str(x)).is_absolute() else str(Path(str(x)).resolve()))
    return out


def _build_one(waveform_csv: Path, freq_csv: Path, family: str, dataset_name: str, run_dir: Path, dataset_order: int) -> pd.DataFrame:
    tmp = build_manifest(
        waveform_csv=waveform_csv,
        freq_csv=freq_csv,
        family=family,
        dataset_name=dataset_name,
        metric_col="waveform_CCC",
        out_path=ROOT / ".tmp_fig_review" / f"_{dataset_name}_{family}_overlay_manifest.csv",
    )
    if tmp.empty:
        return tmp
    tmp = _resolve_data_files(tmp, run_dir)
    tmp["dataset_display_order"] = dataset_order
    rank_order = {"top": 0, "median": 1, "bottom": 2}
    tmp["row_display_order"] = tmp["case_rank"].map(rank_order).fillna(99).astype(int)
    return tmp


def parse_args():
    p = argparse.ArgumentParser(description="Generate a combined dual-regime overlay manifest.")
    p.add_argument("--a-waveform-csv", type=Path, required=True)
    p.add_argument("--a-freq-csv", type=Path, required=True)
    p.add_argument("--a-family", type=str, required=True)
    p.add_argument("--a-dataset-name", type=str, required=True)
    p.add_argument("--a-run-dir", type=Path, required=True)
    p.add_argument("--b-waveform-csv", type=Path, required=True)
    p.add_argument("--b-freq-csv", type=Path, required=True)
    p.add_argument("--b-family", type=str, required=True)
    p.add_argument("--b-dataset-name", type=str, required=True)
    p.add_argument("--b-run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    a = _build_one(args.a_waveform_csv, args.a_freq_csv, args.a_family, args.a_dataset_name, args.a_run_dir, 0)
    b = _build_one(args.b_waveform_csv, args.b_freq_csv, args.b_family, args.b_dataset_name, args.b_run_dir, 1)
    merged = pd.concat([a, b], ignore_index=True)
    merged = merged.sort_values(["dataset_display_order", "row_display_order", "variant_display_order", "variant", "video"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"Saved manifest: {args.out}")
    print(f"Rows: {len(merged)}")


if __name__ == "__main__":
    main()
