#!/usr/bin/env python3
"""Generate a table-ready observation-class semantics table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.observations.semantics import (
    CANONICAL_OBSERVATION_FAMILY_ORDER,
    canonicalize_observation_family,
    get_observation_family_semantics,
)

OBSERVATION_CLASS_ORDER = list(CANONICAL_OBSERVATION_FAMILY_ORDER)


def _current_strength(sem: dict) -> str:
    if sem.get("waveform_primary"):
        return "waveform-primary"
    if sem.get("rate_primary"):
        return "rate-primary"
    if sem.get("helper_heavy"):
        return "helper-heavy"
    if sem.get("nuisance_risk") == "high":
        return "nuisance-limited"
    return "mixed"


def _safe_float(row: pd.Series | None, key: str):
    if row is None or key not in row.index:
        return pd.NA
    val = row[key]
    if pd.isna(val) or str(val).strip() == "":
        return pd.NA
    try:
        return float(val)
    except Exception:
        return pd.NA


def _safe_gap(a, b):
    if pd.isna(a) or pd.isna(b):
        return pd.NA
    return float(a) - float(b)


def build_table(rate_csv: Path, waveform_csv: Path, out_csv: Path) -> pd.DataFrame:
    rate = pd.read_csv(rate_csv)
    wave = pd.read_csv(waveform_csv)
    rate_class_col = "observation_class" if "observation_class" in rate.columns else "family"
    wave_class_col = "observation_class" if "observation_class" in wave.columns else "family"

    rows = []
    for family_name in OBSERVATION_CLASS_ORDER:
        sem = get_observation_family_semantics(family_name)
        display = str(sem.get("display_name") or family_name)

        rate_df = rate[rate[rate_class_col] == display]
        wave_df = wave[wave[wave_class_col] == display]
        rate_row = rate_df.iloc[0] if not rate_df.empty else None
        wave_row = wave_df.iloc[0] if not wave_df.empty else None

        parh_rate_mae = _safe_float(rate_row, "PARH_MAE")
        parh_rate_r = _safe_float(rate_row, "PARH_PearsonR")
        parh_wave_ccc = _safe_float(wave_row, "PARH_CCC")
        parh_wave_dtw = _safe_float(wave_row, "PARH_DTW")
        base_rate_mae = _safe_float(rate_row, "Base_MAE")
        base_wave_ccc = _safe_float(wave_row, "Base_CCC")

        rows.append(
            {
                "observation_class": display,
                "construction": sem.get("construction", ""),
                "domain": sem.get("observation_domain", ""),
                "primary_information": str(sem.get("primary_information", "")).replace("_", " "),
                "secondary_information": str(sem.get("secondary_information", "")).replace("_", " "),
                "nuisance_risk": sem.get("nuisance_risk", ""),
                "current_parh_role": str(sem.get("current_parh_role", "")).replace("family", "observation class"),
                "current_strength": _current_strength(sem),
                "PARH_rate_MAE": parh_rate_mae,
                "PARH_rate_R": parh_rate_r,
                "PARH_waveform_CCC": parh_wave_ccc,
                "PARH_waveform_DTW": parh_wave_dtw,
                "Base_rate_MAE": base_rate_mae,
                "Base_waveform_CCC": base_wave_ccc,
                "rate_gap_vs_base": _safe_gap(parh_rate_mae, base_rate_mae),
                "waveform_gap_vs_base": _safe_gap(parh_wave_ccc, base_wave_ccc),
            }
        )

    table = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False, float_format="%.3f")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate observation-class summary table.")
    parser.add_argument(
        "--rate-csv",
        type=Path,
        default=ROOT / "paper" / "tables_ready" / "T3_rate_main.csv",
    )
    parser.add_argument(
        "--waveform-csv",
        type=Path,
        default=ROOT / "paper" / "tables_ready" / "T4_waveform_main.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "paper" / "tables_ready" / "T2_observation_class_map.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = build_table(args.rate_csv, args.waveform_csv, args.out_csv)
    print(f"Saved: {args.out_csv}")
    print(f"Rows: {len(table)}")
    if not table.empty:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
