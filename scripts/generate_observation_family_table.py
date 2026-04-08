#!/usr/bin/env python3
"""Generate a paper-ready observation-family semantics table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.observations.semantics import (
    canonicalize_observation_family,
    get_observation_family_semantics,
)

FAMILY_ORDER = [
    "of_farneback",
    "of_disp_bridge",
    "profile1d_linear",
    "profile1d_quadratic",
    "profile1d_cubic",
    "dof",
]


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


def build_table(rate_csv: Path, waveform_csv: Path, out_csv: Path) -> pd.DataFrame:
    rate = pd.read_csv(rate_csv)
    wave = pd.read_csv(waveform_csv)

    rows = []
    for family_name in FAMILY_ORDER:
        sem = get_observation_family_semantics(family_name)
        display = str(sem.get("display_name") or family_name)

        rate_row = rate[rate["family"] == display]
        wave_row = wave[wave["family"] == display]
        if rate_row.empty or wave_row.empty:
            continue
        rate_row = rate_row.iloc[0]
        wave_row = wave_row.iloc[0]

        rows.append(
            {
                "family": display,
                "construction": sem.get("construction", ""),
                "domain": sem.get("observation_domain", ""),
                "primary_information": str(sem.get("primary_information", "")).replace("_", " "),
                "secondary_information": str(sem.get("secondary_information", "")).replace("_", " "),
                "nuisance_risk": sem.get("nuisance_risk", ""),
                "current_parh_role": sem.get("current_parh_role", ""),
                "current_strength": _current_strength(sem),
                "PARH_rate_MAE": float(rate_row["PARH_MAE"]),
                "PARH_rate_R": float(rate_row["PARH_PearsonR"]),
                "PARH_waveform_CCC": float(wave_row["PARH_CCC"]),
                "PARH_waveform_DTW": float(wave_row["PARH_DTW"]),
                "Base_rate_MAE": float(rate_row["Base_MAE"]),
                "Base_waveform_CCC": float(wave_row["Base_CCC"]),
                "rate_gap_vs_base": float(rate_row["PARH_MAE"] - rate_row["Base_MAE"]),
                "waveform_gap_vs_base": float(wave_row["PARH_CCC"] - wave_row["Base_CCC"]),
            }
        )

    table = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False, float_format="%.3f")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate observation-family summary table.")
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
        default=ROOT / "paper" / "tables_ready" / "T2_observation_family_map.csv",
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
