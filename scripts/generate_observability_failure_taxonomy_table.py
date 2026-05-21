#!/usr/bin/env python3
"""Generate a paper-ready observability failure taxonomy table.

The input is produced by ``audit_target_observability_failure_modes.py``.  This
script does not create new evidence.  It compresses the diagnostic rows into a
manuscript-facing table that separates current-bank limits, source-selection
room, and video/reference-limited cases.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


MODE_TEXT: Dict[str, Dict[str, str]] = {
    "bounded_or_no_clear_room": {
        "paper_label": "Bounded by current observation bank",
        "interpretation": (
            "Current candidate sources have little oracle room; better source "
            "selection alone is unlikely to solve these trials."
        ),
        "next_need": "Create richer respiratory observations or report low observability.",
    },
    "source_selection_room_posterior_available": {
        "paper_label": "Source-selection room with posterior evidence",
        "interpretation": (
            "A different source can help and the label-free posterior already "
            "contains some usable evidence."
        ),
        "next_need": "Use posterior evidence cautiously as an adaptive-law diagnostic.",
    },
    "source_selection_room_agreement_available": {
        "paper_label": "Source-selection room with agreement evidence",
        "interpretation": (
            "A different source can help and cross-source agreement is partly "
            "informative."
        ),
        "next_need": "Improve agreement-to-state trust without hard source switching.",
    },
    "oracle_room_but_gtfree_evidence_weak": {
        "paper_label": "Oracle room but weak label-free evidence",
        "interpretation": (
            "A better source exists, but current label-free diagnostics "
            "do not identify it reliably."
        ),
        "next_need": "Develop stronger observability features before promotion.",
    },
    "low_target_observability": {
        "paper_label": "Very low label-free observability",
        "interpretation": (
            "Very low label-free support; retained as explicit hard-observability "
            "evidence rather than removed from the evaluation cohort."
        ),
        "next_need": "Create richer respiratory observations or report low observability.",
    },
    "likely_video_or_reference_limited": {
        "paper_label": "Likely video/reference limited",
        "interpretation": (
            "Even the best current source remains poor, suggesting weak visual "
            "respiratory evidence or reference/scale/lag risk."
        ),
        "next_need": "Separate reference-risk reporting from model-error claims.",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    p.add_argument(
        "--failure-csv",
        type=Path,
        default=root / "analysis" / "final_mahnob_tailaligned_observability_failure_modes.csv",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=root / "paper" / "tables_ready" / "T7_observability_failure_taxonomy.csv",
    )
    p.add_argument(
        "--report-out",
        type=Path,
        default=root / "analysis" / "final_mahnob_tailaligned_observability_failure_taxonomy.md",
    )
    return p.parse_args()


def _median(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns:
        return float("nan")
    vals = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return "nan" if not math.isfinite(float(value)) else f"{float(value):.3f}"
    return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    frame = pd.read_csv(args.failure_csv)
    if "failure_mode" not in frame.columns:
        raise SystemExit("failure CSV must contain failure_mode")
    n_total = int(len(frame))
    rows: List[Dict[str, object]] = []
    for mode, sub in frame.groupby("failure_mode", dropna=False):
        text = MODE_TEXT.get(str(mode), {})
        rows.append(
            {
                "failure_mode": str(mode),
                "paper_label": text.get("paper_label", str(mode)),
                "n_trials": int(len(sub)),
                "fraction": float(len(sub) / max(n_total, 1)),
                "median_final_mae_bpm": _median(sub, "final_mae"),
                "median_best_source_mae_bpm": _median(sub, "oracle_best_mae"),
                "median_oracle_room_bpm": _median(sub, "oracle_room_bpm"),
                "median_observability": _median(sub, "target_observability_score"),
                "median_source_spread_bpm": _median(sub, "source_spread_bpm"),
                "interpretation": text.get("interpretation", ""),
                "next_need": text.get("next_need", ""),
            }
        )
    out = pd.DataFrame(rows).sort_values(["n_trials", "failure_mode"], ascending=[False, True])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    report = [
        "# Observability Failure Taxonomy Table",
        "",
        f"- source: `{args.failure_csv}`",
        f"- trials: `{n_total}`",
        "- boundary: paper-facing compression of a diagnostic audit; oracle-best columns are used only to interpret current observation-bank limits.",
        "",
        "## Table",
        "",
        _markdown_table(out),
        "",
        "## Paper Use",
        "",
        "- Use this table to frame MAHNOB as an irregular/low-observability stress regime.",
        "- Do not claim that PARH-OSSM solves MAHNOB strict reconstruction.",
        "- The main scientific claim is that the observation-state decomposition exposes when current camera observations are insufficient and what future respiratory observations must supply.",
    ]
    args.report_out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
