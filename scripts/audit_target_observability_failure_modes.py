#!/usr/bin/env python3
"""Decompose target-side rate failures into interpretable observability modes.

This is a diagnostic-only audit. It may use GT-derived rate-source MAE columns
to label failure modes, but the observability scores themselves are computed
from target-side metadata and source agreement. The output is intended to guide
the next PARH-OSSM observation-law patch, not to promote a new paper-facing
readout.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


SOURCE_ORDER = [
    "final_track_hz",
    "external_output_rate_t",
    "external_rate_posterior_mean_t",
    "native_smoothed_track_hz",
    "state_freq_t",
]


def _finite(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clip01(value: object) -> float:
    val = _finite(value, 0.0)
    return float(np.clip(val, 0.0, 1.0))


def _safe_col(row: pd.Series, name: str, default: float = np.nan) -> float:
    return _finite(row[name], default) if name in row.index else default


def _pivot_rate_decomposition(decomp: pd.DataFrame) -> pd.DataFrame:
    if not {"video", "rate_source", "MAE"}.issubset(decomp.columns):
        raise ValueError("decomposition CSV must contain video, rate_source, and MAE columns")
    mae = decomp.pivot_table(index="video", columns="rate_source", values="MAE", aggfunc="median")
    mae = mae.rename(columns={col: f"mae_{col}" for col in mae.columns})
    med = decomp.pivot_table(index="video", columns="rate_source", values="track_hz_median", aggfunc="median")
    med = med.rename(columns={col: f"track_median_{col}" for col in med.columns})
    out = mae.join(med, how="outer").reset_index()
    mae_cols = [f"mae_{src}" for src in SOURCE_ORDER if f"mae_{src}" in out.columns]
    out["oracle_best_source"] = out[mae_cols].idxmin(axis=1).str.replace("mae_", "", regex=False)
    out["oracle_best_mae"] = out[mae_cols].min(axis=1)
    out["final_mae"] = out.get("mae_final_track_hz", np.nan)
    out["oracle_room_bpm"] = out["final_mae"] - out["oracle_best_mae"]
    return out


def _source_spread_hz(row: pd.Series) -> float:
    vals = []
    for src in SOURCE_ORDER:
        val = _safe_col(row, f"track_median_{src}")
        if math.isfinite(val):
            vals.append(val)
    if len(vals) < 2:
        return float("nan")
    return float(np.nanmax(vals) - np.nanmin(vals))


def _observability_scores(row: pd.Series) -> Dict[str, float]:
    readout_conf = _clip01(
        _safe_col(
            row,
            "external_output_rate_confidence_mean",
            _safe_col(row, "readout_confidence_mean", _safe_col(row, "alpha_mean", 0.0)),
        )
    )
    posterior_conf = _clip01(_safe_col(row, "external_rate_posterior_confidence_mean", _safe_col(row, "posterior_confidence_mean", 0.0)))
    top_gap = _safe_col(row, "external_rate_posterior_top_gap_median", _safe_col(row, "posterior_top_gap_median", 0.0))
    entropy = _safe_col(row, "external_rate_posterior_entropy_median", _safe_col(row, "posterior_entropy_median", 1.0))
    support_count = _safe_col(
        row,
        "readout_support_group_count_median",
        _safe_col(
            row,
            "anchor_support_group_count_median",
            _safe_col(row, "posterior_macro_support_median", 0.0),
        ),
    )
    h1_support = _safe_col(row, "readout_h1_role_support_mean", _safe_col(row, "posterior_h1_role_support_median", 0.0))
    abstain = _safe_col(row, "readout_abstain_pressure_mean", _safe_col(row, "posterior_abstain_pressure_median", 0.0))
    readout_alias = _safe_col(row, "readout_alias_risk_mean", 0.0)
    posterior_alias = _safe_col(row, "posterior_alias_risk_median", 0.0)
    spread_hz = _safe_col(row, "source_spread_hz", np.nan)

    gap_score = _clip01((top_gap - 0.08) / 0.20)
    entropy_score = _clip01((0.98 - entropy) / 0.25)
    posterior_specificity = _clip01(0.55 * gap_score + 0.25 * entropy_score + 0.20 * posterior_conf)
    support_score = _clip01(support_count if support_count <= 1.0 else support_count / 7.0)
    h1_score = _clip01(h1_support)
    alias_safety = _clip01(1.0 - max(readout_alias, posterior_alias))
    abstain_score = _clip01(abstain)
    agreement_score = _clip01(math.exp(-spread_hz / 0.055)) if math.isfinite(spread_hz) else 0.0
    observability = _clip01(
        0.25 * readout_conf
        + 0.20 * posterior_specificity
        + 0.15 * support_score
        + 0.15 * h1_score
        + 0.15 * alias_safety
        + 0.10 * agreement_score
        - 0.10 * abstain_score
    )
    return {
        "readout_confidence_score": readout_conf,
        "posterior_specificity_score": posterior_specificity,
        "support_score": support_score,
        "h1_role_score": h1_score,
        "alias_safety_score": alias_safety,
        "source_agreement_score": agreement_score,
        "abstain_pressure_score": abstain_score,
        "target_observability_score": observability,
    }


def _classify(row: pd.Series) -> str:
    final_mae = _safe_col(row, "final_mae")
    best_mae = _safe_col(row, "oracle_best_mae")
    room = _safe_col(row, "oracle_room_bpm", 0.0)
    observability = _safe_col(row, "target_observability_score", 0.0)
    specificity = _safe_col(row, "posterior_specificity_score", 0.0)
    alias_safety = _safe_col(row, "alias_safety_score", 1.0)
    agreement = _safe_col(row, "source_agreement_score", 0.0)

    if not math.isfinite(final_mae):
        return "missing_metrics"
    if best_mae >= 3.5 and room < 0.35:
        return "likely_video_or_reference_limited"
    if room >= 0.75 and specificity >= 0.35:
        return "source_selection_room_posterior_available"
    if room >= 0.75 and agreement >= 0.35 and alias_safety >= 0.75:
        return "source_selection_room_agreement_available"
    if room >= 0.75:
        return "oracle_room_but_gtfree_evidence_weak"
    if final_mae >= 3.0 and observability < 0.45:
        return "low_target_observability"
    if final_mae >= 3.0 and alias_safety < 0.75:
        return "alias_risk_limited"
    return "bounded_or_no_clear_room"


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 24) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.head(max_rows).iterrows():
        vals: List[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("nan" if not math.isfinite(float(val)) else f"{float(val):.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(frame: pd.DataFrame, out_md: Path) -> None:
    lines = [
        "# Target-Side Observability Failure-Mode Audit",
        "",
        f"- trials: `{len(frame)}`",
        "- boundary: diagnostic-only; labels may use GT-derived oracle room, scores are target-computable.",
        "",
        "## Failure-Mode Counts",
        "",
    ]
    counts = frame["failure_mode"].value_counts().rename_axis("failure_mode").reset_index(name="n")
    lines.append(_markdown_table(counts, max_rows=20))
    lines.extend(["", "## Hardest Trials", ""])
    hard_cols = [
        "video",
        "failure_mode",
        "final_mae",
        "oracle_best_mae",
        "oracle_best_source",
        "oracle_room_bpm",
        "target_observability_score",
        "posterior_specificity_score",
        "alias_safety_score",
        "source_agreement_score",
        "source_spread_bpm",
    ]
    hard_cols = [c for c in hard_cols if c in frame.columns]
    lines.append(_markdown_table(frame.sort_values("final_mae", ascending=False)[hard_cols], max_rows=16))
    lines.extend(["", "## Largest Oracle Room", ""])
    lines.append(_markdown_table(frame.sort_values("oracle_room_bpm", ascending=False)[hard_cols], max_rows=16))
    score_cols = [
        "target_observability_score",
        "posterior_specificity_score",
        "source_agreement_score",
        "alias_safety_score",
        "source_spread_bpm",
        "readout_confidence_score",
        "support_score",
        "h1_role_score",
        "abstain_pressure_score",
    ]
    corr_rows: List[Dict[str, float | str]] = []
    for col in score_cols:
        if col in frame.columns:
            valid_room = frame[[col, "oracle_room_bpm"]].dropna()
            valid_mae = frame[[col, "final_mae"]].dropna()
            corr_rows.append({
                "feature": col,
                "corr_with_oracle_room": float(valid_room[col].corr(valid_room["oracle_room_bpm"]))
                if len(valid_room) >= 2
                else float("nan"),
                "corr_with_final_mae": float(valid_mae[col].corr(valid_mae["final_mae"]))
                if len(valid_mae) >= 2
                else float("nan"),
            })
    if corr_rows:
        corr = pd.DataFrame(corr_rows).sort_values("corr_with_oracle_room", ascending=False)
        lines.extend(["", "## Target-Computable Feature Correlations", ""])
        lines.append(_markdown_table(corr, max_rows=20))
    if "oracle_room_bpm" in frame.columns:
        cohort = frame.assign(room_ge_075=frame["oracle_room_bpm"] >= 0.75)
        keep = [c for c in score_cols if c in cohort.columns] + ["final_mae", "oracle_room_bpm"]
        grouped = cohort.groupby("room_ge_075")[keep].mean(numeric_only=True).reset_index()
        lines.extend(["", "## Mean Scores by Oracle Room >= 0.75 BPM", ""])
        lines.append(_markdown_table(grouped, max_rows=4))
    lines.extend(
        [
            "",
            "## Design Consequence",
            "",
            "- If many rows are `likely_video_or_reference_limited`, forcing a different readout cannot solve the dataset; the model needs uncertainty/observability reporting or reference-lag handling.",
            "- If many rows are `oracle_room_but_gtfree_evidence_weak`, the next patch should improve target-computable observability features before adding another source selector.",
            "- If many rows are `source_selection_room_*`, the observation law can safely learn a shallow target-side arbiter using those evidence channels.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition-csv", required=True, type=Path)
    parser.add_argument("--feature-csv", type=Path, default=None)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    decomp = pd.read_csv(args.decomposition_csv)
    frame = _pivot_rate_decomposition(decomp)
    if args.feature_csv is not None:
        features = pd.read_csv(args.feature_csv)
        drop_cols = [c for c in features.columns if c in frame.columns and c != "video"]
        frame = frame.merge(features.drop(columns=drop_cols), on="video", how="left")
    frame["source_spread_hz"] = frame.apply(_source_spread_hz, axis=1)
    frame["source_spread_bpm"] = frame["source_spread_hz"] * 60.0
    score_rows = frame.apply(_observability_scores, axis=1, result_type="expand")
    frame = pd.concat([frame, score_rows], axis=1)
    frame["failure_mode"] = frame.apply(_classify, axis=1)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    _write_report(frame, args.out_md)
    print(f"> wrote {args.out_csv}")
    print(f"> wrote {args.out_md}")
    print(frame["failure_mode"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
