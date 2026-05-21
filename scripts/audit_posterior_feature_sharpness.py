#!/usr/bin/env python3
"""Audit whether posterior/readout diagnostics actually separate failures.

This script is analysis-only. It may join evaluated MAE with target-GT-derived
audit labels, but it does not select or tune a deployed model. The purpose is
to identify whether target-computable features are sharp enough to support the
adaptive observation law.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_FEATURES = (
    "alpha_mean",
    "alpha_median",
    "ambiguous_alias_fraction",
    "unresolved_p1d_alias_fraction",
    "p1d_half_rescue_fraction",
    "large_downshift_fraction",
    "well_supported_downshift_fraction",
    "guarded_downshift_fraction",
    "high_base_conflict_fraction",
    "posterior_confidence_mean",
    "posterior_entropy_median",
    "posterior_top_gap_median",
    "posterior_macro_support_median",
    "posterior_direct_macro_support_median",
    "posterior_motion_direct_support_median",
    "posterior_alias_risk_median",
    "posterior_independent_timing_support_median",
    "posterior_bridge_timing_preservation_median",
    "posterior_morphology_alias_pressure_median",
    "posterior_h1_role_support_median",
    "posterior_abstain_pressure_median",
    "weak_direct_macro_fraction",
    "alias_risk_mean",
    "h1_role_support_mean",
    "morphology_alias_pressure_mean",
    "abstain_pressure_mean",
    "specific_posterior_correction_fraction",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Evaluated run directory with metrics/.")
    p.add_argument("--name", default="run", help="Label used in the report.")
    p.add_argument("--candidate-gap-trial", type=Path, help="Optional candidate oracle gap CSV.")
    p.add_argument("--bottleneck-trial", type=Path, help="Optional bottleneck audit trial CSV.")
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--report-out", type=Path, required=True)
    return p.parse_args()


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_run(run: Path) -> pd.DataFrame:
    metrics = _read_csv(run / "metrics" / "metrics_freq_domain_raw.csv")
    if "video" not in metrics.columns or "MAE" not in metrics.columns:
        raise ValueError(f"metrics_freq_domain_raw.csv missing video/MAE columns: {run}")
    keep = ["video", "method", "MAE", "RMSE", "PearsonR", "Bias", "gt_bpm_avg", "est_bpm_avg"]
    out = metrics[[c for c in keep if c in metrics.columns]].copy()
    out = out.groupby("video", as_index=False).median(numeric_only=True)
    readout = _read_csv(run / "metrics" / "readout_guard_raw.csv", required=False)
    if not readout.empty and "video" in readout.columns:
        readout_keep = ["video"] + [c for c in DEFAULT_FEATURES if c in readout.columns]
        readout = readout[readout_keep].copy()
        readout = readout.groupby("video", as_index=False).median(numeric_only=True)
        out = out.merge(readout, on="video", how="left")
    return out


def _attach_optional(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.candidate_gap_trial and args.candidate_gap_trial.exists():
        cand = pd.read_csv(args.candidate_gap_trial)
        rename = {
            "oracle_method": "candidate_oracle_method",
            "oracle_family": "candidate_oracle_family",
            "oracle_kind": "candidate_oracle_kind",
        }
        cand = cand.rename(columns=rename)
        keep = [
            "video",
            "oracle_MAE",
            "candidate_oracle_method",
            "candidate_oracle_family",
            "candidate_oracle_kind",
            "fixed_best_method",
            "fixed_best_method_median_MAE",
        ]
        cand = cand[[c for c in keep if c in cand.columns]].copy()
        out = out.merge(cand, on="video", how="left")
    if args.bottleneck_trial and args.bottleneck_trial.exists():
        bot = pd.read_csv(args.bottleneck_trial)
        keep = [
            "video",
            "candidate_room_bpm",
            "candidate_can_solve",
            "posterior_ambiguous",
            "hard_current_failure",
            "bottleneck_class",
            "primary_failure_mode",
        ]
        bot = bot[[c for c in keep if c in bot.columns]].copy()
        out = out.merge(bot, on="video", how="left", suffixes=("", "_audit"))
    if "oracle_MAE" in out.columns:
        out["candidate_room_bpm"] = pd.to_numeric(out["MAE"], errors="coerce") - pd.to_numeric(out["oracle_MAE"], errors="coerce")
    return out


def _safe_corr(a: Iterable[float], b: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 3:
        return float("nan")
    if float(np.nanstd(x[ok])) <= 1e-12 or float(np.nanstd(y[ok])) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = [c for c in ("MAE", "candidate_room_bpm") if c in df.columns]
    for feature in [c for c in DEFAULT_FEATURES if c in df.columns]:
        vals = pd.to_numeric(df[feature], errors="coerce")
        finite = vals[np.isfinite(vals)]
        row = {
            "feature": feature,
            "n": int(finite.size),
            "median": float(finite.median()) if finite.size else float("nan"),
            "std": float(finite.std(ddof=0)) if finite.size else float("nan"),
            "min": float(finite.min()) if finite.size else float("nan"),
            "max": float(finite.max()) if finite.size else float("nan"),
            "saturated_fraction": float(((finite <= 0.02) | (finite >= 0.98)).mean()) if finite.size else float("nan"),
            "low_variance": bool(finite.size >= 3 and float(finite.std(ddof=0)) < 0.03),
        }
        for target in targets:
            row[f"corr_{target}"] = _safe_corr(vals, df[target])
        rows.append(row)
    return pd.DataFrame(rows)


def _group_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "bottleneck_class" not in df.columns:
        return pd.DataFrame()
    features = [c for c in DEFAULT_FEATURES if c in df.columns]
    agg = {"video": "count", "MAE": "median"}
    for feature in features:
        agg[feature] = "median"
    out = df.groupby("bottleneck_class", dropna=False).agg(agg).rename(columns={"video": "n"}).reset_index()
    return out.sort_values("n", ascending=False)


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    sub = df.head(max_rows).copy()
    cols = list(sub.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in sub.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.3f}" if np.isfinite(float(val)) else "nan")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(path: Path, trial: pd.DataFrame, features: pd.DataFrame, groups: pd.DataFrame, args: argparse.Namespace) -> None:
    low_var = features[features["low_variance"] == True].copy()
    high_corr = features.reindex(features["corr_MAE"].abs().sort_values(ascending=False).index) if "corr_MAE" in features else pd.DataFrame()
    lines = [
        f"# Posterior Feature Sharpness Audit - {args.name}",
        "",
        f"- run: `{args.run}`",
        f"- trials: `{len(trial)}`",
        f"- median MAE: `{float(pd.to_numeric(trial['MAE'], errors='coerce').median()):.3f}` BPM",
        "",
        "## Interpretation",
        "",
        "This is an audit, not a selector. Target GT is used only to measure whether target-computable posterior features are sharp enough to explain failures.",
        "",
        "## Low-Variance / Saturated Features",
        "",
        _markdown_table(low_var[["feature", "n", "median", "std", "min", "max", "saturated_fraction"]]),
        "",
        "## Feature Correlation With MAE",
        "",
        _markdown_table(high_corr[["feature", "n", "median", "std", "corr_MAE"]]),
        "",
        "## Bottleneck-Class Medians",
        "",
        _markdown_table(groups),
        "",
        "## Decision Rule For Next Patch",
        "",
        "- Do not add a stricter guard around a saturated feature.",
        "- Promote only reliability features that vary across failure modes and have interpretable directionality.",
        "- If macro support is saturated, redefine support as independent direct timing evidence rather than count of loosely compatible families.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    trial = _attach_optional(_load_run(args.run), args)
    features = _feature_summary(trial)
    groups = _group_summary(trial)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    trial.to_csv(args.out_csv, index=False)
    feature_csv = args.out_csv.with_name(args.out_csv.stem + "_features.csv")
    group_csv = args.out_csv.with_name(args.out_csv.stem + "_groups.csv")
    features.to_csv(feature_csv, index=False)
    groups.to_csv(group_csv, index=False)
    _write_report(args.report_out, trial, features, groups, args)
    print(f"wrote {args.out_csv}")
    print(f"wrote {feature_csv}")
    print(f"wrote {group_csv}")
    print(f"wrote {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
