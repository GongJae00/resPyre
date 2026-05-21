#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Join current MAHNOB rate metrics, candidate-bank oracle gaps, "
            "observability/failure decomposition, and readout-posterior "
            "diagnostics into one bottleneck audit. This is analysis only; it "
            "does not select a deployable model."
        )
    )
    p.add_argument("--current-run", type=Path, required=True)
    p.add_argument("--candidate-gap-trial", type=Path, required=True)
    p.add_argument("--failure-decomp-trial", type=Path, required=True)
    p.add_argument("--out-trial", type=Path, required=True)
    p.add_argument("--out-summary", type=Path, required=True)
    p.add_argument("--report-out", type=Path, required=True)
    p.add_argument("--candidate-good-mae", type=float, default=2.50)
    p.add_argument("--candidate-room-bpm", type=float, default=1.00)
    p.add_argument("--current-hard-mae", type=float, default=4.00)
    p.add_argument("--high-posterior-entropy", type=float, default=0.94)
    p.add_argument("--low-posterior-gap", type=float, default=0.15)
    return p.parse_args()


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise SystemExit(f"missing required CSV: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _first_existing(run: Path, names: List[str]) -> Path | None:
    for name in names:
        p = run / "metrics" / name
        if p.exists():
            return p
    return None


def _safe_num(row: pd.Series, key: str, default: float = float("nan")) -> float:
    try:
        val = float(row.get(key, default))
    except Exception:
        return float(default)
    return val if np.isfinite(val) else float(default)


def _classify(row: pd.Series, args: argparse.Namespace) -> str:
    current = _safe_num(row, "current_MAE")
    oracle = _safe_num(row, "oracle_MAE")
    room = _safe_num(row, "candidate_room_bpm")
    mode = str(row.get("primary_failure_mode", ""))
    entropy = _safe_num(row, "posterior_entropy_median")
    top_gap = _safe_num(row, "posterior_top_gap_median")
    aligned = _safe_num(row, "best_aligned_ccc_z")
    unbounded = _safe_num(row, "best_unbounded_ccc_z")
    abs_unbounded_lag = _safe_num(row, "abs_best_unbounded_lag_sec")
    peer = _safe_num(row, "peer_agreement_median")

    if "unphysical_lag" in mode or (
        np.isfinite(unbounded)
        and unbounded >= 0.65
        and np.isfinite(abs_unbounded_lag)
        and abs_unbounded_lag > 20.0
        and (not np.isfinite(aligned) or aligned < 0.45)
    ):
        return "reference_or_nonstationary_lag"
    if "weak_observability" in mode or (np.isfinite(peer) and peer < 0.15 and np.isfinite(aligned) and aligned < 0.35):
        return "weak_visual_observability"
    if (
        np.isfinite(current)
        and np.isfinite(oracle)
        and current >= oracle + float(args.candidate_room_bpm)
        and oracle <= float(args.candidate_good_mae)
    ):
        if np.isfinite(entropy) and entropy >= float(args.high_posterior_entropy):
            return "candidate_present_but_source_ambiguous"
        return "candidate_present_but_readout_missed"
    if np.isfinite(current) and current >= float(args.current_hard_mae) and np.isfinite(oracle) and oracle > float(args.candidate_good_mae):
        return "candidate_bank_or_observability_limited"
    if np.isfinite(entropy) and entropy >= float(args.high_posterior_entropy) and np.isfinite(top_gap) and top_gap <= float(args.low_posterior_gap):
        return "posterior_ambiguous_preserve"
    return "mixed_or_monitor"


def _join_inputs(args: argparse.Namespace) -> pd.DataFrame:
    rate_path = _first_existing(args.current_run, ["metrics_freq_domain_raw.csv"])
    if rate_path is None:
        raise SystemExit(f"current run has no metrics_freq_domain_raw.csv: {args.current_run}")
    current = _read_csv(rate_path)
    current = current.rename(
        columns={
            "MAE": "current_MAE",
            "RMSE": "current_RMSE",
            "PearsonR": "current_PearsonR",
            "Bias": "current_Bias",
            "est_bpm_avg": "current_est_bpm_avg",
            "gt_bpm_avg": "current_gt_bpm_avg",
            "method": "current_method",
        }
    )
    keep_current = [
        "video",
        "current_method",
        "current_MAE",
        "current_RMSE",
        "current_PearsonR",
        "current_Bias",
        "current_gt_bpm_avg",
        "current_est_bpm_avg",
    ]
    current = current[[c for c in keep_current if c in current.columns]].copy()

    candidate = _read_csv(args.candidate_gap_trial)
    candidate = candidate.rename(
        columns={
            "oracle_method": "candidate_oracle_method",
            "oracle_family": "candidate_oracle_family",
            "oracle_kind": "candidate_oracle_kind",
        }
    )
    candidate = candidate[
        [
            "video",
            "candidate_oracle_method",
            "candidate_oracle_family",
            "candidate_oracle_kind",
            "oracle_MAE",
            "fixed_best_method",
            "fixed_best_method_median_MAE",
            "n_candidates",
        ]
    ].copy()

    failure = _read_csv(args.failure_decomp_trial)
    failure_cols = [
        "video",
        "best_strict_method",
        "best_strict_ccc_z",
        "best_aligned_method",
        "best_aligned_ccc_z",
        "best_aligned_lag_sec",
        "best_unbounded_method",
        "best_unbounded_ccc_z",
        "best_unbounded_lag_sec",
        "best_aligned_family",
        "best_unbounded_family",
        "peer_agreement_median",
        "min_dom_diff_bpm",
        "n_bounded_recoverable",
        "n_unbounded_recoverable",
        "family_dof_aligned_ccc_z",
        "family_of_aligned_ccc_z",
        "family_p1d_aligned_ccc_z",
        "abs_best_aligned_lag_sec",
        "abs_best_unbounded_lag_sec",
        "failure_flags",
        "primary_failure_mode",
    ]
    failure = failure[[c for c in failure_cols if c in failure.columns]].copy()

    readout_path = _first_existing(args.current_run, ["readout_guard_raw.csv"])
    readout = _read_csv(readout_path, required=False) if readout_path is not None else pd.DataFrame()
    if not readout.empty:
        readout_cols = [
            "video",
            "source",
            "alpha_mean",
            "alpha_median",
            "ambiguous_alias_fraction",
            "p1d_half_rescue_fraction",
            "large_downshift_fraction",
            "well_supported_downshift_fraction",
            "guarded_downshift_fraction",
            "posterior_confidence_mean",
            "posterior_entropy_median",
            "posterior_top_gap_median",
            "posterior_macro_support_median",
        ]
        readout = readout[[c for c in readout_cols if c in readout.columns]].copy()
        readout = readout.rename(columns={"source": "readout_source"})

    out = current.merge(candidate, on="video", how="left").merge(failure, on="video", how="left")
    if not readout.empty:
        out = out.merge(readout, on="video", how="left")
    out["candidate_room_bpm"] = pd.to_numeric(out["current_MAE"], errors="coerce") - pd.to_numeric(out["oracle_MAE"], errors="coerce")
    out["candidate_can_solve"] = (
        (pd.to_numeric(out["oracle_MAE"], errors="coerce") <= float(args.candidate_good_mae))
        & (pd.to_numeric(out["candidate_room_bpm"], errors="coerce") >= float(args.candidate_room_bpm))
    )
    out["posterior_ambiguous"] = (
        (pd.to_numeric(out.get("posterior_entropy_median", np.nan), errors="coerce") >= float(args.high_posterior_entropy))
        & (pd.to_numeric(out.get("posterior_top_gap_median", np.nan), errors="coerce") <= float(args.low_posterior_gap))
    )
    out["hard_current_failure"] = pd.to_numeric(out["current_MAE"], errors="coerce") >= float(args.current_hard_mae)
    out["bottleneck_class"] = out.apply(lambda row: _classify(row, args), axis=1)
    return out


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    row: Dict[str, object] = {
        "n_trials": int(len(df)),
        "current_MAE_median": float(pd.to_numeric(df["current_MAE"], errors="coerce").median()),
        "current_MAE_mean": float(pd.to_numeric(df["current_MAE"], errors="coerce").mean()),
        "oracle_MAE_median": float(pd.to_numeric(df["oracle_MAE"], errors="coerce").median()),
        "candidate_room_bpm_median": float(pd.to_numeric(df["candidate_room_bpm"], errors="coerce").median()),
        "candidate_can_solve_frac": float(df["candidate_can_solve"].mean()),
        "hard_current_failure_frac": float(df["hard_current_failure"].mean()),
        "posterior_ambiguous_frac": float(df["posterior_ambiguous"].mean()) if "posterior_ambiguous" in df else float("nan"),
    }
    for cls, frac in df["bottleneck_class"].value_counts(normalize=True).items():
        row[f"frac_{cls}"] = float(frac)
    return pd.DataFrame([row])


def _write_report(path: Path, trial: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    class_counts = (
        trial["bottleneck_class"]
        .value_counts()
        .rename_axis("bottleneck_class")
        .reset_index(name="n")
    )
    class_counts["frac"] = class_counts["n"] / max(int(len(trial)), 1)
    hard = trial.sort_values("current_MAE", ascending=False).head(12)
    cols = [
        "video",
        "current_MAE",
        "oracle_MAE",
        "candidate_room_bpm",
        "candidate_oracle_method",
        "primary_failure_mode",
        "posterior_entropy_median",
        "posterior_top_gap_median",
        "bottleneck_class",
    ]
    hard = hard[[c for c in cols if c in hard.columns]]
    row = summary.iloc[0].to_dict()
    lines = [
        "# Current MAHNOB Bottleneck Audit",
        "",
        "This audit joins current PARH-OSSM performance with candidate-bank oracle",
        "gap, observability/failure decomposition, and readout posterior",
        "diagnostics. It is not a deployable selector.",
        "",
        f"- current run: `{args.current_run}`",
        f"- candidate gap trial CSV: `{args.candidate_gap_trial}`",
        f"- failure decomposition CSV: `{args.failure_decomp_trial}`",
        f"- trial output: `{args.out_trial}`",
        f"- summary output: `{args.out_summary}`",
        "",
        "## Summary",
        "",
        f"- current median rate MAE: `{row.get('current_MAE_median', float('nan')):.3f}` BPM",
        f"- oracle candidate-bank median MAE: `{row.get('oracle_MAE_median', float('nan')):.3f}` BPM",
        f"- median candidate room: `{row.get('candidate_room_bpm_median', float('nan')):.3f}` BPM",
        f"- candidate-present-but-missed fraction: `{100.0 * row.get('candidate_can_solve_frac', 0.0):.1f}%`",
        f"- hard current failure fraction (MAE >= {float(args.current_hard_mae):.1f}): `{100.0 * row.get('hard_current_failure_frac', 0.0):.1f}%`",
        f"- posterior ambiguous fraction: `{100.0 * row.get('posterior_ambiguous_frac', 0.0):.1f}%`",
        "",
        "## Bottleneck Classes",
        "",
        "```csv",
        class_counts.to_csv(index=False).strip(),
        "```",
        "",
        "## Worst Current Trials",
        "",
        "```csv",
        hard.to_csv(index=False).strip(),
        "```",
        "",
        "## Next Design Implication",
        "",
        "- If `candidate_present_but_source_ambiguous` dominates, the candidate bank already contains useful evidence but GT-free source validity is not sharp enough. The safe model action is confidence/abstention, not hard source replacement.",
        "- If `reference_or_nonstationary_lag` dominates, waveform strict metrics should be interpreted with a reference/timing caveat and the model patch should focus on causal phase/rate tracking, not unbounded waveform matching.",
        "- If `candidate_bank_or_observability_limited` dominates, more readout logic will not solve the trial; the observation operators or ROI/preprocessing must expose better respiratory information.",
        "- The next promotable model change must improve one class without regressing the preservation-safe baseline on the ambiguous class.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    trial = _join_inputs(args)
    summary = _summary(trial)
    args.out_trial.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    trial.to_csv(args.out_trial, index=False)
    summary.to_csv(args.out_summary, index=False)
    _write_report(args.report_out, trial, summary, args)
    print(summary.to_string(index=False))
    print(trial["bottleneck_class"].value_counts().to_string())
    print(f"Wrote {args.out_trial}")
    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
