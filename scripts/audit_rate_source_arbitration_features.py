#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_jsonish(value: object) -> Dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _finite_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _nested_get(mapping: Mapping[str, object], path: Iterable[str], default: float = np.nan) -> float:
    cur: object = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return _finite_float(cur, default)


def _dict_values(prefix: str, value: object) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    rows: Dict[str, float] = {}
    for key, raw in value.items():
        val = _finite_float(raw)
        if math.isfinite(val):
            safe_key = str(key).replace(" ", "_").replace("/", "_")
            rows[f"{prefix}_{safe_key}"] = val
    return rows


def _markdown_table(df: pd.DataFrame, *, floatfmt: str = ".3f") -> str:
    if df.empty:
        return ""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(format(float(val), floatfmt) if math.isfinite(float(val)) else "nan")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _estimate_from_pkl(path: Path) -> Optional[Dict[str, object]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    estimates = payload.get("estimates", [])
    if isinstance(estimates, Mapping):
        estimates = [{"method": k, "estimate": v} for k, v in estimates.items()]
    for item in estimates:
        if not isinstance(item, Mapping):
            continue
        est = item.get("estimate")
        if isinstance(est, Mapping):
            return dict(est)
    return None


def _feature_row(data_file: Path) -> Dict[str, object]:
    est = _estimate_from_pkl(data_file)
    if est is None:
        return {"video": data_file.stem, "data_file": str(data_file), "load_error": "no_estimate"}
    meta = _load_jsonish(est.get("meta"))
    diag = _load_jsonish(meta.get("parh_ossm_diagnostics"))
    readout = _load_jsonish(meta.get("decoupled_rate_readout_meta"))
    posterior = _load_jsonish(meta.get("rate_candidate_posterior_meta"))
    calibration = _load_jsonish(meta.get("calibration"))
    anchor_meta = _load_jsonish(meta.get("target_reliability_rate_anchor_meta"))

    row: Dict[str, object] = {
        "video": data_file.stem,
        "data_file": str(data_file),
        "output_rate_source": readout.get("source", ""),
        "anchor_family": calibration.get("anchor_family", ""),
        "anchor": calibration.get("anchor", ""),
        "rate_posterior_output_source": meta.get("rate_posterior_output_source", ""),
        "external_output_rate_hz_median": _finite_float(meta.get("external_output_rate_hz_median")),
        "external_rate_posterior_hz_median": _finite_float(meta.get("external_rate_posterior_hz_median")),
        "external_rate_anchor_hz_median": _finite_float(meta.get("external_rate_anchor_hz_median")),
        "external_output_rate_confidence_mean": _finite_float(meta.get("external_output_rate_confidence_mean")),
        "external_rate_posterior_confidence_mean": _finite_float(meta.get("external_rate_posterior_confidence_mean")),
        "external_rate_posterior_entropy_median": _finite_float(meta.get("external_rate_posterior_entropy_median")),
        "external_rate_posterior_top_gap_median": _finite_float(meta.get("external_rate_posterior_top_gap_median")),
        "external_rate_posterior_blend_active_frac": _nested_get(diag, ["external_rate_posterior_blend_active_frac"]),
        "external_output_rate_blend_active_frac": _nested_get(diag, ["external_output_rate_blend_active_frac"]),
        "state_freq_mean": _nested_get(diag, ["freq_mean"]),
        "state_freq_std": _nested_get(diag, ["freq_std"]),
        "helper_freq_mean": _nested_get(diag, ["helper_freq_mean"]),
        "helper_freq_std": _nested_get(diag, ["helper_freq_std"]),
        "rate_observability_score_mean": _nested_get(diag, ["rate_observability_score_mean"]),
        "mixture_entropy_mean": _nested_get(diag, ["mixture_entropy_mean"]),
        "prior_trust_mean": _nested_get(diag, ["prior_trust_mean"]),
        "prior_collapse_frac": _nested_get(diag, ["prior_collapse_frac"]),
        "q_obs_mean": _nested_get(diag, ["q_obs_mean"]),
        "q_dyn_mean": _nested_get(diag, ["q_dyn_mean"]),
        "q_osc_mean": _nested_get(diag, ["q_osc_mean"]),
        "helper_support_mean": _nested_get(diag, ["helper_support_mean"]),
        "helper_mismatch_mean": _nested_get(diag, ["helper_mismatch_mean"]),
        "nis_mean": _nested_get(diag, ["nis_mean"]),
        "lambda_mean": _nested_get(diag, ["lambda_mean"]),
        "energy_h1": _nested_get(diag, ["energy_h1"]),
        "energy_h2": _nested_get(diag, ["energy_h2"]),
        "energy_baseline": _nested_get(diag, ["energy_baseline"]),
        "energy_residual": _nested_get(diag, ["energy_residual"]),
        "readout_confidence_mean": _nested_get(readout, ["confidence_mean"]),
        "readout_support_group_count_median": _nested_get(readout, ["support_group_count_median"]),
        "readout_alias_risk_mean": _nested_get(readout, ["alias_risk_mean"]),
        "readout_h1_role_support_mean": _nested_get(readout, ["h1_role_support_mean"]),
        "readout_abstain_pressure_mean": _nested_get(readout, ["abstain_pressure_mean"]),
        "posterior_confidence_mean": _nested_get(posterior, ["confidence_mean"]),
        "posterior_entropy_median": _nested_get(posterior, ["posterior_entropy_median"]),
        "posterior_top_gap_median": _nested_get(posterior, ["posterior_top_gap_median"]),
        "posterior_macro_support_median": _nested_get(posterior, ["macro_support_median"]),
        "posterior_direct_macro_support_median": _nested_get(posterior, ["direct_macro_support_median"]),
        "posterior_motion_direct_support_median": _nested_get(posterior, ["motion_direct_support_median"]),
        "posterior_alias_risk_median": _nested_get(posterior, ["alias_risk_median"]),
        "posterior_h1_role_support_median": _nested_get(posterior, ["h1_role_support_median"]),
        "posterior_abstain_pressure_median": _nested_get(posterior, ["abstain_pressure_median"]),
        "anchor_confidence_mean": _nested_get(anchor_meta, ["confidence_mean"]),
        "anchor_support_group_count_median": _nested_get(anchor_meta, ["support_group_count_median"]),
        "anchor_phase_support_median": _nested_get(anchor_meta, ["phase_support_median"]),
        "anchor_temporal_support_median": _nested_get(anchor_meta, ["temporal_support_median"]),
    }
    row.update(_dict_values("family_weight", calibration.get("family_weights")))
    row.update(_dict_values("family_score", calibration.get("family_raw_scores")))
    return row


def _write_report(df: pd.DataFrame, out_md: Path) -> None:
    lines: List[str] = ["# Rate-Source Evidence Feature Audit", ""]
    lines.append(
        "This report is diagnostic-only. It measures whether target-computable "
        "evidence could explain readout failures; it does not define a promoted "
        "source selector."
    )
    lines.append("")
    lines.append(f"- trials: `{len(df)}`")
    if "best_source" in df.columns:
        counts = df["best_source"].value_counts(dropna=False)
        lines.append("- best-source counts:")
        for name, count in counts.items():
            lines.append(f"  - `{name}`: `{int(count)}`")
    if "final_minus_best" in df.columns:
        hard = df.sort_values("final_minus_best", ascending=False).head(8)
        lines.extend(["", "## Largest Available Room", ""])
        cols = [
            "video",
            "best_source",
            "final_minus_best",
            "final_track_hz",
            "external_rate_posterior_mean_t",
            "native_smoothed_track_hz",
            "state_freq_t",
            "posterior_entropy_median",
            "posterior_top_gap_median",
            "readout_alias_risk_mean",
        ]
        cols = [c for c in cols if c in hard.columns]
        lines.append(_markdown_table(hard[cols], floatfmt=".3f"))
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [
        c
        for c in numeric
        if c
        not in {
            "final_track_hz",
            "native_smoothed_track_hz",
            "external_output_rate_t",
            "external_rate_posterior_mean_t",
            "state_freq_t",
            "best_mae",
            "final_minus_best",
            "posterior_mean_minus_final",
        }
    ]
    if "best_source" in df.columns and feature_cols:
        grouped = df.groupby("best_source")[feature_cols].median(numeric_only=True)
        keep = [c for c in feature_cols if grouped[c].max() - grouped[c].min() > 1e-6]
        keep = keep[:20]
        if keep:
            lines.extend(["", "## Median Target-Computable Features by Best Source", ""])
            lines.append(_markdown_table(grouped[keep].reset_index(), floatfmt=".3f"))
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This audit may use per-trial `best_source` labels from the rate-source",
            "decomposition CSV, so it is diagnostic. It is used to design a future",
            "GT-free arbiter; it is not itself a promoted release readout.",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract target-computable PARH/meta features and join them with "
            "rate-source decomposition labels for rate-source arbitration design."
        )
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--pivot-csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    args = parser.parse_args()

    rows = [_feature_row(path) for path in sorted(args.data_dir.glob("*.pkl"))]
    if not rows:
        raise SystemExit(f"no PKL files found in {args.data_dir}")
    features = pd.DataFrame(rows)
    pivot = pd.read_csv(args.pivot_csv)
    merged = pivot.merge(features, on="video", how="left", suffixes=("", "_feature"))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    _write_report(merged, args.out_md)
    print(f"> wrote {args.out_csv}")
    print(f"> wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
