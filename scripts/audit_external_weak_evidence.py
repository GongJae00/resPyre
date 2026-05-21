#!/usr/bin/env python3
"""Audit weak external RR evidence for the final paper execution path.

V4V and SCAMPS are intentionally not pooled with COHFACE/MAHNOB-HCI real
waveform benchmarks. This script makes that boundary executable: it checks the
external manifests, writes a paper-facing supplementary table, and produces a
compact scope figure that documents exactly what evidence each dataset can and
cannot support.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "analysis"
TABLE_DIR = ROOT / "paper" / "tables_ready"
FIGURE_DIR = ROOT / "paper" / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v4v-manifest", type=Path, default=ANALYSIS_DIR / "v4v_rr_rate_manifest.csv")
    p.add_argument("--scamps-manifest", type=Path, default=ANALYSIS_DIR / "scamps_rr_synthetic_manifest.csv")
    p.add_argument("--out-csv", type=Path, default=ANALYSIS_DIR / "external_weak_evidence_audit.csv")
    p.add_argument("--out-md", type=Path, default=ANALYSIS_DIR / "external_weak_evidence_audit.md")
    p.add_argument("--table-out", type=Path, default=TABLE_DIR / "S_T_external_weak_evidence_audit.csv")
    p.add_argument("--figure-out", type=Path, default=FIGURE_DIR / "S_F_external_weak_evidence_summary.pdf")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write missing-manifest rows instead of failing when external manifests are absent.",
    )
    return p.parse_args()


def _read_csv(path: Path, *, allow_missing: bool) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if allow_missing:
        return pd.DataFrame()
    raise FileNotFoundError(f"Required external manifest is missing: {path}")


def _safe_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.median()) if not vals.empty else math.nan


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if not vals.empty else math.nan


def _mode_text(series: pd.Series) -> str:
    vals = series.dropna().astype(str)
    if vals.empty:
        return ""
    return str(vals.mode().iloc[0])


def _split_counts(df: pd.DataFrame) -> str:
    if "split" not in df.columns or df.empty:
        return ""
    counts = df["split"].fillna("missing").astype(str).value_counts().sort_index()
    return "; ".join(f"{k}:{int(v)}" for k, v in counts.items())


def _v4v_row(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "dataset": "V4V",
            "evidence_role": "external_real_rate_only",
            "full_inclusion_stage": "missing_manifest",
            "n_units": 0,
            "valid_path_fraction": math.nan,
            "label_completeness": math.nan,
            "label_summary": "manifest missing",
            "allowed_metrics": "RR rate only after a dedicated external adapter/readout is available",
            "prohibited_metrics": "waveform CCC/DTW; cycle morphology; real waveform reconstruction claims",
            "paper_use": "blocked until manifest is regenerated",
            "claim_boundary": "No waveform claim can be made from V4V.",
        }

    valid_path_fraction = _safe_mean(df.get("video_exists", pd.Series(dtype=float)).astype(float))
    rr_values = pd.to_numeric(df.get("n_rr_values", pd.Series(dtype=float)), errors="coerce")
    label_completeness = float((rr_values > 0).mean()) if len(rr_values) else math.nan
    subjects = int(df["subject"].nunique()) if "subject" in df.columns else math.nan
    label_summary = (
        f"trials={len(df)}, subjects={subjects}, splits={_split_counts(df)}, "
        f"median RR={_safe_median(df['rr_median_bpm']):.2f} bpm, "
        f"median RR-IQR={_safe_median(df['rr_iqr_bpm']):.2f} bpm, "
        f"median label samples={_safe_median(rr_values):.0f}"
    )
    return {
        "dataset": "V4V",
        "evidence_role": "external_real_rate_only",
        "full_inclusion_stage": "mandatory_manifest_and_rate_scope_audit",
        "n_units": int(len(df)),
        "valid_path_fraction": valid_path_fraction,
        "label_completeness": label_completeness,
        "label_summary": label_summary,
        "allowed_metrics": "RR MAE/RMSE/Pearson only if a V4V rate-readout adapter is explicitly run",
        "prohibited_metrics": "waveform CCC/DTW; strict waveform; cycle morphology",
        "paper_use": "supplementary external-rate evidence; not pooled with COHFACE/MAHNOB",
        "claim_boundary": "Supports only timing/rate generalization, never morphology.",
    }


def _scamps_row(df: pd.DataFrame) -> dict[str, object]:
    if df.empty:
        return {
            "dataset": "SCAMPS",
            "evidence_role": "synthetic_controlled_diagnostic",
            "full_inclusion_stage": "missing_manifest",
            "n_units": 0,
            "valid_path_fraction": math.nan,
            "label_completeness": math.nan,
            "label_summary": "manifest missing",
            "allowed_metrics": "synthetic controlled sanity checks only",
            "prohibited_metrics": "real-data generalization performance claims",
            "paper_use": "blocked until manifest is regenerated",
            "claim_boundary": "Synthetic evidence cannot replace real benchmark evidence.",
        }

    mat_exists = df.get("mat_exists", pd.Series(dtype=float)).astype(float)
    has_d_br = df.get("has_d_br", pd.Series(dtype=float)).astype(float)
    has_raw_frames = df.get("has_raw_frames", pd.Series(dtype=float)).astype(float)
    read_errors = df.get("read_error", pd.Series(dtype=str)).fillna("").astype(str)
    n_error = int((read_errors.str.len() > 0).sum())
    label_summary = (
        f"trials={len(df)}, valid MAT={int(mat_exists.sum())}, "
        f"d_br coverage={float(has_d_br.mean()):.3f}, "
        f"raw-frame coverage={float(has_raw_frames.mean()):.3f}, "
        f"median frames={_safe_median(df.get('n_frames', pd.Series(dtype=float))):.0f}, "
        f"common d_br shape={_mode_text(df.get('d_br_shape', pd.Series(dtype=str)))}, "
        f"read errors={n_error}"
    )
    return {
        "dataset": "SCAMPS",
        "evidence_role": "synthetic_controlled_diagnostic",
        "full_inclusion_stage": "mandatory_manifest_and_synthetic_scope_audit",
        "n_units": int(len(df)),
        "valid_path_fraction": float(mat_exists.mean()) if len(mat_exists) else math.nan,
        "label_completeness": float(has_d_br.mean()) if len(has_d_br) else math.nan,
        "label_summary": label_summary,
        "allowed_metrics": "controlled synthetic diagnostic/sanity checks; optional mechanism ablation",
        "prohibited_metrics": "real-data waveform/rate benchmark performance claims",
        "paper_use": "supplementary mechanism-control evidence; not pooled with real data",
        "claim_boundary": "Supports mechanism sanity, not real-world robustness by itself.",
    }


def build_audit(v4v: pd.DataFrame, scamps: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dataset",
        "evidence_role",
        "full_inclusion_stage",
        "n_units",
        "valid_path_fraction",
        "label_completeness",
        "label_summary",
        "allowed_metrics",
        "prohibited_metrics",
        "paper_use",
        "claim_boundary",
    ]
    return pd.DataFrame([_v4v_row(v4v), _scamps_row(scamps)], columns=columns)


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# External Weak Evidence Audit",
        "",
        "This audit is part of the final paper execution path. It prevents external",
        "datasets with weaker labels from being silently treated as full real waveform",
        "benchmarks.",
        "",
        "## Boundary Rule",
        "",
        "- COHFACE and MAHNOB-HCI remain the only current real waveform/rate benchmarks.",
        "- V4V is included only as external RR-rate evidence.",
        "- SCAMPS is included only as synthetic controlled diagnostic evidence.",
        "- Neither V4V nor SCAMPS may be pooled into main real-data waveform tables.",
        "",
        "## Audit Table",
        "",
        "| dataset | role | units | valid paths | label completeness | allowed use | prohibited use |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in df.iterrows():
        valid = row["valid_path_fraction"]
        comp = row["label_completeness"]
        valid_s = "" if pd.isna(valid) else f"{float(valid):.3f}"
        comp_s = "" if pd.isna(comp) else f"{float(comp):.3f}"
        lines.append(
            f"| {row['dataset']} | {row['evidence_role']} | {int(row['n_units'])} | "
            f"{valid_s} | {comp_s} | {row['allowed_metrics']} | {row['prohibited_metrics']} |"
        )
    lines.extend(["", "## Dataset Notes", ""])
    for _, row in df.iterrows():
        lines.append(f"### {row['dataset']}")
        lines.append("")
        lines.append(f"- Full inclusion stage: `{row['full_inclusion_stage']}`")
        lines.append(f"- Label summary: {row['label_summary']}")
        lines.append(f"- Paper use: {row['paper_use']}")
        lines.append(f"- Claim boundary: {row['claim_boundary']}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_audit(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = df.copy()
    plot_df["valid_path_fraction"] = pd.to_numeric(plot_df["valid_path_fraction"], errors="coerce")
    plot_df["label_completeness"] = pd.to_numeric(plot_df["label_completeness"], errors="coerce")
    datasets = plot_df["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.bar(x - width / 2, plot_df["valid_path_fraction"].fillna(0), width, label="valid source paths", color="#2f6f8f")
    ax.bar(x + width / 2, plot_df["label_completeness"].fillna(0), width, label="usable label coverage", color="#d18b2c")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Fraction")
    ax.set_title("External weak-evidence inclusion audit")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    for i, row in plot_df.iterrows():
        role = str(row["evidence_role"]).replace("_", " ")
        ax.text(i, 1.035, role, ha="center", va="bottom", fontsize=8, color="#39434d")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _has_missing_critical_rows(rows: Iterable[dict[str, object]]) -> bool:
    for row in rows:
        if str(row.get("full_inclusion_stage", "")).startswith("missing"):
            return True
    return False


def main() -> int:
    args = parse_args()
    v4v = _read_csv(args.v4v_manifest, allow_missing=bool(args.allow_missing))
    scamps = _read_csv(args.scamps_manifest, allow_missing=bool(args.allow_missing))
    audit = build_audit(v4v, scamps)

    for path in (args.out_csv, args.out_md, args.table_out, args.figure_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out_csv, index=False, float_format="%.6f")
    audit.to_csv(args.table_out, index=False, float_format="%.6f")
    write_markdown(audit, args.out_md)
    plot_audit(audit, args.figure_out)

    print("Wrote:")
    for path in (args.out_csv, args.out_md, args.table_out, args.figure_out):
        print(f"  {path}")

    if _has_missing_critical_rows(audit.to_dict("records")) and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
