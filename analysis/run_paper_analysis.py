#!/usr/bin/env python3
"""Unified analysis pipeline for the QROBF paper.

PRE-RUN steps (--pre-run flag, run BEFORE main.py):
  0a. prepare_cohface_obs  — cache OF/DoF/Profile1D observation signals
  0b. run_noise_analysis   — characterise signal noise properties

POST-RUN steps (default, run AFTER main.py has completed):
  1. plot_performance_summary  — overall perf bar/scatter plots
  2. Innovation EDA            — frame-log diagnostics (NIS, g_t, fail rates)
  3. Instability boundary      — P(QROBF better) vs signal quality decile
  4. Quality stratification    — SNR-based tier assignment + eval metric merge
  5. Quality-stratified figs   — 7 paper figures (PDF+PNG)
  6. Paper directory assembly  — tables/, figures/, PAPER_INDEX.md

Usage:
    # Pre-run (once, before main.py):
    python analysis/run_paper_analysis.py --config configs/cohface_robust_ossm.json --pre-run

    # Post-run (after main.py):
    python analysis/run_paper_analysis.py --run-dir results/cohface_robust_ossm
    python analysis/run_paper_analysis.py --config configs/cohface_robust_ossm.json

All paper outputs go to:
    <run-dir>/paper/
        tables/          ← CSV tables (Table 1 .. N)
        figures/         ← PDF+PNG figures (Fig 1 .. N)
        PAPER_INDEX.md   ← master index of all Tables and Figures
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─── project root ────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# ─── local imports (deferred to avoid heavy GPU startup at import time) ───────


def _step_banner(name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# PRE-RUN steps (run before main.py)
# ─────────────────────────────────────────────────────────────────────────────

def step_prepare_obs(config_path: str) -> bool:
    """Cache OF/DoF/Profile1D observation signals for all COHFACE trials."""
    _step_banner("Step 0a · Prepare COHFACE Observations")
    try:
        import importlib.util, runpy
        # prepare_cohface_obs uses no importable entry point; run via runpy
        script = str(_REPO / "analysis" / "prepare_cohface_obs.py")
        # Temporarily patch sys.argv
        _orig_argv = sys.argv[:]
        sys.argv = [script]
        try:
            runpy.run_path(script, run_name="__main__")
        finally:
            sys.argv = _orig_argv
        return True
    except SystemExit:
        return True  # script calls sys.exit(0) on success
    except Exception as exc:
        print(f"[prepare_obs] Warning: {exc}")
        return False


def step_noise_analysis(config_path: str) -> bool:
    """Characterise signal noise properties across COHFACE trials."""
    _step_banner("Step 0b · Noise Analysis")
    try:
        import runpy
        script = str(_REPO / "analysis" / "run_noise_analysis.py")
        _orig_argv = sys.argv[:]
        sys.argv = [script, "--config", config_path, "--num-samples", "-1",
                    "--output", str(_REPO / "analysis" / "noise_properties")]
        try:
            runpy.run_path(script, run_name="__main__")
        finally:
            sys.argv = _orig_argv
        return True
    except SystemExit:
        return True
    except Exception as exc:
        print(f"[noise_analysis] Warning: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# POST-RUN steps (run after main.py)
# ─────────────────────────────────────────────────────────────────────────────

def step_performance_summary(run_dir: str) -> bool:
    """Generate overall performance bar/scatter summary plots."""
    _step_banner("Step 1 · Performance Summary")
    try:
        from analysis.plot_performance_summary import main as _perf_main
        _orig_argv = sys.argv[:]
        sys.argv = ["plot_performance_summary.py", "--run-dir", run_dir]
        try:
            _perf_main()
        finally:
            sys.argv = _orig_argv
        return True
    except Exception as exc:
        print(f"[perf_summary] Warning: {exc}")
        return False


def step_innovation_eda(run_dir: str) -> bool:
    """Run frame-log innovation EDA.  Returns True if succeeded."""
    _step_banner("Step 2 · Innovation EDA")
    try:
        from analysis.run_innovation_eda import run_innovation_eda
        run_innovation_eda(
            results_dir=str(Path(run_dir).parent),
            run_label=Path(run_dir).name,
            allow_missing=True,
            strict=False,
        )
        return True
    except Exception as exc:
        print(f"[EDA] Warning: {exc}")
        return False


def step_instability_boundary(run_dir: str) -> bool:
    """Run instability-benefit boundary analysis."""
    _step_banner("Step 3 · Instability Boundary")
    try:
        # Import internal helpers directly (main() uses argparse)
        from analysis.plot_instability_boundary import (
            _prepare_metrics, _prepare_quality, _build_pair_table,
            _decile_stats, _plot_probability_curve,
            _plot_delta_deciles, _plot_regime_map,
        )
        out_dir = os.path.join(run_dir, "plots", "boundary")
        os.makedirs(out_dir, exist_ok=True)

        metrics_df = _prepare_metrics(run_dir)
        quality_df = _prepare_quality(run_dir)
        pair_df    = _build_pair_table(metrics_df, quality_df)

        if pair_df.empty:
            print("[boundary] No robust-vs-kfstd pairs — skipping plots.")
            return False

        dec_df = _decile_stats(pair_df, deciles=10)
        pair_df.to_csv(os.path.join(out_dir, "boundary_pairs.csv"), index=False)
        dec_df.to_csv(os.path.join(out_dir, "boundary_deciles.csv"), index=False)

        if not dec_df.empty:
            _plot_probability_curve(dec_df, os.path.join(out_dir, "boundary_prob_curve.png"))
            _plot_delta_deciles(dec_df, os.path.join(out_dir, "boundary_delta_deciles.png"))
            _plot_regime_map(pair_df, os.path.join(out_dir, "boundary_regime_map.png"))

        threshold_u = np.nan
        if not dec_df.empty and "p_better_dual" in dec_df.columns:
            mask = (dec_df["p_better_dual"] >= 0.5) & (dec_df["n_pairs"] >= 3)
            if mask.any():
                threshold_u = float(dec_df.loc[mask, "u_min"].iloc[0])

        report = {
            "schema": "instability_boundary.v1",
            "status": "ok",
            "n_pairs": int(len(pair_df)),
            "threshold_u_p_better_dual_ge_0p5": threshold_u,
            "mean_delta_freq_mae": float(np.nanmean(pair_df["delta_freq_mae"])),
            "mean_p_better_dual": float(np.nanmean(pair_df["better_dual"])),
        }
        with open(os.path.join(out_dir, "boundary_report.json"), "w") as fp:
            json.dump(report, fp, indent=2)
        print(f"[boundary] Saved to {out_dir}")
        return True
    except Exception as exc:
        print(f"[boundary] Warning: {exc}")
        return False


def step_quality_stratification(run_dir: str, out_dir: str) -> Optional[pd.DataFrame]:
    """Compute SNR-based tier assignments + merge evaluation pipeline metrics."""
    _step_banner("Step 4 · Quality Stratification")
    try:
        from analysis.quality_stratification import run_stratification, merge_eval_metrics

        results_dir = str(Path(run_dir).parent)
        run_label   = Path(run_dir).name
        qs_out      = os.path.join(out_dir, "quality_stratification")

        df = run_stratification(results_dir, run_label, qs_out)

        # Merge evaluation pipeline per-trial MAE if available
        eval_csv = os.path.join(run_dir, "metrics", "metrics_freq_domain_raw.csv")
        if os.path.exists(eval_csv):
            df = merge_eval_metrics(df, eval_csv)
            csv_path = Path(qs_out) / "trial_stratification.csv"
            df.to_csv(csv_path, index=False)
            print(f"[qs] Eval metrics merged → {csv_path}")
        else:
            print(f"[qs] Eval CSV not found ({eval_csv}); using proxy metrics only.")

        return df
    except Exception as exc:
        print(f"[qs] Error: {exc}")
        import traceback; traceback.print_exc()
        return None


def step_quality_figures(df: pd.DataFrame, out_dir: str) -> list[Path]:
    """Generate 7 paper-quality figures from stratification DataFrame."""
    _step_banner("Step 5 · Quality-Stratified Figures")
    try:
        from analysis.plot_quality_stratification import (
            run_plots, TIER_LABELS,
        )
        fig_out = os.path.join(out_dir, "quality_stratification")
        # Write a temporary CSV that run_plots() will read
        tmp_csv = os.path.join(fig_out, "trial_stratification.csv")
        if not os.path.exists(tmp_csv):
            df.to_csv(tmp_csv, index=False)
        run_plots(tmp_csv, fig_out)
        figs = sorted(Path(fig_out).glob("fig*.pdf"))
        print(f"[figs] {len(figs)} figures generated in {fig_out}")
        return figs
    except Exception as exc:
        print(f"[figs] Error: {exc}")
        import traceback; traceback.print_exc()
        return []


def step_build_paper_dir(run_dir: str, df: Optional[pd.DataFrame]) -> Path:
    """Assemble paper/ directory with Tables and Figures."""
    _step_banner("Step 6 · Paper Directory")
    paper_dir = Path(run_dir) / "paper"
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Tables ────────────────────────────────────────────────────────────────
    # Table 1: overall frequency-domain performance summary
    freq_csv = Path(run_dir) / "metrics" / "metrics_freq_domain_summary.csv"
    if freq_csv.exists():
        _build_table1(freq_csv, tables_dir)

    # Table 2: per-tier performance (from quality stratification)
    if df is not None:
        _build_table2(df, tables_dir)

    # Table 3: filter diagnostics summary
    diag_csv = Path(run_dir) / "metrics" / "metrics_filter_diagnostics_summary.csv"
    if diag_csv.exists():
        shutil.copy(diag_csv, tables_dir / "table3_filter_diagnostics.csv")
        print(f"[paper] Table 3 → {tables_dir / 'table3_filter_diagnostics.csv'}")

    # ── Figures ───────────────────────────────────────────────────────────────
    qs_dir = Path(run_dir) / "plots" / "quality_stratification"
    fig_map = {
        "fig1_snr_distribution.pdf":    "fig1_snr_distribution.pdf",
        "fig2_tier_bar_OF.pdf":         "fig2_tier_bar_OF.pdf",
        "fig3_tier_bar_all_families.pdf":"fig3_tier_bar_all_families.pdf",
        "fig4_relative_improvement.pdf": "fig4_relative_improvement.pdf",
        "fig5_boxplots.pdf":             "fig5_boxplots.pdf",
        "fig6_scatter_snr_vs_mae.pdf":   "fig6_scatter_snr_vs_mae.pdf",
        "fig7_improvement_heatmap.pdf":  "fig7_improvement_heatmap.pdf",
    }
    for src_name, dst_name in fig_map.items():
        src = qs_dir / src_name
        if src.exists():
            shutil.copy(src, figures_dir / dst_name)

    # Also copy boundary plots
    boundary_dir = Path(run_dir) / "plots" / "boundary"
    for png in boundary_dir.glob("*.png"):
        shutil.copy(png, figures_dir / png.name)

    # Copy overall performance summary
    perf_png = Path(run_dir) / "plots" / "paper_performance_summary.png"
    if perf_png.exists():
        shutil.copy(perf_png, figures_dir / "fig0_overall_performance.png")

    n_tables  = len(list(tables_dir.glob("*.csv")))
    n_figures = len(list(figures_dir.iterdir()))
    print(f"[paper] {n_tables} tables, {n_figures} figures → {paper_dir}")
    return paper_dir


def _build_table1(freq_csv: Path, tables_dir: Path) -> None:
    """Table 1: Overall freq-domain performance (MAE, BPM)."""
    df = pd.read_csv(freq_csv)
    # Keep only MAE columns
    cols = ["method"] + [c for c in df.columns if "mae" in c.lower() or "MAE" in c]
    if "Method" in df.columns:
        df = df.rename(columns={"Method": "method"})
    out = df[[c for c in cols if c in df.columns]]
    out.to_csv(tables_dir / "table1_freq_domain_mae.csv", index=False)
    print(f"[paper] Table 1 → {tables_dir / 'table1_freq_domain_mae.csv'}")


def _build_table2(df: pd.DataFrame, tables_dir: Path) -> None:
    """Table 2: Per-tier MAE for each method family (base / kfstd / QROBF)."""
    from analysis.quality_stratification import METHOD_FAMILIES, TIER_LABELS
    from analysis.plot_quality_stratification import _resolve_families, BASE_COLS_BASE

    FAMILIES, BASE_COLS = _resolve_families(df)
    rows = []
    for tier in TIER_LABELS:
        t = df[df["tier"] == tier]
        n = len(t)
        row: dict = {"tier": tier, "n_trials": n}
        for label, kf_col, qr_col, *_ in FAMILIES:
            base_col = BASE_COLS.get(label, "")
            base_vals = pd.to_numeric(t[base_col], errors="coerce").dropna() if base_col and base_col in t.columns else pd.Series([], dtype=float)
            kf_vals = pd.to_numeric(t[kf_col], errors="coerce").dropna()
            qr_vals = pd.to_numeric(t[qr_col], errors="coerce").dropna()
            row[f"{label}_base_mae"]   = round(base_vals.mean(), 3) if len(base_vals) else float("nan")
            row[f"{label}_kfstd_mae"]  = round(kf_vals.mean(), 3) if len(kf_vals) else float("nan")
            row[f"{label}_qrobf_mae"]  = round(qr_vals.mean(), 3) if len(qr_vals) else float("nan")
            if len(kf_vals) and len(qr_vals) and kf_vals.mean() > 0:
                pct = (qr_vals.mean() - kf_vals.mean()) / kf_vals.mean() * 100.0
                row[f"{label}_pct_change"] = round(pct, 1)
            else:
                row[f"{label}_pct_change"] = float("nan")
        rows.append(row)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(tables_dir / "table2_per_tier_mae.csv", index=False)
    print(f"[paper] Table 2 → {tables_dir / 'table2_per_tier_mae.csv'}")


def step_write_paper_index(paper_dir: Path, run_dir: str, df: Optional[pd.DataFrame]) -> None:
    """Write PAPER_INDEX.md — master index of all Tables and Figures."""
    _step_banner("Step 7 · Paper Index")

    tables  = sorted((paper_dir / "tables").glob("*.csv"))
    figures = sorted((paper_dir / "figures").iterdir())

    tier_summary = ""
    if df is not None:
        try:
            from analysis.quality_stratification import TIER_LABELS
            from analysis.plot_quality_stratification import _resolve_families
            FAMILIES, BASE_COLS = _resolve_families(df)

            lines = ["| Tier | n | " + " | ".join(f"{f[0]} base | {f[0]} kfstd | {f[0]} QROBF | Δ%" for f in FAMILIES) + " |"]
            lines.append("|" + "---|" * (2 + 4 * len(FAMILIES)))
            for tier in TIER_LABELS:
                t = df[df["tier"] == tier]
                n = len(t)
                cells = [f"**{tier}**", str(n)]
                for label, kf_col, qr_col, *_ in FAMILIES:
                    base_col = BASE_COLS.get(label, "")
                    base = pd.to_numeric(t[base_col], errors="coerce").mean() if base_col and base_col in t.columns else float("nan")
                    kf = pd.to_numeric(t[kf_col], errors="coerce").mean()
                    qr = pd.to_numeric(t[qr_col], errors="coerce").mean()
                    pct = (qr - kf) / kf * 100 if (kf > 0 and not np.isnan(kf + qr)) else float("nan")
                    base_str = f"{base:.2f}" if not np.isnan(base) else "N/A"
                    cells += [base_str, f"{kf:.2f}", f"{qr:.2f}", f"{pct:+.1f}%"]
                lines.append("| " + " | ".join(cells) + " |")
            tier_summary = "\n".join(lines)
        except Exception as exc:
            tier_summary = f"_(could not build tier table: {exc})_"

    md_lines = [
        "# Paper Analysis Index",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Quality Tier Summary (freq MAE, BPM)",
        "",
        tier_summary,
        "",
        "## Tables",
        "",
    ]
    TABLE_DESCRIPTIONS = {
        "table1_freq_domain_mae.csv":    "**Table 1** — Overall frequency-domain MAE per method (all 160 trials)",
        "table2_per_tier_mae.csv":       "**Table 2** — Per quality-tier MAE: kfstd vs QROBF with Δ% (4 tiers × 5 families)",
        "table3_filter_diagnostics.csv": "**Table 3** — Filter diagnostics summary (NIS, g_t, α_R, fail rates)",
    }
    for tbl in tables:
        desc = TABLE_DESCRIPTIONS.get(tbl.name, f"**{tbl.stem}**")
        md_lines.append(f"- [{tbl.name}](tables/{tbl.name}) — {desc}")

    md_lines += ["", "## Figures", ""]
    FIGURE_DESCRIPTIONS = {
        "fig0_overall_performance.png":  "**Fig 0** — Overall performance summary (all methods, time+freq domain)",
        "fig1_snr_distribution.pdf":     "**Fig 1** — Spectral SNR distribution (OF signal, 160 trials) with tier colour bands",
        "fig2_tier_bar_OF.pdf":          "**Fig 2** — Per-tier MAE bar chart: OF family (base / kfstd / QROBF)",
        "fig3_tier_bar_all_families.pdf":"**Fig 3** — Per-tier MAE bar chart: all 5 families (2×2 panel)",
        "fig4_relative_improvement.pdf": "**Fig 4** — Relative improvement QROBF vs kfstd (%) per tier × family",
        "fig5_boxplots.pdf":             "**Fig 5** — Box plots (OF, P1D-Linear, P1D-Cubic) per tier",
        "fig6_scatter_snr_vs_mae.pdf":   "**Fig 6** — Scatter: spectral SNR vs freq MAE (kfstd vs QROBF, P1D-Cubic)",
        "fig7_improvement_heatmap.pdf":  "**Fig 7** — Heatmap: % improvement tier × family",
        "boundary_prob_curve.png":       "**Fig 8** — Instability boundary: P(QROBF better) vs signal quality decile",
        "boundary_delta_deciles.png":    "**Fig 9** — Instability boundary: Δ MAE per quality decile",
        "boundary_regime_map.png":       "**Fig 10** — Regime map: trial-level quality vs improvement scatter",
    }
    for fig in figures:
        desc = FIGURE_DESCRIPTIONS.get(fig.name, f"**{fig.stem}**")
        md_lines.append(f"- [{fig.name}](figures/{fig.name}) — {desc}")

    md_lines += [
        "",
        "## Pipeline",
        "",
        "```",
        "# Pre-run (once, before main.py):",
        "python analysis/run_paper_analysis.py --config configs/cohface_robust_ossm.json --pre-run",
        "  Step 0a: prepare_cohface_obs   → dataset trial dirs (obs_*.npy)",
        "  Step 0b: run_noise_analysis    → analysis/noise_properties/",
        "",
        "# After main.py:",
        "python analysis/run_paper_analysis.py --run-dir results/cohface_robust_ossm",
        "  Step 1: Performance summary    → <run>/plots/paper_performance_summary.png",
        "  Step 2: Innovation EDA         → <run>/eda/",
        "  Step 3: Instability boundary   → <run>/plots/boundary/",
        "  Step 4: Quality stratification → <run>/plots/quality_stratification/",
        "  Step 5: Quality figures        → <run>/plots/quality_stratification/fig*.pdf",
        "  Step 6: Paper directory        → <run>/paper/{tables,figures}/",
        "  Step 7: This index             → <run>/paper/PAPER_INDEX.md",
        "```",
        "",
        "Generated by `analysis/run_paper_analysis.py`",
    ]

    index_path = paper_dir / "PAPER_INDEX.md"
    index_path.write_text("\n".join(md_lines))
    print(f"[paper] PAPER_INDEX.md → {index_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Unified QROBF paper analysis pipeline")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--run-dir", help="Path to run directory, e.g. results/cohface_robust_ossm")
    grp.add_argument("--config",  help="Path to config JSON; run-dir derived from results/<run_label>")
    ap.add_argument("--pre-run",       action="store_true",
                    help="Run pre-processing steps only (prepare_obs + noise_analysis). "
                         "Execute BEFORE main.py.")
    ap.add_argument("--skip-perf",     action="store_true", help="Skip performance summary plots")
    ap.add_argument("--skip-eda",      action="store_true", help="Skip innovation EDA step")
    ap.add_argument("--skip-boundary", action="store_true", help="Skip instability boundary step")
    ap.add_argument("--skip-figs",     action="store_true", help="Skip figure generation")
    args = ap.parse_args()

    # Resolve config_path and run_dir
    config_path: Optional[str] = None
    if args.config:
        config_path = str(Path(args.config).resolve())
        with open(config_path) as f:
            cfg = json.load(f)
        run_label    = cfg.get("run_label", Path(args.config).stem)
        results_root = cfg.get("results_dir", "results")
        run_dir      = os.path.join(results_root, run_label)
    else:
        run_dir = args.run_dir

    run_dir = str(Path(run_dir).resolve())

    # ── PRE-RUN mode ─────────────────────────────────────────────────────────
    if args.pre_run:
        if config_path is None:
            sys.exit("[ERROR] --pre-run requires --config (not --run-dir)")
        print(f"\nPRE-RUN mode  · config: {config_path}")
        step_prepare_obs(config_path)
        step_noise_analysis(config_path)
        print(f"\n{'='*60}")
        print("  Pre-run steps done. Now run main.py, then re-run without --pre-run.")
        print(f"{'='*60}\n")
        return

    # ── POST-RUN mode ─────────────────────────────────────────────────────────
    if not os.path.isdir(run_dir):
        sys.exit(f"[ERROR] Run directory not found: {run_dir}")

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nRun directory : {run_dir}")
    print(f"Plots output  : {plots_dir}")

    # Step 1: Overall performance summary
    if not args.skip_perf:
        step_performance_summary(run_dir)

    # Step 2: Innovation EDA
    if not args.skip_eda:
        step_innovation_eda(run_dir)

    # Step 3: Instability boundary
    if not args.skip_boundary:
        step_instability_boundary(run_dir)

    # Step 4 & 5: Quality stratification + figures
    df: Optional[pd.DataFrame] = None
    df = step_quality_stratification(run_dir, plots_dir)
    if df is not None and not args.skip_figs:
        step_quality_figures(df, plots_dir)

    # Step 6: Assemble paper/ directory
    paper_dir = step_build_paper_dir(run_dir, df)

    # Step 7: Write PAPER_INDEX.md
    step_write_paper_index(paper_dir, run_dir, df)

    print(f"\n{'='*60}")
    print(f"  Done!  Paper assets → {paper_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
