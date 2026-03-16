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
  5b. Paper figure suite       — overview / mechanism / waveform figures
  6. Paper directory assembly  — tables/, figures/, PAPER_INDEX.md
  7. Manuscript sync           — copy latest paper assets into notes/.../prism_files

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
from analysis.paper_asset_specs import ABLATION_ROWS, FAMILY_SPECS

# ─── local imports (deferred to avoid heavy GPU startup at import time) ───────


def _step_banner(name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


MANUSCRIPT_TEMPLATE_DIR = _REPO / "notes" / "Template_for_submissions_to_Scientific_Reports__2_"

MANUSCRIPT_SYNC_ITEMS = (
    ("run", "plots/paper/fig1_overview.png", "prism_files/06_comparison/fig1_overview.png"),
    ("run", "plots/paper/fig2_trust_mapping.png", "prism_files/01_prompt/fig2_trust_mapping.png"),
    ("run", "plots/fig_freq_comparison.png", "prism_files/06_comparison/fig_freq_comparison.png"),
    ("run", "plots/fig_time_comparison.png", "prism_files/06_comparison/fig_time_comparison.png"),
    ("run", "plots/paper/waveform_overlay_examples.png", "prism_files/06_comparison/waveform_overlay_examples.png"),
    ("run", "plots/paper/fig4_robust_update.png", "prism_files/01_prompt/fig4_robust_update.png"),
    ("repo", "analysis/noise_properties/global_analysis_OF_Farneback.png", "prism_files/02_eda_fig5/global_analysis_OF_Farneback.png"),
    ("repo", "analysis/noise_properties/global_analysis_Profile1D_Linear.png", "prism_files/02_eda_fig5/global_analysis_Profile1D_Linear.png"),
    ("repo", "analysis/noise_properties/global_analysis_Profile1D_Quad.png", "prism_files/02_eda_fig5/global_analysis_Profile1D_Quad.png"),
    ("repo", "analysis/noise_properties/global_analysis_Profile1D_Cubic.png", "prism_files/02_eda_fig5/global_analysis_Profile1D_Cubic.png"),
    ("repo", "analysis/noise_properties/summary.csv", "prism_files/02_eda_fig5/summary.csv"),
    ("run", "plots/qrobf_diagnostics/of_farneback__robust_ossm_ekf.png", "prism_files/03_failure_fig6/of_farneback__robust_ossm_ekf.png"),
    ("run", "plots/qrobf_diagnostics/profile1d_cubic__robust_ossm_ekf.png", "prism_files/03_failure_fig6/profile1d_cubic__robust_ossm_ekf.png"),
    ("run", "plots/qrobf_diagnostics/qrobf_event_summary.png", "prism_files/03_failure_fig6/qrobf_event_summary.png"),
    ("run", "plots/qrobf_diagnostics/qrobf_event_summary_rates.csv", "prism_files/03_failure_fig6/qrobf_event_summary_rates.csv"),
    ("run", "plots/qrobf_diagnostics/profile1d_cubic__robust_ossm_ekf_event_counts.csv", "prism_files/03_failure_fig6/profile1d_cubic__robust_ossm_ekf_event_counts.csv"),
    ("run", "plots/paper/fig7_ablation.png", "prism_files/06_comparison/fig7_ablation.png"),
    ("run", "plots/quality_stratification/fig3_tier_bar_all_families.png", "prism_files/06_comparison/fig3_tier_bar_all_families.png"),
    ("run", "plots/quality_stratification/fig7_improvement_heatmap.png", "prism_files/06_comparison/fig7_improvement_heatmap.png"),
    ("run", "plots/filter_diagnostics_overview.png", "prism_files/04_calibration_fig8/filter_diagnostics_overview.png"),
    ("run", "plots/filter_diagnostics_heatmap.png", "prism_files/04_calibration_fig8/filter_diagnostics_heatmap.png"),
    ("run", "plots/boundary/boundary_prob_curve.png", "prism_files/04_calibration_fig8/boundary_prob_curve.png"),
    ("run", "plots/boundary/boundary_regime_map.png", "prism_files/04_calibration_fig8/boundary_regime_map.png"),
    ("run", "plots/boundary/boundary_delta_deciles.png", "prism_files/04_calibration_fig8/boundary_delta_deciles.png"),
    ("run", "metrics/metrics_freq_domain_summary.csv", "prism_files/06_tables/metrics_freq_domain_summary.csv"),
    ("run", "metrics/metrics_filter_diagnostics_summary.csv", "prism_files/06_tables/metrics_filter_diagnostics_summary.csv"),
    ("run", "paper/tables/performance_comprehensive.csv", "prism_files/06_tables/performance_comprehensive.csv"),
    ("run", "paper/tables/innovation_diagnostics.csv", "prism_files/06_tables/innovation_diagnostics.csv"),
    ("run", "paper/tables/ablation_profile1d_cubic.csv", "prism_files/06_tables/ablation_profile1d_cubic.csv"),
    ("run", "paper/tables/quality_tier_freq_mae.csv", "prism_files/06_tables/quality_tier_freq_mae.csv"),
    ("run", "paper/tables/filter_calibration_summary.csv", "prism_files/06_tables/filter_calibration_summary.csv"),
    ("run", "paper/tables/waveform_overlay_examples_manifest.csv", "prism_files/06_tables/waveform_overlay_examples_manifest.csv"),
)


def _round_or_nan(value: float, digits: int = 3) -> float:
    if value is None:
        return float("nan")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return round(value, digits) if np.isfinite(value) else float("nan")


def _metric_value(df: pd.DataFrame, method: str, metric: str, suffix: str = "_median") -> float:
    if df.empty or "method" not in df.columns:
        return float("nan")
    row = df.loc[df["method"] == method]
    if row.empty:
        return float("nan")
    col = f"{metric}{suffix}" if f"{metric}{suffix}" in df.columns else metric
    if col not in row.columns:
        return float("nan")
    return _round_or_nan(pd.to_numeric(row.iloc[0][col], errors="coerce"))


def _normalized_family_key(name: str) -> str:
    key = str(name).strip()
    key = key.replace("__robust_ossm_ekf", "")
    key = key.replace("__kfstd", "")
    key = key.replace("profile1d", "Profile1D")
    key = key.replace("of_farneback", "OF_Farneback")
    key = key.replace("dof", "DoF")
    key = key.replace(" linear", "_Linear")
    key = key.replace(" quadratic", "_Quad")
    key = key.replace(" cubic", "_Cubic")
    key = key.replace("-", "_")
    key = key.replace(" ", "_")
    key = key.replace("profile1D", "Profile1D")
    key = key.replace("_linear", "_Linear")
    key = key.replace("_quadratic", "_Quad")
    key = key.replace("_cubic", "_Cubic")
    return key


def _clear_directory(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


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


def step_paper_suite(run_dir: str) -> list[Path]:
    """Generate the manuscript-oriented figure suite under run_dir/plots/paper."""
    _step_banner("Step 5b · Paper Figure Suite")
    out_dir = Path(run_dir) / "plots" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    try:
        from analysis.plot_mechanism_figures import (
            fig_ablation,
            fig_robust_update,
            fig_trust_mapping,
        )
        from analysis.plot_paper_figures import (
            fig_calibration,
            fig_dual_domain_matrix,
            fig_freq_primary,
            fig_overview,
            fig_paper_summary,
            fig_tier_breakdown,
        )

        fig_overview(run_dir, str(out_dir))
        fig_freq_primary(run_dir, str(out_dir))
        fig_calibration(run_dir, str(out_dir))
        fig_tier_breakdown(run_dir, str(out_dir))
        fig_dual_domain_matrix(run_dir, str(out_dir))
        fig_paper_summary(run_dir, str(out_dir))
        fig_trust_mapping(str(out_dir))
        fig_robust_update(str(out_dir))
        fig_ablation(str(out_dir))
        _build_waveform_overlay_examples(run_dir, out_dir / "waveform_overlay_examples")

        generated = sorted(p for p in out_dir.iterdir() if p.is_file())
        print(f"[paper_figs] {len(generated)} files generated in {out_dir}")
        return generated
    except Exception as exc:
        print(f"[paper_figs] Warning: {exc}")
        import traceback; traceback.print_exc()
        return generated


def _build_waveform_overlay_examples(run_dir: str, out_stem: Path) -> Optional[list[Path]]:
    """Build a 3-panel best/median/worst aligned waveform figure for the top QROBF family."""
    import pickle

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from core.evaluation.metrics import calculate_cross_corr_alignment
    from core.utils.common import filter_RW

    freq_summary = Path(run_dir) / "metrics" / "metrics_freq_domain_summary.csv"
    freq_raw = Path(run_dir) / "metrics" / "metrics_freq_domain_raw.csv"
    time_summary = Path(run_dir) / "metrics" / "metrics_time_domain_summary.csv"
    time_raw = Path(run_dir) / "metrics" / "metrics_time_domain_raw.csv"
    if not (freq_summary.exists() and freq_raw.exists() and time_summary.exists() and time_raw.exists()):
        print("[paper_figs] waveform overlays skipped (missing metrics csvs)")
        return None

    freq_summary_df = pd.read_csv(freq_summary)
    time_summary_df = pd.read_csv(time_summary)
    freq_raw_df = pd.read_csv(freq_raw)
    time_raw_df = pd.read_csv(time_raw)

    candidates = []
    for spec in FAMILY_SPECS:
        if spec["family"] == "DoF":
            continue
        freq_mae = _metric_value(freq_summary_df, spec["qrobf_method"], "MAE")
        time_ccc = _metric_value(time_summary_df, spec["qrobf_method"], "CCC")
        time_mae = _metric_value(time_summary_df, spec["qrobf_method"], "MAE")
        if np.isfinite(freq_mae):
            ccc_key = -time_ccc if np.isfinite(time_ccc) else np.inf
            time_key = time_mae if np.isfinite(time_mae) else np.inf
            candidates.append((freq_mae, ccc_key, time_key, spec))
    if not candidates:
        print("[paper_figs] waveform overlays skipped (no candidate QROBF family)")
        return None

    _, _, _, chosen = min(candidates, key=lambda item: item[:3])
    method = chosen["qrobf_method"]
    method_freq = freq_raw_df.loc[freq_raw_df["method"] == method].sort_values("MAE").reset_index(drop=True)
    if method_freq.empty:
        print(f"[paper_figs] waveform overlays skipped (no trial rows for {method})")
        return None

    time_lookup = (
        time_raw_df.loc[time_raw_df["method"] == method, ["video", "CCC", "MAE"]]
        .rename(columns={"MAE": "time_mae"})
        .drop_duplicates(subset=["video"])
        .set_index("video")
        if {"video", "CCC", "MAE"}.issubset(time_raw_df.columns)
        else pd.DataFrame()
    )

    indices = {
        "Best": 0,
        "Median": len(method_freq) // 2,
        "Worst": len(method_freq) - 1,
    }

    def _normalize(sig: np.ndarray) -> np.ndarray:
        arr = np.asarray(sig, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return arr
        arr = arr - np.nanmean(arr)
        scale = np.nanstd(arr)
        return arr / scale if scale > 1e-9 else np.zeros_like(arr)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    manifest_rows = []
    for ax, (rank_label, idx) in zip(axes, indices.items()):
        row = method_freq.iloc[idx]
        video = str(row["video"])
        data_file = str(row.get("data_file", f"data/{video}.pkl"))
        pkl_path = Path(run_dir) / data_file if not os.path.isabs(data_file) else Path(data_file)
        if not pkl_path.exists():
            ax.set_axis_off()
            ax.set_title(f"{rank_label}\nmissing {video}")
            continue

        with open(pkl_path, "rb") as fp:
            trial_payload = pickle.load(fp)

        fps = float(trial_payload.get("fps", 30.0))
        fs_gt = float(trial_payload.get("fs_gt", fps))
        gt_raw = np.asarray(trial_payload.get("gt", []), dtype=np.float64).reshape(-1)
        estimates = trial_payload.get("estimates", [])
        target_est = next((est for est in estimates if est.get("method") == method), None)
        if gt_raw.size == 0 or target_est is None:
            ax.set_axis_off()
            ax.set_title(f"{rank_label}\nunavailable {video}")
            continue

        payload = target_est.get("estimate", target_est)
        est_wave = np.asarray(payload.get("signal_hat", []), dtype=np.float64).reshape(-1)
        if est_wave.size == 0:
            ax.set_axis_off()
            ax.set_title(f"{rank_label}\nno waveform {video}")
            continue

        gt_filt = np.squeeze(filter_RW(gt_raw, fs_gt, lo=0.08, hi=0.5))
        est_filt = np.squeeze(filter_RW(est_wave, fps, lo=0.08, hi=0.5))
        plot_est, plot_gt, lag_sec = calculate_cross_corr_alignment(
            est_filt,
            gt_filt,
            fs_est=fps,
            fs_gt=fs_gt,
        )
        if plot_est.size < 10:
            ax.set_axis_off()
            ax.set_title(f"{rank_label}\nshort {video}")
            continue

        t = np.arange(plot_est.size, dtype=np.float64) / fs_gt
        ax.plot(t, _normalize(plot_gt), color="#4d4d4d", linestyle="--", linewidth=1.4, label="Ground truth")
        ax.plot(t, _normalize(plot_est), color="#c0392b", linewidth=1.5, alpha=0.9, label="QROBF")
        ax.set_title(rank_label, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.text(
            0.02,
            0.98,
            "\n".join(
                part
                for part in (
                    video,
                    f"Freq MAE={float(row['MAE']):.2f} BPM",
                    (
                        f"CCC={float(time_lookup.loc[video, 'CCC']):.2f}"
                        if not time_lookup.empty and video in time_lookup.index
                        else ""
                    ),
                    f"Lag={lag_sec:.2f}s",
                )
                if part
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.92, "pad": 3},
        )
        manifest_rows.append(
            {
                "selection": rank_label.lower(),
                "family": chosen["family"],
                "method": method,
                "video": video,
                "freq_mae_bpm": _round_or_nan(row["MAE"], 3),
                "time_ccc": (
                    _round_or_nan(time_lookup.loc[video, "CCC"], 3)
                    if not time_lookup.empty and video in time_lookup.index
                    else float("nan")
                ),
                "lag_sec": _round_or_nan(lag_sec, 3),
            }
        )

    axes[0].set_ylabel("Normalized amplitude")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Representative aligned waveform overlays from the top QROBF family\n"
        f"{chosen['family']} selected by lowest median frequency-domain MAE",
        fontsize=12,
        fontweight="bold",
        y=1.08,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("png", "pdf"):
        out_path = out_stem.with_suffix(f".{ext}")
        fig.savefig(out_path)
        outputs.append(out_path)
    plt.close(fig)

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(out_stem.with_name(f"{out_stem.name}_manifest.csv"), index=False)

    print(f"[paper_figs] Waveform examples → {out_stem.with_suffix('.png')}")
    return outputs


def step_build_paper_dir(run_dir: str, df: Optional[pd.DataFrame]) -> Path:
    """Assemble paper/ directory with Tables and Figures."""
    _step_banner("Step 6 · Paper Directory")
    paper_dir = Path(run_dir) / "paper"
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _clear_directory(tables_dir)
    _clear_directory(figures_dir)

    # ── Tables ────────────────────────────────────────────────────────────────
    _build_comprehensive_performance_table(run_dir, tables_dir)
    _build_innovation_diagnostics_table(run_dir, tables_dir)
    _build_ablation_table(tables_dir)
    if df is not None:
        _build_quality_tier_table(df, tables_dir)
    _build_filter_calibration_table(run_dir, tables_dir)

    # ── Figures ───────────────────────────────────────────────────────────────
    plots_paper_dir = Path(run_dir) / "plots" / "paper"
    if plots_paper_dir.exists():
        manifest_csv = plots_paper_dir / "waveform_overlay_examples_manifest.csv"
        if manifest_csv.exists():
            shutil.copy(manifest_csv, tables_dir / manifest_csv.name)
        for src in sorted(plots_paper_dir.glob("*")):
            if src.is_file() and src.suffix.lower() in {".png", ".pdf"}:
                shutil.copy(src, figures_dir / src.name)

    qs_dir = Path(run_dir) / "plots" / "quality_stratification"
    for src in sorted(qs_dir.glob("fig*.*")):
        if src.is_file():
            shutil.copy(src, figures_dir / src.name)

    boundary_dir = Path(run_dir) / "plots" / "boundary"
    for src in sorted(boundary_dir.glob("*")):
        if src.is_file() and src.suffix.lower() in {".png", ".pdf"}:
            shutil.copy(src, figures_dir / src.name)

    perf_png = Path(run_dir) / "plots" / "paper_performance_summary.png"
    if perf_png.exists():
        shutil.copy(perf_png, figures_dir / "overall_performance_summary.png")

    n_tables  = len(list(tables_dir.glob("*.csv")))
    n_figures = len(list(figures_dir.iterdir()))
    print(f"[paper] {n_tables} tables, {n_figures} figures → {paper_dir}")
    return paper_dir


def step_sync_manuscript_assets(run_dir: str, manuscript_dir: Optional[Path] = None) -> Optional[Path]:
    """Copy the latest paper assets into the Scientific Reports template directory."""
    _step_banner("Step 8 · Manuscript Asset Sync")
    target_dir = manuscript_dir or MANUSCRIPT_TEMPLATE_DIR
    if not target_dir.exists():
        print(f"[manuscript] skipped (template dir not found: {target_dir})")
        return None

    copied = 0
    missing: list[str] = []
    run_root = Path(run_dir)
    for source_kind, src_rel, dst_rel in MANUSCRIPT_SYNC_ITEMS:
        src_root = run_root if source_kind == "run" else _REPO
        src = src_root / src_rel
        dst = target_dir / dst_rel
        if not src.exists():
            missing.append(str(src))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        copied += 1

    print(f"[manuscript] {copied}/{len(MANUSCRIPT_SYNC_ITEMS)} assets synced → {target_dir}")
    if missing:
        for src in missing[:8]:
            print(f"[manuscript] missing: {src}")
        if len(missing) > 8:
            print(f"[manuscript] ... {len(missing) - 8} more missing assets")
    return target_dir


def _build_comprehensive_performance_table(run_dir: str, tables_dir: Path) -> None:
    """Combined family-level performance table including base, kfstd, and QROBF."""
    freq_csv = Path(run_dir) / "metrics" / "metrics_freq_domain_summary.csv"
    time_csv = Path(run_dir) / "metrics" / "metrics_time_domain_summary.csv"
    if not (freq_csv.exists() and time_csv.exists()):
        print("[paper] performance table skipped (missing summary csvs)")
        return

    freq_df = pd.read_csv(freq_csv)
    time_df = pd.read_csv(time_csv)
    rows = []
    for spec in FAMILY_SPECS:
        for variant, method in (
            ("Base", spec["base_method"]),
            ("kfstd", spec["kfstd_method"]),
            ("QROBF", spec["qrobf_method"]),
        ):
            rows.append(
                {
                    "observation_family": spec["family"],
                    "variant": variant,
                    "freq_mae_bpm": _metric_value(freq_df, method, "MAE"),
                    "freq_rmse_bpm": _metric_value(freq_df, method, "RMSE"),
                    "freq_snr_db": _metric_value(freq_df, method, "SNR_Spec"),
                    "time_ccc": _metric_value(time_df, method, "CCC"),
                    "time_mae_bpm": _metric_value(time_df, method, "MAE"),
                    "time_rmse_bpm": _metric_value(time_df, method, "RMSE"),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "performance_comprehensive.csv", index=False)
    print(f"[paper] performance_comprehensive.csv → {tables_dir / 'performance_comprehensive.csv'}")


def _build_innovation_diagnostics_table(run_dir: str, tables_dir: Path) -> None:
    """Innovation diagnostics table merged with THD/ARCH statistics."""
    eda_csv = Path(run_dir) / "eda" / "innovation_summary.csv"
    noise_csv = _REPO / "analysis" / "noise_properties" / "summary.csv"
    if not eda_csv.exists():
        print("[paper] innovation diagnostics skipped (missing innovation_summary.csv)")
        return

    eda_df = pd.read_csv(eda_csv)
    eda_df["family_key"] = eda_df["method"].map(_normalized_family_key)
    eda_df = eda_df.rename(
        columns={
            "kurtosis": "innovation_kurtosis",
            "t_aic_delta": "delta_aic",
            "t_fit_nu": "nu_fit",
        }
    )

    merged = eda_df[["family_key", "innovation_kurtosis", "delta_aic", "nu_fit", "student_t_justified"]].copy()
    if noise_csv.exists():
        noise_df = pd.read_csv(noise_csv)
        noise_df["family_key"] = noise_df["Method"].map(_normalized_family_key)
        noise_df = noise_df.rename(columns={"THD": "thd", "ARCH_LM_pval": "arch_lm_p"})
        merged = merged.merge(
            noise_df[["family_key", "thd", "arch_lm_p"]],
            on="family_key",
            how="left",
        )
    else:
        merged["thd"] = float("nan")
        merged["arch_lm_p"] = float("nan")

    rows = []
    for spec in FAMILY_SPECS:
        family_key = _normalized_family_key(spec["noise_label"])
        row = merged.loc[merged["family_key"] == family_key]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append(
            {
                "observation_family": spec["family"],
                "kurtosis": _round_or_nan(r["innovation_kurtosis"], 3),
                "delta_aic": _round_or_nan(r["delta_aic"], 3),
                "nu_fit": _round_or_nan(r["nu_fit"], 3),
                "thd": _round_or_nan(r.get("thd", np.nan), 3),
                "arch_lm_p": _round_or_nan(r.get("arch_lm_p", np.nan), 6),
                "student_t_justified": bool(r["student_t_justified"]),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "innovation_diagnostics.csv", index=False)
    print(f"[paper] innovation_diagnostics.csv → {tables_dir / 'innovation_diagnostics.csv'}")


def _build_ablation_table(tables_dir: Path) -> None:
    out = pd.DataFrame(ABLATION_ROWS)
    out.to_csv(tables_dir / "ablation_profile1d_cubic.csv", index=False)
    print(f"[paper] ablation_profile1d_cubic.csv → {tables_dir / 'ablation_profile1d_cubic.csv'}")


def _build_quality_tier_table(df: pd.DataFrame, tables_dir: Path) -> None:
    """Quality-tier performance table without delta columns, plus overall median row."""
    from analysis.plot_quality_stratification import _resolve_families
    from analysis.quality_stratification import TIER_LABELS

    families, base_cols = _resolve_families(df)
    rows = []
    for tier in TIER_LABELS + ["Overall"]:
        if tier == "Overall":
            subset = df
            agg = "median"
        else:
            subset = df.loc[df["tier"] == tier]
            agg = "mean"

        row: dict[str, object] = {
            "tier": tier,
            "n_trials": int(len(subset)),
            "statistic": agg,
        }
        for label, kf_col, qr_col, *_ in families:
            base_col = base_cols.get(label, "")
            base_vals = pd.to_numeric(subset[base_col], errors="coerce").dropna() if base_col and base_col in subset.columns else pd.Series(dtype=float)
            kf_vals = pd.to_numeric(subset[kf_col], errors="coerce").dropna() if kf_col in subset.columns else pd.Series(dtype=float)
            qr_vals = pd.to_numeric(subset[qr_col], errors="coerce").dropna() if qr_col in subset.columns else pd.Series(dtype=float)
            reducer = np.nanmedian if agg == "median" else np.nanmean
            row[f"{label}_base_mae"] = _round_or_nan(reducer(base_vals) if len(base_vals) else np.nan)
            row[f"{label}_kfstd_mae"] = _round_or_nan(reducer(kf_vals) if len(kf_vals) else np.nan)
            row[f"{label}_qrobf_mae"] = _round_or_nan(reducer(qr_vals) if len(qr_vals) else np.nan)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "quality_tier_freq_mae.csv", index=False)
    print(f"[paper] quality_tier_freq_mae.csv → {tables_dir / 'quality_tier_freq_mae.csv'}")


def _build_filter_calibration_table(run_dir: str, tables_dir: Path) -> None:
    diag_csv = Path(run_dir) / "metrics" / "metrics_filter_diagnostics_summary.csv"
    if not diag_csv.exists():
        print("[paper] filter calibration table skipped (missing diagnostics summary)")
        return

    diag_df = pd.read_csv(diag_csv)
    rows = []
    for spec in FAMILY_SPECS:
        row = diag_df.loc[diag_df["method"] == spec["qrobf_method"]]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append(
            {
                "observation_family": spec["family"],
                "variant": "QROBF",
                "fail_total": _round_or_nan(r.get("Fail_Total_median", np.nan), 6),
                "fail_double": _round_or_nan(r.get("Fail_Double_median", np.nan), 6),
                "nis_mean": _round_or_nan(r.get("NIS_Mean_median", np.nan), 6),
                "lambda_lt1_frac": _round_or_nan(r.get("Lambda_LT1_Frac_median", np.nan), 6),
                "coverage95_pct": _round_or_nan(r.get("Coverage95_median", np.nan), 3),
                "stability_sec": _round_or_nan(r.get("Stability_Sec_median", np.nan), 3),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(tables_dir / "filter_calibration_summary.csv", index=False)
    print(f"[paper] filter_calibration_summary.csv → {tables_dir / 'filter_calibration_summary.csv'}")


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

            lines = ["| Tier | n | " + " | ".join(f"{f[0]} base | {f[0]} kfstd | {f[0]} QROBF" for f in FAMILIES) + " |"]
            lines.append("|" + "---|" * (2 + 3 * len(FAMILIES)))
            for tier in TIER_LABELS:
                t = df[df["tier"] == tier]
                n = len(t)
                cells = [f"**{tier}**", str(n)]
                for label, kf_col, qr_col, *_ in FAMILIES:
                    base_col = BASE_COLS.get(label, "")
                    base = pd.to_numeric(t[base_col], errors="coerce").mean() if base_col and base_col in t.columns else float("nan")
                    kf = pd.to_numeric(t[kf_col], errors="coerce").mean()
                    qr = pd.to_numeric(t[qr_col], errors="coerce").mean()
                    base_str = f"{base:.2f}" if not np.isnan(base) else "N/A"
                    cells += [base_str, f"{kf:.2f}", f"{qr:.2f}"]
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
        "performance_comprehensive.csv":   "**Table 2** — Comprehensive family-level performance table (base, kfstd, QROBF; no delta columns)",
        "innovation_diagnostics.csv":      "**Table 3** — Innovation diagnostics supporting the heavy-tail model choice",
        "ablation_profile1d_cubic.csv":    "**Table 4** — Profile1D-Cubic ablation summary (no delta column)",
        "quality_tier_freq_mae.csv":       "**Table 5** — Quality-tier frequency-domain MAE by family (mean per tier, median overall; no delta columns)",
        "filter_calibration_summary.csv":  "**Table 6** — QROBF failure and calibration summary",
        "waveform_overlay_examples_manifest.csv": "**Figure manifest** — Selected best/median/worst trials used in the waveform overlay figure",
    }
    for tbl in tables:
        desc = TABLE_DESCRIPTIONS.get(tbl.name, f"**{tbl.stem}**")
        md_lines.append(f"- [{tbl.name}](tables/{tbl.name}) — {desc}")

    md_lines += ["", "## Figures", ""]
    FIGURE_DESCRIPTIONS = {
        "fig1_overview.png":                 "**Fig 1** — Frequency/time-domain overview across observation families",
        "fig2_freq_primary.png":             "**Fig 2** — Primary frequency-domain comparison for kfstd vs QROBF",
        "waveform_overlay_examples.png":     "**Fig 4** — Representative aligned waveform overlays (best/median/worst trials)",
        "fig2_trust_mapping.png":            "**Mechanism figure** — Deterministic quality-to-trust mapping",
        "fig4_robust_update.png":            "**Mechanism figure** — Student-t robust update behavior",
        "fig7_ablation.png":                 "**Ablation figure** — EKS vs Student-t component contribution",
        "fig1_snr_distribution.png":         "**Tier figure** — Spectral SNR distribution with tier boundaries",
        "fig3_tier_bar_all_families.png":    "**Tier figure** — Per-tier MAE bar chart across all families",
        "fig7_improvement_heatmap.png":      "**Tier figure** — Relative improvement heatmap across tiers",
        "boundary_prob_curve.png":           "**Boundary figure** — P(QROBF better) vs signal-quality decile",
        "boundary_delta_deciles.png":        "**Boundary figure** — Mean MAE delta by quality decile",
        "boundary_regime_map.png":           "**Boundary figure** — Quality/improvement regime map",
        "overall_performance_summary.png":   "**Overview figure** — Aggregated time/frequency performance summary",
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
        "  Step 5b: Paper figure suite    → <run>/plots/paper/",
        "  Step 6: Paper directory        → <run>/paper/{tables,figures}/",
        "  Step 7: This index             → <run>/paper/PAPER_INDEX.md",
        "  Step 8: Manuscript sync        → notes/.../prism_files/",
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
    ap.add_argument("--skip-manuscript-sync", action="store_true",
                    help="Do not copy refreshed figures/tables into the manuscript prism_files directory")
    args = ap.parse_args()

    # Resolve config_path and run_dir
    config_path: Optional[str] = None
    if args.config:
        config_path = str(Path(args.config).resolve())
        with open(config_path) as f:
            cfg = json.load(f)
        run_label    = cfg.get("run_label") or cfg.get("name") or Path(args.config).stem
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
    if not args.skip_figs:
        step_paper_suite(run_dir)

    # Step 6: Assemble paper/ directory
    paper_dir = step_build_paper_dir(run_dir, df)

    # Step 7: Write PAPER_INDEX.md
    step_write_paper_index(paper_dir, run_dir, df)

    # Step 8: Sync manuscript assets
    if not args.skip_manuscript_sync:
        step_sync_manuscript_assets(run_dir)

    print(f"\n{'='*60}")
    print(f"  Done!  Paper assets → {paper_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
