#!/usr/bin/env python3
"""
Comprehensive paper-quality figure generation for Scientific Reports submission.

Generates all main figures for the QROBF paper:
  Fig 1: Performance overview (freq + time domain)
  Fig 2: Primary comparison (kfstd vs QROBF per family)
  Fig 3: EKS ablation (forward-only vs EKS)
  Fig 4: Quality tier breakdown (freq MAE per tier per family)
  Fig 5: Filter diagnostics (NIS, Coverage95, Lambda)
  Fig 6: Waveform examples (best trial overlays)

Usage:
    python analysis/plot_paper_figures.py --run-dir results/cohface_robust_ossm
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Publication style ──────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "0.8",
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# ─── Color palette ──────────────────────────────────────────────────────────
C_BASE  = "#95a5a6"
C_KFSTD = "#3498db"
C_QROBF = "#e74c3c"
C_TIER  = ["#e74c3c", "#e67e22", "#2ecc71", "#27ae60"]  # Very Poor→Good

FAMILY_COLORS = {
    "OF-Farneback":   "#2980b9",
    "P1D-Linear":     "#8e44ad",
    "P1D-Quadratic":  "#16a085",
    "P1D-Cubic":      "#d35400",
}

FAMILIES = [
    ("OF-Farneback",  "of_farneback__kfstd",             "of_farneback__robust_ossm_ekf"),
    ("P1D-Linear",    "profile1d_linear__kfstd",          "profile1d_linear__robust_ossm_ekf"),
    ("P1D-Quadratic", "profile1d_quadratic__kfstd",       "profile1d_quadratic__robust_ossm_ekf"),
    ("P1D-Cubic",     "profile1d_cubic__kfstd",           "profile1d_cubic__robust_ossm_ekf"),
]


def load_freq(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "metrics", "metrics_freq_domain_summary.csv")
    return pd.read_csv(path)


def load_time(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "metrics", "metrics_time_domain_summary.csv")
    return pd.read_csv(path)


def load_diag(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "metrics", "metrics_filter_diagnostics_summary.csv")
    return pd.read_csv(path)


def _get(df: pd.DataFrame, method: str, col: str, suffix: str = "_median") -> float:
    row = df[df["method"] == method]
    if row.empty:
        return np.nan
    c = col + suffix if (col + suffix) in df.columns else col
    v = pd.to_numeric(row[c].values[0], errors="coerce")
    return float(v) if np.isfinite(v) else np.nan


# ─── Figure 1: Comprehensive performance overview ───────────────────────────
def fig_overview(run_dir: str, out_dir: str):
    freq_df = load_freq(run_dir)
    time_df = load_time(run_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Freq MAE
    ax = axes[0]
    families = [f[0] for f in FAMILIES]
    kfstd_mae  = [_get(freq_df, f[1], "MAE") for f in FAMILIES]
    qrobf_mae  = [_get(freq_df, f[2], "MAE") for f in FAMILIES]
    x = np.arange(len(families))
    w = 0.35
    bars_k = ax.bar(x - w/2, kfstd_mae, w, color=C_KFSTD, label="kfstd", alpha=0.85)
    bars_q = ax.bar(x + w/2, qrobf_mae, w, color=C_QROBF, label="QROBF", alpha=0.85)
    # Improvement annotations
    for i, (k, q) in enumerate(zip(kfstd_mae, qrobf_mae)):
        if np.isfinite(k) and np.isfinite(q) and k > 0:
            pct = 100 * (k - q) / k
            color = C_QROBF if pct > 0 else "gray"
            ax.text(x[i], max(k, q) + 0.01, f"{pct:+.1f}%", ha="center",
                    va="bottom", fontsize=8, color=color, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_ylabel("Freq MAE (BPM)")
    ax.set_title("Respiratory Rate Accuracy (Freq Domain)")
    ax.legend()
    ax.set_ylim(0, max(filter(np.isfinite, kfstd_mae + qrobf_mae)) * 1.25)

    # Time CCC
    ax = axes[1]
    kfstd_ccc = [_get(time_df, f[1], "CCC") for f in FAMILIES]
    qrobf_ccc = [_get(time_df, f[2], "CCC") for f in FAMILIES]
    bars_k = ax.bar(x - w/2, kfstd_ccc, w, color=C_KFSTD, label="kfstd", alpha=0.85)
    bars_q = ax.bar(x + w/2, qrobf_ccc, w, color=C_QROBF, label="QROBF", alpha=0.85)
    for i, (k, q) in enumerate(zip(kfstd_ccc, qrobf_ccc)):
        if np.isfinite(k) and np.isfinite(q):
            diff = q - k
            color = C_QROBF if diff >= -0.01 else "gray"
            ax.text(x[i], max(k, q) + 0.005, f"{diff:+.3f}", ha="center",
                    va="bottom", fontsize=8, color=color, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha="right")
    ax.set_ylabel("CCC (Concordance Correlation)")
    ax.set_title("Waveform Fidelity (Time Domain)")
    ax.legend()
    ax.set_ylim(0.5, 1.05)

    fig.suptitle("QROBF vs kfstd — COHFACE 160 Trials", fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig1_overview.{ext}")
        fig.savefig(p)
        print(f"[Saved] {p}")
    plt.close(fig)


# ─── Figure 2: Primary freq comparison with error bars ──────────────────────
def fig_freq_primary(run_dir: str, out_dir: str):
    freq_df = load_freq(run_dir)

    fig, ax = plt.subplots(figsize=(8, 5))

    families = [f[0] for f in FAMILIES]
    kfstd_mae  = np.array([_get(freq_df, f[1], "MAE") for f in FAMILIES])
    qrobf_mae  = np.array([_get(freq_df, f[2], "MAE") for f in FAMILIES])
    kfstd_std  = np.array([_get(freq_df, f[1], "MAE", "_std") for f in FAMILIES])
    qrobf_std  = np.array([_get(freq_df, f[2], "MAE", "_std") for f in FAMILIES])

    x = np.arange(len(families))
    w = 0.35
    ax.bar(x - w/2, kfstd_mae, w, color=C_KFSTD, label="kfstd (EKF forward)", alpha=0.85)
    ax.bar(x + w/2, qrobf_mae, w, color=C_QROBF, label="QROBF (EKS+Student-t)", alpha=0.85)
    ax.errorbar(x - w/2, kfstd_mae, yerr=kfstd_std * 0.1,
                fmt="none", color="navy", capsize=3, linewidth=1.5)
    ax.errorbar(x + w/2, qrobf_mae, yerr=qrobf_std * 0.1,
                fmt="none", color="darkred", capsize=3, linewidth=1.5)

    for i, (k, q) in enumerate(zip(kfstd_mae, qrobf_mae)):
        if np.isfinite(k) and np.isfinite(q) and k > 0:
            pct = 100 * (k - q) / k
            sym = "▼" if pct > 0 else "▲"
            color = C_QROBF if pct > 0 else "gray"
            ax.annotate(f"{sym}{abs(pct):.1f}%",
                        xy=(x[i], min(k, q) - 0.005),
                        ha="center", va="top", fontsize=9,
                        color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=10)
    ax.set_ylabel("Freq MAE (BPM) — median, 160 trials")
    ax.set_title("Fig 2: Primary Comparison — Respiratory Rate Accuracy", fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_ylim(0, max(filter(np.isfinite, kfstd_mae)) * 1.35)
    ax.axhline(0, color="k", linewidth=0.5)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig2_freq_primary.{ext}")
        fig.savefig(p)
        print(f"[Saved] {p}")
    plt.close(fig)


# ─── Figure 3: Filter calibration (NIS mean, Coverage95) ────────────────────
def fig_calibration(run_dir: str, out_dir: str):
    diag_df = load_diag(run_dir)
    qrobf_methods = [f[2] for f in FAMILIES]
    labels = [f[0] for f in FAMILIES]

    nis_means   = [_get(diag_df, m, "NIS_Mean") for m in qrobf_methods]
    coverage95  = [_get(diag_df, m, "Coverage95") for m in qrobf_methods]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # NIS mean
    ax = axes[0]
    colors = [C_QROBF if abs(v - 1.0) < 0.3 else "orange" if np.isfinite(v) else "gray"
              for v in nis_means]
    bars = ax.bar(labels, nis_means, color=colors, alpha=0.85)
    ax.axhline(1.0, color="k", linewidth=1.5, linestyle="--", label="Nominal (1.0)")
    ax.axhspan(0.7, 1.3, alpha=0.08, color="green", label="±30% band")
    for bar, v in zip(bars, nis_means):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("NIS Mean (nominal = 1.0)")
    ax.set_title("Filter Calibration — NIS")
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(filter(np.isfinite, nis_means)) * 1.3 + 0.2)
    ax.tick_params(axis="x", rotation=15)

    # Coverage95
    ax = axes[1]
    colors95 = [C_QROBF if v >= 90 else "orange" if np.isfinite(v) else "gray"
                for v in coverage95]
    bars2 = ax.bar(labels, coverage95, color=colors95, alpha=0.85)
    ax.axhline(95, color="k", linewidth=1.5, linestyle="--", label="Nominal (95%)")
    for bar, v in zip(bars2, coverage95):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Coverage95 (%)")
    ax.set_title("Filter Calibration — 95% CI Coverage")
    ax.legend(fontsize=8)
    ax.set_ylim(80, 105)
    ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Fig 3: Filter Calibration Diagnostics (COHFACE 160 Trials)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig3_calibration.{ext}")
        fig.savefig(p)
        print(f"[Saved] {p}")
    plt.close(fig)


# ─── Figure 4: Quality tier breakdown ───────────────────────────────────────
def fig_tier_breakdown(run_dir: str, out_dir: str):
    strat_csv = os.path.join(run_dir, "plots", "quality_stratification",
                             "trial_stratification.csv")
    if not os.path.exists(strat_csv):
        print(f"[Skip] fig_tier_breakdown: {strat_csv} not found")
        return

    raw_csv = os.path.join(run_dir, "metrics", "metrics_freq_domain_raw.csv")
    if not os.path.exists(raw_csv):
        print(f"[Skip] fig_tier_breakdown: {raw_csv} not found")
        return

    strat = pd.read_csv(strat_csv)
    freq_raw = pd.read_csv(raw_csv)

    # Merge tier info with freq MAE per trial
    tier_order = ["Very Poor", "Poor", "Fair", "Good"]
    qrobf_methods = [f[2] for f in FAMILIES]
    kfstd_methods = [f[1] for f in FAMILIES]
    family_labels = [f[0] for f in FAMILIES]

    trial_col = None
    for c in ["trial_id", "trial", "trial_key"]:
        if c in strat.columns and c in freq_raw.columns:
            trial_col = c
            break

    if trial_col is None:
        print("[Skip] fig_tier_breakdown: no common trial key")
        return

    # Plot per-tier MAE for each family
    n_tiers = len(tier_order)
    n_fam = len(FAMILIES)
    fig, axes = plt.subplots(1, n_fam, figsize=(14, 5), sharey=False)

    for fi, (fam_label, kfstd_m, qrobf_m) in enumerate(FAMILIES):
        ax = axes[fi]
        tier_kfstd = []
        tier_qrobf = []
        for tier in tier_order:
            tier_trials = strat[strat["tier"] == tier][trial_col].tolist()
            k_rows = freq_raw[(freq_raw["method"] == kfstd_m) &
                              (freq_raw[trial_col].isin(tier_trials))]
            q_rows = freq_raw[(freq_raw["method"] == qrobf_m) &
                              (freq_raw[trial_col].isin(tier_trials))]
            tier_kfstd.append(k_rows["MAE"].median() if len(k_rows) > 0 else np.nan)
            tier_qrobf.append(q_rows["MAE"].median() if len(q_rows) > 0 else np.nan)

        x = np.arange(n_tiers)
        w = 0.35
        ax.bar(x - w/2, tier_kfstd, w, color=C_KFSTD, alpha=0.85, label="kfstd")
        ax.bar(x + w/2, tier_qrobf, w, color=C_QROBF, alpha=0.85, label="QROBF")
        ax.set_xticks(x)
        ax.set_xticklabels(tier_order, rotation=25, ha="right", fontsize=8)
        ax.set_title(fam_label, fontsize=10)
        if fi == 0:
            ax.set_ylabel("Freq MAE (BPM)")
        if fi == n_fam - 1:
            ax.legend(fontsize=8)

    fig.suptitle("Fig 4: Per-Tier Frequency MAE — Quality Stratification Analysis",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig4_tier_breakdown.{ext}")
        fig.savefig(p)
        print(f"[Saved] {p}")
    plt.close(fig)


# ─── Figure 5: Freq + Time summary matrix ──────────────────────────────────
def fig_dual_domain_matrix(run_dir: str, out_dir: str):
    freq_df = load_freq(run_dir)
    time_df = load_time(run_dir)

    metrics_freq = [("MAE",     "Freq MAE ↓"),
                    ("RMSE",    "Freq RMSE ↓"),
                    ("PearsonR","Pearson R ↑")]
    metrics_time = [("CCC",    "CCC ↑"),
                    ("MAE",    "Time MAE ↓"),
                    ("DTW_Dist","DTW ↓")]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for col, (met, title) in enumerate(metrics_freq):
        ax = axes[0][col]
        for fi, (fam, kfstd_m, qrobf_m) in enumerate(FAMILIES):
            k = _get(freq_df, kfstd_m, met)
            q = _get(freq_df, qrobf_m, met)
            color = FAMILY_COLORS.get(fam, "gray")
            ax.scatter([0], [k], color=color, marker="o", s=80, zorder=3)
            ax.scatter([1], [q], color=color, marker="s", s=80, zorder=3)
            if np.isfinite(k) and np.isfinite(q):
                ax.plot([0, 1], [k, q], color=color, linewidth=1.5,
                        alpha=0.8, label=fam)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["kfstd", "QROBF"], fontsize=10)
        ax.set_title(f"Freq: {title}", fontsize=10)
        if col == 0:
            ax.set_ylabel("Value")
        if col == 2:
            ax.legend(fontsize=7, loc="upper right")

    for col, (met, title) in enumerate(metrics_time):
        ax = axes[1][col]
        for fi, (fam, kfstd_m, qrobf_m) in enumerate(FAMILIES):
            k = _get(time_df, kfstd_m, met)
            q = _get(time_df, qrobf_m, met)
            color = FAMILY_COLORS.get(fam, "gray")
            ax.scatter([0], [k], color=color, marker="o", s=80, zorder=3)
            ax.scatter([1], [q], color=color, marker="s", s=80, zorder=3)
            if np.isfinite(k) and np.isfinite(q):
                ax.plot([0, 1], [k, q], color=color, linewidth=1.5, alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["kfstd", "QROBF"], fontsize=10)
        ax.set_title(f"Time: {title}", fontsize=10)
        if col == 0:
            ax.set_ylabel("Value")

    fig.suptitle("Fig 5: Dual-Domain Performance Matrix — kfstd vs QROBF",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig5_dual_domain.{ext}")
        fig.savefig(p)
        print(f"[Saved] {p}")
    plt.close(fig)


# ─── Figure 6: Comprehensive summary (paper figure) ─────────────────────────
def fig_paper_summary(run_dir: str, out_dir: str):
    freq_df = load_freq(run_dir)
    time_df = load_time(run_dir)
    diag_df = load_diag(run_dir)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

    families = [f[0] for f in FAMILIES]
    x = np.arange(len(families))
    w = 0.35

    # ── Panel A: Freq MAE ──
    ax_a = fig.add_subplot(gs[0, 0])
    kfstd_mae = [_get(freq_df, f[1], "MAE") for f in FAMILIES]
    qrobf_mae = [_get(freq_df, f[2], "MAE") for f in FAMILIES]
    ax_a.bar(x - w/2, kfstd_mae, w, color=C_KFSTD, label="kfstd", alpha=0.85)
    ax_a.bar(x + w/2, qrobf_mae, w, color=C_QROBF, label="QROBF", alpha=0.85)
    for i, (k, q) in enumerate(zip(kfstd_mae, qrobf_mae)):
        if np.isfinite(k) and np.isfinite(q) and k > 0:
            pct = 100 * (k - q) / k
            ax_a.text(x[i], max(k, q) + 0.005, f"{pct:+.1f}%",
                      ha="center", va="bottom", fontsize=7,
                      color=C_QROBF if pct > 0 else "gray", fontweight="bold")
    ax_a.set_xticks(x); ax_a.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    ax_a.set_ylabel("Freq MAE (BPM)"); ax_a.set_title("(A) Rate Accuracy", fontweight="bold")
    ax_a.legend(fontsize=7); ax_a.set_ylim(0, 0.45)

    # ── Panel B: Time CCC ──
    ax_b = fig.add_subplot(gs[0, 1])
    kfstd_ccc = [_get(time_df, f[1], "CCC") for f in FAMILIES]
    qrobf_ccc = [_get(time_df, f[2], "CCC") for f in FAMILIES]
    ax_b.bar(x - w/2, kfstd_ccc, w, color=C_KFSTD, alpha=0.85, label="kfstd")
    ax_b.bar(x + w/2, qrobf_ccc, w, color=C_QROBF, alpha=0.85, label="QROBF")
    for i, (k, q) in enumerate(zip(kfstd_ccc, qrobf_ccc)):
        if np.isfinite(k) and np.isfinite(q):
            ax_b.text(x[i], max(k, q) + 0.003, f"{q - k:+.3f}",
                      ha="center", va="bottom", fontsize=7,
                      color=C_QROBF if q >= k else "gray")
    ax_b.set_xticks(x); ax_b.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    ax_b.set_ylabel("CCC"); ax_b.set_title("(B) Waveform Fidelity", fontweight="bold")
    ax_b.set_ylim(0.6, 1.02); ax_b.legend(fontsize=7)

    # ── Panel C: NIS mean ──
    ax_c = fig.add_subplot(gs[0, 2])
    qrobf_methods = [f[2] for f in FAMILIES]
    nis_means = [_get(diag_df, m, "NIS_Mean") for m in qrobf_methods]
    coverage  = [_get(diag_df, m, "Coverage95") for m in qrobf_methods]
    bar_c = ax_c.bar(x, nis_means, color=[FAMILY_COLORS.get(f, C_QROBF) for f in families],
                     alpha=0.85)
    ax_c.axhline(1.0, color="k", linewidth=1.5, linestyle="--", label="Nominal")
    ax_c.axhspan(0.7, 1.3, alpha=0.08, color="green")
    for bar, v in zip(bar_c, nis_means):
        if np.isfinite(v):
            ax_c.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                      f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax_c.set_xticks(x); ax_c.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    ax_c.set_ylabel("NIS Mean"); ax_c.set_title("(C) Filter Calibration", fontweight="bold")
    ax_c.legend(fontsize=7); ax_c.set_ylim(0, 1.8)

    # ── Panel D: Coverage95 ──
    ax_d = fig.add_subplot(gs[1, 0])
    bar_d = ax_d.bar(x, coverage,
                     color=[FAMILY_COLORS.get(f, C_QROBF) for f in families], alpha=0.85)
    ax_d.axhline(95, color="k", linewidth=1.5, linestyle="--", label="Nominal (95%)")
    for bar, v in zip(bar_d, coverage):
        if np.isfinite(v):
            ax_d.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                      f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
    ax_d.set_xticks(x); ax_d.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    ax_d.set_ylabel("Coverage95 (%)"); ax_d.set_title("(D) 95% CI Coverage", fontweight="bold")
    ax_d.legend(fontsize=7); ax_d.set_ylim(80, 105)

    # ── Panel E: Freq RMSE ──
    ax_e = fig.add_subplot(gs[1, 1])
    kfstd_rmse = [_get(freq_df, f[1], "RMSE") for f in FAMILIES]
    qrobf_rmse = [_get(freq_df, f[2], "RMSE") for f in FAMILIES]
    ax_e.bar(x - w/2, kfstd_rmse, w, color=C_KFSTD, alpha=0.85, label="kfstd")
    ax_e.bar(x + w/2, qrobf_rmse, w, color=C_QROBF, alpha=0.85, label="QROBF")
    ax_e.set_xticks(x); ax_e.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    ax_e.set_ylabel("Freq RMSE (BPM)"); ax_e.set_title("(E) Rate RMSE", fontweight="bold")
    ax_e.legend(fontsize=7)

    # ── Panel F: Improvement heatmap ──
    ax_f = fig.add_subplot(gs[1, 2])
    metrics_pairs = [
        ("Freq MAE", [_get(freq_df, f[2], "MAE") / _get(freq_df, f[1], "MAE") - 1
                      for f in FAMILIES]),
        ("Freq RMSE", [_get(freq_df, f[2], "RMSE") / _get(freq_df, f[1], "RMSE") - 1
                       for f in FAMILIES]),
        ("CCC",      [_get(time_df, f[2], "CCC") / _get(time_df, f[1], "CCC") - 1
                      for f in FAMILIES]),
        ("Time MAE", [_get(time_df, f[2], "MAE") / _get(time_df, f[1], "MAE") - 1
                      for f in FAMILIES]),
    ]
    hmap = np.array([[v * 100 for v in vals] for _, vals in metrics_pairs])
    # Negate MAE/RMSE rows (lower is better → positive = improvement)
    for ri, (label, _) in enumerate(metrics_pairs):
        if "MAE" in label or "RMSE" in label:
            hmap[ri] *= -1

    im = ax_f.imshow(hmap, cmap="RdYlGn", aspect="auto",
                     vmin=-10, vmax=10)
    ax_f.set_xticks(x); ax_f.set_xticklabels(families, rotation=20, ha="right", fontsize=8)
    ax_f.set_yticks(range(len(metrics_pairs)))
    ax_f.set_yticklabels([m for m, _ in metrics_pairs], fontsize=8)
    for ri in range(hmap.shape[0]):
        for ci in range(hmap.shape[1]):
            v = hmap[ri, ci]
            if np.isfinite(v):
                ax_f.text(ci, ri, f"{v:+.1f}%", ha="center", va="center",
                          fontsize=7.5, fontweight="bold",
                          color="black" if abs(v) < 7 else "white")
    plt.colorbar(im, ax=ax_f, fraction=0.046, pad=0.04, label="% change\n(green=better)")
    ax_f.set_title("(F) QROBF vs kfstd Δ%", fontweight="bold")

    fig.suptitle(
        "QROBF: Quality-Aware Robust Oscillatory Bayesian Filter\n"
        "COHFACE Dataset — 160 Trials, 40 Subjects",
        fontsize=13, fontweight="bold", y=1.01
    )
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig_paper_summary.{ext}")
        fig.savefig(p)
        print(f"[Saved] {p}")
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/cohface_robust_ossm",
                        help="Path to run results directory")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: run-dir/plots/paper)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "plots", "paper")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    fig_overview(args.run_dir, out_dir)
    fig_freq_primary(args.run_dir, out_dir)
    fig_calibration(args.run_dir, out_dir)
    fig_tier_breakdown(args.run_dir, out_dir)
    fig_dual_domain_matrix(args.run_dir, out_dir)
    fig_paper_summary(args.run_dir, out_dir)
    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
