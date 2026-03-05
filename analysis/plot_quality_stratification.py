"""Paper-quality visualizations for Quality Stratification Analysis.

Generates 5 figures for the COHFACE/QROBF paper:
  1. SNR distribution with tier boundaries
  2. Per-tier freq_mae grouped bar chart (all families)
  3. Relative improvement (QROBF vs kfstd) per tier
  4. Box plots per tier (kfstd vs QROBF)
  5. Scatter: SNR vs freq_mae (kfstd vs QROBF)

Usage:
    python analysis/plot_quality_stratification.py \\
        --csv results/cohface_robust_ossm/plots/quality_stratification/trial_stratification.csv \\
        --output results/cohface_robust_ossm/plots/quality_stratification/

Dependencies: matplotlib, seaborn, pandas, numpy, scipy
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────

TIER_LABELS  = ["Very Poor", "Poor", "Fair", "Good"]
TIER_ABBR    = ["VP", "P", "F", "G"]
TIER_COLORS  = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
TIER_ALPHAS  = [0.18, 0.18, 0.18, 0.18]

# Method families: (label, kfstd_col_base, qrobf_col_base, color, marker)
# The _eval suffix variants take priority if present in the DataFrame.
_FAMILIES_BASE = [
    ("OF",       "freq_mae_OF_kfstd",            "freq_mae_OF_qrobf",            "#2980b9", "o"),
    ("P1D-Lin",  "freq_mae_P1D-Linear_kfstd",    "freq_mae_P1D-Linear_qrobf",    "#8e44ad", "s"),
    ("P1D-Quad", "freq_mae_P1D-Quad_kfstd",      "freq_mae_P1D-Quad_qrobf",      "#16a085", "^"),
    ("P1D-Cub",  "freq_mae_P1D-Cubic_kfstd",     "freq_mae_P1D-Cubic_qrobf",     "#d35400", "D"),
    ("DoF",      "freq_mae_DoF_kfstd",            "freq_mae_DoF_qrobf",            "#7f8c8d", "v"),
]
BASE_COLS_BASE = {
    "OF":       "freq_mae_OF_base",
    "P1D-Lin":  "freq_mae_P1D-Linear_base",
    "P1D-Quad": "freq_mae_P1D-Quad_base",
    "P1D-Cub":  "freq_mae_P1D-Cubic_base",
    "DoF":      "freq_mae_DoF_base",
}


def _resolve_families(df: pd.DataFrame):
    """Return (FAMILIES, BASE_COLS) using _eval variants if available."""
    families = []
    base_cols = {}
    for (label, kf_col, qr_col, color, marker) in _FAMILIES_BASE:
        kf = f"{kf_col}_eval" if f"{kf_col}_eval" in df.columns else kf_col
        qr = f"{qr_col}_eval" if f"{qr_col}_eval" in df.columns else qr_col
        families.append((label, kf, qr, color, marker))
        base_raw = BASE_COLS_BASE[label]
        base_cols[label] = f"{base_raw}_eval" if f"{base_raw}_eval" in df.columns else base_raw
    return families, base_cols


FAMILIES    = _FAMILIES_BASE       # will be overridden per-call via _resolve_families
BASE_COLS   = BASE_COLS_BASE

KFSTD_COLOR = "#3498db"
QROBF_COLOR = "#e74c3c"
BASE_COLOR  = "#95a5a6"

# Publication font sizes
TITLE_FS  = 12
LABEL_FS  = 11
TICK_FS   = 9
LEGEND_FS = 9

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "0.8",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _tier_counts(df: pd.DataFrame) -> dict[str, int]:
    return {t: int((df["tier"] == t).sum()) for t in TIER_LABELS}


def _tier_stats(df: pd.DataFrame, col: str) -> dict[str, dict]:
    out = {}
    for t in TIER_LABELS:
        sub  = df.loc[df["tier"] == t, col].dropna()
        out[t] = {
            "mean":   float(sub.mean()) if len(sub) else np.nan,
            "median": float(sub.median()) if len(sub) else np.nan,
            "std":    float(sub.std())  if len(sub) else np.nan,
            "sem":    float(sub.sem())  if len(sub) else np.nan,
            "n":      len(sub),
            "vals":   sub.values,
        }
    return out


def _add_tier_bands(ax: plt.Axes, df: pd.DataFrame,
                    x_col: str = "snr_of_db") -> None:
    """Add colored background bands for each tier."""
    snr = df["snr_of_db"].dropna()
    bounds = [snr.quantile(q) for q in [0.0, 0.20, 0.40, 0.70, 1.00]]
    xmin, xmax = ax.get_xlim()
    for i in range(len(TIER_LABELS)):
        ax.axvspan(bounds[i], bounds[i + 1],
                   color=TIER_COLORS[i], alpha=0.12, zorder=0)
        ax.axvline(bounds[i + 1], color=TIER_COLORS[i],
                   lw=0.8, ls="--", alpha=0.6, zorder=1)


def _significance_bar(ax: plt.Axes, x0: float, x1: float,
                      y: float, p: float, h: float = 0.04) -> None:
    """Draw a significance bracket between x0 and x1."""
    if p < 0.001:
        label = "***"
    elif p < 0.01:
        label = "**"
    elif p < 0.05:
        label = "*"
    else:
        label = "ns"
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle="-", color="0.3", lw=0.8),
    )
    ax.text((x0 + x1) / 2, y + h, label, ha="center", va="bottom",
            fontsize=7, color="0.3")


# ─────────────────────────────────────────────────────────
# Figure 1: SNR Distribution with tier boundaries
# ─────────────────────────────────────────────────────────

def fig_snr_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram of OF-SNR coloured by tier."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    snr = df["snr_of_db"].dropna()
    bins = np.linspace(snr.min() - 0.5, snr.max() + 0.5, 35)

    for tier, color in zip(TIER_LABELS, TIER_COLORS):
        sub = df.loc[df["tier"] == tier, "snr_of_db"].dropna()
        ax.hist(sub, bins=bins, color=color, edgecolor="white",
                linewidth=0.5, alpha=0.85, label=tier)

    # Tier boundary lines
    bounds = [snr.quantile(q) for q in [0.20, 0.40, 0.70]]
    for b, lbl in zip(bounds, ["20th", "40th", "70th"]):
        ax.axvline(b, color="0.35", lw=1.0, ls="--")
        ax.text(b + 0.15, ax.get_ylim()[1] * 0.92, lbl,
                fontsize=7.5, color="0.35", va="top")

    ax.set_xlabel("Spectral SNR of OF Signal (dB)", fontsize=LABEL_FS)
    ax.set_ylabel("Number of Trials", fontsize=LABEL_FS)
    ax.set_title("Signal Quality Distribution (N=160 COHFACE Trials)",
                 fontsize=TITLE_FS, pad=8)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, ncol=2, loc="upper left")

    counts = _tier_counts(df)
    info = "  ".join(f"{t}: n={counts[t]}" for t in TIER_LABELS)
    ax.text(0.99, 0.97, info, transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="0.4")

    plt.tight_layout()
    fig.savefig(out / "fig1_snr_distribution.pdf")
    fig.savefig(out / "fig1_snr_distribution.png")
    plt.close(fig)
    print(f"[plot] Fig1 saved.")


# ─────────────────────────────────────────────────────────
# Figure 2: Per-tier freq_mae grouped bar chart (OF family focus)
# ─────────────────────────────────────────────────────────

def fig_tier_bar_of(df: pd.DataFrame, out: Path) -> None:
    """Grouped bars: base vs kfstd vs QROBF per tier (OF family)."""
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    n_tiers = len(TIER_LABELS)
    w = 0.25
    x = np.arange(n_tiers)

    for i, (col, label, color, hatch) in enumerate([
        ("freq_mae_OF_base",   "Base (OF)",   BASE_COLOR,  ""),
        ("freq_mae_OF_kfstd",  "kfstd",       KFSTD_COLOR, ""),
        ("freq_mae_OF_qrobf",  "QROBF (ours)", QROBF_COLOR, "//"),
    ]):
        means = [_tier_stats(df, col)[t]["mean"] for t in TIER_LABELS]
        sems  = [_tier_stats(df, col)[t]["sem"]  for t in TIER_LABELS]
        bars = ax.bar(x + (i - 1) * w, means, w,
                      label=label, color=color, hatch=hatch,
                      edgecolor="white" if not hatch else color,
                      linewidth=0.6, alpha=0.9)
        ax.errorbar(x + (i - 1) * w, means, yerr=sems,
                    fmt="none", color="0.25", capsize=3, linewidth=1.0, zorder=5)

    # Tier counts
    counts = _tier_counts(df)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{t}\n(n={counts[t]})" for t in TIER_LABELS],
        fontsize=TICK_FS,
    )
    ax.set_ylabel("Freq MAE (BPM)", fontsize=LABEL_FS)
    ax.set_title("Respiratory Rate MAE by Signal Quality Tier — OF Family",
                 fontsize=TITLE_FS, pad=8)
    ax.legend(fontsize=LEGEND_FS, ncol=3, loc="upper right")
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    fig.savefig(out / "fig2_tier_bar_OF.pdf")
    fig.savefig(out / "fig2_tier_bar_OF.png")
    plt.close(fig)
    print(f"[plot] Fig2 saved.")


# ─────────────────────────────────────────────────────────
# Figure 3: All families — per-tier freq_mae (kfstd vs QROBF only)
# ─────────────────────────────────────────────────────────

def fig_tier_bar_all_families(df: pd.DataFrame, out: Path) -> None:
    """4-panel grouped bar (one per tier), all families, kfstd vs QROBF."""
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), sharey=False)

    family_labels = [f[0] for f in FAMILIES]
    x = np.arange(len(FAMILIES))
    w = 0.38

    for ax, tier, color in zip(axes, TIER_LABELS, TIER_COLORS):
        kf_means, kf_sems, qr_means, qr_sems = [], [], [], []
        for fam, kf_col, qr_col, _, _ in FAMILIES:
            k = _tier_stats(df, kf_col)[tier]
            q = _tier_stats(df, qr_col)[tier]
            kf_means.append(k["mean"])
            kf_sems.append(k["sem"])
            qr_means.append(q["mean"])
            qr_sems.append(q["sem"])

        ax.bar(x - w / 2, kf_means, w, label="kfstd",
               color=KFSTD_COLOR, alpha=0.85, edgecolor="white")
        ax.bar(x + w / 2, qr_means, w, label="QROBF",
               color=QROBF_COLOR, alpha=0.85, edgecolor="white", hatch="//")
        ax.errorbar(x - w / 2, kf_means, yerr=kf_sems,
                    fmt="none", color="0.25", capsize=2.5, lw=0.9, zorder=5)
        ax.errorbar(x + w / 2, qr_means, yerr=qr_sems,
                    fmt="none", color="0.25", capsize=2.5, lw=0.9, zorder=5)

        # % improvement annotation
        for xi, (kf, qr) in enumerate(zip(kf_means, qr_means)):
            if kf > 0 and np.isfinite(kf) and np.isfinite(qr):
                imp = (kf - qr) / kf * 100
                col = "#27ae60" if imp >= 0 else "#c0392b"
                ymax = max(kf, qr)
                ax.text(xi, ymax * 1.05, f"{imp:+.0f}%",
                        ha="center", va="bottom", fontsize=6.5,
                        color=col, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(family_labels, fontsize=7, rotation=30, ha="right")
        ax.set_title(f"{tier}", fontsize=LABEL_FS,
                     color=color, fontweight="bold")
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_ylim(bottom=0)
        counts = int((df["tier"] == tier).sum())
        ax.text(0.97, 0.97, f"n={counts}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="0.4")

    axes[0].set_ylabel("Freq MAE (BPM)", fontsize=LABEL_FS)
    handles = [
        mpatches.Patch(color=KFSTD_COLOR, label="kfstd"),
        mpatches.Patch(color=QROBF_COLOR, hatch="//", label="QROBF (ours)"),
    ]
    fig.legend(handles=handles, fontsize=LEGEND_FS, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("RR Freq MAE per Quality Tier — All Signal Families",
                 fontsize=TITLE_FS + 0.5, y=1.06)

    plt.tight_layout()
    fig.savefig(out / "fig3_tier_bar_all_families.pdf", bbox_inches="tight")
    fig.savefig(out / "fig3_tier_bar_all_families.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Fig3 saved.")


# ─────────────────────────────────────────────────────────
# Figure 4: Relative improvement (QROBF vs kfstd) per tier
# ─────────────────────────────────────────────────────────

def fig_relative_improvement(df: pd.DataFrame, out: Path) -> None:
    """Bar chart of % improvement (kfstd→QROBF) per tier, all families."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))

    n_tiers = len(TIER_LABELS)
    n_fam   = len(FAMILIES)
    x       = np.arange(n_tiers)
    group_w = 0.8
    w       = group_w / n_fam

    for i, (fam, kf_col, qr_col, color, _) in enumerate(FAMILIES):
        imps = []
        for tier in TIER_LABELS:
            kf = _tier_stats(df, kf_col)[tier]["mean"]
            qr = _tier_stats(df, qr_col)[tier]["mean"]
            if kf > 0 and np.isfinite(kf) and np.isfinite(qr):
                imps.append((kf - qr) / kf * 100)
            else:
                imps.append(np.nan)
        offset = (i - n_fam / 2 + 0.5) * w
        ax.bar(x + offset, imps, w, label=fam,
               color=color, edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.axhline(0, color="0.2", lw=0.9, ls="-")
    ax.set_xticks(x)
    counts = _tier_counts(df)
    ax.set_xticklabels(
        [f"{t}\n(n={counts[t]})" for t in TIER_LABELS],
        fontsize=TICK_FS,
    )
    ax.set_ylabel("QROBF Improvement over kfstd (%)", fontsize=LABEL_FS)
    ax.set_title("Relative Improvement: QROBF vs kfstd by Signal Quality",
                 fontsize=TITLE_FS, pad=8)
    ax.legend(fontsize=LEGEND_FS, ncol=3, loc="upper left")
    ax.tick_params(labelsize=TICK_FS)

    # Zero line annotation
    ax.text(n_tiers - 0.05, 0.8, "← QROBF worse | QROBF better →",
            ha="right", va="bottom", fontsize=7, color="0.5",
            rotation=0)

    plt.tight_layout()
    fig.savefig(out / "fig4_relative_improvement.pdf")
    fig.savefig(out / "fig4_relative_improvement.png")
    plt.close(fig)
    print(f"[plot] Fig4 saved.")


# ─────────────────────────────────────────────────────────
# Figure 5: Box plots per tier (kfstd vs QROBF, OF + P1D families)
# ─────────────────────────────────────────────────────────

def fig_boxplots(df: pd.DataFrame, out: Path) -> None:
    """Box plot of per-trial freq_mae for kfstd vs QROBF per tier."""
    # Show profile1D families (most interesting)
    selected = [
        ("OF",      "freq_mae_OF_kfstd",         "freq_mae_OF_qrobf"),
        ("P1D-Lin", "freq_mae_P1D-Linear_kfstd",  "freq_mae_P1D-Linear_qrobf"),
        ("P1D-Cub", "freq_mae_P1D-Cubic_kfstd",   "freq_mae_P1D-Cubic_qrobf"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharey=False)

    for ax, (fam, kf_col, qr_col) in zip(axes, selected):
        positions_kf, positions_qr = [], []
        data_kf, data_qr = [], []
        xtick_pos, xtick_labels = [], []

        for j, tier in enumerate(TIER_LABELS):
            sub = df[df["tier"] == tier]
            kf_vals = sub[kf_col].dropna().values
            qr_vals = sub[qr_col].dropna().values

            base_x = j * 3.0
            positions_kf.append(base_x)
            positions_qr.append(base_x + 1.0)
            data_kf.append(kf_vals)
            data_qr.append(qr_vals)
            xtick_pos.append(base_x + 0.5)
            xtick_labels.append(f"{TIER_ABBR[j]}\n(n={len(kf_vals)})")

        def _box_style(bp, color):
            for patch in bp.get("boxes", []):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            for med in bp.get("medians", []):
                med.set_color("black")
                med.set_linewidth(1.5)
            for flier in bp.get("fliers", []):
                flier.set(marker="o", markersize=3.5,
                          alpha=0.5, markerfacecolor=color,
                          markeredgecolor="none")

        bp_kf = ax.boxplot(data_kf, positions=positions_kf, widths=0.8,
                           patch_artist=True, notch=False,
                           showfliers=True, zorder=3)
        bp_qr = ax.boxplot(data_qr, positions=positions_qr, widths=0.8,
                           patch_artist=True, notch=False,
                           showfliers=True, zorder=3)
        _box_style(bp_kf, KFSTD_COLOR)
        _box_style(bp_qr, QROBF_COLOR)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, fontsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_title(f"Family: {fam}", fontsize=LABEL_FS, pad=6)
        if fam == "OF":
            ax.set_ylabel("Freq MAE (BPM)", fontsize=LABEL_FS)

    handles = [
        mpatches.Patch(color=KFSTD_COLOR, alpha=0.75, label="kfstd"),
        mpatches.Patch(color=QROBF_COLOR, alpha=0.75, label="QROBF (ours)"),
    ]
    fig.legend(handles=handles, fontsize=LEGEND_FS, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Per-Trial Freq MAE Distribution per Quality Tier",
                 fontsize=TITLE_FS, y=1.05)
    plt.tight_layout()
    fig.savefig(out / "fig5_boxplots.pdf", bbox_inches="tight")
    fig.savefig(out / "fig5_boxplots.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Fig5 saved.")


# ─────────────────────────────────────────────────────────
# Figure 6: Scatter SNR vs freq_mae (kfstd vs QROBF, P1D-Cubic)
# ─────────────────────────────────────────────────────────

def fig_scatter_snr(df: pd.DataFrame, out: Path) -> None:
    """Scatter: OF-SNR vs per-trial freq_mae, kfstd and QROBF (P1D-Cubic)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for ax, (col, label, color) in zip(axes, [
        ("freq_mae_P1D-Cubic_kfstd",  "kfstd",        KFSTD_COLOR),
        ("freq_mae_P1D-Cubic_qrobf",  "QROBF (ours)", QROBF_COLOR),
    ]):
        snr  = df["snr_of_db"].values
        mae  = df[col].values
        tier = df["tier"].values

        for t, tc in zip(TIER_LABELS, TIER_COLORS):
            mask = tier == t
            ax.scatter(snr[mask], mae[mask], c=tc, s=22, alpha=0.7,
                       edgecolors="none", label=t, zorder=3)

        # Regression line
        valid = np.isfinite(snr) & np.isfinite(mae)
        if valid.sum() > 5:
            slope, intercept, r, p_val, _ = scipy_stats.linregress(
                snr[valid], mae[valid])
            x_line = np.linspace(snr[valid].min(), snr[valid].max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=color, lw=1.8, ls="--", alpha=0.9,
                    label=f"r={r:.2f} (p={p_val:.3f})")

        ax.set_xlabel("Spectral SNR (dB)", fontsize=LABEL_FS)
        ax.set_ylabel("Freq MAE (BPM)", fontsize=LABEL_FS)
        ax.set_title(f"P1D-Cubic: {label}", fontsize=LABEL_FS, pad=6)
        ax.tick_params(labelsize=TICK_FS)
        ax.legend(fontsize=LEGEND_FS - 1, ncol=2)

        # Tier boundaries
        snr_series = df["snr_of_db"].dropna()
        for q, tc in zip([0.20, 0.40, 0.70], TIER_COLORS[:-1]):
            ax.axvline(snr_series.quantile(q), color=tc,
                       lw=0.8, ls=":", alpha=0.7, zorder=1)

    fig.suptitle("Signal SNR vs. RR Estimation Error (P1D-Cubic family)",
                 fontsize=TITLE_FS + 0.5, y=1.02)
    plt.tight_layout()
    fig.savefig(out / "fig6_scatter_snr_vs_mae.pdf", bbox_inches="tight")
    fig.savefig(out / "fig6_scatter_snr_vs_mae.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Fig6 saved.")


# ─────────────────────────────────────────────────────────
# Figure 7: Summary heatmap — % improvement per tier × family
# ─────────────────────────────────────────────────────────

def fig_improvement_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Heatmap: % improvement (QROBF vs kfstd) for tier × family."""
    fam_labels = [f[0] for f in FAMILIES]
    data_matrix = np.zeros((len(TIER_LABELS), len(FAMILIES)))

    for ti, tier in enumerate(TIER_LABELS):
        for fi, (fam, kf_col, qr_col, _, _) in enumerate(FAMILIES):
            kf = _tier_stats(df, kf_col)[tier]["mean"]
            qr = _tier_stats(df, qr_col)[tier]["mean"]
            if kf > 0 and np.isfinite(kf) and np.isfinite(qr):
                data_matrix[ti, fi] = (kf - qr) / kf * 100
            else:
                data_matrix[ti, fi] = np.nan

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    # Symmetric colormap centred on 0
    vmax = np.nanmax(np.abs(data_matrix))
    vmax = max(vmax, 5.0)
    im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto")

    # Annotations
    for ti in range(len(TIER_LABELS)):
        for fi in range(len(FAMILIES)):
            val = data_matrix[ti, fi]
            if np.isfinite(val):
                txt = f"{val:+.1f}%"
                textcolor = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(fi, ti, txt, ha="center", va="center",
                        fontsize=8.5, fontweight="bold", color=textcolor)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("% Improvement (QROBF vs kfstd)", fontsize=LEGEND_FS)
    cbar.ax.tick_params(labelsize=TICK_FS - 1)

    ax.set_xticks(range(len(fam_labels)))
    ax.set_xticklabels(fam_labels, fontsize=TICK_FS)
    ax.set_yticks(range(len(TIER_LABELS)))
    counts = _tier_counts(df)
    ax.set_yticklabels(
        [f"{t} (n={counts[t]})" for t in TIER_LABELS],
        fontsize=TICK_FS,
    )
    ax.set_title("QROBF vs kfstd: % Improvement by Tier × Signal Family",
                 fontsize=TITLE_FS, pad=8)

    plt.tight_layout()
    fig.savefig(out / "fig7_improvement_heatmap.pdf")
    fig.savefig(out / "fig7_improvement_heatmap.png")
    plt.close(fig)
    print(f"[plot] Fig7 saved.")


# ─────────────────────────────────────────────────────────
# Print text summary
# ─────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print(" Quality Stratification Summary")
    print("=" * 70)

    # Tier distribution
    print("\n── Tier Distribution ──────────────────────")
    for t, color in zip(TIER_LABELS, TIER_COLORS):
        sub  = df[df["tier"] == t]
        snr  = sub["snr_of_db"].dropna()
        print(f"  {t:12s}: n={len(sub):3d}  "
              f"SNR [{snr.min():+6.1f}, {snr.max():+6.1f}] dB  "
              f"mean={snr.mean():+6.1f}")

    # Per-family improvement table
    print("\n── base / kfstd / QROBF by Tier (mean freq_mae, BPM) ──")
    for fam, kf_col, qr_col, _, _ in FAMILIES:
        base_col = BASE_COLS.get(fam, "")
        has_base = bool(base_col) and base_col in df.columns
        print(f"\n  Family: {fam}")
        if has_base:
            print(f"  {'Tier':12s} {'base':>8s} {'kfstd':>8s} {'QROBF':>8s} {'Δ (BPM)':>9s} {'Δ%':>8s}")
        else:
            print(f"  {'Tier':12s} {'kfstd':>8s} {'QROBF':>8s} {'Δ (BPM)':>9s} {'Δ%':>8s}")
        for tier in TIER_LABELS:
            base = _tier_stats(df, base_col)[tier]["mean"] if has_base else float("nan")
            kf = _tier_stats(df, kf_col)[tier]["mean"]
            qr = _tier_stats(df, qr_col)[tier]["mean"]
            imp_abs = kf - qr
            imp_pct = imp_abs / kf * 100 if (kf > 0 and np.isfinite(kf)) else np.nan
            if has_base:
                base_str = f"{base:8.3f}" if np.isfinite(base) else "     N/A"
                print(f"  {tier:12s} {base_str} {kf:8.3f} {qr:8.3f} "
                      f"{imp_abs:+9.3f} {imp_pct:+7.1f}%")
            else:
                print(f"  {tier:12s} {kf:8.3f} {qr:8.3f} "
                      f"{imp_abs:+9.3f} {imp_pct:+7.1f}%")

    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def run_plots(csv_path: str,
              output_dir: Optional[str] = None) -> None:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Enforce tier order
    tier_order = {t: i for i, t in enumerate(TIER_LABELS)}
    df["tier_order"] = df["tier"].map(tier_order).fillna(99)
    df = df.sort_values("tier_order").reset_index(drop=True)

    out = Path(output_dir) if output_dir else csv_path.parent
    out.mkdir(parents=True, exist_ok=True)

    # Resolve which metric columns to use (_eval preferred)
    global FAMILIES, BASE_COLS
    FAMILIES, BASE_COLS = _resolve_families(df)
    has_eval = any(c.endswith("_eval") for c in [f[1] for f in FAMILIES])
    metric_src = "evaluation pipeline" if has_eval else "proxy (windowed FFT on signal_hat)"

    print(f"[plot] Input : {csv_path} ({len(df)} rows)")
    print(f"[plot] Output: {out}")
    print(f"[plot] Metric source: {metric_src}")

    print_summary(df)
    fig_snr_distribution(df, out)
    fig_tier_bar_of(df, out)
    fig_tier_bar_all_families(df, out)
    fig_relative_improvement(df, out)
    fig_boxplots(df, out)
    fig_scatter_snr(df, out)
    fig_improvement_heatmap(df, out)

    print(f"\n[plot] All figures saved to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot quality stratification figures")
    parser.add_argument(
        "--csv", required=True,
        help="Path to trial_stratification.csv")
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: same directory as CSV)")
    args = parser.parse_args()
    run_plots(args.csv, args.output)
