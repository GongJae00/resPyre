#!/usr/bin/env python3
"""
Generate paper-quality performance summary plots from run metrics CSVs.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────
# Publication style
# ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
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

# Colours
KFSTD_COLOR = "#3498db"
QROBF_COLOR = "#e74c3c"
BASE_COLOR  = "#95a5a6"

# Method-name abbreviation map for x-axis labels
_ABBREV = {
    "of_farneback":                       "OF-Farn.",
    "of_farneback__kfstd":                "OF\nkfstd",
    "of_farneback__robust_ossm_ekf":      "OF\nQROBF",
    "DoF":                                "DoF",
    "dof__kfstd":                         "DoF\nkfstd",
    "dof__robust_ossm_ekf":               "DoF\nQROBF",
    "profile1D linear":                   "P1D-Lin",
    "profile1d_linear__kfstd":            "Lin\nkfstd",
    "profile1d_linear__robust_ossm_ekf":  "Lin\nQROBF",
    "profile1D quadratic":                "P1D-Quad",
    "profile1d_quadratic__kfstd":         "Quad\nkfstd",
    "profile1d_quadratic__robust_ossm_ekf": "Quad\nQROBF",
    "profile1D cubic":                    "P1D-Cub",
    "profile1d_cubic__kfstd":             "Cub\nkfstd",
    "profile1d_cubic__robust_ossm_ekf":   "Cub\nQROBF",
}


def _abbrev(name: str) -> str:
    return _ABBREV.get(name, name.replace("__", "\n").replace("_", "-"))


def _extract_metric(df: pd.DataFrame, metric: str) -> np.ndarray:
    col = f"{metric}_median" if f"{metric}_median" in df.columns else metric
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)


# ─────────────────────────────────────────────────────────
# Figure A: Freq-domain comparison (paired kfstd vs QROBF)
# ─────────────────────────────────────────────────────────

# Families to compare (family_label, kfstd_method_name, qrobf_method_name, color)
_FAMILIES = [
    ("OF-Farneback",    "of_farneback__kfstd",             "of_farneback__robust_ossm_ekf",             "#2980b9"),
    ("P1D-Linear",      "profile1d_linear__kfstd",          "profile1d_linear__robust_ossm_ekf",          "#8e44ad"),
    ("P1D-Quadratic",   "profile1d_quadratic__kfstd",       "profile1d_quadratic__robust_ossm_ekf",       "#16a085"),
    ("P1D-Cubic",       "profile1d_cubic__kfstd",           "profile1d_cubic__robust_ossm_ekf",           "#d35400"),
]


def _get_paired_metrics(tdf, fdf, metric, domain="freq"):
    """Return (kfstd_vals, qrobf_vals) arrays aligned to _FAMILIES."""
    src = fdf if domain == "freq" else tdf
    col_med = f"{metric}_median" if f"{metric}_median" in src.columns else metric
    method_col = "Method" if "Method" in src.columns else "method"
    lookup = dict(zip(src[method_col].astype(str), pd.to_numeric(src[col_med], errors="coerce")))
    kf_vals, qr_vals = [], []
    for _, kf_m, qr_m, _ in _FAMILIES:
        kf_vals.append(lookup.get(kf_m, np.nan))
        qr_vals.append(lookup.get(qr_m, np.nan))
    return np.array(kf_vals), np.array(qr_vals)


def fig_freq_comparison(fdf: pd.DataFrame, out_dir: str) -> None:
    """Paired bar chart: kfstd vs QROBF freq-domain MAE + RMSE per family."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    x = np.arange(len(_FAMILIES))
    w = 0.38
    fam_labels = [f[0] for f in _FAMILIES]

    for ax, metric, title, unit in [
        (axes[0], "MAE",  "Freq-domain MAE",  "BPM"),
        (axes[1], "RMSE", "Freq-domain RMSE", "BPM"),
    ]:
        kf_vals, qr_vals = _get_paired_metrics(None, fdf, metric, "freq")

        bars_kf = ax.bar(x - w/2, kf_vals, w, label="kfstd",
                         color=KFSTD_COLOR, alpha=0.85, edgecolor="white")
        bars_qr = ax.bar(x + w/2, qr_vals, w, label="QROBF (ours)",
                         color=QROBF_COLOR, alpha=0.85, edgecolor="white", hatch="//")

        # Δ% annotations
        for xi, (kf, qr) in enumerate(zip(kf_vals, qr_vals)):
            if np.isfinite(kf) and np.isfinite(qr) and kf > 0:
                pct = (qr - kf) / kf * 100
                col = "#27ae60" if pct < 0 else "#c0392b"
                ymax = max(kf, qr) * 1.04
                ax.text(xi, ymax, f"{pct:+.1f}%",
                        ha="center", va="bottom", fontsize=8,
                        color=col, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(fam_labels, fontsize=9)
        ax.set_ylabel(f"{title} ({unit})", fontsize=11)
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.18)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=KFSTD_COLOR, label="kfstd (Gaussian oscillator)"),
        mpatches.Patch(color=QROBF_COLOR, hatch="//", label="QROBF (EKS + Student-t)"),
    ]
    fig.legend(handles=handles, fontsize=9, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Frequency-Domain Performance: QROBF vs kfstd (COHFACE, 160 trials, median)",
                 fontsize=12, y=1.06)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_freq_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def fig_time_comparison(tdf: pd.DataFrame, out_dir: str) -> None:
    """Paired bar chart: kfstd vs QROBF time-domain CCC + MAE per family."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    x = np.arange(len(_FAMILIES))
    w = 0.38
    fam_labels = [f[0] for f in _FAMILIES]

    for ax, metric, title, unit, higher_better in [
        (axes[0], "CCC", "Time-domain CCC",       "",    True),
        (axes[1], "MAE", "Time-domain MAE (waveform)", "BPM", False),
    ]:
        kf_vals, qr_vals = _get_paired_metrics(tdf, None, metric, "time")

        bars_kf = ax.bar(x - w/2, kf_vals, w, label="kfstd",
                         color=KFSTD_COLOR, alpha=0.85, edgecolor="white")
        bars_qr = ax.bar(x + w/2, qr_vals, w, label="QROBF (ours)",
                         color=QROBF_COLOR, alpha=0.85, edgecolor="white", hatch="//")

        # Δ% annotations
        for xi, (kf, qr) in enumerate(zip(kf_vals, qr_vals)):
            if np.isfinite(kf) and np.isfinite(qr) and kf > 0:
                if higher_better:
                    pct = (qr - kf) / kf * 100  # positive = QROBF better
                else:
                    pct = (qr - kf) / kf * 100  # positive = QROBF worse (higher MAE)
                col = "#27ae60" if (pct > 0 and higher_better) or (pct < 0 and not higher_better) else "#c0392b"
                ymax = max(kf, qr) * 1.04
                ax.text(xi, ymax, f"{pct:+.1f}%",
                        ha="center", va="bottom", fontsize=8,
                        color=col, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(fam_labels, fontsize=9)
        ylabel = f"{title}" + (f" ({unit})" if unit else "")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, pad=6)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.18)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=KFSTD_COLOR, label="kfstd (Gaussian oscillator)"),
        mpatches.Patch(color=QROBF_COLOR, hatch="//", label="QROBF (EKS + Student-t)"),
    ]
    fig.legend(handles=handles, fontsize=9, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Time-Domain Performance: QROBF vs kfstd (COHFACE, 160 trials, median)",
                 fontsize=12, y=1.06)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_time_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def fig_ranking(fdf: pd.DataFrame, out_dir: str) -> None:
    """Horizontal bar: all methods ranked by freq MAE."""
    method_col = "Method" if "Method" in fdf.columns else "method"
    col_med = "MAE_median" if "MAE_median" in fdf.columns else "MAE"

    methods = fdf[method_col].astype(str).tolist()
    mae_vals = pd.to_numeric(fdf[col_med], errors="coerce").to_numpy(dtype=np.float64)

    # Sort by MAE
    order = np.argsort(mae_vals)
    rank_labels = [_abbrev(methods[i]) for i in order]
    rank_vals = mae_vals[order]

    # Color by method type
    colors = []
    for m in [methods[i] for i in order]:
        if "robust_ossm" in m:
            colors.append(QROBF_COLOR)
        elif "kfstd" in m:
            colors.append(KFSTD_COLOR)
        else:
            colors.append(BASE_COLOR)

    n = len(rank_labels)
    fig_h = max(4.0, n * 0.45)
    fig, ax = plt.subplots(figsize=(7, fig_h))

    y = np.arange(n)
    bars = ax.barh(y, rank_vals, color=colors, edgecolor="white",
                   linewidth=0.5, alpha=0.88)

    # Value annotations
    for yi, v in zip(y, rank_vals):
        if np.isfinite(v):
            ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(rank_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Freq-domain MAE (BPM, lower is better)", fontsize=11)
    ax.set_title("All Methods Ranked by Frequency-Domain MAE\n(COHFACE, 160 trials, median)",
                 fontsize=12, pad=8)
    ax.xaxis.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(color=QROBF_COLOR, label="QROBF"),
        mpatches.Patch(color=KFSTD_COLOR, label="kfstd"),
        mpatches.Patch(color=BASE_COLOR,  label="base / other"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_ranking.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ─────────────────────────────────────────────────────────
# Legacy combined figure (kept for backwards compatibility)
# ─────────────────────────────────────────────────────────

def fig_combined_summary(tdf: pd.DataFrame, fdf: pd.DataFrame, out_dir: str) -> None:
    """2x2 summary overview; redesigned for paper quality."""
    method_col_t = "Method" if "Method" in tdf.columns else "method"
    method_col_f = "Method" if "Method" in fdf.columns else "method"

    methods = tdf[method_col_t].astype(str).tolist()
    x = np.arange(len(methods))
    width = 0.38

    mae_t  = _extract_metric(tdf, "MAE")
    rmse_t = _extract_metric(tdf, "RMSE")
    mae_f  = _extract_metric(fdf, "MAE")
    ccc_t  = _extract_metric(tdf, "CCC")

    abbrev_labels = [_abbrev(m) for m in methods]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()

    # Time-domain bars
    axes[0].bar(x - width / 2, mae_t,  width, label="Time MAE",  color="#2a9d8f", alpha=0.88)
    axes[0].bar(x + width / 2, rmse_t, width, label="Time RMSE", color="#264653", alpha=0.88)
    axes[0].set_title("Waveform Fidelity (Time Domain)", fontsize=12)
    axes[0].set_ylabel("BPM", fontsize=11)
    axes[0].yaxis.grid(True, alpha=0.3)
    axes[0].set_axisbelow(True)
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(abbrev_labels, fontsize=8, rotation=45, ha="right")

    # Freq-domain bars
    axes[1].bar(x, mae_f, width * 1.8, label="Freq MAE", color="#e76f51", alpha=0.88)
    axes[1].set_title("Rate Accuracy (Freq Domain)", fontsize=12)
    axes[1].set_ylabel("BPM", fontsize=11)
    axes[1].yaxis.grid(True, alpha=0.3)
    axes[1].set_axisbelow(True)
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(abbrev_labels, fontsize=8, rotation=45, ha="right")

    # Horizontal ranking by freq MAE
    order = np.argsort(mae_f)
    rank_labels = [abbrev_labels[i] for i in order]
    rank_vals = mae_f[order]
    rank_colors = [
        QROBF_COLOR if "QROBF" in rank_labels[i] or "robust" in methods[order[i]] else
        KFSTD_COLOR if "kfstd" in methods[order[i]] else
        BASE_COLOR
        for i in range(len(order))
    ]
    axes[2].barh(np.arange(len(rank_labels)), rank_vals,
                 color=rank_colors, alpha=0.88, edgecolor="white")
    axes[2].set_yticks(np.arange(len(rank_labels)))
    axes[2].set_yticklabels(rank_labels, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_title("Methods Ranked by Freq MAE (lower=better)", fontsize=12)
    axes[2].set_xlabel("Freq MAE (BPM)", fontsize=11)
    axes[2].xaxis.grid(True, alpha=0.3)
    axes[2].set_axisbelow(True)

    # Scatter: time MAE vs freq MAE
    scatter_colors = [
        QROBF_COLOR if "robust" in m else
        KFSTD_COLOR if "kfstd" in m else
        BASE_COLOR
        for m in methods
    ]
    axes[3].scatter(mae_t, mae_f, c=scatter_colors, s=60, alpha=0.85, zorder=3)
    for i, m in enumerate(methods):
        axes[3].annotate(_abbrev(m), (mae_t[i], mae_f[i]),
                         fontsize=7, alpha=0.75,
                         xytext=(3, 2), textcoords="offset points")
    axes[3].set_xlabel("Time MAE (BPM)", fontsize=11)
    axes[3].set_ylabel("Freq MAE (BPM)", fontsize=11)
    axes[3].set_title("Time vs Freq MAE Trade-off", fontsize=12)
    axes[3].grid(alpha=0.3)
    axes[3].set_axisbelow(True)

    handles = [
        mpatches.Patch(color=QROBF_COLOR, label="QROBF"),
        mpatches.Patch(color=KFSTD_COLOR, label="kfstd"),
        mpatches.Patch(color=BASE_COLOR,  label="base"),
    ]
    axes[3].legend(handles=handles, fontsize=8, loc="upper left")

    run_name = os.path.basename(
        os.path.dirname(os.path.dirname(out_dir)) or out_dir
    )
    fig.suptitle(f"Performance Summary — {run_name}", fontsize=13, y=1.005)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path = os.path.join(out_dir, "paper_performance_summary.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="results/<run_name>")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    t_csv = os.path.join(run_dir, "metrics", "metrics_time_domain_summary.csv")
    f_csv = os.path.join(run_dir, "metrics", "metrics_freq_domain_summary.csv")
    if not (os.path.exists(t_csv) and os.path.exists(f_csv)):
        raise FileNotFoundError("Missing time/freq summary CSVs.")

    tdf = pd.read_csv(t_csv)
    fdf = pd.read_csv(f_csv)
    if "Method" not in tdf.columns:
        tdf["Method"] = tdf["method"]
    if "Method" not in fdf.columns:
        fdf["Method"] = fdf["method"]

    fig_freq_comparison(fdf, out_dir)
    fig_time_comparison(tdf, out_dir)
    fig_ranking(fdf, out_dir)
    fig_combined_summary(tdf, fdf, out_dir)


if __name__ == "__main__":
    main()
