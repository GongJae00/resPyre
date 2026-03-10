#!/usr/bin/env python3
"""
Generate mechanism figures for the QROBF paper.

Fig 2: Quality-to-trust deterministic mapping response curves
Fig 4: Student-t robust update mechanism (lambda curve, gain suppression)
Fig 7: Ablation visualization (EKS vs Student-t contributions)
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


# ─── Fig 2: Quality-to-trust mapping ────────────────────────────────────────
def fig_trust_mapping(out_dir: str):
    """Illustrate how 6D quality signals map to trust parameters."""

    fig = plt.figure(figsize=(13, 7))
    gs = gridspec.GridSpec(2, 3, hspace=0.55, wspace=0.38)

    # Validated regime params (from production config)
    beta_1 = 0.0; beta_2 = 0.0      # alpha_R: disabled
    gamma_1 = 1.9                    # alpha_Q: active
    w_gate_vis = 2.0; w_gate_cons = 1.5
    gate_bias = -100.0               # g_t: disabled
    freq_jitter_decay = 0.62
    thd_max = 0.52; w_h_min = 0.38

    # Active-gating regime (default TrustConfig)
    beta_1_ag = 2.0; beta_2_ag = 1.5
    gate_bias_ag = 1.0

    q = np.linspace(0, 1, 200)

    # Panel 1: alpha_R vs q_out (validated regime: always 1.0; active regime: >1)
    ax = fig.add_subplot(gs[0, 0])
    alpha_R_val = np.ones_like(q)  # alpha_R_max=1.0 → clamped
    alpha_R_ag  = np.clip(1 + beta_1_ag * q + beta_2_ag * 0.2, 1, 20)
    ax.plot(q, alpha_R_val, color="#e74c3c", lw=2.5, label="Validated ($\\alpha_R$=1.0 fixed)")
    ax.plot(q, alpha_R_ag, color="#e74c3c", lw=2.5, ls="--", label="Active-Gating")
    ax.set_xlabel("$q_{\\mathrm{out}}$ (outlier score)")
    ax.set_ylabel("$\\alpha_R$ (R-scale)")
    ax.set_title("Rule 1: R-Scale ($\\alpha_R$)", fontweight="bold")
    ax.legend(); ax.set_ylim(0.9, 4.5)
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")

    # Panel 2: alpha_Q vs q_drift (Rule 2: active in both regimes)
    ax = fig.add_subplot(gs[0, 1])
    alpha_Q = np.clip(1 + gamma_1 * q, 1, 5)
    ax.plot(q, alpha_Q, color="#8e44ad", lw=2.5, label=f"$\\gamma_1$={gamma_1}")
    ax.fill_between(q, 1, alpha_Q, alpha=0.12, color="#8e44ad")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("$q_{\\mathrm{drift}}$ (drift score)")
    ax.set_ylabel("$\\alpha_Q$ (Q-scale)")
    ax.set_title("Rule 2: Q-Scale ($\\alpha_Q$) — Active", fontweight="bold")
    ax.legend(); ax.set_ylim(0.9, 5.2)

    # Panel 3: g_t vs q_vis (Rule 3: validated regime always ~1)
    ax = fig.add_subplot(gs[0, 2])
    logit_val = w_gate_vis * q + w_gate_cons * 0.8 - 0.5 * 0.5 - gate_bias
    logit_ag  = w_gate_vis * q + w_gate_cons * 0.8 - 0.5 * 0.5 - gate_bias_ag
    g_t_val = sigmoid(logit_val)
    g_t_ag  = sigmoid(logit_ag)
    ax.plot(q, g_t_val, color="#27ae60", lw=2.5, label="Validated ($\\approx$1.0)")
    ax.plot(q, g_t_ag,  color="#27ae60", lw=2.5, ls="--", label="Active-Gating")
    ax.set_xlabel("$q_{\\mathrm{vis}}$ (visibility)")
    ax.set_ylabel("$g_t$ (observation gate)")
    ax.set_title("Rule 3: Observation Gate ($g_t$)", fontweight="bold")
    ax.legend(); ax.set_ylim(-0.02, 1.05)
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")

    # Panel 4: g_z vs freq jitter (Rule 4: active via jitter)
    ax = fig.add_subplot(gs[1, 0])
    jitter = np.linspace(0, 1, 200)
    g_z = np.clip(1 * (1 - freq_jitter_decay * jitter), 0, 1)  # g_t=1 in validated
    g_z_floor = np.maximum(g_z, 0.2 * 1)  # g_z_floor=0.2 * g_t
    ax.plot(jitter, g_z,       color="#2980b9", lw=2.5, label="$g_z$ raw")
    ax.plot(jitter, g_z_floor, color="#2980b9", lw=2.5, ls="--",
            label=f"$g_z$ with floor (ratio=0.2)")
    ax.fill_between(jitter, g_z, g_z_floor, alpha=0.15, color="#2980b9")
    ax.set_xlabel("Freq jitter (relative $|\\Delta f|/f$)")
    ax.set_ylabel("$g_z$ (frequency gate)")
    ax.set_title("Rule 4: Freq Gate ($g_z$) — Partially Active", fontweight="bold")
    ax.legend()

    # Panel 5: w_h vs q_harm (Rule 5: active)
    ax = fig.add_subplot(gs[1, 1])
    w_h_raw = 1 - np.clip(q / thd_max, 0, 1)
    w_h = np.maximum(w_h_raw, w_h_min)
    ax.plot(q, w_h_raw, color="#d35400", lw=2.5, ls="--", label="$w_h$ raw")
    ax.plot(q, w_h, color="#d35400", lw=2.5, label=f"$w_h$ with floor={w_h_min}")
    ax.axvline(thd_max, color="gray", lw=1, ls=":", label=f"$\\theta_{{max}}$={thd_max}")
    ax.fill_between(q, w_h_raw, w_h, alpha=0.12, color="#d35400")
    ax.set_xlabel("$q_{\\mathrm{harm}}$ (THD score)")
    ax.set_ylabel("$w_h$ (harmonic weight)")
    ax.set_title("Rule 5: Harmonic Suppression ($w_h$) — Active", fontweight="bold")
    ax.legend()

    # Panel 6: Summary regime comparison
    ax = fig.add_subplot(gs[1, 2])
    rules = ["Rule 1\n$\\alpha_R$", "Rule 2\n$\\alpha_Q$", "Rule 3\n$g_t$",
             "Rule 4\n$g_z$", "Rule 5\n$w_h$"]
    validated = [0.0, 1.0, 0.0, 0.5, 1.0]   # 0=disabled, 0.5=partial, 1=full
    active_gating = [1.0, 1.0, 1.0, 1.0, 1.0]
    x = np.arange(len(rules))
    w = 0.32
    bars_v = ax.bar(x - w/2, validated,    w, color="#e74c3c", alpha=0.85,
                    label="Validated (no-gating)")
    bars_a = ax.bar(x + w/2, active_gating, w, color="#2980b9", alpha=0.85,
                    label="Active-Gating")
    ax.set_xticks(x); ax.set_xticklabels(rules, fontsize=8)
    ax.set_ylabel("Activity level (0=off, 1=full)")
    ax.set_title("Regime Comparison: Active Rules", fontweight="bold")
    ax.set_ylim(0, 1.3); ax.legend()
    ax.text(-0.3, 1.02, "disabled", fontsize=7, color="gray")
    ax.text(-0.3, 0.02, "off", fontsize=7, color="gray")

    fig.suptitle(
        "Fig. 2: Deterministic Trust Allocation — Quality-to-Trust Mapping\n"
        "(5 closed-form rules, no learnable parameters; dashed = Active-Gating regime)",
        fontweight="bold", fontsize=11
    )
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig2_trust_mapping.{ext}")
        fig.savefig(p)
        print(f"[Saved] {p}")
    plt.close(fig)


# ─── Fig 4: Student-t robust update mechanism ────────────────────────────────
def fig_robust_update(out_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    nu = 12.0  # validated setting
    nis_range = np.linspace(0, 30, 500)

    # Panel 1: lambda_t vs NIS_t for different nu
    ax = axes[0]
    for nu_val, ls, col in [(4.0, "-", "#e74c3c"),
                             (12.0, "-", "#2980b9"),
                             (9999.0, "--", "#2ecc71")]:
        lam = (nu_val + 1) / (nu_val + nis_range)
        label = f"$\\nu={nu_val:.0f}$ (Gaussian)" if nu_val > 100 else f"$\\nu={nu_val:.0f}$"
        ax.plot(nis_range, lam, ls=ls, color=col, lw=2.5, label=label)
    ax.axhline(1.0, color="gray", lw=0.8, ls=":", label="Gaussian ($\\lambda=1$)")
    ax.axvline(nu + 1, color="#2980b9", lw=0.8, ls=":")
    ax.set_xlabel("$\\mathrm{NIS}_t = v_t^2 / S_t$")
    ax.set_ylabel("$\\lambda_t$ (scale weight)")
    ax.set_title("(a) Student-t Scale Weight $\\lambda_t$")
    ax.legend(); ax.set_ylim(0, 1.15)
    ax.text(nu + 1.5, 0.9, f"NIS={nu+1:.0f}", fontsize=8, color="#2980b9")

    # Panel 2: R_eff / R ratio (gain suppression)
    ax = axes[1]
    for nu_val, ls, col in [(4.0, "-", "#e74c3c"),
                             (12.0, "-", "#2980b9"),
                             (9999.0, "--", "#2ecc71")]:
        lam = (nu_val + 1) / (nu_val + nis_range)
        r_eff_ratio = 1.0 / lam  # R_eff = R/lambda → ratio R_eff/R = 1/lambda
        ax.plot(nis_range, r_eff_ratio, ls=ls, color=col, lw=2.5,
                label=f"$\\nu={nu_val:.0f}$" if nu_val < 100 else "$\\nu\\to\\infty$")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("$\\mathrm{NIS}_t$")
    ax.set_ylabel("$R_{\\mathrm{eff},t} / R_t$ (gain suppression factor)")
    ax.set_title("(b) Effective Noise Inflation")
    ax.legend(); ax.set_ylim(0.9, 4.5)

    # Panel 3: Real-trial timeline simulation
    ax = axes[2]
    np.random.seed(42)
    T = 100
    t = np.arange(T)
    # Simulate NIS trace with occasional spikes (heavy-tailed)
    nis_trace = np.random.chisquare(1, T) * 0.5
    spike_times = [15, 35, 65, 82]
    for st in spike_times:
        nis_trace[st:st+3] += np.random.exponential(12, 3)
    nis_trace = np.clip(nis_trace, 0, 30)
    lam_trace = (nu + 1) / (nu + nis_trace)
    lam_gauss  = np.ones(T)

    ax2 = ax.twinx()
    ax.fill_between(t, 0, nis_trace, alpha=0.2, color="#95a5a6", label="$\\mathrm{NIS}_t$")
    ax.set_ylabel("$\\mathrm{NIS}_t$", color="#95a5a6")
    ax.tick_params(axis="y", labelcolor="#95a5a6")
    ax.set_ylim(0, 35)

    ax2.plot(t, lam_trace, color="#2980b9", lw=2, label=f"$\\lambda_t$ ($\\nu$=12)")
    ax2.plot(t, lam_gauss, color="#2ecc71", lw=1.5, ls="--", label="$\\lambda_t$ (Gaussian)")
    ax2.set_ylabel("$\\lambda_t$", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_ylim(0, 1.3)
    ax.set_xlabel("Frame $t$")
    ax.set_title("(c) Simulated Trial Timeline")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    fig.suptitle(
        "Fig. 4: Student-t Robust Update Mechanism\n"
        "$\\lambda_t = (\\nu+1)/(\\nu + \\mathrm{NIS}_t)$: suppresses gain under large innovations",
        fontweight="bold", fontsize=11
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig4_robust_update.{ext}")
        fig.savefig(p)
        print(f"[Saved] {p}")
    plt.close(fig)


# ─── Fig 7: Ablation visualization ──────────────────────────────────────────
def fig_ablation(out_dir: str):
    """EKS ablation results for Profile1D-Cubic."""

    ablation_variants = [
        "kfstd\n(baseline)",
        "QROBF\nfwd-only\n$\\nu$=12",
        "QROBF\nEKS+Gauss\n$\\nu\\to\\infty$",
        "QROBF\nEKS+Student-t\n$\\nu$=12\n(full)",
    ]
    mae_vals = [0.220, 0.270, 0.210, 0.210]
    ccc_vals = [0.890, np.nan, 0.880, 0.880]  # approx from metrics
    fail_vals = [np.nan, np.nan, 0.062, 0.062]  # fail_total median

    colors = ["#3498db", "#e74c3c", "#95a5a6", "#27ae60"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    # Panel 1: Freq MAE
    ax = axes[0]
    x = np.arange(len(ablation_variants))
    bars = ax.bar(x, mae_vals, color=colors, alpha=0.88)
    ax.axhline(0.220, color="#3498db", lw=1.5, ls="--", alpha=0.6, label="kfstd reference")
    for bar, v in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    # Annotate percent change vs kfstd
    kfstd_ref = 0.220
    for i, v in enumerate(mae_vals):
        if i > 0 and np.isfinite(v):
            pct = 100 * (v - kfstd_ref) / kfstd_ref
            ax.text(x[i], v - 0.012, f"{pct:+.1f}%",
                    ha="center", va="top", fontsize=8,
                    color="#e74c3c" if pct > 0 else "#27ae60")
    ax.set_xticks(x); ax.set_xticklabels(ablation_variants, fontsize=8)
    ax.set_ylabel("Freq MAE (BPM)")
    ax.set_title("(a) Frequency-Domain Accuracy\nProfile1D-Cubic, 160 trials",
                 fontweight="bold")
    ax.set_ylim(0, 0.35); ax.legend(fontsize=7.5)

    # Panel 2: Time CCC
    ax = axes[1]
    ccc_plot = [v if np.isfinite(v) else 0 for v in ccc_vals]
    bars2 = ax.bar(x, ccc_plot, color=colors, alpha=0.88)
    for bar, v in zip(bars2, ccc_vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ablation_variants, fontsize=8)
    ax.set_ylabel("CCC (Concordance Correlation Coeff)")
    ax.set_title("(b) Time-Domain Waveform Fidelity",
                 fontweight="bold")
    ax.set_ylim(0.7, 1.02)
    ax.text(1, 0.75, "N/A\n(fwd-only\nno EKS)", ha="center", va="center",
            fontsize=7.5, color="gray")

    # Panel 3: Component contribution diagram
    ax = axes[2]
    components = ["kfstd\nforward", "+EKS\n(RTS)", "+Student-t\n($\\nu$=12)"]
    incremental_mae = [0.220, -0.010, 0.000]  # incremental change
    cumulative = [0.220, 0.210, 0.210]
    colors3 = ["#3498db", "#27ae60", "#8e44ad"]

    x3 = np.arange(len(components))
    bars3 = ax.bar(x3, cumulative, color=colors3, alpha=0.88)
    for bar, v, dv in zip(bars3, cumulative, incremental_mae):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if dv != 0:
            ax.text(bar.get_x() + bar.get_width()/2, v/2,
                    f"{dv:+.3f}\nBPM", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    ax.set_xticks(x3); ax.set_xticklabels(components, fontsize=9)
    ax.set_ylabel("Freq MAE (BPM)")
    ax.set_title("(c) Incremental Component Contribution",
                 fontweight="bold")
    ax.set_ylim(0, 0.30)
    ax.annotate("EKS: primary driver\n($-$4.5% MAE)",
                xy=(1, 0.210), xytext=(1.5, 0.245),
                arrowprops=dict(arrowstyle="->", color="#27ae60"),
                fontsize=8, color="#27ae60", fontweight="bold")

    fig.suptitle(
        "Fig. 7: EKS Ablation — Component Contribution Analysis\n"
        "Profile1D-Cubic on COHFACE (160 trials, median freq MAE)",
        fontweight="bold", fontsize=11
    )
    plt.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"fig7_ablation.{ext}")
        fig.savefig(p)
        print(f"[Saved] {p}")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/cohface_robust_ossm/plots/paper",
                        help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fig_trust_mapping(args.out_dir)
    fig_robust_update(args.out_dir)
    fig_ablation(args.out_dir)
    print(f"\nAll mechanism figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
