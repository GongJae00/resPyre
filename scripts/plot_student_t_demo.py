#!/usr/bin/env python3
"""
Figure helper: Student-t robust update intuition.

Outputs a paper-ready PNG showing:
  1) influence function (Gaussian vs Student-t)
  2) lambda / R_eff modulation vs NIS
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=15.0)
    ap.add_argument("--out", default="analysis/figures/student_t_demo.png")
    args = ap.parse_args()

    nu = float(max(args.nu, 1e-6))
    r = np.linspace(-8.0, 8.0, 1200)  # standardized innovation
    nis = np.linspace(0.0, 50.0, 800)

    # Influence function (score wrt residual, up to scaling)
    psi_gauss = r
    psi_t = (nu + 1.0) * r / (nu + r ** 2)

    # VB scale and effective measurement inflation
    lam = (nu + 1.0) / (nu + nis)
    r_eff_scale = 1.0 / np.clip(lam, 1e-9, None)  # R_eff / R

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(r, psi_gauss, label="Gaussian", color="#1d3557", linewidth=2)
    axes[0].plot(r, psi_t, label=f"Student-t (nu={nu:g})", color="#e63946", linewidth=2)
    axes[0].set_title("Influence Function")
    axes[0].set_xlabel("Standardized innovation")
    axes[0].set_ylabel("Influence")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(nis, lam, label="lambda (VB weight)", color="#2a9d8f", linewidth=2)
    axes[1].plot(nis, r_eff_scale, label="R_eff / R", color="#f4a261", linewidth=2)
    axes[1].set_title("Innovation-Adaptive Noise Scaling")
    axes[1].set_xlabel("NIS")
    axes[1].set_ylabel("Scale")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("Student-t Robust Update Mechanism")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.out, dpi=240)
    plt.close(fig)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()

