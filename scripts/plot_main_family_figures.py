#!/usr/bin/env python3
"""Generate main-paper family summary figures from table-ready CSVs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_ORDER = ["OF", "OF_bridge", "P1D_lin", "P1D_quad", "P1D_cub", "DoF"]
VARIANT_ORDER = ["Base", "KFstd", "PARH"]
VARIANT_COLORS = {
    "Base": "#1f3a5f",
    "KFstd": "#c06c2b",
    "PARH": "#116b4f",
}


def ordered_families(families):
    seen = [f for f in FAMILY_ORDER if f in set(families)]
    rest = [f for f in families if f not in seen]
    return seen + rest


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8dde6", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def plot_t3_family_summary(t3_df: pd.DataFrame, out_path: Path):
    families = ordered_families(list(t3_df["family"].astype(str)))
    x = np.arange(len(families))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    panels = [
        ("MAE", "Rate MAE", False),
        ("PearsonR", "Rate Pearson r", True),
    ]

    for ax, (suffix, ylabel, higher_better) in zip(axes, panels):
        style_axes(ax)
        for variant in VARIANT_ORDER:
            y = []
            for family in families:
                row = t3_df.loc[t3_df["family"] == family].iloc[0]
                y.append(float(row[f"{variant}_{suffix}"]))
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=6.5,
                label=variant,
                color=VARIANT_COLORS[variant],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=15)
        ax.set_ylabel(ylabel)
        if not higher_better:
            ax.annotate(
                "lower is better",
                xy=(0.99, 0.96),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=9,
                color="#444",
            )
        else:
            ax.annotate(
                "higher is better",
                xy=(0.99, 0.96),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=9,
                color="#444",
            )
    axes[0].set_title("Oscillatory Output Accuracy Across Families")
    axes[1].set_title("Rate Correlation Across Families")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_activation(t6_df: pd.DataFrame, mech_df: pd.DataFrame, out_path: Path):
    families = ordered_families(list(mech_df["family"].astype(str)))
    mech_df = mech_df.set_index("family")
    t6_df = t6_df.set_index("family")
    x = np.arange(len(families))
    width = 0.32

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)

    ax = axes[0]
    style_axes(ax)
    q_dyn = [float(mech_df.loc[f, "q_dyn_mean_median"]) for f in families]
    q_osc = [float(mech_df.loc[f, "q_osc_mean_median"]) for f in families]
    obs_need = [float(mech_df.loc[f, "obs_nonosc_need_mean_median"]) for f in families]
    ax.plot(x, q_dyn, marker="o", linewidth=2.0, color="#7b4ea3", label=r"$q_{dyn}$")
    ax.plot(x, q_osc, marker="s", linewidth=2.0, color="#1a7f64", label=r"$q_{osc}$")
    ax.plot(x, obs_need, marker="^", linewidth=2.0, color="#c85f2b", label="nonosc need")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Mechanism Activation")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    style_axes(ax)
    for family in families:
        row = t6_df.loc[family]
        ax.scatter(
            float(row["PARH_NIS_Mean"]),
            float(row["PARH_NIS_InBand"]),
            s=70,
            color="#116b4f" if family != "OF_bridge" else "#7b4ea3",
        )
        ax.annotate(family, (float(row["PARH_NIS_Mean"]), float(row["PARH_NIS_InBand"])), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axvline(1.0, color="#888", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Median NIS Mean")
    ax.set_ylabel("Median NIS In-Band")
    ax.set_title("Calibration Position")

    ax = axes[2]
    style_axes(ax)
    baseline = [float(mech_df.loc[f, "baseline_energy_ratio_median"]) for f in families]
    residual = [float(mech_df.loc[f, "residual_energy_ratio_median"]) for f in families]
    ax.bar(x - width / 2, baseline, width=width, color="#5b7db1", label="baseline ratio")
    ax.bar(x + width / 2, residual, width=width, color="#d17a3a", label="residual ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15)
    ax.set_ylabel("Energy Ratio")
    ax.set_title("Baseline vs Residual Share")
    ax.legend(frameon=False, loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate F3/F5 family summary figures.")
    parser.add_argument(
        "--t3-csv",
        type=Path,
        default=root / "paper" / "tables_ready" / "T3_rate_main.csv",
    )
    parser.add_argument(
        "--t6-csv",
        type=Path,
        default=root / "paper" / "tables_ready" / "T6_diagnostics_main.csv",
    )
    parser.add_argument(
        "--mech-csv",
        type=Path,
        default=root / "paper" / "tables_ready" / "T6b_cohface_mechanism_audit.csv",
    )
    parser.add_argument(
        "--out-f3",
        type=Path,
        default=root / "paper" / "figures" / "F3_t3_family_summary.pdf",
    )
    parser.add_argument(
        "--out-f5",
        type=Path,
        default=root / "paper" / "figures" / "F5_mechanism_activation.pdf",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t3_df = pd.read_csv(args.t3_csv)
    t6_df = pd.read_csv(args.t6_csv)
    mech_df = pd.read_csv(args.mech_csv)
    plot_t3_family_summary(t3_df, args.out_f3)
    plot_mechanism_activation(t6_df, mech_df, args.out_f5)
    print(f"Saved {args.out_f3}")
    print(f"Saved {args.out_f5}")


if __name__ == "__main__":
    main()
