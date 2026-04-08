#!/usr/bin/env python3
"""Plot raw OF vs OF_bridge family comparison from table-ready CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.plotting_paper import set_paper_style

VARIANTS = ["Base", "KFstd", "PARH"]
COLORS = {"Base": "#4C6A92", "KFstd": "#C87B2A", "PARH": "#2B8A5B"}
FAMILIES = ["OF", "OF_bridge"]


def _extract_rows(t3: pd.DataFrame, t4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        r3 = t3[t3["family"] == family]
        r4 = t4[t4["family"] == family]
        if r3.empty or r4.empty:
            continue
        r3 = r3.iloc[0]
        r4 = r4.iloc[0]
        for variant in VARIANTS:
            rows.append(
                {
                    "family": family,
                    "variant": variant,
                    "rate_mae": float(r3[f"{variant}_MAE"]),
                    "rate_r": float(r3[f"{variant}_PearsonR"]),
                    "waveform_ccc": float(r4[f"{variant}_CCC"]),
                    "waveform_dtw": float(r4[f"{variant}_DTW"]),
                }
            )
    return pd.DataFrame(rows)


def plot_comparison(t3_csv: Path, t4_csv: Path, out_pdf: Path) -> pd.DataFrame:
    t3 = pd.read_csv(t3_csv)
    t4 = pd.read_csv(t4_csv)
    t3 = t3[t3["dataset"] == "COHFACE"].copy()
    t4 = t4[t4["dataset"] == "COHFACE"].copy()
    data = _extract_rows(t3, t4)

    set_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    metrics = [
        ("rate_mae", "Rate MAE", True),
        ("rate_r", "Rate Pearson r", False),
        ("waveform_ccc", "Waveform CCC", False),
        ("waveform_dtw", "Waveform DTW", True),
    ]
    xpos = [0, 1]
    width = 0.22
    offsets = {"Base": -width, "KFstd": 0.0, "PARH": width}

    for ax, (metric, title, lower_is_better) in zip(axes.ravel(), metrics):
        for variant in VARIANTS:
            ys = []
            for family in FAMILIES:
                row = data[(data["family"] == family) & (data["variant"] == variant)].iloc[0]
                ys.append(float(row[metric]))
            ax.bar(
                [x + offsets[variant] for x in xpos],
                ys,
                width=width,
                color=COLORS[variant],
                label=variant if metric == "rate_mae" else None,
                alpha=0.9,
            )
            for x, y in zip(xpos, ys):
                ax.text(x + offsets[variant], y, f"{y:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xticks(xpos, FAMILIES)
        suffix = " (lower better)" if lower_is_better else " (higher better)"
        ax.set_title(title + suffix)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0, 0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.05, 1.25))
    fig.suptitle("Raw OF vs OF_bridge Observation Construction", y=0.98)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw OF vs OF_bridge comparison.")
    parser.add_argument(
        "--t3-csv",
        type=Path,
        default=ROOT / "paper" / "tables_ready" / "T3_rate_main.csv",
    )
    parser.add_argument(
        "--t4-csv",
        type=Path,
        default=ROOT / "paper" / "tables_ready" / "T4_waveform_main.csv",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=ROOT / "paper" / "figures" / "S_F6_of_construction_comparison.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = plot_comparison(args.t3_csv, args.t4_csv, args.out_pdf)
    print(f"Saved: {args.out_pdf}")
    print(data.to_string(index=False))


if __name__ == "__main__":
    main()
