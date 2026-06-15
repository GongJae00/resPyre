#!/usr/bin/env python3
"""Plot observation-class / preprocessing EDA heatmaps for the manuscript."""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.figure_style import family_label, save_figure, set_manuscript_style, stage_label


STAGE_ORDER = [
    "raw",
    "detrend_only",
    "bandpass_only",
    "sign_align_only",
    "robust_zscore_only",
    "current_preprocess",
    "helper_preprocess",
]

FAMILY_ORDER = ["OF", "OF_bridge", "DoF", "DoF_bridge", "P1D_lin", "P1D_quad", "P1D_cub", "P1D_cons"]


def _parse_args():
    root = ROOT
    parser = argparse.ArgumentParser(description="Plot observation EDA heatmaps.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=root / "analysis" / "cohface_observation_eda_family_stage_summary.csv",
    )
    parser.add_argument(
        "--summary-csv-mahnob",
        type=Path,
        default=root / "analysis" / "mahnob_observation_eda_family_stage_summary.csv",
    )
    parser.add_argument(
        "--delta-csv",
        type=Path,
        default=root / "analysis" / "cohface_preproc_deltas.csv",
    )
    parser.add_argument(
        "--delta-csv-mahnob",
        type=Path,
        default=root / "analysis" / "mahnob_preproc_deltas.csv",
    )
    parser.add_argument(
        "--dataset-rate-csv",
        type=Path,
        default=root / "analysis" / "dataset_rate_distribution_eda.csv",
    )
    parser.add_argument(
        "--dataset-summary-csv",
        type=Path,
        default=root / "analysis" / "dataset_distribution_eda.csv",
    )
    parser.add_argument(
        "--out-main",
        type=Path,
        default=root / "paper" / "figures" / "F2_dataset_and_observation_regime.pdf",
    )
    parser.add_argument(
        "--out-supp",
        type=Path,
        default=root / "paper" / "figures" / "S_F5_preproc_delta_heatmaps.pdf",
    )
    return parser.parse_args()


def _pivot_metric(df: pd.DataFrame, metric: str, fill=np.nan) -> np.ndarray:
    pivot = (
        df.pivot(index="family", columns="stage", values=metric)
        .reindex(index=FAMILY_ORDER, columns=STAGE_ORDER)
    )
    arr = pivot.to_numpy(dtype=float)
    if np.isnan(fill):
        return arr
    arr = np.where(np.isfinite(arr), arr, fill)
    return arr


def _text_color_for_value(cmap_obj, value, vmin, vmax):
    if vmin is None or vmax is None or vmax <= vmin or not np.isfinite(value):
        return "#111111"
    norm = np.clip((float(value) - float(vmin)) / (float(vmax) - float(vmin)), 0.0, 1.0)
    r, g, b, _ = cmap_obj(norm)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.48 else "#111111"


def _draw_heatmap(ax, arr, title, cmap, vmin=None, vmax=None, value_fmt="{:.2f}"):
    masked = np.ma.masked_invalid(arr)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#e8e8e8")
    im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=8, loc="center")
    ax.set_xticks(range(len(STAGE_ORDER)))
    ax.set_xticklabels([stage_label(s) for s in STAGE_ORDER], rotation=15, ha="center", fontsize=8.6)
    ax.set_yticks(range(len(FAMILY_ORDER)))
    ax.set_yticklabels([family_label(f) for f in FAMILY_ORDER], fontsize=8.4)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(STAGE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FAMILY_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    value_fmt.format(val),
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color=_text_color_for_value(cmap_obj, val, vmin, vmax),
                )
            else:
                ax.text(
                    j,
                    i,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="#6b6b6b",
                )
    return im


def _draw_dataset_rate_panel(ax, rate_df: pd.DataFrame):
    order = ["COHFACE", "MAHNOB-HCI", "V4V", "SCAMPS"]
    colors = {
        "COHFACE": "#256d85",
        "MAHNOB-HCI": "#b55a30",
        "V4V": "#4f8f45",
        "SCAMPS": "#7653a6",
    }
    data = []
    labels = []
    for dataset in order:
        vals = pd.to_numeric(rate_df.loc[rate_df["dataset"].eq(dataset), "rate_bpm"], errors="coerce").dropna()
        if not vals.empty:
            data.append(vals.to_numpy())
            labels.append(dataset)
    if not data:
        ax.text(0.5, 0.5, "dataset-rate EDA missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    box = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, dataset in zip(box["boxes"], labels):
        patch.set_facecolor(colors.get(dataset, "#777777"))
        patch.set_alpha(0.72)
    ax.set_ylabel("RR / peak rate (bpm)")
    ax.set_title("Dataset rate-regime distribution", fontsize=11, pad=8, loc="center")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=15, labelsize=8.5)


def _draw_dataset_claim_scope_panel(ax, summary_df: pd.DataFrame):
    order = ["COHFACE", "MAHNOB-HCI", "V4V", "SCAMPS"]
    if summary_df.empty or "dataset" not in summary_df.columns:
        ax.text(0.5, 0.5, "dataset-summary EDA missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    present = set(summary_df["dataset"].astype(str))
    columns = [
        "real\nvideo",
        "resp.\nwaveform",
        "RR/rate\nlabel",
        "main real\nbenchmark",
        "hard-regime\nstress",
        "auxiliary\nonly",
    ]
    # Cell value: 0 unsupported / not claimed, 1 diagnostic or synthetic-only evidence,
    # 2 real headline evidence. Text is intentionally semantic rather than a sample-count proxy.
    status = {
        "COHFACE": [("real", 2), ("wave GT", 2), ("GT peak", 2), ("clean", 2), ("-", 0), ("-", 0)],
        "MAHNOB-HCI": [("real", 2), ("wave GT", 2), ("GT peak", 2), ("hard", 2), ("stress", 2), ("-", 0)],
        "V4V": [("real", 2), ("-", 0), ("RR label", 2), ("-", 0), ("-", 0), ("rate\nonly", 1)],
        "SCAMPS": [("synthetic", 1), ("sim wave", 1), ("sim rate", 1), ("-", 0), ("-", 0), ("control", 1)],
    }
    color_map = {0: "#eef0f4", 1: "#d29532", 2: "#19766f"}
    text_color = {0: "#5b6470", 1: "#111111", 2: "white"}

    arr = np.array([[value for _label, value in status[d]] for d in order], dtype=float)
    rgba = np.empty(arr.shape + (4,), dtype=float)
    for value, color in color_map.items():
        rgba[arr == value] = matplotlib.colors.to_rgba(color)
    ax.imshow(rgba, aspect="auto")
    ax.set_title("Claim-scope evidence map", fontsize=11, pad=8, loc="center")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=0, ha="center", fontsize=7.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8.6)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(length=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i, dataset in enumerate(order):
        for j, cell in enumerate(status[dataset]):
            if dataset not in present:
                txt = "missing"
                color = "#5b6470"
            else:
                txt, value = cell
                color = text_color[value]
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.0, color=color, weight="bold")
    ax.text(
        0.5,
        -0.23,
        "Headline claims use real waveform benchmarks; V4V/SCAMPS remain auxiliary evidence.",
        ha="center",
        va="top",
        fontsize=7.4,
        color="#2a2f36",
        transform=ax.transAxes,
    )


def _draw_main_scope_table(ax, dataset_summary: pd.DataFrame):
    ax.set_axis_off()
    ax.set_title("a  Evaluation scope", fontsize=10.8, weight="bold", loc="left", pad=6)
    colors = {
        "COHFACE": "#19766f",
        "MAHNOB-HCI": "#19766f",
        "V4V": "#d29532",
        "SCAMPS": "#d29532",
    }
    rows = [
        ("COHFACE", "Real waveform + rate", "Clean benchmark"),
        ("MAHNOB-HCI", "Real waveform + rate", "Hard benchmark"),
        ("V4V", "Real rate only", "Auxiliary scope check"),
        ("SCAMPS", "Synthetic control", "Diagnostic context"),
    ]
    n_lookup = {}
    if not dataset_summary.empty and "dataset" in dataset_summary:
        for _, row in dataset_summary.iterrows():
            n_lookup[str(row.get("dataset"))] = row.get("n_units")

    x0 = 0.02
    widths = [0.31, 0.36, 0.31]
    xs = [x0]
    for width in widths[:-1]:
        xs.append(xs[-1] + width)
    header_y = 0.82
    row_h = 0.145
    headers = ["Dataset", "Evidence", "Use in release"]

    ax.add_patch(
        matplotlib.patches.Rectangle(
            (x0, header_y),
            sum(widths),
            0.09,
            facecolor="#1f2933",
            edgecolor="#1f2933",
            transform=ax.transAxes,
        )
    )
    for x, width, header in zip(xs, widths, headers):
        ax.text(
            x + 0.012,
            header_y + 0.045,
            header,
            color="white",
            fontsize=7.7,
            weight="bold",
            va="center",
            transform=ax.transAxes,
        )

    for idx, (dataset, evidence, claim) in enumerate(rows):
        y = header_y - (idx + 1) * row_h
        color = colors[dataset]
        face = "#f8fafc" if idx % 2 == 0 else "#ffffff"
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x0, y),
                sum(widths),
                row_h,
                facecolor=face,
                edgecolor="#c8d0d8",
                linewidth=0.8,
                transform=ax.transAxes,
            )
        )
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (x0, y),
                0.010,
                row_h,
                facecolor=color,
                edgecolor=color,
                transform=ax.transAxes,
            )
        )
        n_value = n_lookup.get(dataset)
        n_text = f"{int(n_value)}" if pd.notna(n_value) else "[verify]"
        dataset_label = f"{dataset}\nN={n_text}"
        values = [dataset_label, evidence, claim]
        for x, width, value in zip(xs, widths, values):
            ax.text(
                x + 0.012,
                y + row_h / 2,
                value,
                fontsize=7.5,
                weight="bold" if value == dataset_label else "normal",
                color=color if value == dataset_label else "#1f2933",
                va="center",
                ha="left",
                linespacing=1.05,
                transform=ax.transAxes,
            )
    ax.text(
        x0,
        0.095,
        "Real waveform claims use COHFACE and MAHNOB-HCI only.",
        fontsize=7.4,
        color="#2a2f36",
        transform=ax.transAxes,
    )
    key_y = 0.035
    key_items = [("COHFACE", "#256d85"), ("MAHNOB-HCI", "#b55a30")]
    key_x = x0
    for label, color in key_items:
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (key_x, key_y),
                0.026,
                0.020,
                facecolor=color,
                edgecolor=color,
                transform=ax.transAxes,
            )
        )
        ax.text(
            key_x + 0.034,
            key_y + 0.010,
            label,
            fontsize=7.2,
            va="center",
            color="#2a2f36",
            transform=ax.transAxes,
        )
        key_x += 0.22


def _stage_summary(summary: pd.DataFrame, metric: str, stages: list[str]) -> list[float]:
    values = []
    for stage in stages:
        vals = pd.to_numeric(summary.loc[summary["stage"].eq(stage), metric], errors="coerce").dropna()
        values.append(float(vals.median()) if not vals.empty else np.nan)
    return values


def _draw_stage_bar_summary(ax, cohface_summary: pd.DataFrame, mahnob_summary: pd.DataFrame, metric: str, title: str, ylabel: str, higher_better: bool, panel_label: str):
    stages = ["raw", "bandpass_only", "current_preprocess", "helper_preprocess"]
    labels = ["Raw", "Band", "Current", "Helper"]
    coh = _stage_summary(cohface_summary, metric, stages)
    mah = _stage_summary(mahnob_summary, metric, stages)
    x = np.arange(len(stages))
    width = 0.36
    bars1 = ax.bar(x - width / 2, coh, width, label="COHFACE", color="#256d85", alpha=0.86)
    bars2 = ax.bar(x + width / 2, mah, width, label="MAHNOB-HCI", color="#b55a30", alpha=0.86)
    ax.set_title(f"{panel_label}  {title}", fontsize=10.8, weight="bold", loc="left", pad=5)
    ax.set_ylabel(ylabel, fontsize=8.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.grid(axis="y", linestyle="--", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylim(0, 1.02)
    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(height + 0.025, 0.98),
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.9,
                    color="#1d252d",
                )


def main():
    args = _parse_args()
    datasets = [("COHFACE", pd.read_csv(args.summary_csv), pd.read_csv(args.delta_csv))]
    if args.summary_csv_mahnob.exists() and args.delta_csv_mahnob.exists():
        datasets.append(("MAHNOB-HCI", pd.read_csv(args.summary_csv_mahnob), pd.read_csv(args.delta_csv_mahnob)))

    args.out_main.parent.mkdir(parents=True, exist_ok=True)
    args.out_supp.parent.mkdir(parents=True, exist_ok=True)

    set_manuscript_style("paper")

    metrics_main = [
        ("corr_wave_best_median", "Waveform Corr (median)", "YlGnBu", 0.0, 1.0, "{:.2f}"),
        ("highfreq_energy_ratio_median", "High-Freq Energy Ratio (median)", "YlOrRd_r", 0.0, 1.0, "{:.2f}"),
    ]
    if args.dataset_rate_csv.exists() and args.dataset_summary_csv.exists():
        rate_df = pd.read_csv(args.dataset_rate_csv)
        summary_df = pd.read_csv(args.dataset_summary_csv)
        if len(datasets) == 2:
            fig = plt.figure(figsize=(7.25, 4.15), constrained_layout=True)
            grid = fig.add_gridspec(
                2,
                2,
                width_ratios=[1.38, 1.0],
                height_ratios=[1.0, 1.0],
                wspace=0.18,
                hspace=0.28,
            )
            main_axes = [
                fig.add_subplot(grid[:, 0]),
                fig.add_subplot(grid[0, 1]),
                fig.add_subplot(grid[1, 1]),
            ]
            _draw_main_scope_table(main_axes[0], summary_df)
            _draw_stage_bar_summary(
                main_axes[1],
                datasets[0][1],
                datasets[1][1],
                "corr_wave_best_median",
                "Waveform evidence",
                "median correlation",
                higher_better=True,
                panel_label="b",
            )
            _draw_stage_bar_summary(
                main_axes[2],
                datasets[0][1],
                datasets[1][1],
                "highfreq_energy_ratio_median",
                "High-frequency burden",
                "median energy ratio",
                higher_better=False,
                panel_label="c",
            )
            save_figure(fig, args.out_main)
        else:
            fig = plt.figure(figsize=(12.8, 3.0 + 4.3 * len(datasets)), constrained_layout=True)
            grid = fig.add_gridspec(len(datasets) + 1, len(metrics_main), height_ratios=[0.78] + [1.0] * len(datasets))
            top_left = fig.add_subplot(grid[0, 0])
            top_right = fig.add_subplot(grid[0, 1])
            axes = np.empty((len(datasets), len(metrics_main)), dtype=object)
            for r in range(len(datasets)):
                for c in range(len(metrics_main)):
                    axes[r, c] = fig.add_subplot(grid[r + 1, c])
            _draw_dataset_rate_panel(top_left, rate_df)
            _draw_dataset_claim_scope_panel(top_right, summary_df)
    else:
        fig, axes = plt.subplots(len(datasets), len(metrics_main), figsize=(12.4, 4.8 * len(datasets)), constrained_layout=True)
        if len(datasets) == 1:
            axes = np.expand_dims(axes, axis=0)
        for r, (dataset_label, summary, _delta) in enumerate(datasets):
            for c, (metric, title, cmap, vmin, vmax, value_fmt) in enumerate(metrics_main):
                ax = axes[r, c]
                arr = _pivot_metric(summary, metric)
                panel_title = title if len(datasets) == 1 else f"{dataset_label}: {title}"
                im = _draw_heatmap(ax, arr, panel_title, cmap, vmin=vmin, vmax=vmax, value_fmt=value_fmt)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                if c > 0:
                    ax.set_yticklabels([])
        fig.suptitle("Dataset and observation-class EDA: claim scope, label regime, and nuisance energy", fontsize=13)
        save_figure(fig, args.out_main)

    metrics_delta = [
        ("corr_wave_best_delta_median", "Waveform Corr Δ vs Raw", "RdYlGn", -0.4, 0.4, "{:+.2f}"),
        ("ccc_wave_best_z_delta_median", "Waveform CCC Δ vs Raw", "RdYlGn", -0.4, 0.4, "{:+.2f}"),
        ("corr_deriv_best_delta_median", "Derivative Corr Δ vs Raw", "RdYlGn", -0.4, 0.7, "{:+.2f}"),
        ("peak_error_hz_delta_median", "Peak Error Hz Δ vs Raw", "RdYlGn_r", -0.08, 0.08, "{:+.3f}"),
        ("lowfreq_energy_ratio_delta_median", "Low-Freq Energy Δ vs Raw", "RdYlGn_r", -0.6, 0.2, "{:+.2f}"),
        ("highfreq_energy_ratio_delta_median", "High-Freq Energy Δ vs Raw", "RdYlGn_r", -0.2, 1.0, "{:+.2f}"),
    ]
    fig, axes = plt.subplots(2 * len(datasets), 3, figsize=(17.2, 8.3 * len(datasets)), constrained_layout=True)
    if len(datasets) == 1:
        axes = np.expand_dims(axes, axis=0).reshape(2, 3)
    for d, (dataset_label, _summary, delta) in enumerate(datasets):
        for idx, (metric, title, cmap, vmin, vmax, value_fmt) in enumerate(metrics_delta):
            r = d * 2 + idx // 3
            c = idx % 3
            ax = axes[r, c]
            arr = _pivot_metric(delta, metric)
            panel_title = f"{dataset_label}: {title}"
            im = _draw_heatmap(ax, arr, panel_title, cmap, vmin=vmin, vmax=vmax, value_fmt=value_fmt)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            if c > 0:
                ax.set_yticklabels([])
    fig.suptitle("Preprocessing-stage deltas relative to raw observation", fontsize=13)
    save_figure(fig, args.out_supp)

    print(f"Saved main figure: {args.out_main}")
    print(f"Saved supplementary figure: {args.out_supp}")


if __name__ == "__main__":
    main()
