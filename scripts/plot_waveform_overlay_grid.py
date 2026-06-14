#!/usr/bin/env python3
"""Plot same-trial waveform overlay grids from a persistent manifest."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import math

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.figure_style import add_metric_box, family_label, save_figure, set_manuscript_style, style_axis, variant_label
from core.evaluation.metrics import calculate_cross_corr_alignment


DEFAULT_VARIANT_ORDER = ["Base", "OSSM-KF", "KFstd", "PARH", "Final"]
VARIANT_COLORS = {
    "Base": "#1f3a5f",
    "OSSM-KF": "#c06c2b",
    "KFstd": "#c06c2b",
    "PARH": "#116b4f",
    "Final": "#7d3aa6",
}

VARIANT_DISPLAY_LABELS = {
    "GT": "GT",
    "Base": variant_label("Base"),
    "OSSM-KF": "OSSM-KF",
    "KFstd": variant_label("KFstd"),
    "PARH": variant_label("PARH"),
    "Final": variant_label("Final"),
}

STYLE_PRESETS = {
    "paper": {
        "figsize_per_col": 4.35,
        "figsize_per_row": 2.95,
        "title_size": 10,
        "label_size": 10,
        "suptitle_size": 13,
        "legend_size": 9,
        "line_gt": 2.0,
        "line_pred": 1.8,
        "grid_alpha": 0.6,
        "metric_box_size": 8.0,
    },
    "review": {
        "figsize_per_col": 5.1,
        "figsize_per_row": 3.35,
        "title_size": 11.5,
        "label_size": 11,
        "suptitle_size": 15,
        "legend_size": 10,
        "line_gt": 2.2,
        "line_pred": 2.0,
        "grid_alpha": 0.75,
        "metric_box_size": 8.7,
    },
}


def _zscore(sig: np.ndarray) -> np.ndarray:
    sig = np.asarray(sig, dtype=float).flatten()
    std = np.std(sig)
    if not np.isfinite(std) or std < 1e-9:
        return sig * 0.0
    return (sig - np.mean(sig)) / std


def _resolve_pkl(row: pd.Series, run_dir: Path) -> Path:
    data_file = Path(str(row["data_file"]))
    if data_file.is_absolute():
        return data_file
    return run_dir / data_file


def _load_estimate(payload: dict, method: str) -> dict:
    aliases = [method]
    if method.endswith("__ossm_kf"):
        aliases.append(method.replace("__ossm_kf", "__kfstd"))
    elif method.endswith("__kfstd"):
        aliases.append(method.replace("__kfstd", "__ossm_kf"))
    for item in payload.get("estimates", []):
        if item.get("method") in aliases:
            return item.get("estimate", {})
    raise KeyError(f"Method {method} not found in payload; tried {aliases}")


def _series_for_row(row: pd.Series, payload: dict) -> tuple[np.ndarray, np.ndarray, float]:
    est = _load_estimate(payload, str(row["method"]))
    variant = str(row["variant"]) if "variant" in row.index else str(row.name)
    output_type = str(row.get("output_type", "z_full" if variant == "PARH" else "signal_hat"))
    if output_type not in est:
        output_type = "z_full" if "z_full" in est else "signal_hat"
    pred = np.asarray(est[output_type], dtype=float).flatten()
    gt = np.asarray(payload["gt"], dtype=float).flatten()
    fs_est = float(payload.get("fps", 20.0))
    fs_gt = float(payload.get("fs_gt", fs_est))
    aligned_pred, aligned_gt, _ = calculate_cross_corr_alignment(pred, gt, fs_est=fs_est, fs_gt=fs_gt)
    return _zscore(aligned_pred), _zscore(aligned_gt), fs_gt


def _window_signals(pred: np.ndarray, gt: np.ndarray, fs: float, window_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = min(len(pred), len(gt))
    pred = pred[:common]
    gt = gt[:common]
    if common == 0:
        return np.array([]), np.array([]), np.array([])
    win = min(common, max(200, int(round(window_sec * fs))))
    start = max((common - win) // 2, 0)
    end = start + win
    t = np.arange(win, dtype=float) / fs
    return t, pred[start:end], gt[start:end]


def _panel_metric_text(row: pd.Series) -> str:
    return (
        f"CCC {float(row['waveform_CCC']):.3f}\n"
        f"wMAE {float(row['waveform_MAE']):.3f}\n"
        f"DTW {float(row['waveform_DTW']):.3f}"
    )


def _variant_order(manifest: pd.DataFrame):
    seen = list(dict.fromkeys(manifest["variant"].astype(str).tolist()))
    if "variant_display_order" in manifest.columns:
        order_df = manifest[["variant", "variant_display_order"]].copy()
        order_df["variant"] = order_df["variant"].astype(str)
        order_df["variant_display_order"] = pd.to_numeric(order_df["variant_display_order"], errors="coerce")
        order_df = order_df.groupby("variant", as_index=False)["variant_display_order"].min()
        fallback_order = {name: idx for idx, name in enumerate(DEFAULT_VARIANT_ORDER)}
        order_df["variant_display_order"] = order_df.apply(
            lambda row: row["variant_display_order"] if pd.notna(row["variant_display_order"]) else fallback_order.get(row["variant"], 99),
            axis=1,
        )
        order_df = order_df.sort_values(["variant_display_order", "variant"])
        return order_df["variant"].tolist()
    ordered = [v for v in DEFAULT_VARIANT_ORDER if v in seen]
    ordered += [v for v in seen if v not in ordered]
    return ordered


def _class_column(manifest: pd.DataFrame) -> str:
    if "observation_class" in manifest.columns:
        return "observation_class"
    if "family" in manifest.columns:
        return "family"
    raise KeyError("Manifest must include either `observation_class` or `family`.")


def _row_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    case_priority = {"top": 0, "median": 1, "bottom": 2}
    class_col = _class_column(manifest)
    cols = [
        c
        for c in [
            "panel_group",
            "dataset",
            class_col,
            "case_rank",
            "video",
            "dataset_display_order",
            "row_display_order",
        ]
        if c in manifest.columns
    ]
    rows = manifest[cols].drop_duplicates().copy()
    if class_col != "observation_class":
        rows = rows.rename(columns={class_col: "observation_class"})
    if "dataset_display_order" not in rows.columns:
        order = {name: idx for idx, name in enumerate(dict.fromkeys(rows["dataset"].astype(str).tolist()))}
        rows["dataset_display_order"] = rows["dataset"].astype(str).map(order).astype(int)
    if "row_display_order" not in rows.columns:
        rows["row_display_order"] = rows["case_rank"].map(case_priority).fillna(99).astype(int)
    rows["case_priority"] = rows["case_rank"].map(case_priority).fillna(99).astype(int)
    rows = rows.sort_values(["dataset_display_order", "row_display_order", "case_priority", "panel_group"]).reset_index(drop=True)
    return rows


def _variant_color(variant: str) -> str:
    if variant in VARIANT_COLORS:
        return VARIANT_COLORS[variant]
    fallback = ["#1f3a5f", "#c06c2b", "#116b4f", "#7d3aa6", "#d95f5f", "#267f99"]
    idx = sum(ord(ch) for ch in variant) % len(fallback)
    return fallback[idx]


def plot_manifest(
    manifest_csv: Path,
    run_dir: Path,
    out_path: Path,
    window_sec: float = 18.0,
    style: str = "paper",
    custom_title: str | None = None,
):
    manifest = pd.read_csv(manifest_csv)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_csv}")

    class_col = _class_column(manifest)
    if class_col != "observation_class":
        manifest = manifest.rename(columns={class_col: "observation_class"})

    style_cfg = STYLE_PRESETS[str(style)]
    set_manuscript_style("review" if style == "review" else "paper")
    row_groups = _row_groups(manifest)
    variants = _variant_order(manifest)
    group_cols = 2 if len(row_groups) >= 6 and len(variants) <= 3 else 1
    plot_rows = int(math.ceil(len(row_groups) / group_cols))
    plot_cols = len(variants) * group_cols
    fig_w = max(10.5, style_cfg["figsize_per_col"] * plot_cols)
    fig_h = max(7.2, style_cfg["figsize_per_row"] * plot_rows)
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(fig_w, fig_h), sharex=False, sharey=True, constrained_layout=True)
    if plot_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if plot_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, row_meta in row_groups.iterrows():
        block_row = r // group_cols
        block_col = r % group_cols
        panel_group = str(row_meta["panel_group"])
        case_rank = str(row_meta.get("case_rank", ""))
        dataset_label = str(row_meta.get("dataset", ""))
        case_df = manifest[manifest["panel_group"] == panel_group].copy()
        case_df = case_df.set_index("variant")
        video_label = str(row_meta.get("video", "")) if not case_df.empty else ""
        for c, variant in enumerate(variants):
            ax = axes[block_row, block_col * len(variants) + c]
            style_axis(ax, grid="y")
            if variant not in case_df.index:
                ax.axis("off")
                continue
            row = case_df.loc[variant]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            payload = pickle.load(open(_resolve_pkl(row, run_dir), "rb"))
            pred, gt, fs = _series_for_row(row, payload)
            t, pred_w, gt_w = _window_signals(pred, gt, fs, window_sec)
            ax.plot(t, gt_w, color="#111111", linewidth=style_cfg["line_gt"], label="GT")
            ax.plot(t, pred_w, color=_variant_color(variant), linewidth=style_cfg["line_pred"], label=variant)
            if block_row == 0:
                title_label = VARIANT_DISPLAY_LABELS.get(variant, variant)
                ax.set_title(title_label, fontsize=style_cfg["title_size"], loc="center")
            add_metric_box(ax, _panel_metric_text(row), loc="upper left", fontsize=style_cfg["metric_box_size"])
            if c == 0:
                class_code = family_label(str(row_meta.get("observation_class", ""))).replace("\n", " ")
                header = " / ".join([part for part in [dataset_label, class_code] if part])
                detail = " / ".join([part for part in [case_rank.capitalize(), video_label] if part])
                ylabel = "\n".join([part for part in [header, detail] if part])
                ax.set_ylabel(ylabel, fontsize=style_cfg["label_size"])
            if block_row == plot_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=style_cfg["label_size"])
            ax.set_ylim(-3.2, 3.2)

    legend_handles = [Line2D([0], [0], color="#111111", linewidth=style_cfg["line_gt"], label=VARIANT_DISPLAY_LABELS["GT"])]
    for variant in variants:
        if variant in set(manifest["variant"].astype(str)):
            legend_handles.append(
                Line2D([0], [0], color=_variant_color(variant), linewidth=style_cfg["line_pred"], label=VARIANT_DISPLAY_LABELS.get(variant, variant))
            )
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(legend_handles), 5),
        frameon=False,
        fontsize=style_cfg["legend_size"],
    )
    dataset = str(manifest.iloc[0]["dataset"])
    unique_blocks = manifest[["dataset", "observation_class"]].drop_duplicates()
    if len(unique_blocks) == 1:
        class_title = family_label(str(manifest.iloc[0]["observation_class"])).replace("\n", " ")
        title = custom_title or f"{dataset} {class_title} Same-Trial Overlay"
    else:
        title = custom_title or "Representative Observation-Class Overlays Across Regimes"
    fig.suptitle(title, fontsize=style_cfg["suptitle_size"], y=1.03)
    save_figure(fig, out_path)


def parse_args():
    root = ROOT
    parser = argparse.ArgumentParser(description="Plot waveform overlay grid from a persistent manifest.")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory that contains the `data/` folder referenced by manifest data_file paths.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=18.0,
    )
    parser.add_argument(
        "--style",
        choices=sorted(STYLE_PRESETS),
        default="paper",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=root / "paper" / "figures" / "F4_waveform_overlay_grid.pdf",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_manifest(
        manifest_csv=args.manifest_csv,
        run_dir=args.run_dir,
        out_path=args.out,
        window_sec=args.window_sec,
        style=args.style,
        custom_title=args.title,
    )
    print(json.dumps({"manifest": str(args.manifest_csv), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
