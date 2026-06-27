#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paperfig.io import DEFAULT_RESULTS_ZIP, ArtifactStore, member_join
from paperfig.panels import annotated_heatmap, dumbbell, horizontal_dotplot
from paperfig.style import (
    DOUBLE_COL_MM,
    METHOD_COLORS,
    PALETTE,
    SUPP_WIDTH_MM,
    clean_axis,
    direct_label,
    figure_size,
    panel_label,
    save_all,
    set_publication_style,
)
from paperfig.waveforms import best_window, extract_estimate, plot_waveform_panel, window_pair, zscore
from scripts.plot_observation_eda import FALLBACK_STAGE_VALUES, MAIN_STAGE_LABELS


FIG_DIR = ROOT / "paper" / "figures"
ANALYSIS_DIR = ROOT / "analysis"
MANIFEST_PATH = ANALYSIS_DIR / "publication_figure_manifest.json"

FINAL_COH = "results/final_full_validation/cohface"
FINAL_MAH = "results/final_full_validation/mahnob_tailaligned"
BASE_COH = (
    "results/20260409_cohface_prod_ofbridge_dofbridge_p1dcons_e2e_policy_narrow/"
    "cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons"
)
BASE_MAH = (
    "results/20260409_mahnob_prod_ofbridge_dofbridge_p1dcons_e2e/"
    "mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons"
)

FAMILY_ORDER = [
    "OF",
    "OF_bridge",
    "DoF",
    "DoF_bridge",
    "P1D_lin",
    "P1D_quad",
    "P1D_cub",
    "P1D_cons",
]

FAMILY_LABEL = {
    "OF": "OF",
    "OF_bridge": "OF bridge",
    "DoF": "DoF",
    "DoF_bridge": "DoF bridge",
    "P1D_lin": "P1D linear",
    "P1D_quad": "P1D quadratic",
    "P1D_cub": "P1D cubic",
    "P1D_cons": "P1D consensus",
}

DIRECT_METHOD = {
    "OF": "of_farneback",
    "OF_bridge": "of_disp_bridge",
    "DoF": "DoF",
    "DoF_bridge": "dof_disp_bridge",
    "P1D_lin": "profile1D linear",
    "P1D_quad": "profile1D quadratic",
    "P1D_cub": "profile1D cubic",
    "P1D_cons": "profile1d_consensus",
}

KF_METHOD = {
    "OF": "of_farneback__kfstd",
    "OF_bridge": "of_disp_bridge__kfstd",
    "DoF": "dof__kfstd",
    "DoF_bridge": "dof_disp_bridge__kfstd",
    "P1D_lin": "profile1d_linear__kfstd",
    "P1D_quad": "profile1d_quadratic__kfstd",
    "P1D_cub": "profile1d_cubic__kfstd",
    "P1D_cons": "profile1d_consensus__kfstd",
}

PARH_METHOD = {
    "OF": "of_farneback__parh_ossm",
    "OF_bridge": "of_disp_bridge__parh_ossm",
    "DoF": "dof__parh_ossm",
    "DoF_bridge": "dof_disp_bridge__parh_ossm",
    "P1D_lin": "profile1d_linear__parh_ossm",
    "P1D_quad": "profile1d_quadratic__parh_ossm",
    "P1D_cub": "profile1d_cubic__parh_ossm",
    "P1D_cons": "profile1d_consensus__parh_ossm",
}


def classify_method(method: str) -> tuple[str | None, str]:
    m = str(method)
    low = m.lower().replace(" ", "_")
    variant = "Base"
    if "__parh_ossm" in low:
        variant = "PARH"
        low = low.split("__parh_ossm")[0]
    elif "__kfstd" in low or "__ossm_kf" in low:
        variant = "OSSM-KF"
        low = low.replace("__ossm_kf", "").split("__kfstd")[0]
    mapping = {
        "of_farneback": "OF",
        "of_disp_bridge": "OF_bridge",
        "dof": "DoF",
        "dof_disp_bridge": "DoF_bridge",
        "profile1d_linear": "P1D_lin",
        "profile1d_quadratic": "P1D_quad",
        "profile1d_cubic": "P1D_cub",
        "profile1d_consensus": "P1D_cons",
        "profile1d": "P1D_lin",
    }
    return mapping.get(low), variant


def _median(s) -> float:
    vals = pd.to_numeric(s, errors="coerce")
    return float(vals.median()) if vals.notna().any() else float("nan")


def load_family_summary(store: ArtifactStore, dataset: str) -> pd.DataFrame:
    run = BASE_COH if dataset == "COHFACE" else BASE_MAH
    freq = store.read_csv(member_join(run, "metrics", "metrics_freq_domain_raw.csv"))
    wave = store.read_csv(member_join(run, "metrics", "metrics_waveform_raw.csv"))
    strict_path = member_join(run, "metrics", "metrics_waveform_strict_raw.csv")
    strict = store.read_csv(strict_path) if store.exists(strict_path) else pd.DataFrame()
    freq[["family", "variant"]] = freq["method"].apply(lambda x: pd.Series(classify_method(x)))
    wave[["family", "variant"]] = wave["method"].apply(lambda x: pd.Series(classify_method(x)))
    if not strict.empty:
        strict[["family", "variant"]] = strict["method"].apply(lambda x: pd.Series(classify_method(x)))
    rows = []
    for family in FAMILY_ORDER:
        for variant in ["Base", "OSSM-KF", "PARH"]:
            rf = freq[freq["family"].eq(family) & freq["variant"].eq(variant)]
            wf = wave[
                wave["family"].eq(family)
                & wave["variant"].eq(variant)
                & wave["causal_or_smoothed"].eq("smoothed")
                & wave["output_type"].eq("signal_hat")
            ]
            sf = pd.DataFrame()
            if not strict.empty:
                sf = strict[
                    strict["family"].eq(family)
                    & strict["variant"].eq(variant)
                    & strict["causal_or_smoothed"].eq("smoothed")
                    & strict["output_type"].eq("signal_hat")
                ]
            if rf.empty and wf.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "family_label": FAMILY_LABEL[family],
                    "variant": variant,
                    "rate_mae": _median(rf["MAE"]) if not rf.empty else np.nan,
                    "rate_r": _median(rf["PearsonR"]) if not rf.empty else np.nan,
                    "wave_ccc": _median(wf["waveform_CCC"]) if not wf.empty else np.nan,
                    "wave_dtw": _median(wf["waveform_DTW"]) if not wf.empty else np.nan,
                    "strict_ccc": _median(sf["strict_CCC"]) if not sf.empty else np.nan,
                    "strict_nmae": _median(sf["strict_NMAE_span"]) if (not sf.empty and "strict_NMAE_span" in sf) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def load_integrated_summary(store: ArtifactStore, dataset: str) -> dict[str, float]:
    run = FINAL_COH if dataset == "COHFACE" else FINAL_MAH
    freq = store.read_csv(member_join(run, "metrics", "metrics_freq_domain_raw.csv"))
    wave = store.read_csv(member_join(run, "metrics", "metrics_waveform_raw.csv"))
    strict = store.read_csv(member_join(run, "metrics", "metrics_waveform_strict_raw.csv"))
    wave = wave[
        wave["causal_or_smoothed"].eq("smoothed")
        & wave["output_type"].eq("z_full")
        & wave["method"].eq("parh_ossm")
    ]
    strict = strict[
        strict["causal_or_smoothed"].eq("smoothed")
        & strict["output_type"].eq("z_full")
        & strict["method"].eq("parh_ossm")
    ]
    return {
        "rate_mae": _median(freq["MAE"]),
        "rate_r": _median(freq["PearsonR"]),
        "wave_ccc": _median(wave["waveform_CCC"]),
        "wave_dtw": _median(wave["waveform_DTW"]),
        "strict_ccc": _median(strict["strict_CCC"]),
        "strict_nmae": _median(strict["strict_NMAE_span"]),
    }


def metric_text(row: pd.Series | dict, prefix: str = "") -> str:
    def get(k):
        return row.get(k, np.nan) if isinstance(row, dict) else row.get(k, np.nan)

    parts = []
    for key, label in [("waveform_CCC", "CCC"), ("waveform_MAE", "wMAE"), ("waveform_DTW", "DTW")]:
        val = get(key)
        if np.isfinite(val):
            parts.append(f"{label} {float(val):.2f}")
    if not parts:
        for key, label in [("wave_ccc", "CCC"), ("wave_dtw", "DTW"), ("rate_mae", "MAE")]:
            val = get(key)
            if np.isfinite(val):
                parts.append(f"{label} {float(val):.2f}")
    return prefix + "\n".join(parts[:3])


def save_manifest(entries: list[dict]) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")


def register(entries: list[dict], fig_id: str, out_pdf: Path, inputs: list[str], width_class: str) -> None:
    rel_pdf = out_pdf.resolve().relative_to(ROOT)
    entries.append(
        {
            "figure_id": fig_id,
            "inputs": inputs,
            "outputs": {
                "pdf": str(rel_pdf),
                "svg": str(rel_pdf.with_suffix(".svg")),
                "png": str(rel_pdf.with_suffix(".png")),
            },
            "generation": "scripts/make_publication_figures.py",
            "width_class": width_class,
        }
    )


def plot_main_f2(entries: list[dict]) -> None:
    stages = MAIN_STAGE_LABELS
    x = np.arange(len(stages))
    out = FIG_DIR / "F2_dataset_and_observation_regime.pdf"
    fig, axes = plt.subplots(1, 2, figsize=figure_size(DOUBLE_COL_MM, 54), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.925, bottom=0.22, top=0.82, wspace=0.24)
    specs = [
        ("corr_wave_best_median", "Waveform evidence", "higher is better"),
        ("highfreq_energy_ratio_median", "High-frequency burden", "lower is better"),
    ]
    colors = {"COHFACE": PALETTE["cohface"], "MAHNOB-HCI": PALETTE["mahnob"]}
    for ax, (metric, title, note) in zip(axes, specs):
        endpoints = []
        for dataset in ["COHFACE", "MAHNOB-HCI"]:
            values = np.asarray(FALLBACK_STAGE_VALUES[dataset][metric], dtype=float)
            ax.plot(x, values, color=colors[dataset], marker="o", linewidth=1.45, markersize=3.7)
            for idx, (xi, yi) in enumerate(zip(x, values)):
                va = "bottom" if yi < 0.94 else "top"
                dy = 0.035 if va == "bottom" else -0.035
                show_endpoint = idx == len(values) - 1 and metric != "highfreq_energy_ratio_median"
                if idx == 0 or show_endpoint or (metric == "highfreq_energy_ratio_median" and idx == 2):
                    ax.text(xi, yi + dy, f"{yi:.2f}", ha="center", va=va, fontsize=5.7, color=colors[dataset])
            endpoints.append((dataset, float(values[-1]), colors[dataset]))
        # Direct labels are separated from the endpoint values so close final
        # stages do not collide in the hard-regime nuisance panel.
        label_positions = [endpoints[0][1], endpoints[1][1]]
        if abs(label_positions[0] - label_positions[1]) < 0.09:
            mid = max(0.055, float(np.mean(label_positions)))
            label_positions = [mid + 0.055, mid - 0.055]
        for (dataset, y_end, color), y_lab in zip(endpoints, label_positions):
            ax.plot([x[-1] + 0.02, x[-1] + 0.12], [y_end, y_lab], color=color, linewidth=0.55, alpha=0.65, clip_on=False)
            direct_label(ax, x[-1] + 0.12, y_lab, dataset, color, dx=0.00)
        ax.set_title(title, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(stages)
        ax.set_ylim(-0.04, 1.08)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_xlim(-0.15, len(stages) - 0.28)
        ax.text(0.98, 0.94, note, transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=PALETTE["muted"])
        clean_axis(ax, "y")
    axes[0].set_ylabel("Median value")
    panel_label(axes[0], "A", -0.14, 1.05)
    panel_label(axes[1], "B", -0.14, 1.05)
    save_all(fig, out)
    register(entries, "Figure 2", out, ["scripts/plot_observation_eda.py:FALLBACK_STAGE_VALUES"], "double")


def plot_s1_rate_diagnostics(store: ArtifactStore, entries: list[dict]) -> None:
    df = load_family_summary(store, "COHFACE")
    y_order = [FAMILY_LABEL[f] for f in FAMILY_ORDER]
    df = df[df["variant"].isin(["Base", "OSSM-KF", "PARH"])].copy()
    out = FIG_DIR / "F3_rate_observation_class_summary.pdf"
    integrated = load_integrated_summary(store, "COHFACE")
    fig, axes = plt.subplots(1, 2, figsize=figure_size(SUPP_WIDTH_MM, 92))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.20, top=0.84, wspace=0.22)
    for ax, xcol, title, xlabel, ref in [
        (axes[0], "rate_mae", "Rate error", "MAE (bpm)", integrated["rate_mae"]),
        (axes[1], "rate_r", "Rate correlation", "Pearson r", integrated["rate_r"]),
    ]:
        horizontal_dotplot(
            ax,
            df,
            y_col="family_label",
            x_col=xcol,
            hue_col="variant",
            y_order=y_order,
            hue_order=["Base", "OSSM-KF", "PARH"],
            xlabel=xlabel,
            direction=None,
        )
        ax.axvline(ref, color=PALETTE["gt"], linestyle="--", linewidth=0.85)
        ax.text(
            0.98,
            1.02,
            "lower is better" if xcol == "rate_mae" else "higher is better",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.2,
            color=PALETTE["muted"],
        )
        ax.set_title(title, loc="left")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["Base"], markeredgecolor=METHOD_COLORS["Base"], label="Base"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["OSSM-KF"], markeredgecolor=METHOD_COLORS["OSSM-KF"], label="OSSM-KF"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS["PARH"], markeredgecolor=METHOD_COLORS["PARH"], label="PARH"),
        Line2D([0], [0], color=PALETTE["gt"], linestyle="--", linewidth=0.9, label="integrated PARH-OSSM"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.58, 0.02))
    panel_label(axes[0], "A", -0.22, 1.05)
    panel_label(axes[1], "B", -0.16, 1.05)
    save_all(fig, out)
    register(entries, "Supplementary Fig. S1", out, [member_join(BASE_COH, "metrics", "metrics_freq_domain_raw.csv")], "supp")


def plot_s2_of_dumbbell(store: ArtifactStore, entries: list[dict]) -> None:
    df = load_family_summary(store, "COHFACE")
    methods = ["Base", "OSSM-KF", "PARH"]
    direct = df[df["family"].eq("OF")].set_index("variant")
    bridge = df[df["family"].eq("OF_bridge")].set_index("variant")
    panels = [
        ("rate_mae", "Rate MAE", "bpm", "lower is better"),
        ("rate_r", "Rate Pearson r", "r", "higher is better"),
        ("wave_ccc", "Aligned waveform CCC", "CCC", "higher is better"),
        ("wave_dtw", "Aligned waveform DTW", "DTW", "lower is better"),
    ]
    out = FIG_DIR / "S_F6_of_construction_comparison.pdf"
    fig, axes = plt.subplots(2, 2, figsize=figure_size(SUPP_WIDTH_MM, 112))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.11, top=0.91, hspace=0.46, wspace=0.28)
    for ax, (metric, title, xlabel, note), label in zip(axes.ravel(), panels, "ABCD"):
        left = [float(direct.loc[m, metric]) if m in direct.index else np.nan for m in methods]
        right = [float(bridge.loc[m, metric]) if m in bridge.index else np.nan for m in methods]
        dumbbell(ax, methods, left, right, left_label="OF", right_label="OF bridge", xlabel=xlabel)
        ax.set_title(title, loc="left")
        ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=PALETTE["muted"])
        panel_label(ax, label, -0.17, 1.08)
    save_all(fig, out)
    register(entries, "Supplementary Fig. S2", out, [member_join(BASE_COH, "metrics", "metrics_freq_domain_raw.csv")], "supp")


def plot_s3_preproc(entries: list[dict]) -> None:
    out = FIG_DIR / "S_F5_preproc_delta_heatmaps.pdf"
    stages = MAIN_STAGE_LABELS
    datasets = ["COHFACE", "MAHNOB-HCI"]
    fig, axes = plt.subplots(2, 1, figsize=figure_size(SUPP_WIDTH_MM, 200))
    fig.subplots_adjust(left=0.16, right=0.92, bottom=0.10, top=0.92, hspace=0.34)
    for ax, dataset, label in zip(axes, datasets, "AB"):
        arr = np.vstack(
            [
                FALLBACK_STAGE_VALUES[dataset]["corr_wave_best_median"],
                FALLBACK_STAGE_VALUES[dataset]["highfreq_energy_ratio_median"],
            ]
        )
        im = annotated_heatmap(
            ax,
            arr,
            row_labels=["Waveform evidence", "High-frequency burden"],
            col_labels=stages,
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            fmt="{:.2f}",
        )
        ax.set_title(dataset, loc="left")
        panel_label(ax, label, -0.12, 1.06)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.028, pad=0.025)
    cbar.set_label("Median diagnostic value")
    save_all(fig, out)
    register(entries, "Supplementary Fig. S3", out, ["scripts/plot_observation_eda.py:FALLBACK_STAGE_VALUES"], "supp")


def load_wave_rows(store: ArtifactStore, run: str, dataset: str) -> pd.DataFrame:
    wave = store.read_csv(member_join(run, "metrics", "metrics_waveform_raw.csv"))
    wave[["family", "variant"]] = wave["method"].apply(lambda x: pd.Series(classify_method(x)))
    wave = wave[wave["causal_or_smoothed"].eq("smoothed") & wave["output_type"].eq("signal_hat")].copy()
    wave["dataset"] = dataset
    return wave


def plot_overlay_grid(
    store: ArtifactStore,
    out: Path,
    *,
    rows: list[dict],
    title: str | None = None,
    window_sec: float = 22.0,
    width_mm: float = SUPP_WIDTH_MM,
    height_mm: float = 135.0,
) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=figure_size(width_mm, height_mm), sharey=False)
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.075, top=0.90 if title else 0.95, hspace=0.34, wspace=0.14)
    col_specs = [("Base", "signal_hat", PALETTE["base"]), ("OSSM-KF", "signal_hat", PALETTE["kf"]), ("PARH", "z_full", PALETTE["parh"])]
    for r, row in enumerate(rows):
        payload = store.read_pickle(row["member"])
        gt_for_window = np.asarray(payload["gt"], dtype=float)
        fs = float(payload.get("fs_gt", payload.get("fps", 20.0)))
        start, end = best_window(gt_for_window, fs, window_sec)
        for c, (variant, output, color) in enumerate(col_specs):
            ax = axes[r, c]
            pred, gt, fs = extract_estimate(payload, row["methods"][variant], output)
            t, pred_w, gt_w = window_pair(pred, gt, fs, start, end, align=True)
            mtext = ""
            if "metric_rows" in row and variant in row["metric_rows"]:
                mtext = metric_text(row["metric_rows"][variant])
            label = f"{row['dataset']} / {row['label']}" if c == 0 else ""
            plot_waveform_panel(
                ax,
                t,
                gt_w,
                pred_w,
                method_label=variant,
                color=color,
                title=variant if r == 0 else None,
                metrics=mtext,
                show_xlabel=r == len(rows) - 1,
                show_ylabel=c == 0,
            )
            if label:
                ax.set_ylabel(f"{label}\nz amplitude")
    if title:
        fig.suptitle(title, y=0.982, fontsize=8.8, fontweight="bold")
    save_all(fig, out)


def metric_rows_for_case(store: ArtifactStore, run: str, video: str, family: str) -> dict[str, pd.Series]:
    wave = store.read_csv(member_join(run, "metrics", "metrics_waveform_raw.csv"))
    wave[["family", "variant"]] = wave["method"].apply(lambda x: pd.Series(classify_method(x)))
    out = {}
    for variant in ["Base", "OSSM-KF", "PARH"]:
        sub = wave[
            wave["video"].eq(video)
            & wave["family"].eq(family)
            & wave["variant"].eq(variant)
            & wave["causal_or_smoothed"].eq("smoothed")
            & wave["output_type"].eq("signal_hat")
        ]
        if not sub.empty:
            out[variant] = sub.iloc[0]
    return out


def construction_overlay(store: ArtifactStore, entries: list[dict], *, kind: str, out_name: str, fig_id: str) -> None:
    family_pairs = {"OF": ("OF", "OF_bridge"), "DoF": ("DoF", "DoF_bridge")}
    direct, bridge = family_pairs[kind]
    rows = []
    for dataset, run, video in [
        ("COHFACE", BASE_COH, "cohface_17_1"),
        ("MAHNOB-HCI", BASE_MAH, "mahnob_1042"),
    ]:
        for family in [direct, bridge]:
            rows.append(
                {
                    "dataset": dataset,
                    "label": FAMILY_LABEL[family],
                    "member": member_join(run, "data", f"{video}.pkl"),
                    "methods": {"Base": DIRECT_METHOD[family], "OSSM-KF": KF_METHOD[family], "PARH": PARH_METHOD[family]},
                    "metric_rows": metric_rows_for_case(store, run, video, family),
                }
            )
    out = FIG_DIR / out_name
    plot_overlay_grid(store, out, rows=rows, title=f"{kind} construction overlay", height_mm=154)
    register(entries, fig_id, out, [r["member"] for r in rows], "supp")


def plot_s6_operating_point(store: ArtifactStore, entries: list[dict]) -> None:
    settings = [
        ("locked_default", "Locked default"),
        ("more_local_windows", "More local windows"),
        ("more_stable_windows", "More stable windows"),
        ("stricter_cross_family_support", "Stricter support"),
        ("looser_cross_family_support", "Looser support"),
    ]
    rows = []
    for key, label in settings:
        for dataset, run_ds in [("COHFACE", "cohface"), ("MAHNOB-HCI", "mahnob_tailaligned")]:
            run = f"results/final_operating_point_sensitivity/{key}/{run_ds}"
            if not store.exists(member_join(run, "metrics", "metrics_freq_domain_raw.csv")):
                continue
            freq = store.read_csv(member_join(run, "metrics", "metrics_freq_domain_raw.csv"))
            wave_path = member_join(run, "metrics", "metrics_waveform_raw.csv")
            strict_path = member_join(run, "metrics", "metrics_waveform_strict_raw.csv")
            wave = store.read_csv(wave_path) if store.exists(wave_path) else pd.DataFrame()
            strict = store.read_csv(strict_path) if store.exists(strict_path) else pd.DataFrame()
            if not wave.empty:
                wave = wave[wave["causal_or_smoothed"].eq("smoothed") & wave["output_type"].eq("z_full")]
            if not strict.empty:
                strict = strict[strict["causal_or_smoothed"].eq("smoothed") & strict["output_type"].eq("z_full")]
            rows.append(
                {
                    "setting": label,
                    "dataset": dataset,
                    "rate_mae": _median(freq["MAE"]),
                    "wave_ccc": _median(wave["waveform_CCC"]) if not wave.empty else np.nan,
                    "strict_ccc": _median(strict["strict_CCC"]) if not strict.empty else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    out = FIG_DIR / "S_F_component_ablation_evidence.pdf"
    fig, axes = plt.subplots(1, 3, figsize=figure_size(SUPP_WIDTH_MM, 90))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.87, wspace=0.30)
    order = [label for _key, label in settings]
    for ax, metric, title, xlabel, note, letter in [
        (axes[0], "rate_mae", "Timing evidence", "Rate MAE (bpm)", "lower is better", "A"),
        (axes[1], "wave_ccc", "Morphology evidence", "Aligned CCC", "higher is better", "B"),
        (axes[2], "strict_ccc", "Strict evidence", "Strict CCC", "higher is better", "C"),
    ]:
        horizontal_dotplot(
            ax,
            df,
            y_col="setting",
            x_col=metric,
            hue_col="dataset",
            y_order=order,
            hue_order=["COHFACE", "MAHNOB-HCI"],
            xlabel=xlabel,
            direction=note,
        )
        for coll, dataset in zip(ax.collections, ["COHFACE", "MAHNOB-HCI"]):
            coll.set_color(PALETTE["cohface"] if dataset == "COHFACE" else PALETTE["mahnob"])
        ax.set_title(title, loc="left")
        panel_label(ax, letter, -0.20, 1.07)
    axes[2].legend(loc="lower right", frameon=False)
    save_all(fig, out)
    register(entries, "Supplementary Fig. S6", out, ["results/final_operating_point_sensitivity/*"], "supp")


def load_decoupled_summary(store: ArtifactStore, run: str) -> dict:
    summary_path = member_join(run, "metrics", "decoupled_system_summary.json")
    if store.exists(summary_path):
        summary = store.read_json(summary_path)
        return {
            "rate_mae": float(summary["rate"]["MAE"]),
            "aligned_ccc": float(summary["waveform"]["waveform_CCC"]),
            "strict_ccc": float(summary["waveform_strict"]["strict_CCC"]),
            "consistency": float(summary["consistency"]["consistency_score"]),
            "track_diff": float(summary["consistency"]["rate_waveform_track_abs_diff_bpm"]),
            "confidence": float(summary["consistency"]["system_confidence_score"]),
        }
    freq = store.read_csv(member_join(run, "metrics", "metrics_freq_domain_raw.csv"))
    wave = store.read_csv(member_join(run, "metrics", "metrics_waveform_raw.csv"))
    strict = store.read_csv(member_join(run, "metrics", "metrics_waveform_strict_raw.csv"))
    cons = store.read_csv(member_join(run, "metrics", "decoupled_system_consistency.csv"))
    return {
        "rate_mae": _median(freq["MAE"]),
        "aligned_ccc": _median(wave["waveform_CCC"]),
        "strict_ccc": _median(strict["strict_CCC"]),
        "consistency": _median(cons["consistency_score"]),
        "track_diff": _median(cons["rate_waveform_track_abs_diff_bpm"]),
        "confidence": _median(cons["system_confidence_score"]),
    }


def plot_s7_within_transfer(store: ArtifactStore, entries: list[dict]) -> None:
    runs = [
        ("COHFACE within", "Default", "results/full_decoupled_validation_suite_v2/cohface_within"),
        ("COHFACE within", "Robust", "results/full_decoupled_validation_suite_v2/cohface_within__robust"),
        ("COHFACE to MAHNOB-HCI", "Default", "results/full_decoupled_validation_suite_v2/cohface_to_mahnob"),
        ("COHFACE to MAHNOB-HCI", "Robust", "results/full_decoupled_validation_suite_v2/cohface_to_mahnob__robust"),
    ]
    rows = []
    for scope, variant, run in runs:
        vals = load_decoupled_summary(store, run)
        vals.update({"scope": scope, "variant": variant})
        rows.append(vals)
    df = pd.DataFrame(rows)
    scope_label_map = {
        "COHFACE within": "COHFACE\nwithin",
        "COHFACE to MAHNOB-HCI": "COHFACE ->\nMAHNOB-HCI",
    }
    df["scope_label"] = df["scope"].map(scope_label_map).fillna(df["scope"])
    out = FIG_DIR / "S_F12_within_transfer_compact_comparison.pdf"
    fig, axes = plt.subplots(2, 2, figsize=figure_size(SUPP_WIDTH_MM, 170))
    axes = axes.ravel()
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.13, top=0.91, hspace=0.46, wspace=0.30)
    metrics = [
        ("aligned_ccc", "Aligned CCC", "higher is better"),
        ("strict_ccc", "Strict CCC", "higher is better"),
        ("consistency", "Consistency", "higher is better"),
        ("confidence", "System confidence", "higher is better"),
    ]
    order = ["COHFACE\nwithin", "COHFACE ->\nMAHNOB-HCI"]
    for ax, (metric, title, note), letter in zip(axes, metrics, "ABCD"):
        horizontal_dotplot(
            ax,
            df,
            y_col="scope_label",
            x_col=metric,
            hue_col="variant",
            y_order=order,
            hue_order=["Default", "Robust"],
            xlabel=title,
            direction=None,
        )
        for coll, color in zip(ax.collections, [PALETTE["base"], PALETTE["parh"]]):
            coll.set_color(color)
        ax.text(0.98, 1.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=PALETTE["muted"])
        ax.set_title(title, loc="left")
        panel_label(ax, letter, -0.22, 1.08)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["base"], markeredgecolor=PALETTE["base"], label="Default"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["parh"], markeredgecolor=PALETTE["parh"], label="Robust"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.60, 0.035))
    save_all(fig, out)
    register(entries, "Supplementary Fig. S7", out, [r[2] for r in runs], "supp")


def plot_s8_state_bundle(store: ArtifactStore, entries: list[dict]) -> None:
    member = member_join(FINAL_COH, "data", "cohface_17_1.pkl")
    payload = store.read_pickle(member)
    est = payload["estimates"][0]["estimate"]
    pred, gt, fs = extract_estimate(payload, "parh_ossm", "z_full")
    start, end = best_window(gt, fs, 24)
    t, pred_w, gt_w = window_pair(pred, gt, fs, start, end, align=True)
    out = FIG_DIR / "S_F9_state_bundle_diagnostics.pdf"
    fig = plt.figure(figsize=figure_size(SUPP_WIDTH_MM, 112))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.0], width_ratios=[1.15, 1.0])
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.88, hspace=0.38, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    plot_waveform_panel(ax, t, gt_w, pred_w, method_label="PARH", color=PALETTE["parh"], show_xlabel=False, show_ylabel=True)
    ax.set_title("Observed reference and reconstructed readout", loc="left")
    panel_label(ax, "A", -0.12, 1.08)
    ax = fig.add_subplot(gs[0, 1])
    diag = est["diagnostics"]
    tt = np.arange(len(diag["q_obs_t"])) / float(payload.get("fps", 20.0))
    for key, label, color in [
        ("q_obs_t", "observation trust", PALETTE["cohface"]),
        ("q_osc_t", "oscillatory support", PALETTE["parh"]),
        ("obs_nonosc_need_eff_t", "non-oscillatory need", PALETTE["mahnob"]),
    ]:
        ax.plot(tt, diag[key], label=label, color=color, linewidth=1.0)
    clean_axis(ax, "y")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Adaptor trajectories", loc="left")
    ax.legend(loc="lower right", frameon=False)
    panel_label(ax, "B", -0.12, 1.08)
    ax = fig.add_subplot(gs[1, 0])
    decomp = est["decomposition"]
    for key, color in [("h1", PALETTE["cohface"]), ("h2", PALETTE["kf"]), ("baseline", PALETTE["muted"]), ("residual", PALETTE["parh"])]:
        arr = zscore(decomp[key])
        ax.plot(np.arange(len(arr)) / float(payload.get("fps", 20.0)), arr, label=key, linewidth=0.9, color=color)
    clean_axis(ax, "y")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("z amplitude")
    ax.set_title("Latent state roles", loc="left")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    panel_label(ax, "C", -0.12, 1.08)
    ax = fig.add_subplot(gs[1, 1])
    mix = np.asarray(diag.get("mixture_t_channels"), dtype=float)
    if mix.ndim == 2:
        im = ax.imshow(mix[:, :: max(1, mix.shape[1] // 220)], aspect="auto", cmap="viridis", vmin=0, vmax=np.nanmax(mix))
        ax.set_yticks(range(8))
        ax.set_yticklabels([FAMILY_LABEL[f] for f in FAMILY_ORDER])
        ax.set_xticks([])
        ax.set_xlabel("Time")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="mix weight")
    ax.set_title("Observation mixture by class", loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "D", -0.12, 1.08)
    save_all(fig, out)
    register(entries, "Supplementary Fig. S8", out, [member], "supp")


def plot_s9_confidence(store: ArtifactStore, entries: list[dict]) -> None:
    run = "results/full_decoupled_validation_suite_v2/cohface_within__robust"
    cons = store.read_csv(member_join(run, "metrics", "consistency_raw.csv"))
    strict_path = member_join(run, "metrics", "metrics_waveform_strict_raw.csv")
    wave_path = member_join(run, "metrics", "metrics_waveform_raw.csv")
    if store.exists(strict_path):
        strict = store.read_csv(strict_path)
        if "causal_or_smoothed" in strict.columns:
            strict = strict[strict["causal_or_smoothed"].eq("smoothed")]
        strict = strict.sort_values("video").drop_duplicates("video", keep="first")
        cons = cons.merge(strict[["video", "strict_CCC"]], on="video", how="left")
    if store.exists(wave_path):
        wave = store.read_csv(wave_path)
        if "causal_or_smoothed" in wave.columns:
            wave = wave[wave["causal_or_smoothed"].eq("smoothed")]
        wave = wave.sort_values("video").drop_duplicates("video", keep="first")
        cons = cons.merge(wave[["video", "waveform_CCC"]], on="video", how="left")
    out = FIG_DIR / "S_F10_decoupled_system_diagnostics.pdf"
    fig, axes = plt.subplots(2, 3, figsize=figure_size(SUPP_WIDTH_MM, 116))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.91, hspace=0.46, wspace=0.30)
    ax = axes[0, 0]
    for col, label, color in [
        ("consistency_score", "consistency", PALETTE["cohface"]),
        ("rate_confidence_score", "rate", PALETTE["kf"]),
        ("waveform_confidence_score", "waveform", PALETTE["parh"]),
        ("system_confidence_score", "system", PALETTE["mahnob"]),
    ]:
        vals = pd.to_numeric(cons[col], errors="coerce").dropna()
        ax.hist(vals, bins=np.linspace(0, 1, 18), histtype="step", linewidth=1.1, color=color, label=label)
    clean_axis(ax, "y")
    ax.set_xlabel("Score")
    ax.set_ylabel("Trials")
    ax.set_title("Confidence distributions", loc="left")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    y_strict = pd.to_numeric(cons.get("strict_CCC", np.nan), errors="coerce")
    ax.scatter(cons["system_confidence_score"], y_strict, s=14, color=PALETTE["parh"], alpha=0.65)
    clean_axis(ax, "both")
    ax.set_xlabel("System confidence")
    ax.set_ylabel("Strict CCC")
    ax.set_title("Confidence vs strict fidelity", loc="left")
    ax = axes[0, 2]
    ax.scatter(cons["rate_waveform_track_abs_diff_bpm"], cons["consistency_score"], s=14, color=PALETTE["cohface"], alpha=0.65)
    clean_axis(ax, "both")
    ax.set_xlabel("Rate-waveform gap (bpm)")
    ax.set_ylabel("Consistency")
    ax.set_title("Agreement diagnostic", loc="left")
    case_videos = ["cohface_39_0", "cohface_40_1", "cohface_9_3"]
    for ax, video, title in zip(axes[1], case_videos, ["Low confidence", "Representative", "High contrast"]):
        member = member_join(FINAL_COH, "data", f"{video}.pkl") if store.exists(member_join(FINAL_COH, "data", f"{video}.pkl")) else member_join(FINAL_COH, "data", "cohface_17_1.pkl")
        payload = store.read_pickle(member)
        pred, gt, fs = extract_estimate(payload, "parh_ossm", "z_full")
        start, end = best_window(gt, fs, 18)
        t, pred_w, gt_w = window_pair(pred, gt, fs, start, end, align=True)
        plot_waveform_panel(ax, t, gt_w, pred_w, method_label="PARH", color=PALETTE["parh"], title=title, show_xlabel=True, show_ylabel=ax is axes[1, 0])
    for ax, label in zip(axes.ravel(), "ABCDEF"):
        panel_label(ax, label, -0.14, 1.08)
    save_all(fig, out)
    register(entries, "Supplementary Fig. S9", out, [member_join(run, "metrics", "consistency_raw.csv")], "supp")


def plot_s10_fallback(store: ArtifactStore, entries: list[dict]) -> None:
    run = "results/full_decoupled_validation_suite_v2/cohface_to_mahnob__robust"
    cons = store.read_csv(member_join(run, "metrics", "consistency_raw.csv"))
    out = FIG_DIR / "S_F11_robust_fallback_diagnostics.pdf"
    fig, axes = plt.subplots(2, 2, figsize=figure_size(SUPP_WIDTH_MM, 110))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.90, hspace=0.42, wspace=0.28)
    ax = axes[0, 0]
    counts = cons["fallback_triggered"].fillna(False).astype(bool).value_counts().reindex([False, True], fill_value=0)
    ax.bar([0, 1], counts.values, color=[PALETTE["base"], PALETTE["parh"]], width=0.58)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["primary", "fallback"])
    ax.set_ylabel("Trials")
    ax.set_title("Fallback activation", loc="left")
    clean_axis(ax, "y")
    ax = axes[0, 1]
    vals = pd.to_numeric(cons["system_confidence_score"], errors="coerce").dropna()
    ax.hist(vals, bins=np.linspace(0, 1, 24), color=PALETTE["parh"], alpha=0.78)
    ax.set_xlabel("System confidence")
    ax.set_ylabel("Trials")
    ax.set_title("Transfer confidence", loc="left")
    clean_axis(ax, "y")
    for ax, video, title in zip(axes[1], ["mahnob_10", "mahnob_1042"], ["Fallback case", "Hard reference case"]):
        member = member_join(FINAL_MAH, "data", f"{video}.pkl") if store.exists(member_join(FINAL_MAH, "data", f"{video}.pkl")) else member_join(FINAL_MAH, "data", "mahnob_1042.pkl")
        payload = store.read_pickle(member)
        pred, gt, fs = extract_estimate(payload, "parh_ossm", "z_full")
        start, end = best_window(gt, fs, 18)
        t, pred_w, gt_w = window_pair(pred, gt, fs, start, end, align=True)
        plot_waveform_panel(ax, t, gt_w, pred_w, method_label="PARH", color=PALETTE["parh"], title=title, show_xlabel=True, show_ylabel=ax is axes[1, 0])
    for ax, label in zip(axes.ravel(), "ABCD"):
        panel_label(ax, label, -0.14, 1.08)
    save_all(fig, out)
    register(entries, "Supplementary Fig. S10", out, [member_join(run, "metrics", "consistency_raw.csv")], "supp")


def method_metric_rows(store: ArtifactStore, dataset: str) -> pd.DataFrame:
    fam = load_family_summary(store, dataset)
    rows = []
    for family in FAMILY_ORDER:
        for variant in ["Base"]:
            sub = fam[fam["family"].eq(family) & fam["variant"].eq(variant)]
            if not sub.empty:
                row = sub.iloc[0].to_dict()
                row["method_label"] = FAMILY_LABEL[family]
                rows.append(row)
    p1d_kf = fam[fam["family"].eq("P1D_quad") & fam["variant"].eq("OSSM-KF")]
    if not p1d_kf.empty:
        row = p1d_kf.iloc[0].to_dict()
        row["method_label"] = "OSSM-KF\n(P1D quadratic)"
        rows.append(row)
    integ = load_integrated_summary(store, dataset)
    integ.update({"dataset": dataset, "method_label": "PARH-OSSM", "family": "PARH-OSSM", "variant": "PARH"})
    rows.append(integ)
    return pd.DataFrame(rows)


def plot_s13_allbase(store: ArtifactStore, entries: list[dict]) -> None:
    df = pd.concat([method_metric_rows(store, "COHFACE"), method_metric_rows(store, "MAHNOB-HCI")], ignore_index=True)
    order = [FAMILY_LABEL[f] for f in FAMILY_ORDER] + ["OSSM-KF\n(P1D quadratic)", "PARH-OSSM"]
    out = FIG_DIR / "F5_mechanism_activation.pdf"
    fig, axes = plt.subplots(1, 3, figsize=figure_size(SUPP_WIDTH_MM, 102))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.13, top=0.89, wspace=0.30)
    for ax, metric, title, xlabel, note, letter in [
        (axes[0], "rate_mae", "Rate error", "MAE (bpm)", "lower is better", "A"),
        (axes[1], "wave_ccc", "Aligned waveform", "CCC", "higher is better", "B"),
        (axes[2], "strict_nmae", "Strict reconstruction", "NMAE / GT span", "lower is better", "C"),
    ]:
        horizontal_dotplot(
            ax,
            df,
            y_col="method_label",
            x_col=metric,
            hue_col="dataset",
            y_order=order,
            hue_order=["COHFACE", "MAHNOB-HCI"],
            xlabel=xlabel,
            direction=note,
        )
        for coll, color in zip(ax.collections, [PALETTE["cohface"], PALETTE["mahnob"]]):
            coll.set_color(color)
        if metric == "strict_nmae":
            ax.set_xscale("log")
        ax.set_title(title, loc="left")
        panel_label(ax, letter, -0.23, 1.06)
    axes[-1].legend(loc="lower right", frameon=False)
    save_all(fig, out)
    register(entries, "Supplementary Fig. S13", out, [BASE_COH, BASE_MAH, FINAL_COH, FINAL_MAH], "supp")


def plot_s14_failure(store: ArtifactStore, entries: list[dict]) -> None:
    out = FIG_DIR / "F6_failure_cases.pdf"
    payload = store.read_pickle(member_join(FINAL_MAH, "data", "mahnob_1042.pkl"))
    est = payload["estimates"][0]["estimate"]
    pred, gt, fs = extract_estimate(payload, "parh_ossm", "z_full")
    start, end = best_window(gt, fs, 18)
    t, pred_w, gt_w = window_pair(pred, gt, fs, start, end, align=True)
    diag = est["diagnostics"]
    fig, axes = plt.subplots(2, 2, figsize=figure_size(SUPP_WIDTH_MM, 112))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.88, hspace=0.40, wspace=0.26)
    plot_waveform_panel(axes[0, 0], t, gt_w, pred_w, method_label="PARH", color=PALETTE["parh"], title="Residual-heavy waveform case", show_ylabel=True)
    tt = np.arange(len(diag["residual_gate_t"])) / float(payload.get("fps", 20.0))
    axes[0, 1].plot(tt, diag["residual_gate_t"], color=PALETTE["parh"], label="residual gate")
    axes[0, 1].plot(tt, diag["obs_nonosc_need_eff_t"], color=PALETTE["mahnob"], label="non-osc need")
    axes[0, 1].set_ylim(-0.02, 1.05)
    axes[0, 1].set_title("Residual diagnostics", loc="left")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend(frameon=False)
    clean_axis(axes[0, 1], "y")
    base_payload = store.read_pickle(member_join(BASE_MAH, "data", "mahnob_1042.pkl"))
    pred2, gt2, fs2 = extract_estimate(base_payload, "of_disp_bridge", "signal_hat")
    start2, end2 = best_window(gt2, fs2, 18)
    t2, pred2_w, gt2_w = window_pair(pred2, gt2, fs2, start2, end2, align=True)
    plot_waveform_panel(axes[1, 0], t2, gt2_w, pred2_w, method_label="Direct", color=PALETTE["base"], title="Weak bridge-observation case", show_xlabel=True, show_ylabel=True)
    rates = load_family_summary(store, "MAHNOB-HCI")
    sub = rates[rates["family"].isin(["OF_bridge", "P1D_quad"])]
    vals = [
        _median(sub[sub["family"].eq("OF_bridge") & sub["variant"].eq("Base")]["rate_mae"]),
        _median(sub[sub["family"].eq("OF_bridge") & sub["variant"].eq("PARH")]["rate_mae"]),
        load_integrated_summary(store, "MAHNOB-HCI")["rate_mae"],
    ]
    axes[1, 1].barh(["OF bridge", "OF bridge PARH", "Integrated PARH"], vals, color=[PALETTE["base"], PALETTE["kf"], PALETTE["parh"]])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Rate MAE (bpm)")
    axes[1, 1].set_title("Metric context", loc="left")
    clean_axis(axes[1, 1], "x")
    for ax, label in zip(axes.ravel(), "ABCD"):
        panel_label(ax, label, -0.14, 1.08)
    save_all(fig, out)
    register(entries, "Supplementary Fig. S14", out, [member_join(FINAL_MAH, "data", "mahnob_1042.pkl")], "supp")


def plot_same_trial_parts(store: ArtifactStore, entries: list[dict]) -> None:
    specs = [
        ("COHFACE", BASE_COH, FINAL_COH, "cohface_17_1"),
        ("MAHNOB-HCI", BASE_MAH, FINAL_MAH, "mahnob_1042"),
    ]
    methods = [
        ("OF", "of_farneback", "signal_hat", "base", PALETTE["base"]),
        ("OF bridge", "of_disp_bridge", "signal_hat", "base", PALETTE["base"]),
        ("DoF", "DoF", "signal_hat", "base", PALETTE["base"]),
        ("DoF bridge", "dof_disp_bridge", "signal_hat", "base", PALETTE["base"]),
        ("P1D quadratic", "profile1D quadratic", "signal_hat", "base", PALETTE["base"]),
        ("OSSM-KF", "profile1d_quadratic__kfstd", "signal_hat", "base", PALETTE["kf"]),
        ("PARH-OSSM", "parh_ossm", "z_full", "final", PALETTE["parh"]),
    ]
    chunks = [methods[:4], methods[4:]]
    for part, chunk in enumerate(chunks, start=1):
        out = FIG_DIR / f"F4_waveform_overlay_grid_part{part}.pdf"
        fig, axes = plt.subplots(len(specs), len(chunk), figsize=figure_size(SUPP_WIDTH_MM, 125), sharey=False)
        if len(chunk) == 1:
            axes = np.expand_dims(axes, axis=1)
        fig.subplots_adjust(left=0.145, right=0.985, bottom=0.12, top=0.88, hspace=0.36, wspace=0.16)
        for r, (dataset, base_run, final_run, video) in enumerate(specs):
            ref_payload = store.read_pickle(member_join(final_run, "data", f"{video}.pkl"))
            _, gt_ref, fs_ref = extract_estimate(ref_payload, "parh_ossm", "z_full")
            start, end = best_window(gt_ref, fs_ref, 20)
            for c, (label, method, output, source, color) in enumerate(chunk):
                payload = ref_payload if source == "final" else store.read_pickle(member_join(base_run, "data", f"{video}.pkl"))
                pred, gt, fs = extract_estimate(payload, method, output)
                t, pred_w, gt_w = window_pair(pred, gt, fs, start, end, align=True)
                ax = axes[r, c]
                plot_waveform_panel(ax, t, gt_w, pred_w, method_label=label, color=color, title=label if r == 0 else None, show_xlabel=r == len(specs) - 1, show_ylabel=c == 0)
                if c == 0:
                    ax.set_ylabel(f"{dataset}\nz amplitude")
        save_all(fig, out)
        register(entries, f"Supplementary Fig. S{10 + part}", out, [BASE_COH, BASE_MAH, FINAL_COH, FINAL_MAH], "supp")
    # Keep a compatibility file for older references; the TeX is patched to use the parts.
    plot_same_trial_compat(store)


def plot_same_trial_compat(store: ArtifactStore) -> None:
    out = FIG_DIR / "F4_waveform_overlay_grid.pdf"
    # A light compatibility contact sheet; manuscript uses split files.
    specs = [
        {"dataset": "COHFACE", "label": "P1D quadratic", "member": member_join(BASE_COH, "data", "cohface_17_1.pkl"), "methods": {"Base": "profile1D quadratic", "OSSM-KF": "profile1d_quadratic__kfstd", "PARH": "profile1d_quadratic__parh_ossm"}},
        {"dataset": "MAHNOB-HCI", "label": "P1D quadratic", "member": member_join(BASE_MAH, "data", "mahnob_1042.pkl"), "methods": {"Base": "profile1D quadratic", "OSSM-KF": "profile1d_quadratic__kfstd", "PARH": "profile1d_quadratic__parh_ossm"}},
    ]
    plot_overlay_grid(store, out, rows=specs, title="Same-trial waveform overlay summary", height_mm=82)


def observation_atlas(store: ArtifactStore, entries: list[dict], *, dataset: str, run: str, video: str, prefix: str, fig_start: int) -> None:
    groups = [FAMILY_ORDER[:4], FAMILY_ORDER[4:]]
    for part, families in enumerate(groups, start=1):
        rows = []
        for family in families:
            rows.append(
                {
                    "dataset": dataset,
                    "label": FAMILY_LABEL[family],
                    "member": member_join(run, "data", f"{video}.pkl"),
                    "methods": {"Base": DIRECT_METHOD[family], "OSSM-KF": KF_METHOD[family], "PARH": PARH_METHOD[family]},
                    "metric_rows": metric_rows_for_case(store, run, video, family),
                }
            )
        out = FIG_DIR / f"{prefix}_part{part}.pdf"
        plot_overlay_grid(store, out, rows=rows, title=f"{dataset} observation-class atlas, part {part}", height_mm=190)
        register(entries, f"Supplementary Fig. S{fig_start + part - 1}", out, [r["member"] for r in rows], "supp")
    # Compatibility unsplit file for older external references.
    rows = []
    for family in FAMILY_ORDER:
        rows.append(
            {
                "dataset": dataset,
                "label": FAMILY_LABEL[family],
                "member": member_join(run, "data", f"{video}.pkl"),
                "methods": {"Base": DIRECT_METHOD[family], "OSSM-KF": KF_METHOD[family], "PARH": PARH_METHOD[family]},
                "metric_rows": metric_rows_for_case(store, run, video, family),
            }
        )
    compat = FIG_DIR / f"{prefix}.pdf"
    plot_overlay_grid(store, compat, rows=rows, title=f"{dataset} observation-class atlas", height_mm=260)


def audit(entries: list[dict]) -> None:
    print(json.dumps(entries, indent=2, sort_keys=True))


def build_all(zip_path: Path) -> list[dict]:
    set_publication_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    with ArtifactStore(zip_path) as store:
        plot_main_f2(entries)
        plot_s1_rate_diagnostics(store, entries)
        plot_s2_of_dumbbell(store, entries)
        plot_s3_preproc(entries)
        construction_overlay(store, entries, kind="OF", out_name="S_F13_of_construction_overlay_grid.pdf", fig_id="Supplementary Fig. S4")
        construction_overlay(store, entries, kind="DoF", out_name="S_F14_dof_construction_overlay_grid.pdf", fig_id="Supplementary Fig. S5")
        plot_s6_operating_point(store, entries)
        plot_s7_within_transfer(store, entries)
        plot_s8_state_bundle(store, entries)
        plot_s9_confidence(store, entries)
        plot_s10_fallback(store, entries)
        plot_same_trial_parts(store, entries)
        plot_s13_allbase(store, entries)
        plot_s14_failure(store, entries)
        observation_atlas(
            store,
            entries,
            dataset="COHFACE",
            run=BASE_COH,
            video="cohface_17_1",
            prefix="S_F15_cohface_observation_class_overlay_atlas",
            fig_start=15,
        )
        observation_atlas(
            store,
            entries,
            dataset="MAHNOB-HCI",
            run=BASE_MAH,
            video="mahnob_1042",
            prefix="S_F16_mahnob_observation_class_overlay_atlas",
            fig_start=17,
        )
    save_manifest(entries)
    return entries


def expected_manifest(zip_path: Path) -> list[dict]:
    entries: list[dict] = []
    outputs = [
        ("Figure 2", "F2_dataset_and_observation_regime.pdf", "double"),
        ("Supplementary Fig. S1", "F3_rate_observation_class_summary.pdf", "supp"),
        ("Supplementary Fig. S2", "S_F6_of_construction_comparison.pdf", "supp"),
        ("Supplementary Fig. S3", "S_F5_preproc_delta_heatmaps.pdf", "supp"),
        ("Supplementary Fig. S4", "S_F13_of_construction_overlay_grid.pdf", "supp"),
        ("Supplementary Fig. S5", "S_F14_dof_construction_overlay_grid.pdf", "supp"),
        ("Supplementary Fig. S6", "S_F_component_ablation_evidence.pdf", "supp"),
        ("Supplementary Fig. S7", "S_F12_within_transfer_compact_comparison.pdf", "supp"),
        ("Supplementary Fig. S8", "S_F9_state_bundle_diagnostics.pdf", "supp"),
        ("Supplementary Fig. S9", "S_F10_decoupled_system_diagnostics.pdf", "supp"),
        ("Supplementary Fig. S10", "S_F11_robust_fallback_diagnostics.pdf", "supp"),
        ("Supplementary Fig. S11", "F4_waveform_overlay_grid_part1.pdf", "supp"),
        ("Supplementary Fig. S12", "F4_waveform_overlay_grid_part2.pdf", "supp"),
        ("Supplementary Fig. S13", "F5_mechanism_activation.pdf", "supp"),
        ("Supplementary Fig. S14", "F6_failure_cases.pdf", "supp"),
        ("Supplementary Fig. S15", "S_F15_cohface_observation_class_overlay_atlas_part1.pdf", "supp"),
        ("Supplementary Fig. S16", "S_F15_cohface_observation_class_overlay_atlas_part2.pdf", "supp"),
        ("Supplementary Fig. S17", "S_F16_mahnob_observation_class_overlay_atlas_part1.pdf", "supp"),
        ("Supplementary Fig. S18", "S_F16_mahnob_observation_class_overlay_atlas_part2.pdf", "supp"),
    ]
    for fig_id, name, width in outputs:
        register(entries, fig_id, FIG_DIR / name, [str(zip_path)], width)
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate publication-quality manuscript figures except Figure 1.")
    ap.add_argument("--zip", "--results-zip", dest="zip", type=Path, default=DEFAULT_RESULTS_ZIP)
    ap.add_argument("--all", action="store_true", help="Regenerate all figures.")
    ap.add_argument("--audit", action="store_true", help="List expected inputs and outputs.")
    args = ap.parse_args()
    if args.audit:
        audit(expected_manifest(args.zip))
        return
    if not args.all:
        ap.error("Use --all to regenerate figures or --audit to list expected outputs.")
    entries = build_all(args.zip)
    print(f"Generated {len(entries)} figure entries.")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
