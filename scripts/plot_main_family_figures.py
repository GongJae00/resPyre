#!/usr/bin/env python3
"""Generate main-paper observation-class summary figures from table-ready CSVs."""

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

from core.evaluation.figure_style import (
    VARIANT_COLORS,
    add_metric_box,
    family_label,
    save_figure,
    set_manuscript_style,
    style_axis,
    variant_label,
)
from components.observations.semantics import (
    CANONICAL_OBSERVATION_FAMILY_ORDER,
    get_observation_family_semantics,
)


FAMILY_ORDER = [
    str(get_observation_family_semantics(name).get("display_name"))
    for name in CANONICAL_OBSERVATION_FAMILY_ORDER
]
VARIANT_ORDER = ["Base", "KFstd", "PARH"]


def classify_method(method_name: str):
    m = str(method_name).lower().replace(" ", "_")
    if "__parh_ossm" in m:
        variant = "PARH"
        family = m.split("__parh_ossm")[0]
    elif "__kfstd" in m:
        variant = "KFstd"
        family = m.split("__kfstd")[0]
    else:
        variant = "Base"
        family = m
    family_map = {
        "of_farneback": "OF",
        "of_disp_bridge": "OF_bridge",
        "of": "OF",
        "dof": "DoF",
        "dof_disp_bridge": "DoF_bridge",
        "profile1d_linear": "P1D_lin",
        "profile1d_quadratic": "P1D_quad",
        "profile1d_cubic": "P1D_cub",
        "profile1d_consensus": "P1D_cons",
    }
    return family_map.get(family), variant


def ordered_families(families, strict=True):
    seen = [f for f in FAMILY_ORDER if f in set(families)]
    if strict:
        return seen
    rest = [f for f in families if f not in seen]
    return seen + rest




def build_family_summary_from_master(master_csv: Path, dataset: str = "COHFACE") -> pd.DataFrame:
    master = pd.read_csv(master_csv)
    master = master[(master["dataset"] == dataset) & (master["variant"].isin(VARIANT_ORDER))].copy()
    master = master[master["family"].isin(FAMILY_ORDER)].copy()
    rows = []
    for family in ordered_families(list(master["family"].astype(str)), strict=True):
        fdf = master[master["family"] == family]
        row = {"dataset": dataset, "family": family}
        for variant in VARIANT_ORDER:
            vdf = fdf[fdf["variant"] == variant]
            row[f"{variant}_MAE"] = float(vdf["rate_MAE"].median()) if not vdf.empty else np.nan
            row[f"{variant}_RMSE"] = float(vdf["rate_RMSE"].median()) if not vdf.empty else np.nan
            row[f"{variant}_PearsonR"] = float(vdf["rate_PearsonR"].median()) if not vdf.empty else np.nan
            row[f"{variant}_N"] = int(len(vdf))
        rows.append(row)
    return pd.DataFrame(rows)


def build_family_summary_from_freq_raw(freq_csv: Path, dataset: str = "COHFACE") -> pd.DataFrame:
    freq = pd.read_csv(freq_csv)
    freq[["family", "variant"]] = freq["method"].apply(lambda m: pd.Series(classify_method(m)))
    freq = freq[freq["family"].isin(FAMILY_ORDER) & freq["variant"].isin(VARIANT_ORDER)].copy()
    rows = []
    for family in ordered_families(list(freq["family"].astype(str)), strict=True):
        fdf = freq[freq["family"] == family]
        row = {"dataset": dataset, "family": family}
        for variant in VARIANT_ORDER:
            vdf = fdf[fdf["variant"] == variant]
            row[f"{variant}_MAE"] = float(pd.to_numeric(vdf["MAE"], errors="coerce").median()) if not vdf.empty else np.nan
            row[f"{variant}_RMSE"] = float(pd.to_numeric(vdf["RMSE"], errors="coerce").median()) if not vdf.empty else np.nan
            row[f"{variant}_PearsonR"] = float(pd.to_numeric(vdf["PearsonR"], errors="coerce").median()) if not vdf.empty else np.nan
            row[f"{variant}_N"] = int(len(vdf))
        rows.append(row)
    return pd.DataFrame(rows)

def _headline_reference(headline_df: pd.DataFrame, dataset: str, suffix: str) -> float | None:
    if headline_df is None or headline_df.empty:
        return None
    if "dataset" not in headline_df.columns or f"PARH_{suffix}" not in headline_df.columns:
        return None
    sub = headline_df[headline_df["dataset"].astype(str).str.upper().str.startswith(dataset.upper())]
    if sub.empty:
        return None
    value = pd.to_numeric(sub.iloc[0].get(f"PARH_{suffix}", np.nan), errors="coerce")
    return float(value) if np.isfinite(value) else None


def plot_t3_family_summary(t3_df: pd.DataFrame, out_path: Path, headline_df: pd.DataFrame | None = None):
    families = ordered_families(list(t3_df["family"].astype(str)), strict=True)
    if not families:
        raise ValueError("T3 summary has no known observation families to plot.")
    t3_df = t3_df[t3_df["family"].isin(families)].copy()
    x = np.arange(len(families))

    set_manuscript_style("paper")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.80, wspace=0.12)
    panels = [
        ("MAE", "Rate MAE", False),
        ("PearsonR", "Rate Pearson r", True),
    ]

    for ax, (suffix, ylabel, higher_better) in zip(axes, panels):
        style_axis(ax, grid="y")
        for variant in VARIANT_ORDER:
            y = []
            for family in families:
                row = t3_df.loc[t3_df["family"] == family].iloc[0]
                value = pd.to_numeric(pd.Series([row.get(f"{variant}_{suffix}", np.nan)]), errors="coerce").iloc[0]
                y.append(float(value) if pd.notna(value) else np.nan)
            y = np.asarray(y, dtype=float)
            if not np.isfinite(y).any():
                continue
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=6.5,
                label="class-local PARH" if variant == "PARH" else variant_label(variant),
                color=VARIANT_COLORS[variant],
            )
        ref_value = _headline_reference(headline_df, "COHFACE", suffix)
        if ref_value is not None:
            ax.axhline(
                ref_value,
                color="#111111",
                linestyle="--",
                linewidth=1.6,
                alpha=0.80,
                label="integrated PARH-OSSM" if suffix == "MAE" else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([family_label(f) for f in families], rotation=0, ha="center")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.04)
        add_metric_box(ax, "lower is better" if not higher_better else "higher is better", loc="upper right", fontsize=8.2)
    axes[0].set_title("Oscillatory output accuracy across observation classes", loc="center")
    axes[1].set_title("Rate correlation across observation classes", loc="center")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=len(handles),
            frameon=False,
            handlelength=2.4,
            columnspacing=1.6,
            borderaxespad=0.0,
        )
    save_figure(fig, out_path)


def plot_mechanism_activation(t6_df: pd.DataFrame, mech_df: pd.DataFrame, out_path: Path):
    families = FAMILY_ORDER
    if "dataset" in t6_df.columns:
        t6_df = t6_df[t6_df["dataset"].astype(str).str.upper().str.startswith("COHFACE")].copy()
    if "family" in t6_df.columns:
        t6_df = t6_df.drop_duplicates("family", keep="first")
    mech_df = mech_df.set_index("family").reindex(families)
    t6_df = t6_df.set_index("family").reindex(families)
    x = np.arange(len(families))
    width = 0.32

    def _family_labels():
        return [family_label(f) for f in families]

    def _shade_missing(ax, missing_idx, label_y):
        for idx in missing_idx:
            ax.axvspan(idx - 0.5, idx + 0.5, color="#d9d9d9", alpha=0.18, zorder=0)
            ax.text(idx, label_y, "N/A", ha="center", va="bottom", fontsize=8, color="#666")

    set_manuscript_style("paper")
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.3), constrained_layout=True)

    ax = axes[0]
    style_axis(ax, grid="y")
    q_dyn = np.asarray([
        pd.to_numeric(pd.Series([mech_df.loc[f, "q_dyn_mean_median"]]), errors="coerce").iloc[0]
        for f in families
    ], dtype=float)
    q_osc = np.asarray([
        pd.to_numeric(pd.Series([mech_df.loc[f, "q_osc_mean_median"]]), errors="coerce").iloc[0]
        for f in families
    ], dtype=float)
    obs_need = np.asarray([
        pd.to_numeric(pd.Series([mech_df.loc[f, "obs_nonosc_need_mean_median"]]), errors="coerce").iloc[0]
        for f in families
    ], dtype=float)
    ax.plot(x, q_dyn, marker="o", linewidth=2.0, color="#7b4ea3", label=r"$q_{dyn}$")
    ax.plot(x, q_osc, marker="s", linewidth=2.0, color="#1a7f64", label=r"$q_{osc}$")
    ax.plot(x, obs_need, marker="^", linewidth=2.0, color="#c85f2b", label="nonosc need")
    ax.set_xticks(x)
    ax.set_xticklabels(_family_labels(), rotation=0, ha="center")
    ax.set_ylim(0.0, 1.05)
    missing_idx = [i for i in range(len(families)) if not (np.isfinite(q_dyn[i]) or np.isfinite(q_osc[i]) or np.isfinite(obs_need[i]))]
    _shade_missing(ax, missing_idx, 0.02)
    ax.set_title("Mechanism activation", loc="center")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    style_axis(ax, grid="both")
    valid_points = []
    missing_families = []
    for family in families:
        row = t6_df.loc[family]
        xval = pd.to_numeric(pd.Series([row.get("PARH_NIS_Mean", np.nan)]), errors="coerce").iloc[0]
        yval = pd.to_numeric(pd.Series([row.get("PARH_NIS_InBand", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(xval) and np.isfinite(yval):
            valid_points.append((family, float(xval), float(yval)))
        else:
            missing_families.append(family)
    for family, xval, yval in valid_points:
        ax.scatter(xval, yval, s=70, color="#116b4f" if family != "OF_bridge" else "#7b4ea3")
        ax.annotate(family_label(family).replace("\n", " "), (xval, yval), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axvline(1.0, color="#888", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Median NIS Mean")
    ax.set_ylabel("Median NIS In-Band")
    ax.set_title("Calibration position", loc="center")
    if missing_families:
        miss = ", ".join(family_label(f).replace("\n", " ") for f in missing_families)
        ax.text(0.02, 0.02, f"N/A: {miss}", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8, color="#666")

    ax = axes[2]
    style_axis(ax, grid="y")
    baseline = np.asarray([
        pd.to_numeric(pd.Series([mech_df.loc[f, "baseline_energy_ratio_median"]]), errors="coerce").iloc[0]
        for f in families
    ], dtype=float)
    residual = np.asarray([
        pd.to_numeric(pd.Series([mech_df.loc[f, "residual_energy_ratio_median"]]), errors="coerce").iloc[0]
        for f in families
    ], dtype=float)
    ax.bar(x - width / 2, baseline, width=width, color="#5b7db1", label="baseline ratio")
    ax.bar(x + width / 2, residual, width=width, color="#d17a3a", label="residual ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(_family_labels(), rotation=0, ha="center")
    finite_vals = np.concatenate([baseline[np.isfinite(baseline)], residual[np.isfinite(residual)]]) if (np.isfinite(baseline).any() or np.isfinite(residual).any()) else np.asarray([1.0])
    ymax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    ax.set_ylim(0.0, ymax * 1.15 if ymax > 0 else 1.0)
    missing_idx = [i for i in range(len(families)) if not (np.isfinite(baseline[i]) or np.isfinite(residual[i]))]
    _shade_missing(ax, missing_idx, ax.get_ylim()[1] * 0.02)
    ax.set_ylabel("Energy Ratio")
    ax.set_title("Baseline vs residual share", loc="center")
    ax.legend(frameon=False, loc="upper right")

    save_figure(fig, out_path)


def _rung_label(rung: str) -> str:
    labels = {
        "direct_OF": "OF\nbase",
        "direct_DoF": "DoF\nbase",
        "direct_P1D_quad": "P1D quad\nbase",
        "direct_P1D_cons": "P1D cons\nbase",
        "OSSM-KF (P1D quad)": "OSSM-KF\n(P1D quad)",
        "PARH-OSSM": "PARH-OSSM",
    }
    return labels.get(str(rung), str(rung).replace("_", " "))


def _allbase_rung_label(rung: str) -> str:
    labels = {
        "Base_OF": "OF\nbase",
        "Base_OF_bridge": "OF bridge\nbase",
        "Base_DoF": "DoF\nbase",
        "Base_DoF_bridge": "DoF bridge\nbase",
        "Base_P1D_lin": "P1D lin\nbase",
        "Base_P1D_quad": "P1D quad\nbase",
        "Base_P1D_cub": "P1D cub\nbase",
        "Base_P1D_cons": "P1D cons\nbase",
        "OSSM-KF (P1D quad)": "OSSM-KF\n(P1D quad)",
        "PARH-OSSM": "PARH-OSSM",
    }
    return labels.get(str(rung), str(rung).replace("_", " "))


def _allbase_ladder_from_master(master_df: pd.DataFrame) -> pd.DataFrame:
    if "family" not in master_df.columns and "observation_class" in master_df.columns:
        master_df = master_df.rename(columns={"observation_class": "family"})
    if "variant" in master_df.columns:
        master_df = master_df.copy()
        master_df["variant"] = master_df["variant"].replace({"class-local PARH": "family_PARH"})
    rows = []
    for dataset in ["COHFACE", "MAHNOB"]:
        ddf = master_df[master_df["dataset"].astype(str) == dataset].copy()
        if ddf.empty:
            continue
        for family in FAMILY_ORDER:
            sub = ddf[(ddf["family"].astype(str) == family) & (ddf["variant"].astype(str) == "Base")]
            if sub.empty:
                continue
            row = sub.iloc[0].copy()
            row["rung"] = f"Base_{family}"
            rows.append(row)
        sub = ddf[(ddf["family"].astype(str) == "P1D_quad") & (ddf["variant"].astype(str) == "OSSM-KF")]
        if not sub.empty:
            row = sub.iloc[0].copy()
            row["rung"] = "OSSM-KF (P1D quad)"
            rows.append(row)
        sub = ddf[(ddf["family"].astype(str) == "PARH-OSSM") & (ddf["variant"].astype(str) == "PARH-OSSM")]
        if not sub.empty:
            row = sub.iloc[0].copy()
            row["rung"] = "PARH-OSSM"
            rows.append(row)
    return pd.DataFrame(rows)


def plot_allbase_fusion_ladder(master_df: pd.DataFrame, out_path: Path):
    """Plot all eight fixed observation operators against comparator and PARH.

    This is the paper-facing mechanism ladder. It keeps the main text honest:
    the fixed Base bank is not collapsed to a single representative when the
    figure's purpose is to show the observation-bank topology and hard-regime
    boundary.
    """
    ladder_df = _allbase_ladder_from_master(master_df)
    rung_order = [f"Base_{f}" for f in FAMILY_ORDER] + ["OSSM-KF (P1D quad)", "PARH-OSSM"]
    datasets = [d for d in ["COHFACE", "MAHNOB"] if d in set(ladder_df["dataset"].astype(str))]
    if not datasets:
        raise ValueError("All-base ladder has no known datasets.")
    y = np.arange(len(rung_order))
    colors = {"COHFACE": "#1f3a5f", "MAHNOB": "#b85c2b"}

    set_manuscript_style("paper")
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 6.4), constrained_layout=False, sharey=True)
    fig.subplots_adjust(left=0.205, right=0.988, bottom=0.105, top=0.825, wspace=0.135)
    panels = [
        ("rate_MAE", "Rate MAE (bpm)", "lower is better", True),
        ("waveform_CCC", "Aligned waveform CCC", "higher is better", False),
        ("strict_NMAE_span", "Strict NMAE / GT span", "lower is better", True),
    ]
    for ax, (metric, ylabel, box, log_y) in zip(axes, panels):
        style_axis(ax, grid="x")
        for d_idx, dataset in enumerate(datasets):
            sub = ladder_df[ladder_df["dataset"].astype(str) == dataset].set_index("rung")
            values = pd.to_numeric(sub.reindex(rung_order)[metric], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(values)
            if not finite.any():
                continue
            y_pos = np.arange(len(rung_order), dtype=float) + (d_idx - (len(datasets) - 1) / 2.0) * 0.18
            ax.plot(
                values[finite],
                y_pos[finite],
                marker="o",
                linewidth=1.6,
                markersize=5.8,
                color=colors.get(dataset, "#555555"),
                label="MAHNOB-HCI" if dataset == "MAHNOB" else dataset,
        )
        ax.axhline(len(FAMILY_ORDER) - 0.5, color="#777777", linestyle=":", linewidth=1.1)
        if log_y:
            ax.set_xscale("log")
        if metric == "waveform_CCC":
            ax.set_xlim(-0.05, 1.02)
        ax.set_yticks(np.arange(len(rung_order)))
        ax.set_yticklabels([_allbase_rung_label(r).replace("\n", " ") for r in rung_order])
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_title(ylabel, loc="center")
        ax.margins(y=0.05)
        add_metric_box(ax, box, loc="lower right" if log_y else "upper right", fontsize=8.0)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.915),
            ncol=len(handles),
            frameon=False,
            handlelength=2.5,
            columnspacing=1.8,
        )
    fig.suptitle("All fixed observations, comparator, and PARH-OSSM on final full-dataset trials", fontsize=13, y=0.985)
    save_figure(fig, out_path)


def plot_fusion_ladder(ladder_df: pd.DataFrame, out_path: Path):
    """Plot a dense paper-facing ladder from fixed operators to PARH-OSSM.

    This replaces sparse mechanism-only plots in the main text. It uses the
    final same-trial comparison layer, so every point has the same provenance
    as the headline tables.
    """
    rung_order = [
        "direct_OF",
        "direct_DoF",
        "direct_P1D_quad",
        "direct_P1D_cons",
        "OSSM-KF (P1D quad)",
        "PARH-OSSM",
    ]
    datasets = [d for d in ["COHFACE", "MAHNOB"] if d in set(ladder_df["dataset"].astype(str))]
    if not datasets:
        raise ValueError("Fusion ladder has no known datasets.")
    ladder_df = ladder_df[ladder_df["rung"].isin(rung_order)].copy()
    x = np.arange(len(rung_order))
    colors = {"COHFACE": "#1f3a5f", "MAHNOB": "#b85c2b"}

    set_manuscript_style("paper")
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.2), constrained_layout=True)
    panels = [
        ("rate_MAE", "Rate MAE (bpm)", "lower is better", True),
        ("waveform_CCC", "Aligned waveform CCC", "higher is better", False),
        ("strict_NMAE_span", "Strict NMAE / GT span", "lower is better", True),
    ]
    for ax, (metric, ylabel, box, log_y) in zip(axes, panels):
        style_axis(ax, grid="y")
        for dataset in datasets:
            sub = ladder_df[ladder_df["dataset"].astype(str) == dataset].set_index("rung")
            y = pd.to_numeric(sub.reindex(rung_order)[metric], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(y)
            if not finite.any():
                continue
            ax.plot(
                x[finite],
                y[finite],
                marker="o",
                linewidth=2.2,
                markersize=6.5,
                color=colors.get(dataset, "#555555"),
                label="MAHNOB-HCI" if dataset == "MAHNOB" else dataset,
            )
        if log_y:
            ax.set_yscale("log")
        if metric == "waveform_CCC":
            ax.set_ylim(-0.05, 1.02)
        ax.set_xticks(x)
        ax.set_xticklabels([_rung_label(r) for r in rung_order], rotation=0, ha="center")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, loc="center")
        add_metric_box(ax, box, loc="upper right", fontsize=8.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=len(handles), frameon=False)
    fig.suptitle("Fixed observations, comparator, and PARH-OSSM on the same final full-dataset trials", fontsize=13)
    save_figure(fig, out_path)


def parse_args():
    root = ROOT
    parser = argparse.ArgumentParser(description="Generate F3/F5 observation-class summary figures.")
    parser.add_argument(
        "--t3-csv",
        type=Path,
        default=root / "paper" / "tables_ready" / "T3_rate_main.csv",
    )
    parser.add_argument(
        "--family-master-csv",
        type=Path,
        default=root / "paper" / "tables_ready" / "COHFACE_full_method_master_table.csv",
    )
    parser.add_argument(
        "--freq-raw-csv",
        type=Path,
        default=root / "results" / "20260409_cohface_prod_ofbridge_dofbridge_p1dcons_e2e_policy_narrow" / "cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons" / "metrics" / "metrics_freq_domain_raw.csv",
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
        "--ladder-csv",
        type=Path,
        default=root / "paper" / "tables_ready" / "T6b_fusion_ladder.csv",
    )
    parser.add_argument(
        "--observation-class-csv",
        type=Path,
        default=root / "paper" / "tables_ready" / "S_T_final_observation_class_comparison.csv",
    )
    parser.add_argument(
        "--allfamily-csv",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--out-f3",
        type=Path,
        default=root / "paper" / "figures" / "F3_rate_observation_class_summary.pdf",
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
    if args.freq_raw_csv.exists():
        t3_family_df = build_family_summary_from_freq_raw(args.freq_raw_csv, dataset="COHFACE")
    elif args.family_master_csv.exists():
        t3_family_df = build_family_summary_from_master(args.family_master_csv, dataset="COHFACE")
    else:
        t3_family_df = t3_df
    if "family" not in t3_family_df.columns and "observation_class" in t3_family_df.columns:
        t3_family_df = t3_family_df.rename(columns={"observation_class": "family"})
    t6_df = pd.read_csv(args.t6_csv)
    mech_df = pd.read_csv(args.mech_csv)
    plot_t3_family_summary(t3_family_df, args.out_f3, headline_df=t3_df)
    observation_class_csv = args.allfamily_csv or args.observation_class_csv
    if observation_class_csv.exists():
        plot_allbase_fusion_ladder(pd.read_csv(observation_class_csv), args.out_f5)
    elif args.ladder_csv.exists():
        plot_fusion_ladder(pd.read_csv(args.ladder_csv), args.out_f5)
    else:
        plot_mechanism_activation(t6_df, mech_df, args.out_f5)
    print(f"Saved {args.out_f3}")
    print(f"Saved {args.out_f5}")


if __name__ == "__main__":
    main()
