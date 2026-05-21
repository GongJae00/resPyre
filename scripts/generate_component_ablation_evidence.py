#!/usr/bin/env python3
"""Generate paper-facing component ablation evidence assets.

The goal is not to invent another model variant. This script consolidates
existing, persistent ablation/evaluation artifacts into a reviewer-facing
table and a compact figure that answer:

1. Which detachable component was introduced?
2. What reference removes or weakens that component?
3. What changed in rate, waveform, and strict morphology metrics?
4. What claim boundary follows from the evidence?
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "paper" / "tables_ready"
FIGURE_DIR = ROOT / "paper" / "figures"
ANALYSIS_DIR = ROOT / "analysis"


@dataclass(frozen=True)
class MetricRow:
    rate_mae: float | None = None
    rate_r: float | None = None
    waveform_ccc: float | None = None
    waveform_dtw: float | None = None
    strict_ccc: float | None = None
    strict_mae: float | None = None


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        fv = float(v)
    except Exception:
        return None
    return None if np.isnan(fv) else fv


def _t5_row(t5: pd.DataFrame, row_id: str, required: bool = True) -> MetricRow:
    rows = t5.loc[t5["row_id"].eq(row_id)]
    if rows.empty:
        if required:
            raise KeyError(f"Missing T5 operator-alignment row: {row_id}")
        return MetricRow()
    row = rows.iloc[0]
    return MetricRow(
        rate_mae=_f(row.get("rate_MAE")),
        rate_r=_f(row.get("rate_PearsonR")),
        waveform_ccc=_f(row.get("waveform_CCC")),
        waveform_dtw=_f(row.get("waveform_DTW")),
        strict_ccc=_f(row.get("strict_CCC")),
        strict_mae=_f(row.get("strict_MAE")),
    )


def _metric_row_from_family(
    t3: pd.DataFrame,
    t4: pd.DataFrame,
    *,
    dataset: str,
    family: str,
    variant: str,
    required: bool = True,
) -> MetricRow:
    prefix = {"Base": "Base", "OSSM-KF": "KFstd", "PARH": "PARH"}[variant]
    ds_names = [dataset]
    if dataset == "MAHNOB-HCI":
        ds_names.append("MAHNOB")
    if dataset == "MAHNOB":
        ds_names.append("MAHNOB-HCI")
    r3_df = t3.loc[t3["dataset"].isin(ds_names) & t3["family"].eq(family)]
    r4_df = t4.loc[t4["dataset"].isin(ds_names) & t4["family"].eq(family)]
    if r3_df.empty or r4_df.empty:
        if required:
            raise KeyError(f"Missing table-ready metric row for dataset={dataset}, family={family}, variant={variant}")
        return MetricRow()
    r3 = r3_df.iloc[0]
    r4 = r4_df.iloc[0]
    return MetricRow(
        rate_mae=_f(r3.get(f"{prefix}_MAE")),
        rate_r=_f(r3.get(f"{prefix}_PearsonR")),
        waveform_ccc=_f(r4.get(f"{prefix}_CCC")),
        waveform_dtw=_f(r4.get(f"{prefix}_DTW")),
    )


def _delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def _component_row(
    *,
    component: str,
    detachable_test: str,
    dataset_scope: str,
    why_included: str,
    reference_name: str,
    promoted_name: str,
    reference: MetricRow,
    promoted: MetricRow,
    interpretation: str,
    paper_use: str,
    caveat: str = "",
) -> dict[str, object]:
    return {
        "component": component,
        "detachable_test": detachable_test,
        "dataset_scope": dataset_scope,
        "why_included": why_included,
        "reference_without_or_weakened": reference_name,
        "promoted_or_stronger_variant": promoted_name,
        "reference_rate_MAE": _fmt(reference.rate_mae),
        "promoted_rate_MAE": _fmt(promoted.rate_mae),
        "delta_rate_MAE": _fmt(_delta(promoted.rate_mae, reference.rate_mae)),
        "reference_rate_R": _fmt(reference.rate_r),
        "promoted_rate_R": _fmt(promoted.rate_r),
        "delta_rate_R": _fmt(_delta(promoted.rate_r, reference.rate_r)),
        "reference_waveform_CCC": _fmt(reference.waveform_ccc),
        "promoted_waveform_CCC": _fmt(promoted.waveform_ccc),
        "delta_waveform_CCC": _fmt(_delta(promoted.waveform_ccc, reference.waveform_ccc)),
        "reference_waveform_DTW": _fmt(reference.waveform_dtw),
        "promoted_waveform_DTW": _fmt(promoted.waveform_dtw),
        "delta_waveform_DTW": _fmt(_delta(promoted.waveform_dtw, reference.waveform_dtw)),
        "reference_strict_CCC": _fmt(reference.strict_ccc),
        "promoted_strict_CCC": _fmt(promoted.strict_ccc),
        "delta_strict_CCC": _fmt(_delta(promoted.strict_ccc, reference.strict_ccc)),
        "interpretation": interpretation,
        "paper_use": paper_use,
        "caveat": caveat,
    }


def _final_metric_row(t3: pd.DataFrame, t4: pd.DataFrame, *, dataset: str) -> MetricRow:
    """Return the final PARH-OSSM row from the current paper package.

    The final full run exports the integrated model as a single method
    (`parh_ossm`) rather than as one row per observation family. That row is
    the correct paper-facing PARH summary; family-level construction rows are
    documented as design/claim-boundary evidence when the family ablation table
    is not present in the current package.
    """
    family_candidates = ["PARH-OSSM", "parh_ossm"]
    r3_df = t3.loc[t3["dataset"].eq(dataset) & t3["family"].isin(family_candidates)]
    r4_df = t4.loc[t4["dataset"].eq(dataset) & t4["family"].isin(family_candidates)]
    if r3_df.empty or r4_df.empty:
        return MetricRow()
    r3 = r3_df.iloc[0]
    r4 = r4_df.iloc[0]
    return MetricRow(
        rate_mae=_f(r3.get("PARH_MAE") if "PARH_MAE" in r3.index else r3.get("Base_MAE")),
        rate_r=_f(r3.get("PARH_PearsonR") if "PARH_PearsonR" in r3.index else r3.get("Base_PearsonR")),
        waveform_ccc=_f(r4.get("PARH_CCC") if "PARH_CCC" in r4.index else r4.get("Base_CCC")),
        waveform_dtw=_f(r4.get("PARH_DTW") if "PARH_DTW" in r4.index else r4.get("Base_DTW")),
    )


def build_component_table(table_dir: Path) -> pd.DataFrame:
    t3 = pd.read_csv(table_dir / "T3_rate_main.csv")
    t4 = pd.read_csv(table_dir / "T4_waveform_main.csv")
    t5 = pd.read_csv(table_dir / "T5_operator_alignment_ablation.csv")

    rows: list[dict[str, object]] = []

    # Observation construction is detachable at the family level.  Older
    # family-ablation packages contain direct OF/DoF/P1D family rows; the final
    # paper package exports only the integrated PARH-OSSM row.  In that case,
    # keep these rows as explicit construction-boundary evidence rather than
    # crashing or silently reusing stale family metrics.
    rows.append(
        _component_row(
            component="OF displacement bridge",
            detachable_test="OF raw vs OF_bridge under the same PARH readout",
            dataset_scope="COHFACE",
            why_included="Converts velocity-like optical flow into a displacement-compatible observation.",
            reference_name="OF + PARH",
            promoted_name="OF_bridge + PARH",
            reference=_metric_row_from_family(t3, t4, dataset="COHFACE", family="OF", variant="PARH", required=False),
            promoted=_metric_row_from_family(t3, t4, dataset="COHFACE", family="OF_bridge", variant="PARH", required=False),
            interpretation="Construction-level evidence: the final package uses OF/OF_bridge inside the joint observation law, while direct family rows are not regenerated in the integrated final run.",
            paper_use="Observation-family construction evidence.",
            caveat="Use T2 and observability audits for final-package interpretation; do not overclaim standalone OF_bridge dominance.",
        )
    )
    rows.append(
        _component_row(
            component="DoF signed-motion bridge",
            detachable_test="DoF raw vs DoF_bridge under the same PARH readout",
            dataset_scope="COHFACE; MAHNOB-HCI",
            why_included="Turns nonnegative burst energy into a signed coarse motion trajectory.",
            reference_name="DoF + PARH",
            promoted_name="DoF_bridge + PARH",
            reference=_metric_row_from_family(t3, t4, dataset="MAHNOB", family="DoF", variant="PARH", required=False),
            promoted=_metric_row_from_family(t3, t4, dataset="MAHNOB", family="DoF_bridge", variant="PARH", required=False),
            interpretation="Construction-level evidence: DoF_bridge is retained as a hard-regime signed-motion probe, not as the main morphology path.",
            paper_use="Hard-regime observation-construction evidence.",
        )
    )
    rows.append(
        _component_row(
            component="P1D consensus",
            detachable_test="best single P1D harmonic vs P1D_cons under the same PARH readout",
            dataset_scope="COHFACE",
            why_included="Aggregates linear/quadratic/cubic profile surrogates to test cross-interpolation stability.",
            reference_name="P1D_quad + PARH",
            promoted_name="P1D_cons + PARH",
            reference=_metric_row_from_family(t3, t4, dataset="COHFACE", family="P1D_quad", variant="PARH", required=False),
            promoted=_metric_row_from_family(t3, t4, dataset="COHFACE", family="P1D_cons", variant="PARH", required=False),
            interpretation="Construction-level evidence: P1D_cons is retained as a stable consensus observation inside the joint law, not as a claim that consensus alone dominates.",
            paper_use="Constructed-family value and boundary.",
        )
    )

    # Filter/state/readout components use the existing intent-aligned ablation table.
    rows.append(
        _component_row(
            component="standard resonator Kalman comparator",
            detachable_test="best single Base vs best single OSSM-KF",
            dataset_scope="MAHNOB-HCI",
            why_included="Tests what is gained by simply attaching an oscillator and standard Kalman filtering.",
            reference_name="best single Base",
            promoted_name="best single OSSM-KF",
            reference=_metric_row_from_family(t3, t4, dataset="MAHNOB", family="DoF_bridge", variant="Base", required=False),
            promoted=_metric_row_from_family(t3, t4, dataset="MAHNOB", family="DoF_bridge", variant="OSSM-KF", required=False),
            interpretation="Comparator-boundary evidence: OSSM-KF is a reference timing channel; the integrated final package does not treat it as the proposed adaptive observation law.",
            paper_use="Comparator baseline and no-black-box-control boundary.",
        )
    )
    rows.append(
        _component_row(
            component="integrated final PARH-OSSM package",
            detachable_test="COHFACE final package vs MAHNOB hard-regime final package",
            dataset_scope="COHFACE; MAHNOB",
            why_included="Separates strong within-dataset behavior from hard-regime target limitations without hiding failures.",
            reference_name="COHFACE final PARH-OSSM",
            promoted_name="MAHNOB final PARH-OSSM",
            reference=_final_metric_row(t3, t4, dataset="COHFACE"),
            promoted=_final_metric_row(t3, t4, dataset="MAHNOB"),
            interpretation="The final model is strong on COHFACE but still limited on MAHNOB; this row supports the paper's observability/target-shift claim boundary.",
            paper_use="Final-package performance boundary.",
            caveat="This is a dataset-shift comparison, not an ablation improvement row.",
        )
    )
    rows.append(
        _component_row(
            component="target-local adaptive observation law",
            detachable_test="shared observation law vs local adaptive observation law",
            dataset_scope="COHFACE ablation",
            why_included="Avoids assuming that one fixed observation operator is reliable for every person/environment.",
            reference_name="shared_observation_law",
            promoted_name="adaptive_observation_law",
            reference=_t5_row(t5, "shared_observation_law", required=False),
            promoted=_t5_row(t5, "adaptive_observation_law", required=False),
            interpretation="Rate and strict morphology improve when the observation law becomes local; waveform CCC tradeoff motivates decoupled readouts.",
            paper_use="Core adaptive-law ablation.",
            caveat="If the paired shared-law row is unavailable, use this as a design-boundary row rather than a numeric delta.",
        )
    )
    rows.append(
        _component_row(
            component="decoupled timing and morphology readouts",
            detachable_test="rate expert vs decoupled z_osc/z_full system",
            dataset_scope="COHFACE ablation",
            why_included="Respiratory timing and waveform morphology are related but not the same optimization target.",
            reference_name="PARH rate expert",
            promoted_name="decoupled system",
            reference=_t5_row(t5, "parh_rate_expert"),
            promoted=_t5_row(t5, "decoupled_system"),
            interpretation="The decoupled readout improves rate MAE, waveform CCC, and strict CCC together; this is the strongest evidence for the final architecture.",
            paper_use="Main architecture ablation.",
        )
    )
    rows.append(
        _component_row(
            component="waveform-specialized expert boundary",
            detachable_test="waveform-only expert vs decoupled system",
            dataset_scope="COHFACE ablation",
            why_included="Tests whether maximizing morphology alone can replace timing-aware estimation.",
            reference_name="waveform expert only",
            promoted_name="decoupled system",
            reference=_t5_row(t5, "waveform_expert_only"),
            promoted=_t5_row(t5, "decoupled_system"),
            interpretation="Waveform-only is competitive for morphology but much worse for rate; this defends not collapsing the paper to a waveform fitter.",
            paper_use="Readout specialization boundary.",
        )
    )
    rows.append(
        _component_row(
            component="robust fallback policy",
            detachable_test="decoupled system vs consistency-first robust fallback",
            dataset_scope="COHFACE ablation",
            why_included="Tests whether a conservative fallback should be promoted as the default output.",
            reference_name="decoupled system",
            promoted_name="robust fallback",
            reference=_t5_row(t5, "decoupled_system", required=False),
            promoted=_t5_row(t5, "robust_decoupled_consistency_first", required=False),
            interpretation="No live final-package robust row is required for the promoted model; keep fallback as diagnostic/hard-regime policy, not the default claim.",
            paper_use="Prevents overclaiming fallback variants.",
        )
    )
    return pd.DataFrame(rows)


def write_markdown(table: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Component Ablation Evidence",
        "",
        "This file is generated from existing paper-ready metrics. Negative `delta_rate_MAE` and negative `delta_waveform_DTW` are improvements; positive `delta_waveform_CCC`, `delta_rate_R`, and `delta_strict_CCC` are improvements.",
        "",
        "| component | detachable test | delta rate MAE | delta waveform CCC | delta strict CCC | interpretation |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['component']} | {row['detachable_test']} | {row['delta_rate_MAE']} | {row['delta_waveform_CCC']} | {row['delta_strict_CCC']} | {row['interpretation']} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_component_figure(t5: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_ids = [
        "base_raw_p1dquad",
        "parh_rate_expert",
        "shared_observation_law",
        "adaptive_observation_law",
        "staged_routed_multihead",
        "waveform_expert_only",
        "decoupled_system",
        "robust_decoupled_consistency_first",
    ]
    labels = [
        "raw\nP1D",
        "PARH\nrate",
        "shared\nlaw",
        "adaptive\nlaw",
        "routed\nmultihead",
        "waveform\nexpert",
        "decoupled\nsystem",
        "robust\nfallback",
    ]
    available_row_ids = [row_id for row_id in row_ids if row_id in set(t5["row_id"].astype(str))]
    if not available_row_ids:
        raise ValueError("No recognized T5 rows available for component evidence figure")
    label_map = dict(zip(row_ids, labels))
    df = t5.set_index("row_id").loc[available_row_ids].reset_index()
    labels = [label_map[row_id] for row_id in available_row_ids]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.8), sharey=True)
    color = "#143b6e"
    accent = "#0d7f72"
    warn = "#c96b17"
    y = np.arange(len(df))

    highlight = df["row_id"].eq("decoupled_system")
    waveform_only = df["row_id"].eq("waveform_expert_only")
    point_colors = np.where(highlight, accent, np.where(waveform_only, warn, color))

    axes[0].scatter(df["rate_MAE"], y, s=54, color=point_colors, zorder=3)
    axes[0].plot(df["rate_MAE"], y, lw=1.2, color=color, alpha=0.45, zorder=2)
    axes[0].set_title("Timing evidence")
    axes[0].set_xlabel("Rate MAE (bpm)")
    axes[0].text(0.98, 0.04, "lower is better", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8)

    axes[1].scatter(df["waveform_CCC"], y, s=54, color=point_colors, zorder=3)
    axes[1].plot(df["waveform_CCC"], y, lw=1.2, color=color, alpha=0.45, zorder=2)
    axes[1].set_title("Morphology evidence")
    axes[1].set_xlabel("Waveform CCC")
    axes[1].text(0.98, 0.04, "higher is better", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8)

    axes[2].scatter(df["strict_CCC"], y, s=54, color=point_colors, zorder=3)
    axes[2].plot(df["strict_CCC"], y, lw=1.2, color=color, alpha=0.45, zorder=2)
    axes[2].set_title("Zero-lag strict evidence")
    axes[2].set_xlabel("Strict CCC")
    axes[2].text(0.98, 0.04, "higher is better", transform=axes[2].transAxes, ha="right", va="bottom", fontsize=8)

    for ax in axes:
        ax.grid(True, axis="x", alpha=0.25, lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Ablation row")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)
    fig.suptitle("Component ablation evidence: why the final readout is decoupled", y=1.01, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate component ablation table and figure.")
    p.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    p.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    p.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    component_table = build_component_table(table_dir)
    out_csv = table_dir / "T5_component_ablation_evidence.csv"
    component_table.to_csv(out_csv, index=False)
    component_table.to_csv(analysis_dir / "component_ablation_evidence.csv", index=False)
    write_markdown(component_table, analysis_dir / "component_ablation_evidence.md")
    plot_component_figure(pd.read_csv(table_dir / "T5_operator_alignment_ablation.csv"), figure_dir / "S_F_component_ablation_evidence.pdf")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {analysis_dir / 'component_ablation_evidence.md'}")
    print(f"Wrote: {figure_dir / 'S_F_component_ablation_evidence.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
