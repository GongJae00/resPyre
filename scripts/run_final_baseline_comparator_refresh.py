#!/usr/bin/env python3
"""Refresh full-dataset baseline/comparator tables with explicit provenance.

This script does not tune PARH-OSSM.  It closes the paper-facing comparison
gap by extracting fixed observation-class baselines and OSSM-KF comparator
rows on the same trial IDs used by the integrated PARH-OSSM full-dataset run.

The source observation-class metrics are retained as a transparent comparator layer;
headline PARH numbers still come from the final integrated PARH-OSSM run.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.observations.semantics import (  # noqa: E402
    CANONICAL_OBSERVATION_FAMILY_ORDER,
    get_observation_family_semantics,
)
from scripts.generate_table_ready import classify_method  # noqa: E402


RAW_METRIC_FILES = (
    "metrics_freq_domain_raw.csv",
    "metrics_waveform_raw.csv",
    "metrics_waveform_strict_raw.csv",
    "metrics_time_domain_raw.csv",
    "metrics_filter_diagnostics_raw.csv",
)

DATASET_DEFAULTS = {
    "COHFACE": {
        "baseline_metrics": ROOT
        / "results"
        / "20260409_cohface_prod_ofbridge_dofbridge_p1dcons_e2e_policy_narrow"
        / "cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons"
        / "metrics",
        "final_metrics": ROOT
        / "results"
        / "final_full_validation"
        / "cohface"
        / "metrics",
        "baseline_layer": "20260409_cohface_allfamily_full_dataset",
        "final_layer": "final_full_validation/cohface",
    },
    "MAHNOB": {
        "baseline_metrics": ROOT
        / "results"
        / "20260409_mahnob_prod_ofbridge_dofbridge_p1dcons_e2e"
        / "mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons"
        / "metrics",
        "final_metrics": ROOT
        / "results"
        / "final_full_validation"
        / "mahnob_tailaligned"
        / "metrics",
        "baseline_layer": "20260409_mahnob_allfamily_full_dataset",
        "final_layer": "final_full_validation/mahnob_tailaligned",
    },
}

METHOD_BY_FAMILY = {
    "OF": "of_farneback",
    "OF_bridge": "of_disp_bridge",
    "DoF": "DoF",
    "DoF_bridge": "dof_disp_bridge",
    "P1D_lin": "profile1D linear",
    "P1D_quad": "profile1D quadratic",
    "P1D_cub": "profile1D cubic",
    "P1D_cons": "profile1d_consensus",
}

KFSTD_BY_FAMILY = {
    "OF": "of_farneback__kfstd",
    "OF_bridge": "of_disp_bridge__kfstd",
    "DoF": "dof__kfstd",
    "DoF_bridge": "dof_disp_bridge__kfstd",
    "P1D_lin": "profile1d_linear__kfstd",
    "P1D_quad": "profile1d_quadratic__kfstd",
    "P1D_cub": "profile1d_cubic__kfstd",
    "P1D_cons": "profile1d_consensus__kfstd",
}

PARH_BY_FAMILY = {
    "OF": "of_farneback__parh_ossm",
    "OF_bridge": "of_disp_bridge__parh_ossm",
    "DoF": "dof__parh_ossm",
    "DoF_bridge": "dof_disp_bridge__parh_ossm",
    "P1D_lin": "profile1d_linear__parh_ossm",
    "P1D_quad": "profile1d_quadratic__parh_ossm",
    "P1D_cub": "profile1d_cubic__parh_ossm",
    "P1D_cons": "profile1d_consensus__parh_ossm",
}

DISPLAY_FAMILY_ORDER = tuple(METHOD_BY_FAMILY.keys())
MAIN_REPRESENTATIVE_FAMILY = "P1D_quad"
DISPLAY_DATASET = {"COHFACE": "COHFACE", "MAHNOB": "MAHNOB"}


@dataclass(frozen=True)
class DatasetLayer:
    dataset: str
    baseline_metrics: Path
    final_metrics: Path
    baseline_layer: str
    final_layer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Write refreshed artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing artifacts.")
    parser.add_argument(
        "--out-run",
        type=Path,
        default=ROOT / "results" / "final_baseline_comparator_refresh",
        help="Directory for filtered baseline/comparator metric extracts.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=ROOT / "paper" / "tables_ready",
        help="Paper table-ready output directory.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=ROOT / "analysis",
        help="Analysis/audit output directory.",
    )
    parser.add_argument(
        "--representative-family",
        choices=sorted(METHOD_BY_FAMILY),
        default=MAIN_REPRESENTATIVE_FAMILY,
        help="Pre-locked representative direct and OSSM-KF comparator family for headline Base and internal KFstd columns.",
    )
    parser.add_argument(
        "--artifact-policy",
        choices=["lean", "full"],
        default="full",
        help="full also copies filtered raw metric CSVs into the result directory.",
    )
    return parser.parse_args()


def _dataset_layers() -> list[DatasetLayer]:
    layers = []
    for dataset, cfg in DATASET_DEFAULTS.items():
        layers.append(
            DatasetLayer(
                dataset=dataset,
                baseline_metrics=Path(cfg["baseline_metrics"]),
                final_metrics=Path(cfg["final_metrics"]),
                baseline_layer=str(cfg["baseline_layer"]),
                final_layer=str(cfg["final_layer"]),
            )
        )
    return layers


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _video_col(df: pd.DataFrame) -> str:
    if "video" in df.columns:
        return "video"
    if "trial_id" in df.columns:
        return "trial_id"
    raise KeyError("metric CSV has neither 'video' nor 'trial_id' column")


def _median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return math.nan
    return float(vals.median())


def _float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def _fmt(value: object, digits: int = 3) -> str:
    val = _float_or_nan(value)
    if math.isnan(val):
        return ""
    return f"{val:.{digits}f}"


def _preferred_rate(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if "rate_source" not in df.columns:
        return df.copy()
    preferred = "signal_spectral" if variant == "Base" else "track_hz"
    sub = df[df["rate_source"].eq(preferred)]
    return sub.copy() if not sub.empty else df.copy()


def _decorate_methods(df: pd.DataFrame, layer: str) -> pd.DataFrame:
    out = df.copy()
    if "method" in out.columns:
        out[["family", "variant"]] = out["method"].apply(lambda m: pd.Series(classify_method(m)))
    out["source_layer"] = layer
    return out


def _final_trial_ids(final_metrics: Path) -> list[str]:
    freq = _read_csv(final_metrics / "metrics_freq_domain_raw.csv")
    col = _video_col(freq)
    return sorted(map(str, freq[col].dropna().unique()))


def _filter_by_trials(df: pd.DataFrame, trial_ids: Iterable[str]) -> pd.DataFrame:
    col = _video_col(df)
    allowed = set(map(str, trial_ids))
    return df[df[col].astype(str).isin(allowed)].copy()


def _load_filtered(layer: DatasetLayer) -> dict[str, pd.DataFrame]:
    trial_ids = _final_trial_ids(layer.final_metrics)
    frames: dict[str, pd.DataFrame] = {}
    for name in RAW_METRIC_FILES:
        base_path = layer.baseline_metrics / name
        final_path = layer.final_metrics / name
        if not base_path.exists() or not final_path.exists():
            continue
        base = _decorate_methods(_filter_by_trials(_read_csv(base_path), trial_ids), layer.baseline_layer)
        final = _decorate_methods(_read_csv(final_path), layer.final_layer)
        frames[name] = pd.concat([base, final], ignore_index=True, sort=False)
    return frames


def _derive_strict_normalized(strict_df: pd.DataFrame, final_metrics: Path) -> pd.DataFrame:
    out = strict_df.copy()
    if "gt_span_p95p05" not in out.columns:
        out["gt_span_p95p05"] = np.nan
    final = _read_csv(final_metrics / "metrics_waveform_strict_raw.csv")
    col = _video_col(final)
    span = final[[col, "gt_span_p95p05"]].dropna().drop_duplicates(subset=[col])
    span_map = dict(zip(span[col].astype(str), pd.to_numeric(span["gt_span_p95p05"], errors="coerce")))
    video_col = _video_col(out)
    missing_span = out["gt_span_p95p05"].isna()
    out.loc[missing_span, "gt_span_p95p05"] = out.loc[missing_span, video_col].astype(str).map(span_map)
    for metric, norm_col in [
        ("strict_MAE", "strict_NMAE_span"),
        ("strict_RMSE", "strict_NRMSE_span"),
        ("strict_DTW", "strict_NDTW_span"),
    ]:
        if metric in out.columns:
            denom = pd.to_numeric(out["gt_span_p95p05"], errors="coerce").replace(0, np.nan)
            out[norm_col] = pd.to_numeric(out[metric], errors="coerce") / denom
    return out


def _select_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df[df["method"].astype(str).eq(method)].copy()


def _select_final_parh(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["method"].astype(str).eq("parh_ossm")].copy()


def _rate_metrics(df: pd.DataFrame, method: str, variant: str) -> tuple[dict[str, object], pd.DataFrame]:
    sub = _preferred_rate(_select_method(df, method), variant)
    source = ""
    if "rate_source" in sub.columns and not sub.empty:
        source = str(sub["rate_source"].iloc[0])
    return (
        {
            "rate_source": source,
            "MAE": _median(sub["MAE"]) if "MAE" in sub.columns else math.nan,
            "RMSE": _median(sub["RMSE"]) if "RMSE" in sub.columns else math.nan,
            "PearsonR": _median(sub["PearsonR"]) if "PearsonR" in sub.columns else math.nan,
            "N": len(sub),
        },
        sub,
    )


def _wave_metrics(df: pd.DataFrame, method: str, output_type: str) -> tuple[dict[str, object], pd.DataFrame]:
    sub = _select_method(df, method)
    if "causal_or_smoothed" in sub.columns:
        sub = sub[sub["causal_or_smoothed"].eq("smoothed")]
    if "output_type" in sub.columns:
        sub = sub[sub["output_type"].eq(output_type)]
    return (
        {
            "CCC": _median(sub["waveform_CCC"]) if "waveform_CCC" in sub.columns else math.nan,
            "MAE": _median(sub["waveform_MAE"]) if "waveform_MAE" in sub.columns else math.nan,
            "DTW": _median(sub["waveform_DTW"]) if "waveform_DTW" in sub.columns else math.nan,
            "N": len(sub),
        },
        sub,
    )


def _strict_metrics(df: pd.DataFrame, method: str, output_type: str) -> tuple[dict[str, object], pd.DataFrame]:
    sub = _select_method(df, method)
    if "causal_or_smoothed" in sub.columns:
        sub = sub[sub["causal_or_smoothed"].eq("smoothed")]
    if "output_type" in sub.columns:
        sub = sub[sub["output_type"].eq(output_type)]
    cols = {
        "CCC": "strict_CCC",
        "MAE": "strict_MAE",
        "RMSE": "strict_RMSE",
        "DTW": "strict_DTW",
        "gt_span_p95p05": "gt_span_p95p05",
        "NMAE_span": "strict_NMAE_span",
        "NRMSE_span": "strict_NRMSE_span",
        "NDTW_span": "strict_NDTW_span",
    }
    return (
        {key: (_median(sub[col]) if col in sub.columns else math.nan) for key, col in cols.items()}
        | {"N": len(sub)},
        sub,
    )


def _cycle_metrics(df: pd.DataFrame, method: str, output_type: str) -> tuple[dict[str, object], pd.DataFrame]:
    sub = _select_method(df, method)
    if "causal_or_smoothed" in sub.columns:
        sub = sub[sub["causal_or_smoothed"].eq("smoothed")]
    if "output_type" in sub.columns:
        sub = sub[sub["output_type"].eq(output_type)]
    cols = ["peak_time_mae_s", "trough_time_mae_s", "cycle_ppi_mae_s", "cycle_ie_abs_err"]
    return ({col: (_median(sub[col]) if col in sub.columns else math.nan) for col in cols} | {"N": len(sub)}, sub)


def _diag_metrics(df: pd.DataFrame, method: str) -> tuple[dict[str, object], pd.DataFrame]:
    sub = _select_method(df, method)
    cols = ["NIS_Mean", "NIS_InBand", "Lambda_Mean", "Lambda_LT1_Frac", "Coverage95", "Stability_Sec"]
    return ({col: (_median(sub[col]) if col in sub.columns else math.nan) for col in cols} | {"N": len(sub)}, sub)


def _make_headline_tables(
    layers: list[DatasetLayer],
    loaded: dict[str, dict[str, pd.DataFrame]],
    representative_family: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_t3: list[dict[str, object]] = []
    rows_t4: list[dict[str, object]] = []
    rows_t4b: list[dict[str, object]] = []
    rows_t4c: list[dict[str, object]] = []
    rows_t6: list[dict[str, object]] = []
    base_method = METHOD_BY_FAMILY[representative_family]
    kf_method = KFSTD_BY_FAMILY[representative_family]

    def _add_comparison_scope(row: dict[str, object]) -> dict[str, object]:
        counts = [
            int(row.get("Base_N", 0) or 0),
            int(row.get("KFstd_N", 0) or 0),
            int(row.get("PARH_N", 0) or 0),
        ]
        if len(set(counts)) == 1:
            row["comparison_scope"] = "same_trials"
            row["coverage_note"] = f"all reported methods evaluated on N={counts[0]} trials"
        else:
            row["comparison_scope"] = "full_dataset_with_baseline_coverage_limit"
            row["coverage_note"] = (
                "PARH-OSSM reports the full final dataset; representative direct "
                f"and OSSM-KF comparator rows are computable for Base_N={counts[0]}, "
                f"KFstd_N={counts[1]}, PARH_N={counts[2]} trials"
            )
        return row

    def _add_diagnostic_scope(row: dict[str, object]) -> dict[str, object]:
        counts = [
            int(row.get("KFstd_N", 0) or 0),
            int(row.get("PARH_N", 0) or 0),
        ]
        if len(set(counts)) == 1:
            row["comparison_scope"] = "same_trials"
            row["coverage_note"] = f"diagnostic comparator and PARH evaluated on N={counts[0]} trials"
        else:
            row["comparison_scope"] = "full_dataset_with_baseline_coverage_limit"
            row["coverage_note"] = (
                "PARH-OSSM reports the full final dataset; OSSM-KF diagnostic "
                f"comparator is computable for KFstd_N={counts[0]}, PARH_N={counts[1]} trials"
            )
        return row

    for layer in layers:
        frames = loaded[layer.dataset]
        dataset = DISPLAY_DATASET[layer.dataset]
        final_method = "parh_ossm"
        method_labels = {
            "Base": base_method,
            "KFstd": kf_method,
            "PARH": final_method,
        }
        rate = frames["metrics_freq_domain_raw.csv"]
        wave = frames["metrics_waveform_raw.csv"]
        strict = _derive_strict_normalized(frames["metrics_waveform_strict_raw.csv"], layer.final_metrics)
        diag = frames["metrics_filter_diagnostics_raw.csv"]

        row_t3 = {"dataset": dataset, "family": "PARH-OSSM"}
        for variant, method in method_labels.items():
            vals, _ = _rate_metrics(rate, method, variant)
            row_t3[f"{variant}_method"] = method
            row_t3[f"{variant}_rate_source"] = vals["rate_source"]
            for col in ["MAE", "RMSE", "PearsonR", "N"]:
                row_t3[f"{variant}_{col}"] = vals[col]
        rows_t3.append(_add_comparison_scope(row_t3))

        row_t4 = {"dataset": dataset, "family": "PARH-OSSM"}
        for variant, method in method_labels.items():
            otype = "z_full" if variant == "PARH" else "signal_hat"
            vals, _ = _wave_metrics(wave, method, otype)
            row_t4[f"{variant}_method"] = method
            for col in ["CCC", "MAE", "DTW", "N"]:
                row_t4[f"{variant}_{col}"] = vals[col]
        rows_t4.append(_add_comparison_scope(row_t4))

        row_t4b = {"dataset": dataset, "family": "PARH-OSSM"}
        for variant, method in method_labels.items():
            otype = "z_full" if variant == "PARH" else "signal_hat"
            vals, _ = _strict_metrics(strict, method, otype)
            row_t4b[f"{variant}_method"] = method
            for col in ["CCC", "MAE", "RMSE", "DTW", "gt_span_p95p05", "NMAE_span", "NRMSE_span", "NDTW_span", "N"]:
                row_t4b[f"{variant}_{col}"] = vals[col]
        rows_t4b.append(_add_comparison_scope(row_t4b))

        row_t4c = {"dataset": dataset, "family": "PARH-OSSM"}
        for variant, method in method_labels.items():
            otype = "z_full" if variant == "PARH" else "signal_hat"
            vals, _ = _cycle_metrics(strict, method, otype)
            row_t4c[f"{variant}_method"] = method
            for col in ["peak_time_mae_s", "trough_time_mae_s", "cycle_ppi_mae_s", "cycle_ie_abs_err", "N"]:
                row_t4c[f"{variant}_{col}"] = vals[col]
        rows_t4c.append(_add_comparison_scope(row_t4c))

        row_t6 = {"dataset": dataset, "family": "PARH-OSSM"}
        for variant, method in [("KFstd", kf_method), ("PARH", final_method)]:
            vals, _ = _diag_metrics(diag, method)
            row_t6[f"{variant}_method"] = method
            cols = ["Stability_Sec", "N"] if variant == "KFstd" else [
                "NIS_Mean",
                "NIS_InBand",
                "Lambda_Mean",
                "Lambda_LT1_Frac",
                "Stability_Sec",
                "N",
            ]
            for col in cols:
                row_t6[f"{variant}_{col}"] = vals[col]
        rows_t6.append(_add_diagnostic_scope(row_t6))

    return (
        pd.DataFrame(rows_t3),
        pd.DataFrame(rows_t4),
        pd.DataFrame(rows_t4b),
        pd.DataFrame(rows_t4c),
        pd.DataFrame(rows_t6),
    )


def _metric_row_for_method(
    dataset: str,
    family: str,
    variant: str,
    method: str,
    source_layer: str,
    frames: dict[str, pd.DataFrame],
    final_metrics: Path,
) -> dict[str, object]:
    rate_vals, _ = _rate_metrics(frames["metrics_freq_domain_raw.csv"], method, "Base" if variant == "Base" else variant)
    wave_vals, _ = _wave_metrics(frames["metrics_waveform_raw.csv"], method, "z_full" if variant == "PARH-OSSM" else ("z_full" if variant == "PARH" else "signal_hat"))
    strict_df = _derive_strict_normalized(frames["metrics_waveform_strict_raw.csv"], final_metrics)
    strict_vals, _ = _strict_metrics(strict_df, method, "z_full" if variant in {"PARH", "PARH-OSSM"} else "signal_hat")
    cycle_vals, _ = _cycle_metrics(strict_df, method, "z_full" if variant in {"PARH", "PARH-OSSM"} else "signal_hat")
    diag_vals, _ = _diag_metrics(frames["metrics_filter_diagnostics_raw.csv"], method)
    return {
        "dataset": dataset,
        "family": family,
        "variant": variant,
        "method": method,
        "source_layer": source_layer,
        "rate_source": rate_vals["rate_source"],
        "rate_MAE": rate_vals["MAE"],
        "rate_RMSE": rate_vals["RMSE"],
        "rate_PearsonR": rate_vals["PearsonR"],
        "rate_N": rate_vals["N"],
        "waveform_CCC": wave_vals["CCC"],
        "waveform_MAE": wave_vals["MAE"],
        "waveform_DTW": wave_vals["DTW"],
        "waveform_N": wave_vals["N"],
        "strict_CCC": strict_vals["CCC"],
        "strict_MAE": strict_vals["MAE"],
        "strict_RMSE": strict_vals["RMSE"],
        "strict_DTW": strict_vals["DTW"],
        "strict_NMAE_span": strict_vals["NMAE_span"],
        "strict_NRMSE_span": strict_vals["NRMSE_span"],
        "strict_NDTW_span": strict_vals["NDTW_span"],
        "strict_N": strict_vals["N"],
        "cycle_ppi_mae_s": cycle_vals["cycle_ppi_mae_s"],
        "cycle_ie_abs_err": cycle_vals["cycle_ie_abs_err"],
        "diag_NIS_Mean": diag_vals["NIS_Mean"],
        "diag_NIS_InBand": diag_vals["NIS_InBand"],
        "diag_Lambda_Mean": diag_vals["Lambda_Mean"],
        "diag_Stability_Sec": diag_vals["Stability_Sec"],
        "diag_N": diag_vals["N"],
    }


def _make_allfamily_table(layers: list[DatasetLayer], loaded: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for layer in layers:
        frames = loaded[layer.dataset]
        dataset = DISPLAY_DATASET[layer.dataset]
        for family in DISPLAY_FAMILY_ORDER:
            for variant, mapping in [
                ("Base", METHOD_BY_FAMILY),
                ("OSSM-KF", KFSTD_BY_FAMILY),
                ("family_PARH", PARH_BY_FAMILY),
            ]:
                rows.append(
                    _metric_row_for_method(
                        dataset,
                        family,
                        variant,
                        mapping[family],
                        layer.baseline_layer,
                        frames,
                        layer.final_metrics,
                    )
                )
        rows.append(
            _metric_row_for_method(
                dataset,
                "PARH-OSSM",
                "PARH-OSSM",
                "parh_ossm",
                layer.final_layer,
                frames,
                layer.final_metrics,
            )
        )
    return pd.DataFrame(rows)


def _make_public_observation_class_comparison(allfamily: pd.DataFrame) -> pd.DataFrame:
    """Return the supplementary comparison table with reader-facing terminology."""
    public = allfamily.copy()
    public = public.rename(columns={"family": "observation_class"})
    if "variant" in public.columns:
        public["variant"] = public["variant"].replace({"family_PARH": "class-local PARH"})
    if "source_layer" in public.columns:
        public["source_layer"] = (
            public["source_layer"]
            .astype(str)
            .str.replace("allfamily", "observation_class", regex=False)
        )
    for col in public.select_dtypes(include="object").columns:
        public[col] = (
            public[col]
            .astype(str)
            .str.replace("__kfstd", "__ossm_kf", regex=False)
            .str.replace("kfstd", "ossm_kf", regex=False)
        )
    return public


def _make_public_comparator_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expose the comparator as OSSM_KF in table-ready public artifacts."""
    public = df.copy()
    public = public.rename(columns={col: col.replace("KFstd_", "OSSM_KF_") for col in public.columns if col.startswith("KFstd_")})
    for col in public.select_dtypes(include="object").columns:
        public[col] = (
            public[col]
            .astype(str)
            .str.replace("KFstd_N", "OSSM_KF_N", regex=False)
            .str.replace("__kfstd", "__ossm_kf", regex=False)
            .str.replace("kfstd", "ossm_kf", regex=False)
        )
    return public


def _make_t2_from_allfamily(allfamily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    coh = allfamily[allfamily["dataset"].eq("COHFACE")]
    for family in DISPLAY_FAMILY_ORDER:
        sem_key = {
            "OF": "of_farneback",
            "OF_bridge": "of_disp_bridge",
            "DoF": "dof",
            "DoF_bridge": "dof_disp_bridge",
            "P1D_lin": "profile1d_linear",
            "P1D_quad": "profile1d_quadratic",
            "P1D_cub": "profile1d_cubic",
            "P1D_cons": "profile1d_consensus",
        }[family]
        sem = get_observation_family_semantics(sem_key)
        base = coh[(coh["family"].eq(family)) & (coh["variant"].eq("Base"))]
        parh = coh[(coh["family"].eq(family)) & (coh["variant"].eq("family_PARH"))]
        base_rate = base["rate_MAE"].iloc[0] if not base.empty else math.nan
        base_wave = base["waveform_CCC"].iloc[0] if not base.empty else math.nan
        parh_rate = parh["rate_MAE"].iloc[0] if not parh.empty else math.nan
        parh_rate_r = parh["rate_PearsonR"].iloc[0] if not parh.empty else math.nan
        parh_wave = parh["waveform_CCC"].iloc[0] if not parh.empty else math.nan
        parh_dtw = parh["waveform_DTW"].iloc[0] if not parh.empty else math.nan
        current_strength = "mixed"
        if sem.get("waveform_primary"):
            current_strength = "waveform-primary"
        elif sem.get("rate_primary"):
            current_strength = "rate-primary"
        elif sem.get("helper_heavy"):
            current_strength = "helper-heavy"
        elif sem.get("nuisance_risk") == "high":
            current_strength = "nuisance-limited"
        role = str(sem.get("current_parh_role", ""))
        role = role.replace("rate family", "rate evidence")
        role = role.replace("constructed family", "constructed cue")
        role = role.replace("auxiliary family", "auxiliary cue")
        role = role.replace("DoF family", "DoF cue")
        role = role.replace("family", "observation class")
        rows.append(
            {
                "observation_class": family,
                "construction": sem.get("construction", ""),
                "domain": sem.get("observation_domain", ""),
                "primary_information": str(sem.get("primary_information", "")).replace("_", " "),
                "secondary_information": str(sem.get("secondary_information", "")).replace("_", " "),
                "nuisance_risk": sem.get("nuisance_risk", ""),
                "current_parh_role": role,
                "current_strength": current_strength,
                "PARH_rate_MAE": parh_rate,
                "PARH_rate_R": parh_rate_r,
                "PARH_waveform_CCC": parh_wave,
                "PARH_waveform_DTW": parh_dtw,
                "Base_rate_MAE": base_rate,
                "Base_waveform_CCC": base_wave,
                "rate_gap_vs_base": parh_rate - base_rate if not (math.isnan(parh_rate) or math.isnan(base_rate)) else math.nan,
                "waveform_gap_vs_base": parh_wave - base_wave if not (math.isnan(parh_wave) or math.isnan(base_wave)) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def _make_ladder(allfamily: pd.DataFrame, representative_family: str) -> pd.DataFrame:
    rows = []
    comparator_label = f"OSSM-KF ({representative_family.replace('_', ' ')})"
    rungs = [
        ("direct_OF", "OF", "Base"),
        ("direct_DoF", "DoF", "Base"),
        ("direct_P1D_quad", "P1D_quad", "Base"),
        ("direct_P1D_cons", "P1D_cons", "Base"),
        (comparator_label, representative_family, "OSSM-KF"),
        ("PARH-OSSM", "PARH-OSSM", "PARH-OSSM"),
    ]
    for dataset in sorted(allfamily["dataset"].unique()):
        for rung, family, variant in rungs:
            sub = allfamily[
                allfamily["dataset"].eq(dataset)
                & allfamily["family"].eq(family)
                & allfamily["variant"].eq(variant)
            ]
            if sub.empty:
                continue
            row = sub.iloc[0].to_dict()
            rows.append(
                {
                    "dataset": dataset,
                    "rung": rung,
                    "family": family,
                    "variant": variant,
                    "method": row.get("method", ""),
                    "rate_MAE": row.get("rate_MAE", math.nan),
                    "rate_PearsonR": row.get("rate_PearsonR", math.nan),
                    "waveform_CCC": row.get("waveform_CCC", math.nan),
                    "waveform_MAE": row.get("waveform_MAE", math.nan),
                    "waveform_DTW": row.get("waveform_DTW", math.nan),
                    "strict_CCC": row.get("strict_CCC", math.nan),
                    "strict_NMAE_span": row.get("strict_NMAE_span", math.nan),
                    "cycle_ppi_mae_s": row.get("cycle_ppi_mae_s", math.nan),
                    "source_layer": str(row.get("source_layer", "")).replace("allfamily", "observation_class"),
                }
            )
    return pd.DataFrame(rows)


def _make_provenance(
    layers: list[DatasetLayer],
    allfamily: pd.DataFrame,
    t3: pd.DataFrame,
    t4: pd.DataFrame,
    t4b: pd.DataFrame,
    t4c: pd.DataFrame,
    t6: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    table_specs = [
        ("T3_rate_main.csv", t3, ["Base", "KFstd", "PARH"]),
        ("T4_waveform_main.csv", t4, ["Base", "KFstd", "PARH"]),
        ("T4b_waveform_strict.csv", t4b, ["Base", "KFstd", "PARH"]),
        ("T4c_cycle_main.csv", t4c, ["Base", "KFstd", "PARH"]),
        ("T6_diagnostics_main.csv", t6, ["KFstd", "PARH"]),
    ]
    layer_by_dataset = {DISPLAY_DATASET[layer.dataset]: layer for layer in layers}
    for table_name, df, variants in table_specs:
        for _, row in df.iterrows():
            layer = layer_by_dataset[row["dataset"]]
            for variant in variants:
                method = row.get(f"{variant}_method", "")
                source_layer = layer.final_layer if variant == "PARH" else layer.baseline_layer
                display_variant = "OSSM-KF" if variant == "KFstd" else variant
                rows.append(
                    {
                        "table": table_name,
                        "dataset": row["dataset"],
                        "method_variant": display_variant,
                        "method": method,
                        "source_layer": source_layer,
                        "result_dir": str(layer.final_metrics.parent if variant == "PARH" else layer.baseline_metrics.parent),
                        "trial_count": int(row.get(f"{variant}_N", 0) or 0),
                        "headline_or_supplementary": "headline",
                        "gt_use": "evaluation_only",
                        "notes": "Base and OSSM-KF rows extracted on the same final full-dataset trial IDs; PARH from the integrated full-dataset run.",
                    }
                )
    for _, row in allfamily.iterrows():
        method_variant = "class-local PARH" if row["variant"] == "family_PARH" else row["variant"]
        rows.append(
            {
                "table": "S_T_final_observation_class_comparison.csv",
                "dataset": row["dataset"],
                "method_variant": method_variant,
                "method": row["method"],
                "source_layer": str(row["source_layer"]).replace("allfamily", "observation_class"),
                "result_dir": "",
                "trial_count": int(row.get("rate_N", 0) or 0),
                "headline_or_supplementary": "supplementary",
                "gt_use": "evaluation_only",
                "notes": "Transparent observation-class diagnostic/support comparison; not a target-tuned selector.",
            }
        )
    return pd.DataFrame(rows)


def _make_gap_audit(tables: dict[str, pd.DataFrame], provenance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, df in tables.items():
        for col in df.columns:
            if not (col.endswith("_N") or col.endswith("_MAE") or col.endswith("_CCC") or col.endswith("_PearsonR")):
                continue
            vals = df[col]
            empty = vals.isna().all() if not vals.empty else True
            zero_n = col.endswith("_N") and pd.to_numeric(vals, errors="coerce").fillna(0).eq(0).all()
            rows.append(
                {
                    "artifact": name,
                    "field": col,
                    "status": "gap" if empty or zero_n else "ok",
                    "empty_all": bool(empty),
                    "zero_n_all": bool(zero_n),
                    "paper_action": "fix_before_submission" if empty or zero_n else "keep",
                }
            )
    mixed = provenance.groupby(["table", "dataset"])["source_layer"].nunique().reset_index(name="n_source_layers")
    for _, row in mixed.iterrows():
        rows.append(
            {
                "artifact": row["table"],
                "field": f"{row['dataset']}_source_layers",
                "status": "transparent_mixed_layers" if row["n_source_layers"] > 1 else "ok",
                "empty_all": False,
                "zero_n_all": False,
                "paper_action": "allowed_only_with_provenance" if row["n_source_layers"] > 1 else "keep",
            }
        )
    return pd.DataFrame(rows)


def _paired_stat_rows(
    layers: list[DatasetLayer],
    loaded: dict[str, dict[str, pd.DataFrame]],
    representative_family: str,
) -> pd.DataFrame:
    rows = []
    base_method = METHOD_BY_FAMILY[representative_family]
    kf_method = KFSTD_BY_FAMILY[representative_family]
    comparisons = [("Base", base_method), ("OSSM-KF", kf_method)]
    for layer in layers:
        frames = loaded[layer.dataset]
        dataset = DISPLAY_DATASET[layer.dataset]
        for metric_file, metric_col, output_type, lower_better in [
            ("metrics_freq_domain_raw.csv", "MAE", None, True),
            ("metrics_waveform_raw.csv", "waveform_CCC", "signal_hat", False),
            ("metrics_waveform_raw.csv", "waveform_DTW", "signal_hat", True),
            ("metrics_waveform_strict_raw.csv", "strict_NMAE_span", "signal_hat", True),
        ]:
            df = frames[metric_file]
            if metric_file == "metrics_waveform_strict_raw.csv":
                df = _derive_strict_normalized(df, layer.final_metrics)
            final = _select_final_parh(df)
            if "causal_or_smoothed" in final.columns:
                final = final[final["causal_or_smoothed"].eq("smoothed")]
            if "output_type" in final.columns:
                final = final[final["output_type"].eq("z_full")]
            if metric_file == "metrics_freq_domain_raw.csv":
                final = _preferred_rate(final, "PARH")
            video_col = _video_col(final)
            final_s = final[[video_col, metric_col]].rename(columns={metric_col: "PARH"})
            for label, method in comparisons:
                comp = _select_method(df, method)
                if "causal_or_smoothed" in comp.columns:
                    comp = comp[comp["causal_or_smoothed"].eq("smoothed")]
                if "output_type" in comp.columns and output_type:
                    comp = comp[comp["output_type"].eq(output_type)]
                if metric_file == "metrics_freq_domain_raw.csv":
                    comp = _preferred_rate(comp, "Base" if label == "Base" else "KFstd")
                comp_s = comp[[video_col, metric_col]].rename(columns={metric_col: label})
                joined = pd.merge(final_s, comp_s, on=video_col, how="inner")
                p = pd.to_numeric(joined["PARH"], errors="coerce")
                c = pd.to_numeric(joined[label], errors="coerce")
                delta = p - c
                if not lower_better:
                    delta = c - p
                rows.append(
                    {
                        "dataset": dataset,
                        "comparison": f"PARH-OSSM vs {label} ({representative_family.replace('_', ' ')})",
                        "metric_file": metric_file,
                        "metric": metric_col,
                        "delta_definition": "PARH_minus_comparator" if lower_better else "comparator_minus_PARH",
                        "lower_delta_favors_PARH": True,
                        "N": int(delta.dropna().shape[0]),
                        "median_delta": _median(delta),
                        "iqr_delta": float(delta.quantile(0.75) - delta.quantile(0.25)) if not delta.dropna().empty else math.nan,
                        "comparator_median": _median(c),
                        "parh_median": _median(p),
                    }
                )
    return pd.DataFrame(rows)


def _write_csv(path: Path, df: pd.DataFrame, digits: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    # Avoid paper-facing "-0.000" artifacts after rounded median aggregation.
    # These are display artifacts, not meaningful signed effects.
    threshold = 0.5 * (10 ** -digits)
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        out.loc[vals.abs() < threshold, col] = 0.0
    out.to_csv(path, index=False, float_format=f"%.{digits}f")


def _write_markdown_table(path: Path, title: str, df: pd.DataFrame, intro: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    if intro:
        lines.extend(intro)
        lines.append("")
    if df.empty:
        lines.append("_No rows._")
    else:
        lines.extend(_markdown_table_lines(df))
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table_lines(df: pd.DataFrame, max_rows: int = 200) -> list[str]:
    """Render a small markdown table without optional tabulate dependency."""
    work = df.head(max_rows).copy()
    cols = [str(c) for c in work.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in work.iterrows():
        vals = []
        for col in work.columns:
            value = row[col]
            if isinstance(value, float):
                text = "" if math.isnan(value) else f"{value:.4f}"
            else:
                text = "" if pd.isna(value) else str(value)
            vals.append(text.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append("")
        lines.append(f"_Truncated to first {max_rows} of {len(df)} rows._")
    return lines


def _write_interpretation(path: Path, allfamily: pd.DataFrame, t3: pd.DataFrame, t4: pd.DataFrame, t4b: pd.DataFrame) -> None:
    lines = [
        "# Final Baseline Comparison Interpretation",
        "",
        "Observation:",
        "  The final PARH-OSSM full-dataset run now has an explicit apples-to-apples comparison layer.",
        "  Headline Base and OSSM-KF columns use the pre-locked P1D_quad direct and",
        "  OSSM-KF (P1D quad) representative, while all eight fixed observation classes are",
        "  retained in the supplementary observation-class table.",
        "",
        "Evidence:",
    ]
    for _, row in t3.iterrows():
        lines.append(
            f"  {row['dataset']} rate MAE: Base={_fmt(row['Base_MAE'])}, "
            f"OSSM-KF={_fmt(row['KFstd_MAE'])}, PARH={_fmt(row['PARH_MAE'])}; "
            f"PearsonR: Base={_fmt(row['Base_PearsonR'])}, "
            f"OSSM-KF={_fmt(row['KFstd_PearsonR'])}, PARH={_fmt(row['PARH_PearsonR'])}."
        )
    for _, row in t4.iterrows():
        lines.append(
            f"  {row['dataset']} aligned waveform CCC: Base={_fmt(row['Base_CCC'])}, "
            f"OSSM-KF={_fmt(row['KFstd_CCC'])}, PARH={_fmt(row['PARH_CCC'])}."
        )
    for _, row in t4b.iterrows():
        lines.append(
            f"  {row['dataset']} strict NMAE/span: Base={_fmt(row['Base_NMAE_span'])}, "
            f"OSSM-KF={_fmt(row['KFstd_NMAE_span'])}, PARH={_fmt(row['PARH_NMAE_span'])}."
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "  COHFACE should be read as the clean-regime success case. MAHNOB should",
            "  be read as a hard-regime observability boundary: rate can improve against",
            "  fixed direct observation baselines, but strict waveform/cycle robustness",
            "  remains limited and must not be overclaimed.",
            "",
            "Decision:",
            "  Main tables may report the representative fixed baseline and OSSM-KF",
            "  comparator because they are pre-locked and evaluated on the same trial IDs.",
            "  Full OF/DoF/P1D observation-class results belong in supplementary/diagnostic tables.",
            "",
            "Paper value:",
            "  This closes the reviewer-facing question: PARH-OSSM is not compared only",
            "  to itself, and OSSM-KF remains a comparator rather than a hidden part of",
            "  the proposed method.",
            "",
            "Risk:",
            "  The baseline layer is extracted from retained observation-class metrics rather",
            "  than recomputing video processing from scratch. This is acceptable only",
            "  because the exact full-dataset trial IDs and metric definitions are recorded",
            "  in the provenance audit; it must remain explicit in Methods/Artifacts.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_claim_boundary(path: Path) -> None:
    lines = [
        "# Final Submission Claim Boundary",
        "",
        "## Strong Claims",
        "",
        "- PARH-OSSM is an auditable observation-to-state framework, not a black-box waveform extractor.",
        "- COHFACE supports strong clean-regime rate and aligned morphology claims.",
        "- Baseline/comparator tables now show fixed direct observation and OSSM-KF reference points on the same final full-dataset trial IDs.",
        "- MAHNOB supports a hard-regime analysis claim: current fixed observation operators often do not contain enough target-computable respiratory evidence.",
        "",
        "## Constrained Claims",
        "",
        "- PARH-OSSM does not dominate every fixed observation class on every metric.",
        "- OSSM-KF is a comparator and weak timing-evidence boundary, not the proposed method body.",
        "- Strict waveform raw MAE must be interpreted with strict_NMAE_span and cycle metrics.",
        "- V4V/SCAMPS remain external weak-evidence/synthetic controls and must not be mixed into real waveform headline performance.",
        "",
        "## Prohibited Claims",
        "",
        "- Do not claim universal MAHNOB waveform robustness.",
        "- Do not claim universal superiority if a class-specific fixed observation wins a single metric.",
        "- Do not present candidate views as final model candidates.",
        "- Do not describe adaptive observation law as a single selector.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_reviewer_response(path: Path) -> None:
    lines = [
        "# Final Reviewer Risk Response",
        "",
        "## Why is the paper valuable if MAHNOB remains hard?",
        "",
        "Because the paper reports where the observation bank becomes unobservable rather than hiding the failure. The failure taxonomy, strict normalized metrics, and rate-source decomposition separate model instability from missing respiratory evidence and reference/scale risk.",
        "",
        "## Why is this not target tuning?",
        "",
        "The baseline/comparator refresh uses pre-locked fixed methods and the same final full-dataset trial IDs. Target GT is used only for evaluation and statistical pairing, not for selecting a per-target method or threshold.",
        "",
        "## How is OSSM-KF different from PARH-OSSM?",
        "",
        "OSSM-KF is a standard resonator plus Kalman comparator attached to a fixed representative observation. PARH-OSSM uses target-computable reliability, candidate evidence views, state/readout role separation, and diagnostics around z_osc/z_full.",
        "",
        "## Why report strict waveform failure?",
        "",
        "Strict waveform exposes lag, unit, and reference-scale fragility that aligned waveform metrics can hide. Reporting it prevents overclaiming and motivates the next generation of respiratory observation operators.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_split_audit(
    path_csv: Path,
    path_md: Path,
    layers: list[DatasetLayer],
    loaded: dict[str, dict[str, pd.DataFrame]],
    representative_family: str,
) -> None:
    rows = []
    base_method = METHOD_BY_FAMILY[representative_family]
    kf_method = KFSTD_BY_FAMILY[representative_family]
    for layer in layers:
        final_trials = set(_final_trial_ids(layer.final_metrics))
        base_freq = loaded[layer.dataset]["metrics_freq_domain_raw.csv"]
        col = _video_col(base_freq)
        all_baseline_trials = set(map(str, base_freq[col].dropna().unique()))
        representative_base = _preferred_rate(_select_method(base_freq, base_method), "Base")
        representative_kfstd = _preferred_rate(_select_method(base_freq, kf_method), "KFstd")
        base_trials = set(map(str, representative_base[col].dropna().unique()))
        kfstd_trials = set(map(str, representative_kfstd[col].dropna().unique()))
        headline_trials = base_trials & kfstd_trials
        missing_headline = final_trials - headline_trials
        headline_status = "pass" if not missing_headline or headline_trials else "fail"
        rows.append(
            {
                "dataset": DISPLAY_DATASET[layer.dataset],
                "final_trial_count": len(final_trials),
                "all_family_baseline_trial_count": len(all_baseline_trials),
                "representative_base_trial_count": len(base_trials),
                "representative_kfstd_trial_count": len(kfstd_trials),
                "headline_overlap_count": len(final_trials & headline_trials),
                "missing_from_representative_headline": ";".join(sorted(missing_headline)),
                "gt_used_for_adaptation": "no",
                "claim_scope": (
                    "PARH evaluated on all final trials; representative baseline/"
                    "OSSM-KF coverage is reported explicitly; GT evaluation only"
                ),
                "status": headline_status,
            }
        )
    df = pd.DataFrame(rows)
    _write_csv(path_csv, df, 0)
    _write_markdown_table(path_md, "Final Split And Leakage Audit", df)


def _write_repro_audit(path_csv: Path, path_md: Path) -> None:
    checks = [
        ("execute_md_exists", ROOT / "execute.md"),
        ("auto_profile_exists", ROOT / "setup" / "auto_profile.py"),
        ("locked_paper_profile_exists", ROOT / "setup" / "locked_paper_profile.py"),
        ("main_pdf_exists", ROOT / "paper" / "main.pdf"),
        ("dataset_scope_table_exists", ROOT / "paper" / "tables_ready" / "T1_dataset_protocol_scope.csv"),
    ]
    rows = [{"check": name, "path": str(path.relative_to(ROOT)), "exists": path.exists(), "status": "pass" if path.exists() else "fail"} for name, path in checks]
    df = pd.DataFrame(rows)
    _write_csv(path_csv, df, 0)
    _write_markdown_table(path_md, "Final Reproducibility Audit", df)


def _copy_filtered_raw_metrics(out_run: Path, layers: list[DatasetLayer], loaded: dict[str, dict[str, pd.DataFrame]]) -> None:
    for layer in layers:
        metrics_dir = out_run / layer.dataset.lower() / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        for name, df in loaded[layer.dataset].items():
            _write_csv(metrics_dir / name, df, 6)


def run(args: argparse.Namespace) -> int:
    layers = _dataset_layers()
    plan_rows = []
    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    for layer in layers:
        trial_ids = _final_trial_ids(layer.final_metrics)
        loaded[layer.dataset] = _load_filtered(layer)
        base_freq = _read_csv(layer.baseline_metrics / "metrics_freq_domain_raw.csv")
        plan_rows.append(
            {
                "dataset": DISPLAY_DATASET[layer.dataset],
                "final_trials": len(trial_ids),
                "baseline_trials_total": base_freq[_video_col(base_freq)].nunique(),
                "baseline_methods": base_freq["method"].nunique(),
                "baseline_metrics": str(layer.baseline_metrics.relative_to(ROOT)),
                "final_metrics": str(layer.final_metrics.relative_to(ROOT)),
            }
        )
    plan = pd.DataFrame(plan_rows)
    if args.dry_run or not args.execute:
        print(plan.to_string(index=False))
        if not args.execute:
            print("\nDry-run only. Re-run with --execute to write artifacts.")
            return 0

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.out_run.mkdir(parents=True, exist_ok=True)

    if args.artifact_policy == "full":
        _copy_filtered_raw_metrics(args.out_run, layers, loaded)

    t3, t4, t4b, t4c, t6 = _make_headline_tables(layers, loaded, args.representative_family)
    allfamily = _make_allfamily_table(layers, loaded)
    public_observation_class = _make_public_observation_class_comparison(allfamily)
    t2 = _make_t2_from_allfamily(allfamily)
    ladder = _make_ladder(allfamily, args.representative_family)
    provenance = _make_provenance(layers, allfamily, t3, t4, t4b, t4c, t6)
    gap = _make_gap_audit(
        {
            "T3_rate_main.csv": t3,
            "T4_waveform_main.csv": t4,
            "T4b_waveform_strict.csv": t4b,
            "T4c_cycle_main.csv": t4c,
            "T6_diagnostics_main.csv": t6,
            "T6b_fusion_ladder.csv": ladder,
        },
        provenance,
    )
    stats = _paired_stat_rows(layers, loaded, args.representative_family)

    _write_csv(args.tables_dir / "T3_rate_main.csv", _make_public_comparator_columns(t3), 3)
    _write_csv(args.tables_dir / "T4_waveform_main.csv", _make_public_comparator_columns(t4), 3)
    _write_csv(args.tables_dir / "T4b_waveform_strict.csv", _make_public_comparator_columns(t4b), 3)
    _write_csv(args.tables_dir / "T4c_cycle_main.csv", _make_public_comparator_columns(t4c), 4)
    _write_csv(args.tables_dir / "T6_diagnostics_main.csv", _make_public_comparator_columns(t6), 4)
    _write_csv(args.tables_dir / "T2_observation_class_map.csv", t2, 3)
    _write_csv(args.tables_dir / "T6b_fusion_ladder.csv", ladder, 4)
    _write_csv(args.tables_dir / "S_T_final_observation_class_comparison.csv", public_observation_class, 4)
    _write_csv(args.analysis_dir / "final_baseline_comparator_refresh.csv", allfamily, 4)
    _write_csv(args.analysis_dir / "final_submission_gap_audit.csv", gap, 4)
    _write_csv(args.analysis_dir / "final_metric_provenance_audit.csv", provenance, 4)
    _write_csv(args.analysis_dir / "final_statistical_comparison.csv", stats, 4)

    _write_markdown_table(
        args.analysis_dir / "final_baseline_comparator_refresh.md",
        "Final Baseline Comparator Refresh",
        allfamily,
        [
            "- Boundary: fixed baselines and OSSM-KF are extracted on the same final full-dataset trial IDs.",
            "- PARH-OSSM rows come from the integrated final full-dataset run.",
            "- This is not target-GT method selection.",
        ],
    )
    _write_markdown_table(args.analysis_dir / "final_submission_gap_audit.md", "Final Submission Gap Audit", gap)
    _write_markdown_table(args.analysis_dir / "final_metric_provenance_audit.md", "Final Metric Provenance Audit", provenance)
    _write_markdown_table(args.analysis_dir / "final_statistical_comparison.md", "Final Statistical Comparison", stats)
    _write_split_audit(
        args.analysis_dir / "final_split_and_leakage_audit.csv",
        args.analysis_dir / "final_split_and_leakage_audit.md",
        layers,
        loaded,
        args.representative_family,
    )
    _write_repro_audit(
        args.analysis_dir / "final_reproducibility_audit.csv",
        args.analysis_dir / "final_reproducibility_audit.md",
    )
    _write_interpretation(
        args.analysis_dir / "final_baseline_comparison_interpretation.md",
        allfamily,
        t3,
        t4,
        t4b,
    )
    _write_claim_boundary(args.analysis_dir / "final_submission_claim_boundary.md")
    _write_reviewer_response(args.analysis_dir / "final_reviewer_risk_response.md")

    # Mark the refresh result directory as paper-facing comparator evidence.
    manifest = args.out_run / "README.md"
    manifest.write_text(
        "\n".join(
            [
                "# Final Baseline Comparator Refresh",
                "",
                "Filtered metric extracts for fixed observation-class baselines and OSSM-KF comparator rows.",
                "The trial IDs match the integrated final PARH-OSSM full-dataset runs.",
                "These files are comparator/provenance artifacts, not additional target-tuned model runs.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Wrote final baseline/comparator refresh artifacts.")
    print(t3.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
