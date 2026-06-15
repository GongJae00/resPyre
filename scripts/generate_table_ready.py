#!/usr/bin/env python3
"""Generate table-ready CSVs for T3/T4/T6 from production run metrics.

Usage:
    python scripts/generate_table_ready.py
    python scripts/generate_table_ready.py --cohface-metrics path/to/cohface/metrics --mahnob-metrics path/to/mahnob/metrics
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "paper" / "tables_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from components.observations.semantics import (
    CANONICAL_OBSERVATION_FAMILY_ORDER,
    get_observation_family_semantics,
    observation_family_order_key,
)

DEFAULT_DATASETS = {
    "COHFACE": ROOT / "results" / "cohface_parh_ossm_prod" / "cohface_parh_ossm_prod" / "metrics",
    "MAHNOB": ROOT / "results" / "mahnob_parh_ossm_prod" / "mahnob_parh_ossm_prod" / "metrics",
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate table-ready CSVs from metrics directories.")
    parser.add_argument(
        "--dataset-metrics",
        action="append",
        default=[],
        help="Explicit dataset metrics mapping in the form DATASET=/abs/or/rel/path/to/metrics. "
             "If provided, only these mappings are used.",
    )
    parser.add_argument(
        "--cohface-metrics",
        type=Path,
        default=DEFAULT_DATASETS["COHFACE"],
        help="Path to COHFACE metrics directory",
    )
    parser.add_argument(
        "--mahnob-metrics",
        type=Path,
        default=DEFAULT_DATASETS["MAHNOB"],
        help="Path to MAHNOB-HCI metrics directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory where table-ready CSVs will be written",
    )
    return parser.parse_args()


def _parse_dataset_metrics_specs(specs):
    datasets = {}
    for spec in specs or []:
        raw = str(spec).strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(
                f"Invalid --dataset-metrics value '{spec}'. Expected DATASET=/path/to/metrics."
            )
        name, path_str = raw.split("=", 1)
        dataset_name = str(name).strip().upper()
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        datasets[dataset_name] = path
    return datasets


def _median_or_nan(series):
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.median())

# Method variant classification
def classify_method(method_name):
    """Return (family, variant) tuple."""
    m = method_name.lower().replace(" ", "_")
    if m in {"parh_ossm", "target_calibrated_multifamily_parh_ossm"}:
        variant = "PARH"
        family = "PARH-OSSM"
    elif "__parh_ossm" in m:
        variant = "PARH"
        family = m.split("__parh_ossm")[0]
    elif "__kfstd" in m:
        variant = "KFstd"
        family = m.split("__kfstd")[0]
    else:
        variant = "Base"
        family = m
    # Normalize family names
    family_map = {
        "of_farneback": "OF",
        "of_disp_bridge": "OF_bridge",
        "of": "OF",
        "dof": "DoF",
        "dof_disp_bridge": "DoF_bridge",
        "profile1d_linear": "P1D_lin",
        "profile1d_linear_bridge": "P1D_lin_bridge",
        "profile1d_quadratic": "P1D_quad",
        "profile1d_quadratic_bridge": "P1D_quad_bridge",
        "profile1d_cubic": "P1D_cub",
        "profile1d_cubic_bridge": "P1D_cub_bridge",
        "profile1d_consensus": "P1D_cons",
        "fusion_of_p1d_quadratic": "Fused_OF+P1D_quad",
        "pair_of_p1d_quadratic": "Pair_OF+P1D_quad",
        "assist_of_p1d_quadratic": "Assist_OF->P1D_quad",
    }
    family = family_map.get(family, family)
    return family, variant


DISPLAY_TO_CANONICAL = {
    str(get_observation_family_semantics(name).get("display_name")): name
    for name in CANONICAL_OBSERVATION_FAMILY_ORDER
}


def _family_display_order_key(family: str):
    canonical = DISPLAY_TO_CANONICAL.get(str(family), str(family))
    return observation_family_order_key(canonical)


def _ordered_display_families(families):
    return sorted((str(f) for f in families), key=_family_display_order_key)


def _best_row(df, primary, ascending, secondary=None):
    work = df.copy()
    if work.empty:
        return None
    if secondary is None:
        secondary = []
    cols = [primary] + secondary
    work = work.dropna(subset=[primary])
    if work.empty:
        return None
    sort_cols = [primary] + secondary
    sort_ascending = [ascending] + [True] * len(secondary)
    if "PearsonR" in secondary:
        idx = secondary.index("PearsonR")
        sort_ascending[idx + 1] = False
    if "waveform_CCC" in secondary:
        idx = secondary.index("waveform_CCC")
        sort_ascending[idx + 1] = False
    work = work.sort_values(sort_cols, ascending=sort_ascending, kind="mergesort")
    return work.iloc[0]


def _aggregate_rate_rows(fr_main):
    """Collapse per-trial rate rows to one median row per method/family/variant."""
    if fr_main.empty:
        return fr_main.copy()
    agg = (
        fr_main.groupby(["method", "family", "variant"], as_index=False)
        .agg(
            MAE=("MAE", _median_or_nan),
            RMSE=("RMSE", _median_or_nan),
            PearsonR=("PearsonR", _median_or_nan),
        )
    )
    return agg


def _aggregate_waveform_rows(wf_main):
    """Collapse per-trial waveform rows to one median row per method/family/variant."""
    if wf_main.empty:
        return wf_main.copy()
    agg = (
        wf_main.groupby(["method", "family", "variant"], as_index=False)
        .agg(
            waveform_CCC=("waveform_CCC", _median_or_nan),
            waveform_MAE=("waveform_MAE", _median_or_nan),
            waveform_DTW=("waveform_DTW", _median_or_nan),
        )
    )
    return agg


def generate_t3(datasets, out_dir):
    """T3: Rate accuracy (MAE, RMSE, PearsonR) — median across trials per dataset×family.

    Policy lock:
    - Base rows use the compatibility signal-spectral rate path.
    - OSSM-KF / PARH rows use track_hz when present.

    This keeps the exported table aligned with the evaluation design docs
    even if future raw CSVs contain mixed rate_source rows for a single family.
    """
    rows = []
    for ds_name, metrics_dir in datasets.items():
        csv_path = metrics_dir / "metrics_freq_domain_raw.csv"
        if not csv_path.exists():
            print(f"  SKIP T3 {ds_name}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df[["family", "variant"]] = df["method"].apply(
            lambda m: pd.Series(classify_method(m))
        )
        for family in _ordered_display_families(df["family"].dropna().unique()):
            fdf = df[df["family"] == family]
            row = {"dataset": ds_name, "family": family}
            for variant in ["Base", "KFstd", "PARH"]:
                vdf = fdf[fdf["variant"] == variant].copy()
                if "rate_source" in vdf.columns:
                    preferred = "signal_spectral" if variant == "Base" else "track_hz"
                    preferred_vdf = vdf[vdf["rate_source"] == preferred]
                    if not preferred_vdf.empty:
                        vdf = preferred_vdf
                row[f"{variant}_rate_source"] = (
                    str(vdf["rate_source"].iloc[0])
                    if (not vdf.empty and "rate_source" in vdf.columns)
                    else ""
                )
                if vdf.empty:
                    for col in ["MAE", "RMSE", "PearsonR"]:
                        row[f"{variant}_{col}"] = np.nan
                else:
                    for col in ["MAE", "RMSE", "PearsonR"]:
                        row[f"{variant}_{col}"] = _median_or_nan(vdf[col])
                row[f"{variant}_N"] = len(vdf)
            rows.append(row)
    t3 = pd.DataFrame(rows)
    out_path = out_dir / "T3_rate_main.csv"
    t3.to_csv(out_path, index=False, float_format="%.3f")
    print(f"  T3 saved: {out_path} ({len(t3)} rows)")
    return t3


def generate_t4(datasets, out_dir):
    """T4: Waveform fidelity (CCC, wMAE, DTW) — unified comparison.
    Main: Base(signal_hat) / OSSM-KF(signal_hat) / PARH(z_full) smoothed only.
    """
    rows = []
    for ds_name, metrics_dir in datasets.items():
        csv_path = metrics_dir / "metrics_waveform_raw.csv"
        if not csv_path.exists():
            print(f"  SKIP T4 {ds_name}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df[["family", "variant"]] = df["method"].apply(
            lambda m: pd.Series(classify_method(m))
        )
        # Main T4: smoothed only
        df_smooth = df[df["causal_or_smoothed"] == "smoothed"]

        for family in _ordered_display_families(df_smooth["family"].dropna().unique()):
            fdf = df_smooth[df_smooth["family"] == family]
            row = {"dataset": ds_name, "family": family}
            for variant, otype in [("Base", "signal_hat"), ("KFstd", "signal_hat"), ("PARH", "z_full")]:
                vdf = fdf[(fdf["variant"] == variant) & (fdf["output_type"] == otype)]
                for col in ["waveform_CCC", "waveform_MAE", "waveform_DTW"]:
                    short = col.replace("waveform_", "")
                    row[f"{variant}_{short}"] = _median_or_nan(vdf[col]) if not vdf.empty else np.nan
                row[f"{variant}_N"] = len(vdf)
            rows.append(row)
    t4 = pd.DataFrame(rows)
    out_path = out_dir / "T4_waveform_main.csv"
    t4.to_csv(out_path, index=False, float_format="%.3f")
    print(f"  T4 saved: {out_path} ({len(t4)} rows)")
    return t4


def _strict_main_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["causal_or_smoothed"] == "smoothed"].copy()
    return pd.concat([
        df[(df["variant"].isin(["Base", "KFstd"])) & (df["output_type"] == "signal_hat")],
        df[(df["variant"] == "PARH") & (df["output_type"] == "z_full")],
    ], ignore_index=True)


def generate_t4_strict(datasets, out_dir):
    """Supplementary strict waveform table.

    Harder than T4 main: zero-lag, unit-preserving, no z-score alignment.
    """
    rows = []
    for ds_name, metrics_dir in datasets.items():
        csv_path = metrics_dir / "metrics_waveform_strict_raw.csv"
        if not csv_path.exists():
            print(f"  SKIP T4-strict {ds_name}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df[["family", "variant"]] = df["method"].apply(lambda m: pd.Series(classify_method(m)))
        df_main = _strict_main_filter(df)
        for family in _ordered_display_families(df_main["family"].dropna().unique()):
            fdf = df_main[df_main["family"] == family]
            row = {"dataset": ds_name, "family": family}
            for variant, otype in [("Base", "signal_hat"), ("KFstd", "signal_hat"), ("PARH", "z_full")]:
                vdf = fdf[(fdf["variant"] == variant) & (fdf["output_type"] == otype)]
                strict_cols = [
                    "strict_CCC",
                    "strict_MAE",
                    "strict_RMSE",
                    "strict_DTW",
                    "gt_span_p95p05",
                    "strict_NMAE_span",
                    "strict_NRMSE_span",
                    "strict_NDTW_span",
                ]
                for col in strict_cols:
                    if col not in vdf.columns:
                        continue
                    short = col.replace("strict_", "")
                    row[f"{variant}_{short}"] = _median_or_nan(vdf[col]) if not vdf.empty else np.nan
                row[f"{variant}_N"] = len(vdf)
            rows.append(row)
    t4s = pd.DataFrame(rows)
    out_path = out_dir / "T4b_waveform_strict.csv"
    t4s.to_csv(out_path, index=False, float_format="%.3f")
    print(f"  T4-strict saved: {out_path} ({len(t4s)} rows)")
    return t4s


def generate_t4_cycle(datasets, out_dir):
    """Supplementary cycle-level waveform timing table."""
    rows = []
    for ds_name, metrics_dir in datasets.items():
        csv_path = metrics_dir / "metrics_waveform_strict_raw.csv"
        if not csv_path.exists():
            print(f"  SKIP T4-cycle {ds_name}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df[["family", "variant"]] = df["method"].apply(lambda m: pd.Series(classify_method(m)))
        df_main = _strict_main_filter(df)
        for family in _ordered_display_families(df_main["family"].dropna().unique()):
            fdf = df_main[df_main["family"] == family]
            row = {"dataset": ds_name, "family": family}
            for variant, otype in [("Base", "signal_hat"), ("KFstd", "signal_hat"), ("PARH", "z_full")]:
                vdf = fdf[(fdf["variant"] == variant) & (fdf["output_type"] == otype)]
                for col in ["peak_time_mae_s", "trough_time_mae_s", "cycle_ppi_mae_s", "cycle_ie_abs_err"]:
                    row[f"{variant}_{col}"] = _median_or_nan(vdf[col]) if not vdf.empty else np.nan
                row[f"{variant}_N"] = len(vdf)
            rows.append(row)
    t4c = pd.DataFrame(rows)
    out_path = out_dir / "T4c_cycle_main.csv"
    t4c.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  T4-cycle saved: {out_path} ({len(t4c)} rows)")
    return t4c


def generate_t2(t3, t4, out_dir):
    """T2: Observation-family semantics and current PARH role map."""
    if t3 is None or t4 is None or t3.empty or t4.empty:
        print("  SKIP T2: T3/T4 inputs missing")
        return pd.DataFrame()
    t3c = t3[t3["dataset"] == "COHFACE"].copy()
    t4c = t4[t4["dataset"] == "COHFACE"].copy()
    order = [
        ("OF", "of_farneback"),
        ("OF_bridge", "of_disp_bridge"),
        ("DoF", "dof"),
        ("DoF_bridge", "dof_disp_bridge"),
        ("P1D_lin", "profile1d_linear"),
        ("P1D_quad", "profile1d_quadratic"),
        ("P1D_cub", "profile1d_cubic"),
        ("P1D_cons", "profile1d_consensus"),
    ]
    rows = []
    for display_name, semantic_key in order:
        rate_df = t3c[t3c["family"] == display_name]
        wave_df = t4c[t4c["family"] == display_name]
        rate_row = rate_df.iloc[0] if not rate_df.empty else None
        wave_row = wave_df.iloc[0] if not wave_df.empty else None
        sem = get_observation_family_semantics(semantic_key)
        current_strength = "mixed"
        if sem.get("waveform_primary"):
            current_strength = "waveform-primary"
        elif sem.get("rate_primary"):
            current_strength = "rate-primary"
        elif sem.get("helper_heavy"):
            current_strength = "helper-heavy"
        elif sem.get("nuisance_risk") == "high":
            current_strength = "nuisance-limited"
        def _safe_float(row, key):
            if row is None or key not in row.index:
                return np.nan
            val = row[key]
            if pd.isna(val) or str(val).strip() == "":
                return np.nan
            try:
                return float(val)
            except Exception:
                return np.nan

        parh_rate_mae = _safe_float(rate_row, "PARH_MAE")
        parh_rate_r = _safe_float(rate_row, "PARH_PearsonR")
        parh_wave_ccc = _safe_float(wave_row, "PARH_CCC")
        parh_wave_dtw = _safe_float(wave_row, "PARH_DTW")
        base_rate_mae = _safe_float(rate_row, "Base_MAE")
        base_wave_ccc = _safe_float(wave_row, "Base_CCC")

        rows.append(
            {
                "observation_class": display_name,
                "construction": sem.get("construction", ""),
                "domain": sem.get("observation_domain", ""),
                "primary_information": str(sem.get("primary_information", "")).replace("_", " "),
                "secondary_information": str(sem.get("secondary_information", "")).replace("_", " "),
                "nuisance_risk": sem.get("nuisance_risk", ""),
                "current_parh_role": str(sem.get("current_parh_role", "")).replace("family", "observation class"),
                "current_strength": current_strength,
                "PARH_rate_MAE": parh_rate_mae,
                "PARH_rate_R": parh_rate_r,
                "PARH_waveform_CCC": parh_wave_ccc,
                "PARH_waveform_DTW": parh_wave_dtw,
                "Base_rate_MAE": base_rate_mae,
                "Base_waveform_CCC": base_wave_ccc,
                "rate_gap_vs_base": parh_rate_mae - base_rate_mae if not (np.isnan(parh_rate_mae) or np.isnan(base_rate_mae)) else np.nan,
                "waveform_gap_vs_base": parh_wave_ccc - base_wave_ccc if not (np.isnan(parh_wave_ccc) or np.isnan(base_wave_ccc)) else np.nan,
            }
        )
    t2 = pd.DataFrame(rows)
    out_path = out_dir / "T2_observation_class_map.csv"
    t2.to_csv(out_path, index=False, float_format="%.3f")
    print(f"  T2 saved: {out_path} ({len(t2)} rows)")
    return t2


def generate_t6(datasets, out_dir):
    """T6: Filter diagnostics by dataset and observation class (OSSM-KF and PARH only)."""
    diag_cols = [
        "NIS_Mean", "NIS_InBand", "Lambda_Mean", "Lambda_LT1_Frac",
        "Coverage95", "Stability_Sec",
    ]
    rows = []
    for ds_name, metrics_dir in datasets.items():
        csv_path = metrics_dir / "metrics_filter_diagnostics_raw.csv"
        if not csv_path.exists():
            print(f"  SKIP T6 {ds_name}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df[["family", "variant"]] = df["method"].apply(
            lambda m: pd.Series(classify_method(m))
        )
        # Only OSSM-KF and PARH have diagnostics in the internal variant map.
        for family in _ordered_display_families(df["family"].dropna().unique()):
            fdf = df[df["family"] == family]
            row = {"dataset": ds_name, "family": family}
            for variant in ["KFstd", "PARH"]:
                vdf = fdf[fdf["variant"] == variant]
                for col in diag_cols:
                    val = _median_or_nan(vdf[col]) if col in vdf.columns and not vdf.empty else np.nan
                    row[f"{variant}_{col}"] = val
                row[f"{variant}_N"] = len(vdf)
            rows.append(row)
    t6 = pd.DataFrame(rows)
    out_path = out_dir / "T6_diagnostics_main.csv"
    t6.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  T6 saved: {out_path} ({len(t6)} rows)")
    return t6


def generate_fusion_ladder(datasets, out_dir):
    """Generate a fusion-ladder table when fusion methods are present.

    Rows:
    1. best single-family Base
    2. best single-family OSSM-KF
    3. best single-family PARH
    4. fused Base
    5. fused OSSM-KF
    6. fused PARH
    """
    ladder_rows = []
    single_families = {"OF", "OF_bridge", "DoF", "DoF_bridge", "P1D_lin", "P1D_lin_bridge", "P1D_quad", "P1D_quad_bridge", "P1D_cub", "P1D_cub_bridge", "P1D_cons"}
    fused_base_family = "Fused_OF+P1D_quad"
    fused_pair_family = "Pair_OF+P1D_quad"
    for ds_name, metrics_dir in datasets.items():
        wf_path = metrics_dir / "metrics_waveform_raw.csv"
        fr_path = metrics_dir / "metrics_freq_domain_raw.csv"
        if not wf_path.exists() or not fr_path.exists():
            continue
        wf = pd.read_csv(wf_path)
        fr = pd.read_csv(fr_path)
        wf[["family", "variant"]] = wf["method"].apply(lambda m: pd.Series(classify_method(m)))
        fr[["family", "variant"]] = fr["method"].apply(lambda m: pd.Series(classify_method(m)))

        wf_smooth = wf[wf["causal_or_smoothed"] == "smoothed"].copy()
        wf_main = pd.concat([
            wf_smooth[(wf_smooth["variant"].isin(["Base", "KFstd"])) & (wf_smooth["output_type"] == "signal_hat")],
            wf_smooth[(wf_smooth["variant"] == "PARH") & (wf_smooth["output_type"] == "z_full")],
        ], ignore_index=True)
        fr_main = fr[(fr["variant"] != "PARH") | (fr["rate_source"] == "track_hz")].copy()
        fr_agg = _aggregate_rate_rows(fr_main)
        wf_agg = _aggregate_waveform_rows(wf_main)

        for block in ["T3_rate", "T4_waveform"]:
            rows = []
            if block == "T3_rate":
                for variant in ["Base", "KFstd", "PARH"]:
                    cands = fr_agg[(fr_agg["variant"] == variant) & (fr_agg["family"].isin(single_families))]
                    best = _best_row(cands, "MAE", True, ["RMSE", "PearsonR"])
                    if best is not None:
                        rows.append({
                            "dataset": ds_name,
                            "block": block,
                            "rung": f"best_single_{variant.lower()}",
                            "method": best["method"],
                            "family": best["family"],
                            "MAE": best["MAE"],
                            "RMSE": best["RMSE"],
                            "PearsonR": best["PearsonR"],
                            "waveform_CCC": np.nan,
                            "waveform_MAE": np.nan,
                            "waveform_DTW": np.nan,
                        })
                for rung, variant, family in [
                    ("fused_base", "Base", fused_base_family),
                    ("fused_kfstd", "KFstd", fused_pair_family),
                    ("fused_parh", "PARH", fused_pair_family),
                ]:
                    cands = fr_agg[(fr_agg["variant"] == variant) & (fr_agg["family"] == family)]
                    best = _best_row(cands, "MAE", True, ["RMSE", "PearsonR"])
                    if best is not None:
                        rows.append({
                            "dataset": ds_name,
                            "block": block,
                            "rung": rung,
                            "method": best["method"],
                            "family": best["family"],
                            "MAE": best["MAE"],
                            "RMSE": best["RMSE"],
                            "PearsonR": best["PearsonR"],
                            "waveform_CCC": np.nan,
                            "waveform_MAE": np.nan,
                            "waveform_DTW": np.nan,
                        })
            else:
                for variant in ["Base", "KFstd", "PARH"]:
                    cands = wf_agg[(wf_agg["variant"] == variant) & (wf_agg["family"].isin(single_families))]
                    best = _best_row(cands, "waveform_MAE", True, ["waveform_DTW", "waveform_CCC"])
                    if best is not None:
                        rows.append({
                            "dataset": ds_name,
                            "block": block,
                            "rung": f"best_single_{variant.lower()}",
                            "method": best["method"],
                            "family": best["family"],
                            "MAE": np.nan,
                            "RMSE": np.nan,
                            "PearsonR": np.nan,
                            "waveform_CCC": best["waveform_CCC"],
                            "waveform_MAE": best["waveform_MAE"],
                            "waveform_DTW": best["waveform_DTW"],
                        })
                for rung, variant, family in [
                    ("fused_base", "Base", fused_base_family),
                    ("fused_kfstd", "KFstd", fused_pair_family),
                    ("fused_parh", "PARH", fused_pair_family),
                ]:
                    cands = wf_agg[(wf_agg["variant"] == variant) & (wf_agg["family"] == family)]
                    best = _best_row(cands, "waveform_MAE", True, ["waveform_DTW", "waveform_CCC"])
                    if best is not None:
                        rows.append({
                            "dataset": ds_name,
                            "block": block,
                            "rung": rung,
                            "method": best["method"],
                            "family": best["family"],
                            "MAE": np.nan,
                            "RMSE": np.nan,
                            "PearsonR": np.nan,
                            "waveform_CCC": best["waveform_CCC"],
                            "waveform_MAE": best["waveform_MAE"],
                            "waveform_DTW": best["waveform_DTW"],
                        })
            ladder_rows.extend(rows)
    ladder = pd.DataFrame(ladder_rows)
    out_path = out_dir / "T6b_fusion_ladder.csv"
    ladder.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  Fusion ladder saved: {out_path} ({len(ladder)} rows)")
    return ladder


if __name__ == "__main__":
    args = _parse_args()
    explicit = _parse_dataset_metrics_specs(args.dataset_metrics)
    datasets = explicit or {
        "COHFACE": args.cohface_metrics,
        "MAHNOB": args.mahnob_metrics,
    }
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Generating table-ready CSVs...")
    t3 = generate_t3(datasets, out_dir)
    t4 = generate_t4(datasets, out_dir)
    t4_strict = generate_t4_strict(datasets, out_dir)
    t4_cycle = generate_t4_cycle(datasets, out_dir)
    t2 = generate_t2(t3, t4, out_dir)
    t6 = generate_t6(datasets, out_dir)
    fusion = generate_fusion_ladder(datasets, out_dir)
    print("\nDone.")
    if t3 is not None and len(t3) > 0:
        print("\n=== T3 Preview ===")
        print(t3.to_string(index=False))
    if t4 is not None and len(t4) > 0:
        print("\n=== T4 Preview ===")
        print(t4.to_string(index=False))
    if t4_strict is not None and len(t4_strict) > 0:
        print("\n=== T4-strict Preview ===")
        print(t4_strict.to_string(index=False))
    if t4_cycle is not None and len(t4_cycle) > 0:
        print("\n=== T4-cycle Preview ===")
        print(t4_cycle.to_string(index=False))
    if t2 is not None and len(t2) > 0:
        print("\n=== T2 Preview ===")
        print(t2.to_string(index=False))
    if t6 is not None and len(t6) > 0:
        print("\n=== T6 Preview ===")
        print(t6.to_string(index=False))
    if fusion is not None and len(fusion) > 0:
        print("\n=== Fusion Ladder Preview ===")
        print(fusion.to_string(index=False))
