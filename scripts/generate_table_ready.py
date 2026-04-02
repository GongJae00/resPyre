#!/usr/bin/env python3
"""Generate table-ready CSVs for T3/T4/T6 from production run metrics.

Usage:
    python scripts/generate_table_ready.py
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "paper" / "tables_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "COHFACE": ROOT / "results" / "cohface_parh_ossm_prod" / "cohface_parh_ossm_prod" / "metrics",
    "MAHNOB": ROOT / "results" / "mahnob_parh_ossm_prod" / "mahnob_parh_ossm_prod" / "metrics",
}

# Method variant classification
def classify_method(method_name):
    """Return (family, variant) tuple."""
    m = method_name.lower().replace(" ", "_")
    if "__parh_ossm" in m:
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
        "of": "OF",
        "dof": "DoF",
        "profile1d_linear": "P1D_lin",
        "profile1d_quadratic": "P1D_quad",
        "profile1d_cubic": "P1D_cub",
    }
    family = family_map.get(family, family)
    return family, variant


def generate_t3():
    """T3: Rate accuracy (MAE, RMSE, PearsonR) — median across trials per dataset×family."""
    rows = []
    for ds_name, metrics_dir in DATASETS.items():
        csv_path = metrics_dir / "metrics_freq_domain_raw.csv"
        if not csv_path.exists():
            print(f"  SKIP T3 {ds_name}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df[["family", "variant"]] = df["method"].apply(
            lambda m: pd.Series(classify_method(m))
        )
        for family in sorted(df["family"].unique()):
            fdf = df[df["family"] == family]
            row = {"dataset": ds_name, "family": family}
            for variant in ["Base", "KFstd", "PARH"]:
                vdf = fdf[fdf["variant"] == variant]
                if vdf.empty:
                    for col in ["MAE", "RMSE", "PearsonR"]:
                        row[f"{variant}_{col}"] = np.nan
                else:
                    for col in ["MAE", "RMSE", "PearsonR"]:
                        row[f"{variant}_{col}"] = vdf[col].median()
                row[f"{variant}_N"] = len(vdf)
            rows.append(row)
    t3 = pd.DataFrame(rows)
    out_path = OUT_DIR / "T3_rate_main.csv"
    t3.to_csv(out_path, index=False, float_format="%.3f")
    print(f"  T3 saved: {out_path} ({len(t3)} rows)")
    return t3


def generate_t4():
    """T4: Waveform fidelity (CCC, wMAE, DTW) — unified comparison.
    Main: Base(signal_hat) / KFstd(signal_hat) / PARH(z_full) smoothed only.
    """
    rows = []
    for ds_name, metrics_dir in DATASETS.items():
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

        for family in sorted(df_smooth["family"].unique()):
            fdf = df_smooth[df_smooth["family"] == family]
            row = {"dataset": ds_name, "family": family}
            for variant, otype in [("Base", "signal_hat"), ("KFstd", "signal_hat"), ("PARH", "z_full")]:
                vdf = fdf[(fdf["variant"] == variant) & (fdf["output_type"] == otype)]
                for col in ["waveform_CCC", "waveform_MAE", "waveform_DTW"]:
                    short = col.replace("waveform_", "")
                    row[f"{variant}_{short}"] = vdf[col].median() if not vdf.empty else np.nan
                row[f"{variant}_N"] = len(vdf)
            rows.append(row)
    t4 = pd.DataFrame(rows)
    out_path = OUT_DIR / "T4_waveform_main.csv"
    t4.to_csv(out_path, index=False, float_format="%.3f")
    print(f"  T4 saved: {out_path} ({len(t4)} rows)")
    return t4


def generate_t6():
    """T6: Filter diagnostics — per dataset×family (KFstd and PARH only)."""
    diag_cols = [
        "NIS_Mean", "NIS_InBand", "Lambda_Mean", "Lambda_LT1_Frac",
        "Coverage95", "Stability_Sec",
    ]
    rows = []
    for ds_name, metrics_dir in DATASETS.items():
        csv_path = metrics_dir / "metrics_filter_diagnostics_raw.csv"
        if not csv_path.exists():
            print(f"  SKIP T6 {ds_name}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        df[["family", "variant"]] = df["method"].apply(
            lambda m: pd.Series(classify_method(m))
        )
        # Only KFstd and PARH have diagnostics
        for family in sorted(df["family"].unique()):
            fdf = df[df["family"] == family]
            row = {"dataset": ds_name, "family": family}
            for variant in ["KFstd", "PARH"]:
                vdf = fdf[fdf["variant"] == variant]
                for col in diag_cols:
                    val = vdf[col].median() if col in vdf.columns and not vdf.empty else np.nan
                    row[f"{variant}_{col}"] = val
                row[f"{variant}_N"] = len(vdf)
            rows.append(row)
    t6 = pd.DataFrame(rows)
    out_path = OUT_DIR / "T6_diagnostics_main.csv"
    t6.to_csv(out_path, index=False, float_format="%.4f")
    print(f"  T6 saved: {out_path} ({len(t6)} rows)")
    return t6


if __name__ == "__main__":
    print("Generating table-ready CSVs...")
    t3 = generate_t3()
    t4 = generate_t4()
    t6 = generate_t6()
    print("\nDone.")
    if t3 is not None and len(t3) > 0:
        print("\n=== T3 Preview ===")
        print(t3.to_string(index=False))
    if t4 is not None and len(t4) > 0:
        print("\n=== T4 Preview ===")
        print(t4.to_string(index=False))
    if t6 is not None and len(t6) > 0:
        print("\n=== T6 Preview ===")
        print(t6.to_string(index=False))
