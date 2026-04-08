#!/usr/bin/env python3
"""Aggregate PARH-OSSM mechanism diagnostics from saved trial PKLs."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.models.heads.parh_ossm import oscillator_PARH_OSSM


def classify_family(method_name: str) -> str:
    m = str(method_name).lower().replace(" ", "_")
    base = m.split("__", 1)[0] if "__" in m else m
    family_map = {
        "of_farneback": "OF",
        "of_disp_bridge": "OF_bridge",
        "of": "OF",
        "dof": "DoF",
        "profile1d_linear": "P1D_lin",
        "profile1d_quadratic": "P1D_quad",
        "profile1d_cubic": "P1D_cub",
    }
    return family_map.get(base, base)


def finite_stats(arr):
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p05": np.nan,
            "p95": np.nan,
        }
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
    }


def safe_array(d, key):
    if not isinstance(d, dict):
        return np.array([], dtype=np.float64)
    try:
        arr = np.asarray(d.get(key, []), dtype=np.float64)
    except Exception:
        return np.array([], dtype=np.float64)
    return arr.reshape(-1)


def energy_ratio(num, den):
    den = np.asarray(den, dtype=np.float64)
    num = np.asarray(num, dtype=np.float64)
    denom = float(np.mean(np.square(den))) if den.size else np.nan
    numer = float(np.mean(np.square(num))) if num.size else np.nan
    if not np.isfinite(denom) or denom <= 1e-12:
        return np.nan
    return numer / denom


def safe_meta_float(d, key):
    if not isinstance(d, dict):
        return np.nan
    try:
        val = float(d.get(key, np.nan))
    except Exception:
        return np.nan
    return val if np.isfinite(val) else np.nan


def parse_trial(pkl_path: Path):
    with pkl_path.open("rb") as fp:
        obj = pickle.load(fp)
    rows = []
    for est in obj.get("estimates", []):
        method = est.get("method")
        if "__parh_ossm" not in str(method):
            continue
        payload = est.get("estimate", est)
        if not isinstance(payload, dict):
            continue
        diag = payload.get("diagnostics", {})
        decomp = payload.get("decomposition", {})
        meta = payload.get("meta", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        obs_meta = meta.get("observation_calibration", {}) if isinstance(meta.get("observation_calibration", {}), dict) else {}
        z_osc = safe_array(payload, "z_osc")
        z_full = safe_array(payload, "z_full")
        z_osc_causal = safe_array(payload, "z_osc_causal")
        z_osc_smoothed = safe_array(payload, "z_osc_smoothed")

        q_obs = safe_array(diag, "q_obs_t")
        q_dyn_raw = safe_array(diag, "q_dyn_raw_t")
        q_dyn = safe_array(diag, "q_dyn_t")
        q_osc = safe_array(diag, "q_osc_t")
        obs_osc_support_t = safe_array(diag, "obs_osc_support_t")
        obs_full_support_t = safe_array(diag, "obs_full_support_t")
        obs_nonosc_need_t = safe_array(diag, "obs_nonosc_need_t")
        obs_nonosc_need_eff_t = safe_array(diag, "obs_nonosc_need_eff_t")
        residual_prior_t = safe_array(diag, "residual_prior_t")
        aper_drive_t = safe_array(diag, "aper_drive_t")
        pi_t = safe_array(diag, "pi_t")
        lambda_t = safe_array(diag, "lambda_t")
        nu_t = safe_array(diag, "nu_t")
        R_t = safe_array(diag, "R_t")
        helper_trust_t = safe_array(diag, "helper_trust_t")
        helper_bias_conf_t = safe_array(diag, "helper_bias_conf_t")
        helper_mismatch_t = safe_array(diag, "helper_mismatch_t")
        freq_rescue_t = safe_array(diag, "freq_rescue_t")
        output_rate_blend_t = safe_array(diag, "output_rate_blend_t")

        baseline = safe_array(decomp, "baseline")
        residual = safe_array(decomp, "residual")

        q_dyn_raw_stats = finite_stats(q_dyn_raw)
        q_dyn_stats = finite_stats(q_dyn)
        q_osc_stats = finite_stats(q_osc)

        row = {
            "video": pkl_path.stem,
            "method": method,
            "family": classify_family(method),
            "q_obs_mean": finite_stats(q_obs)["mean"],
            "q_obs_median": finite_stats(q_obs)["median"],
            "q_dyn_raw_mean": q_dyn_raw_stats["mean"],
            "q_dyn_raw_median": q_dyn_raw_stats["median"],
            "q_dyn_mean": q_dyn_stats["mean"],
            "q_dyn_median": q_dyn_stats["median"],
            "q_osc_mean": q_osc_stats["mean"],
            "q_osc_median": q_osc_stats["median"],
            "obs_osc_support_mean": finite_stats(obs_osc_support_t)["mean"],
            "obs_osc_support_median": finite_stats(obs_osc_support_t)["median"],
            "obs_full_support_mean": finite_stats(obs_full_support_t)["mean"],
            "obs_full_support_median": finite_stats(obs_full_support_t)["median"],
            "obs_nonosc_need_mean": finite_stats(obs_nonosc_need_t)["mean"],
            "obs_nonosc_need_median": finite_stats(obs_nonosc_need_t)["median"],
            "obs_nonosc_need_eff_mean": finite_stats(obs_nonosc_need_eff_t)["mean"],
            "obs_nonosc_need_eff_median": finite_stats(obs_nonosc_need_eff_t)["median"],
            "residual_prior_mean": finite_stats(residual_prior_t)["mean"],
            "residual_prior_median": finite_stats(residual_prior_t)["median"],
            "aper_drive_mean": finite_stats(aper_drive_t)["mean"],
            "aper_drive_median": finite_stats(aper_drive_t)["median"],
            "helper_trust_mean": finite_stats(helper_trust_t)["mean"],
            "helper_trust_median": finite_stats(helper_trust_t)["median"],
            "helper_bias_conf_mean": finite_stats(helper_bias_conf_t)["mean"],
            "helper_bias_conf_median": finite_stats(helper_bias_conf_t)["median"],
            "pi_mean": finite_stats(pi_t)["mean"],
            "pi_median": finite_stats(pi_t)["median"],
            "lambda_mean": finite_stats(lambda_t)["mean"],
            "lambda_median": finite_stats(lambda_t)["median"],
            "lambda_lt095_frac": float(np.mean(lambda_t < 0.95)) if lambda_t.size else np.nan,
            "nu_mean": finite_stats(nu_t)["mean"],
            "nu_median": finite_stats(nu_t)["median"],
            "R_mean": finite_stats(R_t)["mean"],
            "R_median": finite_stats(R_t)["median"],
            "helper_mismatch_mean": finite_stats(helper_mismatch_t)["mean"],
            "helper_mismatch_median": finite_stats(helper_mismatch_t)["median"],
            "freq_rescue_active_frac": float(np.mean(freq_rescue_t > 0.5)) if freq_rescue_t.size else np.nan,
            "output_rate_blend_active_frac": float(np.mean(output_rate_blend_t > 0.5)) if output_rate_blend_t.size else np.nan,
            "obs_fit_corr": safe_meta_float(obs_meta, "fit_corr"),
            "obs_fit_rmse": safe_meta_float(obs_meta, "fit_rmse"),
            "obs_lag_sec": safe_meta_float(obs_meta, "lag_sec"),
            "obs_g_h1": safe_meta_float(obs_meta, "g_h1"),
            "obs_g_h2": safe_meta_float(obs_meta, "g_h2"),
            "obs_g_b": safe_meta_float(obs_meta, "g_b"),
            "obs_g_r": safe_meta_float(obs_meta, "g_r"),
            "Qosc_scale_mean": (1.0 + oscillator_PARH_OSSM.Q_DYN_GAMMA * q_dyn_stats["mean"])
            if np.isfinite(q_dyn_stats["mean"]) else np.nan,
            "Qaper_scale_mean": (1.0 + oscillator_PARH_OSSM.Q_APER_GAMMA * (1.0 - q_osc_stats["mean"]))
            if np.isfinite(q_osc_stats["mean"]) else np.nan,
            "baseline_energy_ratio": energy_ratio(baseline, z_full),
            "residual_energy_ratio": energy_ratio(residual, z_full),
            "max_abs_zfull_minus_zosc": float(np.max(np.abs(z_full - z_osc))) if z_full.size and z_osc.size else np.nan,
            "max_abs_zosc_causal_minus_smoothed": float(np.max(np.abs(z_osc_causal - z_osc_smoothed)))
            if z_osc_causal.size and z_osc_smoothed.size else np.nan,
        }
        rows.append(row)
    return rows


def aggregate_family(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg_cols = [c for c in df.columns if c not in ("video", "method", "family")]
    out = []
    for family, g in df.groupby("family", dropna=False):
        row = {"family": family, "trial_count": int(g["video"].nunique())}
        for col in agg_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if not vals.empty else np.nan
            row[f"{col}_median"] = float(vals.median()) if not vals.empty else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values("family").reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate PARH-OSSM diagnostics from result PKLs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to a results data directory containing trial PKLs",
    )
    parser.add_argument(
        "--trial-out",
        type=Path,
        default=ROOT / "paper" / "manifests" / "parh_mechanism_audit_trials.csv",
        help="Output CSV for per-trial mechanism diagnostics",
    )
    parser.add_argument(
        "--family-out",
        type=Path,
        default=ROOT / "paper" / "tables_ready" / "T6b_mechanism_audit.csv",
        help="Output CSV for family-level aggregated mechanism diagnostics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pkl_files = sorted(args.data_dir.glob("*.pkl"))
    rows = []
    for pkl_path in pkl_files:
        rows.extend(parse_trial(pkl_path))

    trial_df = pd.DataFrame(rows)
    family_df = aggregate_family(trial_df)

    args.trial_out.parent.mkdir(parents=True, exist_ok=True)
    args.family_out.parent.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(args.trial_out, index=False)
    family_df.to_csv(args.family_out, index=False)

    print(f"Saved trial diagnostics: {args.trial_out}")
    print(f"Saved family summary: {args.family_out}")
    print(f"Trials: {len(trial_df)}")
    if not family_df.empty:
        print(family_df.to_string(index=False))


if __name__ == "__main__":
    main()
