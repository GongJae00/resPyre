#!/usr/bin/env python3
"""
Plot instability-benefit boundaries for robust OSSM vs KFSTD.

Inputs (from a completed run):
  - logs/method_quality.csv
  - metrics/metrics_time_domain_raw.csv
  - metrics/metrics_freq_domain_raw.csv
  - metrics/metrics_filter_diagnostics_raw.csv (optional)

Outputs:
  - plots/boundary/boundary_pairs.csv
  - plots/boundary/boundary_deciles.csv
  - plots/boundary/boundary_report.json
  - plots/boundary/boundary_prob_curve.png
  - plots/boundary/boundary_delta_deciles.png
  - plots/boundary/boundary_regime_map.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keep import behavior consistent with other pipeline entry scripts.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from core.pipeline.common import infer_trial_key_from_data_stem


def _safe_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce")


def _derive_trial_key(video: str, data_file: str) -> str:
    data_file = str(data_file or "").strip()
    if data_file:
        stem = os.path.splitext(os.path.basename(data_file))[0]
        dataset_token = stem.split("_", 1)[0] if "_" in stem else None
        key = infer_trial_key_from_data_stem(stem, dataset_token=dataset_token)
        if key:
            return key
    video = str(video or "").strip()
    if "_" in video:
        return video.split("_", 1)[1]
    return video


def _method_parts(name: str) -> Tuple[str, str]:
    m = str(name or "").strip()
    if "__" not in m:
        return m, "base"
    base, variant = m.split("__", 1)
    return base, variant


def _prepare_metrics(run_dir: str) -> pd.DataFrame:
    time_csv = os.path.join(run_dir, "metrics", "metrics_time_domain_raw.csv")
    freq_csv = os.path.join(run_dir, "metrics", "metrics_freq_domain_raw.csv")
    filt_csv = os.path.join(run_dir, "metrics", "metrics_filter_diagnostics_raw.csv")
    if not (os.path.exists(time_csv) and os.path.exists(freq_csv)):
        raise FileNotFoundError("Missing raw time/freq metrics CSVs.")

    tdf = pd.read_csv(time_csv)
    fdf = pd.read_csv(freq_csv)
    ddf = pd.read_csv(filt_csv) if os.path.exists(filt_csv) else pd.DataFrame()

    for df in (tdf, fdf):
        if "method" not in df.columns:
            raise ValueError("Raw metrics CSV must contain 'method' column.")
        if "video" not in df.columns:
            raise ValueError("Raw metrics CSV must contain 'video' column.")
        if "data_file" not in df.columns:
            df["data_file"] = ""
        df["trial_key"] = [
            _derive_trial_key(v, d) for v, d in zip(df["video"], df["data_file"])
        ]

    keep_t = ["method", "trial_key", "video", "data_file", "MAE", "RMSE", "CCC", "DTW_Dist", "SNR_Time"]
    keep_t = [c for c in keep_t if c in tdf.columns]
    tdf = tdf[keep_t].copy()
    tdf = tdf.rename(
        columns={
            "MAE": "time_mae",
            "RMSE": "time_rmse",
            "CCC": "time_ccc",
            "DTW_Dist": "time_dtw",
            "SNR_Time": "time_snr",
        }
    )

    keep_f = ["method", "trial_key", "video", "data_file", "MAE", "RMSE", "PearsonR", "SNR_Spec", "KL_Div"]
    keep_f = [c for c in keep_f if c in fdf.columns]
    fdf = fdf[keep_f].copy()
    fdf = fdf.rename(
        columns={
            "MAE": "freq_mae",
            "RMSE": "freq_rmse",
            "PearsonR": "freq_r",
            "SNR_Spec": "freq_snr",
            "KL_Div": "freq_kl",
        }
    )

    merged = tdf.merge(
        fdf.drop(columns=[c for c in ("video", "data_file") if c in fdf.columns]),
        on=["method", "trial_key"],
        how="inner",
        validate="one_to_one",
    )

    if not ddf.empty:
        if "data_file" not in ddf.columns:
            ddf["data_file"] = ""
        ddf["trial_key"] = [
            _derive_trial_key(v, d) for v, d in zip(ddf.get("video", ""), ddf["data_file"])
        ]
        keep_d = ["method", "trial_key", "Fail_Total", "NIS_Mean", "NIS_Pass", "NIS_OverStrict", "NIS_TrueFail"]
        keep_d = [c for c in keep_d if c in ddf.columns]
        ddf = ddf[keep_d].copy()
        ddf = ddf.rename(
            columns={
                "Fail_Total": "diag_fail_total",
                "NIS_Mean": "diag_nis_mean",
                "NIS_Pass": "diag_nis_pass",
                "NIS_OverStrict": "diag_nis_overstrict",
                "NIS_TrueFail": "diag_nis_truefail",
            }
        )
        merged = merged.merge(
            ddf,
            on=["method", "trial_key"],
            how="left",
            validate="one_to_one",
        )

    return merged


def _prepare_quality(run_dir: str) -> pd.DataFrame:
    q_csv = os.path.join(run_dir, "logs", "method_quality.csv")
    if not os.path.exists(q_csv):
        raise FileNotFoundError("Missing logs/method_quality.csv.")
    qdf = pd.read_csv(q_csv)
    if "method" not in qdf.columns:
        raise ValueError("method_quality.csv must contain 'method'.")
    if "trial" not in qdf.columns:
        raise ValueError("method_quality.csv must contain 'trial'.")
    qdf = qdf.rename(columns={"trial": "trial_key"})
    qdf["base"] = qdf["method"].astype(str).apply(lambda m: _method_parts(m)[0])
    qdf["variant"] = qdf["method"].astype(str).apply(lambda m: _method_parts(m)[1])
    return qdf


def _robust_minmax(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    lo = float(np.nanpercentile(s, 5))
    hi = float(np.nanpercentile(s, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=s.index)
    y = (s - lo) / (hi - lo)
    return y.clip(0.0, 1.0)


def _compute_instability_index(df: pd.DataFrame) -> pd.Series:
    vis_bad = _robust_minmax(1.0 - _safe_float_series(df, "q_vis_mean"))
    drift = _robust_minmax(_safe_float_series(df, "q_drift_mean"))
    out = _robust_minmax(_safe_float_series(df, "q_out_mean"))
    harm = _robust_minmax(_safe_float_series(df, "q_harm_mean"))
    burst = _robust_minmax(_safe_float_series(df, "q_burst_rate"))
    # Emphasize outlier/harmonic/visibility failures.
    u = 0.30 * out + 0.25 * harm + 0.20 * vis_bad + 0.15 * drift + 0.10 * burst
    return pd.to_numeric(u, errors="coerce").clip(0.0, 1.0)


def _build_pair_table(metrics_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    m = metrics_df.copy()
    m["base"] = m["method"].astype(str).apply(lambda x: _method_parts(x)[0])
    m["variant"] = m["method"].astype(str).apply(lambda x: _method_parts(x)[1])

    robust = m[m["variant"].str.contains("robust_ossm", na=False)].copy()
    baseline = m[m["variant"] == "kfstd"].copy()
    if robust.empty or baseline.empty:
        return pd.DataFrame()

    # Join robust metric rows with their KFSTD counterpart (same base, same trial).
    pair = robust.merge(
        baseline[
            [
                "base",
                "trial_key",
                "method",
                "time_mae",
                "freq_mae",
                "diag_fail_total",
                "diag_nis_truefail",
            ]
        ].rename(
            columns={
                "method": "baseline_method",
                "time_mae": "baseline_time_mae",
                "freq_mae": "baseline_freq_mae",
                "diag_fail_total": "baseline_fail_total",
                "diag_nis_truefail": "baseline_nis_truefail",
            }
        ),
        on=["base", "trial_key"],
        how="inner",
        validate="many_to_one",
    )

    # Join quality (only robust rows have these frame-log-based stats).
    q_cols = [
        "method",
        "trial_key",
        "q_vis_mean",
        "q_drift_mean",
        "q_out_mean",
        "q_harm_mean",
        "q_burst_rate",
        "alpha_R_mean",
        "alpha_Q_mean",
        "g_t_mean",
        "g_z_eff_mean",
        "lambda_mean",
    ]
    q_cols = [c for c in q_cols if c in quality_df.columns]
    pair = pair.merge(
        quality_df[q_cols],
        on=["method", "trial_key"],
        how="left",
        validate="many_to_one",
    )
    pair = pair.rename(columns={"method": "robust_method"})
    pair["instability_u"] = _compute_instability_index(pair)

    pair["delta_time_mae"] = pair["time_mae"] - pair["baseline_time_mae"]
    pair["delta_freq_mae"] = pair["freq_mae"] - pair["baseline_freq_mae"]
    pair["delta_fail_total"] = pair.get("diag_fail_total", np.nan) - pair.get("baseline_fail_total", np.nan)
    pair["delta_nis_truefail"] = pair.get("diag_nis_truefail", np.nan) - pair.get("baseline_nis_truefail", np.nan)

    pair["better_time"] = (pair["delta_time_mae"] < 0).astype(float)
    pair["better_freq"] = (pair["delta_freq_mae"] < 0).astype(float)
    pair["better_dual"] = ((pair["delta_time_mae"] < 0) & (pair["delta_freq_mae"] < 0)).astype(float)
    return pair


def _decile_stats(pair_df: pd.DataFrame, deciles: int = 10) -> pd.DataFrame:
    work = pair_df.copy()
    work = work[np.isfinite(work["instability_u"])]
    if work.empty:
        return pd.DataFrame()
    bins = min(max(deciles, 2), int(work["instability_u"].nunique()))
    if bins < 2:
        work["u_bin"] = 0
    else:
        work["u_bin"] = pd.qcut(work["instability_u"], q=bins, labels=False, duplicates="drop")
    agg = (
        work.groupby("u_bin", dropna=True)
        .agg(
            n_pairs=("robust_method", "count"),
            u_mean=("instability_u", "mean"),
            u_min=("instability_u", "min"),
            u_max=("instability_u", "max"),
            p_better_time=("better_time", "mean"),
            p_better_freq=("better_freq", "mean"),
            p_better_dual=("better_dual", "mean"),
            d_time_mae=("delta_time_mae", "mean"),
            d_freq_mae=("delta_freq_mae", "mean"),
            d_fail_total=("delta_fail_total", "mean"),
        )
        .reset_index(drop=True)
    )
    return agg


def _plot_probability_curve(dec_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = dec_df["u_mean"].to_numpy(dtype=np.float64)
    ax.plot(x, dec_df["p_better_time"], marker="o", label="P(better time-MAE)")
    ax.plot(x, dec_df["p_better_freq"], marker="o", label="P(better freq-MAE)")
    ax.plot(x, dec_df["p_better_dual"], marker="o", label="P(better dual)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Instability index U (decile mean)")
    ax.set_ylabel("Probability robust beats KFSTD")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_delta_deciles(dec_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = dec_df["u_mean"].to_numpy(dtype=np.float64)
    ax.plot(x, dec_df["d_time_mae"], marker="o", label="Delta time MAE")
    ax.plot(x, dec_df["d_freq_mae"], marker="o", label="Delta freq MAE")
    if "d_fail_total" in dec_df.columns and dec_df["d_fail_total"].notna().any():
        ax.plot(x, dec_df["d_fail_total"], marker="o", label="Delta fail-total")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Instability index U (decile mean)")
    ax.set_ylabel("Robust - KFSTD (lower is better)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_regime_map(pair_df: pd.DataFrame, out_path: str):
    x = _safe_float_series(pair_df, "q_out_mean")
    y = _safe_float_series(pair_df, "q_harm_mean")
    c = _safe_float_series(pair_df, "delta_time_mae")
    m = pair_df["robust_method"].astype(str).str.contains("_ukf").map({True: "s", False: "o"})

    fig, ax = plt.subplots(figsize=(7, 5))
    for marker in ("o", "s"):
        mask = m == marker
        if not np.any(mask):
            continue
        sc = ax.scatter(
            x[mask], y[mask],
            c=c[mask],
            cmap="coolwarm_r",
            vmin=np.nanpercentile(c, 5) if np.isfinite(c).any() else -1.0,
            vmax=np.nanpercentile(c, 95) if np.isfinite(c).any() else 1.0,
            alpha=0.85,
            s=45,
            marker=marker,
            label="EKF" if marker == "o" else "UKF",
        )
    if np.isfinite(c).any():
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("Delta time MAE (robust - KFSTD)")
    ax.set_xlabel("q_out_mean")
    ax.set_ylabel("q_harm_mean")
    ax.set_title("Regime map (color: robust gain in time MAE)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory (e.g., results/cohface_robust_ossm)")
    ap.add_argument("--out-dir", default=None, help="Output directory for boundary artifacts")
    ap.add_argument("--deciles", type=int, default=10, help="Number of quantile bins")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "plots", "boundary")
    os.makedirs(out_dir, exist_ok=True)

    metrics_df = _prepare_metrics(run_dir)
    quality_df = _prepare_quality(run_dir)
    pair_df = _build_pair_table(metrics_df, quality_df)
    if pair_df.empty:
        report = {
            "schema": "instability_boundary.v1",
            "status": "empty",
            "reason": "No robust-vs-kfstd pairs available.",
            "n_pairs": 0,
        }
        with open(os.path.join(out_dir, "boundary_report.json"), "w", encoding="utf-8") as fp:
            json.dump(report, fp, ensure_ascii=False, indent=2)
        # Write empty placeholders.
        pd.DataFrame().to_csv(os.path.join(out_dir, "boundary_pairs.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, "boundary_deciles.csv"), index=False)
        print(f"[Saved] {os.path.join(out_dir, 'boundary_report.json')} (empty)")
        return

    dec_df = _decile_stats(pair_df, deciles=max(2, int(args.deciles)))

    pair_csv = os.path.join(out_dir, "boundary_pairs.csv")
    dec_csv = os.path.join(out_dir, "boundary_deciles.csv")
    pair_df.to_csv(pair_csv, index=False)
    dec_df.to_csv(dec_csv, index=False)

    # Estimate first decile where robust dual-win probability exceeds 0.5.
    threshold_u = np.nan
    if not dec_df.empty and "p_better_dual" in dec_df.columns:
        mask = (dec_df["p_better_dual"] >= 0.5) & (dec_df["n_pairs"] >= 3)
        if mask.any():
            threshold_u = float(dec_df.loc[mask, "u_min"].iloc[0])

    report = {
        "schema": "instability_boundary.v1",
        "status": "ok",
        "n_pairs": int(len(pair_df)),
        "n_deciles": int(len(dec_df)),
        "threshold_u_p_better_dual_ge_0p5": threshold_u,
        "mean_delta_time_mae": float(np.nanmean(pair_df["delta_time_mae"])),
        "mean_delta_freq_mae": float(np.nanmean(pair_df["delta_freq_mae"])),
        "mean_p_better_time": float(np.nanmean(pair_df["better_time"])),
        "mean_p_better_freq": float(np.nanmean(pair_df["better_freq"])),
        "mean_p_better_dual": float(np.nanmean(pair_df["better_dual"])),
    }
    with open(os.path.join(out_dir, "boundary_report.json"), "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    if not dec_df.empty:
        _plot_probability_curve(dec_df, os.path.join(out_dir, "boundary_prob_curve.png"))
        _plot_delta_deciles(dec_df, os.path.join(out_dir, "boundary_delta_deciles.png"))
    _plot_regime_map(pair_df, os.path.join(out_dir, "boundary_regime_map.png"))

    print(f"[Saved] {pair_csv}")
    print(f"[Saved] {dec_csv}")
    print(f"[Saved] {os.path.join(out_dir, 'boundary_report.json')}")


if __name__ == "__main__":
    main()
