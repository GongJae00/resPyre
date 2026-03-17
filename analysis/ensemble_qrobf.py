"""
Multi-family QROBF Ensemble with quality-aware family selection.

Key insight: raw SNR_Spec has family-level baseline bias (P1D ~9.7dB, OF ~7.6dB),
causing hard_select to almost always pick P1D even when OF is better.
Solution: family-normalized composite quality score.

Selection strategies compared:
  hard_select_snr       — absolute SNR_Spec (baseline, biased)
  hard_select_norm_snr  — family-normalized SNR_Spec z-score
  hard_select_composite — normalized: SNR_Spec + FreqStd_Mean(neg) + Fail_Total(neg)
  hard_select_threshold — norm SNR z-score with margin threshold θ (LOO-CV tuned)
                          if margin(z_best - z_2nd) > θ: pick best-z family
                          else: fallback to P1D-Quadratic (best default)
  soft_mean_norm_b1     — softmax(normalized SNR, β=1) weighted est_bpm_avg
  oracle                — best MAE per trial (upper bound)

All results use windowed-MAE for hard_select/oracle (from eval pipeline).
Soft mean uses single-point |est_avg - gt_avg| (noted separately).

Usage:
    python analysis/ensemble_qrobf.py --run-dir results/cohface_robust_ossm
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import zscore

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

QROBF_METHODS = [
    "of_farneback__robust_ossm_ekf",
    "profile1d_linear__robust_ossm_ekf",
    "profile1d_quadratic__robust_ossm_ekf",
    "profile1d_cubic__robust_ossm_ekf",
]
KFSTD_METHODS = [
    "of_farneback__kfstd",
    "profile1d_linear__kfstd",
    "profile1d_quadratic__kfstd",
    "profile1d_cubic__kfstd",
]
BASE_METHODS = [
    "of_farneback", "profile1D linear", "profile1D quadratic", "profile1D cubic",
]


# ── Feature engineering ───────────────────────────────────────────────────────

def _family_normalize(pivot: pd.DataFrame, methods: list) -> pd.DataFrame:
    """
    Z-score each method column independently (within-family normalization).
    Returns DataFrame of same shape with z-scores.
    """
    out = pivot[methods].copy()
    for m in methods:
        col = out[m].dropna()
        mu, sigma = col.mean(), col.std()
        out[m] = (out[m] - mu) / (sigma + 1e-9)
    return out


def _build_feature_matrix(
    p_snr: pd.DataFrame,
    p_fail: pd.DataFrame,
    p_freq_std: pd.DataFrame,
    p_nis_inband: pd.DataFrame,
    methods: list,
) -> dict:
    """
    For each trial, build a per-method feature vector.
    Returns dict: method -> (n_trials,) normalized feature vector (composite score).
    """
    # Family-normalized features
    z_snr      = _family_normalize(p_snr,      methods)
    z_fail     = _family_normalize(p_fail,      methods)   # lower = better → negate
    z_freq_std = _family_normalize(p_freq_std,  methods)   # lower = better → negate
    z_inband   = _family_normalize(p_nis_inband, methods)  # higher = better

    composite = {}
    for m in methods:
        # Positive contributors: SNR, NIS in-band
        # Negative contributors: fail rate, frequency variability
        score = (
            z_snr[m]
            - z_fail[m]
            - z_freq_std[m]
            + z_inband[m]
        )
        composite[m] = score
    return composite


def _hard_select(score_dict: dict, methods: list, videos: list) -> pd.Series:
    """Pick the method with the highest score per trial."""
    score_df = pd.DataFrame({m: score_dict[m] for m in methods if m in score_dict},
                            index=videos)
    return score_df.idxmax(axis=1)


def _loocv_selection(
    composite: dict,
    p_mae: pd.DataFrame,
    methods: list,
    videos: list,
) -> tuple[pd.Series, float]:
    """
    Leave-one-out logistic regression: for each held-out trial, train a
    per-method linear scorer on the composite features, predict best family.
    Returns (selections Series, mean leave-one-out accuracy).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return None, float("nan")

    n = len(videos)
    # Build feature array: (n_trials, n_methods) — composite score per method
    score_arr = np.column_stack(
        [composite[m].loc[videos].values if isinstance(composite[m], pd.Series)
         else composite[m]
         for m in methods]
    )  # shape (n, 4)

    # Oracle labels (which method is best per trial)
    mae_arr = p_mae.loc[videos, methods].values  # (n, 4)
    oracle_labels = np.array([methods[int(np.nanargmin(mae_arr[i]))] for i in range(n)])

    predictions = []
    correct = 0
    for i in range(n):
        # Leave out trial i
        X_train = np.delete(score_arr, i, axis=0)
        y_train = np.delete(oracle_labels, i)
        X_test  = score_arr[i:i+1]

        # Need at least 2 classes in training
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            pred = methods[int(np.nanargmax(score_arr[i]))]
        else:
            scaler = StandardScaler().fit(X_train)
            clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                     multi_class="multinomial")
            try:
                clf.fit(scaler.transform(X_train), y_train)
                pred = clf.predict(scaler.transform(X_test))[0]
            except Exception:
                pred = methods[int(np.nanargmax(score_arr[i]))]

        predictions.append(pred)
        if pred == oracle_labels[i]:
            correct += 1

    accuracy = correct / n
    return pd.Series(predictions, index=videos), accuracy


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _compute_strategy_mae(
    selections: pd.Series,
    p_mae: pd.DataFrame,
) -> pd.Series:
    """Given per-trial method selections, return the per-trial windowed MAE."""
    maes = []
    for video, method in selections.items():
        if method in p_mae.columns:
            maes.append(p_mae.loc[video, method])
        else:
            maes.append(float("nan"))
    return pd.Series(maes, index=selections.index)


def _strategy_summary(mae_series: pd.Series, label: str) -> dict:
    v = mae_series.dropna()
    return {
        "strategy": label,
        "freq_mae_median": round(float(np.nanmedian(v)), 4),
        "freq_mae_mean":   round(float(np.nanmean(v)),   4),
        "freq_rmse":       round(float(np.sqrt(np.nanmean(v**2))), 4),
        "n": len(v),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run(run_dir: str, output_csv: str) -> pd.DataFrame:
    freq_raw  = Path(run_dir) / "metrics" / "metrics_freq_domain_raw.csv"
    diag_raw  = Path(run_dir) / "metrics" / "metrics_filter_diagnostics_raw.csv"
    if not freq_raw.exists():
        print(f"[ensemble] freq_domain_raw.csv not found"); return pd.DataFrame()

    freq_df = pd.read_csv(freq_raw)
    diag_df = pd.read_csv(diag_raw) if diag_raw.exists() else pd.DataFrame()

    # ── Pivot tables ──────────────────────────────────────────────────────────
    all_methods = QROBF_METHODS + KFSTD_METHODS + BASE_METHODS
    freq_sub = freq_df[freq_df["method"].isin(all_methods)]
    p_mae  = freq_sub.pivot(index="video", columns="method", values="MAE")
    p_snr  = freq_sub.pivot(index="video", columns="method", values="SNR_Spec")
    p_est  = freq_sub.pivot(index="video", columns="method", values="est_bpm_avg")
    p_gt   = freq_sub.pivot(index="video", columns="method", values="gt_bpm_avg")

    gt_bpm = p_gt[QROBF_METHODS].mean(axis=1)
    videos = p_mae.index.tolist()
    n = len(videos)
    print(f"[ensemble] {n} trials × {len(QROBF_METHODS)} QROBF families")

    # Filter diagnostics
    p_fail = p_freq_std = p_nis_inband = None
    if not diag_df.empty:
        diag_sub = diag_df[diag_df["method"].isin(QROBF_METHODS)]
        p_fail      = diag_sub.pivot(index="video", columns="method", values="Fail_Total")
        p_freq_std  = diag_sub.pivot(index="video", columns="method", values="FreqStd_Mean")
        p_nis_inband = diag_sub.pivot(index="video", columns="method", values="NIS_InBand")
        # align index
        for pv in [p_fail, p_freq_std, p_nis_inband]:
            pv = pv.reindex(videos)

    # ── Strategy 1: hard_select_snr (absolute, baseline) ─────────────────────
    sel_snr_abs = p_snr[QROBF_METHODS].idxmax(axis=1)
    mae_snr_abs = _compute_strategy_mae(sel_snr_abs, p_mae)

    # ── Strategy 2: hard_select_norm_snr (family-normalized) ─────────────────
    z_snr = _family_normalize(p_snr, QROBF_METHODS)
    sel_snr_norm = z_snr.idxmax(axis=1)
    mae_snr_norm = _compute_strategy_mae(sel_snr_norm, p_mae)

    # ── Strategy 3: hard_select_composite ────────────────────────────────────
    composite = None
    mae_composite = pd.Series([float("nan")] * n, index=videos)
    if p_fail is not None:
        p_fail_r      = p_fail.reindex(videos)
        p_freq_std_r  = p_freq_std.reindex(videos)
        p_nis_inband_r = p_nis_inband.reindex(videos)
        composite = _build_feature_matrix(
            p_snr.reindex(videos), p_fail_r, p_freq_std_r, p_nis_inband_r, QROBF_METHODS
        )
        sel_composite = _hard_select(composite, QROBF_METHODS, videos)
        mae_composite = _compute_strategy_mae(sel_composite, p_mae)

    # ── Strategy 4: Threshold-based selection (LOO-CV tuned θ) ───────────────
    # For each held-out trial: find best θ on n-1 trials, apply to held-out.
    # Fallback: P1D-Quadratic (best individual family).
    FALLBACK = "profile1d_quadratic__robust_ossm_ekf"
    theta_grid = np.arange(0.0, 2.1, 0.1)

    def _threshold_strategy(z_snr_sub, p_mae_sub, theta, fallback):
        """Apply threshold strategy on a given z_snr pivot and mae pivot."""
        maes = []
        sels = []
        for v in z_snr_sub.index:
            scores = {m: float(z_snr_sub.loc[v, m])
                      for m in QROBF_METHODS
                      if m in z_snr_sub.columns and np.isfinite(z_snr_sub.loc[v, m])}
            if not scores:
                maes.append(float("nan")); sels.append(fallback); continue
            best_m = max(scores, key=scores.get)
            sorted_s = sorted(scores.values(), reverse=True)
            margin = sorted_s[0] - sorted_s[1] if len(sorted_s) > 1 else 99.0
            sel = best_m if margin > theta else fallback
            sels.append(sel)
            maes.append(float(p_mae_sub.loc[v, sel]) if sel in p_mae_sub.columns else float("nan"))
        return pd.Series(maes, index=z_snr_sub.index), pd.Series(sels, index=z_snr_sub.index)

    # LOO-CV to get unbiased threshold and MAE estimates
    loo_maes = []
    loo_sels = []
    for i, v_test in enumerate(videos):
        v_train = [v for j, v in enumerate(videos) if j != i]
        snr_tr = p_snr.reindex(v_train)[QROBF_METHODS]
        mu_tr  = snr_tr.mean(); std_tr = snr_tr.std()
        z_tr   = (snr_tr - mu_tr) / (std_tr + 1e-9)
        # Best θ on training set
        best_theta_loo, best_mae_train = 0.0, 999.0
        p_mae_tr = p_mae.reindex(v_train)
        for theta in theta_grid:
            m_vec, _ = _threshold_strategy(z_tr, p_mae_tr, theta, FALLBACK)
            med = float(np.nanmedian(m_vec))
            if med < best_mae_train:
                best_mae_train = med; best_theta_loo = theta
        # Apply to held-out
        z_test = (p_snr.loc[[v_test]][QROBF_METHODS] - mu_tr) / (std_tr + 1e-9)
        _, sels_test = _threshold_strategy(z_test, p_mae.loc[[v_test]], best_theta_loo, FALLBACK)
        sel = sels_test.iloc[0]
        loo_sels.append(sel)
        loo_maes.append(float(p_mae.loc[v_test, sel]) if sel in p_mae.columns else float("nan"))

    mae_loocv_thresh = pd.Series(loo_maes, index=videos)
    sel_loocv_thresh = pd.Series(loo_sels, index=videos)

    # ── Strategy 5: soft_mean (norm SNR, single-pt metric) ───────────────────
    soft_sp_mae = []
    for video in videos:
        avail = [m for m in QROBF_METHODS if np.isfinite(z_snr.loc[video, m])]
        if not avail:
            soft_sp_mae.append(float("nan")); continue
        snrs = np.array([z_snr.loc[video, m] for m in avail])
        ests = np.array([p_est.loc[video, m] for m in avail])
        w = softmax(snrs)
        ens = float(np.dot(w, ests))
        soft_sp_mae.append(abs(ens - float(gt_bpm.loc[video])))
    mae_soft_norm = pd.Series(soft_sp_mae, index=videos)

    # ── Oracle ────────────────────────────────────────────────────────────────
    mae_oracle = p_mae[QROBF_METHODS].min(axis=1)

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = [
        _strategy_summary(p_mae[m].reindex(videos), f"individual {m}")
        for m in QROBF_METHODS + KFSTD_METHODS + ["profile1D quadratic", "profile1D cubic"]
    ] + [
        _strategy_summary(mae_snr_abs,     "hard_select_snr_abs (biased)"),
        _strategy_summary(mae_snr_norm,    "hard_select_snr_norm"),
        _strategy_summary(mae_composite,        "hard_select_composite"),
        _strategy_summary(mae_loocv_thresh,    "hard_select_threshold (LOO-CV)"),
        _strategy_summary(mae_soft_norm,   "soft_mean_norm (single-pt metric)"),
        _strategy_summary(mae_oracle,      "oracle (upper bound)"),
    ]
    summary = pd.DataFrame(rows).sort_values("freq_mae_median").reset_index(drop=True)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("MULTI-FAMILY QROBF ENSEMBLE — Freq MAE (BPM), n=160")
    print("  hard_select/oracle: windowed MAE  |  soft_mean: single-pt |est_avg-gt_avg|")
    print("=" * 82)
    SEP = {"hard_select_snr_abs (biased)", "hard_select_snr_norm",
           "hard_select_composite", "hard_select_threshold (LOO-CV)",
           "oracle (upper bound)", "soft_mean_norm (single-pt metric)"}
    for _, r in summary.iterrows():
        tag = " ★" if r["strategy"] in SEP else ""
        print(f"  {r['strategy']:<45s}  median={r['freq_mae_median']:.4f}  "
              f"mean={r['freq_mae_mean']:.4f}  rmse={r['freq_rmse']:.4f}{tag}")
    print("=" * 82)

    # ── Selection accuracy (vs oracle) ────────────────────────────────────────
    oracle_labels = p_mae[QROBF_METHODS].reindex(videos).idxmin(axis=1)
    print("\n[Selection accuracy vs oracle]")
    for label, sel in [
        ("hard_select_snr_abs ", sel_snr_abs),
        ("hard_select_snr_norm", sel_snr_norm),
    ]:
        acc = (sel == oracle_labels).mean()
        print(f"  {label}: {acc*100:.1f}%")
    if composite is not None:
        acc_comp = (sel_composite == oracle_labels).mean()
        print(f"  hard_select_composite  : {acc_comp*100:.1f}%")
    acc_thresh = (sel_loocv_thresh == oracle_labels).mean()
    print(f"  hard_select_threshold  : {acc_thresh*100:.1f}% (LOO-CV)")

    print("\n[Oracle: best family per trial]")
    print(oracle_labels.value_counts().to_string())

    print("\n[hard_select_composite picks]")
    if composite is not None:
        print(sel_composite.value_counts().to_string())

    print("\n[hard_select_threshold (LOO-CV) picks]")
    print(sel_loocv_thresh.value_counts().to_string())

    # ── Per-trial CSV ─────────────────────────────────────────────────────────
    out_df = pd.DataFrame({
        "video": videos,
        "gt_bpm": gt_bpm.values,
        "oracle_family": oracle_labels.values,
        "sel_snr_abs": sel_snr_abs.values,
        "sel_snr_norm": sel_snr_norm.values,
        "sel_composite": sel_composite.values if composite is not None else [None]*n,
        "mae_snr_abs": mae_snr_abs.values,
        "mae_snr_norm": mae_snr_norm.values,
        "mae_composite": mae_composite.values,
        "mae_threshold_loocv": mae_loocv_thresh.values,
        "sel_threshold_loocv": sel_loocv_thresh.values,
        "mae_oracle": mae_oracle.values,
    })
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    summary.to_csv(output_csv.replace(".csv", "_summary.csv"), index=False)
    print(f"\n[ensemble] Saved → {output_csv}")

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = str(Path(args.run_dir).resolve())
    if not os.path.isdir(run_dir):
        print(f"ERROR: {run_dir}"); sys.exit(1)
    output_csv = os.path.join(run_dir, "paper", "tables", "ensemble_qrobf.csv")
    run(run_dir, output_csv)


if __name__ == "__main__":
    main()
