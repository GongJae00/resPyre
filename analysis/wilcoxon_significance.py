"""
Wilcoxon Signed-Rank Test for QROBF vs kfstd per-trial comparison.

Usage:
    python analysis/wilcoxon_significance.py --run-dir results/cohface_robust_ossm

Outputs:
    - Console table of p-values and effect sizes per method pair
    - CSV saved to <run-dir>/paper/tables/wilcoxon_significance.csv

Statistical note:
    n=160 paired trials (40 subjects × 4 trials each).
    Two-sided Wilcoxon signed-rank test (non-parametric, no normality assumption).
    Bonferroni correction applied for multiple comparisons.
    Effect size: rank-biserial correlation r = 1 - 2W/(n*(n+1)/2).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ── Method pairs to compare (QROBF vs kfstd for each observation family) ──
METHOD_PAIRS = [
    # (qrobf_name, kfstd_name, observation_family_label)
    ("of_farneback__robust_ossm_ekf",    "of_farneback__kfstd",       "OF-Farneback"),
    ("profile1d_linear__robust_ossm_ekf", "profile1d_linear__kfstd",   "Profile1D-Linear"),
    ("profile1d_quadratic__robust_ossm_ekf", "profile1d_quadratic__kfstd", "Profile1D-Quadratic"),
    ("profile1d_cubic__robust_ossm_ekf",  "profile1d_cubic__kfstd",    "Profile1D-Cubic"),
]

# Metrics to test (smaller = better for all)
METRICS = ["freq_mae", "freq_rmse", "time_mae", "time_rmse"]


def load_per_trial_metrics(run_dir: str) -> pd.DataFrame:
    """Load per-trial metric CSV from <run-dir>/metrics/."""
    candidates = [
        os.path.join(run_dir, "metrics", "metrics_per_trial.csv"),
        os.path.join(run_dir, "metrics", "per_trial_metrics.csv"),
        os.path.join(run_dir, "metrics", "trial_metrics.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"[wilcoxon] Loaded per-trial metrics from: {path}")
            return df
    # Fallback: search recursively for any CSV with 'trial' in the name
    for root, _, files in os.walk(os.path.join(run_dir, "metrics")):
        for f in files:
            if "trial" in f.lower() and f.endswith(".csv"):
                path = os.path.join(root, f)
                df = pd.read_csv(path)
                print(f"[wilcoxon] Loaded per-trial metrics from: {path}")
                return df
    raise FileNotFoundError(
        f"No per-trial metric CSV found in {run_dir}/metrics/. "
        "Run main.py first to generate metrics."
    )


def rank_biserial(stat_W: float, n: int) -> float:
    """Rank-biserial correlation from Wilcoxon W statistic."""
    max_W = n * (n + 1) / 2.0
    if max_W <= 0:
        return 0.0
    return float(1.0 - 2.0 * stat_W / max_W)


def run_wilcoxon(df: pd.DataFrame, output_csv: str) -> pd.DataFrame:
    """Run paired Wilcoxon tests for all method pairs and metrics."""
    results = []
    n_tests = len(METHOD_PAIRS) * len(METRICS)
    bonferroni_alpha = 0.05 / n_tests  # Bonferroni-corrected threshold

    # Detect available column names
    method_col = None
    for c in ["method", "method_name", "head", "estimator"]:
        if c in df.columns:
            method_col = c
            break
    if method_col is None:
        raise ValueError(f"No method column found. Available: {df.columns.tolist()}")

    trial_col = None
    for c in ["trial_uid", "trial_key", "trial", "trial_id"]:
        if c in df.columns:
            trial_col = c
            break

    available_methods = df[method_col].unique().tolist()
    print(f"[wilcoxon] Available methods: {available_methods}")

    for qrobf_name, kfstd_name, family_label in METHOD_PAIRS:
        # Find matching method names (may have prefix variations)
        def find_method(target):
            if target in available_methods:
                return target
            for m in available_methods:
                if target in m or m in target:
                    return m
            return None

        qrobf_key = find_method(qrobf_name)
        kfstd_key = find_method(kfstd_name)

        if qrobf_key is None or kfstd_key is None:
            print(f"[wilcoxon] SKIP {family_label}: "
                  f"{'QROBF not found' if qrobf_key is None else ''} "
                  f"{'kfstd not found' if kfstd_key is None else ''}")
            continue

        df_q = df[df[method_col] == qrobf_key].copy()
        df_k = df[df[method_col] == kfstd_key].copy()

        for metric in METRICS:
            # Find matching metric column
            metric_col = None
            for c in df.columns:
                if metric.lower() in c.lower():
                    metric_col = c
                    break
            if metric_col is None:
                continue

            # Align trials if trial column exists
            if trial_col and trial_col in df.columns:
                df_q_m = df_q.set_index(trial_col)[metric_col].dropna()
                df_k_m = df_k.set_index(trial_col)[metric_col].dropna()
                common_trials = df_q_m.index.intersection(df_k_m.index)
                if len(common_trials) < 8:
                    continue
                vals_q = df_q_m.loc[common_trials].values
                vals_k = df_k_m.loc[common_trials].values
            else:
                # No trial key — assume same ordering
                vals_q = df_q[metric_col].dropna().values
                vals_k = df_k[metric_col].dropna().values
                n_common = min(len(vals_q), len(vals_k))
                if n_common < 8:
                    continue
                vals_q = vals_q[:n_common]
                vals_k = vals_k[:n_common]

            n = len(vals_q)
            diff = vals_k - vals_q  # positive = QROBF is better (lower error)

            # Wilcoxon signed-rank test
            try:
                stat, pval = wilcoxon(diff, alternative='two-sided',
                                      zero_method='wilcox')
            except ValueError:
                # All differences zero
                stat, pval = 0.0, 1.0

            r_effect = rank_biserial(float(stat), n)
            mean_diff = float(np.mean(diff))
            median_diff = float(np.median(diff))

            significant = pval < bonferroni_alpha
            qrobf_better_pct = float(np.mean(diff > 0)) * 100.0

            results.append({
                "family": family_label,
                "metric": metric,
                "n_trials": n,
                "mean_diff (kfstd-QROBF)": round(mean_diff, 4),
                "median_diff (kfstd-QROBF)": round(median_diff, 4),
                "W_stat": round(float(stat), 1),
                "p_value": round(pval, 4),
                "p_bonferroni": round(bonferroni_alpha, 5),
                "significant": significant,
                "effect_r": round(r_effect, 3),
                "QROBF_better_pct": round(qrobf_better_pct, 1),
            })

    if not results:
        print("[wilcoxon] WARNING: No test results produced. Check method names and metric columns.")
        return pd.DataFrame()

    out_df = pd.DataFrame(results)

    # Print summary table
    print("\n" + "=" * 90)
    print("WILCOXON SIGNED-RANK TEST: QROBF vs kfstd")
    print(f"  n_tests={n_tests}, Bonferroni α={bonferroni_alpha:.5f}")
    print("=" * 90)
    print(out_df.to_string(index=False))
    print("=" * 90)

    sig_rows = out_df[out_df["significant"] == True]
    if len(sig_rows) > 0:
        print(f"\n[wilcoxon] {len(sig_rows)}/{len(out_df)} tests significant after Bonferroni correction.")
    else:
        print(f"\n[wilcoxon] No tests significant after Bonferroni correction (n_tests={n_tests}).")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"[wilcoxon] Saved to: {output_csv}")

    return out_df


def main():
    parser = argparse.ArgumentParser(
        description="Wilcoxon signed-rank test: QROBF vs kfstd per-trial comparison"
    )
    parser.add_argument("--run-dir", required=True,
                        help="Path to run directory (e.g. results/cohface_robust_ossm)")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"ERROR: run-dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    df = load_per_trial_metrics(run_dir)
    output_csv = os.path.join(run_dir, "paper", "tables", "wilcoxon_significance.csv")
    run_wilcoxon(df, output_csv)


if __name__ == "__main__":
    main()
