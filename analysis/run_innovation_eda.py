#!/usr/bin/env python3
"""
Frame-log innovation EDA for QROBF.

Uses robust_ossm frame logs (v_t, S_t, nis, lambda_t, fail_*) to produce:
  - innovation_summary.csv/json
  - innovation_raw.csv
  - per-method diagnostic plots
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt

from core.pipeline.common import (
    resolve_target_run_dirs,
    collect_expected_method_trials,
    resolve_frame_logs_for_run,
)


def _safe_mean(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _tail_metrics(x: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {
        "kurtosis": float("nan"),
        "t_aic_delta": float("nan"),
        "t_fit_nu": float("nan"),
        "hill_alpha": float("nan"),
        "shapiro_pval": float("nan"),
    }
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return out

    out["kurtosis"] = float(sps.kurtosis(x))

    try:
        t_params = sps.t.fit(x)
        t_ll = float(np.sum(sps.t.logpdf(x, *t_params)))
        t_aic = 2 * 3 - 2 * t_ll
        g_params = sps.norm.fit(x)
        g_ll = float(np.sum(sps.norm.logpdf(x, *g_params)))
        g_aic = 2 * 2 - 2 * g_ll
        out["t_aic_delta"] = float(g_aic - t_aic)
        out["t_fit_nu"] = float(t_params[0])
    except Exception:
        pass

    try:
        abs_sorted = np.sort(np.abs(x))[::-1]
        k = max(10, int(0.1 * abs_sorted.size))
        k = min(k, abs_sorted.size - 1)
        if k > 1 and abs_sorted[k] > 0:
            log_ratios = np.log(abs_sorted[:k] / abs_sorted[k])
            denom = float(np.sum(log_ratios))
            out["hill_alpha"] = float(k / denom) if denom > 0 else float("inf")
    except Exception:
        pass

    try:
        _, p = sps.shapiro(x[: min(5000, x.size)])
        out["shapiro_pval"] = float(p)
    except Exception:
        pass

    return out


def _diag_plot(x: np.ndarray, title: str, out_path: str):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(x, bins=60, density=True, alpha=0.7, color="#457b9d")
    axes[0].set_title("Innovation Histogram")
    sps.probplot(x, dist="norm", plot=axes[1])
    axes[1].set_title("QQ Plot vs Gaussian")
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


def _load_innovation_from_log(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    fields = list(z["fields"])
    data = z["data"]
    idx = {f: i for i, f in enumerate(fields)}

    def col(name: str, default=np.nan):
        if name not in idx:
            return np.full(data.shape[0], default, dtype=np.float64)
        return np.asarray(data[:, idx[name]], dtype=np.float64)

    v = col("v_t")
    S = col("S_t")
    valid = np.isfinite(v) & np.isfinite(S) & (S > 1e-12)
    if np.any(valid):
        innov = v[valid] / np.sqrt(S[valid])
    else:
        vv = v[np.isfinite(v)]
        innov = (vv - np.mean(vv)) / (np.std(vv) + 1e-9) if vv.size else np.array([], dtype=np.float64)

    out = {
        "innov": innov.astype(np.float64),
        "nis": col("nis"),
        "lambda_t": col("lambda_t"),
        "fail_div": col("fail_diverge", 0.0),
        "fail_slip": col("fail_slip", 0.0),
        "fail_lock": col("fail_lock", 0.0),
        "fail_double": col("fail_double", 0.0),
    }
    return out


def _write_skipped_logs_artifact(eda_dir: str, skipped: List[Dict[str, str]]) -> str:
    payload = {
        "schema": "innovation_eda_skipped_logs.v1",
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "skipped_count": int(len(skipped)),
        "skipped": skipped,
    }
    out_path = os.path.join(eda_dir, "innovation_eda_skipped_logs.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return out_path


def run_innovation_eda(
    results_dir: str,
    run_label: str | None = None,
    allow_missing: bool = False,
):
    target_dirs = resolve_target_run_dirs(results_dir, run_label)
    if not target_dirs:
        print(f"[EDA] No run directories found for '{run_label}' in '{results_dir}'")
        return

    for run_dir in target_dirs:
        eda_dir = os.path.join(run_dir, "eda")
        plots_dir = os.path.join(eda_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        expected = collect_expected_method_trials(run_dir)
        selected, orphans, resolver_diag = resolve_frame_logs_for_run(
            run_dir,
            expected_trials=expected,
            strict=True,
        )
        logs = []
        for method, trial_map in sorted(selected.items()):
            for trial, path in sorted(trial_map.items()):
                logs.append((method, trial, path))

        raw_rows = []
        by_method: Dict[str, List[np.ndarray]] = {}
        skipped_logs: List[Dict[str, str]] = []
        n_expected = int(len(expected))

        # Expected-but-missing logs are explicit skips.
        for item in expected:
            method = str(item.get("method", ""))
            trial = str(item.get("trial", ""))
            if method not in selected or trial not in selected.get(method, {}):
                skipped_logs.append({
                    "method": method,
                    "trial": trial,
                    "path": "",
                    "exception_type": "MissingFrameLog",
                    "short_message": "No canonical frame log selected for expected trial",
                })

        # Orphans are explicit ignores; they never affect EDA aggregate.
        for o in orphans:
            skipped_logs.append({
                "method": str(o.get("method", "")),
                "trial": str(o.get("trial_key", "")),
                "path": str(o.get("path", "")),
                "exception_type": "OrphanIgnored",
                "short_message": str(o.get("reason", "orphan_ignored")),
            })

        for method, trial, path in logs:
            try:
                obj = _load_innovation_from_log(path)
            except Exception as exc:
                skipped_logs.append({
                    "method": str(method),
                    "trial": str(trial),
                    "path": str(path),
                    "exception_type": type(exc).__name__,
                    "short_message": str(exc),
                })
                continue

            innov = obj["innov"]
            nis = obj["nis"]
            lam = obj["lambda_t"]
            fail_total = np.maximum.reduce([obj["fail_div"], obj["fail_slip"], obj["fail_lock"], obj["fail_double"]])

            tm = _tail_metrics(innov)
            raw_rows.append({
                "method": method,
                "trial": trial,
                "n_frames": int(len(innov)),
                "kurtosis": tm["kurtosis"],
                "t_aic_delta": tm["t_aic_delta"],
                "t_fit_nu": tm["t_fit_nu"],
                "hill_alpha": tm["hill_alpha"],
                "shapiro_pval": tm["shapiro_pval"],
                "nis_mean": float(np.nanmean(nis)) if np.isfinite(nis).any() else float("nan"),
                "lambda_mean": float(np.nanmean(lam)) if np.isfinite(lam).any() else float("nan"),
                "lambda_lt1_frac": float(np.nanmean(lam < 1.0)) if np.isfinite(lam).any() else float("nan"),
                "lambda_low_frac": float(np.nanmean(lam < 0.5)) if np.isfinite(lam).any() else float("nan"),
                "fail_total": float(np.nanmean(fail_total)) if np.isfinite(fail_total).any() else float("nan"),
            })

            by_method.setdefault(method, []).append(innov)
        _write_skipped_logs_artifact(eda_dir, skipped_logs)

        raw_df = pd.DataFrame(raw_rows)
        raw_df.to_csv(os.path.join(eda_dir, "innovation_raw.csv"), index=False)

        summary_rows = []
        for method in sorted(by_method.keys()):
            chunks = by_method.get(method, [])
            conc = np.concatenate([c for c in chunks if c.size > 0]) if chunks else np.array([], dtype=np.float64)
            tm = _tail_metrics(conc)
            subset = raw_df[raw_df["method"] == method]
            summary_rows.append({
                "method": method,
                "kurtosis": tm["kurtosis"],
                "t_aic_delta": tm["t_aic_delta"],
                "t_fit_nu": tm["t_fit_nu"],
                "hill_alpha": tm["hill_alpha"],
                "shapiro_pval": tm["shapiro_pval"],
                "nis_mean": _safe_mean(subset["nis_mean"].tolist()),
                "lambda_mean": _safe_mean(subset["lambda_mean"].tolist()),
                "lambda_lt1_frac": _safe_mean(subset["lambda_lt1_frac"].tolist()),
                "lambda_low_frac": _safe_mean(subset["lambda_low_frac"].tolist()),
                "fail_total": _safe_mean(subset["fail_total"].tolist()),
                "student_t_justified": bool(np.isfinite(tm["t_aic_delta"]) and tm["t_aic_delta"] > 10.0 and np.isfinite(tm["kurtosis"]) and tm["kurtosis"] > 1.0),
                "status": "ok",
                "n_expected": n_expected,
                "n_valid": int(len(raw_rows)),
                "n_skipped": int(len(skipped_logs)),
            })

            if conc.size > 20:
                _diag_plot(conc, f"Innovation EDA — {method}", os.path.join(plots_dir, f"{method}.png"))

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).sort_values(by="method").reset_index(drop=True)
            status = "ok" if len(skipped_logs) == 0 else "partial_with_skips"
        else:
            summary_df = pd.DataFrame([{
                "method": "",
                "kurtosis": np.nan,
                "t_aic_delta": np.nan,
                "t_fit_nu": np.nan,
                "hill_alpha": np.nan,
                "shapiro_pval": np.nan,
                "nis_mean": np.nan,
                "lambda_mean": np.nan,
                "lambda_lt1_frac": np.nan,
                "lambda_low_frac": np.nan,
                "fail_total": np.nan,
                "student_t_justified": False,
                "status": "no_valid_logs",
                "n_expected": n_expected,
                "n_valid": 0,
                "n_skipped": int(len(skipped_logs)),
            }])
            status = "no_valid_logs"
        summary_df.to_csv(os.path.join(eda_dir, "innovation_summary.csv"), index=False)
        with open(os.path.join(eda_dir, "innovation_summary.json"), "w", encoding="utf-8") as fp:
            json.dump({
                "schema": "innovation_summary.v2",
                "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
                "status": status,
                "skipped_count": int(len(skipped_logs)),
                "allow_missing": bool(allow_missing),
                "n_expected": n_expected,
                "n_valid": int(len(raw_rows)),
                "selection_policy": resolver_diag.get("selection_policy"),
                "run_instance_started_at_used": resolver_diag.get("run_instance_started_at_used"),
                "warnings": resolver_diag.get("warnings", []),
                "methods": summary_df.to_dict(orient="records"),
            }, fp, ensure_ascii=False, indent=2)
        print(f"[EDA] Saved innovation EDA to {eda_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--run-label", default=None)
    ap.add_argument("--allow-missing", action="store_true",
                    help="Allow unreadable/missing logs and record them in innovation_eda_skipped_logs.json")
    args = ap.parse_args()
    run_innovation_eda(args.results_dir, args.run_label, allow_missing=bool(args.allow_missing))


if __name__ == "__main__":
    main()
