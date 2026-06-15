from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_current_mahnob_bottleneck_audit_classifies_trials(tmp_path: Path) -> None:
    run = tmp_path / "run"
    metrics = run / "metrics"
    metrics.mkdir(parents=True)
    pd.DataFrame(
        {
            "video": ["v_ambiguous", "v_limited"],
            "method": ["m", "m"],
            "MAE": [5.0, 4.5],
            "RMSE": [5.2, 4.7],
            "PearsonR": [0.1, 0.2],
            "Bias": [3.0, -2.0],
            "gt_bpm_avg": [12.0, 12.0],
            "est_bpm_avg": [17.0, 16.5],
        }
    ).to_csv(metrics / "metrics_freq_domain_raw.csv", index=False)
    pd.DataFrame(
        {
            "video": ["v_ambiguous", "v_limited"],
            "source": ["candidate_rate_posterior_calibrated_mean"] * 2,
            "alpha_mean": [0.1, 0.2],
            "posterior_confidence_mean": [0.15, 0.25],
            "posterior_entropy_median": [0.98, 0.80],
            "posterior_top_gap_median": [0.05, 0.40],
        }
    ).to_csv(metrics / "readout_guard_raw.csv", index=False)

    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        {
            "video": ["v_ambiguous", "v_limited"],
            "oracle_method": ["of__kfstd", "dof__kfstd"],
            "oracle_family": ["OF", "DoF"],
            "oracle_kind": ["compat_filter", "compat_filter"],
            "oracle_MAE": [1.5, 3.5],
            "fixed_best_method": ["dof__kfstd", "dof__kfstd"],
            "fixed_best_method_median_MAE": [2.8, 2.8],
            "n_candidates": [24, 24],
        }
    ).to_csv(candidate, index=False)

    failure = tmp_path / "failure.csv"
    pd.DataFrame(
        {
            "video": ["v_ambiguous", "v_limited"],
            "best_aligned_ccc_z": [0.50, 0.50],
            "best_unbounded_ccc_z": [0.60, 0.60],
            "abs_best_unbounded_lag_sec": [5.0, 5.0],
            "peer_agreement_median": [0.30, 0.30],
            "primary_failure_mode": ["mixed_or_partially_observable", "mixed_or_partially_observable"],
        }
    ).to_csv(failure, index=False)

    out_trial = tmp_path / "trial.csv"
    out_summary = tmp_path / "summary.csv"
    out_report = tmp_path / "report.md"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_current_mahnob_bottleneck.py"),
            "--current-run",
            str(run),
            "--candidate-gap-trial",
            str(candidate),
            "--failure-decomp-trial",
            str(failure),
            "--out-trial",
            str(out_trial),
            "--out-summary",
            str(out_summary),
            "--report-out",
            str(out_report),
        ],
        cwd=ROOT,
        check=True,
    )

    trial = pd.read_csv(out_trial)
    classes = dict(zip(trial["video"], trial["bottleneck_class"]))
    assert classes["v_ambiguous"] == "candidate_present_but_source_ambiguous"
    assert classes["v_limited"] == "candidate_bank_or_observability_limited"
    assert "Current MAHNOB Bottleneck Audit" in out_report.read_text(encoding="utf-8")
