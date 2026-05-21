from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_posterior_feature_sharpness_audit_reports_saturated_features(tmp_path: Path) -> None:
    run = tmp_path / "run"
    metrics = run / "metrics"
    metrics.mkdir(parents=True)

    pd.DataFrame(
        {
            "video": ["v_good", "v_mid", "v_bad"],
            "method": ["m", "m", "m"],
            "MAE": [1.0, 3.0, 6.0],
            "RMSE": [1.2, 3.2, 6.4],
            "PearsonR": [0.9, 0.4, -0.1],
            "Bias": [0.2, 1.0, 4.0],
            "gt_bpm_avg": [12.0, 12.0, 12.0],
            "est_bpm_avg": [12.2, 15.0, 18.0],
        }
    ).to_csv(metrics / "metrics_freq_domain_raw.csv", index=False)

    pd.DataFrame(
        {
            "video": ["v_good", "v_mid", "v_bad"],
            "alpha_mean": [0.2, 0.2, 0.2],
            "posterior_confidence_mean": [0.9, 0.5, 0.1],
            "posterior_top_gap_median": [0.7, 0.3, 0.05],
            "posterior_macro_support_median": [1.0, 1.0, 1.0],
            "posterior_direct_macro_support_median": [0.8, 0.4, 0.1],
            "posterior_motion_direct_support_median": [0.7, 0.3, 0.05],
            "posterior_alias_risk_median": [0.1, 0.4, 0.8],
        }
    ).to_csv(metrics / "readout_guard_raw.csv", index=False)

    candidate = tmp_path / "candidate.csv"
    pd.DataFrame(
        {
            "video": ["v_good", "v_mid", "v_bad"],
            "oracle_MAE": [0.8, 1.5, 2.0],
            "oracle_method": ["of", "p1d", "dof"],
            "oracle_family": ["OF", "P1D", "DoF"],
            "oracle_kind": ["base", "base", "base"],
        }
    ).to_csv(candidate, index=False)

    bottleneck = tmp_path / "bottleneck.csv"
    pd.DataFrame(
        {
            "video": ["v_good", "v_mid", "v_bad"],
            "candidate_room_bpm": [0.2, 1.5, 4.0],
            "candidate_can_solve": [True, True, True],
            "posterior_ambiguous": [False, True, True],
            "hard_current_failure": [False, False, True],
            "bottleneck_class": ["solved", "ambiguous", "ambiguous"],
            "primary_failure_mode": ["none", "alias", "alias"],
        }
    ).to_csv(bottleneck, index=False)

    out_csv = tmp_path / "sharpness.csv"
    report = tmp_path / "sharpness.md"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "audit_posterior_feature_sharpness.py"),
            "--run",
            str(run),
            "--name",
            "synthetic",
            "--candidate-gap-trial",
            str(candidate),
            "--bottleneck-trial",
            str(bottleneck),
            "--out-csv",
            str(out_csv),
            "--report-out",
            str(report),
        ],
        cwd=ROOT,
        check=True,
    )

    feature_csv = out_csv.with_name(out_csv.stem + "_features.csv")
    group_csv = out_csv.with_name(out_csv.stem + "_groups.csv")
    features = pd.read_csv(feature_csv)

    macro = features.loc[features["feature"] == "posterior_macro_support_median"].iloc[0]
    assert bool(macro["low_variance"])
    assert float(macro["saturated_fraction"]) == 1.0
    assert out_csv.exists()
    assert group_csv.exists()
    text = report.read_text(encoding="utf-8")
    assert "Posterior Feature Sharpness Audit" in text
    assert "Do not add a stricter guard around a saturated feature" in text
