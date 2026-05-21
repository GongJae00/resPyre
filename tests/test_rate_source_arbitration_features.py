import json
import pickle

import pandas as pd

from scripts.audit_rate_source_arbitration_features import main


def test_rate_source_arbitration_feature_audit_joins_pivot(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    meta = {
        "rate_posterior_output_source": "calibrated_mean",
        "calibration": {
            "anchor": "profile1d_quadratic",
            "anchor_family": "G_P1D_morph",
            "family_weights": {"G_P1D_morph": 0.7, "G_DoF": 0.3},
            "family_raw_scores": {"G_P1D_morph": 0.8, "G_DoF": 0.2},
        },
        "decoupled_rate_readout_meta": {
            "source": "candidate_rate_posterior_calibrated_mean",
            "confidence_mean": 0.4,
            "alias_risk_mean": 0.1,
        },
        "rate_candidate_posterior_meta": {
            "confidence_mean": 0.2,
            "posterior_entropy_median": 0.9,
            "posterior_top_gap_median": 0.15,
        },
        "parh_ossm_diagnostics": {"freq_mean": 0.25, "mixture_entropy_mean": 0.8},
    }
    payload = {
        "estimates": [
            {
                "method": "parh",
                "estimate": {"meta": json.dumps(meta), "track_hz": [0.2, 0.3]},
            }
        ]
    }
    with (data_dir / "trial_a.pkl").open("wb") as f:
        pickle.dump(payload, f)

    pivot = tmp_path / "pivot.csv"
    pd.DataFrame(
        [
            {
                "video": "trial_a",
                "best_source": "external_rate_posterior_mean_t",
                "final_minus_best": 1.0,
                "final_track_hz": 3.0,
            }
        ]
    ).to_csv(pivot, index=False)
    out_csv = tmp_path / "features.csv"
    out_md = tmp_path / "features.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_rate_source_arbitration_features.py",
            "--data-dir",
            str(data_dir),
            "--pivot-csv",
            str(pivot),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ],
    )

    assert main() == 0
    out = pd.read_csv(out_csv)
    assert out.loc[0, "anchor_family"] == "G_P1D_morph"
    assert out.loc[0, "posterior_entropy_median"] == 0.9
    assert out.loc[0, "family_weight_G_P1D_morph"] == 0.7
    assert "diagnostic" in out_md.read_text(encoding="utf-8")
