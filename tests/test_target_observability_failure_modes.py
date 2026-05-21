import pandas as pd

from scripts.audit_target_observability_failure_modes import main


def test_target_observability_audit_labels_oracle_room(tmp_path, monkeypatch):
    decomp = tmp_path / "decomp.csv"
    pd.DataFrame(
        [
            {"video": "trial_a", "rate_source": "final_track_hz", "MAE": 4.0, "track_hz_median": 0.20},
            {"video": "trial_a", "rate_source": "external_rate_posterior_mean_t", "MAE": 2.0, "track_hz_median": 0.22},
            {"video": "trial_a", "rate_source": "native_smoothed_track_hz", "MAE": 3.5, "track_hz_median": 0.21},
        ]
    ).to_csv(decomp, index=False)
    features = tmp_path / "features.csv"
    pd.DataFrame(
        [
            {
                "video": "trial_a",
                "external_output_rate_confidence_mean": 0.45,
                "external_rate_posterior_confidence_mean": 0.55,
                "external_rate_posterior_entropy_median": 0.80,
                "external_rate_posterior_top_gap_median": 0.30,
                "readout_support_group_count_median": 6,
                "readout_alias_risk_mean": 0.02,
                "posterior_alias_risk_median": 0.01,
                "readout_h1_role_support_mean": 0.80,
                "readout_abstain_pressure_mean": 0.05,
            }
        ]
    ).to_csv(features, index=False)
    out_csv = tmp_path / "obs.csv"
    out_md = tmp_path / "obs.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_target_observability_failure_modes.py",
            "--decomposition-csv",
            str(decomp),
            "--feature-csv",
            str(features),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ],
    )

    assert main() == 0
    out = pd.read_csv(out_csv)
    assert out.loc[0, "oracle_best_source"] == "external_rate_posterior_mean_t"
    assert out.loc[0, "oracle_room_bpm"] == 2.0
    assert out.loc[0, "failure_mode"] == "source_selection_room_posterior_available"
    assert "diagnostic-only" in out_md.read_text(encoding="utf-8")


def test_target_observability_audit_labels_reference_limited(tmp_path, monkeypatch):
    decomp = tmp_path / "decomp.csv"
    pd.DataFrame(
        [
            {"video": "trial_b", "rate_source": "final_track_hz", "MAE": 4.0, "track_hz_median": 0.20},
            {"video": "trial_b", "rate_source": "external_output_rate_t", "MAE": 3.9, "track_hz_median": 0.21},
            {"video": "trial_b", "rate_source": "state_freq_t", "MAE": 3.8, "track_hz_median": 0.22},
        ]
    ).to_csv(decomp, index=False)
    out_csv = tmp_path / "obs.csv"
    out_md = tmp_path / "obs.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_target_observability_failure_modes.py",
            "--decomposition-csv",
            str(decomp),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ],
    )

    assert main() == 0
    out = pd.read_csv(out_csv)
    assert out.loc[0, "failure_mode"] == "likely_video_or_reference_limited"
