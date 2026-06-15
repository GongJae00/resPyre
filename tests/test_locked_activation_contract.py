from pathlib import Path

import pytest

from scripts.materialize_calibrated_multifamily_parh_system import (
    _load_windowed_reliability_priors,
    _require_locked_activation,
)


def _release_cfg():
    return {
        "enable_observation_law": True,
        "enable_rate_posterior": True,
        "rate_posterior_output_public_source": "final",
        "parh_input": "multichannel",
        "enable_target_observability_control": True,
    }


def test_windowed_reliability_loader_accepts_start_sec_aliases(tmp_path: Path) -> None:
    csv = tmp_path / "windowed.csv"
    csv.write_text(
        "video,group,start_sec,end_sec,soft_group_weight,z_osc_readout_weight,z_full_readout_weight\n"
        "trial_a,G_OF,0,8,0.70,0.80,0.40\n",
        encoding="utf-8",
    )
    rows = _load_windowed_reliability_priors(csv)
    assert rows["trial_a"][0]["start_sec"] == 0.0
    assert rows["trial_a"][0]["end_sec"] == 8.0
    assert rows["trial_a"][0]["z_osc_readout_weight"] == 0.80


def test_locked_activation_contract_fails_closed_when_windowed_inputs_missing() -> None:
    with pytest.raises(RuntimeError, match="windowed readout reliability"):
        _require_locked_activation(
            video="trial_a",
            cfg=_release_cfg(),
            state_window_rows=[{"group": "G_OF", "start_sec": 0.0, "end_sec": 8.0, "score": 0.8}],
            readout_window_rows=[],
            rate_posterior_meta={"enabled": True},
            output_rate_meta={"enabled": True},
            target_observability_runtime={"target_observability": [1.0]},
            target_observability_meta={"enabled": True},
        )


def test_locked_activation_contract_accepts_active_final_path() -> None:
    _require_locked_activation(
        video="trial_a",
        cfg=_release_cfg(),
        state_window_rows=[{"group": "G_OF", "start_sec": 0.0, "end_sec": 8.0, "score": 0.8}],
        readout_window_rows=[{"group": "G_OF", "start_sec": 0.0, "end_sec": 8.0, "score": 0.8}],
        rate_posterior_meta={"enabled": True},
        output_rate_meta={"enabled": True},
        target_observability_runtime={"target_observability": [1.0]},
        target_observability_meta={"enabled": True},
    )


def test_locked_activation_contract_fails_when_state_role_rows_missing() -> None:
    with pytest.raises(RuntimeError, match="windowed state-role reliability rows"):
        _require_locked_activation(
            video="trial_a",
            cfg=_release_cfg(),
            state_window_rows=[],
            readout_window_rows=[{"group": "G_OF", "start_sec": 0.0, "end_sec": 8.0, "score": 0.8}],
            rate_posterior_meta={"enabled": True},
            output_rate_meta={"enabled": True},
            target_observability_runtime={"target_observability": [1.0]},
            target_observability_meta={"enabled": True},
        )


def test_locked_activation_contract_allows_readout_only_scope_without_state_rows() -> None:
    cfg = {**_release_cfg(), "reliability_prior_scope": "readout_only"}
    _require_locked_activation(
        video="trial_a",
        cfg=cfg,
        state_window_rows=[],
        readout_window_rows=[{"group": "G_OF", "start_sec": 0.0, "end_sec": 8.0, "score": 0.8}],
        rate_posterior_meta={"enabled": True},
        output_rate_meta={"enabled": True},
        target_observability_runtime={"target_observability": [1.0]},
        target_observability_meta={"enabled": True},
    )
