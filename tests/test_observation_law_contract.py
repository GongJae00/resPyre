import json

import numpy as np

from components.models.heads.parh_ossm import oscillator_PARH_OSSM


def test_observation_law_is_multichannel_adaptive_law_not_fallback() -> None:
    fs = 20.0
    n = 240
    t = np.arange(n, dtype=float) / fs
    stack = np.vstack(
        [
            np.sin(2.0 * np.pi * 0.22 * t),
            np.sin(2.0 * np.pi * 0.22 * t + 0.10),
            0.7 * np.sin(2.0 * np.pi * 0.44 * t),
        ]
    )

    head = oscillator_PARH_OSSM()
    head.ENABLE_DYNAMIC_MIXTURE = True
    head.ENABLE_RATE_OBSERVABILITY_MIXTURE = True
    head.ENABLE_RATE_OBSERVABILITY_HELPER = True
    head.ENABLE_RESIDUAL_SEMANTICS = True

    result = head.run(
        stack,
        fs,
        {
            "method_name": "observation_law_contract_smoke",
            "base_method": "of_disp_bridge",
            "observation_law_enabled": True,
            "observation_families": [
                "of_disp_bridge",
                "profile1d_quadratic",
                "profile1d_consensus",
            ],
            "observation_rate_tracks_runtime": {
                "of_disp_bridge": [0.22] * n,
                "profile1d_quadratic": [0.22] * n,
                "profile1d_consensus": [0.22] * n,
            },
        },
    )

    meta = json.loads(result["meta"])
    assert "observation_law" in meta["active_modules"]
    assert "dynamic_soft_observation_mixture" in meta["active_modules"]
    assert meta["observation_law"]["nested_comparator"] == ""
    assert meta["observation_law"]["safety_mode"] == "adaptive_R_eff_and_pi_prior_trust"

    diagnostics = result["diagnostics"]
    assert diagnostics["prior_collapse_t"].shape == (n,)
    assert diagnostics["mixture_entropy_t"].shape == (n,)
    assert diagnostics["pi_t_channels"].shape == stack.shape
    assert diagnostics["mixture_t_channels"].shape == stack.shape
    assert np.all(np.isfinite(diagnostics["prior_collapse_t"]))
    assert np.nanmean(diagnostics["pi_t_channels"]) > 0.2


def test_observation_law_state_role_prior_changes_update_trust() -> None:
    fs = 20.0
    n = 240
    t = np.arange(n, dtype=float) / fs
    shared = np.sin(2.0 * np.pi * 0.22 * t)
    stack = np.vstack([shared, shared, shared])

    role = {
        "h1": np.vstack([
            np.full(n, 1.0),
            np.full(n, 0.15),
            np.full(n, 1.0),
        ]),
        "z_osc": np.vstack([
            np.full(n, 1.0),
            np.full(n, 0.10),
            np.full(n, 1.0),
        ]),
        "z_full": np.ones((3, n), dtype=float),
        "abstain": np.vstack([
            np.zeros(n),
            np.zeros(n),
            np.ones(n),
        ]),
    }

    head = oscillator_PARH_OSSM()
    head.ENABLE_DYNAMIC_MIXTURE = True
    head.ENABLE_RATE_OBSERVABILITY_MIXTURE = True
    head.ENABLE_RATE_OBSERVABILITY_HELPER = True
    head.ENABLE_RESIDUAL_SEMANTICS = True
    head.STATE_ROLE_CONTEXT_POWER = 1.0
    head.STATE_ROLE_RATE_POWER = 1.0
    head.STATE_ROLE_CONTEXT_MULTIPLIER_FLOOR = 0.0
    head.STATE_ROLE_RATE_MULTIPLIER_FLOOR = 0.0
    head.STATE_ROLE_ABSTAIN_R_SCALE = 4.0

    result = head.run(
        stack,
        fs,
        {
            "method_name": "observation_law_state_role_contract",
            "base_method": "of_disp_bridge",
            "observation_law_enabled": True,
            "observation_families": [
                "trusted_rate_channel",
                "weak_rate_channel",
                "abstain_channel",
            ],
            "observation_rate_tracks_runtime": {
                "trusted_rate_channel": [0.22] * n,
                "weak_rate_channel": [0.22] * n,
                "abstain_channel": [0.22] * n,
            },
            "external_state_role_prior_runtime": role,
        },
    )

    meta = json.loads(result["meta"])
    assert meta["dynamic_observation_mixture"]["state_role_prior_runtime_applied"]

    diagnostics = result["diagnostics"]
    for key in (
        "R_eff_t_channels",
        "state_role_context_t_channels",
        "state_role_abstain_t_channels",
        "state_role_h1_t_channels",
        "state_role_zosc_t_channels",
    ):
        assert diagnostics[key].shape == stack.shape
        assert np.all(np.isfinite(diagnostics[key]))

    warm = slice(n // 3, None)
    h1_mean = np.nanmean(diagnostics["state_role_h1_t_channels"][:, warm], axis=1)
    zosc_mean = np.nanmean(diagnostics["state_role_zosc_t_channels"][:, warm], axis=1)
    context_mean = np.nanmean(diagnostics["state_role_context_t_channels"][:, warm], axis=1)
    pi_mean = np.nanmean(diagnostics["pi_t_channels"][:, warm], axis=1)
    r_eff_mean = np.nanmean(diagnostics["R_eff_t_channels"][:, warm], axis=1)

    assert h1_mean[0] > h1_mean[1]
    assert zosc_mean[0] > zosc_mean[1]
    assert context_mean[0] > context_mean[2]
    assert pi_mean[0] > pi_mean[1]
    assert r_eff_mean[2] > r_eff_mean[0]


def test_target_observability_control_gates_update_before_readout() -> None:
    fs = 20.0
    n = 240
    t = np.arange(n, dtype=float) / fs
    stack = np.vstack(
        [
            np.sin(2.0 * np.pi * 0.22 * t),
            np.sin(2.0 * np.pi * 0.22 * t + 0.15),
        ]
    )
    target = np.r_[np.ones(n // 2), np.full(n - n // 2, 0.20)]
    runtime = {
        "target_observability": target,
        "h1_timing": target,
        "h2_morphology": np.ones(n),
        "baseline": np.ones(n),
        "residual": np.ones(n),
        "nuisance": np.r_[np.zeros(n // 2), np.full(n - n // 2, 0.80)],
    }

    head = oscillator_PARH_OSSM()
    head.ENABLE_DYNAMIC_MIXTURE = True
    head.ENABLE_RATE_OBSERVABILITY_MIXTURE = True
    head.ENABLE_RATE_OBSERVABILITY_HELPER = True
    head.ENABLE_RESIDUAL_SEMANTICS = True

    result = head.run(
        stack,
        fs,
        {
            "method_name": "target_observability_control_contract",
            "base_method": "of_disp_bridge",
            "observation_law_enabled": True,
            "observation_families": ["of_disp_bridge", "profile1d_quadratic"],
            "observation_rate_tracks_runtime": {
                "of_disp_bridge": [0.22] * n,
                "profile1d_quadratic": [0.22] * n,
            },
            "external_target_observability_runtime": runtime,
            "enable_target_observability_control_runtime": True,
        },
    )

    meta = json.loads(result["meta"])
    assert "target_observability_control" in meta["active_modules"]
    assert meta["target_observability_control"]["enabled"]
    diagnostics = result["diagnostics"]
    assert diagnostics["target_observability_score_t"].shape == (n,)
    assert diagnostics["target_nuisance_t"].shape == (n,)
    assert np.nanmean(diagnostics["target_observability_score_t"][: n // 2]) > 0.9
    assert np.nanmean(diagnostics["target_observability_score_t"][n // 2 :]) < 0.3
    assert np.nanmean(diagnostics["q_dyn_t"][n // 2 :]) <= np.nanmean(diagnostics["q_dyn_raw_t"][n // 2 :]) + 1e-9


def test_final_observation_law_guards_residual_and_balances_correlated_groups() -> None:
    fs = 20.0
    n = 300
    t = np.arange(n, dtype=float) / fs
    shared = np.sin(2.0 * np.pi * 0.22 * t)
    stack = np.vstack(
        [
            shared,
            shared + 0.04 * np.sin(2.0 * np.pi * 0.44 * t),
            shared + 0.05 * np.sin(2.0 * np.pi * 0.44 * t + 0.15),
        ]
    )
    clean_then_bad = np.r_[np.ones(n // 2), np.full(n - n // 2, 0.15)]
    runtime = {
        "target_observability": clean_then_bad,
        "h1_timing": np.ones(n),
        "h2_morphology": np.ones(n),
        "baseline": np.ones(n),
        "residual": clean_then_bad,
        "nuisance": np.r_[np.zeros(n // 2), np.full(n - n // 2, 0.90)],
    }

    head = oscillator_PARH_OSSM()
    head.ENABLE_DYNAMIC_MIXTURE = True
    head.ENABLE_RATE_OBSERVABILITY_MIXTURE = True
    head.ENABLE_RATE_OBSERVABILITY_HELPER = True
    head.ENABLE_RESIDUAL_SEMANTICS = True
    head.ENABLE_RESIDUAL_IDENTIFIABILITY_GUARD = True
    head.ENABLE_GROUP_BALANCED_FUSION = True
    head.ENABLE_PHASE_ANCHORED_MORPHOLOGY = True

    result = head.run(
        stack,
        fs,
        {
            "method_name": "final_observation_law_contract",
            "base_method": "profile1d_quadratic",
            "observation_law_enabled": True,
            "observation_families": [
                "of_disp_bridge",
                "profile1d_quadratic",
                "profile1d_cubic",
            ],
            "observation_rate_tracks_runtime": {
                "of_disp_bridge": [0.22] * n,
                "profile1d_quadratic": [0.22] * n,
                "profile1d_cubic": [0.22] * n,
            },
            "external_target_observability_runtime": runtime,
            "enable_target_observability_control_runtime": True,
        },
    )

    meta = json.loads(result["meta"])
    assert "residual_identifiability_guard" in meta["active_modules"]
    assert "group_balanced_fusion" in meta["active_modules"]
    assert "z_full_phase_anchored" in result
    diagnostics = result["diagnostics"]
    assert np.nanmean(diagnostics["residual_gate_t"][n // 2 :]) < np.nanmean(diagnostics["residual_gate_t"][: n // 2])
    group_scale = diagnostics["group_balance_scale_t_channels"]
    assert group_scale.shape == stack.shape
    assert np.nanmedian(group_scale[1]) == 2.0
    assert np.nanmedian(group_scale[2]) == 2.0


def test_phase_anchored_morphology_requires_target_observability() -> None:
    fs = 20.0
    n = 240
    t = np.arange(n, dtype=float) / fs
    stack = np.vstack(
        [
            np.sin(2.0 * np.pi * 0.22 * t),
            np.sin(2.0 * np.pi * 0.22 * t + 0.10),
        ]
    )

    head = oscillator_PARH_OSSM()
    head.ENABLE_DYNAMIC_MIXTURE = True
    head.ENABLE_RATE_OBSERVABILITY_MIXTURE = True
    head.ENABLE_RATE_OBSERVABILITY_HELPER = True
    head.ENABLE_RESIDUAL_SEMANTICS = True
    head.ENABLE_PHASE_ANCHORED_MORPHOLOGY = True

    result = head.run(
        stack,
        fs,
        {
            "method_name": "phase_morphology_without_target_evidence_contract",
            "base_method": "profile1d_quadratic",
            "observation_law_enabled": True,
            "observation_families": ["profile1d_quadratic", "profile1d_cubic"],
            "observation_rate_tracks_runtime": {
                "profile1d_quadratic": [0.22] * n,
                "profile1d_cubic": [0.22] * n,
            },
        },
    )

    meta = json.loads(result["meta"])
    phase_meta = meta["phase_anchored_morphology"]
    assert phase_meta["enabled"] is False
    assert phase_meta["reason"] == "missing_target_observability"
    assert "phase_anchored_morphology_readout" not in meta["active_modules"]
    assert "z_full_phase_anchored" in result
