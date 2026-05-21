import numpy as np

from scripts.materialize_calibrated_multifamily_parh_system import (
    _apply_derived_consistency_scaling,
    _calibrated_posterior_mean_readout,
    _preserve_trustworthy_track_rate,
    _rate_hypothesis_graph_anchor,
    _source_validity_guarded_readout,
    _source_validity_rate_readout_runtime,
)


def _posterior(
    mean,
    *,
    conf,
    entropy,
    top_gap,
    support,
    direct,
    macro,
    direct_macro=None,
    motion_direct=0.0,
    p1d_half=0.0,
    alias_risk=None,
    h1_role=None,
    morphology_alias=None,
    abstain=None,
):
    n = len(mean)
    if direct_macro is None:
        direct_macro = macro
    if alias_risk is None:
        alias_risk = p1d_half * (1.0 - motion_direct)
    if h1_role is None:
        h1_role = direct_macro
    if morphology_alias is None:
        morphology_alias = alias_risk
    if abstain is None:
        abstain = 1.0 - h1_role
    return {
        "mean_hz": list(mean),
        "confidence": [conf] * n,
        "entropy": [entropy] * n,
        "top_gap": [top_gap] * n,
        "support": [support] * n,
        "direct_support": [direct] * n,
        "macro_support": [macro] * n,
        "direct_macro_support": [direct_macro] * n,
        "motion_direct_support": [motion_direct] * n,
        "p1d_half_support": [p1d_half] * n,
        "alias_risk": [alias_risk] * n,
        "h1_role_support": [h1_role] * n,
        "morphology_alias_pressure": [morphology_alias] * n,
        "abstain_pressure": [abstain] * n,
    }


def test_calibrated_mean_preserves_well_supported_downshift() -> None:
    base = [0.30] * 16
    posterior = _posterior(
        [0.22] * 16,
        conf=0.55,
        entropy=0.35,
        top_gap=0.45,
        support=0.60,
        direct=0.35,
        macro=0.75,
    )

    out, conf, meta = _calibrated_posterior_mean_readout(
        base,
        [0.40] * 16,
        posterior,
        min_hz=0.08,
        max_hz=0.50,
    )

    assert out is not None
    assert conf is not None
    assert meta["abstention_guard"] == "target_computable_downshift_specificity"
    assert meta["large_downshift_fraction"] == 1.0
    assert meta["well_supported_downshift_fraction"] == 1.0
    assert np.mean(out) < 0.29


def test_calibrated_mean_abstains_from_weak_downshift() -> None:
    base = [0.30] * 16
    posterior = _posterior(
        [0.20] * 16,
        conf=0.18,
        entropy=0.95,
        top_gap=0.05,
        support=0.12,
        direct=0.02,
        macro=0.25,
    )

    out, conf, meta = _calibrated_posterior_mean_readout(
        base,
        [0.75] * 16,
        posterior,
        min_hz=0.08,
        max_hz=0.50,
    )

    assert out is not None
    assert conf is not None
    assert meta["guarded_downshift_fraction"] == 1.0
    assert np.mean(out) > 0.295


def test_specific_calibrated_mean_boosts_only_sharp_posterior() -> None:
    base = [0.30] * 16
    posterior = _posterior(
        [0.24] * 16,
        conf=0.22,
        entropy=0.86,
        top_gap=0.24,
        support=0.55,
        direct=0.18,
        macro=0.80,
        direct_macro=0.65,
    )

    plain, _, plain_meta = _calibrated_posterior_mean_readout(
        base,
        [0.40] * 16,
        posterior,
        min_hz=0.08,
        max_hz=0.50,
        enable_specificity_boost=False,
    )
    boosted, _, boost_meta = _calibrated_posterior_mean_readout(
        base,
        [0.40] * 16,
        posterior,
        min_hz=0.08,
        max_hz=0.50,
        enable_specificity_boost=True,
    )

    assert plain is not None
    assert boosted is not None
    assert plain_meta["specificity_boost_enabled"] is False
    assert boost_meta["specificity_boost_enabled"] is True
    assert boost_meta["specific_posterior_correction_fraction"] == 1.0
    assert np.mean(boosted) < np.mean(plain)


def test_p1d_half_rate_correction_requires_motion_support() -> None:
    base = [0.30] * 16
    unresolved = _posterior(
        [0.18] * 16,
        conf=0.55,
        entropy=0.30,
        top_gap=0.45,
        support=0.70,
        direct=0.30,
        macro=0.80,
        direct_macro=0.65,
        motion_direct=0.05,
        p1d_half=0.80,
    )
    resolved = _posterior(
        [0.18] * 16,
        conf=0.55,
        entropy=0.30,
        top_gap=0.45,
        support=0.70,
        direct=0.30,
        macro=0.80,
        direct_macro=0.65,
        motion_direct=0.55,
        p1d_half=0.80,
    )

    weak, _, weak_meta = _calibrated_posterior_mean_readout(
        base,
        [0.40] * 16,
        unresolved,
        min_hz=0.08,
        max_hz=0.50,
        enable_specificity_boost=True,
        enable_macro_guard=True,
        enable_role_guard=True,
    )
    strong, _, strong_meta = _calibrated_posterior_mean_readout(
        base,
        [0.40] * 16,
        resolved,
        min_hz=0.08,
        max_hz=0.50,
        enable_specificity_boost=True,
        enable_macro_guard=True,
        enable_role_guard=True,
    )

    assert weak is not None
    assert strong is not None
    assert weak_meta["unresolved_p1d_alias_fraction"] == 1.0
    assert strong_meta["p1d_half_rescue_fraction"] == 1.0
    assert np.mean(strong) < np.mean(weak)


def test_low_direct_macro_support_limits_specificity_boost() -> None:
    base = [0.30] * 16
    weak_direct_macro = _posterior(
        [0.24] * 16,
        conf=0.55,
        entropy=0.30,
        top_gap=0.45,
        support=0.70,
        direct=0.30,
        macro=0.80,
        direct_macro=0.20,
    )
    strong_direct_macro = _posterior(
        [0.24] * 16,
        conf=0.55,
        entropy=0.30,
        top_gap=0.45,
        support=0.70,
        direct=0.30,
        macro=0.80,
        direct_macro=0.70,
    )

    weak, _, weak_meta = _calibrated_posterior_mean_readout(
        base,
        [0.40] * 16,
        weak_direct_macro,
        min_hz=0.08,
        max_hz=0.50,
        enable_specificity_boost=True,
        enable_macro_guard=True,
        enable_role_guard=True,
    )
    strong, _, strong_meta = _calibrated_posterior_mean_readout(
        base,
        [0.40] * 16,
        strong_direct_macro,
        min_hz=0.08,
        max_hz=0.50,
        enable_specificity_boost=True,
        enable_macro_guard=True,
        enable_role_guard=True,
    )

    assert weak is not None
    assert strong is not None
    assert weak_meta["weak_direct_macro_fraction"] == 1.0
    assert weak_meta["specific_posterior_correction_fraction"] == 0.0
    assert strong_meta["specific_posterior_correction_fraction"] == 1.0
    assert np.mean(strong) < np.mean(weak)


def test_role_support_limits_rate_correction_even_with_good_macro_support() -> None:
    base = [0.30] * 16
    morphology_only = _posterior(
        [0.22] * 16,
        conf=0.55,
        entropy=0.30,
        top_gap=0.45,
        support=0.70,
        direct=0.30,
        macro=0.80,
        direct_macro=0.70,
        h1_role=0.10,
        morphology_alias=0.60,
        abstain=0.85,
    )
    timing_supported = _posterior(
        [0.22] * 16,
        conf=0.55,
        entropy=0.30,
        top_gap=0.45,
        support=0.70,
        direct=0.30,
        macro=0.80,
        direct_macro=0.70,
        h1_role=0.75,
        morphology_alias=0.05,
        abstain=0.10,
    )

    weak, _, weak_meta = _calibrated_posterior_mean_readout(
        base,
        [0.40] * 16,
        morphology_only,
        min_hz=0.08,
        max_hz=0.50,
        enable_specificity_boost=True,
        enable_macro_guard=True,
        enable_role_guard=True,
    )
    strong, _, strong_meta = _calibrated_posterior_mean_readout(
        base,
        [0.40] * 16,
        timing_supported,
        min_hz=0.08,
        max_hz=0.50,
        enable_specificity_boost=True,
        enable_macro_guard=True,
        enable_role_guard=True,
    )

    assert weak is not None
    assert strong is not None
    assert weak_meta["h1_role_support_mean"] < strong_meta["h1_role_support_mean"]
    assert weak_meta["specific_posterior_correction_fraction"] == 0.0
    assert strong_meta["specific_posterior_correction_fraction"] == 1.0
    assert np.mean(strong) < np.mean(weak)


def test_hypothesis_graph_scores_independent_timing_and_bridge_preservation() -> None:
    anchor, support, mode, detail = _rate_hypothesis_graph_anchor(
        np.asarray([0.24, 0.241, 0.239, 0.12], dtype=float),
        np.asarray([0.8, 0.7, 0.9, 0.9], dtype=float),
        ["G_OF", "G_OF_bridge", "G_DoF", "G_P1D_morph"],
        min_hz=0.08,
        max_hz=0.50,
        rate_ref=0.03,
    )

    assert mode == "rate_hypothesis_graph_p1d_half_resolved"
    assert abs(anchor - 0.24) < 0.03
    assert support > 0.2
    assert detail["best_independent_timing_support"] > 0.7
    assert detail["best_bridge_timing_preservation"] > 0.7
    assert detail["best_h1_role_support"] > detail["best_morphology_alias_pressure"]


def test_motion_timing_track_is_preserved_when_locally_stable() -> None:
    rate, mode = _preserve_trustworthy_track_rate(
        "G_DoF",
        track_rate=0.31,
        stability=0.50,
        anchored_rate=0.18,
        min_hz=0.08,
        max_hz=0.50,
    )

    assert mode == "motion_timing_track_preserved"
    assert rate == 0.31


def test_motion_timing_track_preservation_is_not_std_thresholded() -> None:
    rate, mode = _preserve_trustworthy_track_rate(
        "G_DoF_bridge",
        track_rate=0.29,
        stability=0.02,
        anchored_rate=0.15,
        min_hz=0.08,
        max_hz=0.50,
    )

    assert mode == "motion_timing_track_preserved"
    assert rate == 0.29


def test_non_motion_group_keeps_harmonic_anchor_below_high_stability() -> None:
    rate, mode = _preserve_trustworthy_track_rate(
        "G_P1D_morph",
        track_rate=0.31,
        stability=0.50,
        anchored_rate=0.18,
        min_hz=0.08,
        max_hz=0.50,
    )

    assert mode == "harmonic_anchor"
    assert rate == 0.18


def test_isolated_bridge_observation_is_downweighted() -> None:
    rates = {"G_DoF": 0.20, "G_DoF_bridge": 0.34, "G_P1D_morph": 0.21}
    scores = {"G_DoF": 0.6, "G_DoF_bridge": 0.6, "G_P1D_morph": 0.4}

    counts = _apply_derived_consistency_scaling(rates, scores, rate_ref=0.04)

    assert counts["derived_isolated"] == 1
    assert scores["G_DoF_bridge"] < 0.25


def test_parent_supported_bridge_observation_is_kept() -> None:
    rates = {"G_DoF": 0.31, "G_DoF_bridge": 0.32, "G_P1D_morph": 0.20}
    scores = {"G_DoF": 0.6, "G_DoF_bridge": 0.6, "G_P1D_morph": 0.4}

    counts = _apply_derived_consistency_scaling(rates, scores, rate_ref=0.04)

    assert counts["derived_supported"] == 1
    assert scores["G_DoF_bridge"] == 0.6


def test_source_validity_reports_low_confidence_under_target_ambiguity() -> None:
    n = 120
    fps = 10.0
    t = np.arange(n, dtype=np.float64) / fps
    names = ["dof", "of_farneback", "profile1d_quadratic"]
    source_hz = [0.18, 0.30, 0.42]
    signals = [np.sin(2.0 * np.pi * hz * t) for hz in source_hz]
    rate_tracks = [np.full(n, hz, dtype=np.float64) for hz in source_hz]
    window_rows = [
        {
            "group": group,
            "start_sec": 0.0,
            "end_sec": 12.0,
            "score": 0.60,
            "z_osc_readout_weight": 0.60,
            "rate_phase_score": 1.0,
            "group_support_score": 0.50,
            "macro_timing_support_score": 0.50,
            "event_timing_score": 0.20,
            "harmonic_risk_score": 0.40,
            "abstain_score": 0.50,
        }
        for group in ("G_DoF", "G_OF", "G_P1D_morph")
    ]

    readout, confidence, meta = _source_validity_rate_readout_runtime(
        names,
        signals,
        rate_tracks,
        window_rows,
        n,
        fps,
        {},
        0.08,
        0.50,
    )

    assert readout is not None
    assert confidence is not None
    assert meta["enabled"] is True
    assert meta["source"] == "source_validity_posterior"
    assert meta["source_validity_entropy_median"] > 0.80
    assert meta["confidence_mean"] < 0.20
    assert meta["selected_group_counts"]


def test_source_validity_guard_preserves_base_when_specificity_is_low() -> None:
    base = [0.30] * 20
    source = [0.18] * 20

    out, conf, meta = _source_validity_guarded_readout(
        base,
        [0.80] * 20,
        source,
        [0.15] * 20,
        {
            "source_validity_entropy_median": 0.95,
            "source_validity_top_gap_median": 0.08,
            "confidence_mean": 0.15,
        },
        min_hz=0.08,
        max_hz=0.50,
    )

    assert out is not None
    assert conf is not None
    assert meta["source"] == "source_validity_guarded"
    assert meta["source_validity_guard_alpha_mean"] == 0.0
    assert np.allclose(out, base)
