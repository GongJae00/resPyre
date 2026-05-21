from scripts.extract_target_reliability_graph_features import (
    GROUP_ORDER,
    _event_timing_score,
    _geometric_mean,
    _macro_family,
    method_group,
)
from scripts.materialize_calibrated_multifamily_parh_system import (
    FAMILY_GROUPS,
    _calibrate_stack,
    _channel_context_prior_runtime,
    _decoupled_rate_readout_runtime,
    _group_aware_half_rate_rescue,
    _load_reliability_priors,
    _load_windowed_reliability_priors,
    _rate_hypothesis_graph_anchor,
    _readout_timing_reliability_from_row,
    _regime_observation_law_runtime,
    _sinusoid_phase_fit,
    _split_reliability_scope,
    _timing_reliability_from_row,
    _windowed_rate_anchor_runtime,
)
from components.models.heads.parh_ossm import oscillator_PARH_OSSM


def test_method_group_locks_observation_groups() -> None:
    assert method_group("of_farneback") == "G_OF"
    assert method_group("of_farneback__parh_ossm") == "G_OF"
    assert method_group("of_disp_bridge") == "G_OF_bridge"
    assert method_group("of_disp_bridge__kfstd") == "G_OF_bridge"
    assert method_group("DoF") == "G_DoF"
    assert method_group("dof__parh_ossm") == "G_DoF"
    assert method_group("dof_disp_bridge") == "G_DoF_bridge"
    assert method_group("profile1D linear") == "G_P1D_low"
    assert method_group("profile1d_quadratic__kfstd") == "G_P1D_morph"
    assert method_group("profile1d_cubic__parh_ossm") == "G_P1D_morph"
    assert method_group("profile1d_consensus") == "G_P1D_cons"
    assert len(GROUP_ORDER) == 7


def test_geometric_mean_prevents_single_component_dominance() -> None:
    high_with_one_bad = _geometric_mean([1.0, 1.0, 0.01])
    all_moderate = _geometric_mean([0.45, 0.45, 0.45])
    assert high_with_one_bad < all_moderate


def test_event_timing_score_accepts_bursty_respiration_not_random_noise() -> None:
    import numpy as np

    fs = 20.0
    t = np.arange(1200, dtype=float) / fs
    bursty = np.maximum(0.0, np.sin(2.0 * np.pi * 0.22 * t)) ** 4
    noise = np.random.default_rng(0).normal(size=t.size)

    assert _event_timing_score(bursty, fs, 0.08, 0.50) > 0.80
    assert _event_timing_score(noise, fs, 0.08, 0.50) < 0.40


def test_macro_family_collapses_same_family_variants_for_timing_support() -> None:
    assert _macro_family("G_P1D_low") == "P1D"
    assert _macro_family("G_P1D_morph") == "P1D"
    assert _macro_family("G_P1D_cons") == "P1D"
    assert _macro_family("G_OF") == "OF"
    assert _macro_family("G_OF_bridge") == "OF"


def test_materializer_uses_final_seven_group_semantics() -> None:
    assert FAMILY_GROUPS["of_farneback"] == "G_OF"
    assert FAMILY_GROUPS["of_disp_bridge"] == "G_OF_bridge"
    assert FAMILY_GROUPS["dof"] == "G_DoF"
    assert FAMILY_GROUPS["dof_disp_bridge"] == "G_DoF_bridge"
    assert FAMILY_GROUPS["profile1d_linear"] == "G_P1D_low"
    assert FAMILY_GROUPS["profile1d_quadratic"] == "G_P1D_morph"
    assert FAMILY_GROUPS["profile1d_cubic"] == "G_P1D_morph"
    assert FAMILY_GROUPS["profile1d_consensus"] == "G_P1D_cons"


def test_reliability_csv_loads_trial_group_priors(tmp_path) -> None:
    csv = tmp_path / "rel.csv"
    csv.write_text(
        "video,group,soft_group_weight,reliability_score\n"
        "trial_a,G_OF,0.20,0.5\n"
        "trial_a,G_P1D_cons,0.80,0.9\n",
        encoding="utf-8",
    )
    priors = _load_reliability_priors(csv)
    assert priors["trial_a"]["G_OF"] == 0.20
    assert priors["trial_a"]["G_P1D_cons"] == 0.80


def test_windowed_reliability_csv_loads_runtime_rows(tmp_path) -> None:
    csv = tmp_path / "rel_win.csv"
    csv.write_text(
        "video,group,window_start_sec,window_end_sec,soft_group_weight,timing_reliability_score,timing_reliability_score_v3,morphology_reliability_score\n"
        "trial_a,G_OF,0,10,0.20,0.70,0.75,0.30\n"
        "trial_a,G_P1D_cons,5,15,0.80,0.20,0.25,0.90\n",
        encoding="utf-8",
    )
    priors = _load_windowed_reliability_priors(csv)
    assert len(priors["trial_a"]) == 2
    assert priors["trial_a"][0]["timing_reliability_score"] == 0.70
    assert priors["trial_a"][0]["timing_reliability_score_v3"] == 0.75
    assert priors["trial_a"][1]["morphology_reliability_score"] == 0.90
    mat = _channel_context_prior_runtime(
        ["of_farneback", "profile1d_consensus"],
        priors["trial_a"],
        n_frames=20,
        fps=1.0,
        fallback_priors={"of_farneback": 0.2, "profile1d_consensus": 0.8},
    )
    assert mat is not None
    assert mat[0][2] == 0.30
    assert mat[1][7] == 0.90


def test_arbiter_timing_only_affects_readout_not_state_update() -> None:
    row = {
        "arbiter_timing_score": 0.91,
        "timing_reliability_score_v3": 0.31,
        "timing_reliability_score": 0.29,
        "reliability_score": 0.50,
    }
    assert _timing_reliability_from_row(row) == 0.31
    assert _readout_timing_reliability_from_row(row) == 0.91


def test_readout_only_reliability_scope_blocks_state_prior_leakage() -> None:
    trial_priors = {"G_OF": 0.80, "G_P1D_cons": 0.20}
    window_rows = [
        {
            "group": "G_OF",
            "start_sec": 0.0,
            "end_sec": 10.0,
            "arbiter_timing_score": 0.90,
            "timing_reliability_score_v3": 0.25,
        }
    ]
    state_priors, state_rows, readout_priors, readout_rows = _split_reliability_scope(
        "readout_only",
        trial_priors,
        window_rows,
    )
    assert state_priors == trial_priors
    assert state_rows == [
        {
            "group": "G_OF",
            "start_sec": 0.0,
            "end_sec": 10.0,
            "timing_reliability_score_v3": 0.25,
        }
    ]
    assert readout_priors == trial_priors
    assert readout_rows == window_rows


def test_state_only_reliability_scope_blocks_readout_prior_leakage() -> None:
    trial_priors = {"G_OF": 0.80, "G_P1D_cons": 0.20}
    window_rows = [
        {
            "group": "G_OF",
            "start_sec": 0.0,
            "end_sec": 10.0,
            "h1_timing_weight": 0.70,
            "z_osc_readout_weight": 0.65,
        }
    ]
    state_priors, state_rows, readout_priors, readout_rows = _split_reliability_scope(
        "state_only",
        trial_priors,
        window_rows,
    )
    assert state_priors == trial_priors
    assert state_rows == window_rows
    assert readout_priors == {}
    assert readout_rows == []


def test_windowed_rate_anchor_prefers_reliable_group() -> None:
    import numpy as np

    names = ["of_farneback", "profile1d_consensus"]
    rate_tracks = [
        np.full(60, 0.40, dtype=float),
        np.full(60, 0.20, dtype=float),
    ]
    window_rows = [
        {"group": "G_OF", "start_sec": 0.0, "end_sec": 30.0, "score": 0.10, "rate_phase_score": 0.50},
        {"group": "G_P1D_cons", "start_sec": 0.0, "end_sec": 30.0, "score": 0.90, "rate_phase_score": 1.00},
    ]
    anchor, confidence, meta = _windowed_rate_anchor_runtime(
        names,
        rate_tracks,
        window_rows,
        n_frames=60,
        fps=2.0,
        fallback_priors={"G_OF": 0.10, "G_P1D_cons": 0.90},
        min_hz=0.08,
        max_hz=0.50,
    )
    assert anchor is not None
    assert confidence is not None
    assert meta["enabled"] is True
    assert abs(anchor[5] - 0.20) < 1e-9
    assert confidence[5] > 0.0


def test_windowed_rate_anchor_does_not_half_demote_valid_track() -> None:
    import numpy as np

    fs = 10.0
    n = 600
    t = np.arange(n, dtype=float) / fs
    names = ["dof", "profile1d_consensus"]
    rate_tracks = [
        np.full(n, 0.24, dtype=float),
        np.full(n, 0.24, dtype=float),
    ]
    # A strong subharmonic-looking waveform must not overrule a valid declared
    # timing track; it is confidence context, not the timing readout itself.
    signals = [
        np.sin(2.0 * np.pi * 0.12 * t),
        np.sin(2.0 * np.pi * 0.12 * t),
    ]
    window_rows = [
        {"group": "G_DoF", "start_sec": 0.0, "end_sec": 30.0, "score": 0.90, "rate_phase_score": 1.00},
        {"group": "G_P1D_cons", "start_sec": 0.0, "end_sec": 30.0, "score": 0.90, "rate_phase_score": 1.00},
    ]
    anchor, confidence, meta = _windowed_rate_anchor_runtime(
        names,
        rate_tracks,
        window_rows,
        n_frames=n,
        fps=fs,
        fallback_priors={},
        min_hz=0.08,
        max_hz=0.50,
        signals=signals,
    )
    assert anchor is not None
    assert confidence is not None
    assert meta["enabled"] is True
    assert abs(anchor[5] - 0.24) < 1e-9
    assert meta["anchor_mode_counts"]["direct_track_consensus"] >= 1


def test_sinusoid_phase_fit_scores_matching_frequency() -> None:
    import numpy as np

    fs = 20.0
    t = np.arange(400, dtype=float) / fs
    signal = np.sin(2.0 * np.pi * 0.25 * t)
    score_match, phase = _sinusoid_phase_fit(signal, fs, 0.25)
    score_wrong, _ = _sinusoid_phase_fit(signal, fs, 0.40)
    assert score_match > 0.90
    assert score_match > score_wrong
    assert np.isfinite(phase)


def test_regime_observation_law_boosts_motion_bridge_when_supported() -> None:
    import numpy as np

    fs = 10.0
    n = 300
    t = np.arange(n, dtype=float) / fs
    names = ["of_farneback", "dof", "dof_disp_bridge", "profile1d_quadratic"]
    signals = [
        np.sin(2.0 * np.pi * 0.40 * t),
        np.maximum(0.0, np.sin(2.0 * np.pi * 0.20 * t)),
        np.sin(2.0 * np.pi * 0.20 * t),
        np.sin(2.0 * np.pi * 0.27 * t),
    ]
    rate_tracks = [
        np.full(n, 0.40, dtype=float),
        np.full(n, 0.20, dtype=float),
        np.full(n, 0.20, dtype=float),
        np.full(n, 0.27, dtype=float),
    ]
    window_rows = [
        {"group": "G_OF", "start_sec": 0.0, "end_sec": 30.0, "score": 0.90, "rate_phase_score": 0.40},
        {"group": "G_DoF", "start_sec": 0.0, "end_sec": 30.0, "score": 0.70, "rate_phase_score": 1.00},
        {"group": "G_DoF_bridge", "start_sec": 0.0, "end_sec": 30.0, "score": 0.75, "rate_phase_score": 1.00},
        {"group": "G_P1D_morph", "start_sec": 0.0, "end_sec": 30.0, "score": 0.35, "rate_phase_score": 0.60},
    ]
    context, anchor, confidence, meta = _regime_observation_law_runtime(
        names,
        signals,
        rate_tracks,
        window_rows,
        n_frames=n,
        fps=fs,
        fallback_priors={},
        min_hz=0.08,
        max_hz=0.50,
    )
    assert context is not None
    assert anchor is not None
    assert confidence is not None
    assert meta["enabled"] is True
    assert abs(anchor[10] - 0.20) < 1e-9
    assert confidence[10] > 0.0
    assert context[2][10] > context[0][10]


def test_decoupled_rate_readout_accepts_single_strong_timing_family() -> None:
    import numpy as np

    fs = 10.0
    n = 300
    t = np.arange(n, dtype=float) / fs
    names = ["of_farneback", "dof_disp_bridge", "profile1d_consensus"]
    signals = [
        np.sin(2.0 * np.pi * 0.42 * t),
        np.sin(2.0 * np.pi * 0.18 * t),
        0.2 * np.sin(2.0 * np.pi * 0.31 * t),
    ]
    rate_tracks = [
        np.full(n, 0.42, dtype=float),
        np.full(n, 0.18, dtype=float),
        np.full(n, 0.31, dtype=float),
    ]
    window_rows = [
        {"group": "G_OF", "start_sec": 0.0, "end_sec": 30.0, "score": 0.05, "rate_phase_score": 0.20},
        {"group": "G_DoF_bridge", "start_sec": 0.0, "end_sec": 30.0, "score": 0.90, "rate_phase_score": 1.00},
        {"group": "G_P1D_cons", "start_sec": 0.0, "end_sec": 30.0, "score": 0.05, "rate_phase_score": 0.20},
    ]
    readout, confidence, meta = _decoupled_rate_readout_runtime(
        names,
        signals,
        rate_tracks,
        window_rows,
        n_frames=n,
        fps=fs,
        fallback_priors={},
        min_hz=0.08,
        max_hz=0.50,
    )
    assert readout is not None
    assert confidence is not None
    assert meta["enabled"] is True
    assert abs(readout[10] - 0.18) < 0.01
    assert confidence[10] > 0.0
    assert meta["selected_group_counts"]["G_DoF_bridge"] >= 1


def test_decoupled_rate_readout_uses_timing_not_morphology_score() -> None:
    import numpy as np

    fs = 10.0
    n = 300
    t = np.arange(n, dtype=float) / fs
    names = ["dof", "profile1d_consensus"]
    signals = [
        np.sin(2.0 * np.pi * 0.18 * t),
        np.sin(2.0 * np.pi * 0.32 * t),
    ]
    rate_tracks = [
        np.full(n, 0.18, dtype=float),
        np.full(n, 0.32, dtype=float),
    ]
    window_rows = [
        {
            "group": "G_DoF",
            "start_sec": 0.0,
            "end_sec": 30.0,
            "score": 0.10,
            "timing_reliability_score": 0.10,
            "timing_reliability_score_v3": 0.90,
            "morphology_reliability_score": 0.20,
            "rate_phase_score": 1.00,
        },
        {
            "group": "G_P1D_cons",
            "start_sec": 0.0,
            "end_sec": 30.0,
            "score": 0.90,
            "timing_reliability_score": 0.90,
            "timing_reliability_score_v3": 0.10,
            "morphology_reliability_score": 0.90,
            "rate_phase_score": 0.60,
        },
    ]
    readout, confidence, meta = _decoupled_rate_readout_runtime(
        names,
        signals,
        rate_tracks,
        window_rows,
        n_frames=n,
        fps=fs,
        fallback_priors={},
        min_hz=0.08,
        max_hz=0.50,
    )
    assert readout is not None
    assert confidence is not None
    assert abs(readout[10] - 0.18) < 0.01
    assert meta["selected_group_counts"]["G_DoF"] >= 1


def test_group_aware_half_rate_rescue_requires_motion_double_support() -> None:
    import numpy as np

    rates = np.asarray([0.125, 0.125, 0.25, 0.25], dtype=float)
    weights = np.asarray([0.8, 0.6, 0.7, 0.6], dtype=float)
    groups = ["G_P1D_cons", "G_P1D_morph", "G_DoF", "G_DoF_bridge"]
    rescued, active, strength = _group_aware_half_rate_rescue(
        0.125,
        rates,
        weights,
        groups,
        min_hz=0.08,
        max_hz=0.50,
        rate_ref=0.03,
    )
    assert active is True
    assert strength > 0.0
    assert abs(rescued - 0.25) < 1e-12


def test_rate_hypothesis_graph_resolves_p1d_half_rate_with_motion_support() -> None:
    import numpy as np

    anchor, support, mode, detail = _rate_hypothesis_graph_anchor(
        np.asarray([0.125, 0.125, 0.25, 0.25], dtype=float),
        np.asarray([0.8, 0.7, 0.7, 0.6], dtype=float),
        ["G_P1D_cons", "G_P1D_morph", "G_DoF", "G_DoF_bridge"],
        min_hz=0.08,
        max_hz=0.50,
        rate_ref=0.03,
    )
    assert abs(anchor - 0.25) < 1e-12
    assert support > 0.0
    assert mode == "rate_hypothesis_graph_p1d_half_resolved"
    assert detail["top_candidates"][0]["hz"] == anchor


def test_rate_hypothesis_graph_does_not_double_p1d_without_motion_support() -> None:
    import numpy as np

    anchor, support, mode, _ = _rate_hypothesis_graph_anchor(
        np.asarray([0.125, 0.125], dtype=float),
        np.asarray([0.8, 0.7], dtype=float),
        ["G_P1D_cons", "G_P1D_morph"],
        min_hz=0.08,
        max_hz=0.50,
        rate_ref=0.03,
    )
    assert abs(anchor - 0.125) < 1e-12
    assert support > 0.0
    assert mode == "rate_hypothesis_graph"


def test_external_output_rate_readout_is_bounded_conflict_weighted_correction() -> None:
    import numpy as np

    head = oscillator_PARH_OSSM()
    internal = np.full(100, 0.12, dtype=float)
    readout = np.full(100, 0.25, dtype=float)
    confidence = np.full(100, 0.08, dtype=float)

    out, active = head._external_output_rate_postprocess(
        internal,
        readout,
        confidence,
        alpha_used=0.0,
    )

    assert np.all(out > internal)
    assert np.all(out < readout)
    assert np.all(active > 0.0)
    assert np.all(active < 1.0)


def test_reliability_prior_modulates_canonical_weights() -> None:
    import numpy as np

    t = np.linspace(0.0, 6.0 * np.pi, 240)
    channels = [
        np.sin(t),
        np.sin(t + 0.05),
        np.sin(t + 0.10),
        np.sin(t + 0.15),
    ]
    names = ["of_farneback", "of_disp_bridge", "profile1d_quadratic", "profile1d_consensus"]
    _, _, meta, rows = _calibrate_stack(
        channels,
        names,
        20.0,
        0.08,
        0.50,
        1.0,
        0.08,
        "all",
        "all",
        reliability_group_prior={
            "G_OF": 0.05,
            "G_OF_bridge": 0.05,
            "G_P1D_morph": 0.10,
            "G_P1D_cons": 0.80,
        },
    )
    weights = {row["channel"]: row["canonical_weight"] for row in rows}
    assert meta["target_reliability_graph_enabled"] is True
    assert weights["profile1d_consensus"] > weights["of_farneback"]
