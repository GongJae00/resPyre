import numpy as np

from scripts.materialize_calibrated_multifamily_parh_system import (
    _target_observability_control_runtime,
    _windowed_signal_sqi_observability,
)


def test_target_observability_runtime_uses_source_spread_and_posterior_specificity():
    n = 120
    rate_tracks = [
        np.full(n, 0.20),
        np.full(n, 0.21),
        np.r_[np.full(n // 2, 0.20), np.full(n - n // 2, 0.34)],
    ]
    posterior = {
        "confidence": np.full(n, 0.7),
        "entropy": np.full(n, 0.75),
        "top_gap": np.full(n, 0.30),
        "alias_risk": np.r_[np.zeros(n // 2), np.full(n - n // 2, 0.6)],
        "h1_role_support": np.full(n, 0.8),
        "morphology_role_support": np.full(n, 0.7),
        "abstain_pressure": np.r_[np.zeros(n // 2), np.full(n - n // 2, 0.4)],
    }

    runtime, meta = _target_observability_control_runtime(
        ["of_disp_bridge", "profile1d_quadratic", "dof_disp_bridge"],
        rate_tracks,
        np.full(n, 0.20),
        np.full(n, 0.5),
        posterior,
        n,
        min_hz=0.08,
        max_hz=0.50,
    )

    assert runtime is not None
    assert meta["enabled"]
    spread = np.asarray(runtime["source_spread_hz"], dtype=float)
    alias_safety = np.asarray(runtime["alias_safety"], dtype=float)
    nuisance = np.asarray(runtime["nuisance"], dtype=float)
    assert np.nanmean(spread[n // 2 :]) > np.nanmean(spread[: n // 2])
    assert np.nanmean(alias_safety[n // 2 :]) < np.nanmean(alias_safety[: n // 2])
    assert np.nanmean(nuisance[n // 2 :]) > np.nanmean(nuisance[: n // 2])


def test_signal_sqi_observability_prefers_periodic_respiratory_evidence():
    fps = 30.0
    n = 900
    t = np.arange(n, dtype=float) / fps
    names = ["of_disp_bridge", "profile1d_quadratic", "dof_disp_bridge"]
    rows = [
        {"group": "G_OF_bridge", "start_sec": 0.0, "end_sec": n / fps, "score": 1.0},
        {"group": "G_P1D_morph", "start_sec": 0.0, "end_sec": n / fps, "score": 1.0},
        {"group": "G_DoF_bridge", "start_sec": 0.0, "end_sec": n / fps, "score": 1.0},
    ]
    periodic = [
        np.sin(2.0 * np.pi * 0.20 * t),
        np.sin(2.0 * np.pi * 0.20 * t + 0.08),
        np.sin(2.0 * np.pi * 0.20 * t - 0.05),
    ]
    rng = np.random.default_rng(7)
    noisy = [rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)]
    tracks = [np.full(n, 0.20), np.full(n, 0.20), np.full(n, 0.20)]

    periodic_runtime, periodic_meta = _windowed_signal_sqi_observability(
        names,
        periodic,
        tracks,
        rows,
        n,
        fps,
        min_hz=0.08,
        max_hz=0.50,
    )
    noisy_runtime, noisy_meta = _windowed_signal_sqi_observability(
        names,
        noisy,
        tracks,
        rows,
        n,
        fps,
        min_hz=0.08,
        max_hz=0.50,
    )

    assert periodic_runtime is not None
    assert noisy_runtime is not None
    assert periodic_meta["enabled"]
    assert noisy_meta["enabled"]
    assert np.mean(periodic_runtime["signal_sqi"]) > np.mean(noisy_runtime["signal_sqi"]) + 0.25
    assert np.mean(periodic_runtime["phase_coherence"]) > np.mean(noisy_runtime["phase_coherence"])


def test_target_observability_runtime_accepts_signal_sqi_control():
    fps = 30.0
    n = 900
    t = np.arange(n, dtype=float) / fps
    names = ["of_disp_bridge", "profile1d_quadratic", "dof_disp_bridge"]
    rows = [
        {"group": "G_OF_bridge", "start_sec": 0.0, "end_sec": n / fps, "score": 1.0},
        {"group": "G_P1D_morph", "start_sec": 0.0, "end_sec": n / fps, "score": 1.0},
        {"group": "G_DoF_bridge", "start_sec": 0.0, "end_sec": n / fps, "score": 1.0},
    ]
    periodic = [np.sin(2.0 * np.pi * 0.20 * t + phase) for phase in (0.0, 0.06, -0.03)]
    rng = np.random.default_rng(11)
    noisy = [rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)]
    tracks = [np.full(n, 0.20), np.full(n, 0.20), np.full(n, 0.20)]
    posterior = {
        "confidence": np.full(n, 0.45),
        "entropy": np.full(n, 0.82),
        "top_gap": np.full(n, 0.18),
        "alias_risk": np.full(n, 0.10),
        "h1_role_support": np.full(n, 0.55),
        "morphology_role_support": np.full(n, 0.55),
        "abstain_pressure": np.full(n, 0.10),
    }

    periodic_runtime, periodic_meta = _target_observability_control_runtime(
        names,
        tracks,
        np.full(n, 0.20),
        np.full(n, 0.35),
        posterior,
        n,
        min_hz=0.08,
        max_hz=0.50,
        signals=periodic,
        window_rows=rows,
        fps=fps,
        enable_signal_sqi_observability=True,
    )
    noisy_runtime, noisy_meta = _target_observability_control_runtime(
        names,
        tracks,
        np.full(n, 0.20),
        np.full(n, 0.35),
        posterior,
        n,
        min_hz=0.08,
        max_hz=0.50,
        signals=noisy,
        window_rows=rows,
        fps=fps,
        enable_signal_sqi_observability=True,
    )

    assert periodic_runtime is not None
    assert noisy_runtime is not None
    assert periodic_meta["signal_sqi_observability_enabled"]
    assert noisy_meta["signal_sqi_observability_enabled"]
    assert np.mean(periodic_runtime["signal_sqi"]) > np.mean(noisy_runtime["signal_sqi"]) + 0.25
    assert np.mean(periodic_runtime["target_observability"]) > np.mean(noisy_runtime["target_observability"])
    assert np.mean(periodic_runtime["nuisance"]) < np.mean(noisy_runtime["nuisance"])
