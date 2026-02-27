import os
import tempfile
import json

import numpy as np

from components.models.core.base import OscillatorParams
from components.models.heads.robust_ossm import oscillator_RobustOSSM


def _read_col(log_path, key):
    loaded = np.load(log_path, allow_pickle=True)
    data = loaded["data"]
    fields = list(loaded["fields"])
    return data[:, fields.index(key)]


def test_gating_scope_default_does_not_apply_filter_time_overrides():
    fs = 20.0
    n = 100
    t = np.arange(n, dtype=np.float64)
    sig = np.sin(2.0 * np.pi * 0.25 * t / fs)
    # Force poor quality segment that would normally inflate alpha_R.
    roi_stats_t = [{
        "roi_mean": 100.0,
        "global_mean": 120.0,
        "roi_snr_db": 1.0,
        "valid_ratio": 0.1,
        "roi_cx": 0.5,
        "roi_cy": 0.5,
    } for _ in range(n)]

    params = OscillatorParams(fs=fs, f_min=0.08, f_max=0.5, trace_cap=50.0)
    tmp = tempfile.mkdtemp(prefix="filter_gating_")
    head = oscillator_RobustOSSM(params)
    head.run(
        sig,
        fs,
        {
            "roi_stats_t": roi_stats_t,
            "gating": {"debug": {"disable_gating": True}},
            "aux_save_dir": tmp,
            "trial_key": "gating",
        },
    )
    log_path = os.path.join(tmp, "frame_logs", "gating.npz")
    alpha_r = _read_col(log_path, "alpha_R")
    g_t = _read_col(log_path, "g_t")
    # Default scope is evaluation_only: gating config is audit-only.
    assert not np.allclose(alpha_r, 1.0, atol=1e-8)
    assert not np.allclose(g_t, 1.0, atol=1e-8)


def test_gating_scope_filter_time_disable_overrides_trust():
    fs = 20.0
    n = 100
    t = np.arange(n, dtype=np.float64)
    sig = np.sin(2.0 * np.pi * 0.25 * t / fs)
    roi_stats_t = [{
        "roi_mean": 100.0,
        "global_mean": 120.0,
        "roi_snr_db": 1.0,
        "valid_ratio": 0.1,
        "roi_cx": 0.5,
        "roi_cy": 0.5,
    } for _ in range(n)]

    params = OscillatorParams(fs=fs, f_min=0.08, f_max=0.5, trace_cap=50.0)
    tmp = tempfile.mkdtemp(prefix="filter_gating_scope_")
    head = oscillator_RobustOSSM(params)
    out = head.run(
        sig,
        fs,
        {
            "roi_stats_t": roi_stats_t,
            "gating_scope": "filter_time",
            "gating": {"debug": {"disable_gating": True}},
            "aux_save_dir": tmp,
            "trial_key": "gating",
        },
    )
    log_path = os.path.join(tmp, "frame_logs", "gating.npz")
    alpha_r = _read_col(log_path, "alpha_R")
    g_t = _read_col(log_path, "g_t")
    assert np.allclose(alpha_r, 1.0, atol=1e-8)
    assert np.allclose(g_t, 1.0, atol=1e-8)
    meta = json.loads(out["meta"])
    assert meta.get("gating_scope_used") == "filter_time"


def test_filter_time_prominence_penalty_is_bounded():
    fs = 20.0
    n = 120
    t = np.arange(n, dtype=np.float64)
    sig = np.sin(2.0 * np.pi * 0.25 * t / fs)
    roi_stats_t = [{
        "roi_mean": 120.0,
        "global_mean": 120.0,
        "roi_snr_db": 12.0,
        "valid_ratio": 1.0,
        "roi_cx": 0.5,
        "roi_cy": 0.5,
    } for _ in range(n)]

    params = OscillatorParams(fs=fs, f_min=0.08, f_max=0.5, trace_cap=50.0)
    tmp = tempfile.mkdtemp(prefix="filter_gating_prom_")
    head = oscillator_RobustOSSM(params)
    head.run(
        sig,
        fs,
        {
            "roi_stats_t": roi_stats_t,
            "gating_scope": "filter_time",
            "gating": {
                "profile": "relaxed",
                "spectral": {
                    "peak_ratio_min": 1.0,
                    "prominence_min_db": 1.5,
                    "fwhm_max_hz": 10.0,
                    "fwhm_df_guard": 1.0,
                },
                "tracker": {
                    "std_min_bpm": 0.0,
                    "unique_min": 0.0,
                    "saturation_max": 0.99,
                    "std_is_soft": True,
                    "saturation_margin_hz": 0.0,
                },
                "debug": {"disable_gating": False},
            },
            # Intentionally harsh spectral metadata. This previously collapsed g_t to 0.
            "welch_peak_ratio": 2.0,
            "welch_prom_db": -40.0,
            "welch_fwhm_hz": 0.1,
            "welch_df_hz": 0.25,
            "aux_save_dir": tmp,
            "trial_key": "prom_bound",
        },
    )
    log_path = os.path.join(tmp, "frame_logs", "prom_bound.npz")
    g_t = _read_col(log_path, "g_t")
    alpha_r = _read_col(log_path, "alpha_R")
    # Regression guard: spectral prominence mismatch must not hard-collapse all updates.
    assert float(np.nanmean(g_t)) > 0.05
    assert not np.allclose(g_t, 0.0, atol=1e-8)
    assert np.isfinite(alpha_r).all()
