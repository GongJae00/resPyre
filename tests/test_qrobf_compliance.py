import os
import tempfile

import numpy as np

from components.models.core.base import OscillatorParams
from components.models.core.robust_update import RobustKalmanUpdater
from components.models.core.ssm import OscillatorPredictor, SSMConfig, StateDecoder
from components.models.core.trust import TrustAllocator
from components.models.heads.robust_ossm import oscillator_RobustOSSM
from components.observations.quality import QualityEstimator, normalize_roi_stats_t


def _read_log(path):
    loaded = np.load(path, allow_pickle=True)
    data = loaded["data"]
    fields = list(loaded["fields"])
    return data, fields


def _col(data, fields, name):
    return data[:, fields.index(name)]


def test_roi_stats_normalization_variants_equivalent():
    """list-of-dicts and dict-of-arrays must yield identical per-frame quality behavior."""
    T = 80
    roi_list = []
    for t in range(T):
        roi_list.append({
            "roi_mean": 100.0,
            "global_mean": 120.0,
            "roi_snr_db": 20.0 if t < 40 else 2.0,
            "valid_ratio": 1.0 if t < 40 else 0.2,
            "roi_cx": 0.5,
            "roi_cy": 0.5,
        })

    roi_dict = {
        "roi_mean": [r["roi_mean"] for r in roi_list],
        "global_mean": [r["global_mean"] for r in roi_list],
        "roi_snr_db": [r["roi_snr_db"] for r in roi_list],
        "valid_ratio": [r["valid_ratio"] for r in roi_list],
        "roi_cx": [r["roi_cx"] for r in roi_list],
        "roi_cy": [r["roi_cy"] for r in roi_list],
    }

    # Unit-level schema normalization checks.
    n_list = normalize_roi_stats_t(roi_list, T)
    n_dict = normalize_roi_stats_t(roi_dict, T)
    assert len(n_list) == T and len(n_dict) == T
    assert n_list[10]["valid_ratio"] == n_dict[10]["valid_ratio"]
    assert n_list[60]["roi_snr_db"] == n_dict[60]["roi_snr_db"]

    # Integration-level equivalence through robust_ossm.
    fs = 20.0
    t = np.arange(T)
    sig = np.sin(2.0 * np.pi * 0.25 * t / fs)
    params = OscillatorParams(fs=fs, f_min=0.1, f_max=0.5, trace_cap=20.0)

    tmp1 = tempfile.mkdtemp(prefix="roi_list_")
    head1 = oscillator_RobustOSSM(params)
    head1.run(sig, fs, {"roi_stats_t": roi_list, "aux_save_dir": tmp1, "trial_key": "a"})
    d1, f1 = _read_log(os.path.join(tmp1, "frame_logs", "a.npz"))

    tmp2 = tempfile.mkdtemp(prefix="roi_dict_")
    head2 = oscillator_RobustOSSM(params)
    head2.run(sig, fs, {"roi_stats_t": roi_dict, "aux_save_dir": tmp2, "trial_key": "b"})
    d2, f2 = _read_log(os.path.join(tmp2, "frame_logs", "b.npz"))

    q1 = _col(d1, f1, "q_vis")
    q2 = _col(d2, f2, "q_vis")
    a1 = _col(d1, f1, "alpha_R")
    a2 = _col(d2, f2, "alpha_R")
    assert np.allclose(q1, q2, equal_nan=True)
    assert np.allclose(a1, a2, equal_nan=True)


def test_q_drift_increases_alpha_q():
    """Synthetic center drift should increase q_drift and alpha_Q above 1."""
    fs = 30.0
    qe = QualityEstimator(fs=fs)
    ta = TrustAllocator()

    alpha_q = []
    q_drift_vals = []
    for t in range(60):
        cx = min(1.0, 0.1 + 0.02 * t)  # monotonic drift
        roi = {
            "roi_mean": 100.0,
            "global_mean": 100.0,
            "roi_snr_db": 20.0,
            "valid_ratio": 1.0,
            "roi_cx": cx,
            "roi_cy": 0.5,
        }
        q = qe.update(t, 0.2, roi_stats=roi)
        trust = ta.allocate(q, nis=0.1, current_freq=0.25)
        q_drift_vals.append(q["q_drift"])
        alpha_q.append(trust.alpha_Q)

    assert np.max(q_drift_vals) > 0.0
    assert np.max(alpha_q) > 1.05
    assert alpha_q[1] > alpha_q[0]


def test_gaussian_updater_limit():
    """Gaussian limit must keep lambda=1 and R_eff=R_scaled."""
    updater = RobustKalmanUpdater.gaussian()
    x = np.array([0.1, 0.0, np.log(0.25)])
    P = np.diag([0.2, 0.1, 0.01])
    H = np.array([[1.0, 0.0, 0.0]])
    R_scaled = np.array([[0.35]])
    res = updater.update(x, P, y_t=5.0, H=H, R=R_scaled)
    assert res.lambda_t == 1.0
    assert np.isclose(res.R_eff, R_scaled.item())


def test_robust_updater_caps_extreme_r_eff():
    """R_eff must be capped to avoid pathological UKF measurement-noise explosion."""
    updater = RobustKalmanUpdater(
        nu=3.0,
        vb_iters=3,
        lambda_floor=1e-3,
        r_eff_max_scale=20.0,
    )
    x = np.array([0.0, 0.0, np.log(0.25)])
    P = np.diag([0.2, 0.1, 0.01])
    H = np.array([[1.0, 0.0, 0.0]])
    R_scaled = np.array([[0.5]])
    # Very large innovation to force tiny lambda before clipping.
    res = updater.update(x, P, y_t=1000.0, H=H, R=R_scaled)
    assert res.R_eff <= (20.0 * R_scaled.item() + 1e-12)


def test_state_decoder_handles_extreme_state_without_overflow():
    x = np.array([1e308, -1e308, np.log(0.25)], dtype=np.float64)
    P = np.diag([1.0, 1.0, 0.01]).astype(np.float64)
    state = StateDecoder.decode(x, P)
    assert np.isfinite(state["amp"])
    assert np.isfinite(state["freq_hz"])


def test_ukf_predict_sanitizes_nonfinite_state_and_covariance():
    pred = OscillatorPredictor(SSMConfig())
    x = np.array([np.inf, np.nan, np.nan], dtype=np.float64)
    P = np.array(
        [
            [np.inf, 0.0, 0.0],
            [0.0, np.nan, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    Q = pred.build_Q()
    x_pred, P_pred = pred.predict_ukf(x, P, Q, dt=1.0 / 30.0)
    assert np.all(np.isfinite(x_pred))
    assert np.all(np.isfinite(P_pred))
    assert np.all(np.diag(P_pred) > 0.0)


def test_trust_monotonicity():
    """Worse quality should reduce gate and increase alpha_R; harmonic should reduce w_h."""
    good = {
        "q_vis": 1.0,
        "q_drift": 0.0,
        "q_cons": 1.0,
        "q_out": 0.0,
        "q_harm": 0.0,
        "q_burst": 0.0,
    }
    bad = {
        "q_vis": 0.2,
        "q_drift": 0.0,
        "q_cons": 0.2,
        "q_out": 2.0,
        "q_harm": 0.3,
        "q_burst": 1.0,
    }
    ta1 = TrustAllocator()
    t_good = ta1.allocate(good, nis=0.2, current_freq=0.25)
    ta2 = TrustAllocator()
    t_bad = ta2.allocate(bad, nis=0.2, current_freq=0.25)

    assert t_bad.alpha_R > t_good.alpha_R
    assert t_bad.g_t < t_good.g_t
    assert t_bad.w_h < t_good.w_h


def test_integration_smoke_spike_harmonic():
    """End-to-end smoke test: no NaNs, clamp respected, trace cap respected."""
    fs = 30.0
    n = 240
    t = np.arange(n, dtype=np.float64)
    sig = np.sin(2.0 * np.pi * 0.25 * t / fs)
    sig[60] += 6.0
    sig[120] -= 5.5
    sig[140:210] += 0.6 * np.sin(2.0 * np.pi * 0.50 * t[140:210] / fs)

    roi_stats = []
    for i in range(n):
        roi_stats.append({
            "roi_mean": 100.0,
            "global_mean": 120.0,
            "roi_snr_db": 18.0 if i < 140 else 4.0,
            "valid_ratio": 1.0 if i < 140 else 0.5,
            "roi_cx": float(np.clip(0.5 + 0.08 * np.sin(2.0 * np.pi * i / 40.0), 0.0, 1.0)),
            "roi_cy": 0.5,
        })

    params = OscillatorParams(
        fs=fs,
        f_min=0.08,
        f_max=0.5,
        trace_cap=10.0,
        student_t_nu=10.0,
        predict_method="ekf",
    )
    head = oscillator_RobustOSSM(params)
    tmp = tempfile.mkdtemp(prefix="smoke_qrobf_")
    out = head.run(sig, fs, {"roi_stats_t": roi_stats, "aux_save_dir": tmp, "trial_key": "smoke"})

    assert not np.isnan(out["signal_hat"]).any()
    assert not np.isnan(out["track_hz"]).any()
    assert np.all(out["track_hz"] >= params.f_min - 1e-9)
    assert np.all(out["track_hz"] <= params.f_max + 1e-9)

    data, fields = _read_log(os.path.join(tmp, "frame_logs", "smoke.npz"))
    trace_p = _col(data, fields, "trace_P")
    z = _col(data, fields, "z")
    lo, hi = np.log(params.f_min), np.log(params.f_max)
    assert np.nanmax(trace_p) <= params.trace_cap + 1e-6
    assert np.nanmin(z) >= lo - 1e-9
    assert np.nanmax(z) <= hi + 1e-9
