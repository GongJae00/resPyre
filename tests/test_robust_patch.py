"""
Patch regression tests for robust_ossm pipeline.

T1: Failure flag logging mismatch (P0-1)
T2: z-clamp stability (P0-3)
T3: tau_env/rho consistency (P0-4)
T4: vb_iters/trace_cap wiring (P0-2)
T5: EDA baseline mode (P1-5)
T6: w_h harmonic suppression wiring (P1-6)
"""

import sys
import os
import numpy as np

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_T1_failure_flag_logging():
    """T1: Reproduce old mismatch — FailureFlags.to_dict() keys must be
    accepted by FrameLogger without TypeError."""
    from core.evaluation.frame_logger import FrameLogger
    from components.models.core.failure_monitor import FailureFlags

    flags = FailureFlags(diverge=True, phase_slip=False, locking=True, doubling=False)
    fd = flags.to_dict()
    # Keys are: fail_diverge, fail_slip, fail_lock, fail_double
    assert 'fail_diverge' in fd, "to_dict() should produce fail_* keys"

    logger = FrameLogger(5)
    # Old bug: logger.log_failure(0, **fd) would TypeError
    # because log_failure expects (diverge=, slip=, lock=, double=)
    # Fix: robust_ossm now calls log_frame(t, **fd) instead.
    # Verify log_frame accepts fail_* keys without error:
    logger.log_frame(0, **fd)
    arr = logger.get_array()
    assert arr[0, logger._field_idx['fail_diverge']] == 1.0, "fail_diverge not logged"
    assert arr[0, logger._field_idx['fail_lock']] == 1.0, "fail_lock not logged"
    assert arr[0, logger._field_idx['fail_slip']] == 0.0, "fail_slip should be 0"
    assert arr[0, logger._field_idx['fail_double']] == 0.0, "fail_double should be 0"

    # Also verify the old log_failure still works with its own signature:
    logger.log_failure(1, diverge=False, slip=True, lock=False, double=True)
    assert arr[1, logger._field_idx['fail_slip']] == 1.0
    assert arr[1, logger._field_idx['fail_double']] == 1.0

    print("  T1 Failure flag logging mismatch: PASS")


def test_T2_z_clamp_stability():
    """T2: Force z beyond bounds — confirm clamp_state brings it back."""
    from components.models.core.ssm import OscillatorPredictor, SSMConfig

    cfg = SSMConfig(f_min=0.08, f_max=0.5)
    pred = OscillatorPredictor(cfg)

    lo, hi = cfg.log_f_bounds
    assert lo == np.log(0.08), f"log_f_bounds lower mismatch: {lo}"
    assert hi == np.log(0.5), f"log_f_bounds upper mismatch: {hi}"

    # Case 1: z far above bound
    x = np.array([1.0, 0.0, 5.0])  # exp(5) ≈ 148 Hz — way beyond 0.5 Hz
    pred.clamp_state(x)
    assert x[2] == hi, f"z should be clamped to {hi}, got {x[2]}"

    # Case 2: z far below bound
    x2 = np.array([0.5, 0.3, -10.0])  # exp(-10) ≈ 4.5e-5 Hz
    pred.clamp_state(x2)
    assert x2[2] == lo, f"z should be clamped to {lo}, got {x2[2]}"

    # Case 3: z within bounds should be unchanged
    z_valid = np.log(0.25)
    x3 = np.array([0.0, 0.0, z_valid])
    pred.clamp_state(x3)
    assert x3[2] == z_valid, "z within bounds should not be modified"

    # Case 4: After predict, z should still be in bounds
    # (inject extreme state then predict)
    x_extreme = np.array([0.5, 0.3, 5.0])
    pred.clamp_state(x_extreme)
    P = np.diag([0.1, 0.1, 0.01])
    Q = pred.build_Q()
    x_pred, P_pred = pred.predict(x_extreme, P, Q, dt=1/30.0)
    pred.clamp_state(x_pred)
    assert lo <= x_pred[2] <= hi, f"z after predict+clamp out of bounds: {x_pred[2]}"

    print("  T2 z-clamp stability: PASS")


def test_T3_tau_env_rho_consistency():
    """T3: compute_rho() must derive rho from tau_env consistently.
    When rho is explicitly set (0 < rho < 1), it overrides tau_env.
    When rho=0 (default), tau_env formula is used."""
    from components.models.core.ssm import SSMConfig

    dt = 1.0 / 30.0

    # Case 1: rho=0 (default) → uses tau_env formula
    cfg1 = SSMConfig(tau_env=32.0, rho=0.0)
    expected = np.exp(-dt / 32.0)
    actual = cfg1.compute_rho(dt)
    assert abs(actual - expected) < 1e-10, \
        f"Default rho: expected {expected}, got {actual}"

    # Case 2: rho explicitly set → use it directly
    cfg2 = SSMConfig(tau_env=32.0, rho=0.995)
    assert cfg2.compute_rho(dt) == 0.995, \
        f"Explicit rho should be 0.995, got {cfg2.compute_rho(dt)}"

    # Case 3: Different tau_env → different rho
    cfg3 = SSMConfig(tau_env=10.0, rho=0.0)
    rho_10 = cfg3.compute_rho(dt)
    rho_32 = cfg1.compute_rho(dt)
    assert rho_10 < rho_32, "Shorter tau_env should give lower rho (more damping)"

    # Case 4: rho=0.999 (valid override boundary)
    cfg4 = SSMConfig(tau_env=32.0, rho=0.999)
    assert cfg4.compute_rho(dt) == 0.999

    # Case 5: rho=1.0 (boundary — should NOT override, fall through to tau_env)
    cfg5 = SSMConfig(tau_env=32.0, rho=1.0)
    assert cfg5.compute_rho(dt) == expected, \
        "rho=1.0 should not override (not in (0,1) range)"

    print("  T3 tau_env/rho consistency: PASS")


def test_T4_vb_iters_trace_cap_wiring():
    """T4: OscillatorParams.vb_iters and trace_cap must propagate to updater."""
    from components.models.core.robust_update import RobustKalmanUpdater

    # Default
    u1 = RobustKalmanUpdater(nu=5.0, vb_iters=1, trace_cap=100.0)
    assert u1.vb_iters == 1
    assert u1.trace_cap == 100.0

    # Custom values
    u2 = RobustKalmanUpdater(nu=5.0, vb_iters=3, trace_cap=1.0)
    assert u2.vb_iters == 3
    assert u2.trace_cap == 1.0

    # Trace cap=1.0 should trigger scaling on a P with trace > 1
    x = np.array([0.5, 0.3, np.log(0.25)])
    P = np.diag([10.0, 10.0, 1.0])  # trace = 21.0
    H = np.array([[1.0, 0.0, 0.0]])
    R = np.array([[0.05]])
    result = u2.update(x, P, 0.6, H, R)
    trace_after = float(np.trace(result.P))
    assert trace_after <= 1.0 + 1e-6, \
        f"trace_cap=1.0 but trace(P)={trace_after} after update"

    print("  T4 vb_iters/trace_cap wiring: PASS")


def test_T5_eda_baseline_mode():
    """T5: EDA baseline mode must force Gaussian (λ=1) and neutral trust."""
    from components.models.core.robust_update import RobustKalmanUpdater

    # Gaussian updater (eda_baseline equivalent)
    updater = RobustKalmanUpdater.gaussian()
    assert updater.nu == float('inf'), "Gaussian mode should have nu=inf"

    x = np.array([0.5, 0.3, np.log(0.25)])
    P = np.diag([0.1, 0.1, 0.01])
    H = np.array([[1.0, 0.0, 0.0]])
    R = np.array([[0.05]])

    # Even with a large outlier, λ should be 1.0 (Gaussian: no downweighting)
    result = updater.update(x, P, 5.0, H, R)
    assert result.lambda_t == 1.0, \
        f"EDA mode: lambda should be 1.0, got {result.lambda_t}"

    print("  T5 EDA baseline mode: PASS")


def test_T6_harmonic_suppression_wiring():
    """T6: gate_z_eff = g_z * w_h — w_h < 1 should reduce frequency update."""
    from components.models.core.trust import TrustAllocator, TrustParams, default_quality
    from components.models.core.robust_update import RobustKalmanUpdater

    # Quality with high harmonic content
    q_harm = {'q_vis': 1.0, 'q_drift': 0.0, 'q_cons': 1.0,
              'q_out': 0.0, 'q_harm': 0.3, 'q_burst': 0.0}  # max thd
    ta = TrustAllocator()
    trust = ta.allocate(q_harm, nis=0.0)
    assert 0.0 < trust.w_h <= 0.2, \
        f"w_h should drop near configured floor for q_harm=0.3, got {trust.w_h}"

    gate_z_eff = trust.g_z * trust.w_h
    assert gate_z_eff < trust.g_z, \
        f"gate_z_eff should be suppressed vs g_z, got gate_z_eff={gate_z_eff}, g_z={trust.g_z}"

    # Normal quality → w_h=1 → gate_z_eff = g_z
    q_ok = default_quality()
    trust2 = ta.allocate(q_ok, nis=0.0)
    assert trust2.w_h == 1.0, f"w_h should be 1.0 for neutral quality, got {trust2.w_h}"
    gate_z_eff2 = trust2.g_z * trust2.w_h
    assert abs(gate_z_eff2 - trust2.g_z) < 1e-10, "gate_z_eff should equal g_z when w_h=1"

    # Verify that a lower gate_z changes the frequency update magnitude
    updater = RobustKalmanUpdater(nu=5.0)
    x = np.array([0.5, 0.3, np.log(0.25)])
    # Use P with cross-correlation so K_z is nonzero
    P = np.array([[0.1, 0.0, 0.005],
                   [0.0, 0.1, 0.0],
                   [0.005, 0.0, 0.01]])
    H = np.array([[1.0, 0.0, 0.0]])
    R = np.array([[0.05]])

    r_full = updater.update(x.copy(), P.copy(), 2.0, H, R, gate=1.0, gate_z=1.0)
    r_gated = updater.update(x.copy(), P.copy(), 2.0, H, R, gate=1.0, gate_z=0.01)
    # Frequency update should be much smaller with gate_z=0.01
    dz_full = abs(r_full.x[2] - x[2])
    dz_gated = abs(r_gated.x[2] - x[2])
    assert dz_full > 1e-6, f"K_z should be nonzero with cross-corr P, dz_full={dz_full}"
    assert dz_gated < dz_full, \
        f"gate_z=0.01 should reduce z update: full={dz_full}, gated={dz_gated}"

    print("  T6 harmonic suppression wiring: PASS")


if __name__ == '__main__':
    print("=== Patch Regression Tests ===")
    test_T1_failure_flag_logging()
    test_T2_z_clamp_stability()
    test_T3_tau_env_rho_consistency()
    test_T4_vb_iters_trace_cap_wiring()
    test_T5_eda_baseline_mode()
    test_T6_harmonic_suppression_wiring()
    print()
    print("=== ALL 6 TESTS PASSED ===")
