#!/usr/bin/env python
"""
Phase 7: Extended Bayesian Metrics Tests.

T1: nis_calibration_chi2 — well-calibrated NIS should pass
T2: nis_calibration_chi2 — miscalibrated NIS should fail
T3: coverage_percentage — 100% coverage with tight estimates
T4: coverage_percentage — 0% coverage with terrible estimates
T5: stability_duration — stable track should report long duration
T6: stability_duration — unstable track should report short duration
T7: All 3 metrics are importable and consistent
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_1_nis_calibrated():
    """Well-calibrated NIS ~ χ²(1) should pass the chi2 test."""
    from core.evaluation.metrics import nis_calibration_chi2
    np.random.seed(42)
    # χ²(1) samples → well-calibrated
    nis = np.random.chisquare(df=1, size=500)
    result = nis_calibration_chi2(nis, dof=1)
    assert result['pass_chi2'], \
        f"Well-calibrated NIS should pass, mean={result['mean_nis']:.3f}"
    assert 0.5 < result['mean_nis'] < 1.5, \
        f"Mean NIS should be ~1.0, got {result['mean_nis']}"
    print("  [1] NIS calibrated → PASS: PASS")


def test_2_nis_miscalibrated():
    """Miscalibrated NIS (all huge) should fail."""
    from core.evaluation.metrics import nis_calibration_chi2
    # NIS >> 1 everywhere → filter is overconfident
    nis = np.full(500, 10.0)
    result = nis_calibration_chi2(nis, dof=1)
    assert not result['pass_chi2'], \
        f"Miscalibrated NIS should fail, mean={result['mean_nis']}"
    print("  [2] NIS miscalibrated → FAIL: PASS")


def test_3_coverage_100():
    """Perfect estimates with wide σ should give ~100% coverage."""
    from core.evaluation.metrics import coverage_percentage
    n = 100
    freq_gt = np.linspace(0.15, 0.35, n)
    freq_est = freq_gt  # perfect match
    sigma = np.full(n, 0.1)  # wide uncertainty
    result = coverage_percentage(freq_est, freq_gt, sigma, k=2.0)
    assert result['coverage'] == 100.0, \
        f"Perfect + wide σ should give 100%, got {result['coverage']}%"
    print("  [3] Coverage 100%: PASS")


def test_4_coverage_0():
    """Terrible estimates with tiny σ should give ~0% coverage."""
    from core.evaluation.metrics import coverage_percentage
    n = 100
    freq_gt = np.full(n, 0.25)
    freq_est = np.full(n, 0.50)  # way off
    sigma = np.full(n, 0.001)  # very tight
    result = coverage_percentage(freq_est, freq_gt, sigma, k=2.0)
    assert result['coverage'] < 5.0, \
        f"Bad estimate + tiny σ should give ~0%, got {result['coverage']}%"
    print("  [4] Coverage ~0%: PASS")


def test_5_stability_long():
    """Stable frequency track should report long duration."""
    from core.evaluation.metrics import stability_duration
    fs = 30.0
    # 10 seconds of almost constant frequency
    n = int(10 * fs)
    freq = np.full(n, 0.25) + np.random.randn(n) * 0.001  # tiny jitter
    result = stability_duration(freq, fs, eps_hz=0.02)
    assert result['max_stable_sec'] > 5.0, \
        f"Stable track should give > 5s, got {result['max_stable_sec']:.1f}s"
    assert result['total_stable_pct'] > 90.0, \
        f"Should be > 90% stable, got {result['total_stable_pct']:.1f}%"
    print("  [5] Stability long: PASS")


def test_6_stability_short():
    """Highly unstable track should report short duration."""
    from core.evaluation.metrics import stability_duration
    fs = 30.0
    n = 300
    # Random walk → lots of big jumps
    np.random.seed(99)
    freq = np.cumsum(np.random.randn(n) * 0.1)
    result = stability_duration(freq, fs, eps_hz=0.02)
    assert result['max_stable_sec'] < 1.0, \
        f"Unstable track should give < 1s, got {result['max_stable_sec']:.1f}s"
    print("  [6] Stability short: PASS")


def test_7_import_consistency():
    """All 3 metrics are importable from metrics module."""
    from core.evaluation.metrics import (
        nis_calibration_chi2,
        coverage_percentage,
        stability_duration,
    )
    # Basic smoke: each should return a dict
    r1 = nis_calibration_chi2(np.random.chisquare(1, 100))
    r2 = coverage_percentage([0.25]*10, [0.25]*10, [0.1]*10)
    r3 = stability_duration([0.25]*10, 30.0)
    assert isinstance(r1, dict) and 'mean_nis' in r1
    assert isinstance(r2, dict) and 'coverage' in r2
    assert isinstance(r3, dict) and 'max_stable_sec' in r3
    print("  [7] Import + smoke test: PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 7: EXTENDED BAYESIAN METRICS TESTS")
    print("=" * 60)
    test_1_nis_calibrated()
    test_2_nis_miscalibrated()
    test_3_coverage_100()
    test_4_coverage_0()
    test_5_stability_long()
    test_6_stability_short()
    test_7_import_consistency()
    print("=" * 60)
    print("ALL 7 TESTS PASSED ✅")
    print("=" * 60)
