#!/usr/bin/env python
"""
Phase 0c + Phase 1 acceptance tests.

T1: Extended EDA diagnostics function works
T2: QualityEstimator produces valid 6D vector
T3: QualityEstimator detects outliers
T4: QualityEstimator detects bursts
T5: QualityEstimator THD computation
T6: default_quality() consistency between quality.py and trust.py
T7: QualityEstimator integrates with TrustAllocator
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

QUALITY_KEYS = {'q_vis', 'q_drift', 'q_cons', 'q_out', 'q_harm', 'q_burst'}


def test_1_extended_eda():
    """analyze_step4_residual returns all 5 new diagnostic fields."""
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analysis')))
    from analysis.run_noise_analysis import analyze_step4_residual

    # Generate a signal with heavy tails
    np.random.seed(42)
    from scipy.stats import t as t_dist
    heavy = t_dist.rvs(df=3.0, size=500)
    result = analyze_step4_residual(heavy)

    assert 'kurtosis' in result
    assert 'innovation' in result
    assert 't_aic_delta' in result, "Missing t_aic_delta"
    assert 't_fit_nu' in result, "Missing t_fit_nu"
    assert 'hill_alpha' in result, "Missing hill_alpha"
    assert 'arch_lm_pval' in result, "Missing arch_lm_pval"
    assert 'ljung_box_pval' in result, "Missing ljung_box_pval"
    assert 'shapiro_pval' in result, "Missing shapiro_pval"

    # For heavy-tailed data: kurtosis should be high, shapiro should reject Gaussian
    assert result['kurtosis'] > 1.0, f"κ should be > 1 for t(3), got {result['kurtosis']}"
    if np.isfinite(result['shapiro_pval']):
        assert result['shapiro_pval'] < 0.05, \
            f"Shapiro should reject Gaussian for t(3), p={result['shapiro_pval']}"

    print("  [1] Phase 0c extended EDA diagnostics: PASS")


def test_2_quality_6d_valid():
    """QualityEstimator.update() returns valid 6D dict with correct keys."""
    from components.observations.quality import QualityEstimator

    qe = QualityEstimator(fs=30.0)
    # Run 60 frames (2 seconds)
    for t in range(60):
        q = qe.update(t, np.sin(2 * np.pi * 0.25 * t / 30.0))

    # Check structure
    assert set(q.keys()) == QUALITY_KEYS, f"Keys mismatch: {q.keys()}"
    # All values should be finite
    for k, v in q.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"

    print("  [2] QualityEstimator 6D vector valid: PASS")


def test_3_quality_outlier():
    """q_out should spike toward 1.0 for an extreme observation."""
    from components.observations.quality import QualityEstimator

    qe = QualityEstimator(fs=30.0)
    # Feed 50 normal frames
    for t in range(50):
        q = qe.update(t, 0.5 * np.sin(2 * np.pi * 0.25 * t / 30.0))

    # Now inject a massive outlier
    q_outlier = qe.update(50, 100.0)  # 100x bigger than typical ~0.5
    assert q_outlier['q_out'] > 0.8, \
        f"q_out should be high for extreme outlier, got {q_outlier['q_out']}"
    assert q_outlier['q_out'] <= 1.0, \
        f"q_out must be normalized to [0,1], got {q_outlier['q_out']}"

    # Normal frame after that
    q_normal = qe.update(51, 0.3)
    assert q_normal['q_out'] < q_outlier['q_out'], \
        "q_out should drop back after outlier passes"

    print("  [3] QualityEstimator outlier detection: PASS")


def test_4_quality_burst():
    """q_burst should be 1.0 for a sudden impulse."""
    from components.observations.quality import QualityEstimator

    qe = QualityEstimator(fs=30.0)
    # Feed 30 frames of quiet signal
    for t in range(30):
        qe.update(t, 0.01 * np.random.randn())

    # Inject a huge burst
    q = qe.update(30, 50.0)
    assert q['q_burst'] == 1.0, f"q_burst should be 1.0 for impulse, got {q['q_burst']}"

    # Normal frame
    q2 = qe.update(31, 0.01)
    assert q2['q_burst'] == 0.0, f"q_burst should be 0.0 for normal, got {q2['q_burst']}"

    print("  [4] QualityEstimator burst detection: PASS")


def test_5_quality_thd():
    """q_harm should detect harmonic content."""
    from components.observations.quality import QualityEstimator

    qe = QualityEstimator(fs=30.0, f_min=0.08, f_max=0.5)
    # Signal with strong 2nd harmonic: sin(0.25 Hz) + 0.5*sin(0.5 Hz)
    for t in range(150):  # 5 seconds
        y = np.sin(2 * np.pi * 0.25 * t / 30.0) + 0.5 * np.sin(2 * np.pi * 0.5 * t / 30.0)
        q = qe.update(t, y)

    # q_harm should be > 0 due to harmonic content
    assert q['q_harm'] > 0.0, f"q_harm should detect harmonics, got {q['q_harm']}"

    # Compare with pure sinusoid (no harmonics)
    qe2 = QualityEstimator(fs=30.0, f_min=0.08, f_max=0.5)
    for t in range(150):
        y2 = np.sin(2 * np.pi * 0.25 * t / 30.0)
        q2 = qe2.update(t, y2)

    # Pure sine should have lower q_harm
    assert q2['q_harm'] < q['q_harm'], \
        f"Pure sine THD ({q2['q_harm']}) should be < harmonic ({q['q_harm']})"

    print("  [5] QualityEstimator THD computation: PASS")


def test_6_default_quality_consistency():
    """default_quality() in quality.py and trust.py should match."""
    from components.observations.quality import default_quality as dq1
    from components.models.core.trust import default_quality as dq2

    assert dq1() == dq2(), "default_quality() inconsistent between quality.py and trust.py"

    print("  [6] default_quality consistency: PASS")


def test_7_quality_trust_integration():
    """QualityEstimator output is accepted by TrustAllocator."""
    from components.observations.quality import QualityEstimator
    from components.models.core.trust import TrustAllocator

    qe = QualityEstimator(fs=30.0)
    ta = TrustAllocator()

    # Run 60 frames
    for t in range(60):
        q = qe.update(t, np.sin(2 * np.pi * 0.25 * t / 30.0))

    # Pass quality to trust allocator — should not raise
    try:
        trust = ta.allocate(q, nis=1.5)
    except Exception as e:
        raise AssertionError(f"TrustAllocator.allocate(qe.update(...)) raised: {e}")

    assert hasattr(trust, 'alpha_R'), "Missing alpha_R from trust"
    assert hasattr(trust, 'w_h'), "Missing w_h from trust"

    print("  [7] Quality → Trust integration: PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("PHASE 0c + PHASE 1 TESTS")
    print("=" * 60)
    test_1_extended_eda()
    test_2_quality_6d_valid()
    test_3_quality_outlier()
    test_4_quality_burst()
    test_5_quality_thd()
    test_6_default_quality_consistency()
    test_7_quality_trust_integration()
    print("=" * 60)
    print("ALL 7 TESTS PASSED ✅")
    print("=" * 60)
