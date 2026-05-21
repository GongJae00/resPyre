import numpy as np

from components.models.heads.parh_ossm import oscillator_PARH_OSSM


def test_rate_source_arbiter_moves_only_when_posterior_is_specific() -> None:
    head = oscillator_PARH_OSSM()
    n = 80
    current = np.full(n, 0.30, dtype=float)
    native = np.full(n, 0.25, dtype=float)
    state = np.full(n, 0.25, dtype=float)
    output_conf = np.full(n, 0.20, dtype=float)
    posterior = {
        "mean_hz": np.full(n, 0.22, dtype=float),
        "mode_hz": np.full(n, 0.22, dtype=float),
        "confidence": np.full(n, 0.85, dtype=float),
        "entropy": np.full(n, 0.10, dtype=float),
        "top_gap": np.full(n, 0.85, dtype=float),
        "macro_support": np.full(n, 0.90, dtype=float),
        "direct_macro_support": np.full(n, 0.90, dtype=float),
        "motion_direct_support": np.full(n, 0.80, dtype=float),
        "alias_risk": np.zeros(n, dtype=float),
        "h1_role_support": np.full(n, 0.90, dtype=float),
        "abstain_pressure": np.zeros(n, dtype=float),
    }

    out, diag = head._rate_source_arbiter_v1_postprocess(
        current,
        native,
        state,
        current,
        output_conf,
        posterior,
        alpha_used=0.0,
    )

    assert np.mean(diag["blend"] > 0.0) > 0.95
    assert float(np.median(out)) < 0.30
    assert float(np.median(out)) > 0.22


def test_rate_source_arbiter_abstains_when_posterior_is_ambiguous() -> None:
    head = oscillator_PARH_OSSM()
    n = 80
    current = np.full(n, 0.30, dtype=float)
    native = np.full(n, 0.25, dtype=float)
    state = np.full(n, 0.25, dtype=float)
    output_conf = np.full(n, 0.80, dtype=float)
    posterior = {
        "mean_hz": np.full(n, 0.18, dtype=float),
        "mode_hz": np.full(n, 0.18, dtype=float),
        "confidence": np.full(n, 0.10, dtype=float),
        "entropy": np.full(n, 0.98, dtype=float),
        "top_gap": np.full(n, 0.03, dtype=float),
        "macro_support": np.full(n, 0.20, dtype=float),
        "direct_macro_support": np.full(n, 0.10, dtype=float),
        "alias_risk": np.full(n, 0.80, dtype=float),
        "h1_role_support": np.full(n, 0.10, dtype=float),
        "abstain_pressure": np.full(n, 0.90, dtype=float),
    }

    out, diag = head._rate_source_arbiter_v1_postprocess(
        current,
        native,
        state,
        current,
        output_conf,
        posterior,
        alpha_used=0.0,
    )

    assert np.allclose(out, current)
    assert np.all(diag["blend"] == 0.0)


def test_rate_source_arbiter_v2_blocks_native_state_alias_without_support() -> None:
    head = oscillator_PARH_OSSM()
    n = 80
    current = np.full(n, 0.22, dtype=float)
    native = np.full(n, 0.36, dtype=float)
    state = np.full(n, 0.36, dtype=float)
    output_conf = np.full(n, 0.30, dtype=float)
    posterior = {
        "mean_hz": np.full(n, 0.24, dtype=float),
        "mode_hz": np.full(n, 0.34, dtype=float),
        "confidence": np.full(n, 0.18, dtype=float),
        "entropy": np.full(n, 0.96, dtype=float),
        "top_gap": np.full(n, 0.05, dtype=float),
        "macro_support": np.full(n, 0.25, dtype=float),
        "direct_macro_support": np.full(n, 0.20, dtype=float),
        "motion_direct_support": np.full(n, 0.10, dtype=float),
        "alias_risk": np.full(n, 0.65, dtype=float),
        "h1_role_support": np.full(n, 0.25, dtype=float),
        "abstain_pressure": np.full(n, 0.85, dtype=float),
    }

    out_v1, diag_v1 = head._rate_source_arbiter_v1_postprocess(
        current,
        native,
        state,
        current,
        output_conf,
        posterior,
        alpha_used=0.0,
        guard_version="v1",
    )
    out_v2, diag_v2 = head._rate_source_arbiter_v1_postprocess(
        current,
        native,
        state,
        current,
        output_conf,
        posterior,
        alpha_used=0.0,
        guard_version="v2",
    )

    assert np.mean(diag_v1["blend"] > 0.0) > 0.95
    assert float(np.median(out_v1)) > 0.22
    assert np.allclose(out_v2, current)
    assert np.all(diag_v2["blend"] == 0.0)
    assert float(np.median(diag_v2["native_alias_safety"])) < 0.62


def test_rate_source_arbiter_v3_preserves_observable_native_regime() -> None:
    head = oscillator_PARH_OSSM()
    n = 80
    current = np.full(n, 0.23, dtype=float)
    native = np.full(n, 0.23, dtype=float)
    state = np.full(n, 0.24, dtype=float)
    output = np.full(n, 0.24, dtype=float)
    output_conf = np.full(n, 0.80, dtype=float)
    posterior = {
        "mean_hz": np.full(n, 0.30, dtype=float),
        "mode_hz": np.full(n, 0.30, dtype=float),
        "confidence": np.full(n, 0.40, dtype=float),
        "entropy": np.full(n, 0.55, dtype=float),
        "top_gap": np.full(n, 0.20, dtype=float),
        "macro_support": np.full(n, 0.50, dtype=float),
        "direct_macro_support": np.full(n, 0.50, dtype=float),
        "motion_direct_support": np.full(n, 0.35, dtype=float),
        "alias_risk": np.full(n, 0.10, dtype=float),
        "h1_role_support": np.full(n, 0.60, dtype=float),
        "abstain_pressure": np.full(n, 0.25, dtype=float),
    }

    out, diag = head._rate_source_arbiter_v1_postprocess(
        current,
        native,
        state,
        output,
        output_conf,
        posterior,
        alpha_used=0.0,
        guard_version="v3",
    )

    assert np.allclose(out, current)
    assert np.all(diag["blend"] == 0.0)
    assert float(np.median(diag["current_score"])) > float(np.median(diag["output_score"]))


def test_rate_source_arbiter_v3_allows_supported_external_rate_under_alias_conflict() -> None:
    head = oscillator_PARH_OSSM()
    n = 80
    current = np.full(n, 0.34, dtype=float)
    native = np.full(n, 0.34, dtype=float)
    state = np.full(n, 0.24, dtype=float)
    output = np.full(n, 0.22, dtype=float)
    output_conf = np.full(n, 0.45, dtype=float)
    posterior = {
        "mean_hz": np.full(n, 0.23, dtype=float),
        "mode_hz": np.full(n, 0.23, dtype=float),
        "confidence": np.full(n, 0.75, dtype=float),
        "entropy": np.full(n, 0.25, dtype=float),
        "top_gap": np.full(n, 0.65, dtype=float),
        "macro_support": np.full(n, 0.75, dtype=float),
        "direct_macro_support": np.full(n, 0.65, dtype=float),
        "motion_direct_support": np.full(n, 0.60, dtype=float),
        "alias_risk": np.full(n, 0.10, dtype=float),
        "h1_role_support": np.full(n, 0.75, dtype=float),
        "abstain_pressure": np.full(n, 0.15, dtype=float),
    }

    out, diag = head._rate_source_arbiter_v1_postprocess(
        current,
        native,
        state,
        output,
        output_conf,
        posterior,
        alpha_used=0.0,
        guard_version="v3",
    )

    assert np.mean(diag["blend"] > 0.0) > 0.95
    assert float(np.median(diag["output_score"])) > float(np.median(diag["current_score"]))
    assert float(np.median(out)) < 0.30
    assert float(np.median(out)) > 0.22
