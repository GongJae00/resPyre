from scripts.materialize_calibrated_multifamily_parh_system import _rate_posterior_output_role


def test_rate_posterior_output_roles_lock_paper_boundary() -> None:
    assert _rate_posterior_output_role("off") == "native_state_update_only"
    assert _rate_posterior_output_role("final") == "paper_candidate_bounded_readout"
    assert _rate_posterior_output_role("calibrated_mean") == "paper_candidate_bounded_readout"
    assert _rate_posterior_output_role("source_arbiter_v1") == "diagnostic_or_legacy_not_paper_facing"
    assert _rate_posterior_output_role("source_arbiter_v2") == "diagnostic_or_legacy_not_paper_facing"
    assert _rate_posterior_output_role("source_arbiter_v3") == "diagnostic_or_legacy_not_paper_facing"
    assert _rate_posterior_output_role("mean") == "diagnostic_or_legacy_not_paper_facing"
    assert _rate_posterior_output_role("source_validity_guarded") == "diagnostic_or_legacy_not_paper_facing"
