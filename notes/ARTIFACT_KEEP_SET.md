# Artifact Keep Set

Date: 2026-04-08

## Purpose

This file records the minimum artifact set that should remain in the repository
while the PARH-OSSM redesign is still active.

The goal is to reduce clutter without deleting the current live references or
the most important secondary evidence.

## Keep: results

- `results/20260407_cohface_full_obs_cal_v7`
  - current live single-family COHFACE reference
- `results/20260408_cohface_prod_ofbridge_full`
  - current official six-family COHFACE reference including `OF_bridge`
- `results/20260408_cohface_prod_ofbridge_familyconf_v3`
  - current live COHFACE reference with promoted narrow family-confidence policy
- `results/20260407_cohface_pairfusion_prod`
  - current secondary pair-fusion reference
- `results/20260407_singlefamily_obs_gate_v7`
  - gate evidence for promoted harmonic-only `obs_cal_v7`
- `results/20260407_of_rate_gate_alpha`
  - gate evidence for the rejected stronger OF output-rate policy
- `results/20260407_cohface_full_obs_cal_v7_ofrate_v2`
  - full-dataset no-go confirmation for the rejected stronger OF output-rate policy
- `results/20260407_helper_trust_gate`
  - original helper-trust no-go evidence
- `results/20260407_helper_trust_rescue_only_gate`
  - rescue-only helper-trust no-go evidence
- `results/20260407_of_bridge_gate_subset`
  - OF displacement-bridge promotion gate evidence
- `results/20260407_of_bridge_full`
  - full COHFACE OF raw vs OF bridge validation
- `results/20260408_of_fixed_velocity_prior_gate_v2`
  - corrected raw-OF fixed-velocity-fallback no-go evidence
- `results/20260408_residual_release_v4_gate`
  - observation-driven residual-gap no-go evidence
- `results/20260408_family_confidence_gate_v3`
  - promotion-gate evidence for the current narrow family-confidence policy
- `results/20260408_residual_semantics_gate_rerun`
  - clean rerun for diagnostic-only residual-semantics evidence
- `results/20260408_of_dualtrack_gate`
  - raw-OF plus OF-bridge assistant no-go evidence
- `results/20260408_p1dquad_ofbridge_assist_gate`
  - P1D-quad plus OF-bridge assistant no-go evidence
- `results/20260408_mahnob_subset_ofbridge_gate`
  - MAHNOB irregular-regime subset gate, in progress until explicitly closed

Everything else under `results/` is considered disposable unless a new lock
document explicitly promotes it.

## Keep: analysis

Core reference files:

- `analysis/output_metric_mapping_spec.md`
- `analysis/parh_waveform_protocol_lock.md`
- `analysis/t4_policy_reconciliation_report.md`

Observation EDA:

- `analysis/cohface_preproc_summary.csv`
- `analysis/cohface_preproc_deltas.csv`
- `analysis/cohface_observation_eda_family.csv`
- `analysis/cohface_observation_eda_trials.csv`
- `analysis/cohface_observation_redesign_findings_20260403.md`
- `analysis/cohface_observation_family_map_report_20260408.md`
- `analysis/respyre_benchmark_alignment_note_20260408.md`

Current design-decision reports:

- `analysis/cohface_current_scaffold_report_20260406.md`
- `analysis/cohface_gate_subset_obs_cal_v7_findings_20260407.md`
- `analysis/cohface_gate_output_rate_of_v2_findings_20260407.md`
- `analysis/cohface_full_of_output_rate_v2_findings_20260407.md`
- `analysis/cohface_pairfusion_gate_report_20260407.md`
- `analysis/cohface_pairfusion_full_report_20260407.md`
- `analysis/cohface_assistfusion_gate_report_20260407.md`
- `analysis/cohface_pair_of_velocity_gate_report_20260407.md`
- `analysis/cohface_helper_trust_gate_report_20260407.md`
- `analysis/cohface_of_bridge_gate_report_20260407.md`
- `analysis/cohface_of_bridge_full_report_20260408.md`
- `analysis/cohface_allfamily_ofbridge_report_20260408.md`
- `analysis/cohface_of_fixed_velocity_prior_gate_report_20260408.md`
- `analysis/cohface_residual_release_v4_gate_report_20260408.md`
- `analysis/cohface_residual_semantics_v1_gate_report_20260408.md`
- `analysis/cohface_of_dualtrack_gate_report_20260408.md`
- `analysis/cohface_p1dquad_ofbridge_assist_gate_report_20260408.md`
- `analysis/cohface_profile_harmonic_rate_v1_gate_report_20260408.md`

## Keep: paper manifests and tables

Manifests:

- `paper/manifests/cohface_case_study_manifest.csv`
- `paper/manifests/cohface_parh_mechanism_trials.csv`

Tables:

- `paper/tables_ready/T2_observation_family_map.csv`
- `paper/tables_ready/T3_rate_main.csv`
- `paper/tables_ready/T4_waveform_main.csv`
- `paper/tables_ready/T6_diagnostics_main.csv`
- `paper/tables_ready/T6b_cohface_mechanism_audit.csv`
- `paper/tables_ready/T6b_fusion_ladder.csv`

## Notes

- MAHNOB full remains blocked, but the subset gate directory should be kept
  until its verdict is written down and either promoted or explicitly rejected.
- Temporary smoke runs, duplicate dated CSVs, and superseded gate outputs
  should be removed aggressively.
- `results/20260408_of_fixed_velocity_prior_gate` is superseded by the
  corrected `..._gate_v2` rerun and should not be kept.
