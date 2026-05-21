# PARH Design Boundary Audit (2026-05-21)

This report classifies constants and command-line arguments by design role.
It is a guardrail against accidental dataset-specific hyperparameter search.

## Boundary Rule

- Fixed physiology/state definitions may be part of the method.
- Online-estimated reliability may be part of the method.
- Learned reliability must be trained without target GT selection.
- High-risk knobs cannot be tuned per dataset for paper-facing results.
- Ablation-only knobs can appear in diagnostics, not as promoted defaults.

## Summary

| Category | Count |
|---|---:|
| `ablation_flag` | 22 |
| `estimation_timescale` | 14 |
| `experimental_residual_policy` | 4 |
| `fixed_structure` | 2 |
| `frequency_harmonic_policy` | 2 |
| `high_risk_tuning_arg` | 3 |
| `high_risk_tuning_or_experimental` | 54 |
| `io_or_dataset` | 31 |
| `model_or_metric_boundary_arg` | 15 |
| `observation_equation_policy` | 28 |
| `other_arg` | 36 |
| `preprocessing_policy` | 5 |
| `readout_policy` | 2 |
| `reliability_mapping` | 27 |
| `runtime_resource` | 9 |
| `state_noise_scale` | 6 |

## ablation_flag

| Name | Value | Source | Note |
|---|---|---|---|
| `ENABLE_ADAPT_R` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_BASELINE` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_DISENTANGLED_Q` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_DYNAMIC_MIXTURE` | `False` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_FAMILY_CONFIDENCE` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_FREQ_ADAPT` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_FREQ_RESCUE` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_GROUP_BALANCED_FUSION` | `False` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_HARMONIC2` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_LEGACY_COUPLED_Q` | `False` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_OBS_CAL` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_PHASE_ANCHORED_MORPHOLOGY` | `False` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_RATE_OBSERVABILITY_HELPER` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_RATE_OBSERVABILITY_MIXTURE` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_RESIDUAL` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_RESIDUAL_IDENTIFIABILITY_GUARD` | `False` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_RESIDUAL_SEMANTICS` | `False` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_STUDENT_T` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `ENABLE_TARGET_OBSERVABILITY_CONTROL` | `False` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `P1D_FIXED_FAMILY_PRIOR` | `False` | `components/models/heads/parh_ossm.py` | legacy family-prior switch; off for paper-facing runs unless explicitly ablated |
| `USE_HELPER_PATH` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |
| `USE_LIGHT_OBS_PATH` | `True` | `components/models/heads/parh_ossm.py` | must be locked for paper-facing runs |

## estimation_timescale

| Name | Value | Source | Note |
|---|---|---|---|
| `FREQ_CONFIRM_COUNT` | `3` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `FREQ_INIT_SEC` | `10.0` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `FREQ_MAX_STEP_HZ` | `0.03` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `FREQ_UPDATE_INTERVAL_SEC` | `2.0` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `NU_MAX` | `200.0` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `NU_MIN` | `3.0` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `R_ANCHOR_FRAC` | `0.3` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `TAU_AMP_SEC` | `1.5` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `TAU_KAPPA_SEC` | `3.0` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `TAU_MAD_SEC` | `2.5` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `TAU_RESIDUAL_SEC` | `5.0` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `TAU_R_SEC` | `2.5` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `VB_ITERS` | `3` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |
| `WARMUP_SEC` | `2.5` | `components/models/heads/parh_ossm.py` | physiology/statistics timescale; sensitivity only, no target tuning |

## experimental_residual_policy

| Name | Value | Source | Note |
|---|---|---|---|
| `RESIDUAL_GUARD_NUISANCE_SCALE` | `0.8` | `components/models/heads/parh_ossm.py` | diagnostic unless promoted by four-regime validation |
| `RESIDUAL_GUARD_TRUST_FLOOR` | `0.2` | `components/models/heads/parh_ossm.py` | diagnostic unless promoted by four-regime validation |
| `RESIDUAL_PRIOR_MIN` | `0.1` | `components/models/heads/parh_ossm.py` | diagnostic unless promoted by four-regime validation |
| `RESIDUAL_PRIOR_POWER` | `1.0` | `components/models/heads/parh_ossm.py` | diagnostic unless promoted by four-regime validation |

## fixed_structure

| Name | Value | Source | Note |
|---|---|---|---|
| `STATE_DIM` | `8` | `components/models/heads/parh_ossm.py` | state-space structure or reference unit |
| `_REF_FPS` | `20.0` | `components/models/heads/parh_ossm.py` | state-space structure or reference unit |

## frequency_harmonic_policy

| Name | Value | Source | Note |
|---|---|---|---|
| `HARMONIC_ACF_RATIO` | `1.05` | `components/models/heads/parh_ossm.py` | locked harmonic disambiguation policy; no target-dataset tuning |
| `HARMONIC_POWER_RATIO` | `0.4` | `components/models/heads/parh_ossm.py` | locked harmonic disambiguation policy; no target-dataset tuning |

## high_risk_tuning_arg

| Name | Value | Source | Note |
|---|---|---|---|
| `--max-support-residual` | `` | `scripts/extract_target_reliability_graph_features.py` | candidate for no-sweep lock or ablation-only status |
| `--min-support-corr` | `` | `scripts/extract_target_reliability_graph_features.py` | candidate for no-sweep lock or ablation-only status |
| `--min-support-corr` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | candidate for no-sweep lock or ablation-only status |

## high_risk_tuning_or_experimental

| Name | Value | Source | Note |
|---|---|---|---|
| `DYNAMIC_MIXTURE_CONTEXT_FLOOR` | `0.2` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `DYNAMIC_MIXTURE_GLOBAL_QUALITY_FLOOR` | `0.05` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `DYNAMIC_MIXTURE_MIN_WEIGHT` | `0.02` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `DYNAMIC_MIXTURE_R_WEIGHT` | `0.5` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `DYNAMIC_MIXTURE_TAU_SEC` | `2.0` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `DYNAMIC_MIXTURE_TEMPERATURE` | `0.45` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FAMILY_CONFIDENCE_ALLOWED_FAMILIES` | `'profile1d_quadratic,profile1d_cubic,profile1d_quadratic_bridge,profile1d_cubic_bridge,profile1d_consensus'` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FAMILY_CONFIDENCE_MAX_FIT_RMSE` | `0.2` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FAMILY_CONFIDENCE_MIN_FIT_CORR` | `0.975` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FAMILY_CONFIDENCE_PI_FLOOR` | `0.97` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FAMILY_CONFIDENCE_QDYN_SCALE` | `0.55` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FAMILY_CONFIDENCE_R_SCALE` | `0.85` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FREQ_RESCUE_CONFIRM_COUNT` | `2` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FREQ_RESCUE_HELPER_STD_MAX_HZ` | `0.1` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FREQ_RESCUE_MAX_STEP_HZ` | `0.08` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FREQ_RESCUE_MIN_MISMATCH_HZ` | `0.07` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FREQ_RESCUE_MIN_QDYN` | `0.6` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FREQ_RESCUE_MIN_SUPPORT` | `0.75` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FREQ_RESCUE_POLICY` | `'bridge_v1'` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `FREQ_RESCUE_WINDOW_SEC` | `4.0` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `HELPER_TRUST_MIN_MISMATCH_HZ` | `0.03` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `HELPER_TRUST_MISMATCH_REF_HZ` | `0.06` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `HELPER_TRUST_POLICY` | `'off'` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `HELPER_TRUST_QDYN_FLOOR` | `0.3` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `HELPER_TRUST_RESCUE_MIN` | `0.45` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `HELPER_TRUST_STD_REF_HZ` | `0.07` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `HELPER_TRUST_WINDOW_SEC` | `4.0` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `OUTPUT_RATE_BIAS_MAX_CORR_HZ` | `0.05` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `OUTPUT_RATE_BIAS_MAX_HELPER_STD_HZ` | `0.08` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `OUTPUT_RATE_BIAS_MIN_SIGN_STABILITY` | `0.65` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `OUTPUT_RATE_BIAS_WIN_SEC` | `5.0` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `OUTPUT_RATE_BLEND_ALPHA` | `0.45` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `OUTPUT_RATE_MIN_MISMATCH_HZ` | `0.04` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `OUTPUT_RATE_MIN_QDYN` | `0.45` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `OUTPUT_RATE_MIN_SUPPORT` | `0.72` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `OUTPUT_RATE_POLICY` | `'of_helper_blend_v1'` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `PROFILE_RATE_BLEND_ALPHA` | `0.18` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `PROFILE_RATE_MAX_MISMATCH_HZ` | `0.1` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `PROFILE_RATE_MAX_QDYN` | `0.4` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `PROFILE_RATE_MIN_MISMATCH_HZ` | `0.025` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `PROFILE_RATE_MIN_SUPPORT` | `0.95` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `RATE_OBS_AGREE_REF_HZ` | `0.075` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `RATE_OBS_FLOOR` | `0.18` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `RATE_OBS_HARMONIC_PENALTY` | `0.58` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `RATE_OBS_HELPER_BLEND` | `0.65` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `RATE_OBS_HELPER_MAX_STEP_HZ` | `0.08` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `RATE_OBS_HELPER_MIN_SUPPORT` | `0.28` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `RATE_OBS_POWER` | `1.2` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `RATE_OBS_STD_REF_HZ` | `0.06` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `RATE_OBS_WINDOW_SEC` | `8.0` | `components/models/heads/parh_ossm.py` | do not tune per dataset; promote only after no-sweep validation |
| `TARGET_OBS_NUISANCE_R_SCALE` | `0.55` | `components/models/heads/parh_ossm.py` | target-computable observability control; promote only after no-sweep validation |
| `TARGET_OBS_QDYN_FLOOR` | `0.75` | `components/models/heads/parh_ossm.py` | target-computable observability control; promote only after no-sweep validation |
| `TARGET_OBS_ROLE_POWER` | `0.5` | `components/models/heads/parh_ossm.py` | target-computable observability control; promote only after no-sweep validation |
| `TARGET_OBS_TRUST_FLOOR` | `0.65` | `components/models/heads/parh_ossm.py` | target-computable observability control; promote only after no-sweep validation |

## io_or_dataset

| Name | Value | Source | Note |
|---|---|---|---|
| `--figure-out` | `` | `scripts/audit_external_weak_evidence.py` | I/O, dataset, or reporting path |
| `--out-csv` | `` | `scripts/audit_external_weak_evidence.py` | I/O, dataset, or reporting path |
| `--out-md` | `` | `scripts/audit_external_weak_evidence.py` | I/O, dataset, or reporting path |
| `--table-out` | `` | `scripts/audit_external_weak_evidence.py` | I/O, dataset, or reporting path |
| `--out-csv` | `` | `scripts/audit_final_paper_full_package.py` | I/O, dataset, or reporting path |
| `--out-md` | `` | `scripts/audit_final_paper_full_package.py` | I/O, dataset, or reporting path |
| `--package-csv` | `` | `scripts/audit_final_submission_readiness.py` | I/O, dataset, or reporting path |
| `--reference-csv` | `` | `scripts/audit_final_submission_readiness.py` | I/O, dataset, or reporting path |
| `--candidate-output` | `` | `scripts/extract_target_reliability_graph_features.py` | I/O, dataset, or reporting path |
| `--data-dir` | `` | `scripts/extract_target_reliability_graph_features.py` | I/O, dataset, or reporting path |
| `--dataset-label` | `` | `scripts/extract_target_reliability_graph_features.py` | I/O, dataset, or reporting path |
| `--out-edge` | `` | `scripts/extract_target_reliability_graph_features.py` | I/O, dataset, or reporting path |
| `--out-group` | `` | `scripts/extract_target_reliability_graph_features.py` | I/O, dataset, or reporting path |
| `--out-summary` | `` | `scripts/extract_target_reliability_graph_features.py` | I/O, dataset, or reporting path |
| `--report-out` | `` | `scripts/extract_target_reliability_graph_features.py` | I/O, dataset, or reporting path |
| `--out-dir` | `` | `scripts/generate_table_ready.py` | I/O, dataset, or reporting path |
| `--data-dir` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | I/O, dataset, or reporting path |
| `--enable-target-observability-control` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | I/O, dataset, or reporting path |
| `--name` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | I/O, dataset, or reporting path |
| `--out-run` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | I/O, dataset, or reporting path |
| `--readout-reliability-group-csv` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | I/O, dataset, or reporting path |
| `--reliability-group-csv` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | I/O, dataset, or reporting path |
| `--source-key` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | I/O, dataset, or reporting path |
| `--state-reliability-group-csv` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | I/O, dataset, or reporting path |
| `--analysis-dir` | `` | `scripts/run_final_baseline_comparator_refresh.py` | I/O, dataset, or reporting path |
| `--out-run` | `` | `scripts/run_final_baseline_comparator_refresh.py` | I/O, dataset, or reporting path |
| `--tables-dir` | `` | `scripts/run_final_baseline_comparator_refresh.py` | I/O, dataset, or reporting path |
| `--analysis-csv` | `` | `scripts/run_final_operating_point_sensitivity.py` | I/O, dataset, or reporting path |
| `--cohface-data-dir` | `` | `scripts/run_final_operating_point_sensitivity.py` | I/O, dataset, or reporting path |
| `--commands-out` | `` | `scripts/run_final_operating_point_sensitivity.py` | I/O, dataset, or reporting path |
| `--mahnob-data-dir` | `` | `scripts/run_final_operating_point_sensitivity.py` | I/O, dataset, or reporting path |

## model_or_metric_boundary_arg

| Name | Value | Source | Note |
|---|---|---|---|
| `--max-hz` | `` | `scripts/extract_target_reliability_graph_features.py` | semantic parameter; justify and lock |
| `--max-lag-sec` | `` | `scripts/extract_target_reliability_graph_features.py` | semantic parameter; justify and lock |
| `--min-hz` | `` | `scripts/extract_target_reliability_graph_features.py` | semantic parameter; justify and lock |
| `--window-sec` | `` | `scripts/extract_target_reliability_graph_features.py` | semantic parameter; justify and lock |
| `--window-stride-sec` | `` | `scripts/extract_target_reliability_graph_features.py` | semantic parameter; justify and lock |
| `--enable-decoupled-rate-readout` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | semantic parameter; justify and lock |
| `--enable-rate-hypothesis-graph` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | semantic parameter; justify and lock |
| `--enable-rate-hypothesis-graph-v4` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | semantic parameter; justify and lock |
| `--enable-rate-posterior` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | semantic parameter; justify and lock |
| `--external-rate-evidence-source` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | semantic parameter; justify and lock |
| `--lag-max-sec` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | semantic parameter; justify and lock |
| `--max-hz` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | semantic parameter; justify and lock |
| `--min-hz` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | semantic parameter; justify and lock |
| `--rate-posterior-output-source` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | semantic parameter; justify and lock |
| `--rate-track-source` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | semantic parameter; justify and lock |

## observation_equation_policy

| Name | Value | Source | Note |
|---|---|---|---|
| `OBS_CAL_ALLOWED_FAMILIES` | `'profile1d_quadratic,profile1d_cubic,profile1d_consensus'` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_MAX_FIT_RMSE_NORM` | `1.1` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_MAX_GAIN_AUX` | `0.45` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_MAX_GAIN_B` | `0.35` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_MAX_GAIN_H1` | `1.25` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_MAX_GAIN_H2` | `1.5` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_MAX_GAIN_OSC` | `4.0` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_MAX_GAIN_R` | `0.65` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_MAX_LAG_SEC` | `0.12` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_MIN_FIT_CORR` | `0.45` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_MODE` | `'family_phase_aux'` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_PRIOR_STRENGTH` | `1.5` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_RIDGE` | `0.05` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_SKIP_SEC` | `2.0` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OBS_CAL_WARMUP_SEC` | `12.0` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OF_FIXED_VELOCITY_PRIOR` | `False` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OF_HARMONIC_ONLY` | `True` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OF_MAX_FIT_RMSE_NORM` | `1.0` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OF_MAX_GAIN_H1` | `1.1` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OF_MAX_GAIN_H2` | `0.85` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OF_MAX_LAG_SEC` | `0.06` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OF_MIN_FIT_CORR` | `0.6` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `OF_PRIOR_STRENGTH` | `1.35` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `QUADCUB_HARMONIC_ONLY` | `True` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `QUADCUB_MAX_GAIN_H1` | `1.2` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `QUADCUB_MAX_GAIN_H2` | `1.6` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `QUADCUB_MAX_LAG_SEC` | `0.08` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |
| `QUADCUB_PRIOR_STRENGTH` | `1.7` | `components/models/heads/parh_ossm.py` | allowed only as a locked observation-law policy |

## other_arg

| Name | Value | Source | Note |
|---|---|---|---|
| `--allow-missing` | `` | `scripts/audit_external_weak_evidence.py` | manual review |
| `--scamps-manifest` | `` | `scripts/audit_external_weak_evidence.py` | manual review |
| `--v4v-manifest` | `` | `scripts/audit_external_weak_evidence.py` | manual review |
| `--allow-missing` | `` | `scripts/audit_final_paper_full_package.py` | manual review |
| `--checklist-md` | `` | `scripts/audit_final_submission_readiness.py` | manual review |
| `--package-md` | `` | `scripts/audit_final_submission_readiness.py` | manual review |
| `--reference-md` | `` | `scripts/audit_final_submission_readiness.py` | manual review |
| `--feature-fs` | `` | `scripts/extract_target_reliability_graph_features.py` | manual review |
| `--method` | `` | `scripts/extract_target_reliability_graph_features.py` | manual review |
| `--cohface-metrics` | `` | `scripts/generate_table_ready.py` | manual review |
| `--dataset-metrics` | `` | `scripts/generate_table_ready.py` | manual review |
| `--mahnob-metrics` | `` | `scripts/generate_table_ready.py` | manual review |
| `--anchor-policy` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--canonical-policy` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--enable-derived-consistency-scaling` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--enable-observation-law` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--enable-observation-law-v2` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--enable-phase-anchor-validation` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--enable-regime-observation-law` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--enable-signal-sqi-observability` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--eval-use-track` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--parh-base-method` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--parh-input` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--regime-anchor-policy` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--reliability-prior-scope` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--skip-eval` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--stride` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--win-size` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | manual review |
| `--dry-run` | `` | `scripts/run_final_baseline_comparator_refresh.py` | manual review |
| `--execute` | `` | `scripts/run_final_baseline_comparator_refresh.py` | manual review |
| `--representative-family` | `` | `scripts/run_final_baseline_comparator_refresh.py` | manual review |
| `--analysis-md` | `` | `scripts/run_final_operating_point_sensitivity.py` | manual review |
| `--execute` | `` | `scripts/run_final_operating_point_sensitivity.py` | manual review |
| `--only` | `` | `scripts/run_final_operating_point_sensitivity.py` | manual review |
| `--results-root` | `` | `scripts/run_final_operating_point_sensitivity.py` | manual review |
| `--skip-existing` | `` | `scripts/run_final_operating_point_sensitivity.py` | manual review |

## preprocessing_policy

| Name | Value | Source | Note |
|---|---|---|---|
| `OBS_BLEND_ALPHA` | `0.35` | `components/models/heads/parh_ossm.py` | preprocessing is part of the observation law |
| `OBS_CENTER_MODE` | `'median'` | `components/models/heads/parh_ossm.py` | preprocessing is part of the observation law |
| `OBS_CLIP_Z` | `6.0` | `components/models/heads/parh_ossm.py` | preprocessing is part of the observation law |
| `OBS_FAMILY_POLICY` | `'bridge_v1'` | `components/models/heads/parh_ossm.py` | preprocessing is part of the observation law |
| `OBS_LIGHT_LOWPASS_HZ` | `1.0` | `components/models/heads/parh_ossm.py` | preprocessing is part of the observation law |

## readout_policy

| Name | Value | Source | Note |
|---|---|---|---|
| `PHASE_MORPH_BINS` | `24.0` | `components/models/heads/parh_ossm.py` | phase-anchored morphology readout; fixed structural sensitivity, not target tuning |
| `PHASE_MORPH_MAX_BLEND` | `0.35` | `components/models/heads/parh_ossm.py` | phase-anchored morphology readout; fixed structural sensitivity, not target tuning |

## reliability_mapping

| Name | Value | Source | Note |
|---|---|---|---|
| `GATE_PHASE_SIGMA` | `0.8` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `GATE_WARMUP_SEC` | `0.75` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `QX_ADAPT_GAMMA_LEGACY` | `0.5` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `QX_ADAPT_WARMUP_SEC` | `3.0` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_APER_GAMMA` | `4.0` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_APER_OBS_GAMMA` | `0.0` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_DYN_AMP_WEIGHT` | `0.75` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_DYN_FREQ_REF_HZ` | `0.06` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_DYN_FREQ_WEIGHT` | `1.5` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_DYN_GAMMA` | `0.5` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OBS_JUMP_SCALE` | `3.0` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OBS_MIN` | `0.05` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OBS_ROBUST_Z_SCALE` | `2.0` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OSC_ERR_SIGMA` | `1.25` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OSC_FREQ_REF_HZ` | `0.08` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OSC_FREQ_WEIGHT` | `0.35` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OSC_HELPER_WEIGHT` | `0.45` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OSC_OBS_BAND` | `0.08` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OSC_OBS_MODE` | `'blend_support'` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OSC_OBS_REF` | `0.97` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OSC_OBS_WEIGHT` | `0.0` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `Q_OSC_PHASE_WEIGHT` | `0.2` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `STATE_ROLE_ABSTAIN_R_SCALE` | `0.35` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `STATE_ROLE_CONTEXT_MULTIPLIER_FLOOR` | `0.85` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `STATE_ROLE_CONTEXT_POWER` | `0.5` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `STATE_ROLE_RATE_MULTIPLIER_FLOOR` | `0.8` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |
| `STATE_ROLE_RATE_POWER` | `0.5` | `components/models/heads/parh_ossm.py` | should become normalized or online-estimated where possible |

## runtime_resource

| Name | Value | Source | Note |
|---|---|---|---|
| `--jobs` | `` | `scripts/extract_target_reliability_graph_features.py` | hardware/runtime control; not a model claim |
| `--max-files` | `` | `scripts/extract_target_reliability_graph_features.py` | hardware/runtime control; not a model claim |
| `--artifact-policy` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | hardware/runtime control; not a model claim |
| `--jobs` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | hardware/runtime control; not a model claim |
| `--max-files` | `` | `scripts/materialize_calibrated_multifamily_parh_system.py` | hardware/runtime control; not a model claim |
| `--artifact-policy` | `` | `scripts/run_final_baseline_comparator_refresh.py` | hardware/runtime control; not a model claim |
| `--artifact-policy` | `` | `scripts/run_final_operating_point_sensitivity.py` | hardware/runtime control; not a model claim |
| `--jobs` | `` | `scripts/run_final_operating_point_sensitivity.py` | hardware/runtime control; not a model claim |
| `--max-files` | `` | `scripts/run_final_operating_point_sensitivity.py` | hardware/runtime control; not a model claim |

## state_noise_scale

| Name | Value | Source | Note |
|---|---|---|---|
| `Q_BASELINE_POS` | `0.0001` | `components/models/heads/parh_ossm.py` | state flexibility prior; must be justified and locked |
| `Q_BASELINE_VEL` | `1e-05` | `components/models/heads/parh_ossm.py` | state flexibility prior; must be justified and locked |
| `Q_HARMONIC1_SCALE` | `1.0` | `components/models/heads/parh_ossm.py` | state flexibility prior; must be justified and locked |
| `Q_HARMONIC2_SCALE` | `0.5` | `components/models/heads/parh_ossm.py` | state flexibility prior; must be justified and locked |
| `Q_RESIDUAL_POS` | `0.1` | `components/models/heads/parh_ossm.py` | state flexibility prior; must be justified and locked |
| `Q_RESIDUAL_VEL` | `0.01` | `components/models/heads/parh_ossm.py` | state flexibility prior; must be justified and locked |

## Promotion Warning

High-risk tuning surfaces detected: `57`.

A paper-facing run should either:

1. lock these values before looking at target performance; or
2. move them into online estimation; or
3. report them only as ablation/sensitivity diagnostics.
