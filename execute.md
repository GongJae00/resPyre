# Final Paper Execution Commands

This file keeps only the current paper-facing execution path. Here, `full`
means the full reproducible paper package: dataset links, weak external
evidence, guard audits, real benchmark runs, post-run diagnostics, ablations,
figures/tables, manuscript build, and a final package audit.

## Environment

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PY="${PY:-python}"
eval "$($PY setup/auto_profile.py --mode cpu_batch --device cpu --write-json analysis/final_resource_profile.json)"
eval "$($PY setup/locked_paper_profile.py)"
export COHFACE_PREPARED_DATA_DIR="${COHFACE_PREPARED_DATA_DIR:?set path to prepared COHFACE data directory}"
export MAHNOB_TAILALIGNED_DATA_DIR="${MAHNOB_TAILALIGNED_DATA_DIR:?set path to prepared MAHNOB tail-aligned data directory}"
```

The final paper package is CPU-batch oriented. `--device cpu` avoids accidental
GPU contention from unrelated desktop/deep-learning processes while preserving
parallel CPU throughput (`RESPYRE_JOBS`/`PARALLEL_PROCS`).

## 1. Dataset Links And Experiment Assets

Run this before launching the paper-package validation path. It creates/audits
dataset symlinks and writes the paper-facing dataset-scope, ablation-contract,
external-manifest, and scope figure assets.

```bash
$PY scripts/build_rr_experiment_assets.py \
  --create-symlinks
```

Outputs:

```text
analysis/rr_dataset_symlink_audit.csv
analysis/rr_dataset_scope.csv
analysis/v4v_rr_rate_manifest.csv
analysis/scamps_rr_synthetic_manifest.csv
analysis/rr_ablation_design_contract.csv
analysis/rr_experiment_blueprint.md
analysis/final_resource_profile.json
paper/tables_ready/T1_dataset_protocol_scope.csv
paper/tables_ready/S_T_rr_ablation_design_contract.csv
paper/tables_ready/S_T_external_rr_manifest_summary.csv
paper/figures/S_F_rr_dataset_scope_map.pdf
```

Boundary: COHFACE and MAHNOB-HCI are the real waveform/rate benchmarks. V4V is
external RR-rate-only validation. SCAMPS is synthetic diagnostic/control
evidence. Do not pool V4V or SCAMPS into real waveform performance claims.

## 2. External Weak-Evidence Audit

This is mandatory for the final paper package. It does not add V4V/SCAMPS to
the main real-data performance tables. It records exactly which weak evidence
is allowed and which claims are prohibited.

```bash
$PY scripts/audit_external_weak_evidence.py \
  --v4v-manifest analysis/v4v_rr_rate_manifest.csv \
  --scamps-manifest analysis/scamps_rr_synthetic_manifest.csv \
  --out-csv analysis/external_weak_evidence_audit.csv \
  --out-md analysis/external_weak_evidence_audit.md \
  --table-out paper/tables_ready/S_T_external_weak_evidence_audit.csv \
  --figure-out paper/figures/S_F_external_weak_evidence_summary.pdf
```

Outputs:

```text
analysis/external_weak_evidence_audit.csv
analysis/external_weak_evidence_audit.md
paper/tables_ready/S_T_external_weak_evidence_audit.csv
paper/figures/S_F_external_weak_evidence_summary.pdf
```

## 2b. Dataset Distribution EDA

This paper-facing EDA uses all four audited datasets without changing the
main benchmark boundary. COHFACE/MAHNOB-HCI provide real waveform/rate
regime distributions, V4V provides real RR-rate-only label distribution, and
SCAMPS provides controlled synthetic `d_br` peak-rate distribution. These
outputs support dataset-scope interpretation and must not be treated as
additional real waveform performance rows.

```bash
$PY scripts/generate_dataset_distribution_eda.py
$PY scripts/plot_observation_eda.py
```

Outputs:

```text
analysis/dataset_rate_distribution_eda.csv
analysis/scamps_rr_signal_eda.csv
analysis/dataset_distribution_eda.csv
analysis/dataset_distribution_eda.md
paper/tables_ready/S_T_dataset_distribution_eda.csv
paper/figures/S_F_dataset_distribution_eda.pdf
paper/figures/F2_dataset_and_observation_regime.pdf
paper/figures/S_F5_preproc_delta_heatmaps.pdf
```

## 3. Pre-Full Guard Checks

Run these before launching validation. They do not tune the model; they verify
that the paper-facing path has not drifted into dataset-specific sweeping,
hidden target-GT selection, or broken runtime contracts.

```bash
$PY scripts/audit_parh_design_boundary.py \
  --out-md analysis/pre_full_design_boundary_check.md \
  --json-out analysis/pre_full_design_boundary_check.json

$PY scripts/audit_learning_boundary.py \
  --out-md analysis/pre_full_learning_boundary_check.md \
  --out-csv analysis/pre_full_learning_boundary_check.csv

$PY scripts/audit_pre_full_paper_contract.py \
  --out-csv analysis/pre_full_paper_contract_audit.csv \
  --out-md analysis/pre_full_paper_contract_audit.md

$PY -m pytest -q \
  tests/test_design_boundary_audit.py \
  tests/test_external_weak_evidence_audit.py \
  tests/test_learning_boundary_audit.py \
  tests/test_pre_full_paper_contract_audit.py \
  tests/test_observation_family_order_contract.py \
  tests/test_observation_law_contract.py \
  tests/test_paper_candidate_activation_contract.py \
  tests/test_rate_posterior_output_role.py \
  tests/test_storage_config_runtime.py \
  tests/test_runtime_status_metadata.py
```

## 4. Bounded Operating-Point Sensitivity

This step is recommended before the complete-cohort rerun if we want to check
performance sensitivity without turning the paper into dataset-specific
hyperparameter tuning. The grid is fixed and small. It only changes
semantically interpretable label-free reliability settings:
window length/stride and cross-source support strictness.

Dry-run the plan first:

```bash
$PY scripts/run_final_operating_point_sensitivity.py \
  --max-files 32 \
  --jobs "${RESPYRE_JOBS:-1}" \
  --artifact-policy lean
```

Execute the fixed sensitivity grid:

```bash
$PY scripts/run_final_operating_point_sensitivity.py \
  --execute \
  --skip-existing \
  --max-files 32 \
  --jobs "${RESPYRE_JOBS:-1}" \
  --artifact-policy lean
```

Outputs:

```text
analysis/final_operating_point_sensitivity.csv
analysis/final_operating_point_sensitivity.md
analysis/final_operating_point_sensitivity_commands.sh
analysis/final_operating_point_sensitivity_interpretation.md
analysis/final_operating_point_research_feedback.md
analysis/final_operating_point_decision_record.md
results/final_operating_point_sensitivity/
```

Boundary: this is not a best-of-MAHNOB sweep. If one setting is adopted for the
complete-cohort rerun, it must be justified as a globally defensible operating point
before target-GT performance is used, and the sensitivity table must remain in
the audit trail.

Implementation note: `--artifact-policy lean` still writes strict waveform
metrics during evaluation. Standalone strict regeneration is
only needed when retained PKL artifacts are available.

## 5. Windowed Label-Free Reliability Priors

These CSVs are required by the final paper validation path. They are label-free
with respect to the respiratory reference: they estimate which observation
classes are locally trustworthy from cross-source agreement, timing support,
morphology support, nuisance evidence, and harmonic ambiguity. The materializer
fails closed if these windowed rows are missing.

```bash
$PY scripts/extract_target_reliability_graph_features.py \
  --data-dir "$COHFACE_PREPARED_DATA_DIR" \
  --dataset-label COHFACE \
  --data-dir "$MAHNOB_TAILALIGNED_DATA_DIR" \
  --dataset-label MAHNOB_tailaligned \
  --out-edge analysis/final_priors/cohface_mahnob_target_reliability_windowed_edges.csv \
  --out-group analysis/final_priors/cohface_mahnob_target_computable_reliability_windowed.csv \
  --out-summary analysis/final_priors/cohface_mahnob_target_computable_reliability_windowed_summary.csv \
  --report-out analysis/final_priors/cohface_mahnob_target_computable_reliability_windowed.md \
  --window-sec 30 \
  --window-stride-sec 10 \
  --jobs "${RESPYRE_JOBS:-1}"

$PY scripts/extract_target_reliability_graph_features.py \
  --data-dir "$MAHNOB_TAILALIGNED_DATA_DIR" \
  --dataset-label MAHNOB_tailaligned \
  --out-edge analysis/final_priors/mahnob_target_reliability_windowed_edges.csv \
  --out-group analysis/final_priors/mahnob_target_computable_reliability_windowed.csv \
  --out-summary analysis/final_priors/mahnob_target_computable_reliability_windowed_summary.csv \
  --report-out analysis/final_priors/mahnob_target_computable_reliability_windowed.md \
  --window-sec 30 \
  --window-stride-sec 10 \
  --jobs "${RESPYRE_JOBS:-1}"

cp analysis/final_priors/mahnob_target_computable_reliability_windowed.csv \
   analysis/final_priors/mahnob_state_component_reliability_windowed.csv
```

The MAHNOB windowed reliability CSV carries both readout scores and state-role
weights. The explicit copy keeps the materializer interface honest: state
control and readout control are separate inputs even when they are estimated
from the same label-free window evidence.

## 6. Main Real-Data Paper-Candidate Validation

The final paper-candidate path is the complete-cohort validation. It evaluates
every available COHFACE and MAHNOB-HCI tail-aligned trial from the prepared data
roots, with no `--max-files` cap. Bounded 32-trial runs remain diagnostic
sensitivity checks only and must not be described as the submission result.

The main paper tables report the integrated PARH-OSSM complete-cohort outputs
alongside a transparent same-trial baseline/comparator refresh: pre-specified
`P1D_quad direct`, `OSSM-KF (P1D quad)`, and PARH-OSSM. The full
eight-observation-class comparison remains supplementary diagnostic context and
must not be used as a post hoc best-observation-class main-table selector.

COHFACE:

```bash
$PY scripts/materialize_calibrated_multifamily_parh_system.py \
  --data-dir "$COHFACE_PREPARED_DATA_DIR" \
  --out-run results/final_full_validation/cohface \
  --name parh_ossm \
  --reliability-group-csv analysis/final_priors/cohface_mahnob_target_computable_reliability_windowed.csv \
  --reliability-prior-scope readout_only \
  --parh-input multichannel \
  --external-rate-evidence-source ossm_kfstd \
  --enable-observation-law \
  --enable-rate-posterior \
  --enable-target-observability-control \
  --enable-signal-sqi-observability \
  --rate-posterior-output-source final \
  --eval-use-track \
  --jobs "${RESPYRE_JOBS:-1}" \
  --artifact-policy full
```

MAHNOB tail-aligned:

```bash
$PY scripts/materialize_calibrated_multifamily_parh_system.py \
  --data-dir "$MAHNOB_TAILALIGNED_DATA_DIR" \
  --out-run results/final_full_validation/mahnob_tailaligned \
  --name parh_ossm \
  --state-reliability-group-csv analysis/final_priors/mahnob_state_component_reliability_windowed.csv \
  --readout-reliability-group-csv analysis/final_priors/mahnob_target_computable_reliability_windowed.csv \
  --parh-input multichannel \
  --external-rate-evidence-source ossm_kfstd \
  --enable-observation-law \
  --enable-rate-posterior \
  --enable-target-observability-control \
  --enable-signal-sqi-observability \
  --rate-posterior-output-source final \
  --eval-use-track \
  --jobs "${RESPYRE_JOBS:-1}" \
  --artifact-policy full
```

## 7. Required Post-Run Evaluation And Audits

Regenerate the strict/cycle companions and paper-ready comparison tables
immediately after the two real-data runs. This step is required because the main
aligned waveform table and the strict zero-lag/cycle companions answer
different questions. Do not use `scripts/generate_table_ready.py` alone for the
final submission tables, because the integrated PARH runs contain only the
`parh_ossm` method and would leave Base and OSSM-KF columns empty.

```bash
$PY scripts/generate_waveform_strict_metrics.py \
  --data-dir results/final_full_validation/cohface/data \
  --out-dir results/final_full_validation/cohface/metrics

$PY scripts/generate_waveform_strict_metrics.py \
  --data-dir results/final_full_validation/mahnob_tailaligned/data \
  --out-dir results/final_full_validation/mahnob_tailaligned/metrics

$PY scripts/run_final_baseline_comparator_refresh.py --dry-run

$PY scripts/run_final_baseline_comparator_refresh.py \
  --execute \
  --artifact-policy full

$PY scripts/audit_parh_rate_source_decomposition.py \
  --data-dir results/final_full_validation/cohface/data \
  --out-csv analysis/final_cohface_rate_source_decomposition.csv \
  --out-md analysis/final_cohface_rate_source_decomposition.md

$PY scripts/audit_parh_rate_source_decomposition.py \
  --data-dir results/final_full_validation/mahnob_tailaligned/data \
  --out-csv analysis/final_mahnob_tailaligned_rate_source_decomposition.csv \
  --out-md analysis/final_mahnob_tailaligned_rate_source_decomposition.md

$PY scripts/audit_target_observability_failure_modes.py \
  --decomposition-csv analysis/final_mahnob_tailaligned_rate_source_decomposition.csv \
  --feature-csv results/final_full_validation/mahnob_tailaligned/metrics/readout_guard_raw.csv \
  --out-csv analysis/final_mahnob_tailaligned_observability_failure_modes.csv \
  --out-md analysis/final_mahnob_tailaligned_observability_failure_modes.md

$PY scripts/generate_observability_failure_taxonomy_table.py \
  --failure-csv analysis/final_mahnob_tailaligned_observability_failure_modes.csv \
  --out-csv paper/tables_ready/T7_observability_failure_taxonomy.csv \
  --report-out analysis/final_mahnob_tailaligned_observability_failure_taxonomy.md
```

Additional outputs from the baseline/comparator refresh:

```text
results/final_baseline_comparator_refresh/
analysis/final_baseline_comparator_refresh.csv
analysis/final_baseline_comparator_refresh.md
analysis/final_submission_gap_audit.csv
analysis/final_submission_gap_audit.md
analysis/final_metric_provenance_audit.csv
analysis/final_metric_provenance_audit.md
analysis/final_split_and_leakage_audit.csv
analysis/final_split_and_leakage_audit.md
analysis/final_statistical_comparison.csv
analysis/final_statistical_comparison.md
analysis/final_baseline_comparison_interpretation.md
analysis/final_submission_claim_boundary.md
analysis/final_reviewer_risk_response.md
analysis/final_reproducibility_audit.csv
analysis/final_reproducibility_audit.md
paper/tables_ready/S_T_final_allfamily_comparison.csv
```

## 8. Paper Evidence Assets

After the validation tables are regenerated, rebuild the operator-alignment
ablation companions used to explain which detachable observation/model blocks
are contributing evidence:

```bash
$PY scripts/generate_t5_operator_alignment_ablation.py
$PY scripts/generate_unified_split_ablation.py
```

Outputs:

```text
paper/tables_ready/T5_operator_alignment_ablation.csv
paper/tables_ready/S_T7_unified_split_operator_alignment.csv
```

Then rebuild the component-ablation evidence table and compact supplementary
figure from the fresh table-ready assets:

```bash
$PY scripts/generate_component_ablation_evidence.py
```

Expected outputs:

```text
paper/tables_ready/T5_component_ablation_evidence.csv
analysis/component_ablation_evidence.csv
analysis/component_ablation_evidence.md
paper/figures/S_F_component_ablation_evidence.pdf
```

## 9. Manuscript Build

```bash
(cd paper && latexmk -pdf -interaction=nonstopmode main.tex)

$PY scripts/audit_paper_code_consistency.py \
  --out-csv analysis/paper_code_consistency_audit.csv \
  --out-md analysis/paper_code_consistency_audit.md
```

## 10. Final Package Audit

Run this last. It fails if the final paper package is incomplete.

```bash
$PY scripts/audit_final_paper_full_package.py \
  --out-csv analysis/final_paper_full_package_audit.csv \
  --out-md analysis/final_paper_full_package_audit.md
```

Outputs:

```text
analysis/final_paper_full_package_audit.csv
analysis/final_paper_full_package_audit.md
```

## Boundary

The paper-facing path is:

```text
fixed observation classes
-> adaptive observation law
-> PARH-OSSM
-> final bounded z_osc rate readout
-> z_full waveform diagnostics
```

`ossm_kfstd` is used only as an external timing-evidence comparator channel.
It is not a nested fallback model, and it does not replace the PARH-OSSM state
update or waveform reconstruction path. Legacy source-arbiter and diagnostic
posterior-output variants are not part of the final execution path.
