# Reproduction Commands

This file records the public reproduction path for the PARH-OSSM experiments.
It assumes prepared COHFACE and MAHNOB-HCI inputs are available locally. Raw
datasets are not included in this repository.

## Environment

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
export PY="${PY:-python}"
eval "$($PY setup/auto_profile.py --mode cpu_batch --device cpu --write-json analysis/final_resource_profile.json)"
eval "$($PY setup/locked_parh_profile.py)"
export COHFACE_PREPARED_DATA_DIR="${COHFACE_PREPARED_DATA_DIR:?set COHFACE prepared-data path}"
export MAHNOB_TAILALIGNED_DATA_DIR="${MAHNOB_TAILALIGNED_DATA_DIR:?set MAHNOB-HCI tail-aligned prepared-data path}"
```

## Dataset Scope

```bash
$PY scripts/build_rr_experiment_assets.py --create-symlinks

$PY scripts/audit_external_weak_evidence.py \
  --v4v-manifest analysis/v4v_rr_rate_manifest.csv \
  --scamps-manifest analysis/scamps_rr_synthetic_manifest.csv \
  --out-csv analysis/external_weak_evidence_audit.csv \
  --out-md analysis/external_weak_evidence_audit.md

$PY scripts/generate_dataset_distribution_eda.py
$PY scripts/plot_observation_eda.py
```

COHFACE and MAHNOB-HCI are the real waveform/rate benchmarks. V4V is
rate-only external evidence. SCAMPS is synthetic diagnostic evidence.

## Guard Checks

```bash
$PY scripts/audit_parh_design_boundary.py \
  --out-md analysis/pre_full_design_boundary_check.md \
  --json-out analysis/pre_full_design_boundary_check.json

$PY scripts/audit_learning_boundary.py \
  --out-md analysis/pre_full_learning_boundary_check.md \
  --out-csv analysis/pre_full_learning_boundary_check.csv
```

These checks verify that the locked PARH-OSSM path is not using target labels
for the adaptive observation law.

## Reliability Priors

```bash
mkdir -p analysis/final_priors

$PY scripts/extract_target_reliability_graph_features.py \
  --data-dir "$COHFACE_PREPARED_DATA_DIR" \
  --dataset-label COHFACE \
  --data-dir "$MAHNOB_TAILALIGNED_DATA_DIR" \
  --dataset-label MAHNOB_tailaligned \
  --out-edge analysis/final_priors/cohface_mahnob_target_reliability_windowed_edges.csv \
  --out-group analysis/final_priors/cohface_mahnob_target_computable_reliability_windowed.csv \
  --out-summary analysis/final_priors/cohface_mahnob_target_computable_reliability_windowed_summary.csv \
  --report-out analysis/final_priors/cohface_mahnob_target_computable_reliability_windowed.md

$PY scripts/extract_target_reliability_graph_features.py \
  --data-dir "$MAHNOB_TAILALIGNED_DATA_DIR" \
  --dataset-label MAHNOB_tailaligned \
  --out-edge analysis/final_priors/mahnob_target_reliability_windowed_edges.csv \
  --out-group analysis/final_priors/mahnob_target_computable_reliability_windowed.csv \
  --out-summary analysis/final_priors/mahnob_target_computable_reliability_windowed_summary.csv \
  --report-out analysis/final_priors/mahnob_target_computable_reliability_windowed.md

cp analysis/final_priors/mahnob_target_computable_reliability_windowed.csv \
   analysis/final_priors/mahnob_state_component_reliability_windowed.csv
```

## Main Runs

```bash
$PY scripts/materialize_calibrated_multifamily_parh_system.py \
  --data-dir "$COHFACE_PREPARED_DATA_DIR" \
  --out-run results/final_full_validation/cohface \
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

$PY scripts/materialize_calibrated_multifamily_parh_system.py \
  --data-dir "$MAHNOB_TAILALIGNED_DATA_DIR" \
  --out-run results/final_full_validation/mahnob_tailaligned \
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

## Post-Run Diagnostics

```bash
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
  --out-csv analysis/final_mahnob_tailaligned_observability_failure_taxonomy.csv \
  --report-out analysis/final_mahnob_tailaligned_observability_failure_taxonomy.md

$PY scripts/generate_formal_statistical_tests.py
```

Generated outputs under `analysis/` and `results/` are ignored by Git.
