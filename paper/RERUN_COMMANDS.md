# PARH-OSSM End-to-End Rerun Commands

This file lists the exact commands to execute the current PARH-OSSM pipeline
from sanity checks through reruns, analysis artifacts, and paper rebuild.

## Recommended One-Shot Automation

Run estimation, evaluation, table regeneration, manifests, figures, and PDF
rebuild in one command:

```bash
cd /home/gongjae/Projects/resPyre
python3 scripts/run_parh_e2e.py \
  --config configs/cohface_parh_ossm_prod_ofbridge.json \
  --results results/20260408_cohface_prod_ofbridge_familyconf_v3_e2e
```

To reuse an already-completed run and regenerate artifacts only:

```bash
cd /home/gongjae/Projects/resPyre
python3 scripts/run_parh_e2e.py \
  --config configs/cohface_parh_ossm_prod_ofbridge.json \
  --results results/20260408_cohface_prod_ofbridge_familyconf_v3 \
  --skip-estimate
```

To regenerate combined tables after a new MAHNOB run while keeping the current
COHFACE metrics as peer reference:

```bash
cd /home/gongjae/Projects/resPyre
python3 scripts/run_parh_e2e.py \
  --config configs/mahnob_parh_ossm_subset_ofbridge.json \
  --results results/20260408_mahnob_subset_ofbridge_gate \
  --skip-estimate \
  --peer-metrics COHFACE=results/20260408_cohface_prod_ofbridge_familyconf_v3/cohface_parh_ossm_prod_ofbridge/metrics \
  --paper-suite none
```

## 0. Working directory

```bash
cd /home/gongjae/Projects/resPyre
```

## 1. Sanity check

```bash
python3 -m py_compile \
  components/models/heads/parh_ossm.py \
  components/models/heads/kf_std.py \
  core/pipeline/evaluation_step.py \
  scripts/generate_table_ready.py \
  scripts/generate_case_study_manifest.py \
  scripts/collect_parh_mechanism_audit.py
```

## 2. One-sample smoke run

```bash
python3 main.py \
  --config configs/cohface_parh_ossm_prod.json \
  --debug \
  --results results/20260403_structural_smoke
```

## 3. COHFACE full rerun

```bash
python3 main.py \
  --config configs/cohface_parh_ossm_prod.json \
  --results results/20260403_cohface_rerun
```

## 4. MAHNOB-HCI full rerun

```bash
python3 main.py \
  --config configs/mahnob_parh_ossm_prod.json \
  --results results/20260403_mahnob_rerun
```

## 5. Table-ready CSV regeneration

```bash
python3 scripts/generate_table_ready.py \
  --cohface-metrics results/20260403_cohface_rerun/cohface_parh_ossm_prod/metrics \
  --mahnob-metrics results/20260403_mahnob_rerun/mahnob_parh_ossm_prod/metrics \
  --out-dir paper/tables_ready
```

## 6. Mechanism audit from saved PARH PKLs

### COHFACE

```bash
python3 scripts/collect_parh_mechanism_audit.py \
  --data-dir results/20260403_cohface_rerun/cohface_parh_ossm_prod/data \
  --trial-out paper/manifests/cohface_parh_mechanism_trials.csv \
  --family-out paper/tables_ready/T6b_cohface_mechanism_audit.csv
```

### MAHNOB-HCI

```bash
python3 scripts/collect_parh_mechanism_audit.py \
  --data-dir results/20260403_mahnob_rerun/mahnob_parh_ossm_prod/data \
  --trial-out paper/manifests/mahnob_parh_mechanism_trials.csv \
  --family-out paper/tables_ready/T6b_mahnob_mechanism_audit.csv
```

## 7. Overlay case-study manifest generation

### COHFACE

```bash
python3 scripts/generate_case_study_manifest.py \
  --waveform-csv results/20260403_cohface_rerun/cohface_parh_ossm_prod/metrics/metrics_waveform_raw.csv \
  --freq-csv results/20260403_cohface_rerun/cohface_parh_ossm_prod/metrics/metrics_freq_domain_raw.csv \
  --dataset-name COHFACE \
  --out paper/manifests/cohface_case_study_manifest.csv
```

### MAHNOB-HCI

```bash
python3 scripts/generate_case_study_manifest.py \
  --waveform-csv results/20260403_mahnob_rerun/mahnob_parh_ossm_prod/metrics/metrics_waveform_raw.csv \
  --freq-csv results/20260403_mahnob_rerun/mahnob_parh_ossm_prod/metrics/metrics_freq_domain_raw.csv \
  --dataset-name MAHNOB \
  --out paper/manifests/mahnob_case_study_manifest.csv
```

## 8. Quick file-existence check

```bash
ls results/20260403_cohface_rerun/cohface_parh_ossm_prod/metrics
ls results/20260403_mahnob_rerun/mahnob_parh_ossm_prod/metrics
ls paper/tables_ready
ls paper/manifests
```

## 9. PARH ablation suite

### Smoke ablations first

```bash
python3 scripts/run_parh_ablation.py \
  --config configs/cohface_parh_ossm_prod.json \
  --out-root results/20260403_cohface_ablation_smoke \
  --profiles full no_h2 no_baseline no_residual no_adapt_r no_student_t legacy_q no_freq_adapt no_qdyn no_qosc_release no_helper \
  --debug
```

### Full COHFACE ablations

```bash
python3 scripts/run_parh_ablation.py \
  --config configs/cohface_parh_ossm_prod.json \
  --out-root results/20260403_cohface_ablation_full \
  --profiles full no_h2 no_baseline no_residual no_adapt_r no_student_t legacy_q no_freq_adapt no_qdyn no_qosc_release no_helper
```

## 10. Paper rebuild

```bash
cd /home/gongjae/Projects/resPyre/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## 11. Optional: inspect run status

```bash
cat /home/gongjae/Projects/resPyre/results/20260403_cohface_rerun/cohface_parh_ossm_prod/run_status.json
cat /home/gongjae/Projects/resPyre/results/20260403_mahnob_rerun/mahnob_parh_ossm_prod/run_status.json
```

## Notes

- These commands intentionally write to new rerun directories instead of
  overwriting the older production artifacts.
- Ablations are now exposed through environment-backed profile overrides in
  `scripts/run_parh_ablation.py`.
