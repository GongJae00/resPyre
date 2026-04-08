# Post-Rerun Manuscript Integration Plan

This document defines exactly how the manuscript should be updated after the
current-code COHFACE and MAHNOB reruns complete.

Current status:

- COHFACE rerun: completed
- MAHNOB rerun: pending completion
- manuscript numeric drift remains in parts of `paper/main.tex`, especially in
  the mechanism-audit discussion

## Integration principle

Do not free-write the Results section after rerun.

Update the manuscript in this order:

1. regenerate persistent CSVs and manifests
2. update figure assets
3. update tables
4. update local numeric claims in the text
5. update title/abstract only after the result tables stabilize

## Required rerun artifacts

### COHFACE

- `results/<rerun>/cohface_parh_ossm_prod/metrics/metrics_freq_domain_raw.csv`
- `results/<rerun>/cohface_parh_ossm_prod/metrics/metrics_waveform_raw.csv`
- `results/<rerun>/cohface_parh_ossm_prod/metrics/metrics_filter_diagnostics_raw.csv`
- `results/<rerun>/cohface_parh_ossm_prod/data/*.pkl`

### MAHNOB-HCI

- `results/<rerun>/mahnob_parh_ossm_prod/metrics/metrics_freq_domain_raw.csv`
- `results/<rerun>/mahnob_parh_ossm_prod/metrics/metrics_waveform_raw.csv`
- `results/<rerun>/mahnob_parh_ossm_prod/metrics/metrics_filter_diagnostics_raw.csv`
- `results/<rerun>/mahnob_parh_ossm_prod/data/*.pkl`

### Derived paper artifacts

- `paper/tables_ready/T3_rate_main.csv`
- `paper/tables_ready/T4_waveform_main.csv`
- `paper/tables_ready/T6_diagnostics_main.csv`
- `paper/tables_ready/T6b_cohface_mechanism_audit.csv`
- `paper/tables_ready/T6b_mahnob_mechanism_audit.csv`
- `paper/manifests/cohface_case_study_manifest.csv`
- `paper/manifests/mahnob_case_study_manifest.csv`

## Exact table insertion map

### T1 Dataset and protocol summary

Insert under `Methods`.

Source:
- rerun metadata
- config JSONs
- dataset counts

### T2 Model component and routing map

Insert at the end of the model section.

Source:
- code-verified feature list
- metric routing lock

### T3 Oscillatory output rate accuracy

Replace current COHFACE-only table with:

- panel A: COHFACE
- panel B: MAHNOB-HCI

Source:
- `paper/tables_ready/T3_rate_main.csv`

Text updates:
- Abstract
- Results / oscillatory output
- Discussion / rate interpretation

### T4 Full output waveform fidelity

Replace current COHFACE-only table with:

- panel A: COHFACE
- panel B: MAHNOB-HCI

Source:
- `paper/tables_ready/T4_waveform_main.csv`

Text updates:
- Abstract
- Results / waveform fidelity
- Discussion / waveform-rate trade-off

### T5 Intent-aligned ablation

Insert after T4 or after T6 depending on page pressure.

Source:
- ablation rerun metrics

Must answer:
- what harmonic-2 does
- what baseline does
- what residual does
- what adaptive R does
- what disentangled Q does
- what helper path does

### T6 Calibration diagnostics

Replace current PARH-only limited wording with:

- PARH main columns
- comparator diagnostics only if they are actually available

Source:
- `paper/tables_ready/T6_diagnostics_main.csv`

### T6b Mechanism audit

Keep as a compact mechanism table or move to supplement if page budget is tight.

Source:
- `paper/tables_ready/T6b_cohface_mechanism_audit.csv`
- `paper/tables_ready/T6b_mahnob_mechanism_audit.csv`

## Exact figure insertion map

### F1 Architecture diagram

Location:
- after `PARH-OSSM v1`

Caption must name:
- helper path
- inference path
- oscillator / baseline / residual
- `q_obs`, `q_dyn`, `q_osc`
- `pi_t` vs `lambda_t`
- `z_osc` and `z_full`

### F2 Dataset regime summary

Location:
- Methods or early Results before T3/T4

Caption must justify:
- COHFACE as regular regime
- MAHNOB-HCI as irregular regime

### F3 Family-level T3 summary

Location:
- immediately after T3

Source:
- `metrics_freq_domain_raw.csv`

### F4 Overlay grid

Location:
- immediately after T4

Source:
- case-study manifests
- saved trial PKLs

Selection lock:
- top / median / bottom by waveform CCC
- Base/KFstd use `signal_hat`
- PARH uses `z_full`

### F5 Calibration / mechanism figure

Location:
- after T6/T6b

Source:
- diagnostics CSV
- mechanism audit CSV

### F6 Failure cases

Location:
- late Results or early Discussion

Source:
- case-study manifest
- manually selected from manifest-defined candidates only

## Exact text replacement map

### Abstract

Replace only after:
- T3 finalized
- T4 finalized
- MAHNOB finalized

Do not update abstract from ad hoc notebook outputs.

### Introduction

Usually stable.

Only update:
- contribution bullets
- last paragraph of study scope

### Results

Update in this order:

1. T3 paragraph
2. T4 paragraph
3. T6 paragraph
4. mechanism audit paragraph
5. irregular-regime paragraph
6. ablation paragraph

### Discussion

Update in this order:

1. COHFACE interpretation
2. MAHNOB interpretation
3. decomposition interpretation
4. calibration interpretation
5. limitations

## Submission freeze checklist

Do not freeze `main.tex` until all are true:

- all reruns completed
- all tables regenerated
- overlay manifest regenerated
- mechanism audit regenerated
- ablation table generated
- final figure assets saved under `paper/figures`
- `latexmk` succeeds without warnings
