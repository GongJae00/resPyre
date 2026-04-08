# PARH-OSSM Figure and Table Index

## Scope lock

- Manuscript source: `paper/main.tex`
- Blueprint source: `paper/MANUSCRIPT_BLUEPRINT.md`
- Research masterplan source: `notes/PARH_NEXTGEN_RESEARCH_MASTERPLAN.md`
- Paper redesign source: `paper/PAPER_REDESIGN_LOCK.md`
- Evidence-flow source: `paper/EVIDENCE_FLOW_LOCK.md`
- Current quantitative scope in the live manuscript: COHFACE only
- COHFACE current-code rerun status: completed
- COHFACE active family ladder: `OF`, `OF_bridge`, `DoF`, `P1D_lin`, `P1D_quad`, `P1D_cub`
- COHFACE pair-fusion gate status: completed, secondary-only
- MAHNOB-HCI current-code rerun status: incomplete
- Active table-ready CSVs:
  - `paper/tables_ready/T2_observation_family_map.csv`
  - `paper/tables_ready/T3_rate_main.csv`
  - `paper/tables_ready/T4_waveform_main.csv`
  - `paper/tables_ready/T6_diagnostics_main.csv`
  - `paper/tables_ready/T6b_cohface_mechanism_audit.csv`
  - `paper/tables_ready/T6b_fusion_ladder.csv`

## Current drift note

The live `paper/main.tex` still contains some older pre-`OF_bridge` wording in
places. When regenerating any mechanism table or figure, prefer the current
six-family COHFACE rerun artifacts over legacy production numbers.

## Hard rule

No figure or table may appear in the main manuscript unless its content is
traceable to a persistent artifact path.

## Active main-paper tables

| ID | Current title | Placement in `main.tex` | Source artifact | Status |
|----|---------------|-------------------------|-----------------|--------|
| T3 | COHFACE rate accuracy | Results / COHFACE rate accuracy | `paper/tables_ready/T3_rate_main.csv` | Active |
| T4 | COHFACE waveform fidelity | Results / COHFACE waveform fidelity | `paper/tables_ready/T4_waveform_main.csv` | Active |
| T6 | COHFACE calibration diagnostics | Results / COHFACE calibration diagnostics | `paper/tables_ready/T6_diagnostics_main.csv` | Active |
| T4b | COHFACE mechanism audit | Results / COHFACE mechanism audit | saved PKLs under `results/*/data/*.pkl` | Active |
| T6b | Best-single ladder | Results / rate/waveform synthesis or Supplementary | `paper/tables_ready/T6b_fusion_ladder.csv` | Active, single-family rows only |

## Planned main-paper figures

| ID | Planned content | Recommended placement | Required artifact source | Current status |
|----|-----------------|-----------------------|--------------------------|----------------|
| F1 | PARH-OSSM architecture diagram | After model section | manually prepared vector figure consistent with current code | Planned only |
| F2 | Dataset regime + observation-EDA characterization | After dataset subsection | `paper/figures/F2_dataset_and_observation_regime.pdf` | Active |
| F3 | T3 family summary plot | After T3 table | `paper/figures/F3_t3_family_summary.pdf` generated from `paper/tables_ready/T3_rate_main.csv` | Active |
| F4 | Best/median/worst waveform overlay grid | After T4 table | `paper/manifests/cohface_p1dquad_overlay_manifest.csv` + `paper/figures/F4_waveform_overlay_grid.pdf` + saved PKLs under `results/20260408_cohface_prod_ofbridge_familyconf_v3/cohface_parh_ossm_prod_ofbridge/data` | Active |
| F5 | Mechanism activation / calibration figure | After T6 and mechanism audit | `paper/figures/F5_mechanism_activation.pdf` generated from `paper/tables_ready/T6_diagnostics_main.csv` and `paper/tables_ready/T6b_cohface_mechanism_audit.csv` | Active |
| F6 | Failure-case panel | Late Results or Discussion | `paper/manifests/cohface_failure_case_manifest.csv` + `paper/figures/F6_failure_cases.pdf` + saved PKLs under `results/20260408_cohface_prod_ofbridge_familyconf_v3/cohface_parh_ossm_prod_ofbridge/data` | Active |

## Planned main-paper tables

| ID | Planned content | Recommended placement | Required artifact source | Current status |
|----|-----------------|-----------------------|--------------------------|----------------|
| T1 | Dataset and protocol summary | Methods | config + dataset metadata + rerun metadata | Missing |
| T2 | Observation-family semantics and current PARH role map | End of model section | `paper/tables_ready/T2_observation_family_map.csv` generated from `components/observations/semantics.py`, `paper/tables_ready/T3_rate_main.csv`, and `paper/tables_ready/T4_waveform_main.csv` | Active |
| T5 | Intent-aligned ablation | Results after T4/T6 | dedicated ablation reruns | Missing |
| T6b | Fusion ladder: best single-family vs fused Base/KFstd/PARH | Results after T4 or Supplementary | pair-fusion comparison rerun + current single-family reruns | Active for best-single rows, fused rows secondary only |
| T7 | Regular vs irregular regime stratification | After COHFACE/MAHNOB comparison | completed COHFACE + MAHNOB reruns | Missing |
| T8 | Tuning burden / transfer table | Late Results | controlled transfer experiments | Missing |

## Supplementary figure plan

| ID | Content | Required artifact source |
|----|---------|--------------------------|
| S-F1 | PARH `z_osc` vs `z_full` waveform supplement | saved PARH PKLs |
| S-F2 | Causal vs smoothed overlay | saved PKLs |
| S-F3 | Per-family calibration histograms | saved diagnostics arrays |
| S-F4 | Additional failure cases | case-study manifest |
| S-F5 | Full preprocessing-stage delta heatmaps by family | `analysis/cohface_preproc_summary.csv`, `analysis/cohface_preproc_deltas.csv`, MAHNOB equivalents once complete |
| S-F6 | OF raw vs OF bridge observation comparison | `paper/figures/S_F6_of_construction_comparison.pdf` generated from `paper/tables_ready/T3_rate_main.csv` and `paper/tables_ready/T4_waveform_main.csv` |

## Supplementary table plan

| ID | Content | Required artifact source |
|----|---------|--------------------------|
| S-T1 | Full per-family metric dump | raw metrics CSVs |
| S-T2 | Causal vs smoothed comparison | raw waveform CSVs + PKLs |
| S-T3 | PARH `z_osc` waveform supplement | raw waveform CSV |
| S-T4 | Detailed mechanism audit | PKL diagnostics arrays |
| S-T5 | Case-study manifests | `scripts/generate_case_study_manifest.py`, `scripts/generate_residual_case_manifest.py`, and `scripts/generate_observation_construction_manifest.py` outputs |
| S-T6 | OF raw vs OF bridge ablation ladder | `configs/cohface_of_bridge_gate.json` reruns |

## Planned figure filenames

Use these filenames when the assets are created:

- `paper/figures/F1_architecture.pdf`
- `paper/figures/F2_dataset_and_observation_regime.pdf`
- `paper/figures/F3_t3_family_summary.pdf`
- `paper/figures/F4_waveform_overlay_grid.pdf`
- `paper/figures/F5_mechanism_activation.pdf`
- `paper/figures/F6_failure_cases.pdf`
- `paper/figures/S_F5_preproc_delta_heatmaps.pdf`
- `paper/figures/S_F6_of_construction_comparison.pdf`

## Fusion comparison lock

The first defensible fusion comparison is:

- best single-family reference rows from the current COHFACE rerun
- `fusion_of_p1d_quadratic`
- `pair_of_p1d_quadratic__kfstd`
- `pair_of_p1d_quadratic__parh_ossm`
- assistant fusion remains exploratory only:
  - `analysis/cohface_assistfusion_gate_report_20260407.md`

Gate provenance:

- `analysis/cohface_pairfusion_gate_report_20260407.md`
- `results/20260407_pairfusion_gate/full/cohface_pairfusion_smoke`
- `results/20260407_pairfusion_gate/no_obs_cal/cohface_pairfusion_smoke`

## Overlay figure selection lock

All overlay figures must be generated from a persistent manifest. Use:

- `scripts/generate_case_study_manifest.py`
- `scripts/generate_residual_case_manifest.py`
- `scripts/generate_observation_construction_manifest.py`
- `scripts/generate_failure_case_manifest.py`

Selection policy:

- rank within dataset × family × variant
- use smoothed rows only for primary figures
- Base/KFstd use `signal_hat`
- PARH uses `z_full` for main waveform overlays
- choose top / median / bottom by waveform CCC unless explicitly building a rate overlay

## Source policy

- T3 must trace to `metrics_freq_domain_raw.csv` via `scripts/generate_table_ready.py`
- T4 must trace to `metrics_waveform_raw.csv` via `scripts/generate_table_ready.py`
- T6 must trace to `metrics_filter_diagnostics_raw.csv` via `scripts/generate_table_ready.py`
- Mechanism-audit values must be reproducible from saved PARH PKL payloads
- Overlay figures must trace to a manifest CSV plus saved trial PKLs

## Deferred until rerun

- MAHNOB-HCI main-paper tables
- cross-dataset transfer tables
- irregular-regime stratification
- ablation table
- architecture figure insertion into `main.tex`
