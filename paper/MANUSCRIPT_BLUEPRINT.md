# PARH-OSSM Manuscript Blueprint

This file remains the active submission-oriented blueprint for the current
manuscript. For the broader next-generation research and paper redesign
direction, also follow:

- `notes/PARH_NEXTGEN_RESEARCH_MASTERPLAN.md`
- `paper/PAPER_REDESIGN_LOCK.md`
- `paper/EVIDENCE_FLOW_LOCK.md`

## Purpose

This document is the pre-submission blueprint for turning the current
`paper/main.tex` audit manuscript into a submission-grade PARH-OSSM paper once
the rerun artifacts exist. It is intentionally stricter than a writing outline:
it locks the story, figure/table flow, ablation order, and claim boundaries so
the paper does not drift back into unsupported framing.

## Locked paper thesis

The paper is not about "a more robust Kalman filter."

The paper is about a physiology-aligned decomposition of camera-derived
respiratory motion into:

- oscillatory respiratory drive
- harmonic morphology
- baseline / trend
- aperiodic respiratory residual

The model claim is not universal superiority. The defensible claim is:

- a self-calibrating, auditable state-space scaffold
- a dual-output decomposition that separates rate-oriented and waveform-oriented analysis
- a disentangled adaptation design that can be inspected mechanistically

## Strongest supported current claims

- The repository implements an 8D `K=2` harmonic + baseline + residual model.
- The repository stores `z_osc`, `z_full`, causal outputs, smoothed outputs, and diagnostic arrays.
- The corrected evaluation path now routes T3 through `track_hz` when present.
- The completed current-code COHFACE rerun supports:
  - T3: PARH improves over KFstd after corrected routing
  - T4: PARH improves over Base but remains slightly below KFstd
  - T6: PARH diagnostics can now be reported from persistent arrays
- The first `OF + P1D_quadratic` pair-fusion gate supports:
  - fused PARH waveform improvement over fused Base and fused KFstd
  - a shared-latent multi-observation formulation as a credible next paper axis
- The same fusion axis is still secondary:
  - full pair fusion does not beat the best single-family `P1D_quadratic` route
  - assistant-channel fusion is safe only in inert form and is not yet a gain source
- A new single-family `OF` displacement bridge has passed a 12-trial COHFACE
  promotion gate:
  - it improves OF-family T3 and T4 for `KFstd` and PARH
  - full-dataset validation is now completed
  - promotion decision is locked as an additional observation family rather
    than a replacement for raw `OF`

## Claims that remain forbidden until rerun

- Any MAHNOB quantitative claim until the current-code MAHNOB rerun completes
- Any cross-dataset transfer claim
- Any statement that current PARH is already the final intended model
- Any statement that PARH dominates KFstd on waveform fidelity
- Any claim of significance unless tied to persistent regenerated artifacts
- Any claim that fusion already improves rate over the fused KF baseline
- Any claim that assistant-channel fusion already improves over the best
  single-family PARH route
- Any claim that stacked observation calibration is itself validated
- Any claim that raw `OF` has been replaced by `OF_bridge`

## Final title candidates

Primary:
- `PARH-OSSM: A Physiology-Aligned Harmonic State-Space Decomposition for Camera-Derived Respiratory Motion`

Alternative:
- `A Physiology-Aligned State-Space Decomposition of Camera-Derived Respiratory Motion`

Avoid:
- `NAROSSM`
- `trust-gated`
- `Student-t` in the title
- `optimal`, `first`, `universal`, `superior`

## Recommended manuscript flow

### Abstract

Must contain exactly four ideas:

1. respiratory motion should not be collapsed to one sinusoid
2. PARH-OSSM decomposes oscillatory, harmonic, baseline, and aperiodic components
3. current paper separates rate (`z_osc`) and waveform (`z_full`) evaluation
4. empirical claims are limited to persistent artifacts and dataset regime

### Introduction

Target flow:

1. Why contactless respiration matters
2. Why dominant-frequency-only framing is insufficient
3. Why "one robust innovation magnitude" is conceptually wrong
4. Relation to robust filtering literature
5. Relation to oscillator state-space literature
6. What this paper contributes that is specific and auditable

### Model section

Recommended subsection order:

1. Latent decomposition
2. Observation model and family heterogeneity
3. Harmonic dynamics and conditional linearity
4. Dual outputs and metric routing
5. Disentangled uncertainty adaptation
6. Calibration diagnostics and what they mean

### Methods / evaluation section

Recommended subsection order:

1. Observation families
2. Datasets and regime roles
3. Preprocessing and helper path
4. Evaluation routing policy (T3/T4/T6)
5. Causal vs smoothed policy
6. Statistical testing and case selection rules

### Results section

Recommended final order:

1. R1 dataset regime characterization
2. R2 calibration validation
3. R3 oscillatory output rate results
4. R4 full output waveform results
5. R5 fusion ladder (`best single-family -> fused Base -> fused KFstd -> fused PARH`)
6. R6 regular vs irregular regime stratification
7. R7 intent-aligned ablation
8. R8 transfer / tuning burden
9. R9 preprocessing × mechanism activation
10. R10 failure cases and overlays

### Discussion section

Recommended final order:

1. Why COHFACE gains are limited
2. Why MAHNOB should stress the residual path
3. Why disentangled adaptation matters
4. What the residual is intentionally allowed to capture
5. What remains missing
6. Practical deployment implications
7. Limitations and next steps

## Figure plan

### F1. Architecture diagram

- Placement: immediately after the `PARH-OSSM v1` model description
- Purpose: make the decomposition visually obvious before any numbers appear
- Must show:
  - input observation family
  - inference path
  - helper path
  - oscillator / baseline / residual blocks
  - `q_obs`, `q_dyn`, `q_osc`
  - `pi_t` vs `lambda_t`
  - dual outputs `z_osc`, `z_full`
  - causal forward pass and RTS smoother
- Must not show:
  - speculative IMM/frequency bank blocks unless actually implemented

### F2. Dataset regime characterization

- Placement: after dataset subsection, before main quantitative results
- Preferred form:
  - panel A: trial length distribution
  - panel B: GT rate variability distribution
  - panel C: observation-family quality spread
  - panel D: preprocessing-stage delta heatmap from observation EDA
- Purpose: justify COHFACE as regular regime and MAHNOB as irregular regime
  while also showing that preprocessing effects are family-specific rather than
  interchangeable

### F3. T3 family summary

- Placement: after T3 main table
- Preferred form:
  - paired per-family line plot or compact dot plot for Base / KFstd / PARH
- Metric:
  - MAE primary
  - RMSE secondary
- Purpose: show that corrected `track_hz` routing changes the T3 interpretation

### F4. Waveform overlay grid

- Placement: immediately after T4 main table
- Preferred form:
  - rows: best / median / worst cases
  - columns: Base / KFstd / PARH
  - GT overlaid in every panel
- Selection rule:
  - rank by waveform CCC within family and variant
  - choose top / median / bottom representative trials
- Preferred family priority:
  - `P1D_lin` and `DoF` first
  - add `OF` if page budget permits

### F5. Mechanism activation / calibration

- Placement: after T6 and mechanism audit
- Preferred form:
  - panel A: `q_dyn`, `q_osc`, `lambda_t` distributions
  - panel B: NIS mean vs in-band scatter per family
  - panel C: residual/baseline energy ratio
- Purpose: connect empirical behavior to the intended model story

Current note:

- after the completed COHFACE rerun, regenerate this figure from the latest
  mechanism-audit CSV rather than the older production audit values that still
  appear in parts of `paper/main.tex`

### F6. Failure cases

- Placement: late Results or Discussion
- Preferred form:
  - two short case studies
  - one stationary clean clip where PARH behaves like a conservative oscillator
  - one irregular / artifact-heavy clip where decomposition should matter

## Table plan

### Main paper tables

| ID | Title | Status now | Final role |
|----|-------|------------|------------|
| T1 | Dataset and protocol summary | missing | compact setup table |
| T2 | Observation-family semantics and current PARH role map | active | reviewer orientation table |
| T3 | Oscillatory output rate accuracy | active | main rate table |
| T4 | Full output waveform fidelity | active | main waveform table |
| T5 | Intent-aligned ablation | missing | key mechanistic table |
| T6 | Fusion comparison ladder | missing | best single-family vs fused Base/KFstd/PARH |
| T7 | Calibration diagnostics | active but PARH-only | main diagnostics table |
| T8 | Regular vs irregular regime stratification | missing | bridge between COHFACE and MAHNOB |
| T9 | Tuning burden / transfer | missing | final practical-value table |

### Supplementary tables

- S1 full per-family metrics
- S2 causal vs smoothed comparison
- S3 PARH `z_osc` waveform supplement
- S4 per-family mechanism audit details
- S5 case-study manifest

## Mandatory ablation table design

The ablation table must be intent-aligned, not a random feature toggle dump.

### T5 rows

1. Base
2. KFstd
3. PARH full
4. PARH without harmonic-2
5. PARH without baseline
6. PARH without residual
7. PARH with adaptive `R` off
8. PARH with `q_dyn` off or clipped-low
9. PARH with `q_osc` release off
10. PARH with Student-`t` off
11. PARH with helper path collapsed back to shared path
12. PARH with legacy coupled `Q`
13. OF-family raw proxy vs OF displacement bridge

### T5 columns

- T3 MAE
- T4 waveform CCC
- NIS mean
- NIS in-band
- residual energy ratio
- stability seconds

### T5 reading goal

The table must answer:

- which components help waveform fidelity
- which components help rate tracking
- which components only change diagnostics
- whether helper-path and residual release are actually necessary

## Overlay case selection policy

All overlay figures must come from a persistent case manifest, not manual cherry-picking.

Selection rules:

- use smoothed outputs for primary paper overlays
- rank within dataset × family × variant
- use waveform CCC for T4 overlays
- use rate MAE for T3 overlays only if a rate-focused overlay is added
- top / median / bottom are percentile-based representatives, not arbitrary examples

## Statistical testing plan

Final paper should include:

- paired Wilcoxon signed-rank tests for family-level comparisons
- effect sizes or median paired deltas
- no p-value fishing across every metric in the main text
- strict separation between confirmatory main tables and exploratory supplements

## Reference strengthening priorities

The final paper should cite four literature blocks explicitly:

1. contactless respiration and chest-motion proxies
2. oscillator / conditional-linear state-space modeling
3. robust heavy-tailed and adaptive Kalman filtering
4. waveform-fidelity evaluation / calibration evaluation

Do not overload the paper with deep-learning baselines unless they are either:

- directly compared in reproducible experiments, or
- explicitly framed as related context rather than direct baselines

## Submission blockers

The manuscript should not be frozen until all of the following are true:

- COHFACE rerun completed under current code
- MAHNOB rerun completed under current code
- T3/T4/T6 regenerated from new artifacts
- ablation runs completed
- overlay case manifest generated from persistent metrics
- all figure captions tied to persistent artifact files
- reviewer-risk items in `paper/REJECTION_RISK_REGISTER.md` are addressed
