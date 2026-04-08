# PARH-OSSM Paper Redesign Lock

## Purpose

This file locks the target paper structure for the next manuscript rewrite.
Unlike `paper/MANUSCRIPT_BLUEPRINT.md`, which started as a submission-oriented
outline for the current audited manuscript, this file is the forward-looking
design lock for the stronger paper the project is now aiming for.

Use this file when deciding:

- what the final paper is actually about
- which sections must be rewritten from scratch
- which figures and tables are essential
- which claims are safe
- which claims must remain forbidden

Reference backbone:

- `notes/PARH_REFERENCE_BACKBONE.md`
- `notes/EVALUATION_DESIGN_AND_LIMITS.md`
- `paper/EVIDENCE_FLOW_LOCK.md`

## Final paper identity

The paper is not:

- a generic robust Kalman-filter paper
- a Student-`t` paper
- a trust-gate paper
- a belt-waveform cloning paper

The paper is:

- a physiology-aligned decomposition paper
- an observation-gap-aware state-space paper
- a dual-output respiratory analysis paper
- a self-auditing, reduced-retuning paper

## Final paper question

The paper should answer this question:

How can camera-derived respiratory motion be decomposed into physiologically
meaningful components while explicitly accounting for observation distortion,
dataset regime, and uncertainty pathways, without abandoning interpretable
state-space inference?

## Strong claim hierarchy

### Strongest target claim

Camera-derived respiratory motion should be modelled as a decomposed latent
process with distinct oscillatory, baseline, and aperiodic components, and the
observation channel should be calibrated rather than treated as a direct belt
equivalent.

### Second-level target claim

Within an OSSM-compatible framework, explicit observation calibration and
disentangled adaptation reduce the need for dataset-specific manual retuning.

### Third-level target claim

The decomposition is most valuable in irregular or artifact-heavy regimes,
while clean stationary regimes mainly test do-no-harm behavior.

## Current evidence lock

As of the current repository state:

- COHFACE rerun is completed under the current code.
- The current official COHFACE evidence boundary is the six-family rerun in
  `results/20260408_cohface_prod_ofbridge_familyconf_v3/cohface_parh_ossm_prod_ofbridge`.
- MAHNOB rerun is not yet complete and therefore cannot support quantitative claims.
- COHFACE supports the statement that PARH improves rate tracking over KFstd
  after corrected routing.
- COHFACE does not support a claim of waveform dominance over KFstd.
- Observation-calibration v1 failed a 12-trial COHFACE promotion gate and is
  not an allowed paper claim.
- Observation-calibration v2 (`global_signed_gain`) also failed to show a
  promotable benefit and remains default-off.
- Observation-calibration v3 (`osc_aux_two_gain`) also failed to show a
  promotable benefit and remains default-off.
- Observation-calibration v4 (`family_phase_aux`) showed mixed gate results:
  small quad/cub gains, but slight `OF` and `P1D_linear` regressions.
- Observation-calibration v5 passed the same gate in selective form and
  established the validity of selective family-row calibration.
- Observation-calibration v7 then improved that selective path:
  a harmonic-only `P1D_quad/cub` row improved both T3 absolute error and T4
  waveform fidelity on the same gate subset and is now the promoted default.
- The original output-only `OF` helper blend then became effectively inert
  under the newer scaffold.
- A stronger relaxed-gating variant improved `OF` T3 on the same gate subset
  without changing T4, but the full 160-trial COHFACE rerun regressed `OF`
  T3 and therefore rejected that promotion.
- The first pair-fusion gate (`OF + P1D_quadratic`) supports promotion to a
  full COHFACE comparison run:
  - fused PARH beats fused Base and fused KFstd on waveform
  - rate remains mixed versus fused KFstd
- Full COHFACE pair fusion remains secondary because it does not beat the best
  single-family `P1D_quadratic` route.
- Assistant-channel fusion has now been tested:
  - `v1` harms T3
  - `v2` is safe but inert
  - therefore fusion is not a current headline gain source
- The stacked-setting observation-calibration effect remains mixed and must not
  be presented as a validated source of the pair-fusion gain.
- Frequency rescue passed the same gate only in family-selective form:
  `OF` and `P1D_linear` only.
- The current bridge observation path is also family-selective:
  light path for `OF`, legacy stack for `DoF` and all `P1D` families.
- The current live default is therefore a hybrid scaffold:
  - `OF`: light path + selective frequency rescue + conservative output-only
    helper-rate blending
  - `P1D_linear`: legacy path + selective frequency rescue
  - `P1D_quad/cub`: legacy path + selective harmonic-only observation-row calibration
  - `DoF`: legacy path only
- Additional helper-only and helper-blend observation-routing variants also
  failed to produce a promotable COHFACE gate result.
- A direct OF velocity-row calibration was then tested and failed badly on the
  same gate subset; a fit-quality safety gate now disables such calibration
  when the warm-up fit is weak.
- `OF` helper-trust heuristics were then tested:
  - direct helper-trust-driven `q_dyn` suppression worsened `OF` rate and is rejected
  - rescue-only helper trust was safe but effectively inert
  - rescue-only helper trust plus relaxed rescue improved some `OF` T3 metrics
    but worsened `OF` T4 and therefore also failed promotion
- a new single-family `OF` displacement bridge then produced the first clearly
  positive OF-only gate result:
  - it improved both OF T3 and OF T4 for `KFstd` and PARH on the 12-trial
    COHFACE gate subset
  - the subsequent full COHFACE validation confirmed strong OF T3 gains
  - T4 stayed mixed: slightly better `wMAE` / `DTW`, but lower waveform `CCC`
  - promotion decision: keep raw `OF` and add `OF_bridge` as an additional
    observation family rather than replacing raw `OF`
- the subsequent six-family COHFACE rerun confirmed that promotion:
  - `OF_bridge` is now part of the official single-family ladder
  - `OF_bridge` is the current best PARH T3 family by the locked
    median-based ladder rule
  - `P1D_quadratic` remains the strongest PARH T4 family
  - the corrected ladder therefore now supports an explicit rate-versus-waveform
    family split rather than a single universally best route
- a narrow family-confidence policy has now been promoted on top of that
  six-family scaffold:
  - it only activates for `P1D_quadratic` and `P1D_cubic`
  - it requires excellent warm-up displacement-fit quality
  - it slightly improves T3 absolute error on those families and is
    effectively zero-impact elsewhere
  - the latest official COHFACE tables should therefore be sourced from the
    `familyconf_v3` rerun, not the earlier `ofbridge_full` rerun
- a corrected raw-`OF` fixed velocity prior fallback was then gate-tested:
  - the first attempt was invalid because the fallback path was accidentally
    blocked by the warm-up calibration allowlist
  - the corrected rerun verified that raw `OF` really switched to a velocity
    fallback row
  - it slightly improved `RMSE` / `PearsonR` but worsened primary rate `MAE`
    and strongly degraded waveform fidelity
  - therefore it is rejected as a live raw-`OF` policy
- Residual-release redesign also remains unpromoted:
  - `v1` blended clean support into `q_osc` and mostly closed the residual path
  - `v2` corrected the direction and lowered `q_osc` where oscillatory support
    was weaker, but produced near-zero T3/T4 gains
  - `v3` then added direct `Q_aper` bonus and slightly worsened `OF`/`DoF`
    without helping the main profile families
  - `v4` then replaced the earlier heuristic with an observation-driven
    non-oscillatory gap between oscillator-only support and full-state support;
    this is more interpretable, but it remained effectively inert on the clean
    COHFACE gate subset
- Current evidence therefore rejects further preprocessing-policy switching as
  the main redesign axis; the next step is stronger family-specific
  observation semantics and a more identifiable residual branch.
- Current evidence also rejects helper-trust heuristics as a headline
  redesign axis for `OF`; they may remain diagnostic signals, but they are not
  a promotable source of gain.
- A clean rerun of family-aware residual semantics then confirmed a narrower
  result:
  - family-aware residual priors do change the intended diagnostics
  - but the clean COHFACE scaffold remains almost unchanged in T3/T4
  - the redesign is therefore useful as a diagnostic clarification, not yet as
    a promoted performance policy
- Two additional output-side assistant routes were then rejected:
  - raw `OF` with `OF_bridge` assistant
  - `P1D_quad` with `OF_bridge` assistant
  Both were effectively inert or slightly harmful in T3 while leaving T4
  unchanged.
- Current COHFACE evidence therefore suggests that the next meaningful
  development step is no longer a small helper/assistant tweak. It is either:
  - a more structural multi-observation design, or
  - an irregular-regime subset validation on MAHNOB
- The best-single ladder generator has now been corrected to rank
  family-level medians rather than raw per-trial rows. Any older ladder dump
  produced before that fix should be treated as stale.
- Current `main.tex` still contains some older mechanism-audit language that
  should not be treated as the final paper text.

## Final section flow

### 1. Introduction

Target logic:

- why contactless respiration matters
- why one-sinusoid thinking is insufficient
- why observation distortion matters
- why robustification alone is not enough
- what this paper contributes

### 2. Generative framing

This section should appear before the full method details.

It must explain:

- physiology layer
- camera/projection layer
- extractor/proxy layer
- latent decomposition layer
- output and diagnostic layer

This section is where the final paper becomes conceptually stronger than the
current audit manuscript.

### 3. Model

Target subsections:

- latent decomposition
- calibrated observation model
- harmonic dynamics and conditional linearity
- dual outputs
- disentangled uncertainty pathways
- calibration diagnostics

### 4. Methods

Target subsections:

- datasets and regime roles
- observation families
- preprocessing and helper path
- calibration warm-up protocol
- evaluation routing policy
- causal vs smoothed policy
- statistical testing and manifest policy

### 5. Results

Target subsections:

- dataset regime characterization
- oscillatory-output results
- full-output waveform results
- fusion ladder
- calibration and mechanism results
- regular vs irregular regime comparison
- intent-aligned ablation
- tuning burden / transfer
- failure cases

### 6. Discussion

Target subsections:

- why COHFACE gains are bounded
- why observation modelling is the main bottleneck
- what the residual is allowed to represent
- where multi-family fusion should help
- limitations and deployment implications

## Figure lock

### F1. Generative chain + architecture

Must show:

- physiology
- camera/projection distortion
- observation-family proxies
- calibration block
- helper path
- OSSM latent decomposition
- `z_osc`
- `z_full`
- diagnostics

This should replace a plain block-diagram mentality. It has to explain the
scientific chain, not only the code modules.

### F2. Dataset regime figure

Must justify why COHFACE and MAHNOB play different scientific roles.

It should also include a compact observation-EDA panel showing that
preprocessing effects differ across families. Otherwise the observation-gap
argument remains too abstract.

### F3. T3 family summary

Must make the rate story visible at a glance.

### F4. Overlay grid

Must use manifest-selected best / median / worst examples.

### F5. Mechanism figure

Must connect:

- calibration behavior
- adaptation behavior
- decomposition energy behavior
- rejected observation-routing variants, to show why explicit observation-row
  semantics are needed

### F6. Failure-case figure

Must show where observation mismatch or decomposition limits remain.

## Table lock

### T1

Dataset and protocol summary.

### T2

Observation-family semantics / current PARH role map.

Current active artifact:

- `paper/tables_ready/T2_observation_family_map.csv`

### T3

Oscillatory-output rate results.

### T4

Full-output waveform results.

### T5

Intent-aligned ablation.

### T6

Fusion ladder.

### T7

Calibration diagnostics.

### T8

Regular-vs-irregular regime stratification.

### T9

Tuning burden / transfer.

## Sections that must be rewritten from scratch

- abstract
- last two paragraphs of the introduction
- model section opening
- results framing paragraphs
- discussion opening
- limitations

## Sections that can be partially reused

- contactless respiration motivation
- current code-grounded decomposition description
- metric-routing explanation
- audit/evidence boundary logic

## Safe wording

Use:

- physiology-aligned decomposition
- observation-calibrated state-space model
- dual-output respiratory analysis
- reduced manual retuning burden
- do-no-harm on stationary data
- irregular-regime motivation

Avoid:

- tuning-free in an absolute sense
- full waveform reconstruction in all settings
- universal superiority
- first / optimal / best

## Final paper standard

The paper is ready only when all of the following are true:

- the observation model is explicitly described and justified
- the role of the residual is empirically defended
- COHFACE and MAHNOB are framed as regime-specific evidence, not pooled slogans
- all figures/tables come from persistent files
- the text no longer drifts from the current code or current artifacts
