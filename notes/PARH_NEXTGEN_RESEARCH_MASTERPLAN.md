# PARH-OSSM Next-Generation Research Masterplan

## Purpose

This document is the long-horizon lock for the next-generation PARH-OSSM
research program. It exists to keep the project coherent across future
compaction, refactors, reruns, and manuscript rewrites.

It is intentionally broader than a code TODO list. It locks:

- the scientific north star
- the generative assumptions that the model must explain
- the observation-model redesign direction
- the staged implementation roadmap
- the evaluation structure
- the manuscript structure

Companion documents:

- `notes/OBSERVATION_EDA_PROTOCOL.md`
- `notes/OBSERVATION_MODEL_REDESIGN_SPEC.md`
- `notes/COHFACE_FIRST_RAPID_VALIDATION_ROADMAP.md`
- `notes/PARH_WORK_PACKAGES.md`
- `notes/PARH_REFERENCE_BACKBONE.md`
- `notes/EVALUATION_DESIGN_AND_LIMITS.md`

## North-star objective

The final paper should not claim that a contactless respiratory proxy is
identical to a belt signal.

The stronger and more defensible objective is:

- explain respiratory motion as a physiology-aligned latent decomposition
- explain how camera observations distort that motion
- explain when the distorted proxy can still recover oscillatory rate well
- explain when full waveform recovery is limited by observation physics
- expose those limits through auditable state-space diagnostics

The final contribution is therefore not "a better smoother."

It is a generative account of the chain:

1. physiology
2. projection into camera-observable motion
3. extraction into heterogeneous scalar proxy families
4. latent decomposition and uncertainty adaptation
5. dual outputs for rate and full-motion analysis

## The four alignment axes

The research program must align four things at once.

### 1. Physiological phenomenon

The model must reflect that respiration is not one pure sinusoid. Relevant
structure includes:

- primary oscillatory drive
- harmonic morphology from inhale/exhale asymmetry
- slow drift and posture-related baseline variation
- aperiodic respiratory events
- temporary support breakdown of oscillatory assumptions

### 2. Signal phenomenon

The observed scalar proxy is not the physiology itself. It is the result of:

- projection geometry
- extractor family choice
- sign ambiguity
- gain mismatch
- offset mismatch
- small lag mismatch
- band-limiting and preprocessing distortion

### 3. Real-world gap

The camera world introduces nuisance factors that are neither pure physiology
nor pure white noise:

- ROI drift
- clothing and texture effects
- illumination and contrast changes
- body pose and projection change
- weak thoracic visibility
- family-specific failure modes

### 4. Model concept

Each latent block must mean something precise.

The model concept is only valid if each block has a defensible role:

- harmonic blocks: oscillatory drive and morphology
- baseline block: slow drift / projection trend
- residual block: aperiodic respiratory content, not generic leftover noise
- observation adaptation: confidence about the measurement channel
- process adaptation: novelty in respiratory dynamics
- oscillatory-support adaptation: evidence that the oscillator explanation is breaking down

## Non-negotiable design principles

These principles remain locked unless a later, explicitly better design
supersedes them.

1. Keep OSSM structure.
2. Preserve conditional exact linearity whenever possible.
3. Keep rate-oriented and full-motion outputs separate.
4. Do not collapse observation corruption, dynamical novelty, and oscillatory support into one scalar.
5. Prefer interpretable calibration over opaque flexibility unless there is a clear empirical need.
6. Never claim belt-equivalent waveform reconstruction unless the observation model really supports it.
7. Treat decomposition validity as part of the result, not just end metrics.

## Current state snapshot after the completed COHFACE rerun

This is the current reproducible state to build from.

### What is already strong

- `PARH-OSSM` is already an 8D `K=2` harmonic + baseline + residual model.
- The code stores `z_osc`, `z_full`, causal outputs, smoothed outputs, and diagnostic arrays.
- T3 routing now correctly uses `track_hz` for the model outputs.
- The completed COHFACE rerun shows real rate gains over KFstd for all families.

### What remains weak

- T4 waveform fidelity is still below KFstd on COHFACE.
- Base remains strongest on T3 in four of five COHFACE families.
- The current observation model is still too weak to justify "waveform reconstruction" language.
- Some manuscript text still reflects older mechanism-audit numbers and is therefore stale.
- MAHNOB is still required to test irregular-regime claims.

### What the current COHFACE rerun suggests mechanistically

Compared with the older production audit, the latest COHFACE rerun suggests:

- `q_dyn` is less saturated than before
- `q_osc` is lower than before and the residual path opens more often
- baseline and residual energy shares are still modest
- the model is no longer best described as a fully dormant extension of KFstd
- but the extra decomposition capacity is still not translating into dominant waveform gains on clean data

This is an important shift. The current bottleneck is no longer "the extra
states never activate." It is now closer to "the observation model and
decomposition semantics are still not strong enough to convert that activation
into consistent full-waveform benefit."

## The main scientific gap

The current repository models latent respiratory structure more carefully than
it models the observation channel.

That mismatch is now the main scientific bottleneck.

The current model says more about hidden respiratory dynamics than about how a
camera-derived proxy deviates from belt truth. This is why it can improve rate
tracking while still failing to dominate waveform fidelity.

## COHFACE observation EDA lock

The completed COHFACE observation EDA now gives a concrete observation-side
finding to design around.

Source artifacts:

- `analysis/cohface_observation_eda_trials.csv`
- `analysis/cohface_observation_eda_family.csv`
- `analysis/cohface_preproc_summary.csv`
- `analysis/cohface_preproc_deltas.csv`

Key locked findings:

- `OF` and `DoF` benefit strongly from aggressive oscillatory cleanup.
- `P1D` families are harmed by bandpass-only processing but benefit from the
  full current preprocessing stack.
- the best waveform-alignment stage is not the same across families.
- therefore preprocessing is part of the observation semantics, not a generic
  hygiene step.

Design implication:

- helper-path preprocessing can remain strongly band-limited
- inference-path preprocessing cannot be treated as one universal recipe
- family-aware observation calibration is now a necessity, not a refinement

One concrete code-level symptom was already present in the older scaffold:
the inference preprocess used `detrend + bandpass`, which removed much of the
low-frequency structure that the baseline state was supposed to explain. The
next-generation path must explicitly avoid this contradiction.

## Observation-model redesign direction

The next major upgrade should not abandon OSSM. It should strengthen the
observation model while preserving linear Gaussian structure conditional on
fixed calibration parameters.

### Stage A: stronger single-family observation model

Stage A now has a concrete precursor already in code:

- helper path remains aggressively band-limited for oscillatory evidence
- inference path is being shifted to a lighter observation preprocess so that
  baseline and residual states remain observable

This split is not the final observation model, but it is a necessary
consistency fix before warm-up calibration can be meaningfully interpreted.

For a given family `m`, replace the implicit "sum everything into one scalar"
view with an explicit calibrated observation model:

`y_t^(m) = c_m + s_m [g_osc^(m) z_osc(t) + g_b^(m) b_t + g_r^(m) r_t] + e_t^(m)`

where:

- `s_m` is a sign term
- `c_m` is an offset term
- `g_osc^(m), g_b^(m), g_r^(m)` are family-specific gains
- `e_t^(m)` is measurement noise with adaptive reliability

Optional fixed trial-level lag:

`y_t^(m) <- y_(t - tau_m)^(m)`

with `tau_m` estimated during warm-up only. Because these quantities are fixed
within a trial after warm-up, the filter remains conditionally linear.

This is the most important near-term observation upgrade because it addresses:

- sign ambiguity
- gain mismatch
- offset mismatch
- mild lag mismatch

without needing to abandon OSSM.

### Stage B: multi-family joint fusion

After the single-family model is stable, extend to joint observations:

`y_t = c + H(theta_cal) x_t + v_t`

where `y_t` stacks the available families and `H(theta_cal)` contains
family-specific calibrated rows.

This enables:

- missing-family tolerance
- per-family reliability weighting through `R_t`
- consensus-based self-calibration
- more direct separation of shared respiratory structure from family-specific corruption

The important point is that fusion should be observation-side, not a vague
late fusion heuristic.

### Stage C: subject/session-specific calibration

Warm-up should estimate fixed trial-level calibration parameters without using
ground-truth belt measurements at inference time.

Candidate warm-up estimates:

- sign
- per-family gain
- offset
- lag
- family reliability prior

These can be estimated from:

- cross-family consensus
- helper-path spectral support
- robust warm-up statistics

The calibration must be unsupervised or weakly supervised at deployment time if
the paper is to retain a low-retuning story.

### Stage D: geometry-aware nuisance modelling

This should be added only after Stages A-C are working.

The goal is not to bolt on a large nonlinear subsystem. The goal is to explain
projection-related nuisance explicitly.

Candidate approaches:

- low-frequency nuisance block driven by ROI motion quality
- family-specific nuisance regressors derived from extractor confidence
- pose/projection surrogate terms if these signals are available

The key rule is that nuisance terms should explain observation corruption, not
be allowed to absorb true respiratory structure.

### Stage E: supervised calibration layer

This is optional and should be treated as a later layer, not the core paper's
main identity.

If used, it should be framed as a supervised readout from the decomposed latent
state rather than as a replacement for the state-space model itself.

It becomes attractive only if:

- unsupervised calibration reaches a clear ceiling
- the dataset count is still sufficient to avoid overfitting
- the paper can justify why a supervised layer is needed

## Target model generations

### PARH-OSSM v1.5

Goal:

- stabilize the current scaffold
- fix the single-family observation model
- keep the paper honest and publishable

Required items:

- fixed trial-level sign/gain/offset/lag calibration
- robust warm-up calibration protocol
- cleaner helper/inference path separation
- stronger residual-release logic tied to unexplained observation energy
- explicit current-vs-final claim boundary in the manuscript

### PARH-OSSM v2

Goal:

- joint multi-family fusion while staying within OSSM

Required items:

- stacked multi-family observation model
- per-family adaptive reliability
- cross-family consensus initialization
- unified decomposition output
- missing-data-aware filtering

### PARH-OSSM v3

Goal:

- stronger real-world gap modelling

Required items:

- geometry-aware nuisance terms
- subject/session calibration priors
- optional supervised calibration head
- deployment-oriented causal helper alternative

## Residual semantics that must be enforced

The residual block is the most conceptually fragile part of the model.

If it becomes "whatever improves CCC," the paper weakens immediately.

The residual must instead be constrained to mean:

- non-oscillatory respiratory content
- support breakdown of the harmonic assumption
- transient respiratory events not well described by the harmonic blocks

This means the residual pathway should open when:

- oscillatory support weakens
- helper-path evidence becomes ambiguous
- observation residual is structured rather than impulsive
- the harmonic explanation becomes insufficient but the signal is still respiratory

It should not open merely because:

- there is a single outlier
- the observation gets noisier for a moment
- the model wants a generic flexible branch

## Evaluation redesign

The final evaluation should answer four different questions, not one.

### Q1. Oscillatory accuracy

Use `z_osc`.

Primary metrics:

- rate MAE
- rate RMSE
- Pearson correlation
- spectral SNR

### Q2. Full-motion fidelity

Use `z_full`.

Primary metrics:

- waveform CCC
- waveform MAE
- waveform DTW

### Q3. Decomposition validity

Primary diagnostics:

- residual energy ratio
- baseline energy ratio
- harmonic energy ratio
- `z_full - z_osc` contribution summary
- event overlays in selected irregular clips

### Q4. Calibration and trustworthiness

Primary diagnostics:

- NIS mean
- NIS in-band rate
- `lambda_t` distribution
- stability duration
- family-wise reliability patterns

## Required regime framing

With only COHFACE and MAHNOB-HCI, the paper must make the regime distinction
explicit.

### COHFACE role

- regular / stationary / clean regime
- do-no-harm benchmark
- tests whether added structure damages easy-case performance

### MAHNOB-HCI role

- irregular / challenging / aperiodic regime
- tests whether residual semantics and adaptive decomposition matter

This framing is strong enough for two datasets if and only if the paper avoids
overstating generalisation.

## Experiment ladder

The experiments should be run in this order.

### E1. Current-code reference

- COHFACE current rerun
- MAHNOB current rerun
- regenerate T3/T4/T6
- regenerate mechanism audit
- regenerate overlay manifests

Purpose:

- lock the true baseline before more redesign

### E2. Observation calibration ablation

Single-family runs with:

- no calibration
- sign only
- sign + gain
- sign + gain + offset
- sign + gain + offset + lag

Purpose:

- identify which observation mismatch matters most

### E3. Residual semantics ablation

Runs with:

- no residual block
- residual block but no release
- residual release from old logic
- residual release from unexplained-energy logic

Purpose:

- prove the residual has an interpretable role

### E4. Adaptation disentanglement ablation

Runs with:

- no adaptive R
- no `q_dyn`
- no `q_osc`
- coupled legacy adaptation
- current disentangled adaptation

Purpose:

- show why one innovation-magnitude logic is not enough

### E5. Fusion prototype

Runs with:

- strongest single family
- fixed pair fusion
- full available-family fusion

Purpose:

- test whether observation-side fusion solves more of the waveform gap than more latent flexibility alone

### E6. Transfer/tuning burden experiment

Lock one global configuration and compare:

- single-family PARH
- fused PARH
- KFstd / OSSM comparator settings

Purpose:

- support reduced-retuning claims without overclaiming tuning-free performance

## Code architecture roadmap

The repository should evolve toward clearer boundaries.

### A. Model layer

Own:

- latent state
- dynamics
- adaptive `Q/R`
- observation operator
- decomposition outputs

### B. Calibration layer

Own:

- warm-up estimation of sign/gain/offset/lag
- family confidence priors
- cross-family consensus

### C. Helper layer

Own:

- oscillatory evidence extraction
- local spectral hypotheses
- support-quality signals

### D. Evaluation layer

Own:

- routing of `z_osc` vs `z_full`
- causal vs smoothed separation
- table-ready artifact generation
- regime stratification
- manifest-based case selection

## Manuscript redesign lock

The final paper should follow this logic.

1. Contactless respiration matters, but one-sinusoid thinking is too narrow.
2. Camera-derived respiratory observations are distorted, heterogeneous proxies.
3. A physiology-aligned latent decomposition plus explicit observation handling is therefore required.
4. PARH-OSSM provides that scaffold while preserving interpretable, auditable state-space inference.
5. COHFACE tests do-no-harm behavior; MAHNOB tests irregular-regime value.
6. The paper reports both strengths and limits, including when the observation model remains the bottleneck.

## Reference strengthening map

The final bibliography should be intentionally balanced across these clusters:

- respiratory physiology and irregular breathing phenomena
- contactless respiration / camera-based respiratory proxy extraction
- oscillator and harmonic state-space modelling
- robust filtering and adaptive noise calibration
- multimodal or multi-observation fusion
- measurement calibration, nuisance modelling, and projection distortion

Do not add references only to inflate count. Each reference cluster must map to
a concrete paragraph or design choice in the paper.

## Immediate next actions

1. Treat the completed COHFACE rerun as the new current reference, not the older production run.
2. Finish MAHNOB rerun under the current code.
3. Run the locked ablation suite.
4. Add observation-calibration design documents before major code refactors.
5. Rewrite the manuscript around the observation-model bottleneck rather than around generic robustness.

## Final warning

If the project keeps adding latent flexibility without strengthening the
observation model, it will become harder to interpret and harder to defend.

The next serious step is not "more adaptation."

It is "better observation modelling inside an OSSM-compatible framework."
