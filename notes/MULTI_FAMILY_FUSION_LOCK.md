# Multi-Family Fusion Lock

Date: 2026-04-07

## Purpose

This note locks the intended multi-family fusion direction for PARH-OSSM and
the exact comparison logic needed to make the resulting paper defensible.

The goal is not "combine more inputs and hope the metric improves."

The goal is:

- preserve one shared physiology-aligned latent state
- model each family as a different observation of that same state
- exploit cross-family agreement without erasing family-specific meaning
- compare fusion fairly against both single-family and fused non-state-space
  baselines

## Core rule

Do not use naive averaging as the main fusion model.

Bad main design:

- average all raw proxies
- average all preprocessed proxies
- average all estimated rates

These can be supplementary baselines, but not the primary PARH fusion design.

The primary design should be observation-side fusion inside the OSSM.

Current evidence boundary:

- naive scalar fusion is not enough for a main claim
- direct stacked peer-observation pair fusion is still not sufficient on
  COHFACE
- asymmetric assistant-channel fusion is the next exploratory candidate, but
  it is not yet a promoted source of gain

## Intended fusion model

Use a stacked observation model:

`y_t = c + H(theta_cal) x_t + v_t`

where:

- `x_t` is the shared 8D PARH state
- `y_t` stacks the available family observations
- `H(theta_cal)` contains one calibrated observation row per family
- `R_t` is family-specific and adaptive

This means:

- shared respiratory structure is latent
- disagreement between families stays observable
- reliability weighting happens in the filter, not only after the fact

## Family inclusion policy

Do not fuse all families blindly in the first prototype.

Stage-1 fusion should prioritise the families with the clearest complementary
semantics:

- `OF` for oscillatory/rate evidence
- `P1D_quadratic` or `P1D_cubic` for displacement-like morphology

Secondary candidates:

- `P1D_linear` if it improves rate stability without reducing waveform fidelity

Deferred family:

- `DoF` should not be part of the first promoted fusion unless nuisance
  handling improves, because it is the most contamination-prone family

## Fusion ladder

The paper should use a staged fusion ladder, not jump directly to full fusion.

### L1. Best single-family baseline

For each dataset and metric block, identify the best single-family input among:

- `OF`
- `DoF`
- `P1D_linear`
- `P1D_quadratic`
- `P1D_cubic`

This is needed because reviewers will reasonably ask whether the best
single-family proxy is already sufficient.

### L2. Fused-base baseline

Add a non-state-space fusion baseline using the same family subset selected for
PARH fusion.

Acceptable fused-base candidates:

- reliability-weighted average of normalised proxy signals
- reliability-weighted median of normalised proxy signals
- rate-level consensus from per-family base estimates

The point is to show whether gains come from fusion alone or from the OSSM.

### L3. Fused KFstd

Run the same fused observation subset through a simpler state-space baseline.

This isolates the value of:

- fusion
- state-space smoothing

without PARH's extra decomposition semantics.

### L4. Fused PARH

Run the same fused observation subset through PARH-OSSM.

This is the key comparison for the paper:

- same fused observations
- simpler KFstd versus decomposed PARH latent structure

## Required comparisons

The main fusion ablation should therefore contain at least these rows:

1. best single-family Base
2. best single-family KFstd
3. best single-family PARH
4. fused Base
5. fused KFstd
6. fused PARH

Optional supplementary rows:

- pair fusion (`OF + P1D_quad`)
- pair fusion (`OF + P1D_cubic`)
- full available-family fusion

## Why this is the right comparison

Without the fused Base row, any gain can be dismissed as "more inputs."

Without the fused KFstd row, any gain can be dismissed as "fusion plus generic
Kalman smoothing."

Without the best single-family rows, any gain can be dismissed as "you changed
the task by using more measurements."

This exact ladder is therefore required for a strong paper.

## Metric interpretation under fusion

Keep the same metric split:

- `T3`: `z_osc -> track_hz`
- `T4`: `z_full`
- `T6`: diagnostics

Fusion is not allowed to blur this separation.

The fused model should still produce:

- one oscillatory output for rate
- one full-motion output for waveform
- one set of per-family or aggregated diagnostics

## Best first fusion candidate

The most defensible first fusion candidate is:

- `OF + P1D_quadratic`

Why:

- `OF` carries strong oscillatory/rate evidence
- `P1D_quadratic` carries strong displacement/morphology evidence
- their strengths are complementary rather than redundant

The second candidate is:

- `OF + P1D_cubic`

Use `P1D_linear` only if it improves stability and does not dilute morphology.
Keep `DoF` out of the first promoted fusion unless new nuisance handling makes
its contribution interpretable.

## Paper writing rule

The paper should not say:

- "we fused all signals and performance improved"

The paper should say:

- each family offers a different partial view of respiratory motion
- fusion is formulated as a multi-observation state-space model over a shared
  latent respiratory decomposition
- the ablation ladder separates the effect of more measurements from the
  effect of better modelling

## Implementation priority

1. keep current single-family scaffold and results as the reference baseline
2. implement pair fusion first (`OF + P1D_quadratic`)
3. add fused Base and fused KFstd comparators
4. only then evaluate broader family sets

## Current implementation snapshot

As of 2026-04-07, the first fusion prototype is implemented in code.

Implemented method keys:

- `fusion_of_p1d_quadratic`
- `pair_of_p1d_quadratic__kfstd`
- `pair_of_p1d_quadratic__parh_ossm`

Current roles:

- `fusion_of_p1d_quadratic`:
  - scalar fused-base comparator
  - converts OF to a displacement-like surrogate through demeaned integration
  - band-limits and robustly scales both channels
  - spectral-peak-weighted scalar fusion

- `pair_of_p1d_quadratic__kfstd`:
  - stacked multichannel KF baseline over a shared 2D oscillator state

- `pair_of_p1d_quadratic__parh_ossm`:
  - stacked multichannel PARH over the shared 8D latent decomposition
  - channel-wise preprocessing and warm-up calibration
  - aggregated helper frequency/support diagnostics

Current evidence boundary:

- 1-trial COHFACE smoke passes end-to-end
- 12-trial COHFACE gate has now completed
- the gate report is locked in:
  - `analysis/cohface_pairfusion_gate_report_20260407.md`
- a displacement-bridge pair variant was tested and rejected for promotion
- a velocity-aware `OF` peer-observation variant was tested and rejected for
  promotion
- the rejection report is locked in:
  - `analysis/cohface_pair_of_velocity_gate_report_20260407.md`

## New lock after pair-fusion failures

Do not spend more time on symmetric peer-observation fusion variants.

Assistant-channel fusion has now been tested and is kept only as an
exploratory secondary path.

The last tested assistant prototype was:

- primary waveform channel: `P1D_quadratic`
- assistant oscillatory channel: `OF`

Interpretation:

- `P1D_quadratic` should own direct waveform-state observation
- `OF` should help frequency, oscillatory support, or post-filter rate output
- `OF` should not be treated as an equally direct waveform observation unless
  a stronger observation model proves otherwise

Assistant-channel reports:

- `analysis/cohface_assistfusion_gate_report_20260407.md`

Locked outcome:

- `of_rate_assistant_v1`: rejected, harms T3
- `of_rate_assistant_v2`: safe but inert on COHFACE
- assistant fusion therefore remains exploratory and not headline-worthy

## Gate outcome snapshot

The `OF + P1D_quadratic` pair-fusion gate supports promotion to a full
COHFACE comparison run.

`full` profile, 12-trial subset:

- waveform CCC:
  - fused Base: `0.798`
  - pair KFstd: `0.823`
  - pair PARH: `0.838`
- waveform MAE:
  - fused Base: `0.496`
  - pair KFstd: `0.478`
  - pair PARH: `0.425`
- waveform DTW:
  - fused Base: `0.404`
  - pair KFstd: `0.383`
  - pair PARH: `0.344`

Rate remains mixed:

- pair KFstd and pair PARH tie on MAE (`0.440`)
- pair PARH is slightly worse on RMSE
- pair PARH is slightly worse on PearsonR

## Promotion lock

Promoted:

- full COHFACE pair-fusion comparison

Not promoted:

- MAHNOB fusion
- broad multi-family fusion beyond the first pair
- observation-calibration as a claimed source of fusion gain
- assistant-channel fusion as a headline source of gain

## Observation-calibration lock in the stacked setting

`full` versus `no_obs_cal` on the same 12-trial gate remains mixed:

- waveform CCC: small improvement
- waveform DTW: small improvement
- waveform MAE: small regression
- rate MAE/RMSE: tiny improvement
- rate PearsonR: small regression

Therefore:

- the pair-fusion scaffold is promotable
- the stacked observation-calibration mechanism is still not a standalone paper
  claim
- the best current COHFACE story still centers on the strongest single-family
  route, currently `P1D_quadratic`
