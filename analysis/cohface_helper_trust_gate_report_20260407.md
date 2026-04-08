# COHFACE Helper-Trust Gate Report

Date: 2026-04-07

## Purpose

This note locks the outcome of the `OF` helper-trust experiments on the
12-trial COHFACE gate subset.

The helper-trust idea was tested because the current bottleneck is still
`OF` rate behavior under a scaffold where `OF` remains helper-heavy rather than
observation-row-heavy.

The original question was:

- can helper trust make `OF` rescue or helper-rate refinement safer
  without harming waveform behavior?

## Profiles tested

### Original helper-trust policies

- `helper_trust_of_v1`
  - helper trust computed from helper support and helper-frequency stability
  - helper trust directly suppressed `q_dyn`
- `helper_trust_of_v1_rescue_v2`
  - same as above, plus `of_v2` rescue thresholds

### Rescue-only helper-trust policies

- `helper_trust_rescue_only_v1`
  - helper trust still computed and logged
  - helper trust no longer suppresses `q_dyn`
  - trust only used when deciding whether rescue is safe
- `helper_trust_rescue_only_v1_rescue_v2`
  - same rescue-only trust logic
  - plus `of_v2` rescue thresholds

## Locked verdict

### 1. Helper trust must not suppress `q_dyn`

`helper_trust_of_v1` is a no-go.

It lowered `OF` `q_dyn` as intended, but that removed useful oscillatory
adaptation and worsened `OF` rate behavior. The mechanism change happened, but
in the wrong direction.

### 2. Rescue-only helper trust is safe but mostly inert

`helper_trust_rescue_only_v1` is effectively inert.

Against the same gate subset baseline:

- `OF waveform_CCC`: `+0.000793`
- `OF waveform_MAE`: `-0.001345`
- `OF waveform_DTW`: `-0.000411`
- `OF rate MAE`: `0.000000`
- `OF rate RMSE`: `0.000000`
- `OF rate PearsonR`: `0.000000`

This is acceptable as a safety property, but it is not promotable because it
does not create a real gain source.

### 3. Rescue-only helper trust plus `of_v2` rescue is mixed, not promotable

`helper_trust_rescue_only_v1_rescue_v2` is also a no-go as a live promotion.

Relative to the gate baseline:

- `OF waveform_CCC`: `-0.017374`
- `OF waveform_MAE`: `+0.021987`
- `OF waveform_DTW`: `-0.004800`
- `OF rate MAE`: `-0.040000`
- `OF rate RMSE`: `-0.040000`
- `OF rate PearsonR`: `+0.070000`
- `OF rate Bias`: `+0.031974`

Interpretation:

- T3 moves in a partially positive direction
- but T4 gets clearly worse
- the net result is not a no-harm promotion

## Mechanism interpretation

For `helper_trust_rescue_only_v1_rescue_v2`, the `OF` family shows:

- `q_dyn_mean_median`: `0.599051`
- `helper_trust_mean_median`: `0.799009`
- `helper_bias_conf_mean_median`: `0.375007`
- `helper_mismatch_mean_median`: `0.064157`
- `freq_rescue_active_frac_median`: `0.002062`
- `output_rate_blend_active_frac_median`: `0.276107`

This matters because it confirms that the rescue-only design behaved as
intended:

- helper trust no longer suppressed `q_dyn`
- trust and bias confidence only acted as rescue-side evidence

However, even this narrower use still did not create a clean promotion. The
problem is therefore not just unsafe gating logic. The deeper bottleneck is
that `OF` helper evidence still does not map cleanly enough into the shared
rate/waveform scaffold.

## Final design lock from this experiment

- helper-trust-driven `q_dyn` suppression is rejected
- rescue-only helper trust is allowed to remain as an experimental profile
- rescue-only helper trust is not promoted as a live default
- the next `OF` redesign step should target observation semantics directly,
  not helper-trust heuristics

## Consequence for the live scaffold

The live scaffold remains:

- `OF`: light path + selective internal frequency rescue + conservative
  output-only helper-rate blending
- `P1D_linear`: legacy path + selective frequency rescue
- `P1D_quad/cub`: legacy path + selective harmonic-only observation-row
  calibration
- `DoF`: legacy path only
