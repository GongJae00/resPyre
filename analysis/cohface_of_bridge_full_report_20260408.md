# COHFACE OF-Bridge Full Validation Report

Date: 2026-04-08

## Purpose

This report records the full 160-trial COHFACE validation of the single-family
 `OF` displacement bridge.

Reference run:

- `results/20260407_of_bridge_full/cohface_of_bridge_gate`

Question:

- should `of_disp_bridge` replace raw `OF` outright,
- be rejected,
- or be promoted as an additional observation family?

## Headline conclusion

`OF_bridge` is promotable as a new observation family, but not as a strict
replacement for raw `OF`.

Why:

- it substantially improves OF-family rate estimation for both `KFstd` and
  PARH
- it keeps waveform fidelity in the same range, with slightly better
  `waveform_MAE` and `DTW` for PARH
- but it does not improve the OF-family headline waveform `CCC`
- therefore the correct promotion is:
  - keep raw `OF`
  - add `OF_bridge`
  - do not silently overwrite the old OF route

## Full COHFACE medians

### Raw `OF`

Base:

- `CCC 0.711`
- `waveform_MAE 0.576`
- `DTW 0.452`
- `rate MAE 0.290`
- `rate RMSE 0.390`
- `rate r 0.890`

KFstd:

- `CCC 0.773`
- `waveform_MAE 0.516`
- `DTW 0.403`
- `rate MAE 0.550`
- `rate RMSE 0.650`
- `rate r 0.730`

PARH:

- `CCC 0.791`
- `waveform_MAE 0.511`
- `DTW 0.403`
- `rate MAE 0.510`
- `rate RMSE 0.565`
- `rate r 0.740`

### `OF_bridge`

Base:

- `CCC 0.685`
- `waveform_MAE 0.629`
- `DTW 0.466`
- `rate MAE 0.245`
- `rate RMSE 0.325`
- `rate r 0.900`

KFstd:

- `CCC 0.789`
- `waveform_MAE 0.503`
- `DTW 0.396`
- `rate MAE 0.260`
- `rate RMSE 0.335`
- `rate r 0.835`

PARH:

- `CCC 0.777`
- `waveform_MAE 0.504`
- `DTW 0.399`
- `rate MAE 0.295`
- `rate RMSE 0.370`
- `rate r 0.850`

## Direct PARH comparison

`of_disp_bridge__parh_ossm` minus `of_farneback__parh_ossm`:

- `waveform_CCC -0.0138`
- `waveform_MAE -0.0062`
- `waveform_DTW -0.0031`
- `rate MAE -0.215`
- `rate RMSE -0.195`
- `rate PearsonR +0.110`
- `rate Bias -0.412`

Interpretation:

- T3 improvement is large and clear
- T4 is mixed:
  - `CCC` regresses modestly
  - `wMAE` and `DTW` improve slightly
- this is enough to justify promotion as a new family, but not enough to claim
  that raw `OF` is obsolete

## Trial-level behavior

Per-trial raw-vs-bridge PARH comparison:

- `rate MAE` improved on `58.8%` of trials
- `rate RMSE` improved on `58.1%` of trials
- `rate PearsonR` improved on `51.3%` of trials
- absolute bias improved on `86.9%` of trials
- `waveform_CCC` improved on only `43.1%` of trials
- `waveform_MAE` improved on `46.3%` of trials
- `waveform_DTW` improved on `42.5%` of trials

Interpretation:

- the bridge consistently reduces OF rate bias and large-rate failures
- waveform gains are not dominant enough to justify replacing raw `OF`

## Diagnostics

PARH calibration summary:

Raw `OF`:

- `NIS_Mean 0.937`
- `NIS_InBand 0.929`
- `Lambda_Mean 1.019`
- `strict_pass_rate 0.531`
- `relaxed_pass_rate 0.975`

`OF_bridge`:

- `NIS_Mean 0.841`
- `NIS_InBand 0.950`
- `Lambda_Mean 1.015`
- `strict_pass_rate 0.069`
- `relaxed_pass_rate 0.994`

Interpretation:

- the bridge is not unstable
- but it is more over-strict under the current strict NIS split
- that makes it a useful observation family, not yet a clean calibration win

Mechanism summary from PKL audit:

- `q_dyn_mean_median`: `0.594 -> 0.561`
- `q_osc_mean_median`: `0.837 -> 0.737`
- `R_mean_median`: `0.287 -> 0.064`
- `output_rate_blend_active_frac_median`: `0.271 -> 0.153`

Interpretation:

- the bridge makes the OF family look more observation-consistent and less
  noisy to the filter
- but the resulting signal is not a uniformly better waveform proxy

## Promotion decision

Decision:

- promote `OF_bridge` as an additional observation family
- do not replace raw `OF`
- rerun the all-family COHFACE ladder with both `OF` and `OF_bridge`
- keep MAHNOB blocked until that all-family COHFACE rerun is complete

This preserves honesty:

- raw `OF` still has the best OF-family waveform `CCC` for PARH
- `OF_bridge` is clearly better for OF-family rate tracking and comparable on
  other waveform metrics

The correct paper framing is therefore:

- raw `OF`: canonical velocity-like optical-flow proxy
- `OF_bridge`: displacement-compatible OF-derived family that improves OF rate
  tracking and narrows the OF observation mismatch
