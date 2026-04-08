# COHFACE Gate Subset Findings — Observation Calibration v7

Date: 2026-04-07

## Scope

- Dataset: COHFACE gate subset (12 trials)
- Baseline: current promoted scaffold (`full`, former selective `obs_cal_v5`)
- Experimental variants:
  - `obs_cal_v7_harmonic_only`
  - `obs_cal_v7_lowaux`

Run root:

- `results/20260407_singlefamily_obs_gate_v7`

## Executive verdict

`obs_cal_v7_harmonic_only` is the first post-v5 observation-row update that
clearly improves the target `P1D_quadratic` and `P1D_cubic` families on both
T3 and T4 without touching the other families.

`obs_cal_v7_lowaux` is not promotable.

## Exact median deltas: `obs_cal_v7_harmonic_only - full`

Only `P1D_quadratic` and `P1D_cubic` change.

### `profile1d_quadratic__parh_ossm`

- rate:
  - `MAE -0.035`
  - `RMSE -0.040`
  - `PearsonR -0.005`
- waveform:
  - `CCC +0.011845`
  - `MAE -0.004716`
  - `DTW -0.004255`

### `profile1d_cubic__parh_ossm`

- rate:
  - `MAE -0.030`
  - `RMSE -0.050`
  - `PearsonR +0.005`
- waveform:
  - `CCC +0.009475`
  - `MAE -0.013575`
  - `DTW -0.004194`

Interpretation:

- waveform improves clearly for both higher-order profile families
- absolute rate error also improves for both families
- the tiny `quadratic` PearsonR drop is negligible relative to the gains in
  absolute error and waveform fidelity

## Exact median deltas: `obs_cal_v7_lowaux - full`

### `profile1d_quadratic__parh_ossm`

- rate:
  - `MAE -0.015`
  - `RMSE -0.020`
  - `PearsonR +0.000`
- waveform:
  - `CCC +0.000848`
  - `MAE +0.000844`
  - `DTW +0.005658`

### `profile1d_cubic__parh_ossm`

- rate:
  - `MAE -0.010`
  - `RMSE -0.025`
  - `PearsonR +0.005`
- waveform:
  - `CCC +0.000848`
  - `MAE +0.000445`
  - `DTW +0.003353`

Interpretation:

- low auxiliary leakage is still too much auxiliary leakage
- it gives little or no waveform gain and slightly worsens T4 distance metrics

## Mechanistic interpretation

This result supports a stronger semantic statement about the profile families:

- `P1D_quadratic` and `P1D_cubic` are primarily harmonic-morphology
  observations under the current preprocessing chain
- they benefit from bounded `h1/h2` visibility reshaping
- they do **not** benefit from additional baseline/residual visibility in the
  warm-up observation row

This matches the fact that these channels already enter the model through the
legacy band-limited preprocessing stack rather than the light observation path.

## Promotion decision

Promote:

- selective harmonic-only warm-up observation calibration for
  `P1D_quadratic` and `P1D_cubic`

Do not promote:

- low-aux variant
- broader family activation
- any new fusion claim

## Next step

The next expensive run should be a full COHFACE rerun under the new promoted
single-family scaffold:

- `OF`: unchanged
- `P1D_linear`: unchanged
- `P1D_quadratic/cubic`: harmonic-only warm-up observation row
- `DoF`: unchanged
