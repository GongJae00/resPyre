# COHFACE Family-Confidence Gate Report (2026-04-08)

## Purpose

Close the remaining `P1D_quad/cub` gap by trusting those observations slightly
more when warm-up observation calibration already indicates a very strong
displacement-like fit.

The policy is deliberately narrow:

- allowed families: `profile1d_quadratic`, `profile1d_cubic`
- only when warm-up calibration is enabled
- only when `obs_domain == displacement`
- only when warm-up fit is already excellent

## Variants

- `full`
  - current live scaffold before family-confidence promotion
- `family_confidence_v1`
  - `fit_corr >= 0.97`
  - `fit_rmse <= 0.22`
  - `pi_floor = 0.95`
  - `q_dyn_scale = 0.75`
  - `R_scale = 1.00`
- `family_confidence_v2`
  - `fit_corr >= 0.975`
  - `fit_rmse <= 0.20`
  - `pi_floor = 0.965`
  - `q_dyn_scale = 0.65`
  - `R_scale = 0.92`
- `family_confidence_v3`
  - `fit_corr >= 0.975`
  - `fit_rmse <= 0.20`
  - `pi_floor = 0.97`
  - `q_dyn_scale = 0.55`
  - `R_scale = 0.85`

## 12-trial COHFACE gate summary

### `profile1d_quadratic__parh_ossm`

Baseline `full`:

- rate: `MAE 0.285`, `RMSE 0.370`, `r 0.930`
- waveform: `CCC 0.870235`, `wMAE 0.408708`, `DTW 0.327242`

`v1 - full`:

- rate: `MAE -0.005`, `RMSE -0.005`, `r -0.005`
- waveform: `CCC +0.000152`, `wMAE -0.000351`, `DTW +0.000351`

`v2 - full`:

- rate: `MAE -0.005`, `RMSE -0.010`, `r -0.005`
- waveform: `CCC +0.000110`, `wMAE -0.000405`, `DTW +0.000552`

`v3 - full`:

- rate: `MAE -0.010`, `RMSE -0.010`, `r -0.005`
- waveform: `CCC +0.000074`, `wMAE -0.000451`, `DTW +0.000715`

### `profile1d_cubic__parh_ossm`

Baseline `full`:

- rate: `MAE 0.295`, `RMSE 0.380`, `r 0.925`
- waveform: `CCC 0.870886`, `wMAE 0.412217`, `DTW 0.328569`

`v1 - full`:

- rate: `MAE -0.005`, `RMSE -0.005`, `r +0.000`
- waveform: `CCC +0.000158`, `wMAE -0.000309`, `DTW +0.000092`

`v2 - full`:

- rate: `MAE -0.005`, `RMSE -0.010`, `r +0.000`
- waveform: `CCC +0.000127`, `wMAE -0.000305`, `DTW +0.000157`

`v3 - full`:

- rate: `MAE -0.010`, `RMSE -0.010`, `r +0.000`
- waveform: `CCC +0.000101`, `wMAE -0.000306`, `DTW +0.000227`

## Interpretation

The effect is small, but it is consistent.

- `v1` gave a weak positive signal.
- `v2` improved absolute rate error slightly more without touching other
  families.
- `v3` was the strongest version on the target families and still remained
  zero-impact on `OF`, `DoF`, and `P1D_linear` in the same gate.

The patch does **not** close the Base gap by itself. The gain is too small for
that. But it is a reasonable promotion candidate because:

- it only activates on already well-fit profile families
- it is grounded in observation semantics, not generic retuning
- it slightly reduces over-conservative filtering on the strongest waveform
  families

## Decision

Promote `family_confidence_v3` into the live scaffold as a low-risk candidate
and require one full COHFACE rerun before any paper claim or MAHNOB expansion.
