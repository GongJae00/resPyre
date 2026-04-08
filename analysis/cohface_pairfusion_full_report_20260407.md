# COHFACE Pair-Fusion Full Report

Date: 2026-04-07

## Run

- results root:
  - `results/20260407_cohface_pairfusion_prod/cohface_pairfusion_prod`
- status:
  - `completed`

Primary metric sources:

- `metrics/metrics_freq_domain_raw.csv`
- `metrics/metrics_waveform_raw.csv`
- `metrics/metrics_filter_diagnostics_raw.csv`

## Main outcome

The promoted `OF + P1D_quadratic` pair-fusion scaffold is useful, but it is not
yet the main winning path on COHFACE.

It does beat the corresponding fused KF and fused scalar comparator, but it
does not beat the best single-family `P1D_quadratic` route.

## Fused comparison

### Rate (T3)

- fused Base (`fusion_of_p1d_quadratic`)
  - MAE: `0.245`
  - RMSE: `0.330`
  - PearsonR: `0.890`
- fused KF (`pair_of_p1d_quadratic__kfstd`)
  - MAE: `0.480`
  - RMSE: `0.580`
  - PearsonR: `0.770`
- fused PARH (`pair_of_p1d_quadratic__parh_ossm`)
  - MAE: `0.445`
  - RMSE: `0.525`
  - PearsonR: `0.790`

Interpretation:

- fused PARH beats fused KF on all three primary rate metrics
- fused PARH does not beat fused Base on rate

### Waveform (T4, PARH uses `z_full`)

- fused Base (`fusion_of_p1d_quadratic`)
  - CCC: `0.757`
  - MAE: `0.545`
  - DTW: `0.423`
- fused KF (`pair_of_p1d_quadratic__kfstd`)
  - CCC: `0.808`
  - MAE: `0.480`
  - DTW: `0.382`
- fused PARH (`pair_of_p1d_quadratic__parh_ossm`)
  - CCC: `0.814`
  - MAE: `0.460`
  - DTW: `0.379`

Interpretation:

- fused PARH beats fused KF and fused Base on waveform
- the waveform margin over fused KF is real but modest

## Best single-family gap

### Best single-family overall for rate

- method:
  - `profile1D quadratic`
- metrics:
  - MAE: `0.195`
  - RMSE: `0.270`
  - PearsonR: `0.950`

### Best single-family PARH for rate

- method:
  - `profile1d_quadratic__parh_ossm`
- metrics:
  - MAE: `0.285`
  - RMSE: `0.375`
  - PearsonR: `0.885`

### Fused PARH vs best single-family rate

Against best single overall:

- MAE: `0.445 - 0.195 = +0.250`
- RMSE: `0.525 - 0.270 = +0.255`
- PearsonR: `0.790 - 0.950 = -0.160`

Against best single PARH:

- MAE: `0.445 - 0.285 = +0.160`
- RMSE: `0.525 - 0.375 = +0.150`
- PearsonR: `0.790 - 0.885 = -0.095`

### Best single-family overall for waveform

The best single-family waveform route remains `P1D_quadratic`.

Metric-wise best rows:

- lowest MAE:
  - `profile1d_quadratic__kfstd`
  - `0.416`
- lowest DTW:
  - `profile1d_quadratic__kfstd`
  - `0.343`
- highest CCC:
  - `profile1d_quadratic__parh_ossm`
  - `0.853`

### Fused PARH vs best single-family waveform

Against best single overall MAE/DTW and best single CCC:

- CCC: `0.814 - 0.853 = -0.039`
- MAE: `0.460 - 0.416 = +0.044`
- DTW: `0.379 - 0.343 = +0.036`

Against best single PARH:

- CCC: `0.814 - 0.853 = -0.039`
- MAE: `0.460 - 0.422 = +0.038`
- DTW: `0.379 - 0.344 = +0.035`

## Diagnostic interpretation

The fused PARH diagnostics are healthy:

- NIS mean: `0.963`
- Lambda mean: `1.023`
- strict pass: `0.869`

So the pair-fusion shortfall is not a calibration-collapse problem.

It is primarily an information/modelling problem:

- `P1D_quadratic` alone is already very strong on COHFACE
- the current pair observation model does not extract enough extra value from
  the `OF` channel to offset the added observation mismatch

## Working conclusion

Allowed paper use:

- pair fusion as an ablation/secondary result
- evidence that shared-latent multichannel fusion is viable and beats fused KF
  on waveform

Not yet allowed:

- using pair fusion as the main headline result
- claiming that fusion beats the best single-family route on COHFACE
