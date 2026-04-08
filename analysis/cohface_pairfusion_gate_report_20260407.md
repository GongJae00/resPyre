# COHFACE Pair-Fusion Gate Report

Date: 2026-04-07

## Scope

This note records the first promotion gate for the `OF + P1D_quadratic`
pair-fusion scaffold under:

- fused scalar baseline: `fusion_of_p1d_quadratic`
- stacked multichannel KF baseline: `pair_of_p1d_quadratic__kfstd`
- stacked multichannel PARH: `pair_of_p1d_quadratic__parh_ossm`

The gate used the 12-trial COHFACE subset from
`scripts/run_parh_gate_subset.py`.

Profiles:

- `full`
- `no_obs_cal`

## Persistent artifact roots

- `results/20260407_pairfusion_gate/full/cohface_pairfusion_smoke`
- `results/20260407_pairfusion_gate/no_obs_cal/cohface_pairfusion_smoke`

## Primary finding

The pair-fusion scaffold is promotable for a full COHFACE comparison run.

This is supported by the `full` profile:

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

So the stacked multichannel PARH clearly improves waveform fidelity over both
the scalar fused comparator and the stacked multichannel KF baseline on this
gate subset.

## Rate-side finding

The rate story is promising but not yet dominant.

For the same `full` profile:

- rate MAE:
  - pair KFstd: `0.440`
  - pair PARH: `0.440`
  - fused Base: `0.490`
- rate RMSE:
  - pair KFstd: `0.505`
  - pair PARH: `0.540`
- rate PearsonR:
  - pair KFstd: `0.685`
  - pair PARH: `0.625`

This means the first pair-fusion PARH gate win is primarily a waveform/T4
result, not yet a clean T3-rate win over the stacked KF baseline.

## Observation-calibration sensitivity

`full` versus `no_obs_cal` for the stacked pair PARH:

- waveform CCC: `+0.0044`
- waveform DTW: `-0.0038`
- waveform MAE: `+0.0050`
- rate MAE: `-0.005`
- rate RMSE: `-0.010`
- rate PearsonR: `-0.080`

Interpretation:

- warm-up observation calibration is not the main source of the pair-fusion gain
- its effect remains mixed
- the pair-fusion promotion should therefore be framed as a multichannel
  observation result, not as an observation-calibration success claim

## Promotion decision

Allowed next step:

- full COHFACE pair-fusion comparison run

Not yet allowed:

- MAHNOB pair-fusion run
- claim that observation calibration is a validated contributor in the stacked
  setting
- claim that fused PARH already dominates fused KFstd in both waveform and rate

## Working paper claim boundary

What is currently defensible:

- camera-proxy fusion should be formulated as a shared-latent multi-observation
  state-space model rather than naive scalar averaging
- `OF + P1D_quadratic` pair fusion improves waveform fidelity over both fused
  Base and fused KFstd on the COHFACE gate subset

What is still not defensible:

- universal fusion superiority
- rate dominance of fused PARH over fused KFstd
- MAHNOB fusion claims
