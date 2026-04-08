# COHFACE residual semantics v1 gate

Date: 2026-04-08

Reference runs:
- `full`: `/home/gongjae/Projects/resPyre/results/20260408_residual_semantics_gate_rerun/full/cohface_parh_ossm_prod_ofbridge`
- `residual_semantics_v1`: `/home/gongjae/Projects/resPyre/results/20260408_residual_semantics_gate_rerun/residual_semantics_v1/cohface_parh_ossm_prod_ofbridge`

## Goal

Test whether a family-aware residual prior can make the residual branch more
interpretable without harming the clean COHFACE scaffold.

## What changed

- `residual_prior_scale` now depends on family semantics.
- `obs_nonosc_need_eff`, `residual_prior_t`, and `aper_drive_t` are saved.
- Gate profile:
  - `RESPYRE_PARH_ENABLE_RESIDUAL_SEMANTICS=1`
  - `RESPYRE_PARH_Q_OSC_OBS_MODE=penalize_nonosc_gap_v1`
  - `RESPYRE_PARH_Q_OSC_OBS_WEIGHT=0.22`
  - `RESPYRE_PARH_Q_APER_OBS_GAMMA=0.50`
  - `RESPYRE_PARH_RESIDUAL_PRIOR_MIN=0.10`
  - `RESPYRE_PARH_RESIDUAL_PRIOR_POWER=1.00`

## Method-level deltas vs `full`

- `of_farneback__parh_ossm`
  - rate: `MAE +0.000`, `RMSE +0.000`, `r -0.005`
  - waveform: `CCC +0.0001`, `MAE -0.0002`, `DTW -0.0001`
- `of_disp_bridge__parh_ossm`
  - rate: `MAE +0.010`, `RMSE +0.010`, `r -0.005`
  - waveform: `CCC +0.0007`, `MAE -0.0021`, `DTW -0.0017`
- `profile1d_linear__parh_ossm`
  - rate: `MAE -0.010`, `RMSE -0.015`, `r +0.010`
  - waveform: effectively unchanged
- `profile1d_quadratic__parh_ossm`
  - rate/waveform: unchanged
- `profile1d_cubic__parh_ossm`
  - rate/waveform: unchanged
- `dof__parh_ossm`
  - rate: `MAE +0.035`, `RMSE +0.030`, `r -0.040`
  - waveform: slightly worse

## Mechanism shift

The redesign did what it was supposed to do diagnostically:

- `residual_prior_mean_mean` was reduced by family:
  - `DoF -0.70`
  - `OF -0.30`
  - `OF_bridge -0.45`
  - `P1D_lin -0.55`
  - `P1D_quad -0.90`
  - `P1D_cub -0.88`
- `obs_nonosc_need_eff` and `aper_drive` both dropped accordingly.
- Residual energy barely moved on clean COHFACE.

## Verdict

`no-go` for live promotion.

Interpretability improved, but the clean COHFACE scaffold did not gain enough
to justify a new default policy. Keep it as a diagnostic branch only.
