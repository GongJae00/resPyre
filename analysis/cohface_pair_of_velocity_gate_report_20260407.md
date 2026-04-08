# COHFACE Pair-OF Velocity Gate Report

Date: 2026-04-07

## Question

Can `OF + P1D_quadratic` pair fusion be improved by giving the `OF` channel a
velocity-aware warm-up observation row instead of treating it as a lagged
displacement-like peer observation?

Test run:

- baseline gate: `results/20260407_pair_of_velocity_gate/full`
- velocity-aware gate: `results/20260407_pair_of_velocity_gate/obs_cal_of_velocity_v1`

Config:

- `/home/gongjae/Projects/resPyre/configs/cohface_pairfusion_smoke.json`
- 12-trial COHFACE subset via `scripts/run_parh_gate_subset.py`

## What changed

The raw `OF` channel was kept as a direct pair observation, but its warm-up
family prior/observation row was changed from lagged displacement semantics to
velocity semantics:

- harmonic part observes the time derivative of the lagged harmonic projection
- auxiliary part observes `[b_dot, r_dot]` instead of `[b, r]`

This was enabled only in the gate profile:

- `RESPYRE_PARH_OBS_CAL_ALLOWED_FAMILIES=of_farneback,profile1d_quadratic,profile1d_cubic`
- `RESPYRE_PARH_OBS_CAL_MODE=family_phase_aux`

## Median results for `pair_of_p1d_quadratic__parh_ossm`

Baseline `full`:

- rate MAE: `0.440`
- rate RMSE: `0.540`
- rate PearsonR: `0.625`
- z_full CCC: `0.837`
- z_full MAE: `0.443`
- z_full DTW: `0.339`

Velocity-aware `obs_cal_of_velocity_v1`:

- rate MAE: `0.505`
- rate RMSE: `0.595`
- rate PearsonR: `0.815`
- z_full CCC: `0.824`
- z_full MAE: `0.467`
- z_full DTW: `0.362`

Delta (`velocity-aware - baseline`):

- rate MAE: `+0.065`
- rate RMSE: `+0.055`
- rate PearsonR: `+0.190`
- z_full CCC: `-0.013`
- z_full MAE: `+0.024`
- z_full DTW: `+0.023`

## Verdict

`No-go` for promotion.

Why:

- the `OF` velocity-aware peer row improves correlation but worsens the
  primary absolute rate errors
- waveform fidelity also degrades
- this means `OF` still does not behave like a healthy peer measurement for
  the shared pair-update state

## Interpretation

This does **not** invalidate the broader multi-family idea.

It invalidates the narrower assumption that raw `OF` should directly
participate as a peer observation row in the same stacked update as
`P1D_quadratic`.

Current evidence now rejects two direct pair strategies on COHFACE:

1. `OF -> displacement bridge -> peer observation`
2. `OF -> velocity-aware peer observation`

## Next design

The remaining defensible direction is asymmetric fusion:

- `P1D_quadratic` remains the primary waveform/morphology observation
- `OF` becomes an assistant channel
- `OF` influences helper frequency evidence, oscillatory support, or rate
  output refinement
- `OF` does **not** directly pull the shared waveform state as a full peer
  measurement

This will be referred to as assistant-channel fusion in follow-up design docs.
