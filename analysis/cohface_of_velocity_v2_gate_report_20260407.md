# COHFACE OF Velocity Observation Gate Report (2026-04-07)

## Verdict

Direct OF velocity-row calibration is `no-go` as a promoted observation-model
change.

## First result without fit gating

Profile:

- `obs_cal_of_velocity_v2`
- enables OF warm-up velocity-domain observation calibration alongside the
  already-promoted `P1D_quad/cub` calibration
- OF prior is harmonic-only, short-lag, low-`h2`

Observed median deltas versus `full` on the 12-trial COHFACE gate subset:

- `OF` rate `MAE +0.875`
- `OF` rate `RMSE +0.845`
- `OF` rate `PearsonR -0.230`
- `OF` waveform `CCC -0.122`
- `OF` waveform `MAE +0.137`
- `OF` waveform `DTW +0.032`

Warm-up calibration diagnostics on a representative trial showed why:

- `obs_domain = velocity`
- `g_h1 = 0.141`
- `g_h2 = 0.052`
- `lag_sec = -0.05`
- `fit_corr = 0.177`
- `fit_rmse = 1.348`

Interpretation:

- the OF warm-up fit is too weak to justify a direct observation-row rewrite
- the fitted row is effectively a poor surrogate rather than a valid sensor
  model

## Safety change

A fit-quality gate was added to warm-up observation calibration:

- family priors now include minimum acceptable fit correlation and maximum
  normalized fit RMSE
- OF-specific thresholds are stricter than generic thresholds
- if the fitted row does not meet those criteria, calibration is disabled and
  the method falls back to the fixed live scaffold

Result after the safeguard:

- the same experimental OF profile becomes inert on the tested subset
- representative OF meta now reports:
  - `enabled = False`
  - `observation_model_type = fixed_sum_bridge_v1`
- subset medians then match the `full` baseline exactly for `OF`:
  - rate `MAE 0.67`, `RMSE 0.84`, `PearsonR 0.72`
  - waveform `CCC 0.7809`, `MAE 0.5277`, `DTW 0.3826`

## Conclusion

The correct conclusion is not that OF is useless.

The correct conclusion is:

- OF is not currently a valid direct full-waveform observation row
- OF is still useful as oscillatory/helper evidence
- the next OF redesign should target:
  - stronger helper-path dynamics semantics
  - or a different latent/observation formulation
  - not another free or weakly-constrained direct observation-row fit
