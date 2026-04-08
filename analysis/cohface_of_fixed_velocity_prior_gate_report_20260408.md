# COHFACE Raw-OF Fixed Velocity Prior Gate Report

Date: 2026-04-08

## Purpose

Test whether raw `OF` should fall back to a conservative velocity-domain
observation prior instead of the current fixed displacement-style bridge
fallback when warm-up observation calibration is unavailable.

Profiles compared:

- `full`
- `of_fixed_velocity_prior_v1`

Gate dataset:

- 12-trial COHFACE subset used by the existing single-family gate runner

Result roots:

- `results/20260408_of_fixed_velocity_prior_gate_v2/full`
- `results/20260408_of_fixed_velocity_prior_gate_v2/of_fixed_velocity_prior_v1`

## Important implementation note

The first run under `results/20260408_of_fixed_velocity_prior_gate` was invalid
as a comparison artifact because the fallback prior was accidentally blocked by
the same `OBS_CAL_ALLOWED_FAMILIES` filter used for warm-up calibration.

That wiring bug has been fixed. Only `results/20260408_of_fixed_velocity_prior_gate_v2`
should be treated as valid evidence for this experiment.

## Metadata verification

Representative trial `cohface_3_2`, raw `OF` PARH:

- `full`
  - `observation_model_type = fixed_sum_bridge_v1`
  - `fallback_mode = fixed_sum`
  - `obs_domain = displacement`
- `of_fixed_velocity_prior_v1`
  - `observation_model_type = fixed_of_velocity_prior_v1`
  - `fallback_mode = of_velocity_prior_v1`
  - `obs_domain = velocity`

The experimental switch therefore took effect in the corrected rerun.

## Gate summary for raw OF PARH

Median over the 12-trial gate subset:

- `full`
  - rate: `MAE 0.670`, `RMSE 0.840`, `PearsonR 0.720`
  - waveform: `CCC 0.781`, `MAE 0.528`, `DTW 0.383`
  - diagnostics: `NIS 0.908`, `NIS_InBand 0.927`, `Lambda 1.020`, `Pr(lambda<1) 0.081`, `Stability 12.225 s`
- `of_fixed_velocity_prior_v1`
  - rate: `MAE 0.715`, `RMSE 0.760`, `PearsonR 0.740`
  - waveform: `CCC 0.662`, `MAE 0.663`, `DTW 0.471`
  - diagnostics: `NIS 0.896`, `NIS_InBand 0.936`, `Lambda 1.017`, `Pr(lambda<1) 0.077`, `Stability 8.150 s`

Delta (`velocity_prior - full`):

- rate
  - `MAE +0.045`
  - `RMSE -0.080`
  - `PearsonR +0.020`
- waveform
  - `CCC -0.119`
  - `MAE +0.135`
  - `DTW +0.088`

## Verdict

`No-go`.

The fixed velocity prior changes raw `OF` semantics in the intended direction,
but the resulting trade-off is not acceptable:

- slight T3 shape improvement (`RMSE`, `PearsonR`)
- worse primary absolute T3 error (`MAE`)
- large T4 collapse

This is not a promotable fallback for the live raw-`OF` route.

## Implication

Current `OF` design lock remains:

- raw `OF` stays helper-heavy under the existing live scaffold
- `OF_bridge` remains the promoted OF-derived observation family for rate gains
- raw `OF` direct velocity semantics remain experimental only

The next `OF` redesign should not be a fixed fallback row alone. It must
either:

- model velocity/displacement mismatch more explicitly with phase-consistent
  dynamics, or
- keep `OF` as a support/helper family while bridged observations carry the
  displacement-compatible route.
