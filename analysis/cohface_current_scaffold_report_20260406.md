# COHFACE Current Scaffold Report

Date: 2026-04-06

Reference run:

- `results/20260406_cohface_full_current_scaffold/cohface_parh_ossm_prod`

## Completion status

- run status: completed
- steps: estimate, evaluate, metadata
- tables regenerated
- mechanism audit regenerated
- observation EDA regenerated
- case-study manifest regenerated

## Current headline result

The current scaffold is materially better aligned than the older conservative
PARH scaffold, but it is still not the final observation-calibrated model.

What improved:

- `OF` waveform now beats `KFstd`
- `P1D_linear` waveform is roughly tied with a slight edge for PARH
- internal mechanism diagnostics no longer show near-saturated `q_osc`
- residual release is no longer completely dormant

What is still not solved:

- Base remains the strongest rate baseline in 4 of 5 families
- `P1D_quadratic` and `P1D_cubic` waveform fidelity still trail `KFstd`
- `DoF` remains weak and unstable
- explicit observation calibration is still missing from production behavior

## T3: rate summary

From `paper/tables_ready/T3_rate_main.csv`:

- PARH beats `KFstd` on `DoF`, `P1D_linear`, `P1D_quadratic`, `P1D_cubic`
- PARH loses to `KFstd` on `OF`
- Base remains best on `OF` and all `P1D` families
- PARH beats Base only on `DoF`

## T4: waveform summary

From `paper/tables_ready/T4_waveform_main.csv`:

- `OF`: PARH `CCC 0.791` vs `KFstd 0.773`
- `P1D_linear`: PARH `CCC 0.727` vs `KFstd 0.725`
- `P1D_quadratic`: PARH `CCC 0.847` vs `KFstd 0.853`
- `P1D_cubic`: PARH `CCC 0.843` vs `KFstd 0.851`
- `DoF`: PARH `CCC 0.571` vs `KFstd 0.575`

Interpretation:

- the family-aware bridge step helped `OF` substantially
- it did not yet close the gap on the stronger `P1D` families

## T6/T6b: mechanism summary

From `paper/tables_ready/T6_diagnostics_main.csv` and
`paper/tables_ready/T6b_cohface_20260406_mechanism_audit.csv`:

- `NIS mean` is roughly `0.84` to `0.94`
- `NIS in-band` is roughly `0.93` to `0.96`
- `lambda mean` remains near `1.01` to `1.02`
- `q_dyn mean` is now roughly `0.56` to `0.69`
- `q_osc mean` is now roughly `0.73` to `0.83`
- residual energy ratio is roughly `0.7%` to `1.8%`
- `freq_rescue` is active only in the intended families and only at a very low
  fraction

Interpretation:

- the adaptive system is no longer behaving like the old almost-dormant
  decomposition scaffold
- but the observation channel is still the main bottleneck

## Current lock

- current scaffold is strong enough for COHFACE-based iteration
- current scaffold is not strong enough to justify MAHNOB yet
- next step is not another broad rerun
- next step is a better observation-model gate

## Immediate next target

The next promotable model change must be richer than a single global scalar but
safer than free component-wise warm-up regression.

That means:

1. family-specific observation semantics
2. bounded oscillatory calibration
3. optional warm-up lag handling
4. no promotion to production until the 12-trial gate passes
