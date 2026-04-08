# COHFACE Residual-Release v4 Gate Report

Date: 2026-04-08

## Purpose

Test a more interpretable residual-release pathway for PARH-OSSM.

Instead of using only oscillatory-support degradation, `residual_release_v4`
computes an observation-driven non-oscillatory need from the gap between:

- oscillator-only observation support
- full-state observation support

This is intended to move the residual branch closer to an identifiable
observation-based mechanism rather than a generic release heuristic.

Profiles compared:

- `full`
- `residual_release_v4`

Gate dataset:

- 12-trial COHFACE subset

Result roots:

- `results/20260408_residual_release_v4_gate/full`
- `results/20260408_residual_release_v4_gate/residual_release_v4`

## Implementation summary

New diagnostic quantities:

- `obs_full_support_t`
- updated `obs_nonosc_need_t` based on a weighted combination of:
  - non-oscillatory gap: `obs_full_support - obs_osc_support`
  - residual unexplainedness: `1 - obs_full_support`

Experimental gate profile:

- `RESPYRE_PARH_Q_OSC_OBS_MODE=penalize_nonosc_gap_v1`
- `RESPYRE_PARH_Q_OSC_OBS_WEIGHT=0.22`
- `RESPYRE_PARH_Q_APER_OBS_GAMMA=0.50`

## Gate result

The new policy is effectively inert on clean COHFACE.

Representative median deltas (`residual_release_v4 - full`):

- raw `OF` PARH:
  - rate `MAE 0.670 -> 0.670`
  - rate `RMSE 0.840 -> 0.840`
  - waveform `CCC 0.78091 -> 0.78085`
  - waveform `MAE 0.52774 -> 0.52800`
- `OF_bridge` PARH:
  - no meaningful change
- `P1D_quadratic` PARH:
  - no meaningful change
- `P1D_cubic` PARH:
  - no meaningful change

## Diagnostic interpretation

Raw `OF` PARH medians:

- `full`
  - `q_osc_mean 0.8543`
  - `obs_full_support_mean 0.9500`
  - `obs_nonosc_need_mean 0.0627`
- `residual_release_v4`
  - `q_osc_mean 0.8447`
  - `obs_full_support_mean 0.9501`
  - `obs_nonosc_need_mean 0.0391`

This confirms that the new formulation changes the semantics of the residual
diagnostic, but on COHFACE the clean-data regime still leaves very little
structured non-oscillatory need to exploit.

## Verdict

`No-go` for live promotion, but `keep` as a diagnostic redesign step.

Why:

- it does not improve T3 or T4 on the clean gate subset
- it does provide a more interpretable observation-driven quantity
- that quantity may still become useful on MAHNOB or in later residual-branch
  redesigns

## Implication

Current live residual design remains unchanged.

The next residual redesign should not merely scale `Q_aper` harder. It should
move toward a more identifiable residual branch, potentially using:

- explicit non-oscillatory support diagnostics in tables/figures
- event-oriented residual case studies
- cleaner separation between observation mismatch and true aperiodic content
