# COHFACE Observation-Redesign Findings

Date:

- 2026-04-03

Scope:

- completed COHFACE rerun under `results/20260403_cohface_rerun`
- full observation EDA from:
  - `analysis/cohface_observation_eda_trials.csv`
  - `analysis/cohface_observation_eda_family.csv`
  - `analysis/cohface_preproc_summary.csv`
  - `analysis/cohface_preproc_deltas.csv`
- smoke ablation from:
  - `results/20260403_obs_path_ablation_smoke_v2/full`
  - `results/20260403_obs_path_ablation_smoke_v2/legacy_obs_path`

## Why this note exists

This note locks the first observation-side findings that materially change the
next PARH redesign direction.

The key question was whether a single global observation preprocess is
reasonable. The answer from COHFACE is already no.

## Full-EDA lock

Best waveform-alignment stage by family, using `corr_wave_best_median` from
`analysis/cohface_preproc_summary.csv`:

- `OF`: `bandpass_only` = `0.6871`
- `DoF`: `helper_preprocess` = `0.5281`
- `P1D_lin`: `current_preprocess` = `0.6324`
- `P1D_quad`: `current_preprocess` = `0.7598`
- `P1D_cub`: `current_preprocess` = `0.7541`

Best derivative-alignment stage by family, using `corr_deriv_best_median`:

- `OF`: `helper_preprocess` = `0.6567`
- `DoF`: `helper_preprocess` = `0.4521`
- `P1D_lin`: `current_preprocess` = `0.5678`
- `P1D_quad`: `helper_preprocess` = `0.7433`
- `P1D_cub`: `helper_preprocess` = `0.7396`

Most important negative finding:

- `bandpass_only` harms all `P1D` families badly.
  - `P1D_lin` waveform-corr delta vs raw median: `-0.0632`
  - `P1D_quad` waveform-corr delta vs raw median: `-0.1907`
  - `P1D_cub` waveform-corr delta vs raw median: `-0.1765`

Most important positive finding:

- aggressive oscillatory cleanup strongly helps `OF` and `DoF`.
  - `OF current_preprocess` waveform-corr delta vs raw median: `+0.3412`
  - `OF current_preprocess` derivative-corr delta vs raw median: `+0.5952`
  - `DoF helper_preprocess` waveform-corr delta vs raw median: `+0.3841`
  - `DoF helper_preprocess` derivative-corr delta vs raw median: `+0.4254`

## Smoke ablation lock

The first structural code correction introduced a PARH-specific light
observation path and compared it against the legacy preprocess on a 1-sample
COHFACE smoke run.

After a second iteration, the observation path became family-aware:

- `OF` and `DoF` use the light observation path
- `P1D` families keep the stronger legacy/current preprocess

Smoke comparison: `full` versus `legacy_obs_path`

- `OF`
  - waveform CCC: `0.7842` vs `0.7060`
  - waveform MAE: `0.5452` vs `0.6051`
  - rate MAE: `0.15` vs `0.16`
- `DoF`
  - waveform CCC: `0.3073` vs `0.3663`
  - waveform MAE: `0.9375` vs `0.9067`
  - rate MAE: `1.37` vs `3.13`
- `P1D_lin`
  - waveform CCC: identical `0.9363`
  - waveform MAE: identical `0.2753`
  - rate MAE: identical `0.10`
- `P1D_quad`
  - waveform CCC: identical `0.9050`
  - waveform MAE: identical `0.3302`
  - rate MAE: identical `0.09`
- `P1D_cub`
  - waveform CCC: identical `0.9031`
  - waveform MAE: identical `0.3335`
  - rate MAE: identical `0.09`

## Locked interpretation

The observation redesign direction is now constrained by evidence:

- one global preprocess is not defensible
- `P1D` families already behave like stronger waveform proxies under the current
  preprocessing stack
- `OF` benefits from a lighter inference path once helper evidence remains
  band-limited
- `DoF` exposes an explicit rate-versus-waveform trade-off under the light path

Therefore the next serious step is not another generic preprocess tweak.

It is:

- family-aware observation calibration
- warm-up sign/gain/offset estimation
- explicit observation rows
- later, multi-family fusion

## Immediate code implication

The interim family-aware observation path is justified as a bridge step.

It is not the final answer.

The final answer still requires replacing the fixed observation row with a
calibrated observation model.
