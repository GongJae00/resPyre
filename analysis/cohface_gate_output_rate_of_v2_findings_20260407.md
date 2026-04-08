# COHFACE Gate Subset Findings — Output Rate OF v2

Date: 2026-04-07

## Scope

- Dataset: COHFACE gate subset (12 trials)
- Baseline for comparison: current promoted scaffold (`full`, with
  harmonic-only `obs_cal_v7` on `P1D_quad/cub`)
- Experimental variants:
  - `output_rate_of_v1`
  - `output_rate_of_strong`
  - `output_rate_of_strong_relaxed`

Run root:

- `results/20260407_of_rate_gate_alpha`

## Executive verdict

The original output-only `OF` helper blend (`v1`) is now effectively inert
under the newer scaffold.

The stronger relaxed-gating variant is promotable:

- `blend_alpha = 0.65`
- `min_support = 0.66`
- `min_qdyn = 0.35`
- `min_mismatch_hz = 0.03`

It improves `OF` T3 on the same gate subset without changing waveform metrics.

## Exact median deltas: `output_rate_of_v1 - full`

For `of_farneback__parh_ossm`:

- rate:
  - `MAE +0.000`
  - `RMSE +0.000`
  - `PearsonR +0.000`
  - `Bias +0.000`
- waveform:
  - `CCC +0.000000`
  - `MAE +0.000000`
  - `DTW +0.000000`

Interpretation:

- after the newer observation-row promotions, the old blend is too conservative
  to move the final rate output

## Exact median deltas: `output_rate_of_strong - full`

For `of_farneback__parh_ossm`:

- rate:
  - `MAE -0.030`
  - `RMSE -0.030`
  - `PearsonR +0.130`
  - `Bias -0.144883`
- waveform:
  - `CCC +0.000000`
  - `MAE +0.000000`
  - `DTW +0.000000`

Interpretation:

- stronger blending is already enough to recover useful OF rate information
- waveform remains unchanged, which matches the design intent of an output-side
  T3-only refinement

## Exact median deltas: `output_rate_of_strong_relaxed - full`

For `of_farneback__parh_ossm`:

- rate:
  - `MAE -0.060`
  - `RMSE -0.030`
  - `PearsonR +0.200`
  - `Bias -0.181325`
- waveform:
  - `CCC +0.000000`
  - `MAE +0.000000`
  - `DTW +0.000000`

Interpretation:

- OF still tends to overestimate rate under the current scaffold
- helper evidence is often closer than the latent phase track, but the old gate
  did not activate often enough
- relaxing the gate while increasing blend strength corrects part of that bias
  without touching waveform-state inference

## Promotion decision

Promote:

- `OF` output-only helper blending with:
  - `blend_alpha = 0.65`
  - `min_support = 0.66`
  - `min_qdyn = 0.35`
  - `min_mismatch_hz = 0.03`

Do not keep as default:

- the older `output_rate_of_v1` thresholds

## Next step

The next expensive run should be a new full COHFACE rerun under the promoted
hybrid scaffold:

- `OF`: light path + selective internal freq rescue + stronger output-only
  helper-rate blending
- `P1D_linear`: legacy path + selective freq rescue
- `P1D_quad/cub`: legacy path + selective harmonic-only `obs_cal_v7`
- `DoF`: legacy path only
