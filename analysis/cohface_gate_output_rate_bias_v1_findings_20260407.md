# COHFACE Gate Findings — OF Bias-Aware Output Rate v1

Date: 2026-04-07

## Scope

- Baseline gate profile:
  - `results/20260407_of_output_bias_gate/full`
- Experimental gate profile:
  - `results/20260407_of_output_bias_gate/output_rate_of_bias_v1`

Both use the same promoted single-family scaffold:

- `P1D_quad/cub`: harmonic-only `obs_cal_v7`
- `OF`: unchanged inference path
- only the output-side `OF` rate policy differs

## Change under test

Experimental policy:

- `RESPYRE_PARH_OUTPUT_RATE_POLICY=of_helper_bias_v1`
- correction uses recent signed helper-track mismatch
- correction is bounded and only activates when mismatch sign is stable over a
  recent horizon
- waveform-state inference remains unchanged

## Gate verdict

`mixed / no-go`

The bias-aware policy improves some secondary rate properties on the 12-trial
COHFACE gate subset, but it does **not** improve the primary absolute rate
error.

## Exact median deltas: `output_rate_of_bias_v1 - full`

For `of_farneback__parh_ossm`:

- rate:
  - `MAE +0.040`
  - `RMSE -0.080`
  - `PearsonR +0.055`
  - `Bias -0.0311`
- waveform:
  - `CCC +0.0000`
  - `MAE +0.0000`
  - `DTW +0.0000`

## Interpretation

The new policy is safer than the previously over-relaxed helper blend because
it does not change T4 and it reduces some large deviations. However, it still
fails the main criterion for promotion:

- it does not lower `OF` rate `MAE`

That means the current `OF` bottleneck is not just instantaneous helper
overreaction. The remaining mismatch is more structural:

- helper evidence is useful for some horizons
- but the current state/observation track is still not aligned enough with the
  helper to turn that into consistent absolute rate gains

## Promotion decision

Do not promote:

- `of_helper_bias_v1`

Keep live default:

- conservative `of_helper_blend_v1`

## Next step

Shift the main redesign axis away from output-only `OF` helper blending and
toward:

- residual-release semantics driven by reliable unexplained observation content
- then, if needed, a stricter `OF` observation reinterpretation rather than more
  output-side heuristics
