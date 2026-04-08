# COHFACE Full Findings — OF Output Rate Strong Relaxed Gate

Date: 2026-04-07

## Scope

- Baseline full run:
  - `results/20260407_cohface_full_obs_cal_v7`
- Experimental full run:
  - `results/20260407_cohface_full_obs_cal_v7_ofrate_v2`

Both runs use the same promoted single-family scaffold:

- `P1D_quad/cub`: harmonic-only `obs_cal_v7`
- `P1D_linear`: unchanged
- `DoF`: unchanged

The only intended difference was the `OF` output-rate policy.

## Change under test

Experimental policy:

- `blend_alpha = 0.65`
- `min_support = 0.66`
- `min_qdyn = 0.35`
- `min_mismatch_hz = 0.03`

This policy passed the 12-trial COHFACE gate subset.

## Full-run verdict

`no-go`

The stronger relaxed-gating OF output-rate policy does **not** survive the
full 160-trial COHFACE rerun.

It leaves waveform unchanged, but it regresses OF T3 relative to the
conservative live scaffold.

## Exact median deltas: `full_ofrate_v2 - full_obs_cal_v7`

For `of_farneback__parh_ossm`:

- rate:
  - `MAE +0.140`
  - `RMSE +0.170`
  - `PearsonR -0.005`
  - `Bias +0.043543`
- waveform:
  - `CCC +0.000000`
  - `MAE +0.000000`
  - `DTW +0.000000`

For other PARH families:

- `P1D_linear`, `P1D_quadratic`, `P1D_cubic`, and `DoF` are unchanged as
  expected

## Interpretation

The subset gate result was real but not stable enough.

Most likely explanation:

- the stronger relaxed gate overfits a subset where helper corrections are
  locally helpful
- across the full dataset, the same relaxed gate activates in too many segments
  where helper evidence is not actually safer than the latent phase track

This means:

- output-side helper integration is still the right design axis for `OF`
- but the current stronger relaxed policy is too permissive

## Promotion decision

Do not promote:

- stronger relaxed OF output-rate blending

Keep as live default:

- conservative `OF` output-rate helper blend

## Next step

The next `OF` redesign should avoid a static stronger gate.

Candidate directions:

- bias-aware helper blending
- trial-level or warm-up-calibrated helper trust
- bounded corrections that only activate when helper and latent tracks stay
  consistent over longer horizons
