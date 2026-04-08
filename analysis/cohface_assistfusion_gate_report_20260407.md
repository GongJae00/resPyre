# COHFACE Assistant-Fusion Gate Report

Date: 2026-04-07

## Question

Can `OF` help `P1D_quadratic` in an asymmetric way if it is used only as an
assistant channel for rate support rather than as a peer waveform
observation?

Two assistant policies were tested on a 12-trial COHFACE gate subset:

- `of_rate_assistant_v1`: unconditional helper-rate median blend
- `of_rate_assistant_v2`: gated helper-rate assist, only allowed when the
  primary `P1D_quadratic` route looks locally weak

Config:

- `/home/gongjae/Projects/resPyre/configs/cohface_assistfusion_smoke.json`
- subset driver:
  - `/home/gongjae/Projects/resPyre/scripts/run_parh_gate_subset.py`

Results:

- `results/20260407_assistfusion_gate/full`
- `results/20260407_assistfusion_gate_v2/full`
- `results/20260407_assistfusion_gate_v2/assistant_of_v2`

## Baseline reference

Reference route:

- `profile1d_quadratic__parh_ossm`

Median results:

- rate MAE: `0.310`
- rate RMSE: `0.395`
- rate PearsonR: `0.935`
- z_full CCC: `0.859`
- z_full MAE: `0.433`
- z_full DTW: `0.336`

## Assistant v1

Method:

- `assist_of_p1d_quadratic__parh_ossm`

Median results:

- rate MAE: `0.475`
- rate RMSE: `0.530`
- rate PearsonR: `0.935`
- z_full CCC: `0.859`
- z_full MAE: `0.433`
- z_full DTW: `0.336`

Delta versus baseline:

- rate MAE: `+0.165`
- rate RMSE: `+0.135`
- rate PearsonR: `+0.000`
- z_full CCC: `+0.000`
- z_full MAE: `+0.000`
- z_full DTW: `+0.000`

Verdict:

- `No-go`

Interpretation:

- unconditional `OF` assistance corrupts T3 without helping T4
- this confirms that `OF` should not be blended into the output rate track by
  default when `P1D_quadratic` is already stable

## Assistant v2

Method:

- `assist_of_p1d_quadratic__parh_ossm`
- with `RESPYRE_PARH_ASSISTANT_POLICY=of_rate_assistant_v2`

Median results:

- rate MAE: `0.310`
- rate RMSE: `0.395`
- rate PearsonR: `0.935`
- z_full CCC: `0.859`
- z_full MAE: `0.433`
- z_full DTW: `0.336`

Delta versus baseline:

- rate MAE: `+0.000`
- rate RMSE: `+0.000`
- rate PearsonR: `+0.000`
- z_full CCC: `+0.000`
- z_full MAE: `+0.000`
- z_full DTW: `+0.000`

Verdict:

- `Promotion-safe but inert`

Interpretation:

- a conservative assistant gate can avoid harming the primary route
- however, it still does not extract extra value from `OF` on this COHFACE
  subset
- `OF` as an assistant channel therefore remains exploratory rather than a
  promoted headline design

## Locked conclusion

Current evidence rejects three stronger `OF` fusion uses on COHFACE:

1. peer stacked fusion
2. displacement-bridge peer fusion
3. unconditional assistant output blending

The remaining allowed statement is weaker:

- `OF` can be kept as an optional, conservative assistant path
- but this assistant path is not yet a source of measurable gain on COHFACE

## Next implication

Do not spend more time on `OF + P1D_quadratic` fusion as the main winning path
until a materially stronger observation model or a more irregular dataset
regime justifies it.

Main effort should return to:

- single-family observation modelling
- family-specific observation semantics
- later MAHNOB confirmation once the single-family path is stronger
