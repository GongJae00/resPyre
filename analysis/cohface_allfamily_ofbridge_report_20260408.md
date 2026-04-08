# COHFACE All-Family OF-Bridge Report

Date: 2026-04-08

## Purpose

This report records the official six-family COHFACE rerun after promoting
`OF_bridge` as an additional observation family.

Reference run:

- `results/20260408_cohface_prod_ofbridge_full/cohface_parh_ossm_prod_ofbridge`

Derived artifacts:

- `paper/tables_ready/T3_rate_main.csv`
- `paper/tables_ready/T4_waveform_main.csv`
- `paper/tables_ready/T6_diagnostics_main.csv`
- `paper/tables_ready/T6b_cohface_mechanism_audit.csv`
- `paper/manifests/cohface_case_study_manifest.csv`

## Headline conclusion

The current COHFACE evidence now supports a six-family single-family ladder:

- raw `OF`
- `OF_bridge`
- `DoF`
- `P1D_linear`
- `P1D_quadratic`
- `P1D_cubic`

`OF_bridge` is now a validated additional observation family. It materially
improves OF-family rate tracking, but it does not replace raw `OF` as the
best OF-family waveform-CCC route.

The strongest overall single-family story remains asymmetric:

- rate: best absolute route is still raw `P1D_quadratic` Base
- waveform: strongest PARH route is still `P1D_quadratic`
- observation construction matters: `OF_bridge` changes the OF-family
  rate/waveform trade-off in a reproducible way

## Official COHFACE T3 summary

Median T3 rate accuracy from `paper/tables_ready/T3_rate_main.csv`:

- `DoF`
  - Base: `MAE 2.400`, `RMSE 4.015`, `r 0.370`
  - KFstd: `MAE 3.110`, `RMSE 3.230`, `r 0.300`
  - PARH: `MAE 2.110`, `RMSE 2.310`, `r 0.390`
- `OF`
  - Base: `MAE 0.290`, `RMSE 0.390`, `r 0.890`
  - KFstd: `MAE 0.550`, `RMSE 0.650`, `r 0.730`
  - PARH: `MAE 0.510`, `RMSE 0.565`, `r 0.740`
- `OF_bridge`
  - Base: `MAE 0.245`, `RMSE 0.325`, `r 0.900`
  - KFstd: `MAE 0.260`, `RMSE 0.335`, `r 0.835`
  - PARH: `MAE 0.295`, `RMSE 0.370`, `r 0.850`
- `P1D_linear`
  - Base: `MAE 0.335`, `RMSE 0.435`, `r 0.840`
  - KFstd: `MAE 0.620`, `RMSE 0.820`, `r 0.655`
  - PARH: `MAE 0.465`, `RMSE 0.590`, `r 0.715`
- `P1D_quadratic`
  - Base: `MAE 0.195`, `RMSE 0.270`, `r 0.950`
  - KFstd: `MAE 0.485`, `RMSE 0.580`, `r 0.800`
  - PARH: `MAE 0.295`, `RMSE 0.375`, `r 0.890`
- `P1D_cubic`
  - Base: `MAE 0.220`, `RMSE 0.275`, `r 0.940`
  - KFstd: `MAE 0.500`, `RMSE 0.605`, `r 0.790`
  - PARH: `MAE 0.310`, `RMSE 0.385`, `r 0.875`

Interpretation:

- PARH beats KFstd on MAE/RMSE in five of the six families.
- `OF_bridge` is the only family where PARH loses the absolute T3 race to both
  Base and KFstd while still improving Pearson correlation over KFstd.
- Base remains the best absolute rate route on `OF`, `OF_bridge`, and all
  `P1D` families.
- `DoF` remains the only family where PARH is the strongest overall T3 route.

## Official COHFACE T4 summary

Median T4 waveform fidelity from `paper/tables_ready/T4_waveform_main.csv`:

- `DoF`
  - Base: `CCC 0.438`, `MAE 0.792`, `DTW 0.551`
  - KFstd: `CCC 0.575`, `MAE 0.731`, `DTW 0.460`
  - PARH: `CCC 0.571`, `MAE 0.736`, `DTW 0.459`
- `OF`
  - Base: `CCC 0.711`, `MAE 0.576`, `DTW 0.452`
  - KFstd: `CCC 0.773`, `MAE 0.516`, `DTW 0.403`
  - PARH: `CCC 0.791`, `MAE 0.511`, `DTW 0.403`
- `OF_bridge`
  - Base: `CCC 0.685`, `MAE 0.629`, `DTW 0.466`
  - KFstd: `CCC 0.789`, `MAE 0.503`, `DTW 0.396`
  - PARH: `CCC 0.777`, `MAE 0.504`, `DTW 0.399`
- `P1D_linear`
  - Base: `CCC 0.578`, `MAE 0.655`, `DTW 0.531`
  - KFstd: `CCC 0.725`, `MAE 0.536`, `DTW 0.424`
  - PARH: `CCC 0.727`, `MAE 0.543`, `DTW 0.424`
- `P1D_quadratic`
  - Base: `CCC 0.811`, `MAE 0.484`, `DTW 0.368`
  - KFstd: `CCC 0.853`, `MAE 0.416`, `DTW 0.343`
  - PARH: `CCC 0.854`, `MAE 0.425`, `DTW 0.346`
- `P1D_cubic`
  - Base: `CCC 0.806`, `MAE 0.490`, `DTW 0.371`
  - KFstd: `CCC 0.851`, `MAE 0.419`, `DTW 0.344`
  - PARH: `CCC 0.848`, `MAE 0.425`, `DTW 0.346`

Interpretation:

- `OF` remains the clearest PARH waveform win over KFstd.
- `P1D_linear` is effectively tied.
- `P1D_quadratic` is the strongest single-family PARH waveform route by CCC,
  but KFstd still keeps slightly lower waveform MAE/DTW.
- `OF_bridge` shows the intended observation-semantics trade-off:
  it improves OF-family rate markedly, but waveform CCC stays slightly below
  raw `OF` PARH and slightly below `OF_bridge` KFstd.

## Calibration and mechanism interpretation

From `paper/tables_ready/T6_diagnostics_main.csv`:

- `OF_bridge` PARH diagnostics:
  - `NIS_Mean 0.841`
  - `NIS_InBand 0.950`
  - `Lambda_Mean 1.015`
  - `Lambda_LT1_Frac 0.000`
  - `Stability_Sec 18.575`

Relative to raw `OF` PARH:

- `NIS_Mean` decreases (`0.937 -> 0.841`)
- `NIS_InBand` increases (`0.929 -> 0.950`)
- `Lambda_LT1_Frac` falls (`0.086 -> 0.000`)
- stability remains shorter than the profile families

From `paper/tables_ready/T6b_cohface_mechanism_audit.csv`:

- raw `OF`
  - `q_dyn_mean_median 0.594`
  - `q_osc_mean_median 0.831`
  - `Qosc_scale_mean_median 1.297`
  - `Qaper_scale_mean_median 1.677`
- `OF_bridge`
  - `q_dyn_mean_median 0.561`
  - `q_osc_mean_median 0.788`
  - `Qosc_scale_mean_median 1.281`
  - `Qaper_scale_mean_median 1.847`

Interpretation:

- `OF_bridge` makes the OF-family observation more internally consistent to
  the filter
- the bridge lowers apparent oscillatory support somewhat and reduces the need
  for strong posterior downweighting
- this helps T3 more than T4, which is exactly what should happen if the
  bridge is a better rate-oriented displacement-compatible observation but not
  yet a perfect waveform surrogate

## Official best-single ladder

Corrected median-based best-single rows from
`paper/tables_ready/T6b_fusion_ladder.csv`:

- T3 best Base: `profile1D quadratic`
- T3 best KFstd: `of_disp_bridge__kfstd`
- T3 best PARH: `of_disp_bridge__parh_ossm`
- T4 best Base: `profile1D quadratic`
- T4 best KFstd: `profile1d_quadratic__kfstd`
- T4 best PARH: `profile1d_quadratic__parh_ossm`

This is the key scientific split:

- rate-optimal PARH family is now `OF_bridge`
- waveform-optimal PARH family is still `P1D_quadratic`

That split reinforces the dual-output and observation-family semantics story.

## Promotion decision

Promote:

- `OF_bridge` as an official additional observation family
- the six-family COHFACE ladder as the current manuscript evidence boundary

Do not promote:

- `OF_bridge` as a replacement for raw `OF`
- fusion as a main result
- MAHNOB quantitative claims before a current-code rerun

## Next research step

The next bottleneck is no longer whether observation construction matters.
That is now demonstrated. The next bottleneck is how to make observation
construction stronger without destroying waveform semantics.

The immediate next tasks are:

1. finalise the six-family COHFACE paper tables and narrative
2. use the corrected best-single ladder in the manuscript
3. push residual identifiability and OF-family observation semantics further
4. only then open the MAHNOB current-code rerun
