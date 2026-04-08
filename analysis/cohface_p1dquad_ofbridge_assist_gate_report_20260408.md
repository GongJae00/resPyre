# COHFACE P1D_quad plus OF-bridge assistant gate

Date: 2026-04-08

Run:
- `/home/gongjae/Projects/resPyre/results/20260408_p1dquad_ofbridge_assist_gate/cohface_p1dquad_ofbridge_assist_gate`

## Goal

Keep `P1D_quad` as the waveform-primary family while letting `OF_bridge`
provide output-only rate assistance.

## Compared methods

- `profile1d_quadratic__parh_ossm`
- `assist_ofbridge_p1d_quadratic__parh_ossm`
- `of_disp_bridge__parh_ossm`

## Result

`assist_ofbridge_p1d_quadratic__parh_ossm` was effectively identical in T4 and
slightly worse in T3 than `profile1d_quadratic__parh_ossm`.

### P1D_quad PARH -> P1D_quad + OF_bridge assistant PARH

- waveform:
  - `CCC 0.872 -> 0.872`
  - `MAE 0.417 -> 0.417`
  - `DTW 0.331 -> 0.331`
- rate:
  - `MAE 0.275 -> 0.285`
  - `RMSE 0.360 -> 0.365`
  - `PearsonR 0.925 -> 0.925`

## Interpretation

This assistant route also remained effectively inert. The current assistant
policy is not yet a useful mechanism for merging `OF_bridge` rate evidence
into the strongest waveform family.

## Verdict

`no-go`.

The strongest current story remains:
- `OF_bridge` as a rate-oriented observation family
- `P1D_quad` as the strongest waveform family

If these are to be combined later, it will require a stronger multi-observation
design than the current assistant gate.
