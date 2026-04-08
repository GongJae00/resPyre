# COHFACE OF-Bridge Gate Report

Date: 2026-04-07

## Purpose

This note records the first direct gate comparison between the canonical raw
`OF-Farneback` family and the new single-family `of_disp_bridge` family on the
12-trial COHFACE gate subset.

The question is narrow:

- if raw `OF` is a velocity-like proxy and therefore mismatched to the shared
  displacement-oriented PARH latent state,
- does a simple OF-to-displacement bridge improve the single-family OF route
  enough to justify a full COHFACE validation run?

## Compared methods

Gate config:

- `configs/cohface_of_bridge_gate.json`

Methods:

- `of_farneback`
- `of_disp_bridge`
- `of_farneback__kfstd`
- `of_disp_bridge__kfstd`
- `of_farneback__parh_ossm`
- `of_disp_bridge__parh_ossm`

Result root:

- `results/20260407_of_bridge_gate_subset/full/cohface_of_bridge_gate`

## Main gate numbers

All values below are medians over the 12-trial gate subset.

### Base

- raw `OF`: `CCC 0.731`, `wMAE 0.577`, `DTW 0.407`, `rate MAE 0.625`, `RMSE 1.160`, `r 0.765`
- `OF bridge`: `CCC 0.718`, `wMAE 0.577`, `DTW 0.445`, `rate MAE 0.350`, `RMSE 0.635`, `r 0.905`

Interpretation:

- the displacement bridge is already useful for raw rate estimation
- but without filtering it does not improve waveform fidelity

### KFstd

- raw `OF`: `CCC 0.771`, `wMAE 0.499`, `DTW 0.401`, `rate MAE 0.560`, `RMSE 0.645`, `r 0.635`
- `OF bridge`: `CCC 0.854`, `wMAE 0.379`, `DTW 0.343`, `rate MAE 0.225`, `RMSE 0.325`, `r 0.875`

Interpretation:

- the bridge strongly improves both T3 and T4 for the standard KF route

### PARH

- raw `OF`: `CCC 0.781`, `wMAE 0.528`, `DTW 0.383`, `rate MAE 0.670`, `RMSE 0.840`, `r 0.720`
- `OF bridge`: `CCC 0.841`, `wMAE 0.397`, `DTW 0.343`, `rate MAE 0.320`, `RMSE 0.365`, `r 0.910`

Interpretation:

- the bridge improves both waveform and rate metrics substantially for PARH
- this is the first OF-side redesign that looks like a promotable observation
  semantics step rather than a heuristic tweak

## Calibration / diagnostics

PARH diagnostics on the same subset:

- raw `OF`: `NIS_Mean 0.908`, `Lambda_Mean 1.020`, `strict_pass 0.417`
- `OF bridge`: `NIS_Mean 0.829`, `Lambda_Mean 1.015`, `strict_pass 0.167`

Interpretation:

- the bridge route is not failing catastrophically
- but it is also more over-strict under the current calibration split
- full-dataset validation is required before any promotion because the bridge
  may be helping task metrics at the cost of calibration conservatism

## Decision

Decision:

- subset result is `promising`
- not yet promoted
- full 160-trial COHFACE validation is required next

Why this is different from earlier OF redesign attempts:

- earlier OF work mostly changed helper trust, helper rescue, or output-rate
  postprocessing
- those routes improved some subset T3 numbers but usually harmed T4 or failed
  on the full dataset
- the OF displacement bridge instead changes the observation semantics itself

This makes it a materially stronger candidate for promotion.

## Promotion criteria for the full COHFACE run

The full COHFACE bridge route should only be promoted if:

- `of_disp_bridge__parh_ossm` improves OF T3 absolute error relative to raw
  `of_farneback__parh_ossm`
- `of_disp_bridge__parh_ossm` improves OF T4 waveform fidelity relative to raw
  `of_farneback__parh_ossm`
- the bridge does not create obvious diagnostic collapse or pathological
  instability

If those conditions fail, the bridge must remain experimental.
