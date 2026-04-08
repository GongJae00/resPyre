# COHFACE Observation Family Map Report

Date: 2026-04-08

## Purpose

This note fixes the current live interpretation of the six active COHFACE
observation families under the official reference run:

- `results/20260408_cohface_prod_ofbridge_familyconf_v3/cohface_parh_ossm_prod_ofbridge`

It is the textual companion to:

- `paper/tables_ready/T2_observation_family_map.csv`
- `paper/tables_ready/T3_rate_main.csv`
- `paper/tables_ready/T4_waveform_main.csv`

## Locked interpretation

- `OF` remains a raw velocity-like optical-flow surrogate.
  - It is still helper-heavy and rate-oriented.
  - It improves waveform over raw Base and KFstd, but it is not the best T3 family.
- `OF_bridge` is an OF-derived displacement-compatible constructed family.
  - It is now the strongest rate-oriented constructed family.
  - It should be treated as an additional family, not as a replacement for raw `OF`.
- `P1D_lin` remains a conservative displacement family.
  - It is not the strongest family on either T3 or T4, but it remains a useful
    mid-strength displacement route.
- `P1D_quad` is the current waveform-primary harmonic family.
  - It is the strongest current PARH family on T4.
  - It is also rate-competitive and effectively tied with `OF_bridge` in T3 MAE.
- `P1D_cub` behaves similarly to `P1D_quad` but remains slightly weaker on both
  T3 and T4.
- `DoF` remains nuisance-limited.
  - It improves over its own Base and KFstd routes in T3 and T4.
  - It is still not a main-family candidate for the live paper story.

## Current live numbers

PARH medians from the current official COHFACE table-ready CSVs:

- `OF`: T3 `MAE 0.510`, T4 `CCC 0.791`
- `OF_bridge`: T3 `MAE 0.295`, T4 `CCC 0.777`
- `P1D_lin`: T3 `MAE 0.465`, T4 `CCC 0.727`
- `P1D_quad`: T3 `MAE 0.295`, T4 `CCC 0.855`
- `P1D_cub`: T3 `MAE 0.300`, T4 `CCC 0.849`
- `DoF`: T3 `MAE 2.110`, T4 `CCC 0.571`

## Why this matters

The live paper is no longer just about a shared latent state. It is also about
how the observation family is constructed before that latent state is applied.

The current strongest paper-level statement is therefore:

- observation construction changes performance materially;
- the best rate family and the best waveform family are not the same;
- family-specific observation semantics are part of the model, not just part of
  preprocessing.
