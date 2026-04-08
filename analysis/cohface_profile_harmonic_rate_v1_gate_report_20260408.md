# COHFACE Gate Report: `profile_harmonic_rate_v1`

## Scope

- Dataset: COHFACE 12-trial gate subset
- Baseline profile: `full`
- Experimental profile: `profile_harmonic_rate_v1`
- Run root:
  - `results/20260408_profile_harmonic_rate_gate/full/cohface_parh_ossm_prod_ofbridge`
  - `results/20260408_profile_harmonic_rate_gate/profile_harmonic_rate_v1/cohface_parh_ossm_prod_ofbridge`

## Intent

Test whether a very mild output-only helper-frequency blend for
`profile1d_quadratic` and `profile1d_cubic` can improve T3 without harming T4.

Policy:

- `RESPYRE_PARH_OUTPUT_RATE_POLICY=hybrid_semantics_v1`
- `RESPYRE_PARH_PROFILE_RATE_BLEND_ALPHA=0.18`
- `RESPYRE_PARH_PROFILE_RATE_MIN_SUPPORT=0.95`
- `RESPYRE_PARH_PROFILE_RATE_MAX_QDYN=0.40`
- `RESPYRE_PARH_PROFILE_RATE_MIN_MISMATCH_HZ=0.025`
- `RESPYRE_PARH_PROFILE_RATE_MAX_MISMATCH_HZ=0.10`

## Result

No-go.

The profile-harmonic output-rate blend did not improve the intended families on
the gate subset.

Observed median deltas (`profile_harmonic_rate_v1 - full`):

- `profile1d_quadratic__parh_ossm`
  - rate `MAE +0.010`
  - rate `RMSE +0.000`
  - rate `PearsonR +0.000`
  - waveform `CCC +0.000`
  - waveform `wMAE +0.000`
  - waveform `DTW +0.000`
- `profile1d_cubic__parh_ossm`
  - rate `MAE +0.005`
  - rate `RMSE +0.000`
  - rate `PearsonR +0.000`
  - waveform `CCC +0.000`
  - waveform `wMAE +0.000`
  - waveform `DTW +0.000`
- all other PARH families
  - no measurable change

## Interpretation

For the current live scaffold, the profile harmonic families are already
well-calibrated enough that an output-only helper blend is either inert or
slightly harmful. The next profitable axis remains observation construction,
not post-hoc rate blending.

## Promotion decision

- `profile_harmonic_rate_v1`: rejected
- live reference remains:
  - `results/20260408_cohface_prod_ofbridge_familyconf_v3/cohface_parh_ossm_prod_ofbridge`
