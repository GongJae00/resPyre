# Observation-Model Redesign Specification

## Purpose

This document defines the next serious modelling step for PARH-OSSM.

The goal is to strengthen the observation model while preserving the OSSM
identity of the project.

Current gate status:

- warm-up observation calibration v1 was tested on a 12-trial COHFACE gate
  subset and failed the promotion gate
- warm-up observation calibration v2 (`global_signed_gain`) was also tested on
  a 12-trial COHFACE gate subset and was neutral-to-slightly-harmful relative
  to the current scaffold
- warm-up observation calibration v3 (`osc_aux_two_gain`) was also tested on a
  12-trial COHFACE gate subset and remained neutral-to-harmful
- warm-up observation calibration v4 (`family_phase_aux`) was the first mixed
  result: small gains on `P1D_quad/cub`, but slight regressions on `OF` and
  `P1D_linear`
- warm-up observation calibration v5, a selective `family_phase_aux` policy,
  passed the same gate and established selective family-row calibration as a
  valid direction
- warm-up observation calibration v7 then improved that selective policy:
  a harmonic-only row for `P1D_quad/cub` improved both T3 absolute error and
  T4 waveform fidelity on the same gate subset
- frequency rescue was subset-gated and passed only in family-selective form:
  enabled for `OF` and `P1D_linear`, disabled for `P1D_quad`, `P1D_cub`,
  and `DoF`
- `OF` output-rate refinement was then retested under stronger output-only
  helper blending:
  - the original `v1` policy became effectively inert under the newer
    scaffold
  - a stronger, more relaxed gate improved `OF` T3 on the 12-trial gate
    subset without changing T4
  - but the same stronger policy regressed `OF` T3 on the full 160-trial
    COHFACE rerun and therefore failed promotion
- a newer bias-aware output-only `OF` policy was then subset-gated:
  - it kept T4 unchanged
  - it improved `RMSE`, `PearsonR`, and reduced median bias on the 12-trial
    gate subset
  - but it still worsened the primary `MAE` metric and therefore also failed
    promotion
- the current live production path is therefore:
  - `OF`: light observation path
    plus conservative output-only helper-rate blending
  - `DoF`: legacy stack
  - `P1D_linear`: legacy stack
  - `P1D_quad/cub`: legacy stack plus selective harmonic-only warm-up
    observation-row calibration
- helper-only observation-routing variants for `P1D_quad` and `P1D_cub` showed
  mild rate gains but slight waveform regressions and therefore failed
  promotion
- helper-only routing for all `P1D` families clearly failed because
  `P1D_linear` regressed strongly
- helper-only or helper-blend routing for `DoF` produced mixed waveform-rate
  trade-offs and also failed promotion
- preprocessing-policy switching is therefore no longer the main redesign axis;
  the next promotable step must change the observation row itself
- output-only `OF` helper policies are also no longer the main redesign axis;
  the next promotable step should instead improve residual-release semantics or
  move `OF` observation modelling itself
- `OF` helper-trust heuristics have now also been tested:
  - direct helper-trust-driven `q_dyn` suppression is rejected
  - rescue-only helper trust is safe but inert
  - rescue-only helper trust plus relaxed `OF` rescue improves some T3 numbers
    but harms T4 and therefore also fails promotion
  - helper trust may remain as a diagnostic signal, but it is not the next
    promotable design axis
- a new single-family `OF` displacement bridge has now produced the first
  clearly positive OF-only gate result:
  - it worsened the raw base waveform proxy slightly
  - but it substantially improved both T3 and T4 for `KFstd` and PARH on the
    12-trial COHFACE gate subset
  - this makes OF observation semantics, not helper heuristics, the current
    most credible OF redesign axis
  - full-dataset validation is now completed:
    - T3 improvement survives strongly on the full dataset
    - T4 is mixed, with slight `wMAE` / `DTW` gains but lower waveform `CCC`
  - the correct promotion is therefore as an additional family, not as a
      raw `OF` replacement
- a corrected raw-`OF` fixed velocity prior fallback has now also been tested:
  - the first attempt was invalid because a wiring bug let the fallback get
    blocked by the warm-up calibration allowlist
  - after fixing that bug, the fallback really switched raw `OF` from
    `fixed_sum_bridge_v1` to `fixed_of_velocity_prior_v1`
  - it slightly improved `RMSE` and `PearsonR` on the 12-trial gate subset
    but worsened primary rate `MAE` and strongly degraded waveform fidelity
  - it therefore also fails promotion as a live raw-`OF` fallback

This is the critical design shift away from:

- treating camera proxies as direct waveform surrogates
- relying on preprocessing to absorb most observation mismatch
- adding more latent flexibility before the measurement channel is explained

## Design target

The target model should explain this chain:

1. hidden respiratory motion
2. projection into visible thoracic motion
3. extractor-family-specific scalar proxy
4. calibrated observation equation
5. uncertainty-aware state estimation

## Keep versus change

### Keep

- OSSM structure
- conditional exact linearity
- `K=2` harmonic morphology
- baseline state
- residual state
- dual outputs `z_osc` and `z_full`
- disentangled uncertainty adaptation

### Change

- observation operator
- warm-up calibration
- fusion logic
- nuisance handling
- residual release semantics

Current residual-release truth:

- direct clean-support blending into `q_osc` is wrong for clean COHFACE because
  it pushes `q_osc` upward
- one-sided unexplained-observation penalty moves `q_osc` in the right
  direction but is too weak to change T3/T4 materially
- direct `Q_aper` bonus on top of that also failed promotion on the gate subset

Therefore residual redesign remains open, but the next step must improve
residual identifiability rather than add more release heuristics alone.

The latest clean COHFACE rerun confirmed that a family-aware residual prior can
make this branch more interpretable without making it more useful:

- `residual_semantics_v1` reduced effective non-oscillatory drive exactly as
  intended by family semantics
- but T3/T4 stayed unchanged for `P1D_quad/cub`, shifted only slightly for
  `P1D_linear`, and worsened `DoF`
- this branch therefore remains diagnostic-only

Current OF observation truth:

- a direct OF velocity-domain observation row can be written down and calibrated
  in warm-up
- but current warm-up fits are too weak to support promotion
- a fit-quality gate is now required before any family-specific observation row
  is allowed to replace the fixed live scaffold

For OF specifically, the design lock is now:

- helper-path and oscillatory evidence remain the primary role
- direct observation-row replacement is experimental only
- a single-family OF-to-displacement bridge is now the first OF observation
  redesign promoted from subset gate to full COHFACE validation
- after full validation, it is promoted as a new OF-derived observation family
  rather than a replacement for raw `OF`
- if full validation confirms the gate result, the OF family should be
  treated as two validated constructions with different strengths:
  - raw `OF`: helper-heavy motion waveform proxy
  - `OF_bridge`: rate-oriented displacement-compatible proxy
- two output-side assistant variants were then tested and both failed:
  - raw `OF` with `OF_bridge` as a rate assistant was effectively inert and
    slightly worse in T3
  - `P1D_quad` with `OF_bridge` as a rate assistant was also inert and slightly
    worse in T3
- therefore the current assistant gate is not the right mechanism for final
  multi-family integration
  reinterpreted in two tracks:
  - raw OF as a velocity-like helper-heavy proxy
  - bridged OF as a displacement-compatible single-family observation route
- a fixed velocity prior fallback for raw `OF` is now explicitly rejected as a
  live policy because it harms waveform fidelity too strongly on the corrected
  gate rerun
- the current official COHFACE live scaffold is now:
  - six-family ladder with `OF_bridge` included
  - plus a narrow `family_confidence_v3` refinement for
    `P1D_quadratic` / `P1D_cubic`
  - that refinement only activates when warm-up calibration already indicates
    an excellent displacement fit
  - it is a low-risk observation-confidence refinement, not a new observation
    row family

## Current observation limitation

The current implementation effectively treats the observation as a single
preprocessed scalar aligned to a fixed sum of latent components.

This is too weak because it does not explicitly represent:

- sign ambiguity
- family-dependent gain
- family-dependent offset
- mild lag mismatch
- family-specific sensitivity to oscillatory vs non-oscillatory components

There was also a structural inconsistency in the earlier scaffold:

- the inference path used `detrend + bandpass`
- the latent model simultaneously claimed to estimate baseline and residual terms

This meant the observation pipeline was removing part of the structure that the
latent state was supposed to explain. The redesign must remove this mismatch.

## Stage-1 redesigned observation model

For a single family `m`, use:

`y_t^(m) = c_m + s_m [g_h1^(m) h1_t + g_h2^(m) h2_t + g_b^(m) b_t + g_r^(m) r_t] + v_t^(m)`

with:

- `s_m ∈ {-1, +1}` fixed after warm-up
- `c_m` fixed after warm-up
- `g_h1^(m), g_h2^(m), g_b^(m), g_r^(m)` fixed after warm-up
- `v_t^(m) ~ N(0, R_t^(m))`

Equivalent matrix form:

`y_t^(m) = c_m + H_m(theta_cal) x_t + v_t^(m)`

This preserves linearity once `theta_cal` is estimated.

Current bridge step already implemented:

- `OF`: family-aware light observation path
- `DoF`: legacy/current preprocess retained
- `P1D` families: legacy/current preprocess retained

This is only a transition step justified by COHFACE EDA and smoke ablations.
It is not a substitute for explicit warm-up calibration.

Additional routed-observation experiments have now also been tested and
rejected at the 12-trial COHFACE gate:

- helper-only routing for quad/cub profile families
- helper-only routing for quad/cub plus `DoF`
- helper-only routing for all `P1D` families
- helper/legacy blend routing for quad/cub profile families
- helper/legacy blend routing for quad/cub plus `DoF`

Those results matter because they show that observation mismatch is not solved
by exposing a different preprocessed scalar alone.

The first three regression-style warm-up calibration attempts should be treated
as failed prototypes rather than locked designs. The promoted design is a
bounded selective observation row.

Why `DoF` is not on the light path:

- the subset gate showed that the lighter observation path improves `OF`
  waveform and rate behavior
- the same gate showed that `DoF` regresses under the lighter path and should
  remain on the legacy stack until the observation row is explicitly calibrated

## Why component-wise gains matter

The gains should not be a single global scalar if avoidable.

Reason:

- oscillatory blocks may be strongly visible in some families
- baseline drift may project differently than oscillatory motion
- residual events may be over- or under-expressed depending on the extractor

The gate evidence now supports a stronger constraint: calibration freedom must
be selective. Broad calibration across all families regresses already-strong
families. Bounded calibration on the families that still show an observation
gap is useful.

## Stage-1 warm-up calibration

Status:

- prototype v1 implemented, gate-tested, and rejected
- constrained v2 global signed-gain calibration implemented, gate-tested, and
  rejected
- constrained v3 oscillator-versus-auxiliary two-gain calibration
  implemented, gate-tested, and rejected
- bounded family-phase auxiliary v4 implemented, gate-tested, and kept
  experimental because it was mixed
- bounded selective family-phase auxiliary v5 implemented, gate-tested, and
  promoted for `P1D_quad/cub`

Warm-up should estimate:

- sign
- offset
- scale
- optional lag
- family reliability prior

### Sign estimation

Candidate rule:

- compare warm-up signal to helper-path oscillatory template
- choose sign that maximises phase consistency / spectral agreement

### Offset estimation

Candidate rule:

- robust warm-up median after detrending choice is fixed

### Gain estimation

Candidate rule:

- a single global signed gain on `z_full` has now been tested and found too
  weak to justify promotion
- a static oscillatory-vs-auxiliary two-gain split on `[z_osc, b+r]` has also
  been tested and found too weak to justify promotion
- helper-only and helper-blend routing variants have also been tested and found
  insufficient for promotion
- broad family-phase auxiliary calibration was also tested and found mixed
- the promoted design uses bounded family-specific visibility weights over
  harmonic and auxiliary latent content, but only for the two higher-order
  profile families where gate evidence is positive
- only re-introduce stronger baseline/residual calibration if subset-level
  evidence supports it
- avoid unconstrained regression onto all latent components in early versions

### Lag estimation

Candidate rule:

- search a small bounded lag during warm-up only
- choose lag maximising correlation or coherence with the helper template
- current gate evidence shows that lag is only occasionally used; the main
  useful degree of freedom is harmonic visibility plus bounded auxiliary gain

### Reliability prior

Candidate rule:

- derive from spectral concentration, clipping fraction, and cross-family agreement

## Stage-2 multi-family fusion

Once single-family calibration is stable, stack multiple families:

`y_t = c + H(theta_cal) x_t + v_t`

where:

- `y_t` is the vector of all available family observations
- `H(theta_cal)` contains one row per family
- `R_t` is diagonal or block-diagonal with family-specific adaptive entries

## Benefits of fusion

- better shared oscillatory support
- better warm-up calibration through consensus
- robustness when one family fails
- explicit modelling of family disagreement

## Stage-3 nuisance extension

Add nuisance observation terms only after calibration is stable.

Possible form:

`y_t^(m) = c_m + H_m x_t + G_m n_t + v_t^(m)`

where `n_t` is a low-dimensional nuisance state or nuisance regressor.

Good nuisance candidates:

- ROI quality
- low-frequency global motion contamination
- extractor instability indicator

Bad nuisance candidates:

- any term that simply duplicates the residual block
- any term whose meaning cannot be diagnosed in saved outputs

## Residual-release redesign

Residual release should depend less on self-consistency of the oscillator and
more on unexplained observation structure.

Target ingredients:

- oscillatory support
- helper ambiguity
- structured observation residual
- cross-family disagreement

The residual should open when the signal still looks respiratory but the
oscillator alone no longer explains it.

## Relation to preprocessing

Preprocessing should support the observation model, not replace it.

This means:

- sign alignment in preprocessing should eventually become a calibration output
- family-specific scaling should not be buried in z-scoring alone
- helper preprocessing should be explicitly labelled as helper evidence, not as the final observation

Immediate implication for the code:

- helper path can remain strongly band-limited
- inference path should be light and preserve low-frequency structure
- the observation model must absorb sign/gain/offset semantics instead of
  outsourcing them entirely to preprocessing

## What success looks like

A successful redesign should show all of these:

1. Base is no longer trivially strongest on COHFACE T3 in most families.
2. PARH waveform fidelity closes the gap to KFstd on COHFACE.
3. Mechanism audit shows interpretable shifts in baseline/residual usage.
4. Residual semantics become more valuable on MAHNOB than on COHFACE.
5. The manuscript can explain why gains appear, not only that they appear.

## Minimum implementation plan

### Package 1

Single-family calibrated observation rows with:

- sign
- offset
- bounded family-specific harmonic visibility
- bounded auxiliary gain on `b+r`
- optional bounded lag selection

### Package 2

Warm-up reliability prior and family-specific nuisance checks.

### Package 3

Updated diagnostics arrays:

- calibrated sign
- calibrated gains
- calibrated lag
- residual-release cause summary

### Package 4

Fusion prototype over the existing five families.

## Immediate redesign lock

The current evidence supports these concrete decisions:

1. keep `bridge_v1` as the observation-path scaffold
2. keep selective `freq_rescue` for `OF` and `P1D_linear`
3. keep selective `obs_cal_v5` for `P1D_quad/cub`
4. stop promoting broad observation-calibration policies unless they improve
   the gated families without regressing `OF` or `P1D_linear`
5. treat helper preprocessing as evidence for calibration and support
   estimation, not as a sufficient replacement for the live observation row
6. block MAHNOB until a full COHFACE rerun confirms the promoted default

The next candidate should therefore look more like:

`y_t^(m) = c_m + s_m [g_{1}^{(m)} h_{1,t} + g_{2}^{(m)} h_{2,t} + g_{\mathrm{aux}}^{(m)} (b_t + r_t)] + v_t^(m)`

with optional bounded warm-up lag selection and bounded visibility weights,
rather than another global preprocessing switch.

## Hard warning

Do not add a learned black-box head before Package 1 works.

If the observation model is still weak, adding more latent or learned capacity
will make the system less interpretable while leaving the paper scientifically
fragile.
