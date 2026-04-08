# PARH-OSSM Immediate Work Packages

## Purpose

This file translates the research masterplan into concrete implementation
packages.

These packages are ordered for COHFACE-first rapid iteration and then MAHNOB
confirmation.

## WP0. Current-state lock

Deliverables:

- keep the completed COHFACE rerun as the live reference
- keep MAHNOB blocked until the redesigned COHFACE gate passes
- regenerate T3/T4/T6/T6b and manifests

Purpose:

- prevent redesign decisions from drifting away from current reproducible artifacts

## WP1. Observation EDA pipeline

Deliverables:

- script(s) to export per-trial observation/preprocessing diagnostics
- per-family preprocessing summary CSVs
- stage-by-stage plots for representative trials

Primary files to add:

- `scripts/run_observation_eda.py`
- `scripts/summarize_preproc_eda.py`

Success condition:

- we can state exactly what each family and each preprocessing stage changes

Current state:

- smoke EDA already shows strongly family-specific preprocessing effects
- PARH code now uses an OF-only light observation path plus a stronger helper path
- `DoF` and all `P1D` families remain on the legacy stack
- full COHFACE EDA is now complete and confirms that helper-style observation
  exposure helps some families and harms others
- preprocessing-routing experiments alone are no longer a sufficient redesign
  axis

## WP2. Calibrated single-family observation model

Deliverables:

- warm-up sign/gain/offset calibration
- optional warm-up lag search
- calibrated observation row in PARH
- saved calibration diagnostics

Primary files to change:

- `components/models/heads/parh_ossm.py`
- `components/models/core/base.py`
- config schema for calibration options

Success condition:

- PARH no longer relies on preprocessing alone to absorb sign/scale mismatch
- baseline/residual states are no longer contradicted by the inference preprocess

Current state:

- family-aware bridge observation path is in production
- `freq_rescue` is subset-gated positive only for `OF` and `P1D_linear`
- warm-up observation calibration v1 was subset-gated on COHFACE and failed
- warm-up observation calibration v2 (`global_signed_gain`) was subset-gated
  on COHFACE and stayed neutral-to-slightly-harmful
- warm-up observation calibration v3 (`osc_aux_two_gain`) was subset-gated on
  COHFACE and stayed neutral-to-harmful
- warm-up observation calibration v4 (`family_phase_aux`) produced small
  quad/cub gains but also slight `OF` and `P1D_linear` regressions, so it was
  not promotable as a general policy
- warm-up observation calibration v5 passed the same gate in selective form:
  enabled for `P1D_quad/cub`, disabled for `OF`, `P1D_linear`, and `DoF`
- warm-up observation calibration v7 then improved that path by making the
  `P1D_quad/cub` row harmonic-only rather than auxiliary-aware
- code default now uses selective harmonic-only `obs_cal_v7`
- helper-only routing for quad/cub profile families slightly improved rate but
  slightly harmed waveform and therefore failed promotion
- helper-only routing for all `P1D` families clearly failed because
  `P1D_linear` collapsed
- helper-only or helper-blend routing for `DoF` produced mixed trade-offs and
  also failed promotion
- next iteration should move to bounded family-specific observation-row
  semantics rather than more preprocessing-policy switching

## WP3. Residual semantics redesign

Deliverables:

- unexplained-observation-driven residual release
- saved residual-release cause diagnostics
- residual-specific ablation rows

Success condition:

- residual becomes empirically interpretable rather than a flexible leftover

## WP4. COHFACE-first benchmark loop

Deliverables:

- subset benchmark
- full COHFACE rerun
- updated T3/T4/T6/T6b
- updated overlay manifest

Success condition:

- redesign improves coherence and narrows the Base gap without harming schema stability

Current gate lock:

- selective `freq_rescue` now passes the 12-trial COHFACE gate subset
- OF-only light observation path also passes after rolling `DoF` back to the
  legacy stack
- selective harmonic-only `obs_cal_v7` now passes the same gate and is the
  promoted default for `P1D_quad/cub`
- a full COHFACE rerun under the current scaffold has completed and is now the
  live reference
- a stronger relaxed `OF` output-rate blend passed the 12-trial gate subset
  but failed the full 160-trial COHFACE rerun and therefore remains
  experimental only
- a newer bias-aware `OF` output-rate policy then improved subset `RMSE`,
  `PearsonR`, and bias while leaving T4 unchanged, but it still worsened subset
  `MAE` and therefore also remains experimental only
- `OF` helper-trust redesign then also failed promotion:
  - direct helper-trust-driven `q_dyn` suppression harmed `OF` rate
  - rescue-only helper trust was safe but effectively inert
  - rescue-only helper trust plus relaxed rescue improved some `OF` T3 numbers
    but harmed T4 and therefore also failed promotion
- residual-release `v1` also failed promotion because clean observation support
  mostly pushed `q_osc` upward and closed the residual branch more tightly
- residual-release `v2` fixed that direction and lowered `q_osc` for `OF` and
  `P1D_linear`, but T3/T4 changes were effectively zero
- residual-release `v3` then added direct `Q_aper` bonus from unexplained clean
  observation need, but it slightly worsened `OF`/`DoF` without helping the
  main profile families
- residual-release `v4` now uses an observation-driven non-oscillatory gap
  between oscillator-only support and full-state support
  - this is more interpretable than the earlier heuristics
  - but on the clean COHFACE gate it remains effectively inert
  - therefore it is kept as a diagnostic redesign step, not a promoted live policy
- additional helper-only and helper-blend observation-routing variants have now
  failed the same gate and should remain experimental only
- the next expensive run is not MAHNOB yet; it is an observation-semantics
  step that can improve `OF` rate and waveform coherence without leaning on
  helper-trust heuristics or subset-only rescue behavior
- that next observation-semantics step is now concretely defined:
  single-family `of_disp_bridge`
- `of_disp_bridge` produced a strongly positive 12-trial COHFACE gate result
  for `KFstd` and PARH and is now under full COHFACE validation
- that full validation is now completed and supports promotion of
  `OF_bridge` as an additional observation family
- the next expensive run is therefore an all-family COHFACE rerun that adds
  `OF_bridge` to the single-family comparison ladder
- a corrected raw-`OF` fixed velocity prior fallback was then gate-tested:
  - the first run was invalid because the fallback was accidentally blocked by
    the warm-up calibration allowlist
  - the corrected rerun verified that raw `OF` really switched to a velocity
    fallback row
  - it slightly improved `RMSE` / `PearsonR` but worsened primary rate `MAE`
    and strongly degraded waveform fidelity
  - therefore it remains rejected
- MAHNOB remains blocked until that gate produces a promotable observation step
- residual semantics is still unfinished; the next redesign must make residual
  content more identifiable, not only easier to release
- a family-aware residual prior (`residual_semantics_v1`) was then rerun
  cleanly after an earlier stale gate artifact:
  - it changed the intended diagnostics (`obs_nonosc_need_eff`,
    `residual_prior`, `aper_drive`)
  - but on clean COHFACE it was effectively inert for the main profile
    families, slightly harmful for `DoF`, and only marginally helpful for
    `P1D_linear`
  - it therefore remains a diagnostic branch, not a live policy
- raw `OF` plus `OF_bridge` as an output-only assistant path was also gate
  tested and failed:
  - waveform was unchanged
  - rate became slightly worse than raw `OF PARH`
  - therefore this is not yet a viable dual-track OF family
- `P1D_quad` plus `OF_bridge` as an output-only assistant path was then gate
  tested and also failed:
  - waveform stayed identical
  - rate became slightly worse than `P1D_quad PARH`
  - therefore the current assistant gate is not yet a meaningful fusion path
- current COHFACE evidence therefore suggests that easy output-side assistant
  heuristics are exhausted; the next meaningful step is no longer another small
  COHFACE helper tweak but either:
  - a more structural multi-observation design, or
  - an irregular-regime subset check on MAHNOB

## WP5. Multi-family fusion prototype

Deliverables:

- stacked observation prototype
- family-specific `R_t`
- missing-family handling

Success condition:

- waveform gap narrows more than with single-family redesign alone

## WP6. MAHNOB confirmation

Deliverables:

- full MAHNOB rerun under redesigned model
- irregular-regime analysis
- T7 regime table

Success condition:

- gains or trade-offs on irregular data are interpretable and consistent with the model story

## WP7. Manuscript rewrite

Deliverables:

- rewritten abstract
- rewritten generative framing section
- rewritten model section around calibrated observation modelling
- rewritten results/discussion with COHFACE vs MAHNOB regime logic
- updated figure/table plan and captions

Success condition:

- the paper explains why the model should work, not only that some metrics moved
