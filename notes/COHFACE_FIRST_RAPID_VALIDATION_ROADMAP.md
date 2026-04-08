# COHFACE-First Rapid Validation Roadmap

## Purpose

This file defines the fast iteration loop to use before expensive MAHNOB runs.

COHFACE is the rapid validation environment for:

- observation-model redesign sanity
- preprocessing choices
- calibration stability
- ablation correctness
- saved-payload schema integrity

It is not the final proof environment for irregular respiratory behavior.

## Guiding rule

Every major redesign should first answer:

"Does this make the model more coherent on clean data without obviously harming
rate tracking or creating metric-routing regressions?"

before asking:

"Does this help irregular respiratory behavior on MAHNOB?"

## Fast loop

### Step 1. EDA update

Run or regenerate observation/preprocessing EDA for the targeted family set.

### Step 2. Structural smoke

Run a single-trial debug check and inspect:

- payload fields
- diagnostics arrays
- calibration fields
- decomposition outputs

### Step 3. COHFACE subset sweep

Use a small but fixed subset for quick comparison before the full rerun.

### Step 4. Full COHFACE rerun

Only after the subset behaves sensibly.

### Step 5. Update T3/T4/T6/T6b and case manifest

Treat this as mandatory, not optional.

## COHFACE success criteria

At minimum, a redesign should satisfy:

- no schema breakage
- no evaluation routing regressions
- PARH remains stronger than KFstd on at least the strongest T3 families
- Base does not widen its T3 lead over PARH
- T4 does not collapse on strong motion families
- mechanism audit changes in the direction predicted by the redesign

## First concrete redesign lesson already observed

The first smoke ablation comparing the new `light_obs_path` against the legacy
preprocess showed a non-uniform effect:

- `OF` improved in both waveform and rate
- `DoF` improved strongly in rate but not in waveform
- `P1D` families lost waveform quality under a single global light path

Implication:

- a single universal observation preprocess is unlikely to be optimal
- the next step must be family-aware calibration, not merely "more" or "less"
  preprocessing

## What to inspect after each COHFACE rerun

- `paper/tables_ready/T3_rate_main.csv`
- `paper/tables_ready/T4_waveform_main.csv`
- `paper/tables_ready/T6_diagnostics_main.csv`
- `paper/tables_ready/T6b_cohface_mechanism_audit.csv`
- `paper/manifests/cohface_case_study_manifest.csv`

## Escalation to MAHNOB

Only escalate a redesign to MAHNOB once:

- the design rationale is documented
- COHFACE mechanism changes match expectation
- the model is not obviously harming clean regimes
- the observation-model or residual-semantic change is substantial enough to justify a long rerun
