# Observation-Family and Preprocessing EDA Protocol

## Purpose

This document defines the mandatory exploratory analysis protocol for the next
phase of PARH-OSSM development.

The objective is not "look at a few plots."

The objective is to explain, with evidence, how each observation family and
each preprocessing stage changes the relationship between:

- latent respiratory motion
- camera-derived proxy signal
- belt reference waveform
- rate estimation metrics

This is the gatekeeper analysis before a serious observation-model redesign.

## Why this EDA is mandatory

The current repository already models latent oscillatory structure in detail.
What remains under-modelled is the observation channel.

Without observation-focused EDA, the project risks:

- adding latent flexibility without fixing the real bottleneck
- mislabelling observation distortion as physiology
- overfitting to COHFACE or MAHNOB quirks
- writing a paper that talks about physiology while the actual gains come from preprocessing heuristics

## Observation families to analyse

Treat these as distinct measurement families, not interchangeable inputs.

### OF-Farnebäck

What it likely measures:

- projected vertical respiratory velocity / displacement surrogate

Expected strengths:

- strong oscillatory support
- good rate fidelity
- better robustness than simpler motion proxies

Expected distortions:

- sign ambiguity
- gain mismatch
- ROI drift sensitivity
- low-frequency projection drift

### DoF

What it likely measures:

- thresholded motion energy

Expected strengths:

- simplicity
- sensitivity to large motion

Expected distortions:

- coarse amplitude semantics
- threshold saturation
- stronger noise sensitivity
- weak waveform fidelity

### Profile1D linear / quadratic / cubic

What they likely measure:

- frame-to-frame vertical profile displacement surrogate

Expected strengths:

- good sensitivity to breathing-related chest shifts
- potentially strong waveform tracking in clean clips

Expected distortions:

- interpolation-induced bias
- lag/misalignment
- scale instability
- family-specific harmonic distortion

## Preprocessing stages to analyse separately

The analysis must not jump directly from raw signal to final metric.

Each stage must be measured.

### Stage P0. Raw family signal

Signal exactly as produced by the observation extractor.

### Stage P1. Detrending only

Test whether simple linear detrending removes the dominant drift component.

### Stage P2. Bandpass only

Current physiological bandpass without robust z-score.

### Stage P3. Sign alignment only

Warm-up sign correction without other normalization.

### Stage P4. Robust z-score only

Median/MAD scaling and clipping without bandpass.

### Stage P5. Current inference preprocessing

Current repository `_preprocess` output.

### Stage P6. Current helper preprocessing

Current repository `_helper_preprocess` output.

### Stage P7. Candidate redesigned preprocessing variants

At minimum:

- no clipping
- lower clipping
- no bandpass
- causal filter substitute
- family-specific bandpass variants

## Core EDA questions

The analysis must answer all of these questions.

### Q1. What physical quantity is each family closest to?

Test whether each family behaves more like:

- displacement
- velocity
- motion energy
- envelope proxy

Required analyses:

- correlation with GT waveform
- correlation with GT derivative
- lag scan
- harmonic content comparison

### Q2. What does preprocessing actually fix?

For each stage, measure whether it improves or worsens:

- sign consistency
- lag consistency
- harmonic fidelity
- rate stability
- waveform alignment

Current early evidence already supports the need for this question.
On the initial COHFACE smoke subset:

- `OF` and `DoF` benefited strongly from aggressive band-limited shaping
- `P1D` families could be harmed by bandpass-only processing while still
  benefiting from the full current preprocessing stack
- a globally lighter observation path helped `OF` and `DoF` rate behavior but
  hurt `P1D` waveform fidelity

This means preprocessing is not a uniform nuisance-removal step. It is part of
the observation semantics and must be analysed family by family.

### Q3. Where does each family fail?

Characterise failure modes:

- weak amplitude
- drift
- sign flips
- clipping/saturation
- noisy instability
- motion bursts
- family-specific aliasing

### Q4. What part of the gap is observation-side versus latent-side?

This is the most important question.

We need evidence for whether poor waveform fidelity is mainly driven by:

- wrong observation semantics
- missing calibration
- missing nuisance terms
- insufficient latent flexibility

### Q5. Can Base beat model-based methods mainly because the observation family is already strong?

This must be tested directly.

If Base wins on some families, determine whether it wins because:

- the family already produces an excellent oscillatory proxy
- state-space smoothing damages clean measurements
- rate-only evaluation rewards spectral simplicity
- waveform alignment is dominated by preprocessing rather than modelling

## Trial-level measurements to compute

These measurements must be computed for every:

- dataset
- family
- trial
- preprocessing stage

### Observation-vs-GT measurements

- best lag to GT waveform
- best lag to GT derivative
- CCC to GT waveform after allowed alignment
- Pearson correlation to GT waveform
- Pearson correlation to GT derivative
- scale ratio to GT
- offset drift magnitude
- harmonic ratio mismatch
- spectral peak mismatch
- phase-lock summary

### Cross-family measurements

- family-to-family correlation
- family-to-family lag
- family-to-family spectral agreement
- cross-family sign agreement
- cross-family consensus confidence

### Signal-quality measurements

- variance
- robust scale
- clipped fraction
- spectral peak sharpness
- harmonic concentration
- low-frequency drift energy
- high-frequency contamination
- stationarity score

## Required summary outputs

### Dataset-level CSVs

- per-trial observation EDA
- per-family preprocessing summary
- cross-family consensus summary

### Figures

- family raw/preprocessed example strips
- stage-by-stage spectrum comparison
- lag/scale distributions per family
- derivative-vs-waveform alignment panel
- family consensus heatmaps
- preprocessing gain/loss plots

### Tables

- family distortion taxonomy
- preprocessing benefit table
- family suitability map for `z_osc` and `z_full`

## COHFACE-first rapid iteration protocol

Because MAHNOB is slow, COHFACE should be used first for fast iteration.

The purpose of COHFACE is not to prove final superiority.

The purpose is to quickly validate:

- whether a redesign is internally coherent
- whether a new observation-calibration idea is stable
- whether Base still dominates easy cases
- whether we accidentally harm clean oscillatory tracking

### COHFACE-only fast gates

A redesign should pass these before MAHNOB:

1. no metric-routing regression
2. no failure in saved payload schema
3. no drop in PARH-vs-KFstd T3 on most families
4. no catastrophic T4 collapse
5. interpretable change in mechanism audit

## MAHNOB confirmation role

MAHNOB is the confirmation stage for:

- irregular respiratory content
- residual semantics
- observation corruption handling
- family disagreement handling

No redesign should be declared successful without MAHNOB, but no redesign
should wait for MAHNOB before basic internal logic is tested on COHFACE.

## Link to the next model design

This EDA directly informs:

- sign/gain/offset/lag calibration
- family-specific observation rows
- fusion design
- nuisance design
- residual release logic

If a planned model component is not motivated by this EDA, it should be treated
as speculative.
