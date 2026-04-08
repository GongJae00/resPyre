# PARH-OSSM Reference Backbone

## Purpose

This file maps the core claims of the PARH-OSSM program to the reference
clusters that must support them.

It is not a bibliography dump. It is a defence map.

Each cluster answers one reviewer question:

- why this modelling choice?
- why this decomposition?
- why this evaluation?
- why these limitations?

## Cluster A. State-space modelling principle

Question:

- Why do observation definition and state definition matter so much here?

Core references:

- Harvey, `Forecasting, Structural Time Series Models and the Kalman Filter`
- Durbin and Koopman, `Time Series Analysis by State Space Methods`
- Särkkä, `Bayesian Filtering and Smoothing`

Paper role:

- justify that a state-space model is only meaningful when the latent state and
  observation equations correspond to the structure generating the data
- support the claim that observation modelling is not a cosmetic detail

## Cluster B. Oscillatory / resonator state-space modelling

Question:

- Why use an OSSM-style resonator model at all?

Core references:

- Särkkä et al., `DRIFTER`
- Solin and Särkkä, periodic covariance / state-space link
- Beck et al., state-space oscillator models

Paper role:

- justify resonator state-space modelling for narrow-band or quasi-periodic
  signals
- support keeping exact linear inference conditional on frequency guidance

## Cluster C. Respiratory variability and morphology

Question:

- Why is respiration not one clean sinusoid?

Core references:

- Oku, `Temporal variations in the pattern of breathing`
- van den Bosch et al., `Breathing variability—implications for anaesthesiology and intensive care`

Paper role:

- justify oscillatory plus aperiodic thinking
- justify that breath-to-breath variability carries physiological meaning
- justify why a residual path is not merely "noise"

## Cluster D. Contactless respiration observation gap

Question:

- Why is camera observation modelling a central issue rather than a nuisance footnote?

Core references:

- Boccignone et al., `Remote Respiration Measurement with RGB Cameras: A Review and Benchmark`
- the original motion-family papers already used in `paper/refs.bib`

Paper role:

- justify heterogeneous observation families
- justify that different extractors measure different respiratory proxies
- justify that COHFACE is comparatively easy and MAHNOB is comparatively difficult

## Cluster E. Robust and adaptive filtering

Question:

- Why separate observation reliability, dynamical novelty, and posterior robustification?

Core references:

- Mehra
- Mohamed and Schwarz
- Pich{\'e} et al.
- Roth et al.
- Huang et al.

Paper role:

- justify adaptive `R`
- justify posterior heavy-tailed safeguards
- justify why "one innovation magnitude" is conceptually insufficient

## Cluster F. Evaluation and benchmark fairness

Question:

- Why separate rate and waveform evaluation?

Core references:

- Lin (CCC)
- contactless respiration benchmark/review sources

Paper role:

- justify `z_osc` for T3
- justify `z_full` for T4
- justify why waveform and rate should not be conflated

## Minimum reference promises for the final paper

The final paper must explicitly support all of the following:

1. state/observation alignment principle
2. resonator suitability for quasi-periodic signals
3. respiratory variability and non-sinusoidal structure
4. camera observation heterogeneity
5. disentangled adaptation logic
6. dual-output evaluation logic

If any of these six points lacks a clear reference trail, the paper is not yet
defensible.
