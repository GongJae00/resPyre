# ResPyre Benchmark Alignment Note

Date: 2026-04-08

Reference document read directly:

- `/home/gongjae/바탕화면/공유폴더/ResPyre.pdf`

Current repository reference:

- `paper/refs.bib` entry `Boccignone2025Review`

## What the benchmark contributes

The `resPyre` benchmark is the correct external reference for:

- the high-level taxonomy of RGB-respiration methods
  - motion-based
  - rPPG-based
  - deep learning-based
- the choice of public benchmark datasets
  - BP4D+
  - COHFACE
  - MAHNOB-HCI
- the baseline observation families we reused
  - OF
  - DoF
  - Profile1D

It is therefore the right benchmark paper to cite when motivating why the
field needs reproducible evaluation and why motion-based observation families
remain competitive.

## Why the current PARH paper intentionally diverges

The benchmark treats every method as a respiratory-waveform estimator first.
Respiration rate is then derived from the estimated waveform through spectral
analysis. This is a sensible benchmarking choice for heterogeneous extractors,
but it is not fully aligned with the present PARH design.

PARH uses an explicit latent decomposition:

- oscillatory respiratory drive
- harmonic morphology
- baseline / trend
- aperiodic residual

Because of that decomposition, the current paper uses a split evaluation:

- `T3`: rate metrics from the oscillatory output (`z_osc` or `track_hz`)
- `T4`: waveform metrics from the full output (`z_full`)
- `T6`: calibration / mechanism diagnostics

This divergence is intentional and should be framed as a model-alignment choice,
not as an incompatibility with the benchmark.

## Why `OF_bridge` exists but no `DoF_bridge` does

The benchmark families are not semantically equivalent.

- Raw `OF` retains signed directional information from dense motion.
  - This makes an OF-derived displacement-compatible bridge at least
    identifiable in principle.
- `DoF` is a thresholded motion-energy count.
  - It has already collapsed sign, phase, and direct displacement meaning.
  - A naive ``bridge'' would therefore be much less physically grounded.

Current repository evidence supports:

- raw `OF` as a velocity-like helper-heavy family
- `OF_bridge` as an additional displacement-compatible constructed family
- `DoF` as a nuisance-limited auxiliary family

## Current paper-level implication

The correct relationship to `resPyre` is:

- keep the benchmark families and reproducibility ethos
- explicitly state where the PARH paper changes evaluation routing
- argue that family-specific observation semantics are part of the model, not
  just part of preprocessing

This note should remain aligned with:

- `paper/main.tex`
- `paper/PAPER_REDESIGN_LOCK.md`
- `notes/OBSERVATION_FAMILY_SEMANTICS_LOCK.md`
