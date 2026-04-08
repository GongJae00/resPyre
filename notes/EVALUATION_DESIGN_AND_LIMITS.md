# PARH-OSSM Evaluation Design and Limits

## Purpose

This document locks the intended meaning of the current evaluation stack and
records the remaining caveats that must stay visible in the manuscript.

It is not enough for metrics to be implemented.
They must also be scientifically aligned with the model outputs.

## Output-specific evaluation logic

### T3: oscillatory output

Target:

- `z_osc`
- `track_hz`
- windowed BPM trajectory

What it measures:

- respiratory rate tracking
- oscillatory stability
- dominant-band behavior

What it does not measure:

- full waveform fidelity
- baseline semantics
- aperiodic residual fidelity

Current code path:

- `core/pipeline/evaluation_step.py`
- when `eval.use_track=true` and `track_hz` exists, rate metrics use the
  windowed track rather than a re-spectralized `signal_hat`

### T4: full respiratory waveform

Target:

- Base / KFstd: `signal_hat`
- PARH: `z_full`

What it measures:

- band-limited respiratory waveform fidelity after affine mismatch removal
- morphology consistency after alignment

Protocol:

- bandpass
- z-score
- cross-correlation alignment
- whole-trial CCC / MAE / DTW

Why this is defensible:

- camera proxies and belt signals do not live in the same absolute gain/offset
  space
- direct raw-amplitude comparison would punish observation mismatch more than
  respiratory morphology mismatch
- the metric therefore focuses on respiratory-motion fidelity rather than
  sensor-unit identity

What it does not justify:

- claims of exact belt reconstruction
- claims about absolute baseline amplitude fidelity

### T6: calibration and diagnostics

Target:

- NIS behavior
- trust / robustification behavior
- track stability
- coverage behavior

Current source priority:

- saved payload diagnostics first
- frame logs second

Why this matters:

- paper claims about self-auditing behavior must come from persistent saved
  arrays, not transient console summaries

## Current code-grounded evaluation verdict

### What is aligned

- `T3` is now routed through `track_hz` when available.
- `T4` main comparison separates Base/KFstd `signal_hat` from PARH `z_full`.
- `T6` can now read saved PARH diagnostics arrays.

### What was corrected

- the generic legacy time-domain block in `evaluation_step.py` used BPM-oriented
  helpers for waveform arrays; it has now been switched to direct waveform CCC /
  MAE / RMSE calculations

### What remains limited

- spectral-shape metrics in the frequency CSV still come from the bandpassed
  `signal_hat` spectrum, not from `track_hz`
- comparator diagnostics are still richer for PARH than for Base
- waveform evaluation intentionally removes gain/offset mismatch, so it is not
  an absolute observation-model score

## Manuscript implications

### Safe main-paper framing

- T3 demonstrates oscillatory tracking
- T4 demonstrates respiratory waveform morphology fidelity under observation
  mismatch
- T6 demonstrates calibration / diagnostic behavior

### Unsafe framing

- using T4 to imply raw observation equivalence
- using T3 to imply full waveform recovery
- treating spectral-shape metrics as if they came from the same object as the
  track-based rate metrics

## Reviewer-facing caveat that should remain explicit

The evaluation intentionally separates:

- what can be judged from oscillatory phase tracking
- what can be judged from aligned respiratory morphology
- what can be judged from filter diagnostics

This separation is not a weakness.
It is required by the dual-output model design.
