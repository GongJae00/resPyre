# Final Reviewer Risk Response

## Why is the paper valuable if MAHNOB remains hard?

Because the paper reports where the observation bank becomes unobservable rather than hiding the failure. The failure taxonomy, strict normalized metrics, and rate-source decomposition separate model instability from missing respiratory evidence and reference/scale risk.

## Why is this not target tuning?

The baseline/comparator refresh uses pre-locked fixed methods and the same final full-dataset trial IDs. Target GT is used only for evaluation and statistical pairing, not for selecting a per-target method or threshold.

## How is OSSM-KF different from PARH-OSSM?

OSSM-KF is a standard resonator plus Kalman comparator attached to a fixed representative observation. PARH-OSSM uses target-computable reliability, candidate evidence views, state/readout role separation, and diagnostics around z_osc/z_full.

## Why report strict waveform failure?

Strict waveform exposes lag, unit, and reference-scale fragility that aligned waveform metrics can hide. Reporting it prevents overclaiming and motivates the next generation of respiratory observation operators.
