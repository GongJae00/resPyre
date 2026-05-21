# Dataset Distribution EDA

This report complements the main COHFACE/MAHNOB-HCI metrics with label and
rate-regime evidence from V4V and SCAMPS. It is not a hidden training or
target-tuning step.

## Summary

| dataset | role | N | median RR bpm | IQR | duration median | paper use | boundary |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| COHFACE | primary real waveform/rate | 160 | 12.00 | 4.00 | 61.2 | main benchmark | supports real waveform/rate claims |
| MAHNOB-HCI | hard real waveform/rate | 525 | 8.00 | 8.00 | 118.0 | hard-regime benchmark | supports observability-boundary analysis |
| V4V | external real rate-only | 724 | 13.82 | 4.67 |  | supplementary rate-only scope | no waveform or morphology claims |
| SCAMPS | synthetic controlled diagnostic | 2800 | nan | nan |  | supplementary synthetic control | no real-data performance claims |

## Interpretation

- COHFACE and MAHNOB-HCI remain the only real waveform/rate benchmarks.
- V4V contributes real RR-rate label distribution and external timing scope, but no waveform morphology evidence.
- SCAMPS contributes controlled synthetic breathing-signal coverage and a sanity-check rate distribution, not real-world robustness.
- The distribution view is useful because it shows that label availability is not the same as observation observability.
