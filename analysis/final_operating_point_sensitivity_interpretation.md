# Final Operating-Point Sensitivity Interpretation

This report interprets `analysis/final_operating_point_sensitivity.csv`.
The study is a bounded operating-point sensitivity check, not a target-specific
hyperparameter sweep.

## Summary Decision

Keep `locked_default` as the paper-facing operating point.

Rationale: no alternative setting produces a globally defensible improvement.
`more_local_windows` slightly improves COHFACE/MAHNOB rate MAE but damages
MAHNOB rate correlation and COHFACE strict fidelity. `more_stable_windows`
improves MAHNOB rate MAE and PearsonR slightly but worsens COHFACE rate MAE and
does not improve MAHNOB waveform/strict behavior. Tightening or loosening
cross-family support produces no meaningful downstream change.

## Metric Readout

| operating point | dataset | rate MAE | rate R | aligned CCC | strict CCC | NMAE span | guard alpha | abstain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| locked_default | COHFACE | 0.280 | 0.920 | 0.870 | 0.182 | 0.579 | 0.252 | 0.036 |
| locked_default | MAHNOB_tailaligned | 3.365 | 0.280 | 0.333 | 0.000 | 0.236 | 0.188 | 0.117 |
| more_local_windows | COHFACE | 0.275 | 0.910 | 0.868 | 0.171 | 0.588 | 0.242 | 0.038 |
| more_local_windows | MAHNOB_tailaligned | 3.355 | 0.140 | 0.344 | 0.000 | 0.236 | 0.185 | 0.125 |
| more_stable_windows | COHFACE | 0.345 | 0.920 | 0.864 | 0.187 | 0.590 | 0.240 | 0.030 |
| more_stable_windows | MAHNOB_tailaligned | 3.280 | 0.305 | 0.329 | 0.000 | 0.236 | 0.189 | 0.117 |
| stricter_cross_family_support | COHFACE | 0.280 | 0.920 | 0.870 | 0.182 | 0.579 | 0.252 | 0.036 |
| stricter_cross_family_support | MAHNOB_tailaligned | 3.365 | 0.280 | 0.333 | 0.000 | 0.236 | 0.188 | 0.117 |
| looser_cross_family_support | COHFACE | 0.280 | 0.920 | 0.870 | 0.182 | 0.579 | 0.252 | 0.036 |
| looser_cross_family_support | MAHNOB_tailaligned | 3.365 | 0.280 | 0.333 | 0.000 | 0.236 | 0.188 | 0.117 |

## Interpretation

### Locked Default

`locked_default` remains the most defensible operating point. It preserves
COHFACE rate performance and aligned morphology while keeping MAHNOB behavior
honest: rate MAE remains high, aligned waveform remains weak, and strict
zero-lag waveform remains essentially unsolved. This is scientifically safer
than selecting a setting that improves one MAHNOB scalar while damaging other
metrics or source-domain preservation.

### More Local Windows

Shorter windows give a marginal MAE change but worsen MAHNOB PearsonR
substantially (`0.280 -> 0.140`) and increase MAHNOB abstain pressure
(`0.117 -> 0.125`). This suggests the more local reliability view can follow
within-trial variation, but in MAHNOB it also follows noisy or ambiguous
evidence. It should not be promoted to the paper default.

### More Stable Windows

Longer windows slightly improve MAHNOB rate MAE (`3.365 -> 3.280`) and PearsonR
(`0.280 -> 0.305`), but COHFACE rate MAE worsens (`0.280 -> 0.345`), aligned
CCC drops on both datasets, and strict MAHNOB remains unchanged. This is a
classic stability/locality tradeoff rather than a robust model improvement.

### Cross-Family Support Thresholds

Both stricter and looser support thresholds are numerically indistinguishable
from `locked_default` after downstream materialization. This means the current
readout is not fragile to small support-threshold perturbations, but it also
means these thresholds are not the active bottleneck. The evidence points back
to observability/readout limits rather than a missing support-threshold tune.

## Paper-Facing Implication

The paper should not claim that sensitivity improves MAHNOB. The correct claim
is narrower and stronger: the final operating point is not an arbitrary
target-tuned setting, and the hard-regime failure persists across reasonable
target-computable reliability settings. That supports the claim that MAHNOB
strict waveform failure is an observability/evaluation boundary rather than a
single hyperparameter accident.

## Final-Full Recommendation

Run final full with the existing locked default. Do not change `execute.md`
defaults based on this sensitivity table.
