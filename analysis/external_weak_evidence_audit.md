# External Weak Evidence Audit

This audit is part of the final paper execution path. It prevents external
datasets with weaker labels from being silently treated as full real waveform
benchmarks.

## Boundary Rule

- COHFACE and MAHNOB-HCI remain the only current real waveform/rate benchmarks.
- V4V is included only as external RR-rate evidence.
- SCAMPS is included only as synthetic controlled diagnostic evidence.
- Neither V4V nor SCAMPS may be pooled into main real-data waveform tables.

## Audit Table

| dataset | role | units | valid paths | label completeness | allowed use | prohibited use |
| --- | --- | ---: | ---: | ---: | --- | --- |
| V4V | external_real_rate_only | 724 | 1.000 | 1.000 | RR MAE/RMSE/Pearson only if a V4V rate-readout adapter is explicitly run | waveform CCC/DTW; strict waveform; cycle morphology |
| SCAMPS | synthetic_controlled_diagnostic | 2800 | 1.000 | 1.000 | controlled synthetic diagnostic/sanity checks; optional mechanism ablation | real-data waveform/rate benchmark performance claims |

## Dataset Notes

### V4V

- Full inclusion stage: `mandatory_manifest_and_rate_scope_audit`
- Label summary: trials=724, subjects=100, splits=train:724, median RR=13.82 bpm, median RR-IQR=5.44 bpm, median label samples=820
- Paper use: supplementary external-rate evidence; not pooled with COHFACE/MAHNOB
- Claim boundary: Supports only timing/rate generalization, never morphology.

### SCAMPS

- Full inclusion stage: `mandatory_manifest_and_synthetic_scope_audit`
- Label summary: trials=2800, valid MAT=2800, d_br coverage=1.000, raw-frame coverage=1.000, median frames=600, common d_br shape=600x1, read errors=0
- Paper use: supplementary mechanism-control evidence; not pooled with real data
- Claim boundary: Supports mechanism sanity, not real-world robustness by itself.
