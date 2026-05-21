# Final Operating-Point Sensitivity

This is a bounded sensitivity report, not a best-of-sweep selector.
The grid is fixed before looking at target labels and changes only
semantically interpretable target-computable reliability settings:
window length/stride and cross-family support strictness.

Paper rule: do not claim a tuned MAHNOB optimum from this table.
Use it to check whether the locked operating point is fragile and to
justify one final full rerun only if the same setting is defensible
before target-GT performance is inspected.

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

## Locked Grid

- `locked_default`: window `30s/10s`, support corr `0.25`, residual `1.25`; current locked paper setting; balanced locality and stability.
- `more_local_windows`: window `20s/5s`, support corr `0.25`, residual `1.25`; more local target reliability; tests within-trial adaptation without new labels.
- `more_stable_windows`: window `45s/15s`, support corr `0.25`, residual `1.25`; longer reliability windows; tests whether MAHNOB needs more stable evidence aggregation.
- `stricter_cross_family_support`: window `30s/10s`, support corr `0.3`, residual `1`; higher agreement requirement; tests whether weak cross-family edges should abstain.
- `looser_cross_family_support`: window `30s/10s`, support corr `0.2`, residual `1.5`; lower agreement requirement; tests whether hard regimes are under-supported.
