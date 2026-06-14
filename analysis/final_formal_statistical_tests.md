# Formal Paired Statistical Tests

Positive effects favor PARH-OSSM. Wilcoxon tests are two-sided signed-rank tests on paired trial-level deltas; confidence intervals are percentile bootstrap intervals for the median paired effect. q-values use Benjamini-Hochberg correction.

| dataset | comparison | metric | N | median effect | 95% CI | Wilcoxon p | BH q | dz | positive fraction |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| COHFACE | PARH-OSSM vs P1D quad direct | rate_MAE | 160 | -0.06 | [-0.11, -0.03] | 0.0833 | 0.12 | 0.0715 | 0.35 |
| COHFACE | PARH-OSSM vs OSSM-KF (P1D quad) | rate_MAE | 160 | 0.01 | [-0.02, 0.1] | 0.00412 | 0.00706 | 0.19 | 0.512 |
| COHFACE | PARH-OSSM vs P1D quad direct | aligned_CCC | 160 | 0.02824 | [0.01953, 0.04064] | 3.16e-11 | 9.47e-11 | 0.49 | 0.762 |
| COHFACE | PARH-OSSM vs OSSM-KF (P1D quad) | aligned_CCC | 160 | 0.00328 | [-0.003873, 0.01233] | 0.117 | 0.151 | 0.189 | 0.537 |
| COHFACE | PARH-OSSM vs P1D quad direct | strict_span_NMAE | 160 | 52.74 | [41.63, 67.29] | 5.24e-28 | 3.14e-27 | 0.624 | 1 |
| COHFACE | PARH-OSSM vs OSSM-KF (P1D quad) | strict_span_NMAE | 160 | 0.2726 | [0.2314, 0.3282] | 2.26e-26 | 1.16e-25 | 0.333 | 0.969 |
| COHFACE | PARH-OSSM vs P1D quad direct | cycle_PPI_MAE | 160 | 0.254 | [0.1186, 0.3591] | 3.16e-15 | 1.26e-14 | 0.602 | 0.769 |
| COHFACE | PARH-OSSM vs OSSM-KF (P1D quad) | cycle_PPI_MAE | 160 | 0.05583 | [0.025, 0.1219] | 5.49e-09 | 1.41e-08 | 0.431 | 0.644 |
| MAHNOB | PARH-OSSM vs P1D quad direct | rate_MAE | 488 | 4.2 | [3.94, 4.43] | 7.09e-81 | 1.28e-79 | 1.9 | 0.971 |
| MAHNOB | PARH-OSSM vs OSSM-KF (P1D quad) | rate_MAE | 488 | 1.85 | [1.615, 2.11] | 1.08e-54 | 7.8e-54 | 0.92 | 0.82 |
| MAHNOB | PARH-OSSM vs P1D quad direct | rate_RMSE | 488 | 5.095 | [4.72, 5.365] | 1.39e-81 | 4.99e-80 | 2.19 | 0.994 |
| MAHNOB | PARH-OSSM vs OSSM-KF (P1D quad) | rate_RMSE | 488 | 1.97 | [1.695, 2.285] | 5.12e-57 | 4.6e-56 | 0.962 | 0.834 |
| MAHNOB | PARH-OSSM vs P1D quad direct | aligned_CCC | 488 | -0.01718 | [-0.03123, 0.0008332] | 4.71e-05 | 9.42e-05 | -0.251 | 0.457 |
| MAHNOB | PARH-OSSM vs OSSM-KF (P1D quad) | aligned_CCC | 488 | 0.004795 | [-0.01181, 0.01366] | 0.385 | 0.426 | -0.0844 | 0.51 |
| MAHNOB | PARH-OSSM vs P1D quad direct | strict_CCC | 488 | -0.0006662 | [-0.002446, 0.0006228] | 0.247 | 0.297 | -0.0403 | 0.475 |
| MAHNOB | PARH-OSSM vs OSSM-KF (P1D quad) | strict_CCC | 488 | 5.724e-06 | [-8.763e-06, 2.43e-05] | 0.391 | 0.426 | 0.0148 | 0.516 |
| MAHNOB | PARH-OSSM vs P1D quad direct | strict_span_NMAE | 488 | -0.03054 | [-0.03805, -0.01282] | 0.00192 | 0.00346 | 0.161 | 0.391 |
| MAHNOB | PARH-OSSM vs OSSM-KF (P1D quad) | strict_span_NMAE | 488 | -0.07622 | [-0.08304, -0.06877] | 9.13e-66 | 1.1e-64 | -0.984 | 0.123 |
| MAHNOB | PARH-OSSM vs P1D quad direct | cycle_PPI_MAE | 479 | 0.2866 | [0.1872, 0.3497] | 0.0981 | 0.131 | -0.192 | 0.637 |
| MAHNOB | PARH-OSSM vs OSSM-KF (P1D quad) | cycle_PPI_MAE | 479 | 0.04855 | [-0.01111, 0.08983] | 0.0236 | 0.0369 | -0.294 | 0.534 |
| MAHNOB | PARH-OSSM vs P1D quad direct | cycle_IE_error | 477 | 0.1172 | [0.09606, 0.139] | 3.26e-26 | 1.47e-25 | 0.295 | 0.73 |
| MAHNOB | PARH-OSSM vs OSSM-KF (P1D quad) | cycle_IE_error | 479 | 0.05816 | [0.04007, 0.08075] | 1.71e-12 | 6.17e-12 | 0.107 | 0.637 |

Full machine-readable table: `analysis/final_formal_statistical_tests.csv`.
