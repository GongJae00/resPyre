# T4 Waveform Comparison Policy
# Date: 2026-04-02
# Status: LOCKED
# Decision: Option A — Unified waveform comparison

## Selected Option
**Option A: Unified waveform comparison policy**

## Decision Rationale
1. **Reviewer expectation**: A main comparison table (Base / KFstd / PARH-OSSM) must use a single evaluation protocol. Reviewers will ask "why wasn't the same metric applied to all methods?"
2. **Paper story**: The thesis is that PARH-OSSM's decomposed z_full reconstructs the full respiratory waveform better than signal_hat from simpler filters. This claim requires showing Base/KFstd signal_hat CCC alongside PARH-OSSM z_full CCC under the same protocol.
3. **Pipeline cost**: Adding signal_hat waveform evaluation to Block 1b is ~20 lines. Minimal risk.
4. **T4 strength**: With unified comparison, T4 becomes a strong main table showing clear separation between methods.

## T4 Source CSV
`metrics/metrics_waveform_raw.csv` — single file per dataset

## Included Methods (main T4)
| Method Variant | output_type | Role |
|---------------|------------|------|
| Base (all families) | signal_hat | Baseline |
| KFstd (all families) | signal_hat | Prior art |
| PARH-OSSM (all families) | z_full | Proposed |

## Supplementary Only
| Method Variant | output_type | Role |
|---------------|------------|------|
| PARH-OSSM (all families) | z_osc | Internal decomposition reference |

## Included Metrics
- waveform_CCC (Lin's concordance)
- waveform_MAE (mean absolute error after zscore)
- waveform_DTW (dynamic time warping, normalized)
- latency_ms (cross-correlation lag)

## Mandatory CSV Columns
video, method, output_type, causal_or_smoothed, waveform_CCC, waveform_MAE, waveform_DTW, latency_ms, data_file

## Waveform Protocol (same for all methods)
```
bandpass(lo=0.08, hi=0.5) → zscore → cross-corr alignment → CCC/wMAE/DTW
```

## Hard Rules
1. Main T4 uses ONLY: Base(signal_hat) / KFstd(signal_hat) / PARH-OSSM(z_full)
2. PARH-OSSM z_osc waveform rows are supplement/reference ONLY
3. All methods evaluated under identical protocol (no special treatment)
4. causal_or_smoothed: main T4 reports "smoothed" as primary
5. T4 source is ONE CSV per dataset (metrics_waveform_raw.csv)
6. Statistic: median across trials per family

## Main Paper Wording
"Table 4 compares waveform reconstruction fidelity across all methods using the same evaluation protocol (bandpass → z-score → alignment → CCC). For Base and KFstd, signal_hat is the primary output; for PARH-OSSM, z_full (= z_osc + baseline + residual) provides the full respiratory trajectory including aperiodic components."

## Supplement Wording
"Supplementary Table S-T4 additionally reports PARH-OSSM z_osc waveform metrics, showing the contribution of baseline and residual states to waveform fidelity improvement."
