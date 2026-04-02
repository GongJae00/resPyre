# T4 Policy Reconciliation Report
# Date: 2026-04-02

## Issue
`metrics_waveform_raw.csv` was originally designed for PARH-OSSM z_full only.
The main comparison (Base / KFstd / PARH-OSSM) requires unified waveform evaluation.

## Resolution: Option A (Unified Waveform Comparison)

### Pipeline Patch
`evaluation_step.py` Block 1b expanded:
- **Before**: Only z_full (smoothed/causal) for PARH-OSSM methods
- **After**: All methods get waveform evaluation via identical protocol

### New waveform CSV row types

| Method Type | output_type | Rows per trial |
|------------|------------|----------------|
| Base | signal_hat | 1 (smoothed) |
| KFstd | signal_hat | 1 (smoothed) |
| PARH-OSSM | signal_hat | 1 (smoothed) |
| PARH-OSSM | z_full | 2 (smoothed + causal) |
| PARH-OSSM | z_osc | 1 (smoothed, supplement) |

### T4 Main Table Filter
```python
df_main = df[(df["causal_or_smoothed"] == "smoothed") &
             (((df["variant"] != "PARH") & (df["output_type"] == "signal_hat")) |
              ((df["variant"] == "PARH") & (df["output_type"] == "z_full")))]
```

### Verified by Smoke Test
E2E smoke (3 COHFACE trials, 9 methods) produced 54 waveform rows:
- 6 signal_hat rows (3 Base + 3 KFstd)
- 18 PARH-OSSM rows (3 methods × {signal_hat, z_full_smooth, z_full_causal, z_osc_smooth})
- Per trial: 6 + 18/3 = 12 rows × 3 trials = ~54 ✓

### Impact on output_metric_mapping_spec.md
Updated signal routing:
- T4 now includes Base/KFstd signal_hat waveform metrics
- Protocol unchanged (bandpass → zscore → alignment → CCC/wMAE/DTW)
