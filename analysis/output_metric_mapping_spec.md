# Output-Metric Mapping Specification
# PARH-OSSM v1
# Date: 2026-03-31 (T4 policy updated: 2026-04-02)

## Dual Output Definition

| Output | Formula | Purpose |
|--------|---------|---------|
| z_osc | h_c^(1) + h_c^(2) | Rate estimation (oscillatory component only) |
| z_full | h_c^(1) + h_c^(2) + b_t + r_t | Waveform reconstruction (all components) |

## Metric-to-Output Mapping

### z_osc metrics (Table T3: Frequency Performance)
| Metric | Description | Computed on |
|--------|-------------|-------------|
| RR MAE (BPM) | Mean absolute error of respiratory rate | z_osc → track_hz → windowed BPM |
| RR RMSE (BPM) | Root mean square error of respiratory rate | z_osc → track_hz → windowed BPM |
| Pearson r | Correlation of rate trajectory with GT | z_osc → track_hz → windowed BPM |
| MAPE (%) | Mean absolute percentage error | z_osc → track_hz → windowed BPM |
| SNR_Spec (dB) | Spectral signal-to-noise ratio | z_osc spectrum |

### Waveform metrics (Table T4: Unified Waveform Comparison)
**Policy (locked 2026-04-02):** Option A — Unified Comparison.
All methods evaluated under identical protocol; output_type distinguishes signal source.

| Method Variant | Signal Source | output_type | Smoothing |
|---------------|--------------|-------------|-----------|
| Base | signal_hat | signal_hat | smoothed |
| KFstd | signal_hat | signal_hat | smoothed |
| PARH-OSSM | z_full | z_full | smoothed |
| PARH-OSSM (supp) | z_osc | z_osc | smoothed |

| Metric | Description | Computed on |
|--------|-------------|-------------|
| CCC | Lin's concordance correlation coefficient | signal vs GT waveform (sample-level) |
| wMAE | Waveform mean absolute error | signal vs GT waveform (sample-level) |
| DTW | Dynamic time warping distance (normalised) | signal vs GT waveform |

**T4 Main Table filter:**
```python
df_main = df[(df["causal_or_smoothed"] == "smoothed") &
             (((df["variant"] != "PARH") & (df["output_type"] == "signal_hat")) |
              ((df["variant"] == "PARH") & (df["output_type"] == "z_full")))]
```

### Filter diagnostics (Table T6: Calibration)
| Metric | Description | Source |
|--------|-------------|--------|
| NIS mean | Mean normalised innovation squared | Forward pass diagnostics |
| NIS in-band % | Fraction of NIS within chi-sq bounds | Forward pass diagnostics |
| pi_t mean | Mean prior trust | q_obs → pi_t mapping |
| lambda_t < 1 % | Fraction of Student-t downweighted | VB update |
| nu median | Median degrees of freedom | Kurtosis adaptation |

## Hard Rules (MUST NOT violate)

1. **CCC is for z_full ONLY, not z_osc.** CCC measures waveform fidelity including baseline and aperiodic components.
2. **Freq MAE/RMSE is for z_osc ONLY, not z_full.** Rate estimation comes from the oscillatory component's instantaneous phase.
3. **DTW is for z_full ONLY.** Dynamic time warping compares full waveform morphology.
4. **Causal and smoothed metrics MUST be kept separate.** Never mix causal forward-pass results with RTS-smoothed results in the same table row.
5. **Never apply waveform CCC to BPM sequences.** The existing LinCorr function in metrics.py operates on BPM windows, which is a different use case.

## Waveform Evaluation Protocol (locked)

### 1. Resampling
- If GT fs != estimate fs: GT is resampled to estimate fs via `np.interp` inside `calculate_cross_corr_alignment()`.
- Both signals operate at the estimate's fps after alignment.

### 2. Normalization
- Both z_full and GT are bandpass filtered (`filter_RW`, [0.08, 0.5] Hz) before comparison.
- Both are then standard z-scored: `(x - mean(x)) / (std(x) + 1e-9)`.
- This is implemented in `evaluation_step.py` Block 1b (lines ~1127-1135).

### 3. Alignment
- Max-lag cross-correlation via `calculate_cross_corr_alignment(est_norm, gt_norm, fs_est, fs_gt)`.
- Returns aligned segments of equal length + lag in seconds.
- Same function used for existing time-domain metrics (Block 1).

### 4. Overlapping segment rule
- After alignment, metrics are computed on the full overlapping segment.
- No sub-windowing for waveform CCC/wMAE/DTW — these are whole-trial metrics.
- If aligned length < 10 samples, metrics are skipped (NaN).

### 5. DTW dependency
- Uses `calculate_dtw_distance()` from `core/evaluation/metrics.py`.
- If `fastdtw` is not installed, returns NaN gracefully (try/except).

## Pipeline Changes Made

### core/evaluation/metrics.py
Added functions:
- `waveform_ccc(sig_est, sig_gt)` — sample-level CCC for z_full
- `waveform_mae(sig_est, sig_gt)` — sample-level MAE for z_full
- `waveform_dtw(sig_est, sig_gt)` — normalised DTW for z_full
- `compute_dual_output_metrics(result_dict, gt_signal, fs)` — orchestrator

### core/pipeline/evaluation_step.py (PATCHED 2026-03-31, EXPANDED 2026-04-02)
Block 1b: Unified waveform metrics for ALL methods (CCC/wMAE/DTW).
- **Before (2026-03-31):** Only z_full (smoothed/causal) for PARH-OSSM methods
- **After (2026-04-02):** All methods get waveform evaluation via identical protocol
- Base/KFstd: output_type='signal_hat' (smoothed)
- PARH-OSSM: output_type='z_full' (smoothed + causal), 'z_osc' (smoothed supplement)
- Outputs: `metrics_waveform_raw.csv` with columns: video, method, output_type, causal_or_smoothed, waveform_CCC, waveform_MAE, waveform_DTW, latency_ms

### components/models/heads/parh_ossm.py
Result dict now contains:
- `z_osc`, `z_full` (smoothed, primary)
- `z_osc_causal`, `z_full_causal` (forward-pass only)
- `z_osc_smoothed`, `z_full_smoothed` (aliases)
- `track_hz`, `track_hz_causal`
- `decomposition` dict with h1, h2, baseline, residual
- `diagnostics` dict with R_t, pi_t, lambda_t, nu_t, q_obs_t, q_dyn_t, q_osc_t, freq_t, nis_empirical_t
