# PARH-OSSM Waveform Evaluation Protocol Lock
# Date: 2026-04-01
# Status: LOCKED

## Protocol (as implemented in evaluation_step.py Block 1b)

```
Input: z_full (smoothed or causal) from PARH-OSSM result
       gt_signal from dataset ground truth

Step 1 — Bandpass filter
    z_filt = filter_RW(z_full, fps, lo=min_hz, hi=max_hz)
    gt_filt = filter_RW(gt, fs_gt, lo=min_hz, hi=max_hz)
    Default: min_hz=0.08, max_hz=0.5

Step 2 — Z-score normalization
    z_norm = (z_filt - mean(z_filt)) / (std(z_filt) + 1e-9)
    gt_norm = (gt_filt - mean(gt_filt)) / (std(gt_filt) + 1e-9)

Step 3 — Cross-correlation alignment
    z_aligned, gt_aligned, lag = calculate_cross_corr_alignment(
        z_norm, gt_norm, fs_est=fps, fs_gt=fs_gt)

Step 4 — Whole-trial metrics
    CCC  = waveform_ccc(z_aligned, gt_aligned)    # Lin's concordance
    wMAE = waveform_mae(z_aligned, gt_aligned)     # mean absolute error
    DTW  = waveform_dtw(z_aligned, gt_aligned)     # dynamic time warping
```

## Code References

| Step | File | Lines |
|------|------|-------|
| Bandpass + zscore (GT) | evaluation_step.py | Block 1 existing GT processing |
| Bandpass + zscore (z_full) | evaluation_step.py | ~1130-1138 |
| Cross-corr alignment | evaluation_step.py | ~1139-1141 |
| CCC/wMAE/DTW computation | evaluation_step.py | ~1142-1144 |
| waveform_ccc() | core/evaluation/metrics.py | :661-679 |
| waveform_mae() | core/evaluation/metrics.py | :681-697 |
| waveform_dtw() | core/evaluation/metrics.py | :699-716 |

## Metrics Definitions

### CCC (Lin's Concordance Correlation Coefficient)
```
CCC = (2 * cov(x,y)) / (var(x) + var(y) + (mean(x) - mean(y))^2)
```
Range: [-1, 1]. Perfect agreement = 1.

### wMAE (Waveform Mean Absolute Error)
```
wMAE = mean(|x - y|)
```
After zscore normalization, this is scale-free.

### DTW (Dynamic Time Warping distance)
```
DTW = fastdtw(x, y, radius=1) / len(x)
```
Normalized by signal length. Returns NaN if fastdtw not installed.

## Why This Protocol Matters

Raw z_full vs raw GT comparison yields CCC ≈ 0.003 (COHFACE) due to:
- Different sampling rates (fps ≈ 20 Hz vs fs_gt = 32 Hz)
- Different DC offsets and scales
- Phase misalignment

After protocol (bandpass + zscore + alignment): CCC ≈ 0.83-0.88.

## Variants Evaluated

| Variant | Description | CSV Column |
|---------|------------|------------|
| smoothed | RTS smoother output (z_full_smoothed) | causal_or_smoothed='smoothed' |
| causal | Forward-only filter output (z_full_causal) | causal_or_smoothed='causal' |

Paper T4 reports **smoothed** as primary, causal in supplementary.
