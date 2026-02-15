
import pandas as pd
import numpy as np
import sys

# Load metrics
df = pd.read_csv("results/cohface_robust_ossm/metrics/metrics_freq_domain_raw.csv")

# Filter relevant methods
robust = df[df['method'] == 'profile1d_cubic__robust_ossm']
kfstd = df[df['method'] == 'profile1d_cubic__kfstd']

print(f"Robust Count: {len(robust)}")
print(f"KFStd Count: {len(kfstd)}")

# Join on video
merged = pd.merge(robust, kfstd, on='video', suffixes=('_rob', '_kf'))

# Compare MAE
merged['mae_diff'] = merged['MAE_rob'] - merged['MAE_kf']

print("\n=== MAE Comparison (Robust - KFStd) ===")
print(merged['mae_diff'].describe())

print("\n=== Cases where Robust is significantly worse (> 1 BPM) ===")
worse = merged[merged['mae_diff'] > 1.0]
print(worse[['video', 'MAE_rob', 'MAE_kf', 'mae_diff']].sort_values('mae_diff', ascending=False).head(10))

print("\n=== Cases where Robust is significantly better (< -1 BPM) ===")
better = merged[merged['mae_diff'] < -1.0]
print(better[['video', 'MAE_rob', 'MAE_kf', 'mae_diff']].sort_values('mae_diff').head(10))

# Check correlation with SNR
if 'SNR_Spec_rob' in merged.columns:
    print("\n=== Correlation of MAE Diff with SNR (Robust) ===")
    print(merged[['mae_diff', 'SNR_Spec_rob']].corr())
