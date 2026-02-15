
import numpy as np
import sys
import os

trial_key = "38_0"
log_path = f"results/cohface_robust_ossm/aux/profile1d_cubic__robust_ossm/frame_logs/{trial_key}.npz"

if not os.path.exists(log_path):
    print(f"Log not found: {log_path}")
    sys.exit(1)

data = np.load(log_path)
print(f"Loaded {trial_key}")
print("Keys:", list(data.keys()))

# Extract fields
arr = data['data']
fields = data['fields']
print("Fields:", fields)

def get_col(name):
    idx = np.where(fields == name)[0]
    if len(idx) == 0:
        return np.zeros(len(arr))
    return arr[:, idx[0]]

x1 = get_col('x1')
z = get_col('z')
lambda_t = get_col('lambda_t')
nis = get_col('nis')
g_t = get_col('g_t')
v_t = get_col('v_t')
S_t = get_col('S_t')

print("\n=== Statistics ===")
print(f"Mean Lambda (Trust): {np.mean(lambda_t):.4f} (Low = Outlier)")
print(f"Mean NIS: {np.mean(nis):.4f}")
print(f"Mean Gate g_t: {np.mean(g_t):.4f}")
print(f"Mean Innovation v_t: {np.mean(np.abs(v_t)):.4f}")

# Check for divergence
div_indices = np.where(lambda_t < 0.1)[0]
if len(div_indices) > 0:
    print(f"\nPotential Loss of Tracking (Lambda < 0.1) in {len(div_indices)} frames")
    print("Indices:", div_indices[:10], "..." if len(div_indices)>10 else "")

# Frequency Track stats
freqs = np.exp(z) * 60.0
print(f"\nFrequency Track (BPM): Mean={np.mean(freqs):.2f}, Std={np.std(freqs):.2f}")
print(f"Range: [{np.min(freqs):.2f}, {np.max(freqs):.2f}]")
