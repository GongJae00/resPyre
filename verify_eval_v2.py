
import os
import shutil
import pickle
import numpy as np
import pandas as pd
from scipy import signal

# Define paths
TEST_DIR = "test_results_v3"
DATA_DIR = os.path.join(TEST_DIR, "dummy_dataset", "data")
METRICS_DIR = os.path.join(TEST_DIR, "dummy_dataset", "metrics")
EVAL_STEP_PATH = "core.pipeline.evaluation_step"

# Cleanup
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

# Generate Dummy Data
def generate_dummy_trial(filename, bias_bpm=0, noise_level=0.1, latency_sec=0.5, nonlinear_warp=False):
    fs = 30.0
    duration = 60.0 # 60 seconds
    t = np.arange(0, duration, 1/fs)
    
    # Ground Truth: 15 BPM (0.25 Hz)
    resp_freq = 0.25
    gt_sig = np.sin(2 * np.pi * resp_freq * t)
    
    # Estimate: 
    # 1. Apply Latency (Shift)
    # Latency = 0.5s -> 15 samples
    lag_samples = int(latency_sec * fs)
    # If latency is positive, est is DELAYED, so est(t) = gt(t - lag)
    # We simulate this by rolling gt
    est_sig_base = np.roll(gt_sig, lag_samples)
    
    # 2. Add Bias to Frequency? 
    # For waveform, frequency shift kills CCC. Let's keep freq same for waveform test,
    # but maybe modulate phase for DTW test.
    est_sig = est_sig_base.copy()
    
    if nonlinear_warp:
        # Warp speed in middle
        # Simple simulation: just add some random phase noise
        phase_jitter = np.cumsum(np.random.normal(0, 0.05, size=len(t)))
        est_sig = np.sin(2 * np.pi * resp_freq * t + phase_jitter)
        # Re-apply lag
        est_sig = np.roll(est_sig, lag_samples)

    # Add Noise
    est_sig += np.random.normal(0, noise_level, size=len(t))
    
    # Create Picke Structure
    data = {
        'gt': gt_sig,
        'fps': fs,
        'fs_gt': fs,
        'estimates': [
            {
                'method': 'DummyMethod',
                'estimate': {'signal_hat': est_sig}
            }
        ]
    }
    
    with open(os.path.join(DATA_DIR, f"{filename}.pkl"), 'wb') as f:
        pickle.dump(data, f)

# Create 2 trials
# Trial 1: Pure Latency (0.5s), No Warping
generate_dummy_trial('trial_1_latency', latency_sec=0.5, nonlinear_warp=False)
# Trial 2: Latency + Non-linear Warping
generate_dummy_trial('trial_2_warping', latency_sec=0.2, nonlinear_warp=True)

print(">> Dummy data created.")

# Run Evaluation
print(">> Running Evaluation Step...")
from core.pipeline.evaluation_step import run_evaluation

try:
    # Run
    run_evaluation(TEST_DIR, run_label=None)
    
    # Check Results
    summary_file = os.path.join(METRICS_DIR, 'metrics_time_domain_summary.txt')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            print("\n>> Time Domain Summary Content:")
            print(f.read())
            
        # Verify CSV for new columns
        df = pd.read_csv(os.path.join(METRICS_DIR, 'metrics_time_domain_raw.csv'))
        if 'Latency' in df.columns and 'DTW_Dist' in df.columns:
            print("[SUCCESS] 'Latency' and 'DTW_Dist' columns found in output CSV!")
            print(df[['video', 'Latency', 'DTW_Dist']])
        else:
            print("[FAILURE] New columns missing from CSV!")
    else:
        print("[FAILURE] Summary file not found!")
        
except Exception as e:
    print(f"[ERROR] Evaluation failed: {e}")
    import traceback
    traceback.print_exc()

print(">> Verification script finished.")
