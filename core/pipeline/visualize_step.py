import os
import glob
import pickle
import numpy as np
from core.utils.common import tqdm
from core.evaluation.plotting import plot_comprehensive_result

def run_visualization(results_dir: str, run_label: str = None):
    """
    Generates plots for all result files in the directory.
    """
    search_pattern = os.path.join(results_dir, "*")
    if run_label:
        from core.pipeline.common import _sanitize_run_label
        label = _sanitize_run_label(run_label)
        search_pattern = os.path.join(results_dir, f"{label}_*")

    candidate_dirs = glob.glob(search_pattern)
    target_dirs = [d for d in candidate_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'data'))]

    if not target_dirs:
        print(f"> Visualization: No result directories found matching '{run_label}'")
        return

    print(f"\n> Starting Visualization for {len(target_dirs)} dataset(s)...")

    for d_dir in target_dirs:
        dataset_name = os.path.basename(d_dir)
        data_dir = os.path.join(d_dir, 'data')
        plot_dir = os.path.join(d_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        if not pkl_files:
            continue

        print(f"   >> Generating plots for {dataset_name} in {plot_dir}")
        
        for pkl_path in tqdm(pkl_files, desc="Plotting"):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
            except Exception:
                continue
            
            # Extract data
            fname = os.path.splitext(os.path.basename(pkl_path))[0]
            
            # Assuming data structure from runner.py
            # data = {'estimates': [...], 'gt': ..., 'fps': ...}
            # estimates = [{'name': ..., 'signal_hat': ..., 'track_hz': ...}]
            
            estimates = data.get('estimates', [])
            gt_signal = data.get('gt') # This might be raw signal
            fps = data.get('fps', 30.0)
            
            # Construct time array
            # Length should match estimates
            if not estimates:
                continue
                
            # Use length of first estimate
            n_samples = len(estimates[0]['signal_hat'])
            times = np.arange(n_samples) / fps
            
            # Normalize GT for visual comparison if possible
            if gt_signal is not None and len(gt_signal) == n_samples:
                gt_sig = (gt_signal - np.mean(gt_signal)) / (np.std(gt_signal) + 1e-6)
            else:
                gt_sig = None

            # Generate plot for each method
            for est in estimates:
                method_name = est['name']
                sig_hat = est['signal_hat']
                track_hz = est['track_hz']
                
                # Normalize est signal
                sig_hat = (sig_hat - np.mean(sig_hat)) / (np.std(sig_hat) + 1e-6)
                
                bpm_est = track_hz * 60.0
                
                save_name = f"{fname}_{method_name}.html"
                save_path = os.path.join(plot_dir, save_name)
                
                plot_comprehensive_result(
                    times, 
                    sig_hat, 
                    bpm_est, 
                    signal_gt=gt_sig, 
                    bpm_gt=None, # GT BPM usually not directly available unless computed
                    title=f"{fname} : {method_name}",
                    save_path=save_path
                )
