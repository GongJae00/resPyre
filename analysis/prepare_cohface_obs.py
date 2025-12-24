#!/usr/bin/env python3
"""
Data Preparation Script: Cache Observation Signals for COHFACE.
Purpose: Pre-calculate OF, DoF, and Profile1D (3 types) to avoid redundant heavy computation.
Saves .npy files directly in the trial directories.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils.common import tqdm, get_chest_ROI
from components.observations.methods import OF_Model, profile1D_Model, DoF_Model
from components.datasets.impl import COHFACE

def main():
    # Force the dataset location if symlinked
    custom_root = os.environ.get('RESPIRE_DATA_DIR')
    if not custom_root:
        # Default to repo_root/dataset
        repo_root = Path(__file__).resolve().parent.parent
        custom_root = str(repo_root / 'dataset')
    
    print(f"Using Data Directory: {custom_root}")
    ds = COHFACE()
    ds.load_dataset()

    methods = [
        ('of', OF_Model()),
        ('dof', DoF_Model()),
        ('p1d_linear', profile1D_Model('linear')),
        ('p1d_quad', profile1D_Model('quadratic')),
        ('p1d_cubic', profile1D_Model('cubic'))
    ]

    print(f"Starting preparation for {len(ds.data)} samples...")

    for i, item in enumerate(tqdm(ds.data, desc="Caching Observations")):
        video_path = item['video_path']
        trial_dir = os.path.dirname(video_path)
        
        # Check if all files including metadata exist to skip
        all_exist = True
        for name, _ in methods:
            if not os.path.exists(os.path.join(trial_dir, f"obs_{name}.npy")):
                all_exist = False
                break
        if not os.path.exists(os.path.join(trial_dir, "obs_meta.json")):
            all_exist = False
        
        if all_exist:
            continue

        try:
            import json
            from analysis.run_noise_analysis import analyze_step1_raw
            
            # Extract ROIs (heavy part)
            frames, fs, _ = get_chest_ROI(video_path, ds.name)
            data_dict = {'chest_rois': frames, 'fps': fs}

            # Cache Raw Kurtosis (Step 1)
            raw_res = analyze_step1_raw(frames)
            with open(os.path.join(trial_dir, "obs_meta.json"), 'w') as f:
                json.dump({'fps': fs, 'raw_kurt': float(raw_res['kurtosis'])}, f)

            for name, model in methods:
                out_path = os.path.join(trial_dir, f"obs_{name}.npy")
                if os.path.exists(out_path):
                    continue
                
                # Compute (heavy part)
                y_raw = model.process(data_dict)
                if isinstance(y_raw, list): y_raw = np.array(y_raw)
                y_raw = y_raw.flatten()
                
                # Save
                np.save(out_path, y_raw)
                
        except Exception as e:
            print(f"\n[Error] Failed processing {video_path}: {e}")
            continue

    print("\nCaching Complete!")

if __name__ == '__main__':
    main()
