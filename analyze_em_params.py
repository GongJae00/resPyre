
import os
import glob
import pickle
import numpy as np
import pandas as pd
import json

RESULTS_DIR = "results/p1d_cubic_tuning/cohface_p1d_cubic_tuning/data"

def analyze_em_params():
    pkl_files = glob.glob(os.path.join(RESULTS_DIR, "*.pkl"))
    if not pkl_files:
        print("No result files found.")
        return

    records = []
    
    print(f"Scanning {len(pkl_files)} files for EM parameters...")
    
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            estimates = data.get('estimates', [])
            for est in estimates:
                method = est.get('method')
                payload = est.get('estimate', est)
                
                # Check for meta in payload
                # wrapped_method _store_npz puts meta in 'meta' key of npz, 
                # but main.py saves the return of process().
                # OscillatorWrappedMethod.process returns `result` dict which has `meta`.
                
                if isinstance(payload, dict):
                    meta = payload.get('meta')
                    if isinstance(meta, str):
                        try:
                            meta = json.loads(meta)
                        except:
                            meta = {}
                            
                    if meta and isinstance(meta, dict):
                        params = meta.get('params', {})
                        # For KFStd, usage is qx. For UKFFreq, usage is qf.
                        # We try both or logic based on method name.
                        
                        q = None
                        if 'ukffreq' in method:
                            q = params.get('qf')
                        else:
                            q = params.get('qx')
                            
                        r = params.get('rv_floor')
                        
                        if q is not None and r is not None:
                            records.append({
                                'method': method,
                                'Q': float(q),
                                'R': float(r),
                                'video': os.path.basename(pkl_path)
                            })
                            
        except Exception as e:
            continue

    if not records:
        print("No EM parameters found in results.")
        return

    df = pd.DataFrame(records)
    
    print("\n[Global EM Parameter Analysis]")
    print(df.groupby('method')[['Q', 'R']].describe())
    
    print("\n[Proposed Defaults (Median)]")
    summary = df.groupby('method')[['Q', 'R']].median()
    print(summary)

    # Specific Recommendations
    print("\n[Recommendations for config.json]")
    for method, row in summary.iterrows():
        head = method.split('__')[-1]
        print(f"  > {head}: q={row['Q']:.2e}, r={row['R']:.2e}")

if __name__ == "__main__":
    analyze_em_params()
