#!/usr/bin/env python3
"""
Advanced Signal Analysis Script for ResPyre.
Purpose: Systematic 4-Step Diagnosis of Non-Linear/Non-Gaussian properties.
Steps: 1. Raw Data, 2. Observation Logic, 3. Preprocessed Signal, 4. Residual/Noise.
"""

import sys
import os
import argparse
import numpy as np
import scipy.stats as sps
import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils.config import load_config
from core.utils.common import tqdm, get_chest_ROI
from components.observations.methods import OF_Model, profile1D_Model, DoF_Model
from components.datasets.impl import BP4D, COHFACE, MAHNOB

def _build_datasets(dataset_configs):
    datasets = []
    for d_cfg in dataset_configs:
        name = d_cfg['name'].lower()
        if name == 'bp4d': ds = BP4D()
        elif name == 'cohface': ds = COHFACE()
        elif name == 'mahnob': ds = MAHNOB()
        else: raise ValueError(f"Unknown dataset: {name}")
        ds.configure(d_cfg)
        datasets.append(ds)
    return datasets

def analyze_step1_raw(frames):
    """Step 1: Raw Data Analysis (Pixel Dynamics)"""
    # Sample intensity over time from first ROI's center
    # frames is a list of PIL Images or numpy arrays
    sample_frame = np.array(frames[0])
    h, w = sample_frame.shape[:2]
    pixel_series = []
    for f in frames:
        arr = np.array(f)
        pixel_series.append(np.mean(arr[h//2-5:h//2+5, w//2-5:w//2+5, :]))
    
    pixel_series = np.array(pixel_series)
    pixel_series_norm = (pixel_series - np.mean(pixel_series)) / (np.std(pixel_series) + 1e-9)
    kurt = sps.kurtosis(pixel_series_norm)
    return {'kurtosis': kurt}

def analyze_step2_obs(y_raw, fs):
    """Step 2: Observation Signal (Harmonics/Non-linearity)"""
    n = len(y_raw)
    freqs = np.fft.rfftfreq(n, 1/fs)
    fft_mag = np.abs(np.fft.rfft(y_raw))
    mask = (freqs >= 0.1) & (freqs <= 0.5)
    if not np.any(mask): return {'thd': 0}
    f0_idx = np.argmax(fft_mag[mask]) + np.where(mask)[0][0]
    p0 = fft_mag[f0_idx]**2
    p_harmonics = 0
    for h in [2, 3]:
        idx_h = np.argmin(np.abs(freqs - h * freqs[f0_idx]))
        p_harmonics += fft_mag[idx_h]**2
    return {'thd': p_harmonics / (p0 + 1e-9)}

def analyze_step3_preproc(y_clean):
    """Step 3: Preprocessed Signal (Impulse detection)"""
    threshold = 3.0
    impulse_rate = np.mean(np.abs(y_clean) > threshold)
    return {'impulse_rate': impulse_rate}

def analyze_step4_residual(y_clean):
    """Step 4: Residual/Noise Analysis — Extended EDA (Phase 0c).

    Returns:
        kurtosis: excess kurtosis (0 = Gaussian)
        innovation: normalized innovation sequence
        t_aic_delta: ΔAIC = AIC_gauss - AIC_t  (negative → Student-t better)
        t_fit_nu: fitted Student-t ν  (lower = heavier tails)
        hill_alpha: Hill tail-index estimator  (< 4 → heavy tail confirmed)
        arch_lm_pval: ARCH-LM p-value  (< 0.05 → heteroscedastic)
        ljung_box_pval: Ljung-Box p-value on v²  (< 0.05 → serial correlation)
        shapiro_pval: Shapiro-Wilk p-value  (< 0.05 → reject Gaussian)
    """
    innovation = np.diff(y_clean)
    innovation_norm = (innovation - np.mean(innovation)) / (np.std(innovation) + 1e-9)
    kurt = sps.kurtosis(innovation_norm)
    n = len(innovation_norm)

    result = {'kurtosis': kurt, 'innovation': innovation_norm}

    # ── 1. Student-t vs Gaussian AIC ──
    try:
        from scipy.stats import t as t_dist, norm as norm_dist
        import scipy.optimize as opt
        
        # Helper for robust Student-t fit
        def robust_t_fit(data_norm):
            def nll(params):
                nu, loc, scale = params
                if nu <= 1.05 or scale <= 1e-6:
                    return 1e10
                val = -np.sum(t_dist.logpdf(data_norm, df=nu, loc=loc, scale=scale))
                if nu > 100:
                    val += (nu - 100) * 0.01  # weak penalty to keep nu bounded if Data is Gaussian
                return val
                
            res = opt.minimize(nll, x0=[5.0, 0.0, 1.0], method='L-BFGS-B', bounds=[(1.1, 500), (None, None), (1e-6, None)])
            if res.success:
                return res.x
            else:
                return t_dist.fit(data_norm)

        t_params = robust_t_fit(innovation_norm)
        t_ll = np.sum(t_dist.logpdf(innovation_norm, *t_params))

        t_aic = 2 * 3 - 2 * t_ll  # k=3 params
        # Fit Gaussian (2 params: loc, scale)
        g_params = norm_dist.fit(innovation_norm)
        g_ll = np.sum(norm_dist.logpdf(innovation_norm, *g_params))
        g_aic = 2 * 2 - 2 * g_ll  # k=2 params
        result['t_aic_delta'] = float(g_aic - t_aic)  # positive → t is better
        result['t_fit_nu'] = float(t_params[0])
    except Exception:
        result['t_aic_delta'] = 0.0
        result['t_fit_nu'] = float('inf')

    # ── 2. Hill estimator (tail index) ──
    try:
        abs_sorted = np.sort(np.abs(innovation_norm))[::-1]
        # Use top 10% of data for tail estimation
        k = max(10, int(0.1 * n))
        k = min(k, n - 1)
        if k > 1 and abs_sorted[k] > 0:
            log_ratios = np.log(abs_sorted[:k] / abs_sorted[k])
            alpha = float(k / np.sum(log_ratios)) if np.sum(log_ratios) > 0 else float('inf')
        else:
            alpha = float('inf')
        result['hill_alpha'] = alpha
    except Exception:
        result['hill_alpha'] = float('inf')

    # ── 3. ARCH-LM test (heteroscedasticity in innovations) ──
    try:
        from statsmodels.stats.diagnostic import het_arch
        v_sq = innovation_norm ** 2
        lm_stat, lm_pval, _, _ = het_arch(v_sq, nlags=min(10, n // 5))
        result['arch_lm_pval'] = float(lm_pval)
    except Exception:
        result['arch_lm_pval'] = float('nan')

    # ── 4. Ljung-Box on v² (serial correlation in variance) ──
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        v_sq = innovation_norm ** 2
        lb_result = acorr_ljungbox(v_sq, lags=[min(10, n // 5)], return_df=True)
        result['ljung_box_pval'] = float(lb_result['lb_pvalue'].iloc[-1])
    except Exception:
        result['ljung_box_pval'] = float('nan')

    # ── 5. Shapiro-Wilk (subsample for large n) ──
    try:
        from scipy.stats import shapiro
        sub = innovation_norm[:min(5000, n)]  # Shapiro limit
        _, sw_pval = shapiro(sub)
        result['shapiro_pval'] = float(sw_pval)
    except Exception:
        result['shapiro_pval'] = float('nan')

    return result

def analyze_residuals(signal_val, fs):
    detrended = signal.detrend(signal_val)
    nyq = 0.5 * fs
    b, a = signal.butter(2, [0.08 / nyq, 0.5 / nyq], btype='bandpass')
    resp = signal.filtfilt(b, a, detrended)
    return (resp - np.mean(resp)) / (np.std(resp) + 1e-9)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-n', '--num-samples', type=int, default=3)
    parser.add_argument('--output', default='analysis/noise_properties')
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    datasets = _build_datasets(cfg['datasets'])
    os.makedirs(args.output, exist_ok=True)
    
    methods = [
        ('OF_Farneback', OF_Model()),
        ('DoF', DoF_Model()),
        ('Profile1D_Linear', profile1D_Model('linear')),
        ('Profile1D_Quad', profile1D_Model('quadratic')),
        ('Profile1D_Cubic', profile1D_Model('cubic'))
    ]
    results_summary = {m[0]: {
        'raw_kurt': [], 'thd': [], 'imp_rate': [], 'noise_kurt': [],
        'all_innovations': [], 'all_spectra': [],
        # Phase 0c extended EDA
        't_aic_delta': [], 't_fit_nu': [],
        'hill_alpha': [], 'arch_lm_pval': [],
        'ljung_box_pval': [], 'shapiro_pval': [],
    } for m in methods}
    
    method_to_cache = {
        'OF_Farneback': 'obs_of.npy',
        'DoF': 'obs_dof.npy',
        'Profile1D_Linear': 'obs_p1d_linear.npy',
        'Profile1D_Quad': 'obs_p1d_quad.npy',
        'Profile1D_Cubic': 'obs_p1d_cubic.npy'
    }

    for ds in datasets:
        ds.load_dataset()
        samples = ds.data[:args.num_samples] if args.num_samples > 0 else ds.data
        for i, sample in enumerate(tqdm(samples, desc="Analyzing Samples")):
            trial_dir = os.path.dirname(sample['video_path'])
            meta_path = os.path.join(trial_dir, "obs_meta.json")
            
            # 1. Try to load metadata
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                fs = meta['fps']
                res1 = {'kurtosis': meta['raw_kurt']}
            else:
                frames, fs, _ = get_chest_ROI(sample['video_path'], ds.name)
                res1 = analyze_step1_raw(frames)
            
            for name, model in methods:
                cache_file = os.path.join(trial_dir, method_to_cache.get(name, ''))
                if os.path.exists(cache_file):
                    y_raw = np.load(cache_file)
                else:
                    frames, fs, _ = get_chest_ROI(sample['video_path'], ds.name)
                    data_dict = {'chest_rois': frames, 'fps': fs}
                    y_raw = model.process(data_dict)
                    if isinstance(y_raw, list): y_raw = np.array(y_raw)
                    y_raw = y_raw.flatten()
                
                # Spectral for THD visualization
                n = len(y_raw)
                freqs = np.fft.rfftfreq(n, 1/fs)
                fft_mag = np.abs(np.fft.rfft(y_raw))
                # Store normalized spectrum in range 0.1-1.0 Hz
                mask = (freqs >= 0.05) & (freqs <= 1.0)
                if np.any(mask):
                    results_summary[name]['all_spectra'].append((freqs[mask], fft_mag[mask] / (np.max(fft_mag[mask]) + 1e-9)))
                
                res2 = analyze_step2_obs(y_raw, fs)
                y_clean = analyze_residuals(y_raw, fs)
                res3 = analyze_step3_preproc(y_clean)
                res4 = analyze_step4_residual(y_clean)
                
                results_summary[name]['raw_kurt'].append(res1['kurtosis'])
                results_summary[name]['thd'].append(res2['thd'])
                results_summary[name]['imp_rate'].append(res3['impulse_rate'])
                results_summary[name]['noise_kurt'].append(res4['kurtosis'])
                results_summary[name]['all_innovations'].append(res4['innovation'])
                # Phase 0c extended diagnostics
                results_summary[name]['t_aic_delta'].append(res4.get('t_aic_delta', 0.0))
                results_summary[name]['t_fit_nu'].append(res4.get('t_fit_nu', float('inf')))
                results_summary[name]['hill_alpha'].append(res4.get('hill_alpha', float('inf')))
                results_summary[name]['arch_lm_pval'].append(res4.get('arch_lm_pval', float('nan')))
                results_summary[name]['ljung_box_pval'].append(res4.get('ljung_box_pval', float('nan')))
                results_summary[name]['shapiro_pval'].append(res4.get('shapiro_pval', float('nan')))

    # Global Aggregated Plots
    for name, s in results_summary.items():
        if not s['all_innovations']: continue
        
        agg_innov = np.concatenate(s['all_innovations'])
        avg_kurt = np.mean(s['noise_kurt'])
        avg_thd = np.mean(s['thd'])
        
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        
        # 1. Histogram
        sns.histplot(agg_innov, kde=True, ax=axes[0], color='blue', stat="density")
        axes[0].set_title(f"Innovation Dist\n(Kurt={avg_kurt:.2f})")
        
        # 2. Q-Q Plot
        sps.probplot(agg_innov, dist="norm", plot=axes[1])
        axes[1].set_title("Global Q-Q Plot")
        
        # 3. Phase Portrait
        step = max(1, len(agg_innov) // 5000)
        axes[2].scatter(agg_innov[::step][:-1], agg_innov[::step][1:], alpha=0.1, s=1, color='green')
        axes[2].set_title("Global Phase Portrait")
        
        # 4. Aggregated Spectrum (THD Visual)
        if s['all_spectra']:
            # For simplicity, interpolate all to a common frequency grid
            common_f = np.linspace(0.05, 1.0, 200)
            all_mags = []
            for f, m in s['all_spectra']:
                all_mags.append(np.interp(common_f, f, m))
            avg_mag = np.mean(all_mags, axis=0)
            axes[3].plot(common_f, avg_mag, color='red')
            axes[3].set_title(f"Avg Spectrum\n(THD={avg_thd:.4f})")
            axes[3].set_xlabel("Freq (Hz)")
            # Find fundamental and draw markers
            f0 = common_f[np.argmax(avg_mag)]
            axes[3].axvline(f0, color='gray', linestyle='--', alpha=0.5, label='Fundamental')
            for h in [2]:
                if h*f0 <= 1.0: axes[3].axvline(h*f0, color='orange', linestyle=':', alpha=0.5, label=f'{h}x harmonic')

        # 5. Stats info
        axes[4].text(0.1, 0.5, f"Method: {name}\nN Samples: {len(s['raw_kurt'])}\nRaw Kurt: {np.mean(s['raw_kurt']):.2f}\nAvg THD: {avg_thd:.4f}\nImpulse%: {np.mean(s['imp_rate'])*100:.2f}%", 
                     fontsize=12, verticalalignment='center')
        axes[4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f"global_analysis_{name}.png"))
        plt.close()

    import json
    import csv

    # Helper: safe mean that handles inf/nan
    def _safe_mean(lst):
        clean = [x for x in lst if np.isfinite(x)]
        return float(np.mean(clean)) if clean else float('nan')

    # Prepare stats for saving
    final_stats = []
    for name, s in results_summary.items():
        if not s['raw_kurt']: continue
        row = {
            'Method': name,
            'Raw_Kurt': float(np.mean(s['raw_kurt'])),
            'THD': float(np.mean(s['thd'])),
            'Impulse_Rate': float(np.mean(s['imp_rate'])),
            'Noise_Kurt': float(np.mean(s['noise_kurt'])),
            # Phase 0c extended diagnostics
            't_AIC_delta': _safe_mean(s['t_aic_delta']),
            't_fit_nu': _safe_mean(s['t_fit_nu']),
            'Hill_alpha': _safe_mean(s['hill_alpha']),
            'ARCH_LM_pval': _safe_mean(s['arch_lm_pval']),
            'LjungBox_pval': _safe_mean(s['ljung_box_pval']),
            'Shapiro_pval': _safe_mean(s['shapiro_pval']),
        }
        # Verdict: Student-t justified if ΔAIC > 10 AND κ > 3
        row['Student_t_justified'] = (
            row['t_AIC_delta'] > 10 and row['Noise_Kurt'] > 3
        ) if np.isfinite(row['t_AIC_delta']) else False
        final_stats.append(row)

    # 1. Save JSON
    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(final_stats, f, indent=4)

    # 2. Save CSV
    with open(os.path.join(args.output, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=final_stats[0].keys())
        writer.writeheader()
        writer.writerows(final_stats)

    # 3. Console output
    print("\n" + "=" * 130)
    hdr = (f"{'Method':<20} | {'Kurt':>6} | {'THD':>7} | {'Imp%':>5} "
           f"| {'ΔAIC':>8} | {'ν_fit':>6} | {'Hill':>6} "
           f"| {'ARCH-p':>7} | {'LB-p':>7} | {'SW-p':>7} | {'Student-t?':>10}")
    print(hdr)
    print("-" * 130)
    for row in final_stats:
        verdict = "✅ YES" if row['Student_t_justified'] else "  no"
        print(f"{row['Method']:<20} | {row['Noise_Kurt']:6.2f} | {row['THD']:7.4f} | {row['Impulse_Rate']*100:4.1f}% "
              f"| {row['t_AIC_delta']:8.1f} | {row['t_fit_nu']:6.2f} | {row['Hill_alpha']:6.2f} "
              f"| {row['ARCH_LM_pval']:7.4f} | {row['LjungBox_pval']:7.4f} | {row['Shapiro_pval']:7.4f} | {verdict:>10}")
    print("=" * 130)
    print(f"Results, Plots, JSON, and CSV saved to: {args.output}")

if __name__ == '__main__': main()
