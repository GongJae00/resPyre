import os
import glob
import pickle
import json
import copy
import numpy as np
import pandas as pd
from scipy import signal as sps
from typing import List, Dict, Optional

from core.evaluation import metrics as metrics_lib
from core.evaluation.metrics import getErrors
from core.utils.common import tqdm, filter_RW, sig_windowing, sig_to_RPM

PRIMARY_METRICS = ['RMSE', 'MAE', 'MAPE', 'R', 'SNR']

def _format_scalar(value, decimals=3):
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{float(value):.{decimals}f}"
    return str(value)

def _render_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    def _fmt_row(cells):
        return "|" + "|".join(f" {str(cell).ljust(col_widths[i])} " for i, cell in enumerate(cells)) + "|"

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines = [separator, _fmt_row(headers), separator]
    for row in rows:
        lines.append(_fmt_row(row))
    lines.append(separator)
    return "\n".join(lines)

def _spectral_estimate_on_grid(filtered_sig, fps_val, window_size, centers, min_hz, max_hz):
    """
    Computes frequency-domain RPM estimates on a specified time grid (centers) using Welch's method.
    Matches the logic from Commit 2986ef5.
    """
    if filtered_sig is None or filtered_sig.size == 0 or not centers.size:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    
    total_frames = filtered_sig.shape[-1]
    win_frames = int(round(window_size * fps_val))
    half_window = 0.5 * window_size
    
    valid_centers = []
    rpm_values = []
    
    # Sig to RPM helper logic
    welch_win = max(1, int(round(window_size / 1.5)))
    
    for center in centers:
        start_time = center - half_window
        start_idx = int(round(start_time * fps_val))
        end_idx = start_idx + win_frames
        
        if start_idx < 0 or end_idx > total_frames:
            continue
            
        segment = filtered_sig[..., start_idx:end_idx]
        if segment.shape[-1] != win_frames:
            continue
            
        # Extract peak frequency using Welch
        # If multi-dimensional, we stack and mean them as in the old code
        if segment.ndim > 1:
            rpm = sig_to_RPM(segment, fps_val, welch_win, min_hz, max_hz)
            rpm_val = float(np.nanmean(rpm))
        else:
            rpm = sig_to_RPM(segment[np.newaxis, :], fps_val, welch_win, min_hz, max_hz)
            rpm_val = float(rpm)
            
        rpm_values.append(rpm_val)
        valid_centers.append(center)
        
    return np.asarray(rpm_values), np.asarray(valid_centers)

def _blind_spectral_snr(sig, fs, min_hz, max_hz):
    """
    Computes a blind spectral SNR (Peak-to-Median power ratio in dB) matching legacy logic.
    """
    from core.utils.common import Welch_rpm
    if sig is None or sig.size == 0:
        return float('nan')
    try:
        # Match legacy Welch_rpm usage
        sig_arr = np.asarray(sig, dtype=np.float64)
        if sig_arr.ndim == 1:
            sig_arr = sig_arr[np.newaxis, :]
        
        # Use a standard 20s window for holistic trial SNR
        freqs_bpm, power = Welch_rpm(sig_arr, fs, 20.0, min_hz, max_hz)
        if power.size == 0:
            return float('nan')
        
        mean_p = np.mean(power, axis=0) if power.ndim > 1 else power
        peak_p = np.max(mean_p)
        med_p = np.median(mean_p)
        eps = 1e-12
        snr_db = 10.0 * np.log10(max(peak_p, eps) / max(med_p, eps))
        return float(snr_db)
    except Exception:
        return float('nan')

def run_evaluation(results_dir: str, run_label: str = None, win_size: float = 30.0, stride: float = 1.0, min_hz: float = 0.08, max_hz: float = 0.5):
    """
    Scans results and computes aggregate metrics using a strict Dual-Domain approach:
    1. Time Domain (Waveform Fidelity): CCC, Aligned MAE/RMSE, SNR, I:E Ratio, PPI MAE.
    2. Freq Domain (Rate Accuracy): MAE, RMSE, MAPE, Spectral SNR, Bland-Altman, KL-Div, Entropy.
    """
    from core.utils.common import get_SNR, Welch_rpm
    from core.evaluation.metrics import (
        calculate_cross_corr_alignment, bland_altman_stats, 
        calculate_spectral_snr, calculate_breathing_dynamics, calculate_spectral_shape_metrics,
        calculate_dtw_distance
    )

    # 1. Identify target directories
    search_pattern = os.path.join(results_dir, "*")
    if run_label:
        from core.pipeline.common import _sanitize_run_label
        label = _sanitize_run_label(run_label)
        search_pattern = os.path.join(results_dir, f"{label}*")

    candidate_dirs = glob.glob(search_pattern)
    target_dirs = [d for d in candidate_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'data'))]

    if not target_dirs:
        print(f"> Evaluation: No result directories found matching '{run_label}' in '{results_dir}'")
        return

    print(f"\n> Starting Dual-Domain Evaluation for {len(target_dirs)} dataset(s)...")
    
    for d_dir in target_dirs:
        dataset_name = os.path.basename(d_dir)
        data_dir = os.path.join(d_dir, 'data')
        pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        
        if not pkl_files:
            continue

        print(f"\n>> Evaluating Dataset: {dataset_name} ({len(pkl_files)} trials)")
        
        all_time_domain_records = []
        all_freq_domain_records = []
        
        # Metrics definitions
        time_metrics_list = ['CCC', 'MAE', 'RMSE', 'SNR_Time', 'Latency', 'DTW_Dist', 'IE_Err', 'PPI_MAE']
        freq_metrics_list = ['MAE', 'RMSE', 'MAPE', 'PearsonR', 'SNR_Spec', 'Bias', 'LoA_Width', 'KL_Div', 'Entropy_Err']

        for pkl_path in tqdm(pkl_files, desc="Calculating Metrics"):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {pkl_path}: {e}")
                continue

            gt_signal = data.get('gt')
            fps = data.get('fps', 30.0)
            fs_gt = data.get('fs_gt', fps)
            fname = os.path.splitext(os.path.basename(pkl_path))[0]
            
            if gt_signal is None or not isinstance(gt_signal, np.ndarray):
                continue
            
            # Preprocess GT for both domains
            # 1. Bandpass Filter (0.08 - 0.5 Hz)
            gt_filt = filter_RW(gt_signal, fs_gt, lo=min_hz, hi=max_hz)
            # 2. Z-score Normalize (for Time Domain) - Ensure 1D
            gt_norm = (gt_filt - np.mean(gt_filt)) / (np.std(gt_filt) + 1e-9)
            gt_norm = gt_norm.flatten()
            
            estimates = data.get('estimates', [])
            
            for est in estimates:
                method_name = est.get('method', est.get('name', 'unknown'))
                payload = est.get('estimate', est)
                if not isinstance(payload, dict): continue

                sig_hat = payload.get('signal_hat')
                if sig_hat is None: continue
                
                # Preprocess Estimate
                est_filt = filter_RW(sig_hat, fps, lo=min_hz, hi=max_hz)
                est_norm = (est_filt - np.mean(est_filt)) / (np.std(est_filt) + 1e-9)
                est_norm = est_norm.flatten()

                # =========================================================
                # Block 1: Time Domain (Waveform Fidelity & Dynamics)
                # =========================================================
                # 1. Cross-Correlation Alignment
                est_aligned, gt_aligned_t, lag_sec = calculate_cross_corr_alignment(est_norm, gt_norm, fs_est=fps, fs_gt=fs_gt)
                
                if len(est_aligned) > 10: # Ensure valid length
                    # Compute Waveform Metrics
                    # CCC
                    ccc_val = metrics_lib.LinCorr(est_aligned, gt_aligned_t)
                    # MAE / RMSE (on Z-scored aligned signals)
                    errs_time = getErrors(est_aligned, gt_aligned_t, None, None, ['MAE', 'RMSE'])
                    mae_time = errs_time[0]
                    rmse_time = errs_time[1]
                    # SNR (Time Domain)
                    residual = gt_aligned_t - est_aligned
                    p_sig = np.sum(gt_aligned_t**2)
                    p_res = np.sum(residual**2)
                    snr_time = 10 * np.log10(p_sig / (p_res + 1e-9))
                    
                    # New Critical Metrics: Latency & DTW
                    # Latency in ms. lag_sec > 0 means Est is delayed (starts after GT).
                    latency_ms = lag_sec * 1000.0
                    
                    # DTW Distance (Non-linear robustness)
                    dtw_dist = metrics_lib.calculate_dtw_distance(est_aligned, gt_aligned_t)
                    
                    # Physiological Dynamics (I:E Ratio, PPI)
                    # Compute on Aligned signals to match phases? 
                    # Dynamics are independent of lag, but clean signals are better.
                    # Use est_norm/gt_norm (full length) or aligned? 
                    # Let's use aligned to ensure we look at same segment, 
                    # but find_peaks handles shift.
                    gt_ie, gt_ppi, _ = calculate_breathing_dynamics(gt_norm, fs_gt)
                    est_ie, est_ppi, _ = calculate_breathing_dynamics(est_norm, fps)
                    
                    ie_err = abs(est_ie - gt_ie) if (np.isfinite(est_ie) and np.isfinite(gt_ie)) else np.nan
                    ppi_mae = abs(est_ppi - gt_ppi) if (np.isfinite(est_ppi) and np.isfinite(gt_ppi)) else np.nan

                    rec_time = {
                        'video': fname, 'method': method_name,
                        'CCC': ccc_val, 'MAE': mae_time, 'RMSE': rmse_time,
                        'SNR_Time': snr_time, 
                        'Latency': latency_ms, 'DTW_Dist': dtw_dist,
                        'IE_Err': ie_err, 'PPI_MAE': ppi_mae,
                        '_sig_aligned_pair': (est_aligned, gt_aligned_t),
                        'data_file': os.path.relpath(pkl_path, d_dir)
                    }
                    all_time_domain_records.append(rec_time)

                # =========================================================
                # Block 2: Frequency Domain (Rate Accuracy & Spectral Shape)
                # =========================================================
                # Use Sliding Window (30s window, 1s stride)
                # We need to re-window both GT and Est with the same parameters
                
                # GT RPM Sequence
                gt_win, t_gt = sig_windowing(gt_filt, fs_gt, win_size, stride=stride)
                gt_rpms = sig_to_RPM(gt_win, fs_gt, int(win_size/1.5), min_hz, max_hz).reshape(-1)
                
                # Est RPM Sequence
                est_win, t_est = sig_windowing(est_filt, fps, win_size, stride=stride)
                est_rpms = sig_to_RPM(est_win, fps, int(win_size/1.5), min_hz, max_hz).reshape(-1)
                
                # Align RPM sequences by time centers
                min_len_r = min(len(gt_rpms), len(est_rpms))
                if min_len_r > 5: # Need enough points for stats
                    gt_rpm_seq = gt_rpms[:min_len_r]
                    est_rpm_seq = est_rpms[:min_len_r]
                    
                    # Compute Rate Metrics
                    errs_freq = getErrors(est_rpm_seq, gt_rpm_seq, None, None, ['MAE', 'RMSE', 'MAPE', 'PearsonR'])
                    
                    # Spectral SNR (De Haan) - Averaged over windows
                    # Spectral Shape (KL Div, Entropy) - Computed on Average PSD of the whole trial for robustness
                    # (Computing KL per window is noisy).
                    
                    # 1. Spectral SNR Map
                    snr_vals = []
                    for w_sig in est_win:
                        s_val = calculate_spectral_snr(w_sig.flatten(), fps, min_hz, max_hz)
                        if np.isfinite(s_val): snr_vals.append(s_val)
                    avg_spec_snr = np.mean(snr_vals) if snr_vals else np.nan
                    
                    # 2. Global PSD Shape Comparison
                    # Re-compute full Welch for shape comparison
                    # Use standard 30s window Welch on full signal
                    _, p_gt = Welch_rpm(gt_filt, fs_gt, win_size, min_hz, max_hz)
                    _, p_est = Welch_rpm(est_filt, fps, win_size, min_hz, max_hz)
                    
                    # Handle multi-channel return from Welch_rpm if any
                    if p_gt.ndim > 1: p_gt = np.mean(p_gt, axis=0)
                    if p_est.ndim > 1: p_est = np.mean(p_est, axis=0)
                    
                    # Ensure same length? Welch_rpm outputs based on nfft. 
                    # If fs differs, length differs. Resample PSD or ensure fs matches?
                    # Assuming fps ~= fs_gt for this metric, or interpolated.
                    # If lengths differ, we cannot compute KL directly.
                    # Interpolate Est PSD to GT PSD bins if needed.
                    if len(p_gt) != len(p_est):
                        p_est = np.interp(np.linspace(0, 1, len(p_gt)), np.linspace(0, 1, len(p_est)), p_est)

                    kl_div, ent_err, _ = calculate_spectral_shape_metrics(p_est, p_gt)
                    
                    # Bland-Altman
                    bias, lower_loa, upper_loa = bland_altman_stats(est_rpm_seq, gt_rpm_seq)
                    loa_width = upper_loa - lower_loa
                    
                    rec_freq = {
                        'video': fname, 'method': method_name,
                        'MAE': errs_freq[0], 'RMSE': errs_freq[1], 'MAPE': errs_freq[2], 
                        'PearsonR': errs_freq[3],
                        'SNR_Spec': avg_spec_snr,
                        'Bias': bias, 'LoA_Width': loa_width,
                        'KL_Div': kl_div, 'Entropy_Err': ent_err,
                        '_rpm_pair': (est_rpm_seq, gt_rpm_seq),
                        'data_file': os.path.relpath(pkl_path, d_dir)
                    }
                    all_freq_domain_records.append(rec_freq)

        # Save Logic
        metrics_dir = os.path.join(d_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        def _save_domain_v2(records, label, metric_keys):
            if not records: return
            df = pd.DataFrame(records)
            
            # Numeric conversion
            for m in metric_keys: df[m] = pd.to_numeric(df[m], errors='coerce')
            
            # Sort
            methods_in_df = df['method'].unique()
            sorted_methods = sorted(methods_in_df, key=_method_sort_key)
            
            # Group stats
            summary_stats = df.groupby('method')[metric_keys].agg(['median', 'std'])
            summary_stats.columns = [f'{m}_{s}' for m, s in summary_stats.columns]
            summary_stats = summary_stats.reset_index()
            summary_stats['Method'] = summary_stats['method']
            
            summary_stats['sort_key'] = summary_stats['Method'].apply(_method_sort_key)
            summary_stats = summary_stats.sort_values('sort_key').drop(columns=['sort_key'])
            
            # Format Table
            table_headers = ['Method'] + [f"{m} (median±std)" for m in metric_keys]
            table_rows = []
            for _, row in summary_stats.iterrows():
                tr = [row['Method']]
                for m in metric_keys:
                    val = row[f'{m}_median']
                    std = row[f'{m}_std']
                    # Format nice scalars
                    tr.append(f"{_format_scalar(val)} (±{_format_scalar(std, decimals=2)})")
                table_rows.append(tr)
            
            table_str = _render_table(table_headers, table_rows)
            txt_name = f'metrics_{label}_summary.txt'
            
            title = "Time Domain (Waveform Fidelity)" if label == 'time_domain' else "Freq Domain (Rate Accuracy)"
            
            with open(os.path.join(metrics_dir, txt_name), 'w', encoding='utf-8') as f:
                f.write(f"# Dual-Domain Eval | {title}\n{table_str}\n")
            
            print(f"\n   [{title}]")
            print(table_str)
            
            # CSVs
            df_clean = df.drop(columns=[c for c in df.columns if c.startswith('_')], errors='ignore')
            df_clean.to_csv(os.path.join(metrics_dir, f'metrics_{label}_raw.csv'), index=False)
            summary_stats.to_csv(os.path.join(metrics_dir, f'metrics_{label}_summary.csv'), index=False)
            
            # Save Pickles for plots
            # Reconstruct legacy structure for plotting compatibility
            method_metrics = {}
            for rec in records:
                m = rec['method']
                entry = {
                    'video': rec['video'],
                    'metrics': [rec[k] for k in metric_keys],
                    'source_label': label,
                    'data_file': rec.get('data_file')
                }
                if label == 'time_domain':
                    entry['pair'] = rec.get('_sig_aligned_pair')
                else:
                    entry['pair'] = rec.get('_rpm_pair')
                    
                method_metrics.setdefault(m, []).append(entry)
                
            with open(os.path.join(metrics_dir, f'metrics_{label}.pkl'), 'wb') as f:
                pickle.dump([metric_keys, method_metrics], f)

        _save_domain_v2(all_time_domain_records, 'time_domain', time_metrics_list)
        _save_domain_v2(all_freq_domain_records, 'freq_domain', freq_metrics_list)


def _method_sort_key(method_name: str):
    """
    Returns a sort key that groups by base method and puts base before variants.
    Example Order:
    1. DoF
    2. dof__kfstd
    3. dof__ukffreq
    4. OF_Model
    5. of_farneback__kfstd
    ...
    """
    name = method_name.lower().replace(' ', '_')
    
    # 1. Determine the base family
    if 'dof' in name:
        family = 10
    elif 'of_farneback' in name or 'of_model' in name:
        family = 20
    elif 'profile1d_linear' in name:
        family = 30
    elif 'profile1d_quadratic' in name:
        family = 40
    elif 'profile1d_cubic' in name:
        family = 50
    else:
        family = 99
        
    # 2. Determine sub-order (Base < KFstd < UKFfreq)
    if '__kfstd' in name:
        sub = 2
    elif '__ukffreq' in name:
        sub = 3
    elif '__agakf' in name:
        sub = 4
    else:
        sub = 1 # Base model
        
    return (family, sub, name)
