import os
import glob
import pickle
import pandas as pd
import numpy as np
from core.utils.common import tqdm
from scipy import signal as sps
from core.evaluation.plotting_paper import plot_summary_mae_boxplot, plot_summary_scatter_gt, plot_trace_paper
from core.utils.common import filter_RW, sig_windowing, sig_to_RPM

def run_visualization(results_dir: str, run_label: str = None, win_size: float = 30.0, stride: float = 1.0, min_hz: float = 0.08, max_hz: float = 0.5):
    """
    Generates summary plots (PNG) from aggregated metrics and sample traces.
    Restores family-wise overlays and best-sample aligned overlays.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    search_pattern = os.path.join(results_dir, "*")
    if run_label:
        from core.pipeline.common import _sanitize_run_label
        label = _sanitize_run_label(run_label)
        search_pattern = os.path.join(results_dir, f"{label}*")

    candidate_dirs = glob.glob(search_pattern)
    target_dirs = [d for d in candidate_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'data'))]

    if not target_dirs:
        print(f"> Visualization: No result directories found matching '{run_label}'")
        return

    print(f"\n> Starting Visualization for {len(target_dirs)} dataset(s)...")

    for d_dir in target_dirs:
        dataset_name = os.path.basename(d_dir)
        csv_path = os.path.join(d_dir, 'metrics', 'metrics_raw.csv')
        plot_dir = os.path.join(d_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        metrics_pkl = os.path.join(d_dir, 'metrics', 'metrics_time_domain.pkl')
        
        # 1. Generate Summary Plots (PNG)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'MAE' in df.columns:
                    print(f"   >> Generating paper-style summary plots for {dataset_name}...")
                    
                    boxplot_path = os.path.join(plot_dir, "summary_mae_boxplot.png")
                    gt_col = 'gt_bpm_avg' if 'gt_bpm_avg' in df.columns else 'gt_bpm'
                    est_col = 'est_bpm_avg' if 'est_bpm_avg' in df.columns else 'est_bpm'
                    
                    plot_summary_mae_boxplot(df, boxplot_path, title=f"Error Distribution: {dataset_name}")
                    
                    scatter_path = os.path.join(plot_dir, "summary_scatter_gt.png")
                    scatter_df = df.copy()
                    if gt_col != 'gt_bpm': scatter_df['gt_bpm'] = scatter_df[gt_col]
                    if est_col != 'est_bpm': scatter_df['est_bpm'] = scatter_df[est_col]
                    
                    plot_summary_scatter_gt(scatter_df, scatter_path, title=f"Prediction vs Ground Truth: {dataset_name}")
                    
                    print(f"      [Saved] {boxplot_path}")
                    print(f"      [Saved] {scatter_path}")
            except Exception as e:
                print(f"   !! Failed to generate summary plots: {e}")
 
         # 2. Family Overlays logic
        if os.path.exists(metrics_pkl):
            print(f"   >> Generating family-wise overlays for {dataset_name}...")
            _save_family_overlays(d_dir, metrics_pkl)
            
            # 3. Best Sample Overlays (Lag-corrected, Min-Max normalized)
            print(f"   >> Generating best-sample aligned overlays for {dataset_name}...")
            _save_best_overlays_aligned(d_dir, metrics_pkl, stride)

        # 4. Trace Plots (Best/Median/Worst)
        print("   >> Generating standard best/median/worst sample trace plots...")
        summary_plot_dir = os.path.join(plot_dir, 'summary')
        os.makedirs(summary_plot_dir, exist_ok=True)
        _generate_bmw_trace_plots(d_dir, csv_path, plot_dir, min_hz, max_hz, win_size, stride, summary_plot_dir)

def _save_family_overlays(run_dir, metrics_pkl):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from pathlib import Path

    METRIC_NAME = "MAE"
    METHOD_COLORS = {"__kfstd": "#1b9e77", "__agakf": "#e7298a", "__ukffreq": "#2c7bb6"}

    try:
        with open(metrics_pkl, 'rb') as f:
            headers, method_metrics = pickle.load(f)
        
        if METRIC_NAME not in headers:
            return
        idx = headers.index(METRIC_NAME)
        
        # Dynamic Family Grouping: Base model first, then its variants
        all_methods = list(method_metrics.keys())
        roots = ["dof", "of_farneback", "of_model", "profile1d_linear", "profile1d_quadratic", "profile1d_cubic"]
        
        FAMILIES = {}
        remaining_methods = set(all_methods)
        
        # We normalize to lowercase for matching
        for r in roots:
            # Match methods that belong to this base root
            family = [m for m in all_methods if m.lower().replace(' ', '_').startswith(r)]
            if family:
                FAMILIES[r] = sorted(family, key=_method_sort_key)
                remaining_methods -= set(family)
                
        # Add leftover methods as their own families
        for m in sorted(list(remaining_methods)):
            FAMILIES[m] = [m]
            
        output_dir = Path(run_dir) / 'plots' / 'family_overlays'
        output_dir.mkdir(parents=True, exist_ok=True)
        results_root = Path(run_dir)

        for family_name, methods in FAMILIES.items():
            base_method = methods[0]
            family_records = []
            for m in methods:
                if m in method_metrics:
                    for rec in method_metrics[m]:
                        family_records.append({'method': m, 'rec': rec, 'score': rec['metrics'][idx]})
            
            if not family_records:
                continue
                
            best_entry = min(family_records, key=lambda x: x['score'])
            
            fig = plt.figure(figsize=(16, 9))
            gs = GridSpec(2, 2, height_ratios=[1.2, 1.0], figure=fig, hspace=0.35, wspace=0.25)
            ax_base = fig.add_subplot(gs[0, 0])
            ax_family = fig.add_subplot(gs[0, 1])
            title_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

            rec = best_entry['rec']
            pair = rec.get('pair')
            if pair:
                est = np.atleast_1d(np.squeeze(pair[0])).astype(float)
                gt = np.atleast_1d(np.squeeze(pair[1])).astype(float)
                t = rec.get('times_est', np.arange(len(est)))
                
                def _nm(x): 
                    mn, mx = np.nanmin(x), np.nanmax(x)
                    return (x - mn) / (mx - mn + 1e-9) * 2 - 1
                
                ax_base.plot(t, _nm(gt), color="#000000", linestyle="--", linewidth=2.0, alpha=0.8, label="GT")
                ax_base.plot(t, _nm(est), color="#1b9e77", linewidth=1.4, label=f"Best: {best_entry['method']}")
                ax_base.set_title(f"Best Trial Signal ({best_entry['rec']['video']})")
                ax_base.legend()

            trial_path = results_root / rec['data_file']
            if trial_path.exists():
                with open(trial_path, 'rb') as tf:
                    trial_data = pickle.load(tf)
                
                fps = trial_data.get('fps', 30.0)
                gt_sig = trial_data.get('gt', [])
                if len(gt_sig) > 0:
                    ax_family.plot(np.arange(len(gt_sig))/fps, _nm(gt_sig), color="#000000", linestyle="--", label="GT")
                
                for est_data in trial_data.get('estimates', []):
                    m_name = est_data.get('method')
                    color = None
                    for suffix, c in METHOD_COLORS.items():
                        if m_name.endswith(suffix):
                            color = c
                            break
                    
                    is_in_family = any(m_name == fm for fm in methods)
                    if is_in_family:
                        est_sig = est_data.get('estimate', est_data)
                        if isinstance(est_sig, dict): est_sig = est_sig.get('signal_hat')
                        if est_sig is not None:
                            ax_family.plot(np.arange(len(est_sig))/fps, _nm(est_sig), label=m_name, color=color, alpha=0.7)
                
                ax_family.set_title(f"Family Comparison: {family}")
                ax_family.legend(fontsize=8)

            fig.savefig(output_dir / f"{family}.png", dpi=200)
            plt.close(fig)
    except Exception as e:
        print(f"      !! Family overlays failed: {e}")

def _save_best_overlays_aligned(run_dir, metrics_pkl, stride):
    """
    Produces a Standalone plot of the BEST trial's estimated vs ground-truth WAVEFORMS.
    Loads the original pkl to get full-resolution signals, filters GT, and aligns via cross-correlation.
    """
    import matplotlib.pyplot as plt
    try:
        with open(metrics_pkl, 'rb') as f:
            headers, method_metrics = pickle.load(f)
        
        plot_dir = os.path.join(run_dir, 'plots')
        data_dir = os.path.join(run_dir, 'data')
        
        def _minmax(x):
            lo, hi = np.nanmin(x), np.nanmax(x)
            if hi - lo < 1e-9: return np.zeros_like(x)
            return (x - lo) / (hi - lo) * 2 - 1

        for method, recs in method_metrics.items():
            if not recs: continue
            # Find trial with HIGHEST CCC (best waveform correlation)
            # rec['metrics'][0] is CCC
            best_rec = max(recs, key=lambda r: r['metrics'][0])
            fname = best_rec['video']
            pkl_path = os.path.join(data_dir, f"{fname}.pkl")
            
            if not os.path.exists(pkl_path):
                continue
                
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract raw signals
            fps = data.get('fps', 30.0)
            gt_raw = data.get('gt')
            estimates = data.get('estimates', [])
            target_est = next((e for e in estimates if e.get('method') == method), None)
            
            if gt_raw is None or target_est is None:
                continue
            
            payload = target_est.get('estimate', target_est)
            est_wave = payload.get('signal_hat')
            if est_wave is None:
                continue
            
            # Ensure 1D and float
            est_wave = np.atleast_1d(np.squeeze(est_wave)).astype(np.float64)
            gt_raw = np.atleast_1d(np.squeeze(gt_raw)).astype(np.float64)
                
            # Filter GT to respiratory band using correct GT sampling rate
            f_lo, f_hi = 0.08, 0.5
            fs_gt = data.get('fs_gt', fps)
            gt_filt = np.squeeze(filter_RW(gt_raw, fs_gt, lo=f_lo, hi=f_hi))
            
            # Filter estimate signal similarly before alignment to remove broadband noise
            est_filt = np.squeeze(filter_RW(est_wave, fps, lo=f_lo, hi=f_hi))

            # Cross-correlation based alignment with frequency-aware resampling
            from core.evaluation.metrics import calculate_cross_corr_alignment
            plot_est, plot_gt, lag_sec = calculate_cross_corr_alignment(est_filt, gt_filt, fs_est=fps, fs_gt=fs_gt)
            
            if plot_est.size < 10: continue # Too short to plot
            
            # Time axis based on fs_gt (since signals are aligned/resampled to fs_gt)
            plot_t = np.arange(len(plot_est)) / fs_gt

            plt.figure(figsize=(12, 5))
            plt.plot(plot_t, _minmax(plot_gt), label='Ground Truth (filtered)', color='gray', linestyle='--', alpha=0.6)
            plt.plot(plot_t, _minmax(plot_est), label=f'Estimated (lag={lag_sec:.3f}s)', color='blue', alpha=0.8)
            plt.title(f"Best Waveform Alignment: {method}\nTrial: {fname}")
            plt.xlabel("Time (s) [relative]")
            plt.ylabel("Normalized Amplitude")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_name = f"overlay_best_waveform_aligned_{method.replace(' ', '_')}.png"
            plt.savefig(os.path.join(plot_dir, save_name), dpi=200)
            plt.close()
    except Exception as e:
        print(f"      !! Best waveform overlays failed: {e}")

def _generate_bmw_trace_plots(d_dir, csv_path, plot_dir, min_hz, max_hz, win_size, stride, summary_plot_dir=None):
    if not os.path.exists(csv_path): return
    import shutil
    df = pd.read_csv(csv_path)
    if 'MAE' not in df.columns: return
    data_dir = os.path.join(d_dir, 'data')
    if 'Method' in df.columns:
        methods = df['Method'].unique()
        method_col = 'Method'
    elif 'method' in df.columns:
        methods = df['method'].unique()
        method_col = 'method'
    else:
        return
        
    sorted_methods = sorted(methods, key=_method_sort_key)
    for m in sorted_methods:
        m_df = df[df[method_col] == m].sort_values(by='MAE').reset_index(drop=True)
        if m_df.empty: continue
        n = len(m_df)
        indices = {'best': 0, 'median': n // 2, 'worst': n - 1}
        m_clean = m.replace(' ', '_')
        m_plot_dir = os.path.join(plot_dir, m_clean)
        os.makedirs(m_plot_dir, exist_ok=True)
        for label, idx in indices.items():
            row = m_df.iloc[idx]
            fname = row['video']
            pkl_path = os.path.join(data_dir, fname + ".pkl")
            if not os.path.exists(pkl_path): continue
            try:
                with open(pkl_path, 'rb') as f: data = pickle.load(f)
                save_path = os.path.join(m_plot_dir, f"{label}_{fname}.png")
                _plot_single_trace(data, m, label, fname, row['MAE'], save_path, min_hz, max_hz, win_size, stride)
                
                # Aggregate 'best' to summary folder
                if label == 'best' and summary_plot_dir:
                    shutil.copy(save_path, os.path.join(summary_plot_dir, f"trace_best_{m_clean}_{fname}.png"))
            except Exception as e: 
                print(f"      !! Failed trace plot for {m} {label}: {e}")
                continue

def _plot_single_trace(data, method_name, label, fname, mae_val, save_path, min_hz, max_hz, win_size, stride):
    estimates, gt_signal = data.get('estimates', []), data.get('gt')
    fps, fs_gt = data.get('fps', 30.0), data.get('fs_gt', data.get('fps', 30.0))
    if not estimates or gt_signal is None: return
    filt_gt = filter_RW(gt_signal, fs_gt, lo=min_hz, hi=max_hz)
    gt_win, t_gt = sig_windowing(filt_gt, fs_gt, win_size, stride=stride)
    gt_rpm_series = sig_to_RPM(gt_win, fs_gt, int(win_size/1.5), min_hz, max_hz).reshape(-1)
    target_est = next((e for e in estimates if e.get('method') == method_name), None)
    if not target_est: return
    payload = target_est.get('estimate', target_est)
    sig_hat = payload.get('signal_hat')
    bpm_est = np.asarray(payload.get('track_hz', []), dtype=np.float64) * 60.0
    
    if sig_hat is None: return
    
    pred_times = payload.get('times_hz')
    if pred_times is None or (isinstance(pred_times, np.ndarray) and pred_times.size == 0):
        pred_times = np.arange(sig_hat.size) / fps
    pred_times = np.asarray(pred_times, dtype=np.float64)
    # If no BPM track (Base Models), we only plot the waveform part
    if bpm_est.size == 0:
        bpm_est = None
        bpm_gt_interp = None
    else:
        bpm_gt_interp = np.interp(pred_times, t_gt[:len(gt_rpm_series)], gt_rpm_series) if gt_rpm_series.size > 0 else None
    def _zn(x): return (x - np.mean(x)) / (np.std(x) + 1e-6)
    sig_hat_norm, gt_sig_norm = _zn(sig_hat), _zn(filt_gt)
    if len(gt_sig_norm.shape) > 1: gt_sig_norm = gt_sig_norm[0]
    gt_sig_resampled = np.interp(pred_times, np.arange(len(gt_sig_norm))/fs_gt, gt_sig_norm)
    plot_trace_paper(times=pred_times, sig_est=sig_hat_norm, bpm_est=bpm_est, sig_gt=gt_sig_resampled, bpm_gt=bpm_gt_interp, title=f"Sample: {label} ({fname})\nMethod: {method_name} | MAE: {mae_val:.2f} BPM", save_path=save_path)

def _method_sort_key(method_name: str):
    """Hierarchical sort key for grouping methods: Base < KFstd < UKFfreq < AGAKF"""
    name = str(method_name).lower().replace(' ', '_')
    if 'dof' in name: family = 10
    elif 'of_farneback' in name or 'of_model' in name: family = 20
    elif 'profile1d_linear' in name: family = 30
    elif 'profile1d_quadratic' in name: family = 40
    elif 'profile1d_cubic' in name: family = 50
    else: family = 99

    if '__kfstd' in name: sub = 2
    elif '__ukffreq' in name: sub = 3
    elif '__agakf' in name: sub = 4
    else: sub = 1
    return (family, sub, name)
