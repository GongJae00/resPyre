import os
import glob
import pickle
import json
import pandas as pd
import numpy as np
from core.utils.common import tqdm
from scipy import signal as sps
from core.evaluation.plotting_paper import plot_summary_mae_boxplot, plot_summary_scatter_gt, plot_trace_paper
from core.utils.common import filter_RW, sig_windowing, sig_to_RPM
from core.pipeline.common import (
    resolve_target_run_dirs,
    collect_expected_method_trials,
    resolve_frame_logs_for_run,
)


def _write_alignment_guide(plot_dir: str):
    """Write a small guide clarifying which plots are lag-aligned."""
    path = os.path.join(plot_dir, "alignment_guide.txt")
    lines = [
        "QROBF Visualization Alignment Guide",
        "",
        "[Aligned (lag-corrected)]",
        "- overlay_best_waveform_aligned_*.png",
        "- trace_*_aligned_*.png",
        "- family_overlays/* left panel (Best aligned pair from metrics pkl)",
        "",
        "[Raw / not lag-corrected]",
        "- trace_*_*.png (without _aligned suffix)",
        "- qrobf_diagnostics/*.png (state/trust/failure timelines with event markers)",
        "- family_overlays/* right panel (raw family waveform overlays)",
    ]
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def _build_scatter_df(run_dir: str, time_df: pd.DataFrame) -> pd.DataFrame:
    """Build scatter dataframe with gt_bpm/est_bpm columns.

    Priority:
      1) Already present in time-domain raw metrics
      2) Fallback to freq-domain raw metrics columns (gt_bpm_avg/est_bpm_avg)
    """
    def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        out["gt_bpm"] = pd.to_numeric(out["gt_bpm"], errors="coerce")
        out["est_bpm"] = pd.to_numeric(out["est_bpm"], errors="coerce")
        out = out.dropna(subset=["gt_bpm", "est_bpm", "method"])
        return out

    if {'gt_bpm', 'est_bpm', 'method'}.issubset(time_df.columns):
        cand = _sanitize(time_df[['gt_bpm', 'est_bpm', 'method']].copy())
        if not cand.empty:
            return cand
    if {'gt_bpm_avg', 'est_bpm_avg', 'method'}.issubset(time_df.columns):
        out = time_df[['gt_bpm_avg', 'est_bpm_avg', 'method']].copy()
        out = out.rename(columns={'gt_bpm_avg': 'gt_bpm', 'est_bpm_avg': 'est_bpm'})
        out = _sanitize(out)
        if not out.empty:
            return out

    freq_csv = os.path.join(run_dir, 'metrics', 'metrics_freq_domain_raw.csv')
    if os.path.exists(freq_csv):
        try:
            fdf = pd.read_csv(freq_csv)
            if {'gt_bpm_avg', 'est_bpm_avg', 'method'}.issubset(fdf.columns):
                out = fdf[['gt_bpm_avg', 'est_bpm_avg', 'method']].copy()
                out = out.rename(columns={'gt_bpm_avg': 'gt_bpm', 'est_bpm_avg': 'est_bpm'})
                out = _sanitize(out)
                if not out.empty:
                    return out
        except Exception:
            pass
    return pd.DataFrame()


def run_visualization(
    results_dir: str,
    run_label: str = None,
    win_size: float = 30.0,
    stride: float = 1.0,
    min_hz: float = 0.08,
    max_hz: float = 0.5,
    frame_log_strict: bool = True,
):
    """
    Generates summary plots (PNG) from aggregated metrics and sample traces.
    Restores family-wise overlays and best-sample aligned overlays.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    target_dirs = resolve_target_run_dirs(results_dir, run_label)

    if not target_dirs:
        print(f"> Visualization: No result directories found matching '{run_label}'")
        return

    print(f"\n> Starting Visualization for {len(target_dirs)} dataset(s)...")

    for d_dir in target_dirs:
        dataset_name = os.path.basename(d_dir)
        csv_path = os.path.join(d_dir, 'metrics', 'metrics_time_domain_raw.csv')
        if not os.path.exists(csv_path):
            # Backward compatibility for the older metrics filename.
            csv_path = os.path.join(d_dir, 'metrics', 'metrics_raw.csv')
        plot_dir = os.path.join(d_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        _write_alignment_guide(plot_dir)
        metrics_pkl = os.path.join(d_dir, 'metrics', 'metrics_time_domain.pkl')
        
        # 1. Generate Summary Plots (PNG)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'MAE' in df.columns:
                    print(f"   >> Generating summary plots for {dataset_name}...")
                    
                    boxplot_path = os.path.join(plot_dir, "summary_mae_boxplot.png")
                    
                    plot_summary_mae_boxplot(df, boxplot_path, title=f"Error Distribution: {dataset_name}")
                    print(f"      [Saved] {boxplot_path}")

                    scatter_df = _build_scatter_df(d_dir, df)
                    if not scatter_df.empty and {'gt_bpm', 'est_bpm', 'method'}.issubset(scatter_df.columns):
                        scatter_path = os.path.join(plot_dir, "summary_scatter_gt.png")
                        plot_summary_scatter_gt(scatter_df, scatter_path, title=f"Prediction vs Ground Truth: {dataset_name}")
                        print(f"      [Saved] {scatter_path}")
                    else:
                        print("      [Skip] summary_scatter_gt (missing gt/est bpm columns in time+freq metrics)")
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

        # 5. Diagnostics overlays and summary figures for robust/filter behavior.
        print("   >> Generating filter diagnostics overview plots...")
        _save_filter_diagnostics_overview(d_dir)
        print("   >> Generating filter diagnostics heatmap...")
        _save_filter_diag_heatmap(d_dir)
        print("   >> Generating latency distribution plots...")
        _save_latency_distribution(d_dir)
        print("   >> Generating trust/failure timeline overlays...")
        _save_trust_failure_overlays(d_dir, frame_log_strict=bool(frame_log_strict))

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
        roots = ["dof", "dof_disp_bridge", "of_farneback", "of_model", "of_disp_bridge", "profile1d_linear", "profile1d_quadratic", "profile1d_cubic"]
        
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
                ax_base.set_title(f"Best Trial Signal (Aligned) ({best_entry['rec']['video']})")
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
                
                ax_family.set_title(f"Family Comparison (Raw, no lag correction): {family_name}")
                ax_family.legend(fontsize=8)

            safe_family = str(family_name).replace(' ', '_')
            fig.savefig(output_dir / f"{safe_family}.png", dpi=200)
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
            plt.title(f"Best Waveform Alignment (Lag-corrected): {method}\nTrial: {fname}")
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
                _plot_single_trace(data, m, label, fname, row['MAE'], save_path, min_hz, max_hz, win_size, stride, aligned=False)
                save_path_aligned = os.path.join(m_plot_dir, f"{label}_aligned_{fname}.png")
                _plot_single_trace(data, m, label, fname, row['MAE'], save_path_aligned, min_hz, max_hz, win_size, stride, aligned=True)
                
                # Aggregate 'best' to summary folder
                if label == 'best' and summary_plot_dir:
                    shutil.copy(save_path, os.path.join(summary_plot_dir, f"trace_best_raw_{m_clean}_{fname}.png"))
                    shutil.copy(save_path_aligned, os.path.join(summary_plot_dir, f"trace_best_aligned_{m_clean}_{fname}.png"))
            except Exception as e: 
                print(f"      !! Failed trace plot for {m} {label}: {e}")
                continue

def _plot_single_trace(data, method_name, label, fname, mae_val, save_path, min_hz, max_hz, win_size, stride, aligned=False):
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
    def _zn(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return (x - np.mean(x)) / (np.std(x) + 1e-6)

    sig_hat_norm = _zn(sig_hat)
    gt_sig_norm = _zn(filt_gt)
    if len(gt_sig_norm.shape) > 1:
        gt_sig_norm = gt_sig_norm[0]

    if not aligned:
        # If no BPM track (Base Models), we only plot the waveform part
        if bpm_est.size == 0:
            bpm_est_plot = None
            bpm_gt_interp = None
        else:
            bpm_est_plot = bpm_est
            bpm_gt_interp = np.interp(pred_times, t_gt[:len(gt_rpm_series)], gt_rpm_series) if gt_rpm_series.size > 0 else None
        gt_sig_plot = np.interp(pred_times, np.arange(len(gt_sig_norm))/fs_gt, gt_sig_norm)
        title = (
            f"Sample: {label} ({fname}) [Raw, no lag correction]\n"
            f"Method: {method_name} | MAE: {mae_val:.2f} BPM"
        )
        plot_trace_paper(
            times=pred_times, sig_est=sig_hat_norm, bpm_est=bpm_est_plot,
            sig_gt=gt_sig_plot, bpm_gt=bpm_gt_interp, title=title, save_path=save_path
        )
        return

    # Aligned waveform variant (lag-corrected)
    from core.evaluation.metrics import calculate_cross_corr_alignment
    est_filt = np.squeeze(filter_RW(sig_hat, fps, lo=min_hz, hi=max_hz))
    plot_est, plot_gt, lag_sec = calculate_cross_corr_alignment(est_filt, filt_gt, fs_est=fps, fs_gt=fs_gt)
    if plot_est.size < 10:
        return
    times_aligned = np.arange(len(plot_est), dtype=np.float64) / fs_gt
    est_plot = _zn(plot_est)
    gt_plot = _zn(plot_gt)
    title = (
        f"Sample: {label} ({fname}) [Aligned, lag={lag_sec:.3f}s]\n"
        f"Method: {method_name} | MAE: {mae_val:.2f} BPM"
    )
    # Keep BPM panel disabled for aligned variant to avoid misinterpreting
    # track values after lag-resampling transform.
    plot_trace_paper(
        times=times_aligned, sig_est=est_plot, bpm_est=None,
        sig_gt=gt_plot, bpm_gt=None, title=title, save_path=save_path
    )


def _save_filter_diagnostics_overview(run_dir: str):
    import matplotlib.pyplot as plt

    csv_path = os.path.join(run_dir, 'metrics', 'metrics_filter_diagnostics_raw.csv')
    if not os.path.exists(csv_path):
        print("      [Skip] filter diagnostics csv not found")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"      !! Failed to load filter diagnostics csv: {e}")
        return

    required = {'method', 'Coverage95', 'Fail_Total', 'NIS_Mean'}
    if not required.issubset(df.columns):
        print("      [Skip] diagnostics csv missing required columns")
        return

    out_path = os.path.join(run_dir, 'plots', 'filter_diagnostics_overview.png')
    methods_all = sorted(df['method'].dropna().unique(), key=_method_sort_key)

    def _safe_nanmedian(series: pd.Series) -> float:
        arr = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float('nan')
        return float(np.median(finite))

    cov_all = [_safe_nanmedian(df.loc[df['method'] == m, 'Coverage95']) for m in methods_all]
    fail_all = [_safe_nanmedian(df.loc[df['method'] == m, 'Fail_Total']) for m in methods_all]
    nis_all = [_safe_nanmedian(df.loc[df['method'] == m, 'NIS_Mean']) for m in methods_all]

    applicable = []
    for m, c, f, n in zip(methods_all, cov_all, fail_all, nis_all):
        if np.isfinite(c) or np.isfinite(f) or np.isfinite(n):
            applicable.append((m, c, f, n))
    if not applicable:
        print("      [Skip] no applicable diagnostics rows")
        return

    methods = [x[0] for x in applicable]
    cov = [x[1] for x in applicable]
    fail = [x[2] for x in applicable]
    nis = [x[3] for x in applicable]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].bar(x, cov, color="#2a9d8f")
    axes[0].set_ylabel("Coverage95 (%)")
    axes[0].set_title("Filter Diagnostics Summary")
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x, fail, color="#e76f51")
    axes[1].set_ylabel("Failure Rate")
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].bar(x, nis, color="#264653")
    axes[2].set_ylabel("NIS Mean")
    axes[2].axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"      [Saved] {out_path}")


def _save_filter_diag_heatmap(run_dir: str):
    import matplotlib.pyplot as plt

    csv_path = os.path.join(run_dir, 'metrics', 'metrics_filter_diagnostics_summary.csv')
    if not os.path.exists(csv_path):
        print("      [Skip] filter diagnostics summary csv not found")
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"      !! Failed to load diagnostics summary csv: {e}")
        return

    key_map = {
        'Fail_Total_median': 'Fail_Total',
        'NIS_Mean_median': 'NIS_Mean',
        'NIS_OverStrict_median': 'NIS_OverStrict',
        'NIS_TrueFail_median': 'NIS_TrueFail',
        'Coverage95_median': 'Coverage95',
        'Lambda_LT1_Frac_median': 'Lambda_LT1_Frac',
    }
    cols = [c for c in key_map.keys() if c in df.columns]
    if not cols or 'Method' not in df.columns:
        print("      [Skip] diagnostics summary missing required columns")
        return

    mat = []
    methods = []
    for _, row in df.iterrows():
        vals = [pd.to_numeric(row[c], errors='coerce') for c in cols]
        if not np.isfinite(np.asarray(vals, dtype=np.float64)).any():
            continue
        methods.append(str(row['Method']))
        mat.append(vals)
    if not mat:
        print("      [Skip] no finite diagnostics summary values")
        return

    arr = np.asarray(mat, dtype=np.float64)
    # robust per-column normalization for comparability
    norm = np.zeros_like(arr)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        lo = np.nanpercentile(col[finite], 10)
        hi = np.nanpercentile(col[finite], 90)
        if hi - lo < 1e-9:
            norm[finite, j] = 0.5
        else:
            norm[finite, j] = np.clip((col[finite] - lo) / (hi - lo), 0.0, 1.0)

    out_path = os.path.join(run_dir, 'plots', 'filter_diagnostics_heatmap.png')
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(methods))))
    im = ax.imshow(norm, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([key_map[c] for c in cols], rotation=30, ha='right')
    ax.set_title('Filter Diagnostics Heatmap (normalized)')
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Normalized score')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"      [Saved] {out_path}")


def _save_latency_distribution(run_dir: str):
    import matplotlib.pyplot as plt

    csv_path = os.path.join(run_dir, 'metrics', 'metrics_time_domain_raw.csv')
    if not os.path.exists(csv_path):
        print("      [Skip] time-domain raw csv not found")
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"      !! Failed to load time-domain raw csv: {e}")
        return
    if 'Latency' not in df.columns or 'method' not in df.columns:
        print("      [Skip] latency columns missing")
        return

    out = df[['method', 'Latency']].copy()
    out['Latency'] = pd.to_numeric(out['Latency'], errors='coerce')
    out = out.dropna(subset=['Latency', 'method'])
    if out.empty:
        print("      [Skip] no finite latency rows")
        return

    methods = sorted(out['method'].unique(), key=_method_sort_key)
    out_path = os.path.join(run_dir, 'plots', 'latency_distribution.png')
    fig, ax = plt.subplots(figsize=(12, 6))
    data = [out.loc[out['method'] == m, 'Latency'].to_numpy(dtype=np.float64) for m in methods]
    # Matplotlib compatibility:
    # - newer versions accept `tick_labels`
    # - older versions use `labels`
    try:
        ax.boxplot(data, tick_labels=methods, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=methods, showfliers=True)
    ax.set_title('Latency Distribution by Method')
    ax.set_ylabel('Latency (ms)')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"      [Saved] {out_path}")


def _event_indices(x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(arr) & (arr > threshold))[0]


def _contiguous_runs(mask: np.ndarray):
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, cuts)
    return [(int(g[0]), int(g[-1])) for g in groups if g.size > 0]


def _mark_events_on_axes(axes, t: np.ndarray, event_idx: np.ndarray, color: str, max_lines: int = 160):
    if event_idx.size == 0:
        return
    idx = event_idx
    if idx.size > max_lines:
        pick = np.linspace(0, idx.size - 1, num=max_lines, dtype=int)
        idx = idx[pick]
    x_vals = t[idx]
    for ax in axes:
        for xv in x_vals:
            ax.axvline(xv, color=color, alpha=0.16, linewidth=0.8)


def _write_empty_qrobf_artifacts(out_dir: str, resolver_diag: dict, orphans: list):
    import matplotlib.pyplot as plt

    summary_cols = [
        "method", "n_trials", "n_frames", "div", "slip", "lock", "double",
        "gate_collapse_runs", "harm_suppress_runs",
        "selection_policy", "run_instance_started_at_used", "n_orphan_logs_ignored",
    ]
    rates_cols = summary_cols + [
        "div_rate", "slip_rate", "lock_rate", "double_rate",
        "gate_collapse_rate", "harm_suppress_rate",
    ]
    summary_csv = os.path.join(out_dir, "qrobf_event_summary.csv")
    rates_csv = os.path.join(out_dir, "qrobf_event_summary_rates.csv")
    summary_json = os.path.join(out_dir, "qrobf_event_summary.json")
    summary_png = os.path.join(out_dir, "qrobf_event_summary.png")

    pd.DataFrame(columns=summary_cols).to_csv(summary_csv, index=False)
    pd.DataFrame(columns=rates_cols).to_csv(rates_csv, index=False)

    payload = {
        "schema_version": "qrobf_event_summary.v1",
        "status": "empty",
        "reason": "no_canonical_logs",
        "n_trials": 0,
        "n_events": 0,
        "selection_policy": str((resolver_diag or {}).get("selection_policy", "")),
        "run_instance_started_at_used": str((resolver_diag or {}).get("run_instance_started_at_used", "")),
        "n_orphan_logs_ignored": int(len(orphans or [])),
        "resolver_diag": resolver_diag or {},
    }
    with open(summary_json, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, "No canonical logs", ha="center", va="center", fontsize=16)
    ax.set_title("QROBF Event Summary")
    plt.tight_layout()
    plt.savefig(summary_png, dpi=180)
    plt.close(fig)
    print(f"      [Saved] {summary_png}")


def _save_trust_failure_overlays(run_dir: str, frame_log_strict: bool = True):
    import matplotlib.pyplot as plt

    out_dir = os.path.join(run_dir, 'plots', 'qrobf_diagnostics')
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []

    expected_all = collect_expected_method_trials(run_dir)
    expected = []
    for item in expected_all:
        method = str(item.get("method", "")).strip()
        trial = str(item.get("trial", "")).strip()
        if not method or not trial:
            continue
        log_dir = os.path.join(run_dir, "aux", method.replace(" ", "_"), "frame_logs")
        if os.path.isdir(log_dir):
            expected.append({"method": method, "trial": trial})
    resolution = resolve_frame_logs_for_run(
        run_dir,
        expected_trials=expected,
        strict=bool(frame_log_strict),
        allow_empty=True,
    )
    selected = resolution.get("canonical", {})
    orphans = resolution.get("extras", [])
    resolver_diag = resolution.get("diag", {}) if isinstance(resolution, dict) else {}
    method_to_logs = {}
    for method, trial_map in sorted(selected.items()):
        method_to_logs[method] = [trial_map[k] for k in sorted(trial_map.keys())]

    orphan_csv = os.path.join(out_dir, "qrobf_orphan_logs.csv")
    if orphans:
        pd.DataFrame(orphans).to_csv(orphan_csv, index=False)
    else:
        pd.DataFrame(columns=["method", "trial_key", "path", "filename", "suffix", "mtime", "reason"]).to_csv(orphan_csv, index=False)

    if not method_to_logs:
        print("      [Info] no canonical frame logs found; writing placeholder qrobf summary artifacts")
        _write_empty_qrobf_artifacts(out_dir, resolver_diag=resolver_diag, orphans=orphans)
        return

    for method, method_paths in method_to_logs.items():
        try:
            rep = None
            agg_counts = {
                "div": 0,
                "slip": 0,
                "lock": 0,
                "double": 0,
                "gate_collapse_runs": 0,
                "harm_suppress_runs": 0,
            }
            n_trials = 0
            n_frames_total = 0
            event_rows = []
            for path in method_paths:
                z = np.load(path, allow_pickle=True)
                fields = list(z['fields'])
                data = z['data']
                idx = {f: i for i, f in enumerate(fields)}

                def _col(name: str, default=np.nan):
                    if name not in idx:
                        return np.full(data.shape[0], default, dtype=np.float64)
                    return np.asarray(data[:, idx[name]], dtype=np.float64)

                t = _col('t')
                if not np.isfinite(t).any():
                    t = np.arange(data.shape[0], dtype=np.float64)

                y_t = _col('y_t')
                y_pred = _col('y_pred')
                q_vis = _col('q_vis')
                q_drift = _col('q_drift')
                q_harm = _col('q_harm')
                alpha_R = _col('alpha_R')
                g_t = _col('g_t')
                g_z = _col('g_z')
                w_h = _col('w_h')
                g_z_eff = g_z * w_h
                nis = _col('nis')
                fail_div = _col('fail_diverge', default=0.0)
                fail_slip = _col('fail_slip', default=0.0)
                fail_lock = _col('fail_lock', default=0.0)
                fail_double = _col('fail_double', default=0.0)

                idx_div = _event_indices(fail_div)
                idx_slip = _event_indices(fail_slip)
                idx_lock = _event_indices(fail_lock)
                idx_double = _event_indices(fail_double)
                gate_collapse_runs = _contiguous_runs(np.isfinite(g_t) & (g_t < 0.20))
                harm_suppress_runs = _contiguous_runs(np.isfinite(g_z_eff) & (g_z_eff < 0.20))

                if rep is None:
                    rep = {
                        "t": t,
                        "y_t": y_t,
                        "y_pred": y_pred,
                        "q_vis": q_vis,
                        "q_drift": q_drift,
                        "q_harm": q_harm,
                        "alpha_R": alpha_R,
                        "g_t": g_t,
                        "g_z_eff": g_z_eff,
                        "nis": nis,
                        "fail_div": fail_div,
                        "fail_slip": fail_slip,
                        "fail_lock": fail_lock,
                        "fail_double": fail_double,
                        "idx_div": idx_div,
                        "idx_slip": idx_slip,
                        "idx_lock": idx_lock,
                        "idx_double": idx_double,
                    }

                trial = os.path.splitext(os.path.basename(path))[0]
                n_trials += 1
                n_frames_total += int(data.shape[0])
                agg_counts["div"] += int(idx_div.size)
                agg_counts["slip"] += int(idx_slip.size)
                agg_counts["lock"] += int(idx_lock.size)
                agg_counts["double"] += int(idx_double.size)
                agg_counts["gate_collapse_runs"] += int(len(gate_collapse_runs))
                agg_counts["harm_suppress_runs"] += int(len(harm_suppress_runs))

                for label, idx_arr in (
                    ("fail_diverge", idx_div),
                    ("fail_slip", idx_slip),
                    ("fail_lock", idx_lock),
                    ("fail_double", idx_double),
                ):
                    for i in idx_arr.tolist():
                        event_rows.append({
                            "method": method,
                            "trial": trial,
                            "event": label,
                            "frame": int(i),
                            "time": float(t[i]),
                            "nis": float(nis[i]) if np.isfinite(nis[i]) else np.nan,
                            "g_t": float(g_t[i]) if np.isfinite(g_t[i]) else np.nan,
                            "g_z_eff": float(g_z_eff[i]) if np.isfinite(g_z_eff[i]) else np.nan,
                            "q_harm": float(q_harm[i]) if np.isfinite(q_harm[i]) else np.nan,
                        })

            if rep is None:
                continue

            fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
            axes[0].plot(rep["t"], rep["y_t"], label='y_t', color='#1d3557', alpha=0.9)
            axes[0].plot(rep["t"], rep["y_pred"], label='y_pred', color='#457b9d', alpha=0.8)
            axes[0].legend(loc='upper right', ncol=2)
            axes[0].set_ylabel('Signal')

            axes[1].plot(rep["t"], rep["q_vis"], label='q_vis', color='#2a9d8f')
            axes[1].plot(rep["t"], rep["q_drift"], label='q_drift', color='#8ab17d')
            axes[1].plot(rep["t"], rep["q_harm"], label='q_harm', color='#e9c46a')
            axes[1].set_ylabel('Quality')
            axes[1].legend(loc='upper right', ncol=3)

            axes[2].plot(rep["t"], rep["alpha_R"], label='alpha_R', color='#e76f51')
            axes[2].plot(rep["t"], rep["g_t"], label='g_t', color='#264653')
            axes[2].plot(rep["t"], rep["g_z_eff"], label='g_z_eff', color='#6d597a')
            axes[2].set_ylabel('Trust/Gates')
            axes[2].legend(loc='upper right', ncol=3)

            axes[3].plot(rep["t"], rep["nis"], label='NIS', color='#1d3557')
            axes[3].plot(rep["t"], rep["fail_div"], label='fail_div', color='#d00000', alpha=0.8)
            axes[3].plot(rep["t"], rep["fail_slip"], label='fail_slip', color='#f77f00', alpha=0.8)
            axes[3].plot(rep["t"], rep["fail_lock"], label='fail_lock', color='#fcbf49', alpha=0.8)
            axes[3].plot(rep["t"], rep["fail_double"], label='fail_double', color='#9d4edd', alpha=0.8)
            axes[3].set_ylabel('Failure/NIS')
            axes[3].set_xlabel('Frame')
            axes[3].legend(loc='upper right', ncol=5, fontsize=8)

            # Overlay failure event markers across all panels.
            _mark_events_on_axes(axes, rep["t"], rep["idx_div"], '#d00000')
            _mark_events_on_axes(axes, rep["t"], rep["idx_slip"], '#f77f00')
            _mark_events_on_axes(axes, rep["t"], rep["idx_lock"], '#fcbf49')
            _mark_events_on_axes(axes, rep["t"], rep["idx_double"], '#9d4edd')

            # Shade gate/harmonic suppression windows on trust panel.
            rep_gate_runs = _contiguous_runs(np.isfinite(rep["g_t"]) & (rep["g_t"] < 0.20))
            rep_harm_runs = _contiguous_runs(np.isfinite(rep["g_z_eff"]) & (rep["g_z_eff"] < 0.20))
            for s, e in rep_gate_runs[:80]:
                axes[2].axvspan(rep["t"][s], rep["t"][e], color='#264653', alpha=0.08)
            for s, e in rep_harm_runs[:80]:
                axes[2].axvspan(rep["t"][s], rep["t"][e], color='#6d597a', alpha=0.07)

            axes[3].text(
                0.995, 0.02,
                f"agg(div={agg_counts['div']} slip={agg_counts['slip']} lock={agg_counts['lock']} dbl={agg_counts['double']})",
                transform=axes[3].transAxes,
                ha='right', va='bottom', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'),
            )

            fig.suptitle(
                f"QROBF Diagnostics — {method} "
                f"(representative timeline; aggregated over {n_trials} trial(s))"
            )
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            out_path = os.path.join(out_dir, f"{method}.png")
            plt.savefig(out_path, dpi=220)
            plt.close(fig)

            # Save event summary artifacts for reproducible figure/table support.
            counts_path = os.path.join(out_dir, f"{method}_event_counts.csv")
            pd.DataFrame([{
                "method": method,
                "n_trials": int(n_trials),
                "n_frames": int(n_frames_total),
                **agg_counts,
            }]).to_csv(counts_path, index=False)
            summary_rows.append({
                "method": method,
                "n_trials": int(n_trials),
                "n_frames": int(n_frames_total),
                **agg_counts,
            })

            events_path = os.path.join(out_dir, f"{method}_events.csv")
            event_cols = ["method", "trial", "event", "frame", "time", "nis", "g_t", "g_z_eff", "q_harm"]
            pd.DataFrame(event_rows, columns=event_cols).to_csv(events_path, index=False)
        except Exception as e:
            print(f"      !! diagnostics overlay failed for {method}: {e}")
            continue

    # Cross-method summary for figure/table generation.
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty:
            summary_df = summary_df.sort_values(by="method", key=lambda s: s.map(_method_sort_key))
            summary_df["selection_policy"] = str(resolver_diag.get("selection_policy", ""))
            summary_df["run_instance_started_at_used"] = str(resolver_diag.get("run_instance_started_at_used", ""))
            summary_df["n_orphan_logs_ignored"] = int(len(orphans))
            summary_csv = os.path.join(out_dir, "qrobf_event_summary.csv")
            summary_df.to_csv(summary_csv, index=False)

        # Add rate columns for fair method comparison.
        denom = summary_df["n_frames"].replace(0, np.nan)
        for key in ("div", "slip", "lock", "double"):
            summary_df[f"{key}_rate"] = summary_df[key] / denom
        summary_df["gate_collapse_rate"] = summary_df["gate_collapse_runs"] / denom
        summary_df["harm_suppress_rate"] = summary_df["harm_suppress_runs"] / denom
        summary_rate_csv = os.path.join(out_dir, "qrobf_event_summary_rates.csv")
        summary_df.to_csv(summary_rate_csv, index=False)
        summary_json = os.path.join(out_dir, "qrobf_event_summary.json")
        with open(summary_json, "w", encoding="utf-8") as fp:
            json.dump({
                "schema_version": "qrobf_event_summary.v1",
                "status": "ok",
                "reason": "",
                "n_trials": int(summary_df["n_trials"].sum()),
                "n_events": int(summary_df[["div", "slip", "lock", "double"]].sum().sum()),
                "selection_policy": str(resolver_diag.get("selection_policy", "")),
                "run_instance_started_at_used": str(resolver_diag.get("run_instance_started_at_used", "")),
                "n_orphan_logs_ignored": int(len(orphans)),
                "resolver_diag": resolver_diag,
                "methods": summary_df.to_dict(orient="records"),
            }, fp, ensure_ascii=False, indent=2)

            # Compact visual summary (failure rates + gate/harm run rates).
            methods = summary_df["method"].tolist()
            x = np.arange(len(methods), dtype=np.float64)
            width = 0.18

            fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            axes[0].bar(x - 1.5 * width, summary_df["div_rate"], width, label="diverge", color="#d00000")
            axes[0].bar(x - 0.5 * width, summary_df["slip_rate"], width, label="phase_slip", color="#f77f00")
            axes[0].bar(x + 0.5 * width, summary_df["lock_rate"], width, label="locking", color="#fcbf49")
            axes[0].bar(x + 1.5 * width, summary_df["double_rate"], width, label="doubling", color="#9d4edd")
            axes[0].set_ylabel("Failure Rate per Frame")
            axes[0].set_title("QROBF Failure Event Summary")
            axes[0].legend(loc="upper right", ncol=4, fontsize=9)
            axes[0].grid(axis="y", alpha=0.3)

            axes[1].bar(x - 0.25 * width, summary_df["gate_collapse_rate"], width * 1.2, label="gate collapse", color="#264653")
            axes[1].bar(x + 0.25 * width, summary_df["harm_suppress_rate"], width * 1.2, label="harmonic suppression", color="#6d597a")
            axes[1].set_ylabel("Run Rate per Frame")
            axes[1].set_xlabel("Method")
            axes[1].legend(loc="upper right", ncol=2, fontsize=9)
            axes[1].grid(axis="y", alpha=0.3)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(methods, rotation=45, ha="right")

            plt.tight_layout()
            summary_png = os.path.join(out_dir, "qrobf_event_summary.png")
            plt.savefig(summary_png, dpi=220)
            plt.close(fig)
            print(f"      [Saved] {summary_png}")

            legend_lines = [
                "QROBF Diagnostics Event Legend",
                "",
                "[Failure Labels]",
                "- diverge (red): fail_diverge > 0.5",
                "- phase_slip (orange): fail_slip > 0.5",
                "- locking (yellow): fail_lock > 0.5",
                "- doubling (purple): fail_double > 0.5",
                "",
                "[Operational Windows]",
                "- gate collapse: g_t < 0.20 contiguous runs",
                "- harmonic suppression: g_z_eff < 0.20 contiguous runs",
                "",
                "[Output Files]",
                "- *_event_counts.csv : method-level aggregate counts across all trials",
                "- *_events.csv       : per-frame event rows across all trials",
                "- qrobf_event_summary.csv / _rates.csv / .png : aggregate summaries",
            ]
            legend_path = os.path.join(out_dir, "qrobf_event_legend.txt")
            with open(legend_path, "w", encoding="utf-8") as fp:
                fp.write("\n".join(legend_lines) + "\n")

def _method_sort_key(method_name: str):
    """Shared ordering with execution/evaluation: family + variant."""
    name = str(method_name).lower().replace(' ', '_')
    base = name.split('__', 1)[0] if '__' in name else name

    if base in ('of_model', 'of', 'of_farneback'):
        family = 10
    elif base in ('of_disp_bridge', 'of_displacement_bridge', 'of_bridge'):
        family = 15
    elif base in ('dof_disp_bridge', 'dof_bridge', 'dof_displacement_bridge'):
        family = 25
    elif base == 'dof':
        family = 20
    elif base.startswith('profile1d_linear'):
        family = 30
    elif base.startswith('profile1d_quadratic'):
        family = 40
    elif base.startswith('profile1d_cubic'):
        family = 50
    elif base.startswith('pair_of_p1d_quadratic'):
        family = 60
    elif base.startswith('assist_of_p1d_quadratic'):
        family = 68
    elif base.startswith('fusion_of_p1d_quadratic'):
        family = 70
    else:
        family = 99

    if '__' not in name:
        sub = 0
    else:
        head = name.split('__', 1)[1]
        if 'kfstd' in head:
            sub = 10
        elif 'robust_ossm_ekf' in head or head.endswith('_ekf') or head == 'robust_ossm':
            sub = 20
        elif 'robust_ossm_ukf' in head or 'ukffreq' in head or head.endswith('_ukf'):
            sub = 30
        elif 'agakf' in head:
            sub = 40
        else:
            sub = 90
    return (family, sub, name)
