import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def set_paper_style():
    """Sets a clean, scientific plotting style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })

def plot_summary_mae_boxplot(df: pd.DataFrame, save_path: str, title: str = "MAE Distribution"):
    """Generates a professional boxplot of MAE by method."""
    plt.figure(figsize=(12, 7))
    set_paper_style()
    
    # Sort methods hierarchically: Group by base, then Base < KFstd < UKFfreq
    methods = df['method'].unique()
    order = sorted(methods, key=_method_sort_key)
    
    ax = sns.boxplot(data=df, x='method', y='MAE', order=order, palette='viridis', hue='method', legend=False)
    sns.stripplot(data=df, x='method', y='MAE', order=order, color=".3", size=3, alpha=0.5, ax=ax)
    
    plt.title(title)
    plt.xlabel("Method")
    plt.ylabel("BPM Error (MAE)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(save_path)
    plt.close()

def plot_summary_scatter_gt(df: pd.DataFrame, save_path: str, title: str = "Est BPM vs Ground Truth"):
    """Generates a professional scatter plot of Est BPM vs GT BPM."""
    work = df.copy()
    work["gt_bpm"] = pd.to_numeric(work["gt_bpm"], errors="coerce")
    work["est_bpm"] = pd.to_numeric(work["est_bpm"], errors="coerce")
    work = work.dropna(subset=["gt_bpm", "est_bpm", "method"]).reset_index(drop=True)

    plt.figure(figsize=(10, 8))
    set_paper_style()

    if work.empty:
        plt.title(f"{title}\n(no valid points)")
        plt.xlabel("Ground Truth BPM")
        plt.ylabel("Estimated BPM")
        plt.grid(alpha=0.3)
        plt.savefig(save_path)
        plt.close()
        return

    hue_order = sorted(work['method'].unique(), key=_method_sort_key)

    # When points collapse (common in debug mode with one trial), add tiny deterministic
    # jitter so markers/labels remain readable while preserving trend.
    gt = work["gt_bpm"].to_numpy(dtype=np.float64)
    est = work["est_bpm"].to_numpy(dtype=np.float64)
    gt_span = float(np.nanmax(gt) - np.nanmin(gt)) if gt.size else 0.0
    est_span = float(np.nanmax(est) - np.nanmin(est)) if est.size else 0.0
    dup_ratio = 1.0 - (work[["gt_bpm", "est_bpm"]].drop_duplicates().shape[0] / max(len(work), 1))
    use_jitter = dup_ratio > 0.0 or gt_span < 1e-6 or est_span < 1e-6
    if use_jitter:
        x_j = max(gt_span * 0.02, 0.03)
        y_j = max(est_span * 0.02, 0.03)
        method_codes = pd.Categorical(work["method"], categories=hue_order).codes
        phase = method_codes * 1.61803398875 + np.arange(len(work), dtype=np.float64) * 0.73
        work["gt_bpm_plot"] = work["gt_bpm"] + x_j * np.sin(phase)
        work["est_bpm_plot"] = work["est_bpm"] + y_j * np.cos(phase)
    else:
        work["gt_bpm_plot"] = work["gt_bpm"]
        work["est_bpm_plot"] = work["est_bpm"]

    sns.scatterplot(
        data=work,
        x='gt_bpm_plot',
        y='est_bpm_plot',
        hue='method',
        hue_order=hue_order,
        alpha=0.8,
        s=62,
    )

    # Add identity line
    min_val = min(work['gt_bpm'].min(), work['est_bpm'].min())
    max_val = max(work['gt_bpm'].max(), work['est_bpm'].max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.8, label='Identity (GT)')

    if len(work) <= 40:
        for _, row in work.iterrows():
            lbl = str(row["method"]).split("__", 1)[-1][:16]
            plt.annotate(
                lbl,
                (row["gt_bpm_plot"], row["est_bpm_plot"]),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
                alpha=0.75,
            )

    title_suffix = " (jittered for visibility)" if use_jitter else ""
    plt.title(title + title_suffix)
    plt.xlabel("Ground Truth BPM")
    plt.ylabel("Estimated BPM")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.savefig(save_path)
    plt.close()

def plot_trace_paper(times, sig_est, bpm_est, sig_gt=None, bpm_gt=None, title="Signal Trace", save_path=None):
    """Generates a professional multi-panel trace plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    set_paper_style()
    
    # 1. Signal
    ax1.plot(times, sig_est, label='Estimated', color='blue', alpha=0.8)
    if sig_gt is not None:
        ax1.plot(times, sig_gt, label='Ground Truth', color='orange', linestyle='--', alpha=0.6)
    ax1.set_ylabel("Normalized Amplitude")
    ax1.set_title("Breathing Waveform")
    ax1.legend(loc='upper right')
    
    # 2. BPM Tracking
    if bpm_est is not None:
        ax2.plot(times, bpm_est, label='Estimated BPM', color='green', linewidth=2)
        if bpm_gt is not None:
            if np.isscalar(bpm_gt):
                ax2.axhline(bpm_gt, color='red', linestyle='--', label=f'GT BPM ({bpm_gt:.1f})', alpha=0.7)
            else:
                ax2.plot(times, bpm_gt, color='red', linestyle='--', label='GT BPM', alpha=0.7)
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, "No Real-time Tracking Available\n(Base Model / Raw Observation)", 
                 ha='center', va='center', transform=ax2.transAxes, fontsize=14, color='gray')
            
    ax2.set_ylabel("BPM")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Respiratory Rate Tracking")
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def _method_sort_key(method_name: str):
    """Hierarchical sort key for grouping methods."""
    name = str(method_name).lower().replace(' ', '_')
    if 'of_farneback' in name or 'of_model' in name: family = 10
    elif 'dof' in name: family = 20
    elif 'profile1d_linear' in name: family = 30
    elif 'profile1d_quadratic' in name: family = 40
    elif 'profile1d_cubic' in name: family = 50
    else: family = 99

    if '__kfstd' in name:
        sub = 2
    elif '__robust_ossm_ekf' in name or name.endswith('__ekf'):
        sub = 3
    elif '__robust_ossm_ukf' in name or '__ukffreq' in name or name.endswith('__ukf'):
        sub = 4
    elif '__agakf' in name:
        sub = 5
    else:
        sub = 1
    return (family, sub, name)
