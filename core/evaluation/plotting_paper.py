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
    plt.figure(figsize=(10, 8))
    set_paper_style()
    
    hue_order = sorted(df['method'].unique(), key=_method_sort_key)
    sns.scatterplot(data=df, x='gt_bpm', y='est_bpm', hue='method', hue_order=hue_order, alpha=0.7, s=50)
    
    # Add identity line
    min_val = min(df['gt_bpm'].min(), df['est_bpm'].min())
    max_val = max(df['gt_bpm'].max(), df['est_bpm'].max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', alpha=0.8, label='Identity (GT)')
    
    plt.title(title)
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
