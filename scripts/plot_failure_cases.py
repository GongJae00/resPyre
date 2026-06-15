#!/usr/bin/env python3
"""Plot main failure cases from a persistent manifest."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.figure_style import add_metric_box, save_figure, set_manuscript_style, style_axis
from core.evaluation.metrics import calculate_cross_corr_alignment


COLORS = {
    'GT': '#111111',
    'PARH': '#116b4f',
    'PARH_alt': '#7b4ea3',
    'osc': '#d17a3a',
    'q_osc': '#1a7f64',
    'nonosc': '#c85f2b',
}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()


def _zscore(sig):
    sig = np.asarray(sig, dtype=float).flatten()
    std = np.std(sig)
    if not np.isfinite(std) or std < 1e-9:
        return sig * 0.0
    return (sig - np.mean(sig)) / std


def _load_payload(run_dir: Path, rel_data_file: str):
    with open(run_dir / rel_data_file, 'rb') as f:
        return pickle.load(f)


def _load_estimate(payload: dict, method: str):
    for item in payload.get('estimates', []):
        if item.get('method') == method:
            return item.get('estimate', {})
    raise KeyError(method)


def _aligned_window(pred, gt, fs_est, fs_gt, window_sec=18.0):
    pred_a, gt_a, _ = calculate_cross_corr_alignment(pred, gt, fs_est=fs_est, fs_gt=fs_gt)
    pred_a = _zscore(pred_a)
    gt_a = _zscore(gt_a)
    common = min(len(pred_a), len(gt_a))
    if common == 0:
        return np.array([]), np.array([]), np.array([])
    pred_a = pred_a[:common]
    gt_a = gt_a[:common]
    fs = fs_gt
    win = min(common, max(200, int(round(window_sec * fs))))
    start = max((common - win) // 2, 0)
    end = start + win
    t = np.arange(win, dtype=float) / fs
    return t, pred_a[start:end], gt_a[start:end]


def _residual_case(ax_wave, ax_diag, row: pd.Series, run_dir: Path):
    payload = _load_payload(run_dir, str(row['data_file']))
    est = _load_estimate(payload, str(row['method']))
    fs = float(payload.get('fps', 20.0))
    fs_gt = float(payload.get('fs_gt', fs))
    gt = np.asarray(payload['gt'], dtype=float)
    z_full = np.asarray(est['z_full'], dtype=float)
    z_osc = np.asarray(est['z_osc'], dtype=float)
    t, zfull_w, gt_w = _aligned_window(z_full, gt, fs, fs_gt)
    t2, zosc_w, _ = _aligned_window(z_osc, gt, fs, fs_gt)

    ax_wave.plot(t, gt_w, color=COLORS['GT'], linewidth=2.0, label='GT')
    ax_wave.plot(t, zfull_w, color=COLORS['PARH'], linewidth=1.8, label=r'$z_{full}$')
    ax_wave.plot(t2, zosc_w, color=COLORS['osc'], linewidth=1.6, alpha=0.95, label=r'$z_{osc}$')
    ax_wave.set_title('Residual-heavy PARH case', fontsize=10.5, loc='left')
    add_metric_box(ax_wave, f"CCC {float(row['waveform_CCC']):.3f}\nrate MAE {float(row['rate_MAE']):.2f}", loc='upper left')
    ax_wave.set_ylabel('Aligned amplitude')
    style_axis(ax_wave, grid='y')
    ax_wave.set_ylim(-3.2, 3.2)

    diag = est['diagnostics']
    q_osc = np.asarray(diag['q_osc_t'], dtype=float)
    nonosc = np.asarray(diag['obs_nonosc_need_t'], dtype=float)
    tt = np.arange(len(q_osc), dtype=float) / fs
    ax_diag.plot(tt, q_osc, color=COLORS['q_osc'], linewidth=1.7, label=r'$q_{osc}$')
    ax_diag.plot(tt, nonosc, color=COLORS['nonosc'], linewidth=1.7, label='nonosc need')
    ax_diag.set_title('Residual diagnostics', fontsize=10.5, loc='left')
    add_metric_box(ax_diag, f"residual ratio {float(row['residual_energy_ratio']):.3f}\nnonosc mean {float(row['obs_nonosc_need_mean']):.3f}", loc='lower right')
    ax_diag.set_ylabel('Diagnostic value')
    ax_diag.set_ylim(0.0, 1.05)
    style_axis(ax_diag, grid='y')


def _bridge_case(ax_wave, ax_bar, row: pd.Series, run_dir: Path):
    payload = _load_payload(run_dir, str(row['data_file']))
    raw_est = _load_estimate(payload, 'of_farneback__parh_ossm')
    bridge_est = _load_estimate(payload, 'of_disp_bridge__parh_ossm')
    fs = float(payload.get('fps', 20.0))
    fs_gt = float(payload.get('fs_gt', fs))
    gt = np.asarray(payload['gt'], dtype=float)

    t, raw_w, gt_w = _aligned_window(np.asarray(raw_est['z_full'], dtype=float), gt, fs, fs_gt)
    t2, bridge_w, _ = _aligned_window(np.asarray(bridge_est['z_full'], dtype=float), gt, fs, fs_gt)

    ax_wave.plot(t, gt_w, color=COLORS['GT'], linewidth=2.0, label='GT')
    ax_wave.plot(t, raw_w, color=COLORS['PARH'], linewidth=1.8, label='raw OF PARH')
    ax_wave.plot(t2, bridge_w, color=COLORS['PARH_alt'], linewidth=1.8, label='OF_bridge PARH')
    ax_wave.set_title('Worst OF bridge case', fontsize=10.5, loc='left')
    add_metric_box(ax_wave, f"ΔCCC {float(row['delta_waveform_CCC']):+.3f}\nΔrate MAE {float(row['delta_rate_MAE']):+.2f}", loc='upper left')
    style_axis(ax_wave, grid='y')
    ax_wave.set_ylim(-3.2, 3.2)

    labels = ['Rate MAE', 'Rate RMSE', 'Wave CCC']
    raw_vals = [float(row['raw_rate_MAE']), float(row['raw_rate_RMSE']), float(row['raw_waveform_CCC'])]
    bridge_vals = [float(row['bridge_rate_MAE']), float(row['bridge_rate_RMSE']), float(row['bridge_waveform_CCC'])]
    x = np.arange(len(labels))
    width = 0.33
    ax_bar.bar(x - width / 2, raw_vals, width=width, color=COLORS['PARH'], label='raw OF PARH')
    ax_bar.bar(x + width / 2, bridge_vals, width=width, color=COLORS['PARH_alt'], label='OF_bridge PARH')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_title('Per-trial metric comparison', fontsize=10.5, loc='left')
    style_axis(ax_bar, grid='y')


def _plot_placeholder(out_path: Path, title: str, message: str) -> None:
    set_manuscript_style('paper')
    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    ax.axis('off')
    ax.text(0.5, 0.62, title, ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(0.5, 0.42, message, ha='center', va='center', fontsize=10.5)
    save_figure(fig, out_path)


def plot_failure_cases(manifest_csv: Path, run_dir: Path, out_path: Path):
    manifest = _safe_read_csv(manifest_csv)
    if manifest.empty or 'panel_kind' not in manifest.columns:
        _plot_placeholder(out_path, 'Failure cases unavailable', 'No residual/bridge failure case satisfied the current selection rule.')
        return

    res_rows = manifest[manifest['panel_kind'].isin(['residual_heavy_of', 'residual_heavy_parh'])]
    bridge_rows = manifest[manifest['panel_kind'] == 'of_bridge_failure']
    if res_rows.empty or bridge_rows.empty:
        _plot_placeholder(out_path, 'Failure cases unavailable', 'Current run did not produce both residual-heavy and bridge-failure exemplars.')
        return

    res_row = res_rows.iloc[0]
    bridge_row = bridge_rows.iloc[0]

    set_manuscript_style('paper')
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), constrained_layout=True)
    _residual_case(axes[0, 0], axes[0, 1], res_row, run_dir)
    _bridge_case(axes[1, 0], axes[1, 1], bridge_row, run_dir)

    axes[0, 0].legend(frameon=False, loc='upper right')
    axes[0, 1].legend(frameon=False, loc='upper right')
    axes[1, 0].legend(frameon=False, loc='upper right')
    axes[1, 1].legend(frameon=False, loc='upper right')

    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Metric value')
    axes[0, 1].set_xlabel('Time (s)')
    fig.suptitle('COHFACE failure cases: residual activation and OF bridge limitation', fontsize=13, y=1.02)
    save_figure(fig, out_path)


def parse_args():
    root = ROOT
    parser = argparse.ArgumentParser(description='Plot main failure cases from a persistent manifest.')
    parser.add_argument('--manifest-csv', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=root / 'paper' / 'figures' / 'F6_failure_cases.pdf')
    return parser.parse_args()


def main():
    args = parse_args()
    plot_failure_cases(manifest_csv=args.manifest_csv, run_dir=args.run_dir, out_path=args.out)
    print(json.dumps({'manifest': str(args.manifest_csv), 'out': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
