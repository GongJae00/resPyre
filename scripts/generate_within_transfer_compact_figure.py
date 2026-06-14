#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.figure_style import set_manuscript_style, style_axis, add_metric_box, save_figure


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _load_transfer_baseline(run_dir: Path) -> dict:
    metrics = run_dir / 'metrics'
    freq = pd.read_csv(metrics / 'metrics_freq_domain_raw.csv')
    wave = pd.read_csv(metrics / 'metrics_waveform_raw.csv')
    strict = pd.read_csv(metrics / 'metrics_waveform_strict_raw.csv')
    cons = pd.read_csv(metrics / 'decoupled_system_consistency.csv')
    method = 'p1dquad_temporal_decoupled_system'
    freq = freq[freq['method'] == method]
    wave = wave[(wave['method'] == method) & (wave['causal_or_smoothed'] == 'smoothed')]
    strict = strict[(strict['method'] == method) & (strict['causal_or_smoothed'] == 'smoothed')]
    cons = cons[cons['method'] == method]
    return {
        'rate_mae': float(pd.to_numeric(freq['MAE'], errors='coerce').median()),
        'aligned_ccc': float(pd.to_numeric(wave['waveform_CCC'], errors='coerce').median()),
        'strict_ccc': float(pd.to_numeric(strict['strict_CCC'], errors='coerce').median()),
        'consistency': float(pd.to_numeric(cons['consistency_score'], errors='coerce').median()),
        'track_diff': float(pd.to_numeric(cons['rate_waveform_track_abs_diff_bpm'], errors='coerce').median()),
        'system_confidence': float(pd.to_numeric(cons['system_confidence_score'], errors='coerce').median()),
    }


def _load_summary_or_run(path: Path) -> dict:
    if path.is_dir():
        return _load_transfer_baseline(path)
    return _read_json(path)


def _flat_values(summary: dict) -> dict:
    if 'rate_mae' in summary:
        return summary
    return {
        'rate_mae': float(summary['rate']['MAE']),
        'aligned_ccc': float(summary['waveform']['waveform_CCC']),
        'strict_ccc': float(summary['waveform_strict']['strict_CCC']),
        'consistency': float(summary['consistency']['consistency_score']),
        'track_diff': float(summary['consistency']['rate_waveform_track_abs_diff_bpm']),
        'system_confidence': float(summary['consistency']['system_confidence_score']),
    }


def build_figure(
    within_default_summary: Path,
    within_robust_summary: Path,
    transfer_default_run: Path,
    transfer_robust_summary: Path,
    out_pdf: Path,
):
    within_default = _flat_values(_load_summary_or_run(within_default_summary))
    within_robust = _flat_values(_load_summary_or_run(within_robust_summary))
    transfer_default = _load_transfer_baseline(transfer_default_run)
    transfer_robust = _flat_values(_load_summary_or_run(transfer_robust_summary))

    rows = [
        ('Within / Default', within_default),
        ('Within / Robust', within_robust),
        ('Transfer / Default', transfer_default),
        ('Transfer / Robust', transfer_robust),
    ]

    df = pd.DataFrame({name: vals for name, vals in rows}).T
    display = pd.DataFrame(index=df.index)
    display['rate_mae'] = -df['rate_mae']
    display['aligned_ccc'] = df['aligned_ccc']
    display['strict_ccc'] = df['strict_ccc']
    display['consistency'] = df['consistency']
    display['track_diff'] = -df['track_diff']
    display['system_confidence'] = df['system_confidence']

    norm = display.copy()
    for col in display.columns:
        vals = display[col].to_numpy(dtype=float)
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        if hi - lo < 1e-12:
            norm[col] = 0.5
        else:
            norm[col] = (vals - lo) / (hi - lo)

    set_manuscript_style('paper')
    fig = plt.figure(figsize=(16.4, 5.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    im = ax1.imshow(norm.to_numpy(dtype=float), aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    ax1.set_yticks(np.arange(len(norm.index)))
    ax1.set_yticklabels(norm.index)
    ax1.set_xticks(np.arange(len(norm.columns)))
    ax1.set_xticklabels(['-Rate\nMAE', 'Aligned\nCCC', 'Strict\nCCC', 'Consis-\ntency', '-|Δ|\nbpm', 'System\nconf.'])
    ax1.set_title('Within/transfer compact comparison')
    metric_order = ['rate_mae', 'aligned_ccc', 'strict_ccc', 'consistency', 'track_diff', 'system_confidence']
    for i in range(norm.shape[0]):
        for j, key in enumerate(metric_order):
            raw = float(df.iloc[i][key])
            txt = f"{raw:.3f}" if abs(raw) >= 1e-3 else f"{raw:.1e}"
            ax1.text(j, i, txt, ha='center', va='center', color='white', fontsize=8)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Within this panel: better is brighter')

    x = np.arange(2, dtype=float)
    width = 0.34
    aligned_base = np.array([df.loc['Within / Default', 'aligned_ccc'], df.loc['Transfer / Default', 'aligned_ccc']])
    aligned_rob = np.array([df.loc['Within / Robust', 'aligned_ccc'], df.loc['Transfer / Robust', 'aligned_ccc']])
    strict_base = np.array([df.loc['Within / Default', 'strict_ccc'], df.loc['Transfer / Default', 'strict_ccc']])
    strict_rob = np.array([df.loc['Within / Robust', 'strict_ccc'], df.loc['Transfer / Robust', 'strict_ccc']])
    ax2.bar(x - width / 2, aligned_base, width=width, color='#9AA5B1', label='Default aligned CCC')
    ax2.bar(x + width / 2, aligned_rob, width=width, color='#7D3AA6', label='Robust aligned CCC')
    ax2.plot(x - width / 2, strict_base, color='#4C6A92', marker='o', lw=1.5, label='Default strict CCC')
    ax2.plot(x + width / 2, strict_rob, color='#C87B2A', marker='o', lw=1.5, label='Robust strict CCC')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['COHFACE within', 'COHFACE→MAHNOB'])
    ax2.set_title('Robustness trade-off on waveform fidelity')
    ax2.set_ylabel('CCC')
    style_axis(ax2, 'y')
    ax2.legend(frameon=False, loc='upper right')
    add_metric_box(
        ax2,
        f"transfer aligned: {aligned_base[1]:.3f}→{aligned_rob[1]:.3f}\n"
        f"transfer cons.: {float(df.loc['Transfer / Default', 'consistency']):.3f}→{float(df.loc['Transfer / Robust', 'consistency']):.3f}\n"
        f"within strict: {strict_base[0]:.3f}→{strict_rob[0]:.3f}",
        loc='lower left'
    )

    fig.suptitle('Compact within-vs-transfer view of default and robust decoupled systems', y=1.02)
    save_figure(fig, out_pdf)


def write_report(
    within_default_summary: Path,
    within_robust_summary: Path,
    transfer_default_run: Path,
    transfer_robust_summary: Path,
    out_md: Path,
    figure_rel: str,
):
    within_default = _flat_values(_load_summary_or_run(within_default_summary))
    within_robust = _flat_values(_load_summary_or_run(within_robust_summary))
    transfer_default = _load_transfer_baseline(transfer_default_run)
    transfer_robust = _flat_values(_load_summary_or_run(transfer_robust_summary))
    lines = [
        '# Within vs Transfer Compact Comparison',
        '',
        f'- Figure: `{figure_rel}`',
        f'- Within default summary: `{within_default_summary}`',
        f'- Within robust summary: `{within_robust_summary}`',
        f'- Transfer default run: `{transfer_default_run}`',
        f'- Transfer robust summary: `{transfer_robust_summary}`',
        '',
        '## Key values',
        f"- Within aligned CCC: `{within_default['aligned_ccc']:.3f} -> {within_robust['aligned_ccc']:.3f}`",
        f"- Within strict CCC: `{within_default['strict_ccc']:.3f} -> {within_robust['strict_ccc']:.3f}`",
        f"- Within consistency: `{within_default['consistency']:.3f} -> {within_robust['consistency']:.3f}`",
        f"- Transfer aligned CCC: `{transfer_default['aligned_ccc']:.3f} -> {transfer_robust['aligned_ccc']:.3f}`",
        f"- Transfer strict CCC: `{transfer_default['strict_ccc']:.2e} -> {transfer_robust['strict_ccc']:.2e}`",
        f"- Transfer consistency: `{transfer_default['consistency']:.3f} -> {transfer_robust['consistency']:.3f}`",
        '',
        '## Interpretation',
        '- The robust fallback barely changes the easy within-dataset regime.',
        '- The same mechanism yields a visible aligned-waveform and consistency gain under cross-dataset transfer.',
        '- The remaining gap is strict transfer waveform fidelity, which stays near zero and remains the main unresolved bottleneck.',
        '',
    ]
    out_md.write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Generate a compact within-vs-transfer comparison figure for default and robust decoupled systems.')
    ap.add_argument('--within-default-summary', required=True)
    ap.add_argument('--within-robust-summary', required=True)
    ap.add_argument('--transfer-default-run', required=True)
    ap.add_argument('--transfer-robust-summary', required=True)
    ap.add_argument('--out-pdf', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args()

    build_figure(
        Path(args.within_default_summary).resolve(),
        Path(args.within_robust_summary).resolve(),
        Path(args.transfer_default_run).resolve(),
        Path(args.transfer_robust_summary).resolve(),
        Path(args.out_pdf).resolve(),
    )
    write_report(
        Path(args.within_default_summary).resolve(),
        Path(args.within_robust_summary).resolve(),
        Path(args.transfer_default_run).resolve(),
        Path(args.transfer_robust_summary).resolve(),
        Path(args.out_md).resolve(),
        str(Path(args.out_pdf).resolve()),
    )


if __name__ == '__main__':
    main()
