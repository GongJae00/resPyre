#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
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


METHOD_LABELS = {
    'temporal_fusion_comparator__torch': 'Temporal',
    'adaptive_observation_law__torch': 'Adaptive fallback',
    'profile1D quadratic': 'Raw-base fallback',
}
METHOD_COLORS = {
    'temporal_fusion_comparator__torch': '#2B8A5B',
    'adaptive_observation_law__torch': '#C87B2A',
    'profile1D quadratic': '#4C6A92',
}
ORDERED_METHODS = [
    'temporal_fusion_comparator__torch',
    'adaptive_observation_law__torch',
    'profile1D quadratic',
]


def _z(x):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    if sigma < 1e-8:
        return arr - mu
    return (arr - mu) / sigma


def _fmt(value: float, digits: int = 3) -> str:
    try:
        val = float(value)
    except Exception:
        return 'n/a'
    if not np.isfinite(val):
        return 'n/a'
    return f'{val:.{digits}f}'


def _case_ylim(*series: np.ndarray) -> tuple[float, float]:
    vals = [np.asarray(s, dtype=np.float64).reshape(-1) for s in series if np.asarray(s).size]
    if not vals:
        return (-1.0, 1.0)
    y = np.concatenate(vals)
    lo, hi = np.percentile(y, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(y)), float(np.max(y))
    span = max(hi - lo, 1e-6)
    pad = 0.10 * span
    return lo - pad, hi + pad


def _load_consistency(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / 'metrics' / 'consistency_raw.csv')


def _load_summary(run_dir: Path) -> dict:
    return json.loads((run_dir / 'metrics' / 'decoupled_system_summary.json').read_text(encoding='utf-8'))


def _load_source_estimate(source_run: Path, video: str, method: str):
    obj = pickle.loads((source_run / 'data' / f'{video}.pkl').read_bytes())
    fps = float(obj.get('fps', 1.0))
    gt = _z(obj.get('gt', np.array([], dtype=np.float64)))
    estimate = None
    for item in obj.get('estimates', []):
        if item.get('method') == method:
            estimate = item.get('estimate', {})
            break
    if estimate is None:
        raise KeyError(f'method {method} not found in {source_run}/data/{video}.pkl')
    signal = estimate.get('signal_hat')
    if signal is None and method.endswith('__parh_ossm'):
        signal = estimate.get('z_full')
    signal = _z(signal if signal is not None else np.array([], dtype=np.float64))
    n = min(len(gt), len(signal))
    t = np.arange(n, dtype=np.float64) / fps
    return t, gt[:n], signal[:n]


def _selection_counts(df: pd.DataFrame) -> pd.Series:
    counts = df['selected_waveform_method'].fillna('unknown').value_counts()
    for name in ORDERED_METHODS:
        if name not in counts.index:
            counts.loc[name] = 0
    return counts[ORDERED_METHODS]


def _pick_cases(df_transfer: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    fallback = df_transfer[df_transfer['fallback_triggered'] == True].copy()
    if fallback.empty:
        raise RuntimeError('no fallback-triggered rows found')
    raw = fallback[fallback['selected_waveform_method'] == 'profile1D quadratic'].copy()
    adaptive = fallback[fallback['selected_waveform_method'] == 'adaptive_observation_law__torch'].copy()
    if raw.empty:
        raw = fallback.sort_values('system_confidence_score', ascending=False).head(1)
    else:
        raw = raw.sort_values('consistency_score', ascending=False).head(1)
    if adaptive.empty:
        adaptive = fallback.sort_values('system_confidence_score', ascending=True).head(1)
    else:
        adaptive = adaptive.sort_values('consistency_score', ascending=False).head(1)
    return raw.iloc[0], adaptive.iloc[0]


def build_figure(
    within_default: Path,
    within_robust: Path,
    transfer_default_metrics: Path,
    transfer_robust: Path,
    out_pdf: Path,
):
    set_manuscript_style('paper')
    df_within = _load_consistency(within_robust)
    df_transfer = _load_consistency(transfer_robust)
    summary_within_default = _load_summary(within_default)
    summary_within_robust = _load_summary(within_robust)
    summary_transfer_robust = _load_summary(transfer_robust)

    freq_transfer = pd.read_csv(transfer_default_metrics / 'metrics_freq_domain_raw.csv')
    wave_transfer = pd.read_csv(transfer_default_metrics / 'metrics_waveform_raw.csv')
    strict_transfer = pd.read_csv(transfer_default_metrics / 'metrics_waveform_strict_raw.csv')
    cons_transfer = pd.read_csv(transfer_default_metrics / 'decoupled_system_consistency.csv')
    method = 'p1dquad_temporal_decoupled_system'
    wave_transfer = wave_transfer[(wave_transfer['method'] == method) & (wave_transfer['causal_or_smoothed'] == 'smoothed')]
    strict_transfer = strict_transfer[(strict_transfer['method'] == method) & (strict_transfer['causal_or_smoothed'] == 'smoothed')]
    cons_transfer = cons_transfer[cons_transfer['method'] == method]

    baseline_transfer = {
        'aligned_ccc': float(pd.to_numeric(wave_transfer['waveform_CCC'], errors='coerce').median()),
        'strict_ccc': float(pd.to_numeric(strict_transfer['strict_CCC'], errors='coerce').median()),
        'consistency': float(pd.to_numeric(cons_transfer['consistency_score'], errors='coerce').median()),
        'track_diff': float(pd.to_numeric(cons_transfer['rate_waveform_track_abs_diff_bpm'], errors='coerce').median()),
    }

    within_counts = _selection_counts(df_within)
    transfer_counts = _selection_counts(df_transfer)
    raw_case, adaptive_case = _pick_cases(df_transfer)
    manifest = json.loads((transfer_robust / 'metrics' / 'decoupled_system_manifest.json').read_text(encoding='utf-8'))
    robust_source_run = Path(manifest['waveform_run'])

    fig = plt.figure(figsize=(15.0, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    x = np.arange(2)
    width = 0.22
    for idx, key in enumerate(ORDERED_METHODS):
        vals = np.array([
            float(within_counts[key]) / max(len(df_within), 1),
            float(transfer_counts[key]) / max(len(df_transfer), 1),
        ])
        ax1.bar(x + (idx - 1) * width, vals, width=width, color=METHOD_COLORS[key], label=METHOD_LABELS[key])
    ax1.set_xticks(x)
    ax1.set_xticklabels(['COHFACE within', 'COHFACE→MAHNOB'])
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title('Waveform selection mix under robust fallback', loc='center')
    ax1.set_ylabel('Fraction of trials')
    ax1.legend(frameon=False, loc='upper right', fontsize=8.1)
    style_axis(ax1, 'y')
    add_metric_box(
        ax1,
        f"within fallback={int(df_within['fallback_triggered'].sum())}/{len(df_within)}\ntransfer fallback={int(df_transfer['fallback_triggered'].sum())}/{len(df_transfer)}",
        fontsize=7.7,
    )

    bins = np.linspace(0.0, 0.5, 16)
    for key in ORDERED_METHODS:
        vals = pd.to_numeric(
            df_transfer.loc[df_transfer['selected_waveform_method'] == key, 'system_confidence_score'],
            errors='coerce'
        ).dropna().to_numpy(dtype=float)
        if vals.size:
            ax2.hist(vals, bins=bins, alpha=0.32, color=METHOD_COLORS[key])
    ax2.axvline(0.17, color='black', linestyle='--', lw=1.2)
    ax2.set_title('Transfer confidence under fallback', loc='center')
    ax2.set_xlabel('System confidence score')
    ax2.set_ylabel('Count')
    style_axis(ax2, 'y')
    add_metric_box(
        ax2,
        f"median score={_fmt(df_transfer['system_confidence_score'].median())}\nthreshold=0.17",
        fontsize=7.8,
    )

    baseline_vals = [
        float(summary_within_default['waveform']['waveform_CCC']),
        float(summary_within_default['waveform_strict']['strict_CCC']),
        float(summary_within_default['consistency']['consistency_score']),
        baseline_transfer['aligned_ccc'],
        baseline_transfer['strict_ccc'],
        baseline_transfer['consistency'],
    ]
    robust_vals = [
        float(summary_within_robust['waveform']['waveform_CCC']),
        float(summary_within_robust['waveform_strict']['strict_CCC']),
        float(summary_within_robust['consistency']['consistency_score']),
        float(summary_transfer_robust['waveform']['waveform_CCC']),
        float(summary_transfer_robust['waveform_strict']['strict_CCC']),
        float(summary_transfer_robust['consistency']['consistency_score']),
    ]
    xpos = np.array([0, 1, 2, 4, 5, 6], dtype=float)
    ax3.bar(xpos - 0.17, baseline_vals, width=0.32, color='#9AA5B1', label='Default')
    ax3.bar(xpos + 0.17, robust_vals, width=0.32, color='#7D3AA6', label='Robust fallback')
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(['Aligned', 'Strict', 'Cons.', 'Aligned', 'Strict', 'Cons.'], rotation=20, ha='right')
    ax3.set_title('Default vs robust: within and transfer', loc='center')
    ax3.set_ylabel('Median score')
    style_axis(ax3, 'y')
    ax3.legend(frameon=False, loc='upper right', fontsize=8.1)
    ax3.text(1.0, -0.22, 'COHFACE within', transform=ax3.get_xaxis_transform(), ha='center', va='top', fontsize=8.6)
    ax3.text(5.0, -0.22, 'COHFACE→MAHNOB', transform=ax3.get_xaxis_transform(), ha='center', va='top', fontsize=8.6)
    add_metric_box(
        ax3,
        f"transfer aligned={_fmt(baseline_transfer['aligned_ccc'])}→{_fmt(summary_transfer_robust['waveform']['waveform_CCC'])}\ntransfer cons.={_fmt(baseline_transfer['consistency'])}→{_fmt(summary_transfer_robust['consistency']['consistency_score'])}",
        loc='lower left',
        fontsize=7.6,
    )

    def plot_case(ax, row: pd.Series, title: str):
        video = str(row['video'])
        selected_method = str(row['selected_waveform_method'])
        t_gt, gt, temporal = _load_source_estimate(robust_source_run, video, 'temporal_fusion_comparator__torch')
        _, _, chosen = _load_source_estimate(robust_source_run, video, selected_method)
        n = min(len(t_gt), len(gt), len(temporal), len(chosen))
        ax.plot(t_gt[:n], gt[:n], color='black', lw=1.6, label='GT')
        ax.plot(t_gt[:n], temporal[:n], color='#2B8A5B', lw=1.05, label='Primary temporal')
        ax.plot(t_gt[:n], chosen[:n], color='#7D3AA6', lw=1.05, label='Selected waveform')
        ax.set_title(f"{title}\n{video}", loc='center')
        ax.set_xlabel('Time (s)')
        style_axis(ax, 'y')
        ax.set_ylim(*_case_ylim(gt[:n], temporal[:n], chosen[:n]))
        delta_txt = _fmt(row['rate_waveform_track_abs_diff_bpm'])
        txt = (
            f"selected={METHOD_LABELS.get(selected_method, selected_method)}\n"
            f"sys={_fmt(row['system_confidence_score'])}\n"
            f"cons={_fmt(row['consistency_score'])}\n"
            f"|Δ bpm|={delta_txt}"
        )
        add_metric_box(ax, txt, loc='upper right', fontsize=7.4)

    plot_case(ax4, raw_case, 'Raw-base fallback case')
    plot_case(ax5, adaptive_case, 'Adaptive fallback case')
    ax4.set_ylabel('Normalized amplitude')
    ax4.legend(frameon=False, loc='lower left', fontsize=8.1)

    fallback_rate = np.array([
        float(df_within['fallback_triggered'].mean()),
        float(df_transfer['fallback_triggered'].mean()),
    ])
    primary_rate = 1.0 - fallback_rate
    ax6.bar([0, 1], primary_rate, color='#2B8A5B', label='Primary temporal kept')
    ax6.bar([0, 1], fallback_rate, bottom=primary_rate, color='#7D3AA6', label='Fallback used')
    ax6.set_xticks([0, 1])
    ax6.set_xticklabels(['COHFACE within', 'COHFACE→MAHNOB'])
    ax6.set_ylim(0.0, 1.0)
    ax6.set_title('Fallback activation rate', loc='center')
    ax6.set_ylabel('Fraction of trials')
    style_axis(ax6, 'y')
    add_metric_box(
        ax6,
        f"within strict CCC={_fmt(summary_within_robust['waveform_strict']['strict_CCC'])}\ntransfer strict CCC={summary_transfer_robust['waveform_strict']['strict_CCC']:.2e}",
        fontsize=7.7,
    )

    fig.suptitle('Robust fallback diagnostics: selection, confidence, and hard-regime cases', y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.11, top=0.90, wspace=0.30, hspace=0.40)
    save_figure(fig, out_pdf)


def write_report(
    within_default: Path,
    within_robust: Path,
    transfer_default_metrics: Path,
    transfer_robust: Path,
    out_md: Path,
    figure_rel: str,
):
    df_within = _load_consistency(within_robust)
    df_transfer = _load_consistency(transfer_robust)
    summary_within_default = _load_summary(within_default)
    summary_within_robust = _load_summary(within_robust)
    summary_transfer_robust = _load_summary(transfer_robust)
    raw_case, adaptive_case = _pick_cases(df_transfer)

    wave_transfer = pd.read_csv(transfer_default_metrics / 'metrics_waveform_raw.csv')
    strict_transfer = pd.read_csv(transfer_default_metrics / 'metrics_waveform_strict_raw.csv')
    cons_transfer = pd.read_csv(transfer_default_metrics / 'decoupled_system_consistency.csv')
    method = 'p1dquad_temporal_decoupled_system'
    wave_transfer = wave_transfer[(wave_transfer['method'] == method) & (wave_transfer['causal_or_smoothed'] == 'smoothed')]
    strict_transfer = strict_transfer[(strict_transfer['method'] == method) & (strict_transfer['causal_or_smoothed'] == 'smoothed')]
    cons_transfer = cons_transfer[cons_transfer['method'] == method]

    lines = [
        '# Robust Fallback Diagnostics',
        '',
        f'- Figure: `{figure_rel}`',
        f'- Within baseline: `{within_default}`',
        f'- Within robust: `{within_robust}`',
        f'- Transfer baseline metrics: `{transfer_default_metrics}`',
        f'- Transfer robust: `{transfer_robust}`',
        '',
        '## Aggregate deltas',
        f"- COHFACE within aligned CCC: `{_fmt(summary_within_default['waveform']['waveform_CCC'])} -> {_fmt(summary_within_robust['waveform']['waveform_CCC'])}`",
        f"- COHFACE within strict CCC: `{_fmt(summary_within_default['waveform_strict']['strict_CCC'])} -> {_fmt(summary_within_robust['waveform_strict']['strict_CCC'])}`",
        f"- COHFACE within fallback triggered: `{int(df_within['fallback_triggered'].sum())}/{len(df_within)}`",
        f"- COHFACE->MAHNOB aligned CCC: `{_fmt(pd.to_numeric(wave_transfer['waveform_CCC'], errors='coerce').median())} -> {_fmt(summary_transfer_robust['waveform']['waveform_CCC'])}`",
        f"- COHFACE->MAHNOB strict CCC: `{pd.to_numeric(strict_transfer['strict_CCC'], errors='coerce').median():.2e} -> {summary_transfer_robust['waveform_strict']['strict_CCC']:.2e}`",
        f"- COHFACE->MAHNOB consistency: `{_fmt(pd.to_numeric(cons_transfer['consistency_score'], errors='coerce').median())} -> {_fmt(summary_transfer_robust['consistency']['consistency_score'])}`",
        f"- COHFACE->MAHNOB fallback triggered: `{int(df_transfer['fallback_triggered'].sum())}/{len(df_transfer)}`",
        '',
        '## Selection mix',
    ]
    for label, df in [('Within', df_within), ('Transfer', df_transfer)]:
        counts = _selection_counts(df)
        parts = [f"{METHOD_LABELS.get(name, name)}: {int(counts[name])}" for name in counts.index]
        lines.append(f"- {label}: " + ', '.join(parts))
    lines.extend([
        '',
        '## Representative fallback cases',
        f"- Raw-base fallback case: `{raw_case['video']}` selected `{METHOD_LABELS.get(raw_case['selected_waveform_method'], raw_case['selected_waveform_method'])}` with consistency `{_fmt(raw_case['consistency_score'])}`",
        f"- Adaptive fallback case: `{adaptive_case['video']}` selected `{METHOD_LABELS.get(adaptive_case['selected_waveform_method'], adaptive_case['selected_waveform_method'])}` with consistency `{_fmt(adaptive_case['consistency_score'])}`",
        '',
        '## Interpretation',
        '- The robust fallback preserves the easy COHFACE regime by almost never firing.',
        '- In hard transfer, the fallback is activated selectively and improves aligned waveform fidelity and rate-waveform consistency without changing the rate expert.',
        '- The remaining weakness is strict transfer reconstruction, which still stays near zero even after fallback.',
        '',
    ])
    out_md.write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Generate robust fallback diagnostics figure and report.')
    ap.add_argument('--within-default-run', required=True)
    ap.add_argument('--within-robust-run', required=True)
    ap.add_argument('--transfer-default-run', required=True)
    ap.add_argument('--transfer-robust-run', required=True)
    ap.add_argument('--out-pdf', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args()

    within_default = Path(args.within_default_run).resolve()
    within_robust = Path(args.within_robust_run).resolve()
    transfer_default_metrics = Path(args.transfer_default_run).resolve() / 'metrics'
    transfer_robust = Path(args.transfer_robust_run).resolve()
    out_pdf = Path(args.out_pdf).resolve()
    out_md = Path(args.out_md).resolve()

    build_figure(within_default, within_robust, transfer_default_metrics, transfer_robust, out_pdf)
    write_report(within_default, within_robust, transfer_default_metrics, transfer_robust, out_md, str(out_pdf))


if __name__ == '__main__':
    main()
