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


def _z(x):
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    s = float(np.std(arr))
    if s < 1e-8:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / s


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


def _load_run(run_dir: Path):
    metrics = run_dir / 'metrics'
    consistency = pd.read_csv(metrics / 'consistency_raw.csv')
    wave = pd.read_csv(metrics / 'metrics_waveform_raw.csv')
    strict = pd.read_csv(metrics / 'metrics_waveform_strict_raw.csv')
    freq = pd.read_csv(metrics / 'metrics_freq_domain_raw.csv')
    summary = json.loads((metrics / 'decoupled_system_summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((metrics / 'decoupled_system_manifest.json').read_text(encoding='utf-8'))
    method = str(summary['system_name'])
    wave = wave[(wave['method'] == method) & (wave['causal_or_smoothed'] == 'smoothed')].copy()
    strict = strict[(strict['method'] == method) & (strict['causal_or_smoothed'] == 'smoothed')].copy()
    freq = freq[freq['method'] == method].copy()
    if 'output_type' in wave.columns:
        wave = wave[(wave['output_type'] == 'signal_hat') | (wave['output_type'] == 'z_full')].copy()
        wave = wave.sort_values(['video', 'output_type']).drop_duplicates('video', keep='first')
    if 'output_type' in strict.columns:
        strict = strict[(strict['output_type'] == 'signal_hat') | (strict['output_type'] == 'z_full')].copy()
        strict = strict.sort_values(['video', 'output_type']).drop_duplicates('video', keep='first')
    merged = consistency.merge(freq[['video', 'MAE', 'RMSE', 'PearsonR']], on='video', how='left')
    merged = merged.merge(wave[['video', 'waveform_CCC', 'waveform_MAE', 'waveform_DTW']], on='video', how='left')
    merged = merged.merge(strict[['video', 'strict_CCC', 'strict_MAE', 'strict_DTW', 'peak_time_mae_s', 'trough_time_mae_s', 'cycle_ppi_mae_s']], on='video', how='left')
    return summary, manifest, merged


def _case_rows(df: pd.DataFrame) -> dict[str, pd.Series]:
    work = df.sort_values('system_confidence_score').reset_index(drop=True)
    if work.empty:
        raise RuntimeError('no rows in decoupled system diagnostics table')
    return {
        'Low confidence': work.iloc[0],
        'Median confidence': work.iloc[len(work) // 2],
        'High confidence': work.iloc[-1],
    }


def _load_case(run_dir: Path, data_file: str):
    path = run_dir / data_file
    obj = pickle.loads(path.read_bytes())
    est = obj['estimates'][0]['estimate']
    fps = float(obj.get('fps', 1.0))
    gt = _z(obj.get('gt', np.array([], dtype=np.float64)))
    sig = _z(est.get('signal_hat', np.array([], dtype=np.float64)))
    n = min(len(gt), len(sig))
    t = np.arange(n, dtype=np.float64) / fps
    return t, gt[:n], sig[:n]


def build_figure(run_dir: Path, summary: dict, df: pd.DataFrame, out_pdf: Path):
    set_manuscript_style('paper')
    fig = plt.figure(figsize=(14.4, 8.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.10])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    axs_bottom = [fig.add_subplot(gs[1, i]) for i in range(3)]

    score_cols = [
        ('consistency_score', 'Consistency', '#4C6A92'),
        ('rate_confidence_score', 'Rate conf.', '#C87B2A'),
        ('waveform_confidence_score', 'Wave conf.', '#2B8A5B'),
        ('system_confidence_score', 'System conf.', '#7D3AA6'),
    ]
    bins = np.linspace(0.0, 1.0, 16)
    for col, label, color in score_cols:
        vals = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy(dtype=float)
        if vals.size:
            ax1.hist(vals, bins=bins, alpha=0.28, color=color, label=label, density=True)
            ax1.axvline(float(np.median(vals)), color=color, lw=1.2, linestyle='--')
    ax1.set_title('Confidence and consistency distributions', loc='center')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Density')
    ax1.legend(frameon=False, ncol=2, loc='upper right', fontsize=8.2)
    style_axis(ax1, 'y')
    add_metric_box(
        ax1,
        f"median system={_fmt(df['system_confidence_score'].median())}\nmedian consistency={_fmt(df['consistency_score'].median())}",
        loc='upper left',
        fontsize=7.8,
    )

    sc = ax2.scatter(
        df['system_confidence_score'],
        df['strict_CCC'],
        c=df['MAE'],
        cmap='viridis_r',
        s=34,
        alpha=0.85,
        edgecolors='none',
    )
    ax2.set_title('Confidence vs strict waveform fidelity', loc='center')
    ax2.set_xlabel('System confidence score')
    ax2.set_ylabel('Strict CCC')
    style_axis(ax2, 'both')
    add_metric_box(
        ax2,
        f"median strict CCC={_fmt(df['strict_CCC'].median())}\nmedian rate MAE={_fmt(df['MAE'].median())}",
        fontsize=7.8,
    )
    cbar = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.03)
    cbar.set_label('Rate MAE')

    sc2 = ax3.scatter(
        df['rate_waveform_track_abs_diff_bpm'],
        df['strict_CCC'],
        c=df['waveform_CCC'],
        cmap='plasma',
        s=34,
        alpha=0.85,
        edgecolors='none',
    )
    ax3.set_title('Agreement vs strict waveform fidelity', loc='center')
    ax3.set_xlabel('|rate track - waveform track| bpm')
    ax3.set_ylabel('Strict CCC')
    style_axis(ax3, 'both')
    add_metric_box(
        ax3,
        f"median |Δ bpm|={_fmt(df['rate_waveform_track_abs_diff_bpm'].median())}\nmedian aligned CCC={_fmt(df['waveform_CCC'].median())}",
        fontsize=7.8,
    )
    cbar2 = fig.colorbar(sc2, ax=ax3, fraction=0.046, pad=0.03)
    cbar2.set_label('Aligned waveform CCC')

    cases = _case_rows(df)
    for ax, title in zip(axs_bottom, ['Low confidence', 'Median confidence', 'High confidence']):
        row = cases[title]
        t, gt, sig = _load_case(run_dir, str(row['data_file']))
        ax.plot(t, gt, color='black', lw=1.6, label='GT')
        ax.plot(t, sig, color='#2B8A5B', lw=1.15, label='Decoupled waveform')
        ax.set_title(f"{title} case\n{row['video']}", loc='center')
        style_axis(ax, 'y')
        ax.set_xlabel('Time (s)')
        ax.set_ylim(*_case_ylim(gt, sig))
        if ax is axs_bottom[0]:
            ax.set_ylabel('Normalized amplitude')
        txt = (
            f"sys={_fmt(row['system_confidence_score'])}\n"
            f"cons={_fmt(row['consistency_score'])}\n"
            f"strict CCC={_fmt(row['strict_CCC'])}\n"
            f"|Δ bpm|={_fmt(row['rate_waveform_track_abs_diff_bpm'])}"
        )
        add_metric_box(ax, txt, loc='upper right', fontsize=7.6)
    axs_bottom[0].legend(frameon=False, loc='lower left', fontsize=8.2)

    fig.suptitle('Decoupled system confidence, agreement, and case diagnostics', y=0.985)
    fig.subplots_adjust(left=0.055, right=0.982, bottom=0.075, top=0.90, wspace=0.28, hspace=0.34)
    save_figure(fig, out_pdf)


def write_report(summary: dict, manifest: dict, df: pd.DataFrame, out_md: Path, figure_rel: str):
    cases = _case_rows(df)
    lines = [
        '# Decoupled System Diagnostics',
        '',
        f"- Run: `{manifest['name']}`",
        f"- Source rate run: `{manifest['rate_run']}`",
        f"- Source waveform run: `{manifest['waveform_run']}`",
        f"- Rate method: `{manifest['rate_method']}`",
        f"- Waveform method: `{manifest['waveform_method']}`",
        f"- Figure: `{figure_rel}`",
        '',
        '## Median aggregate diagnostics',
        f"- consistency score: `{_fmt(df['consistency_score'].median())}`",
        f"- rate confidence score: `{_fmt(df['rate_confidence_score'].median())}`",
        f"- waveform confidence score: `{_fmt(df['waveform_confidence_score'].median())}`",
        f"- system confidence score: `{_fmt(df['system_confidence_score'].median())}`",
        f"- rate-vs-waveform track abs diff (bpm): `{_fmt(df['rate_waveform_track_abs_diff_bpm'].median())}`",
        f"- strict CCC: `{_fmt(df['strict_CCC'].median())}`",
        f"- aligned waveform CCC: `{_fmt(df['waveform_CCC'].median())}`",
        f"- rate MAE: `{_fmt(df['MAE'].median())}`",
        '',
        '## Case selection',
    ]
    for title, row in cases.items():
        lines.extend([
            f"### {title}: `{row['video']}`",
            f"- system confidence score: `{_fmt(row['system_confidence_score'])}`",
            f"- consistency score: `{_fmt(row['consistency_score'])}`",
            f"- rate MAE: `{_fmt(row['MAE'])}`",
            f"- strict CCC: `{_fmt(row['strict_CCC'])}`",
            f"- rate-vs-waveform track abs diff (bpm): `{_fmt(row['rate_waveform_track_abs_diff_bpm'])}`",
            '',
        ])
    lines.extend([
        '## Interpretation',
        '- The best current decoupled system stays strong because the strongest rate expert and strongest waveform expert remain mostly rate-waveform consistent on the same split.',
        '- `system_confidence_score` is intentionally conservative: it collapses when either rate stability, waveform stability, or cross-output consistency drops.',
        '- The lower row shows selected low-, median-, and high-confidence trajectories from the promoted best decoupled system itself.',
        '',
    ])
    out_md.write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Generate confidence/consistency diagnostics for a decoupled system run.')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out-pdf', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    out_pdf = Path(args.out_pdf).resolve()
    out_md = Path(args.out_md).resolve()
    summary, manifest, df = _load_run(run_dir)
    build_figure(run_dir, summary, df, out_pdf)
    write_report(summary, manifest, df, out_md, str(out_pdf))


if __name__ == '__main__':
    main()
