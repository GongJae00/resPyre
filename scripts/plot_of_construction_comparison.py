#!/usr/bin/env python3
"""Plot raw OF vs OF_bridge family comparison from table-ready CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.figure_style import (
    VARIANT_COLORS,
    add_bar_labels,
    family_label,
    save_figure,
    set_manuscript_style,
    style_axis,
    variant_label,
)

VARIANTS = ['Base', 'KFstd', 'PARH']
FAMILIES = ['OF', 'OF_bridge']


def _extract_rows(t3: pd.DataFrame, t4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in FAMILIES:
        r3 = t3[t3['family'] == family]
        r4 = t4[t4['family'] == family]
        if r3.empty or r4.empty:
            continue
        r3 = r3.iloc[0]
        r4 = r4.iloc[0]
        for variant in VARIANTS:
            rows.append(
                {
                    'family': family,
                    'variant': variant,
                    'rate_mae': float(r3[f'{variant}_MAE']),
                    'rate_r': float(r3[f'{variant}_PearsonR']),
                    'waveform_ccc': float(r4[f'{variant}_CCC']),
                    'waveform_dtw': float(r4[f'{variant}_DTW']),
                }
            )
    return pd.DataFrame(rows)


def _extract_rows_from_long(comparison: pd.DataFrame) -> pd.DataFrame:
    variant_map = {'Base': 'Base', 'OSSM-KF': 'KFstd', 'class-local PARH': 'PARH', 'PARH': 'PARH'}
    comparison = comparison.copy()
    comparison['variant_canonical'] = comparison['variant'].map(variant_map)
    sub = comparison[
        (comparison['dataset'].astype(str) == 'COHFACE')
        & (comparison['observation_class'].isin(FAMILIES))
        & (comparison['variant_canonical'].isin(VARIANTS))
    ].copy()
    rows = []
    for family in FAMILIES:
        for variant in VARIANTS:
            vdf = sub[(sub['observation_class'] == family) & (sub['variant_canonical'] == variant)]
            if vdf.empty:
                continue
            row = vdf.iloc[0]
            rows.append(
                {
                    'family': family,
                    'variant': variant,
                    'rate_mae': float(row['rate_MAE']),
                    'rate_r': float(row['rate_PearsonR']),
                    'waveform_ccc': float(row['waveform_CCC']),
                    'waveform_dtw': float(row['waveform_DTW']),
                }
            )
    return pd.DataFrame(rows)


def _plot_placeholder(out_pdf: Path, message: str) -> pd.DataFrame:
    set_manuscript_style('paper')
    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    ax.axis('off')
    ax.text(0.5, 0.60, 'OF construction comparison unavailable', ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(0.5, 0.42, message, ha='center', va='center', fontsize=10.5)
    save_figure(fig, out_pdf)
    return pd.DataFrame()


def plot_comparison(t3_csv: Path, t4_csv: Path, out_pdf: Path, comparison_csv: Path | None = None) -> pd.DataFrame:
    if comparison_csv is not None and comparison_csv.exists():
        data = _extract_rows_from_long(pd.read_csv(comparison_csv))
        if not data.empty and set(data['family']) == set(FAMILIES):
            return _plot_data(data, out_pdf)

    t3 = pd.read_csv(t3_csv)
    t4 = pd.read_csv(t4_csv)
    t3 = t3[t3['dataset'] == 'COHFACE'].copy()
    t4 = t4[t4['dataset'] == 'COHFACE'].copy()
    data = _extract_rows(t3, t4)
    if data.empty or set(data['family']) != set(FAMILIES):
        return _plot_placeholder(out_pdf, 'Current table-ready CSVs do not contain both raw OF and OF_bridge families.')
    return _plot_data(data, out_pdf)


def _plot_data(data: pd.DataFrame, out_pdf: Path) -> pd.DataFrame:
    set_manuscript_style('paper')
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.4), constrained_layout=True)
    metrics = [
        ('rate_mae', 'Rate MAE', True),
        ('rate_r', 'Rate Pearson r', False),
        ('waveform_ccc', 'Waveform CCC', False),
        ('waveform_dtw', 'Waveform DTW', True),
    ]
    xpos = [0, 1]
    width = 0.22
    offsets = {'Base': -width, 'KFstd': 0.0, 'PARH': width}

    for ax, (metric, title, lower_is_better) in zip(axes.ravel(), metrics):
        style_axis(ax, grid='y')
        max_y = float(data[metric].max())
        for variant in VARIANTS:
            ys = []
            for family in FAMILIES:
                row = data[(data['family'] == family) & (data['variant'] == variant)].iloc[0]
                ys.append(float(row[metric]))
            bars = ax.bar(
                [x + offsets[variant] for x in xpos],
                ys,
                width=width,
                color=VARIANT_COLORS[variant],
                label=variant_label(variant) if metric == 'rate_mae' else None,
                alpha=0.92,
                edgecolor='white',
                linewidth=0.8,
            )
            ax.set_ylim(0, max(max_y * 1.18, ax.get_ylim()[1]))
            add_bar_labels(ax, bars)
        ax.set_xticks(xpos, [family_label(f) for f in FAMILIES])
        suffix = ' (lower better)' if lower_is_better else ' (higher better)'
        ax.set_title(title + suffix, loc='left')
        ax.tick_params(axis='x', pad=8)

    axes[0, 0].legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(1.08, 1.18))
    fig.suptitle('Raw OF vs OF bridge observation construction', y=1.01)
    save_figure(fig, out_pdf)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot raw OF vs OF_bridge comparison.')
    parser.add_argument('--t3-csv', type=Path, default=ROOT / 'paper' / 'tables_ready' / 'T3_rate_main.csv')
    parser.add_argument('--t4-csv', type=Path, default=ROOT / 'paper' / 'tables_ready' / 'T4_waveform_main.csv')
    parser.add_argument('--comparison-csv', type=Path, default=ROOT / 'paper' / 'tables_ready' / 'S_T_final_observation_class_comparison.csv')
    parser.add_argument('--out-pdf', type=Path, default=ROOT / 'paper' / 'figures' / 'S_F6_of_construction_comparison.pdf')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = plot_comparison(args.t3_csv, args.t4_csv, args.out_pdf, comparison_csv=args.comparison_csv)
    print(f'Saved: {args.out_pdf}')
    if data.empty:
        print(json.dumps({'status': 'placeholder', 'out': str(args.out_pdf)}, indent=2))
    else:
        print(data.to_string(index=False))


if __name__ == '__main__':
    main()
