#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILE_SPECS = {
    'rate': 'metrics_freq_domain_raw.csv',
    'waveform': 'metrics_waveform_raw.csv',
    'waveform_strict': 'metrics_waveform_strict_raw.csv',
}

ROWS = [
    ('raw_base_p1dquad', 'Raw base (P1D_quad)', 'profile1D quadratic', None),
    ('parh_rate_expert', 'PARH rate expert', 'profile1d_quadratic__parh_ossm', None),
    ('shared_observation_law', 'Shared observation law', 'shared_observation_law__torch', None),
    ('adaptive_observation_law', 'Adaptive observation law', 'adaptive_observation_law__torch', None),
    ('staged_routed_multioutput', 'Staged routed multi-output', 'adaptive_observation_law_staged_routed_multihead__torch', None),
    ('waveform_expert_temporal', 'Waveform expert only', 'temporal_fusion_comparator__torch', None),
    ('adaptive_temporal_decoupled', 'Adaptive-rate decoupled system', None, 'adaptive_rate_temporal_waveform__decoupled_system'),
    ('decoupled_system', 'Decoupled system', None, 'p1dquad_rate_temporal_waveform__decoupled_system'),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Generate same-split unified ablation companion table.')
    ap.add_argument(
        '--source-run',
        default=str(ROOT / 'results' / 'cohface_rate_supervised_routed_full_gate_v1'),
        help='Run directory containing same-split source-method metrics.',
    )
    ap.add_argument(
        '--decoupled-root',
        default=str(ROOT / 'results' / 'cohface_decoupled_system_gate_v2'),
        help='Root directory containing decoupled-system child runs.',
    )
    ap.add_argument(
        '--out-csv',
        default=str(ROOT / 'paper' / 'tables_ready' / 'S_T7_unified_split_operator_alignment.csv'),
    )
    ap.add_argument(
        '--out-md',
        default=str(ROOT / 'analysis' / 'unified_split_operator_alignment.md'),
    )
    return ap.parse_args()


def _median_numeric(df: pd.DataFrame) -> dict[str, float]:
    nums = df.select_dtypes(include='number')
    med = nums.median(numeric_only=True)
    return {k: float(v) for k, v in med.items() if pd.notna(v)}


def _load_source_method(metrics_dir: Path, method: str) -> dict[str, float]:
    rate = pd.read_csv(metrics_dir / FILE_SPECS['rate'])
    wave = pd.read_csv(metrics_dir / FILE_SPECS['waveform'])
    strict = pd.read_csv(metrics_dir / FILE_SPECS['waveform_strict'])

    r = _median_numeric(rate[rate['method'] == method])
    w = wave[(wave['method'] == method) & (wave['causal_or_smoothed'] == 'smoothed')].copy()
    if 'output_type' in w.columns:
        w = w[(w['output_type'] == 'signal_hat') | (w['output_type'] == 'z_full')].sort_values(['video', 'output_type']).drop_duplicates('video', keep='first')
    w = _median_numeric(w)
    s = strict[(strict['method'] == method) & (strict['causal_or_smoothed'] == 'smoothed')].copy()
    if 'output_type' in s.columns:
        s = s[(s['output_type'] == 'signal_hat') | (s['output_type'] == 'z_full')].sort_values(['video', 'output_type']).drop_duplicates('video', keep='first')
    s = _median_numeric(s)

    return {
        'rate_MAE': r.get('MAE'),
        'rate_RMSE': r.get('RMSE'),
        'rate_PearsonR': r.get('PearsonR'),
        'waveform_CCC': w.get('waveform_CCC'),
        'waveform_MAE': w.get('waveform_MAE'),
        'waveform_DTW': w.get('waveform_DTW'),
        'strict_CCC': s.get('strict_CCC'),
        'strict_MAE': s.get('strict_MAE'),
        'strict_DTW': s.get('strict_DTW'),
        'cycle_ppi_mae_s': s.get('cycle_ppi_mae_s'),
    }


def _load_decoupled(metrics_dir: Path) -> dict[str, float]:
    rate = _median_numeric(pd.read_csv(metrics_dir / FILE_SPECS['rate']))
    wave = _median_numeric(pd.read_csv(metrics_dir / FILE_SPECS['waveform']))
    strict = _median_numeric(pd.read_csv(metrics_dir / FILE_SPECS['waveform_strict']))
    consistency = _median_numeric(pd.read_csv(metrics_dir / 'consistency_raw.csv'))
    return {
        'rate_MAE': rate.get('MAE'),
        'rate_RMSE': rate.get('RMSE'),
        'rate_PearsonR': rate.get('PearsonR'),
        'waveform_CCC': wave.get('waveform_CCC'),
        'waveform_MAE': wave.get('waveform_MAE'),
        'waveform_DTW': wave.get('waveform_DTW'),
        'strict_CCC': strict.get('strict_CCC'),
        'strict_MAE': strict.get('strict_MAE'),
        'strict_DTW': strict.get('strict_DTW'),
        'cycle_ppi_mae_s': strict.get('cycle_ppi_mae_s'),
        'consistency_score': consistency.get('consistency_score'),
        'system_confidence_score': consistency.get('system_confidence_score'),
    }


def _fmt(v: float | None) -> str:
    if v is None:
        return ''
    return f'{float(v):.3f}'


def _has_core_metrics(payload: dict[str, float]) -> bool:
    return all(payload.get(k) is not None for k in ("rate_MAE", "waveform_CCC", "strict_CCC"))


def main() -> None:
    args = parse_args()
    source_metrics = Path(args.source_run).resolve() / 'metrics'
    dec_root = Path(args.decoupled_root).resolve()
    rows = []
    skipped = []
    for row_id, label, source_method, decoupled_name in ROWS:
        if source_method is not None:
            payload = _load_source_method(source_metrics, source_method)
            payload['artifact_scope'] = 'same_split_source_run'
        else:
            payload = _load_decoupled(dec_root / decoupled_name / 'metrics')
            payload['artifact_scope'] = 'same_split_decoupled_system'
        if not _has_core_metrics(payload):
            skipped.append((label, payload['artifact_scope'], 'missing core rate/waveform/strict metric'))
            continue
        payload['row_id'] = row_id
        payload['label'] = label
        rows.append(payload)
    out_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    lines = [
        '# Unified split operator-alignment companion',
        '',
        f'- source run: `{Path(args.source_run).resolve()}`',
        f'- decoupled root: `{Path(args.decoupled_root).resolve()}`',
        '',
        'This table is the same-split companion to the main T5 mechanistic ablation.',
        'Unlike the main T5 table, every row here comes from one bounded COHFACE diagnostic split or from decoupled systems materialized and re-evaluated on that same split.',
        '',
        '| row | scope | rate MAE | rate r | aligned CCC | strict CCC | strict MAE | T4c PPI (s) | consistency | system conf. |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for _, row in out_df.iterrows():
        lines.append(
            f"| {row['label']} | {row['artifact_scope']} | {_fmt(row.get('rate_MAE'))} | {_fmt(row.get('rate_PearsonR'))} | {_fmt(row.get('waveform_CCC'))} | {_fmt(row.get('strict_CCC'))} | {_fmt(row.get('strict_MAE'))} | {_fmt(row.get('cycle_ppi_mae_s'))} | {_fmt(row.get('consistency_score'))} | {_fmt(row.get('system_confidence_score'))} |"
        )
    lines.extend([
        '',
        '## Reading rule',
        '- Use this table to check whether the operator-alignment ordering still holds when provenance is restricted to one split.',
        '- Use the main T5 table to keep the fuller mechanistic ladder, including lines that were validated in separate completed gates.',
        '',
    ])
    if skipped:
        lines.extend([
            '## Skipped incomplete rows',
            '',
            'Rows below were omitted because at least one core metric was missing in the live artifact.',
            '',
            '| row | scope | reason |',
            '| --- | --- | --- |',
        ])
        for label, scope, reason in skipped:
            lines.append(f'| {label} | {scope} | {reason} |')
        lines.append('')
    out_md = Path(args.out_md).resolve()
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
