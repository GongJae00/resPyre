#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUITE_ROOT = ROOT / 'results' / 'full_decoupled_validation_suite_v2'
DEFAULT_TABLE_OUT = ROOT / 'paper' / 'tables_ready' / 'S_T8_strict_scale_safe_hard_regime.csv'
DEFAULT_REPORT_OUT = ROOT / 'analysis' / 'strict_scale_safe_report_full_decoupled_validation_suite_v2.md'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Generate a scale-safe strict-waveform companion table for MAHNOB-heavy hard regimes.')
    p.add_argument('--suite-root', type=Path, default=DEFAULT_SUITE_ROOT)
    p.add_argument('--table-out', type=Path, default=DEFAULT_TABLE_OUT)
    p.add_argument('--report-out', type=Path, default=DEFAULT_REPORT_OUT)
    return p.parse_args()


def _read_payload_with_fallback(suite_root: Path, run_name: str, rel_path: str) -> dict:
    rel = Path(rel_path)
    candidates = [
        suite_root / run_name / rel,
        suite_root / f'{run_name}__robust' / 'data' / rel.name,
        suite_root / f'{run_name}__adaptiveonly_fallback' / 'data' / rel.name,
    ]
    for path in candidates:
        if path.exists():
            return pickle.loads(path.read_bytes())
    tried = ', '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'could not find GT payload for {run_name}/{rel_path}; tried: {tried}')


def _enrich_source_strict(suite_root: Path, run_name: str) -> pd.DataFrame:
    run_dir = suite_root / run_name
    df = pd.read_csv(run_dir / 'metrics' / 'metrics_waveform_strict_raw.csv')

    scale_cache: dict[str, tuple[float, float]] = {}
    spans: list[tuple[float, float]] = []
    for rel_path in df['data_file']:
        if rel_path not in scale_cache:
            payload = _read_payload_with_fallback(suite_root, run_name, str(rel_path))
            gt = np.asarray(payload['gt'], dtype=float)
            q05, q95 = np.nanpercentile(gt, [5, 95])
            q25, q75 = np.nanpercentile(gt, [25, 75])
            scale_cache[rel_path] = (
                float(max(abs(q95 - q05), 1e-8)),
                float(max(abs(q75 - q25), 1e-8)),
            )
        spans.append(scale_cache[rel_path])

    df['gt_span_p95p05'] = [span for span, _ in spans]
    df['gt_iqr'] = [iqr for _, iqr in spans]
    df['strict_NMAE_span'] = df['strict_MAE'] / df['gt_span_p95p05']
    df['strict_NRMSE_span'] = df['strict_RMSE'] / df['gt_span_p95p05']
    df['strict_NDTW_span'] = df['strict_DTW'] / df['gt_span_p95p05']

    cols = [
        'data_file',
        'method',
        'strict_CCC',
        'strict_MAE',
        'strict_DTW',
        'strict_NMAE_span',
        'strict_NRMSE_span',
        'strict_NDTW_span',
    ]
    return (
        df[cols]
        .groupby(['data_file', 'method'], as_index=False)
        .median(numeric_only=True)
    )


def _medians(frame: pd.DataFrame) -> dict[str, float]:
    med = frame[
        [
            'strict_CCC',
            'strict_MAE',
            'strict_DTW',
            'strict_NMAE_span',
            'strict_NRMSE_span',
            'strict_NDTW_span',
        ]
    ].median()
    return {
        'strict_CCC': float(med['strict_CCC']),
        'strict_MAE_raw': float(med['strict_MAE']),
        'strict_DTW_raw': float(med['strict_DTW']),
        'strict_NMAE_span': float(med['strict_NMAE_span']),
        'strict_NRMSE_span': float(med['strict_NRMSE_span']),
        'strict_NDTW_span': float(med['strict_NDTW_span']),
    }


def _append_direct(rows: list[dict[str, object]], scenario: str, role: str, source_df: pd.DataFrame, method: str) -> None:
    rows.append({
        'scenario': scenario,
        'role': role,
        **_medians(source_df[source_df['method'] == method]),
    })


def _append_reconstructed(rows: list[dict[str, object]], scenario: str, role: str, source_df: pd.DataFrame, consistency_csv: Path) -> None:
    selection = pd.read_csv(consistency_csv)
    merged = selection.merge(
        source_df,
        left_on=['data_file', 'selected_waveform_method'],
        right_on=['data_file', 'method'],
        how='left',
    )
    rows.append({'scenario': scenario, 'role': role, **_medians(merged)})


def main() -> int:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    table_out = args.table_out.resolve()
    report_out = args.report_out.resolve()

    ctm = _enrich_source_strict(suite_root, 'cohface_to_mahnob')
    mwi = _enrich_source_strict(suite_root, 'mahnob_within')

    rows: list[dict[str, object]] = []
    _append_direct(rows, 'COHFACE->MAHNOB', 'default decoupled', ctm, 'p1dquad_temporal_decoupled_system')
    _append_reconstructed(rows, 'COHFACE->MAHNOB', 'consistency-first robust', ctm, suite_root / 'cohface_to_mahnob__robust' / 'metrics' / 'consistency_raw.csv')
    _append_reconstructed(rows, 'COHFACE->MAHNOB', 'waveform-first robust', ctm, suite_root / 'cohface_to_mahnob__adaptiveonly_fallback' / 'metrics' / 'consistency_raw.csv')
    _append_direct(rows, 'MAHNOB within', 'adaptive+temporal decoupled', mwi, 'adaptive_temporal_decoupled_system__torch')
    _append_direct(rows, 'MAHNOB within', 'p1dquad+temporal decoupled', mwi, 'p1dquad_temporal_decoupled_system')
    _append_reconstructed(rows, 'MAHNOB within', 'consistency-first robust', mwi, suite_root / 'mahnob_within__robust' / 'metrics' / 'consistency_raw.csv')
    _append_reconstructed(rows, 'MAHNOB within', 'waveform-first robust', mwi, suite_root / 'mahnob_within__adaptiveonly_fallback' / 'metrics' / 'consistency_raw.csv')

    out = pd.DataFrame(rows)
    table_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(table_out, index=False)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        '\n'.join([
            '# Strict Scale-Safe Summary',
            '',
            'Absolute strict MAE/DTW values are unit-preserving and therefore',
            'dataset-scale dependent. For MAHNOB-heavy scenarios, cross-regime',
            'comparison should use `strict_CCC` together with the span-normalized',
            'metrics below.',
            '',
            '```csv',
            out.to_csv(index=False).strip(),
            '```',
            '',
            f'Source table: `{table_out}`',
        ]),
        encoding='utf-8',
    )
    print(f'Wrote {table_out}')
    print(f'Wrote {report_out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
