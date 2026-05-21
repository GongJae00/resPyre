#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load(run_dir: Path):
    m = run_dir / 'metrics'
    return {
        'rate': pd.read_csv(m / 'metrics_freq_domain_summary.csv'),
        'wave': pd.read_csv(m / 'metrics_waveform_summary.csv'),
        'strict': pd.read_csv(m / 'metrics_waveform_strict_summary.csv'),
        'diag': pd.read_csv(m / 'metrics_filter_diagnostics_summary.csv'),
    }


def _find(df: pd.DataFrame, method: str):
    hit = df[df['method'] == method]
    return None if hit.empty else hit.iloc[0]


def _row(bundle, method: str, source_run: str, row_id: str, intent: str):
    r = _find(bundle['rate'], method)
    w = _find(bundle['wave'], method)
    s = _find(bundle['strict'], method)
    d = _find(bundle['diag'], method)
    if r is None or s is None:
        raise KeyError(f'missing required summaries for method={method}')
    return {
        'row_id': row_id,
        'intent': intent,
        'source_run': source_run,
        'method': method,
        'rate_MAE': r['MAE_median'],
        'rate_RMSE': r['RMSE_median'],
        'rate_PearsonR': r['PearsonR_median'],
        'waveform_CCC': None if w is None else w['waveform_CCC_median'],
        'waveform_MAE': None if w is None else w['waveform_MAE_median'],
        'waveform_DTW': None if w is None else w['waveform_DTW_median'],
        'strict_CCC': s['strict_CCC_median'],
        'strict_MAE': s['strict_MAE_median'],
        'strict_DTW': s['strict_DTW_median'],
        'cycle_ppi_mae_s': s['cycle_ppi_mae_s_median'],
        'NIS_Mean': None if d is None else d['NIS_Mean_median'],
        'NIS_InBand': None if d is None else d['NIS_InBand_median'],
        'Lambda_Mean': None if d is None else d['Lambda_Mean_median'],
        'Stability_Sec': None if d is None else d['Stability_Sec_median'],
    }


def _row_from_gate_table(row: pd.Series, row_id: str, intent: str, source_run: str = 'decoupled_system_gate'):
    return {
        'row_id': row_id,
        'intent': intent,
        'source_run': source_run,
        'method': row.get('method'),
        'rate_MAE': row.get('rate_MAE'),
        'rate_RMSE': row.get('rate_RMSE'),
        'rate_PearsonR': row.get('rate_PearsonR'),
        'waveform_CCC': row.get('waveform_CCC'),
        'waveform_MAE': row.get('waveform_MAE'),
        'waveform_DTW': row.get('waveform_DTW'),
        'strict_CCC': row.get('strict_CCC'),
        'strict_MAE': row.get('strict_MAE'),
        'strict_DTW': row.get('strict_DTW'),
        'cycle_ppi_mae_s': None,
        'NIS_Mean': None,
        'NIS_InBand': None,
        'Lambda_Mean': None,
        'Stability_Sec': None,
    }


def _maybe_append_summary(rows: list[dict], run_path: str | None, method: str, source_run: str, row_id: str, intent: str) -> None:
    if not run_path:
        return
    path = Path(run_path)
    if not path.exists():
        return
    try:
        bundle = _load(path)
        rows.append(_row(bundle, method, source_run, row_id, intent))
    except Exception as exc:
        print(f"  SKIP {row_id}: {exc}")


def _build_from_decoupled_table(decoupled_run: Path, shared_adaptive_run: str | None = None) -> list[dict]:
    table_path = decoupled_run / 'decoupled_system_gate_table.csv'
    if not table_path.exists():
        raise FileNotFoundError(f"Missing decoupled gate table: {table_path}")
    dec_tbl = pd.read_csv(table_path)
    rows: list[dict] = []
    role_to_row = {
        'raw_base_p1dquad': ('base_raw_p1dquad', 'raw single-family reference'),
        'rate_expert_p1dquad_parh': ('parh_rate_expert', 'structured rate expert'),
        'learned_adaptive': ('adaptive_observation_law', 'local observation adaptation control'),
        'learned_staged_routed': ('staged_routed_multihead', 'coupled routed multi-output'),
        'waveform_expert_temporal': ('waveform_expert_only', 'waveform-specialized learned expert'),
        'honest_best_overall': ('decoupled_system', 'current best honest overall system'),
    }
    for role, (row_id, intent) in role_to_row.items():
        hit = dec_tbl[dec_tbl['role'].astype(str).eq(role)]
        if hit.empty:
            print(f"  SKIP {row_id}: role={role} not found in {table_path}")
            continue
        rows.append(_row_from_gate_table(hit.iloc[0], row_id=row_id, intent=intent))
    _maybe_append_summary(
        rows,
        shared_adaptive_run,
        'shared_observation_law__torch',
        'shared_adaptive_gate',
        'shared_observation_law',
        'wrong shared operator assumption',
    )
    return rows


def _paper_ready_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the operator-alignment table focused on paper-facing metrics.

    The strict raw MAE/DTW columns are native-scale dependent and are preserved
    in the dedicated strict companion artifacts, not in this ablation table.
    Fully empty diagnostic columns are also removed so table-ready CSV readers
    do not mistake "not applicable" fields for missing generated results.
    """

    scale_risk_cols = {"strict_MAE", "strict_DTW"}
    keep_cols: list[str] = []
    for col in df.columns:
        if col in scale_risk_cols:
            continue
        series = df[col]
        if series.isna().all():
            continue
        keep_cols.append(col)
    return df.loc[:, keep_cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prod-run')
    ap.add_argument(
        '--shared-adaptive-run',
        help='Optional paired shared/adaptive source run. Omitted by default to avoid mixing incompatible stale gates.',
    )
    ap.add_argument('--resonator-prior-run')
    ap.add_argument('--latent-regularized-run')
    ap.add_argument('--staged-routed-run')
    ap.add_argument('--temporal-run')
    ap.add_argument('--decoupled-run', default=str(ROOT / 'results' / 'cohface_decoupled_system_gate_v2'))
    ap.add_argument('--out-csv', default=str(ROOT / 'paper' / 'tables_ready' / 'T5_operator_alignment_ablation.csv'))
    ap.add_argument('--out-md', default=str(ROOT / 'analysis' / 'cohface_t5_operator_alignment_ablation.md'))
    args = ap.parse_args()

    rows = _build_from_decoupled_table(Path(args.decoupled_run), args.shared_adaptive_run)
    _maybe_append_summary(
        rows,
        args.latent_regularized_run,
        'adaptive_observation_law_latent_regularized__torch',
        'latent_regularized_gate',
        'latent_regularized',
        'adaptive plus weak latent regularization',
    )
    _maybe_append_summary(
        rows,
        args.resonator_prior_run,
        'adaptive_observation_law_resonator_prior__torch',
        'resonator_prior_gate',
        'resonator_prior',
        'adaptive plus resonator prior branch',
    )

    df = _paper_ready_columns(pd.DataFrame(rows))
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    lines = [
        '# T5 Operator-Alignment Ablation',
        '',
        'This is the current intent-aligned ablation under the revised thesis.',
        '',
        'Important provenance note:',
        '- rows come from the strongest available completed reruns for each question',
        '- most rows come from the decoupled-system gate table so they share one live provenance',
        '- `shared_observation_law` is included when its live paired source run is available',
        '- unavailable historical latent/resonator rows are omitted rather than silently copied from stale artifacts',
        '- native-scale strict raw MAE/DTW and fully empty diagnostic fields are omitted from this paper-facing ablation table',
        '- this table is reviewer-facing and mechanistic, not a single unified leaderboard',
        '',
        df.to_string(index=False),
        '',
    ]
    Path(args.out_md).write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
