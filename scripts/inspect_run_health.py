#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _bytes_human(num: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    value = float(max(int(num), 0))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num}B"


def _parse_iso8601(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        if ts.endswith('Z'):
            ts = ts[:-1] + '+00:00'
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _minutes_since(ts: Optional[str]) -> Optional[float]:
    dt = _parse_iso8601(ts)
    if dt is None:
        return None
    return (_utc_now() - dt.astimezone(timezone.utc)).total_seconds() / 60.0


def _count_pkls(run_dir: Path) -> int:
    data_dir = run_dir / 'data'
    if not data_dir.exists():
        return 0
    return sum(1 for _ in data_dir.glob('*.pkl'))


def _count_metric_csvs(run_dir: Path) -> int:
    metrics_dir = run_dir / 'metrics'
    if not metrics_dir.exists():
        return 0
    return sum(1 for _ in metrics_dir.glob('*.csv'))


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob('*'):
        if not child.is_file():
            continue
        try:
            total += child.stat().st_size
        except OSError:
            continue
    return total


def _eval_timing_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / 'metrics' / 'evaluation_timing.json'
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    timings = payload.get('timings_sec') or {}
    if not isinstance(timings, dict) or not timings:
        return {}
    hotspots = {
        k: float(v) for k, v in timings.items()
        if k != 'save_metrics_sec' and isinstance(v, (int, float))
    }
    if not hotspots:
        return {}
    hot_key, hot_val = max(hotspots.items(), key=lambda kv: kv[1])
    return {
        'eval_total_sec': float(payload.get('total_sec', 0.0) or 0.0),
        'eval_hotspot': str(hot_key),
        'eval_hotspot_sec': float(hot_val),
    }


def _latest_mtime_iso(run_dir: Path) -> Optional[str]:
    latest: Optional[float] = None
    for child in run_dir.rglob('*'):
        if not child.is_file():
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        latest = mtime if latest is None else max(latest, mtime)
    if latest is None:
        return None
    return datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()


def _load_status(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def _device_hint(runtime: Dict[str, Any]) -> str:
    gpu = runtime.get('gpu') or {}
    if isinstance(gpu, list):
        if gpu:
            first = gpu[0]
            if isinstance(first, dict) and first.get('name'):
                return f"gpu:{first['name']}"
        gpu = {}
    if isinstance(gpu, dict) and gpu.get('available') and gpu.get('name'):
        return f"gpu:{gpu['name']}"
    py_exec = runtime.get('python_executable') or ''
    if py_exec:
        return f"py:{Path(py_exec).name}"
    return '-'


def _summarize_run(status_path: Path, results_root: Path, stale_minutes: float) -> Dict[str, Any]:
    run_dir = status_path.parent
    status = _load_status(status_path)
    heartbeat_source = status.get('heartbeat_at') or status.get('updated_at') or status.get('completed_at')
    heartbeat_age_min = _minutes_since(heartbeat_source)
    started_age_min = _minutes_since(status.get('started_at'))
    steps = status.get('steps') or []
    completed_steps = status.get('completed_steps') or []
    runtime = status.get('runtime') or {}
    sub_sizes = {name: _dir_size_bytes(run_dir / name) for name in ('data', 'metrics', 'aux', 'plots', 'logs')}
    row = {
        'run_dir': str(run_dir.relative_to(results_root)),
        'status': status.get('status', 'unknown'),
        'heartbeat_age_min': None if heartbeat_age_min is None else round(heartbeat_age_min, 1),
        'started_age_min': None if started_age_min is None else round(started_age_min, 1),
        'progress': f"{len(completed_steps)}/{len(steps)}",
        'pkl_count': _count_pkls(run_dir),
        'metric_csv_count': _count_metric_csvs(run_dir),
        'last_file_update': _latest_mtime_iso(run_dir),
        'device_hint': _device_hint(runtime),
        'duration_sec': status.get('duration_sec'),
        'size_bytes': int(sum(sub_sizes.values())),
        'size_h': _bytes_human(sum(sub_sizes.values())),
        'data_size_h': _bytes_human(sub_sizes.get('data', 0)),
        'aux_size_h': _bytes_human(sub_sizes.get('aux', 0)),
        'metrics_size_h': _bytes_human(sub_sizes.get('metrics', 0)),
    }
    row.update(_eval_timing_summary(run_dir))
    row['stale'] = (
        row['status'] == 'running'
        and heartbeat_age_min is not None
        and heartbeat_age_min > stale_minutes
    )
    return row


def _format_cell(value: Any) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        if math.isnan(value):
            return '-'
        return f'{value:.1f}'
    return str(value)


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print('No run_status.json files found.')
        return
    columns = [
        'run_dir',
        'status',
        'progress',
        'pkl_count',
        'metric_csv_count',
        'size_h',
        'aux_size_h',
        'heartbeat_age_min',
        'stale',
        'device_hint',
        'eval_hotspot',
    ]
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(_format_cell(row.get(col))))
    header = '  '.join(col.ljust(widths[col]) for col in columns)
    print(header)
    print('  '.join('-' * widths[col] for col in columns))
    for row in rows:
        print('  '.join(_format_cell(row.get(col)).ljust(widths[col]) for col in columns))


def main() -> None:
    ap = argparse.ArgumentParser(description='Inspect run_status heartbeat/progress across results bundles.')
    ap.add_argument('--results-root', default='results')
    ap.add_argument('--stale-minutes', type=float, default=30.0)
    ap.add_argument('--json', action='store_true', help='Emit JSON instead of a text table.')
    args = ap.parse_args()

    results_root = Path(args.results_root).resolve()
    status_paths = sorted(results_root.rglob('run_status.json'))
    rows = [_summarize_run(path, results_root, args.stale_minutes) for path in status_paths]
    rows.sort(key=lambda r: (r['status'] != 'running', r['run_dir']))

    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        _print_table(rows)


if __name__ == '__main__':
    main()
