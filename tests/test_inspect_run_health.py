import json
import os
import tempfile
from pathlib import Path

from scripts.inspect_run_health import _summarize_run


def test_summarize_run_marks_stale_running_job():
    tmp = tempfile.mkdtemp(prefix='inspect_run_health_')
    results_root = Path(tmp)
    run_dir = results_root / 'bundle' / 'testrun'
    (run_dir / 'data').mkdir(parents=True, exist_ok=True)
    (run_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    (run_dir / 'data' / 'a.pkl').write_bytes(b'0')
    (run_dir / 'metrics' / 'metrics_freq_domain_raw.csv').write_text('x\n', encoding='utf-8')
    status = {
        'status': 'running',
        'steps': ['estimate', 'evaluate'],
        'completed_steps': ['estimate'],
        'heartbeat_at': '2000-01-01T00:00:00+00:00',
        'started_at': '2000-01-01T00:00:00+00:00',
        'runtime': {'python_executable': '/tmp/python', 'gpu': {'available': False}},
    }
    with open(run_dir / 'run_status.json', 'w', encoding='utf-8') as fp:
        json.dump(status, fp)

    row = _summarize_run(run_dir / 'run_status.json', results_root, stale_minutes=0.01)
    assert row['run_dir'] == os.path.join('bundle', 'testrun')
    assert row['progress'] == '1/2'
    assert row['pkl_count'] == 1
    assert row['metric_csv_count'] == 1
    assert row['stale'] is True
