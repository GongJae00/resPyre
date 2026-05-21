import json
import tempfile
from pathlib import Path
import importlib.util


def _load_prune_module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'prune_results.py'
    spec = importlib.util.spec_from_file_location('_prune_results_mod', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_running_status_is_protected_by_default():
    mod = _load_prune_module()
    tmp = Path(tempfile.mkdtemp(prefix='prune_policy_'))
    run = tmp / 'testrun' / 'payload'
    run.mkdir(parents=True, exist_ok=True)
    status = {
        'status': 'running',
        'heartbeat_at': '2000-01-01T00:00:00Z',
    }
    (run / 'run_status.json').write_text(json.dumps(status), encoding='utf-8')
    assert mod.is_running(tmp / 'testrun') is True


def test_stale_running_requires_explicit_override():
    mod = _load_prune_module()
    tmp = Path(tempfile.mkdtemp(prefix='prune_policy_stale_'))
    run = tmp / 'testrun' / 'payload'
    run.mkdir(parents=True, exist_ok=True)
    status = {
        'status': 'running',
        'heartbeat_at': '2000-01-01T00:00:00Z',
    }
    (run / 'run_status.json').write_text(json.dumps(status), encoding='utf-8')
    assert mod.is_running(tmp / 'testrun', allow_stale_running=True, stale_minutes=1.0) is False
