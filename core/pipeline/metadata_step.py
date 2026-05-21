import json
import os
import subprocess
import pickle
import socket
import getpass
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from core.pipeline.common import (
    _dataset_results_dir,
    _sanitize_run_label,
    resolve_target_run_dirs,
    _atomic_json_dump,
)

def _load_json(path: Path) -> Optional[Dict]:
    if path is None or not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    except Exception:
        return None


def _parse_estimate_meta(meta_obj) -> Dict:
    if isinstance(meta_obj, dict):
        return meta_obj
    if isinstance(meta_obj, str):
        txt = meta_obj.strip()
        if txt.startswith("{") and txt.endswith("}"):
            try:
                out = json.loads(txt)
                if isinstance(out, dict):
                    return out
            except Exception:
                return {}
    return {}


def _collect_method_unused_from_data(run_dir: Path) -> List[str]:
    keys = set()
    data_dir = run_dir / "data"
    if not data_dir.exists():
        return []
    for pkl_path in sorted(data_dir.glob("*.pkl")):
        try:
            with open(pkl_path, "rb") as fp:
                obj = pickle.load(fp)
        except Exception:
            continue
        estimates = obj.get("estimates", []) if isinstance(obj, dict) else []
        for entry in estimates:
            payload = entry.get("estimate", entry) if isinstance(entry, dict) else {}
            if not isinstance(payload, dict):
                continue
            meta = _parse_estimate_meta(payload.get("meta"))
            for key in meta.get("unused_config_keys", []) if isinstance(meta, dict) else []:
                sval = str(key).strip()
                if sval:
                    keys.add(sval)
    return sorted(keys)

def _resolve_run_dirs_allow_incomplete(results_dir: str, run_label: Optional[str]) -> List[str]:
    """Resolve run directories even when `data/` is missing.

    This is used for failure-safe metadata emission.
    """
    if not run_label:
        return []
    label = _sanitize_run_label(run_label)
    if not label:
        return []
    exact = os.path.join(results_dir, label)
    if os.path.isdir(exact):
        return [exact]
    candidates = sorted(
        d for d in Path(results_dir).glob(f"{label}_*")
        if d.is_dir()
    )
    return [str(d) for d in candidates]

def bootstrap_run_dirs(results_dir: str, run_label: Optional[str], dataset_names: Optional[Iterable[str]] = None) -> List[str]:
    """Create deterministic run directories ahead of execution.

    This guarantees a location for run_status/metadata even if estimation fails.
    """
    label = _sanitize_run_label(run_label) if run_label else None
    if not label:
        return []
    names = [str(n) for n in (dataset_names or []) if str(n).strip()]
    single_dataset = len(names) <= 1
    run_dirs: List[str] = []
    if not names:
        run_dirs.append(_dataset_results_dir(results_dir, label))
        return run_dirs
    for ds_name in names:
        dir_name = label if single_dataset else f"{label}_{ds_name.upper()}"
        run_dirs.append(_dataset_results_dir(results_dir, dir_name))
    # preserve order but deduplicate
    seen = set()
    out = []
    for d in run_dirs:
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return out

def write_run_status(
    run_dir: str,
    *,
    status: str,
    command: str = "",
    config_path: str = "",
    steps: Optional[Iterable[str]] = None,
    completed_steps: Optional[Iterable[str]] = None,
    error_summary: str = "",
    error_traceback: str = "",
    heartbeat_only: bool = False,
    extra: Optional[Dict] = None,
) -> str:
    """Write a run_status.json payload to `run_dir`."""
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    status_path = p / "run_status.json"
    prev = _load_json(status_path)
    prev = prev if isinstance(prev, dict) else {}
    started_at = prev.get("started_at")
    prev_run_instance = prev.get("run_instance_started_at")
    prev_completed_at = prev.get("completed_at")
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    status_l = str(status).lower()
    if status_l == "running" and not heartbeat_only:
        run_instance_started_at = now_iso
        completed_at = None
    else:
        run_instance_started_at = prev_run_instance or started_at or now_iso
        if status_l in {"completed", "failed"}:
            completed_at = now_iso
        else:
            completed_at = prev_completed_at
    runtime = prev.get("runtime") if isinstance(prev.get("runtime"), dict) else {}
    if not runtime:
        runtime = capture_runtime_context()
    heartbeat_seq = int(prev.get("heartbeat_seq", 0) or 0) + 1
    payload = {
        "schema_version": "run_status.v2",
        "status": str(status),
        "started_at": started_at or now_iso,
        "run_instance_started_at": run_instance_started_at,
        "completed_at": completed_at,
        "updated_at": now_iso,
        "heartbeat_at": now_iso,
        "heartbeat_seq": heartbeat_seq,
        "duration_sec": _duration_seconds(started_at or now_iso, completed_at),
        "command": command or prev.get("command", ""),
        "config_path": config_path or prev.get("config_path", ""),
        "steps": list(steps or prev.get("steps", [])),
        "completed_steps": list(completed_steps or prev.get("completed_steps", [])),
        "error_summary": error_summary or prev.get("error_summary", ""),
        "error_traceback": error_traceback or prev.get("error_traceback", ""),
        "runtime": runtime,
    }
    if extra:
        payload["extra"] = dict(extra)
    _atomic_json_dump(payload, str(status_path), indent=2)
    return str(status_path.resolve())

def mark_run_status_bulk(
    run_dirs: Iterable[str],
    *,
    status: str,
    command: str = "",
    config_path: str = "",
    steps: Optional[Iterable[str]] = None,
    completed_steps: Optional[Iterable[str]] = None,
    error_summary: str = "",
    error_traceback: str = "",
    heartbeat_only: bool = False,
    extra: Optional[Dict] = None,
) -> List[str]:
    """Write run_status.json for multiple run directories."""
    written = []
    for d in run_dirs:
        written.append(
            write_run_status(
                d,
                status=status,
                command=command,
                config_path=config_path,
                steps=steps,
                completed_steps=completed_steps,
                error_summary=error_summary,
                error_traceback=error_traceback,
                heartbeat_only=heartbeat_only,
                extra=extra,
            )
        )
    return written


def touch_run_status_bulk(run_dirs: Iterable[str]) -> List[str]:
    return mark_run_status_bulk(run_dirs, status="running", heartbeat_only=True)


class RunStatusHeartbeat:
    def __init__(self, run_dirs: Iterable[str], interval_sec: float = 60.0):
        self.run_dirs = [str(d) for d in run_dirs]
        self.interval_sec = max(5.0, float(interval_sec))
        self._stop = threading.Event()
        self._thread = None

    def _loop(self):
        while not self._stop.wait(self.interval_sec):
            try:
                touch_run_status_bulk(self.run_dirs)
            except Exception:
                pass

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, name="run-status-heartbeat", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(2.0, self.interval_sec))
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


def _git_commit(cwd) -> Optional[str]:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(cwd), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _git_branch(cwd) -> Optional[str]:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd=str(cwd), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _gpu_summary() -> List[Dict[str, str]]:
    try:
        out = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=name,driver_version,memory.total,memory.free',
                '--format=csv,noheader,nounits',
            ],
            stderr=subprocess.DEVNULL,
        ).decode('utf-8', errors='ignore').strip()
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4:
            continue
        rows.append({
            'name': parts[0],
            'driver_version': parts[1],
            'memory_total_mib': parts[2],
            'memory_free_mib': parts[3],
        })
    return rows


def capture_runtime_context() -> Dict[str, object]:
    return {
        'host': socket.gethostname(),
        'user': getpass.getuser(),
        'pid': int(os.getpid()),
        'ppid': int(os.getppid()),
        'cwd': os.getcwd(),
        'python_executable': os.sys.executable,
        'python_realpath': os.path.realpath(os.sys.executable),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'pythonpath_env': os.environ.get('PYTHONPATH', ''),
        'pyenv_version_file': (Path(os.getcwd()) / '.python-version').read_text(encoding='utf-8').strip() if (Path(os.getcwd()) / '.python-version').exists() else '',
        'git_commit': _git_commit(Path(os.getcwd())),
        'git_branch': _git_branch(Path(os.getcwd())),
        'gpu': _gpu_summary(),
    }


def _duration_seconds(started_at: Optional[str], completed_at: Optional[str]) -> Optional[float]:
    def _parse(ts: Optional[str]):
        if not ts:
            return None
        try:
            return datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        except Exception:
            return None
    a = _parse(started_at)
    b = _parse(completed_at)
    if a is None or b is None:
        return None
    return max(0.0, float((b - a).total_seconds()))


def _detect_eval_settings(run_dir: Path) -> Optional[Dict]:
    candidates = [
        run_dir / 'metrics' / 'eval_settings.json',
        run_dir / 'eval_settings.json'
    ]
    for path in candidates:
        data = _load_json(path)
        if data:
            return data
    return None

def _default_artifacts(run_dir: Path) -> Dict[str, str]:
    artifacts = {}
    metrics_dir = run_dir / 'metrics'
    logs_dir = run_dir / 'logs'
    plots_dir = run_dir / 'plots'
    
    if metrics_dir.exists():
        summary_time = metrics_dir / 'metrics_time_domain_summary.txt'
        if summary_time.exists():
            artifacts['metrics_time_summary'] = str(summary_time.resolve())
        summary_freq = metrics_dir / 'metrics_freq_domain_summary.txt'
        if summary_freq.exists():
            artifacts['metrics_freq_summary'] = str(summary_freq.resolve())
        pkl_time = metrics_dir / 'metrics_time_domain.pkl'
        if pkl_time.exists():
            artifacts['metrics_time_pickle'] = str(pkl_time.resolve())
        pkl_freq = metrics_dir / 'metrics_freq_domain.pkl'
        if pkl_freq.exists():
            artifacts['metrics_freq_pickle'] = str(pkl_freq.resolve())
        diag_summary = metrics_dir / 'metrics_filter_diagnostics_summary.txt'
        if diag_summary.exists():
            artifacts['metrics_filter_diag_summary'] = str(diag_summary.resolve())
        diag_raw = metrics_dir / 'metrics_filter_diagnostics_raw.csv'
        if diag_raw.exists():
            artifacts['metrics_filter_diag_raw'] = str(diag_raw.resolve())
        diag_pkl = metrics_dir / 'metrics_filter_diagnostics.pkl'
        if diag_pkl.exists():
            artifacts['metrics_filter_diag_pickle'] = str(diag_pkl.resolve())
        calib_split = metrics_dir / 'metrics_filter_calibration_split.csv'
        if calib_split.exists():
            artifacts['metrics_filter_calibration_split_csv'] = str(calib_split.resolve())
        calib_split_all = metrics_dir / 'metrics_filter_calibration_split_all.csv'
        if calib_split_all.exists():
            artifacts['metrics_filter_calibration_split_all_csv'] = str(calib_split_all.resolve())
        calib_split_json = metrics_dir / 'metrics_filter_calibration_split.json'
        if calib_split_json.exists():
            artifacts['metrics_filter_calibration_split_json'] = str(calib_split_json.resolve())
        calib_split_txt = metrics_dir / 'metrics_filter_calibration_split.txt'
        if calib_split_txt.exists():
            artifacts['metrics_filter_calibration_split_txt'] = str(calib_split_txt.resolve())
        resolver_diag = metrics_dir / 'resolver_diag.json'
        if resolver_diag.exists():
            artifacts['frame_log_resolver_diag'] = str(resolver_diag.resolve())
        # Legacy single-file name
        legacy_pkl = metrics_dir / 'metrics.pkl'
        if legacy_pkl.exists():
            artifacts['metrics_pickle_legacy'] = str(legacy_pkl.resolve())
        eval_settings = metrics_dir / 'eval_settings.json'
        if eval_settings.exists():
            artifacts['eval_settings'] = str(eval_settings.resolve())
        evaluation_timing = metrics_dir / 'evaluation_timing.json'
        if evaluation_timing.exists():
            artifacts['evaluation_timing_json'] = str(evaluation_timing.resolve())
    
    if logs_dir.exists():
        csv = logs_dir / 'method_quality.csv'
        if csv.exists():
            artifacts['method_quality_csv'] = str(csv.resolve())
        js = logs_dir / 'method_quality_summary.json'
        if js.exists():
            artifacts['method_quality_json'] = str(js.resolve())
        orphan_csv = logs_dir / 'method_quality_orphan_logs.csv'
        if orphan_csv.exists():
            artifacts['method_quality_orphan_csv'] = str(orphan_csv.resolve())
        frame_manifest = logs_dir / 'frame_log_manifest.json'
        if frame_manifest.exists():
            artifacts['frame_log_manifest'] = str(frame_manifest.resolve())
        cfg_usage = logs_dir / 'config_usage.json'
        if cfg_usage.exists():
            artifacts['config_usage_json'] = str(cfg_usage.resolve())

    status_json = run_dir / 'run_status.json'
    if status_json.exists():
        artifacts['run_status'] = str(status_json.resolve())
    
    if plots_dir.exists():
        # Count all plot-like artifacts recursively (PNG + HTML + PDF).
        plots = (
            list(plots_dir.rglob("*.png")) +
            list(plots_dir.rglob("*.html")) +
            list(plots_dir.rglob("*.pdf"))
        )
        artifacts['plots_count'] = len(plots)
        artifacts['plots_dir'] = str(plots_dir.resolve())

    eda_dir = run_dir / 'eda'
    if eda_dir.exists():
        summary_json = eda_dir / 'innovation_summary.json'
        summary_csv = eda_dir / 'innovation_summary.csv'
        skipped_json = eda_dir / 'innovation_eda_skipped_logs.json'
        if summary_json.exists():
            artifacts['eda_innovation_summary_json'] = str(summary_json.resolve())
        if summary_csv.exists():
            artifacts['eda_innovation_summary_csv'] = str(summary_csv.resolve())
        if skipped_json.exists():
            artifacts['eda_innovation_skipped_json'] = str(skipped_json.resolve())
        artifacts['eda_dir'] = str(eda_dir.resolve())

    return artifacts

def run_metadata_generation(
    results_dir: str,
    run_label: str = None,
    command: str = '',
    notes: str = '',
    allow_incomplete: bool = False,
):
    """
    Generates metadata.json for the run.
    """
    # Locate specific run directory matches if label provided, or assume results_dir IS the run dir if it has data?
    # Runner.py creates a subdirectory `run_label` inside `results_dir`. 
    # But usually results_dir passed to this function is the ROOT results dir.
    
    # Let's check how other steps work.
    # evaluation_step scans subdirs.
    
    target_dirs = resolve_target_run_dirs(results_dir, run_label)
    if not target_dirs and allow_incomplete:
        target_dirs = _resolve_run_dirs_allow_incomplete(results_dir, run_label)

    if not target_dirs:
        print(f"> Metadata: No result directories found matching '{run_label}'")
        return

    print(f"\n> Generating Metadata for {len(target_dirs)} runs...")

    for d_dir in target_dirs:
        run_dir = Path(d_dir).resolve()
        
        payload = {
            'created': datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            'run_dir': str(run_dir),
            'command': command,
            'notes': notes,
            'git_commit': _git_commit(run_dir), # Try to get git commit from CWD of run or project root
            'git_branch': _git_branch(run_dir),
            'artifacts': _default_artifacts(run_dir),
            'paths': {
                'metrics': str((run_dir / 'metrics').resolve()),
                'data': str((run_dir / 'data').resolve()),
            }
        }
        
        eval_settings = _detect_eval_settings(run_dir)
        unused_merged = []
        if eval_settings:
            payload['eval_settings'] = eval_settings
            gating = eval_settings.get('gating', {})
            payload['gating'] = gating
            unused_keys = eval_settings.get('unused_config_keys', [])
            if isinstance(unused_keys, list):
                unused_merged.extend([str(x) for x in unused_keys])

        cfg_usage = _load_json(run_dir / 'logs' / 'config_usage.json')
        if isinstance(cfg_usage, dict):
            payload['config_usage'] = cfg_usage
            cfg_unused = cfg_usage.get('unused_config_keys', [])
            if isinstance(cfg_unused, list):
                unused_merged.extend([str(x) for x in cfg_unused])
            method_unused = cfg_usage.get('method_unused_config_keys', [])
            if isinstance(method_unused, list):
                unused_merged.extend([str(x) for x in method_unused])
            top_unused = cfg_usage.get('top_level_unused_keys', [])
            if isinstance(top_unused, list):
                unused_merged.extend([f"top.{str(x)}" for x in top_unused])

        # Fallback aggregation from per-estimate metadata, even when config_usage
        # was not generated (e.g., partial/failed runs).
        method_unused_data = _collect_method_unused_from_data(run_dir)
        if method_unused_data:
            payload['method_unused_config_keys'] = method_unused_data
            unused_merged.extend(method_unused_data)

        run_status = _load_json(run_dir / 'run_status.json')
        if isinstance(run_status, dict):
            payload['run_status'] = run_status
            payload['status'] = run_status.get('status')
            runtime = run_status.get('runtime')
            if isinstance(runtime, dict):
                payload['runtime'] = runtime
            payload['heartbeat_at'] = run_status.get('heartbeat_at')
            payload['duration_sec'] = run_status.get('duration_sec')
        if unused_merged:
            seen = set()
            ordered = []
            for x in unused_merged:
                if x in seen:
                    continue
                seen.add(x)
                ordered.append(x)
            payload['unused_config_keys'] = ordered

        metadata_path = run_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"   >> Metadata written to {metadata_path}")
