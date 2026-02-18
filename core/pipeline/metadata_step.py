import json
import os
import subprocess
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from core.pipeline.common import (
    _dataset_results_dir,
    _sanitize_run_label,
    resolve_target_run_dirs,
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
) -> str:
    """Write a run_status.json payload to `run_dir`."""
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    status_path = p / "run_status.json"
    prev = _load_json(status_path)
    started_at = None
    prev_run_instance = None
    if isinstance(prev, dict):
        started_at = prev.get("started_at")
        prev_run_instance = prev.get("run_instance_started_at")
    now_iso = datetime.utcnow().isoformat() + "Z"
    if str(status).lower() == "running":
        run_instance_started_at = now_iso
    else:
        run_instance_started_at = prev_run_instance or started_at or now_iso
    payload = {
        "schema_version": "run_status.v1",
        "status": str(status),
        "started_at": started_at or now_iso,
        "run_instance_started_at": run_instance_started_at,
        "updated_at": now_iso,
        "command": command,
        "config_path": config_path,
        "steps": list(steps or []),
        "completed_steps": list(completed_steps or []),
        "error_summary": error_summary or "",
        "error_traceback": error_traceback or "",
    }
    with open(status_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
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
            )
        )
    return written

def _git_commit(cwd) -> Optional[str]:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(cwd), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None

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
        # Legacy single-file name
        legacy_pkl = metrics_dir / 'metrics.pkl'
        if legacy_pkl.exists():
            artifacts['metrics_pickle_legacy'] = str(legacy_pkl.resolve())
        eval_settings = metrics_dir / 'eval_settings.json'
        if eval_settings.exists():
            artifacts['eval_settings'] = str(eval_settings.resolve())
    
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
            'created': datetime.utcnow().isoformat() + 'Z',
            'run_dir': str(run_dir),
            'command': command,
            'notes': notes,
            'git_commit': _git_commit(run_dir), # Try to get git commit from CWD of run or project root
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
