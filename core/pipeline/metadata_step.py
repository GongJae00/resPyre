import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

def _load_json(path: Path) -> Optional[Dict]:
    if path is None or not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    except Exception:
        return None

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
        summary = metrics_dir / 'metrics_track_summary.txt'
        if summary.exists():
            artifacts['metrics_summary'] = str(summary.resolve())
        pkl = metrics_dir / 'metrics.pkl'
        if pkl.exists():
            artifacts['metrics_pickle'] = str(pkl.resolve())
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
    
    if plots_dir.exists():
        # Just count plots or list first few
        plots = list(plots_dir.glob("*.html"))
        artifacts['plots_count'] = len(plots)
        artifacts['plots_dir'] = str(plots_dir.resolve())

    return artifacts

def run_metadata_generation(results_dir: str, run_label: str = None, command: str = '', notes: str = ''):
    """
    Generates metadata.json for the run.
    """
    # Locate specific run directory matches if label provided, or assume results_dir IS the run dir if it has data?
    # Runner.py creates a subdirectory `run_label` inside `results_dir`. 
    # But usually results_dir passed to this function is the ROOT results dir.
    
    # Let's check how other steps work.
    # evaluation_step scans subdirs.
    
    search_pattern = os.path.join(results_dir, "*")
    if run_label:
        from core.pipeline.common import _sanitize_run_label
        label = _sanitize_run_label(run_label)
        search_pattern = os.path.join(results_dir, f"{label}_*")

    import glob
    candidate_dirs = glob.glob(search_pattern)
    target_dirs = [d for d in candidate_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'data'))]

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
        if eval_settings:
            payload['eval_settings'] = eval_settings
            gating = eval_settings.get('gating', {})
            payload['gating'] = gating

        metadata_path = run_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"   >> Metadata written to {metadata_path}")
