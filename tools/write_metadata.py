#!/usr/bin/env python3
"""
Utility script to generate results/<run>/metadata.json after evaluate/metrics.

Example:
  python tools/write_metadata.py --run results/cohface_motion_oscillator \
    --command "python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics"
"""

from __future__ import annotations

import argparse
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


def _git_commit() -> Optional[str]:
	try:
		return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(Path(__file__).resolve().parent.parent), stderr=subprocess.DEVNULL).decode().strip()
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
		ms = logs_dir / 'methods_seen.txt'
		if ms.exists():
			artifacts['methods_seen'] = str(ms.resolve())
	return artifacts


def main():
	parser = argparse.ArgumentParser(description="Generate metadata.json for a results run directory")
	parser.add_argument('--run', required=True, help='results/<run_label> path')
	parser.add_argument('--command', help='Command used to produce this run (string)')
	parser.add_argument('--notes', help='Optional notes to embed in metadata')
	args = parser.parse_args()

	run_dir = Path(args.run).resolve()
	if not run_dir.exists():
		raise SystemExit(f"Run directory {run_dir} does not exist")

	payload = {
		'created': datetime.utcnow().isoformat() + 'Z',
		'run_dir': str(run_dir),
		'command': args.command or '',
		'notes': args.notes or '',
		'git_commit': _git_commit(),
		'artifacts': _default_artifacts(run_dir),
		'paths': {
			'autotune_dir': str((Path('runs/autotune')).resolve()),
			'em_params_dir': str((Path('runs/em_params')).resolve()),
			'em_logs': str((Path('runs/em_logs')).resolve()),
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
	print(f"> metadata written to {metadata_path}")


if __name__ == '__main__':
	main()
