#!/usr/bin/env python3
"""Optuna-based tuner for resPyre oscillator heads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pickle

from config_loader import load_config
from riv.optim.em_kalman import EMKalmanTrainer, save_em_params, log_em_result

try:
	import mlflow  # type: ignore
except Exception:  # pragma: no cover - optional dependency
	mlflow = None

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cohface_motion_oscillator.json"

BASE_METHODS = {
    'of_farneback',
    'dof',
    'profile1d_linear',
    'profile1d_quadratic',
    'profile1d_cubic'
}
SUFFIX_FAMILY = {
    '__ukffreq': 'ukffreq',
    '__pll': 'pll',
    '__kfstd': 'kfstd',
    '__spec_ridge': 'spec_ridge'
}
ALLOWABLE_SUFFIXES = tuple(SUFFIX_FAMILY.keys())
DEFAULT_WEIGHTS = {
    'mae': 0.85,
    'rmse': 0.15
}
TRIAL_CSV_FIELDS = [
    'trial', 'objective', 'MAE_bpm_med', 'RMSE_bpm_med', 'R_mean', 'SNR_med',
    'edge_sat', 'nan_rate', 'jerk_hzps', 'em_q', 'em_r', 'em_ll', 'params'
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuner for resPyre oscillators")
    parser.add_argument('-c', '--config', default=str(DEFAULT_CONFIG), help='Base config JSON (default: %(default)s)')
    parser.add_argument('--output', default='runs/optuna', help='Output root for studies')
    parser.add_argument('--methods', nargs='*', help='Subset of methods to tune (default: allowlist 20)')
    parser.add_argument('--families', nargs='*', choices=sorted(set(SUFFIX_FAMILY.values())), help='Restrict to specific families')
    parser.add_argument('--n-trials', type=int, default=50, help='Trials per method')
    parser.add_argument('--timeout', type=int, help='Optional timeout (seconds) per method')
    parser.add_argument('--sampler-seed', type=int, default=42, help='Seed for Optuna sampler')
    parser.add_argument('--no-prune', action='store_true', help='Disable Optuna pruner')
    parser.add_argument('--keep-artifacts', action='store_true', help='Keep per-trial results directories')
    parser.add_argument('--skip-leaderboard', action='store_true', help='Skip leaderboard/bundle generation')
    parser.add_argument('--bundle', action='store_true', help='Force bundle creation even if leaderboard skipped')
    parser.add_argument('--list', action='store_true', help='List tuned methods and exit')
    parser.add_argument('--num-shards', type=int, default=1, help='Split allowlist into this many shards (default: 1)')
    parser.add_argument('--shard-index', type=int, default=0, help='Shard index to run (0-based)')
    parser.add_argument('--em-mode', choices=['off', 'trial', 'best'], default='off', help='Run EM-based Kalman gain learning per trial or only for the best config')
    parser.add_argument('--mlflow-uri', help='Optional MLflow tracking URI for trial logging')
    parser.add_argument('--mlflow-experiment', default='respyre-optuna', help='MLflow experiment name (default: %(default)s)')
    weight_help_msg = (
        "Objective = 0.85*MAE + 0.15*RMSE (기본). MAE가 1차 판단지표, RMSE는 대형오류 억제를 위한 보조 항. 모든 값은 bpm 단위."
    )
    parser.add_argument(
        '--weight-mae',
        dest='weight_mae',
        type=float,
        help=f"{weight_help_msg} MAE 가중치 덮어쓰기는 이 옵션을 사용하세요 (기본: 0.85)."
    )
    parser.add_argument(
        '--weight-rmse',
        dest='weight_rmse',
        type=float,
        help=f"{weight_help_msg} RMSE 가중치 덮어쓰기는 이 옵션을 사용하세요 (기본: 0.15)."
    )
    return parser.parse_args()


def _extract_method_names(method_entries: Sequence) -> List[str]:
    names = []
    for entry in method_entries:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict) and entry.get('name'):
            names.append(entry['name'])
    return names


def _method_family(method: str) -> Optional[str]:
    lname = method.lower()
    for suffix, family in SUFFIX_FAMILY.items():
        if lname.endswith(suffix):
            return family
    return None


def _allowlist_methods(method_names: Sequence[str], explicit: Optional[Sequence[str]] = None) -> List[str]:
    filtered = []
    selection = set(n.lower() for n in explicit) if explicit else None
    for name in method_names:
        parts = name.split('__', 1)
        base = parts[0].lower()
        has_suffix = len(parts) > 1
        if base in BASE_METHODS and not has_suffix:
            continue
        family = _method_family(name)
        if not family:
            continue
        if selection and name.lower() not in selection:
            continue
        filtered.append(name)
    dedup = []
    seen = set()
    for name in filtered:
        if name.lower() in seen:
            continue
        dedup.append(name)
        seen.add(name.lower())
    if len(dedup) != 20:
        print(f"> Warning: allowlist resolved to {len(dedup)} methods (expected 20)")
    return dedup


def _group_methods_by_base(methods: Sequence[str]) -> List[List[str]]:
    groups: List[List[str]] = []
    order: List[str] = []
    by_base: Dict[str, List[str]] = {}
    for name in methods:
        base = name.split('__', 1)[0].lower()
        if base not in by_base:
            by_base[base] = []
            order.append(base)
        by_base[base].append(name)
    for base in order:
        groups.append(by_base.get(base, []))
    return groups


def _select_shard_methods(methods: Sequence[str], num_shards: int, shard_index: int) -> List[str]:
    if num_shards <= 1:
        return list(methods)
    num_shards = max(1, num_shards)
    shard_index = max(0, min(shard_index, num_shards - 1))
    groups = _group_methods_by_base(methods)
    shards: List[List[str]] = [[] for _ in range(num_shards)]
    for idx, group in enumerate(groups):
        target = idx % num_shards
        shards[target].extend(group)
    return shards[shard_index]


@dataclass
class ParamSpec:
    path: str
    kind: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[Sequence] = None
    log: bool = False


# WHY: Adult respiration (0.1–0.4 Hz) at 64 Hz → gentle drift, narrow noise floors; trim non-identifiable/unused params.
FAMILY_PARAM_SPACE: Dict[str, List[ParamSpec]] = {
    'ukffreq': [
        ParamSpec('oscillator.qf', 'float', 1.5e-4, 6e-4, log=True),
        ParamSpec('oscillator.rv_floor', 'float', 0.02, 0.06),
        ParamSpec('oscillator.tau_env', 'float', 24.0, 42.0),
        ParamSpec('oscillator.qx', 'float', 5e-5, 3e-4, log=True),
        ParamSpec('oscillator.ukf_alpha', 'float', 0.05, 0.12),
        ParamSpec('oscillator.ukf_beta', 'choice', choices=[2.0]),
    ],
    'pll': [
        ParamSpec('oscillator.pll_zeta', 'float', 0.7, 1.0),
        ParamSpec('oscillator.pll_ttrack', 'float', 4.0, 8.0),
        ParamSpec('oscillator.pll_kp_min', 'float', 5e-5, 5e-3, log=True),
        ParamSpec('oscillator.pll_ki_min', 'float', 5e-6, 5e-4, log=True),
    ],
    'kfstd': [
        ParamSpec('oscillator.qx', 'float', 1e-4, 6e-4, log=True),
        ParamSpec('oscillator.rv_floor', 'float', 0.02, 0.05, log=True),
        ParamSpec('oscillator.post_smooth_alpha', 'float', 0.85, 0.95),
    ],
    'spec_ridge': [
        ParamSpec('oscillator.stft_win', 'int', 10, 14),
        ParamSpec('oscillator.spec_overlap', 'float', 0.88, 0.93),
        ParamSpec('oscillator.spec_nfft_factor', 'choice', choices=[1, 2]),
        ParamSpec('oscillator.spec_peak_smooth_len', 'int', 1, 3),
        ParamSpec('oscillator.spec_subbin_interp', 'choice', choices=['parabolic', 'none']),
        ParamSpec('oscillator.ridge_penalty', 'float', 150.0, 350.0),
    ],
}

# WHY: Seed each head near physiologic mid-points so Optuna explores narrow, safe bands.
FAMILY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    'ukffreq': {
        'oscillator.qf': 3e-4,
        'oscillator.qx': 1e-4,
        'oscillator.rv_floor': 0.03,
        'oscillator.tau_env': 32.0,
        'oscillator.ukf_alpha': 0.08,
        'oscillator.ukf_beta': 2.0,
        'oscillator.ukf_kappa': 0.0,
    },
    'pll': {
        'oscillator.pll_zeta': 0.85,
        'oscillator.pll_ttrack': 5.5,
        'oscillator.pll_kp_min': 3e-4,
        'oscillator.pll_ki_min': 2e-5,
    },
    'kfstd': {
        'oscillator.qx': 2.5e-4,
        'oscillator.rv_floor': 0.03,
        'oscillator.post_smooth_alpha': 0.9,
    },
    'spec_ridge': {
        'oscillator.stft_win': 12,
        'oscillator.spec_overlap': 0.92,
        'oscillator.spec_nfft_factor': 2,
        'oscillator.spec_peak_smooth_len': 3,
        'oscillator.spec_subbin_interp': 'parabolic',
        'oscillator.ridge_penalty': 250.0,
    },
}


def suggest_params(trial: optuna.trial.Trial, family: str) -> Dict[str, float]:
    specs = FAMILY_PARAM_SPACE.get(family, [])
    params: Dict[str, float] = {}
    for spec in specs:
        name = f"param:{spec.path}"
        if spec.kind == 'choice' and spec.choices is not None:
            params[spec.path] = trial.suggest_categorical(name, list(spec.choices))
        elif spec.kind == 'int':
            params[spec.path] = int(trial.suggest_int(name, int(spec.low), int(spec.high)))
        elif spec.kind == 'float':
            params[spec.path] = float(trial.suggest_float(name, float(spec.low), float(spec.high), log=spec.log))
        else:
            raise ValueError(f"Unsupported ParamSpec kind: {spec.kind}")
    return params


def _set_nested(target: Dict, path: str, value) -> None:
    keys = path.split('.')
    node = target
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _stringify_percentile(arr: List[float], fn=np.nanmedian) -> float:
    if not arr:
        return float('nan')
    vals = np.asarray(arr, dtype=np.float64)
    if vals.size == 0:
        return float('nan')
    return float(fn(vals))


def locate_metrics_file(results_root: Path) -> Optional[Path]:
    if not results_root.exists():
        return None
    for path in results_root.rglob('metrics.pkl'):
        return path
    for path in results_root.rglob('metrics_1w.pkl'):
        return path
    return None


def load_method_records(metrics_path: Path, method: str):
    with open(metrics_path, 'rb') as fp:
        metric_names, method_metrics = pickle.load(fp)
    records = method_metrics.get(method, []) if isinstance(method_metrics, dict) else []
    return metric_names, records


def _extract_metric(values: List[float], agg=np.nanmedian) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float('nan')
    return float(agg(arr))


def summarize_records(metric_names: Sequence[str], records: Sequence[Dict]) -> Dict[str, float]:
    name_to_idx = {name: idx for idx, name in enumerate(metric_names)}
    mae_vals: List[float] = []
    rmse_vals: List[float] = []
    r_vals: List[float] = []
    snr_vals: List[float] = []
    edge_vals: List[float] = []
    nan_vals: List[float] = []
    jerk_vals: List[float] = []
    idx_mae = name_to_idx.get('MAE')
    idx_r = name_to_idx.get('R') or name_to_idx.get('PCC') or name_to_idx.get('PearsonR')
    idx_snr = name_to_idx.get('SNR')

    def _ae_samples_bpm(record):
        arr = record.get('ae_bpm')
        if arr is not None:
            arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        else:
            arr = record.get('ae_hz')
            if arr is not None:
                arr = np.asarray(arr, dtype=np.float64).reshape(-1) * 60.0
        if arr is None:
            return None
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return arr

    def _median_abs_error(samples, metrics):
        if samples is not None:
            return float(np.nanmedian(samples))
        if idx_mae is not None and len(metrics) > idx_mae:
            return float(metrics[idx_mae])
        return None

    def _rmse_error(samples):
        if samples is None:
            return None
        rmse = float(np.sqrt(np.mean(np.square(samples))))
        if not np.isfinite(rmse):
            return None
        return rmse

    def _corr_value(record, metrics, idx, keys):
        for key in keys:
            if key in record and record[key] is not None:
                return float(record[key])
        if idx is not None and len(metrics) > idx:
            return float(metrics[idx])
        return None

    def _edge_fraction(record):
        track_stats = record.get('track_stats') or {}
        edge_val = track_stats.get('edge_saturation_fraction')
        if edge_val is None:
            edge_val = track_stats.get('saturation_fraction')
        return float(edge_val) if edge_val is not None else None

    def _nan_rate(record, pred_arr):
        track_stats = record.get('track_stats') or {}
        for key in ('nan_rate', 'nan_fraction', 'nan_frac'):
            if key in track_stats and track_stats[key] is not None:
                return float(track_stats[key])
        if record.get('nan_rate') is not None:
            return float(record['nan_rate'])
        if pred_arr is not None and pred_arr.size:
            mask = np.isnan(pred_arr)
            if mask.size:
                return float(np.mean(mask))
        return None

    def _sample_dt(record):
        times = record.get('times_est')
        if times is None:
            return None
        arr = np.asarray(times, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            return None
        diffs = np.diff(arr)
        finite = diffs[np.isfinite(diffs)]
        if finite.size == 0:
            return None
        dt = float(np.nanmedian(finite))
        if not np.isfinite(dt) or dt <= 0:
            return None
        return dt

    def _jerk_hzps(pred_arr, dt):
        if pred_arr is None or pred_arr.size == 0 or dt is None:
            return None
        freq = np.asarray(pred_arr, dtype=np.float64).reshape(-1) / 60.0
        freq = freq[np.isfinite(freq)]
        if freq.size < 2:
            return None
        jerks = np.abs(np.diff(freq)) / dt
        jerks = jerks[np.isfinite(jerks)]
        if jerks.size == 0:
            return None
        return float(np.nanmedian(jerks))

    for record in records:
        metrics = record.get('metrics') or []
        ae_samples = _ae_samples_bpm(record)
        mae_stat = _median_abs_error(ae_samples, metrics)
        if mae_stat is not None:
            mae_vals.append(mae_stat)
        rmse_stat = _rmse_error(ae_samples)
        if rmse_stat is not None:
            rmse_vals.append(rmse_stat)
        r_stat = _corr_value(record, metrics, idx_r, ('r', 'pcc', 'pearson'))
        if r_stat is not None:
            r_vals.append(r_stat)
        if idx_snr is not None and idx_snr < len(metrics):
            try:
                snr_stat = float(metrics[idx_snr])
            except Exception:
                snr_stat = None
            if snr_stat is not None and np.isfinite(snr_stat):
                snr_vals.append(snr_stat)
        edge_stat = _edge_fraction(record)
        if edge_stat is not None:
            edge_vals.append(edge_stat)
        pred_arr = None
        pair = record.get('pair') or []
        if isinstance(pair, (list, tuple)) and pair and pair[0] is not None:
            pred_arr = np.asarray(pair[0], dtype=np.float64).reshape(-1)
            if pred_arr.size == 0:
                pred_arr = None
        nan_stat = _nan_rate(record, pred_arr)
        if nan_stat is not None:
            nan_vals.append(nan_stat)
        dt = _sample_dt(record)
        jerk_stat = _jerk_hzps(pred_arr, dt)
        if jerk_stat is not None:
            jerk_vals.append(jerk_stat)
    return {
        'MAE_bpm_med': _extract_metric(mae_vals, np.nanmedian),
        'RMSE_bpm_med': _extract_metric(rmse_vals, np.nanmedian),
        'R_mean': _extract_metric(r_vals, np.nanmean),
        'SNR_med': _extract_metric(snr_vals, np.nanmedian),
        'edge_sat': _extract_metric(edge_vals, np.nanmean),
        'nan_rate': _extract_metric(nan_vals, np.nanmean),
        'jerk_hzps': _extract_metric(jerk_vals, np.nanmedian),
    }


def combine_objective(summary: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    terms = {}
    objective = 0.0
    def _add_term(key: str, value: float, transform=None):
        nonlocal objective
        weight = float(weights.get(key, 0.0))
        if weight == 0.0 or not np.isfinite(value):
            terms[key] = float('nan')
            return
        term_val = transform(value) if transform else value
        objective += weight * term_val
        terms[key] = term_val

    _add_term('mae', summary.get('MAE_bpm_med', float('nan')))
    _add_term('rmse', summary.get('RMSE_bpm_med', float('nan')))

    if not np.isfinite(objective):
        objective = 1e6
    return objective, terms


@dataclass
class StudyArgs:
    base_cfg: Dict
    config_path: Path
    output_root: Path
    weights: Dict[str, float]
    n_trials: int
    timeout: Optional[int]
    sampler_seed: int
    pruner_enabled: bool
    keep_artifacts: bool
    dataset_name: str
    em_mode: str
    mlflow_uri: Optional[str]
    mlflow_experiment: Optional[str]


class MethodStudy:
    def __init__(self, method: str, family: str, args: StudyArgs):
        self.method = method
        self.family = family
        self.args = args
        self.study_dir = args.output_root / family / method
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.study_dir / 'artifacts'
        self.artifacts_dir.mkdir(exist_ok=True)
        self.trials_csv = self.study_dir / 'trials.csv'
        self.best_path = self.study_dir / 'best.json'
        self.dataset_name = args.dataset_name or 'UNKNOWN'
        self.em_mode = args.em_mode or 'off'
        self.em_trainer = EMKalmanTrainer() if self.em_mode != 'off' else None
        self.mlflow_uri = args.mlflow_uri
        self.mlflow_experiment = args.mlflow_experiment
        self.mlflow_enabled = False
        if self.mlflow_uri and mlflow is not None:
            try:
                mlflow.set_tracking_uri(self.mlflow_uri)
                mlflow.set_experiment(self.mlflow_experiment or 'respyre-optuna')
                self.mlflow_enabled = True
            except Exception as exc:  # pragma: no cover - optional path
                print(f"> Warning: failed to configure MLflow ({exc}); logging disabled.")
        elif self.mlflow_uri and mlflow is None:
            print("> Warning: mlflow package not installed; MLflow logging disabled.")

    def optimize(self):
        sampler = optuna.samplers.TPESampler(seed=self.args.sampler_seed)
        pruner = None if not self.args.pruner_enabled else optuna.pruners.MedianPruner(n_startup_trials=max(1, int(self.args.n_trials * 0.1)))
        storage = f"sqlite:///{(self.study_dir / 'study.db').as_posix()}"
        study = optuna.create_study(
            study_name=self.method,
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True
        )
        study.optimize(self._objective, n_trials=self.args.n_trials, timeout=self.args.timeout, gc_after_trial=True)

    def _objective(self, trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial, self.family)
        summary, em_result, trial_root = self._run_trial(trial.number, params)
        objective, _ = combine_objective(summary, self.args.weights)
        self._record_trial(trial.number, objective, summary, params, em_result)
        self._update_best(objective, summary, params, em_result, trial_root)
        self._log_mlflow(trial.number, objective, summary, params, em_result)
        self._cleanup_trial(trial_root)
        return objective

    def _record_trial(self, trial_num: int, objective: float, summary: Dict[str, float], params: Dict[str, float], em_result: Optional[Dict]):
        row = {
            'trial': trial_num,
            'objective': objective,
            'MAE_bpm_med': summary.get('MAE_bpm_med'),
            'RMSE_bpm_med': summary.get('RMSE_bpm_med'),
            'R_mean': summary.get('R_mean'),
            'SNR_med': summary.get('SNR_med'),
            'edge_sat': summary.get('edge_sat'),
            'nan_rate': summary.get('nan_rate'),
            'jerk_hzps': summary.get('jerk_hzps'),
            'em_q': em_result.get('q') if em_result else None,
            'em_r': em_result.get('r') if em_result else None,
            'em_ll': em_result.get('ll') if em_result else None,
            'params': json.dumps(params, sort_keys=True)
        }
        write_header = not self.trials_csv.exists()
        with open(self.trials_csv, 'a', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=TRIAL_CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _update_best(self, objective: float, summary: Dict[str, float], params: Dict[str, float], em_result: Optional[Dict], trial_root: Optional[Path]):
        best = None
        if self.best_path.exists():
            try:
                with open(self.best_path, 'r', encoding='utf-8') as fp:
                    best = json.load(fp)
            except Exception:
                best = None
        if best and best.get('objective') <= objective:
            return
        payload = {
            'method': self.method,
            'family': self.family,
            'objective': objective,
            'metrics': summary,
            'params': params,
            'created': datetime.utcnow().isoformat() + 'Z'
        }
        with open(self.best_path, 'w', encoding='utf-8') as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        if self.em_mode in ('trial', 'best'):
            if self.em_mode == 'best' or em_result is None:
                results_dir = trial_root / 'results' if trial_root else None
                em_result = self._run_em_learning(results_dir) if results_dir else None
            if em_result:
                save_em_params(self.dataset_name, self.method, em_result)
                log_em_result(self.dataset_name, self.method, em_result, source="optuna_best")

    def _run_trial(self, trial_number: int, params: Dict[str, float]):
        cfg = json.loads(json.dumps(self.args.base_cfg))
        cfg['methods'] = [self.method]
        results_dir = self.artifacts_dir / f"trial_{trial_number:04d}" / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        cfg['results_dir'] = str(results_dir)
        cfg.setdefault('name', f"optuna_{self.method}")
        self._apply_family_defaults(cfg)
        self._apply_track_enforcement(cfg)
        for path, value in params.items():
            _set_nested(cfg, path, value)
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"optuna_{self.method}_"))
        cfg_path = tmp_dir / 'config.json'
        with open(cfg_path, 'w', encoding='utf-8') as fp:
            json.dump(cfg, fp, ensure_ascii=False, indent=2)
        cmd = [sys.executable, str(REPO_ROOT / 'run_all.py'), '-c', str(cfg_path), '-s', 'estimate', 'evaluate', 'metrics', '--no-profile-steps']
        trial_root = Path(cfg['results_dir']).parent
        try:
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            shutil.rmtree(trial_root, ignore_errors=True)
            raise optuna.TrialPruned(f"Pipeline failed: {exc}")
        metrics_path = locate_metrics_file(Path(cfg['results_dir']))
        if not metrics_path:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            shutil.rmtree(trial_root, ignore_errors=True)
            raise optuna.TrialPruned('metrics file missing')
        metric_names, records = load_method_records(metrics_path, self.method)
        summary = summarize_records(metric_names, records)
        em_result = None
        if self.em_mode == 'trial':
            em_result = self._run_em_learning(Path(cfg['results_dir']))
            if em_result:
                log_em_result(self.dataset_name, self.method, em_result, source=f"optuna_trial_{trial_number:04d}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return summary, em_result, trial_root

    def _apply_track_enforcement(self, cfg: Dict) -> None:
        gating = cfg.setdefault('gating', {})
        gating.setdefault('debug', {})['disable_gating'] = True
        common = gating.setdefault('common', {})
        common['use_track'] = True
        tracker = gating.setdefault('tracker', {})
        tracker['std_is_soft'] = True
        tracker['std_min_bpm'] = 0.0
        tracker['unique_min'] = 0.0
        tracker['saturation_max'] = 1.0
        cfg.setdefault('eval', {}).setdefault('use_track', True)

    def _apply_family_defaults(self, cfg: Dict) -> None:
        defaults = FAMILY_DEFAULTS.get(self.family)
        if not defaults:
            return
        for path, value in defaults.items():
            keys = path.split('.')
            node = cfg
            for key in keys[:-1]:
                if key not in node or not isinstance(node[key], dict):
                    node[key] = {}
                node = node[key]
            node.setdefault(keys[-1], value)

    def _collect_tracks(self, results_dir: Optional[Path]) -> List[np.ndarray]:
        tracks: List[np.ndarray] = []
        if results_dir is None:
            return tracks
        aux_dir = results_dir / 'aux' / self.method
        if not aux_dir.exists():
            return tracks
        for npz in aux_dir.glob('*.npz'):
            try:
                with np.load(npz, allow_pickle=True) as data:
                    if 'track_hz' in data:
                        track = np.asarray(data['track_hz'], dtype=np.float64)
                        if track.size:
                            tracks.append(track)
            except Exception:
                continue
        return tracks

    def _run_em_learning(self, results_dir: Optional[Path]) -> Optional[Dict]:
        if not self.em_trainer or results_dir is None:
            return None
        tracks = self._collect_tracks(results_dir)
        if not tracks:
            return None
        stacked = np.concatenate(tracks)
        return self.em_trainer.fit(stacked)

    def _cleanup_trial(self, trial_root: Optional[Path]) -> None:
        if self.args.keep_artifacts:
            return
        if trial_root and trial_root.exists():
            shutil.rmtree(trial_root, ignore_errors=True)

    def _log_mlflow(self, trial_num: int, objective: float, summary: Dict[str, float], params: Dict[str, float], em_result: Optional[Dict]):
        if not self.mlflow_enabled:
            return
        metrics = {'objective': float(objective)}
        for key, value in summary.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                metrics[key] = float(value)
        if em_result:
            for key in ('q', 'r', 'll'):
                val = em_result.get(key)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    metrics[f'em_{key}'] = float(val)
        params_payload = {
            'method': self.method,
            'family': self.family,
            'dataset': self.dataset_name
        }
        params_payload.update({k: v for k, v in params.items()})
        run_name = f"{self.method}_trial{trial_num:04d}"
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params_payload)
                mlflow.log_metrics(metrics)
        except Exception as exc:  # pragma: no cover
            print(f"> Warning: MLflow logging failed ({exc})")


def aggregate_best_entries(output_root: Path, allowed_methods: Optional[Sequence[str]] = None) -> List[Dict[str, str]]:
    rows = []
    allowed = {name.lower(): name for name in allowed_methods} if allowed_methods else None
    for family_dir in output_root.iterdir() if output_root.exists() else []:
        if not family_dir.is_dir() or family_dir.name.startswith('_'):
            continue
        for method_dir in family_dir.iterdir():
            best_path = method_dir / 'best.json'
            if not best_path.exists():
                continue
            with open(best_path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            rel_path = str(best_path.relative_to(REPO_ROOT))
            method_name = data.get('method', method_dir.name)
            if allowed and method_name.lower() not in allowed:
                continue
            rows.append({
                'method': method_name,
                'family': data.get('family', family_dir.name),
                'objective': data.get('objective'),
                'MAE_bpm_med': data.get('metrics', {}).get('MAE_bpm_med'),
                'R_mean': data.get('metrics', {}).get('R_mean'),
                'SNR_med': data.get('metrics', {}).get('SNR_med'),
                'edge_sat': data.get('metrics', {}).get('edge_sat'),
                'nan_rate': data.get('metrics', {}).get('nan_rate'),
                'jerk_hzps': data.get('metrics', {}).get('jerk_hzps'),
                'best_json_path': rel_path
            })
    if allowed and len(rows) != len(allowed):
        present = {row['method'].lower() for row in rows}
        missing = sorted(name for key, name in allowed.items() if key not in present)
        if missing:
            print(f"> Warning: leaderboard missing best.json for {len(missing)} methods: {', '.join(missing)}")
    rows.sort(key=lambda r: (
        r['objective'] if isinstance(r.get('objective'), (int, float)) else float('inf'),
        r.get('MAE_bpm_med', float('inf')),
        -(r.get('R_mean') or -float('inf')),
        -(r.get('SNR_med') or -float('inf'))
    ))
    return rows


def update_leaderboard(output_root: Path, allowed_methods: Optional[Sequence[str]] = None) -> Path:
    rows = aggregate_best_entries(output_root, allowed_methods)
    if not rows:
        raise RuntimeError("No best.json files found; run tuning first")
    dashboards = output_root / 'dashboards'
    dashboards.mkdir(parents=True, exist_ok=True)
    leaderboard = dashboards / 'leaderboard.csv'
    with open(leaderboard, 'w', newline='', encoding='utf-8') as fp:
        fieldnames = ['method', 'family', 'objective', 'MAE_bpm_med', 'R_mean', 'SNR_med', 'edge_sat', 'nan_rate', 'jerk_hzps', 'best_json_path']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return leaderboard


def create_bundle(output_root: Path, config_path: Path, allowed_methods: Optional[Sequence[str]] = None) -> Path:
    leaderboard = output_root / 'dashboards' / 'leaderboard.csv'
    if not leaderboard.exists():
        raise RuntimeError("leaderboard.csv missing; cannot bundle")
    with open(leaderboard, 'r', encoding='utf-8') as fp:
        reader = csv.DictReader(fp)
        entries = list(reader)
    if allowed_methods:
        allowed = {name.lower() for name in allowed_methods}
        entries = [row for row in entries if row['method'].lower() in allowed]
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M')
    bundle_dir = output_root / '_bundles' / f"best20_{timestamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_dir / 'manifest.json'
    manifest = {
        'created': datetime.utcnow().isoformat() + 'Z',
        'config': str(config_path),
        'methods': {row['method']: row['best_json_path'] for row in entries}
    }
    with open(manifest_path, 'w', encoding='utf-8') as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    apply_script = bundle_dir / 'apply_all.sh'
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'CONFIG="{config_path}"',
        'MANIFEST="$(dirname "$0")/manifest.json"',
        'RESULTS_ROOT="results/paper_best"',
        'ROOT_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"',
        "python - \"$CONFIG\" \"$MANIFEST\" \"$RESULTS_ROOT\" \"$ROOT_DIR\" <<'PY'",
        "import json, pathlib, subprocess, sys",
        "config, manifest_path, results_root, repo_root = sys.argv[1:5]",
        "repo = pathlib.Path(repo_root).resolve()",
        "manifest = json.load(open(manifest_path))",
        "methods = manifest.get('methods', {})",
        "results_path = pathlib.Path(results_root)",
        "if not results_path.is_absolute():",
        "    results_path = repo / results_path",
        "results_path.mkdir(parents=True, exist_ok=True)",
        "for method, best_rel in methods.items():",
        "    best_json = repo / best_rel",
        "    out_dir = results_path / method",
        "    out_dir.mkdir(parents=True, exist_ok=True)",
        "    cmd = [sys.executable, str(repo / 'run_all.py'), '-c', config, '-s', 'estimate', 'evaluate', 'metrics',",
        "           '--auto_discover_methods=false', '--methods', method, '--override-from', str(best_json),",
        "           '--override', 'profile=paper', '-d', str(out_dir)]",
        "    subprocess.run(cmd, check=True, cwd=repo)",
        "PY",
        "",
    ]
    script = "\n".join(script_lines)
    apply_script.write_text(script, encoding='utf-8')
    os.chmod(apply_script, 0o755)
    readme = bundle_dir / 'README.txt'
    readme.write_text("Run ./apply_all.sh to re-generate best methods under results/paper_best", encoding='utf-8')
    return bundle_dir


def main():
    args = parse_args()
    if args.num_shards is None or args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < num_shards")
    base_cfg = load_config(args.config)
    dataset_entries = base_cfg.get('datasets') or []
    dataset_name = 'UNKNOWN'
    if dataset_entries:
        dataset_name = str(dataset_entries[0].get('name') or 'unknown').upper()
    method_names = _extract_method_names(base_cfg.get('methods', []))
    allowlist = _allowlist_methods(method_names, args.methods)
    shard_methods = _select_shard_methods(allowlist, args.num_shards, args.shard_index)
    if args.list:
        print("\n".join(shard_methods))
        return
    if args.num_shards > 1:
        print(f"> Shard {args.shard_index}/{args.num_shards}: {len(shard_methods)} methods")
        if shard_methods:
            print("  -> " + ", ".join(shard_methods))
    if not shard_methods:
        print("> No methods assigned to this shard. Exiting.")
        return
    weights_cfg = ((base_cfg.get('optuna') or {}).get('objective') or {}).get('weights', {})
    cli_weight_overrides = {key: getattr(args, f'weight_{key}', None) for key in DEFAULT_WEIGHTS}
    cli_weights = {key: value for key, value in cli_weight_overrides.items() if value is not None}
    weights = {**DEFAULT_WEIGHTS, **weights_cfg, **cli_weights}
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    study_args = StudyArgs(
        base_cfg=base_cfg,
        config_path=Path(args.config).resolve(),
        output_root=output_root,
        weights=weights,
        n_trials=args.n_trials,
        timeout=args.timeout,
        sampler_seed=args.sampler_seed,
        pruner_enabled=not args.no_prune,
        keep_artifacts=args.keep_artifacts,
        dataset_name=dataset_name,
        em_mode=args.em_mode,
        mlflow_uri=args.mlflow_uri,
        mlflow_experiment=args.mlflow_experiment,
    )
    for method in shard_methods:
        family = _method_family(method)
        if not family:
            continue
        if args.families and family not in args.families:
            continue
        print(f"\n>>> Tuning {method} ({family})")
        MethodStudy(method, family, study_args).optimize()
    if not args.skip_leaderboard:
        leaderboard = update_leaderboard(output_root, shard_methods)
        print(f"> Leaderboard written to {leaderboard}")
    if (args.bundle or not args.skip_leaderboard) and (output_root / 'dashboards' / 'leaderboard.csv').exists():
        bundle = create_bundle(output_root, study_args.config_path, shard_methods)
        print(f"> Bundle created at {bundle}")


if __name__ == '__main__':
    main()
