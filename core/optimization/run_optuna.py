#!/usr/bin/env python3
"""Paper-grade Optuna tuner for ResPyre/QROBF.

This runner is aligned with the current pipeline contracts:
- main.py orchestration (estimate/evaluate/metadata)
- metrics CSV summaries in results/<run>/metrics
- method-quality artifacts in results/<run>/logs
- run_status metadata for trial health

Primary goal is robust, unattended tuning with audit-grade artifacts.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import fnmatch
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = REPO_ROOT / "configs" / "cohface_robust_ossm.json"

from core.utils.config import load_config
from core.pipeline.common import resolve_target_run_dirs


SUFFIX_FAMILY = {
    "__robust_ossm_ekf": "robust_ossm_ekf",
    "__robust_ossm_ukf": "robust_ossm_ukf",
    "__kfstd": "kfstd",
    "__ukffreq": "ukffreq",
}

DEFAULT_OBJECTIVE_WEIGHTS = {
    # Trajectory fidelity
    "time_mae": 0.08,
    "time_dtw": 0.05,
    "time_rmse": 0.03,
    # Rate fidelity
    "freq_mae": 0.24,
    "freq_rmse": 0.10,
    # Stability / failures
    "fail_rate": 0.12,
    "fail_lock": 0.09,
    "fail_double": 0.08,
    "fail_slip": 0.04,
    "invalid_rate": 0.04,
    "clip_rate": 0.03,
    # Calibration
    "nis_truefail": 0.19,
    "nis_overstrict": 0.01,
    "nis_mean_dev": 0.10,
    "coverage_dev": 0.07,
    # Trust-shaping quality
    "alpha_R_excess": 0.08,
    "g_z_eff_deficit": 0.10,
    # Auxiliary quality
    "ccc_penalty": 0.01,
    "snr_penalty": 0.01,
    # Baseline-relative (hinge): penalize if robust is worse than kfstd references.
    "vs_kfstd_time_mae": 0.10,
    "vs_kfstd_freq_mae": 0.18,
    # Hard-constraint excess (normalized)
    "constraint_penalty": 0.14,
}

DEFAULT_OBJECTIVE_CONSTRAINTS = {
    # Upper-bound constraints.
    "fail_rate_max": 0.45,
    "fail_lock_max": 0.30,
    "fail_double_max": 0.18,
    "fail_slip_max": 0.01,
    "invalid_rate_max": 0.03,
    "clip_rate_max": 0.20,
    "nis_truefail_max": 0.75,
    "nis_mean_dev_max": 0.60,
    "alpha_R_mean_max": 2.0,
    "lambda_low_frac_max": 0.35,
    "vs_kfstd_time_mae_max": 0.0,
    "vs_kfstd_freq_mae_max": 0.0,
    # Lower-bound constraints.
    "coverage95_min": 0.90,
    "g_z_eff_mean_min": 0.18,
    # Soft target for |coverage95 - target| term.
    "coverage95_target": 0.95,
    "nis_mean_target": 1.0,
    "alpha_R_target": 1.6,
    "g_z_eff_target": 0.24,
}


@dataclass
class ParamSpec:
    path: str
    kind: str
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[Sequence[Any]] = None
    log: bool = False


DEFAULT_PARAM_SPACE: Dict[str, List[ParamSpec]] = {
    "robust_ossm_ekf": [
        # Oscillator core
        ParamSpec("oscillator.qx", "float", 2e-5, 3e-3, log=True),
        ParamSpec("oscillator.qf", "float", 2e-6, 5e-4, log=True),
        ParamSpec("oscillator.rv_floor", "float", 2e-3, 4e-2, log=True),
        ParamSpec("oscillator.rv_mad_scale", "float", 0.6, 1.8),
        ParamSpec("oscillator.tau_env", "float", 20.0, 120.0),
        ParamSpec("oscillator.init_margin_hz", "float", 0.0, 0.08),
        ParamSpec("oscillator.student_t_nu", "float", 4.0, 20.0),
        ParamSpec("oscillator.vb_iters", "int", 1, 4),
        ParamSpec("oscillator.trace_cap", "float", 40.0, 260.0),
        ParamSpec("oscillator.lambda_floor", "float", 8e-4, 8e-3, log=True),
        ParamSpec("oscillator.r_eff_max_scale", "float", 20.0, 100.0),
        ParamSpec("oscillator.g_z_eff_floor_ratio", "float", 0.06, 0.25),
        ParamSpec("oscillator.post_smooth_alpha", "float", 0.72, 0.94),
        ParamSpec("oscillator.rv_auto", "choice", choices=[True, False]),
        ParamSpec("oscillator.rv", "float", 1e-4, 0.2, log=True),
        ParamSpec("oscillator.detrend", "choice", choices=[True]),
        ParamSpec("oscillator.bandpass", "choice", choices=[True]),
        ParamSpec("oscillator.zscore", "choice", choices=[True]),
        # Spectral-guidance controls
        ParamSpec("oscillator.spec_guidance_strength", "float", 0.5, 1.6),
        ParamSpec("oscillator.spec_guidance_offset", "float", 0.1, 0.45),
        ParamSpec("oscillator.spec_guidance_confidence_scale", "float", 2.0, 10.0),
        ParamSpec("oscillator.spec_guidance_snr_scale", "float", 2.0, 8.0),
        # Trust mapping
        ParamSpec("trust.beta_1", "float", 0.8, 2.6),
        ParamSpec("trust.beta_2", "float", 0.6, 2.4),
        ParamSpec("trust.gamma_1", "float", 1.0, 3.2),
        ParamSpec("trust.w_gate_vis", "float", 0.8, 2.4),
        ParamSpec("trust.w_gate_cons", "float", 0.8, 2.2),
        ParamSpec("trust.w_gate_nis", "float", 0.2, 1.2),
        ParamSpec("trust.gate_bias", "float", 0.6, 1.6),
        ParamSpec("trust.freq_jitter_decay", "float", 0.6, 1.1),
        ParamSpec("trust.thd_max", "float", 0.25, 0.5),
        ParamSpec("trust.w_h_min", "float", 0.15, 0.35),
        ParamSpec("trust.g_z_floor", "float", 0.06, 0.2),
        ParamSpec("trust.nis_hard_gate", "float", 10.0, 30.0),
        ParamSpec("trust.alpha_R_max", "float", 10.0, 30.0),
        ParamSpec("trust.alpha_Q_max", "float", 2.5, 6.0),
        # Quality estimator
        ParamSpec("quality.vis_eps", "float", 1e-8, 1e-3, log=True),
        ParamSpec("quality.vis_snr_low_db", "float", -12.0, -4.0),
        ParamSpec("quality.vis_snr_high_db", "float", 12.0, 22.0),
        ParamSpec("quality.vis_blend_contrast", "float", 0.0, 1.0),
        ParamSpec("quality.vis_blend_snr", "float", 0.0, 1.0),
        ParamSpec("quality.vis_blend_valid", "float", 0.0, 1.0),
        ParamSpec("quality.drift_scale", "float", 0.08, 0.5),
        ParamSpec("quality.cons_window", "int", 24, 64),
        ParamSpec("quality.hampel_k", "float", 1.0, 3.0),
        ParamSpec("quality.hampel_thresh", "float", 2.0, 4.2),
        ParamSpec("quality.harm_window_sec", "float", 3.0, 7.0),
        ParamSpec("quality.harm_harmonics", "int", 1, 4),
        ParamSpec("quality.burst_sigma", "float", 1.5, 5.0),
        ParamSpec("quality.burst_window", "int", 8, 40),
        # Preprocessing
        ParamSpec("preproc.robust_zscore.enabled", "choice", choices=[True]),
        ParamSpec("preproc.robust_zscore.clip", "float", 2.0, 4.0),
        ParamSpec("preproc.robust_zscore.eps", "float", 1e-8, 1e-3, log=True),
        ParamSpec("preproc.sign_align.enabled", "choice", choices=[True]),
        ParamSpec("preproc.sign_align.seconds", "float", 8.0, 14.0),
    ],
    "robust_ossm_ukf": [
        # Oscillator core
        ParamSpec("oscillator.qx", "float", 2e-5, 3e-3, log=True),
        ParamSpec("oscillator.qf", "float", 2e-6, 5e-4, log=True),
        ParamSpec("oscillator.rv_floor", "float", 2e-3, 4e-2, log=True),
        ParamSpec("oscillator.rv_mad_scale", "float", 0.6, 1.8),
        ParamSpec("oscillator.tau_env", "float", 20.0, 120.0),
        ParamSpec("oscillator.init_margin_hz", "float", 0.0, 0.08),
        ParamSpec("oscillator.student_t_nu", "float", 4.0, 20.0),
        ParamSpec("oscillator.vb_iters", "int", 1, 4),
        ParamSpec("oscillator.trace_cap", "float", 40.0, 260.0),
        ParamSpec("oscillator.lambda_floor", "float", 8e-4, 8e-3, log=True),
        ParamSpec("oscillator.r_eff_max_scale", "float", 20.0, 100.0),
        ParamSpec("oscillator.g_z_eff_floor_ratio", "float", 0.06, 0.25),
        ParamSpec("oscillator.post_smooth_alpha", "float", 0.72, 0.94),
        ParamSpec("oscillator.rv_auto", "choice", choices=[True, False]),
        ParamSpec("oscillator.rv", "float", 1e-4, 0.2, log=True),
        ParamSpec("oscillator.detrend", "choice", choices=[True]),
        ParamSpec("oscillator.bandpass", "choice", choices=[True]),
        ParamSpec("oscillator.zscore", "choice", choices=[True]),
        # UKF sigma-point controls
        ParamSpec("oscillator.ukf_alpha", "float", 8e-4, 3e-2, log=True),
        ParamSpec("oscillator.ukf_beta", "float", 1.6, 3.2),
        ParamSpec("oscillator.ukf_kappa", "float", -0.2, 0.8),
        # Spectral-guidance controls
        ParamSpec("oscillator.spec_guidance_strength", "float", 0.5, 1.6),
        ParamSpec("oscillator.spec_guidance_offset", "float", 0.1, 0.45),
        ParamSpec("oscillator.spec_guidance_confidence_scale", "float", 2.0, 10.0),
        ParamSpec("oscillator.spec_guidance_snr_scale", "float", 2.0, 8.0),
        # Trust mapping
        ParamSpec("trust.beta_1", "float", 0.8, 2.6),
        ParamSpec("trust.beta_2", "float", 0.6, 2.4),
        ParamSpec("trust.gamma_1", "float", 1.0, 3.2),
        ParamSpec("trust.w_gate_vis", "float", 0.8, 2.4),
        ParamSpec("trust.w_gate_cons", "float", 0.8, 2.2),
        ParamSpec("trust.w_gate_nis", "float", 0.2, 1.2),
        ParamSpec("trust.gate_bias", "float", 0.6, 1.6),
        ParamSpec("trust.freq_jitter_decay", "float", 0.6, 1.1),
        ParamSpec("trust.thd_max", "float", 0.25, 0.5),
        ParamSpec("trust.w_h_min", "float", 0.15, 0.35),
        ParamSpec("trust.g_z_floor", "float", 0.06, 0.2),
        ParamSpec("trust.nis_hard_gate", "float", 10.0, 30.0),
        ParamSpec("trust.alpha_R_max", "float", 10.0, 30.0),
        ParamSpec("trust.alpha_Q_max", "float", 2.5, 6.0),
        # Quality estimator
        ParamSpec("quality.vis_eps", "float", 1e-8, 1e-3, log=True),
        ParamSpec("quality.vis_snr_low_db", "float", -12.0, -4.0),
        ParamSpec("quality.vis_snr_high_db", "float", 12.0, 22.0),
        ParamSpec("quality.vis_blend_contrast", "float", 0.0, 1.0),
        ParamSpec("quality.vis_blend_snr", "float", 0.0, 1.0),
        ParamSpec("quality.vis_blend_valid", "float", 0.0, 1.0),
        ParamSpec("quality.drift_scale", "float", 0.08, 0.5),
        ParamSpec("quality.cons_window", "int", 24, 64),
        ParamSpec("quality.hampel_k", "float", 1.0, 3.0),
        ParamSpec("quality.hampel_thresh", "float", 2.0, 4.2),
        ParamSpec("quality.harm_window_sec", "float", 3.0, 7.0),
        ParamSpec("quality.harm_harmonics", "int", 1, 4),
        ParamSpec("quality.burst_sigma", "float", 1.5, 5.0),
        ParamSpec("quality.burst_window", "int", 8, 40),
        # Preprocessing
        ParamSpec("preproc.robust_zscore.enabled", "choice", choices=[True]),
        ParamSpec("preproc.robust_zscore.clip", "float", 2.0, 4.0),
        ParamSpec("preproc.robust_zscore.eps", "float", 1e-8, 1e-3, log=True),
        ParamSpec("preproc.sign_align.enabled", "choice", choices=[True]),
        ParamSpec("preproc.sign_align.seconds", "float", 8.0, 14.0),
    ],
    "kfstd": [
        ParamSpec("oscillator.qx", "float", 1e-6, 2e-2, log=True),
        ParamSpec("oscillator.rv_floor", "float", 1e-4, 0.2, log=True),
        ParamSpec("oscillator.tau_env", "float", 4.0, 120.0),
        ParamSpec("oscillator.post_smooth_alpha", "float", 0.0, 0.98),
        ParamSpec("oscillator.spec_guidance_strength", "float", 0.0, 2.0),
        ParamSpec("oscillator.spec_guidance_offset", "float", 0.0, 1.0),
        ParamSpec("preproc.robust_zscore.clip", "float", 1.5, 6.0),
        ParamSpec("preproc.robust_zscore.eps", "float", 1e-8, 1e-4, log=True),
    ],
    "ukffreq": [
        ParamSpec("oscillator.qx", "float", 1e-6, 2e-2, log=True),
        ParamSpec("oscillator.qf", "float", 1e-7, 5e-3, log=True),
        ParamSpec("oscillator.rv_floor", "float", 1e-4, 0.2, log=True),
        ParamSpec("oscillator.tau_env", "float", 4.0, 120.0),
        ParamSpec("oscillator.ukf_alpha", "float", 1e-4, 0.8, log=True),
        ParamSpec("oscillator.ukf_beta", "float", 1.0, 4.0),
        ParamSpec("oscillator.ukf_kappa", "float", -1.0, 3.0),
        ParamSpec("oscillator.post_smooth_alpha", "float", 0.0, 0.98),
        ParamSpec("oscillator.spec_guidance_strength", "float", 0.0, 2.0),
        ParamSpec("oscillator.spec_guidance_offset", "float", 0.0, 1.0),
        ParamSpec("preproc.robust_zscore.clip", "float", 1.5, 6.0),
        ParamSpec("preproc.robust_zscore.eps", "float", 1e-8, 1e-4, log=True),
    ],
}

DEFAULT_FAMILY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "robust_ossm_ekf": {
        "oscillator.predict_method": "ekf",
        "oscillator.eda_baseline": False,
        "oscillator.no_autotune": True,
        "oscillator.em_mode": "off",
        "oscillator.qx": 2.2e-4,
        "oscillator.qf": 5e-5,
        "oscillator.rv_floor": 0.012,
        "oscillator.rv_mad_scale": 0.9,
        "oscillator.tau_env": 52.0,
        "oscillator.student_t_nu": 12.0,
        "oscillator.vb_iters": 2,
        "oscillator.trace_cap": 120.0,
        "oscillator.lambda_floor": 0.004,
        "oscillator.r_eff_max_scale": 25.0,
        "oscillator.g_z_eff_floor_ratio": 0.24,
        "oscillator.post_smooth_alpha": 0.86,
        "oscillator.spec_guidance_strength": 0.95,
        "oscillator.spec_guidance_offset": 0.22,
        "trust.beta_1": 1.2,
        "trust.beta_2": 0.85,
        "trust.gamma_1": 1.9,
        "trust.w_gate_vis": 1.6,
        "trust.w_gate_cons": 1.3,
        "trust.w_gate_nis": 0.45,
        "trust.gate_bias": 0.72,
        "trust.freq_jitter_decay": 0.62,
        "trust.thd_max": 0.52,
        "trust.w_h_min": 0.38,
        "trust.g_z_floor": 0.2,
        "trust.nis_hard_gate": 24.0,
        "trust.alpha_R_max": 8.0,
        "trust.alpha_Q_max": 3.2,
        "gating.profile": "relaxed",
        "gating.tracker.std_min_bpm": 0.55,
        "gating.tracker.unique_min": 0.05,
        "gating.tracker.saturation_max": 0.42,
        "gating.tracker.std_is_soft": True,
        "gating.tracker.saturation_margin_hz": 0.02,
        "gating.spectral.peak_ratio_min": 1.25,
        "gating.spectral.prominence_min_db": 1.5,
        "gating.spectral.fwhm_max_hz": 0.55,
        "gating.spectral.fwhm_df_guard": 1.5,
        "preproc.robust_zscore.enabled": True,
        "preproc.robust_zscore.clip": 2.5,
        "preproc.sign_align.enabled": True,
        "preproc.sign_align.seconds": 10.0,
    },
    "robust_ossm_ukf": {
        "oscillator.predict_method": "ukf",
        "oscillator.eda_baseline": False,
        "oscillator.no_autotune": True,
        "oscillator.em_mode": "off",
        "oscillator.qx": 3.5e-4,
        "oscillator.qf": 8e-5,
        "oscillator.rv_floor": 0.015,
        "oscillator.rv_mad_scale": 1.0,
        "oscillator.tau_env": 56.0,
        "oscillator.student_t_nu": 10.0,
        "oscillator.vb_iters": 2,
        "oscillator.trace_cap": 150.0,
        "oscillator.lambda_floor": 0.0015,
        "oscillator.r_eff_max_scale": 50.0,
        "oscillator.g_z_eff_floor_ratio": 0.16,
        "oscillator.post_smooth_alpha": 0.84,
        "oscillator.spec_guidance_strength": 1.15,
        "oscillator.spec_guidance_offset": 0.32,
        "oscillator.ukf_alpha": 0.005,
        "oscillator.ukf_beta": 2.0,
        "oscillator.ukf_kappa": 0.0,
        "trust.beta_1": 1.4,
        "trust.beta_2": 1.1,
        "trust.gamma_1": 2.2,
        "trust.w_gate_vis": 1.5,
        "trust.w_gate_cons": 1.4,
        "trust.w_gate_nis": 0.7,
        "trust.gate_bias": 0.85,
        "trust.freq_jitter_decay": 0.8,
        "trust.thd_max": 0.4,
        "trust.w_h_min": 0.25,
        "trust.g_z_floor": 0.12,
        "trust.nis_hard_gate": 16.0,
        "trust.alpha_R_max": 20.0,
        "trust.alpha_Q_max": 5.0,
        "preproc.robust_zscore.enabled": True,
        "preproc.robust_zscore.clip": 2.5,
        "preproc.sign_align.enabled": True,
        "preproc.sign_align.seconds": 10.0,
    },
}

DEFAULT_REPRO_GUARD_TRACKED: Tuple[str, ...] = (
    "main.py",
    "core/optimization/run_optuna.py",
    "core/pipeline/runner.py",
    "core/pipeline/wrapped_method.py",
    "core/pipeline/evaluation_step.py",
    "components/models/core/base.py",
    "components/models/heads/robust_ossm.py",
)


class ReproGuardViolation(RuntimeError):
    """Raised when tracked code/config files change during an Optuna run."""


TRIAL_INDEX_FIELDS = [
    "timestamp",
    "method",
    "family",
    "trial_number",
    "objective",
    "objective_raw",
    "status",
    "duration_s",
    "split_mode",
    "split_n_selected_trials",
    "norm_mode",
    "prune_stage",
    "run_dir",
    "trial_dir",
    "config_fingerprint",
    "manifest_path",
    "metrics_summary_path",
    "params_json",
    "time_mae",
    "time_dtw",
    "freq_mae",
    "fail_rate",
    "fail_lock",
    "fail_double",
    "invalid_rate",
    "clip_rate",
    "g_t_mean",
    "g_z_eff_mean",
    "alpha_R_mean",
    "nis_mean",
    "lambda_low_frac",
    "coverage95",
    "nis_truefail",
    "nis_overstrict",
    "constraint_penalty",
]


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _normalize_method_name(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def method_family(method_name: str) -> Optional[str]:
    lname = _normalize_method_name(method_name)
    for suffix, family in SUFFIX_FAMILY.items():
        if lname.endswith(suffix):
            return family
    return None


def _method_base_key(method_name: str) -> str:
    lname = _normalize_method_name(method_name)
    return lname.split("__", 1)[0] if "__" in lname else lname


def _resolve_kfstd_reference_for_method(
    method_name: str,
    reference_cfg: Optional[Dict[str, Dict[str, float]]],
) -> Dict[str, float]:
    if not isinstance(reference_cfg, dict):
        return {}
    method_key = _normalize_method_name(method_name)
    base_key = _method_base_key(method_name)
    raw = reference_cfg.get(method_key)
    if not isinstance(raw, dict):
        raw = reference_cfg.get(base_key)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, float] = {}
    for k in ("time_mae", "freq_mae", "fail_rate"):
        v = _float_or_nan(raw.get(k))
        if np.isfinite(v) and v > 0:
            out[k] = float(v)
    return out


def _json_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_rel(path: Path) -> str:
    p = path.resolve()
    try:
        rel = p.relative_to(REPO_ROOT.resolve())
        return rel.as_posix()
    except Exception:
        return str(p)


def _guard_snapshot(paths: Sequence[Path]) -> Dict[str, str]:
    snap: Dict[str, str] = {}
    for p in paths:
        key = _repo_rel(p)
        if not p.exists():
            snap[key] = "__missing__"
            continue
        try:
            snap[key] = _file_sha256(p)
        except Exception:
            snap[key] = "__unreadable__"
    return snap


def load_repro_guard(base_cfg: Dict[str, Any], config_path: Path, cli_mode: Optional[str] = None) -> Dict[str, Any]:
    optuna_cfg = base_cfg.get("optuna") or {}
    raw = optuna_cfg.get("repro_guard")

    enabled = True
    mode = str(cli_mode or "").strip().lower()
    tracked_extra: List[str] = []

    if isinstance(raw, dict):
        if "enabled" in raw:
            enabled = bool(raw.get("enabled"))
        if not mode:
            mode = str(raw.get("mode", "abort")).strip().lower()
        extras = raw.get("tracked_paths")
        if isinstance(extras, list):
            tracked_extra = [str(x) for x in extras if str(x).strip()]

    if not mode:
        mode = "abort"
    if mode not in {"off", "warn", "abort"}:
        mode = "abort"
    if mode == "off":
        enabled = False

    tracked_paths: List[Path] = [REPO_ROOT / p for p in DEFAULT_REPRO_GUARD_TRACKED]
    tracked_paths.append(config_path.resolve())
    for item in tracked_extra:
        p = Path(item)
        if not p.is_absolute():
            p = REPO_ROOT / p
        tracked_paths.append(p.resolve())

    dedup: Dict[str, Path] = {}
    for p in tracked_paths:
        dedup[_repo_rel(p)] = p
    ordered = [dedup[k] for k in sorted(dedup.keys())]
    baseline = _guard_snapshot(ordered)

    return {
        "enabled": bool(enabled),
        "mode": mode,
        "tracked_paths": sorted(baseline.keys()),
        "baseline": baseline,
        "warned": False,
        "last_check": "",
        "last_changed_paths": [],
    }


def check_repro_guard(guard: Dict[str, Any], *, context: str) -> None:
    if not isinstance(guard, dict) or not bool(guard.get("enabled", False)):
        return
    baseline = guard.get("baseline")
    if not isinstance(baseline, dict) or not baseline:
        return

    changed: List[str] = []
    for key, expected in baseline.items():
        p = Path(key) if os.path.isabs(str(key)) else (REPO_ROOT / str(key))
        if not p.exists():
            current = "__missing__"
        else:
            try:
                current = _file_sha256(p)
            except Exception:
                current = "__unreadable__"
        if str(current) != str(expected):
            changed.append(str(key))

    guard["last_check"] = now_iso()
    guard["last_changed_paths"] = list(changed)
    if not changed:
        return

    listed = ", ".join(changed[:8])
    if len(changed) > 8:
        listed += f", ... (+{len(changed) - 8} more)"
    msg = f"Repro guard detected code/config changes during tuning [{context}]: {listed}"
    mode = str(guard.get("mode", "abort")).strip().lower()
    if mode == "warn":
        if not bool(guard.get("warned", False)):
            print(f"> WARNING: {msg}")
            guard["warned"] = True
        return
    raise ReproGuardViolation(msg)


def _deep_set(target: Dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    node = target
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def _float_or_nan(v: Any) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float("nan")


def _normalize_coverage95_scale(v: Any) -> float:
    """Normalize coverage95 to ratio scale [0, 1] when needed.

    Evaluation artifacts may store coverage as either:
      - ratio in [0, 1], or
      - percent in [0, 100].
    Objective/constraints are defined on ratio scale, so percent values are
    converted by dividing by 100 when detected.
    """
    x = _float_or_nan(v)
    if not np.isfinite(x):
        return float("nan")
    # Treat typical percent encoding (e.g., 93.3, 100.0) as 0.933 / 1.0.
    if abs(x) > 1.5 and abs(x) <= 100.0:
        x = x / 100.0
    return float(x)


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as fp:
        content = fp.read().replace("\x00", "")
    if not content.strip():
        return []
    return list(csv.DictReader(io.StringIO(content)))


def _select_method_row(rows: Sequence[Dict[str, str]], method: str) -> Optional[Dict[str, str]]:
    if not rows:
        return None
    target = _normalize_method_name(method)
    for row in rows:
        m = _normalize_method_name(row.get("method", row.get("Method", "")))
        if m == target:
            return row
    return None


def _row_value(row: Dict[str, str], *keys: str) -> Any:
    for key in keys:
        if key in row and str(row.get(key, "")).strip() != "":
            return row.get(key)
    return np.nan


def _safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _safe_median(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


_TRIAL_KEY_REGEX = re.compile(r"(?:^|[/\\\\])([A-Za-z0-9]+)_([A-Za-z0-9]+)(?:\.[A-Za-z0-9]+)?$")


def _trial_key_from_raw_token(token: Any) -> str:
    s = str(token or "").strip()
    if not s:
        return ""
    s = os.path.splitext(os.path.basename(s))[0]
    parts = [p for p in s.split("_") if p]
    if len(parts) >= 2:
        return f"{parts[-2]}_{parts[-1]}"
    return ""


def _extract_trial_key_from_row(row: Dict[str, str]) -> str:
    tk = _trial_key_from_raw_token(row.get("trial", ""))
    if tk:
        return tk
    tk = _trial_key_from_raw_token(row.get("trial_key", ""))
    if tk:
        return tk
    tk = _trial_key_from_raw_token(row.get("data_file", ""))
    if tk:
        return tk
    tk = _trial_key_from_raw_token(row.get("video", ""))
    if tk:
        return tk
    return ""


def _trial_subject(trial_key: str) -> str:
    s = str(trial_key or "").strip()
    if "_" in s:
        return s.split("_", 1)[0]
    return s


def _method_rows(rows: Sequence[Dict[str, str]], method: str) -> List[Dict[str, str]]:
    target = _normalize_method_name(method)
    out = []
    for r in rows:
        m = _normalize_method_name(r.get("method", r.get("Method", "")))
        if m == target:
            out.append(r)
    return out


def _collect_trial_metric_table(run_dir: Path, method: str) -> Dict[str, Dict[str, float]]:
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    table: Dict[str, Dict[str, float]] = {}

    def ensure_row(trial_key: str) -> Dict[str, float]:
        row = table.setdefault(str(trial_key), {})
        return row

    time_rows = _method_rows(_read_csv_rows(metrics_dir / "metrics_time_domain_raw.csv"), method)
    for r in time_rows:
        tk = _extract_trial_key_from_row(r)
        if not tk:
            continue
        row = ensure_row(tk)
        row["time_mae"] = _float_or_nan(_row_value(r, "MAE_median", "MAE"))
        row["time_rmse"] = _float_or_nan(_row_value(r, "RMSE_median", "RMSE"))
        row["time_ccc"] = _float_or_nan(_row_value(r, "CCC_median", "CCC"))
        row["time_dtw"] = _float_or_nan(_row_value(r, "DTW_Dist_median", "DTW_Dist"))

    freq_rows = _method_rows(_read_csv_rows(metrics_dir / "metrics_freq_domain_raw.csv"), method)
    for r in freq_rows:
        tk = _extract_trial_key_from_row(r)
        if not tk:
            continue
        row = ensure_row(tk)
        row["freq_mae"] = _float_or_nan(_row_value(r, "MAE_median", "MAE"))
        row["freq_rmse"] = _float_or_nan(_row_value(r, "RMSE_median", "RMSE"))
        row["freq_snr"] = _float_or_nan(_row_value(r, "SNR_Spec_median", "SNR_Spec"))

    filt_rows = _method_rows(_read_csv_rows(metrics_dir / "metrics_filter_diagnostics_raw.csv"), method)
    for r in filt_rows:
        tk = _extract_trial_key_from_row(r)
        if not tk:
            continue
        row = ensure_row(tk)
        row["fail_rate"] = _float_or_nan(_row_value(r, "Fail_Total_median", "Fail_Total"))
        row["fail_div"] = _float_or_nan(_row_value(r, "Fail_Div_median", "Fail_Div"))
        row["fail_slip"] = _float_or_nan(_row_value(r, "Fail_Slip_median", "Fail_Slip"))
        row["fail_lock"] = _float_or_nan(_row_value(r, "Fail_Lock_median", "Fail_Lock"))
        row["fail_double"] = _float_or_nan(_row_value(r, "Fail_Double_median", "Fail_Double"))
        row["nis_mean"] = _float_or_nan(_row_value(r, "NIS_Mean_median", "NIS_Mean"))
        row["nis_truefail"] = _float_or_nan(_row_value(r, "NIS_TrueFail_median", "NIS_TrueFail"))
        row["nis_overstrict"] = _float_or_nan(_row_value(r, "NIS_OverStrict_median", "NIS_OverStrict"))
        row["coverage95"] = _float_or_nan(_row_value(r, "Coverage95_median", "Coverage95"))
        row["lambda_low_frac"] = _float_or_nan(_row_value(r, "Lambda_LowFrac_median", "Lambda_LowFrac"))

    mq_rows = _method_rows(_read_csv_rows(logs_dir / "method_quality.csv"), method)
    for r in mq_rows:
        tk = _extract_trial_key_from_row(r)
        if not tk:
            continue
        row = ensure_row(tk)
        row["invalid_rate"] = _float_or_nan(r.get("invalid_row_rate"))
        row["clip_rate"] = (
            _float_or_nan(r.get("freq_low_clip_rate")) +
            _float_or_nan(r.get("freq_high_clip_rate")) +
            _float_or_nan(r.get("z_low_clip_rate")) +
            _float_or_nan(r.get("z_high_clip_rate"))
        )
        row["q_vis_mean"] = _float_or_nan(r.get("q_vis_mean"))
        row["g_t_mean"] = _float_or_nan(r.get("g_t_mean"))
        row["g_z_eff_mean"] = _float_or_nan(r.get("g_z_eff_mean"))
        row["alpha_R_mean"] = _float_or_nan(r.get("alpha_R_mean"))
        row["lambda_mean"] = _float_or_nan(r.get("lambda_mean"))
        row["lambda_lt1_frac"] = _float_or_nan(r.get("lambda_lt1_frac"))
        if not np.isfinite(row.get("lambda_low_frac", np.nan)):
            row["lambda_low_frac"] = _float_or_nan(r.get("lambda_low_frac"))

    return table


def _aggregate_metrics_from_trials(
    trial_metrics: Dict[str, Dict[str, float]],
    selected_trials: Sequence[str],
) -> Dict[str, float]:
    sel = [str(x) for x in selected_trials if str(x) in trial_metrics]
    if not sel:
        return {}

    def vals(metric: str) -> List[float]:
        out: List[float] = []
        for tk in sel:
            out.append(_float_or_nan(trial_metrics.get(tk, {}).get(metric)))
        return out

    return {
        "time_mae": _safe_median(vals("time_mae")),
        "time_rmse": _safe_median(vals("time_rmse")),
        "time_ccc": _safe_median(vals("time_ccc")),
        "time_dtw": _safe_median(vals("time_dtw")),
        "freq_mae": _safe_median(vals("freq_mae")),
        "freq_rmse": _safe_median(vals("freq_rmse")),
        "freq_snr": _safe_median(vals("freq_snr")),
        "fail_rate": _safe_median(vals("fail_rate")),
        "fail_div": _safe_median(vals("fail_div")),
        "fail_slip": _safe_median(vals("fail_slip")),
        "fail_lock": _safe_median(vals("fail_lock")),
        "fail_double": _safe_median(vals("fail_double")),
        "nis_mean": _safe_median(vals("nis_mean")),
        "nis_truefail": _safe_median(vals("nis_truefail")),
        "nis_overstrict": _safe_median(vals("nis_overstrict")),
        "coverage95": _safe_median(vals("coverage95")),
        "invalid_rate": _safe_mean(vals("invalid_rate")),
        "clip_rate": _safe_mean(vals("clip_rate")),
        "q_vis_mean": _safe_mean(vals("q_vis_mean")),
        "g_t_mean": _safe_mean(vals("g_t_mean")),
        "g_z_eff_mean": _safe_mean(vals("g_z_eff_mean")),
        "alpha_R_mean": _safe_mean(vals("alpha_R_mean")),
        "lambda_mean": _safe_mean(vals("lambda_mean")),
        "lambda_lt1_frac": _safe_mean(vals("lambda_lt1_frac")),
        "lambda_low_frac": _safe_mean(vals("lambda_low_frac")),
        "n_method_quality_rows": float(len(sel)),
        "n_method_quality_valid_rows": float(len(sel)),
    }


def _assign_subject_folds(subjects: Sequence[str], n_folds: int) -> Dict[str, int]:
    n_folds = max(1, int(n_folds))
    out: Dict[str, int] = {}
    for idx, subj in enumerate(sorted({str(s) for s in subjects if str(s).strip()})):
        out[subj] = int(idx % n_folds)
    return out


def _select_split_trials(
    trial_keys: Sequence[str],
    *,
    split_cfg: Optional[Dict[str, Any]],
    trial_number: int,
) -> Tuple[List[str], Dict[str, Any], List[List[str]]]:
    keys = sorted({str(k) for k in trial_keys if str(k).strip()})
    cfg = split_cfg if isinstance(split_cfg, dict) else {}
    mode = str(cfg.get("mode", "none")).strip().lower()
    if mode in {"", "none", "off", "disabled"} or len(keys) == 0:
        return keys, {"mode": "none", "n_total_trials": len(keys)}, [keys]

    if mode != "subject_kfold":
        return keys, {"mode": "none", "reason": f"unsupported_mode:{mode}", "n_total_trials": len(keys)}, [keys]

    n_folds = int(max(2, int(cfg.get("n_folds", 5))))
    subjects = [_trial_subject(k) for k in keys]
    fold_map = _assign_subject_folds(subjects, n_folds=n_folds)
    val_fold = int(trial_number % n_folds)

    fold_trials: List[List[str]] = [[] for _ in range(n_folds)]
    for tk in keys:
        subj = _trial_subject(tk)
        fid = int(fold_map.get(subj, 0) % n_folds)
        fold_trials[fid].append(tk)

    selected = list(fold_trials[val_fold])
    if not selected:
        selected = keys

    split_info = {
        "mode": "subject_kfold",
        "n_folds": int(n_folds),
        "val_fold": int(val_fold),
        "n_total_trials": int(len(keys)),
        "n_selected_trials": int(len(selected)),
        "n_subjects": int(len(set(subjects))),
    }
    return selected, split_info, fold_trials


def _extract_method_entries(method_cfg: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for entry in method_cfg:
        if isinstance(entry, str):
            name = str(entry)
            out[name] = {"name": name}
            continue
        if isinstance(entry, dict) and entry.get("name"):
            out[str(entry["name"])] = json.loads(json.dumps(entry))
    return out


def discover_tunable_methods(
    method_entries: Dict[str, Dict[str, Any]],
    explicit_methods: Optional[Sequence[str]] = None,
    families: Optional[Sequence[str]] = None,
) -> List[str]:
    items = sorted(method_entries.keys(), key=lambda x: _normalize_method_name(x))

    if explicit_methods:
        wanted = {_normalize_method_name(x) for x in explicit_methods}
        selected = [m for m in items if _normalize_method_name(m) in wanted]
        return [m for m in selected if method_family(m)]

    family_filtered = [m for m in items if method_family(m)]
    if families:
        allow = {str(f).strip().lower() for f in families}
        return [m for m in family_filtered if method_family(m) in allow]

    robust_only = [
        m for m in family_filtered
        if method_family(m) in {"robust_ossm_ekf", "robust_ossm_ukf"}
    ]
    if robust_only:
        return robust_only
    return family_filtered


def shard_methods(methods: Sequence[str], num_shards: int, shard_index: int) -> List[str]:
    if num_shards <= 1:
        return list(methods)
    shards: List[List[str]] = [[] for _ in range(max(1, num_shards))]
    for idx, method in enumerate(methods):
        shards[idx % num_shards].append(method)
    shard_index = max(0, min(shard_index, num_shards - 1))
    return shards[shard_index]


def suggest_params(
    trial: optuna.trial.Trial,
    family: str,
    search_space: Dict[str, List[ParamSpec]],
    method_name: str = "",
) -> Dict[str, Any]:
    specs = _resolve_search_specs(search_space, family=family, method_name=method_name)
    params: Dict[str, Any] = {}
    for spec in specs:
        name = f"param:{spec.path}"
        if spec.kind == "choice":
            params[spec.path] = trial.suggest_categorical(name, list(spec.choices or []))
        elif spec.kind == "int":
            params[spec.path] = int(trial.suggest_int(name, int(spec.low), int(spec.high)))
        elif spec.kind == "float":
            params[spec.path] = float(
                trial.suggest_float(name, float(spec.low), float(spec.high), log=bool(spec.log))
            )
        else:
            raise ValueError(f"Unsupported ParamSpec kind: {spec.kind}")
    return params


def _resolve_method_scoped_payload(mapping: Dict[str, Any], method_name: str) -> Tuple[Any, str]:
    """Resolve method-scoped payload from an override mapping.

    Precedence:
      1) exact method-name key
      2) wildcard pattern key (fnmatch), most specific first
    """
    if not isinstance(mapping, dict):
        return None, ""
    target = _normalize_method_name(method_name)
    if not target:
        return None, ""

    exact_matches: List[Tuple[str, Any]] = []
    pattern_matches: List[Tuple[int, str, Any]] = []

    for raw_key, payload in mapping.items():
        key_norm = _normalize_method_name(raw_key)
        if not key_norm:
            continue
        if key_norm == target:
            exact_matches.append((str(raw_key), payload))
            continue
        if any(ch in key_norm for ch in ("*", "?", "[")):
            if fnmatch.fnmatch(target, key_norm):
                # Specificity score: more literal chars = tighter match.
                literal_len = len(key_norm.replace("*", "").replace("?", "").replace("[", "").replace("]", ""))
                pattern_matches.append((literal_len, str(raw_key), payload))

    if exact_matches:
        key, payload = exact_matches[0]
        return payload, key
    if pattern_matches:
        pattern_matches.sort(key=lambda x: x[0], reverse=True)
        _, key, payload = pattern_matches[0]
        return payload, key
    return None, ""


def _resolve_search_specs(
    search_space: Dict[str, List[ParamSpec]],
    *,
    family: str,
    method_name: str,
) -> List[ParamSpec]:
    payload, _ = _resolve_method_scoped_payload(search_space, method_name)
    if isinstance(payload, list) and payload:
        return list(payload)
    return list(search_space.get(family, []))


def _resolve_family_defaults_for_method(
    family_defaults: Dict[str, Dict[str, Any]],
    *,
    family: str,
    method_name: str,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    fam_defaults = family_defaults.get(family, {})
    if isinstance(fam_defaults, dict):
        merged.update(json.loads(json.dumps(fam_defaults)))

    payload, _ = _resolve_method_scoped_payload(family_defaults, method_name)
    if isinstance(payload, dict):
        merged.update(json.loads(json.dumps(payload)))
    return merged


def _apply_tuned_param(method_entry: Dict[str, Any], cfg: Dict[str, Any], path: str, value: Any) -> None:
    if path.startswith("oscillator."):
        _deep_set(method_entry, f"params.{path}", value)
    elif path.startswith("trust."):
        _deep_set(method_entry, f"params.{path}", value)
    elif path.startswith("quality."):
        _deep_set(method_entry, f"params.{path}", value)
    elif path.startswith("preproc."):
        _deep_set(method_entry, f"params.{path}", value)
    elif path.startswith("eval."):
        _deep_set(cfg, path, value)
    elif path.startswith("gating."):
        _deep_set(cfg, path, value)
    else:
        # Explicit fallback: keep trial tuning method-scoped.
        _deep_set(method_entry, f"params.oscillator.{path}", value)


def collect_trial_metrics(
    run_dir: Path,
    method: str,
    *,
    split_cfg: Optional[Dict[str, Any]] = None,
    trial_number: int = 0,
) -> Tuple[Dict[str, float], List[Dict[str, str]], Dict[str, Any]]:
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"

    time_rows = _read_csv_rows(metrics_dir / "metrics_time_domain_summary.csv")
    freq_rows = _read_csv_rows(metrics_dir / "metrics_freq_domain_summary.csv")
    filt_rows = _read_csv_rows(metrics_dir / "metrics_filter_diagnostics_summary.csv")

    time_row = _select_method_row(time_rows, method) or {}
    freq_row = _select_method_row(freq_rows, method) or {}
    filt_row = _select_method_row(filt_rows, method) or {}

    mq_rows_all = _read_csv_rows(logs_dir / "method_quality.csv")
    mq_rows = [
        r for r in mq_rows_all
        if _normalize_method_name(r.get("method", "")) == _normalize_method_name(method)
    ]
    mq_valid = [r for r in mq_rows if not _truthy(r.get("missing_frame_log", False))]

    invalid_rate = _safe_mean([_float_or_nan(r.get("invalid_row_rate")) for r in mq_valid])
    clip_rate = _safe_mean([
        _float_or_nan(r.get("freq_low_clip_rate")) +
        _float_or_nan(r.get("freq_high_clip_rate")) +
        _float_or_nan(r.get("z_low_clip_rate")) +
        _float_or_nan(r.get("z_high_clip_rate"))
        for r in mq_valid
    ])

    q_vis_mean = _safe_mean([_float_or_nan(r.get("q_vis_mean")) for r in mq_valid])
    g_t_mean = _safe_mean([_float_or_nan(r.get("g_t_mean")) for r in mq_valid])
    g_z_eff_mean = _safe_mean([_float_or_nan(r.get("g_z_eff_mean")) for r in mq_valid])
    alpha_r_mean = _safe_mean([_float_or_nan(r.get("alpha_R_mean")) for r in mq_valid])
    lambda_mean = _safe_mean([_float_or_nan(r.get("lambda_mean")) for r in mq_valid])
    lambda_lt1_frac = _safe_mean([_float_or_nan(r.get("lambda_lt1_frac")) for r in mq_valid])
    lambda_low_frac = _safe_mean([_float_or_nan(r.get("lambda_low_frac")) for r in mq_valid])

    metrics_summary_based = {
        "time_mae": _float_or_nan(_row_value(time_row, "MAE_median", "MAE")),
        "time_rmse": _float_or_nan(_row_value(time_row, "RMSE_median", "RMSE")),
        "time_ccc": _float_or_nan(_row_value(time_row, "CCC_median", "CCC")),
        "time_dtw": _float_or_nan(_row_value(time_row, "DTW_Dist_median", "DTW_Dist")),
        "freq_mae": _float_or_nan(_row_value(freq_row, "MAE_median", "MAE")),
        "freq_rmse": _float_or_nan(_row_value(freq_row, "RMSE_median", "RMSE")),
        "freq_snr": _float_or_nan(_row_value(freq_row, "SNR_Spec_median", "SNR_Spec")),
        "fail_rate": _float_or_nan(_row_value(filt_row, "Fail_Total_median", "Fail_Total")),
        "fail_div": _float_or_nan(_row_value(filt_row, "Fail_Div_median", "Fail_Div")),
        "fail_slip": _float_or_nan(_row_value(filt_row, "Fail_Slip_median", "Fail_Slip")),
        "fail_lock": _float_or_nan(_row_value(filt_row, "Fail_Lock_median", "Fail_Lock")),
        "fail_double": _float_or_nan(_row_value(filt_row, "Fail_Double_median", "Fail_Double")),
        "nis_mean": _float_or_nan(_row_value(filt_row, "NIS_Mean_median", "NIS_Mean")),
        "nis_truefail": _float_or_nan(_row_value(filt_row, "NIS_TrueFail_median", "NIS_TrueFail")),
        "nis_overstrict": _float_or_nan(_row_value(filt_row, "NIS_OverStrict_median", "NIS_OverStrict")),
        "coverage95": _float_or_nan(_row_value(filt_row, "Coverage95_median", "Coverage95")),
        "invalid_rate": invalid_rate,
        "clip_rate": clip_rate,
        "q_vis_mean": q_vis_mean,
        "g_t_mean": g_t_mean,
        "g_z_eff_mean": g_z_eff_mean,
        "alpha_R_mean": alpha_r_mean,
        "lambda_mean": lambda_mean,
        "lambda_lt1_frac": lambda_lt1_frac,
        "lambda_low_frac": lambda_low_frac,
        "n_method_quality_rows": float(len(mq_rows)),
        "n_method_quality_valid_rows": float(len(mq_valid)),
    }

    trial_metrics = _collect_trial_metric_table(run_dir, method)
    all_trial_keys = sorted(trial_metrics.keys())
    selected_trials, split_info, fold_trials = _select_split_trials(
        all_trial_keys,
        split_cfg=split_cfg,
        trial_number=trial_number,
    )
    split_metrics = _aggregate_metrics_from_trials(trial_metrics, selected_trials)
    metrics = dict(metrics_summary_based)
    split_used = bool(split_metrics and split_info.get("mode") != "none")
    if split_used:
        # Split-aware objective uses the selected validation fold aggregation.
        metrics.update(split_metrics)

    fold_metrics: List[Dict[str, Any]] = []
    if split_info.get("mode") == "subject_kfold":
        for fid, fold_keys in enumerate(fold_trials):
            if not fold_keys:
                continue
            fm = _aggregate_metrics_from_trials(trial_metrics, fold_keys)
            if not fm:
                continue
            fold_metrics.append({
                "fold_id": int(fid),
                "n_trials": int(len(fold_keys)),
                "metrics": fm,
            })

    quality_summary: Dict[str, Any] = {
        "method": method,
        "n_rows": int(len(mq_rows)),
        "n_valid_rows": int(len(mq_valid)),
        "q_vis_mean": q_vis_mean,
        "g_t_mean": g_t_mean,
        "g_z_eff_mean": g_z_eff_mean,
        "alpha_R_mean": alpha_r_mean,
        "lambda_mean": lambda_mean,
        "invalid_rate_mean": invalid_rate,
        "clip_rate_mean": clip_rate,
        "split_info": split_info,
        "split_mode_used": str(split_info.get("mode", "none")),
        "split_selection_used": bool(split_used),
        "selected_trials": selected_trials,
        "fold_metrics": fold_metrics,
        "metrics_source": "split_selected_trials" if split_used else "summary_csv",
    }

    mq_summary_path = logs_dir / "method_quality_summary.json"
    if mq_summary_path.exists():
        try:
            with open(mq_summary_path, "r", encoding="utf-8") as fp:
                mq_summary = json.load(fp)
            quality_summary["method_quality_summary"] = mq_summary.get("method_summary", [])
            quality_summary["resolver_diag"] = mq_summary.get("resolver_diag")
        except Exception:
            pass

    return metrics, mq_rows, quality_summary


def _history_terms_from_study(trial: optuna.trial.Trial) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for t in trial.study.trials:
        if int(getattr(t, "number", -1)) >= int(trial.number):
            continue
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        raw = t.user_attrs.get("objective_terms_raw")
        if isinstance(raw, dict):
            out.append({str(k): _float_or_nan(v) for k, v in raw.items()})
    return out


def _history_debug_objectives_from_study(trial: optuna.trial.Trial) -> List[float]:
    out: List[float] = []
    for t in trial.study.trials:
        if int(getattr(t, "number", -1)) >= int(trial.number):
            continue
        dbg = t.user_attrs.get("debug_stage")
        if not isinstance(dbg, dict):
            continue
        val = _float_or_nan(dbg.get("objective_used", dbg.get("objective_raw")))
        if np.isfinite(val):
            out.append(float(val))
    return out


def _trial_state_to_status(state: Any) -> str:
    if state == optuna.trial.TrialState.COMPLETE:
        return "completed"
    if state == optuna.trial.TrialState.PRUNED:
        return "pruned"
    if state == optuna.trial.TrialState.FAIL:
        return "failed"
    if state == optuna.trial.TrialState.RUNNING:
        return "running"
    return "pending"


def _trial_objectives_from_frozen_trial(trial: optuna.trial.FrozenTrial, failure_objective: float) -> Tuple[float, float]:
    ua = trial.user_attrs if isinstance(trial.user_attrs, dict) else {}
    obj_raw = _float_or_nan(ua.get("objective_raw"))
    obj_used = _float_or_nan(ua.get("objective_used"))
    if np.isfinite(obj_raw) and np.isfinite(obj_used):
        return float(obj_used), float(obj_raw)
    if np.isfinite(obj_used) and not np.isfinite(obj_raw):
        return float(obj_used), float(obj_used)
    if np.isfinite(obj_raw) and not np.isfinite(obj_used):
        return float(obj_raw), float(obj_raw)

    trial_val = _float_or_nan(getattr(trial, "value", np.nan))
    if np.isfinite(trial_val):
        return float(trial_val), float(trial_val)

    fallback = float(failure_objective)
    return fallback, fallback


def _decode_optuna_params(raw_params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(raw_params, dict):
        return out
    for key, val in raw_params.items():
        k = str(key)
        if k.startswith("param:"):
            out[k[len("param:"):]] = val
        else:
            out[k] = val
    return out


def _duration_from_frozen_trial(trial: optuna.trial.FrozenTrial) -> float:
    dt0 = getattr(trial, "datetime_start", None)
    dt1 = getattr(trial, "datetime_complete", None)
    if dt0 is None or dt1 is None:
        return float("nan")
    try:
        return float(max(0.0, (dt1 - dt0).total_seconds()))
    except Exception:
        return float("nan")


def _build_tpe_sampler(seed: int, n_startup_trials: int, n_ei_candidates: int) -> optuna.samplers.TPESampler:
    kwargs: Dict[str, Any] = {"seed": int(seed)}
    sig = inspect.signature(optuna.samplers.TPESampler.__init__)
    if "n_startup_trials" in sig.parameters:
        kwargs["n_startup_trials"] = int(max(5, n_startup_trials))
    if "n_ei_candidates" in sig.parameters:
        kwargs["n_ei_candidates"] = int(max(8, n_ei_candidates))
    if "multivariate" in sig.parameters:
        kwargs["multivariate"] = True
    if "group" in sig.parameters:
        kwargs["group"] = True
    if "constant_liar" in sig.parameters:
        kwargs["constant_liar"] = True
    return optuna.samplers.TPESampler(**kwargs)


def _build_pruner(prune_cfg: Dict[str, Any], n_trials: int) -> Optional[optuna.pruners.BasePruner]:
    kind = str(prune_cfg.get("pruner", "percentile")).strip().lower()
    startup = int(max(8, int(prune_cfg.get("n_startup_trials", max(8, int(n_trials * 0.2))))))
    warmup_steps = int(max(0, int(prune_cfg.get("n_warmup_steps", 0))))
    interval_steps = int(max(1, int(prune_cfg.get("interval_steps", 1))))
    if kind in {"off", "none", "disabled"}:
        return None
    if kind == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=startup,
            n_warmup_steps=warmup_steps,
            interval_steps=interval_steps,
        )
    percentile = float(prune_cfg.get("percentile", 55.0))
    percentile = float(np.clip(percentile, 0.0, 100.0))
    return optuna.pruners.PercentilePruner(
        percentile=percentile,
        n_startup_trials=startup,
        n_warmup_steps=warmup_steps,
        interval_steps=interval_steps,
    )


def _weighted_sum_terms(terms: Dict[str, float], weights: Dict[str, float]) -> float:
    obj = 0.0
    for key, value in terms.items():
        w = float(weights.get(key, 0.0))
        if not np.isfinite(value):
            continue
        obj += w * float(value)
    if not np.isfinite(obj):
        return float("inf")
    return float(obj)


def normalize_objective_terms(
    terms_raw: Dict[str, float],
    weights: Dict[str, float],
    history_terms: Sequence[Dict[str, float]],
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    conf = cfg if isinstance(cfg, dict) else {}
    enabled = bool(conf.get("enabled", False))
    mode = str(conf.get("mode", "robust_zscore")).strip().lower()
    min_history = int(max(1, int(conf.get("min_history", 8))))
    eps = float(max(1e-12, float(conf.get("eps", 1e-6))))
    clip = conf.get("clip", 4.0)
    clip_v = float(clip) if clip is not None else None
    fallback = str(conf.get("fallback", "raw")).strip().lower()

    used = dict(terms_raw)
    per_term: Dict[str, Dict[str, Any]] = {}
    if not enabled or mode in {"none", "off", "disabled"}:
        return used, {
            "enabled": False,
            "mode": "raw",
            "n_history": int(len(history_terms)),
            "per_term": per_term,
        }

    for key, raw_value in terms_raw.items():
        if abs(float(weights.get(key, 0.0))) <= 0.0:
            per_term[key] = {"mode": "unused_weight"}
            continue
        rv = _float_or_nan(raw_value)
        hist_vals = np.asarray(
            [_float_or_nan(h.get(key)) for h in history_terms],
            dtype=np.float64,
        )
        hist_vals = hist_vals[np.isfinite(hist_vals)]
        if not np.isfinite(rv):
            per_term[key] = {"mode": "nonfinite_raw"}
            continue
        if hist_vals.size < min_history:
            if fallback == "raw":
                used[key] = float(rv)
                per_term[key] = {"mode": "fallback_raw", "n_hist": int(hist_vals.size)}
                continue
            used[key] = 0.0
            per_term[key] = {"mode": "fallback_zero", "n_hist": int(hist_vals.size)}
            continue

        med = float(np.median(hist_vals))
        mad = float(np.median(np.abs(hist_vals - med)))
        scale = float(max(1.4826 * mad, eps))
        z = float((rv - med) / scale)
        if clip_v is not None and np.isfinite(clip_v):
            z = float(np.clip(z, -abs(clip_v), abs(clip_v)))
        used[key] = z
        per_term[key] = {
            "mode": "robust_zscore",
            "n_hist": int(hist_vals.size),
            "median": med,
            "mad": mad,
            "scale": scale,
        }

    return used, {
        "enabled": True,
        "mode": mode,
        "n_history": int(len(history_terms)),
        "min_history": int(min_history),
        "clip": clip_v,
        "fallback": fallback,
        "per_term": per_term,
    }


def _constraint_penalty(metrics: Dict[str, float], constraints: Dict[str, float]) -> float:
    if not constraints:
        return 0.0

    penalty = 0.0

    def _upper(metric_key: str, bound_key: str) -> None:
        nonlocal penalty
        bound = constraints.get(bound_key, None)
        if bound is None:
            return
        b = _float_or_nan(bound)
        # Allow zero upper-bound constraints (e.g., "no worse than reference").
        if not np.isfinite(b):
            return
        x = _float_or_nan(metrics.get(metric_key))
        if not np.isfinite(x):
            penalty += 1.0
            return
        if x > b:
            penalty += float((x - b) / max(abs(b), 1e-6))

    def _lower(metric_key: str, bound_key: str) -> None:
        nonlocal penalty
        bound = constraints.get(bound_key, None)
        if bound is None:
            return
        if metric_key == "coverage95":
            b = _normalize_coverage95_scale(bound)
        else:
            b = _float_or_nan(bound)
        if not np.isfinite(b) or b <= 0:
            return
        if metric_key == "coverage95":
            x = _normalize_coverage95_scale(metrics.get(metric_key))
        else:
            x = _float_or_nan(metrics.get(metric_key))
        if not np.isfinite(x):
            penalty += 1.0
            return
        if x < b:
            penalty += float((b - x) / max(abs(b), 1e-6))

    _upper("fail_rate", "fail_rate_max")
    _upper("fail_div", "fail_div_max")
    _upper("fail_slip", "fail_slip_max")
    _upper("fail_lock", "fail_lock_max")
    _upper("fail_double", "fail_double_max")
    _upper("invalid_rate", "invalid_rate_max")
    _upper("clip_rate", "clip_rate_max")
    _upper("nis_truefail", "nis_truefail_max")
    _upper("nis_mean_dev", "nis_mean_dev_max")
    _upper("alpha_R_mean", "alpha_R_mean_max")
    _upper("lambda_low_frac", "lambda_low_frac_max")
    _upper("vs_kfstd_time_mae", "vs_kfstd_time_mae_max")
    _upper("vs_kfstd_freq_mae", "vs_kfstd_freq_mae_max")
    _lower("coverage95", "coverage95_min")
    _lower("g_z_eff_mean", "g_z_eff_mean_min")

    return float(max(0.0, penalty))


def compute_objective(
    metrics: Dict[str, float],
    weights: Dict[str, float],
    constraints: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    time_mae = metrics.get("time_mae")
    time_rmse = metrics.get("time_rmse")
    time_dtw = metrics.get("time_dtw")
    freq_mae = metrics.get("freq_mae")
    freq_rmse = metrics.get("freq_rmse")
    fail_rate = metrics.get("fail_rate")
    fail_div = metrics.get("fail_div")
    fail_slip = metrics.get("fail_slip")
    fail_lock = metrics.get("fail_lock")
    fail_double = metrics.get("fail_double")
    invalid_rate = metrics.get("invalid_rate")
    clip_rate = metrics.get("clip_rate")
    ccc = metrics.get("time_ccc")
    snr = metrics.get("freq_snr")
    nis_mean = metrics.get("nis_mean")
    nis_truefail = metrics.get("nis_truefail")
    nis_overstrict = metrics.get("nis_overstrict")
    alpha_r_mean = metrics.get("alpha_R_mean")
    g_z_eff_mean = metrics.get("g_z_eff_mean")
    kfstd_time_mae_ref = metrics.get("kfstd_time_mae_ref")
    kfstd_freq_mae_ref = metrics.get("kfstd_freq_mae_ref")
    coverage95 = _normalize_coverage95_scale(metrics.get("coverage95"))
    coverage_target = _normalize_coverage95_scale((constraints or {}).get("coverage95_target", 0.95))
    if not np.isfinite(coverage_target):
        coverage_target = 0.95
    nis_mean_target = _float_or_nan((constraints or {}).get("nis_mean_target", 1.0))
    if not np.isfinite(nis_mean_target) or nis_mean_target <= 0:
        nis_mean_target = 1.0
    alpha_r_target = _float_or_nan((constraints or {}).get("alpha_R_target", 1.6))
    if not np.isfinite(alpha_r_target) or alpha_r_target <= 0:
        alpha_r_target = 1.6
    g_z_eff_target = _float_or_nan((constraints or {}).get("g_z_eff_target", 0.24))
    if not np.isfinite(g_z_eff_target) or g_z_eff_target <= 0:
        g_z_eff_target = 0.24

    def safe(v: Any, fb: float) -> float:
        x = _float_or_nan(v)
        return fb if not np.isfinite(x) else float(x)

    time_mae_safe = safe(time_mae, 5.0)
    freq_mae_safe = safe(freq_mae, 5.0)
    ref_time = safe(kfstd_time_mae_ref, np.nan)
    ref_freq = safe(kfstd_freq_mae_ref, np.nan)
    vs_kfstd_time_mae = 0.0
    if np.isfinite(ref_time) and ref_time > 0:
        vs_kfstd_time_mae = float(max(0.0, (time_mae_safe - ref_time) / ref_time))
    vs_kfstd_freq_mae = 0.0
    if np.isfinite(ref_freq) and ref_freq > 0:
        vs_kfstd_freq_mae = float(max(0.0, (freq_mae_safe - ref_freq) / ref_freq))
    nis_mean_dev = float(abs(safe(nis_mean, nis_mean_target) - nis_mean_target) / max(abs(nis_mean_target), 1e-6))
    alpha_r_excess = float(max(0.0, (safe(alpha_r_mean, alpha_r_target) - alpha_r_target) / max(alpha_r_target, 1e-6)))
    g_z_eff_deficit = float(max(0.0, (g_z_eff_target - safe(g_z_eff_mean, 0.0)) / max(g_z_eff_target, 1e-6)))

    metrics_for_constraints = dict(metrics)
    metrics_for_constraints["vs_kfstd_time_mae"] = vs_kfstd_time_mae
    metrics_for_constraints["vs_kfstd_freq_mae"] = vs_kfstd_freq_mae
    metrics_for_constraints["nis_mean_dev"] = nis_mean_dev

    terms = {
        "time_mae": time_mae_safe,
        "time_rmse": safe(time_rmse, 5.0),
        "time_dtw": safe(time_dtw, 2.0),
        "freq_mae": freq_mae_safe,
        "freq_rmse": safe(freq_rmse, 5.0),
        "fail_rate": safe(fail_rate, 1.0),
        "fail_div": safe(fail_div, 1.0),
        "fail_slip": safe(fail_slip, 1.0),
        "fail_lock": safe(fail_lock, 1.0),
        "fail_double": safe(fail_double, 1.0),
        "invalid_rate": safe(invalid_rate, 1.0),
        "clip_rate": safe(clip_rate, 1.0),
        "ccc_penalty": 1.0 - float(np.clip(safe(ccc, 0.0), -1.0, 1.0)),
        "snr_penalty": 1.0 / (max(safe(snr, 0.0), 0.0) + 1.0),
        "nis_mean_dev": nis_mean_dev,
        "nis_truefail": safe(nis_truefail, 1.0),
        "nis_overstrict": safe(nis_overstrict, 0.0),
        "alpha_R_excess": alpha_r_excess,
        "g_z_eff_deficit": g_z_eff_deficit,
        "vs_kfstd_time_mae": vs_kfstd_time_mae,
        "vs_kfstd_freq_mae": vs_kfstd_freq_mae,
        "coverage_dev": abs(safe(coverage95, 0.0) - coverage_target),
        "constraint_penalty": _constraint_penalty(metrics_for_constraints, constraints or {}),
    }

    objective = 0.0
    for key, term in terms.items():
        w = float(weights.get(key, 0.0))
        objective += w * float(term)

    if not np.isfinite(objective):
        objective = 1e6
    return float(objective), terms


@dataclass
class StudyArgs:
    base_cfg: Dict[str, Any]
    config_path: Path
    output_root: Path
    n_trials: int
    timeout: Optional[int]
    sampler_seed: int
    pruner_enabled: bool
    keep_artifacts: bool
    em_mode: str
    failure_objective: float
    objective_weights: Dict[str, float]
    objective_constraints: Dict[str, float]
    objective_kfstd_reference: Dict[str, Dict[str, float]]
    objective_normalization: Dict[str, Any]
    search_space: Dict[str, List[ParamSpec]]
    family_defaults: Dict[str, Dict[str, Any]]
    split_cfg: Dict[str, Any]
    pruning_cfg: Dict[str, Any]
    repro_guard: Dict[str, Any]


class MethodStudy:
    def __init__(
        self,
        method: str,
        family: str,
        method_entry: Dict[str, Any],
        args: StudyArgs,
    ):
        self.method = method
        self.family = family
        self.method_entry = json.loads(json.dumps(method_entry))
        self.args = args

        self.study_dir = args.output_root / family / _normalize_method_name(method)
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.trials_root = self.study_dir / "trials"
        self.trials_root.mkdir(parents=True, exist_ok=True)
        self.study_db = self.study_dir / "study.db"
        self.best_path = self.study_dir / "best.json"
        self.trial_index_path = self.study_dir / "trial_index.csv"
        self.global_index_path = args.output_root / "trial_index.csv"

    def _attach_kfstd_reference(self, metrics: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        out = dict(metrics or {})
        ref = _resolve_kfstd_reference_for_method(
            self.method,
            self.args.objective_kfstd_reference,
        )
        if "time_mae" in ref:
            out["kfstd_time_mae_ref"] = float(ref["time_mae"])
        if "freq_mae" in ref:
            out["kfstd_freq_mae_ref"] = float(ref["freq_mae"])
        if "fail_rate" in ref:
            out["kfstd_fail_rate_ref"] = float(ref["fail_rate"])
        return out, ref

    def optimize(self) -> None:
        check_repro_guard(self.args.repro_guard, context=f"before_study:{self.method}")
        prune_cfg = self.args.pruning_cfg if isinstance(self.args.pruning_cfg, dict) else {}
        startup_trials = int(max(8, int(prune_cfg.get("sampler_startup_trials", max(8, int(self.args.n_trials * 0.2))))))
        n_ei_candidates = int(max(16, int(prune_cfg.get("sampler_ei_candidates", 64))))
        sampler = _build_tpe_sampler(
            seed=self.args.sampler_seed,
            n_startup_trials=startup_trials,
            n_ei_candidates=n_ei_candidates,
        )
        pruner = _build_pruner(prune_cfg, n_trials=self.args.n_trials) if self.args.pruner_enabled else None

        study = optuna.create_study(
            study_name=_normalize_method_name(self.method),
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=f"sqlite:///{self.study_db.as_posix()}",
            load_if_exists=True,
        )
        self._reconcile_trial_records(study)
        if not study.trials:
            self._enqueue_seed_trial(study)
        try:
            study.optimize(self._objective, n_trials=self.args.n_trials, timeout=self.args.timeout, gc_after_trial=True)
        finally:
            # Best-effort repair for index/manifest gaps if the run exits early.
            self._reconcile_trial_records(study)

    def _enqueue_seed_trial(self, study: optuna.study.Study) -> None:
        """Enqueue family defaults as trial #0 warm-start seed (fresh studies only)."""
        specs = _resolve_search_specs(self.args.search_space, family=self.family, method_name=self.method)
        defaults = _resolve_family_defaults_for_method(
            self.args.family_defaults, family=self.family, method_name=self.method
        )
        seed: Dict[str, Any] = {}
        for spec in specs:
            val = defaults.get(spec.path)
            if val is None:
                continue
            key = f"param:{spec.path}"
            if spec.kind == "choice":
                if val in (spec.choices or []):
                    seed[key] = val
            elif spec.kind in ("float", "int"):
                lo, hi = float(spec.low), float(spec.high)
                cval = float(val) if spec.kind == "float" else int(round(float(val)))
                if lo <= cval <= hi:
                    seed[key] = cval
        if seed:
            try:
                study.enqueue_trial(seed)
            except Exception:
                pass  # Non-fatal: optimization proceeds without warm-start

    def _objective(self, trial: optuna.trial.Trial) -> float:
        check_repro_guard(self.args.repro_guard, context=f"trial_start:{self.method}:{trial.number}")
        started_ts = time.time()
        trial_number = int(trial.number)
        trial_dir = self.trials_root / f"trial_{trial_number:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        params = suggest_params(
            trial,
            self.family,
            self.args.search_space,
            method_name=self.method,
        )
        run_label = f"trial_{trial_number:05d}"
        run_root = trial_dir / "run_workspace"
        run_results_root = run_root / "results"
        run_results_root.mkdir(parents=True, exist_ok=True)

        trial_cfg = self._build_trial_cfg(params=params, run_label=run_label, run_results_root=run_results_root)
        cfg_fingerprint = _json_hash(trial_cfg)[:16]

        trial_cfg_path = trial_dir / "trial_config.json"
        with open(trial_cfg_path, "w", encoding="utf-8") as fp:
            json.dump(trial_cfg, fp, ensure_ascii=False, indent=2)
        manifest_stub, metrics_stub = self._write_trial_stub_artifacts(
            trial=trial,
            trial_dir=trial_dir,
            cfg_fingerprint=cfg_fingerprint,
            params=params,
        )
        self._upsert_trial_index(
            path=self.trial_index_path,
            trial_number=trial_number,
            objective=np.nan,
            objective_raw=np.nan,
            status="running",
            duration_s=0.0,
            run_dir=None,
            trial_dir=trial_dir,
            cfg_fingerprint=cfg_fingerprint,
            manifest_path=manifest_stub,
            metrics_summary_path=metrics_stub,
            params=params,
            metrics={},
            split_info={},
            norm_meta={"mode": "raw"},
            prune_stage="",
        )
        self._upsert_trial_index(
            path=self.global_index_path,
            trial_number=trial_number,
            objective=np.nan,
            objective_raw=np.nan,
            status="running",
            duration_s=0.0,
            run_dir=None,
            trial_dir=trial_dir,
            cfg_fingerprint=cfg_fingerprint,
            manifest_path=manifest_stub,
            metrics_summary_path=metrics_stub,
            params=params,
            metrics={},
            split_info={},
            norm_meta={"mode": "raw"},
            prune_stage="",
        )

        status = "completed"
        err_msg = ""
        objective = float(self.args.failure_objective)
        objective_raw = float(self.args.failure_objective)
        metrics: Dict[str, float] = {}
        terms_raw: Dict[str, float] = {}
        terms_used: Dict[str, float] = {}
        objective_norm_meta: Dict[str, Any] = {"enabled": False, "mode": "raw"}
        quality_summary: Dict[str, Any] = {}
        mq_rows: List[Dict[str, str]] = []
        run_dir: Optional[Path] = None
        pruned_exc: Optional[optuna.TrialPruned] = None
        prune_stage = ""
        debug_stage_payload: Dict[str, Any] = {}

        try:
            prune_cfg = self.args.pruning_cfg if isinstance(self.args.pruning_cfg, dict) else {}
            run_debug_stage = bool(self.args.pruner_enabled) and bool(prune_cfg.get("debug_stage", False))
            if run_debug_stage:
                debug_run_label = f"{run_label}__debug"
                debug_run_root = trial_dir / "run_workspace_debug"
                debug_run_results_root = debug_run_root / "results"
                debug_run_results_root.mkdir(parents=True, exist_ok=True)
                debug_cfg = self._build_trial_cfg(
                    params=params,
                    run_label=debug_run_label,
                    run_results_root=debug_run_results_root,
                )
                debug_cfg_path = trial_dir / "trial_config_debug.json"
                with open(debug_cfg_path, "w", encoding="utf-8") as fp:
                    json.dump(debug_cfg, fp, ensure_ascii=False, indent=2)

                self._run_pipeline(debug_cfg_path, trial_dir, debug_mode=True)
                debug_run_dir = self._resolve_run_dir(debug_run_results_root, debug_run_label)
                metrics_dbg, _, quality_dbg = collect_trial_metrics(
                    debug_run_dir,
                    self.method,
                    split_cfg=self.args.split_cfg,
                    trial_number=trial_number,
                )
                metrics_dbg_ref, kfstd_ref_dbg = self._attach_kfstd_reference(metrics_dbg)
                objective_dbg_raw, terms_dbg_raw = compute_objective(
                    metrics_dbg_ref,
                    self.args.objective_weights,
                    self.args.objective_constraints,
                )
                hist_terms_dbg = _history_terms_from_study(trial)
                terms_dbg_used, norm_dbg_meta = normalize_objective_terms(
                    terms_dbg_raw,
                    self.args.objective_weights,
                    hist_terms_dbg,
                    self.args.objective_normalization,
                )
                objective_dbg = _weighted_sum_terms(terms_dbg_used, self.args.objective_weights)
                if not np.isfinite(objective_dbg):
                    objective_dbg = float(self.args.failure_objective)

                debug_hist = _history_debug_objectives_from_study(trial)
                dbg_quantile = prune_cfg.get("debug_prune_quantile", None)
                dbg_min_hist = int(max(1, int(prune_cfg.get("debug_prune_min_history", 16))))
                dbg_margin = float(prune_cfg.get("debug_prune_margin", 0.0))
                if dbg_quantile is not None:
                    qv = _float_or_nan(dbg_quantile)
                    if np.isfinite(qv):
                        qv = float(np.clip(qv, 0.01, 0.99))
                        if len(debug_hist) >= dbg_min_hist:
                            cutoff = float(np.quantile(np.asarray(debug_hist, dtype=np.float64), qv))
                            if np.isfinite(cutoff) and objective_dbg > (cutoff + dbg_margin):
                                status = "pruned"
                                prune_stage = "debug_quantile"
                                objective = float(objective_dbg)
                                objective_raw = float(objective_dbg_raw)
                                terms_raw = terms_dbg_raw
                                terms_used = terms_dbg_used
                                objective_norm_meta = norm_dbg_meta
                                metrics = metrics_dbg_ref
                                quality_summary = quality_dbg
                                debug_stage_payload = {
                                    "enabled": True,
                                    "objective_raw": float(objective_dbg_raw),
                                    "objective_used": float(objective_dbg),
                                    "objective_terms_raw": terms_dbg_raw,
                                    "objective_terms_used": terms_dbg_used,
                                    "normalization": norm_dbg_meta,
                                    "kfstd_reference": kfstd_ref_dbg,
                                    "split_info": quality_dbg.get("split_info", {}),
                                    "run_dir": str(debug_run_dir),
                                    "quantile_cutoff": float(cutoff),
                                    "quantile_level": float(qv),
                                }
                                trial.set_user_attr("debug_stage", debug_stage_payload)
                                pruned_exc = optuna.TrialPruned("Pruned by debug-stage quantile gate")
                                raise pruned_exc

                trial.report(float(objective_dbg), step=0)
                debug_stage_payload = {
                    "enabled": True,
                    "objective_raw": float(objective_dbg_raw),
                    "objective_used": float(objective_dbg),
                    "objective_terms_raw": terms_dbg_raw,
                    "objective_terms_used": terms_dbg_used,
                    "normalization": norm_dbg_meta,
                    "kfstd_reference": kfstd_ref_dbg,
                    "split_info": quality_dbg.get("split_info", {}),
                    "run_dir": str(debug_run_dir),
                }
                trial.set_user_attr("debug_stage", debug_stage_payload)
                if trial.should_prune():
                    status = "pruned"
                    prune_stage = "debug_stage"
                    objective = float(objective_dbg)
                    objective_raw = float(objective_dbg_raw)
                    terms_raw = terms_dbg_raw
                    terms_used = terms_dbg_used
                    objective_norm_meta = norm_dbg_meta
                    metrics = metrics_dbg_ref
                    quality_summary = quality_dbg
                    pruned_exc = optuna.TrialPruned("Pruned after debug-stage objective")
                    raise pruned_exc

            self._run_pipeline(trial_cfg_path, trial_dir, debug_mode=False)
            run_dir = self._resolve_run_dir(run_results_root, run_label)
            run_status = self._read_run_status(run_dir)
            if str(run_status.get("status", "")).lower() != "completed":
                raise RuntimeError(f"Pipeline run not completed: status={run_status.get('status')}")

            metrics, mq_rows, quality_summary = collect_trial_metrics(
                run_dir,
                self.method,
                split_cfg=self.args.split_cfg,
                trial_number=trial_number,
            )
            metrics_ref, kfstd_ref = self._attach_kfstd_reference(metrics)
            metrics = metrics_ref
            objective_raw, terms_raw = compute_objective(
                metrics_ref,
                self.args.objective_weights,
                self.args.objective_constraints,
            )
            hist_terms = _history_terms_from_study(trial)
            terms_used, objective_norm_meta = normalize_objective_terms(
                terms_raw,
                self.args.objective_weights,
                hist_terms,
                self.args.objective_normalization,
            )
            objective = _weighted_sum_terms(terms_used, self.args.objective_weights)
            if not np.isfinite(objective):
                objective = float(self.args.failure_objective)
            metrics["constraint_penalty"] = float(terms_raw.get("constraint_penalty", np.nan))
            if kfstd_ref:
                metrics["kfstd_ref_count"] = float(len(kfstd_ref))

            report_step = 1 if run_debug_stage else 0
            trial.report(float(objective), step=report_step)
            if self.args.pruner_enabled and trial.should_prune():
                status = "pruned"
                prune_stage = "full_objective"
                pruned_exc = optuna.TrialPruned("Pruned after full objective report")
                raise pruned_exc
        except optuna.TrialPruned as exc:
            status = "pruned"
            err_msg = str(exc)
            if pruned_exc is None:
                pruned_exc = exc
        except Exception as exc:
            status = "failed"
            err_msg = str(exc)
            objective = float(self.args.failure_objective)
            objective_raw = float(self.args.failure_objective)

        finished_ts = time.time()
        duration_s = float(max(0.0, finished_ts - started_ts))

        manifest_path, metrics_summary_path = self._write_trial_artifacts(
            trial=trial,
            trial_dir=trial_dir,
            trial_cfg=trial_cfg,
            cfg_fingerprint=cfg_fingerprint,
            params=params,
            status=status,
            error_message=err_msg,
            objective=objective,
            objective_raw=objective_raw,
            objective_terms_raw=terms_raw,
            objective_terms_used=terms_used if terms_used else terms_raw,
            objective_normalization=objective_norm_meta,
            metrics=metrics,
            mq_rows=mq_rows,
            quality_summary=quality_summary,
            run_dir=run_dir,
            duration_s=duration_s,
            prune_stage=prune_stage,
            debug_stage=debug_stage_payload,
        )

        self._upsert_trial_index(
            path=self.trial_index_path,
            trial_number=trial_number,
            objective=objective,
            objective_raw=objective_raw,
            status=status,
            duration_s=duration_s,
            run_dir=run_dir,
            trial_dir=trial_dir,
            cfg_fingerprint=cfg_fingerprint,
            manifest_path=manifest_path,
            metrics_summary_path=metrics_summary_path,
            params=params,
            metrics=metrics,
            split_info=quality_summary.get("split_info", {}),
            norm_meta=objective_norm_meta,
            prune_stage=prune_stage,
        )
        self._upsert_trial_index(
            path=self.global_index_path,
            trial_number=trial_number,
            objective=objective,
            objective_raw=objective_raw,
            status=status,
            duration_s=duration_s,
            run_dir=run_dir,
            trial_dir=trial_dir,
            cfg_fingerprint=cfg_fingerprint,
            manifest_path=manifest_path,
            metrics_summary_path=metrics_summary_path,
            params=params,
            metrics=metrics,
            split_info=quality_summary.get("split_info", {}),
            norm_meta=objective_norm_meta,
            prune_stage=prune_stage,
        )

        if status == "completed":
            self._update_best(
                trial.number,
                objective,
                params,
                metrics,
                terms_used if terms_used else terms_raw,
                manifest_path,
            )

        if not self.args.keep_artifacts and run_root.exists():
            shutil.rmtree(run_root, ignore_errors=True)
        debug_root = trial_dir / "run_workspace_debug"
        if not self.args.keep_artifacts and debug_root.exists():
            shutil.rmtree(debug_root, ignore_errors=True)

        trial.set_user_attr("objective_raw", float(objective_raw))
        trial.set_user_attr("objective_used", float(objective))
        trial.set_user_attr("objective_terms_raw", terms_raw)
        trial.set_user_attr("objective_terms_used", terms_used if terms_used else terms_raw)
        trial.set_user_attr("objective_normalization", objective_norm_meta)
        trial.set_user_attr("split_info", quality_summary.get("split_info", {}))

        if status == "failed":
            trial.set_user_attr("failed", True)
            trial.set_user_attr("error", err_msg)
        if status == "pruned":
            trial.set_user_attr("pruned", True)
            trial.set_user_attr("prune_stage", prune_stage or "unknown")
            if pruned_exc is not None:
                raise pruned_exc
        return float(objective)

    def _build_trial_cfg(self, params: Dict[str, Any], run_label: str, run_results_root: Path) -> Dict[str, Any]:
        cfg = json.loads(json.dumps(self.args.base_cfg))

        method_cfg = json.loads(json.dumps(self.method_entry))
        if not isinstance(method_cfg, dict):
            method_cfg = {"name": self.method}
        method_cfg.setdefault("name", self.method)

        # Family-specific defaults first.
        defaults = _resolve_family_defaults_for_method(
            self.args.family_defaults,
            family=self.family,
            method_name=self.method,
        )
        for path, value in defaults.items():
            _apply_tuned_param(method_cfg, cfg, path, value)

        # Unattended reproducibility defaults.
        _deep_set(method_cfg, "params.oscillator.no_autotune", True)
        if self.args.em_mode == "off":
            _deep_set(method_cfg, "params.oscillator.em_mode", None)
        else:
            # WrappedMethod supports explicit experimental EM modes ("online"/"trial").
            # For unattended tuning we map trial/best onto per-trial EM execution.
            _deep_set(method_cfg, "params.oscillator.em_mode", "trial")

        for path, value in params.items():
            _apply_tuned_param(method_cfg, cfg, path, value)

        cfg["methods"] = [method_cfg]
        cfg["results_dir"] = str(run_results_root)
        cfg["name"] = run_label
        cfg["steps"] = ["estimate", "evaluate", "metadata"]

        eval_cfg = cfg.setdefault("eval", {})
        eval_cfg["frame_log_strict"] = True
        eval_cfg["allow_missing"] = False

        if not cfg.get("gating_scope"):
            cfg["gating_scope"] = "evaluation_only"

        # Keep strict-key usage relaxed by default for tuning sweeps.
        config_block = cfg.setdefault("config", {})
        config_block.setdefault("strict_key_usage", False)

        return cfg

    def _run_pipeline(self, trial_cfg_path: Path, trial_dir: Path, *, debug_mode: bool = False) -> None:
        if debug_mode:
            stdout_path = trial_dir / "stdout.debug.log"
            stderr_path = trial_dir / "stderr.debug.log"
        else:
            stdout_path = trial_dir / "stdout.log"
            stderr_path = trial_dir / "stderr.log"
        cmd = [sys.executable, str(REPO_ROOT / "main.py"), "--config", str(trial_cfg_path)]
        if debug_mode:
            cmd.append("--debug")
        with open(stdout_path, "w", encoding="utf-8") as out_fp, open(stderr_path, "w", encoding="utf-8") as err_fp:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=out_fp, stderr=err_fp)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Pipeline failed with return code {proc.returncode}. "
                f"See logs: {stdout_path} / {stderr_path}"
            )

    def _resolve_run_dir(self, run_results_root: Path, run_label: str) -> Path:
        dirs = resolve_target_run_dirs(str(run_results_root), run_label)
        if dirs:
            return Path(dirs[0])
        # Fallback for older/wider naming.
        direct = run_results_root / run_label
        if direct.exists():
            return direct
        raise FileNotFoundError(
            f"Unable to resolve run directory under '{run_results_root}' for run_label '{run_label}'"
        )

    def _read_run_status(self, run_dir: Path) -> Dict[str, Any]:
        path = run_dir / "run_status.json"
        if not path.exists():
            raise FileNotFoundError(f"run_status.json missing: {path}")
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def _write_trial_stub_artifacts(
        self,
        *,
        trial: optuna.trial.Trial,
        trial_dir: Path,
        cfg_fingerprint: str,
        params: Dict[str, Any],
    ) -> Tuple[str, str]:
        manifest_path = trial_dir / "trial_manifest.json"
        metrics_summary_path = trial_dir / "metrics_summary.json"
        if not manifest_path.exists():
            payload = {
                "schema_version": "optuna_trial_manifest.v1",
                "generated_at": now_iso(),
                "method": self.method,
                "family": self.family,
                "trial_number": int(trial.number),
                "status": "running",
                "params": params,
                "config_fingerprint": cfg_fingerprint,
                "base_config_path": str(self.args.config_path),
                "study_db": str(self.study_db),
                "run_dir": "",
                "incomplete": True,
                "trial_stub": True,
            }
            with open(manifest_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
        if not metrics_summary_path.exists():
            payload = {
                "schema_version": "optuna_trial_metrics.v1",
                "generated_at": now_iso(),
                "method": self.method,
                "trial_number": int(trial.number),
                "status": "running",
                "objective": float("nan"),
                "objective_raw": float("nan"),
                "metrics": {},
                "incomplete": True,
                "trial_stub": True,
            }
            with open(metrics_summary_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
        return str(manifest_path), str(metrics_summary_path)

    def _write_trial_artifacts(
        self,
        *,
        trial: optuna.trial.Trial,
        trial_dir: Path,
        trial_cfg: Dict[str, Any],
        cfg_fingerprint: str,
        params: Dict[str, Any],
        status: str,
        error_message: str,
        objective: float,
        objective_raw: float,
        objective_terms_raw: Dict[str, float],
        objective_terms_used: Dict[str, float],
        objective_normalization: Dict[str, Any],
        metrics: Dict[str, float],
        mq_rows: List[Dict[str, str]],
        quality_summary: Dict[str, Any],
        run_dir: Optional[Path],
        duration_s: float,
        prune_stage: str,
        debug_stage: Dict[str, Any],
    ) -> Tuple[str, str]:
        run_status_obj = {}
        if run_dir and (run_dir / "run_status.json").exists():
            try:
                with open(run_dir / "run_status.json", "r", encoding="utf-8") as fp:
                    run_status_obj = json.load(fp)
            except Exception:
                run_status_obj = {}

        manifest = {
            "schema_version": "optuna_trial_manifest.v1",
            "generated_at": now_iso(),
            "method": self.method,
            "family": self.family,
            "trial_number": int(trial.number),
            "status": status,
            "error_message": error_message,
            "duration_s": duration_s,
            "objective": float(objective),
            "objective_raw": float(objective_raw),
            "objective_terms_raw": objective_terms_raw,
            "objective_terms_used": objective_terms_used,
            "objective_normalization": objective_normalization,
            "params": params,
            "config_fingerprint": cfg_fingerprint,
            "code_commit": _git_commit(),
            "base_config_path": str(self.args.config_path),
            "study_db": str(self.study_db),
            "run_dir": str(run_dir) if run_dir else "",
            "run_status": run_status_obj,
            "objective_weights": self.args.objective_weights,
            "objective_constraints": self.args.objective_constraints,
            "split_cfg": self.args.split_cfg,
            "split_info": quality_summary.get("split_info", {}),
            "prune_stage": prune_stage,
            "debug_stage": debug_stage,
            "repro_guard": {
                "enabled": bool((self.args.repro_guard or {}).get("enabled", False)),
                "mode": str((self.args.repro_guard or {}).get("mode", "off")),
                "tracked_paths": list((self.args.repro_guard or {}).get("tracked_paths", [])),
                "last_check": str((self.args.repro_guard or {}).get("last_check", "")),
                "last_changed_paths": list((self.args.repro_guard or {}).get("last_changed_paths", [])),
            },
        }
        manifest_path = trial_dir / "trial_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, ensure_ascii=False, indent=2)

        metrics_payload = {
            "schema_version": "optuna_trial_metrics.v1",
            "generated_at": now_iso(),
            "method": self.method,
            "trial_number": int(trial.number),
            "status": status,
            "objective": float(objective),
            "objective_raw": float(objective_raw),
            "objective_terms_raw": objective_terms_raw,
            "objective_terms_used": objective_terms_used,
            "objective_normalization": objective_normalization,
            "metrics": metrics,
            "split_info": quality_summary.get("split_info", {}),
            "prune_stage": prune_stage,
            "incomplete": status != "completed",
        }
        metrics_summary_path = trial_dir / "metrics_summary.json"
        with open(metrics_summary_path, "w", encoding="utf-8") as fp:
            json.dump(metrics_payload, fp, ensure_ascii=False, indent=2)

        # Trial detail table: per-trial method_quality rows for this method.
        detail_csv = trial_dir / "metrics_detail.csv"
        if mq_rows:
            cols = sorted({k for r in mq_rows for k in r.keys()})
            with open(detail_csv, "w", encoding="utf-8", newline="") as fp:
                w = csv.DictWriter(fp, fieldnames=cols)
                w.writeheader()
                for row in mq_rows:
                    w.writerow(row)
        else:
            with open(detail_csv, "w", encoding="utf-8", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(["method", "trial", "status", "note"])
                w.writerow([self.method, trial.number, status, "no_method_quality_rows"])

        quality_path = trial_dir / "quality_trust_summary.json"
        with open(quality_path, "w", encoding="utf-8") as fp:
            json.dump(quality_summary, fp, ensure_ascii=False, indent=2)

        cfg_path = trial_dir / "config_used.json"
        with open(cfg_path, "w", encoding="utf-8") as fp:
            json.dump(trial_cfg, fp, ensure_ascii=False, indent=2)

        return str(manifest_path), str(metrics_summary_path)

    def _empty_trial_index_row(self) -> Dict[str, Any]:
        return {k: "" for k in TRIAL_INDEX_FIELDS}

    def _trial_index_row_key(self, row: Dict[str, Any]) -> Tuple[str, str, str]:
        method = _normalize_method_name(row.get("method", ""))
        family = str(row.get("family", "")).strip().lower()
        trial_number = _float_or_nan(row.get("trial_number"))
        if np.isfinite(trial_number):
            number_key = str(int(trial_number))
        else:
            number_key = str(row.get("trial_number", "")).strip()
        return method, family, number_key

    def _upsert_trial_index_row(self, path: Path, row: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        norm_row = self._empty_trial_index_row()
        for key in TRIAL_INDEX_FIELDS:
            if key in row:
                norm_row[key] = row.get(key)

        lock_path = path.with_suffix(".lock")
        with open(lock_path, "w") as _lock_fp:
            fcntl.flock(_lock_fp, fcntl.LOCK_EX)

            rows = _read_csv_rows(path)
            out_rows: List[Dict[str, Any]] = []
            key_new = self._trial_index_row_key(norm_row)
            replaced = False
            for item in rows:
                cur = self._empty_trial_index_row()
                for key in TRIAL_INDEX_FIELDS:
                    cur[key] = item.get(key, "")
                if self._trial_index_row_key(cur) == key_new:
                    out_rows.append(norm_row)
                    replaced = True
                else:
                    out_rows.append(cur)
            if not replaced:
                out_rows.append(norm_row)

            def _sort_key(r: Dict[str, Any]) -> Tuple[str, str, int, str]:
                tn = _float_or_nan(r.get("trial_number"))
                tni = int(tn) if np.isfinite(tn) else 10 ** 9
                return (
                    _normalize_method_name(r.get("method", "")),
                    str(r.get("family", "")).strip().lower(),
                    tni,
                    str(r.get("timestamp", "")),
                )

            out_rows.sort(key=_sort_key)
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8", newline="") as fp:
                w = csv.DictWriter(fp, fieldnames=TRIAL_INDEX_FIELDS)
                w.writeheader()
                for item in out_rows:
                    w.writerow({k: item.get(k, "") for k in TRIAL_INDEX_FIELDS})
            os.replace(tmp_path, path)

    def _upsert_trial_index(
        self,
        *,
        path: Path,
        trial_number: int,
        objective: float,
        objective_raw: float,
        status: str,
        duration_s: float,
        run_dir: Optional[Path],
        trial_dir: Path,
        cfg_fingerprint: str,
        manifest_path: str,
        metrics_summary_path: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        split_info: Dict[str, Any],
        norm_meta: Dict[str, Any],
        prune_stage: str,
    ) -> None:
        row = {
            "timestamp": now_iso(),
            "method": self.method,
            "family": self.family,
            "trial_number": int(trial_number),
            "objective": _float_or_nan(objective),
            "objective_raw": _float_or_nan(objective_raw),
            "status": str(status or ""),
            "duration_s": _float_or_nan(duration_s),
            "split_mode": str(split_info.get("mode", "none")) if isinstance(split_info, dict) else "none",
            "split_n_selected_trials": int(split_info.get("n_selected_trials", 0)) if isinstance(split_info, dict) else 0,
            "norm_mode": str(norm_meta.get("mode", "raw")) if isinstance(norm_meta, dict) else "raw",
            "prune_stage": str(prune_stage or ""),
            "run_dir": str(run_dir) if run_dir else "",
            "trial_dir": str(trial_dir),
            "config_fingerprint": str(cfg_fingerprint or ""),
            "manifest_path": str(manifest_path or ""),
            "metrics_summary_path": str(metrics_summary_path or ""),
            "params_json": json.dumps(params or {}, ensure_ascii=False, sort_keys=True),
            "time_mae": _float_or_nan((metrics or {}).get("time_mae")),
            "time_dtw": _float_or_nan((metrics or {}).get("time_dtw")),
            "freq_mae": _float_or_nan((metrics or {}).get("freq_mae")),
            "fail_rate": _float_or_nan((metrics or {}).get("fail_rate")),
            "fail_lock": _float_or_nan((metrics or {}).get("fail_lock")),
            "fail_double": _float_or_nan((metrics or {}).get("fail_double")),
            "invalid_rate": _float_or_nan((metrics or {}).get("invalid_rate")),
            "clip_rate": _float_or_nan((metrics or {}).get("clip_rate")),
            "g_t_mean": _float_or_nan((metrics or {}).get("g_t_mean")),
            "g_z_eff_mean": _float_or_nan((metrics or {}).get("g_z_eff_mean")),
            "alpha_R_mean": _float_or_nan((metrics or {}).get("alpha_R_mean")),
            "nis_mean": _float_or_nan((metrics or {}).get("nis_mean")),
            "lambda_low_frac": _float_or_nan((metrics or {}).get("lambda_low_frac")),
            "coverage95": _float_or_nan((metrics or {}).get("coverage95")),
            "nis_truefail": _float_or_nan((metrics or {}).get("nis_truefail")),
            "nis_overstrict": _float_or_nan((metrics or {}).get("nis_overstrict")),
            "constraint_penalty": _float_or_nan((metrics or {}).get("constraint_penalty")),
        }
        self._upsert_trial_index_row(path, row)

    def _load_trial_index_map(self, path: Path) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for row in _read_csv_rows(path):
            if _normalize_method_name(row.get("method", "")) != _normalize_method_name(self.method):
                continue
            if str(row.get("family", "")).strip().lower() != str(self.family).strip().lower():
                continue
            tn = _float_or_nan(row.get("trial_number"))
            if not np.isfinite(tn):
                continue
            norm = self._empty_trial_index_row()
            for key in TRIAL_INDEX_FIELDS:
                norm[key] = row.get(key, "")
            out[int(tn)] = norm
        return out

    def _write_reconciled_trial_artifacts(
        self,
        *,
        frozen_trial: optuna.trial.FrozenTrial,
        trial_dir: Path,
        status: str,
        objective: float,
        objective_raw: float,
        params: Dict[str, Any],
        split_info: Dict[str, Any],
        norm_meta: Dict[str, Any],
        prune_stage: str,
        existing_row: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, Dict[str, Any], str]:
        trial_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = trial_dir / "trial_manifest.json"
        metrics_path = trial_dir / "metrics_summary.json"

        existing_manifest: Dict[str, Any] = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as fp:
                    existing_manifest = json.load(fp)
            except Exception:
                existing_manifest = {}
        run_dir = str((existing_row or {}).get("run_dir", "") or existing_manifest.get("run_dir", ""))

        keep_manifest = bool(existing_manifest) and not bool(existing_manifest.get("trial_stub", False))
        if not keep_manifest:
            payload = {
                "schema_version": "optuna_trial_manifest.v1",
                "generated_at": now_iso(),
                "method": self.method,
                "family": self.family,
                "trial_number": int(frozen_trial.number),
                "status": status,
                "error_message": str((frozen_trial.user_attrs or {}).get("error", "")),
                "duration_s": _duration_from_frozen_trial(frozen_trial),
                "objective": float(objective),
                "objective_raw": float(objective_raw),
                "objective_terms_raw": (frozen_trial.user_attrs or {}).get("objective_terms_raw", {}),
                "objective_terms_used": (frozen_trial.user_attrs or {}).get("objective_terms_used", {}),
                "objective_normalization": norm_meta,
                "params": params,
                "config_fingerprint": str((existing_row or {}).get("config_fingerprint", "")),
                "code_commit": _git_commit(),
                "base_config_path": str(self.args.config_path),
                "study_db": str(self.study_db),
                "run_dir": run_dir,
                "objective_weights": self.args.objective_weights,
                "objective_constraints": self.args.objective_constraints,
                "split_cfg": self.args.split_cfg,
                "split_info": split_info,
                "prune_stage": str(prune_stage or ""),
                "debug_stage": (frozen_trial.user_attrs or {}).get("debug_stage", {}),
                "reconciled": True,
            }
            with open(manifest_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)

        metrics_payload: Dict[str, Any] = {}
        if metrics_path.exists():
            try:
                with open(metrics_path, "r", encoding="utf-8") as fp:
                    metrics_payload = json.load(fp)
            except Exception:
                metrics_payload = {}
        metrics_obj = metrics_payload.get("metrics", {}) if isinstance(metrics_payload, dict) else {}
        if not isinstance(metrics_obj, dict):
            metrics_obj = {}
        keep_metrics = bool(metrics_payload) and "metrics" in metrics_payload and not bool(metrics_payload.get("trial_stub", False))
        if not keep_metrics:
            payload = {
                "schema_version": "optuna_trial_metrics.v1",
                "generated_at": now_iso(),
                "method": self.method,
                "trial_number": int(frozen_trial.number),
                "status": status,
                "objective": float(objective),
                "objective_raw": float(objective_raw),
                "objective_terms_raw": (frozen_trial.user_attrs or {}).get("objective_terms_raw", {}),
                "objective_terms_used": (frozen_trial.user_attrs or {}).get("objective_terms_used", {}),
                "objective_normalization": norm_meta,
                "metrics": metrics_obj,
                "split_info": split_info,
                "prune_stage": str(prune_stage or ""),
                "incomplete": status != "completed",
                "reconciled": True,
            }
            with open(metrics_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)

        return str(manifest_path), str(metrics_path), metrics_obj, run_dir

    def _reconcile_trial_records(self, study: optuna.study.Study) -> None:
        try:
            frozen_trials = list(study.get_trials(deepcopy=False))
        except Exception:
            return
        if not frozen_trials:
            return

        per_method_rows = self._load_trial_index_map(self.trial_index_path)
        global_rows = self._load_trial_index_map(self.global_index_path)

        for ft in frozen_trials:
            trial_number = int(getattr(ft, "number", -1))
            if trial_number < 0:
                continue
            db_status = _trial_state_to_status(getattr(ft, "state", None))
            if db_status in {"pending", "running"}:
                continue

            existing = per_method_rows.get(trial_number, {})
            existing_status = str(existing.get("status", "")).strip().lower()
            manifest_str = str(existing.get("manifest_path", "")).strip()
            metrics_str = str(existing.get("metrics_summary_path", "")).strip()
            manifest_ok = bool(manifest_str) and Path(manifest_str).exists()
            metrics_ok = bool(metrics_str) and Path(metrics_str).exists()
            needs_patch = (
                not existing or
                existing_status in {"", "running", "pending"} or
                existing_status != db_status or
                (not manifest_ok) or
                (not metrics_ok)
            )
            if not needs_patch:
                continue

            objective, objective_raw = _trial_objectives_from_frozen_trial(ft, self.args.failure_objective)
            params = _decode_optuna_params(getattr(ft, "params", {}) or {})
            split_info = (ft.user_attrs or {}).get("split_info", {})
            if not isinstance(split_info, dict):
                split_info = {}
            norm_meta = (ft.user_attrs or {}).get("objective_normalization", {})
            if not isinstance(norm_meta, dict):
                norm_meta = {"mode": "raw"}
            prune_stage = str((ft.user_attrs or {}).get("prune_stage", ""))
            trial_dir = self.trials_root / f"trial_{trial_number:05d}"

            manifest_path, metrics_path, metrics_obj, run_dir_val = self._write_reconciled_trial_artifacts(
                frozen_trial=ft,
                trial_dir=trial_dir,
                status=db_status,
                objective=objective,
                objective_raw=objective_raw,
                params=params,
                split_info=split_info,
                norm_meta=norm_meta,
                prune_stage=prune_stage,
                existing_row=existing,
            )

            row = self._empty_trial_index_row()
            for key in TRIAL_INDEX_FIELDS:
                if key in existing:
                    row[key] = existing.get(key, "")
            row.update({
                "timestamp": now_iso(),
                "method": self.method,
                "family": self.family,
                "trial_number": int(trial_number),
                "objective": _float_or_nan(objective),
                "objective_raw": _float_or_nan(objective_raw),
                "status": db_status,
                "duration_s": _float_or_nan(_duration_from_frozen_trial(ft)),
                "split_mode": str(split_info.get("mode", "none")),
                "split_n_selected_trials": int(split_info.get("n_selected_trials", 0)),
                "norm_mode": str(norm_meta.get("mode", "raw")),
                "prune_stage": prune_stage,
                "run_dir": run_dir_val,
                "trial_dir": str(trial_dir),
                "manifest_path": manifest_path,
                "metrics_summary_path": metrics_path,
                "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
            })
            metric_fields = (
                "time_mae",
                "time_dtw",
                "freq_mae",
                "fail_rate",
                "fail_lock",
                "fail_double",
                "invalid_rate",
                "clip_rate",
                "g_t_mean",
                "g_z_eff_mean",
                "alpha_R_mean",
                "nis_mean",
                "lambda_low_frac",
                "coverage95",
                "nis_truefail",
                "nis_overstrict",
                "constraint_penalty",
            )
            for mk in metric_fields:
                mv = _float_or_nan(metrics_obj.get(mk)) if isinstance(metrics_obj, dict) else float("nan")
                if np.isfinite(mv):
                    row[mk] = mv
            terms_raw = (ft.user_attrs or {}).get("objective_terms_raw", {})
            if isinstance(terms_raw, dict):
                cp = _float_or_nan(terms_raw.get("constraint_penalty"))
                if np.isfinite(cp):
                    row["constraint_penalty"] = cp

            self._upsert_trial_index_row(self.trial_index_path, row)
            g_existing = global_rows.get(trial_number, {})
            g_row = self._empty_trial_index_row()
            for key in TRIAL_INDEX_FIELDS:
                if key in g_existing:
                    g_row[key] = g_existing.get(key, "")
            for key in TRIAL_INDEX_FIELDS:
                if key in row:
                    g_row[key] = row.get(key, "")
            self._upsert_trial_index_row(self.global_index_path, g_row)

    def _update_best(
        self,
        trial_number: int,
        objective: float,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        terms: Dict[str, float],
        manifest_path: str,
    ) -> None:
        current_best = None
        if self.best_path.exists():
            try:
                with open(self.best_path, "r", encoding="utf-8") as fp:
                    current_best = json.load(fp)
            except Exception:
                current_best = None

        if current_best and _float_or_nan(current_best.get("objective")) <= objective:
            return

        payload = {
            "schema_version": "optuna_best.v1",
            "updated_at": now_iso(),
            "method": self.method,
            "family": self.family,
            "trial_number": int(trial_number),
            "objective": float(objective),
            "params": params,
            "metrics": metrics,
            "objective_terms": terms,
            "manifest_path": manifest_path,
        }
        with open(self.best_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)


def aggregate_best_entries(output_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not output_root.exists():
        return rows
    for fam_dir in sorted(output_root.iterdir()):
        if not fam_dir.is_dir() or fam_dir.name.startswith("_"):
            continue
        for method_dir in sorted(fam_dir.iterdir()):
            best_path = method_dir / "best.json"
            if not best_path.exists():
                continue
            try:
                with open(best_path, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
            except Exception:
                continue
            rows.append({
                "family": payload.get("family", fam_dir.name),
                "method": payload.get("method", method_dir.name),
                "objective": _float_or_nan(payload.get("objective")),
                "trial_number": payload.get("trial_number"),
                "time_mae": _float_or_nan((payload.get("metrics") or {}).get("time_mae")),
                "freq_mae": _float_or_nan((payload.get("metrics") or {}).get("freq_mae")),
                "fail_rate": _float_or_nan((payload.get("metrics") or {}).get("fail_rate")),
                "fail_lock": _float_or_nan((payload.get("metrics") or {}).get("fail_lock")),
                "fail_double": _float_or_nan((payload.get("metrics") or {}).get("fail_double")),
                "invalid_rate": _float_or_nan((payload.get("metrics") or {}).get("invalid_rate")),
                "clip_rate": _float_or_nan((payload.get("metrics") or {}).get("clip_rate")),
                "g_z_eff_mean": _float_or_nan((payload.get("metrics") or {}).get("g_z_eff_mean")),
                "alpha_R_mean": _float_or_nan((payload.get("metrics") or {}).get("alpha_R_mean")),
                "nis_mean": _float_or_nan((payload.get("metrics") or {}).get("nis_mean")),
                "coverage95": _float_or_nan((payload.get("metrics") or {}).get("coverage95")),
                "nis_truefail": _float_or_nan((payload.get("metrics") or {}).get("nis_truefail")),
                "nis_overstrict": _float_or_nan((payload.get("metrics") or {}).get("nis_overstrict")),
                "best_json_path": str(best_path.relative_to(output_root)),
            })

    rows.sort(key=lambda r: (_float_or_nan(r.get("objective")), _normalize_method_name(str(r.get("method", "")))))
    return rows


def write_leaderboard(output_root: Path) -> Path:
    rows = aggregate_best_entries(output_root)
    out_dir = output_root / "dashboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leaderboard.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as fp:
        cols = [
            "family", "method", "objective", "trial_number",
            "time_mae", "freq_mae", "fail_rate", "fail_lock", "fail_double",
            "invalid_rate", "clip_rate",
            "g_z_eff_mean", "alpha_R_mean", "nis_mean",
            "coverage95", "nis_truefail", "nis_overstrict",
            "best_json_path",
        ]
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})
    return out_path


def write_trial_diagnostics(output_root: Path) -> Path:
    out_dir = output_root / "dashboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trial_diagnostics.csv"
    rows = _read_csv_rows(output_root / "trial_index.csv")
    cols = [
        "timestamp",
        "family",
        "method",
        "trial_number",
        "status",
        "objective",
        "objective_raw",
        "prune_stage",
        "norm_mode",
        "time_mae",
        "freq_mae",
        "fail_rate",
        "fail_lock",
        "fail_double",
        "nis_truefail",
        "nis_overstrict",
        "nis_mean",
        "g_z_eff_mean",
        "alpha_R_mean",
        "coverage95",
        "constraint_penalty",
        "term_nis_mean_dev",
        "term_g_z_eff_deficit",
        "term_alpha_R_excess",
        "term_vs_kfstd_time_mae",
        "term_vs_kfstd_freq_mae",
        "term_constraint_penalty",
        "manifest_path",
        "metrics_summary_path",
    ]
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        manifest_path = str(r.get("manifest_path", "")).strip()
        terms_raw: Dict[str, Any] = {}
        if manifest_path and os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
                terms_raw = payload.get("objective_terms_raw", {}) if isinstance(payload, dict) else {}
                if not isinstance(terms_raw, dict):
                    terms_raw = {}
            except Exception:
                terms_raw = {}
        out_rows.append({
            "timestamp": str(r.get("timestamp", "")),
            "family": str(r.get("family", "")),
            "method": str(r.get("method", "")),
            "trial_number": _float_or_nan(r.get("trial_number")),
            "status": str(r.get("status", "")),
            "objective": _float_or_nan(r.get("objective")),
            "objective_raw": _float_or_nan(r.get("objective_raw")),
            "prune_stage": str(r.get("prune_stage", "")),
            "norm_mode": str(r.get("norm_mode", "")),
            "time_mae": _float_or_nan(r.get("time_mae")),
            "freq_mae": _float_or_nan(r.get("freq_mae")),
            "fail_rate": _float_or_nan(r.get("fail_rate")),
            "fail_lock": _float_or_nan(r.get("fail_lock")),
            "fail_double": _float_or_nan(r.get("fail_double")),
            "nis_truefail": _float_or_nan(r.get("nis_truefail")),
            "nis_overstrict": _float_or_nan(r.get("nis_overstrict")),
            "nis_mean": _float_or_nan(r.get("nis_mean")),
            "g_z_eff_mean": _float_or_nan(r.get("g_z_eff_mean")),
            "alpha_R_mean": _float_or_nan(r.get("alpha_R_mean")),
            "coverage95": _float_or_nan(r.get("coverage95")),
            "constraint_penalty": _float_or_nan(r.get("constraint_penalty")),
            "term_nis_mean_dev": _float_or_nan(terms_raw.get("nis_mean_dev")),
            "term_g_z_eff_deficit": _float_or_nan(terms_raw.get("g_z_eff_deficit")),
            "term_alpha_R_excess": _float_or_nan(terms_raw.get("alpha_R_excess")),
            "term_vs_kfstd_time_mae": _float_or_nan(terms_raw.get("vs_kfstd_time_mae")),
            "term_vs_kfstd_freq_mae": _float_or_nan(terms_raw.get("vs_kfstd_freq_mae")),
            "term_constraint_penalty": _float_or_nan(terms_raw.get("constraint_penalty")),
            "manifest_path": manifest_path,
            "metrics_summary_path": str(r.get("metrics_summary_path", "")),
        })
    out_rows.sort(
        key=lambda x: (
            _normalize_method_name(x.get("method", "")),
            _float_or_nan(x.get("trial_number")),
            _float_or_nan(x.get("objective")),
        )
    )
    with open(out_path, "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in cols})
    return out_path


def write_method_health_summary(output_root: Path) -> Path:
    out_dir = output_root / "dashboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "method_health_summary.csv"
    rows = _read_csv_rows(output_root / "trial_index.csv")
    by_method: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (str(row.get("family", "")), _normalize_method_name(row.get("method", "")))
        by_method.setdefault(key, []).append(row)

    cols = [
        "family",
        "method",
        "n_trials",
        "n_completed",
        "n_pruned",
        "n_failed",
        "best_objective",
        "best_time_mae",
        "best_freq_mae",
        "best_fail_rate",
        "best_fail_lock",
        "best_nis_truefail",
        "best_g_z_eff_mean",
        "best_alpha_R_mean",
    ]
    out_rows: List[Dict[str, Any]] = []
    for (family, method), items in sorted(by_method.items()):
        n_trials = len(items)
        n_completed = sum(1 for r in items if str(r.get("status", "")).strip().lower() == "completed")
        n_pruned = sum(1 for r in items if str(r.get("status", "")).strip().lower() == "pruned")
        n_failed = sum(1 for r in items if str(r.get("status", "")).strip().lower() == "failed")
        completed = [r for r in items if str(r.get("status", "")).strip().lower() == "completed"]
        completed.sort(key=lambda r: _float_or_nan(r.get("objective")))
        best = completed[0] if completed else {}
        out_rows.append({
            "family": family,
            "method": method,
            "n_trials": int(n_trials),
            "n_completed": int(n_completed),
            "n_pruned": int(n_pruned),
            "n_failed": int(n_failed),
            "best_objective": _float_or_nan(best.get("objective")),
            "best_time_mae": _float_or_nan(best.get("time_mae")),
            "best_freq_mae": _float_or_nan(best.get("freq_mae")),
            "best_fail_rate": _float_or_nan(best.get("fail_rate")),
            "best_fail_lock": _float_or_nan(best.get("fail_lock")),
            "best_nis_truefail": _float_or_nan(best.get("nis_truefail")),
            "best_g_z_eff_mean": _float_or_nan(best.get("g_z_eff_mean")),
            "best_alpha_R_mean": _float_or_nan(best.get("alpha_R_mean")),
        })
    with open(out_path, "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in cols})
    return out_path


def export_best_preset(
    *,
    base_cfg: Dict[str, Any],
    config_path: Path,
    output_root: Path,
    destination: Optional[str],
) -> Optional[Path]:
    best_rows = aggregate_best_entries(output_root)
    if not best_rows:
        return None

    best_map: Dict[str, Dict[str, Any]] = {}
    for row in best_rows:
        rel = row.get("best_json_path")
        if not rel:
            continue
        best_path = output_root / str(rel)
        if not best_path.exists():
            continue
        try:
            with open(best_path, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
            best_map[str(payload.get("method", ""))] = payload
        except Exception:
            continue

    cfg = json.loads(json.dumps(base_cfg))
    methods = cfg.get("methods") or []
    new_methods: List[Any] = []
    updates: Dict[str, Dict[str, Any]] = {}

    for entry in methods:
        if isinstance(entry, str):
            method_name = entry
            method_obj = {"name": method_name}
        elif isinstance(entry, dict):
            method_name = str(entry.get("name", ""))
            method_obj = json.loads(json.dumps(entry))
        else:
            new_methods.append(entry)
            continue

        best = best_map.get(method_name)
        if not best:
            new_methods.append(method_obj if isinstance(entry, dict) else entry)
            continue

        params = best.get("params", {}) or {}
        for path, value in params.items():
            _apply_tuned_param(method_obj, cfg, str(path), value)
        updates[method_name] = params
        new_methods.append(method_obj)

    cfg["methods"] = new_methods
    cfg.setdefault("optuna_export", {})
    cfg["optuna_export"] = {
        "generated_at": now_iso(),
        "source_config": str(config_path),
        "output_root": str(output_root),
        "n_methods_updated": int(len(updates)),
        "updated_methods": sorted(list(updates.keys())),
    }

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if destination:
        dest = Path(destination)
        if dest.suffix.lower() != ".json":
            dest.mkdir(parents=True, exist_ok=True)
            out_path = dest / f"{config_path.stem}_optuna_best_{ts}.json"
        else:
            out_path = dest
            out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = config_path.parent / "presets"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{config_path.stem}_optuna_best_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2)

    diff_path = out_path.with_suffix(".diff.json")
    with open(diff_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "generated_at": now_iso(),
                "source_config": str(config_path),
                "preset_config": str(out_path),
                "updated_methods": updates,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    return out_path


def _parse_param_specs(raw_specs: Any) -> List[ParamSpec]:
    out: List[ParamSpec] = []
    if not isinstance(raw_specs, list):
        return out
    for item in raw_specs:
        if not isinstance(item, dict) or "path" not in item or "kind" not in item:
            continue
        out.append(
            ParamSpec(
                path=str(item.get("path")),
                kind=str(item.get("kind")),
                low=item.get("low"),
                high=item.get("high"),
                choices=item.get("choices"),
                log=bool(item.get("log", False)),
            )
        )
    return out


def load_search_space(base_cfg: Dict[str, Any]) -> Dict[str, List[ParamSpec]]:
    space = {k: list(v) for k, v in DEFAULT_PARAM_SPACE.items()}
    optuna_cfg = base_cfg.get("optuna") or {}
    raw = optuna_cfg.get("search_space")
    if isinstance(raw, dict):
        for family, specs in raw.items():
            parsed = _parse_param_specs(specs)
            if parsed:
                space[str(family)] = parsed
    return space


def load_family_defaults(base_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = json.loads(json.dumps(DEFAULT_FAMILY_DEFAULTS))
    optuna_cfg = base_cfg.get("optuna") or {}
    raw = optuna_cfg.get("family_defaults")
    if isinstance(raw, dict):
        for family, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            fam = str(family)
            base = out.get(fam, {})
            merged = json.loads(json.dumps(base))
            merged.update(payload)
            out[fam] = merged
    return out


def load_objective_weights(base_cfg: Dict[str, Any], cli_overrides: Dict[str, Optional[float]]) -> Dict[str, float]:
    out = dict(DEFAULT_OBJECTIVE_WEIGHTS)
    optuna_cfg = base_cfg.get("optuna") or {}
    objective_cfg = optuna_cfg.get("objective") or {}
    if isinstance(objective_cfg.get("weights"), dict):
        for key, val in objective_cfg["weights"].items():
            try:
                out[str(key)] = float(val)
            except Exception:
                continue
    for key, val in cli_overrides.items():
        if val is None:
            continue
        out[key] = float(val)
    return out


def load_objective_constraints(base_cfg: Dict[str, Any]) -> Dict[str, float]:
    out = dict(DEFAULT_OBJECTIVE_CONSTRAINTS)
    optuna_cfg = base_cfg.get("optuna") or {}
    objective_cfg = optuna_cfg.get("objective") or {}
    raw = objective_cfg.get("constraints")
    if raw is None:
        return out
    if isinstance(raw, dict):
        for key, val in raw.items():
            try:
                out[str(key)] = float(val)
            except Exception:
                continue
        return out
    # Any non-dict explicit value disables constraints.
    return {}


def load_objective_kfstd_reference(base_cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    optuna_cfg = base_cfg.get("optuna") or {}
    objective_cfg = optuna_cfg.get("objective") or {}
    raw = objective_cfg.get("kfstd_reference")
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        key = _normalize_method_name(k)
        ref: Dict[str, float] = {}
        for mk in ("time_mae", "freq_mae", "fail_rate"):
            val = _float_or_nan(v.get(mk))
            if np.isfinite(val) and val > 0:
                ref[mk] = float(val)
        if ref:
            out[key] = ref
    return out


def load_objective_normalization(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": False,
        "mode": "robust_zscore",
        "min_history": 8,
        "eps": 1e-6,
        "clip": 4.0,
        "fallback": "raw",
    }
    optuna_cfg = base_cfg.get("optuna") or {}
    objective_cfg = optuna_cfg.get("objective") or {}
    for key in ("normalization", "robust_zscore"):
        raw = objective_cfg.get(key)
        if isinstance(raw, dict):
            out.update(raw)
    return out


def load_split_cfg(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "mode": "none",
        "n_folds": 5,
    }
    optuna_cfg = base_cfg.get("optuna") or {}
    raw = optuna_cfg.get("split")
    if isinstance(raw, dict):
        out.update(raw)
    mode = str(out.get("mode", "none")).strip().lower()
    if mode not in {"none", "subject_kfold"}:
        out["mode"] = "none"
        out["reason"] = f"unsupported_split_mode:{mode}"
    out["n_folds"] = int(max(2, int(out.get("n_folds", 5)))) if out.get("mode") == "subject_kfold" else 1
    return out


def load_pruning_cfg(base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        # Enable low-cost debug-stage pass to allow meaningful pruning before full runs.
        "debug_stage": True,
        # Sampler/pruner defaults tuned for high-dimensional robust search.
        "pruner": "percentile",
        "percentile": 55.0,
        "n_startup_trials": 10,
        "n_warmup_steps": 0,
        "interval_steps": 1,
        "sampler_startup_trials": 10,
        "sampler_ei_candidates": 64,
        # Deterministic debug gate before full trial execution.
        "debug_prune_quantile": 0.80,
        "debug_prune_min_history": 16,
        "debug_prune_margin": 0.02,
    }
    optuna_cfg = base_cfg.get("optuna") or {}
    raw = optuna_cfg.get("pruning")
    if isinstance(raw, dict):
        out.update(raw)
    out["debug_stage"] = bool(out.get("debug_stage", False))
    out["n_startup_trials"] = int(max(1, int(out.get("n_startup_trials", 10))))
    out["n_warmup_steps"] = int(max(0, int(out.get("n_warmup_steps", 0))))
    out["interval_steps"] = int(max(1, int(out.get("interval_steps", 1))))
    out["sampler_startup_trials"] = int(max(1, int(out.get("sampler_startup_trials", 10))))
    out["sampler_ei_candidates"] = int(max(8, int(out.get("sampler_ei_candidates", 64))))
    qv = _float_or_nan(out.get("debug_prune_quantile", np.nan))
    if np.isfinite(qv):
        out["debug_prune_quantile"] = float(np.clip(qv, 0.01, 0.99))
    else:
        out["debug_prune_quantile"] = None
    out["debug_prune_min_history"] = int(max(1, int(out.get("debug_prune_min_history", 16))))
    out["debug_prune_margin"] = float(max(0.0, float(out.get("debug_prune_margin", 0.0))))
    return out


def write_tuning_contract(
    *,
    output_root: Path,
    config_path: Path,
    selected_methods: Sequence[str],
    search_space: Dict[str, List[ParamSpec]],
    family_defaults: Dict[str, Dict[str, Any]],
    objective_weights: Dict[str, float],
    objective_constraints: Dict[str, float],
    objective_normalization: Dict[str, Any],
    split_cfg: Dict[str, Any],
    pruning_cfg: Dict[str, Any],
    repro_guard: Dict[str, Any],
) -> Path:
    out_dir = output_root / "dashboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tuning_contract.json"
    space_payload: Dict[str, List[Dict[str, Any]]] = {}
    for fam, specs in (search_space or {}).items():
        fam_rows: List[Dict[str, Any]] = []
        for spec in specs:
            fam_rows.append({
                "path": str(spec.path),
                "kind": str(spec.kind),
                "low": spec.low,
                "high": spec.high,
                "choices": list(spec.choices) if isinstance(spec.choices, (list, tuple)) else spec.choices,
                "log": bool(spec.log),
            })
        space_payload[str(fam)] = fam_rows

    payload = {
        "schema_version": "optuna_tuning_contract.v1",
        "generated_at": now_iso(),
        "config_path": str(config_path),
        "selected_methods": list(selected_methods),
        "search_space": space_payload,
        "family_defaults": family_defaults,
        "objective_weights": objective_weights,
        "objective_constraints": objective_constraints,
        "objective_normalization": objective_normalization,
        "split_cfg": split_cfg,
        "pruning_cfg": pruning_cfg,
        "repro_guard": {
            "enabled": bool((repro_guard or {}).get("enabled", False)),
            "mode": str((repro_guard or {}).get("mode", "off")),
            "tracked_paths": list((repro_guard or {}).get("tracked_paths", [])),
        },
    }
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-grade Optuna tuner for QROBF")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="Base config JSON")
    parser.add_argument("--output", default="runs/optuna", help="Optuna output root")
    parser.add_argument("--methods", nargs="*", help="Explicit methods to tune")
    parser.add_argument("--families", nargs="*", choices=sorted(set(SUFFIX_FAMILY.values())), help="Restrict families")
    parser.add_argument("--n-trials", type=int, default=50, help="Trials per method")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout seconds per method")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--no-prune", action="store_true", help="Disable Optuna pruner")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep per-trial run_workspace artifacts")
    parser.add_argument("--list", action="store_true", help="List selected methods and exit")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--em-mode", choices=["off", "trial", "best"], default="off")
    parser.add_argument("--failure-objective", type=float, default=1e6)
    parser.add_argument("--export-best-preset", default="", help="Output .json path or directory for merged best preset")
    parser.add_argument(
        "--repro-guard",
        choices=["off", "warn", "abort"],
        default=None,
        help="Guard mode for code/config changes during tuning (default: abort).",
    )

    # Optional per-term overrides.
    parser.add_argument("--w-time-mae", type=float, default=None)
    parser.add_argument("--w-time-rmse", type=float, default=None)
    parser.add_argument("--w-time-dtw", type=float, default=None)
    parser.add_argument("--w-freq-mae", type=float, default=None)
    parser.add_argument("--w-freq-rmse", type=float, default=None)
    parser.add_argument("--w-fail-rate", type=float, default=None)
    parser.add_argument("--w-invalid-rate", type=float, default=None)
    parser.add_argument("--w-clip-rate", type=float, default=None)
    parser.add_argument("--w-ccc-penalty", type=float, default=None)
    parser.add_argument("--w-snr-penalty", type=float, default=None)
    parser.add_argument("--w-nis-truefail", type=float, default=None)
    parser.add_argument("--w-nis-overstrict", type=float, default=None)
    parser.add_argument("--w-nis-mean-dev", type=float, default=None)
    parser.add_argument("--w-alpha-r-excess", type=float, default=None)
    parser.add_argument("--w-g-z-eff-deficit", type=float, default=None)
    parser.add_argument("--w-coverage-dev", type=float, default=None)
    parser.add_argument("--w-constraint-penalty", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise SystemExit("--shard-index must satisfy 0 <= index < num_shards")

    base_cfg = load_config(args.config)
    method_entries = _extract_method_entries(base_cfg.get("methods", []))

    selected = discover_tunable_methods(
        method_entries,
        explicit_methods=args.methods,
        families=args.families,
    )
    selected = shard_methods(selected, args.num_shards, args.shard_index)

    if args.list:
        for m in selected:
            fam = method_family(m)
            print(f"{m}\t{fam}")
        return

    if not selected:
        print("> No tunable methods selected. "
              "Hint: ensure config has wrapped methods like '*__robust_ossm_ekf' / '*__robust_ossm_ukf'.")
        return

    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cli_weights = {
        "time_mae": args.w_time_mae,
        "time_rmse": args.w_time_rmse,
        "time_dtw": args.w_time_dtw,
        "freq_mae": args.w_freq_mae,
        "freq_rmse": args.w_freq_rmse,
        "fail_rate": args.w_fail_rate,
        "invalid_rate": args.w_invalid_rate,
        "clip_rate": args.w_clip_rate,
        "ccc_penalty": args.w_ccc_penalty,
        "snr_penalty": args.w_snr_penalty,
        "nis_truefail": args.w_nis_truefail,
        "nis_overstrict": args.w_nis_overstrict,
        "nis_mean_dev": args.w_nis_mean_dev,
        "alpha_R_excess": args.w_alpha_r_excess,
        "g_z_eff_deficit": args.w_g_z_eff_deficit,
        "coverage_dev": args.w_coverage_dev,
        "constraint_penalty": args.w_constraint_penalty,
    }
    objective_weights = load_objective_weights(base_cfg, cli_weights)
    objective_constraints = load_objective_constraints(base_cfg)
    objective_kfstd_reference = load_objective_kfstd_reference(base_cfg)
    objective_normalization = load_objective_normalization(base_cfg)
    split_cfg = load_split_cfg(base_cfg)
    pruning_cfg = load_pruning_cfg(base_cfg)
    repro_guard = load_repro_guard(base_cfg, Path(args.config).resolve(), cli_mode=args.repro_guard)
    check_repro_guard(repro_guard, context="startup")

    study_args = StudyArgs(
        base_cfg=base_cfg,
        config_path=Path(args.config).resolve(),
        output_root=output_root,
        n_trials=int(args.n_trials),
        timeout=args.timeout,
        sampler_seed=int(args.sampler_seed),
        pruner_enabled=not bool(args.no_prune),
        keep_artifacts=bool(args.keep_artifacts),
        em_mode=str(args.em_mode),
        failure_objective=float(args.failure_objective),
        objective_weights=objective_weights,
        objective_constraints=objective_constraints,
        objective_kfstd_reference=objective_kfstd_reference,
        objective_normalization=objective_normalization,
        search_space=load_search_space(base_cfg),
        family_defaults=load_family_defaults(base_cfg),
        split_cfg=split_cfg,
        pruning_cfg=pruning_cfg,
        repro_guard=repro_guard,
    )

    print(f"> Output root: {output_root}")
    print(f"> Methods selected ({len(selected)}):")
    for m in selected:
        print(f"  - {m} [{method_family(m)}]")
    print(f"> Optuna split cfg: {study_args.split_cfg}")
    print(f"> Optuna objective normalization: {study_args.objective_normalization}")
    print(f"> Optuna pruning cfg: {study_args.pruning_cfg}")
    if study_args.repro_guard.get("enabled", False):
        print(
            f"> Repro guard: mode={study_args.repro_guard.get('mode')} "
            f"tracked={len(study_args.repro_guard.get('tracked_paths', []))}"
        )
    else:
        print("> Repro guard: disabled")
    print(f"> Optuna kfstd references: {len(study_args.objective_kfstd_reference)} entries")
    if args.keep_artifacts:
        print("> Note: --keep-artifacts is enabled; this increases disk I/O and usually slows trials.")
    contract_path = write_tuning_contract(
        output_root=output_root,
        config_path=study_args.config_path,
        selected_methods=selected,
        search_space=study_args.search_space,
        family_defaults=study_args.family_defaults,
        objective_weights=study_args.objective_weights,
        objective_constraints=study_args.objective_constraints,
        objective_normalization=study_args.objective_normalization,
        split_cfg=study_args.split_cfg,
        pruning_cfg=study_args.pruning_cfg,
        repro_guard=study_args.repro_guard,
    )
    print(f"> Tuning contract: {contract_path}")

    for method in selected:
        family = method_family(method)
        if not family:
            print(f"> Skip {method}: unknown family")
            continue
        entry = method_entries.get(method)
        if not entry:
            print(f"> Skip {method}: method entry missing")
            continue
        if family not in study_args.search_space:
            print(f"> Skip {method}: no search-space for family '{family}'")
            continue

        print(f"\n>>> Tuning {method} ({family})")
        MethodStudy(method=method, family=family, method_entry=entry, args=study_args).optimize()

    leaderboard = write_leaderboard(output_root)
    trial_diag = write_trial_diagnostics(output_root)
    method_health = write_method_health_summary(output_root)
    print(f"> Leaderboard: {leaderboard}")
    print(f"> Trial diagnostics: {trial_diag}")
    print(f"> Method health summary: {method_health}")

    if args.export_best_preset:
        preset = export_best_preset(
            base_cfg=base_cfg,
            config_path=study_args.config_path,
            output_root=output_root,
            destination=args.export_best_preset,
        )
        if preset:
            print(f"> Exported best preset: {preset}")
        else:
            print("> Best preset export skipped: no best.json entries found")


if __name__ == "__main__":
    main()
