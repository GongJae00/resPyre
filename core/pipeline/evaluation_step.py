import os
import glob
import pickle
import json
import copy
import hashlib
import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.stats import chi2
from typing import List, Dict, Optional, Tuple

from core.evaluation import metrics as metrics_lib
from core.evaluation.metrics import getErrors, waveform_ccc, waveform_mae, waveform_dtw
from core.utils.common import tqdm, filter_RW, sig_windowing, sig_to_RPM
from core.pipeline.common import (
    resolve_target_run_dirs,
    resolve_frame_logs_for_run,
    resolve_frame_log_path,
    collect_expected_method_trials,
    infer_trial_key_from_data_stem,
)

PRIMARY_METRICS = ['RMSE', 'MAE', 'MAPE', 'R', 'SNR']
METHOD_QUALITY_SCHEMA_VERSION = "method_quality.v1"
CONFIG_USAGE_SCHEMA_VERSION = "config_usage.v1"


def _resolve_calibration_policy(eval_cfg: Optional[Dict]) -> Dict[str, float]:
    """Calibration policy used to separate strict-test failure vs practical miscalibration.

    Strict test:
    - NIS mean chi-square consistency test (`NIS_Pass` / `NIS_Pass_Strict`)

    Relaxed practical test:
    - |mean_nis - dof| <= nis_mean_tol_relaxed
    - |nis_inband - nis_inband_target| <= nis_inband_tol_relaxed

    This lets us label:
    - NIS_OverStrict: strict fail but relaxed pass
    - NIS_TrueFail: strict fail and relaxed fail
    """
    cfg = eval_cfg if isinstance(eval_cfg, dict) else {}
    cal = cfg.get("calibration", {}) if isinstance(cfg.get("calibration", {}), dict) else {}

    def _pick(name: str, default: float) -> float:
        if name in cal:
            try:
                return float(cal.get(name, default))
            except Exception:
                return float(default)
        try:
            return float(cfg.get(name, default))
        except Exception:
            return float(default)

    policy = {
        "dof": max(1.0, _pick("nis_dof", 1.0)),
        "alpha_strict": float(np.clip(_pick("nis_alpha_strict", 0.05), 1e-6, 0.5)),
        "inband_target": float(np.clip(_pick("nis_inband_target", 0.95), 0.0, 1.0)),
        "mean_tol_relaxed": max(0.0, _pick("nis_mean_tol_relaxed", 0.35)),
        "inband_tol_relaxed": max(0.0, _pick("nis_inband_tol_relaxed", 0.08)),
    }
    return policy

def _format_scalar(value, decimals=3):
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        abs_v = abs(float(value))
        # Preserve very small non-zero values instead of rounding to 0.000.
        if 0.0 < abs_v < 10 ** (-decimals):
            return f"{float(value):.2e}"
        return f"{float(value):.{decimals}f}"
    return str(value)

def _render_table(headers, rows):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    def _fmt_row(cells):
        return "|" + "|".join(f" {str(cell).ljust(col_widths[i])} " for i, cell in enumerate(cells)) + "|"

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines = [separator, _fmt_row(headers), separator]
    for row in rows:
        lines.append(_fmt_row(row))
    lines.append(separator)
    return "\n".join(lines)

def _flatten_leaf_key_paths(d: Dict, prefix: str = "") -> List[str]:
    out: List[str] = []
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten_leaf_key_paths(v, prefix=key))
        else:
            out.append(key)
    return out


def _build_config_usage(
    eval_cfg: Optional[Dict],
    gating_cfg: Optional[Dict],
    gating_scope: str = "evaluation_only",
) -> Dict[str, List[str]]:
    eval_cfg = eval_cfg if isinstance(eval_cfg, dict) else {}
    gating_cfg = gating_cfg if isinstance(gating_cfg, dict) else {}
    scope = str(gating_scope or "evaluation_only").strip().lower()
    if scope not in {"evaluation_only", "filter_time"}:
        raise ValueError(
            f"Invalid gating_scope '{gating_scope}'. Allowed values: ['evaluation_only', 'filter_time']"
        )

    eval_used = {
        "win_size", "stride", "min_hz", "max_hz",
        "calibration",
        "nis_dof", "nis_alpha_strict", "nis_inband_target",
        "nis_mean_tol_relaxed", "nis_inband_tol_relaxed",
    }
    eval_seen = set(eval_cfg.keys())
    eval_unused = sorted(k for k in eval_seen if k not in eval_used)

    gating_supported = {
        "profile",
        "debug.disable_gating",
        "spectral.peak_ratio_min",
        "spectral.prominence_min_db",
        "spectral.fwhm_df_guard",
        "spectral.fwhm_max_hz",
        "tracker.std_min_bpm",
        "tracker.unique_min",
        "tracker.saturation_max",
        "tracker.std_is_soft",
        "tracker.saturation_margin_hz",
    }
    gating_seen = set(_flatten_leaf_key_paths(gating_cfg))
    if scope == "filter_time":
        gating_used = sorted(k for k in gating_seen if k in gating_supported)
        gating_unused = sorted(k for k in gating_seen if k not in gating_supported)
    else:
        gating_used = []
        gating_unused = sorted(gating_seen)

    unused_all = [f"eval.{k}" for k in eval_unused] + [f"gating.{k}" for k in gating_unused]
    return {
        "gating_scope": scope,
        "eval_used_keys": sorted(eval_used),
        "eval_unused_keys": eval_unused,
        "gating_used_keys": gating_used,
        "gating_unused_keys": gating_unused,
        "unused_config_keys": sorted(unused_all),
    }


def _collect_method_unused_config_keys(run_dir: str) -> List[str]:
    keys = set()
    for pkl_path in sorted(glob.glob(os.path.join(run_dir, "data", "*.pkl"))):
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
            meta_obj = payload.get("meta")
            meta = _parse_estimate_meta(meta_obj)
            for k in meta.get("unused_config_keys", []) if isinstance(meta, dict) else []:
                sval = str(k).strip()
                if sval:
                    keys.add(sval)
    return sorted(keys)


def _write_config_usage_artifact(
    logs_dir: str,
    usage_payload: Dict[str, List[str]],
    strict_key_usage: bool = False,
) -> Dict:
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, "config_usage.json")
    existing = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fp:
                loaded = json.load(fp)
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    payload = {
        "schema_version": CONFIG_USAGE_SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "strict_key_usage": bool(strict_key_usage),
        **usage_payload,
    }

    # Preserve top-level audit keys written by main() when present.
    for key in ("top_level_used_keys", "top_level_unused_keys"):
        if key in existing and key not in payload:
            payload[key] = existing[key]

    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    # Strict mode: any unused config key becomes an explicit error.
    strict_unused = list(payload.get("unused_config_keys", []) or [])
    top_unused = payload.get("top_level_unused_keys", [])
    if isinstance(top_unused, list):
        strict_unused.extend([f"top.{str(k)}" for k in top_unused])
    strict_unused = sorted(set(str(x) for x in strict_unused if str(x).strip()))
    if strict_key_usage and strict_unused:
        raise ValueError(
            "strict_key_usage=True and unused config keys were found: "
            + ", ".join(strict_unused)
        )
    return payload


def _load_frame_log_npz_strict(path: str) -> Dict[str, np.ndarray]:
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"Failed to load frame log '{path}': {exc}") from exc

    if "fields" not in z or "data" not in z:
        raise ValueError(
            f"Malformed frame log '{path}': required keys ['fields', 'data'] not found."
        )
    fields = [str(f) for f in z["fields"]]
    data = np.asarray(z["data"])
    if data.ndim != 2:
        raise ValueError(f"Malformed frame log '{path}': data must be 2D, got ndim={data.ndim}.")
    if data.shape[1] != len(fields):
        raise ValueError(
            f"Malformed frame log '{path}': column mismatch data.shape[1]={data.shape[1]} vs len(fields)={len(fields)}."
        )
    idx = {f: i for i, f in enumerate(fields)}
    return {"fields": np.asarray(fields, dtype=object), "data": data, "idx": idx}


def _series_stats(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a)
    if a.size == 0:
        return {
            "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
            "median": np.nan, "p95": np.nan, "nan_rate": np.nan
        }
    if not finite.any():
        return {
            "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan,
            "median": np.nan, "p95": np.nan, "nan_rate": 1.0
        }
    af = a[finite]
    return {
        "mean": float(np.mean(af)),
        "std": float(np.std(af)),
        "min": float(np.min(af)),
        "max": float(np.max(af)),
        "median": float(np.median(af)),
        "p95": float(np.percentile(af, 95)),
        "nan_rate": float(np.mean(~finite)),
    }


def _method_quality_columns() -> List[str]:
    return [
        "schema_version", "method", "trial", "n_frames",
        "missing_frame_log", "missing_reason",
        "frame_log_filename_used", "frame_log_resolution_mode", "frame_log_suffix_used",
        "welch_df_hz",
        "q_vis_mean", "q_vis_std", "q_vis_nan_rate",
        "q_drift_mean", "q_cons_mean", "q_out_mean", "q_harm_mean", "q_burst_rate",
        "alpha_R_mean", "alpha_Q_mean", "g_t_mean", "g_z_mean", "w_h_mean", "g_z_eff_mean",
        "lambda_mean", "lambda_lt1_frac", "lambda_low_frac",
        "qx_base_mean", "qf_base_mean", "rv_base_mean",
        "qx_used_mean", "qf_used_mean", "rv_scaled_mean", "R_post_mean",
        "freq_low_clip_rate", "freq_high_clip_rate", "z_low_clip_rate", "z_high_clip_rate",
        "invalid_row_rate",
    ]


def _empty_method_quality_row(method: str, trial: str, reason: str) -> Dict[str, float]:
    row = {k: np.nan for k in _method_quality_columns()}
    row.update({
        "schema_version": METHOD_QUALITY_SCHEMA_VERSION,
        "method": str(method),
        "trial": str(trial),
        "n_frames": 0,
        "missing_frame_log": True,
        "missing_reason": str(reason),
        "frame_log_filename_used": "",
        "frame_log_resolution_mode": "missing",
        "frame_log_suffix_used": -1,
    })
    return row


def _summarize_method_quality_from_log(
    path: str,
    method: str,
    trial: str,
    min_hz: float,
    max_hz: float,
    resolution_info: Optional[Dict[str, object]] = None,
    welch_df_hz: float = np.nan,
) -> Dict[str, float]:
    z_lo = float(np.log(max(min_hz, 1e-6)))
    z_hi = float(np.log(max(max_hz, min_hz + 1e-6)))
    obj = _load_frame_log_npz_strict(path)
    data = obj["data"]
    idx = obj["idx"]

    def col(name: str, default=np.nan):
        if name not in idx:
            return np.full(data.shape[0], default, dtype=np.float64)
        return np.asarray(data[:, idx[name]], dtype=np.float64)

    q_vis = col("q_vis")
    q_drift = col("q_drift")
    q_cons = col("q_cons")
    q_out = col("q_out")
    q_harm = col("q_harm")
    q_burst = col("q_burst")
    alpha_R = col("alpha_R")
    alpha_Q = col("alpha_Q")
    g_t = col("g_t")
    g_z = col("g_z")
    w_h = col("w_h")
    g_z_eff = col("g_z_eff", default=np.nan)
    if not np.isfinite(g_z_eff).any():
        g_z_eff = g_z * w_h
    lam = col("lambda_t")
    qx_base = col("qx_base")
    qf_base = col("qf_base")
    rv_base = col("rv_base")
    qx_used = col("qx_used", default=np.nan)
    if not np.isfinite(qx_used).any():
        qx_used = col("qx_eff")
    qf_used = col("qf_used", default=np.nan)
    if not np.isfinite(qf_used).any():
        qf_used = col("qf_eff")
    rv_scaled = col("rv_scaled", default=np.nan)
    if not np.isfinite(rv_scaled).any():
        rv_scaled = col("rv_eff")
    r_post = col("R_post", default=np.nan)
    if not np.isfinite(r_post).any():
        r_post = col("R_eff")

    freq_hz = col("freq_hz")
    z_val = col("z")
    invalid_row = (
        ~np.isfinite(q_vis) |
        ~np.isfinite(alpha_R) |
        ~np.isfinite(alpha_Q) |
        ~np.isfinite(g_t) |
        ~np.isfinite(lam)
    )

    vis_stats = _series_stats(q_vis)
    lam_f = lam[np.isfinite(lam)]
    q_burst_f = q_burst[np.isfinite(q_burst)]
    freq_f = freq_hz[np.isfinite(freq_hz)]
    z_f = z_val[np.isfinite(z_val)]

    row = {
        "schema_version": METHOD_QUALITY_SCHEMA_VERSION,
        "method": method,
        "trial": trial,
        "n_frames": int(data.shape[0]),
        "missing_frame_log": False,
        "missing_reason": "",
        "frame_log_filename_used": "",
        "frame_log_resolution_mode": "",
        "frame_log_suffix_used": -1,
        "welch_df_hz": float(welch_df_hz) if np.isfinite(welch_df_hz) else np.nan,
        "q_vis_mean": vis_stats["mean"],
        "q_vis_std": vis_stats["std"],
        "q_vis_nan_rate": vis_stats["nan_rate"],
        "q_drift_mean": _series_stats(q_drift)["mean"],
        "q_cons_mean": _series_stats(q_cons)["mean"],
        "q_out_mean": _series_stats(q_out)["mean"],
        "q_harm_mean": _series_stats(q_harm)["mean"],
        "q_burst_rate": float(np.mean(q_burst_f > 0.5)) if q_burst_f.size else np.nan,
        "alpha_R_mean": _series_stats(alpha_R)["mean"],
        "alpha_Q_mean": _series_stats(alpha_Q)["mean"],
        "g_t_mean": _series_stats(g_t)["mean"],
        "g_z_mean": _series_stats(g_z)["mean"],
        "w_h_mean": _series_stats(w_h)["mean"],
        "g_z_eff_mean": _series_stats(g_z_eff)["mean"],
        "lambda_mean": _series_stats(lam)["mean"],
        "lambda_lt1_frac": float(np.mean(lam_f < 1.0)) if lam_f.size else np.nan,
        "lambda_low_frac": float(np.mean(lam_f < 0.5)) if lam_f.size else np.nan,
        "qx_base_mean": _series_stats(qx_base)["mean"],
        "qf_base_mean": _series_stats(qf_base)["mean"],
        "rv_base_mean": _series_stats(rv_base)["mean"],
        "qx_used_mean": _series_stats(qx_used)["mean"],
        "qf_used_mean": _series_stats(qf_used)["mean"],
        "rv_scaled_mean": _series_stats(rv_scaled)["mean"],
        "R_post_mean": _series_stats(r_post)["mean"],
        "freq_low_clip_rate": float(np.mean(freq_f <= (min_hz + 1e-6))) if freq_f.size else np.nan,
        "freq_high_clip_rate": float(np.mean(freq_f >= (max_hz - 1e-6))) if freq_f.size else np.nan,
        "z_low_clip_rate": float(np.mean(z_f <= (z_lo + 1e-6))) if z_f.size else np.nan,
        "z_high_clip_rate": float(np.mean(z_f >= (z_hi - 1e-6))) if z_f.size else np.nan,
        "invalid_row_rate": float(np.mean(invalid_row)) if invalid_row.size else np.nan,
    }
    if isinstance(resolution_info, dict):
        row["frame_log_filename_used"] = str(resolution_info.get("frame_log_filename_used", "") or "")
        row["frame_log_resolution_mode"] = str(resolution_info.get("frame_log_resolution_mode", "") or "")
        try:
            row["frame_log_suffix_used"] = int(resolution_info.get("frame_log_suffix_used", -1))
        except Exception:
            row["frame_log_suffix_used"] = -1
    return row


def _parse_estimate_meta(meta_obj) -> Dict:
    if isinstance(meta_obj, dict):
        return meta_obj
    if isinstance(meta_obj, str):
        txt = meta_obj.strip()
        if txt.startswith("{") and txt.endswith("}"):
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
    return {}


def _collect_expected_method_trials(run_dir: str) -> List[Dict[str, object]]:
    expected_pairs = collect_expected_method_trials(run_dir)
    welch_by_pair: Dict[Tuple[str, str], float] = {}
    for pkl_path in sorted(glob.glob(os.path.join(run_dir, "data", "*.pkl"))):
        try:
            with open(pkl_path, "rb") as fp:
                obj = pickle.load(fp)
        except Exception:
            continue
        fname = os.path.splitext(os.path.basename(pkl_path))[0]
        dataset_token = fname.split("_", 1)[0] if "_" in fname else None
        trial_key = infer_trial_key_from_data_stem(fname, dataset_token=dataset_token)
        for entry in obj.get("estimates", []) if isinstance(obj, dict) else []:
            method = entry.get("method") if isinstance(entry, dict) else None
            if method:
                payload = entry.get("estimate", entry) if isinstance(entry, dict) else {}
                est_meta = _parse_estimate_meta(payload.get("meta") if isinstance(payload, dict) else None)
                welch_df_hz = np.nan
                if isinstance(est_meta, dict):
                    try:
                        welch_df_hz = float(est_meta.get("welch_df_hz", np.nan))
                    except Exception:
                        welch_df_hz = np.nan
                welch_by_pair[(str(method), str(trial_key))] = welch_df_hz
    expected: List[Dict[str, object]] = []
    seen = set()
    for item in expected_pairs:
        method = str(item.get("method", "")).strip()
        trial = str(item.get("trial", "")).strip()
        if not method or not trial:
            continue
        key = (method, trial)
        if key in seen:
            continue
        seen.add(key)
        expected.append({
            "method": method,
            "trial": trial,
            "welch_df_hz": welch_by_pair.get(key, np.nan),
        })
    return expected


def _filter_expected_trials_for_frame_logs(
    run_dir: str,
    expected_trials: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Keep only expected trials for methods that have frame-log directories."""
    out: List[Dict[str, object]] = []
    for item in expected_trials:
        method = str(item.get("method", "")).strip()
        trial = str(item.get("trial", "")).strip()
        if not method or not trial:
            continue
        log_dir = os.path.join(run_dir, "aux", method.replace(" ", "_"), "frame_logs")
        if os.path.isdir(log_dir):
            out.append(item)
    return out


def _compute_method_quality_records(
    run_dir: str,
    min_hz: float,
    max_hz: float,
    expected_trials: Optional[List[Dict[str, object]]] = None,
    selected_logs: Optional[Dict[str, Dict[str, str]]] = None,
    selected_details: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
) -> pd.DataFrame:
    cols = _method_quality_columns()
    expected = expected_trials if isinstance(expected_trials, list) else _collect_expected_method_trials(run_dir)
    rows: List[Dict[str, float]] = []
    selected_logs = selected_logs if isinstance(selected_logs, dict) else {}
    selected_details = selected_details if isinstance(selected_details, dict) else {}

    # Ensure expected trials are represented; emit explicit missing markers.
    for item in expected:
        method = str(item.get("method", "")).strip()
        trial = str(item.get("trial", "")).strip()
        try:
            welch_df_hz = float(item.get("welch_df_hz", np.nan))
        except Exception:
            welch_df_hz = np.nan
        if not method or not trial:
            continue
        method_paths = selected_logs.get(method)
        if method_paths is None:
            method_paths = selected_logs.get(method.replace(" ", "_"), {})
        method_details = selected_details.get(method)
        if method_details is None:
            method_details = selected_details.get(method.replace(" ", "_"), {})
        path = method_paths.get(trial) if isinstance(method_paths, dict) else None
        resolution = method_details.get(trial, {}) if isinstance(method_details, dict) else {}
        if not path or not os.path.exists(path):
            row = _empty_method_quality_row(method, trial, "frame_log_missing")
            row["welch_df_hz"] = welch_df_hz
            row["frame_log_filename_used"] = str(resolution.get("frame_log_filename_used", "") or "")
            row["frame_log_resolution_mode"] = str(resolution.get("frame_log_resolution_mode", "missing"))
            try:
                row["frame_log_suffix_used"] = int(resolution.get("frame_log_suffix_used", -1))
            except Exception:
                row["frame_log_suffix_used"] = -1
            rows.append(row)
            continue
        try:
            rows.append(
                _summarize_method_quality_from_log(
                    path, method, trial, min_hz, max_hz,
                    resolution_info=resolution,
                    welch_df_hz=welch_df_hz,
                )
            )
        except Exception as exc:
            row = _empty_method_quality_row(method, trial, f"frame_log_unreadable:{exc}")
            row["welch_df_hz"] = welch_df_hz
            row["frame_log_filename_used"] = str(resolution.get("frame_log_filename_used", "") or "")
            row["frame_log_resolution_mode"] = str(resolution.get("frame_log_resolution_mode", ""))
            try:
                row["frame_log_suffix_used"] = int(resolution.get("frame_log_suffix_used", -1))
            except Exception:
                row["frame_log_suffix_used"] = -1
            rows.append(row)

    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def _write_method_quality_artifacts(
    run_dir: str,
    eval_settings: Dict,
    expected_trials: Optional[List[Dict[str, object]]] = None,
    frame_log_strict: bool = True,
) -> None:
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    expected_raw = expected_trials if isinstance(expected_trials, list) else _collect_expected_method_trials(run_dir)
    expected = _filter_expected_trials_for_frame_logs(run_dir, expected_raw)
    resolution = resolve_frame_logs_for_run(
        run_dir,
        expected_trials=[{"method": str(x.get("method", "")), "trial": str(x.get("trial", ""))} for x in expected],
        strict=bool(frame_log_strict),
        allow_empty=True,
    )
    selected = resolution.get("canonical", {})
    orphans = resolution.get("extras", [])
    missing_trials = resolution.get("missing", [])
    resolver_diag = resolution.get("diag", {}) if isinstance(resolution, dict) else {}
    selected_details = resolver_diag.get("selected_details", {}) if isinstance(resolver_diag, dict) else {}

    min_hz = float(eval_settings.get("min_hz", 0.08))
    max_hz = float(eval_settings.get("max_hz", 0.5))
    df = _compute_method_quality_records(
        run_dir,
        min_hz=min_hz,
        max_hz=max_hz,
        expected_trials=expected,
        selected_logs=selected,
        selected_details=selected_details,
    )
    csv_path = os.path.join(logs_dir, "method_quality.csv")
    df.to_csv(csv_path, index=False)
    orphan_csv_path = os.path.join(logs_dir, "method_quality_orphan_logs.csv")
    orphan_cols = ["method", "trial_key", "path", "filename", "suffix", "mtime", "reason"]
    if orphans:
        pd.DataFrame(orphans).to_csv(orphan_csv_path, index=False)
    else:
        pd.DataFrame(columns=orphan_cols).to_csv(orphan_csv_path, index=False)

    method_summary = []
    resolution_counts: Dict[str, int] = {}
    if not df.empty:
        grouped = df.groupby("method", dropna=False)
        for method, g in grouped:
            g_valid = g.loc[~pd.to_numeric(g["missing_frame_log"], errors="coerce").fillna(False).astype(bool)]
            method_summary.append({
                "method": str(method),
                "n_trials": int(len(g)),
                "n_frames_total": int(pd.to_numeric(g["n_frames"], errors="coerce").fillna(0).sum()),
                "n_trials_missing_logs": int(pd.to_numeric(g["missing_frame_log"], errors="coerce").fillna(0).sum()),
                "q_vis_mean": float(pd.to_numeric(g_valid["q_vis_mean"], errors="coerce").mean()),
                "alpha_R_mean": float(pd.to_numeric(g_valid["alpha_R_mean"], errors="coerce").mean()),
                "alpha_Q_mean": float(pd.to_numeric(g_valid["alpha_Q_mean"], errors="coerce").mean()),
                "g_t_mean": float(pd.to_numeric(g_valid["g_t_mean"], errors="coerce").mean()),
                "lambda_mean": float(pd.to_numeric(g_valid["lambda_mean"], errors="coerce").mean()),
                "R_post_mean": float(pd.to_numeric(g_valid["R_post_mean"], errors="coerce").mean()),
                "invalid_row_rate_mean": float(pd.to_numeric(g_valid["invalid_row_rate"], errors="coerce").mean()),
                "welch_df_hz_mean": float(pd.to_numeric(g_valid["welch_df_hz"], errors="coerce").mean()),
            })
        for mode in df["frame_log_resolution_mode"].fillna("missing").astype(str).tolist():
            resolution_counts[mode] = int(resolution_counts.get(mode, 0) + 1)

    fp_source = {
        "schema_version": METHOD_QUALITY_SCHEMA_VERSION,
        "eval_settings": eval_settings,
        "columns": _method_quality_columns(),
    }
    fp_bytes = json.dumps(fp_source, ensure_ascii=False, sort_keys=True).encode("utf-8")
    config_fingerprint = hashlib.sha256(fp_bytes).hexdigest()[:16]
    artifact_fp = hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()[:16]
    n_trials_total = int(len(df))
    n_trials_missing = int(pd.to_numeric(df["missing_frame_log"], errors="coerce").fillna(0).sum()) if not df.empty else 0
    n_trials_with_logs = int(max(0, n_trials_total - n_trials_missing))
    valid_df = df.loc[~pd.to_numeric(df["missing_frame_log"], errors="coerce").fillna(False).astype(bool)] if not df.empty else df
    summary = {
        "schema_version": METHOD_QUALITY_SCHEMA_VERSION,
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "config_fingerprint": config_fingerprint,
        "artifact_fingerprint": artifact_fp,
        "columns": _method_quality_columns(),
        "n_rows": int(len(df)),
        "n_methods": int(len(method_summary)),
        "n_trials_total": n_trials_total,
        "n_trials_total_expected": n_trials_total,
        "n_trials_with_logs": n_trials_with_logs,
        "n_trials_missing_logs": n_trials_missing,
        "n_trials_missing_expected_in_resolver": int(len(missing_trials)),
        "n_orphan_logs_ignored": int(len(orphans)),
        "selection_policy": resolver_diag.get("selection_policy"),
        "run_instance_started_at_used": resolver_diag.get("run_instance_started_at_used"),
        "resolver_diag": resolver_diag,
        "method_summary": method_summary,
        "frame_log_resolution_counts": resolution_counts,
        "frame_log_fallback_count": int(
            resolution_counts.get("fallback_base", 0) +
            resolution_counts.get("fallback_suffix", 0)
        ),
        "welch_df_hz_stats": {
            "mean": float(pd.to_numeric(valid_df["welch_df_hz"], errors="coerce").mean()) if not valid_df.empty else np.nan,
            "min": float(pd.to_numeric(valid_df["welch_df_hz"], errors="coerce").min()) if not valid_df.empty else np.nan,
            "max": float(pd.to_numeric(valid_df["welch_df_hz"], errors="coerce").max()) if not valid_df.empty else np.nan,
        },
        "notes": {
            "units": {
                "freq": "Hz",
                "q_*": "[0,1] or score",
                "alpha_*": "unitless scale",
                "lambda_t": "unitless Student-t latent scale",
                "R_post": "measurement variance",
            }
        },
    }
    with open(os.path.join(logs_dir, "method_quality_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


def _infer_trial_key_from_fname(fname: str, dataset_token: Optional[str] = None) -> str:
    """Infer trial key used by aux/frame_logs from a data filename stem.

    Example:
      fname='cohface_1_0' -> trial_key='1_0'
    """
    stem = str(fname or "").strip()
    if not stem:
        return ""
    if dataset_token:
        prefix = f"{dataset_token.lower()}_"
        if stem.lower().startswith(prefix):
            return stem[len(prefix):]
    if "_" in stem:
        return stem.split("_", 1)[1]
    return stem


def _load_frame_log(
    run_dir: str,
    method_name: str,
    trial_key: str,
    cache: Dict[str, Dict[str, np.ndarray]],
    selected_logs: Optional[Dict[str, Dict[str, str]]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Load frame log for (method, trial) with cache."""
    path = None
    if isinstance(selected_logs, dict):
        method_paths = selected_logs.get(str(method_name))
        if method_paths is None:
            method_paths = selected_logs.get(str(method_name).replace(" ", "_"))
        if isinstance(method_paths, dict):
            path = method_paths.get(str(trial_key))
        if not path:
            return None
    else:
        method_slug = str(method_name).replace(" ", "_")
        aux_dir = os.path.join(run_dir, "aux", method_slug)
        path, _ = resolve_frame_log_path(aux_dir, trial_key, strict=True)
        if path is None:
            return None
    if path in cache:
        return cache[path]
    try:
        out = _load_frame_log_npz_strict(path)
        cache[path] = out
        return out
    except Exception as exc:
        raise ValueError(f"Malformed frame log '{path}': {exc}") from exc


def _instantaneous_freq_hz(sig: np.ndarray, fs: float, min_hz: float, max_hz: float) -> np.ndarray:
    """Approximate instantaneous frequency from analytic phase (Hz)."""
    x = np.asarray(sig, dtype=np.float64).reshape(-1)
    if x.size < 4 or fs <= 0:
        return np.full(x.shape, np.nan, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    analytic = sps.hilbert(x)
    phase = np.unwrap(np.angle(analytic))
    inst = np.diff(phase) / (2.0 * np.pi) * fs
    if inst.size == 0:
        return np.full(x.shape, np.nan, dtype=np.float64)
    inst = np.concatenate([[inst[0]], inst])
    inst = np.clip(inst, min_hz, max_hz)
    return inst.astype(np.float64)


def _compute_filter_diag_record(
    log_obj: Optional[Dict[str, np.ndarray]],
    method_name: str,
    fname: str,
    data_file_rel: str,
    fps: float,
    est_track_hz: np.ndarray,
    gt_inst_hz: np.ndarray,
    calibration_policy: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute per-trial filter diagnostics from frame logs and tracks."""
    rec = {
        "video": fname,
        "method": method_name,
        "Fail_Total": np.nan,
        "Fail_Div": np.nan,
        "Fail_Slip": np.nan,
        "Fail_Lock": np.nan,
        "Fail_Double": np.nan,
        "NIS_Mean": np.nan,
        "NIS_Pass": np.nan,
        "NIS_Pass_Strict": np.nan,
        "NIS_Pass_Relaxed": np.nan,
        "NIS_OverStrict": np.nan,
        "NIS_TrueFail": np.nan,
        "NIS_InBand": np.nan,
        "NIS_InBand_Target": np.nan,
        "NIS_InBand_DevAbs": np.nan,
        "NIS_Mean_DevAbs": np.nan,
        "NIS_CI_Lower": np.nan,
        "NIS_CI_Upper": np.nan,
        "NIS_pval": np.nan,
        "Lambda_Mean": np.nan,
        "Lambda_LT1_Frac": np.nan,
        "Lambda_LowFrac": np.nan,
        "Coverage95": np.nan,
        "FreqStd_Mean": np.nan,
        "Stability_Sec": np.nan,
        "n_frames": np.nan,
        "data_file": data_file_rel,
    }

    # Stability can be measured from any method that outputs track_hz.
    if est_track_hz is not None:
        est_track = np.asarray(est_track_hz, dtype=np.float64).reshape(-1)
        est_track = est_track[np.isfinite(est_track)]
        if est_track.size > 1:
            st = metrics_lib.stability_duration(est_track, fs=max(float(fps), 1.0), eps_hz=0.02)
            rec["Stability_Sec"] = float(st.get("max_stable_sec", np.nan))

    if log_obj is None:
        return rec

    arr = log_obj["data"]
    idx = log_obj["idx"]
    n = float(arr.shape[0])
    rec["n_frames"] = n

    def col(name: str) -> Optional[np.ndarray]:
        i = idx.get(name)
        if i is None:
            return None
        return np.asarray(arr[:, i], dtype=np.float64)

    fail_cols = {
        "Fail_Div": col("fail_diverge"),
        "Fail_Slip": col("fail_slip"),
        "Fail_Lock": col("fail_lock"),
        "Fail_Double": col("fail_double"),
    }
    for key, values in fail_cols.items():
        if values is not None and values.size:
            rec[key] = float(np.nanmean(values))
    fail_vals = [rec[k] for k in ("Fail_Div", "Fail_Slip", "Fail_Lock", "Fail_Double") if np.isfinite(rec[k])]
    if fail_vals:
        rec["Fail_Total"] = float(max(fail_vals))

    cal = calibration_policy if isinstance(calibration_policy, dict) else {}
    dof = int(max(1, round(float(cal.get("dof", 1.0)))))
    alpha_strict = float(np.clip(cal.get("alpha_strict", 0.05), 1e-6, 0.5))
    inband_target = float(np.clip(cal.get("inband_target", 0.95), 0.0, 1.0))
    mean_tol_relaxed = float(max(0.0, cal.get("mean_tol_relaxed", 0.35)))
    inband_tol_relaxed = float(max(0.0, cal.get("inband_tol_relaxed", 0.08)))
    rec["NIS_InBand_Target"] = inband_target

    nis = col("nis")
    if nis is not None and np.isfinite(nis).any():
        nis_cal = metrics_lib.nis_calibration_chi2(nis, dof=dof, alpha=alpha_strict)
        rec["NIS_Mean"] = float(nis_cal.get("mean_nis", np.nan))
        strict_pass = bool(nis_cal.get("pass_chi2", False))
        rec["NIS_Pass"] = 1.0 if strict_pass else 0.0
        rec["NIS_Pass_Strict"] = rec["NIS_Pass"]
        rec["NIS_pval"] = float(nis_cal.get("pval", np.nan))
        rec["NIS_CI_Lower"] = float(nis_cal.get("ci_lower", np.nan))
        rec["NIS_CI_Upper"] = float(nis_cal.get("ci_upper", np.nan))
        if np.isfinite(rec["NIS_Mean"]):
            rec["NIS_Mean_DevAbs"] = float(abs(rec["NIS_Mean"] - float(dof)))

        nis_f = nis[np.isfinite(nis)]
        if nis_f.size > 0:
            lo = float(chi2.ppf(0.025, dof))
            hi = float(chi2.ppf(0.975, dof))
            rec["NIS_InBand"] = float(np.mean((nis_f >= lo) & (nis_f <= hi)))
            rec["NIS_InBand_DevAbs"] = float(abs(rec["NIS_InBand"] - inband_target))

        relaxed_pass = False
        if np.isfinite(rec["NIS_Mean_DevAbs"]) and np.isfinite(rec["NIS_InBand_DevAbs"]):
            relaxed_pass = bool(
                rec["NIS_Mean_DevAbs"] <= mean_tol_relaxed and
                rec["NIS_InBand_DevAbs"] <= inband_tol_relaxed
            )
        rec["NIS_Pass_Relaxed"] = 1.0 if relaxed_pass else 0.0
        rec["NIS_OverStrict"] = 1.0 if ((not strict_pass) and relaxed_pass) else 0.0
        rec["NIS_TrueFail"] = 1.0 if ((not strict_pass) and (not relaxed_pass)) else 0.0

    lam = col("lambda_t")
    if lam is not None and np.isfinite(lam).any():
        rec["Lambda_Mean"] = float(np.nanmean(lam))
        rec["Lambda_LT1_Frac"] = float(np.nanmean(lam < 1.0))
        rec["Lambda_LowFrac"] = float(np.nanmean(lam < 0.5))

    # Coverage requires track + per-frame sigma + GT instantaneous frequency.
    freq_std = col("freq_std_hz")
    if freq_std is not None and est_track_hz is not None and gt_inst_hz is not None:
        est_track = np.asarray(est_track_hz, dtype=np.float64).reshape(-1)
        sig_track = np.asarray(freq_std, dtype=np.float64).reshape(-1)
        gt_track = np.asarray(gt_inst_hz, dtype=np.float64).reshape(-1)
        finite_sig = sig_track[np.isfinite(sig_track)]
        if finite_sig.size:
            rec["FreqStd_Mean"] = float(np.mean(finite_sig))
        m = min(est_track.size, sig_track.size, gt_track.size)
        if m > 10:
            cov = metrics_lib.coverage_percentage(est_track[:m], gt_track[:m], sig_track[:m], k=2.0)
            rec["Coverage95"] = float(cov.get("coverage", np.nan))

    return rec

def run_evaluation(
    results_dir: str,
    run_label: str = None,
    win_size: float = 30.0,
    stride: float = 1.0,
    min_hz: float = 0.08,
    max_hz: float = 0.5,
    gating: Optional[Dict] = None,
    eval_cfg: Optional[Dict] = None,
    gating_scope: str = "evaluation_only",
    strict_key_usage: bool = False,
    frame_log_strict: bool = True,
):
    """
    Scans results and computes aggregate metrics using a strict Dual-Domain approach:
    1. Time Domain (Waveform Fidelity): CCC, Aligned MAE/RMSE, SNR, I:E Ratio, PPI MAE.
    2. Freq Domain (Rate Accuracy): MAE, RMSE, MAPE, Spectral SNR, Bland-Altman, KL-Div, Entropy.
    """
    from core.utils.common import get_SNR, Welch_rpm
    from core.evaluation.metrics import (
        calculate_cross_corr_alignment, bland_altman_stats, 
        calculate_spectral_snr, calculate_breathing_dynamics, calculate_spectral_shape_metrics,
        calculate_dtw_distance
    )

    # 1. Identify target directories
    target_dirs = resolve_target_run_dirs(results_dir, run_label)

    if not target_dirs:
        print(f"> Evaluation: No result directories found matching '{run_label}' in '{results_dir}'")
        return

    print(f"\n> Starting Dual-Domain Evaluation for {len(target_dirs)} dataset(s)...")
    
    for d_dir in target_dirs:
        dataset_name = os.path.basename(d_dir)
        data_dir = os.path.join(d_dir, 'data')
        metrics_dir = os.path.join(d_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)

        scope = str(gating_scope or "evaluation_only").strip().lower()
        if scope not in {"evaluation_only", "filter_time"}:
            raise ValueError(
                f"Invalid gating_scope '{gating_scope}'. Allowed values: ['evaluation_only', 'filter_time']"
            )
        calibration_policy = _resolve_calibration_policy(eval_cfg)

        # Persist evaluation settings + config usage for audit/metadata.
        config_usage = _build_config_usage(eval_cfg, gating, gating_scope=scope)
        method_unused = _collect_method_unused_config_keys(d_dir)
        if method_unused:
            combined_unused = set(config_usage.get("unused_config_keys", []))
            combined_unused.update(method_unused)
            config_usage["method_unused_config_keys"] = sorted(method_unused)
            config_usage["unused_config_keys"] = sorted(combined_unused)
        eval_settings = {
            'win_size': float(win_size),
            'stride': float(stride),
            'min_hz': float(min_hz),
            'max_hz': float(max_hz),
            'frame_log_strict': bool(frame_log_strict),
            'gating_scope': scope,
            'gating': copy.deepcopy(gating) if isinstance(gating, dict) else {},
            'calibration': calibration_policy,
            'gating_note': (
                "gating is evaluation-audit metadata; filter-time behavior is unchanged "
                "unless wrapped methods explicitly run with gating_scope='filter_time'."
            ),
            **config_usage,
        }
        _write_config_usage_artifact(
            os.path.join(d_dir, 'logs'),
            config_usage,
            strict_key_usage=bool(strict_key_usage),
        )
        try:
            with open(os.path.join(metrics_dir, 'eval_settings.json'), 'w', encoding='utf-8') as fp:
                json.dump(eval_settings, fp, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"> Warning: failed to write eval_settings.json for {dataset_name}: {exc}")

        pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
        
        if not pkl_files:
            _write_method_quality_artifacts(d_dir, eval_settings, expected_trials=[])
            print(f"> Evaluation: no trial PKLs found in '{data_dir}' (wrote empty method_quality artifacts)")
            continue

        print(f"\n>> Evaluating Dataset: {dataset_name} ({len(pkl_files)} trials)")
        
        all_time_domain_records = []
        all_freq_domain_records = []
        all_filter_diag_records = []
        all_waveform_records = []   # z_full waveform metrics (PARH-OSSM dual output)
        frame_log_cache: Dict[str, Dict[str, np.ndarray]] = {}
        expected_trials = _collect_expected_method_trials(d_dir)
        expected_trials_for_logs = _filter_expected_trials_for_frame_logs(d_dir, expected_trials)
        resolution = resolve_frame_logs_for_run(
            d_dir,
            expected_trials=[{"method": str(x.get("method", "")), "trial": str(x.get("trial", ""))} for x in expected_trials_for_logs],
            strict=bool(frame_log_strict),
            allow_empty=True,
        )
        selected_logs = resolution.get("canonical", {})
        resolver_diag = resolution.get("diag", {}) if isinstance(resolution, dict) else {}
        try:
            with open(os.path.join(metrics_dir, "resolver_diag.json"), "w", encoding="utf-8") as fp:
                json.dump({
                    "schema_version": "frame_log_resolver_diag.v1",
                    "frame_log_strict": bool(frame_log_strict),
                    "resolver": resolver_diag,
                    "missing": resolution.get("missing", []),
                    "extras": resolution.get("extras", []),
                }, fp, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"> Warning: failed to write resolver_diag.json for {dataset_name}: {exc}")
        
        # Metrics definitions
        time_metrics_list = ['CCC', 'MAE', 'RMSE', 'SNR_Time', 'Latency', 'DTW_Dist', 'IE_Err', 'PPI_MAE']
        freq_metrics_list = ['MAE', 'RMSE', 'MAPE', 'PearsonR', 'SNR_Spec', 'Bias', 'LoA_Width', 'KL_Div', 'Entropy_Err']
        filter_diag_metrics_list = [
            'Fail_Total', 'Fail_Div', 'Fail_Slip', 'Fail_Lock', 'Fail_Double',
            'NIS_Mean', 'NIS_Pass', 'NIS_OverStrict', 'NIS_TrueFail', 'NIS_Pass_Relaxed',
            'NIS_InBand', 'NIS_Mean_DevAbs', 'NIS_InBand_DevAbs', 'NIS_pval',
            'Lambda_Mean', 'Lambda_LT1_Frac', 'Lambda_LowFrac',
            'Coverage95', 'FreqStd_Mean', 'Stability_Sec',
        ]

        for pkl_path in tqdm(pkl_files, desc="Calculating Metrics"):
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {pkl_path}: {e}")
                continue

            gt_signal = data.get('gt')
            fps = data.get('fps', 30.0)
            fs_gt = data.get('fs_gt', fps)
            fname = os.path.splitext(os.path.basename(pkl_path))[0]
            
            if gt_signal is None or not isinstance(gt_signal, np.ndarray):
                continue
            
            # Preprocess GT for both domains
            # 1. Bandpass Filter (0.08 - 0.5 Hz)
            gt_filt = filter_RW(gt_signal, fs_gt, lo=min_hz, hi=max_hz)
            # 2. Z-score Normalize (for Time Domain) - Ensure 1D
            gt_norm = (gt_filt - np.mean(gt_filt)) / (np.std(gt_filt) + 1e-9)
            gt_norm = gt_norm.flatten()
            gt_inst_hz = _instantaneous_freq_hz(gt_filt, fs_gt, min_hz=min_hz, max_hz=max_hz)
            
            estimates = data.get('estimates', [])
            
            for est in estimates:
                method_name = est.get('method', est.get('name', 'unknown'))
                payload = est.get('estimate', est)
                if not isinstance(payload, dict): continue

                sig_hat = payload.get('signal_hat')
                if sig_hat is None: continue
                sig_hat = np.asarray(sig_hat).flatten()
                if sig_hat.size < 2: continue
                data_file_rel = os.path.relpath(pkl_path, d_dir)
                est_track_hz = np.asarray(payload.get('track_hz', []), dtype=np.float64)
                
                # Preprocess Estimate
                est_filt = filter_RW(sig_hat, fps, lo=min_hz, hi=max_hz)
                est_norm = (est_filt - np.mean(est_filt)) / (np.std(est_filt) + 1e-9)
                est_norm = est_norm.flatten()

                # =========================================================
                # Block 1: Time Domain (Waveform Fidelity & Dynamics)
                # =========================================================
                # 1. Cross-Correlation Alignment
                est_aligned, gt_aligned_t, lag_sec = calculate_cross_corr_alignment(est_norm, gt_norm, fs_est=fps, fs_gt=fs_gt)
                
                if len(est_aligned) > 10: # Ensure valid length
                    # Compute Waveform Metrics
                    # CCC
                    ccc_val = metrics_lib.LinCorr(est_aligned, gt_aligned_t)
                    # MAE / RMSE (on Z-scored aligned signals)
                    errs_time = getErrors(est_aligned, gt_aligned_t, None, None, ['MAE', 'RMSE'])
                    mae_time = errs_time[0]
                    rmse_time = errs_time[1]
                    # SNR (Time Domain)
                    residual = gt_aligned_t - est_aligned
                    p_sig = np.sum(gt_aligned_t**2)
                    p_res = np.sum(residual**2)
                    snr_time = 10 * np.log10(p_sig / (p_res + 1e-9))
                    
                    # New Critical Metrics: Latency & DTW
                    # Latency in ms. lag_sec > 0 means Est is delayed (starts after GT).
                    latency_ms = lag_sec * 1000.0
                    
                    # DTW Distance (Non-linear robustness)
                    dtw_dist = metrics_lib.calculate_dtw_distance(est_aligned, gt_aligned_t)
                    
                    # Physiological Dynamics (I:E Ratio, PPI)
                    # Compute on Aligned signals to match phases? 
                    # Dynamics are independent of lag, but clean signals are better.
                    # Use est_norm/gt_norm (full length) or aligned? 
                    # Let's use aligned to ensure we look at same segment, 
                    # but find_peaks handles shift.
                    gt_ie, gt_ppi, _ = calculate_breathing_dynamics(gt_norm, fs_gt)
                    est_ie, est_ppi, _ = calculate_breathing_dynamics(est_norm, fps)
                    
                    ie_err = abs(est_ie - gt_ie) if (np.isfinite(est_ie) and np.isfinite(gt_ie)) else np.nan
                    ppi_mae = abs(est_ppi - gt_ppi) if (np.isfinite(est_ppi) and np.isfinite(gt_ppi)) else np.nan

                    rec_time = {
                        'video': fname, 'method': method_name,
                        'CCC': ccc_val, 'MAE': mae_time, 'RMSE': rmse_time,
                        'SNR_Time': snr_time, 
                        'Latency': latency_ms, 'DTW_Dist': dtw_dist,
                        'IE_Err': ie_err, 'PPI_MAE': ppi_mae,
                        '_sig_aligned_pair': (est_aligned, gt_aligned_t),
                        'gt_bpm_avg': np.nan,
                        'est_bpm_avg': np.nan,
                        'data_file': data_file_rel,
                    }
                    all_time_domain_records.append(rec_time)

                # =========================================================
                # Block 1b: Unified Waveform Metrics (T4 comparative)
                # =========================================================
                # All methods get waveform CCC/wMAE/DTW via identical protocol:
                #   bandpass → zscore → cross-corr alignment → metrics
                # output_type distinguishes the signal source:
                #   Base/KFstd: signal_hat
                #   PARH-OSSM:  z_full (primary), z_osc (supplement)
                _wf_candidates = {}

                # --- signal_hat waveform (all methods) ---
                if est_norm is not None and est_norm.size >= 10 and gt_signal is not None:
                    _wf_candidates[('signal_hat', 'smoothed')] = est_norm

                # --- z_full waveform (PARH-OSSM only) ---
                _zf_smooth = payload.get('z_full_smoothed') if payload.get('z_full_smoothed') is not None else payload.get('z_full')
                _zf_causal = payload.get('z_full_causal')
                if _zf_smooth is not None:
                    _zfs = np.asarray(_zf_smooth, dtype=np.float64).flatten()
                    if _zfs.size >= 10:
                        _zfs_filt = filter_RW(_zfs, fps, lo=min_hz, hi=max_hz)
                        _zfs_norm = (_zfs_filt - np.mean(_zfs_filt)) / (np.std(_zfs_filt) + 1e-9)
                        _wf_candidates[('z_full', 'smoothed')] = _zfs_norm.flatten()
                if _zf_causal is not None:
                    _zfc = np.asarray(_zf_causal, dtype=np.float64).flatten()
                    if _zfc.size >= 10:
                        _zfc_filt = filter_RW(_zfc, fps, lo=min_hz, hi=max_hz)
                        _zfc_norm = (_zfc_filt - np.mean(_zfc_filt)) / (np.std(_zfc_filt) + 1e-9)
                        _wf_candidates[('z_full', 'causal')] = _zfc_norm.flatten()

                # --- z_osc waveform (PARH-OSSM supplement) ---
                _zo_smooth = payload.get('z_osc_smoothed') if payload.get('z_osc_smoothed') is not None else payload.get('z_osc')
                if _zo_smooth is not None and payload.get('z_full') is not None:
                    _zos = np.asarray(_zo_smooth, dtype=np.float64).flatten()
                    if _zos.size >= 10:
                        _zos_filt = filter_RW(_zos, fps, lo=min_hz, hi=max_hz)
                        _zos_norm = (_zos_filt - np.mean(_zos_filt)) / (np.std(_zos_filt) + 1e-9)
                        _wf_candidates[('z_osc', 'smoothed')] = _zos_norm.flatten()

                for (_otype, _variant_label), _wf_sig in _wf_candidates.items():
                    if gt_signal is None:
                        continue
                    _wf_aligned, _gt_aligned_w, _lag_w = calculate_cross_corr_alignment(
                        _wf_sig, gt_norm, fs_est=fps, fs_gt=fs_gt
                    )
                    if len(_wf_aligned) > 10:
                        _w_ccc = waveform_ccc(_wf_aligned, _gt_aligned_w)
                        _w_mae = waveform_mae(_wf_aligned, _gt_aligned_w)
                        _w_dtw = waveform_dtw(_wf_aligned, _gt_aligned_w)
                        rec_wf = {
                            'video': fname, 'method': method_name,
                            'output_type': _otype,
                            'causal_or_smoothed': _variant_label,
                            'waveform_CCC': _w_ccc,
                            'waveform_MAE': _w_mae,
                            'waveform_DTW': _w_dtw,
                            'latency_ms': _lag_w * 1000.0,
                            'data_file': data_file_rel,
                        }
                        all_waveform_records.append(rec_wf)

                # =========================================================
                # Block 2: Frequency Domain (Rate Accuracy & Spectral Shape)
                # =========================================================
                # Use Sliding Window (30s window, 1s stride)
                # We need to re-window both GT and Est with the same parameters
                
                # GT RPM Sequence
                gt_win, t_gt = sig_windowing(gt_filt, fs_gt, win_size, stride=stride)
                gt_rpms = sig_to_RPM(gt_win, fs_gt, int(win_size/1.5), min_hz, max_hz).reshape(-1)
                
                # Est RPM Sequence
                est_win, t_est = sig_windowing(est_filt, fps, win_size, stride=stride)
                est_rpms = sig_to_RPM(est_win, fps, int(win_size/1.5), min_hz, max_hz).reshape(-1)
                
                # Align RPM sequences by time centers
                min_len_r = min(len(gt_rpms), len(est_rpms))
                if min_len_r > 5: # Need enough points for stats
                    gt_rpm_seq = gt_rpms[:min_len_r]
                    est_rpm_seq = est_rpms[:min_len_r]
                    
                    # Compute Rate Metrics
                    errs_freq = getErrors(est_rpm_seq, gt_rpm_seq, None, None, ['MAE', 'RMSE', 'MAPE', 'PearsonR'])
                    
                    # Spectral SNR (De Haan) - Averaged over windows
                    # Spectral Shape (KL Div, Entropy) - Computed on Average PSD of the whole trial for robustness
                    # (Computing KL per window is noisy).
                    
                    # 1. Spectral SNR Map
                    snr_vals = []
                    for w_sig in est_win:
                        s_val = calculate_spectral_snr(w_sig.flatten(), fps, min_hz, max_hz)
                        if np.isfinite(s_val): snr_vals.append(s_val)
                    avg_spec_snr = np.mean(snr_vals) if snr_vals else np.nan
                    
                    # 2. Global PSD Shape Comparison
                    # Re-compute full Welch for shape comparison
                    # Use standard 30s window Welch on full signal
                    _, p_gt = Welch_rpm(gt_filt, fs_gt, win_size, min_hz, max_hz)
                    _, p_est = Welch_rpm(est_filt, fps, win_size, min_hz, max_hz)
                    
                    # Handle multi-channel return from Welch_rpm if any
                    if p_gt.ndim > 1: p_gt = np.mean(p_gt, axis=0)
                    if p_est.ndim > 1: p_est = np.mean(p_est, axis=0)
                    
                    # Ensure same length? Welch_rpm outputs based on nfft. 
                    # If fs differs, length differs. Resample PSD or ensure fs matches?
                    # Assuming fps ~= fs_gt for this metric, or interpolated.
                    # If lengths differ, we cannot compute KL directly.
                    # Interpolate Est PSD to GT PSD bins if needed.
                    if len(p_gt) != len(p_est):
                        p_est = np.interp(np.linspace(0, 1, len(p_gt)), np.linspace(0, 1, len(p_est)), p_est)

                    kl_div, ent_err, _ = calculate_spectral_shape_metrics(p_est, p_gt)
                    
                    # Bland-Altman
                    bias, lower_loa, upper_loa = bland_altman_stats(est_rpm_seq, gt_rpm_seq)
                    loa_width = upper_loa - lower_loa
                    
                    rec_freq = {
                        'video': fname, 'method': method_name,
                        'MAE': errs_freq[0], 'RMSE': errs_freq[1], 'MAPE': errs_freq[2], 
                        'PearsonR': errs_freq[3],
                        'SNR_Spec': avg_spec_snr,
                        'Bias': bias, 'LoA_Width': loa_width,
                        'KL_Div': kl_div, 'Entropy_Err': ent_err,
                        'gt_bpm_avg': float(np.nanmean(gt_rpm_seq)) if gt_rpm_seq.size else np.nan,
                        'est_bpm_avg': float(np.nanmean(est_rpm_seq)) if est_rpm_seq.size else np.nan,
                        '_rpm_pair': (est_rpm_seq, gt_rpm_seq),
                        'data_file': data_file_rel
                    }
                    all_freq_domain_records.append(rec_freq)

                # =========================================================
                # Block 3: Filter Diagnostics (Failure / Calibration)
                # =========================================================
                dataset_token = fname.split('_', 1)[0] if '_' in fname else None
                trial_key = _infer_trial_key_from_fname(fname, dataset_token=dataset_token)
                log_obj = _load_frame_log(
                    d_dir, method_name, trial_key, frame_log_cache,
                    selected_logs=selected_logs,
                )

                # Align GT instantaneous frequency to estimator track length.
                gt_track_interp = None
                if est_track_hz.size > 0 and gt_inst_hz.size > 0:
                    t_est = np.arange(est_track_hz.size, dtype=np.float64) / max(float(fps), 1e-6)
                    t_gt = np.arange(gt_inst_hz.size, dtype=np.float64) / max(float(fs_gt), 1e-6)
                    gt_track_interp = np.interp(t_est, t_gt, gt_inst_hz)

                diag_rec = _compute_filter_diag_record(
                    log_obj=log_obj,
                    method_name=method_name,
                    fname=fname,
                    data_file_rel=data_file_rel,
                    fps=float(fps),
                    est_track_hz=est_track_hz if est_track_hz.size > 0 else None,
                    gt_inst_hz=gt_track_interp,
                    calibration_policy=calibration_policy,
                )
                all_filter_diag_records.append(diag_rec)

        # Save Logic
        metrics_dir = os.path.join(d_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        def _save_domain_v2(records, label, metric_keys):
            if not records: return
            df = pd.DataFrame(records)

            # Filter diagnostics are only applicable when frame-log derived metrics exist.
            # Keep raw CSV complete, but exclude non-applicable rows from summary table.
            summary_df = df
            if label == 'filter_diagnostics':
                diag_keys = ['Fail_Total', 'NIS_Mean', 'Lambda_Mean', 'Coverage95']
                has_any = np.zeros(len(df), dtype=bool)
                for k in diag_keys:
                    if k in df.columns:
                        vals = pd.to_numeric(df[k], errors='coerce').to_numpy(dtype=np.float64)
                        has_any |= np.isfinite(vals)
                summary_df = df.loc[has_any].copy()
                if summary_df.empty:
                    summary_df = df.copy()
            
            # Numeric conversion
            for m in metric_keys:
                df[m] = pd.to_numeric(df[m], errors='coerce')
                summary_df[m] = pd.to_numeric(summary_df[m], errors='coerce')
            
            # Sort
            methods_in_df = summary_df['method'].unique()
            sorted_methods = sorted(methods_in_df, key=_method_sort_key)
            
            # Group stats
            summary_stats = summary_df.groupby('method')[metric_keys].agg(['median', 'std'])
            summary_stats.columns = [f'{m}_{s}' for m, s in summary_stats.columns]
            summary_stats = summary_stats.reset_index()
            summary_stats['Method'] = summary_stats['method']
            
            summary_stats['sort_key'] = summary_stats['Method'].apply(_method_sort_key)
            summary_stats = summary_stats.sort_values('sort_key').drop(columns=['sort_key'])
            
            # Format Table
            table_headers = ['Method'] + [f"{m} (median±std)" for m in metric_keys]
            table_rows = []
            for _, row in summary_stats.iterrows():
                tr = [row['Method']]
                for m in metric_keys:
                    val = row[f'{m}_median']
                    std = row[f'{m}_std']
                    # Format nice scalars
                    tr.append(f"{_format_scalar(val)} (±{_format_scalar(std, decimals=2)})")
                table_rows.append(tr)
            
            table_str = _render_table(table_headers, table_rows)
            txt_name = f'metrics_{label}_summary.txt'
            
            if label == 'time_domain':
                title = "Time Domain (Waveform Fidelity)"
            elif label == 'freq_domain':
                title = "Freq Domain (Rate Accuracy)"
            elif label == 'filter_diagnostics':
                title = "Filter Diagnostics (Failure/Calibration)"
            elif label == 'waveform':
                title = "z_full Waveform (CCC/wMAE/DTW — PARH-OSSM dual output)"
            else:
                title = label
            
            with open(os.path.join(metrics_dir, txt_name), 'w', encoding='utf-8') as f:
                f.write(f"# Dual-Domain Eval | {title}\n{table_str}\n")
            
            print(f"\n   [{title}]")
            print(table_str)
            
            # CSVs
            df_clean = df.drop(columns=[c for c in df.columns if c.startswith('_')], errors='ignore')
            df_clean.to_csv(os.path.join(metrics_dir, f'metrics_{label}_raw.csv'), index=False)
            summary_stats.to_csv(os.path.join(metrics_dir, f'metrics_{label}_summary.csv'), index=False)
            
            # Save Pickles for plots
            # Reconstruct legacy structure for plotting compatibility
            method_metrics = {}
            for rec in records:
                m = rec['method']
                entry = {
                    'video': rec['video'],
                    'metrics': [rec[k] for k in metric_keys],
                    'source_label': label,
                    'data_file': rec.get('data_file')
                }
                if label == 'time_domain':
                    entry['pair'] = rec.get('_sig_aligned_pair')
                else:
                    entry['pair'] = rec.get('_rpm_pair')
                    
                method_metrics.setdefault(m, []).append(entry)
                
            with open(os.path.join(metrics_dir, f'metrics_{label}.pkl'), 'wb') as f:
                pickle.dump([metric_keys, method_metrics], f)

        _save_domain_v2(all_time_domain_records, 'time_domain', time_metrics_list)
        _save_domain_v2(all_freq_domain_records, 'freq_domain', freq_metrics_list)
        _save_domain_v2(all_filter_diag_records, 'filter_diagnostics', filter_diag_metrics_list)

        # ── Unified waveform metrics (T4 comparative) ──
        # All methods: signal_hat. PARH-OSSM additionally: z_full, z_osc.
        if all_waveform_records:
            wf_metrics_keys = ['waveform_CCC', 'waveform_MAE', 'waveform_DTW', 'latency_ms']
            _save_domain_v2(all_waveform_records, 'waveform', wf_metrics_keys)

        # Explicitly separate strict-test failures from practical miscalibration.
        try:
            fdf = pd.DataFrame(all_filter_diag_records)
            if not fdf.empty and 'method' in fdf.columns:
                def _safe_mean(series) -> float:
                    arr = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64, copy=False)
                    arr = arr[np.isfinite(arr)]
                    return float(np.mean(arr)) if arr.size else float("nan")

                def _safe_median(series) -> float:
                    arr = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64, copy=False)
                    arr = arr[np.isfinite(arr)]
                    return float(np.median(arr)) if arr.size else float("nan")

                out_rows = []
                for method, g in fdf.groupby('method', dropna=False):
                    s_strict = pd.to_numeric(g.get('NIS_Pass', np.nan), errors='coerce')
                    s_relaxed = pd.to_numeric(g.get('NIS_Pass_Relaxed', np.nan), errors='coerce')
                    s_over = pd.to_numeric(g.get('NIS_OverStrict', np.nan), errors='coerce')
                    s_true = pd.to_numeric(g.get('NIS_TrueFail', np.nan), errors='coerce')
                    s_mean = pd.to_numeric(g.get('NIS_Mean', np.nan), errors='coerce')
                    s_inband = pd.to_numeric(g.get('NIS_InBand', np.nan), errors='coerce')
                    has_calibration = bool(np.isfinite(s_mean).any() and np.isfinite(s_inband).any())
                    out_rows.append({
                        "method": str(method),
                        "n_trials": int(len(g)),
                        "calibration_applicable": bool(has_calibration),
                        "applicability_reason": "" if has_calibration else "no_finite_nis_metrics",
                        "strict_pass_rate": _safe_mean(s_strict) if has_calibration else np.nan,
                        "relaxed_pass_rate": _safe_mean(s_relaxed) if has_calibration else np.nan,
                        "overstrict_rate": _safe_mean(s_over) if has_calibration else np.nan,
                        "true_fail_rate": _safe_mean(s_true) if has_calibration else np.nan,
                        "nis_mean_median": _safe_median(s_mean) if has_calibration else np.nan,
                        "nis_inband_median": _safe_median(s_inband) if has_calibration else np.nan,
                    })
                split_all_df = pd.DataFrame(out_rows)
                if not split_all_df.empty:
                    split_all_df["sort_key"] = split_all_df["method"].apply(_method_sort_key)
                    split_all_df = (
                        split_all_df
                        .sort_values("sort_key")
                        .drop(columns=["sort_key"])
                        .reset_index(drop=True)
                    )
                split_df = split_all_df.loc[
                    split_all_df["calibration_applicable"].astype(bool)
                ].copy()
                split_csv = os.path.join(metrics_dir, "metrics_filter_calibration_split.csv")
                split_all_csv = os.path.join(metrics_dir, "metrics_filter_calibration_split_all.csv")
                split_json = os.path.join(metrics_dir, "metrics_filter_calibration_split.json")
                split_df.to_csv(split_csv, index=False)
                split_all_df.to_csv(split_all_csv, index=False)
                # Human-readable summary table for quick paper/debug review.
                txt_path = os.path.join(metrics_dir, "metrics_filter_calibration_split.txt")
                if not split_df.empty:
                    table_headers = [
                        "Method", "strict_pass", "relaxed_pass", "overstrict", "true_fail", "nis_mean_med", "nis_inband_med"
                    ]
                    table_rows = []
                    for _, row in split_df.iterrows():
                        table_rows.append([
                            str(row.get("method", "")),
                            _format_scalar(row.get("strict_pass_rate", np.nan)),
                            _format_scalar(row.get("relaxed_pass_rate", np.nan)),
                            _format_scalar(row.get("overstrict_rate", np.nan)),
                            _format_scalar(row.get("true_fail_rate", np.nan)),
                            _format_scalar(row.get("nis_mean_median", np.nan)),
                            _format_scalar(row.get("nis_inband_median", np.nan)),
                        ])
                    with open(txt_path, "w", encoding="utf-8") as fp:
                        fp.write("# Filter Calibration Split (applicable methods only)\n")
                        fp.write(_render_table(table_headers, table_rows) + "\n")
                else:
                    with open(txt_path, "w", encoding="utf-8") as fp:
                        fp.write("# Filter Calibration Split (applicable methods only)\n")
                        fp.write("No applicable methods with finite NIS metrics.\n")
                with open(split_json, "w", encoding="utf-8") as fp:
                    json.dump({
                        "schema_version": "filter_calibration_split.v1",
                        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
                        "policy": calibration_policy,
                        "n_methods_all": int(len(split_all_df)),
                        "n_methods_applicable": int(len(split_df)),
                        "n_methods_non_applicable": int(len(split_all_df) - len(split_df)),
                        "rows": split_df.to_dict(orient="records"),
                        "rows_all": split_all_df.to_dict(orient="records"),
                    }, fp, ensure_ascii=False, indent=2)
                if not split_df.empty:
                    print("\n   [Calibration Split (Strict vs Relaxed)]")
                    print(_render_table(
                        ["Method", "strict_pass", "relaxed_pass", "overstrict", "true_fail"],
                        [
                            [
                                str(r["method"]),
                                _format_scalar(r["strict_pass_rate"]),
                                _format_scalar(r["relaxed_pass_rate"]),
                                _format_scalar(r["overstrict_rate"]),
                                _format_scalar(r["true_fail_rate"]),
                            ]
                            for _, r in split_df.iterrows()
                        ],
                    ))
        except Exception as exc:
            print(f"> Warning: failed to write filter calibration split artifacts: {exc}")

        _write_method_quality_artifacts(
            d_dir,
            eval_settings,
            expected_trials=expected_trials_for_logs,
            frame_log_strict=bool(frame_log_strict),
        )


def _method_sort_key(method_name: str):
    def _normalize(name: str) -> str:
        return str(name).lower().replace(' ', '_')

    name = _normalize(method_name)
    base = name.split('__', 1)[0] if '__' in name else name

    # family order: OF -> DoF -> profile1d linear -> quadratic -> cubic
    if base in ('of_model', 'of', 'of_farneback'):
        family = 10
    elif base == 'dof':
        family = 20
    elif base.startswith('profile1d_linear'):
        family = 30
    elif base.startswith('profile1d_quadratic'):
        family = 40
    elif base.startswith('profile1d_cubic'):
        family = 50
    else:
        family = 99

    # within family: base -> kfstd -> ekf/robust_ossm -> ukf/ukffreq -> others
    if '__' not in name:
        sub = 0
    else:
        head = name.split('__', 1)[1]
        if 'kfstd' in head:
            sub = 10
        elif 'robust_ossm_ekf' in head or head.endswith('_ekf') or head == 'robust_ossm':
            sub = 20
        elif 'robust_ossm_ukf' in head or 'ukffreq' in head or head.endswith('_ukf'):
            sub = 30
        elif 'agakf' in head:
            sub = 40
        else:
            sub = 90

    return (family, sub, name)
