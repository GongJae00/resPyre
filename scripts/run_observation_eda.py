#!/usr/bin/env python3
"""Run observation-class and preprocessing EDA from saved trial PKLs.

This script is designed for the next-generation PARH-OSSM redesign. It focuses
on the observation channel: raw class proxies, preprocessing stages, alignment
to the ground-truth waveform, and spectrum-level distortion characteristics.
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.models.core.base import OscillatorParams
from components.models.heads.parh_ossm import oscillator_PARH_OSSM
from core.evaluation.metrics import concordance_correlation_coefficient
from core.utils.config import load_config


FAMILY_INFO = {
    "OF": {
        "cache": "obs_of.npy",
        "base_methods": {"of_farneback", "of"},
    },
    "OF_bridge": {
        "cache": "obs_of_bridge.npy",
        "base_methods": {"of_disp_bridge", "of_displacement_bridge", "of_bridge"},
    },
    "DoF": {
        "cache": "obs_dof.npy",
        "base_methods": {"dof"},
    },
    "DoF_bridge": {
        "cache": "obs_dof_bridge_v2.npy",
        "base_methods": {"dof_disp_bridge", "dof_bridge"},
    },
    "P1D_lin": {
        "cache": "obs_p1d_linear.npy",
        "base_methods": {"profile1d_linear", "profile1d linear"},
    },
    "P1D_quad": {
        "cache": "obs_p1d_quad.npy",
        "base_methods": {"profile1d_quadratic", "profile1d quadratic"},
    },
    "P1D_cub": {
        "cache": "obs_p1d_cubic.npy",
        "base_methods": {"profile1d_cubic", "profile1d cubic"},
    },
    "P1D_cons": {
        "cache": "obs_p1d_consensus_v1.npy",
        "base_methods": {"profile1d_consensus", "profile1d_cons", "p1d_cons"},
    },
}

STAGE_ORDER = [
    "raw",
    "detrend_only",
    "bandpass_only",
    "sign_align_only",
    "robust_zscore_only",
    "current_preprocess",
    "helper_preprocess",
]


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility wrapper for NumPy versions without np.trapezoid."""
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x))


def canonical_family(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if "__" in key:
        key = key.split("__", 1)[0]
    family_map = {
        "of_farneback": "OF",
        "of_disp_bridge": "OF_bridge",
        "of": "OF",
        "dof": "DoF",
        "dof_disp_bridge": "DoF_bridge",
        "profile1d_linear": "P1D_lin",
        "profile1d_linear_bridge": "P1D_lin_bridge",
        "profile1d linear": "P1D_lin",
        "profile1d_quadratic": "P1D_quad",
        "profile1d_quadratic_bridge": "P1D_quad_bridge",
        "profile1d quadratic": "P1D_quad",
        "profile1d_cubic": "P1D_cub",
        "profile1d_cubic_bridge": "P1D_cub_bridge",
        "profile1d cubic": "P1D_cub",
        "profile1d_consensus": "P1D_cons",
    }
    return family_map.get(key, key)


def variant_from_method(name: str) -> str:
    key = str(name).lower()
    if "__parh_ossm" in key:
        return "PARH"
    if "__kfstd" in key:
        return "KFstd"
    return "Base"


def family_base_method(name: str) -> bool:
    return variant_from_method(name) == "Base"


def safe_float(value, default=np.nan) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    return v if np.isfinite(v) else float(default)


def robust_stats(signal: np.ndarray) -> Dict[str, float]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "mad": np.nan,
            "sigma_hat": np.nan,
        }
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma_hat = 1.4826 * mad
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": med,
        "mad": mad,
        "sigma_hat": float(sigma_hat),
    }


def zscore(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd


def overlap_by_lag(x: np.ndarray, y: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    if lag > 0:
        return x[lag:], y[: n - lag]
    if lag < 0:
        lag = abs(lag)
        return x[: n - lag], y[lag:]
    return x, y


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 3:
        return np.nan
    x = x[finite]
    y = y[finite]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < 1e-8 or sy < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def best_signed_lag_metrics(signal: np.ndarray, ref: np.ndarray, fs: float, max_lag_sec: float) -> Dict[str, float]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    y = np.asarray(ref, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    max_lag = int(max(0, round(max_lag_sec * max(fs, 1.0))))
    best = {
        "corr": np.nan,
        "ccc_z": np.nan,
        "lag_samples": 0.0,
        "lag_sec": 0.0,
        "sign": np.nan,
    }
    best_corr = -np.inf
    for sign in (1.0, -1.0):
        xs = sign * x
        for lag in range(-max_lag, max_lag + 1):
            xa, ya = overlap_by_lag(xs, y, lag)
            if xa.size < 8 or ya.size < 8:
                continue
            corr = safe_corr(zscore(xa), zscore(ya))
            if not np.isfinite(corr):
                continue
            if corr > best_corr:
                best_corr = corr
                best["corr"] = float(corr)
                best["ccc_z"] = float(concordance_correlation_coefficient(zscore(xa), zscore(ya)))
                best["lag_samples"] = float(lag)
                best["lag_sec"] = float(lag / max(fs, 1e-6))
                best["sign"] = float(sign)
    return best


def spectral_summary(signal: np.ndarray, fs: float, f_min: float, f_max: float) -> Dict[str, float]:
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < 8 or fs <= 0:
        return {
            "peak_hz": np.nan,
            "peak_power": np.nan,
            "peak_sharpness": np.nan,
            "harmonic_ratio": np.nan,
            "lowfreq_energy_ratio": np.nan,
            "band_energy_ratio": np.nan,
            "highfreq_energy_ratio": np.nan,
        }
    nperseg = min(x.size, max(int(round(fs * 15.0)), 64))
    nperseg = max(nperseg, 8)
    freqs, psd = sps.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    psd = np.asarray(psd, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    total = trapz(psd, freqs) if freqs.size > 1 else float(np.sum(psd))
    if not np.isfinite(total) or total <= 1e-12:
        total = np.nan

    band_mask = (freqs >= f_min) & (freqs <= f_max)
    low_mask = freqs < f_min
    high_mask = freqs > f_max
    band_psd = psd[band_mask]
    band_freq = freqs[band_mask]

    if band_psd.size == 0:
        peak_hz = np.nan
        peak_power = np.nan
        sharpness = np.nan
        harmonic_ratio = np.nan
    else:
        peak_idx = int(np.argmax(band_psd))
        peak_hz = float(band_freq[peak_idx])
        peak_power = float(band_psd[peak_idx])
        med_power = float(np.median(band_psd)) if band_psd.size else np.nan
        sharpness = peak_power / max(med_power, 1e-12) if np.isfinite(med_power) else np.nan
        if band_freq.size > 1 and np.isfinite(peak_hz):
            df = float(np.median(np.diff(band_freq)))
            tol = max(2.0 * df, 0.02)
            fundamental = float(np.sum(band_psd[np.abs(band_freq - peak_hz) <= tol]))
            harmonic = float(np.sum(band_psd[np.abs(band_freq - 2.0 * peak_hz) <= tol]))
            harmonic_ratio = harmonic / max(fundamental, 1e-12)
        else:
            harmonic_ratio = np.nan

    def band_energy(mask: np.ndarray) -> float:
        if np.count_nonzero(mask) < 1:
            return np.nan
        if freqs[mask].size > 1:
            val = trapz(psd[mask], freqs[mask])
        else:
            val = float(np.sum(psd[mask]))
        if not np.isfinite(total) or total <= 1e-12:
            return np.nan
        return val / total

    return {
        "peak_hz": peak_hz,
        "peak_power": peak_power,
        "peak_sharpness": float(sharpness) if np.isfinite(sharpness) else np.nan,
        "harmonic_ratio": float(harmonic_ratio) if np.isfinite(harmonic_ratio) else np.nan,
        "lowfreq_energy_ratio": band_energy(low_mask),
        "band_energy_ratio": band_energy(band_mask),
        "highfreq_energy_ratio": band_energy(high_mask),
    }


def resample_gt(gt: np.ndarray, fs_gt: float, n_target: int, fs_obs: float) -> np.ndarray:
    y = np.asarray(gt, dtype=np.float64).reshape(-1)
    if n_target <= 0:
        return np.array([], dtype=np.float64)
    if y.size == 0 or fs_gt <= 0 or fs_obs <= 0:
        return np.zeros(n_target, dtype=np.float64)
    t_src = np.arange(y.size, dtype=np.float64) / fs_gt
    t_dst = np.arange(n_target, dtype=np.float64) / fs_obs
    return np.interp(t_dst, t_src, y, left=float(y[0]), right=float(y[-1]))


def load_head_from_config(config_path: Optional[Path]) -> oscillator_PARH_OSSM:
    cfg = load_config(str(config_path)) if config_path else load_config(None)
    osc_cfg = copy.deepcopy(cfg.get("oscillator", {}))
    valid_fields = set(OscillatorParams().__dict__.keys())
    osc_kwargs = {k: v for k, v in osc_cfg.items() if k in valid_fields}
    head = oscillator_PARH_OSSM(params=OscillatorParams(**osc_kwargs))
    head.preproc_cfg = copy.deepcopy(cfg.get("preproc", {})) if isinstance(cfg.get("preproc"), dict) else {}
    return head


def load_roi_quality(trial_dir: Path) -> Dict[str, float]:
    path = trial_dir / "obs_roi_stats_v1.npz"
    if not path.exists():
        return {
            "roi_snr_db_mean": np.nan,
            "valid_ratio_mean": np.nan,
            "center_disp_mean": np.nan,
            "center_disp_std": np.nan,
            "roi_mean_mean": np.nan,
            "roi_std_mean": np.nan,
        }
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return {
            "roi_snr_db_mean": np.nan,
            "valid_ratio_mean": np.nan,
            "center_disp_mean": np.nan,
            "center_disp_std": np.nan,
            "roi_mean_mean": np.nan,
            "roi_std_mean": np.nan,
        }
    out = {}
    for key, col in (
        ("roi_snr_db_mean", "col__roi_snr_db"),
        ("valid_ratio_mean", "col__valid_ratio"),
        ("center_disp_mean", "col__center_disp"),
        ("roi_mean_mean", "col__roi_mean"),
        ("roi_std_mean", "col__roi_std"),
    ):
        arr = np.asarray(data[col], dtype=np.float64).reshape(-1) if col in data else np.array([], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        out[key] = float(np.mean(arr)) if arr.size else np.nan
    center = np.asarray(data["col__center_disp"], dtype=np.float64).reshape(-1) if "col__center_disp" in data else np.array([], dtype=np.float64)
    center = center[np.isfinite(center)]
    out["center_disp_std"] = float(np.std(center)) if center.size else np.nan
    return out


def method_index(estimates: Iterable[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for item in estimates:
        method = str(item.get("method", ""))
        est = item.get("estimate", {})
        out[method] = est if isinstance(est, dict) else {}
    return out


def find_base_signalhat(estimates: Dict[str, Dict], family: str) -> np.ndarray:
    candidates = FAMILY_INFO[family]["base_methods"]
    for method_name, payload in estimates.items():
        base = str(method_name).lower()
        if "__" in base:
            continue
        if base in candidates:
            return np.asarray(payload.get("signal_hat", []), dtype=np.float64)
    return np.array([], dtype=np.float64)


def raw_signal_path(trial_dir: Path, family: str) -> Path:
    return trial_dir / FAMILY_INFO[family]["cache"]


def sign_align_only(x: np.ndarray, fs: float, head: oscillator_PARH_OSSM) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    if y.size == 0:
        return y
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y)
    sign_cfg = head.preproc_cfg.get("sign_align", {}) if isinstance(head.preproc_cfg, dict) else {}
    if not sign_cfg.get("enabled") or fs <= 0 or y.size < 2:
        return y
    sign_seconds = safe_float(sign_cfg.get("seconds", 12.0), default=12.0)
    seg_len = int(min(y.size, max(1, round(sign_seconds * fs))))
    if seg_len <= 1:
        return y
    coarse = head._coarse_freq(y[:seg_len], fs)
    if np.isfinite(coarse) and coarse > 0.0:
        t = np.arange(seg_len, dtype=np.float64) / fs
        ref = np.cos(2.0 * np.pi * coarse * t)
        if float(np.dot(y[:seg_len], ref)) < 0.0:
            y = -y
    return y


def robust_zscore_only(x: np.ndarray, head: oscillator_PARH_OSSM) -> Tuple[np.ndarray, Dict[str, float]]:
    y = np.asarray(x, dtype=np.float64).copy()
    if y.size == 0:
        return y, {"clipped_frac": np.nan, "sigma_hat": np.nan}
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y)
    robust_cfg = head.preproc_cfg.get("robust_zscore", {}) if isinstance(head.preproc_cfg, dict) else {}
    enabled = bool(robust_cfg.get("enabled", True))
    if not enabled:
        return y, {"clipped_frac": 0.0, "sigma_hat": np.nan}
    eps = safe_float(robust_cfg.get("eps", 1e-6), default=1e-6)
    clip = robust_cfg.get("clip", 5.0)
    clip = None if clip is None else safe_float(clip, default=np.nan)
    stats = robust_stats(y)
    denom = max(stats["sigma_hat"], eps) if np.isfinite(stats["sigma_hat"]) else eps
    z = (y - stats["median"]) / max(denom, 1e-8)
    clipped_frac = 0.0
    if clip is not None and np.isfinite(clip) and clip > 0.0:
        clipped_frac = float(np.mean(np.abs(z) >= clip)) if z.size else 0.0
        z = np.clip(z, -clip, clip)
    return z, {"clipped_frac": clipped_frac, "sigma_hat": stats["sigma_hat"]}


def bandpass_only(x: np.ndarray, fs: float, f_min: float, f_max: float) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).copy()
    if y.size == 0:
        return y
    if np.any(np.isnan(y)):
        y = np.nan_to_num(y)
    if fs <= 0:
        return y
    nyq = 0.5 * fs
    low = max(f_min, 0.01)
    high = min(f_max, nyq - 1e-3)
    if high <= low:
        return y
    b, a = sps.butter(2, [low / nyq, high / nyq], btype="bandpass")
    return sps.filtfilt(b, a, y, method="gust")


def preprocess_stage(raw: np.ndarray, fs: float, head: oscillator_PARH_OSSM, stage: str) -> Tuple[np.ndarray, Dict[str, float]]:
    if stage == "raw":
        return np.asarray(raw, dtype=np.float64).copy(), {"clipped_frac": 0.0}
    if stage == "detrend_only":
        y = np.asarray(raw, dtype=np.float64).copy()
        if np.any(np.isnan(y)):
            y = np.nan_to_num(y)
        return sps.detrend(y, type="linear"), {"clipped_frac": 0.0}
    if stage == "bandpass_only":
        return bandpass_only(raw, fs, head.params.f_min, head.params.f_max), {"clipped_frac": 0.0}
    if stage == "sign_align_only":
        return sign_align_only(raw, fs, head), {"clipped_frac": 0.0}
    if stage == "robust_zscore_only":
        return robust_zscore_only(raw, head)
    if stage == "current_preprocess":
        y = head._preprocess(raw, fs)
        meta = head._last_preproc_meta.get("robust_z", {}) if isinstance(head._last_preproc_meta, dict) else {}
        return y, {
            "clipped_frac": safe_float(meta.get("clipped_frac"), default=np.nan),
            "sigma_hat": safe_float(meta.get("sigma_hat"), default=np.nan),
        }
    if stage == "helper_preprocess":
        return head._helper_preprocess(raw, fs), {"clipped_frac": 0.0}
    raise ValueError(f"Unknown stage '{stage}'")


def scale_ratio(sig: np.ndarray, ref: np.ndarray) -> float:
    sx = robust_stats(sig)["sigma_hat"]
    sy = robust_stats(ref)["sigma_hat"]
    if not np.isfinite(sx) or not np.isfinite(sy) or sy <= 1e-8:
        return np.nan
    return float(sx / sy)


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n == 0:
        return np.nan
    d = x[:n] - y[:n]
    finite = np.isfinite(d)
    if not np.any(finite):
        return np.nan
    return float(np.sqrt(np.mean(np.square(d[finite]))))


def build_trial_rows(
    pkl_path: Path,
    head: oscillator_PARH_OSSM,
    stages: List[str],
    max_lag_sec: float,
    dataset_override: Optional[str] = None,
) -> List[Dict[str, float]]:
    with pkl_path.open("rb") as fp:
        obj = pickle.load(fp)

    gt = np.asarray(obj.get("gt", []), dtype=np.float64)
    fs_gt = safe_float(obj.get("fs_gt", np.nan))
    fs_obs = safe_float(obj.get("fps", np.nan))
    video_path = Path(str(obj.get("video_path", "")))
    dataset_name = dataset_override or video_path.parts[-4].upper() if len(video_path.parts) >= 4 else "UNKNOWN"
    trial_dir = video_path.parent
    quality = load_roi_quality(trial_dir)
    est_map = method_index(obj.get("estimates", []))

    rows: List[Dict[str, float]] = []
    for family in FAMILY_INFO:
        cache_path = raw_signal_path(trial_dir, family)
        if not cache_path.exists():
            continue
        raw = np.asarray(np.load(cache_path), dtype=np.float64).reshape(-1)
        if raw.size == 0:
            continue
        gt_rs = resample_gt(gt, fs_gt, raw.size, fs_obs)
        gt_diff = np.gradient(gt_rs) if gt_rs.size else np.array([], dtype=np.float64)
        gt_spec = spectral_summary(gt_rs, fs_obs, head.params.f_min, head.params.f_max)
        base_signalhat = find_base_signalhat(est_map, family)

        for stage in stages:
            stage_sig, stage_meta = preprocess_stage(raw, fs_obs, head, stage)
            wave = best_signed_lag_metrics(stage_sig, gt_rs, fs_obs, max_lag_sec=max_lag_sec)
            deriv = best_signed_lag_metrics(np.gradient(stage_sig), gt_diff, fs_obs, max_lag_sec=max_lag_sec)
            spec = spectral_summary(stage_sig, fs_obs, head.params.f_min, head.params.f_max)
            stats = robust_stats(stage_sig)
            row = {
                "dataset": dataset_name,
                "video": pkl_path.stem,
                "subject": video_path.parent.parent.name if video_path.parent.parent else "",
                "trial": video_path.parent.name if video_path.parent else "",
                "family": family,
                "stage": stage,
                "video_path": str(video_path),
                "cache_path": str(cache_path),
                "n_samples": int(stage_sig.size),
                "duration_sec": float(stage_sig.size / max(fs_obs, 1e-6)),
                "fs_obs": fs_obs,
                "fs_gt": fs_gt,
                "stage_mean": stats["mean"],
                "stage_std": stats["std"],
                "stage_median": stats["median"],
                "stage_mad": stats["mad"],
                "stage_sigma_hat": stats["sigma_hat"],
                "stage_clipped_frac": safe_float(stage_meta.get("clipped_frac"), default=np.nan),
                "scale_ratio_to_gt": scale_ratio(stage_sig, gt_rs),
                "corr_wave_best": wave["corr"],
                "ccc_wave_best_z": wave["ccc_z"],
                "lag_wave_sec": wave["lag_sec"],
                "sign_wave_best": wave["sign"],
                "corr_deriv_best": deriv["corr"],
                "ccc_deriv_best_z": deriv["ccc_z"],
                "lag_deriv_sec": deriv["lag_sec"],
                "sign_deriv_best": deriv["sign"],
                "peak_hz": spec["peak_hz"],
                "gt_peak_hz": gt_spec["peak_hz"],
                "peak_error_hz": (spec["peak_hz"] - gt_spec["peak_hz"])
                if np.isfinite(spec["peak_hz"]) and np.isfinite(gt_spec["peak_hz"]) else np.nan,
                "peak_power": spec["peak_power"],
                "peak_sharpness": spec["peak_sharpness"],
                "harmonic_ratio": spec["harmonic_ratio"],
                "lowfreq_energy_ratio": spec["lowfreq_energy_ratio"],
                "band_energy_ratio": spec["band_energy_ratio"],
                "highfreq_energy_ratio": spec["highfreq_energy_ratio"],
                "vs_saved_base_rmse": rmse(stage_sig, base_signalhat),
                "vs_saved_base_corr": safe_corr(zscore(stage_sig), zscore(base_signalhat[: stage_sig.size])),
                **quality,
            }
            rows.append(row)
    return rows


def aggregate_family_stage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    metric_cols = [
        c for c in df.columns
        if c not in {
            "dataset", "video", "subject", "trial", "family", "stage",
            "video_path", "cache_path",
        }
    ]
    out = []
    for (dataset, family, stage), g in df.groupby(["dataset", "family", "stage"], dropna=False):
        row = {"dataset": dataset, "family": family, "stage": stage, "trial_count": int(g["video"].nunique())}
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(vals.mean()) if not vals.empty else np.nan
            row[f"{col}_median"] = float(vals.median()) if not vals.empty else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values(["dataset", "family", "stage"]).reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run observation/preprocessing EDA over saved trial PKLs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Results data directory containing trial PKLs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json",
        help="Config used to reproduce current preprocess settings",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset label override",
    )
    parser.add_argument(
        "--trial-out",
        type=Path,
        default=ROOT / "analysis" / "observation_eda_trials.csv",
        help="Output CSV for per-trial stage rows",
    )
    parser.add_argument(
        "--family-out",
        type=Path,
        default=ROOT / "analysis" / "observation_eda_family_stage_summary.csv",
        help="Output CSV for per-family stage summaries",
    )
    parser.add_argument(
        "--stages",
        nargs="*",
        default=STAGE_ORDER,
        help="Preprocessing stages to compute",
    )
    parser.add_argument(
        "--max-lag-sec",
        type=float,
        default=3.0,
        help="Maximum lag search for alignment diagnostics",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of trial PKLs to process for smoke testing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    head = load_head_from_config(args.config)
    pkl_files = sorted(args.data_dir.glob("*.pkl"))
    if args.limit and args.limit > 0:
        pkl_files = pkl_files[: int(args.limit)]

    rows: List[Dict[str, float]] = []
    for pkl_path in pkl_files:
        rows.extend(
            build_trial_rows(
                pkl_path=pkl_path,
                head=head,
                stages=list(args.stages),
                max_lag_sec=float(args.max_lag_sec),
                dataset_override=args.dataset_name,
            )
        )

    trial_df = pd.DataFrame(rows)
    family_df = aggregate_family_stage(trial_df)

    args.trial_out.parent.mkdir(parents=True, exist_ok=True)
    args.family_out.parent.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(args.trial_out, index=False)
    family_df.to_csv(args.family_out, index=False)

    print(f"Saved trial EDA: {args.trial_out}")
    print(f"Saved family-stage summary: {args.family_out}")
    print(f"Trial rows: {len(trial_df)}")
    if not family_df.empty:
        print(family_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
