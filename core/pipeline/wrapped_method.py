import copy
import os
from typing import Any, Dict, Optional, Tuple, List, Mapping

import numpy as np
from scipy import signal as sps
from .common import _deep_merge_dict, derive_trial_identifiers

from components.observations.methods import (
    OF_Model,
    OFDisplacementBridge_Model,
    DoF_Model,
    profile1D_Model,
    OFP1DQuadraticPair_Model,
    AssistOFP1DQuadratic_Model,
    AssistOFBridgeOF_Model,
    AssistOFBridgeP1DQuadratic_Model,
    FusionOFP1DQuadratic_Model,
)
from components.models import OscillatorParams, build_head

ROI_STATS_CACHE_SCHEMA = "roi_stats_cache.v1"
ROI_STATS_CACHE_FILE = "obs_roi_stats_v1.npz"
ROI_STATS_DEFAULTS: Dict[str, float] = {
    "roi_mean": 0.0,
    "roi_std": 0.0,
    "roi_snr_db": 0.0,
    "roi_cx": 0.5,
    "roi_cy": 0.5,
    "valid_ratio": 0.0,
    "center_disp": 0.0,
    "global_mean": 1.0,
}

def _normalize_base(name: str) -> str:
    key = name.lower()
    if key in ("of_model", "of", "of_farneback"):
        return "of_farneback"
    if key in ("of_disp_bridge", "of_displacement_bridge", "of_bridge"):
        return "of_disp_bridge"
    if key == "dof":
        return "dof"
    if key in ("profile1d_linear", "profile1d linear", "profile1d-linear"):
        return "profile1d_linear"
    if key in ("profile1d_quadratic", "profile1d quadratic", "profile1d-quadratic"):
        return "profile1d_quadratic"
    if key in ("profile1d_cubic", "profile1d cubic", "profile1d-cubic"):
        return "profile1d_cubic"
    if key in (
        "pair_of_p1d_quadratic",
        "of_p1d_quadratic_pair",
        "stack_of_p1d_quadratic",
    ):
        return "pair_of_p1d_quadratic"
    if key in (
        "fusion_of_p1d_quadratic",
        "of_p1d_quadratic_fusion",
        "fused_of_p1d_quadratic",
    ):
        return "fusion_of_p1d_quadratic"
    if key in (
        "assist_of_p1d_quadratic",
        "of_p1d_quadratic_assist",
        "assistant_of_p1d_quadratic",
    ):
        return "assist_of_p1d_quadratic"
    if key in (
        "assist_ofbridge_of",
        "of_bridge_assist_of",
        "assistant_ofbridge_of",
    ):
        return "assist_ofbridge_of"
    if key in (
        "assist_ofbridge_p1d_quadratic",
        "of_bridge_assist_p1d_quadratic",
        "assistant_ofbridge_p1d_quadratic",
    ):
        return "assist_ofbridge_p1d_quadratic"
    raise ValueError(f"Unknown base method '{name}' for oscillator wrapper")


def _normalize_head(name: str) -> str:
    key = name.lower().replace("-", "")
    if key in ("kfstd", "kf_std"):
        return "kfstd"
    if key in ("ukffreq", "ukf_freq"):
        return "ukffreq"
    if key in ("agakf", "ag_akf"):
        return "agakf"
    if key in ("robust_ossm", "robust_bayesian", "robustossm"):
        return "robust_ossm"
    if key in ("robust_ossm_ekf", "robustossmekf"):
        return "robust_ossm_ekf"
    if key in ("robust_ossm_ukf", "robustossmukf"):
        return "robust_ossm_ukf"
    if key in ("simple_bandpass", "simplebandpass", "bandpass"):
        return "simple_bandpass"
    if key in ("narossm", "narossm"):
        return "narossm"
    if key in ("parh_ossm", "parhossm", "parh-ossm"):
        return "parh_ossm"
    raise ValueError(f"Unknown oscillator head '{name}'")


def _normalize_gating_scope(value: Optional[str]) -> str:
    scope = str(value or "evaluation_only").strip().lower()
    allowed = {"evaluation_only", "filter_time"}
    if scope not in allowed:
        raise ValueError(
            f"Invalid gating_scope '{value}'. Allowed values: {sorted(allowed)}"
        )
    return scope


def _build_base(base_key: str):
    if base_key == "of_farneback":
        base = OF_Model()
        base.name = "of_farneback"
        return base
    if base_key == "of_disp_bridge":
        base = OFDisplacementBridge_Model()
        base.name = "of_disp_bridge"
        return base
    if base_key == "dof":
        base = DoF_Model()
        base.name = "dof"
        return base
    if base_key == "profile1d_linear":
        base = profile1D_Model("linear")
        base.name = "profile1d_linear"
        return base
    if base_key == "profile1d_quadratic":
        base = profile1D_Model("quadratic")
        base.name = "profile1d_quadratic"
        return base
    if base_key == "profile1d_cubic":
        base = profile1D_Model("cubic")
        base.name = "profile1d_cubic"
        return base
    if base_key == "pair_of_p1d_quadratic":
        base = OFP1DQuadraticPair_Model()
        base.name = "pair_of_p1d_quadratic"
        return base
    if base_key == "fusion_of_p1d_quadratic":
        base = FusionOFP1DQuadratic_Model()
        base.name = "fusion_of_p1d_quadratic"
        return base
    if base_key == "assist_of_p1d_quadratic":
        base = AssistOFP1DQuadratic_Model()
        base.name = "assist_of_p1d_quadratic"
        return base
    if base_key == "assist_ofbridge_of":
        base = AssistOFBridgeOF_Model()
        base.name = "assist_ofbridge_of"
        return base
    if base_key == "assist_ofbridge_p1d_quadratic":
        base = AssistOFBridgeP1DQuadratic_Model()
        base.name = "assist_ofbridge_p1d_quadratic"
        return base
    raise ValueError(f"Unsupported base key '{base_key}'")


def _roi_stats_cache_path(video_path: Optional[str]) -> Optional[str]:
    if not video_path:
        return None
    trial_dir = os.path.dirname(str(video_path))
    if not trial_dir:
        return None
    return os.path.join(trial_dir, ROI_STATS_CACHE_FILE)


def _roi_summary_from_stats(stats_t: List[Dict[str, float]]) -> Tuple[float, float, float]:
    if not stats_t:
        return float("nan"), float("nan"), float("nan")
    means: List[float] = []
    stds: List[float] = []
    for frame in stats_t:
        m = frame.get("roi_mean", np.nan)
        s = frame.get("roi_std", np.nan)
        if np.isfinite(m):
            means.append(float(m))
        if np.isfinite(s):
            stds.append(float(s))
    if not means:
        return float("nan"), float("nan"), float("nan")
    mean_intensity = float(np.nanmean(np.asarray(means, dtype=np.float64)))
    std_intensity = float(np.nanmean(np.asarray(stds, dtype=np.float64))) if stds else float("nan")
    if np.isfinite(std_intensity) and std_intensity > 0:
        snr_db = float(20.0 * np.log10(abs(mean_intensity) / max(std_intensity, 1e-6)))
    else:
        snr_db = float("nan")
    return mean_intensity, std_intensity, snr_db


def _coerce_roi_stats_frame(raw: Mapping[str, Any]) -> Dict[str, float]:
    frame = dict(ROI_STATS_DEFAULTS)
    for key in ROI_STATS_DEFAULTS:
        if key not in raw:
            continue
        val = raw.get(key)
        try:
            if isinstance(val, np.ndarray):
                if val.size == 0:
                    continue
                if val.size != 1:
                    raise ValueError(f"roi_stats key '{key}' expected scalar, got shape={val.shape}")
                val = val.reshape(-1)[0]
            elif isinstance(val, (list, tuple)):
                if len(val) == 0:
                    continue
                if len(val) != 1:
                    raise ValueError(f"roi_stats key '{key}' expected scalar, got sequence length={len(val)}")
                val = val[0]
            fv = float(val)
            if np.isfinite(fv):
                frame[key] = fv
        except Exception:
            continue
    frame["roi_cx"] = float(np.clip(frame["roi_cx"], 0.0, 1.0))
    frame["roi_cy"] = float(np.clip(frame["roi_cy"], 0.0, 1.0))
    frame["valid_ratio"] = float(np.clip(frame["valid_ratio"], 0.0, 1.0))
    frame["center_disp"] = max(frame["center_disp"], 0.0)
    if abs(frame["global_mean"]) < 1e-6 or not np.isfinite(frame["global_mean"]):
        frame["global_mean"] = 1.0
    return frame


def compute_roi_stats_time_series(rois: Optional[list]) -> Tuple[List[Dict[str, float]], float, float, float]:
    """
    Compute per-frame ROI statistics for quality estimation and legacy scalars.

    Returns:
        roi_stats_t: List of dicts with keys
            (roi_mean, roi_std, roi_snr_db, roi_cx, roi_cy, valid_ratio, center_disp, global_mean)
        mean_intensity: global average mean
        std_intensity: global average std
        snr_db: global SNR
    """
    if not rois:
        return [], float("nan"), float("nan"), float("nan")

    stats_t = []
    all_means: List[float] = []
    all_stds: List[float] = []
    prev_center: Optional[Tuple[float, float]] = None

    def _gradient_center(gray_img: np.ndarray, valid_mask: np.ndarray) -> Tuple[float, float]:
        """Content-based center proxy in [0,1]x[0,1] from gradient magnitude."""
        if gray_img.ndim != 2 or gray_img.shape[0] < 2 or gray_img.shape[1] < 2:
            return 0.5, 0.5
        if not np.any(valid_mask):
            return 0.5, 0.5
        fill_val = float(np.nanmedian(gray_img[valid_mask]))
        filled = np.where(valid_mask, gray_img, fill_val).astype(np.float64, copy=False)
        gy, gx = np.gradient(filled)
        mag = np.sqrt(gx * gx + gy * gy)
        mag = np.where(np.isfinite(mag), mag, 0.0)
        total = float(np.sum(mag))
        if total <= 1e-12:
            return 0.5, 0.5
        xs = np.linspace(0.0, 1.0, mag.shape[1], dtype=np.float64)
        ys = np.linspace(0.0, 1.0, mag.shape[0], dtype=np.float64)
        cx = float((np.sum(mag, axis=0) @ xs) / total)
        cy = float((np.sum(mag, axis=1) @ ys) / total)
        return float(np.clip(cx, 0.0, 1.0)), float(np.clip(cy, 0.0, 1.0))

    try:
        for roi in rois:
            frame_stats = {
                "roi_mean": 0.0, "roi_std": 0.0, "roi_snr_db": 0.0,
                "roi_cx": 0.5, "roi_cy": 0.5, "valid_ratio": 0.0,
                "center_disp": 0.0,
            }

            if roi is None:
                stats_t.append(frame_stats)
                continue

            roi_payload = roi if isinstance(roi, dict) else {}
            roi_obj = roi
            if roi_payload:
                for key in ("roi", "crop", "image", "frame"):
                    if key in roi_payload and roi_payload[key] is not None:
                        roi_obj = roi_payload[key]
                        break

            arr = np.asarray(roi_obj, dtype=np.float32)
            if arr.size == 0:
                stats_t.append(frame_stats)
                continue

            if arr.ndim == 3:
                gray = np.nanmean(arr, axis=2)
            elif arr.ndim == 2:
                gray = arr
            else:
                gray = arr.reshape(-1)

            valid_mask = np.isfinite(gray)
            valid_ratio = float(np.sum(valid_mask) / gray.size) if gray.size else 0.0

            if np.any(valid_mask):
                valid_vals = gray[valid_mask]
                f_mean = float(np.mean(valid_vals))
                f_std = float(np.std(valid_vals))
                f_snr = (
                    float(20.0 * np.log10(abs(f_mean) / max(f_std, 1e-6)))
                    if f_mean > 1e-9 else 0.0
                )
            else:
                f_mean = 0.0
                f_std = 0.0
                f_snr = 0.0

            cx, cy = _gradient_center(gray, valid_mask) if gray.ndim == 2 else (0.5, 0.5)

            try:
                if "roi_cx" in roi_payload:
                    cx = float(roi_payload["roi_cx"])
                if "roi_cy" in roi_payload:
                    cy = float(roi_payload["roi_cy"])
            except Exception:
                pass
            if not np.isfinite(cx):
                cx = 0.5
            if not np.isfinite(cy):
                cy = 0.5
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))

            if prev_center is None:
                center_disp = 0.0
            else:
                center_disp = float(np.hypot(cx - prev_center[0], cy - prev_center[1]))
            prev_center = (cx, cy)

            frame_stats.update({
                "roi_mean": f_mean,
                "roi_std": f_std,
                "roi_snr_db": f_snr,
                "roi_cx": cx,
                "roi_cy": cy,
                "valid_ratio": valid_ratio,
                "center_disp": center_disp,
            })

            stats_t.append(frame_stats)
            if np.isfinite(f_mean):
                all_means.append(f_mean)
            if np.isfinite(f_std):
                all_stds.append(f_std)

        if not all_means:
            return stats_t, float("nan"), float("nan"), float("nan")

        means = np.asarray(all_means, dtype=np.float64)
        stds = np.asarray(all_stds, dtype=np.float64)
        mean_intensity = float(np.nanmean(means))
        std_intensity = float(np.nanmean(stds))
        snr_db = (
            float(20.0 * np.log10(abs(mean_intensity) / max(std_intensity, 1e-6)))
            if std_intensity > 0 else float("nan")
        )

        global_mean = float(np.nanmedian(means)) if means.size else 1.0
        if not np.isfinite(global_mean) or abs(global_mean) < 1e-6:
            global_mean = 1.0
        for fs in stats_t:
            fs["global_mean"] = global_mean

        return stats_t, mean_intensity, std_intensity, snr_db

    except Exception:
        return [], float("nan"), float("nan"), float("nan")


def _save_roi_stats_cache(video_path: Optional[str], fs: float, stats_t: List[Dict[str, float]]) -> Optional[str]:
    cache_path = _roi_stats_cache_path(video_path)
    if not cache_path or not stats_t:
        return None
    keys = sorted(ROI_STATS_DEFAULTS.keys())
    payload: Dict[str, np.ndarray] = {
        "schema_version": np.asarray([ROI_STATS_CACHE_SCHEMA], dtype=np.str_),
        "n_frames": np.asarray([len(stats_t)], dtype=np.int32),
        "fps": np.asarray([float(fs)], dtype=np.float32),
    }
    for key in keys:
        payload[f"col__{key}"] = np.asarray(
            [float(frame.get(key, ROI_STATS_DEFAULTS[key])) for frame in stats_t],
            dtype=np.float32,
        )
    tmp_path = f"{cache_path}.tmp.npz"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, cache_path)
    return cache_path


def save_roi_stats_cache(video_path: Optional[str], fs: float, stats_t: List[Dict[str, float]]) -> Optional[str]:
    """Public wrapper for ROI stats cache writing (used by prep scripts)."""
    return _save_roi_stats_cache(video_path, fs, stats_t)


def _load_roi_stats_cache(
    video_path: Optional[str],
    *,
    expected_len: Optional[int],
    fs: Optional[float],
) -> Optional[Tuple[List[Dict[str, float]], str]]:
    cache_path = _roi_stats_cache_path(video_path)
    if not cache_path or not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            schema = ""
            if "schema_version" in npz:
                schema_arr = np.asarray(npz["schema_version"]).reshape(-1)
                if schema_arr.size > 0:
                    schema = str(schema_arr[0])
            if schema != ROI_STATS_CACHE_SCHEMA:
                return None
            n_frames = int(np.asarray(npz["n_frames"]).reshape(-1)[0]) if "n_frames" in npz else 0
            if n_frames <= 0:
                return None
            if expected_len is not None and int(expected_len) != n_frames:
                return None
            if fs is not None and "fps" in npz:
                cached_fs = float(np.asarray(npz["fps"]).reshape(-1)[0])
                if np.isfinite(cached_fs) and abs(cached_fs - float(fs)) > 1e-3:
                    return None

            stats_t: List[Dict[str, float]] = []
            for t in range(n_frames):
                frame_raw: Dict[str, Any] = {}
                for key in ROI_STATS_DEFAULTS:
                    col = f"col__{key}"
                    if col not in npz:
                        continue
                    arr = np.asarray(npz[col]).reshape(-1)
                    if t < arr.size:
                        frame_raw[key] = arr[t]
                stats_t.append(_coerce_roi_stats_frame(frame_raw))
        return stats_t, cache_path
    except Exception:
        return None


from components.observations.methods import MethodBase

class OscillatorWrappedMethod(MethodBase):
    """Wraps an existing chest-based method with an oscillator head."""

    def __init__(
        self,
        base_key: str,
        head_key: str,
        osc_params: Optional[OscillatorParams] = None,
        save_payload: Optional[Dict[str, bool]] = None,
        preproc_cfg: Optional[Dict] = None,
        ensemble_cfg: Optional[Dict] = None,
        gating_cfg: Optional[Dict] = None,
        quality_cfg: Optional[Dict] = None,
        trust_cfg: Optional[Dict] = None,
        gating_scope: str = "evaluation_only",
    ):
        super().__init__()
        self.base_key = base_key
        self.head_key = head_key
        self.name = f"{base_key}__{head_key}"
        self.data_type = "chest"
        self.base_method = _build_base(base_key)
        self.osc_head = build_head(head_key, params=osc_params)
        print(f"> Initialized {self.name} with head_params: {self.osc_head.params.__dict__}")
        self.save_payload = save_payload or {"npz": True}
        self._base_meta = {"base_method": base_key}
        if hasattr(self.base_method, "component_names"):
            self._base_meta["observation_families"] = list(getattr(self.base_method, "component_names"))
        self.preproc_cfg = copy.deepcopy(preproc_cfg) if isinstance(preproc_cfg, dict) else {}
        self.gating_cfg = copy.deepcopy(gating_cfg) if isinstance(gating_cfg, dict) else {}
        self.quality_cfg = copy.deepcopy(quality_cfg) if isinstance(quality_cfg, dict) else {}
        self.trust_cfg = copy.deepcopy(trust_cfg) if isinstance(trust_cfg, dict) else {}
        self.gating_scope = _normalize_gating_scope(gating_scope)
        self.ensemble_cfg = copy.deepcopy(ensemble_cfg) if isinstance(ensemble_cfg, dict) else {}
        self._unused_config_keys: List[str] = []
        if self.ensemble_cfg:
            # Explicitly track currently-unused config blocks to avoid silent ignore.
            self._unused_config_keys.append("method.ensemble")
            print(
                f"> Warning: {self.name} received 'ensemble' config, "
                "but this wrapper currently runs single-head inference (diagnostic-only config)."
            )
        setattr(self.osc_head, "preproc_cfg", copy.deepcopy(self.preproc_cfg))

    def _base_obs_cache_path(self, video_path: Optional[str]) -> Optional[str]:
        if not video_path:
            return None
        trial_dir = os.path.dirname(str(video_path))
        if not trial_dir:
            return None
        if self.base_key == "of_farneback":
            return os.path.join(trial_dir, "obs_of.npy")
        if self.base_key == "of_disp_bridge":
            return os.path.join(trial_dir, "obs_of_bridge.npy")
        if self.base_key == "dof":
            return os.path.join(trial_dir, "obs_dof.npy")
        if self.base_key == "profile1d_linear":
            return os.path.join(trial_dir, "obs_p1d_linear.npy")
        if self.base_key == "profile1d_quadratic":
            return os.path.join(trial_dir, "obs_p1d_quad.npy")
        if self.base_key == "profile1d_cubic":
            return os.path.join(trial_dir, "obs_p1d_cubic.npy")
        if self.base_key == "pair_of_p1d_quadratic":
            return None
        if self.base_key == "fusion_of_p1d_quadratic":
            return None
        if self.base_key == "assist_of_p1d_quadratic":
            return None
        if self.base_key == "assist_ofbridge_of":
            return None
        if self.base_key == "assist_ofbridge_p1d_quadratic":
            return None
        return None

    def can_run_without_chest_rois(self, data: Dict) -> bool:
        """
        Cache-only eligibility check.
        Requires both:
          1) base observation cache (.npy)
          2) ROI stats metadata source (in-memory roi_stats_t OR disk roi cache)
        """
        if not isinstance(data, dict):
            return False
        video_path = data.get("video_path")
        obs_cache = self._base_obs_cache_path(video_path)
        has_obs_cache = bool(obs_cache and os.path.exists(obs_cache))

        roi_stats_t = data.get("roi_stats_t")
        has_roi_mem = False
        if isinstance(roi_stats_t, (list, tuple)):
            has_roi_mem = len(roi_stats_t) > 0
        elif isinstance(roi_stats_t, dict):
            has_roi_mem = len(roi_stats_t) > 0
        roi_cache_path = _roi_stats_cache_path(video_path)
        has_roi_disk = bool(roi_cache_path and os.path.exists(roi_cache_path))
        return bool(has_obs_cache and (has_roi_mem or has_roi_disk))

    def _roi_stats_time_series(self, rois: Optional[list]) -> Tuple[List[Dict[str, float]], float, float, float]:
        return compute_roi_stats_time_series(rois)

    def _roi_intensity_stats(self, rois: Optional[list]) -> Tuple[float, float, float]:
        # Backwards compatibility wrapper
        _, m, s, snr = self._roi_stats_time_series(rois)
        return m, s, snr


    def _signal_spectral_meta(self, signal_arr: np.ndarray, fs: float, f_min: float, f_max: float) -> Dict[str, float]:
        meta: Dict[str, float] = {}
        try:
            nperseg = min(signal_arr.size, max(64, int(round(fs * 4.0))))
            if nperseg >= 8:
                freqs, psd = sps.welch(signal_arr, fs=fs, nperseg=nperseg)
                band_mask = (freqs >= max(f_min, 1e-3)) & (freqs <= max(f_max, f_min + 1e-3))
                if np.any(band_mask):
                    band_freqs = freqs[band_mask]
                    band_psd = psd[band_mask]
                    idx = int(np.argmax(band_psd))
                    peak_hz = float(band_freqs[idx])
                    peak_power = float(band_psd[idx])
                    median_power = float(np.median(band_psd) + 1e-9)
                    ratio = peak_power / median_power
                    ratio_safe = max(ratio, 1e-12)
                    meta["welch_peak_hz"] = peak_hz
                    meta["welch_peak_ratio"] = ratio
                    meta["welch_peak_db"] = float(10.0 * np.log10(ratio_safe))
                    # crude prominence proxy
                    meta["welch_prom_db"] = float(10.0 * np.log10(max(peak_power - median_power, 1e-9) / median_power))
                    # Half-power peak width proxy (FWHM-like).
                    if band_psd.size >= 3:
                        peak_power_safe = max(float(peak_power), 1e-12)
                        half_power = 0.5 * peak_power_safe
                        left = idx
                        right = idx
                        while left > 0 and float(band_psd[left]) >= half_power:
                            left -= 1
                        while right < (band_psd.size - 1) and float(band_psd[right]) >= half_power:
                            right += 1
                        f_left = float(band_freqs[left])
                        f_right = float(band_freqs[right])
                        meta["welch_fwhm_hz"] = float(max(f_right - f_left, 0.0))
                    if band_freqs.size > 1:
                        df = float(np.mean(np.diff(band_freqs)))
                        meta["welch_df_hz"] = df
        except Exception:
            pass
        return meta

    def _store_npz(self, data: Dict, result: Dict[str, np.ndarray]):
        aux_dir = data.get("aux_save_dir")
        trial_key = data.get("trial_key")
        if not aux_dir or not trial_key:
            return
        os.makedirs(aux_dir, exist_ok=True)
        payload = {
            "signal_hat": np.asarray(result["signal_hat"], dtype=np.float32),
            "track_hz": np.asarray(result["track_hz"], dtype=np.float32),
            "rr_hz": np.array([result["rr_hz"]], dtype=np.float32),
            "rr_bpm": np.array([result["rr_bpm"]], dtype=np.float32),
            "meta": np.array([result["meta"]], dtype=object),
        }
        np.savez_compressed(os.path.join(aux_dir, f"{trial_key}.npz"), **payload)
        components = result.get("components")
        if components:
            comp_dir = os.path.join(aux_dir, "components", trial_key)
            os.makedirs(comp_dir, exist_ok=True)
            for comp in components:
                comp_result = comp['result']
                head_name = comp['name']
                comp_payload = {
                    "signal_hat": np.asarray(comp_result["signal_hat"], dtype=np.float32),
                    "track_hz": np.asarray(comp_result["track_hz"], dtype=np.float32),
                    "rr_hz": np.array([comp_result["rr_hz"]], dtype=np.float32),
                    "rr_bpm": np.array([comp_result["rr_bpm"]], dtype=np.float32),
                    "meta": np.array([comp_result["meta"]], dtype=object),
                }
                np.savez_compressed(os.path.join(comp_dir, f"{head_name}.npz"), **comp_payload)

    def process(self, data: Dict) -> np.ndarray:
        # Execute base method first to obtain motion proxy y(t).
        base_signal = np.asarray(self.base_method.process(data), dtype=np.float64)
        if base_signal.ndim == 0:
            base_signal = base_signal.reshape(1)
        elif base_signal.ndim > 2:
            base_signal = np.asarray(base_signal, dtype=np.float64).reshape(-1)
        fs = float(data.get("fps", self.osc_head.params.fs))
        dataset_label = data.get("dataset_name") or data.get("dataset") or data.get("dataset_slug") or "unknown"
        if not str(data.get("trial_key") or "").strip():
            short_key, full_key = derive_trial_identifiers(data, dataset_name=dataset_label, sample_index=0)
            data["trial_key"] = short_key
            data["trial_key_full"] = full_key
            data["trial_uid"] = full_key
        meta = dict(self._base_meta)
        if hasattr(self.base_method, "get_runtime_meta"):
            try:
                runtime_meta = self.base_method.get_runtime_meta()
                if isinstance(runtime_meta, dict) and runtime_meta:
                    meta.update(copy.deepcopy(runtime_meta))
                    primary_family = runtime_meta.get("primary_observation_family_runtime")
                    if isinstance(primary_family, str) and primary_family.strip():
                        meta["base_method"] = primary_family.strip()
                        meta["assistant_base_method"] = self.base_key
            except Exception:
                pass
        trial_key = data.get("trial_key")
        meta.update({
            "head": self.head_key,
            "fs": fs,
            "dataset": dataset_label,
            "dataset_slug": dataset_label,
            "trial_key": trial_key,
            "trial_key_full": data.get("trial_key_full"),
            "trial_uid": data.get("trial_uid"),
            "subject": data.get("subject"),
            "trial": data.get("trial"),
            "method_name": self.name,
            "data_file": data.get("video_path"),
            "aux_save_dir": data.get("aux_save_dir"),
            "gating_scope": self.gating_scope,
        })
        if self._unused_config_keys:
            meta["unused_config_keys"] = list(self._unused_config_keys)
        if self.gating_cfg:
            meta["gating"] = copy.deepcopy(self.gating_cfg)
        if self.quality_cfg:
            meta["quality"] = copy.deepcopy(self.quality_cfg)
        if self.trust_cfg:
            meta["trust"] = copy.deepcopy(self.trust_cfg)
        # Base-signal diagnostics passed to oscillator heads
        if base_signal.size:
            if base_signal.ndim == 1:
                abs_sig = np.abs(base_signal)
                meta.update({
                    "signal_mean": float(np.nanmean(base_signal)),
                    "signal_std": float(np.nanstd(base_signal)),
                    "signal_ptp": float(np.nanmax(base_signal) - np.nanmin(base_signal)),
                    "signal_energy": float(np.nanmean(base_signal ** 2)),
                    "signal_abs_mean": float(np.nanmean(abs_sig)),
                    "signal_abs_std": float(np.nanstd(abs_sig)),
                    "signal_pos_fraction": float(np.mean(base_signal >= 0.0)),
                })
                meta.update(self._signal_spectral_meta(base_signal, fs, getattr(self.osc_head.params, "f_min", 0.08), getattr(self.osc_head.params, "f_max", 0.5)))
            elif base_signal.ndim == 2:
                channel_stats = []
                obs_families = list(meta.get("observation_families") or [])
                for idx in range(base_signal.shape[0]):
                    sig_i = np.asarray(base_signal[idx], dtype=np.float64).reshape(-1)
                    abs_sig = np.abs(sig_i)
                    rec = {
                        "channel_index": int(idx),
                        "family": obs_families[idx] if idx < len(obs_families) else f"ch{idx}",
                        "signal_mean": float(np.nanmean(sig_i)) if sig_i.size else float("nan"),
                        "signal_std": float(np.nanstd(sig_i)) if sig_i.size else float("nan"),
                        "signal_ptp": float(np.nanmax(sig_i) - np.nanmin(sig_i)) if sig_i.size else float("nan"),
                        "signal_energy": float(np.nanmean(sig_i ** 2)) if sig_i.size else float("nan"),
                        "signal_abs_mean": float(np.nanmean(abs_sig)) if sig_i.size else float("nan"),
                        "signal_abs_std": float(np.nanstd(abs_sig)) if sig_i.size else float("nan"),
                        "signal_pos_fraction": float(np.mean(sig_i >= 0.0)) if sig_i.size else float("nan"),
                    }
                    rec.update(self._signal_spectral_meta(sig_i, fs, getattr(self.osc_head.params, "f_min", 0.08), getattr(self.osc_head.params, "f_max", 0.5)))
                    channel_stats.append(rec)
                meta["signal_channels"] = channel_stats
                meta["signal_ndim"] = 2
                meta["signal_shape"] = [int(base_signal.shape[0]), int(base_signal.shape[1])]
        roi_stats_source = "computed"
        cache_path_used: Optional[str] = None
        roi_stats_t = data.get("roi_stats_t")
        has_chest_rois = bool(data.get("chest_rois"))
        if isinstance(roi_stats_t, list) and roi_stats_t:
            roi_stats_source = "memory_cache"
        else:
            expected_len = len(data.get("chest_rois") or [])
            loaded = _load_roi_stats_cache(
                data.get("video_path"),
                expected_len=expected_len if expected_len > 0 else None,
                fs=fs,
            )
            if loaded is not None:
                roi_stats_t, cache_path_used = loaded
                roi_stats_source = "disk_cache"
            else:
                if not has_chest_rois:
                    raise ValueError(
                        f"{self.name}: cannot build roi_stats_t without chest ROIs "
                        f"(video_path='{data.get('video_path', '')}'). "
                        "Provide chest_rois or precompute obs_roi_stats_v1.npz."
                    )
                roi_stats_t, _, _, _ = self._roi_stats_time_series(data.get("chest_rois"))
                if roi_stats_t:
                    try:
                        cache_path_used = _save_roi_stats_cache(data.get("video_path"), fs, roi_stats_t)
                    except Exception:
                        cache_path_used = None
                roi_stats_source = "computed"

        # Share per-sample ROI stats across wrapped methods to avoid recomputation.
        data["roi_stats_t"] = roi_stats_t if isinstance(roi_stats_t, list) else []
        roi_mean, roi_std, roi_snr_db = _roi_summary_from_stats(data["roi_stats_t"])
        data["roi_intensity_mean"] = roi_mean
        data["roi_intensity_std"] = roi_std
        data["roi_intensity_snr_db"] = roi_snr_db
        data["roi_stats_source"] = roi_stats_source
        if cache_path_used:
            data["roi_stats_cache_path"] = cache_path_used
        meta.update({
            "roi_stats_t": data["roi_stats_t"],
            "roi_intensity_mean": roi_mean,
            "roi_intensity_std": roi_std,
            "roi_intensity_snr_db": roi_snr_db,
            "roi_stats_source": roi_stats_source,
        })
        if cache_path_used:
            meta["roi_stats_cache_path"] = cache_path_used
        # Integrate Automatic EM Tuning (explicit experimental mode only).
        em_mode = getattr(self.osc_head.params, "em_mode", None)
        if hasattr(self, "params") and isinstance(self.params, dict):
             em_mode = em_mode or self.params.get("em_mode")
        em_mode_norm = str(em_mode or "").strip().lower()
        em_enabled = em_mode_norm in ("online", "trial")
        meta["em_mode_used"] = em_mode_norm if em_enabled else "off"
        meta["em_unavailable"] = False
        meta["em_qx"] = np.nan
        meta["em_qf"] = np.nan
        meta["em_rv"] = np.nan
        meta["em_iters"] = np.nan
        meta["em_converged"] = False
        run_head = self.osc_head
        if em_enabled and base_signal.size > 0 and base_signal.ndim == 1:
            try:
                from core.optimization.em_kalman import EMKalmanTrainer, EMConfig
                # Per-trial head instance prevents cross-trial parameter leakage.
                run_head = build_head(self.head_key, params=copy.deepcopy(self.osc_head.params))
                setattr(run_head, "preproc_cfg", copy.deepcopy(self.preproc_cfg))

                init_q = getattr(run_head.params, "qx", None) or getattr(run_head.params, "qf", 1e-4)
                init_r = getattr(run_head.params, "rv_floor", 0.01)
                em_cfg = EMConfig(init_q=float(init_q), init_r=float(init_r), max_iters=15)
                trainer = EMKalmanTrainer(em_cfg)
                em_result = trainer.fit([base_signal])
                opt_q = em_result.get("q")
                opt_r = em_result.get("r")
                if opt_q is not None:
                    if self.head_key == "ukffreq":
                        run_head.params.qf = float(opt_q)
                        meta["em_qf"] = float(opt_q)
                    else:
                        run_head.params.qx = float(opt_q)
                        meta["em_qx"] = float(opt_q)
                if opt_r is not None:
                    run_head.params.rv_floor = float(opt_r)
                    meta["em_rv"] = float(opt_r)
                meta["em_iters"] = float(em_result.get("n_iters", np.nan))
                meta["em_converged"] = bool(em_result.get("converged", False))
            except ImportError:
                meta["em_unavailable"] = True
                meta["em_mode_used"] = f"{em_mode_norm}_unavailable"
            except Exception as e:
                meta["em_mode_used"] = f"{em_mode_norm}_failed"
                meta["em_error"] = str(e)
        elif em_enabled and base_signal.ndim != 1:
            meta["em_mode_used"] = f"{em_mode_norm}_unsupported_multichannel"

        result = run_head.run(base_signal, fs, meta)
        if self.save_payload.get("npz", True):
            self._store_npz(data, result)
        return result


def create_wrapped_method(
    method_name: str,
    params: Optional[Dict] = None,
    oscillator_defaults: Optional[Dict] = None,
    preproc_defaults: Optional[Dict] = None,
    gating_defaults: Optional[Dict] = None,
    quality_defaults: Optional[Dict] = None,
    trust_defaults: Optional[Dict] = None,
    gating_scope_default: str = "evaluation_only",
) -> OscillatorWrappedMethod:
    if "__" not in method_name:
        raise ValueError("Wrapped method names must use `<base>__<head>` convention")
    base_part, head_part = method_name.split("__", 1)
    base_key = _normalize_base(base_part)
    head_key = _normalize_head(head_part)

    params = params or {}
    preproc_cfg = copy.deepcopy(preproc_defaults) if isinstance(preproc_defaults, dict) else {}
    gating_cfg = copy.deepcopy(gating_defaults) if isinstance(gating_defaults, dict) else {}
    quality_cfg = copy.deepcopy(quality_defaults) if isinstance(quality_defaults, dict) else {}
    trust_cfg = copy.deepcopy(trust_defaults) if isinstance(trust_defaults, dict) else {}

    # Also extract sub-configs from nested params block (config pattern: entry["params"]["trust"] etc.)
    _nested = params.get("params") if isinstance(params.get("params"), dict) else {}
    if isinstance(_nested.get("preproc"), dict):
        _v = _nested["preproc"]
        preproc_cfg = _deep_merge_dict(preproc_cfg, _v) if preproc_cfg else copy.deepcopy(_v)
    if isinstance(_nested.get("gating"), dict):
        _v = _nested["gating"]
        gating_cfg = _deep_merge_dict(gating_cfg, _v) if gating_cfg else copy.deepcopy(_v)
    if isinstance(_nested.get("quality"), dict):
        _v = _nested["quality"]
        quality_cfg = _deep_merge_dict(quality_cfg, _v) if quality_cfg else copy.deepcopy(_v)
    if isinstance(_nested.get("trust"), dict):
        _v = _nested["trust"]
        trust_cfg = _deep_merge_dict(trust_cfg, _v) if trust_cfg else copy.deepcopy(_v)

    if isinstance(params.get("preproc"), dict):
        if preproc_cfg:
            preproc_cfg = _deep_merge_dict(preproc_cfg, params["preproc"])
        else:
            preproc_cfg = copy.deepcopy(params["preproc"])
    if isinstance(params.get("gating"), dict):
        if gating_cfg:
            gating_cfg = _deep_merge_dict(gating_cfg, params["gating"])
        else:
            gating_cfg = copy.deepcopy(params["gating"])
    if isinstance(params.get("quality"), dict):
        if quality_cfg:
            quality_cfg = _deep_merge_dict(quality_cfg, params["quality"])
        else:
            quality_cfg = copy.deepcopy(params["quality"])
    if isinstance(params.get("trust"), dict):
        if trust_cfg:
            trust_cfg = _deep_merge_dict(trust_cfg, params["trust"])
        else:
            trust_cfg = copy.deepcopy(params["trust"])
    gating_scope = _normalize_gating_scope(params.get("gating_scope", gating_scope_default))
    # Flatten nested parameter dictionaries.
    merged_params: Dict[str, Any] = {}
    
    # 1. Primary parameter blocks
    for key in ("params", "head_params", "oscillator", "oscillator_params"):
        val = params.get(key)
        if isinstance(val, dict):
            # Special case: if it contains an 'oscillator' block itself (common in main config)
            if "oscillator" in val and isinstance(val["oscillator"], dict):
                merged_params.update(val["oscillator"])
            else:
                merged_params.update(val)

    # 1.5. Apply global oscillator defaults only for missing fields.
    if isinstance(oscillator_defaults, dict):
        for field, value in oscillator_defaults.items():
            if field not in merged_params:
                merged_params[field] = copy.deepcopy(value)
                
    # 2. Top-level overrides
    merged_params.update({k: v for k, v in params.items() if k not in ("name", "params", "head_params", "oscillator", "oscillator_params", "preproc")})

    ensemble_cfg = params.get("ensemble")
    # 3. Extract valid OscillatorParams fields
    osc_kwargs = {}
    valid_fields = set(OscillatorParams().__dict__.keys())
    for field, value in merged_params.items():
        if field in valid_fields:
            osc_kwargs[field] = value
            
    osc_params = OscillatorParams(**osc_kwargs) if osc_kwargs else None
    save_payload = params.get("save_payload")
    return OscillatorWrappedMethod(
        base_key,
        head_key,
        osc_params=osc_params,
        save_payload=save_payload,
        preproc_cfg=preproc_cfg,
        ensemble_cfg=ensemble_cfg,
        gating_cfg=gating_cfg,
        quality_cfg=quality_cfg,
        trust_cfg=trust_cfg,
        gating_scope=gating_scope,
    )
