import copy
import os
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from scipy import signal as sps
from .common import _deep_merge_dict, derive_trial_identifiers

from components.observations.methods import OF_Model, DoF_Model, profile1D_Model
from components.models import OscillatorParams, build_head

def _normalize_base(name: str) -> str:
    key = name.lower()
    if key in ("of_model", "of", "of_farneback"):
        return "of_farneback"
    if key == "dof":
        return "dof"
    if key in ("profile1d_linear", "profile1d linear", "profile1d-linear"):
        return "profile1d_linear"
    if key in ("profile1d_quadratic", "profile1d quadratic", "profile1d-quadratic"):
        return "profile1d_quadratic"
    if key in ("profile1d_cubic", "profile1d cubic", "profile1d-cubic"):
        return "profile1d_cubic"
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
    raise ValueError(f"Unsupported base key '{base_key}'")


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

    def _roi_stats_time_series(self, rois: Optional[list]) -> Tuple[List[Dict[str, float]], float, float, float]:
        """
        Compute per-frame ROI statistics for quality estimation and legacy scalars.
        
        Returns:
            roi_stats_t: List of dicts with keys (roi_mean, roi_std, roi_snr_db, roi_cx, roi_cy, valid_ratio)
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
                # Default "invalid" frame stats
                frame_stats = {
                    "roi_mean": 0.0, "roi_std": 0.0, "roi_snr_db": 0.0,
                    "roi_cx": 0.5, "roi_cy": 0.5, "valid_ratio": 0.0,
                    "center_disp": 0.0,
                }
                
                if roi is None:
                    stats_t.append(frame_stats)
                    continue

                # Dict-like ROI payloads can carry precomputed center/quality.
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

                # Convert to gray for stable stats / center proxy.
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

                # Approach B (spec): compute drift proxy from ROI content.
                cx, cy = _gradient_center(gray, valid_mask) if gray.ndim == 2 else (0.5, 0.5)

                # Allow external ROI metadata to override when present.
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

            # Aggregate for legacy scalars
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

            # Fallback global reference (full-frame mean unavailable):
            # robust constant baseline from ROI means.
            global_mean = float(np.nanmedian(means)) if means.size else 1.0
            if not np.isfinite(global_mean) or abs(global_mean) < 1e-6:
                global_mean = 1.0
            for fs in stats_t:
                fs["global_mean"] = global_mean
            
            return stats_t, mean_intensity, std_intensity, snr_db

        except Exception:
            # Fallback
            return [], float("nan"), float("nan"), float("nan")

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
                    meta["welch_peak_hz"] = peak_hz
                    meta["welch_peak_ratio"] = ratio
                    meta["welch_peak_db"] = float(10.0 * np.log10(ratio))
                    # crude prominence proxy
                    meta["welch_prom_db"] = float(10.0 * np.log10(max(peak_power - median_power, 1e-9) / median_power))
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
        base_signal = self.base_method.process(data)
        base_signal = np.asarray(base_signal, dtype=np.float64).reshape(-1)
        fs = float(data.get("fps", self.osc_head.params.fs))
        dataset_label = data.get("dataset_name") or data.get("dataset") or data.get("dataset_slug") or "unknown"
        if not str(data.get("trial_key") or "").strip():
            short_key, full_key = derive_trial_identifiers(data, dataset_name=dataset_label, sample_index=0)
            data["trial_key"] = short_key
            data["trial_key_full"] = full_key
            data["trial_uid"] = full_key
        meta = dict(self._base_meta)
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
        roi_stats_t, roi_mean, roi_std, roi_snr_db = self._roi_stats_time_series(data.get("chest_rois"))
        meta.update({
            "roi_stats_t": roi_stats_t,
            "roi_intensity_mean": roi_mean,
            "roi_intensity_std": roi_std,
            "roi_intensity_snr_db": roi_snr_db
        })
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
        if em_enabled and base_signal.size > 0:
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

        result = run_head.run(base_signal, fs, meta)
        if self.save_payload.get("npz", True):
            self._store_npz(data, result)
        return result


def create_wrapped_method(
    method_name: str,
    params: Optional[Dict] = None,
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
