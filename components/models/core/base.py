import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy import signal as sps
from scipy.signal import hilbert

from ..autotune.params_autotune import AutoTuneRepository
from core.optimization.em_kalman import load_em_params

_AUTOTUNE_REPO = AutoTuneRepository()

class PLLAdaptiveController:
    """Simple adaptive gain controller driven by SNR."""

    def __init__(self, zeta: float, ttrack: float, fs: float):
        self.zeta = float(max(zeta, 1e-3))
        self.ttrack = float(max(ttrack, 1e-3))
        self.fs = float(max(fs, 1.0))

    def gains(self, snr: float, kp_min: float, ki_min: float) -> Tuple[float, float]:
        snr = max(float(snr), 0.0)
        snr_scale = np.clip(np.log1p(snr) / np.log(10.0), 0.2, 2.0)
        omega_n = 2.0 * np.pi * snr_scale / self.ttrack
        kp = max(2.0 * self.zeta * omega_n / self.fs, kp_min)
        ki = max((omega_n ** 2) / (self.fs ** 2), ki_min)
        return kp, ki


@dataclass
class OscillatorParams:
    """Configuration shared across oscillator-based heads."""

    fs: float = 64.0
    f_min: float = 0.08
    f_max: float = 0.50
    init_margin_hz: float = 0.01  # interior guard in Hz to prevent boundary-locked initialisation
    rho: float = 0.0  # legacy manual override
    tau_env: float = 30.0
    qx: float = 1e-4  # baseline state noise (can be overridden if needed)
    qx_scale: float = 0.3
    rv: float = 0.1  # legacy manual override
    rv_auto: bool = True
    rv_mad_scale: float = 1.2
    rv_floor: float = 0.03  # observation noise floor, typically 0.02-0.05
    qf: float = 5e-5  # frequency random-walk noise; physiologic default (~1-3 bpm drift/30s). Recommended sweep [5e-6, 5e-4], up to 1e-3 only for very low SNR
    qf_override: Optional[float] = None
    qx_override: Optional[float] = None
    rv_floor_override: Optional[float] = None
    tau_env_override: Optional[float] = None
    ukf_alpha: float = 1e-3
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0
    spec_overlap: float = 0.5
    spec_nfft_factor: int = 1
    spec_peak_smooth_len: int = 1
    spec_subbin_interp: str = "parabolic"
    post_smooth_alpha: float = 0.0
    stft_win: float = 12.0
    stft_hop: float = 1.0
    ridge_penalty: float = 250.0
    spec_guidance_strength: float = 0.8
    spec_guidance_offset: float = 0.1
    spec_guidance_confidence_scale: float = 5.0
    spec_guidance_snr_scale: float = 3.5
    pll_autogain: bool = True
    pll_kp: float = 0.0  # legacy manual override
    pll_ki: float = 0.0  # legacy manual override
    pll_kp_min: float = 0.0
    pll_ki_min: float = 0.0
    pll_zeta: float = 0.9
    pll_ttrack: float = 7.0
    # Robust / Attention specific
    student_t_nu: float = 4.0  # Degrees of freedom for Student-t (lower = heavier tails)
    attention_win: int = 5     # Window size for attention/reliability calculation
    attention_alpha: float = 0.1 # Learning rate for attention weights
    detrend: bool = True
    bandpass: bool = True
    zscore: bool = True

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class _BaseOscillatorHead:
    """Shared helpers for oscillator heads."""

    head_key: str = "base"

    def __init__(self, params: Optional[OscillatorParams] = None):
        self.params = params or OscillatorParams()
        self._last_sigma_y = None
        self._last_init_freq = None
        self._last_signal_std = None
        self._last_snr = None
        self.preproc_cfg: Dict = {}
        self._last_preproc_meta: Dict = {}
        self._autotune_repo = _AUTOTUNE_REPO
        self._autotune_cache: Dict[Tuple[str, str], bool] = {}

    def _preprocess(self, signal: np.ndarray, fs: float) -> np.ndarray:
        p = self.params
        x = np.asarray(signal, dtype=np.float64).copy()
        if x.size == 0:
            return x
        if np.any(np.isnan(x)):
            x = np.nan_to_num(x)
        if p.detrend:
            x = sps.detrend(x, type="linear")
        if p.bandpass and fs > 0:
            nyq = 0.5 * fs
            low = max(p.f_min, 0.01)
            high = min(p.f_max, nyq - 1e-3)
            if high > low:
                b, a = sps.butter(2, [low / nyq, high / nyq], btype="bandpass")
                x = sps.filtfilt(b, a, x, method="gust")
        preproc_cfg = getattr(self, "preproc_cfg", {}) or {}
        self._last_preproc_meta = {}
        sign_cfg = preproc_cfg.get("sign_align", {})
        sign_seconds = float(sign_cfg.get("seconds", 12.0)) if isinstance(sign_cfg, dict) else 12.0
        if sign_cfg.get("enabled") and fs > 0 and x.size:
            seg_len = int(min(x.size, max(1, round(sign_seconds * fs))))
            if seg_len > 1:
                coarse = self._coarse_freq(x[:seg_len], fs)
                if np.isfinite(coarse) and coarse > 0.0:
                    t = np.arange(seg_len, dtype=np.float64) / fs
                    ref = np.cos(2.0 * np.pi * coarse * t)
                    dot = float(np.dot(x[:seg_len], ref))
                    if dot < 0:
                        x = -x
        median = float(np.median(x)) if x.size else 0.0
        if not np.isfinite(median):
            median = 0.0
        abs_dev = np.abs(x - median)
        mad = float(np.median(abs_dev)) if abs_dev.size else 0.0
        if not np.isfinite(mad) or mad < 0.0:
            mad = 0.0
        sigma_hat = 1.4826 * mad
        if not np.isfinite(sigma_hat) or sigma_hat < 0.0:
            sigma_hat = 0.0
        self._last_sigma_y = float(sigma_hat if sigma_hat > 0.0 else 0.0)
        signal_std = float(np.std(x)) if x.size else 0.0
        self._last_signal_std = signal_std
        self._last_snr = float(signal_std / max(self._last_sigma_y, 1e-6)) if self._last_sigma_y else 0.0
        robust_cfg = preproc_cfg.get("robust_zscore", {}) or {}
        robust_flag = robust_cfg.get("enabled")
        use_robust = bool(p.zscore) and (bool(robust_flag) if robust_flag is not None else True)
        eps = float(robust_cfg.get("eps", 1e-6))
        if not np.isfinite(eps) or eps <= 0.0:
            eps = 1e-6
        clip_raw = robust_cfg.get("clip", 3.5)
        clip_val = None
        if clip_raw is not None:
            try:
                clip_val = float(clip_raw)
            except (TypeError, ValueError):
                clip_val = None
        denom = max(sigma_hat, eps)
        clipped_frac = 0.0
        if use_robust:
            z = (x - median) / denom
            if clip_val is not None and clip_val > 0.0:
                clipped_frac = float(np.mean(np.abs(z) >= clip_val)) if z.size else 0.0
                x = np.clip(z, -clip_val, clip_val)
            else:
                clip_val = None
                x = z
        elif p.zscore:
            std = np.std(x)
            mean = np.mean(x)
            if std > 1e-8:
                x = (x - mean) / (std + 1e-8)
            else:
                x = x - mean
        self._last_preproc_meta = {
            "robust_z": {
                "enabled": bool(use_robust),
                "med": float(median),
                "mad": float(mad),
                "sigma_hat": float(sigma_hat),
                "clip": None if clip_val is None else float(clip_val),
                "clipped_frac": float(clipped_frac if use_robust and clip_val is not None else 0.0)
            }
        }
        return x

    def _apply_post_smoothing(self, track: np.ndarray) -> np.ndarray:
        alpha = getattr(self.params, "post_smooth_alpha", None)
        if alpha is None or (not np.isfinite(alpha)):
            return track
        if alpha <= 0.0 or alpha >= 1.0 or track.size < 2:
            return track
        smoothed = np.asarray(track, dtype=np.float64).copy()
        coeff = float(alpha)
        for idx in range(1, smoothed.size):
            smoothed[idx] = coeff * smoothed[idx - 1] + (1.0 - coeff) * smoothed[idx]
        return smoothed

    def _timebase(self, n: int, fs: float) -> np.ndarray:
        if fs <= 0:
            return np.arange(n)
        return np.arange(n) / fs

    def _dataset_from_meta(self, meta: Optional[Dict]) -> str:
        if isinstance(meta, dict):
            dataset = meta.get("dataset") or meta.get("dataset_name") or meta.get("dataset_slug")
            if dataset:
                return str(dataset).lower()
            data_file = meta.get("data_file")
            if isinstance(data_file, str) and data_file:
                token = os.path.basename(data_file).split('_')[0]
                if token:
                    return token.lower()
        return "unknown"

    def _method_identifier(self, meta: Optional[Dict]) -> str:
        if isinstance(meta, dict):
            name = meta.get("method_name")
            if isinstance(name, str) and name:
                return name
            base = meta.get("base_method") or "unknown"
        else:
            base = "unknown"
        return f"{base}__{self.head_key}"

    def _maybe_apply_autotune(self, meta: Optional[Dict]):
        dataset = self._dataset_from_meta(meta)
        method_id = self._method_identifier(meta)
        cache_key = (dataset, method_id)
        if cache_key in self._autotune_cache:
            return
        # 1) Load dataset/head-specific autotune overrides (top-K heuristics).
        overrides = self._autotune_repo.load_params(dataset, method_id)
        if not overrides:
            overrides = self._autotune_repo.load_params(dataset, self.head_key)
        if overrides:
            for field, value in overrides.items():
                if hasattr(self.params, field) and value is not None:
                    try:
                        setattr(self.params, field, float(value))
                    except Exception:
                        setattr(self.params, field, value)
        # 2) EM-based Q/R estimates always take precedence for qx/rv_floor,
        # ensuring that principled EM training cannot be silently overridden
        # by heuristic autotune files.
        em_overrides = load_em_params(dataset, method_id)
        if em_overrides:
            if 'q' in em_overrides:
                setattr(self.params, 'qx_override', float(em_overrides['q']))
            if 'r' in em_overrides:
                setattr(self.params, 'rv_floor_override', float(em_overrides['r']))
        # 3) Clamp overrides so mis-estimated EM/auto-tune values cannot explode the
        # oscillator random walk and destabilise tracks.
        base_qx = self.params.qx if (self.params.qx and self.params.qx > 0) else None
        if base_qx is None or not np.isfinite(base_qx):
            # fallback: derive a small, conservative base from qx_scale
            base_qx = max((self.params.qx_scale or 0.0) * 1e-4, 5e-5)
        qx_override = getattr(self.params, "qx_override", None)
        if qx_override is not None and np.isfinite(qx_override) and qx_override > 0:
            qx_cap = max(5.0 * base_qx, base_qx + 5e-5)
            self.params.qx_override = float(np.clip(qx_override, 1e-7, qx_cap))
        base_rv_floor = self.params.rv_floor if (self.params.rv_floor and self.params.rv_floor > 0) else 0.03
        rv_override = getattr(self.params, "rv_floor_override", None)
        if rv_override is not None and np.isfinite(rv_override) and rv_override > 0:
            rv_cap = max(5.0 * base_rv_floor, base_rv_floor + 0.02)
            # Prevent EM/autotune from collapsing the observation noise far below the
            # dataset-tuned base; very small R makes trackers jittery and hurts MAE.
            rv_floor_min = max(0.5 * base_rv_floor, 1e-3)
            self.params.rv_floor_override = float(np.clip(rv_override, rv_floor_min, rv_cap))
        base_qf = self.params.qf if (self.params.qf and self.params.qf > 0) else 5e-5
        qf_override = getattr(self.params, "qf_override", None)
        if qf_override is not None and np.isfinite(qf_override) and qf_override > 0:
            qf_cap = max(5.0 * base_qf, 1e-3)
            self.params.qf_override = float(np.clip(qf_override, 1e-7, qf_cap))
        # 4) Apply a light default smoother to tracker heads when none was set;
        # helps stabilise MAE/RMSE without affecting spectral metrics.
        if self.head_key in ("kfstd", "ukffreq", "pll"):
            alpha = getattr(self.params, "post_smooth_alpha", 0.0)
            if not (alpha > 0.0 and alpha < 1.0):
                self.params.post_smooth_alpha = 0.88
        self._autotune_cache[cache_key] = True

    def _log_autotune_stats(self, meta: Optional[Dict], freq: float):
        if self._autotune_repo is None:
            return
        dataset = self._dataset_from_meta(meta)
        method_id = self._method_identifier(meta)
        mad = self._last_sigma_y if self._last_sigma_y is not None else 0.0
        snr = self._last_snr if self._last_snr is not None else 0.0
        freq_val = float(freq) if np.isfinite(freq) else 0.0
        extra = {
            "snr": snr,
            "mad": mad,
            "signal_std": self._last_signal_std if self._last_signal_std is not None else 0.0,
            "init_margin": float(getattr(self.params, "init_margin_hz", 0.0))
        }
        try:
            self._autotune_repo.record_stats(dataset, method_id, mad, snr, freq_val, extra)
        except Exception:
            pass

    def _welch_candidate(self, signal: np.ndarray, fs: float) -> Tuple[float, float, Dict]:
        p = self.params
        if signal.size < 8 or fs <= 0:
            return float("nan"), 0.0, {}
        win = int(min(max(fs * 20, fs), signal.size))
        if win < 8:
            win = min(256, signal.size)
        freqs, power = sps.welch(signal, fs=fs, nperseg=win)
        mask = (freqs >= p.f_min) & (freqs <= p.f_max)
        if not np.any(mask):
            return float("nan"), 0.0, {}
        sub_freqs = freqs[mask]
        sub_power = power[mask]
        idx = int(np.argmax(sub_power))
        peak_power = float(sub_power[idx])
        median_power = float(np.median(sub_power) + 1e-9)
        peak_ratio = peak_power / median_power
        confidence = np.clip(peak_ratio / 4.0, 0.0, 5.0)
        return float(sub_freqs[idx]), float(confidence), {"label": "welch", "peak_ratio": peak_ratio}

    def _autocorr_candidate(self, signal: np.ndarray, fs: float) -> Tuple[float, float, Dict]:
        p = self.params
        if signal.size < 16 or fs <= 0:
            return float("nan"), 0.0, {}
        autocorr = sps.correlate(signal, signal, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        lag_min = int(max(1, round(fs / max(p.f_max, 1e-3))))
        lag_max = int(min(signal.size - 1, round(fs / max(p.f_min, 1e-3))))
        if lag_max <= lag_min:
            return float("nan"), 0.0, {}
        window = autocorr[lag_min:lag_max]
        if window.size == 0:
            return float("nan"), 0.0, {}
        idx = int(np.argmax(window))
        lag = lag_min + idx
        freq = fs / lag if lag > 0 else float("nan")
        confidence = float(window[idx] / max(autocorr[0], 1e-9))
        return float(freq), float(confidence), {"label": "autocorr", "lag": lag}

    def _hilbert_candidate(self, signal: np.ndarray, fs: float) -> Tuple[float, float, Dict]:
        p = self.params
        if signal.size < 16 or fs <= 0:
            return float("nan"), 0.0, {}
        try:
            analytic = hilbert(signal)
            phase = np.unwrap(np.angle(analytic))
            inst_freq = np.diff(phase)
            inst_freq = (fs / (2.0 * np.pi)) * inst_freq
            valid = inst_freq[(inst_freq >= p.f_min) & (inst_freq <= p.f_max)]
            if valid.size == 0:
                return float("nan"), 0.0, {}
            freq = float(np.median(valid))
            spread = float(np.std(valid))
            confidence = float(1.0 / (1.0 + spread))
            return freq, confidence, {"label": "hilbert", "spread": spread}
        except Exception:
            return float("nan"), 0.0, {}

    def _coarse_freq(self, signal: np.ndarray, fs: float, meta: Optional[Dict] = None) -> float:
        p = self.params
        candidates: List[Tuple[float, float, Dict]] = []
        candidates.append(self._welch_candidate(signal, fs))
        candidates.append(self._autocorr_candidate(signal, fs))
        candidates.append(self._hilbert_candidate(signal, fs))
        # External candidates supplied by upstream motion stage
        if isinstance(meta, dict):
            ext_hz = meta.get("welch_peak_hz")
            ext_ratio = meta.get("welch_peak_ratio")
            ext_db = meta.get("welch_peak_db")
            ext_prom = meta.get("welch_prom_db")
            confidence = None
            if isinstance(ext_ratio, (int, float, np.floating)) and np.isfinite(ext_ratio):
                confidence = float(max(ext_ratio, 1e-3))
            elif isinstance(ext_db, (int, float, np.floating)) and np.isfinite(ext_db):
                confidence = float(max(10.0 ** (ext_db / 10.0), 1e-3))
            if confidence is not None and isinstance(ext_hz, (int, float, np.floating)) and np.isfinite(ext_hz):
                candidates.append((float(ext_hz), confidence, {"label": "external_welch", "peak_db": ext_db, "peak_ratio": ext_ratio, "prom_db": ext_prom}))
            ext_custom = meta.get("external_coarse_candidates")
            if isinstance(ext_custom, (list, tuple)):
                for cand in ext_custom:
                    if isinstance(cand, dict):
                        hz = cand.get("hz")
                        conf = cand.get("confidence", cand.get("conf", cand.get("weight")))
                        if isinstance(hz, (int, float, np.floating)) and np.isfinite(hz) and isinstance(conf, (int, float, np.floating)) and np.isfinite(conf):
                            candidates.append((float(hz), float(max(conf, 1e-6)), {"label": cand.get("label", "external")}))
        valid = [(freq, conf, meta) for freq, conf, meta in candidates if np.isfinite(freq) and freq > 0 and np.isfinite(conf) and conf > 0]
        if not valid:
            freq_final = 0.5 * (p.f_min + p.f_max)
            self._last_init_freq = {"raw_hz": freq_final}
            return freq_final
        conf_values = np.asarray([max(v[1], 1e-6) for v in valid], dtype=np.float64)
        freqs = np.asarray([v[0] for v in valid], dtype=np.float64)
        weights = conf_values / np.sum(conf_values)
        blended = float(np.sum(freqs * weights))
        best_idx = int(np.argmax([v[1] for v in valid]))
        freq_raw = float(valid[best_idx][0])
        margin = getattr(p, "init_margin_hz", 0.0) or 0.0
        try:
            margin = float(margin)
        except (TypeError, ValueError):
            margin = 0.0
        if not np.isfinite(margin) or margin < 0.0:
            margin = 0.0
        band = float(p.f_max) - float(p.f_min)
        if band > 0.0 and (margin * 2.0) >= band:
            margin = max(0.0, 0.5 * band - 1e-8)
        freq = blended
        if margin > 0.0 and p.f_max > p.f_min:
            # Start oscillators inside the band to avoid edge-locking in low SNR regimes.
            interior_low = float(p.f_min) + margin
            interior_high = float(p.f_max) - margin
            if interior_low <= interior_high:
                freq = float(np.clip(freq, interior_low, interior_high))
        freq_interior = float(freq)
        freq_final = float(np.clip(freq, p.f_min, p.f_max))
        candidate_meta = []
        for freq_val, conf_val, meta in valid:
            entry = {"hz": float(freq_val), "confidence": float(conf_val)}
            if isinstance(meta, dict):
                entry.update(meta)
            candidate_meta.append(entry)
        max_conf = float(np.max([v[1] for v in valid]))
        weighted_conf = float(np.sum(weights * np.asarray([v[1] for v in valid], dtype=np.float64)))
        self._last_init_freq = {
            "raw_hz": float(freq_raw),
            "interior_hz": float(freq_interior),
            "final_hz": float(freq_final),
            "candidates": candidate_meta,
            "confidence_weighted": weighted_conf,
            "confidence_max": float(max_conf)
        }
        return freq_final

    def _effective_params(self, fs: float, meta: Optional[Dict] = None) -> Dict[str, float]:
        p = self.params
        fs = fs or p.fs
        tau_override = p.tau_env_override if (p.tau_env_override is not None and p.tau_env_override > 0) else None
        tau = max(tau_override if tau_override is not None else p.tau_env, 1e-3)
        rho = np.exp(-1.0 / max(fs * tau, 1e-6))
        if p.rho and p.rho > 0:
            rho = np.clip(p.rho, 0.0, 0.999999)
        if p.qx and p.qx > 0:
            qx = p.qx
        else:
            qx = max((p.qx_scale or 0.0) * (1.0 - rho ** 2), 1e-8)
        if p.qx_override is not None and p.qx_override > 0:
            qx = float(p.qx_override)
        rv_floor = p.rv_floor_override if (p.rv_floor_override is not None and p.rv_floor_override > 0) else p.rv_floor
        if p.rv_auto:
            sigma = self._last_sigma_y if (self._last_sigma_y is not None and np.isfinite(self._last_sigma_y)) else 1.0
            # Large sigma values occasionally appear on outlier trials (e.g., extreme motion);
            # cap to avoid inflating the measurement noise by orders of magnitude.
            sigma_cap = 10.0
            if sigma > sigma_cap:
                sigma = sigma_cap
            rv = max((p.rv_mad_scale * sigma) ** 2, rv_floor)
        else:
            rv = max(p.rv, rv_floor)
        # Upper-bound rv so the filter does not become unresponsive on high-variance clips.
        rv_cap = max(50.0 * rv_floor, rv_floor + 1.0)
        rv = min(rv, rv_cap)
        # Base (pre-guidance) frequency diffusion level used to bound qf.
        qf_base = p.qf if (p.qf and p.qf > 0) else 5e-5
        qf = qf_base
        if p.qf_override is not None and p.qf_override > 0:
            qf = float(p.qf_override)
            qf_base = qf
        # Meta-driven scaling: motion energy / ROI variance / SNR hints from the base method.
        if isinstance(meta, dict):
            try:
                sig_std = float(meta.get("signal_std", float("nan")))
            except Exception:
                sig_std = float("nan")
            try:
                abs_mean = float(meta.get("signal_abs_mean", float("nan")))
            except Exception:
                abs_mean = float("nan")
            try:
                roi_std = float(meta.get("roi_intensity_std", float("nan")))
            except Exception:
                roi_std = float("nan")
            try:
                roi_mean = float(meta.get("roi_intensity_mean", float("nan")))
            except Exception:
                roi_mean = float("nan")
            motion_ratios = []
            if np.isfinite(sig_std) and abs_mean > 1e-6:
                motion_ratios.append(sig_std / abs_mean)
            if np.isfinite(roi_std) and roi_mean > 1e-6:
                motion_ratios.append(roi_std / roi_mean)
            motion_level = float(np.nanmedian(motion_ratios)) if motion_ratios else 1.0
            motion_level = float(np.clip(motion_level, 0.3, 4.0))
            # Scale qx/qf/rv with motion_level (higher motion -> looser filter)
            scale = float(np.clip(1.0 + 0.35 * (motion_level - 1.0), 0.7, 2.5))
            qx *= scale
            qf *= scale
            rv *= (0.8 + 0.4 * scale)
            # ROI SNR hint
            roi_snr_db = meta.get("roi_intensity_snr_db")
            if isinstance(roi_snr_db, (int, float, np.floating)) and np.isfinite(roi_snr_db):
                if roi_snr_db < 5.0:
                    rv *= 1.5
                    qx *= 1.2
                    qf *= 1.2
                elif roi_snr_db > 15.0:
                    rv *= 0.85
            # Spectral peak confidence hint from base stage
            peak_ratio = meta.get("welch_peak_ratio")
            if isinstance(peak_ratio, (int, float, np.floating)) and np.isfinite(peak_ratio):
                if peak_ratio > 3.0:
                    qx *= 0.85
                    qf *= 0.85
                    rv *= 0.9
                elif peak_ratio < 1.5:
                    qx *= 1.15
                    qf *= 1.15
                    rv *= 1.1
            # Motion-dependent smoother (only if not explicitly set to a strong value)
            alpha = getattr(self.params, "post_smooth_alpha", 0.0)
            target_alpha = alpha
            if motion_level > 1.2:
                target_alpha = max(target_alpha, min(0.98, 0.9 + 0.03 * (motion_level - 1.0)))
            elif motion_level < 0.9:
                target_alpha = max(target_alpha, 0.85)
            if np.isfinite(target_alpha) and 0.0 < target_alpha < 1.0:
                self.params.post_smooth_alpha = float(target_alpha)
        # Guardrails to prevent runaway variances
        qx = float(np.clip(qx, 1e-7, 1e-1))
        qf = float(np.clip(qf, 1e-7, 5e-3))
        rv = float(np.clip(rv, rv_floor, rv_cap))
        qx, qf, rv = self._apply_spectral_guidance(qx, qf, rv, rv_floor, qf_base)
        return {
            'rho': float(rho),
            'qx': float(qx),
            'rv': float(rv),
            'qf': float(qf)
        }

    def _spectral_guidance_score(self) -> float:
        info = getattr(self, "_last_init_freq", {}) or {}
        candidates = info.get("candidates") if isinstance(info, dict) else None
        best_conf = 0.0
        if isinstance(candidates, (list, tuple)):
            for cand in candidates:
                if isinstance(cand, dict):
                    try:
                        val = float(cand.get('confidence', 0.0))
                    except (TypeError, ValueError):
                        val = 0.0
                    if np.isfinite(val):
                        best_conf = max(best_conf, val)
        conf_scale = getattr(self.params, 'spec_guidance_confidence_scale', 5.0) or 5.0
        if conf_scale <= 0.0:
            conf_scale = 5.0
        conf_norm = float(np.clip(best_conf / conf_scale, 0.0, 1.0))
        snr_val = max(self._last_snr or 0.0, 0.0)
        snr_scale = getattr(self.params, 'spec_guidance_snr_scale', 3.5) or 3.5
        if snr_scale <= 0.0:
            snr_scale = 3.5
        snr_norm = float(np.clip(np.tanh(snr_val / snr_scale), 0.0, 1.0))
        return float(0.6 * conf_norm + 0.4 * snr_norm)

    def _apply_spectral_guidance(self, qx: float, qf: float, rv: float, rv_floor: float, qf_base: float) -> Tuple[float, float, float]:
        strength = float(getattr(self.params, 'spec_guidance_strength', 0.0) or 0.0)
        if strength <= 0.0:
            # Even when guidance is nominally off, keep a minimal random-walk
            # variance so that trackers never become perfectly static and
            # frequency-locked at band edges.
            qf_floor_rel = max(0.1 * float(qf_base), 1e-7)
            qf = max(qf, qf_floor_rel)
            return qx, qf, rv
        offset = float(getattr(self.params, 'spec_guidance_offset', 0.0) or 0.0)
        score = np.clip(self._spectral_guidance_score(), 0.0, 1.0)
        centered = score - offset
        # Bound qf between a floor/ceiling derived from the configured level so
        # that spectral guidance cannot collapse it to ~0 or explode it.
        qf_floor_rel = max(0.1 * float(qf_base), 1e-7)
        qf_ceil_rel = max(10.0 * float(qf_base), 1e-3)
        if centered >= 0.0:
            tighten = 1.0 / max(1.0 + strength * centered, 1e-6)
            qf = max(qf * tighten, qf_floor_rel)
            qx = max(qx * (0.7 * tighten + 0.3), 1e-9)
            rv = max(rv * (0.6 * tighten + 0.4), rv_floor)
        else:
            loosen = 1.0 + strength * (-centered)
            qf = min(qf * loosen, qf_ceil_rel)
            qx = min(qx * (0.5 * loosen + 0.5), 1.0)
            rv = max(min(rv * (0.5 * loosen + 0.5), 10.0), rv_floor)
        return qx, qf, rv

    def _package(
        self,
        signal_hat: np.ndarray,
        track_hz: np.ndarray,
        meta_extra: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        track = np.asarray(track_hz, dtype=np.float64)
        signal_arr = np.asarray(signal_hat, dtype=np.float64)
        valid = track[~np.isnan(track)]
        rr_hz = float(np.nanmedian(valid)) if valid.size else float("nan")
        rr_bpm = float(rr_hz * 60.0) if np.isfinite(rr_hz) else float("nan")
        finite_track = track[np.isfinite(track)]
        if finite_track.size:
            sat_mask = (finite_track <= (self.params.f_min + 1e-6)) | (finite_track >= (self.params.f_max - 1e-6))
            frac_saturated = float(np.mean(sat_mask))
            dyn_range = float(np.percentile(finite_track, 95) - np.percentile(finite_track, 5))
        else:
            frac_saturated = float("nan")
            dyn_range = float("nan")
        meta = {
            "head": self.head_key,
            "params": self.params.to_dict(),
            "track_frac_saturated": frac_saturated,
            "track_dyn_range_hz": dyn_range
        }
        snr_val = self._last_snr if (self._last_snr is not None and np.isfinite(self._last_snr)) else float('nan')
        meta["snr_estimate"] = float(snr_val) if np.isfinite(snr_val) else float('nan')
        init_freq = getattr(self, "_last_init_freq", None)
        if isinstance(init_freq, dict):
            raw_val = init_freq.get("raw_hz")
            if raw_val is not None and np.isfinite(raw_val):
                meta["init_freq_raw_hz"] = float(raw_val)
            interior_val = init_freq.get("interior_hz")
            if interior_val is not None and np.isfinite(interior_val):
                meta["init_freq_interior_hz"] = float(interior_val)
            final_val = init_freq.get("final_hz")
            if final_val is not None and np.isfinite(final_val):
                meta["init_freq_final_hz"] = float(final_val)
            meta["coarse_candidates"] = init_freq.get("candidates")
        preproc_meta = getattr(self, "_last_preproc_meta", None)
        if isinstance(preproc_meta, dict):
            meta.update(preproc_meta)
        if meta_extra:
            meta.update(meta_extra)
        f0 = meta.get("f0", rr_hz)
        self._log_autotune_stats(meta, f0 if isinstance(f0, (int, float)) else rr_hz)
        return {
            "signal_hat": signal_arr,
            "track_hz": track,
            "rr_hz": rr_hz,
            "rr_bpm": rr_bpm,
            "meta": json.dumps(meta),
        }

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        raise NotImplementedError
