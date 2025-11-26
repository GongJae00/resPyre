import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
from scipy import signal as sps
from scipy.signal import hilbert

from .params_autotune import AutoTuneRepository
from riv.optim.em_kalman import load_em_params

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
            self.params.rv_floor_override = float(np.clip(rv_override, 1e-6, rv_cap))
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

    def _coarse_freq(self, signal: np.ndarray, fs: float) -> float:
        p = self.params
        candidates: List[Tuple[float, float, Dict]] = []
        candidates.append(self._welch_candidate(signal, fs))
        candidates.append(self._autocorr_candidate(signal, fs))
        candidates.append(self._hilbert_candidate(signal, fs))
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

    def _effective_params(self, fs: float) -> Dict[str, float]:
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


class oscillator_KFstd(_BaseOscillatorHead):
    head_key = "kfstd"

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        p = self.params
        fs = fs or p.fs
        self._maybe_apply_autotune(meta)
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        dt = 1.0 / fs
        freq0 = self._coarse_freq(y, fs)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))
        omega0 = 2.0 * np.pi * freq0
        eff = self._effective_params(fs)
        rho = eff['rho']
        qx = eff['qx']
        rv = eff['rv']

        cos_w = np.cos(omega0 * dt)
        sin_w = np.sin(omega0 * dt)
        F = rho * np.array([[cos_w, -sin_w], [sin_w, cos_w]], dtype=np.float64)
        C = np.array([[1.0, 0.0]], dtype=np.float64)
        Q = qx * np.eye(2, dtype=np.float64)
        R = np.array([[rv]], dtype=np.float64)

        x_filt = np.zeros((n, 2), dtype=np.float64)
        P_filt = np.zeros((n, 2, 2), dtype=np.float64)
        x_pred = np.zeros((n, 2), dtype=np.float64)
        P_pred = np.zeros((n, 2, 2), dtype=np.float64)

        x = np.zeros(2, dtype=np.float64)
        # Somata-style initial covariance: start from the state noise level.
        P = Q.copy()

        I = np.eye(2, dtype=np.float64)
        for t in range(n):
            # Prediction
            x = F @ x
            P = F @ P @ F.T + Q
            x_pred[t] = x
            P_pred[t] = P

            # Update (standard linear Kalman filter for 1D observation)
            y_t = y[t]
            S = float(C @ P @ C.T + R)
            if S <= 1e-12 or not np.isfinite(S):
                S = 1e-12
            K = (P @ C.T) / S
            innovation = float(y_t - (C @ x)[0])
            x = x + (K[:, 0] * innovation)
            P = (I - K @ C) @ P
            P = 0.5 * (P + P.T)
            for i in range(2):
                if P[i, i] < 1e-10 or not np.isfinite(P[i, i]):
                    P[i, i] = 1e-10

            x_filt[t] = x
            P_filt[t] = P

        x_smooth = np.copy(x_filt)
        P_smooth = np.copy(P_filt)
        F_T = F.T
        for t in range(n - 2, -1, -1):
            try:
                P_inv = np.linalg.pinv(P_pred[t + 1])
            except np.linalg.LinAlgError:
                P_inv = np.linalg.inv(P_pred[t + 1] + 1e-9 * np.eye(2))
            G = P_filt[t] @ F_T @ P_inv
            x_smooth[t] += G @ (x_smooth[t + 1] - x_pred[t + 1])
            P_smooth[t] += G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T

        # Derive amplitude/phase and instantaneous frequency from smoothed state.
        x1 = x_smooth[:, 0]
        x2 = x_smooth[:, 1]
        if n > 1:
            phase = np.unwrap(np.arctan2(x2, x1))
            dphi = np.diff(phase)
            inst_freq = (fs / (2.0 * np.pi)) * dphi
            track_hz = np.empty(n, dtype=np.float64)
            if inst_freq.size:
                track_hz[0] = inst_freq[0]
                track_hz[1:] = inst_freq
            else:
                track_hz[:] = freq0
        else:
            track_hz = np.full(n, freq0, dtype=np.float64)

        # Constrain and stabilise the frequency track within the respiratory band.
        track_hz = np.asarray(track_hz, dtype=np.float64)
        bad_mask = ~np.isfinite(track_hz)
        if np.any(bad_mask):
            track_hz[bad_mask] = freq0
        track_hz = np.clip(track_hz, p.f_min, p.f_max)
        track_hz = self._apply_post_smoothing(track_hz)

        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        meta_payload["freq_source"] = "kf_phase"
        meta_payload.setdefault("is_constant_track", False)
        return self._package(x1, track_hz, meta_payload)


class oscillator_UKF_freq(_BaseOscillatorHead):
    head_key = "ukffreq"

    def __init__(self, params: Optional[OscillatorParams] = None):
        super().__init__(params=params)
        p = self.params
        self.alpha = float(p.ukf_alpha if (p.ukf_alpha and p.ukf_alpha > 0) else 1e-3)
        self.beta = float(p.ukf_beta if p.ukf_beta else 2.0)
        self.kappa = float(p.ukf_kappa if p.ukf_kappa is not None else 0.0)

    def _sigma_points(self, x: np.ndarray, P: np.ndarray):
        n = x.size
        lam = self.alpha ** 2 * (n + self.kappa) - n
        c = max(n + lam, 1e-9)
        P = 0.5 * (P + P.T)
        jitter = 1e-9 * max(1.0, float(np.max(np.diag(P))))
        for _ in range(6):
            try:
                L = np.linalg.cholesky(P + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.clip(eigvals, 1e-10, None)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        sqrtP = np.sqrt(c) * L
        sigma = np.zeros((2 * n + 1, n), dtype=np.float64)
        sigma[0] = x
        for i in range(n):
            sigma[i + 1] = x + sqrtP[:, i]
            sigma[n + i + 1] = x - sqrtP[:, i]
        Wm = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
        Wc = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
        Wm[0] = lam / c
        Wc[0] = lam / c + (1.0 - self.alpha ** 2 + self.beta)
        return sigma, Wm, Wc

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        p = self.params
        fs = fs or p.fs
        self._maybe_apply_autotune(meta)
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        dt = 1.0 / fs
        freq0 = self._coarse_freq(y, fs)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))
        eff = self._effective_params(fs)
        rho = eff['rho']
        qx = eff['qx']
        rv = eff['rv']
        qf = max(eff['qf'], 1e-10)

        log_f_min = np.log(max(p.f_min, 1e-4))
        log_f_max = np.log(max(p.f_max, p.f_min + 1e-4))
        log_f0 = np.log(np.clip(freq0, p.f_min, p.f_max))

        def transition(x_state):
            x1, x2, log_f = x_state
            freq = np.clip(np.exp(log_f), p.f_min, p.f_max)
            omega = 2.0 * np.pi * freq
            cos_w = np.cos(omega * dt)
            sin_w = np.sin(omega * dt)
            rot = rho * np.array([[cos_w, -sin_w], [sin_w, cos_w]], dtype=np.float64)
            xy = rot @ np.array([x1, x2])
            new_state = np.array([xy[0], xy[1], log_f], dtype=np.float64)
            new_state[2] = np.clip(new_state[2], log_f_min, log_f_max)
            return new_state

        def observe(x_state):
            return np.array([x_state[0]], dtype=np.float64)

        x = np.array([0.0, 0.0, log_f0], dtype=np.float64)
        P = np.diag([1.0, 1.0, 0.25 ** 2]).astype(np.float64)
        Q = np.diag([qx, qx, qf]).astype(np.float64)
        R = np.array([[rv]], dtype=np.float64)

        states = np.zeros((n, 3), dtype=np.float64)
        I = np.eye(3, dtype=np.float64)
        for t in range(n):
            sigma, Wm, Wc = self._sigma_points(x, P)
            sigma_pred = np.array([transition(sp) for sp in sigma], dtype=np.float64)
            x_pred = np.sum(Wm[:, None] * sigma_pred, axis=0)
            P_pred = Q.copy()
            for i in range(sigma_pred.shape[0]):
                diff = sigma_pred[i] - x_pred
                P_pred += Wc[i] * np.outer(diff, diff)

            Z = np.array([observe(sp) for sp in sigma_pred], dtype=np.float64)
            z_pred = np.sum(Wm[:, None] * Z, axis=0)
            S = R.copy()
            Pxz = np.zeros((3, 1), dtype=np.float64)
            for i in range(Z.shape[0]):
                z_diff = Z[i] - z_pred
                x_diff = sigma_pred[i] - x_pred
                S += Wc[i] * np.outer(z_diff, z_diff)
                Pxz += Wc[i] * np.outer(x_diff, z_diff)
            if S.shape == (1, 1):
                S[0, 0] = float(max(S[0, 0], 1e-12))
            K = Pxz @ np.linalg.pinv(S)
            innovation = y[t] - z_pred[0]
            x = x_pred + (K[:, 0] * innovation)
            P = (I - K @ np.array([[1.0, 0.0, 0.0]])) @ P_pred @ (I - K @ np.array([[1.0, 0.0, 0.0]])).T + K @ R @ K.T
            P = 0.5 * (P + P.T)
            for i in range(3):
                if P[i, i] < 1e-8:
                    P[i, i] = 1e-8
            x[2] = float(np.clip(x[2], log_f_min, log_f_max))

            states[t] = x

        track_hz = np.clip(np.exp(states[:, 2]), p.f_min, p.f_max)
        track_hz = self._apply_post_smoothing(track_hz)
        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        qf_eff = float(qf)
        meta_payload["qf_eff"] = qf_eff
        try:
            finite_track = track_hz[np.isfinite(track_hz)]
            freq_med = float(np.nanmedian(finite_track)) if finite_track.size else float(freq0)
            sigma_bpm_30s = 60.0 * freq_med * float(np.sqrt(max(qf_eff, 0.0) * fs * 30.0))
            if np.isfinite(sigma_bpm_30s):
                meta_payload["qf_sigma_bpm_30s"] = sigma_bpm_30s
                meta_payload.setdefault("qf_note", "sigma_bpm_30s approximates 30s breathing drift")
        except Exception:
            pass
        meta_payload.setdefault("is_constant_track", False)
        return self._package(states[:, 0], track_hz, meta_payload)


class oscillator_Spec_ridge(_BaseOscillatorHead):
    head_key = "spec_ridge"

    def _track_ridge(self, freqs: np.ndarray, magnitudes: np.ndarray, penalty: float):
        n_freqs, n_times = magnitudes.shape
        if n_times == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int32)
        cost = magnitudes[:, 0].astype(np.float64)
        backptr = np.zeros((n_freqs, n_times), dtype=np.int32)
        freq_diff = (freqs[:, None] - freqs[None, :]) ** 2
        scale = penalty / max((freqs[-1] - freqs[0]) ** 2, 1e-6)
        for t in range(1, n_times):
            transition = cost[None, :] - scale * freq_diff
            idx = np.argmax(transition, axis=1)
            cost = transition[np.arange(n_freqs), idx] + magnitudes[:, t]
            backptr[:, t] = idx
        ridge = np.zeros(n_times, dtype=np.int32)
        ridge[-1] = int(np.argmax(cost))
        for t in range(n_times - 1, 0, -1):
            ridge[t - 1] = backptr[ridge[t], t]
        return freqs[ridge], ridge

    def _median_smooth(self, arr: np.ndarray, width: int) -> np.ndarray:
        width = max(1, int(width))
        if width % 2 == 0:
            width += 1
        if arr.size <= 1 or width <= 1:
            return arr
        pad = width // 2
        padded = np.pad(arr, (pad, pad), mode='edge')
        out = np.empty_like(arr, dtype=np.float64)
        for idx in range(arr.size):
            out[idx] = np.median(padded[idx:idx + width])
        return out

    def _subbin_refine(self, freqs: np.ndarray, magnitudes: np.ndarray, ridge_idx: np.ndarray, mode: str) -> np.ndarray:
        if mode not in ('parabolic',):
            return freqs[ridge_idx]
        if freqs.size < 3 or magnitudes.shape[0] < 3:
            return freqs[ridge_idx]
        refined = freqs[ridge_idx].astype(np.float64).copy()
        df = np.diff(freqs)
        df = np.append(df, df[-1])
        for t, idx in enumerate(ridge_idx):
            if idx <= 0 or idx >= freqs.size - 1:
                continue
            y1 = magnitudes[idx - 1, t]
            y2 = magnitudes[idx, t]
            y3 = magnitudes[idx + 1, t]
            denom = (y1 - 2.0 * y2 + y3)
            if not np.isfinite(denom) or abs(denom) < 1e-12:
                continue
            shift = 0.5 * (y1 - y3) / denom
            refined[t] = freqs[idx] + shift * df[idx]
        return refined

    def _multi_resolution_tracks(self, signal: np.ndarray, fs: float) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        p = self.params
        base_win = max(16, int(p.stft_win * fs))
        scales = [0.75, 1.0, 1.5]
        results = []
        for scale in scales:
            win = max(16, int(base_win * scale))
            overlap_frac = float(getattr(p, "spec_overlap", 0.5))
            overlap_frac = float(np.clip(overlap_frac, 0.0, 0.95))
            hop = max(1, int(round(win * (1.0 - overlap_frac))))
            hop = min(hop, win - 1)
            nfft_factor = max(1, int(getattr(p, "spec_nfft_factor", 1)))
            nfft = max(win, int(win * nfft_factor))
            freqs, times, Zxx = sps.stft(
                signal,
                fs=fs,
                nperseg=win,
                noverlap=win - hop,
                detrend=False,
                padded=False,
                nfft=nfft
            )
            results.append((freqs, times, Zxx))
        return results

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        p = self.params
        fs = fs or p.fs
        self._maybe_apply_autotune(meta)
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        multi_tracks = []
        reliabilities = []
        sample_times = self._timebase(n, fs)
        for freqs, times, Zxx in self._multi_resolution_tracks(y, fs):
            mask = (freqs >= p.f_min) & (freqs <= p.f_max)
            if not np.any(mask):
                continue
            sub_freqs = freqs[mask]
            magnitudes = np.abs(Zxx[mask, :]).astype(np.float64)
            ridge_freqs, ridge_idx = self._track_ridge(sub_freqs, magnitudes, p.ridge_penalty)
            subbin_mode = (getattr(p, "spec_subbin_interp", "parabolic") or "").strip().lower()
            if ridge_idx.size and subbin_mode in ('parabolic',):
                ridge_freqs = self._subbin_refine(sub_freqs, magnitudes, ridge_idx, subbin_mode)
            if ridge_freqs.size >= 3:
                time_step = times[1] - times[0] if times.size > 1 else p.stft_hop
                kernel = max(3, int(round(1.0 / max(time_step, 1e-6))))
                ridge_freqs = self._median_smooth(ridge_freqs, kernel)
            if times.size < 2:
                track = np.full(n, np.clip(ridge_freqs[0], p.f_min, p.f_max), dtype=np.float64)
            else:
                track = np.interp(sample_times, times, ridge_freqs, left=ridge_freqs[0], right=ridge_freqs[-1])
                track = np.clip(track, p.f_min, p.f_max)
            if track.size >= 3:
                track = self._median_smooth(track, 5)
            peak_smooth = max(1, int(getattr(p, "spec_peak_smooth_len", 1)))
            if peak_smooth > 1:
                track = self._median_smooth(track, peak_smooth)
            track = self._apply_post_smoothing(track)
            consistency = float(np.std(np.diff(track))) if track.size > 2 else 0.0
            power_ratio = float(np.max(magnitudes) / max(np.median(magnitudes), 1e-9))
            reliability = float(np.exp(-consistency * 5.0) * np.tanh(power_ratio / 5.0))
            multi_tracks.append(track)
            reliabilities.append(reliability)

        if not multi_tracks:
            track = np.full(n, 0.5 * (p.f_min + p.f_max), dtype=np.float64)
            meta_payload = dict(meta or {})
            meta_payload.setdefault("is_constant_track", False)
            return self._package(y, track, meta_payload)

        weights = np.asarray(reliabilities, dtype=np.float64)
        weights = weights / max(np.sum(weights), 1e-6)
        stacked = np.vstack(multi_tracks)
        track = np.average(stacked, axis=0, weights=weights)
        track = np.clip(track, p.f_min, p.f_max)
        meta_payload = dict(meta or {})
        meta_payload["multi_resolution_used"] = True
        meta_payload["reliability_score"] = float(np.max(reliabilities))
        meta_payload.setdefault("is_constant_track", False)
        return self._package(y, track, meta_payload)


class oscillator_PLL(_BaseOscillatorHead):
    head_key = "pll"

    def _estimate_pll_snr(self, analytic: np.ndarray) -> float:
        """Robust SNR estimate based on analytic envelope dispersion."""
        if analytic.size == 0:
            return 0.0
        amp = np.abs(analytic)
        finite = amp[np.isfinite(amp)]
        if finite.size == 0:
            return 0.0
        p75 = float(np.percentile(finite, 75))
        p25 = float(np.percentile(finite, 25))
        noise = max(p25, 1e-6)
        snr_lin = (max(p75, noise) ** 2) / (noise ** 2)
        return float(np.clip(np.log10(snr_lin + 1e-9) * 10.0, 0.0, 30.0))

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        p = self.params
        fs = fs or p.fs
        self._maybe_apply_autotune(meta)
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        dt = 1.0 / fs
        analytic = hilbert(y)
        freq0 = self._coarse_freq(y, fs)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))
        omega = 2.0 * np.pi * freq0
        phase_nco = float(np.angle(analytic[0]))
        integrator = 0.0

        pll_snr = self._estimate_pll_snr(analytic)
        if p.pll_autogain or p.pll_kp <= 0 or p.pll_ki <= 0:
            controller = PLLAdaptiveController(p.pll_zeta, p.pll_ttrack, fs)
            gain_snr = pll_snr if np.isfinite(pll_snr) else (self._last_snr or 0.0)
            kp, ki = controller.gains(gain_snr, p.pll_kp_min, p.pll_ki_min)
        else:
            kp = max(p.pll_kp, float(p.pll_kp_min))
            ki = max(p.pll_ki, float(p.pll_ki_min))

        track = np.zeros(n, dtype=np.float64)
        osc = np.zeros(n, dtype=np.float64)
        omega_min = 2.0 * np.pi * p.f_min
        omega_max = 2.0 * np.pi * p.f_max
        phase_noise = 0.0

        for t in range(n):
            phase_y = float(np.angle(analytic[t]))
            err_raw = np.arctan2(np.sin(phase_y - phase_nco), np.cos(phase_y - phase_nco))
            # simple phase-noise shaping
            err = err_raw - 0.1 * phase_noise
            phase_noise = err_raw
            integrator_candidate = integrator + ki * err
            omega_candidate = omega + kp * err + integrator_candidate
            omega_clamped = float(np.clip(omega_candidate, omega_min, omega_max))
            if omega_clamped != omega_candidate:
                integrator_candidate += omega_clamped - omega_candidate
            integrator = np.clip(integrator_candidate, -np.pi, np.pi)
            omega = omega_clamped
            phase_nco = ((phase_nco + omega * dt + np.pi) % (2.0 * np.pi)) - np.pi
            track[t] = omega / (2.0 * np.pi)
            osc[t] = np.cos(phase_nco)

        track = np.clip(track, p.f_min, p.f_max)
        track = self._apply_post_smoothing(track)
        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        meta_payload["pll_snr_db"] = float(pll_snr)
        meta_payload.setdefault("is_constant_track", False)
        consistency = float(np.std(np.diff(track))) if track.size > 2 else 0.0
        reliability = float(np.exp(-consistency * 10.0))
        meta_payload["reliability_score"] = reliability
        return self._package(osc, track, meta_payload)


HEAD_REGISTRY = {
    "kfstd": oscillator_KFstd,
    "kf_std": oscillator_KFstd,
    "ukffreq": oscillator_UKF_freq,
    "ukf_freq": oscillator_UKF_freq,
    "spec_ridge": oscillator_Spec_ridge,
    "specridge": oscillator_Spec_ridge,
    "pll": oscillator_PLL,
}


def build_head(head_key: str, params: Optional[OscillatorParams] = None):
    key = head_key.lower()
    if key not in HEAD_REGISTRY:
        raise ValueError(f"Unknown oscillator head '{head_key}'")
    cls = HEAD_REGISTRY[key]
    return cls(params=params)
