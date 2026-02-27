"""
6D Quality Estimation Module (Phase 1).

Computes a per-frame quality vector used by TrustAllocator (§D):
    q_vis   – ROI visibility / brightness ratio
    q_drift – ROI center displacement rate
    q_cons  – sub-ROI cross-correlation consistency
    q_out   – Hampel outlier score normalized to [0, 1]
    q_harm  – total harmonic distortion (THD) from Welch
    q_burst – impulse/burst binary flag

All computations are NumPy/SciPy only — no external dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
import numpy as np

ROI_SCALAR_DEFAULTS: Dict[str, float] = {
    'roi_mean': 1.0,
    'global_mean': 1.0,
    'roi_std': 0.0,
    'roi_snr_db': 0.0,
    'valid_ratio': 1.0,
    'roi_cx': 0.5,
    'roi_cy': 0.5,
    'center_disp': 0.0,
}


def _is_sequence_like(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    return isinstance(value, (list, tuple, np.ndarray))


def _to_float_scalar(value: Any, key: str, t: int, *, allow_index: bool = True) -> float:
    """Convert value to scalar float.

    If a sequence is provided where a scalar is expected, index by frame `t`
    when possible; otherwise raise a clear ValueError.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.size == 1:
            return float(value.reshape(()).item())
        if allow_index and t < value.shape[0]:
            return _to_float_scalar(value[t], key, t, allow_index=False)
        raise ValueError(
            f"roi_stats key '{key}' at frame {t} expected scalar; got ndarray shape={value.shape}"
        )
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _to_float_scalar(value[0], key, t, allow_index=False)
        if allow_index and t < len(value):
            return _to_float_scalar(value[t], key, t, allow_index=False)
        raise ValueError(
            f"roi_stats key '{key}' at frame {t} expected scalar; got sequence length={len(value)}"
        )
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(
            f"roi_stats key '{key}' at frame {t} is not float-coercible: {type(value).__name__}"
        ) from exc


def _coerce_frame_mapping(mapping: Mapping[str, Any], t: int) -> Dict[str, float]:
    frame: Dict[str, float] = {}
    for key, value in mapping.items():
        frame[key] = _to_float_scalar(value, key, t, allow_index=True)
    return frame


def normalize_roi_stats_t(roi_stats_t: Any, T: int) -> List[Dict[str, float]]:
    """Normalize ROI metadata into a per-frame list of scalar dicts.

    Supported inputs:
      A) None -> [{}] * T
      B) list/tuple of dicts -> pad/trim to T
      C) dict of arrays/lists/scalars -> converted to list-of-dicts by frame index
    """
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if roi_stats_t is None:
        return [{} for _ in range(T)]

    out: List[Dict[str, float]] = []

    if isinstance(roi_stats_t, (list, tuple)):
        for t in range(T):
            if t >= len(roi_stats_t) or roi_stats_t[t] is None:
                out.append({})
                continue
            frame_raw = roi_stats_t[t]
            if not isinstance(frame_raw, Mapping):
                raise ValueError(
                    f"roi_stats_t[{t}] must be mapping/dict, got {type(frame_raw).__name__}"
                )
            out.append(_coerce_frame_mapping(frame_raw, t))
        return out

    if isinstance(roi_stats_t, Mapping):
        has_sequence = any(_is_sequence_like(v) for v in roi_stats_t.values())
        for t in range(T):
            if has_sequence:
                frame = {k: _to_float_scalar(v, k, t, allow_index=True) for k, v in roi_stats_t.items()}
            else:
                frame = _coerce_frame_mapping(roi_stats_t, t)
            out.append(frame)
        return out

    raise ValueError(
        "roi_stats_t must be None, list/tuple of dicts, or dict-of-arrays/scalars; "
        f"got {type(roi_stats_t).__name__}"
    )


@dataclass
class QualityConfig:
    """Tuneable thresholds for quality estimation."""
    # q_vis
    vis_eps: float = 1e-6             # denominator guard
    vis_snr_low_db: float = -5.0      # SNR lower bound mapped to 0
    vis_snr_high_db: float = 5.0      # SNR upper bound mapped to 1
    vis_blend_contrast: float = 0.20  # blend weight for contrast cue
    vis_blend_snr: float = 0.50       # blend weight for SNR cue
    vis_blend_valid: float = 0.30     # blend weight for valid-ratio cue

    # q_drift
    drift_scale: float = 0.25         # normalized center-units/s → [0,1] mapping scale

    # q_cons
    cons_window: int = 30             # frames for sub-ROI cross-corr

    # q_out (Hampel, normalized)
    hampel_k: float = 1.4826         # MAD → σ scaling constant
    hampel_thresh: float = 3.0       # threshold in σ units

    # q_harm
    harm_window_sec: float = 4.0     # Welch window length in seconds
    harm_harmonics: int = 2          # number of harmonics to sum (2f, 3f)

    # q_burst
    burst_sigma: float = 3.0         # threshold for burst detection
    burst_window: int = 15           # local σ estimation window (frames)


def default_quality() -> Dict[str, float]:
    """Neutral quality vector — all dimensions at 'trustworthy' values.
    
    This is used when quality estimation is not available (placeholder mode).
    """
    return {
        'q_vis': 1.0,    # fully visible
        'q_drift': 0.0,  # no drift
        'q_cons': 1.0,   # fully consistent
        'q_out': 0.0,    # no outlier
        'q_harm': 0.0,   # no harmonic distortion
        'q_burst': 0.0,  # no burst
    }


class QualityEstimator:
    """Online 6D quality estimation from observation signal + ROI metadata.

    Usage:
        qe = QualityEstimator(fs=30.0)
        for t in range(T):
            q = qe.update(t, y[t], roi_stats=roi_meta_t)
            # q is a dict with keys: q_vis, q_drift, q_cons, q_out, q_harm, q_burst
    """

    def __init__(self, fs: float = 30.0,
                 cfg: Optional[QualityConfig] = None,
                 f_min: float = 0.08, f_max: float = 0.5):
        self.fs = fs
        self.cfg = cfg or QualityConfig()
        self.f_min = f_min
        self.f_max = f_max

        # Running buffers
        self._y_buf: list = []
        self._roi_centers: list = []

        # Welch cache
        self._welch_win = int(self.cfg.harm_window_sec * fs)

    def reset(self):
        """Clear all running buffers."""
        self._y_buf.clear()
        self._roi_centers.clear()

    def _roi_stats_for_frame(self, roi_stats: Any, t: int) -> Dict[str, float]:
        """Select/normalize ROI stats for frame `t`.

        Accepts per-frame dict, list-of-dicts, or dict-of-arrays.
        """
        if roi_stats is None:
            return {}
        if isinstance(roi_stats, Mapping):
            frame = {}
            for key, value in roi_stats.items():
                if _is_sequence_like(value):
                    frame[key] = _to_float_scalar(value, key, t, allow_index=True)
                else:
                    frame[key] = _to_float_scalar(value, key, t, allow_index=False)
            return frame
        if isinstance(roi_stats, (list, tuple)):
            if t >= len(roi_stats) or roi_stats[t] is None:
                return {}
            if not isinstance(roi_stats[t], Mapping):
                raise ValueError(
                    f"roi_stats[{t}] must be mapping/dict, got {type(roi_stats[t]).__name__}"
                )
            return _coerce_frame_mapping(roi_stats[t], t)
        raise ValueError(
            f"roi_stats must be mapping/list/tuple/None, got {type(roi_stats).__name__}"
        )

    def update(self, t: int, y_t: float,
               roi_stats: Any = None) -> Dict[str, float]:
        """Compute 6D quality vector for frame t.

        Args:
            t: frame index
            y_t: current observation value
            roi_stats: optional dict with keys:
                'roi_mean': mean intensity of ROI
                'global_mean': mean intensity of full frame
                'roi_cx', 'roi_cy': ROI center coordinates

        Returns:
            dict with keys q_vis, q_drift, q_cons, q_out, q_harm, q_burst
        """
        self._y_buf.append(float(y_t))
        roi = self._roi_stats_for_frame(roi_stats, t)

        # ── q_vis: ROI brightness ratio + SNR + Validity ──
        roi_mean = float(roi.get('roi_mean', ROI_SCALAR_DEFAULTS['roi_mean']))
        global_mean = float(roi.get('global_mean', ROI_SCALAR_DEFAULTS['global_mean']))
        if not np.isfinite(roi_mean):
            roi_mean = ROI_SCALAR_DEFAULTS['roi_mean']
        if not np.isfinite(global_mean) or abs(global_mean) < self.cfg.vis_eps:
            global_mean = max(abs(roi_mean), ROI_SCALAR_DEFAULTS['global_mean'])
        
        # 1. Contrast-based visibility (Legacy)
        q_contrast = float(np.clip(roi_mean / (global_mean + self.cfg.vis_eps), 0.0, 1.0))

        # 2. SNR-based visibility (calibrated)
        q_snr = 1.0
        if 'roi_snr_db' in roi:
            snr = float(roi.get('roi_snr_db', ROI_SCALAR_DEFAULTS['roi_snr_db']))
            if np.isfinite(snr):
                lo = float(self.cfg.vis_snr_low_db)
                hi = float(max(self.cfg.vis_snr_high_db, lo + 1e-6))
                q_snr = float(np.clip((snr - lo) / (hi - lo), 0.0, 1.0))

        # 3. Validity ratio
        q_valid = 1.0
        if 'valid_ratio' in roi:
            valid = float(roi.get('valid_ratio', ROI_SCALAR_DEFAULTS['valid_ratio']))
            if np.isfinite(valid):
                q_valid = float(np.clip(valid, 0.0, 1.0))

        # Conservative blend:
        #   - weighted blend avoids pathological collapse when one cue is noisy
        #   - guard term keeps low-SNR or low-valid frames penalized
        w_c = float(self.cfg.vis_blend_contrast)
        w_s = float(self.cfg.vis_blend_snr)
        w_v = float(self.cfg.vis_blend_valid)
        w_sum = max(w_c + w_s + w_v, 1e-9)
        q_blend = (w_c * q_contrast + w_s * q_snr + w_v * q_valid) / w_sum
        q_guard = min(q_snr, q_valid)
        q_score = 0.5 * q_blend + 0.5 * q_guard
        q_vis = float(np.clip(q_score, 0.0, 1.0)) if np.isfinite(q_score) else 1.0

        # ── q_drift: ROI center displacement rate ──
        cx = float(roi.get('roi_cx', ROI_SCALAR_DEFAULTS['roi_cx']))
        cy = float(roi.get('roi_cy', ROI_SCALAR_DEFAULTS['roi_cy']))
        if not np.isfinite(cx):
            cx = ROI_SCALAR_DEFAULTS['roi_cx']
        if not np.isfinite(cy):
            cy = ROI_SCALAR_DEFAULTS['roi_cy']
        self._roi_centers.append((cx, cy))
        center_disp = float(roi.get('center_disp', np.nan))
        if np.isfinite(center_disp):
            drift_pps = center_disp * self.fs
            q_drift = float(np.clip(drift_pps / self.cfg.drift_scale, 0.0, 1.0))
        elif len(self._roi_centers) >= 2:
            dx = cx - self._roi_centers[-2][0]
            dy = cy - self._roi_centers[-2][1]
            drift_pps = np.sqrt(dx**2 + dy**2) * self.fs
            q_drift = float(np.clip(drift_pps / self.cfg.drift_scale, 0.0, 1.0))
        else:
            q_drift = 0.0

        # ── q_cons: sub-ROI cross-correlation consistency ──
        # Approximated from observation buffer autocorrelation
        q_cons = self._compute_consistency()

        # ── q_out: Hampel outlier score ──
        q_out = self._compute_outlier(y_t)

        # ── q_harm: THD from recent spectral content ──
        q_harm = self._compute_thd()

        # ── q_burst: impulse/burst detection ──
        q_burst = self._compute_burst(y_t)

        return {
            'q_vis': q_vis,
            'q_drift': q_drift,
            'q_cons': q_cons,
            'q_out': q_out,
            'q_harm': q_harm,
            'q_burst': q_burst,
        }

    def _compute_consistency(self) -> float:
        """Cross-correlation consistency from observation buffer.
        
        Uses split-half correlation: split recent window into two halves
        and compute normalized cross-correlation. High correlation → consistent.
        """
        cfg = self.cfg
        n = len(self._y_buf)
        if n < cfg.cons_window:
            return 1.0  # not enough data → assume consistent

        window = np.array(self._y_buf[-cfg.cons_window:])
        half = cfg.cons_window // 2
        a, b = window[:half], window[half:2*half]
        # Normalize
        a_norm = a - np.mean(a)
        b_norm = b - np.mean(b)
        denom = np.sqrt(np.sum(a_norm**2) * np.sum(b_norm**2))
        if denom < 1e-12:
            return 1.0
        corr = float(np.sum(a_norm * b_norm) / denom)
        return float(np.clip(corr, 0.0, 1.0))

    def _compute_outlier(self, y_t: float) -> float:
        """Hampel outlier score normalized to [0, 1].

        raw_score = |y_t - median| / (k * MAD + eps)
        q_out = clip(raw_score / hampel_thresh, 0, 1)

        This keeps the trust wiring numerically stable while preserving
        monotonic outlier sensitivity.
        """
        cfg = self.cfg
        n = len(self._y_buf)
        if n < 5:
            return 0.0

        buf = np.array(self._y_buf[-min(n, 100):])  # last 100 frames
        med = np.median(buf)
        mad = np.median(np.abs(buf - med))
        sigma = cfg.hampel_k * mad + 1e-9
        score = abs(y_t - med) / sigma
        score = score / max(cfg.hampel_thresh, 1e-6)
        return float(np.clip(score, 0.0, 1.0))

    def _compute_thd(self) -> float:
        """THD from Welch PSD on recent observation buffer.
        
        THD = (P_2f + P_3f + ...) / P_f0
        """
        cfg = self.cfg
        n = len(self._y_buf)
        if n < self._welch_win:
            return 0.0

        from scipy.signal import welch
        window = np.array(self._y_buf[-self._welch_win:])
        freqs, psd = welch(window, fs=self.fs,
                           nperseg=min(len(window), 256),
                           noverlap=None)

        # Find fundamental in respiratory band
        mask = (freqs >= self.f_min) & (freqs <= self.f_max)
        if not np.any(mask):
            return 0.0

        psd_masked = psd[mask]
        freqs_masked = freqs[mask]
        f0_idx = np.argmax(psd_masked)
        f0 = freqs_masked[f0_idx]
        p0 = psd_masked[f0_idx]

        if p0 < 1e-12:
            return 0.0

        # Sum harmonic power
        p_harm = 0.0
        for h in range(2, 2 + cfg.harm_harmonics):
            fh = h * f0
            idx = np.argmin(np.abs(freqs - fh))
            if idx < len(psd):
                p_harm += psd[idx]

        thd = float(p_harm / p0)
        return float(np.clip(thd, 0.0, 1.0))

    def _compute_burst(self, y_t: float) -> float:
        """Burst/impulse detection: |y_t| > k * σ_local."""
        cfg = self.cfg
        n = len(self._y_buf)
        if n < cfg.burst_window:
            return 0.0

        local = np.array(self._y_buf[-cfg.burst_window:])
        sigma = np.std(local) + 1e-9
        return 1.0 if abs(y_t - np.mean(local)) > cfg.burst_sigma * sigma else 0.0
