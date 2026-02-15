"""
6D Quality Estimation Module (Phase 1).

Computes a per-frame quality vector used by TrustAllocator (§D):
    q_vis   – ROI visibility / brightness ratio
    q_drift – ROI center displacement rate
    q_cons  – sub-ROI cross-correlation consistency
    q_out   – Hampel outlier score (observation-level)
    q_harm  – total harmonic distortion (THD) from Welch
    q_burst – impulse/burst binary flag

All computations are NumPy/SciPy only — no external dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class QualityConfig:
    """Tuneable thresholds for quality estimation."""
    # q_vis
    vis_eps: float = 1e-6             # denominator guard

    # q_drift
    drift_scale: float = 5.0          # pixels/s → [0,1] mapping scale

    # q_cons
    cons_window: int = 30             # frames for sub-ROI cross-corr

    # q_out (Hampel)
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
        self._roi_means: list = []
        self._global_means: list = []

        # Welch cache
        self._welch_win = int(self.cfg.harm_window_sec * fs)

    def reset(self):
        """Clear all running buffers."""
        self._y_buf.clear()
        self._roi_centers.clear()
        self._roi_means.clear()
        self._global_means.clear()

    def update(self, t: int, y_t: float,
               roi_stats: Optional[Dict[str, float]] = None) -> Dict[str, float]:
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
        roi = roi_stats or {}

        # ── q_vis: ROI brightness ratio ──
        roi_mean = roi.get('roi_mean', 1.0)
        global_mean = roi.get('global_mean', 1.0)
        q_vis = float(np.clip(
            roi_mean / (global_mean + self.cfg.vis_eps), 0.0, 1.0
        ))

        # ── q_drift: ROI center displacement rate ──
        cx = roi.get('roi_cx', 0.0)
        cy = roi.get('roi_cy', 0.0)
        self._roi_centers.append((cx, cy))
        if len(self._roi_centers) >= 2:
            dx = cx - self._roi_centers[-2][0]
            dy = cy - self._roi_centers[-2][1]
            drift_pps = np.sqrt(dx**2 + dy**2) * self.fs  # pixels per second
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
        """Hampel outlier score: |y_t - median| / (k * MAD).
        
        Returns: score ∈ [0, ∞). Values > threshold indicate outlier.
        For trust rules, raw score is returned (not clipped).
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
        return float(score)

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
