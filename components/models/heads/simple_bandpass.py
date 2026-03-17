"""
Simple Bandpass + Welch Spectral Peak Baseline.

Signal-processing-only baseline: no Kalman filter, no state estimation.
Pipeline per trial:
    1. Detrend + bandpass (0.08-0.50 Hz) + robust z-score
    2. Sliding-window Welch PSD, peak detection within [f_min, f_max]
    3. Constant-within-window frequency track output

This method documents the performance achievable with classical signal
processing alone, serving as an explicit lower-bound reference for
Kalman-filter-based heads (kfstd, QROBF).

Reference:
    Procházka et al. (2017), "Breathing Analysis Using Chest Movement
    and Resampling Processes", MDPI Sensors — standard BPF+Welch pipeline.
"""

from typing import Dict, Optional
import numpy as np
from scipy import signal as sps

from ..core.base import _BaseOscillatorHead, OscillatorParams


class oscillator_SimpleBandpass(_BaseOscillatorHead):
    """Bandpass filter + Welch spectral peak (no Kalman filter).

    Frequency track is computed from a sliding window Welch PSD.
    Within each window, the frequency is held constant at the spectral peak.

    This is the standard signal-processing baseline referenced in respiratory
    rate estimation literature. It requires no filter tuning beyond the
    bandpass corners and window length.
    """

    head_key = "simple_bandpass"

    def __init__(self, params: Optional[OscillatorParams] = None):
        super().__init__(params=params)
        # Window length for Welch PSD (seconds); default 30 s matches eval window
        self._welch_win_sec = 30.0
        # Stride for sliding Welch (seconds); 1 s gives a per-second frequency estimate
        self._welch_stride_sec = 1.0

    def run(self, signal: np.ndarray, fs: float,
            meta: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Run the simple bandpass + spectral peak pipeline.

        Args:
            signal: raw 1D observation signal
            fs: sampling frequency (Hz)
            meta: optional metadata dict

        Returns:
            Standard head output dict with signal_hat, track_hz, etc.
        """
        p = self.params
        fs = float(fs or p.fs)

        # Preprocessing (detrend, bandpass, robust z-score — same as other heads)
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        # ── Sliding-window Welch peak detection ──
        win_len = max(int(round(self._welch_win_sec * fs)), 8)
        stride = max(int(round(self._welch_stride_sec * fs)), 1)

        # Compute global Welch estimate for coarse frequency (used for short signals)
        f_global = self._welch_peak(y, fs)

        freq_track = np.full(n, f_global, dtype=np.float64)

        # Slide over the signal in stride-sized steps
        half_win = win_len // 2
        for t in range(0, n, stride):
            t_start = max(0, t - half_win)
            t_end = min(n, t + half_win)
            seg = y[t_start:t_end]
            if seg.size >= 8:
                f_peak = self._welch_peak(seg, fs)
            else:
                f_peak = f_global
            # Fill this stride block with the window's peak frequency
            block_end = min(t + stride, n)
            freq_track[t:block_end] = f_peak

        # Forward fill any remaining frames (tail)
        if n > 0:
            last_valid = freq_track[np.flatnonzero(np.isfinite(freq_track))]
            if last_valid.size > 0:
                bad = ~np.isfinite(freq_track)
                freq_track[bad] = last_valid[-1]

        # Constrain to physiological band
        freq_track = np.clip(freq_track, p.f_min, p.f_max)

        meta_payload = dict(meta or {})
        meta_payload.update({
            "f0": f_global,
            "head": self.head_key,
            "freq_source": "welch_sliding",
        })
        meta_payload.setdefault("is_constant_track", False)

        # signal_hat: bandpass-filtered signal (already in y)
        return self._package(y, freq_track, meta_payload)

    def _welch_peak(self, seg: np.ndarray, fs: float) -> float:
        """Find dominant frequency in seg via Welch PSD within [f_min, f_max]."""
        p = self.params
        try:
            # nperseg: use full segment or 256, whichever is smaller
            nperseg = min(len(seg), max(int(fs * 15), 64))
            nperseg = max(nperseg, 8)
            freqs, psd = sps.welch(seg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        except Exception:
            return float(np.clip(p.f_min + (p.f_max - p.f_min) / 2, p.f_min, p.f_max))

        # Restrict to physiological band
        band_mask = (freqs >= p.f_min) & (freqs <= p.f_max)
        if not np.any(band_mask):
            return float(np.clip((p.f_min + p.f_max) / 2, p.f_min, p.f_max))

        band_psd = psd[band_mask]
        band_freq = freqs[band_mask]
        peak_idx = int(np.argmax(band_psd))

        # Sub-bin parabolic interpolation for better frequency resolution
        if 0 < peak_idx < len(band_psd) - 1:
            alpha = band_psd[peak_idx - 1]
            beta = band_psd[peak_idx]
            gamma = band_psd[peak_idx + 1]
            denom = alpha - 2.0 * beta + gamma
            if abs(denom) > 1e-12:
                delta = 0.5 * (alpha - gamma) / denom
                delta = float(np.clip(delta, -1.0, 1.0))
                df = band_freq[1] - band_freq[0] if len(band_freq) > 1 else 0.0
                f_peak = float(band_freq[peak_idx]) + delta * float(df)
            else:
                f_peak = float(band_freq[peak_idx])
        else:
            f_peak = float(band_freq[peak_idx])

        return float(np.clip(f_peak, p.f_min, p.f_max))
