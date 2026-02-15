"""
Online Failure Mode Monitor.

Spec §8: Auto-labels each frame with failure mode flags.
Modes: divergence, phase slip, locking, doubling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FailureConfig:
    """Thresholds for failure mode detection."""
    # Divergence: trace(P) > τ_div or NIS > χ²_thresh for ≥ div_window frames
    tau_div: float = 100.0
    nis_chi2_thresh: float = 10.83   # χ²(1, 0.999)
    div_window: int = 5

    # Phase slip: |Δφ| > slip_thresh in single step
    slip_thresh: float = np.pi / 2   # radians

    # Locking: σ_freq < ε_lock for ≥ lock_window frames
    epsilon_lock: float = 0.005      # Hz
    lock_window: int = 30

    # Doubling: |f_track − 2·f_ref| < δ or |f_track − f_ref/2| < δ
    doubling_delta: float = 0.03     # Hz


@dataclass
class FailureFlags:
    """Per-frame failure mode labels."""
    diverge: bool = False
    phase_slip: bool = False
    locking: bool = False
    doubling: bool = False

    def any_active(self) -> bool:
        return self.diverge or self.phase_slip or self.locking or self.doubling

    def to_dict(self) -> Dict[str, bool]:
        return {
            'fail_diverge': self.diverge,
            'fail_slip': self.phase_slip,
            'fail_lock': self.locking,
            'fail_double': self.doubling,
        }


class FailureMonitor:
    """Online failure mode detection.

    Operates on decoded filter state per frame. Stateful:
    tracks running windows for divergence and locking detection.
    """

    def __init__(self, cfg: Optional[FailureConfig] = None,
                 f_ref: Optional[float] = None):
        """
        Args:
            cfg: detection thresholds
            f_ref: reference frequency (Hz) for doubling detection.
                   If None, uses the initial frequency estimate.
        """
        self.cfg = cfg or FailureConfig()
        self.f_ref = f_ref

        # Running state for window-based detection
        self._nis_exceed_count = 0
        self._trace_exceed_count = 0
        self._lock_count = 0
        self._prev_phase = None
        self._freq_history: List[float] = []

    def update(self, state: Dict[str, float],
               nis: float,
               trace_P: float) -> FailureFlags:
        """Check failure modes for current frame.

        Args:
            state: decoded state from StateDecoder.decode()
                   Must contain: freq_hz, phase_rad, freq_std_hz
            nis: normalized innovation squared
            trace_P: trace of covariance matrix

        Returns:
            FailureFlags for this frame.
        """
        c = self.cfg
        flags = FailureFlags()

        freq = state.get('freq_hz', 0.0)
        phase = state.get('phase_rad', 0.0)
        freq_std = state.get('freq_std_hz', 0.0)

        # Set reference frequency on first call if not provided
        if self.f_ref is None and freq > 0:
            self.f_ref = freq

        # ── Divergence ──
        # Trace explosion or persistent NIS overflow
        if trace_P > c.tau_div:
            self._trace_exceed_count += 1
        else:
            self._trace_exceed_count = max(0, self._trace_exceed_count - 1)

        if nis > c.nis_chi2_thresh:
            self._nis_exceed_count += 1
        else:
            self._nis_exceed_count = max(0, self._nis_exceed_count - 1)

        if (self._trace_exceed_count >= c.div_window or
                self._nis_exceed_count >= c.div_window):
            flags.diverge = True

        # ── Phase slip ──
        if self._prev_phase is not None:
            delta_phase = abs(_angle_diff(phase, self._prev_phase))
            if delta_phase > c.slip_thresh:
                flags.phase_slip = True
        self._prev_phase = phase

        # ── Locking ──
        # Frequency barely changing (low variance AND low actual change)
        self._freq_history.append(freq)
        if len(self._freq_history) > c.lock_window:
            self._freq_history = self._freq_history[-c.lock_window:]
        if len(self._freq_history) >= c.lock_window:
            freq_window = np.array(self._freq_history[-c.lock_window:])
            freq_sigma = float(np.std(freq_window))
            if freq_sigma < c.epsilon_lock:
                self._lock_count += 1
                if self._lock_count >= c.lock_window:
                    flags.locking = True
            else:
                self._lock_count = 0

        # ── Doubling ──
        if self.f_ref is not None and self.f_ref > 0:
            if abs(freq - 2.0 * self.f_ref) < c.doubling_delta:
                flags.doubling = True
            elif abs(freq - 0.5 * self.f_ref) < c.doubling_delta:
                flags.doubling = True

        return flags

    def reset(self, f_ref: Optional[float] = None):
        """Reset between trials."""
        self._nis_exceed_count = 0
        self._trace_exceed_count = 0
        self._lock_count = 0
        self._prev_phase = None
        self._freq_history = []
        if f_ref is not None:
            self.f_ref = f_ref
        else:
            self.f_ref = None


def _angle_diff(a: float, b: float) -> float:
    """Signed angular difference in [-π, π]."""
    d = a - b
    return float((d + np.pi) % (2 * np.pi) - np.pi)
