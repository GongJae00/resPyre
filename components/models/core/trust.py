"""
Trust Allocation Module.

Spec §D: Maps 6D quality vector → trust parameters for robust update.
All rules are deterministic. No attention mechanism.

Quality vector (6D MVP):
    q_vis, q_drift, q_cons, q_out, q_harm, q_burst
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TrustParams:
    """Trust parameters produced per frame."""
    alpha_R: float = 1.0     # R scaling (≥1 → distrust observation)
    alpha_Q: float = 1.0     # Q scaling (≥1 → allow more process drift)
    g_t: float = 1.0         # observation update gate [0, 1]
    g_z: float = 1.0         # frequency update gate [0, 1]
    w_h: float = 1.0         # harmonic suppression weight [0, 1]


@dataclass
class TrustConfig:
    """Tuneable hyper-parameters for trust wiring rules."""
    # R-scale: α_R = 1 + β₁·q_out + β₂·(1−q_vis)
    beta_1: float = 2.0       # outlier sensitivity
    beta_2: float = 1.5       # visibility sensitivity

    # Q-scale: α_Q = 1 + γ₁·q_drift
    gamma_1: float = 1.0      # drift sensitivity

    # Observation gate: σ(w₁·q_vis + w₂·q_cons − w₃·NIS − b)
    w_gate_vis: float = 2.0
    w_gate_cons: float = 1.5
    w_gate_nis: float = 0.5
    gate_bias: float = 1.0

    # Frequency gate
    freq_jitter_decay: float = 0.8   # how much jitter reduces g_z

    # Harmonic suppression: w_h = 1 − clip(q_harm/thd_max, 0, 1)
    thd_max: float = 0.3

    # NIS hard gate threshold
    nis_hard_gate: float = 15.0

    # Clamps
    alpha_R_max: float = 20.0
    alpha_Q_max: float = 5.0


class TrustAllocator:
    """Deterministic trust allocation from 6D quality vector.

    Maps quality metrics to filter control params via closed-form rules.
    No learnable parameters — ensures interpretability for Scientific Reports.
    """

    def __init__(self, cfg: Optional[TrustConfig] = None):
        self.cfg = cfg or TrustConfig()
        self._prev_freq = None  # for freq jitter detection

    def allocate(self, quality: Dict[str, float],
                 nis: float = 0.0,
                 current_freq: Optional[float] = None) -> TrustParams:
        """Compute trust parameters from quality vector.

        Args:
            quality: dict with keys {q_vis, q_drift, q_cons, q_out, q_harm, q_burst}
                     Values in [0, 1] or raw scale depending on metric.
            nis: Normalized Innovation Squared from previous update.
            current_freq: current frequency estimate (Hz) for jitter detection.

        Returns:
            TrustParams with α_R, α_Q, g_t, g_z, w_h
        """
        c = self.cfg

        q_vis = float(quality.get('q_vis', 1.0))
        q_drift = float(quality.get('q_drift', 0.0))
        q_cons = float(quality.get('q_cons', 1.0))
        q_out = float(quality.get('q_out', 0.0))
        q_harm = float(quality.get('q_harm', 0.0))
        q_burst = float(quality.get('q_burst', 0.0))

        # ── Rule 1: R-scale ──
        # Higher outlier score or lower visibility → inflate R → distrust y_t
        alpha_R = 1.0 + c.beta_1 * q_out + c.beta_2 * max(1.0 - q_vis, 0.0)
        # Burst events also inflate R
        alpha_R += 3.0 * q_burst
        alpha_R = float(np.clip(alpha_R, 1.0, c.alpha_R_max))

        # ── Rule 2: Q-scale ──
        # High drift → allow more state flexibility
        alpha_Q = 1.0 + c.gamma_1 * q_drift
        alpha_Q = float(np.clip(alpha_Q, 1.0, c.alpha_Q_max))

        # ── Rule 3: Observation gate ──
        # g_t = σ(w₁·q_vis + w₂·q_cons − w₃·NIS − b)
        logit = (c.w_gate_vis * q_vis +
                 c.w_gate_cons * q_cons -
                 c.w_gate_nis * nis -
                 c.gate_bias)
        g_t = float(_sigmoid(logit))

        # Hard NIS gate
        if nis > c.nis_hard_gate:
            g_t = 0.0

        # ── Rule 4: Frequency gate ──
        # g_z = g_t · (1 − freq_jitter_ratio)
        freq_jitter = 0.0
        if current_freq is not None and self._prev_freq is not None:
            # jitter = |Δf| / f  — relative frequency change
            delta_f = abs(current_freq - self._prev_freq)
            freq_jitter = delta_f / max(abs(self._prev_freq), 1e-6)
            freq_jitter = min(freq_jitter, 1.0)
        g_z = g_t * max(1.0 - c.freq_jitter_decay * freq_jitter, 0.0)

        if current_freq is not None:
            self._prev_freq = current_freq

        # ── Rule 5: Harmonic suppression ──
        # w_h = 1 − clip(q_harm / thd_max, 0, 1)
        w_h = 1.0 - float(np.clip(q_harm / max(c.thd_max, 1e-6), 0.0, 1.0))

        return TrustParams(
            alpha_R=alpha_R,
            alpha_Q=alpha_Q,
            g_t=g_t,
            g_z=g_z,
            w_h=w_h,
        )

    def reset(self):
        """Reset internal state between trials."""
        self._prev_freq = None


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = np.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = np.exp(x)
        return z / (1.0 + z)


def default_quality() -> Dict[str, float]:
    """Return a neutral quality vector (fully trusted observation)."""
    return {
        'q_vis': 1.0,
        'q_drift': 0.0,
        'q_cons': 1.0,
        'q_out': 0.0,
        'q_harm': 0.0,
        'q_burst': 0.0,
    }
