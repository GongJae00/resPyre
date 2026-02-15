"""
Oscillatory State-Space Model (SSM) — prediction module.

Spec §B: 3D state [x1, x2, z=log(f)] with rotation transition.
Supports both EKF (Jacobian) and UKF (sigma points) prediction.

This module handles ONLY prediction. Update is in robust_update.py.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class SSMConfig:
    """Configuration for the oscillatory SSM."""
    f_min: float = 0.08          # Hz, minimum respiratory frequency
    f_max: float = 0.50          # Hz, maximum respiratory frequency
    rho: float = 0.999           # damping factor exp(-dt/tau_env)
    qx: float = 1e-4             # process noise for oscillator states
    qf: float = 1e-6             # process noise for log-frequency
    rv_floor: float = 0.01       # base measurement noise R
    tau_env: float = 32.0        # envelope time constant (seconds)

    # UKF parameters
    ukf_alpha: float = 1e-3
    ukf_beta: float = 2.0
    ukf_kappa: float = 0.0

    def compute_rho(self, dt: float) -> float:
        """Damping factor per sample.

        If rho is explicitly set (0 < rho < 1), use it directly.
        Otherwise derive from tau_env: rho = exp(-dt/tau_env).
        """
        if 0.0 < self.rho < 1.0:
            return float(self.rho)
        return float(np.exp(-dt / max(self.tau_env, 1e-3)))

    @property
    def log_f_bounds(self):
        """Precomputed log-frequency clamp bounds."""
        return np.log(max(self.f_min, 1e-4)), np.log(max(self.f_max, self.f_min + 1e-4))


class StateDecoder:
    """Extracts interpretable quantities from raw state vector.

    State: x = [x1, x2, z] where z = log(f_hz).
    """

    @staticmethod
    def decode(x: np.ndarray, P: np.ndarray) -> Dict[str, float]:
        """Decode state vector into amplitude, phase, frequency + uncertainties.

        Returns:
            amp: envelope amplitude √(x1² + x2²)
            phase_rad: instantaneous phase atan2(x2, x1)
            freq_hz: instantaneous frequency exp(z)
            amp_std: approximate amplitude uncertainty
            freq_std_hz: approximate frequency uncertainty
        """
        x1, x2 = float(x[0]), float(x[1])
        z = float(x[2]) if x.size >= 3 else 0.0

        amp = np.sqrt(x1**2 + x2**2)
        phase = np.arctan2(x2, x1)
        freq = np.exp(z)

        # Propagate uncertainty (1st-order)
        # amp = sqrt(x1^2 + x2^2)  →  Jacobian ∂amp/∂x = [x1/amp, x2/amp, 0]
        if amp > 1e-9:
            J_amp = np.array([x1 / amp, x2 / amp, 0.0])
            amp_var = float(J_amp @ P @ J_amp)
        else:
            amp_var = float(P[0, 0])
        amp_std = np.sqrt(max(amp_var, 0.0))

        # freq = exp(z)  → Jacobian ∂freq/∂x = [0, 0, exp(z)]
        freq_var = float(freq**2 * P[2, 2]) if P.shape[0] >= 3 else 0.0
        freq_std = np.sqrt(max(freq_var, 0.0))

        return {
            'amp': float(amp),
            'phase_rad': float(phase),
            'freq_hz': float(freq),
            'amp_std': float(amp_std),
            'freq_std_hz': float(freq_std),
            'x1': x1,
            'x2': x2,
            'z': z,
        }


class OscillatorPredictor:
    """3D oscillatory SSM prediction step.

    State: x = [x1, x2, z]  where z = log(f_hz)
    Transition:
        [x1, x2]_{t+1} = ρ · R(θ_t) · [x1, x2]_t + w_x
        z_{t+1} = z_t + w_z
    where θ_t = 2π · exp(z_t) · dt
    Observation: y_t = H · x_t = x1_t
    """

    def __init__(self, cfg: Optional[SSMConfig] = None):
        self.cfg = cfg or SSMConfig()
        self.H = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)  # observe x1
        self.n_state = 3
        self.decoder = StateDecoder()

    def clamp_state(self, x: np.ndarray) -> np.ndarray:
        """Clamp log-frequency z to valid range [log(f_min), log(f_max)].

        Must be called after every state mutation (prediction, update)
        to prevent exp(z) explosion or collapse.
        """
        lo, hi = self.cfg.log_f_bounds
        x[2] = np.clip(x[2], lo, hi)
        return x

    def build_Q(self, qx: Optional[float] = None, qf: Optional[float] = None,
                alpha_Q: float = 1.0) -> np.ndarray:
        """Construct process noise matrix with optional trust scaling."""
        _qx = qx if qx is not None else self.cfg.qx
        _qf = qf if qf is not None else self.cfg.qf
        return np.diag([alpha_Q * _qx, alpha_Q * _qx, alpha_Q * _qf]).astype(np.float64)

    def build_R(self, rv: Optional[float] = None, alpha_R: float = 1.0) -> np.ndarray:
        """Construct measurement noise matrix with optional trust scaling."""
        _rv = rv if rv is not None else self.cfg.rv_floor
        return np.array([[alpha_R * _rv]], dtype=np.float64)

    def init_state(self, freq0: float) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize state and covariance from initial frequency estimate.

        Args:
            freq0: initial frequency estimate (Hz), typically from _coarse_freq().

        Returns:
            x0: initial state [0, 0, log(freq0)]
            P0: initial covariance (diagonal)
        """
        freq0 = float(np.clip(freq0, self.cfg.f_min, self.cfg.f_max))
        log_f0 = np.log(freq0)
        x0 = np.array([0.0, 0.0, log_f0], dtype=np.float64)
        # Generous initial uncertainty: oscillator amplitude unknown,
        # frequency roughly 1 octave uncertain.
        P0 = np.diag([1.0, 1.0, 0.25**2]).astype(np.float64)
        return x0, P0

    # ------------------------------------------------------------------
    # EKF Prediction
    # ------------------------------------------------------------------

    def predict_ekf(self, x: np.ndarray, P: np.ndarray,
                    Q: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """EKF prediction: linearized transition with Jacobian.

        Returns (x_pred, P_pred).
        """
        rho = self.cfg.compute_rho(dt)
        x1, x2, z = float(x[0]), float(x[1]), float(x[2])
        freq = np.exp(np.clip(z, np.log(self.cfg.f_min), np.log(self.cfg.f_max)))
        theta = 2.0 * np.pi * freq * dt

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Transition
        x1_new = rho * (cos_t * x1 - sin_t * x2)
        x2_new = rho * (sin_t * x1 + cos_t * x2)
        z_new = z  # random walk (mean = z)

        x_pred = np.array([x1_new, x2_new, z_new], dtype=np.float64)

        # Jacobian F = ∂f/∂x evaluated at (x1, x2, z)
        # ∂x1_new/∂x1 = ρ cos(θ),  ∂x1_new/∂x2 = -ρ sin(θ)
        # ∂x1_new/∂z  = ρ(-sin(θ)·x1 - cos(θ)·x2) · dθ/dz
        # where dθ/dz = 2π·dt·exp(z) = 2π·dt·freq = θ
        dtheta_dz = theta  # = 2π·freq·dt
        F = np.array([
            [rho * cos_t,  -rho * sin_t,
             rho * (-sin_t * x1 - cos_t * x2) * dtheta_dz],
            [rho * sin_t,   rho * cos_t,
             rho * (cos_t * x1 - sin_t * x2) * dtheta_dz],
            [0.0,           0.0,          1.0]
        ], dtype=np.float64)

        P_pred = F @ P @ F.T + Q
        P_pred = 0.5 * (P_pred + P_pred.T)

        return x_pred, P_pred

    # ------------------------------------------------------------------
    # UKF Prediction
    # ------------------------------------------------------------------

    def _sigma_points(self, x: np.ndarray, P: np.ndarray):
        """Compute sigma points and weights (Merwe's scaled form)."""
        n = x.size
        alpha = self.cfg.ukf_alpha
        beta = self.cfg.ukf_beta
        kappa = self.cfg.ukf_kappa
        lam = alpha**2 * (n + kappa) - n
        c = max(n + lam, 1e-9)

        # Robust Cholesky with progressive jitter
        P_sym = 0.5 * (P + P.T)
        jitter = 1e-9 * max(1.0, float(np.max(np.diag(P_sym))))
        L = None
        for _ in range(6):
            try:
                L = np.linalg.cholesky(P_sym + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        if L is None:
            eigvals, eigvecs = np.linalg.eigh(P_sym)
            eigvals = np.clip(eigvals, 1e-10, None)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        sqrtP = np.sqrt(c) * L
        sigma = np.zeros((2 * n + 1, n), dtype=np.float64)
        sigma[0] = x
        for i in range(n):
            sigma[i + 1] = x + sqrtP[:, i]
            sigma[n + i + 1] = x - sqrtP[:, i]

        Wm = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
        Wc = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
        Wm[0] = lam / c
        Wc[0] = lam / c + (1.0 - alpha**2 + beta)
        return sigma, Wm, Wc

    def _transition_fn(self, x_state: np.ndarray, dt: float) -> np.ndarray:
        """Nonlinear transition for a single sigma point."""
        rho = self.cfg.compute_rho(dt)
        x1, x2, z = x_state
        freq = np.clip(np.exp(z), self.cfg.f_min, self.cfg.f_max)
        theta = 2.0 * np.pi * freq * dt
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x1_new = rho * (cos_t * x1 - sin_t * x2)
        x2_new = rho * (sin_t * x1 + cos_t * x2)
        z_new = np.clip(z, np.log(self.cfg.f_min), np.log(self.cfg.f_max))
        return np.array([x1_new, x2_new, z_new], dtype=np.float64)

    def predict_ukf(self, x: np.ndarray, P: np.ndarray,
                    Q: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """UKF prediction via unscented transform.

        Returns (x_pred, P_pred).
        """
        sigma, Wm, Wc = self._sigma_points(x, P)
        sigma_pred = np.array([self._transition_fn(sp, dt) for sp in sigma],
                              dtype=np.float64)
        x_pred = np.sum(Wm[:, None] * sigma_pred, axis=0)
        P_pred = Q.copy()
        for i in range(sigma_pred.shape[0]):
            diff = sigma_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(diff, diff)
        P_pred = 0.5 * (P_pred + P_pred.T)
        return x_pred, P_pred

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray, P: np.ndarray,
                Q: np.ndarray, dt: float,
                method: str = "ekf") -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch to EKF or UKF prediction."""
        if method == "ukf":
            return self.predict_ukf(x, P, Q, dt)
        return self.predict_ekf(x, P, Q, dt)

    def observe(self, x: np.ndarray) -> float:
        """Predicted observation H·x = x1."""
        return float(x[0])

    def innovation(self, x_pred: np.ndarray, P_pred: np.ndarray,
                   y_t: float, R: np.ndarray) -> Tuple[float, float, float]:
        """Compute innovation, innovation variance, and NIS.

        Returns (v_t, S_t, NIS)  where all are scalars (1D obs).
        """
        y_pred = self.observe(x_pred)
        v_t = y_t - y_pred
        # S = H P H' + R  (scalar for 1D obs)
        S_t = float(self.H @ P_pred @ self.H.T + R)
        S_t = max(S_t, 1e-12)
        NIS = v_t**2 / S_t
        return v_t, S_t, NIS

    def gaussian_update(self, x_pred: np.ndarray, P_pred: np.ndarray,
                        y_t: float, R: np.ndarray,
                        gate: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Standard Gaussian Kalman update (baseline, no Student-t).

        Uses Joseph form for numerical stability.

        Returns (x_upd, P_upd, v_t, NIS).
        """
        v_t, S_t, NIS = self.innovation(x_pred, P_pred, y_t, R)

        # Kalman gain: K = P_pred H' / S
        K = (P_pred @ self.H.T) / S_t  # (3, 1)

        # Gated state update
        x_upd = x_pred + gate * K[:, 0] * v_t

        # Joseph form: P = (I - g·K·H) P (I - g·K·H)' + g²·K·R·K'
        I = np.eye(self.n_state, dtype=np.float64)
        M = I - gate * (K @ self.H)
        P_upd = M @ P_pred @ M.T + gate**2 * K @ R @ K.T
        P_upd = 0.5 * (P_upd + P_upd.T)

        # Floor eigenvalues
        for i in range(self.n_state):
            if P_upd[i, i] < 1e-10:
                P_upd[i, i] = 1e-10

        # Clamp log-freq to valid range
        x_upd[2] = np.clip(x_upd[2],
                           np.log(self.cfg.f_min),
                           np.log(self.cfg.f_max))

        return x_upd, P_upd, v_t, NIS
