"""
Student-t Robust Optimal Kalman Update — Algorithm 1.

Spec §C (§5.3–5.6): Scale-mixture derivation.
Core equation: R_eff = R / λ, where λ = (ν+1)/(ν + v²/S).

This module handles ONLY the measurement update step.
Prediction is in ssm.py. Trust allocation is in trust.py.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class UpdateResult:
    """Output of a single robust Kalman update step."""
    x: np.ndarray       # updated state
    P: np.ndarray       # updated covariance
    v_t: float          # innovation (scalar)
    S_t: float          # innovation variance (with base R)
    nis: float          # normalized innovation squared
    lambda_t: float     # VB scale weight
    R_eff: float        # effective measurement noise
    K: np.ndarray       # Kalman gain vector


class RobustKalmanUpdater:
    """Student-t VB/EM Robust Kalman update.

    Implements Algorithm 1 from spec §5.5:

        1) v = y − H m⁻,   S = H P⁻ Hᵀ + R_t
        2) λ = (ν + 1) / (ν + v²/S)           ← VB E-step
        3) R_eff = R_t / λ                     ← scale-mixture
        4) K = P⁻ Hᵀ / (H P⁻ Hᵀ + R_eff)     ← standard Kalman gain
        5) m = m⁻ + g · K · v                  ← gated update
        6) Joseph form covariance
        7) PSD projection
        8) Trace cap

    Identity: K = P⁻Hᵀ/(HP⁻Hᵀ + R/λ) ≡ λ·P⁻Hᵀ/(λ·HP⁻Hᵀ + R)
    """

    def __init__(self, nu: float = 5.0, vb_iters: int = 1,
                 eig_floor: float = 1e-9, trace_cap: float = 100.0,
                 lambda_floor: float = 1e-3, r_eff_max_scale: float = 80.0):
        """
        Args:
            nu: Student-t degrees of freedom. Lower = heavier tails.
                ν → ∞ recovers standard Gaussian KF.
            vb_iters: VB E-step iterations for λ refinement.
                      Typically 1 is sufficient for 1D observation.
            eig_floor: minimum eigenvalue for PSD projection.
            trace_cap: maximum trace(P) before scaling down.
            lambda_floor: lower bound for λ to avoid extreme R inflation.
            r_eff_max_scale: cap R_eff to at most r_eff_max_scale * R_scaled.
        """
        self.nu = float(nu)
        self.vb_iters = max(1, int(vb_iters))
        self.eig_floor = float(eig_floor)
        self.trace_cap = float(trace_cap)
        self.lambda_floor = float(max(lambda_floor, 1e-9))
        self.r_eff_max_scale = float(max(r_eff_max_scale, 1.0))

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray,
               y_t: float, H: np.ndarray, R: np.ndarray,
               gate: float = 1.0, gate_z: float = 1.0) -> UpdateResult:
        """Execute Algorithm 1: Student-t robust Kalman update.

        Args:
            x_pred: predicted state (n,)
            P_pred: predicted covariance (n, n)
            y_t: scalar observation
            H: observation matrix (1, n)
            R: base measurement noise (1, 1) matrix
            gate: observation trust gate g_t ∈ [0, 1]
            gate_z: separate frequency gate g_z ∈ [0, 1]

        Returns:
            UpdateResult with updated state, covariance, and diagnostics.
        """
        n = x_pred.size
        R_val = float(R[0, 0]) if R.ndim == 2 else float(R)

        # ── Step 1: Innovation with base R ──
        y_pred = (H @ x_pred).item()
        v = y_t - y_pred

        # ── Step 2. Innovation variance S = H P⁻ Hᵀ + R ──
        #    (Scalar for 1D observation)
        S_base = (H @ P_pred @ H.T).item() + R_val
        S_base = max(S_base, 1e-12) # Ensure S_base is not zero for NIS calculation
        nis_base = (v ** 2) / S_base

        # ── Step 3. VB Iteration for λ (Algorithm 1, lines 4-6)── 
        #    Initialize λ = 1.0 (or previous)
        lambda_t = 1.0

        # If nu is infinite (Gaussian), lambda stays 1.0
        if self.nu > 1e12:
            pass
        else:
            for _ in range(self.vb_iters):
                # E-step: λ = (ν + 1) / (ν + v²/S_t)
                # where S_t is the current total variance R/λ + HP Hᵀ
                
                R_eff = R_val / lambda_t
                S_eff = (H @ P_pred @ H.T).item() + R_eff
                S_eff = max(S_eff, 1e-12) # Ensure S_eff is not zero
                mahal_sq = (v ** 2) / S_eff
                lambda_t = (self.nu + 1.0) / (self.nu + mahal_sq)

        # ── Step 4. Final update with converged λ ──
        #    R_eff = R / λ, with robust caps for numerical stability.
        lambda_t = float(np.clip(lambda_t, self.lambda_floor, 1e6))
        R_eff = R_val / lambda_t
        if np.isfinite(self.r_eff_max_scale) and self.r_eff_max_scale > 0:
            R_eff_cap = R_val * self.r_eff_max_scale
            if np.isfinite(R_eff_cap) and R_eff_cap > 0:
                R_eff = float(min(R_eff, R_eff_cap))
        HP = (H @ P_pred @ H.T).item()
        S_eff_final = HP + R_eff
        S_eff_final = max(S_eff_final, 1e-12) # Ensure S_eff_final is not zero
        K = (P_pred @ H.T) / S_eff_final  # (n, 1)

        # ── Step 5: Gated state update ──
        # Separate gates for observation (g_t) and frequency (g_z)
        delta_x = K[:, 0] * v
        g_vec = np.full(n, gate, dtype=np.float64)
        if n >= 3:
            g_vec[2] = gate_z  # frequency component uses separate gate
        x_upd = x_pred + g_vec * delta_x

        # ── Step 6: Joseph form covariance ──
        # P = (I - g·K·H) P (I - g·K·H)' + g²·K·R_eff·K'
        I_n = np.eye(n, dtype=np.float64)
        gK = g_vec[:, None] * K  # (n, 1) element-wise gated
        M = I_n - gK @ H
        R_eff_mat = np.array([[R_eff]], dtype=np.float64)
        P_upd = M @ P_pred @ M.T + gK @ R_eff_mat @ gK.T

        # ── Step 7: PSD projection ──
        P_upd = self._psd_project(P_upd)

        # ── Step 8: Trace cap ──
        P_upd = self._trace_cap(P_upd)

        # NIS for logging (computed with base S, not R_eff)
        nis = float(nis_base)

        return UpdateResult(
            x=x_upd, P=P_upd,
            v_t=v, S_t=S_base, nis=nis,
            lambda_t=lambda_t, R_eff=R_eff,
            K=K[:, 0]
        )

    def _psd_project(self, P: np.ndarray) -> np.ndarray:
        """Symmetrize and clamp eigenvalues to ensure PSD."""
        P = 0.5 * (P + P.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.clip(eigvals, self.eig_floor, None)
            P = eigvecs @ np.diag(eigvals) @ eigvecs.T
        except np.linalg.LinAlgError:
            # Fallback: diagonal floor only
            for i in range(P.shape[0]):
                P[i, i] = max(P[i, i], self.eig_floor)
        return P

    def _trace_cap(self, P: np.ndarray) -> np.ndarray:
        """Prevent covariance explosion."""
        tr = float(np.trace(P))
        if tr > self.trace_cap and tr > 0:
            P = P * (self.trace_cap / tr)
        return P

    # ------------------------------------------------------------------
    # Gaussian recovery convenience
    # ------------------------------------------------------------------
    @classmethod
    def gaussian(cls) -> "RobustKalmanUpdater":
        """Create an updater that behaves as standard Gaussian KF (ν→∞)."""
        return cls(nu=float('inf'), vb_iters=1)
