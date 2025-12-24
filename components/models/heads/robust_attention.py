from typing import Dict, Optional, Tuple
import numpy as np
from .ukf_freq import oscillator_UKF_freq

class oscillator_Robust_Attention(oscillator_UKF_freq):
    head_key = "robust_attention"

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        p = self.params
        fs = fs or p.fs
        self._maybe_apply_autotune(meta)
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        dt = 1.0 / fs
        freq0 = self._coarse_freq(y, fs, meta)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))
        eff = self._effective_params(fs, meta)
        rho = eff['rho']
        qx = eff['qx']
        rv = eff['rv']
        qf = max(eff['qf'], 1e-10)

        # Robust / Student-t parameters
        nu = float(getattr(p, 'student_t_nu', 4.0)) # Degrees of freedom
        
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
            # Random walk is added in the additive noise step, here just dynamics
            return new_state

        def observe(x_state):
            return np.array([x_state[0]], dtype=np.float64)

        x = np.array([0.0, 0.0, log_f0], dtype=np.float64)
        P = np.diag([1.0, 1.0, 0.25 ** 2]).astype(np.float64)
        Q = np.diag([qx, qx, qf]).astype(np.float64)
        # R is adaptive now, but we start with base rv
        R_base = np.array([[rv]], dtype=np.float64)

        states = np.zeros((n, 3), dtype=np.float64)
        weights_hist = np.zeros(n, dtype=np.float64)
        
        I = np.eye(3, dtype=np.float64)
        
        for t in range(n):
            # 1. Prediction Step (Standard UKF)
            # --------------------------------
            sigma, Wm, Wc = self._sigma_points(x, P)
            sigma_pred = np.array([transition(sp) for sp in sigma], dtype=np.float64)
            x_pred = np.sum(Wm[:, None] * sigma_pred, axis=0)
            
            # Amplitude Gating for Frequency Noise:
            # If amplitude is low, signal is noise -> Don't let frequency drift (reduce Q_f)
            amplitude = np.sqrt(x_pred[0]**2 + x_pred[1]**2)
            amp_gate = np.clip(amplitude / (np.sqrt(rv) * 2.0), 0.1, 1.0) # 0.1 at noise floor, 1.0 at signal
            
            # Adaptive Q
            Q_t = Q.copy()
            Q_t[2, 2] *= amp_gate # Reduce freq walk when signal is weak
            
            P_pred = Q_t.copy()
            for i in range(sigma_pred.shape[0]):
                diff = sigma_pred[i] - x_pred
                P_pred += Wc[i] * np.outer(diff, diff)

            # 2. Update Step (Robust / Student-t)
            # -----------------------------------
            # Recalculate sigma points for update (optional, but standard UKF reuses pred)
            # We reuse sigma_pred for Z calculation
            Z = np.array([observe(sp) for sp in sigma_pred], dtype=np.float64)
            z_pred = np.sum(Wm[:, None] * Z, axis=0)
            
            # Innovation Covariance
            S = R_base.copy() # Start with base R
            Pxz = np.zeros((3, 1), dtype=np.float64)
            
            for i in range(Z.shape[0]):
                z_diff = Z[i] - z_pred
                x_diff = sigma_pred[i] - x_pred
                S += Wc[i] * np.outer(z_diff, z_diff)
                Pxz += Wc[i] * np.outer(x_diff, z_diff)
            
            if S.shape == (1, 1):
                S_val = float(max(S[0, 0], 1e-12))
            else:
                S_val = 1e-12

            # Calculate robust weight w_t
            innovation = y[t] - z_pred[0]
            d_sq = (innovation ** 2) / S_val
            
            # Student-t Weighting: w = (nu + 1) / (nu + d^2)
            # This down-weights the observation noise R (effectively INCREASING R) for outliers
            w_t = (nu + 1) / (nu + d_sq)
            weights_hist[t] = w_t
            
            # Re-calculate S and K with robust R
            R_eff = R_base / max(w_t, 1e-4) # Avoid div by zero
            ensure_S = S_val - R_base[0,0] # S without R
            # New S_eff = (HPH') + R_eff
            S_eff = ensure_S + R_eff[0,0]
            
            K = Pxz / S_eff
            
            # State Update
            x = x_pred + (K[:, 0] * innovation)
            P = P_pred - K @ (S_eff * np.eye(1)) @ K.T
            
            # Stabilize P
            P = 0.5 * (P + P.T)
            for i in range(3):
                if P[i, i] < 1e-8: P[i, i] = 1e-8
            
            x[2] = float(np.clip(x[2], log_f_min, log_f_max))
            states[t] = x

        track_hz = np.clip(np.exp(states[:, 2]), p.f_min, p.f_max)
        track_hz = self._apply_post_smoothing(track_hz)
        
        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        meta_payload["robust_weights_mean"] = float(np.mean(weights_hist))
        meta_payload["is_constant_track"] = False
        
        return self._package(states[:, 0], track_hz, meta_payload)
