from typing import Dict, Optional
import numpy as np
from ..core.base import _BaseOscillatorHead

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
        eff = self._effective_params(fs, meta)
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
