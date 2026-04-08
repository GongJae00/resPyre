from typing import Dict, Optional
import numpy as np
from ..core.base import _BaseOscillatorHead

class oscillator_KFstd(_BaseOscillatorHead):
    head_key = "kfstd"

    _REF_FPS: float = 20.0  # reference fps used during original tuning

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        p = self.params
        fs = fs or p.fs
        self._maybe_apply_autotune(meta)
        signal_arr = np.asarray(signal, dtype=np.float64)
        multichannel = signal_arr.ndim == 2
        if multichannel:
            channels = [self._preprocess(np.asarray(signal_arr[i], dtype=np.float64).reshape(-1), fs) for i in range(signal_arr.shape[0])]
            n = min((ch.size for ch in channels), default=0)
            if n > 0:
                y = np.vstack([ch[:n] for ch in channels])
                ref = y[0]
                for i in range(1, y.shape[0]):
                    if y[i].size == ref.size and y[i].size > 8:
                        corr = float(np.corrcoef(ref, y[i])[0, 1])
                        if np.isfinite(corr) and corr < 0.0:
                            y[i] = -y[i]
            else:
                y = np.zeros((signal_arr.shape[0], 0), dtype=np.float64)
        else:
            y = self._preprocess(signal_arr, fs)
            n = y.size
        if n == 0:
            empty_sig = np.array([], dtype=np.float64) if not multichannel else np.zeros((signal_arr.shape[0], 0), dtype=np.float64)
            return self._package(empty_sig, np.array([], dtype=np.float64), meta)

        dt = 1.0 / fs
        init_len = min(n, max(int(10.0 * fs), int(4.0 * fs)))
        if multichannel:
            freq_candidates = []
            for i in range(y.shape[0]):
                yi = np.asarray(y[i, :init_len], dtype=np.float64)
                if yi.size:
                    fi = self._coarse_freq(yi, fs)
                    if np.isfinite(fi):
                        freq_candidates.append(float(fi))
            freq0 = float(np.median(freq_candidates)) if freq_candidates else float(self._coarse_freq(y[0, :init_len], fs))
        else:
            freq0 = self._coarse_freq(y[:init_len], fs)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))
        omega0 = 2.0 * np.pi * freq0
        eff = self._effective_params(fs, meta)
        rho = eff['rho']
        # Scale process noise to maintain consistent per-second diffusion
        dt_ref = 1.0 / self._REF_FPS
        qx = eff['qx'] * (dt / dt_ref)
        rv = eff['rv']
        # Convert post_smooth_alpha to fps-invariant form
        alpha_base = float(eff.get('post_smooth_alpha_base', getattr(self.params, 'post_smooth_alpha', 0.0) or 0.0))
        if 0.0 < alpha_base < 1.0:
            _tau_smooth = -1.0 / (np.log(max(alpha_base, 1e-6)) * self._REF_FPS)
            alpha_used = float(np.exp(-1.0 / max(_tau_smooth * fs, 1e-6)))
        else:
            alpha_used = float(eff.get('post_smooth_alpha_used', alpha_base))

        cos_w = np.cos(omega0 * dt)
        sin_w = np.sin(omega0 * dt)
        F = rho * np.array([[cos_w, -sin_w], [sin_w, cos_w]], dtype=np.float64)
        m_obs = int(y.shape[0]) if multichannel else 1
        C = np.zeros((m_obs, 2), dtype=np.float64)
        C[:, 0] = 1.0
        Q = qx * np.eye(2, dtype=np.float64)
        R = rv * np.eye(m_obs, dtype=np.float64)

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

            # Update (standard linear Kalman filter for 1D / stacked observation)
            y_t = np.asarray(y[:, t], dtype=np.float64) if multichannel else np.array([float(y[t])], dtype=np.float64)
            y_pred = C @ x
            innovation = y_t - y_pred
            S = C @ P @ C.T + R
            S = 0.5 * (S + S.T)
            try:
                S_inv = np.linalg.pinv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S + 1e-9 * np.eye(m_obs, dtype=np.float64))
            K = P @ C.T @ S_inv
            x = x + K @ innovation
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
        track_hz = self._apply_post_smoothing(track_hz, alpha_override=alpha_used)

        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        meta_payload["f0_raw"] = freq0
        meta_payload["freq_source"] = "kf_phase"
        meta_payload["warmup_frames"] = init_len
        meta_payload["post_smooth_alpha_base"] = alpha_base
        meta_payload["post_smooth_alpha_used"] = alpha_used
        meta_payload["observation_mode"] = "stacked_multichannel" if multichannel else "single_channel"
        meta_payload["n_observation_channels"] = int(m_obs)
        meta_payload.setdefault("is_constant_track", False)
        return self._package(x1, track_hz, meta_payload)
