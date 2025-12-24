from typing import Dict, Optional
import numpy as np
from ..core.base import _BaseOscillatorHead, OscillatorParams

class oscillator_UKF_freq(_BaseOscillatorHead):
    head_key = "ukffreq"

    def __init__(self, params: Optional[OscillatorParams] = None):
        super().__init__(params=params)
        p = self.params
        self.alpha = float(p.ukf_alpha if (p.ukf_alpha and p.ukf_alpha > 0) else 1e-3)
        self.beta = float(p.ukf_beta if p.ukf_beta else 2.0)
        self.kappa = float(p.ukf_kappa if p.ukf_kappa is not None else 0.0)

    def _sigma_points(self, x: np.ndarray, P: np.ndarray):
        n = x.size
        lam = self.alpha ** 2 * (n + self.kappa) - n
        c = max(n + lam, 1e-9)
        P = 0.5 * (P + P.T)
        jitter = 1e-9 * max(1.0, float(np.max(np.diag(P))))
        for _ in range(6):
            try:
                L = np.linalg.cholesky(P + jitter * np.eye(n))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.clip(eigvals, 1e-10, None)
            L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        sqrtP = np.sqrt(c) * L
        sigma = np.zeros((2 * n + 1, n), dtype=np.float64)
        sigma[0] = x
        for i in range(n):
            sigma[i + 1] = x + sqrtP[:, i]
            sigma[n + i + 1] = x - sqrtP[:, i]
        Wm = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
        Wc = np.full(2 * n + 1, 1.0 / (2.0 * c), dtype=np.float64)
        Wm[0] = lam / c
        Wc[0] = lam / c + (1.0 - self.alpha ** 2 + self.beta)
        return sigma, Wm, Wc

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
            new_state[2] = np.clip(new_state[2], log_f_min, log_f_max)
            return new_state

        def observe(x_state):
            return np.array([x_state[0]], dtype=np.float64)

        x = np.array([0.0, 0.0, log_f0], dtype=np.float64)
        P = np.diag([1.0, 1.0, 0.25 ** 2]).astype(np.float64)
        Q = np.diag([qx, qx, qf]).astype(np.float64)
        R = np.array([[rv]], dtype=np.float64)

        states = np.zeros((n, 3), dtype=np.float64)
        I = np.eye(3, dtype=np.float64)
        for t in range(n):
            sigma, Wm, Wc = self._sigma_points(x, P)
            sigma_pred = np.array([transition(sp) for sp in sigma], dtype=np.float64)
            x_pred = np.sum(Wm[:, None] * sigma_pred, axis=0)
            P_pred = Q.copy()
            for i in range(sigma_pred.shape[0]):
                diff = sigma_pred[i] - x_pred
                P_pred += Wc[i] * np.outer(diff, diff)

            Z = np.array([observe(sp) for sp in sigma_pred], dtype=np.float64)
            z_pred = np.sum(Wm[:, None] * Z, axis=0)
            S = R.copy()
            Pxz = np.zeros((3, 1), dtype=np.float64)
            for i in range(Z.shape[0]):
                z_diff = Z[i] - z_pred
                x_diff = sigma_pred[i] - x_pred
                S += Wc[i] * np.outer(z_diff, z_diff)
                Pxz += Wc[i] * np.outer(x_diff, z_diff)
            if S.shape == (1, 1):
                S[0, 0] = float(max(S[0, 0], 1e-12))
            K = Pxz @ np.linalg.pinv(S)
            innovation = y[t] - z_pred[0]
            x = x_pred + (K[:, 0] * innovation)
            P = (I - K @ np.array([[1.0, 0.0, 0.0]])) @ P_pred @ (I - K @ np.array([[1.0, 0.0, 0.0]])).T + K @ R @ K.T
            P = 0.5 * (P + P.T)
            for i in range(3):
                if P[i, i] < 1e-8:
                    P[i, i] = 1e-8
            x[2] = float(np.clip(x[2], log_f_min, log_f_max))

            states[t] = x

        track_hz = np.clip(np.exp(states[:, 2]), p.f_min, p.f_max)
        track_hz = self._apply_post_smoothing(track_hz)
        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        qf_eff = float(qf)
        meta_payload["qf_eff"] = qf_eff
        try:
            finite_track = track_hz[np.isfinite(track_hz)]
            freq_med = float(np.nanmedian(finite_track)) if finite_track.size else float(freq0)
            sigma_bpm_30s = 60.0 * freq_med * float(np.sqrt(max(qf_eff, 0.0) * fs * 30.0))
            if np.isfinite(sigma_bpm_30s):
                meta_payload["qf_sigma_bpm_30s"] = sigma_bpm_30s
                meta_payload.setdefault("qf_note", "sigma_bpm_30s approximates 30s breathing drift")
        except Exception:
            pass
        meta_payload.setdefault("is_constant_track", False)
        return self._package(states[:, 0], track_hz, meta_payload)
