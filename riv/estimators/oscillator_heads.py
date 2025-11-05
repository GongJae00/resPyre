import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
from scipy import signal as sps
from scipy.signal import hilbert


@dataclass
class OscillatorParams:
    """Configuration shared across oscillator-based heads."""

    fs: float = 64.0
    f_min: float = 0.08
    f_max: float = 0.50
    rho: float = 0.0  # legacy manual override
    tau_env: float = 30.0
    qx: float = 0.0  # legacy manual override
    qx_scale: float = 0.3
    rv: float = 0.1  # legacy manual override
    rv_auto: bool = True
    rv_mad_scale: float = 1.2
    rv_floor: float = 0.08
    qf: float = 1e-7
    stft_win: float = 12.0
    stft_hop: float = 1.0
    ridge_penalty: float = 250.0
    pll_autogain: bool = True
    pll_kp: float = 0.0  # legacy manual override
    pll_ki: float = 0.0  # legacy manual override
    pll_zeta: float = 0.9
    pll_ttrack: float = 5.0
    detrend: bool = True
    bandpass: bool = True
    zscore: bool = True

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class _BaseOscillatorHead:
    """Shared helpers for oscillator heads."""

    head_key: str = "base"

    def __init__(self, params: Optional[OscillatorParams] = None):
        self.params = params or OscillatorParams()
        self._last_sigma_y = None

    def _preprocess(self, signal: np.ndarray, fs: float) -> np.ndarray:
        p = self.params
        x = np.asarray(signal, dtype=np.float64).copy()
        if x.size == 0:
            return x
        if np.any(np.isnan(x)):
            x = np.nan_to_num(x)
        if p.detrend:
            x = sps.detrend(x, type="linear")
        if p.bandpass and fs > 0:
            nyq = 0.5 * fs
            low = max(p.f_min, 0.01)
            high = min(p.f_max, nyq - 1e-3)
            if high > low:
                b, a = sps.butter(2, [low / nyq, high / nyq], btype="bandpass")
                x = sps.filtfilt(b, a, x, method="gust")
        median = np.median(x)
        mad = np.median(np.abs(x - median)) / 0.6745
        if not np.isfinite(mad) or mad <= 1e-6:
            std = np.std(x)
            mad = std if std > 1e-6 else 1.0
        self._last_sigma_y = float(mad)
        if p.zscore:
            std = np.std(x)
            if std > 1e-8:
                x = (x - np.mean(x)) / (std + 1e-8)
            else:
                x = x - np.mean(x)
        return x

    def _timebase(self, n: int, fs: float) -> np.ndarray:
        if fs <= 0:
            return np.arange(n)
        return np.arange(n) / fs

    def _coarse_freq(self, signal: np.ndarray, fs: float) -> float:
        p = self.params
        if signal.size < 8:
            return 0.5 * (p.f_min + p.f_max)
        win = int(min(max(fs * 20, fs), signal.size))
        if win < 8:
            win = min(256, signal.size)
        freqs, power = sps.welch(signal, fs=fs, nperseg=win)
        mask = (freqs >= p.f_min) & (freqs <= p.f_max)
        if not np.any(mask):
            return 0.5 * (p.f_min + p.f_max)
        freq = float(freqs[mask][np.argmax(power[mask])])
        if not np.isfinite(freq):
            freq = 0.5 * (p.f_min + p.f_max)
        return float(np.clip(freq, p.f_min, p.f_max))

    def _effective_params(self, fs: float) -> Dict[str, float]:
        p = self.params
        fs = fs or p.fs
        tau = max(p.tau_env, 1e-3)
        rho = np.exp(-1.0 / max(fs * tau, 1e-6))
        if p.rho and p.rho > 0:
            rho = np.clip(p.rho, 0.0, 0.999999)
        if p.qx and p.qx > 0:
            qx = p.qx
        else:
            qx = max((p.qx_scale or 0.0) * (1.0 - rho ** 2), 1e-8)
        if p.rv_auto:
            sigma = self._last_sigma_y if (self._last_sigma_y is not None and np.isfinite(self._last_sigma_y)) else 1.0
            rv = max((p.rv_mad_scale * sigma) ** 2, p.rv_floor)
        else:
            rv = max(p.rv, p.rv_floor)
        qf = p.qf if (p.qf and p.qf > 0) else 1e-7
        return {
            'rho': float(rho),
            'qx': float(qx),
            'rv': float(rv),
            'qf': float(qf)
        }

    def _package(
        self,
        signal_hat: np.ndarray,
        track_hz: np.ndarray,
        meta_extra: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        track = np.asarray(track_hz, dtype=np.float64)
        signal_arr = np.asarray(signal_hat, dtype=np.float64)
        valid = track[~np.isnan(track)]
        rr_hz = float(np.nanmedian(valid)) if valid.size else float("nan")
        rr_bpm = float(rr_hz * 60.0) if np.isfinite(rr_hz) else float("nan")
        finite_track = track[np.isfinite(track)]
        if finite_track.size:
            sat_mask = (finite_track <= (self.params.f_min + 1e-6)) | (finite_track >= (self.params.f_max - 1e-6))
            frac_saturated = float(np.mean(sat_mask))
            dyn_range = float(np.percentile(finite_track, 95) - np.percentile(finite_track, 5))
        else:
            frac_saturated = float("nan")
            dyn_range = float("nan")
        meta = {
            "head": self.head_key,
            "params": self.params.to_dict(),
            "track_frac_saturated": frac_saturated,
            "track_dyn_range_hz": dyn_range
        }
        if meta_extra:
            meta.update(meta_extra)
        return {
            "signal_hat": signal_arr,
            "track_hz": track,
            "rr_hz": rr_hz,
            "rr_bpm": rr_bpm,
            "meta": json.dumps(meta),
        }

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class oscillator_KFstd(_BaseOscillatorHead):
    head_key = "kfstd"

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        p = self.params
        fs = fs or p.fs
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        dt = 1.0 / fs
        freq0 = self._coarse_freq(y, fs)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))
        omega0 = 2.0 * np.pi * freq0
        eff = self._effective_params(fs)
        rho = eff['rho']
        qx = eff['qx']
        rv = eff['rv']

        cos_w = np.cos(omega0 * dt)
        sin_w = np.sin(omega0 * dt)
        A = rho * np.array([[cos_w, -sin_w], [sin_w, cos_w]], dtype=np.float64)
        C = np.array([[1.0, 0.0]], dtype=np.float64)
        Q = qx * np.eye(2, dtype=np.float64)
        R = np.array([[rv]], dtype=np.float64)

        x_filt = np.zeros((n, 2), dtype=np.float64)
        P_filt = np.zeros((n, 2, 2), dtype=np.float64)
        x_pred = np.zeros((n, 2), dtype=np.float64)
        P_pred = np.zeros((n, 2, 2), dtype=np.float64)

        x = np.zeros(2, dtype=np.float64)
        P = np.eye(2, dtype=np.float64)

        I = np.eye(2, dtype=np.float64)
        for t in range(n):
            x = A @ x
            P = A @ P @ A.T + Q
            x_pred[t] = x
            P_pred[t] = P

            y_t = y[t]
            S = C @ P @ C.T + R
            K = (P @ C.T) / S
            innovation = y_t - (C @ x)[0]
            x = x + (K[:, 0] * innovation)
            P = (I - K @ C) @ P @ (I - K @ C).T + K @ R @ K.T

            x_filt[t] = x
            P_filt[t] = P

        x_smooth = np.copy(x_filt)
        P_smooth = np.copy(P_filt)
        A_T = A.T
        for t in range(n - 2, -1, -1):
            try:
                P_inv = np.linalg.pinv(P_pred[t + 1])
            except np.linalg.LinAlgError:
                P_inv = np.linalg.inv(P_pred[t + 1] + 1e-9 * np.eye(2))
            G = P_filt[t] @ A_T @ P_inv
            x_smooth[t] += G @ (x_smooth[t + 1] - x_pred[t + 1])
            P_smooth[t] += G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T

        track_hz = np.full(n, freq0, dtype=np.float64)
        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        meta_payload.setdefault("is_constant_track", True)
        return self._package(x_smooth[:, 0], track_hz, meta_payload)


class oscillator_UKF_freq(_BaseOscillatorHead):
    head_key = "ukffreq"

    def __init__(self, params: Optional[OscillatorParams] = None):
        super().__init__(params=params)
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0

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
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        dt = 1.0 / fs
        freq0 = self._coarse_freq(y, fs)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))
        eff = self._effective_params(fs)
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
        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        meta_payload.setdefault("is_constant_track", False)
        return self._package(states[:, 0], track_hz, meta_payload)


class oscillator_Spec_ridge(_BaseOscillatorHead):
    head_key = "spec_ridge"

    def _track_ridge(self, freqs: np.ndarray, magnitudes: np.ndarray, penalty: float) -> np.ndarray:
        n_freqs, n_times = magnitudes.shape
        if n_times == 0:
            return np.zeros(0, dtype=np.float64)
        cost = magnitudes[:, 0].astype(np.float64)
        backptr = np.zeros((n_freqs, n_times), dtype=np.int32)
        freq_diff = (freqs[:, None] - freqs[None, :]) ** 2
        scale = penalty / max((freqs[-1] - freqs[0]) ** 2, 1e-6)
        for t in range(1, n_times):
            transition = cost[None, :] - scale * freq_diff
            idx = np.argmax(transition, axis=1)
            cost = transition[np.arange(n_freqs), idx] + magnitudes[:, t]
            backptr[:, t] = idx
        ridge = np.zeros(n_times, dtype=np.int32)
        ridge[-1] = int(np.argmax(cost))
        for t in range(n_times - 1, 0, -1):
            ridge[t - 1] = backptr[ridge[t], t]
        return freqs[ridge]

    def _median_smooth(self, arr: np.ndarray, width: int) -> np.ndarray:
        width = max(1, int(width))
        if width % 2 == 0:
            width += 1
        if arr.size <= 1 or width <= 1:
            return arr
        pad = width // 2
        padded = np.pad(arr, (pad, pad), mode='edge')
        out = np.empty_like(arr, dtype=np.float64)
        for idx in range(arr.size):
            out[idx] = np.median(padded[idx:idx + width])
        return out

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        p = self.params
        fs = fs or p.fs
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        win = max(16, int(p.stft_win * fs))
        hop = max(1, int(p.stft_hop * fs))
        hop = min(hop, win - 1)
        freqs, times, Zxx = sps.stft(y, fs=fs, nperseg=win, noverlap=win - hop, detrend=False, padded=False)
        mask = (freqs >= p.f_min) & (freqs <= p.f_max)
        if not np.any(mask):
            track = np.full(n, 0.5 * (p.f_min + p.f_max), dtype=np.float64)
            meta_payload = dict(meta or {})
            meta_payload["stft_bins"] = int(np.sum(mask))
            meta_payload.setdefault("is_constant_track", False)
            return self._package(y, track, meta_payload)

        sub_freqs = freqs[mask]
        magnitudes = np.abs(Zxx[mask, :]).astype(np.float64)
        ridge_freqs = self._track_ridge(sub_freqs, magnitudes, p.ridge_penalty)

        if ridge_freqs.size >= 3:
            time_step = times[1] - times[0] if times.size > 1 else p.stft_hop
            kernel = max(3, int(round(1.0 / max(time_step, 1e-6))))
            ridge_freqs = self._median_smooth(ridge_freqs, kernel)

        sample_times = self._timebase(n, fs)
        if times.size < 2:
            track = np.full(n, np.clip(ridge_freqs[0], p.f_min, p.f_max), dtype=np.float64)
        else:
            track = np.interp(sample_times, times, ridge_freqs, left=ridge_freqs[0], right=ridge_freqs[-1])
            track = np.clip(track, p.f_min, p.f_max)
        if track.size >= 3:
            track = self._median_smooth(track, 5)
        meta_payload = dict(meta or {})
        meta_payload["stft_bins"] = int(np.sum(mask))
        meta_payload.setdefault("is_constant_track", False)
        return self._package(y, track, meta_payload)


class oscillator_PLL(_BaseOscillatorHead):
    head_key = "pll"

    def run(self, signal: np.ndarray, fs: float, meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        p = self.params
        fs = fs or p.fs
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        dt = 1.0 / fs
        analytic = hilbert(y)
        freq0 = self._coarse_freq(y, fs)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))
        omega = 2.0 * np.pi * freq0
        phase_nco = float(np.angle(analytic[0]))
        integrator = 0.0

        if p.pll_autogain or p.pll_kp <= 0 or p.pll_ki <= 0:
            omega_n = 2.0 * np.pi / max(p.pll_ttrack, 1e-3)
            kp = 2.0 * p.pll_zeta * omega_n * dt
            ki = (omega_n * dt) ** 2
        else:
            kp = p.pll_kp
            ki = p.pll_ki

        track = np.zeros(n, dtype=np.float64)
        osc = np.zeros(n, dtype=np.float64)
        omega_min = 2.0 * np.pi * p.f_min
        omega_max = 2.0 * np.pi * p.f_max

        for t in range(n):
            phase_y = float(np.angle(analytic[t]))
            err = np.arctan2(np.sin(phase_y - phase_nco), np.cos(phase_y - phase_nco))
            integrator_candidate = integrator + ki * err
            omega_candidate = omega + kp * err + integrator_candidate
            omega_clamped = float(np.clip(omega_candidate, omega_min, omega_max))
            if omega_clamped != omega_candidate:
                integrator_candidate += omega_clamped - omega_candidate
            integrator = np.clip(integrator_candidate, -np.pi, np.pi)
            omega = omega_clamped
            phase_nco = ((phase_nco + omega * dt + np.pi) % (2.0 * np.pi)) - np.pi
            track[t] = omega / (2.0 * np.pi)
            osc[t] = np.cos(phase_nco)

        track = np.clip(track, p.f_min, p.f_max)
        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        meta_payload.setdefault("is_constant_track", False)
        return self._package(osc, track, meta_payload)


HEAD_REGISTRY = {
    "kfstd": oscillator_KFstd,
    "kf_std": oscillator_KFstd,
    "ukffreq": oscillator_UKF_freq,
    "ukf_freq": oscillator_UKF_freq,
    "spec_ridge": oscillator_Spec_ridge,
    "specridge": oscillator_Spec_ridge,
    "pll": oscillator_PLL,
}


def build_head(head_key: str, params: Optional[OscillatorParams] = None):
    key = head_key.lower()
    if key not in HEAD_REGISTRY:
        raise ValueError(f"Unknown oscillator head '{head_key}'")
    cls = HEAD_REGISTRY[key]
    return cls(params=params)
