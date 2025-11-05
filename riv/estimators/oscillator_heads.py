import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy import signal as sps
from scipy.signal import hilbert


@dataclass
class OscillatorParams:
    """Configuration shared across oscillator-based heads."""

    fs: float = 64.0
    f_min: float = 0.08
    f_max: float = 0.5
    rho: float = 0.995
    qx: float = 1e-4
    rv: float = 5e-2
    qf: float = 1e-5
    stft_win: float = 6.0
    stft_hop: float = 0.5
    ridge_lambda: float = 10.0
    pll_kp: float = 0.005
    pll_ki: float = 4e-5
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
        nperseg = int(min(max(fs * 8, fs), signal.size))
        if nperseg < 8:
            nperseg = min(256, signal.size)
        freqs, power = sps.welch(signal, fs=fs, nperseg=nperseg)
        mask = (freqs >= p.f_min) & (freqs <= p.f_max)
        if not np.any(mask):
            return 0.5 * (p.f_min + p.f_max)
        idx = np.argmax(power[mask])
        freq = float(freqs[mask][idx])
        return float(np.clip(freq, p.f_min, p.f_max))

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
        meta = {
            "head": self.head_key,
            "params": self.params.to_dict(),
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
        omega0 = 2.0 * np.pi * freq0
        rho = float(np.clip(p.rho, 0.9, 0.9999))

        cos_w = np.cos(omega0 * dt)
        sin_w = np.sin(omega0 * dt)
        A = rho * np.array([[cos_w, -sin_w], [sin_w, cos_w]], dtype=np.float64)
        C = np.array([[1.0, 0.0]], dtype=np.float64)
        Q = p.qx * np.eye(2, dtype=np.float64)
        R = np.array([[p.rv]], dtype=np.float64)

        x_filt = np.zeros((n, 2), dtype=np.float64)
        P_filt = np.zeros((n, 2, 2), dtype=np.float64)
        x_pred = np.zeros((n, 2), dtype=np.float64)
        P_pred = np.zeros((n, 2, 2), dtype=np.float64)

        x = np.zeros(2, dtype=np.float64)
        P = np.eye(2, dtype=np.float64)

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
            P = (np.eye(2) - K @ C) @ P

            x_filt[t] = x
            P_filt[t] = P

        x_smooth = np.copy(x_filt)
        P_smooth = np.copy(P_filt)
        for t in range(n - 2, -1, -1):
            P_inv = np.linalg.pinv(P_pred[t + 1])
            G = P_filt[t] @ A.T @ P_inv
            x_smooth[t] += G @ (x_smooth[t + 1] - x_pred[t + 1])
            P_smooth[t] += G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T

        phase = np.unwrap(np.arctan2(x_smooth[:, 1], x_smooth[:, 0]))
        dphi = np.gradient(phase, dt)
        track_hz = np.clip(dphi / (2.0 * np.pi), p.f_min, p.f_max)
        return self._package(x_smooth[:, 0], track_hz, {"f0": freq0, **(meta or {})})


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
        c = n + lam
        sqrtP = np.linalg.cholesky((c) * P)
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
        rho = float(np.clip(p.rho, 0.9, 0.9999))
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
        P = np.diag([1.0, 1.0, 0.1]).astype(np.float64)
        Q = np.diag([p.qx, p.qx, p.qf]).astype(np.float64)
        R = np.array([[max(p.rv, 1e-6)]], dtype=np.float64)

        states = np.zeros((n, 3), dtype=np.float64)
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

            K = Pxz @ np.linalg.pinv(S)
            innovation = y[t] - z_pred[0]
            x = x_pred + (K[:, 0] * innovation)
            P = P_pred - K @ S @ K.T
            x[2] = float(np.clip(x[2], log_f_min, log_f_max))

            states[t] = x

        track_hz = np.clip(np.exp(states[:, 2]), p.f_min, p.f_max)
        return self._package(states[:, 0], track_hz, {"f0": freq0, **(meta or {})})


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
            return self._package(y, track, {"stft_bins": int(np.sum(mask)), **(meta or {})})

        sub_freqs = freqs[mask]
        magnitudes = np.abs(Zxx[mask, :]).astype(np.float64)
        ridge_freqs = self._track_ridge(sub_freqs, magnitudes, p.ridge_lambda)

        def _median_smooth(arr: np.ndarray, width: int) -> np.ndarray:
            width = max(1, int(width))
            if width % 2 == 0:
                width += 1
            if arr.size <= 1 or width <= 1:
                return arr
            half = width // 2
            padded = np.pad(arr, (half, half), mode='edge')
            smoothed = np.empty_like(arr, dtype=np.float64)
            for idx in range(arr.size):
                smoothed[idx] = np.median(padded[idx:idx + width])
            return smoothed

        if ridge_freqs.size >= 3:
            kernel = 5 if ridge_freqs.size >= 5 else 3
            ridge_freqs = _median_smooth(ridge_freqs, kernel)

        sample_times = self._timebase(n, fs)
        if times.size < 2:
            track = np.full(n, np.clip(ridge_freqs[0], p.f_min, p.f_max), dtype=np.float64)
        else:
            track = np.interp(sample_times, times, ridge_freqs, left=ridge_freqs[0], right=ridge_freqs[-1])
            track = np.clip(track, p.f_min, p.f_max)
        if track.size >= 3:
            kernel = 5 if track.size >= 5 else 3
            track = _median_smooth(track, kernel)
        return self._package(y, track, {"stft_bins": int(np.sum(mask)), **(meta or {})})


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
        omega = 2.0 * np.pi * np.clip(freq0, p.f_min, p.f_max)
        phase_nco = float(np.angle(analytic[0]))
        integrator = 0.0
        integrator_limit = np.pi

        track = np.zeros(n, dtype=np.float64)
        osc = np.zeros(n, dtype=np.float64)
        omega_min = 2.0 * np.pi * p.f_min
        omega_max = 2.0 * np.pi * p.f_max
        kp = p.pll_kp
        ki = p.pll_ki
        if kp <= 0 or ki <= 0:
            fn = 0.04
            zeta = 0.7
            omega_n = 2.0 * np.pi * fn
            kp = 2.0 * zeta * omega_n * dt
            ki = (omega_n ** 2) * (dt ** 2)

        for t in range(n):
            ref = np.exp(-1j * phase_nco)
            err = np.angle(analytic[t] * ref)
            err = np.arctan2(np.sin(err), np.cos(err))
            integrator_candidate = np.clip(integrator + err, -integrator_limit, integrator_limit)
            omega_candidate = omega + kp * err + ki * integrator_candidate
            omega_clamped = float(np.clip(omega_candidate, omega_min, omega_max))
            if omega_clamped == omega_candidate:
                integrator = integrator_candidate
            else:
                integrator = np.clip(integrator, -integrator_limit, integrator_limit)
            omega = omega_clamped
            phase_nco = ((phase_nco + omega * dt + np.pi) % (2.0 * np.pi)) - np.pi
            track[t] = omega / (2.0 * np.pi)
            osc[t] = np.cos(phase_nco)

        return self._package(osc, track, {"f0": freq0, **(meta or {})})


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
