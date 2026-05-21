"""
NAROSSM v5c — Noise-Adaptive Robust Oscillatory State-Space Model.

A 2D oscillatory Kalman filter with:
  1. Innovation-based adaptive measurement noise R_t (Mehra 1970, stabilised)
  2. Innovation-based adaptive process noise Q_t (R-based bandwidth control)
  3. Running-kurtosis adaptive Student-t degrees of freedom ν_t
  4. Student-t VB robust update (outlier downweighting via λ_t)
  5. Phase-coherence & amplitude-stability gating (NIS-free trust)
  6. RTS backward smoother (exact for linear 2D system)
  7. Harmonic-aware frequency initialization (prevents 2nd harmonic lock)
  8. Self-gating frequency adaptation (rate-limited, confirmation-based)

Design notes (empirical findings from v6–v7 experimentation):
  - Adaptive R is the primary working mechanism: Mehra-based innovation
    covariance tracking correctly handles heteroscedastic noise.
  - After R adaptation normalises residuals, Student-t ν → 200 (Gaussian).
    This is correct behaviour: it serves as a diagnostic confirming that
    R adaptation has captured the noise structure.
  - Frequency adaptation is self-gating: on COHFACE, it fires on < 5%
    of trials (freq_std ≤ 0.005 for stable breathing). It helps DoF
    where genuine frequency changes exist, but is mostly inert on clean
    signals — a desirable property.
  - LTI convergence (v6) was tested but removed: the improvement on
    DoF was marginal (2.185 → 2.185) while slightly degrading P1D-Cubic
    (0.220 → 0.225). The adaptive mechanisms are already self-regulating.
"""

from typing import Dict, Optional
import numpy as np
import os
import hashlib
from ..core.base import _BaseOscillatorHead
from core.evaluation.frame_logger import FrameLogger
from core.pipeline.common import (
    derive_trial_identifiers,
    sanitize_trial_key,
    update_frame_log_manifest,
)


def _angle_wrap(x: float) -> float:
    """Wrap angle to [-π, π]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class oscillator_NAROSSM(_BaseOscillatorHead):
    """NAROSSM v5c: Noise-Adaptive Robust Oscillatory State-Space Model."""

    head_key = "narossm"

    # ── Time-scale reference ──
    # All time constants are defined in seconds and converted to per-sample
    # values at runtime using the actual sampling rate (fs). This ensures
    # consistent physical behaviour across different fps (e.g. 20 vs 61).
    _REF_FPS: float = 20.0  # reference fps used during original tuning

    # ── Adaptive R hyperparameters ──
    TAU_R_SEC: float = 2.5            # R adaptation time constant (seconds)
    R_ANCHOR_FRAC: float = 0.3       # R floor = fraction of R_init

    # ── Adaptive Q (symmetric R-based bandwidth control) ──
    ADAPTIVE_Q: bool = True
    QX_ADAPT_GAMMA: float = 0.5      # Power-law exponent (sqrt relationship)
    QX_MIN_SCALE: float = 0.1        # Min Q scale (10% of base)
    QX_MAX_SCALE: float = 3.0        # Max Q scale (3x base)
    QX_ADAPT_WARMUP_SEC: float = 3.0 # Wait for R_t to stabilize (seconds)

    # ── Student-t / kurtosis ──
    TAU_KAPPA_SEC: float = 2.5       # Kurtosis EMA time constant (seconds)
    NU_MIN: float = 3.0
    NU_MAX: float = 200.0
    KAPPA_EPS: float = 0.01
    LAMBDA_FLOOR: float = 1e-3

    # ── Numerical safety ──
    TRACE_CAP: float = 100.0
    EIG_FLOOR: float = 1e-9
    WARMUP_SEC: float = 1.5          # Warmup duration (seconds)

    # ── Frequency adaptation (self-gating, rate-limited) ──
    FREQ_ADAPT: bool = True
    FREQ_UPDATE_INTERVAL_SEC: float = 3.0
    FREQ_UPDATE_MIN_BUF_SEC: float = 3.0
    FREQ_MAX_RATE: float = 0.010
    FREQ_MIN_DELTA: float = 0.05
    FREQ_CONFIRM_COUNT: int = 3

    # ── Phase-coherence & amplitude-stability gate ──
    GATE_PHASE_SIGMA: float = 0.8
    GATE_AMP_SIGMA: float = 3.0
    TAU_AMP_SEC: float = 1.5         # Amplitude baseline time constant (seconds)
    GATE_FLOOR: float = 0.05
    GATE_THRESHOLD: float = 0.65
    GATE_WARMUP_SEC: float = 0.75    # Gate warmup duration (seconds)

    # ── Harmonic-aware initialisation (v5) ──
    HARMONIC_POWER_RATIO: float = 0.15  # Subharmonic needs 15% of peak power (conservative; autocorr confirms)

    # ── LTI convergence (v6: tested, disabled in v5c) ──
    # When R_t/R_init stays low, all adaptations freeze → pure LTI filter.
    # Removed in v5c: adaptive mechanisms are already self-regulating.
    LTI_CONVERGE: bool = False
    TAU_LTI_SEC: float = 5.0          # LTI r_ratio EMA time constant (seconds)
    LTI_ENTER_THRESH: float = 0.4     # r_ratio_ema below this → LTI candidate
    LTI_ENTER_SEC: float = 10.0       # Sustained stability needed (seconds)
    TAU_LTI_SURPRISE_SEC: float = 0.5 # Surprise EMA time constant (seconds)
    LTI_EXIT_SURPRISE: float = 2.5    # Surprise EMA above this → exit LTI

    def run(
        self,
        signal: np.ndarray,
        fs: float,
        meta: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        p = self.params
        fs = fs or p.fs
        self._maybe_apply_autotune(meta)
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        dt = 1.0 / fs

        # ── Compute per-sample constants from time-based hyperparameters ──
        ALPHA_R = float(np.exp(-dt / max(self.TAU_R_SEC, 1e-6)))
        ALPHA_KAPPA = float(np.exp(-dt / max(self.TAU_KAPPA_SEC, 1e-6)))
        GATE_ALPHA_AMP = float(np.exp(-dt / max(self.TAU_AMP_SEC, 1e-6)))
        WARMUP_FRAMES = max(1, int(self.WARMUP_SEC * fs))
        QX_ADAPT_WARMUP = max(1, int(self.QX_ADAPT_WARMUP_SEC * fs))
        FREQ_UPDATE_INTERVAL = max(1, int(self.FREQ_UPDATE_INTERVAL_SEC * fs))
        GATE_WARMUP = max(1, int(self.GATE_WARMUP_SEC * fs))
        LTI_ALPHA = float(np.exp(-dt / max(self.TAU_LTI_SEC, 1e-6)))
        LTI_ENTER_FRAMES = max(1, int(self.LTI_ENTER_SEC * fs))
        LTI_SURPRISE_ALPHA = float(np.exp(-dt / max(self.TAU_LTI_SURPRISE_SEC, 1e-6)))

        # Scale process noise qx to maintain consistent per-second diffusion
        # across different sampling rates. qx was tuned at _REF_FPS.
        dt_ref = 1.0 / self._REF_FPS
        qx_scale_factor = dt / dt_ref  # < 1 for higher fps → smaller per-step noise

        # ── Frame logger ──
        _EXTRA_FIELDS = [
            'S_t', 'R_eff', 'R_scaled', 'K_x1', 'K_x2',
            'qx_eff', 'qx_base', 'rv_base', 'rv_scaled',
            'q_phase', 'q_amp', 'nu_t', 'r_ratio',
            'lti_mode',
        ]
        logger = FrameLogger(n, extra_fields=_EXTRA_FIELDS)

        # ── Initial frequency (harmonic-aware) ──
        freq0_raw = self._coarse_freq(y, fs)
        freq0 = self._harmonic_refine(freq0_raw, y, fs)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))

        # ── Effective parameters ──
        eff = self._effective_params(fs, meta)
        qx = eff['qx'] * qx_scale_factor  # fps-normalized process noise
        R_init = eff['rv']
        rho = eff['rho']

        # ── State-space matrices ──
        H = np.array([[1.0, 0.0]], dtype=np.float64)
        I2 = np.eye(2, dtype=np.float64)

        # ── Storage arrays ──
        x_filt = np.zeros((n, 2), dtype=np.float64)
        P_filt = np.zeros((n, 2, 2), dtype=np.float64)
        x_pred_arr = np.zeros((n, 2), dtype=np.float64)
        P_pred_arr = np.zeros((n, 2, 2), dtype=np.float64)
        F_arr = np.zeros((n, 2, 2), dtype=np.float64)

        # Diagnostic traces
        diag_nis = np.zeros(n, dtype=np.float64)
        diag_lambda = np.ones(n, dtype=np.float64)
        diag_R = np.full(n, R_init, dtype=np.float64)
        diag_nu = np.full(n, self.NU_MAX, dtype=np.float64)
        diag_freq = np.full(n, freq0, dtype=np.float64)
        diag_gate = np.ones(n, dtype=np.float64)
        diag_phase_coh = np.ones(n, dtype=np.float64)
        diag_amp_stab = np.ones(n, dtype=np.float64)
        diag_qx = np.full(n, qx, dtype=np.float64)  # v5: adaptive Q tracking

        # ── Initial state ──
        x = np.zeros(2, dtype=np.float64)
        P = qx * I2.copy()

        # ── Adaptive R state ──
        R_t = float(R_init)
        C_e = float(R_init)

        # ── Adaptive Q state (v5: R-based) ──
        qx_adaptive = float(qx)     # Start with base qx

        # ── Kurtosis / Student-t state ──
        m2_ema = 1.0
        m4_ema = 3.0
        nu_t = float(self.NU_MAX)

        # ── Frequency state ──
        f_t = freq0
        freq_window = []
        FREQ_BUF_SIZE = max(1, int(10.0 * fs))
        freq_confirm = 0

        # ── Gate state ──
        amp_baseline = 0.0
        prev_phase = None

        # ── LTI convergence state (v6) ──
        lti_mode = False
        lti_stable_count = 0
        r_ratio_ema = 1.0
        frozen_R = float(R_init)
        frozen_C_e = float(R_init)
        frozen_Q = float(qx)
        surprise_ema = 1.0
        diag_lti = np.zeros(n, dtype=np.float64)  # 1.0 = LTI mode active

        # ══════════════════════════════════════════
        #  FORWARD PASS
        # ══════════════════════════════════════════
        for t in range(n):
            # ── 1. Frequency adaptation (self-gating, rate-limited) ──
            if self.FREQ_ADAPT and not lti_mode and t > 0 and t % FREQ_UPDATE_INTERVAL == 0:
                f_t, freq_confirm = self._adapt_frequency(
                    f_t, fs, p, freq_window, FREQ_BUF_SIZE, freq_confirm
                )
            f_t = float(np.clip(f_t, p.f_min, p.f_max))
            diag_freq[t] = f_t

            # ── 2. Build transition matrix (damped rotation) ──
            omega = 2.0 * np.pi * f_t * dt
            cos_w = np.cos(omega)
            sin_w = np.sin(omega)
            F = rho * np.array([[cos_w, -sin_w], [sin_w, cos_w]], dtype=np.float64)
            F_arr[t] = F

            # ── 3. Predict (with adaptive Q) ──
            Q_t = qx_adaptive * I2
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q_t
            x_pred_arr[t] = x_pred
            P_pred_arr[t] = P_pred

            # ── 4. Innovation ──
            y_t = float(y[t])
            y_pred_t = float((H @ x_pred).item())
            e_t = y_t - y_pred_t

            # ── 5. Innovation variance tracking & Adaptive R ──
            HP = float((H @ P_pred @ H.T).item())
            if t >= WARMUP_FRAMES:
                # Always track C_e (even in LTI mode) so that on LTI exit
                # R_t can immediately recover to the correct noise level.
                C_e = ALPHA_R * C_e + (1.0 - ALPHA_R) * (e_t ** 2)
                if not lti_mode:
                    R_mehra = C_e - HP
                    R_floor = self.R_ANCHOR_FRAC * R_init
                    R_t = max(R_mehra, R_floor)
            # In LTI mode: R_t stays frozen, but C_e tracks for fast recovery

            diag_R[t] = R_t
            S_t = HP + R_t
            S_t = max(S_t, 1e-12)

            # ── 6. Adaptive Q (SKIP in LTI mode — qx_adaptive stays frozen) ──
            if not lti_mode and self.ADAPTIVE_Q and t >= QX_ADAPT_WARMUP:
                r_ratio = R_t / max(R_init, 1e-12)
                q_scale = float(np.clip(
                    r_ratio ** self.QX_ADAPT_GAMMA,
                    self.QX_MIN_SCALE, self.QX_MAX_SCALE
                ))
                qx_adaptive = qx * q_scale
            diag_qx[t] = qx_adaptive

            # ── 7. Empirical NIS (always — used for surprise detection) ──
            C_e_safe = max(C_e, 1e-12)
            nis_empirical = (e_t ** 2) / C_e_safe
            diag_nis[t] = nis_empirical

            # ── 8. Adaptive ν_t (SKIP in LTI mode) ──
            if not lti_mode and t >= WARMUP_FRAMES:
                e_std = e_t / max(np.sqrt(C_e_safe), 1e-12)
                m2_ema = ALPHA_KAPPA * m2_ema + (1.0 - ALPHA_KAPPA) * (e_std ** 2)
                m4_ema = ALPHA_KAPPA * m4_ema + (1.0 - ALPHA_KAPPA) * (e_std ** 4)
                m2_safe = max(m2_ema, 1e-12)
                kurtosis_excess = (m4_ema / (m2_safe ** 2)) - 3.0
                kurtosis_excess = max(kurtosis_excess, 0.0)
                if kurtosis_excess > self.KAPPA_EPS:
                    nu_t = 6.0 / kurtosis_excess + 4.0
                else:
                    nu_t = self.NU_MAX
                nu_t = float(np.clip(nu_t, self.NU_MIN, self.NU_MAX))
            # In LTI mode: nu_t stays at last value (typically NU_MAX)

            diag_nu[t] = nu_t

            # ── 9. Student-t VB weight (λ=1.0 in LTI mode) ──
            if lti_mode:
                lambda_t = 1.0
            elif nu_t < self.NU_MAX - 1.0:
                lambda_t = (nu_t + 1.0) / (nu_t + nis_empirical)
                lambda_t = float(np.clip(lambda_t, self.LAMBDA_FLOOR, 1e6))
            else:
                lambda_t = 1.0
            diag_lambda[t] = lambda_t

            # ── 10. Kalman gain with R_eff ──
            R_eff = R_t / lambda_t
            S_eff = HP + R_eff
            S_eff = max(S_eff, 1e-12)
            K = (P_pred @ H.T) / S_eff  # (2, 1)

            # ── 11. Phase-coherence & amplitude-stability gate ──
            #        (SKIP in LTI mode — g_t = 1.0 always)
            g_t = 1.0
            q_phase = 1.0
            q_amp = 1.0

            if not lti_mode and t >= GATE_WARMUP:
                delta_phi_expected = omega
                x_trial = x_pred + K[:, 0] * e_t
                phi_trial = np.arctan2(x_trial[1], x_trial[0])
                phi_pred = np.arctan2(x_pred[1], x_pred[0])

                if prev_phase is not None:
                    delta_phi_actual = _angle_wrap(phi_trial - prev_phase)
                    phase_dev = abs(_angle_wrap(delta_phi_actual - delta_phi_expected))
                    q_phase = float(np.exp(-0.5 * (phase_dev / self.GATE_PHASE_SIGMA) ** 2))

                amp_trial = np.sqrt(x_trial[0] ** 2 + x_trial[1] ** 2)
                if amp_baseline > 1e-6:
                    amp_change = abs(amp_trial - amp_baseline) / amp_baseline
                    q_amp = float(np.exp(-0.5 * (amp_change / self.GATE_AMP_SIGMA) ** 2))

                q_raw = q_phase * q_amp
                if q_raw >= self.GATE_THRESHOLD:
                    g_t = 1.0
                else:
                    g_t = max(q_raw, self.GATE_FLOOR)

            diag_gate[t] = g_t
            diag_phase_coh[t] = q_phase
            diag_amp_stab[t] = q_amp

            # ── 12. Kalman update (standard when g_t=1.0 in LTI mode) ──
            x_upd = x_pred + g_t * K[:, 0] * e_t
            gK = g_t * K
            M = I2 - gK @ H
            R_eff_mat = np.array([[R_eff]], dtype=np.float64)
            P_upd = M @ P_pred @ M.T + (g_t ** 2) * K @ R_eff_mat @ K.T

            # PSD projection
            P_upd = 0.5 * (P_upd + P_upd.T)
            eigvals, eigvecs = np.linalg.eigh(P_upd)
            eigvals = np.clip(eigvals, self.EIG_FLOOR, None)
            P_upd = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Trace cap
            tr = float(np.trace(P_upd))
            if tr > self.TRACE_CAP and tr > 0:
                P_upd = P_upd * (self.TRACE_CAP / tr)

            x = x_upd
            P = P_upd
            x_filt[t] = x
            P_filt[t] = P

            # ── 13. Update gate state ──
            amp_current = np.sqrt(x[0] ** 2 + x[1] ** 2)
            if t == 0:
                amp_baseline = amp_current
            else:
                amp_baseline = (GATE_ALPHA_AMP * amp_baseline +
                                (1.0 - GATE_ALPHA_AMP) * amp_current)
            prev_phase = np.arctan2(x[1], x[0])

            # ── 14. Update frequency buffer ──
            freq_window.append(float(x_filt[t, 0]))
            if len(freq_window) > FREQ_BUF_SIZE:
                freq_window.pop(0)

            # ── 15. LTI convergence check (v6) ──
            if self.LTI_CONVERGE and t >= QX_ADAPT_WARMUP:
                r_ratio_now = R_t / max(R_init, 1e-12)
                r_ratio_ema = (LTI_ALPHA * r_ratio_ema
                               + (1.0 - LTI_ALPHA) * r_ratio_now)

                if not lti_mode:
                    # ── Check ENTER condition: sustained low r_ratio ──
                    if r_ratio_ema < self.LTI_ENTER_THRESH:
                        lti_stable_count += 1
                    else:
                        lti_stable_count = 0
                    if lti_stable_count >= LTI_ENTER_FRAMES:
                        lti_mode = True
                        frozen_R = R_t
                        frozen_C_e = C_e
                        frozen_Q = qx_adaptive
                else:
                    # ── Check EXIT condition: surprise detection ──
                    # Compare current innovation to frozen C_e (fixed reference).
                    # If innovations are consistently much larger than when we
                    # entered LTI mode, the signal has changed → unfreeze.
                    surprise_nis = (e_t ** 2) / max(frozen_C_e, 1e-12)
                    surprise_ema = (LTI_SURPRISE_ALPHA * surprise_ema
                                    + (1.0 - LTI_SURPRISE_ALPHA) * surprise_nis)
                    if surprise_ema > self.LTI_EXIT_SURPRISE:
                        lti_mode = False
                        lti_stable_count = 0
                        surprise_ema = 1.0
                        # Immediately recover R_t from tracked C_e
                        R_mehra = C_e - HP
                        R_floor = self.R_ANCHOR_FRAC * R_init
                        R_t = max(R_mehra, R_floor)

            diag_lti[t] = 1.0 if lti_mode else 0.0

            # ── Frame logging ──
            logger.log_state(
                t, x, P,
                y_t=y_t,
                y_pred=y_pred_t,
                v_t=e_t,
                nis=nis_empirical,
                lambda_t=lambda_t,
            )
            logger.log_trust(
                t,
                alpha_R=1.0,
                alpha_Q=qx_adaptive / max(qx, 1e-12),
                g_t=g_t,
                g_z=1.0,
                w_h=1.0,
            )
            logger.log_frame(t, fail_diverge=0, fail_slip=0, fail_lock=0, fail_double=0)
            logger.log_frame(
                t,
                S_t=S_eff,
                R_eff=R_eff,
                R_scaled=R_t,
                K_x1=float(K[0, 0]),
                K_x2=float(K[1, 0]),
                qx_eff=qx_adaptive,
                qx_base=qx,
                rv_base=R_init,
                rv_scaled=R_t,
                q_phase=q_phase,
                q_amp=q_amp,
                nu_t=nu_t,
                r_ratio=R_t / max(R_init, 1e-12),
                lti_mode=1.0 if lti_mode else 0.0,
            )

        # ══════════════════════════════════════════
        #  RTS BACKWARD SMOOTHER
        # ══════════════════════════════════════════
        x_smooth = np.copy(x_filt)
        P_smooth = np.copy(P_filt)

        for t in range(n - 2, -1, -1):
            F_next = F_arr[min(t + 1, n - 1)]
            P_pred_next = P_pred_arr[t + 1]
            try:
                P_pred_inv = np.linalg.pinv(P_pred_next)
            except np.linalg.LinAlgError:
                P_pred_inv = np.linalg.inv(P_pred_next + 1e-9 * I2)
            G = P_filt[t] @ F_next.T @ P_pred_inv
            x_smooth[t] += G @ (x_smooth[t + 1] - x_pred_arr[t + 1])
            P_smooth[t] += G @ (P_smooth[t + 1] - P_pred_arr[t + 1]) @ G.T

        # ══════════════════════════════════════════
        #  OUTPUT: signal_hat + track_hz
        # ══════════════════════════════════════════
        signal_hat = x_smooth[:, 0].copy()

        x1s = x_smooth[:, 0]
        x2s = x_smooth[:, 1]
        if n > 1:
            phase = np.unwrap(np.arctan2(x2s, x1s))
            dphi = np.diff(phase)
            inst_freq = (fs / (2.0 * np.pi)) * dphi
            track_hz = np.empty(n, dtype=np.float64)
            track_hz[0] = inst_freq[0] if inst_freq.size else freq0
            track_hz[1:] = inst_freq
        else:
            track_hz = np.full(n, freq0, dtype=np.float64)

        bad = ~np.isfinite(track_hz)
        if np.any(bad):
            track_hz[bad] = freq0
        track_hz = np.clip(track_hz, p.f_min, p.f_max)

        # Convert post_smooth_alpha to fps-invariant form.
        # The original alpha=0.88 was tuned at 20fps → tau = -1/ln(0.88) / 20 ≈ 0.39s.
        # At any fps, we want the same physical time constant.
        _alpha_cfg = getattr(p, 'post_smooth_alpha', 0.0) or 0.0
        if not (0.0 < _alpha_cfg < 1.0):
            _alpha_cfg = 0.88
        _tau_smooth = -1.0 / (np.log(max(_alpha_cfg, 1e-6)) * self._REF_FPS)
        alpha_used = float(np.exp(-1.0 / max(_tau_smooth * fs, 1e-6)))
        track_hz = self._apply_post_smoothing(track_hz, alpha_override=alpha_used)

        # ── Package output with diagnostics ──
        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        meta_payload["f0_raw"] = float(freq0_raw)
        meta_payload["f0_harmonic_refined"] = (freq0 != freq0_raw)
        meta_payload["freq_source"] = "narossm_phase"
        meta_payload["post_smooth_alpha_used"] = alpha_used
        meta_payload["narossm_diagnostics"] = {
            "nis_mean": float(np.mean(diag_nis)),
            "nis_median": float(np.median(diag_nis)),
            "lambda_lt1_frac": float(np.mean(diag_lambda < 1.0)),
            "lambda_mean": float(np.mean(diag_lambda)),
            "R_mean": float(np.mean(diag_R)),
            "R_std": float(np.std(diag_R)),
            "nu_mean": float(np.mean(diag_nu)),
            "nu_median": float(np.median(diag_nu)),
            "freq_mean": float(np.mean(diag_freq)),
            "freq_std": float(np.std(diag_freq)),
            "gate_mean": float(np.mean(diag_gate)),
            "gate_lt09_frac": float(np.mean(diag_gate < 0.9)),
            "gate_lt05_frac": float(np.mean(diag_gate < 0.5)),
            "phase_coh_mean": float(np.mean(diag_phase_coh)),
            "amp_stab_mean": float(np.mean(diag_amp_stab)),
            # v5 diagnostics
            "qx_adaptive_mean": float(np.mean(diag_qx)),
            "qx_adaptive_std": float(np.std(diag_qx)),
            "qx_adaptive_min": float(np.min(diag_qx)),
            "qx_adaptive_max": float(np.max(diag_qx)),
            "qx_scale_mean": float(np.mean(diag_qx) / max(qx, 1e-12)),
            # v6 LTI convergence diagnostics
            "lti_fraction": float(np.mean(diag_lti)),
            "lti_enter_frame": int(np.argmax(diag_lti)) if np.any(diag_lti > 0) else -1,
            "r_ratio_ema_final": float(r_ratio_ema),
        }

        # ── Save frame log ──
        aux_dir = (meta or {}).get('aux_save_dir')
        save_frame_logs = bool((meta or {}).get('save_frame_logs', True))
        trial_key = str((meta or {}).get('trial_key') or "").strip()
        if not trial_key:
            short_key, _ = derive_trial_identifiers(
                {"dataset_name": (meta or {}).get("dataset"),
                 "subject": (meta or {}).get("subject"),
                 "trial": (meta or {}).get("trial"),
                 "video_path": (meta or {}).get("data_file")},
                dataset_name=str((meta or {}).get("dataset", "")),
                sample_index=0,
            )
            trial_key = short_key
        if aux_dir and trial_key and save_frame_logs:
            log_dir = os.path.join(aux_dir, 'frame_logs')
            os.makedirs(log_dir, exist_ok=True)
            base_key = sanitize_trial_key(trial_key, fallback="trial")
            suffix_int = 0
            out_path = os.path.join(log_dir, f"{base_key}.npz")
            while os.path.exists(out_path):
                suffix_int += 1
                out_path = os.path.join(log_dir, f"{base_key}_{suffix_int}.npz")
            logger.save(out_path)
            sha256 = ""
            try:
                h = hashlib.sha256()
                with open(out_path, "rb") as fp:
                    for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                        h.update(chunk)
                sha256 = h.hexdigest()
            except Exception:
                pass
            update_frame_log_manifest(
                aux_dir=aux_dir,
                base_trial_key=base_key,
                actual_filename=os.path.basename(out_path),
                suffix=suffix_int,
                sha256=sha256,
            )

        return self._package(signal_hat, track_hz, meta_payload)

    # ──────────────────────────────────────────
    #  Harmonic-aware frequency refinement (v5)
    # ──────────────────────────────────────────
    def _harmonic_refine(
        self,
        freq0: float,
        y: np.ndarray,
        fs: float,
    ) -> float:
        """Detect and correct harmonic confusion in frequency initialisation.

        For low BPM breathing (< 10 BPM), the 2nd harmonic often dominates
        the Welch PSD because chest motion waveforms are non-sinusoidal.
        This method analyses the raw Welch PSD (not the blended freq0) to
        detect if the dominant peak has a subharmonic (fundamental) with
        significant power, and prefers the fundamental if so.

        Uses autocorrelation as a tiebreaker since autocorrelation naturally
        identifies the fundamental period, not harmonics.
        """
        p = self.params

        try:
            from scipy.signal import welch
            # Use long window for fine frequency resolution (critical for
            # distinguishing fundamental from harmonic at low BPM).
            # 60s window → df ≈ 0.017 Hz at 20fps, enough to resolve
            # the difference between 0.12 Hz and 0.24 Hz.
            nperseg = min(len(y), int(60.0 * fs))
            if nperseg < 8:
                return freq0

            freqs, psd = welch(y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            band = (freqs >= p.f_min) & (freqs <= p.f_max)
            if not np.any(band):
                return freq0

            f_b = freqs[band]
            p_b = psd[band]
            if len(f_b) < 3:
                return freq0
            df = f_b[1] - f_b[0]

            # Find the dominant Welch peak (raw, not blended)
            peak_idx = int(np.argmax(p_b))
            f_dominant = float(f_b[peak_idx])
            p_dominant = float(p_b[peak_idx])

            # Check subharmonic at f_dominant/2
            f_sub = f_dominant / 2.0
            if f_sub < p.f_min:
                return freq0  # Subharmonic out of respiratory band

            # Find power at the subharmonic (±2 bins tolerance)
            sub_mask = np.abs(f_b - f_sub) <= 2 * df
            if not np.any(sub_mask):
                return freq0

            sub_candidates = np.where(sub_mask)[0]
            best_sub_local = int(np.argmax(p_b[sub_candidates]))
            best_sub_idx = sub_candidates[best_sub_local]
            p_sub = float(p_b[best_sub_idx])
            f_sub_best = float(f_b[best_sub_idx])

            # Is the subharmonic strong enough to be the true fundamental?
            if p_sub < self.HARMONIC_POWER_RATIO * p_dominant:
                return freq0  # Subharmonic too weak — dominant is the real peak

            # Subharmonic has significant power → likely harmonic confusion.
            # Use autocorrelation as tiebreaker (autocorr naturally finds
            # the fundamental period, not harmonics).
            freq_ac, conf_ac, _ = self._autocorr_candidate(y, fs)
            if np.isfinite(freq_ac) and conf_ac > 0:
                dist_to_sub = abs(freq_ac - f_sub_best)
                dist_to_dom = abs(freq_ac - f_dominant)
                if dist_to_sub < dist_to_dom:
                    # Autocorrelation confirms: subharmonic is the fundamental
                    return float(np.clip(f_sub_best, p.f_min, p.f_max))

            # Strong subharmonic power even without autocorr confirmation
            if p_sub >= 0.5 * p_dominant:
                return float(np.clip(f_sub_best, p.f_min, p.f_max))

        except Exception:
            pass

        return freq0

    # ──────────────────────────────────────────
    #  Frequency adaptation (self-gating, rate-limited)
    # ──────────────────────────────────────────
    def _adapt_frequency(
        self,
        f_prev: float,
        fs: float,
        p,
        freq_window: list,
        buf_size: int,
        confirm_count: int = 0,
    ) -> tuple:
        """Self-gating spectral frequency adaptation.

        Only adapts when spectral evidence consistently shows a different
        frequency. On stable breathing, no adaptation occurs.
        """
        min_buf = max(int(self.FREQ_UPDATE_MIN_BUF_SEC * fs), 30)
        if len(freq_window) < min_buf:
            return f_prev, confirm_count

        buf = np.array(freq_window[-buf_size:], dtype=np.float64)

        try:
            from scipy.signal import welch
            nperseg = min(len(buf), int(4.0 * fs))
            if nperseg < 8:
                return f_prev, confirm_count
            freqs, psd = welch(buf, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            band_mask = (freqs >= p.f_min) & (freqs <= p.f_max)
            if not np.any(band_mask):
                return f_prev, confirm_count
            psd_band = psd[band_mask]
            freqs_band = freqs[band_mask]
            peak_idx = np.argmax(psd_band)
            f_spec = freqs_band[peak_idx]

            peak_power = psd_band[peak_idx]
            bg_power = np.median(psd_band)
            snr = peak_power / max(bg_power, 1e-12)
            alpha_f = float(np.clip(snr / (snr + 5.0), 0.05, 0.8))
        except Exception:
            return f_prev, confirm_count

        spectral_delta = abs(f_spec - f_prev)
        if spectral_delta < self.FREQ_MIN_DELTA:
            confirm_count = max(confirm_count - 1, 0)
            return f_prev, confirm_count

        confirm_count += 1
        if confirm_count < self.FREQ_CONFIRM_COUNT:
            return f_prev, confirm_count

        f_target = alpha_f * f_spec + (1.0 - alpha_f) * f_prev

        delta = f_target - f_prev
        if abs(delta) > self.FREQ_MAX_RATE:
            f_new = f_prev + self.FREQ_MAX_RATE * np.sign(delta)
        else:
            f_new = f_target

        confirm_count = 0
        return float(np.clip(f_new, p.f_min, p.f_max)), confirm_count
