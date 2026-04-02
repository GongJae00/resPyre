"""
PARH-OSSM v1 — Physiology-Aligned Regime-Adaptive Harmonic Oscillatory SSM.

An 8D linear SSM that decomposes camera-derived respiratory motion into:
  - Oscillatory drive:  K=2 harmonics  [h_c^(1), h_s^(1), h_c^(2), h_s^(2)]
  - Baseline trend:     constant-velocity  [b_t, ḃ_t]
  - Aperiodic residual: damped random walk  [r_t, ṙ_t]

Key properties:
  1. Exactly linear conditional on external f_t → standard KF + exact RTS.
  2. K=2 harmonic absorbs inhale/exhale asymmetry into model state.
  3. Disentangled uncertainty via 3 quality scores:
       q_obs → prior trust π_t (observation reliability)
       q_dyn → Q_osc scaling (dynamical novelty)
       q_osc → Q_aper scaling (oscillatory support)
  4. Dual output: z_osc (rate estimation), z_full (waveform reconstruction).
  5. Causal + smoothed outputs both stored.
  6. Per-component ablation flags for intent-aligned experiments.

State vector (8D):
  x = [h_c^(1), h_s^(1), h_c^(2), h_s^(2), b, ḃ, r, ṙ]

Observation model:
  y_t = H @ x_t + v_t,   H = [1, 0, 1, 0, 1, 0, 1, 0]

Design principle:
  R_t self-calibration (Mehra) is the PRIMARY mechanism.
  π_t = f(q_obs) is prior trust (pre-innovation quality).
  λ_t is posterior robustification (Student-t, post-innovation).
  After R_t normalises residuals, ν → ∞ (Gaussian) — confirming
  that adaptation has captured the noise structure.

Quality score boundary (v1):
  - R_selfcal is primary adaptive R (existing Mehra-based).
  - π_t = f(q_obs) as prior trust → R_eff = R_selfcal / π_t.
  - Q_osc = Q_osc_0 * g(q_dyn): oscillatory process noise.
  - Q_aper = Q_aper_0 * h(1 - q_osc): aperiodic absorbs non-oscillatory surprise.
  - λ_t from Student-t VB: posterior outlier downweighting.
  This is a conservative design: R_selfcal remains the main scale,
  q_obs provides mild prior trust modulation. The paper narrative
  and code are aligned on this boundary.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from ..core.base import _BaseOscillatorHead

MODEL_VERSION = "parh_ossm_v1"


def _angle_wrap(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class oscillator_PARH_OSSM(_BaseOscillatorHead):
    """PARH-OSSM v1: Physiology-Aligned Regime-Adaptive Harmonic SSM."""

    head_key = "parh_ossm"

    # ── Dimensions ──
    STATE_DIM: int = 8
    HC1, HS1, HC2, HS2, B, BDOT, R, RDOT = range(8)

    # ── Time-scale reference ──
    _REF_FPS: float = 20.0

    # ── Adaptive R (Mehra self-calibration — PRIMARY mechanism) ──
    TAU_R_SEC: float = 2.5
    R_ANCHOR_FRAC: float = 0.3

    # ── EW-MAD robust scale ──
    TAU_MAD_SEC: float = 2.5  # Same as R for consistency

    # ── Kurtosis / Student-t ──
    TAU_KAPPA_SEC: float = 3.0
    NU_MIN: float = 3.0
    NU_MAX: float = 200.0
    VB_ITERS: int = 3

    # ── Quality score hyperparameters ──
    # q_obs
    Q_OBS_ROBUST_Z_SCALE: float = 2.0   # robust_z sensitivity
    Q_OBS_JUMP_SCALE: float = 3.0       # signal jump sensitivity
    Q_OBS_MIN: float = 0.05             # floor

    # q_dyn
    Q_DYN_FREQ_WEIGHT: float = 5.0      # freq deviation sensitivity
    Q_DYN_AMP_WEIGHT: float = 2.0       # amplitude slope sensitivity
    Q_DYN_FREQ_REF_HZ: float = 0.02     # reference freq step

    # q_osc (oscillatory support)
    Q_OSC_PHASE_WEIGHT: float = 0.5
    Q_OSC_AMP_WEIGHT: float = 0.5
    GATE_PHASE_SIGMA: float = 0.8
    GATE_AMP_SIGMA: float = 3.0
    TAU_AMP_SEC: float = 1.5
    GATE_WARMUP_SEC: float = 0.75

    # ── Disentangled Q mapping ──
    Q_DYN_GAMMA: float = 0.5            # Q_osc scaling strength from q_dyn
    Q_APER_GAMMA: float = 1.0           # Q_aper scaling strength from (1-q_osc)

    # ── Legacy coupled Q (ablation only) ──
    QX_ADAPT_GAMMA_LEGACY: float = 0.5

    # ── Frequency adaptation ──
    FREQ_UPDATE_INTERVAL_SEC: float = 2.0
    FREQ_CONFIRM_COUNT: int = 3
    FREQ_MAX_STEP_HZ: float = 0.03

    # ── Harmonic-aware init ──
    HARMONIC_POWER_RATIO: float = 0.15

    # ── Warmup ──
    WARMUP_SEC: float = 2.5
    QX_ADAPT_WARMUP_SEC: float = 3.0
    FREQ_INIT_SEC: float = 10.0   # Warm-up window for initial frequency estimation

    # ── Component-specific process noise scales ──
    Q_HARMONIC1_SCALE: float = 1.0
    Q_HARMONIC2_SCALE: float = 0.5
    Q_BASELINE_POS: float = 1e-4
    Q_BASELINE_VEL: float = 1e-5
    Q_RESIDUAL_POS: float = 0.1
    Q_RESIDUAL_VEL: float = 0.01

    # ── Damping for aperiodic residual ──
    TAU_RESIDUAL_SEC: float = 5.0

    # ── Ablation flags ──
    ENABLE_HARMONIC2: bool = True
    ENABLE_BASELINE: bool = True
    ENABLE_RESIDUAL: bool = True
    ENABLE_ADAPT_R: bool = True
    ENABLE_DISENTANGLED_Q: bool = True   # v1 default: disentangled
    ENABLE_LEGACY_COUPLED_Q: bool = False  # ablation: old R_t/R_init coupled
    ENABLE_STUDENT_T: bool = True
    ENABLE_FREQ_ADAPT: bool = True

    # ─────────────────────────────────────────────
    #  SSM MATRIX BUILDERS
    # ─────────────────────────────────────────────

    def _build_F(self, omega: float, dt: float, rho: float) -> np.ndarray:
        """8x8 state transition: block-diagonal harmonic + baseline + residual."""
        F = np.zeros((8, 8), dtype=np.float64)

        cos1 = np.cos(omega * dt)
        sin1 = np.sin(omega * dt)
        F[0, 0] = rho * cos1;  F[0, 1] = -rho * sin1
        F[1, 0] = rho * sin1;  F[1, 1] = rho * cos1

        if self.ENABLE_HARMONIC2:
            cos2 = np.cos(2.0 * omega * dt)
            sin2 = np.sin(2.0 * omega * dt)
            F[2, 2] = rho * cos2;  F[2, 3] = -rho * sin2
            F[3, 2] = rho * sin2;  F[3, 3] = rho * cos2

        if self.ENABLE_BASELINE:
            F[4, 4] = 1.0;  F[4, 5] = dt
            F[5, 5] = 1.0

        if self.ENABLE_RESIDUAL:
            alpha_r = np.exp(-dt / max(self.TAU_RESIDUAL_SEC, 1e-6))
            F[6, 6] = alpha_r;  F[6, 7] = dt * alpha_r
            F[7, 7] = alpha_r

        return F

    def _build_Q_disentangled(
        self, qx: float, dt: float,
        q_dyn: float, q_osc: float,
    ) -> np.ndarray:
        """8x8 process noise with disentangled quality-driven scaling.

        Q_osc components scaled by g(q_dyn):
            Q_osc = Q_osc_0 * (1 + Q_DYN_GAMMA * q_dyn)
        Q_aper components scaled by h(1 - q_osc):
            Q_aper = Q_aper_0 * (1 + Q_APER_GAMMA * (1 - q_osc))
        """
        Q = np.zeros((8, 8), dtype=np.float64)

        # Oscillatory noise: higher when dynamics are changing (q_dyn high)
        osc_scale = 1.0 + self.Q_DYN_GAMMA * q_dyn
        q_h1 = qx * self.Q_HARMONIC1_SCALE * osc_scale
        Q[0, 0] = q_h1;  Q[1, 1] = q_h1

        if self.ENABLE_HARMONIC2:
            q_h2 = qx * self.Q_HARMONIC2_SCALE * osc_scale
            Q[2, 2] = q_h2;  Q[3, 3] = q_h2

        # Baseline (always slow, independent of quality scores)
        if self.ENABLE_BASELINE:
            dt_scale = dt / (1.0 / self._REF_FPS)
            Q[4, 4] = self.Q_BASELINE_POS * dt_scale
            Q[5, 5] = self.Q_BASELINE_VEL * dt_scale

        # Aperiodic residual: absorbs when oscillatory support is low
        if self.ENABLE_RESIDUAL:
            aper_scale = 1.0 + self.Q_APER_GAMMA * (1.0 - q_osc)
            Q[6, 6] = self.Q_RESIDUAL_POS * qx * aper_scale
            Q[7, 7] = self.Q_RESIDUAL_VEL * qx * aper_scale

        return Q

    def _build_Q_legacy_coupled(
        self, qx: float, dt: float, r_ratio: float,
    ) -> np.ndarray:
        """Legacy coupled Q: all components scale with R_t/R_init ratio.
        Kept for ablation comparison only.
        """
        q_dyn_scale = 1.0 + self.QX_ADAPT_GAMMA_LEGACY * max(r_ratio - 1.0, 0.0)
        q_res_scale = max(1.0, 2.0 - q_dyn_scale)

        Q = np.zeros((8, 8), dtype=np.float64)
        q_h1 = qx * self.Q_HARMONIC1_SCALE * q_dyn_scale
        Q[0, 0] = q_h1;  Q[1, 1] = q_h1
        if self.ENABLE_HARMONIC2:
            q_h2 = qx * self.Q_HARMONIC2_SCALE * q_dyn_scale
            Q[2, 2] = q_h2;  Q[3, 3] = q_h2
        if self.ENABLE_BASELINE:
            dt_scale = dt / (1.0 / self._REF_FPS)
            Q[4, 4] = self.Q_BASELINE_POS * dt_scale
            Q[5, 5] = self.Q_BASELINE_VEL * dt_scale
        if self.ENABLE_RESIDUAL:
            Q[6, 6] = self.Q_RESIDUAL_POS * qx * q_res_scale
            Q[7, 7] = self.Q_RESIDUAL_VEL * qx * q_res_scale
        return Q

    def _build_H(self) -> np.ndarray:
        """1x8 observation matrix: y = h_c1 + h_c2 + b + r."""
        H = np.zeros((1, 8), dtype=np.float64)
        H[0, self.HC1] = 1.0
        if self.ENABLE_HARMONIC2:
            H[0, self.HC2] = 1.0
        if self.ENABLE_BASELINE:
            H[0, self.B] = 1.0
        if self.ENABLE_RESIDUAL:
            H[0, self.R] = 1.0
        return H

    # ─────────────────────────────────────────────
    #  OUTPUT EXTRACTION
    # ─────────────────────────────────────────────

    def _extract_outputs(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Dual output from state trajectory.
        z_osc = h_c1 + h_c2 (rate estimation)
        z_full = z_osc + b + r (waveform fidelity)
        """
        z_osc = x[:, self.HC1].copy()
        if self.ENABLE_HARMONIC2:
            z_osc += x[:, self.HC2]
        z_full = z_osc.copy()
        if self.ENABLE_BASELINE:
            z_full += x[:, self.B]
        if self.ENABLE_RESIDUAL:
            z_full += x[:, self.R]
        return z_osc, z_full

    def _compute_inst_freq(self, x: np.ndarray, fs: float, freq0: float) -> np.ndarray:
        """Inst. freq from fundamental harmonic phase: atan2(h_s1, h_c1)."""
        n = x.shape[0]
        if n < 2:
            return np.full(n, freq0, dtype=np.float64)
        phase = np.unwrap(np.arctan2(x[:, self.HS1], x[:, self.HC1]))
        dphi = np.diff(phase)
        inst_freq = (fs / (2.0 * np.pi)) * dphi
        track_hz = np.empty(n, dtype=np.float64)
        track_hz[0] = inst_freq[0] if inst_freq.size else freq0
        track_hz[1:] = inst_freq
        return track_hz

    # ─────────────────────────────────────────────
    #  QUALITY SCORES
    # ─────────────────────────────────────────────

    def _compute_q_obs(
        self, e_t: float, robust_scale: float,
        y_t: float, y_prev: float, signal_scale: float,
    ) -> float:
        """q_obs(t): observation reliability [0, 1].

        Based on:
          1. Robust standardised innovation: |e_t| / robust_scale
          2. Signal jump: |y_t - y_prev| / signal_scale
        Higher q_obs = more trustworthy observation.
        """
        # Robust z-score of innovation
        robust_z = abs(e_t) / max(robust_scale, 1e-12)
        term_z = np.exp(-0.5 * (robust_z / self.Q_OBS_ROBUST_Z_SCALE) ** 2)

        # Signal jump detector
        jump = abs(y_t - y_prev) / max(signal_scale, 1e-12)
        term_jump = np.exp(-0.5 * (jump / self.Q_OBS_JUMP_SCALE) ** 2)

        q_obs = float(np.clip(term_z * term_jump, self.Q_OBS_MIN, 1.0))
        return q_obs

    def _compute_q_dyn(
        self, freq_candidate: float, freq_t: float,
        amp_t: float, amp_baseline: float,
    ) -> float:
        """q_dyn(t): dynamical novelty [0, 1].

        Higher q_dyn = physiology is changing → increase Q_osc.
        Based on:
          1. Frequency deviation from current track
          2. Amplitude envelope change
        """
        freq_dev = abs(freq_candidate - freq_t) / max(self.Q_DYN_FREQ_REF_HZ, 1e-6)
        amp_slope = abs(amp_t - amp_baseline) / max(amp_baseline, 1e-6)

        raw = self.Q_DYN_FREQ_WEIGHT * freq_dev + self.Q_DYN_AMP_WEIGHT * amp_slope
        q_dyn = float(np.clip(1.0 - np.exp(-raw), 0.0, 1.0))
        return q_dyn

    def _compute_q_osc(
        self, phase_coh: float, amp_stab: float,
    ) -> float:
        """q_osc(t): oscillatory support [0, 1].

        Higher q_osc = current segment is well-described by oscillator.
        Based on:
          1. Phase coherence (Gaussian kernel on phase prediction error)
          2. Amplitude stability (Gaussian kernel on amplitude deviation)
        """
        q_osc = float(np.clip(
            self.Q_OSC_PHASE_WEIGHT * phase_coh + self.Q_OSC_AMP_WEIGHT * amp_stab,
            0.0, 1.0
        ))
        return q_osc

    # ─────────────────────────────────────────────
    #  ROBUST SCALE & STUDENT-T
    # ─────────────────────────────────────────────

    @staticmethod
    def _ew_mad_update(abs_e: float, mad_prev: float, alpha: float) -> float:
        """Exponentially-weighted MAD: robust innovation scale."""
        return alpha * mad_prev + (1.0 - alpha) * abs_e

    def _student_t_vb_update(
        self, e_t: float, S_eff: float, nu_t: float,
        x: np.ndarray, P: np.ndarray, H: np.ndarray, I_D: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Student-t VB robust update. Returns (x_upd, P_upd, lambda_t)."""
        lambda_t = 1.0
        for _ in range(self.VB_ITERS):
            lambda_t = (nu_t + 1.0) / (nu_t + (e_t ** 2) / max(S_eff * lambda_t, 1e-12))
            lambda_t = float(np.clip(lambda_t, 0.01, 100.0))

        R_eff_vb = S_eff / max(lambda_t, 1e-12)
        S_vb = float(H @ P @ H.T) + R_eff_vb
        if S_vb <= 1e-12 or not np.isfinite(S_vb):
            S_vb = 1e-12
        K_vb = (P @ H.T) / S_vb
        x_upd = x + K_vb[:, 0] * e_t
        P_upd = (I_D - K_vb @ H) @ P
        P_upd = 0.5 * (P_upd + P_upd.T)
        return x_upd, P_upd, lambda_t

    # ─────────────────────────────────────────────
    #  MAIN RUN
    # ─────────────────────────────────────────────

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
        D = self.STATE_DIM

        # ── Per-sample EMA constants ──
        ALPHA_R = float(np.exp(-dt / max(self.TAU_R_SEC, 1e-6)))
        ALPHA_MAD = float(np.exp(-dt / max(self.TAU_MAD_SEC, 1e-6)))
        ALPHA_KAPPA = float(np.exp(-dt / max(self.TAU_KAPPA_SEC, 1e-6)))
        ALPHA_AMP = float(np.exp(-dt / max(self.TAU_AMP_SEC, 1e-6)))
        WARMUP = max(1, int(self.WARMUP_SEC * fs))
        QX_WARMUP = max(1, int(self.QX_ADAPT_WARMUP_SEC * fs))
        FREQ_INTERVAL = max(1, int(self.FREQ_UPDATE_INTERVAL_SEC * fs))
        GATE_WARMUP = max(1, int(self.GATE_WARMUP_SEC * fs))

        # fps-invariant process noise
        dt_ref = 1.0 / self._REF_FPS
        qx = (p.qx if hasattr(p, 'qx') else 0.005) * (dt / dt_ref)

        # ── Initial frequency (warm-up window only, NOT full signal) ──
        init_len = min(n, max(int(self.FREQ_INIT_SEC * fs), int(4.0 * fs)))
        y_init = y[:init_len]
        freq0_raw = self._coarse_freq(y_init, fs)
        freq0 = self._harmonic_refine(freq0_raw, y_init, fs)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))
        omega0 = 2.0 * np.pi * freq0

        # ── Effective parameters ──
        eff = self._effective_params(fs, meta)
        rho = eff['rho']
        R_init = eff['rv']

        # ── Matrices ──
        H = self._build_H()
        I_D = np.eye(D, dtype=np.float64)

        # ── Storage ──
        x_filt = np.zeros((n, D), dtype=np.float64)
        P_filt = np.zeros((n, D, D), dtype=np.float64)
        x_pred_arr = np.zeros((n, D), dtype=np.float64)
        P_pred_arr = np.zeros((n, D, D), dtype=np.float64)
        F_arr = np.zeros((n, D, D), dtype=np.float64)

        # Diagnostic arrays (per-sample)
        diag_R = np.zeros(n, dtype=np.float64)
        diag_nis = np.ones(n, dtype=np.float64)
        diag_pi = np.ones(n, dtype=np.float64)     # prior trust
        diag_lambda = np.ones(n, dtype=np.float64)  # posterior Student-t
        diag_nu = np.full(n, self.NU_MAX, dtype=np.float64)
        diag_freq = np.full(n, freq0, dtype=np.float64)
        diag_q_obs = np.ones(n, dtype=np.float64)
        diag_q_dyn = np.zeros(n, dtype=np.float64)
        diag_q_osc = np.ones(n, dtype=np.float64)

        # ── Init state ──
        x = np.zeros(D, dtype=np.float64)
        Q0 = self._build_Q_disentangled(qx, dt, q_dyn=0.0, q_osc=1.0)
        P = Q0.copy()

        # ── Running statistics ──
        R_t = R_init
        C_e = R_init       # EMA of squared innovations
        mad_e = np.sqrt(R_init)  # EW-MAD robust scale (in units of |e|)
        kurtosis_ema = 3.0
        nu_t = self.NU_MAX
        omega_t = omega0
        freq_t = freq0

        # Signal scale for q_obs jump detector
        signal_scale = max(float(np.std(y[:min(n, int(3.0 * fs))])), 1e-6)
        y_prev = y[0] if n > 0 else 0.0

        # Phase/amplitude state for q_osc
        phase_prev = 0.0
        amp_baseline = 0.0
        gate_init = False

        # Frequency adaptation state
        freq_candidate = freq0
        freq_confirm = 0

        # ══════════════════════════════════════════
        #  FORWARD PASS
        # ══════════════════════════════════════════
        for t in range(n):
            # Build F
            F = self._build_F(omega_t, dt, rho)
            F_arr[t] = F

            # Initial Q (updated below if disentangled Q is active)
            Q = self._build_Q_disentangled(qx, dt, q_dyn=0.0, q_osc=1.0)

            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            x_pred_arr[t] = x_pred
            P_pred_arr[t] = P_pred

            # Innovation
            y_t = y[t]
            e_t = y_t - float(H @ x_pred)

            # Model-based innovation variance
            HP = float(H @ P_pred @ H.T)

            # ── Empirical innovation variance (EMA) ──
            C_e = ALPHA_R * C_e + (1.0 - ALPHA_R) * (e_t ** 2)
            C_e_safe = max(C_e, 1e-12)

            # ── EW-MAD robust scale ──
            mad_e = self._ew_mad_update(abs(e_t), mad_e, ALPHA_MAD)
            robust_scale = max(mad_e * 1.4826, 1e-12)  # MAD → σ conversion

            # ── Empirical NIS ──
            nis_empirical = (e_t ** 2) / C_e_safe

            # ── Adaptive R (Mehra self-calibration — PRIMARY) ──
            if self.ENABLE_ADAPT_R and t >= WARMUP:
                R_mehra = C_e - HP
                R_floor = self.R_ANCHOR_FRAC * R_init
                R_t = max(R_mehra, R_floor)
            else:
                R_t = R_init

            # ── Phase coherence & amplitude stability (for q_osc) ──
            hc1_pred = x_pred[self.HC1]
            hs1_pred = x_pred[self.HS1]
            amp_t = float(np.sqrt(hc1_pred ** 2 + hs1_pred ** 2))
            phase_t = float(np.arctan2(hs1_pred, hc1_pred))

            if t >= GATE_WARMUP and gate_init:
                expected_phase = phase_prev + omega_t * dt
                phase_err = abs(_angle_wrap(phase_t - expected_phase))
                phase_coh = float(np.exp(-0.5 * (phase_err / self.GATE_PHASE_SIGMA) ** 2))

                amp_baseline = ALPHA_AMP * amp_baseline + (1.0 - ALPHA_AMP) * amp_t
                amp_dev = abs(amp_t - amp_baseline) / max(amp_baseline, 1e-6)
                amp_stab = float(np.exp(-0.5 * (amp_dev / self.GATE_AMP_SIGMA) ** 2))
            else:
                phase_coh = 1.0
                amp_stab = 1.0
                if not gate_init and amp_t > 1e-8:
                    amp_baseline = amp_t
                    gate_init = True

            phase_prev = phase_t

            # ── Compute 3 quality scores ──
            q_obs_t = self._compute_q_obs(e_t, robust_scale, y_t, y_prev, signal_scale)
            q_dyn_t = self._compute_q_dyn(freq_candidate, freq_t, amp_t, amp_baseline) if t >= WARMUP else 0.0
            q_osc_t = self._compute_q_osc(phase_coh, amp_stab)

            # ── Prior trust π_t = f(q_obs_t) ──
            # Conservative: π_t = q_obs_t. Low q_obs inflates R_eff,
            # reducing Kalman gain for unreliable observations.
            pi_t = q_obs_t

            # ── Adaptive Q ──
            if t >= QX_WARMUP:
                if self.ENABLE_DISENTANGLED_Q:
                    Q = self._build_Q_disentangled(qx, dt, q_dyn_t, q_osc_t)
                elif self.ENABLE_LEGACY_COUPLED_Q:
                    r_ratio = R_t / max(R_init, 1e-12)
                    Q = self._build_Q_legacy_coupled(qx, dt, r_ratio)
                # else: keep default Q (no adaptation)

                # Re-predict with updated Q
                P_pred = F @ P @ F.T + Q
                P_pred_arr[t] = P_pred

            # ── Effective R with prior trust ──
            R_eff = R_t / max(pi_t, 1e-6)
            S_eff = float(H @ P_pred @ H.T) + R_eff
            if S_eff <= 1e-12 or not np.isfinite(S_eff):
                S_eff = 1e-12

            # ── Kurtosis → Student-t ν ──
            if self.ENABLE_STUDENT_T and t >= WARMUP:
                e_std = e_t / max(np.sqrt(C_e_safe), 1e-12)
                kurtosis_ema = ALPHA_KAPPA * kurtosis_ema + (1.0 - ALPHA_KAPPA) * (e_std ** 4)
                excess_k = max(kurtosis_ema - 3.0, 0.0)
                nu_t = max(self.NU_MIN, 6.0 / excess_k + 4.0) if excess_k > 0.01 else self.NU_MAX
                nu_t = float(np.clip(nu_t, self.NU_MIN, self.NU_MAX))

            # ── Update ──
            if self.ENABLE_STUDENT_T and nu_t < self.NU_MAX - 1.0:
                x, P, lambda_t = self._student_t_vb_update(
                    e_t, S_eff, nu_t, x_pred.copy(), P_pred.copy(), H, I_D
                )
            else:
                lambda_t = 1.0
                K = (P_pred @ H.T) / S_eff
                x = x_pred + K[:, 0] * e_t
                P = (I_D - K @ H) @ P_pred
                P = 0.5 * (P + P.T)

            # Covariance floor
            for i in range(D):
                if P[i, i] < 1e-12 or not np.isfinite(P[i, i]):
                    P[i, i] = 1e-12

            x_filt[t] = x
            P_filt[t] = P

            # ── Frequency adaptation ──
            if self.ENABLE_FREQ_ADAPT and t > 0 and t % FREQ_INTERVAL == 0:
                win_start = max(0, t - int(5.0 * fs))
                y_win = y[win_start:t + 1]
                if len(y_win) >= int(2.0 * fs):
                    f_spec = self._coarse_freq(y_win, fs)
                    f_spec = float(np.clip(f_spec, p.f_min, p.f_max))
                    if abs(f_spec - freq_candidate) < 0.02:
                        freq_confirm += 1
                    else:
                        freq_candidate = f_spec
                        freq_confirm = 1
                    if freq_confirm >= self.FREQ_CONFIRM_COUNT:
                        step = np.clip(freq_candidate - freq_t,
                                       -self.FREQ_MAX_STEP_HZ, self.FREQ_MAX_STEP_HZ)
                        freq_t = float(np.clip(freq_t + step, p.f_min, p.f_max))
                        omega_t = 2.0 * np.pi * freq_t
                        freq_confirm = 0

            # ── Store diagnostics ──
            diag_R[t] = R_t
            diag_nis[t] = nis_empirical
            diag_pi[t] = pi_t
            diag_lambda[t] = lambda_t
            diag_nu[t] = nu_t
            diag_freq[t] = freq_t
            diag_q_obs[t] = q_obs_t
            diag_q_dyn[t] = q_dyn_t
            diag_q_osc[t] = q_osc_t
            y_prev = y_t

        # ══════════════════════════════════════════
        #  CAUSAL OUTPUTS (before smoothing)
        # ══════════════════════════════════════════
        z_osc_causal, z_full_causal = self._extract_outputs(x_filt)
        track_hz_causal = self._compute_inst_freq(x_filt, fs, freq0)
        bad = ~np.isfinite(track_hz_causal)
        if np.any(bad):
            track_hz_causal[bad] = freq0
        track_hz_causal = np.clip(track_hz_causal, p.f_min, p.f_max)

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
                P_pred_inv = np.linalg.inv(P_pred_next + 1e-9 * I_D)
            G = P_filt[t] @ F_next.T @ P_pred_inv
            x_smooth[t] += G @ (x_smooth[t + 1] - x_pred_arr[t + 1])
            P_smooth[t] += G @ (P_smooth[t + 1] - P_pred_arr[t + 1]) @ G.T

        # ══════════════════════════════════════════
        #  SMOOTHED OUTPUTS
        # ══════════════════════════════════════════
        z_osc_smoothed, z_full_smoothed = self._extract_outputs(x_smooth)
        track_hz_smoothed = self._compute_inst_freq(x_smooth, fs, freq0)
        bad = ~np.isfinite(track_hz_smoothed)
        if np.any(bad):
            track_hz_smoothed[bad] = freq0
        track_hz_smoothed = np.clip(track_hz_smoothed, p.f_min, p.f_max)

        # Post-smoothing of freq track (fps-invariant)
        _alpha_cfg = getattr(p, 'post_smooth_alpha', 0.0) or 0.0
        if not (0.0 < _alpha_cfg < 1.0):
            _alpha_cfg = 0.88
        _tau_smooth = -1.0 / (np.log(max(_alpha_cfg, 1e-6)) * self._REF_FPS)
        alpha_used = float(np.exp(-1.0 / max(_tau_smooth * fs, 1e-6)))
        track_hz_smoothed = self._apply_post_smoothing(track_hz_smoothed, alpha_override=alpha_used)

        # Primary output: smoothed z_osc for rate estimation
        signal_hat = z_osc_smoothed
        track_hz = track_hz_smoothed

        # ══════════════════════════════════════════
        #  PACKAGE
        # ══════════════════════════════════════════
        # Determine which adaptive modules are active
        active_modules = []
        if self.ENABLE_ADAPT_R: active_modules.append("adapt_R")
        if self.ENABLE_DISENTANGLED_Q: active_modules.append("disentangled_Q")
        if self.ENABLE_LEGACY_COUPLED_Q: active_modules.append("legacy_coupled_Q")
        if self.ENABLE_STUDENT_T: active_modules.append("student_t")
        if self.ENABLE_FREQ_ADAPT: active_modules.append("freq_adapt")
        if self.ENABLE_HARMONIC2: active_modules.append("harmonic2")
        if self.ENABLE_BASELINE: active_modules.append("baseline")
        if self.ENABLE_RESIDUAL: active_modules.append("residual")

        meta_payload = dict(meta or {})
        meta_payload["f0"] = freq0
        meta_payload["f0_raw"] = float(freq0_raw)
        meta_payload["freq_source"] = "parh_ossm_phase"
        meta_payload["post_smooth_alpha_used"] = alpha_used
        meta_payload["model_version"] = MODEL_VERSION
        meta_payload["output_semantics"] = "smoothed"
        meta_payload["warmup_frames"] = WARMUP
        meta_payload["active_modules"] = active_modules

        # Scalar diagnostics (JSON-safe)
        meta_payload["parh_ossm_diagnostics"] = {
            "nis_mean": float(np.mean(diag_nis)),
            "nis_median": float(np.median(diag_nis)),
            "pi_mean": float(np.mean(diag_pi)),
            "pi_lt09_frac": float(np.mean(diag_pi < 0.9)),
            "lambda_mean": float(np.mean(diag_lambda)),
            "lambda_lt1_frac": float(np.mean(diag_lambda < 1.0)),
            "R_mean": float(np.mean(diag_R)),
            "R_std": float(np.std(diag_R)),
            "nu_mean": float(np.mean(diag_nu)),
            "nu_median": float(np.median(diag_nu)),
            "freq_mean": float(np.mean(diag_freq)),
            "freq_std": float(np.std(diag_freq)),
            "q_obs_mean": float(np.mean(diag_q_obs)),
            "q_dyn_mean": float(np.mean(diag_q_dyn)),
            "q_osc_mean": float(np.mean(diag_q_osc)),
            "energy_h1": float(np.mean(x_smooth[:, self.HC1] ** 2)),
            "energy_h2": float(np.mean(x_smooth[:, self.HC2] ** 2)) if self.ENABLE_HARMONIC2 else 0.0,
            "energy_baseline": float(np.mean(x_smooth[:, self.B] ** 2)) if self.ENABLE_BASELINE else 0.0,
            "energy_residual": float(np.mean(x_smooth[:, self.R] ** 2)) if self.ENABLE_RESIDUAL else 0.0,
        }

        result = self._package(signal_hat, track_hz, meta_payload)

        # Numpy arrays (not JSON-serialisable — attached after _package)
        result["z_osc"] = z_osc_smoothed
        result["z_full"] = z_full_smoothed
        result["z_osc_causal"] = z_osc_causal
        result["z_full_causal"] = z_full_causal
        result["z_osc_smoothed"] = z_osc_smoothed
        result["z_full_smoothed"] = z_full_smoothed
        result["track_hz_causal"] = track_hz_causal
        result["decomposition"] = {
            "h1": x_smooth[:, self.HC1].copy(),
            "h2": x_smooth[:, self.HC2].copy() if self.ENABLE_HARMONIC2 else np.zeros(n),
            "baseline": x_smooth[:, self.B].copy() if self.ENABLE_BASELINE else np.zeros(n),
            "residual": x_smooth[:, self.R].copy() if self.ENABLE_RESIDUAL else np.zeros(n),
        }
        result["diagnostics"] = {
            "R_t": diag_R,
            "pi_t": diag_pi,
            "lambda_t": diag_lambda,
            "nu_t": diag_nu,
            "q_obs_t": diag_q_obs,
            "q_dyn_t": diag_q_dyn,
            "q_osc_t": diag_q_osc,
            "freq_t": diag_freq,
            "nis_empirical_t": diag_nis,
        }

        return result

    # ─────────────────────────────────────────────
    #  HARMONIC-AWARE FREQUENCY REFINEMENT
    # ─────────────────────────────────────────────

    def _harmonic_refine(self, freq0: float, y: np.ndarray, fs: float) -> float:
        """Detect and correct harmonic confusion in frequency init."""
        p = self.params
        try:
            from scipy.signal import welch
            nperseg = min(len(y), int(60.0 * fs))
            if nperseg < 8:
                return freq0
            freqs, psd = welch(y, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            band = (freqs >= p.f_min) & (freqs <= p.f_max)
            if not np.any(band):
                return freq0
            f_b, p_b = freqs[band], psd[band]
            if len(f_b) < 3:
                return freq0
            peak_idx = int(np.argmax(p_b))
            f_dominant = float(f_b[peak_idx])
            p_dominant = float(p_b[peak_idx])
            f_sub = f_dominant / 2.0
            if f_sub < p.f_min:
                return freq0
            sub_idx = int(np.argmin(np.abs(f_b - f_sub)))
            p_sub = float(p_b[sub_idx])
            if p_sub >= self.HARMONIC_POWER_RATIO * p_dominant:
                lag_dom = int(round(fs / f_dominant))
                lag_sub = int(round(fs / f_sub))
                if lag_sub < len(y) - 1:
                    y_norm = y - np.mean(y)
                    acf = np.correlate(y_norm, y_norm, mode='full')
                    acf = acf[len(y_norm) - 1:]
                    acf = acf / max(acf[0], 1e-12)
                    if lag_sub < len(acf) and lag_dom < len(acf):
                        if acf[lag_sub] > acf[lag_dom] * 0.8:
                            return f_sub
        except Exception:
            pass
        return freq0
