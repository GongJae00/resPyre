"""
Quality-Aware Robust Oscillatory Bayesian Filter — Integration Head.

Wires: SSM predict → Trust allocate → Robust update → Decode → Log.
Register as 'robust_ossm' in HEAD_REGISTRY.

Patch notes (P0–P1):
  P0-1: log_failure → log_frame to match fail_* schema keys
  P0-2: vb_iters / trace_cap wired from OscillatorParams
  P0-3: clamp_state() after predict AND update (z bounds)
  P0-4: rho → tau_env conversion for consistent damping
  P1-5: eda_baseline mode (bypass trust, force Gaussian ν→∞)
  P1-6: w_h wired into gate_z_eff = g_z * w_h
  P1-7: extra_fields in FrameLogger for EDA diagnostics
"""

from typing import Dict, Optional
import numpy as np

from ..core.base import _BaseOscillatorHead, OscillatorParams
from ..core.ssm import OscillatorPredictor, SSMConfig, StateDecoder
from ..core.robust_update import RobustKalmanUpdater
from ..core.trust import TrustAllocator, TrustConfig, TrustParams
from ..core.failure_monitor import FailureMonitor, FailureConfig
from core.evaluation.frame_logger import FrameLogger
from components.observations.quality import QualityEstimator, default_quality

# Extra audit fields beyond the default 23-field FRAME_SCHEMA.
# These enable Phase-0 EDA and post-hoc diagnostics.
_EXTRA_FIELDS = [
    'S_t', 'R_eff', 'R_scaled',        # innovation / noise diagnostics
    'K_x1', 'K_x2', 'K_z',             # Kalman gain components
    'q_vis', 'q_drift', 'q_cons',       # quality vector (placeholder)
    'q_out', 'q_harm', 'q_burst',
    'qx_eff', 'qf_eff', 'rv_eff',      # effective noise params used
]


class oscillator_RobustOSSM(_BaseOscillatorHead):
    """Quality-Aware Robust Oscillatory Bayesian Filter.

    Full pipeline per frame:
        1. SSM predict (EKF or UKF)
        2. Trust allocate from quality vector
        3. Robust Kalman update (Student-t, Algorithm 1)
        4. Clamp z = log(f) to valid range
        5. State decode (amp, phase, freq)
        6. Failure monitor
        7. Frame logging (core + extra diagnostics)
    """
    head_key = "robust_ossm"

    def __init__(self, params: Optional[OscillatorParams] = None):
        super().__init__(params=params)
        p = self.params

        # SSM config from OscillatorParams
        self.ssm_cfg = SSMConfig(
            f_min=p.f_min,
            f_max=p.f_max,
            rho=0.0,  # 0 = "not overridden", compute_rho uses tau_env
            qx=p.qx if p.qx else 1e-4,
            qf=p.qf if p.qf else 1e-6,
            rv_floor=p.rv_floor if p.rv_floor else 0.01,
            tau_env=p.tau_env if p.tau_env else 32.0,
            ukf_alpha=p.ukf_alpha if p.ukf_alpha else 1e-3,
            ukf_beta=p.ukf_beta if p.ukf_beta else 2.0,
            ukf_kappa=p.ukf_kappa if p.ukf_kappa is not None else 0.0,
        )

        # Student-t ν from params (default from OscillatorParams)
        self.nu = float(getattr(p, 'student_t_nu', 5.0) or 5.0)

        # Prediction method
        self.predict_method = getattr(p, 'predict_method', 'ekf') or 'ekf'

        # EDA baseline mode (P1-5)
        self.eda_baseline = bool(getattr(p, 'eda_baseline', False))

    def run(self, signal: np.ndarray, fs: float,
            meta: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Run the full robust Bayesian filter.

        Args:
            signal: raw 1D observation signal from base method
            fs: sampling frequency
            meta: metadata from wrapped_method (spectral, ROI stats, etc.)

        Returns:
            Standard head output dict with signal_hat, track_hz, etc.
        """
        p = self.params
        fs = fs or p.fs
        self._maybe_apply_autotune(meta)

        # ── Preprocessing ──
        y = self._preprocess(signal, fs)
        n = y.size
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        dt = 1.0 / fs

        # ── Initial frequency estimate ──
        freq0 = self._coarse_freq(y, fs, meta)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))

        # ── Effective noise params (meta-driven) ──
        eff = self._effective_params(fs, meta)

        # P0-4: Convert eff['rho'] → tau_env so compute_rho() is consistent.
        # tau_eff = -dt / ln(rho_eff).  rho=0 means "use tau_env" (default).
        rho_eff = float(eff.get('rho', 0.0))
        if 0.0 < rho_eff < 1.0:
            tau_eff = -dt / np.log(rho_eff)
            self.ssm_cfg.tau_env = max(tau_eff, 1e-3)
            self.ssm_cfg.rho = 0.0  # let compute_rho derive from tau_env
        # Other effective params
        self.ssm_cfg.qx = eff['qx']
        self.ssm_cfg.qf = max(eff.get('qf', self.ssm_cfg.qf), 1e-10)
        self.ssm_cfg.rv_floor = eff['rv']

        # ── Instantiate modules ──
        predictor = OscillatorPredictor(self.ssm_cfg)

        # P0-2: Wire vb_iters and trace_cap from OscillatorParams
        # P1-5: EDA baseline → Gaussian updater (ν→∞)
        if self.eda_baseline:
            updater = RobustKalmanUpdater.gaussian()
        else:
            updater = RobustKalmanUpdater(
                nu=self.nu,
                vb_iters=int(p.vb_iters),
                eig_floor=1e-9,
                trace_cap=float(p.trace_cap),
            )

        trust_alloc = TrustAllocator()
        monitor = FailureMonitor(f_ref=freq0)
        decoder = StateDecoder()

        # P1-7: FrameLogger with extended diagnostic fields
        logger = FrameLogger(n, extra_fields=_EXTRA_FIELDS)

        # Phase 1: Quality estimator (uses live signal + ROI metadata)
        qe = QualityEstimator(fs=fs, f_min=p.f_min, f_max=p.f_max)
        
        # Retrieve per-frame ROI stats if available (P1)
        roi_stats_seq = (meta or {}).get('roi_stats_t', [])
        if not isinstance(roi_stats_seq, (list, tuple)):
            roi_stats_seq = []

        # ── Initialize state ──
        x, P = predictor.init_state(freq0)
        Q = predictor.build_Q()
        R = predictor.build_R()
        H = predictor.H

        # ── Storage ──
        x1_out = np.zeros(n, dtype=np.float64)
        freq_track = np.zeros(n, dtype=np.float64)
        nis_prev = 0.0

        # ── Main filter loop ──
        for t in range(n):
            # 1. Predict
            x_pred, P_pred = predictor.predict(x, P, Q, dt,
                                                method=self.predict_method)
            # P0-3: clamp z after prediction
            predictor.clamp_state(x_pred)

            # 2. Trust allocate
            # Phase 1: use real quality estimator when possible
            # Extract per-frame ROI statistics safely
            frame_roi_stats = roi_stats_seq[t] if t < len(roi_stats_seq) else {}
            quality = qe.update(t, y[t], roi_stats=frame_roi_stats)
            
            current_freq = float(np.exp(x_pred[2]))  # safe: z already clamped
            if self.eda_baseline:
                # P1-5: bypass trust → neutral params
                trust = TrustParams()  # defaults: all 1.0
            else:
                trust = trust_alloc.allocate(quality, nis=nis_prev,
                                              current_freq=current_freq)

            # 3. Robust update (Algorithm 1)
            R_scaled = predictor.build_R(alpha_R=trust.alpha_R)

            # Only re-predict with scaled Q if trust actually modified it
            if not self.eda_baseline and trust.alpha_Q != 1.0:
                Q_scaled = predictor.build_Q(alpha_Q=trust.alpha_Q)
                x_pred, P_pred = predictor.predict(x, P, Q_scaled, dt,
                                                    method=self.predict_method)
                predictor.clamp_state(x_pred)

            # P1-6: Wire harmonic suppression into frequency gate
            # gate_z_eff = g_z * w_h (structure-preserving: w_h=1 when no harmonic → no change)
            gate_z_eff = trust.g_z * trust.w_h

            result = updater.update(
                x_pred, P_pred, y[t], H, R_scaled,
                gate=trust.g_t, gate_z=gate_z_eff,
            )

            # 4. Post-update clamp (P0-3)
            predictor.clamp_state(result.x)

            # 5. Accept state
            x = result.x
            P = result.P
            nis_prev = result.nis

            # 6. Decode
            state = decoder.decode(x, P)

            # 7. Failure monitor
            flags = monitor.update(state, result.nis, float(np.trace(P)))

            # Handle divergence → reinitialize
            if flags.diverge:
                x, P = predictor.init_state(freq0)
                monitor.reset(f_ref=freq0)
                trust_alloc.reset()

            # 8. Logging — core state
            logger.log_state(
                t, x, P,
                y_t=y[t],
                y_pred=float(H @ x_pred),
                v_t=result.v_t,
                nis=result.nis,
                lambda_t=result.lambda_t,
            )
            logger.log_trust(
                t,
                alpha_R=trust.alpha_R,
                alpha_Q=trust.alpha_Q,
                g_t=trust.g_t,
                g_z=trust.g_z,
                w_h=trust.w_h,
            )
            # P0-1: Use log_frame() with fail_* keys (matches FRAME_SCHEMA)
            # instead of log_failure() which expects diverge/slip/lock/double args.
            logger.log_frame(t, **flags.to_dict())

            # P1-7: Log extended diagnostics
            logger.log_frame(
                t,
                S_t=result.S_t,
                R_eff=result.R_eff,
                R_scaled=float(R_scaled[0, 0]),
                K_x1=float(result.K[0]),
                K_x2=float(result.K[1]),
                K_z=float(result.K[2]) if len(result.K) >= 3 else 0.0,
                q_vis=quality['q_vis'],
                q_drift=quality['q_drift'],
                q_cons=quality['q_cons'],
                q_out=quality['q_out'],
                q_harm=quality['q_harm'],
                q_burst=quality['q_burst'],
                qx_eff=self.ssm_cfg.qx,
                qf_eff=self.ssm_cfg.qf,
                rv_eff=self.ssm_cfg.rv_floor,
            )

            # 9. Store outputs
            x1_out[t] = state['x1']
            freq_track[t] = state['freq_hz']

        # ── Post-processing ──
        freq_track = np.clip(freq_track, p.f_min, p.f_max)
        freq_track = self._apply_post_smoothing(freq_track)

        # ── Save frame log ──
        aux_dir = (meta or {}).get('aux_save_dir')
        trial_key = (meta or {}).get('trial_key')
        if aux_dir and trial_key:
            import os
            log_dir = os.path.join(aux_dir, 'frame_logs')
            os.makedirs(log_dir, exist_ok=True)
            logger.save(os.path.join(log_dir, f"{trial_key}.npz"))

        # ── Package output ──
        meta_payload = dict(meta or {})
        # Remove large time-series from payload to avoid bloating logs
        meta_payload.pop('roi_stats_t', None)
        
        meta_payload.update({
            'f0': freq0,
            'head': self.head_key,
            'student_t_nu': self.nu if not self.eda_baseline else float('inf'),
            'predict_method': self.predict_method,
            'eda_baseline': self.eda_baseline,
        })
        meta_payload.setdefault('is_constant_track', False)

        return self._package(x1_out, freq_track, meta_payload)
