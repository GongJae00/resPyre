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
from core.pipeline.common import (
    derive_trial_identifiers,
    sanitize_trial_key,
    update_frame_log_manifest,
)
from components.observations.quality import QualityConfig, QualityEstimator, normalize_roi_stats_t

# Extra audit fields beyond the default 23-field FRAME_SCHEMA.
# These enable Phase-0 EDA and post-hoc diagnostics.
_EXTRA_FIELDS = [
    'S_t', 'R_eff', 'R_scaled',        # innovation / noise diagnostics
    'K_x1', 'K_x2', 'K_z',             # Kalman gain components
    'q_vis', 'q_drift', 'q_cons',       # quality vector (placeholder)
    'q_out', 'q_harm', 'q_burst',
    'freq_std_hz', 'amp_std',
    'g_z_eff',
    'g_z_eff_raw',
    'qx_eff', 'qf_eff', 'rv_eff',
    'qx_base', 'qf_base', 'rv_base',   # base params before trust scaling
    'qx_used', 'qf_used',              # effective process-noise values used
    'rv_scaled', 'R_post',             # measurement-noise after trust / robust scaling
    'post_smooth_alpha_base', 'post_smooth_alpha_used',
]


def _bounded_nonnegative_ratio(numer: float, denom: float, *, cap: float = 1.0) -> float:
    """Return a finite, non-negative ratio clipped to [0, cap]."""
    if not np.isfinite(numer):
        return 0.0
    if not np.isfinite(denom) or denom <= 0.0:
        return 0.0
    ratio = max(0.0, float(numer) / float(denom))
    if np.isfinite(cap) and cap > 0.0:
        ratio = min(ratio, float(cap))
    return float(ratio)


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
        alpha_base = float(eff.get('post_smooth_alpha_base', getattr(self.params, 'post_smooth_alpha', 0.0) or 0.0))
        alpha_used = float(eff.get('post_smooth_alpha_used', alpha_base))

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
                lambda_floor=float(getattr(p, "lambda_floor", 1e-3)),
                r_eff_max_scale=float(getattr(p, "r_eff_max_scale", 80.0)),
            )

        trust_cfg_obj = None
        trust_cfg_raw = (meta or {}).get('trust', {})
        if isinstance(trust_cfg_raw, dict):
            allowed_t = set(TrustConfig.__dataclass_fields__.keys())
            t_kwargs = {k: v for k, v in trust_cfg_raw.items() if k in allowed_t}
            if t_kwargs:
                trust_cfg_obj = TrustConfig(**t_kwargs)
        trust_alloc = TrustAllocator(cfg=trust_cfg_obj)
        alpha_r_clip_max = float(max(getattr(trust_alloc.cfg, "alpha_R_max", 50.0), 1.0))
        monitor = FailureMonitor(f_ref=freq0)
        decoder = StateDecoder()

        # P1-7: FrameLogger with extended diagnostic fields
        logger = FrameLogger(n, extra_fields=_EXTRA_FIELDS)

        # Phase 1: Quality estimator (uses live signal + ROI metadata)
        quality_cfg_obj = None
        quality_cfg_raw = (meta or {}).get('quality', {})
        if isinstance(quality_cfg_raw, dict):
            allowed = set(QualityConfig.__dataclass_fields__.keys())
            q_kwargs = {k: v for k, v in quality_cfg_raw.items() if k in allowed}
            if q_kwargs:
                quality_cfg_obj = QualityConfig(**q_kwargs)
        qe = QualityEstimator(fs=fs, cfg=quality_cfg_obj, f_min=p.f_min, f_max=p.f_max)
        gating_cfg = (meta or {}).get('gating', {})
        gating_scope = str((meta or {}).get('gating_scope', 'evaluation_only')).strip().lower()
        if gating_scope not in {'evaluation_only', 'filter_time'}:
            raise ValueError(
                f"Invalid gating_scope '{gating_scope}'. Allowed values: ['evaluation_only', 'filter_time']"
            )
        gating_supported = {
            'profile',
            'debug.disable_gating',
            'spectral.peak_ratio_min',
            'spectral.prominence_min_db',
            'spectral.fwhm_max_hz',
            'spectral.fwhm_df_guard',
            'tracker.std_min_bpm',
            'tracker.unique_min',
            'tracker.saturation_max',
            'tracker.std_is_soft',
            'tracker.saturation_margin_hz',
        }
        gating_seen = set()
        gating_consumed = []
        gating_unused = []
        if isinstance(gating_cfg, dict):
            def _leaf_paths(d: Dict, prefix: str = ""):
                out = []
                for k, v in d.items():
                    key = f"{prefix}.{k}" if prefix else str(k)
                    if isinstance(v, dict):
                        out.extend(_leaf_paths(v, key))
                    else:
                        out.append(key)
                return out
            gating_seen = set(_leaf_paths(gating_cfg))
        if gating_scope == 'filter_time':
            gating_consumed = sorted(x for x in gating_seen if x in gating_supported)
            gating_unused = sorted(x for x in gating_seen if x not in gating_supported)
        else:
            # Evaluation-only scope: no gating key is consumed in filter-time path.
            gating_consumed = []
            gating_unused = sorted(gating_seen)

        profile_name = str((gating_cfg or {}).get('profile', 'paper')).strip().lower()
        if profile_name in {'relaxed', 'loose'}:
            spec_penalty_gain = 0.75
            tracker_penalty_gain = 0.75
        elif profile_name in {'strict', 'hard'}:
            spec_penalty_gain = 1.25
            tracker_penalty_gain = 1.25
        else:
            spec_penalty_gain = 1.0
            tracker_penalty_gain = 1.0
        
        # Retrieve per-frame ROI stats if available (P1), normalized to canonical schema.
        roi_stats_seq = normalize_roi_stats_t((meta or {}).get('roi_stats_t'), n)
        freq_hist_bpm = []

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
                # Optional filter-time gating overrides from config/meta.
                # Kept deterministic and structure-preserving.
                if gating_scope == 'filter_time' and isinstance(gating_cfg, dict):
                    debug_cfg = gating_cfg.get('debug', {})
                    gating_disabled = isinstance(debug_cfg, dict) and bool(debug_cfg.get('disable_gating', False))
                    if gating_disabled:
                        trust = TrustParams()
                    if not gating_disabled:
                        spec_cfg = gating_cfg.get('spectral', {})
                        peak_ratio = (meta or {}).get('welch_peak_ratio')
                        if (
                            isinstance(spec_cfg, dict) and
                            isinstance(peak_ratio, (int, float, np.floating)) and
                            np.isfinite(peak_ratio)
                        ):
                            ratio_min = spec_cfg.get('peak_ratio_min')
                            if isinstance(ratio_min, (int, float, np.floating)) and ratio_min > 0:
                                deficit = _bounded_nonnegative_ratio(
                                    float(ratio_min) - float(peak_ratio),
                                    float(ratio_min),
                                    cap=1.0,
                                )
                                if deficit > 0.0:
                                    trust.alpha_R = float(np.clip(
                                        trust.alpha_R * (1.0 + 0.5 * spec_penalty_gain * deficit),
                                        1.0,
                                        alpha_r_clip_max,
                                    ))
                                    trust.g_t = float(np.clip(
                                        trust.g_t * (1.0 - 0.6 * spec_penalty_gain * deficit),
                                        0.0,
                                        1.0,
                                    ))
                                    trust.g_z = float(np.clip(
                                        trust.g_z * (1.0 - 0.8 * spec_penalty_gain * deficit),
                                        0.0,
                                        1.0,
                                    ))

                        if isinstance(spec_cfg, dict):
                            prom_db = (meta or {}).get('welch_prom_db')
                            prom_min = spec_cfg.get('prominence_min_db')
                            if (
                                isinstance(prom_db, (int, float, np.floating)) and np.isfinite(prom_db) and
                                isinstance(prom_min, (int, float, np.floating))
                            ):
                                # NOTE:
                                # `welch_prom_db` can be strongly negative when the spectral peak is broad/weak.
                                # Raw relative deficit can explode and collapse gates to 0.0.
                                # We bound the deficit to keep the spectral gate effect soft and stable.
                                # Map prominence mismatch onto a wide dB span so that
                                # moderately negative `welch_prom_db` does not immediately
                                # saturate to full penalty on every frame.
                                prom_scale_db = max(abs(float(prom_min)), 20.0)
                                prom_def = _bounded_nonnegative_ratio(
                                    float(prom_min) - float(prom_db),
                                    prom_scale_db,
                                    cap=1.0,
                                )
                                if prom_def > 0.0:
                                    trust.alpha_R = float(np.clip(
                                        trust.alpha_R * (1.0 + 0.4 * spec_penalty_gain * prom_def),
                                        1.0,
                                        alpha_r_clip_max,
                                    ))
                                    trust.g_t = float(np.clip(
                                        trust.g_t * (1.0 - 0.45 * spec_penalty_gain * prom_def),
                                        0.0,
                                        1.0,
                                    ))
                                    trust.g_z = float(np.clip(
                                        trust.g_z * (1.0 - 0.65 * spec_penalty_gain * prom_def),
                                        0.0,
                                        1.0,
                                    ))

                            fwhm_hz = (meta or {}).get('welch_fwhm_hz')
                            fwhm_max = spec_cfg.get('fwhm_max_hz')
                            fwhm_df_guard = spec_cfg.get('fwhm_df_guard')
                            welch_df_hz = (meta or {}).get('welch_df_hz')
                            fwhm_guard_ok = True
                            if (
                                isinstance(fwhm_df_guard, (int, float, np.floating)) and
                                isinstance(welch_df_hz, (int, float, np.floating)) and
                                np.isfinite(welch_df_hz) and float(fwhm_df_guard) > 0.0
                            ):
                                min_resolved = float(fwhm_df_guard) * float(welch_df_hz)
                                if isinstance(fwhm_max, (int, float, np.floating)):
                                    fwhm_guard_ok = float(fwhm_max) > min_resolved
                            if (
                                fwhm_guard_ok and
                                isinstance(fwhm_hz, (int, float, np.floating)) and np.isfinite(fwhm_hz) and
                                isinstance(fwhm_max, (int, float, np.floating)) and float(fwhm_max) > 0.0
                            ):
                                fwhm_excess = _bounded_nonnegative_ratio(
                                    float(fwhm_hz) - float(fwhm_max),
                                    float(fwhm_max),
                                    cap=2.0,
                                )
                                if fwhm_excess > 0.0:
                                    trust.alpha_R = float(np.clip(
                                        trust.alpha_R * (1.0 + 0.35 * spec_penalty_gain * fwhm_excess),
                                        1.0,
                                        alpha_r_clip_max,
                                    ))
                                    trust.g_t = float(np.clip(
                                        trust.g_t * (1.0 - 0.25 * spec_penalty_gain * fwhm_excess),
                                        0.0,
                                        1.0,
                                    ))
                                    trust.g_z = float(np.clip(
                                        trust.g_z * (1.0 - 0.7 * spec_penalty_gain * fwhm_excess),
                                        0.0,
                                        1.0,
                                    ))

                        tracker_cfg = gating_cfg.get('tracker', {})
                        if isinstance(tracker_cfg, dict):
                            freq_hist_bpm.append(current_freq * 60.0)
                            if len(freq_hist_bpm) > int(max(fs, 1.0) * 10):
                                freq_hist_bpm = freq_hist_bpm[-int(max(fs, 1.0) * 10):]

                            hist = np.asarray(freq_hist_bpm, dtype=np.float64)
                            if hist.size >= 5:
                                std_min_bpm = tracker_cfg.get('std_min_bpm')
                                if isinstance(std_min_bpm, (int, float, np.floating)) and std_min_bpm > 0:
                                    std_bpm = float(np.std(hist))
                                    if std_bpm < float(std_min_bpm):
                                        if bool(tracker_cfg.get('std_is_soft', True)):
                                            scale = float(np.clip(std_bpm / float(std_min_bpm), 0.05, 1.0))
                                            scale = float(scale ** tracker_penalty_gain)
                                            trust.g_z = float(np.clip(trust.g_z * scale, 0.0, 1.0))
                                        else:
                                            trust.g_z = 0.0

                                unique_min = tracker_cfg.get('unique_min')
                                if isinstance(unique_min, (int, float, np.floating)) and unique_min > 0:
                                    uniq_ratio = float(np.unique(np.round(hist, 2)).size / hist.size)
                                    if uniq_ratio < float(unique_min):
                                        scale = float(np.clip(uniq_ratio / float(unique_min), 0.05, 1.0))
                                        scale = float(scale ** tracker_penalty_gain)
                                        trust.g_z = float(np.clip(trust.g_z * scale, 0.0, 1.0))

                                saturation_max = tracker_cfg.get('saturation_max')
                                if isinstance(saturation_max, (int, float, np.floating)) and 0 <= saturation_max < 1:
                                    margin_hz = float(tracker_cfg.get('saturation_margin_hz', 0.0) or 0.0)
                                    hist_hz = hist / 60.0
                                    sat = (
                                        (hist_hz <= (p.f_min + margin_hz)) |
                                        (hist_hz >= (p.f_max - margin_hz))
                                    )
                                    sat_ratio = float(np.mean(sat))
                                    if sat_ratio > float(saturation_max):
                                        excess = sat_ratio - float(saturation_max)
                                        trust.alpha_R = float(np.clip(
                                            trust.alpha_R * (1.0 + tracker_penalty_gain * excess),
                                            1.0,
                                            alpha_r_clip_max,
                                        ))
                                        trust.g_z = float(np.clip(
                                            trust.g_z * (1.0 - tracker_penalty_gain * excess),
                                            0.0,
                                            1.0,
                                        ))

            # 3. Robust update (Algorithm 1)
            R_scaled = predictor.build_R(alpha_R=trust.alpha_R)
            qx_base = float(self.ssm_cfg.qx)
            qf_base = float(self.ssm_cfg.qf)
            rv_base = float(self.ssm_cfg.rv_floor)
            qx_used = qx_base * float(trust.alpha_Q)
            qf_used = qf_base * float(trust.alpha_Q)
            rv_scaled = float(R_scaled[0, 0])

            # Only re-predict with scaled Q if trust actually modified it
            if not self.eda_baseline and trust.alpha_Q != 1.0:
                Q_scaled = predictor.build_Q(alpha_Q=trust.alpha_Q)
                x_pred, P_pred = predictor.predict(x, P, Q_scaled, dt,
                                                    method=self.predict_method)
                predictor.clamp_state(x_pred)

            # P1-6: Wire harmonic suppression into frequency gate
            # gate_z_eff = g_z * w_h (structure-preserving: w_h=1 when no harmonic → no change)
            gate_z_eff_raw = float(trust.g_z * trust.w_h)
            gate_z_floor_ratio = float(max(getattr(p, "g_z_eff_floor_ratio", 0.0), 0.0))
            if trust.g_t > 0.0 and gate_z_floor_ratio > 0.0:
                gate_z_eff = max(gate_z_eff_raw, gate_z_floor_ratio * trust.g_t)
            else:
                gate_z_eff = gate_z_eff_raw
            gate_z_eff = float(np.clip(gate_z_eff, 0.0, 1.0))

            result = updater.update(
                x_pred, P_pred, y[t], H, R_scaled,
                gate=trust.g_t, gate_z=gate_z_eff,
            )

            # 4. Post-update clamp (P0-3)
            x_candidate = predictor.clamp_state(np.asarray(result.x, dtype=np.float64).copy())
            P_candidate = predictor.sanitize_covariance(np.asarray(result.P, dtype=np.float64))

            # 5. Accept state (or safe re-init on numeric blow-up)
            if (not np.all(np.isfinite(x_candidate))) or (not np.all(np.isfinite(P_candidate))):
                x, P = predictor.init_state(freq0)
                monitor.reset(f_ref=freq0)
                trust_alloc.reset()
                nis_prev = 0.0
            else:
                x = x_candidate
                P = P_candidate
                nis_prev = float(result.nis) if np.isfinite(result.nis) else 0.0

            # 6. Decode
            state = decoder.decode(x, P)

            # 7. Failure monitor
            flags = monitor.update(state, result.nis, float(np.trace(P)))

            # 8. Logging — core state
            logger.log_state(
                t, x, P,
                y_t=y[t],
                y_pred=(H @ x_pred).item(),
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
                g_z_eff=gate_z_eff,
                g_z_eff_raw=gate_z_eff_raw,
                q_vis=quality['q_vis'],
                q_drift=quality['q_drift'],
                q_cons=quality['q_cons'],
                q_out=quality['q_out'],
                q_harm=quality['q_harm'],
                q_burst=quality['q_burst'],
                freq_std_hz=state.get('freq_std_hz', np.nan),
                amp_std=state.get('amp_std', np.nan),
                qx_eff=qx_used,
                qf_eff=qf_used,
                rv_eff=rv_scaled,
                qx_base=qx_base,
                qf_base=qf_base,
                rv_base=rv_base,
                qx_used=qx_used,
                qf_used=qf_used,
                rv_scaled=rv_scaled,
                R_post=result.R_eff,
                post_smooth_alpha_base=alpha_base,
                post_smooth_alpha_used=alpha_used,
            )

            # 9. Store outputs
            x1_out[t] = state['x1']
            freq_track[t] = state['freq_hz']

            # Handle divergence for the *next* step after logging current frame.
            if flags.diverge:
                x, P = predictor.init_state(freq0)
                monitor.reset(f_ref=freq0)
                trust_alloc.reset()

        # ── Post-processing ──
        freq_track = np.clip(freq_track, p.f_min, p.f_max)
        freq_track = self._apply_post_smoothing(freq_track, alpha_override=alpha_used)

        # ── Save frame log ──
        aux_dir = (meta or {}).get('aux_save_dir')
        trial_key = str((meta or {}).get('trial_key') or "").strip()
        trial_key_full = (meta or {}).get('trial_key_full')
        trial_uid = (meta or {}).get('trial_uid')
        if not trial_key:
            short_key, full_key = derive_trial_identifiers(
                {
                    "dataset_name": (meta or {}).get("dataset"),
                    "subject": (meta or {}).get("subject"),
                    "trial": (meta or {}).get("trial"),
                    "video_path": (meta or {}).get("data_file"),
                },
                dataset_name=str((meta or {}).get("dataset", "")),
                sample_index=0,
            )
            trial_key = short_key
            trial_key_full = trial_key_full or full_key
            trial_uid = trial_uid or full_key
        if aux_dir and trial_key:
            import os
            import hashlib
            log_dir = os.path.join(aux_dir, 'frame_logs')
            os.makedirs(log_dir, exist_ok=True)
            base_key = sanitize_trial_key(trial_key, fallback="trial")
            suffix = ""
            suffix_int = 0
            out_path = os.path.join(log_dir, f"{base_key}.npz")
            while os.path.exists(out_path):
                suffix_int += 1
                suffix = f"_{suffix_int}"
                out_path = os.path.join(log_dir, f"{base_key}{suffix}.npz")
            logger.save(out_path)
            # Deterministic resolution contract: latest saved file must be pinned in manifest.
            sha256 = ""
            try:
                h = hashlib.sha256()
                with open(out_path, "rb") as fp:
                    for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                        h.update(chunk)
                sha256 = h.hexdigest()
            except Exception:
                # Hash is optional; manifest update remains mandatory.
                sha256 = ""
            update_frame_log_manifest(
                aux_dir=aux_dir,
                base_trial_key=base_key,
                actual_filename=os.path.basename(out_path),
                suffix=suffix_int,
                sha256=sha256,
            )
        else:
            suffix = ""

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
            'no_autotune': bool(getattr(self.params, 'no_autotune', False)),
            'gating_scope_used': gating_scope,
            'gating_consumed_keys': gating_consumed,
            'gating_unused_keys': gating_unused,
            'trial_key': trial_key if aux_dir else (meta_payload.get('trial_key') or trial_key),
            'trial_key_full': trial_key_full,
            'trial_uid': trial_uid,
            'trial_key_suffix': suffix,
            'post_smooth_alpha_base': alpha_base,
            'post_smooth_alpha_used': alpha_used,
        })
        inherited_unused = meta_payload.get('unused_config_keys', [])
        merged_unused = []
        if isinstance(inherited_unused, list):
            merged_unused.extend([str(x) for x in inherited_unused])
        merged_unused.extend([f"gating.{k}" for k in gating_unused])
        if merged_unused:
            # Deduplicate but keep deterministic order.
            seen = set()
            ordered = []
            for key in merged_unused:
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(key)
            meta_payload['unused_config_keys'] = ordered
        meta_payload.setdefault('is_constant_track', False)

        return self._package(x1_out, freq_track, meta_payload)
