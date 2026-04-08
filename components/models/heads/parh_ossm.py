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

import os
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import signal as sps
from components.observations.semantics import get_observation_family_semantics
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
    Q_DYN_FREQ_WEIGHT: float = 1.5      # freq deviation sensitivity
    Q_DYN_AMP_WEIGHT: float = 0.75      # amplitude slope sensitivity
    Q_DYN_FREQ_REF_HZ: float = 0.06     # reference freq step

    # q_osc (oscillatory support)
    Q_OSC_PHASE_WEIGHT: float = 0.2
    Q_OSC_HELPER_WEIGHT: float = 0.45
    Q_OSC_FREQ_WEIGHT: float = 0.35
    Q_OSC_OBS_WEIGHT: float = 0.0
    Q_OSC_OBS_MODE: str = "blend_support"
    Q_OSC_OBS_REF: float = 0.97
    Q_OSC_OBS_BAND: float = 0.08
    GATE_PHASE_SIGMA: float = 0.8
    Q_OSC_ERR_SIGMA: float = 1.25
    Q_OSC_FREQ_REF_HZ: float = 0.08
    TAU_AMP_SEC: float = 1.5
    GATE_WARMUP_SEC: float = 0.75

    # ── Disentangled Q mapping ──
    Q_DYN_GAMMA: float = 0.5            # Q_osc scaling strength from q_dyn
    Q_APER_GAMMA: float = 4.0           # Q_aper scaling strength from (1-q_osc)
    Q_APER_OBS_GAMMA: float = 0.0       # extra Q_aper boost from clean unexplained observation need

    # ── Legacy coupled Q (ablation only) ──
    QX_ADAPT_GAMMA_LEGACY: float = 0.5

    # ── Frequency adaptation ──
    FREQ_UPDATE_INTERVAL_SEC: float = 2.0
    FREQ_CONFIRM_COUNT: int = 3
    FREQ_MAX_STEP_HZ: float = 0.03
    ENABLE_FREQ_RESCUE: bool = True
    FREQ_RESCUE_WINDOW_SEC: float = 4.0
    FREQ_RESCUE_MIN_SUPPORT: float = 0.75
    FREQ_RESCUE_MIN_QDYN: float = 0.60
    FREQ_RESCUE_MIN_MISMATCH_HZ: float = 0.07
    FREQ_RESCUE_HELPER_STD_MAX_HZ: float = 0.10
    FREQ_RESCUE_CONFIRM_COUNT: int = 2
    FREQ_RESCUE_MAX_STEP_HZ: float = 0.08
    FREQ_RESCUE_POLICY: str = "bridge_v1"

    # ── Harmonic-aware init ──
    HARMONIC_POWER_RATIO: float = 0.40
    HARMONIC_ACF_RATIO: float = 1.05

    # ── Warmup ──
    WARMUP_SEC: float = 2.5
    QX_ADAPT_WARMUP_SEC: float = 3.0
    FREQ_INIT_SEC: float = 10.0   # Warm-up window for initial frequency estimation

    # ── Observation path (keeps baseline/residual visible) ──
    OBS_LIGHT_LOWPASS_HZ: float = 1.0
    OBS_CLIP_Z: float = 6.0
    OBS_CENTER_MODE: str = "median"
    OBS_BLEND_ALPHA: float = 0.35

    # ── Warm-up observation calibration ──
    ENABLE_OBS_CAL: bool = True
    OBS_CAL_WARMUP_SEC: float = 12.0
    OBS_CAL_SKIP_SEC: float = 2.0
    OBS_CAL_MODE: str = "family_phase_aux"
    OBS_CAL_RIDGE: float = 5e-2
    OBS_CAL_PRIOR_STRENGTH: float = 1.5
    OBS_CAL_MAX_GAIN_OSC: float = 4.0
    OBS_CAL_MAX_GAIN_AUX: float = 0.45
    OBS_CAL_MAX_GAIN_B: float = 0.35
    OBS_CAL_MAX_GAIN_R: float = 0.65
    OBS_CAL_MAX_GAIN_H1: float = 1.25
    OBS_CAL_MAX_GAIN_H2: float = 1.50
    OBS_CAL_MAX_LAG_SEC: float = 0.12
    OBS_CAL_MIN_FIT_CORR: float = 0.45
    OBS_CAL_MAX_FIT_RMSE_NORM: float = 1.10
    OBS_CAL_ALLOWED_FAMILIES: str = "profile1d_quadratic,profile1d_cubic"
    QUADCUB_HARMONIC_ONLY: bool = True
    QUADCUB_PRIOR_STRENGTH: float = 1.70
    QUADCUB_MAX_GAIN_H1: float = 1.20
    QUADCUB_MAX_GAIN_H2: float = 1.60
    QUADCUB_MAX_LAG_SEC: float = 0.08
    OF_HARMONIC_ONLY: bool = True
    OF_PRIOR_STRENGTH: float = 1.35
    OF_MAX_GAIN_H1: float = 1.10
    OF_MAX_GAIN_H2: float = 0.85
    OF_MAX_LAG_SEC: float = 0.06
    OF_MIN_FIT_CORR: float = 0.60
    OF_MAX_FIT_RMSE_NORM: float = 1.00
    OF_FIXED_VELOCITY_PRIOR: bool = False

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
    USE_HELPER_PATH: bool = True
    USE_LIGHT_OBS_PATH: bool = True
    OBS_FAMILY_POLICY: str = "bridge_v1"
    OUTPUT_RATE_POLICY: str = "of_helper_blend_v1"
    OUTPUT_RATE_BLEND_ALPHA: float = 0.45
    OUTPUT_RATE_MIN_SUPPORT: float = 0.72
    OUTPUT_RATE_MIN_QDYN: float = 0.45
    OUTPUT_RATE_MIN_MISMATCH_HZ: float = 0.04
    OUTPUT_RATE_BIAS_WIN_SEC: float = 5.0
    OUTPUT_RATE_BIAS_MIN_SIGN_STABILITY: float = 0.65
    OUTPUT_RATE_BIAS_MAX_HELPER_STD_HZ: float = 0.08
    OUTPUT_RATE_BIAS_MAX_CORR_HZ: float = 0.05
    PROFILE_RATE_BLEND_ALPHA: float = 0.18
    PROFILE_RATE_MIN_SUPPORT: float = 0.95
    PROFILE_RATE_MAX_QDYN: float = 0.40
    PROFILE_RATE_MIN_MISMATCH_HZ: float = 0.025
    PROFILE_RATE_MAX_MISMATCH_HZ: float = 0.10

    # ── Helper trust (OF-only experimental) ──
    HELPER_TRUST_POLICY: str = "off"
    HELPER_TRUST_WINDOW_SEC: float = 4.0
    HELPER_TRUST_STD_REF_HZ: float = 0.07
    HELPER_TRUST_QDYN_FLOOR: float = 0.30
    HELPER_TRUST_MIN_MISMATCH_HZ: float = 0.03
    HELPER_TRUST_MISMATCH_REF_HZ: float = 0.06
    HELPER_TRUST_RESCUE_MIN: float = 0.45

    # ── Family-conditioned confidence policy (experimental) ──
    ENABLE_FAMILY_CONFIDENCE: bool = True
    FAMILY_CONFIDENCE_ALLOWED_FAMILIES: str = "profile1d_quadratic,profile1d_cubic"
    FAMILY_CONFIDENCE_MIN_FIT_CORR: float = 0.975
    FAMILY_CONFIDENCE_MAX_FIT_RMSE: float = 0.20
    FAMILY_CONFIDENCE_PI_FLOOR: float = 0.97
    FAMILY_CONFIDENCE_QDYN_SCALE: float = 0.55
    FAMILY_CONFIDENCE_R_SCALE: float = 0.85

    # ── Family-aware residual semantics (experimental) ──
    ENABLE_RESIDUAL_SEMANTICS: bool = False
    RESIDUAL_PRIOR_MIN: float = 0.10
    RESIDUAL_PRIOR_POWER: float = 1.0

    def __init__(self, params=None):
        super().__init__(params=params)
        self._apply_env_overrides()

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        raw = str(raw).strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.getenv(name)
        if raw is None:
            return float(default)
        try:
            return float(raw)
        except Exception:
            return float(default)

    @staticmethod
    def _env_str(name: str, default: str) -> str:
        raw = os.getenv(name)
        if raw is None:
            return str(default)
        return str(raw).strip()

    def _apply_env_overrides(self) -> None:
        bool_fields = {
            "ENABLE_HARMONIC2": self.ENABLE_HARMONIC2,
            "ENABLE_BASELINE": self.ENABLE_BASELINE,
            "ENABLE_RESIDUAL": self.ENABLE_RESIDUAL,
            "ENABLE_ADAPT_R": self.ENABLE_ADAPT_R,
            "ENABLE_DISENTANGLED_Q": self.ENABLE_DISENTANGLED_Q,
            "ENABLE_LEGACY_COUPLED_Q": self.ENABLE_LEGACY_COUPLED_Q,
            "ENABLE_STUDENT_T": self.ENABLE_STUDENT_T,
            "ENABLE_FREQ_ADAPT": self.ENABLE_FREQ_ADAPT,
            "ENABLE_FREQ_RESCUE": self.ENABLE_FREQ_RESCUE,
            "USE_HELPER_PATH": self.USE_HELPER_PATH,
            "USE_LIGHT_OBS_PATH": self.USE_LIGHT_OBS_PATH,
            "ENABLE_OBS_CAL": self.ENABLE_OBS_CAL,
            "QUADCUB_HARMONIC_ONLY": self.QUADCUB_HARMONIC_ONLY,
            "OF_HARMONIC_ONLY": self.OF_HARMONIC_ONLY,
            "OF_FIXED_VELOCITY_PRIOR": self.OF_FIXED_VELOCITY_PRIOR,
            "ENABLE_FAMILY_CONFIDENCE": self.ENABLE_FAMILY_CONFIDENCE,
            "ENABLE_RESIDUAL_SEMANTICS": self.ENABLE_RESIDUAL_SEMANTICS,
        }
        for field, default in bool_fields.items():
            setattr(self, field, self._env_bool(f"RESPYRE_PARH_{field}", default))

        float_fields = {
            "Q_DYN_GAMMA": self.Q_DYN_GAMMA,
            "Q_APER_GAMMA": self.Q_APER_GAMMA,
            "Q_APER_OBS_GAMMA": self.Q_APER_OBS_GAMMA,
            "Q_OSC_OBS_WEIGHT": self.Q_OSC_OBS_WEIGHT,
            "Q_OSC_OBS_REF": self.Q_OSC_OBS_REF,
            "Q_OSC_OBS_BAND": self.Q_OSC_OBS_BAND,
            "OBS_CAL_RIDGE": self.OBS_CAL_RIDGE,
            "OBS_CAL_PRIOR_STRENGTH": self.OBS_CAL_PRIOR_STRENGTH,
            "OBS_CAL_MAX_GAIN_AUX": self.OBS_CAL_MAX_GAIN_AUX,
            "OBS_CAL_MAX_GAIN_B": self.OBS_CAL_MAX_GAIN_B,
            "OBS_CAL_MAX_GAIN_R": self.OBS_CAL_MAX_GAIN_R,
            "FREQ_RESCUE_MAX_STEP_HZ": self.FREQ_RESCUE_MAX_STEP_HZ,
            "HARMONIC_POWER_RATIO": self.HARMONIC_POWER_RATIO,
            "OBS_BLEND_ALPHA": self.OBS_BLEND_ALPHA,
            "OBS_CAL_MAX_GAIN_H1": self.OBS_CAL_MAX_GAIN_H1,
            "OBS_CAL_MAX_GAIN_H2": self.OBS_CAL_MAX_GAIN_H2,
            "OBS_CAL_MAX_LAG_SEC": self.OBS_CAL_MAX_LAG_SEC,
            "OBS_CAL_MIN_FIT_CORR": self.OBS_CAL_MIN_FIT_CORR,
            "OBS_CAL_MAX_FIT_RMSE_NORM": self.OBS_CAL_MAX_FIT_RMSE_NORM,
            "QUADCUB_PRIOR_STRENGTH": self.QUADCUB_PRIOR_STRENGTH,
            "QUADCUB_MAX_GAIN_H1": self.QUADCUB_MAX_GAIN_H1,
            "QUADCUB_MAX_GAIN_H2": self.QUADCUB_MAX_GAIN_H2,
            "QUADCUB_MAX_LAG_SEC": self.QUADCUB_MAX_LAG_SEC,
            "OF_PRIOR_STRENGTH": self.OF_PRIOR_STRENGTH,
            "OF_MAX_GAIN_H1": self.OF_MAX_GAIN_H1,
            "OF_MAX_GAIN_H2": self.OF_MAX_GAIN_H2,
            "OF_MAX_LAG_SEC": self.OF_MAX_LAG_SEC,
            "OF_MIN_FIT_CORR": self.OF_MIN_FIT_CORR,
            "OF_MAX_FIT_RMSE_NORM": self.OF_MAX_FIT_RMSE_NORM,
            "OUTPUT_RATE_BLEND_ALPHA": self.OUTPUT_RATE_BLEND_ALPHA,
            "OUTPUT_RATE_MIN_SUPPORT": self.OUTPUT_RATE_MIN_SUPPORT,
            "OUTPUT_RATE_MIN_QDYN": self.OUTPUT_RATE_MIN_QDYN,
            "OUTPUT_RATE_MIN_MISMATCH_HZ": self.OUTPUT_RATE_MIN_MISMATCH_HZ,
            "OUTPUT_RATE_BIAS_WIN_SEC": self.OUTPUT_RATE_BIAS_WIN_SEC,
            "OUTPUT_RATE_BIAS_MIN_SIGN_STABILITY": self.OUTPUT_RATE_BIAS_MIN_SIGN_STABILITY,
            "OUTPUT_RATE_BIAS_MAX_HELPER_STD_HZ": self.OUTPUT_RATE_BIAS_MAX_HELPER_STD_HZ,
            "OUTPUT_RATE_BIAS_MAX_CORR_HZ": self.OUTPUT_RATE_BIAS_MAX_CORR_HZ,
            "PROFILE_RATE_BLEND_ALPHA": self.PROFILE_RATE_BLEND_ALPHA,
            "PROFILE_RATE_MIN_SUPPORT": self.PROFILE_RATE_MIN_SUPPORT,
            "PROFILE_RATE_MAX_QDYN": self.PROFILE_RATE_MAX_QDYN,
            "PROFILE_RATE_MIN_MISMATCH_HZ": self.PROFILE_RATE_MIN_MISMATCH_HZ,
            "PROFILE_RATE_MAX_MISMATCH_HZ": self.PROFILE_RATE_MAX_MISMATCH_HZ,
            "HELPER_TRUST_WINDOW_SEC": self.HELPER_TRUST_WINDOW_SEC,
            "HELPER_TRUST_STD_REF_HZ": self.HELPER_TRUST_STD_REF_HZ,
            "HELPER_TRUST_QDYN_FLOOR": self.HELPER_TRUST_QDYN_FLOOR,
            "HELPER_TRUST_MIN_MISMATCH_HZ": self.HELPER_TRUST_MIN_MISMATCH_HZ,
            "HELPER_TRUST_MISMATCH_REF_HZ": self.HELPER_TRUST_MISMATCH_REF_HZ,
            "HELPER_TRUST_RESCUE_MIN": self.HELPER_TRUST_RESCUE_MIN,
            "FAMILY_CONFIDENCE_MIN_FIT_CORR": self.FAMILY_CONFIDENCE_MIN_FIT_CORR,
            "FAMILY_CONFIDENCE_MAX_FIT_RMSE": self.FAMILY_CONFIDENCE_MAX_FIT_RMSE,
            "FAMILY_CONFIDENCE_PI_FLOOR": self.FAMILY_CONFIDENCE_PI_FLOOR,
            "FAMILY_CONFIDENCE_QDYN_SCALE": self.FAMILY_CONFIDENCE_QDYN_SCALE,
            "FAMILY_CONFIDENCE_R_SCALE": self.FAMILY_CONFIDENCE_R_SCALE,
            "RESIDUAL_PRIOR_MIN": self.RESIDUAL_PRIOR_MIN,
            "RESIDUAL_PRIOR_POWER": self.RESIDUAL_PRIOR_POWER,
        }
        for field, default in float_fields.items():
            setattr(self, field, self._env_float(f"RESPYRE_PARH_{field}", default))

        self.OBS_CAL_MODE = self._env_str("RESPYRE_PARH_OBS_CAL_MODE", self.OBS_CAL_MODE)
        self.OBS_FAMILY_POLICY = self._env_str("RESPYRE_PARH_OBS_FAMILY_POLICY", self.OBS_FAMILY_POLICY)
        self.FREQ_RESCUE_POLICY = self._env_str("RESPYRE_PARH_FREQ_RESCUE_POLICY", self.FREQ_RESCUE_POLICY)
        self.OUTPUT_RATE_POLICY = self._env_str("RESPYRE_PARH_OUTPUT_RATE_POLICY", self.OUTPUT_RATE_POLICY)
        self.Q_OSC_OBS_MODE = self._env_str("RESPYRE_PARH_Q_OSC_OBS_MODE", self.Q_OSC_OBS_MODE)
        self.HELPER_TRUST_POLICY = self._env_str("RESPYRE_PARH_HELPER_TRUST_POLICY", self.HELPER_TRUST_POLICY)
        self.OBS_CAL_ALLOWED_FAMILIES = self._env_str(
            "RESPYRE_PARH_OBS_CAL_ALLOWED_FAMILIES",
            self.OBS_CAL_ALLOWED_FAMILIES,
        )
        self.FAMILY_CONFIDENCE_ALLOWED_FAMILIES = self._env_str(
            "RESPYRE_PARH_FAMILY_CONFIDENCE_ALLOWED_FAMILIES",
            self.FAMILY_CONFIDENCE_ALLOWED_FAMILIES,
        )

    @staticmethod
    def _base_method_from_meta(run_meta: Dict, family_override: Optional[str] = None) -> str:
        if family_override is not None:
            base_method = str(family_override or "").lower()
        else:
            base_method = str(run_meta.get("base_method") or run_meta.get("method_name") or "").lower()
        if "__" in base_method:
            base_method = base_method.split("__", 1)[0]
        return base_method

    @staticmethod
    def _observation_family_semantics(base_method: str) -> Dict[str, object]:
        return get_observation_family_semantics(base_method)

    def _freq_rescue_allowed(self, base_method: str) -> bool:
        """Allow rescue only where gate-subset evidence is positive."""
        sem = self._observation_family_semantics(base_method)
        return bool(sem.get("allow_freq_rescue", False))

    def _helper_trust_allowed(self, base_method: str) -> bool:
        """Restrict helper-trust experiments to OF-like families for now."""
        policy = str(self.HELPER_TRUST_POLICY or "off").strip().lower()
        if policy == "off":
            return False
        sem = self._observation_family_semantics(base_method)
        return bool(sem.get("allow_helper_trust", False))

    def _helper_trust_scales_qdyn(self, base_method: str) -> bool:
        """Only the original OF helper-trust policy is allowed to suppress q_dyn.

        Gate evidence showed that broad helper-trust-driven q_dyn suppression
        hurts OF rate. Rescue-only policies still compute helper trust and bias
        confidence, but they leave q_dyn unchanged and use trust only when
        deciding whether rescue is safe.
        """
        if not self._helper_trust_allowed(base_method):
            return False
        policy = str(self.HELPER_TRUST_POLICY or "off").strip().lower()
        return policy == "of_v1"

    def _family_confidence_policy(
        self,
        base_method: str,
        obs_cal: Dict[str, float],
    ) -> Dict[str, float]:
        """Warm-up-fit-qualified confidence policy for strong profile families.

        The current Base gap on `P1D_quad/cub` suggests these families are
        already close to the intended displacement-like semantics. When warm-up
        calibration confirms that closeness, we can safely keep a higher prior
        trust floor, damp q_dyn slightly, and very gently shrink effective R so
        the filter does not overreact to small innovation spikes on otherwise
        very clean trials.
        """
        policy = {
            "enabled": False,
            "pi_floor": 0.0,
            "qdyn_scale": 1.0,
            "r_scale": 1.0,
        }
        if not bool(self.ENABLE_FAMILY_CONFIDENCE):
            return policy
        sem = self._observation_family_semantics(base_method)
        base = str(sem.get("canonical_family", base_method or "")).lower().strip()
        if not bool(sem.get("allow_family_confidence", False)):
            return policy
        allowed = {
            token.strip().lower()
            for token in str(self.FAMILY_CONFIDENCE_ALLOWED_FAMILIES or "").split(",")
            if token.strip()
        }
        if allowed and base not in allowed:
            return policy
        if not bool(obs_cal.get("enabled", False)):
            return policy
        if str(obs_cal.get("obs_domain", "displacement")).strip().lower() != "displacement":
            return policy
        fit_corr = float(obs_cal.get("fit_corr", np.nan))
        fit_rmse = float(obs_cal.get("fit_rmse", np.nan))
        if (not np.isfinite(fit_corr)) or fit_corr < float(self.FAMILY_CONFIDENCE_MIN_FIT_CORR):
            return policy
        if (not np.isfinite(fit_rmse)) or fit_rmse > float(self.FAMILY_CONFIDENCE_MAX_FIT_RMSE):
            return policy
        policy["enabled"] = True
        policy["pi_floor"] = float(np.clip(self.FAMILY_CONFIDENCE_PI_FLOOR, 0.0, 1.0))
        policy["qdyn_scale"] = float(np.clip(self.FAMILY_CONFIDENCE_QDYN_SCALE, 0.0, 1.0))
        policy["r_scale"] = float(np.clip(self.FAMILY_CONFIDENCE_R_SCALE, 0.5, 1.0))
        return policy

    def _freq_rescue_params(self, base_method: str) -> Dict[str, float]:
        """Family-aware rescue thresholds.

        The current bottleneck is OF rate, not waveform. This policy allows
        OF-specific rescue thresholds to be loosened without touching the
        profile families unless explicit evidence supports it.
        """
        params = {
            "min_support": float(self.FREQ_RESCUE_MIN_SUPPORT),
            "min_qdyn": float(self.FREQ_RESCUE_MIN_QDYN),
            "min_mismatch_hz": float(self.FREQ_RESCUE_MIN_MISMATCH_HZ),
            "helper_std_max_hz": float(self.FREQ_RESCUE_HELPER_STD_MAX_HZ),
            "confirm_count": int(self.FREQ_RESCUE_CONFIRM_COUNT),
            "max_step_hz": float(self.FREQ_RESCUE_MAX_STEP_HZ),
        }
        policy = str(self.FREQ_RESCUE_POLICY or "bridge_v1").strip().lower()
        sem = self._observation_family_semantics(base_method)
        base = str(sem.get("canonical_family", base_method or "")).lower().strip()
        if policy == "of_v2" and str(sem.get("family_group", "")).startswith("optical_flow"):
            params.update({
                "min_support": 0.68,
                "min_qdyn": 0.45,
                "min_mismatch_hz": 0.045,
                "helper_std_max_hz": 0.12,
                "confirm_count": 1,
                "max_step_hz": min(0.10, max(0.05, float(self.FREQ_RESCUE_MAX_STEP_HZ))),
            })
        return params

    def _output_rate_postprocess(
        self,
        track_hz: np.ndarray,
        helper_freq: np.ndarray,
        helper_support: np.ndarray,
        q_dyn: np.ndarray,
        base_method: str,
        alpha_used: float,
        fs: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Output-only rate refinement.

        This preserves waveform-state dynamics while allowing family-specific
        helper evidence to improve T3 when the helper is locally more reliable.
        """
        out = np.asarray(track_hz, dtype=np.float64).copy()
        active = np.zeros(out.size, dtype=np.float64)
        policy = str(self.OUTPUT_RATE_POLICY or "bridge_v1").strip().lower()
        sem = self._observation_family_semantics(base_method)
        base = str(sem.get("canonical_family", base_method or "")).lower().strip()
        if policy not in {"of_helper_blend_v1", "of_helper_bias_v1", "hybrid_semantics_v1"} or out.size == 0:
            return out, active

        helper = np.asarray(helper_freq, dtype=np.float64).copy()
        if helper.size != out.size or helper.size == 0:
            return out, active
        helper_bad = ~np.isfinite(helper)
        if np.any(helper_bad):
            helper[helper_bad] = out[helper_bad]
        helper = np.clip(helper, self.params.f_min, self.params.f_max)
        helper = self._apply_post_smoothing(helper, alpha_override=alpha_used)

        support = np.asarray(helper_support, dtype=np.float64).reshape(-1)
        qdyn = np.asarray(q_dyn, dtype=np.float64).reshape(-1)
        if support.size != out.size:
            support = np.full(out.size, np.nanmedian(support) if support.size else 0.0, dtype=np.float64)
        if qdyn.size != out.size:
            qdyn = np.full(out.size, np.nanmedian(qdyn) if qdyn.size else 0.0, dtype=np.float64)
        mismatch = np.abs(helper - out)
        if str(sem.get("family_group", "")) == "profile1d_harmonic" and policy == "hybrid_semantics_v1":
            gate = (
                np.isfinite(helper)
                & np.isfinite(out)
                & np.isfinite(support)
                & np.isfinite(qdyn)
                & (support >= float(self.PROFILE_RATE_MIN_SUPPORT))
                & (qdyn <= float(self.PROFILE_RATE_MAX_QDYN))
                & (mismatch >= float(self.PROFILE_RATE_MIN_MISMATCH_HZ))
                & (mismatch <= float(self.PROFILE_RATE_MAX_MISMATCH_HZ))
            )
            if not np.any(gate):
                return out, active
            blend_alpha = float(np.clip(self.PROFILE_RATE_BLEND_ALPHA, 0.0, 1.0))
            out[gate] = (1.0 - blend_alpha) * out[gate] + blend_alpha * helper[gate]
            out = np.clip(out, self.params.f_min, self.params.f_max)
            active[gate] = 1.0
            return out, active

        if policy in {"of_helper_blend_v1", "hybrid_semantics_v1"}:
            if not bool(sem.get("allow_output_rate_refine", False)):
                return out, active
            gate = (
                np.isfinite(helper)
                & np.isfinite(out)
                & np.isfinite(support)
                & np.isfinite(qdyn)
                & (support >= float(self.OUTPUT_RATE_MIN_SUPPORT))
                & (qdyn >= float(self.OUTPUT_RATE_MIN_QDYN))
                & (mismatch >= float(self.OUTPUT_RATE_MIN_MISMATCH_HZ))
            )
            if not np.any(gate):
                return out, active

            blend_alpha = float(np.clip(self.OUTPUT_RATE_BLEND_ALPHA, 0.0, 1.0))
            out[gate] = (1.0 - blend_alpha) * out[gate] + blend_alpha * helper[gate]
            out = np.clip(out, self.params.f_min, self.params.f_max)
            active[gate] = 1.0
            return out, active

        # Bias-aware helper integration for OF.
        # Instead of following the instantaneous helper, only apply a bounded
        # correction when helper-track mismatch is stable over a recent horizon.
        win = max(3, int(round(float(self.OUTPUT_RATE_BIAS_WIN_SEC) * max(float(fs), 1e-6))))
        min_frames = max(6, win // 2)
        signed_mismatch = helper - out
        blend_alpha = float(np.clip(self.OUTPUT_RATE_BLEND_ALPHA, 0.0, 1.0))
        max_corr = max(float(self.OUTPUT_RATE_BIAS_MAX_CORR_HZ), 1e-6)
        min_sign_stability = float(np.clip(self.OUTPUT_RATE_BIAS_MIN_SIGN_STABILITY, 0.0, 1.0))
        max_helper_std = max(float(self.OUTPUT_RATE_BIAS_MAX_HELPER_STD_HZ), 1e-6)
        min_mismatch = max(float(self.OUTPUT_RATE_MIN_MISMATCH_HZ), 1e-6)
        for t in range(out.size):
            start = max(0, t - win + 1)
            mm_win = signed_mismatch[start:t + 1]
            support_win = support[start:t + 1]
            qdyn_win = qdyn[start:t + 1]
            helper_win = helper[start:t + 1]
            valid = (
                np.isfinite(mm_win)
                & np.isfinite(support_win)
                & np.isfinite(qdyn_win)
                & np.isfinite(helper_win)
            )
            if int(np.count_nonzero(valid)) < min_frames:
                continue
            mm_valid = mm_win[valid]
            support_valid = support_win[valid]
            qdyn_valid = qdyn_win[valid]
            helper_valid = helper_win[valid]
            if np.nanmedian(support_valid) < float(self.OUTPUT_RATE_MIN_SUPPORT):
                continue
            if np.nanmedian(qdyn_valid) < float(self.OUTPUT_RATE_MIN_QDYN):
                continue
            signed_strong = mm_valid[np.abs(mm_valid) >= min_mismatch]
            if signed_strong.size < max(4, min_frames // 3):
                continue
            sign_stability = abs(float(np.mean(np.sign(signed_strong))))
            if sign_stability < min_sign_stability:
                continue
            helper_std = float(np.nanstd(helper_valid))
            if not np.isfinite(helper_std) or helper_std > max_helper_std:
                continue
            corr = float(np.nanmedian(signed_strong))
            if not np.isfinite(corr):
                continue
            corr = float(np.clip(corr, -max_corr, max_corr))
            if abs(corr) < min_mismatch:
                continue
            if np.sign(corr) != np.sign(signed_mismatch[t]):
                continue
            out[t] = float(np.clip(out[t] + blend_alpha * corr, self.params.f_min, self.params.f_max))
            active[t] = 1.0
        return out, active

    def _assistant_rate_postprocess(
        self,
        track_hz: np.ndarray,
        primary_helper_freq: np.ndarray,
        assistant_helper_freq: Optional[np.ndarray],
        helper_support: np.ndarray,
        q_dyn: np.ndarray,
        alpha_used: float,
        assistant_policy: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Output-only assistant-channel refinement.

        Unlike direct fusion, this path only nudges the final rate output when
        the primary observation appears locally weak and the assistant helper is
        both stable and closer than the primary helper to the current rate
        track. It never alters waveform-state inference.
        """
        out = np.asarray(track_hz, dtype=np.float64).copy()
        active = np.zeros(out.size, dtype=np.float64)
        policy = str(assistant_policy or "").strip().lower()
        if policy != "of_rate_assistant_v2" or assistant_helper_freq is None or out.size == 0:
            return out, active

        primary = np.asarray(primary_helper_freq, dtype=np.float64).copy()
        assistant = np.asarray(assistant_helper_freq, dtype=np.float64).copy()
        if primary.size != out.size or assistant.size != out.size:
            return out, active

        bad_primary = ~np.isfinite(primary)
        bad_assistant = ~np.isfinite(assistant)
        if np.any(bad_primary):
            primary[bad_primary] = out[bad_primary]
        if np.any(bad_assistant):
            assistant[bad_assistant] = primary[bad_assistant]

        primary = np.clip(primary, self.params.f_min, self.params.f_max)
        assistant = np.clip(assistant, self.params.f_min, self.params.f_max)
        primary = self._apply_post_smoothing(primary, alpha_override=alpha_used)
        assistant = self._apply_post_smoothing(assistant, alpha_override=alpha_used)

        support = np.asarray(helper_support, dtype=np.float64).reshape(-1)
        qdyn = np.asarray(q_dyn, dtype=np.float64).reshape(-1)
        if support.size != out.size:
            fill = np.nanmedian(support) if support.size else 0.0
            support = np.full(out.size, fill, dtype=np.float64)
        if qdyn.size != out.size:
            fill = np.nanmedian(qdyn) if qdyn.size else 0.0
            qdyn = np.full(out.size, fill, dtype=np.float64)

        mismatch_primary = np.abs(primary - out)
        mismatch_assistant = np.abs(assistant - out)
        helper_disagreement = np.abs(assistant - primary)
        assistant_slew = np.abs(np.diff(assistant, prepend=assistant[0]))

        gate = (
            np.isfinite(out)
            & np.isfinite(primary)
            & np.isfinite(assistant)
            & np.isfinite(support)
            & np.isfinite(qdyn)
            & (support <= 0.78)
            & (qdyn >= 0.30)
            & (mismatch_primary >= 0.045)
            & (helper_disagreement >= 0.020)
            & (assistant_slew <= 0.030)
            & (mismatch_assistant <= np.minimum(0.75 * mismatch_primary, 0.080))
        )
        if not np.any(gate):
            return out, active

        blend_alpha = 0.20
        out[gate] = (1.0 - blend_alpha) * out[gate] + blend_alpha * assistant[gate]
        out = np.clip(out, self.params.f_min, self.params.f_max)
        active[gate] = 1.0
        return out, active

    def _observation_family_mode(self, base_method: str) -> str:
        """Select the inference-path observation semantics for each family.

        Policies are intentionally conservative and only promote families with
        explicit gate evidence. The helper mode reuses the oscillatory helper
        preprocessing as the observation path, which suppresses low-frequency
        nuisance but also weakens baseline/residual visibility.
        """
        sem = self._observation_family_semantics(base_method)
        base = str(sem.get("canonical_family", base_method or "")).lower().strip()
        policy = str(self.OBS_FAMILY_POLICY or "bridge_v1").strip().lower()
        if not base:
            return "light"

        if policy == "legacy_all":
            return "legacy"
        if policy == "helper_all":
            return "helper"
        if policy == "all_p1d_helper":
            if base.startswith("profile1d_"):
                return "helper"
            if base == "dof":
                return "legacy"
            return "light"
        if policy == "quadcub_blend":
            if base in {"profile1d_quadratic", "profile1d_cubic"}:
                return "blend"
            if base.startswith("profile1d_") or base == "dof":
                return "legacy"
            return "light"
        if policy == "quadcub_dof_blend":
            if base in {"profile1d_quadratic", "profile1d_cubic", "dof"}:
                return "blend"
            if base.startswith("profile1d_"):
                return "legacy"
            return "light"
        if policy == "quadcub_helper":
            if base in {"profile1d_quadratic", "profile1d_cubic"}:
                return "helper"
            if base.startswith("profile1d_") or base == "dof":
                return "legacy"
            return "light"
        if policy == "quadcub_dof_helper":
            if base in {"profile1d_quadratic", "profile1d_cubic", "dof"}:
                return "helper"
            if base.startswith("profile1d_"):
                return "legacy"
            return "light"

        # bridge_v1: current promoted scaffold.
        # The OF displacement bridge is closer to a displacement-like profile
        # proxy than to raw OF velocity semantics, so keep it on the legacy
        # inference stack rather than the light OF path.
        return str(sem.get("default_inference_mode", "light"))

    def _obs_cal_family_prior(
        self,
        base_method: str,
        *,
        apply_allowed_filter: bool = True,
    ) -> Dict[str, float]:
        """Family-specific bounded observation-row prior.

        The goal is not to learn a free regression row, but to encode a small
        family-aware visibility prior over the fundamental harmonic, second
        harmonic, and auxiliary non-oscillatory content. A small fixed lag is
        represented as a phase shift in the harmonic observation row.
        """
        sem = self._observation_family_semantics(base_method)
        base = str(sem.get("canonical_family", base_method or "")).lower().strip()
        prior = {
            "enabled": False,
            "obs_domain": str(sem.get("observation_domain", "displacement")),
            "g_h1": 1.0,
            "g_h2": 0.5,
            "g_aux": 0.15,
            "g_b": 0.12,
            "g_r": 0.18,
            "max_g_h1": float(self.OBS_CAL_MAX_GAIN_H1),
            "max_g_h2": float(self.OBS_CAL_MAX_GAIN_H2 if self.ENABLE_HARMONIC2 else 0.0),
            "max_g_aux": float(self.OBS_CAL_MAX_GAIN_AUX),
            "max_g_b": float(self.OBS_CAL_MAX_GAIN_B),
            "max_g_r": float(self.OBS_CAL_MAX_GAIN_R),
            "prior_strength": float(self.OBS_CAL_PRIOR_STRENGTH),
            "max_lag_sec": min(0.15, float(self.OBS_CAL_MAX_LAG_SEC)),
        }
        if base == "of_disp_bridge":
            prior.update({
                "enabled": True,
                "obs_domain": str(sem.get("observation_domain", "displacement")),
                "g_h1": 1.00,
                "g_h2": 0.85,
                "g_aux": 0.12,
                "g_b": 0.10,
                "g_r": 0.18,
                "max_g_h1": min(float(self.OBS_CAL_MAX_GAIN_H1), 1.20),
                "max_g_h2": min(float(self.OBS_CAL_MAX_GAIN_H2 if self.ENABLE_HARMONIC2 else 0.0), 1.35),
                "max_g_aux": min(float(self.OBS_CAL_MAX_GAIN_AUX), 0.30),
                "max_g_b": min(float(self.OBS_CAL_MAX_GAIN_B), 0.25),
                "max_g_r": min(float(self.OBS_CAL_MAX_GAIN_R), 0.35),
                "prior_strength": max(float(self.OBS_CAL_PRIOR_STRENGTH), 1.10),
                "max_lag_sec": min(0.08, float(self.OBS_CAL_MAX_LAG_SEC)),
                "min_fit_corr": max(float(self.OF_MIN_FIT_CORR), float(self.OBS_CAL_MIN_FIT_CORR)),
                "max_fit_rmse_norm": min(float(self.OF_MAX_FIT_RMSE_NORM), float(self.OBS_CAL_MAX_FIT_RMSE_NORM)),
            })
        elif str(sem.get("family_group", "")).startswith("optical_flow"):
            harmonic_only = bool(self.OF_HARMONIC_ONLY)
            prior.update({
                "enabled": True,
                "obs_domain": str(sem.get("observation_domain", "velocity")),
                "g_h1": 1.00,
                "g_h2": 0.45 if self.ENABLE_HARMONIC2 else 0.0,
                "g_aux": 0.0 if harmonic_only else 0.08,
                "g_b": 0.0 if harmonic_only else 0.05,
                "g_r": 0.0 if harmonic_only else 0.10,
                "max_g_h1": min(float(self.OF_MAX_GAIN_H1), float(self.OBS_CAL_MAX_GAIN_H1), 1.10),
                "max_g_h2": min(
                    float(self.OF_MAX_GAIN_H2),
                    float(self.OBS_CAL_MAX_GAIN_H2 if self.ENABLE_HARMONIC2 else 0.0),
                    0.90,
                ),
                "max_g_aux": 0.0 if harmonic_only else min(float(self.OBS_CAL_MAX_GAIN_AUX), 0.18),
                "max_g_b": 0.0 if harmonic_only else min(float(self.OBS_CAL_MAX_GAIN_B), 0.10),
                "max_g_r": 0.0 if harmonic_only else min(float(self.OBS_CAL_MAX_GAIN_R), 0.20),
                "prior_strength": max(float(self.OF_PRIOR_STRENGTH), float(self.OBS_CAL_PRIOR_STRENGTH), 1.10),
                "max_lag_sec": min(float(self.OF_MAX_LAG_SEC), float(self.OBS_CAL_MAX_LAG_SEC)),
                "min_fit_corr": max(float(self.OF_MIN_FIT_CORR), float(self.OBS_CAL_MIN_FIT_CORR)),
                "max_fit_rmse_norm": min(float(self.OF_MAX_FIT_RMSE_NORM), float(self.OBS_CAL_MAX_FIT_RMSE_NORM)),
            })
        elif base == "profile1d_linear":
            prior.update({
                "enabled": True,
                "obs_domain": str(sem.get("observation_domain", "displacement")),
                "g_h1": 1.00,
                "g_h2": 0.35,
                "g_aux": 0.10,
                "g_b": 0.08,
                "g_r": 0.10,
                "max_g_aux": min(float(self.OBS_CAL_MAX_GAIN_AUX), 0.30),
                "max_g_b": min(float(self.OBS_CAL_MAX_GAIN_B), 0.20),
                "max_g_r": min(float(self.OBS_CAL_MAX_GAIN_R), 0.25),
                "prior_strength": max(float(self.OBS_CAL_PRIOR_STRENGTH), 1.0),
                "max_lag_sec": min(0.12, float(self.OBS_CAL_MAX_LAG_SEC)),
                "min_fit_corr": float(self.OBS_CAL_MIN_FIT_CORR),
                "max_fit_rmse_norm": float(self.OBS_CAL_MAX_FIT_RMSE_NORM),
            })
        elif str(sem.get("family_group", "")) == "profile1d_harmonic":
            harmonic_only = bool(self.QUADCUB_HARMONIC_ONLY)
            prior.update({
                "enabled": True,
                "obs_domain": str(sem.get("observation_domain", "displacement")),
                "g_h1": 1.00,
                "g_h2": 1.05,
                "g_aux": 0.0 if harmonic_only else 0.18,
                "g_b": 0.0 if harmonic_only else 0.12,
                "g_r": 0.0 if harmonic_only else 0.32,
                "max_g_h1": min(float(self.QUADCUB_MAX_GAIN_H1), float(self.OBS_CAL_MAX_GAIN_H1), 1.25),
                "max_g_h2": min(
                    float(self.QUADCUB_MAX_GAIN_H2),
                    float(self.OBS_CAL_MAX_GAIN_H2 if self.ENABLE_HARMONIC2 else 0.0),
                    1.60,
                ),
                "max_g_aux": 0.0 if harmonic_only else min(float(self.OBS_CAL_MAX_GAIN_AUX), 0.45),
                "max_g_b": 0.0 if harmonic_only else min(float(self.OBS_CAL_MAX_GAIN_B), 0.30),
                "max_g_r": 0.0 if harmonic_only else min(float(self.OBS_CAL_MAX_GAIN_R), 0.75),
                "prior_strength": max(
                    float(self.QUADCUB_PRIOR_STRENGTH) if harmonic_only else float(self.OBS_CAL_PRIOR_STRENGTH),
                    1.25,
                ),
                "max_lag_sec": min(
                    float(self.QUADCUB_MAX_LAG_SEC) if harmonic_only else 0.12,
                    float(self.OBS_CAL_MAX_LAG_SEC),
                ),
                "min_fit_corr": float(self.OBS_CAL_MIN_FIT_CORR),
                "max_fit_rmse_norm": float(self.OBS_CAL_MAX_FIT_RMSE_NORM),
            })
        if apply_allowed_filter:
            allowed = {
                token.strip().lower()
                for token in str(self.OBS_CAL_ALLOWED_FAMILIES or "").split(",")
                if token.strip()
            }
            if allowed and base not in allowed:
                prior["enabled"] = False
        return prior

    def _fallback_observation_calibration(
        self,
        base_method: str,
        default: Dict[str, float],
        y_fit: Optional[np.ndarray] = None,
        y_helper_fit: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Return a family-aware fixed observation prior when calibration is unavailable.

        This is intentionally conservative. It is currently only enabled as an
        experimental raw-OF fallback because raw OF is known to be velocity-like
        while the generic fixed-sum observation row is displacement-like.
        """
        out = dict(default)
        out["family"] = str(base_method or "").lower().strip()
        out["fallback_mode"] = "fixed_sum"

        base = out["family"]
        if not bool(self.OF_FIXED_VELOCITY_PRIOR):
            return out
        if (not base.startswith("of_")) or base == "of_disp_bridge":
            return out

        prior = self._obs_cal_family_prior(base, apply_allowed_filter=False)
        if not bool(prior.get("enabled", False)):
            return out

        obs_sign = 1.0
        try:
            if y_fit is not None and y_helper_fit is not None:
                y_fit = np.asarray(y_fit, dtype=np.float64).reshape(-1)
                y_helper_fit = np.asarray(y_helper_fit, dtype=np.float64).reshape(-1)
                valid = np.isfinite(y_fit) & np.isfinite(y_helper_fit)
                if int(np.count_nonzero(valid)) >= 8:
                    corr = float(np.corrcoef(y_fit[valid], y_helper_fit[valid])[0, 1])
                    if np.isfinite(corr) and corr < 0.0:
                        obs_sign = -1.0
        except Exception:
            obs_sign = 1.0

        g_h1 = float(prior.get("g_h1", 1.0))
        g_h2 = float(prior.get("g_h2", 0.0 if not self.ENABLE_HARMONIC2 else 1.0))
        g_aux = float(prior.get("g_aux", 0.0))
        g_b = float(prior.get("g_b", 0.0)) if self.ENABLE_BASELINE else 0.0
        g_r = float(prior.get("g_r", 0.0)) if self.ENABLE_RESIDUAL else 0.0
        g_osc = float(0.5 * (g_h1 + g_h2) if self.ENABLE_HARMONIC2 else g_h1)

        out.update({
            "enabled": False,
            "mode": "family_phase_aux",
            "obs_domain": "velocity",
            "obs_sign": float(obs_sign),
            "g_osc": g_osc,
            "g_b": g_b,
            "g_r": g_r,
            "g_h1": g_h1,
            "g_h2": g_h2 if self.ENABLE_HARMONIC2 else 0.0,
            "g_aux": g_aux,
            "g_osc_signed": float(obs_sign * g_osc),
            "g_b_signed": float(obs_sign * g_b) if self.ENABLE_BASELINE else 0.0,
            "g_r_signed": float(obs_sign * g_r) if self.ENABLE_RESIDUAL else 0.0,
            "g_h1_signed": float(obs_sign * g_h1),
            "g_h2_signed": float(obs_sign * g_h2) if self.ENABLE_HARMONIC2 else 0.0,
            "g_aux_signed": float(obs_sign * g_aux),
            "lag_sec": 0.0,
            "fallback_mode": "of_velocity_prior_v1",
        })
        return out

    def _preprocess(
        self,
        signal: np.ndarray,
        fs: float,
        family_override: Optional[str] = None,
    ) -> np.ndarray:
        """Inference path: preserve low-frequency structure for baseline/residual states.

        Unlike the generic oscillator heads, PARH should not remove the very
        baseline/trend content that its latent state is designed to represent.
        The helper path remains aggressively band-limited; the inference path is
        only lightly low-pass filtered and robustly scaled.
        """
        run_meta = getattr(self, "_current_run_meta", {}) or {}
        base_method = self._base_method_from_meta(run_meta, family_override=family_override)

        if not self.USE_LIGHT_OBS_PATH:
            return super()._preprocess(signal, fs)

        obs_mode = self._observation_family_mode(base_method)

        if obs_mode == "legacy":
            y = super()._preprocess(signal, fs)
            preproc_meta = getattr(self, "_last_preproc_meta", {}) or {}
            preproc_meta["observation_preprocess"] = {
                "mode": "family_aware_legacy_stack",
                "family": base_method,
                "policy": str(self.OBS_FAMILY_POLICY),
                "lowpass_enabled": False,
                "legacy_base_preprocess": True,
            }
            self._last_preproc_meta = preproc_meta
            return y

        if obs_mode == "helper":
            y = self._helper_preprocess(signal, fs)
            sigma_hat = float(np.std(y)) if y.size else 0.0
            self._last_sigma_y = float(max(sigma_hat, 0.0))
            self._last_signal_std = float(np.std(y)) if y.size else 0.0
            self._last_snr = float(self._last_signal_std / max(self._last_sigma_y, 1e-6)) if self._last_sigma_y else 0.0
            self._last_preproc_meta = {
                "robust_z": {
                    "enabled": True,
                    "med": float(np.median(y)) if y.size else 0.0,
                    "mad": float(np.median(np.abs(y - np.median(y)))) if y.size else 0.0,
                    "sigma_hat": float(max(sigma_hat, 0.0)),
                    "clip": None,
                    "clipped_frac": 0.0,
                },
                "observation_preprocess": {
                    "mode": "family_aware_helper_observation",
                    "family": base_method,
                    "policy": str(self.OBS_FAMILY_POLICY),
                    "helper_like": True,
                    "lowpass_enabled": False,
                    "legacy_base_preprocess": False,
                },
            }
            return y

        if obs_mode == "blend":
            y_legacy = super()._preprocess(signal, fs)
            y_helper = self._helper_preprocess(signal, fs)
            if y_helper.size != y_legacy.size or y_helper.size == 0:
                return y_legacy
            if np.all(np.isfinite(y_legacy)) and np.all(np.isfinite(y_helper)) and y_legacy.size > 8:
                corr = float(np.corrcoef(y_legacy, y_helper)[0, 1])
                if np.isfinite(corr) and corr < 0.0:
                    y_helper = -y_helper
            legacy_std = float(np.std(y_legacy)) if y_legacy.size else 0.0
            helper_std = float(np.std(y_helper)) if y_helper.size else 0.0
            if np.isfinite(legacy_std) and legacy_std > 1e-8 and np.isfinite(helper_std) and helper_std > 1e-8:
                y_helper = y_helper * (legacy_std / helper_std)
            alpha = float(np.clip(self.OBS_BLEND_ALPHA, 0.0, 1.0))
            y = (1.0 - alpha) * y_legacy + alpha * y_helper
            sigma_hat = float(np.std(y)) if y.size else 0.0
            self._last_sigma_y = float(max(sigma_hat, 0.0))
            self._last_signal_std = float(np.std(y)) if y.size else 0.0
            self._last_snr = float(self._last_signal_std / max(self._last_sigma_y, 1e-6)) if self._last_sigma_y else 0.0
            self._last_preproc_meta = {
                "robust_z": {
                    "enabled": True,
                    "med": float(np.median(y)) if y.size else 0.0,
                    "mad": float(np.median(np.abs(y - np.median(y)))) if y.size else 0.0,
                    "sigma_hat": float(max(sigma_hat, 0.0)),
                    "clip": None,
                    "clipped_frac": 0.0,
                },
                "observation_preprocess": {
                    "mode": "family_aware_blend_observation",
                    "family": base_method,
                    "policy": str(self.OBS_FAMILY_POLICY),
                    "blend_alpha": float(alpha),
                    "legacy_component": True,
                    "helper_component": True,
                    "lowpass_enabled": False,
                    "legacy_base_preprocess": False,
                },
            }
            return y

        x = np.asarray(signal, dtype=np.float64).copy()
        if x.size == 0:
            return x
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        preproc_cfg = getattr(self, "preproc_cfg", {}) or {}
        obs_cfg = preproc_cfg.get("parh_observation", {}) if isinstance(preproc_cfg.get("parh_observation"), dict) else {}
        robust_cfg = preproc_cfg.get("robust_zscore", {}) if isinstance(preproc_cfg.get("robust_zscore"), dict) else {}

        lowpass_enabled = bool(obs_cfg.get("lowpass_enabled", True))
        cutoff_default = max(2.0 * float(self.params.f_max), self.OBS_LIGHT_LOWPASS_HZ)
        try:
            lowpass_hz = float(obs_cfg.get("lowpass_hz", cutoff_default))
        except Exception:
            lowpass_hz = float(cutoff_default)
        if lowpass_enabled and fs > 0.0:
            nyq = 0.5 * fs
            cutoff = min(lowpass_hz, nyq - 1e-3)
            if cutoff > 0.0 and cutoff < nyq:
                b, a = sps.butter(2, cutoff / nyq, btype="lowpass")
                x = sps.filtfilt(b, a, x, method="gust")

        center_mode = str(obs_cfg.get("center", self.OBS_CENTER_MODE)).strip().lower()
        if center_mode == "mean":
            center = float(np.mean(x))
        else:
            center = float(np.median(x))
        if not np.isfinite(center):
            center = 0.0
        x = x - center

        abs_dev = np.abs(x)
        mad = float(np.median(abs_dev)) if abs_dev.size else 0.0
        if not np.isfinite(mad) or mad < 0.0:
            mad = 0.0
        sigma_hat = float(1.4826 * mad)
        if not np.isfinite(sigma_hat) or sigma_hat < 0.0:
            sigma_hat = 0.0

        robust_scale = bool(obs_cfg.get("robust_scale", True))
        eps = robust_cfg.get("eps", 1e-6)
        try:
            eps = float(eps)
        except Exception:
            eps = 1e-6
        eps = max(eps, 1e-6)

        clip_raw = obs_cfg.get("clip", robust_cfg.get("clip", self.OBS_CLIP_Z))
        clip_val = None
        if clip_raw is not None:
            try:
                clip_val = float(clip_raw)
            except Exception:
                clip_val = None

        clipped_frac = 0.0
        if robust_scale:
            denom = max(sigma_hat, eps)
            x = x / denom
            if clip_val is not None and clip_val > 0.0:
                clipped_frac = float(np.mean(np.abs(x) >= clip_val)) if x.size else 0.0
                x = np.clip(x, -clip_val, clip_val)

        self._last_sigma_y = float(sigma_hat if sigma_hat > 0.0 else 0.0)
        signal_std = float(np.std(x)) if x.size else 0.0
        self._last_signal_std = signal_std
        self._last_snr = float(signal_std / max(self._last_sigma_y, 1e-6)) if self._last_sigma_y else 0.0
        self._last_preproc_meta = {
            "robust_z": {
                "enabled": bool(robust_scale),
                "med": float(center),
                "mad": float(mad),
                "sigma_hat": float(sigma_hat),
                "clip": None if clip_val is None else float(clip_val),
                "clipped_frac": float(clipped_frac if robust_scale and clip_val is not None else 0.0),
            },
            "observation_preprocess": {
                "mode": "family_aware_light_lowpass_robust_scale",
                "family": base_method or "unknown",
                "policy": str(self.OBS_FAMILY_POLICY),
                "lowpass_enabled": bool(lowpass_enabled),
                "lowpass_hz": float(lowpass_hz),
                "center_mode": center_mode,
                "robust_scale": bool(robust_scale),
            },
        }
        return x

    def _helper_preprocess(self, signal: np.ndarray, fs: float) -> np.ndarray:
        """Separate helper path: stronger oscillatory cleanup than inference path."""
        x = np.asarray(signal, dtype=np.float64).copy()
        if x.size == 0:
            return x
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = sps.detrend(x, type="linear")
        if fs > 0.0:
            nyq = 0.5 * fs
            low = max(self.params.f_min, 0.05)
            high = min(self.params.f_max, nyq - 1e-3)
            if high > low:
                b, a = sps.butter(3, [low / nyq, high / nyq], btype="bandpass")
                x = sps.filtfilt(b, a, x, method="gust")
        mean = float(np.mean(x)) if x.size else 0.0
        std = float(np.std(x)) if x.size else 0.0
        if np.isfinite(std) and std > 1e-8:
            x = (x - mean) / std
        else:
            x = x - mean
        return x

    def _helper_features(
        self, signal: np.ndarray, fs: float, freq0: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build helper-path signal, envelope, and local frequency evidence."""
        y_helper = self._helper_preprocess(signal, fs)
        if y_helper.size == 0:
            return y_helper, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        analytic = sps.hilbert(y_helper)
        helper_amp = np.abs(analytic).astype(np.float64)
        helper_phase = np.unwrap(np.angle(analytic))
        if helper_phase.size > 1:
            inst = np.diff(helper_phase) * (fs / (2.0 * np.pi))
            helper_freq = np.empty(helper_phase.size, dtype=np.float64)
            helper_freq[0] = inst[0]
            helper_freq[1:] = inst
            k = min(helper_freq.size // 2 * 2 + 1, 5)
            if k >= 3:
                helper_freq = sps.medfilt(helper_freq, kernel_size=k)
        else:
            helper_freq = np.full(helper_phase.size, freq0, dtype=np.float64)
        bad = ~np.isfinite(helper_freq)
        if np.any(bad):
            helper_freq[bad] = freq0
        helper_freq = np.clip(helper_freq, self.params.f_min, self.params.f_max)
        return y_helper, helper_amp, helper_freq

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
        obs_nonosc_need: float = 0.0,
        residual_prior_scale: float = 1.0,
    ) -> np.ndarray:
        """8x8 process noise with disentangled quality-driven scaling.

        Q_osc components scaled by g(q_dyn):
            Q_osc = Q_osc_0 * (1 + Q_DYN_GAMMA * q_dyn)
        Q_aper components scaled by a family-aware residual drive:
            aper_drive = s_res * (1 - q_osc)
            Q_aper = Q_aper_0 * (1 + Q_APER_GAMMA * aper_drive)
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
            residual_prior_scale = float(np.clip(residual_prior_scale, 0.0, 1.0))
            aper_drive = residual_prior_scale * max(1.0 - q_osc, 0.0)
            aper_obs_drive = residual_prior_scale * float(np.clip(obs_nonosc_need, 0.0, 1.0))
            aper_scale = (
                1.0
                + self.Q_APER_GAMMA * aper_drive
                + self.Q_APER_OBS_GAMMA * aper_obs_drive
            )
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
        """1x8 default observation matrix: y = h_c1 + h_c2 + b + r."""
        return self._build_H_calibrated(1.0, 1.0, 1.0)

    def _build_H_calibrated(
        self,
        g_osc_signed: float,
        g_b_signed: float,
        g_r_signed: float,
    ) -> np.ndarray:
        """1x8 calibrated observation matrix.

        The harmonic blocks share the oscillatory gain. Baseline/residual keep
        their own signed gains. Once fixed after warm-up, inference remains
        conditionally linear.
        """
        H = np.zeros((1, 8), dtype=np.float64)
        H[0, self.HC1] = float(g_osc_signed)
        if self.ENABLE_HARMONIC2:
            H[0, self.HC2] = float(g_osc_signed)
        if self.ENABLE_BASELINE:
            H[0, self.B] = float(g_b_signed)
        if self.ENABLE_RESIDUAL:
            H[0, self.R] = float(g_r_signed)
        return H

    def _build_H_family_visibility(
        self,
        g_h1_signed: float,
        g_h2_signed: float,
        g_aux_signed: float,
        lag_sec: float,
        freq_hz: float,
    ) -> np.ndarray:
        """1x8 observation row with bounded harmonic phase visibility.

        A small family-specific lag is represented as a phase shift in the
        harmonic observation projection. This keeps inference linear once the
        warm-up parameters are fixed.
        """
        H = np.zeros((1, 8), dtype=np.float64)
        phi1 = 2.0 * np.pi * float(freq_hz) * float(lag_sec)
        H[0, self.HC1] = float(g_h1_signed) * np.cos(phi1)
        H[0, self.HS1] = float(g_h1_signed) * np.sin(phi1)
        if self.ENABLE_HARMONIC2:
            phi2 = 2.0 * phi1
            H[0, self.HC2] = float(g_h2_signed) * np.cos(phi2)
            H[0, self.HS2] = float(g_h2_signed) * np.sin(phi2)
        if self.ENABLE_BASELINE:
            H[0, self.B] = float(g_aux_signed)
        if self.ENABLE_RESIDUAL:
            H[0, self.R] = float(g_aux_signed)
        return H

    def _build_H_family_visibility_split(
        self,
        g_h1_signed: float,
        g_h2_signed: float,
        g_b_signed: float,
        g_r_signed: float,
        lag_sec: float,
        freq_hz: float,
    ) -> np.ndarray:
        """1x8 observation row with family-specific harmonic and aux visibility."""
        H = np.zeros((1, 8), dtype=np.float64)
        phi1 = 2.0 * np.pi * float(freq_hz) * float(lag_sec)
        H[0, self.HC1] = float(g_h1_signed) * np.cos(phi1)
        H[0, self.HS1] = float(g_h1_signed) * np.sin(phi1)
        if self.ENABLE_HARMONIC2:
            phi2 = 2.0 * phi1
            H[0, self.HC2] = float(g_h2_signed) * np.cos(phi2)
            H[0, self.HS2] = float(g_h2_signed) * np.sin(phi2)
        if self.ENABLE_BASELINE:
            H[0, self.B] = float(g_b_signed)
        if self.ENABLE_RESIDUAL:
            H[0, self.R] = float(g_r_signed)
        return H

    def _build_H_family_velocity(
        self,
        g_h1_signed: float,
        g_h2_signed: float,
        g_aux_signed: float,
        lag_sec: float,
        freq_hz: float,
    ) -> np.ndarray:
        """1x8 observation row for velocity-like families such as raw OF."""
        H = np.zeros((1, 8), dtype=np.float64)
        omega = 2.0 * np.pi * float(freq_hz)
        phi1 = omega * float(lag_sec)
        H[0, self.HC1] = float(g_h1_signed) * omega * np.sin(phi1)
        H[0, self.HS1] = -float(g_h1_signed) * omega * np.cos(phi1)
        if self.ENABLE_HARMONIC2:
            phi2 = 2.0 * phi1
            H[0, self.HC2] = float(g_h2_signed) * (2.0 * omega) * np.sin(phi2)
            H[0, self.HS2] = -float(g_h2_signed) * (2.0 * omega) * np.cos(phi2)
        if self.ENABLE_BASELINE:
            H[0, self.BDOT] = float(g_aux_signed)
        if self.ENABLE_RESIDUAL:
            H[0, self.RDOT] = float(g_aux_signed)
        return H

    def _build_H_family_velocity_split(
        self,
        g_h1_signed: float,
        g_h2_signed: float,
        g_b_signed: float,
        g_r_signed: float,
        lag_sec: float,
        freq_hz: float,
    ) -> np.ndarray:
        """1x8 velocity-like observation row with split aux visibility."""
        H = np.zeros((1, 8), dtype=np.float64)
        omega = 2.0 * np.pi * float(freq_hz)
        phi1 = omega * float(lag_sec)
        H[0, self.HC1] = float(g_h1_signed) * omega * np.sin(phi1)
        H[0, self.HS1] = -float(g_h1_signed) * omega * np.cos(phi1)
        if self.ENABLE_HARMONIC2:
            phi2 = 2.0 * phi1
            H[0, self.HC2] = float(g_h2_signed) * (2.0 * omega) * np.sin(phi2)
            H[0, self.HS2] = -float(g_h2_signed) * (2.0 * omega) * np.cos(phi2)
        if self.ENABLE_BASELINE:
            H[0, self.BDOT] = float(g_b_signed)
        if self.ENABLE_RESIDUAL:
            H[0, self.RDOT] = float(g_r_signed)
        return H

    def _build_H_from_obs_cal(
        self,
        obs_cal: Dict[str, float],
        freq_hz: float,
    ) -> np.ndarray:
        """Build the active observation row for the current frame."""
        mode = str(obs_cal.get("mode", "")).strip().lower()
        obs_domain = str(obs_cal.get("obs_domain", "displacement")).strip().lower()
        fallback_mode = str(obs_cal.get("fallback_mode", "")).strip().lower()
        if (not bool(obs_cal.get("enabled", False))) and fallback_mode == "of_velocity_prior_v1":
            return self._build_H_family_velocity_split(
                obs_cal.get("g_h1_signed", obs_cal.get("g_osc_signed", 1.0)),
                obs_cal.get("g_h2_signed", obs_cal.get("g_osc_signed", 1.0)),
                obs_cal.get("g_b_signed", obs_cal.get("g_aux_signed", 0.0)),
                obs_cal.get("g_r_signed", obs_cal.get("g_aux_signed", 0.0)),
                obs_cal.get("lag_sec", 0.0),
                freq_hz,
            )
        if bool(obs_cal.get("enabled", False)) and mode == "family_phase_split_aux":
            if obs_domain == "velocity":
                return self._build_H_family_velocity_split(
                    obs_cal.get("g_h1_signed", obs_cal.get("g_osc_signed", 1.0)),
                    obs_cal.get("g_h2_signed", obs_cal.get("g_osc_signed", 1.0)),
                    obs_cal.get("g_b_signed", obs_cal.get("g_aux_signed", 1.0)),
                    obs_cal.get("g_r_signed", obs_cal.get("g_aux_signed", 1.0)),
                    obs_cal.get("lag_sec", 0.0),
                    freq_hz,
                )
            return self._build_H_family_visibility_split(
                obs_cal.get("g_h1_signed", obs_cal.get("g_osc_signed", 1.0)),
                obs_cal.get("g_h2_signed", obs_cal.get("g_osc_signed", 1.0)),
                obs_cal.get("g_b_signed", obs_cal.get("g_aux_signed", 1.0)),
                obs_cal.get("g_r_signed", obs_cal.get("g_aux_signed", 1.0)),
                obs_cal.get("lag_sec", 0.0),
                freq_hz,
            )
        if bool(obs_cal.get("enabled", False)) and mode == "family_phase_aux":
            if obs_domain == "velocity":
                return self._build_H_family_velocity(
                    obs_cal.get("g_h1_signed", obs_cal.get("g_osc_signed", 1.0)),
                    obs_cal.get("g_h2_signed", obs_cal.get("g_osc_signed", 1.0)),
                    obs_cal.get("g_aux_signed", obs_cal.get("g_b_signed", 1.0)),
                    obs_cal.get("lag_sec", 0.0),
                    freq_hz,
                )
            return self._build_H_family_visibility(
                obs_cal.get("g_h1_signed", obs_cal.get("g_osc_signed", 1.0)),
                obs_cal.get("g_h2_signed", obs_cal.get("g_osc_signed", 1.0)),
                obs_cal.get("g_aux_signed", obs_cal.get("g_b_signed", 1.0)),
                obs_cal.get("lag_sec", 0.0),
                freq_hz,
            )
        return self._build_H_calibrated(
            obs_cal.get("g_osc_signed", 1.0),
            obs_cal.get("g_b_signed", 1.0),
            obs_cal.get("g_r_signed", 1.0),
        )

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
        self, helper_freq_t: float, freq_t: float,
        helper_amp_t: float, helper_amp_baseline: float,
    ) -> float:
        """q_dyn(t): dynamical novelty [0, 1].

        Higher q_dyn = physiology is changing → increase Q_osc.
        Based on:
          1. Helper-path local frequency deviation from the current track
          2. Helper-path envelope change
        """
        freq_dev = abs(helper_freq_t - freq_t) / max(self.Q_DYN_FREQ_REF_HZ, 1e-6)
        amp_rel = abs(helper_amp_t - helper_amp_baseline) / max(helper_amp_baseline, 0.25)
        amp_dev = np.log1p(amp_rel)

        raw = (
            self.Q_DYN_FREQ_WEIGHT * (freq_dev ** 2) +
            self.Q_DYN_AMP_WEIGHT * (amp_dev ** 2)
        )
        q_dyn = float(np.clip(1.0 - np.exp(-raw), 0.0, 1.0))
        return q_dyn

    def _compute_helper_trust(
        self,
        helper_freq_win: np.ndarray,
        support_win: np.ndarray,
        track_freq_win: np.ndarray,
        base_method: str,
    ) -> Tuple[float, float]:
        """Estimate whether helper evidence is locally trustworthy.

        helper_trust:
          high when helper support is strong and helper frequency is stable.
        helper_bias_conf:
          high only when helper-track mismatch is both large enough and
          sign-consistent over the recent horizon.

        This is OF-only for now because direct OF observation replacement
        failed; the helper remains evidence, not a peer waveform sensor.
        """
        if not self._helper_trust_allowed(base_method):
            return 1.0, 0.0
        hf = np.asarray(helper_freq_win, dtype=np.float64).reshape(-1)
        sup = np.asarray(support_win, dtype=np.float64).reshape(-1)
        tf = np.asarray(track_freq_win, dtype=np.float64).reshape(-1)
        n = min(hf.size, sup.size, tf.size)
        if n <= 0:
            return 1.0, 0.0
        hf = hf[-n:]
        sup = sup[-n:]
        tf = tf[-n:]
        valid = np.isfinite(hf) & np.isfinite(sup) & np.isfinite(tf)
        if int(np.count_nonzero(valid)) < max(4, n // 2):
            return 1.0, 0.0

        hf = hf[valid]
        sup = sup[valid]
        tf = tf[valid]

        helper_std = float(np.std(hf))
        support_med = float(np.median(np.clip(sup, 0.0, 1.0)))
        stability_term = float(np.exp(
            -0.5 * (helper_std / max(float(self.HELPER_TRUST_STD_REF_HZ), 1e-6)) ** 2
        ))
        helper_trust = float(np.clip(
            0.55 * support_med + 0.45 * stability_term,
            0.0,
            1.0,
        ))

        signed_mismatch = hf - tf
        strong = signed_mismatch[np.abs(signed_mismatch) >= max(float(self.HELPER_TRUST_MIN_MISMATCH_HZ), 1e-6)]
        if strong.size < max(4, n // 3):
            return helper_trust, 0.0
        sign_stability = abs(float(np.mean(np.sign(strong))))
        mismatch_mag = float(np.median(np.abs(strong)))
        mismatch_term = float(np.clip(
            mismatch_mag / max(float(self.HELPER_TRUST_MISMATCH_REF_HZ), 1e-6),
            0.0,
            1.0,
        ))
        helper_bias_conf = float(np.clip(helper_trust * sign_stability * mismatch_term, 0.0, 1.0))
        return helper_trust, helper_bias_conf

    def _compute_q_osc(
        self,
        phase_coh: float,
        helper_support: float,
        freq_lock: float,
        obs_osc_support: float = 1.0,
        obs_full_support: float = 1.0,
        q_obs: float = 1.0,
    ) -> Tuple[float, float]:
        """q_osc(t): oscillatory support [0, 1].

        Higher q_osc = current segment is well-described by oscillator.
        Based on:
          1. Phase coherence (Gaussian kernel on phase prediction error)
          2. Helper-path oscillator fit
          3. Helper-path local frequency lock
        """
        base = (
            self.Q_OSC_PHASE_WEIGHT * phase_coh +
            self.Q_OSC_HELPER_WEIGHT * helper_support +
            self.Q_OSC_FREQ_WEIGHT * freq_lock
        )
        q_obs_clip = float(np.clip(q_obs, 0.0, 1.0))
        obs_support_clip = float(np.clip(obs_osc_support, 0.0, 1.0))
        obs_full_clip = float(np.clip(obs_full_support, 0.0, 1.0))
        obs_w = float(np.clip(self.Q_OSC_OBS_WEIGHT, 0.0, 1.0))
        mode = str(self.Q_OSC_OBS_MODE or "blend_support").strip().lower()

        if mode == "penalize_unexplained_v1":
            band = max(float(self.Q_OSC_OBS_BAND), 1e-6)
            ref = float(np.clip(self.Q_OSC_OBS_REF, 0.05, 1.0))
            clean_unexplained = q_obs_clip * float(np.clip((ref - obs_support_clip) / band, 0.0, 1.0))
            q_osc = float(np.clip(base - obs_w * clean_unexplained, 0.05, 1.0))
            return q_osc, clean_unexplained

        if mode == "penalize_nonosc_gap_v1":
            nonosc_gap = float(np.clip(obs_full_clip - obs_support_clip, 0.0, 1.0))
            unexplained_gap = float(np.clip(1.0 - obs_full_clip, 0.0, 1.0))
            clean_nonosc_need = q_obs_clip * float(np.clip(0.75 * nonosc_gap + 0.25 * unexplained_gap, 0.0, 1.0))
            q_osc = float(np.clip(base - obs_w * clean_nonosc_need, 0.05, 1.0))
            return q_osc, clean_nonosc_need

        clean_obs_support = q_obs_clip * obs_support_clip + (1.0 - q_obs_clip) * 1.0
        q_osc = float(np.clip((1.0 - obs_w) * base + obs_w * clean_obs_support, 0.05, 1.0))
        return q_osc, (1.0 - clean_obs_support)

    def _residual_prior_scale(self, base_method: str) -> float:
        if not bool(self.ENABLE_RESIDUAL_SEMANTICS):
            return 1.0
        sem = self._observation_family_semantics(base_method)
        scale = float(sem.get("residual_observability", 1.0))
        scale = float(np.clip(scale, 0.0, 1.0))
        scale = max(scale, float(self.RESIDUAL_PRIOR_MIN))
        power = max(float(self.RESIDUAL_PRIOR_POWER), 1e-6)
        return float(np.clip(scale ** power, 0.0, 1.0))

    # ─────────────────────────────────────────────
    #  ROBUST SCALE & STUDENT-T
    # ─────────────────────────────────────────────

    @staticmethod
    def _ew_mad_update(abs_e: float, mad_prev: float, alpha: float) -> float:
        """Exponentially-weighted MAD: robust innovation scale."""
        return alpha * mad_prev + (1.0 - alpha) * abs_e

    def _student_t_vb_update(
        self, e_t: float, HPH: float, R_eff: float, nu_t: float,
        x: np.ndarray, P: np.ndarray, H: np.ndarray, I_D: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Student-t VB robust update. Returns (x_upd, P_upd, lambda_t)."""
        lambda_t = 1.0
        for _ in range(self.VB_ITERS):
            S_latent = HPH + (R_eff / max(lambda_t, 1e-12))
            lambda_t = (nu_t + 1.0) / (nu_t + (e_t ** 2) / max(S_latent, 1e-12))
            lambda_t = float(np.clip(lambda_t, 0.01, 100.0))

        R_eff_vb = R_eff / max(lambda_t, 1e-12)
        S_vb = HPH + R_eff_vb
        if S_vb <= 1e-12 or not np.isfinite(S_vb):
            S_vb = 1e-12
        K_vb = (P @ H.T) / S_vb
        x_upd = x + K_vb[:, 0] * e_t
        P_upd = (I_D - K_vb @ H) @ P
        P_upd = 0.5 * (P_upd + P_upd.T)
        return x_upd, P_upd, lambda_t

    def _student_t_vb_update_multichannel(
        self,
        innovation: np.ndarray,
        R_eff_diag: np.ndarray,
        nu_t: float,
        x: np.ndarray,
        P: np.ndarray,
        H: np.ndarray,
        I_D: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Multivariate Student-t VB update for stacked observations."""
        innovation = np.asarray(innovation, dtype=np.float64).reshape(-1)
        R_diag = np.asarray(R_eff_diag, dtype=np.float64).reshape(-1)
        m_obs = innovation.size
        if m_obs == 0:
            return x, P, 1.0
        lambda_t = 1.0
        for _ in range(self.VB_ITERS):
            R_vb = np.diag(R_diag / max(lambda_t, 1e-12))
            S_vb = H @ P @ H.T + R_vb
            S_vb = 0.5 * (S_vb + S_vb.T)
            try:
                S_inv = np.linalg.pinv(S_vb)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S_vb + 1e-9 * np.eye(m_obs, dtype=np.float64))
            quad = float(innovation.T @ S_inv @ innovation)
            lambda_t = (nu_t + float(m_obs)) / max(nu_t + quad, 1e-12)
            lambda_t = float(np.clip(lambda_t, 0.01, 100.0))

        R_vb = np.diag(R_diag / max(lambda_t, 1e-12))
        S_vb = H @ P @ H.T + R_vb
        S_vb = 0.5 * (S_vb + S_vb.T)
        try:
            S_inv = np.linalg.pinv(S_vb)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S_vb + 1e-9 * np.eye(m_obs, dtype=np.float64))
        K_vb = P @ H.T @ S_inv
        x_upd = x + K_vb @ innovation
        P_upd = (I_D - K_vb @ H) @ P
        P_upd = 0.5 * (P_upd + P_upd.T)
        return x_upd, P_upd, lambda_t

    def _run_warmup_provisional_filter(
        self,
        y: np.ndarray,
        fs: float,
        freq0: float,
        qx: float,
        dt: float,
        rho: float,
        R_init: float,
    ) -> np.ndarray:
        """Short provisional forward pass used only for warm-up calibration."""
        n_cal = int(min(y.size, max(int(self.OBS_CAL_WARMUP_SEC * fs), int(4.0 * fs))))
        if n_cal <= 0:
            return np.zeros((0, self.STATE_DIM), dtype=np.float64)
        x = np.zeros(self.STATE_DIM, dtype=np.float64)
        P = self._build_Q_disentangled(qx, dt, q_dyn=0.0, q_osc=1.0).copy()
        H0 = self._build_H()
        I_D = np.eye(self.STATE_DIM, dtype=np.float64)
        omega = 2.0 * np.pi * freq0
        Q = self._build_Q_disentangled(qx, dt, q_dyn=0.0, q_osc=1.0)
        x_hist = np.zeros((n_cal, self.STATE_DIM), dtype=np.float64)
        for t in range(n_cal):
            F = self._build_F(omega, dt, rho)
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            e_t = float(y[t] - (H0 @ x_pred))
            S = float(H0 @ P_pred @ H0.T) + max(R_init, 1e-9)
            if not np.isfinite(S) or S <= 1e-12:
                S = 1e-12
            K = (P_pred @ H0.T) / S
            x = x_pred + K[:, 0] * e_t
            P = (I_D - K @ H0) @ P_pred
            P = 0.5 * (P + P.T)
            x_hist[t] = x
        return x_hist

    def _warmup_observation_calibration(
        self,
        y: np.ndarray,
        y_helper: np.ndarray,
        fs: float,
        freq0: float,
        qx: float,
        dt: float,
        rho: float,
        R_init: float,
        family_override: Optional[str] = None,
    ) -> Dict[str, float]:
        """Estimate fixed trial-level observation parameters from warm-up only."""
        run_meta = getattr(self, "_current_run_meta", {}) or {}
        base_method = self._base_method_from_meta(run_meta, family_override=family_override)
        default = {
            "offset": 0.0,
            "obs_sign": 1.0,
            "obs_domain": "displacement",
            "g_osc": 1.0,
            "g_b": 1.0,
            "g_r": 1.0,
            "g_h1": 1.0,
            "g_h2": 1.0 if self.ENABLE_HARMONIC2 else 0.0,
            "g_aux": 1.0 if (self.ENABLE_BASELINE or self.ENABLE_RESIDUAL) else 0.0,
            "g_osc_signed": 1.0,
            "g_b_signed": 1.0,
            "g_r_signed": 1.0,
            "g_h1_signed": 1.0,
            "g_h2_signed": 1.0 if self.ENABLE_HARMONIC2 else 0.0,
            "g_aux_signed": 1.0 if (self.ENABLE_BASELINE or self.ENABLE_RESIDUAL) else 0.0,
            "lag_sec": 0.0,
            "fit_corr": np.nan,
            "fit_rmse": np.nan,
            "n_frames": 0,
            "mode": str(self.OBS_CAL_MODE),
            "enabled": bool(self.ENABLE_OBS_CAL),
            "family": base_method,
            "fallback_mode": "fixed_sum",
        }
        if (not self.ENABLE_OBS_CAL) or y.size < max(int(4.0 * fs), 24):
            return self._fallback_observation_calibration(base_method, default)

        x_hist = self._run_warmup_provisional_filter(y, fs, freq0, qx, dt, rho, R_init)
        if x_hist.shape[0] < max(int(4.0 * fs), 24):
            return self._fallback_observation_calibration(base_method, default)

        z_osc, _ = self._extract_outputs(x_hist)
        b = x_hist[:, self.B].copy() if self.ENABLE_BASELINE else np.zeros(x_hist.shape[0], dtype=np.float64)
        r = x_hist[:, self.R].copy() if self.ENABLE_RESIDUAL else np.zeros(x_hist.shape[0], dtype=np.float64)

        n_cal = x_hist.shape[0]
        skip = min(n_cal // 2, int(max(0.0, self.OBS_CAL_SKIP_SEC) * fs))
        if skip >= n_cal - 8:
            skip = 0
        y_fit = np.asarray(y[skip:n_cal], dtype=np.float64)
        y_helper_fit = np.asarray(y_helper[skip:n_cal], dtype=np.float64) if y_helper.size >= n_cal else y_fit
        z_fit = np.asarray(z_osc[skip:n_cal], dtype=np.float64)
        b_fit = np.asarray(b[skip:n_cal], dtype=np.float64)
        r_fit = np.asarray(r[skip:n_cal], dtype=np.float64)

        finite = np.isfinite(y_fit) & np.isfinite(z_fit) & np.isfinite(b_fit) & np.isfinite(r_fit)
        if np.count_nonzero(finite) < 16:
            return default
        y_fit = y_fit[finite]
        y_helper_fit = y_helper_fit[finite]
        z_fit = z_fit[finite]
        b_fit = b_fit[finite]
        r_fit = r_fit[finite]

        allowed_families = {
            "of_farneback",
            "profile1d_linear",
            "profile1d_quadratic",
            "profile1d_cubic",
        }
        if base_method not in allowed_families:
            default["enabled"] = False
            return self._fallback_observation_calibration(base_method, default, y_fit, y_helper_fit)

        ridge = max(float(self.OBS_CAL_RIDGE), 1e-8)
        prior_strength = max(float(self.OBS_CAL_PRIOR_STRENGTH), 0.0)
        mode = str(self.OBS_CAL_MODE).strip().lower()
        z_full_fit = z_fit + b_fit + r_fit
        if mode in {"family_phase_aux", "family_phase_split_aux"}:
            prior = self._obs_cal_family_prior(base_method)
            if not bool(prior.get("enabled", False)):
                default["enabled"] = False
                return self._fallback_observation_calibration(base_method, default, y_fit, y_helper_fit)
            obs_domain = str(prior.get("obs_domain", "displacement")).strip().lower()
            prior_strength_eff = max(float(prior.get("prior_strength", prior_strength)), 0.0)
            g_h1_max = max(float(prior.get("max_g_h1", self.OBS_CAL_MAX_GAIN_H1)), 1e-6)
            g_h2_max = max(
                float(prior.get("max_g_h2", self.OBS_CAL_MAX_GAIN_H2 if self.ENABLE_HARMONIC2 else 0.0)),
                0.0,
            )
            g_aux_max = max(float(prior.get("max_g_aux", self.OBS_CAL_MAX_GAIN_AUX)), 0.0)
            g_b_max = max(float(prior.get("max_g_b", self.OBS_CAL_MAX_GAIN_B)), 0.0)
            g_r_max = max(float(prior.get("max_g_r", self.OBS_CAL_MAX_GAIN_R)), 0.0)

            hc1_fit = np.asarray(x_hist[:, self.HC1][skip:n_cal], dtype=np.float64)[finite]
            hs1_fit = np.asarray(x_hist[:, self.HS1][skip:n_cal], dtype=np.float64)[finite]
            if self.ENABLE_HARMONIC2:
                hc2_fit = np.asarray(x_hist[:, self.HC2][skip:n_cal], dtype=np.float64)[finite]
                hs2_fit = np.asarray(x_hist[:, self.HS2][skip:n_cal], dtype=np.float64)[finite]
            else:
                hc2_fit = np.zeros_like(hc1_fit)
                hs2_fit = np.zeros_like(hs1_fit)
            bdot_fit = np.asarray(x_hist[:, self.BDOT][skip:n_cal], dtype=np.float64)[finite] if self.ENABLE_BASELINE else np.zeros_like(b_fit)
            rdot_fit = np.asarray(x_hist[:, self.RDOT][skip:n_cal], dtype=np.float64)[finite] if self.ENABLE_RESIDUAL else np.zeros_like(r_fit)
            aux_fit = (bdot_fit + rdot_fit) if obs_domain == "velocity" else (b_fit + r_fit)

            max_lag_sec = max(0.0, min(float(prior.get("max_lag_sec", 0.0)), float(self.OBS_CAL_MAX_LAG_SEC)))
            lag_step_sec = 1.0 / max(fs, 1e-6)
            lag_count = max(0, int(np.floor(max_lag_sec / lag_step_sec)))
            lag_grid = np.arange(-lag_count, lag_count + 1, dtype=int) / max(fs, 1e-6)
            if lag_grid.size == 0:
                lag_grid = np.array([0.0], dtype=np.float64)

            y_std = float(np.std(y_fit)) if y_fit.size else 1.0
            y_std = max(y_std, 1e-6)
            best = None
            for obs_sign in (1.0, -1.0):
                for lag_sec in lag_grid:
                    phi1 = 2.0 * np.pi * float(freq0) * float(lag_sec)
                    phi2 = 2.0 * phi1
                    if obs_domain == "velocity":
                        omega = 2.0 * np.pi * float(freq0)
                        p1 = omega * (np.sin(phi1) * hc1_fit - np.cos(phi1) * hs1_fit)
                        p2 = (
                            (2.0 * omega) * (np.sin(phi2) * hc2_fit - np.cos(phi2) * hs2_fit)
                            if self.ENABLE_HARMONIC2 else np.zeros_like(hc1_fit)
                        )
                    else:
                        p1 = np.cos(phi1) * hc1_fit + np.sin(phi1) * hs1_fit
                        p2 = np.cos(phi2) * hc2_fit + np.sin(phi2) * hs2_fit if self.ENABLE_HARMONIC2 else np.zeros_like(p1)

                    if mode == "family_phase_split_aux":
                        A = np.column_stack([
                            np.ones_like(y_fit),
                            obs_sign * p1,
                            obs_sign * p2,
                            obs_sign * (bdot_fit if obs_domain == "velocity" else b_fit),
                            obs_sign * (rdot_fit if obs_domain == "velocity" else r_fit),
                        ])
                        prior_beta = np.array([
                            0.0,
                            float(prior.get("g_h1", 1.0)),
                            float(prior.get("g_h2", 0.0)) if self.ENABLE_HARMONIC2 else 0.0,
                            float(prior.get("g_b", 0.0)) if self.ENABLE_BASELINE else 0.0,
                            float(prior.get("g_r", 0.0)) if self.ENABLE_RESIDUAL else 0.0,
                        ], dtype=np.float64)
                    else:
                        A = np.column_stack([
                            np.ones_like(y_fit),
                            obs_sign * p1,
                            obs_sign * p2,
                            obs_sign * aux_fit,
                        ])
                        prior_beta = np.array([
                            0.0,
                            float(prior.get("g_h1", 1.0)),
                            float(prior.get("g_h2", 0.0)) if self.ENABLE_HARMONIC2 else 0.0,
                            float(prior.get("g_aux", 0.0)),
                        ], dtype=np.float64)

                    reg = ridge * np.eye(A.shape[1], dtype=np.float64)
                    reg[0, 0] = 0.0
                    rhs = A.T @ y_fit
                    if prior_strength_eff > 0.0:
                        reg[1:, 1:] += prior_strength_eff * np.eye(A.shape[1] - 1, dtype=np.float64)
                        rhs[1:] += prior_strength_eff * prior_beta[1:]
                    try:
                        beta = np.linalg.solve(A.T @ A + reg, rhs)
                    except np.linalg.LinAlgError:
                        beta, *_ = np.linalg.lstsq(A, y_fit, rcond=None)

                    g_h1 = float(np.clip(beta[1], 0.0, g_h1_max))
                    g_h2 = float(np.clip(beta[2], 0.0, g_h2_max)) if self.ENABLE_HARMONIC2 else 0.0
                    if mode == "family_phase_split_aux":
                        g_b = float(np.clip(beta[3], 0.0, g_b_max)) if self.ENABLE_BASELINE else 0.0
                        g_r = float(np.clip(beta[4], 0.0, g_r_max)) if self.ENABLE_RESIDUAL else 0.0
                        aux_b_fit = bdot_fit if obs_domain == "velocity" else b_fit
                        aux_r_fit = rdot_fit if obs_domain == "velocity" else r_fit
                        offset = float(np.mean(y_fit - (obs_sign * (g_h1 * p1 + g_h2 * p2 + g_b * aux_b_fit + g_r * aux_r_fit))))
                        y_hat = offset + obs_sign * (g_h1 * p1 + g_h2 * p2 + g_b * aux_b_fit + g_r * aux_r_fit)
                    else:
                        g_aux = float(np.clip(beta[3], 0.0, g_aux_max))
                        g_b = g_aux if self.ENABLE_BASELINE else 0.0
                        g_r = g_aux if self.ENABLE_RESIDUAL else 0.0
                        offset = float(np.mean(y_fit - (obs_sign * (g_h1 * p1 + g_h2 * p2 + g_aux * aux_fit))))
                        y_hat = offset + obs_sign * (g_h1 * p1 + g_h2 * p2 + g_aux * aux_fit)

                    fit_corr = float(np.corrcoef(y_fit, y_hat)[0, 1]) if y_fit.size > 3 else np.nan
                    fit_rmse = float(np.sqrt(np.mean((y_fit - y_hat) ** 2))) if y_fit.size > 0 else np.nan
                    if not np.isfinite(fit_corr):
                        fit_corr = -1.0
                    rmse_norm = fit_rmse / y_std if np.isfinite(fit_rmse) else np.inf
                    if mode == "family_phase_split_aux":
                        prior_dev = (
                            abs(g_h1 - prior_beta[1]) / max(g_h1_max, 1e-6) +
                            (abs(g_h2 - prior_beta[2]) / max(g_h2_max, 1.0) if self.ENABLE_HARMONIC2 else 0.0) +
                            (abs(g_b - prior_beta[3]) / max(g_b_max, 1e-6) if self.ENABLE_BASELINE else 0.0) +
                            (abs(g_r - prior_beta[4]) / max(g_r_max, 1e-6) if self.ENABLE_RESIDUAL else 0.0)
                        )
                    else:
                        prior_dev = (
                            abs(g_h1 - prior_beta[1]) / max(g_h1_max, 1e-6) +
                            (abs(g_h2 - prior_beta[2]) / max(g_h2_max, 1.0) if self.ENABLE_HARMONIC2 else 0.0) +
                            abs(g_aux - prior_beta[3]) / max(g_aux_max, 1e-6)
                        )
                    lag_pen = abs(float(lag_sec)) / max(max_lag_sec, 1e-6) if max_lag_sec > 0.0 else 0.0
                    aux_pen = 0.0
                    if mode == "family_phase_split_aux":
                        aux_pen = (
                            (g_b / max(g_b_max, 1e-6) if self.ENABLE_BASELINE else 0.0) +
                            (g_r / max(g_r_max, 1e-6) if self.ENABLE_RESIDUAL else 0.0)
                        )
                    score = fit_corr - 0.12 * rmse_norm - 0.04 * prior_dev - 0.03 * lag_pen - 0.015 * aux_pen
                    candidate = {
                        "score": float(score),
                        "offset": offset,
                        "obs_sign": float(obs_sign),
                        "g_h1": g_h1,
                        "g_h2": g_h2,
                        "g_aux": max(g_b, g_r),
                        "g_b": g_b,
                        "g_r": g_r,
                        "lag_sec": float(lag_sec),
                        "fit_corr": float(fit_corr),
                        "fit_rmse": float(fit_rmse),
                    }
                    if best is None or candidate["score"] > best["score"]:
                        best = candidate

            if best is None:
                return self._fallback_observation_calibration(base_method, default, y_fit, y_helper_fit)

            min_fit_corr = float(prior.get("min_fit_corr", self.OBS_CAL_MIN_FIT_CORR))
            max_fit_rmse_norm = float(prior.get("max_fit_rmse_norm", self.OBS_CAL_MAX_FIT_RMSE_NORM))
            best_fit_corr = float(best.get("fit_corr", np.nan))
            best_fit_rmse = float(best.get("fit_rmse", np.nan))
            best_rmse_norm = best_fit_rmse / y_std if np.isfinite(best_fit_rmse) else np.inf
            if (not np.isfinite(best_fit_corr)) or best_fit_corr < min_fit_corr or best_rmse_norm > max_fit_rmse_norm:
                default["enabled"] = False
                return self._fallback_observation_calibration(base_method, default, y_fit, y_helper_fit)

            g_h1 = float(best["g_h1"])
            g_h2 = float(best["g_h2"])
            g_aux = float(best["g_aux"])
            g_b = float(best.get("g_b", g_aux if self.ENABLE_BASELINE else 0.0))
            g_r = float(best.get("g_r", g_aux if self.ENABLE_RESIDUAL else 0.0))
            obs_sign = float(best["obs_sign"])
            g_h1_signed = obs_sign * g_h1
            g_h2_signed = obs_sign * g_h2
            g_aux_signed = obs_sign * g_aux
            g_osc = float(0.5 * (g_h1 + g_h2) if self.ENABLE_HARMONIC2 else g_h1)
            g_osc_signed = obs_sign * g_osc
            g_b_signed = obs_sign * g_b if self.ENABLE_BASELINE else 0.0
            g_r_signed = obs_sign * g_r if self.ENABLE_RESIDUAL else 0.0
            offset = float(best["offset"])
            fit_corr = float(best["fit_corr"])
            fit_rmse = float(best["fit_rmse"])
            lag_sec = float(best["lag_sec"])
        elif mode == "global_signed_gain":
            A = np.column_stack([
                np.ones_like(y_fit),
                z_full_fit,
            ])
            reg = ridge * np.eye(A.shape[1], dtype=np.float64)
            reg[0, 0] = 0.0
            try:
                beta = np.linalg.solve(A.T @ A + reg, A.T @ y_fit)
            except np.linalg.LinAlgError:
                beta, *_ = np.linalg.lstsq(A, y_fit, rcond=None)

            offset = float(beta[0])
            g_signed = float(np.clip(beta[1], -self.OBS_CAL_MAX_GAIN_OSC, self.OBS_CAL_MAX_GAIN_OSC))
            if abs(g_signed) < 0.05:
                corr_helper = float(np.corrcoef(y_fit, y_helper_fit)[0, 1]) if y_fit.size > 3 else np.nan
                corr_full = float(np.corrcoef(y_fit, z_full_fit)[0, 1]) if y_fit.size > 3 else np.nan
                corr_ref = corr_full if np.isfinite(corr_full) else corr_helper
                g_signed = -1.0 if np.isfinite(corr_ref) and corr_ref < 0.0 else 1.0

            obs_sign = 1.0 if g_signed >= 0.0 else -1.0
            g_mag = abs(g_signed)
            g_osc_signed = g_signed
            g_b_signed = g_signed if self.ENABLE_BASELINE else 0.0
            g_r_signed = g_signed if self.ENABLE_RESIDUAL else 0.0
            g_osc = g_mag
            g_b = g_mag if self.ENABLE_BASELINE else 0.0
            g_r = g_mag if self.ENABLE_RESIDUAL else 0.0
            y_hat = offset + g_signed * z_full_fit
        elif mode == "osc_aux_two_gain":
            aux_fit = b_fit + r_fit
            A = np.column_stack([
                np.ones_like(y_fit),
                z_fit,
                aux_fit,
            ])
            reg = ridge * np.eye(A.shape[1], dtype=np.float64)
            reg[0, 0] = 0.0
            try:
                beta = np.linalg.solve(A.T @ A + reg, A.T @ y_fit)
            except np.linalg.LinAlgError:
                beta, *_ = np.linalg.lstsq(A, y_fit, rcond=None)

            offset = float(beta[0])
            g_osc_signed = float(np.clip(beta[1], -self.OBS_CAL_MAX_GAIN_OSC, self.OBS_CAL_MAX_GAIN_OSC))
            g_aux_signed = float(np.clip(beta[2], -self.OBS_CAL_MAX_GAIN_AUX, self.OBS_CAL_MAX_GAIN_AUX))

            if abs(g_osc_signed) < 0.05:
                corr_helper = float(np.corrcoef(y_fit, y_helper_fit)[0, 1]) if y_fit.size > 3 else np.nan
                corr_osc = float(np.corrcoef(y_fit, z_fit)[0, 1]) if y_fit.size > 3 else np.nan
                corr_ref = corr_osc if np.isfinite(corr_osc) else corr_helper
                g_osc_signed = -1.0 if np.isfinite(corr_ref) and corr_ref < 0.0 else 1.0

            obs_sign = 1.0 if g_osc_signed >= 0.0 else -1.0
            g_osc = abs(g_osc_signed)
            g_aux = g_aux_signed * obs_sign
            g_b_signed = g_aux_signed if self.ENABLE_BASELINE else 0.0
            g_r_signed = g_aux_signed if self.ENABLE_RESIDUAL else 0.0
            g_b = g_aux if self.ENABLE_BASELINE else 0.0
            g_r = g_aux if self.ENABLE_RESIDUAL else 0.0
            y_hat = offset + g_osc_signed * z_fit + g_aux_signed * aux_fit
        else:
            A = np.column_stack([
                np.ones_like(y_fit),
                z_fit,
                b_fit,
                r_fit,
            ])
            reg = ridge * np.eye(A.shape[1], dtype=np.float64)
            reg[0, 0] = 0.0
            try:
                beta = np.linalg.solve(A.T @ A + reg, A.T @ y_fit)
            except np.linalg.LinAlgError:
                beta, *_ = np.linalg.lstsq(A, y_fit, rcond=None)

            offset = float(beta[0])
            g_osc_signed = float(np.clip(beta[1], -self.OBS_CAL_MAX_GAIN_OSC, self.OBS_CAL_MAX_GAIN_OSC))
            g_b_signed = float(np.clip(beta[2], -self.OBS_CAL_MAX_GAIN_AUX, self.OBS_CAL_MAX_GAIN_AUX)) if self.ENABLE_BASELINE else 0.0
            g_r_signed = float(np.clip(beta[3], -self.OBS_CAL_MAX_GAIN_AUX, self.OBS_CAL_MAX_GAIN_AUX)) if self.ENABLE_RESIDUAL else 0.0

            if abs(g_osc_signed) < 0.05:
                corr_helper = float(np.corrcoef(y_fit, y_helper_fit)[0, 1]) if y_fit.size > 3 else np.nan
                if np.isfinite(corr_helper) and corr_helper < 0.0:
                    g_osc_signed = -1.0
                else:
                    g_osc_signed = 1.0
            obs_sign = 1.0 if g_osc_signed >= 0.0 else -1.0
            g_osc = abs(g_osc_signed)
            g_b = g_b_signed * obs_sign if self.ENABLE_BASELINE else 0.0
            g_r = g_r_signed * obs_sign if self.ENABLE_RESIDUAL else 0.0
            y_hat = offset + g_osc_signed * z_fit + g_b_signed * b_fit + g_r_signed * r_fit

        if mode != "family_phase_aux":
            fit_corr = float(np.corrcoef(y_fit, y_hat)[0, 1]) if y_fit.size > 3 else np.nan
            fit_rmse = float(np.sqrt(np.mean((y_fit - y_hat) ** 2))) if y_fit.size > 0 else np.nan
            lag_sec = 0.0

        return {
            "offset": offset,
            "obs_sign": obs_sign,
            "obs_domain": obs_domain if mode in {"family_phase_aux", "family_phase_split_aux"} else "displacement",
            "g_osc": g_osc,
            "g_b": g_b,
            "g_r": g_r,
            "g_h1": g_h1 if mode in {"family_phase_aux", "family_phase_split_aux"} else g_osc,
            "g_h2": g_h2 if mode in {"family_phase_aux", "family_phase_split_aux"} else (g_osc if self.ENABLE_HARMONIC2 else 0.0),
            "g_aux": g_aux if mode in {"family_phase_aux", "family_phase_split_aux"} else max(g_b, g_r),
            "g_osc_signed": g_osc_signed,
            "g_b_signed": g_b_signed,
            "g_r_signed": g_r_signed,
            "g_h1_signed": g_h1_signed if mode in {"family_phase_aux", "family_phase_split_aux"} else g_osc_signed,
            "g_h2_signed": g_h2_signed if mode in {"family_phase_aux", "family_phase_split_aux"} else (g_osc_signed if self.ENABLE_HARMONIC2 else 0.0),
            "g_aux_signed": g_aux_signed if mode in {"family_phase_aux", "family_phase_split_aux"} else max(g_b_signed, g_r_signed),
            "lag_sec": float(lag_sec),
            "fit_corr": fit_corr,
            "fit_rmse": fit_rmse,
            "n_frames": int(y_fit.size),
            "mode": mode,
            "enabled": True,
        }

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
        run_meta = dict(meta or {})
        assistant_policy = str(run_meta.pop("assistant_policy_runtime", "") or "").strip().lower()
        assistant_families = list(run_meta.pop("assistant_observation_families_runtime", []) or [])
        assistant_signal_map = run_meta.pop("assistant_signals_runtime", {}) or {}
        self._current_run_meta = dict(run_meta)
        base_method = self._base_method_from_meta(self._current_run_meta)
        signal_arr = np.asarray(signal, dtype=np.float64)
        multichannel = signal_arr.ndim == 2
        observation_families = list(self._current_run_meta.get("observation_families") or [])
        if multichannel:
            n_channels = int(signal_arr.shape[0])
            if len(observation_families) != n_channels:
                observation_families = [f"{base_method}_ch{i}" for i in range(n_channels)]
        else:
            observation_families = [base_method]
        freq_rescue_allowed = bool(
            self.ENABLE_FREQ_ADAPT
            and self.ENABLE_FREQ_RESCUE
            and self._freq_rescue_allowed(base_method)
        )
        freq_rescue_params = self._freq_rescue_params(base_method)
        if multichannel:
            y_channels = []
            raw_channels = []
            for idx, family in enumerate(observation_families):
                raw_i = np.asarray(signal_arr[idx], dtype=np.float64).reshape(-1)
                raw_channels.append(raw_i)
                y_i = self._preprocess(raw_i, fs, family_override=family)
                y_channels.append(np.asarray(y_i, dtype=np.float64).reshape(-1))
            n = min((yi.size for yi in y_channels), default=0)
            if n > 0:
                y = np.vstack([yi[:n] for yi in y_channels])
                raw_channels = [ri[:n] for ri in raw_channels]
            else:
                y = np.zeros((len(y_channels), 0), dtype=np.float64)
                raw_channels = [np.array([], dtype=np.float64) for _ in y_channels]
        else:
            y = self._preprocess(signal_arr, fs)
            n = y.size
            raw_channels = [np.asarray(signal_arr, dtype=np.float64).reshape(-1)]
        if n == 0:
            return self._package(y, np.array([], dtype=np.float64), meta)

        assistant_raw_channels = []
        assistant_family_list = []
        if (not multichannel) and assistant_families and isinstance(assistant_signal_map, dict):
            for family in assistant_families:
                raw_a = assistant_signal_map.get(family)
                if raw_a is None:
                    continue
                arr_a = np.asarray(raw_a, dtype=np.float64).reshape(-1)
                if arr_a.size == 0:
                    continue
                arr_a = arr_a[:n]
                assistant_raw_channels.append(arr_a)
                assistant_family_list.append(str(family))

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

        # ── Initial frequency (warm-up window only, helper path only) ──
        init_len = min(n, max(int(self.FREQ_INIT_SEC * fs), int(4.0 * fs)))
        if multichannel:
            helper_init_tracks = []
            for raw_i in raw_channels:
                y_init_helper_i = self._helper_preprocess(raw_i[:init_len], fs)
                if y_init_helper_i.size == 0:
                    continue
                fi_raw = self._coarse_freq(y_init_helper_i, fs)
                fi = self._harmonic_refine(fi_raw, y_init_helper_i, fs)
                if np.isfinite(fi):
                    helper_init_tracks.append(float(fi))
            if helper_init_tracks:
                freq0_raw = float(np.median(helper_init_tracks))
                freq0 = float(np.median(helper_init_tracks))
            else:
                y_init_helper = self._helper_preprocess(raw_channels[0][:init_len], fs)
                if y_init_helper.size == 0:
                    y_init_helper = y[0, :init_len]
                freq0_raw = self._coarse_freq(y_init_helper, fs)
                freq0 = self._harmonic_refine(freq0_raw, y_init_helper, fs)
        else:
            y_init_helper = self._helper_preprocess(signal_arr[:init_len], fs)
            if y_init_helper.size == 0:
                y_init_helper = y[:init_len]
            freq0_raw = self._coarse_freq(y_init_helper, fs)
            freq0 = self._harmonic_refine(freq0_raw, y_init_helper, fs)
        freq0 = float(np.clip(freq0, p.f_min, p.f_max))
        omega0 = 2.0 * np.pi * freq0
        if self.USE_HELPER_PATH:
            if multichannel:
                y_helper_channels = []
                helper_amp_channels = []
                helper_freq_channels = []
                for raw_i in raw_channels:
                    y_helper_i, helper_amp_i, helper_freq_i = self._helper_features(raw_i, fs, freq0)
                    if y_helper_i.size != n:
                        y_helper_i = np.asarray(raw_i[:n], dtype=np.float64)
                        helper_amp_i = np.abs(y_helper_i).astype(np.float64)
                        helper_freq_i = np.full(n, freq0, dtype=np.float64)
                    y_helper_channels.append(y_helper_i[:n])
                    helper_amp_channels.append(helper_amp_i[:n])
                    helper_freq_channels.append(helper_freq_i[:n])
                y_helper = np.vstack(y_helper_channels)
                helper_amp = np.vstack(helper_amp_channels)
                helper_freq = np.vstack(helper_freq_channels)
                helper_amp_consensus = np.nanmedian(helper_amp, axis=0)
                helper_freq_consensus = np.nanmedian(helper_freq, axis=0)
            else:
                y_helper, helper_amp, helper_freq = self._helper_features(signal_arr, fs, freq0)
                if y_helper.size != n:
                    y_helper = y.copy()
                    helper_amp = np.abs(y_helper).astype(np.float64)
                    helper_freq = np.full(n, freq0, dtype=np.float64)
                helper_amp_consensus = helper_amp.copy()
                helper_freq_consensus = helper_freq.copy()
        else:
            if multichannel:
                y_helper = y.copy()
                helper_amp = np.abs(y_helper).astype(np.float64)
                helper_freq = np.full((y.shape[0], n), freq0, dtype=np.float64)
                helper_amp_consensus = np.nanmedian(helper_amp, axis=0)
                helper_freq_consensus = np.full(n, freq0, dtype=np.float64)
            else:
                y_helper = y.copy()
                analytic = sps.hilbert(y_helper) if y_helper.size else np.array([], dtype=np.complex128)
                helper_amp = np.abs(analytic).astype(np.float64) if y_helper.size else np.array([], dtype=np.float64)
                helper_freq = np.full(n, freq0, dtype=np.float64)
                helper_amp_consensus = helper_amp.copy()
                helper_freq_consensus = helper_freq.copy()

        output_helper_freq_consensus = helper_freq_consensus.copy()
        assistant_helper_freq_consensus = None
        if (
            (not multichannel)
            and assistant_policy in {"of_rate_assistant_v1", "of_rate_assistant_v2"}
            and assistant_raw_channels
        ):
            assistant_freq_tracks = []
            for raw_a in assistant_raw_channels:
                _, _, helper_freq_a = self._helper_features(raw_a, fs, freq0)
                if helper_freq_a.size != n:
                    helper_freq_a = np.full(n, freq0, dtype=np.float64)
                assistant_freq_tracks.append(np.asarray(helper_freq_a[:n], dtype=np.float64))
            if assistant_freq_tracks:
                assistant_helper_freq_consensus = np.nanmedian(
                    np.vstack(assistant_freq_tracks),
                    axis=0,
                )
                bad_assistant = ~np.isfinite(assistant_helper_freq_consensus)
                if np.any(bad_assistant):
                    assistant_helper_freq_consensus[bad_assistant] = helper_freq_consensus[bad_assistant]
                if assistant_policy == "of_rate_assistant_v1":
                    output_helper_freq_consensus = np.nanmedian(
                        np.vstack([output_helper_freq_consensus, assistant_helper_freq_consensus]),
                        axis=0,
                    )
                    bad = ~np.isfinite(output_helper_freq_consensus)
                    if np.any(bad):
                        output_helper_freq_consensus[bad] = helper_freq_consensus[bad]

        # ── Effective parameters ──
        eff = self._effective_params(fs, run_meta)
        rho = eff['rho']
        R_init = eff['rv']

        if multichannel:
            obs_cal = [
                self._warmup_observation_calibration(
                    np.asarray(y[idx], dtype=np.float64),
                    np.asarray(y_helper[idx], dtype=np.float64),
                    fs, freq0, qx, dt, rho, R_init,
                    family_override=observation_families[idx],
                )
                for idx in range(y.shape[0])
            ]
        else:
            obs_cal = self._warmup_observation_calibration(y, y_helper, fs, freq0, qx, dt, rho, R_init)

        if multichannel:
            family_confidence = [
                self._family_confidence_policy(family, oc)
                for family, oc in zip(observation_families, obs_cal)
            ]
        else:
            family_confidence = self._family_confidence_policy(base_method, obs_cal)

        # ── Matrices ──
        if multichannel:
            H = np.vstack([self._build_H_from_obs_cal(oc, freq0) for oc in obs_cal])
        else:
            H = self._build_H_from_obs_cal(obs_cal, freq0)
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
        diag_obs_osc_support = np.ones(n, dtype=np.float64)
        diag_obs_full_support = np.ones(n, dtype=np.float64)
        diag_obs_nonosc_need = np.zeros(n, dtype=np.float64)
        diag_obs_nonosc_need_eff = np.zeros(n, dtype=np.float64)
        diag_residual_prior = np.ones(n, dtype=np.float64)
        diag_aper_drive = np.zeros(n, dtype=np.float64)
        diag_helper_support = np.ones(n, dtype=np.float64)
        diag_helper_trust = np.ones(n, dtype=np.float64)
        diag_helper_bias_conf = np.zeros(n, dtype=np.float64)
        diag_helper_freq = helper_freq_consensus.copy()
        diag_helper_mismatch = np.full(n, np.nan, dtype=np.float64)
        diag_freq_rescue = np.zeros(n, dtype=np.float64)
        diag_output_rate_blend = np.zeros(n, dtype=np.float64)
        diag_q_dyn_raw = np.zeros(n, dtype=np.float64)
        if multichannel:
            n_obs = y.shape[0]
            diag_R_channels = np.zeros((n_obs, n), dtype=np.float64)
            diag_q_obs_channels = np.ones((n_obs, n), dtype=np.float64)
        else:
            n_obs = 1
            diag_R_channels = None
            diag_q_obs_channels = None

        # ── Init state ──
        x = np.zeros(D, dtype=np.float64)
        Q0 = self._build_Q_disentangled(qx, dt, q_dyn=0.0, q_osc=1.0)
        P = Q0.copy()

        # ── Running statistics ──
        if multichannel:
            R_t = np.full(n_obs, R_init, dtype=np.float64)
            C_e = np.full(n_obs, R_init, dtype=np.float64)
            mad_e = np.full(n_obs, np.sqrt(R_init), dtype=np.float64)
        else:
            R_t = R_init       # EMA of squared innovations
            C_e = R_init       # EMA of squared innovations
            mad_e = np.sqrt(R_init)  # EW-MAD robust scale (in units of |e|)
        kurtosis_ema = 3.0
        nu_t = self.NU_MAX
        omega_t = omega0
        freq_t = freq0

        # Signal scale for q_obs jump detector
        if multichannel:
            signal_scale = np.array(
                [max(float(np.std(y[idx, :min(n, int(3.0 * fs))])), 1e-6) for idx in range(n_obs)],
                dtype=np.float64,
            )
            y_prev = y[:, 0].copy() if n > 0 else np.zeros(n_obs, dtype=np.float64)
        else:
            signal_scale = max(float(np.std(y[:min(n, int(3.0 * fs))])), 1e-6)
            y_prev = y[0] if n > 0 else 0.0

        # Phase/amplitude state for q_osc
        phase_prev = 0.0
        helper_amp_baseline = float(np.median(helper_amp_consensus[:max(1, min(init_len, helper_amp_consensus.size))])) if helper_amp_consensus.size else 1.0
        helper_amp_baseline = max(helper_amp_baseline, 0.25)
        gate_init = False

        # Frequency adaptation state
        freq_candidate = freq0
        freq_confirm = 0

        # ══════════════════════════════════════════
        #  FORWARD PASS
        # ══════════════════════════════════════════
        for t in range(n):
            if multichannel:
                H = np.vstack([self._build_H_from_obs_cal(oc, freq_t) for oc in obs_cal])
            else:
                H = self._build_H_from_obs_cal(obs_cal, freq_t)
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
            if multichannel:
                offset_vec = np.asarray([oc["offset"] for oc in obs_cal], dtype=np.float64)
                y_t = np.asarray(y[:, t], dtype=np.float64) - offset_vec
                y_pred = H @ x_pred
                e_t = y_t - y_pred
                HP = H @ P_pred @ H.T
                HP_diag = np.clip(np.diag(HP), 0.0, np.inf)
                C_e = ALPHA_R * C_e + (1.0 - ALPHA_R) * (e_t ** 2)
                C_e_safe = np.maximum(C_e, 1e-12)
                mad_e = ALPHA_MAD * mad_e + (1.0 - ALPHA_MAD) * np.abs(e_t)
                robust_scale = np.maximum(mad_e * 1.4826, 1e-12)
                nis_empirical = float(np.mean((e_t ** 2) / C_e_safe))
            else:
                y_t = y[t] - obs_cal["offset"]
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
                if multichannel:
                    R_mehra = C_e - HP_diag
                    R_floor = self.R_ANCHOR_FRAC * R_init
                    R_t = np.maximum(R_mehra, R_floor)
                else:
                    R_mehra = C_e - HP
                    R_floor = self.R_ANCHOR_FRAC * R_init
                    R_t = max(R_mehra, R_floor)
            else:
                if multichannel:
                    R_t = np.full(n_obs, R_init, dtype=np.float64)
                else:
                    R_t = R_init

            # ── Phase coherence & amplitude stability (for q_osc) ──
            hc1_pred = x_pred[self.HC1]
            hs1_pred = x_pred[self.HS1]
            phase_t = float(np.arctan2(hs1_pred, hc1_pred))
            helper_amp_t = float(helper_amp_consensus[t]) if t < helper_amp_consensus.size else helper_amp_baseline
            helper_freq_t = float(helper_freq_consensus[t]) if t < helper_freq_consensus.size else freq_t
            y_osc_pred = float(hc1_pred + (x_pred[self.HC2] if self.ENABLE_HARMONIC2 else 0.0))
            freq_lock = float(np.exp(
                -0.5 * ((helper_freq_t - freq_t) / max(self.Q_OSC_FREQ_REF_HZ, 1e-6)) ** 2
            ))

            if t >= GATE_WARMUP and gate_init:
                expected_phase = phase_prev + omega_t * dt
                phase_err = abs(_angle_wrap(phase_t - expected_phase))
                phase_coh = float(np.exp(-0.5 * (phase_err / self.GATE_PHASE_SIGMA) ** 2))
            else:
                phase_coh = 1.0
                if not gate_init and helper_amp_t > 1e-8:
                    helper_amp_baseline = max(helper_amp_t, 0.25)
                    gate_init = True

            phase_prev = phase_t
            if multichannel:
                helper_support_vec = []
                obs_osc_support_vec = []
                obs_full_support_vec = []
                for idx in range(n_obs):
                    y_helper_t = float(y_helper[idx, t]) if t < y_helper.shape[1] else float(y_t[idx])
                    H_osc = H[idx].copy()
                    if self.ENABLE_BASELINE:
                        H_osc[self.B] = 0.0
                    if self.ENABLE_RESIDUAL:
                        H_osc[self.R] = 0.0
                    y_osc_pred_i = float(H_osc @ x_pred)
                    helper_err = abs(y_helper_t - y_osc_pred_i)
                    helper_support_vec.append(float(np.exp(
                        -0.5 * (helper_err / max(self.Q_OSC_ERR_SIGMA, 1e-6)) ** 2
                    )))
                    obs_err_i = abs(float(y_t[idx]) - y_osc_pred_i)
                    obs_sigma_i = max(float(robust_scale[idx]), float(self.Q_OSC_ERR_SIGMA), 1e-6)
                    obs_osc_support_vec.append(float(np.exp(
                        -0.5 * (obs_err_i / obs_sigma_i) ** 2
                    )))
                    y_full_pred_i = float(H[idx] @ x_pred)
                    obs_err_full_i = abs(float(y_t[idx]) - y_full_pred_i)
                    obs_full_support_vec.append(float(np.exp(
                        -0.5 * (obs_err_full_i / obs_sigma_i) ** 2
                    )))
                helper_support = float(np.nanmean(helper_support_vec)) if helper_support_vec else 1.0
                obs_osc_support = float(np.nanmean(obs_osc_support_vec)) if obs_osc_support_vec else 1.0
                obs_full_support = float(np.nanmean(obs_full_support_vec)) if obs_full_support_vec else 1.0
            else:
                y_helper_t = float(y_helper[t]) if t < y_helper.size else y_t
                helper_err = abs(y_helper_t - y_osc_pred)
                helper_support = float(np.exp(
                    -0.5 * (helper_err / max(self.Q_OSC_ERR_SIGMA, 1e-6)) ** 2
                ))
                H_osc = np.asarray(H, dtype=np.float64).reshape(-1).copy()
                if self.ENABLE_BASELINE:
                    H_osc[self.B] = 0.0
                if self.ENABLE_RESIDUAL:
                    H_osc[self.R] = 0.0
                y_osc_pred_obs = float(H_osc @ x_pred)
                obs_err = abs(float(y_t) - y_osc_pred_obs)
                obs_sigma = max(float(robust_scale), float(self.Q_OSC_ERR_SIGMA), 1e-6)
                obs_osc_support = float(np.exp(
                    -0.5 * (obs_err / obs_sigma) ** 2
                ))
                y_full_pred = float(H @ x_pred)
                obs_err_full = abs(float(y_t) - y_full_pred)
                obs_full_support = float(np.exp(
                    -0.5 * (obs_err_full / obs_sigma) ** 2
                ))

            if t > 0:
                helper_start = max(0, t - int(round(float(self.HELPER_TRUST_WINDOW_SEC) * fs)) + 1)
                helper_freq_hist = np.asarray(diag_helper_freq[helper_start:t], dtype=np.float64)
                track_freq_hist = np.asarray(diag_freq[helper_start:t], dtype=np.float64)
                support_hist = np.asarray(diag_helper_support[helper_start:t], dtype=np.float64)
                helper_freq_win = np.concatenate([helper_freq_hist, np.array([helper_freq_t], dtype=np.float64)])
                track_freq_win = np.concatenate([track_freq_hist, np.array([freq_t], dtype=np.float64)])
                support_win = np.concatenate([support_hist, np.array([helper_support], dtype=np.float64)])
                helper_trust_t, helper_bias_conf_t = self._compute_helper_trust(
                    helper_freq_win,
                    support_win,
                    track_freq_win,
                    base_method,
                )
            else:
                helper_trust_t, helper_bias_conf_t = (1.0, 0.0)

            # ── Compute 3 quality scores ──
            if multichannel:
                q_obs_vec = np.array([
                    self._compute_q_obs(float(e_t[idx]), float(robust_scale[idx]), float(y_t[idx]), float(y_prev[idx]), float(signal_scale[idx]))
                    for idx in range(n_obs)
                ], dtype=np.float64)
                q_obs_t = float(np.mean(q_obs_vec))
                residual_prior_t = float(np.mean([
                    self._residual_prior_scale(family) for family in observation_families
                ])) if observation_families else 1.0
            else:
                q_obs_t = self._compute_q_obs(e_t, robust_scale, y_t, y_prev, signal_scale)
                residual_prior_t = self._residual_prior_scale(base_method)
            q_dyn_raw_t = (
                self._compute_q_dyn(helper_freq_t, freq_t, helper_amp_t, helper_amp_baseline)
                if t >= WARMUP else 0.0
            )
            q_dyn_t = float(q_dyn_raw_t)
            if multichannel:
                conf_qdyn_scale = float(np.mean([
                    float(fc.get("qdyn_scale", 1.0))
                    for fc in family_confidence
                    if bool(fc.get("enabled", False))
                ])) if family_confidence else 1.0
            else:
                conf_qdyn_scale = float(family_confidence.get("qdyn_scale", 1.0)) if bool(family_confidence.get("enabled", False)) else 1.0
            q_dyn_t = float(np.clip(q_dyn_t * conf_qdyn_scale, 0.0, 1.0))
            if self._helper_trust_scales_qdyn(base_method):
                qdyn_floor = float(np.clip(self.HELPER_TRUST_QDYN_FLOOR, 0.0, 1.0))
                qdyn_scale = qdyn_floor + (1.0 - qdyn_floor) * float(np.clip(helper_trust_t, 0.0, 1.0))
                q_dyn_t = float(np.clip(q_dyn_raw_t * qdyn_scale, 0.0, 1.0))
            if t >= GATE_WARMUP:
                q_osc_t, obs_nonosc_need = self._compute_q_osc(
                    phase_coh,
                    helper_support,
                    freq_lock,
                    obs_osc_support=obs_osc_support,
                    obs_full_support=obs_full_support,
                    q_obs=q_obs_t,
                )
            else:
                q_osc_t = 1.0
                obs_full_support = 1.0
                obs_nonosc_need = 0.0
            obs_nonosc_need_eff = float(np.clip(residual_prior_t * obs_nonosc_need, 0.0, 1.0))
            aper_drive_t = float(np.clip(residual_prior_t * max(1.0 - q_osc_t, 0.0), 0.0, 1.0))
            helper_amp_baseline = ALPHA_AMP * helper_amp_baseline + (1.0 - ALPHA_AMP) * helper_amp_t

            # ── Prior trust π_t = f(q_obs_t) ──
            # Conservative: π_t = q_obs_t. Low q_obs inflates R_eff,
            # reducing Kalman gain for unreliable observations.
            if multichannel:
                pi_t = q_obs_vec
                if family_confidence:
                    floors = np.array([
                        float(fc.get("pi_floor", 0.0)) if bool(fc.get("enabled", False)) else 0.0
                        for fc in family_confidence
                    ], dtype=np.float64)
                    if floors.size == pi_t.size:
                        pi_t = np.maximum(pi_t, floors)
            else:
                pi_t = q_obs_t
                if bool(family_confidence.get("enabled", False)):
                    pi_t = max(float(pi_t), float(family_confidence.get("pi_floor", 0.0)))

            # ── Adaptive Q ──
            if t >= QX_WARMUP:
                if self.ENABLE_DISENTANGLED_Q:
                    Q = self._build_Q_disentangled(
                        qx,
                        dt,
                        q_dyn_t,
                        q_osc_t,
                        obs_nonosc_need=obs_nonosc_need_eff,
                        residual_prior_scale=residual_prior_t,
                    )
                elif self.ENABLE_LEGACY_COUPLED_Q:
                    r_ratio = float(np.mean(R_t)) / max(R_init, 1e-12) if multichannel else (R_t / max(R_init, 1e-12))
                    Q = self._build_Q_legacy_coupled(qx, dt, r_ratio)
                # else: keep default Q (no adaptation)

                # Re-predict with updated Q
                P_pred = F @ P @ F.T + Q
                P_pred_arr[t] = P_pred

            # ── Effective R with prior trust ──
            if multichannel:
                R_eff = np.asarray(R_t, dtype=np.float64) / np.maximum(np.asarray(pi_t, dtype=np.float64), 1e-6)
                if family_confidence:
                    r_scales = np.array([
                        float(fc.get("r_scale", 1.0)) if bool(fc.get("enabled", False)) else 1.0
                        for fc in family_confidence
                    ], dtype=np.float64)
                    if r_scales.size == R_eff.size:
                        R_eff = R_eff * r_scales
                S_eff = H @ P_pred @ H.T + np.diag(R_eff)
                S_eff = 0.5 * (S_eff + S_eff.T)
            else:
                R_eff = R_t / max(pi_t, 1e-6)
                if bool(family_confidence.get("enabled", False)):
                    R_eff = float(R_eff) * float(family_confidence.get("r_scale", 1.0))
                S_eff = float(H @ P_pred @ H.T) + R_eff
                if S_eff <= 1e-12 or not np.isfinite(S_eff):
                    S_eff = 1e-12

            # ── Kurtosis → Student-t ν ──
            if self.ENABLE_STUDENT_T and t >= WARMUP:
                if multichannel:
                    e_std = np.asarray(e_t, dtype=np.float64) / np.maximum(np.sqrt(C_e_safe), 1e-12)
                    e_std_4 = float(np.mean(e_std ** 4))
                    kurtosis_ema = ALPHA_KAPPA * kurtosis_ema + (1.0 - ALPHA_KAPPA) * e_std_4
                else:
                    e_std = e_t / max(np.sqrt(C_e_safe), 1e-12)
                    kurtosis_ema = ALPHA_KAPPA * kurtosis_ema + (1.0 - ALPHA_KAPPA) * float(e_std ** 4)
                excess_k = max(kurtosis_ema - 3.0, 0.0)
                nu_t = max(self.NU_MIN, 6.0 / excess_k + 4.0) if excess_k > 0.01 else self.NU_MAX
                nu_t = float(np.clip(nu_t, self.NU_MIN, self.NU_MAX))

            # ── Update ──
            if self.ENABLE_STUDENT_T and nu_t < self.NU_MAX - 1.0:
                if multichannel:
                    x, P, lambda_t = self._student_t_vb_update_multichannel(
                        e_t, R_eff, nu_t, x_pred.copy(), P_pred.copy(), H, I_D
                    )
                else:
                    x, P, lambda_t = self._student_t_vb_update(
                        e_t, HP, R_eff, nu_t, x_pred.copy(), P_pred.copy(), H, I_D
                    )
            else:
                lambda_t = 1.0
                if multichannel:
                    try:
                        S_inv = np.linalg.pinv(S_eff)
                    except np.linalg.LinAlgError:
                        S_inv = np.linalg.pinv(S_eff + 1e-9 * np.eye(n_obs, dtype=np.float64))
                    K = P_pred @ H.T @ S_inv
                    x = x_pred + K @ e_t
                    P = (I_D - K @ H) @ P_pred
                else:
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
                if multichannel:
                    y_win = np.asarray(helper_freq_consensus[win_start:t + 1], dtype=np.float64)
                else:
                    y_win = y_helper[win_start:t + 1]
                if len(y_win) >= int(2.0 * fs):
                    helper_start = max(0, t - int(self.FREQ_RESCUE_WINDOW_SEC * fs))
                    if multichannel:
                        helper_win = np.asarray(helper_freq_consensus[helper_start:t + 1], dtype=np.float64)
                        f_spec = float(np.clip(np.nanmedian(helper_win), p.f_min, p.f_max))
                    else:
                        f_spec = self._coarse_freq(y_win, fs)
                        f_spec = float(np.clip(f_spec, p.f_min, p.f_max))
                        helper_win = np.asarray(helper_freq[helper_start:t + 1], dtype=np.float64)
                    support_hist = np.asarray(diag_helper_support[helper_start:t], dtype=np.float64)
                    qdyn_hist = np.asarray(diag_q_dyn[helper_start:t], dtype=np.float64)
                    trust_hist = np.asarray(diag_helper_trust[helper_start:t], dtype=np.float64)
                    bias_hist = np.asarray(diag_helper_bias_conf[helper_start:t], dtype=np.float64)
                    support_win = np.concatenate([support_hist, np.array([helper_support], dtype=np.float64)])
                    qdyn_win = np.concatenate([qdyn_hist, np.array([q_dyn_t], dtype=np.float64)])
                    trust_win = np.concatenate([trust_hist, np.array([helper_trust_t], dtype=np.float64)])
                    bias_win = np.concatenate([bias_hist, np.array([helper_bias_conf_t], dtype=np.float64)])

                    helper_med = freq_t
                    helper_std = np.inf
                    valid_helper = helper_win[np.isfinite(helper_win)]
                    if valid_helper.size >= max(3, int(0.5 * fs)):
                        helper_med = float(np.clip(np.median(valid_helper), p.f_min, p.f_max))
                    helper_std = float(np.std(valid_helper))
                    support_med = float(np.nanmedian(support_win)) if support_win.size else 0.0
                    qdyn_med = float(np.nanmedian(qdyn_win)) if qdyn_win.size else 0.0
                    trust_med = float(np.nanmedian(trust_win)) if trust_win.size else 1.0
                    bias_conf_med = float(np.nanmedian(bias_win)) if bias_win.size else 0.0
                    helper_mismatch = abs(helper_med - freq_t)

                    rescue_mode = bool(
                        freq_rescue_allowed
                        and np.isfinite(helper_std)
                        and helper_std <= float(freq_rescue_params["helper_std_max_hz"])
                        and support_med >= float(freq_rescue_params["min_support"])
                        and qdyn_med >= float(freq_rescue_params["min_qdyn"])
                        and helper_mismatch >= float(freq_rescue_params["min_mismatch_hz"])
                    )
                    if rescue_mode and self._helper_trust_allowed(base_method):
                        rescue_mode = bool(
                            trust_med >= float(self.HELPER_TRUST_RESCUE_MIN)
                            and bias_conf_med >= float(self.HELPER_TRUST_RESCUE_MIN)
                        )
                    if rescue_mode:
                        diag_freq_rescue[t] = 1.0
                        if abs(f_spec - helper_med) <= 0.08:
                            f_candidate = 0.5 * (f_spec + helper_med)
                        else:
                            f_candidate = helper_med
                        confirm_tol = 0.04
                        confirm_target = int(freq_rescue_params["confirm_count"])
                        step_cap = float(freq_rescue_params["max_step_hz"])
                    else:
                        f_candidate = f_spec
                        confirm_tol = 0.02
                        confirm_target = self.FREQ_CONFIRM_COUNT
                        step_cap = self.FREQ_MAX_STEP_HZ

                    if abs(f_candidate - freq_candidate) < confirm_tol:
                        freq_confirm += 1
                    else:
                        freq_candidate = f_candidate
                        freq_confirm = 1
                    if freq_confirm >= confirm_target:
                        step = np.clip(freq_candidate - freq_t, -step_cap, step_cap)
                        freq_t = float(np.clip(freq_t + step, p.f_min, p.f_max))
                        omega_t = 2.0 * np.pi * freq_t
                        freq_confirm = 0

            # ── Store diagnostics ──
            diag_R[t] = float(np.mean(R_t)) if multichannel else R_t
            diag_nis[t] = nis_empirical
            diag_pi[t] = float(np.mean(pi_t)) if multichannel else pi_t
            diag_lambda[t] = lambda_t
            diag_nu[t] = nu_t
            diag_freq[t] = freq_t
            diag_q_obs[t] = q_obs_t
            diag_q_dyn_raw[t] = q_dyn_raw_t
            diag_q_dyn[t] = q_dyn_t
            diag_q_osc[t] = q_osc_t
            diag_obs_osc_support[t] = obs_osc_support
            diag_obs_full_support[t] = obs_full_support
            diag_obs_nonosc_need[t] = obs_nonosc_need
            diag_obs_nonosc_need_eff[t] = obs_nonosc_need_eff
            diag_residual_prior[t] = residual_prior_t
            diag_aper_drive[t] = aper_drive_t
            diag_helper_support[t] = helper_support
            diag_helper_trust[t] = helper_trust_t
            diag_helper_bias_conf[t] = helper_bias_conf_t
            diag_helper_freq[t] = helper_freq_t
            if np.isfinite(helper_freq_t):
                diag_helper_mismatch[t] = abs(helper_freq_t - freq_t)
            if multichannel:
                diag_R_channels[:, t] = np.asarray(R_t, dtype=np.float64)
                diag_q_obs_channels[:, t] = np.asarray(q_obs_vec, dtype=np.float64)
                y_prev = y_t
            else:
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
        output_rate_base_method = base_method
        if multichannel and any(str(f).lower().startswith("of_") for f in observation_families):
            output_rate_base_method = "of_farneback"
        if (
            (not multichannel)
            and any(str(f).lower().startswith("of_") for f in assistant_family_list)
            and assistant_policy == "of_rate_assistant_v1"
        ):
            output_rate_base_method = "of_farneback"
        track_hz_smoothed, diag_output_rate_blend = self._output_rate_postprocess(
            track_hz_smoothed,
            output_helper_freq_consensus,
            diag_helper_support,
            diag_q_dyn,
            output_rate_base_method,
            alpha_used,
            fs,
        )
        track_hz_smoothed, diag_assistant_rate_blend = self._assistant_rate_postprocess(
            track_hz_smoothed,
            helper_freq_consensus,
            assistant_helper_freq_consensus,
            diag_helper_support,
            diag_q_dyn,
            alpha_used,
            assistant_policy,
        )
        diag_output_rate_blend = np.maximum(diag_output_rate_blend, diag_assistant_rate_blend)

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
        if freq_rescue_allowed:
            active_modules.append("freq_rescue")
        if self._helper_trust_allowed(base_method):
            active_modules.append("helper_trust")
        if self.ENABLE_HARMONIC2: active_modules.append("harmonic2")
        if self.ENABLE_BASELINE: active_modules.append("baseline")
        if self.ENABLE_RESIDUAL: active_modules.append("residual")
        if self.ENABLE_RESIDUAL and self.ENABLE_RESIDUAL_SEMANTICS:
            active_modules.append("residual_semantics")
        if self.USE_HELPER_PATH: active_modules.append("helper_path")
        if self.USE_LIGHT_OBS_PATH: active_modules.append("light_obs_path")
        if (any(bool(oc["enabled"]) for oc in obs_cal) if multichannel else obs_cal["enabled"]):
            active_modules.append("obs_cal")
        if multichannel:
            active_modules.append("multichannel_observation")

        meta_payload = dict(run_meta)
        meta_payload["f0"] = freq0
        meta_payload["f0_raw"] = float(freq0_raw)
        meta_payload["freq_source"] = "parh_ossm_phase"
        meta_payload["post_smooth_alpha_used"] = alpha_used
        meta_payload["model_version"] = MODEL_VERSION
        meta_payload["output_semantics"] = "smoothed"
        meta_payload["warmup_frames"] = WARMUP
        meta_payload["active_modules"] = active_modules
        meta_payload["helper_path_enabled"] = bool(self.USE_HELPER_PATH)
        meta_payload["freq_rescue_enabled"] = bool(freq_rescue_allowed)
        meta_payload["freq_rescue_family_allowed"] = bool(self._freq_rescue_allowed(base_method))
        meta_payload["freq_rescue_policy"] = str(self.FREQ_RESCUE_POLICY)
        meta_payload["helper_trust_policy"] = str(self.HELPER_TRUST_POLICY)
        meta_payload["output_rate_policy"] = str(self.OUTPUT_RATE_POLICY)
        meta_payload["observation_family_policy"] = str(self.OBS_FAMILY_POLICY)
        meta_payload["q_osc_obs_mode"] = str(self.Q_OSC_OBS_MODE)
        meta_payload["q_osc_obs_weight"] = float(self.Q_OSC_OBS_WEIGHT)
        meta_payload["q_osc_obs_ref"] = float(self.Q_OSC_OBS_REF)
        meta_payload["q_osc_obs_band"] = float(self.Q_OSC_OBS_BAND)
        meta_payload["q_aper_obs_gamma"] = float(self.Q_APER_OBS_GAMMA)
        meta_payload["enable_residual_semantics"] = bool(self.ENABLE_RESIDUAL_SEMANTICS)
        meta_payload["residual_prior_min"] = float(self.RESIDUAL_PRIOR_MIN)
        meta_payload["residual_prior_power"] = float(self.RESIDUAL_PRIOR_POWER)
        meta_payload["primary_observation_semantics"] = self._observation_family_semantics(base_method)
        if assistant_family_list:
            meta_payload["assistant_channel_policy"] = assistant_policy or "unknown"
            meta_payload["assistant_observation_families"] = list(assistant_family_list)
            meta_payload["assistant_observation_semantics"] = [
                self._observation_family_semantics(family) for family in assistant_family_list
            ]
            meta_payload["assistant_helper_freq_mean"] = float(np.mean(output_helper_freq_consensus)) if output_helper_freq_consensus.size else float("nan")
            meta_payload["assistant_helper_freq_std"] = float(np.std(output_helper_freq_consensus)) if output_helper_freq_consensus.size else float("nan")
            if assistant_helper_freq_consensus is not None and np.size(assistant_helper_freq_consensus):
                meta_payload["assistant_raw_helper_freq_mean"] = float(np.mean(assistant_helper_freq_consensus))
                meta_payload["assistant_raw_helper_freq_std"] = float(np.std(assistant_helper_freq_consensus))
        if multichannel:
            obs_enabled = any(bool(oc["enabled"]) for oc in obs_cal)
            modes = sorted({str(oc.get("mode", self.OBS_CAL_MODE)) for oc in obs_cal})
            fallback_modes = sorted({str(oc.get("fallback_mode", "fixed_sum")) for oc in obs_cal if str(oc.get("fallback_mode", "")).strip()})
            meta_payload["observation_model_type"] = (
                f"stacked_multichannel_{'+'.join(modes)}_family_observation" if obs_enabled
                else (
                    f"stacked_multichannel_fixed_{'+'.join(fallback_modes)}_family_observation"
                    if fallback_modes else "stacked_multichannel_fixed_family_observation"
                )
            )
            meta_payload["observation_families"] = list(observation_families)
            meta_payload["observation_family_semantics"] = [
                self._observation_family_semantics(family) for family in observation_families
            ]
            meta_payload["n_observation_channels"] = int(n_obs)
            meta_payload["observation_calibration_channels"] = []
            for family, oc in zip(observation_families, obs_cal):
                meta_payload["observation_calibration_channels"].append({
                    "family": str(family),
                    "enabled": bool(oc["enabled"]),
                    "mode": str(oc.get("mode", self.OBS_CAL_MODE)),
                    "obs_domain": str(oc.get("obs_domain", "displacement")),
                    "offset": float(oc["offset"]),
                    "obs_sign": float(oc["obs_sign"]),
                    "g_osc": float(oc["g_osc"]),
                    "g_b": float(oc["g_b"]),
                    "g_r": float(oc["g_r"]),
                    "g_h1": float(oc.get("g_h1", oc["g_osc"])),
                    "g_h2": float(oc.get("g_h2", oc["g_osc"] if self.ENABLE_HARMONIC2 else 0.0)),
                    "g_aux": float(oc.get("g_aux", max(oc["g_b"], oc["g_r"]))),
                    "g_osc_signed": float(oc["g_osc_signed"]),
                    "g_b_signed": float(oc["g_b_signed"]),
                    "g_r_signed": float(oc["g_r_signed"]),
                    "g_h1_signed": float(oc.get("g_h1_signed", oc["g_osc_signed"])),
                    "g_h2_signed": float(oc.get("g_h2_signed", oc["g_osc_signed"] if self.ENABLE_HARMONIC2 else 0.0)),
                    "g_aux_signed": float(oc.get("g_aux_signed", max(oc["g_b_signed"], oc["g_r_signed"]))),
                    "lag_sec": float(oc.get("lag_sec", 0.0)),
                    "fit_corr": float(oc["fit_corr"]) if np.isfinite(oc["fit_corr"]) else float("nan"),
                    "fit_rmse": float(oc["fit_rmse"]) if np.isfinite(oc["fit_rmse"]) else float("nan"),
                    "n_frames": int(oc["n_frames"]),
                    "fallback_mode": str(oc.get("fallback_mode", "fixed_sum")),
                })
            meta_payload["observation_calibration"] = {
                "enabled": bool(obs_enabled),
                "mode": "+".join(modes),
                "obs_domain": "+".join(sorted({str(oc.get("obs_domain", "displacement")) for oc in obs_cal})),
                "offset": float(np.nanmean([oc["offset"] for oc in obs_cal])),
                "obs_sign": float(np.nanmean([oc["obs_sign"] for oc in obs_cal])),
                "g_osc": float(np.nanmean([oc["g_osc"] for oc in obs_cal])),
                "g_b": float(np.nanmean([oc["g_b"] for oc in obs_cal])),
                "g_r": float(np.nanmean([oc["g_r"] for oc in obs_cal])),
                "g_h1": float(np.nanmean([oc.get("g_h1", oc["g_osc"]) for oc in obs_cal])),
                "g_h2": float(np.nanmean([oc.get("g_h2", oc["g_osc"] if self.ENABLE_HARMONIC2 else 0.0) for oc in obs_cal])),
                "g_aux": float(np.nanmean([oc.get("g_aux", max(oc["g_b"], oc["g_r"])) for oc in obs_cal])),
                "g_osc_signed": float(np.nanmean([oc["g_osc_signed"] for oc in obs_cal])),
                "g_b_signed": float(np.nanmean([oc["g_b_signed"] for oc in obs_cal])),
                "g_r_signed": float(np.nanmean([oc["g_r_signed"] for oc in obs_cal])),
                "g_h1_signed": float(np.nanmean([oc.get("g_h1_signed", oc["g_osc_signed"]) for oc in obs_cal])),
                "g_h2_signed": float(np.nanmean([oc.get("g_h2_signed", oc["g_osc_signed"] if self.ENABLE_HARMONIC2 else 0.0) for oc in obs_cal])),
                "g_aux_signed": float(np.nanmean([oc.get("g_aux_signed", max(oc["g_b_signed"], oc["g_r_signed"])) for oc in obs_cal])),
                "lag_sec": float(np.nanmean([oc.get("lag_sec", 0.0) for oc in obs_cal])),
                "fit_corr": float(np.nanmean([oc["fit_corr"] for oc in obs_cal if np.isfinite(oc["fit_corr"])])) if any(np.isfinite(oc["fit_corr"]) for oc in obs_cal) else float("nan"),
                "fit_rmse": float(np.nanmean([oc["fit_rmse"] for oc in obs_cal if np.isfinite(oc["fit_rmse"])])) if any(np.isfinite(oc["fit_rmse"]) for oc in obs_cal) else float("nan"),
                "n_frames": int(np.nanmean([oc["n_frames"] for oc in obs_cal])),
                "fallback_mode": "+".join(fallback_modes) if fallback_modes else "fixed_sum",
            }
        else:
            fallback_mode = str(obs_cal.get("fallback_mode", "fixed_sum"))
            if obs_cal["enabled"]:
                meta_payload["observation_model_type"] = (
                    f"warmup_calibrated_{str(obs_cal.get('mode', self.OBS_CAL_MODE))}_family_aware_observation"
                )
            else:
                if fallback_mode == "of_velocity_prior_v1":
                    meta_payload["observation_model_type"] = "fixed_of_velocity_prior_v1"
                else:
                    meta_payload["observation_model_type"] = (
                        f"fixed_sum_{str(self.OBS_FAMILY_POLICY)}" if self.USE_LIGHT_OBS_PATH
                        else "fixed_sum_legacy_preprocess"
                    )
            meta_payload["observation_calibration"] = {
                "enabled": bool(obs_cal["enabled"]),
                "mode": str(obs_cal.get("mode", self.OBS_CAL_MODE)),
                "obs_domain": str(obs_cal.get("obs_domain", "displacement")),
                "offset": float(obs_cal["offset"]),
                "obs_sign": float(obs_cal["obs_sign"]),
                "g_osc": float(obs_cal["g_osc"]),
                "g_b": float(obs_cal["g_b"]),
                "g_r": float(obs_cal["g_r"]),
                "g_h1": float(obs_cal.get("g_h1", obs_cal["g_osc"])),
                "g_h2": float(obs_cal.get("g_h2", obs_cal["g_osc"] if self.ENABLE_HARMONIC2 else 0.0)),
                "g_aux": float(obs_cal.get("g_aux", max(obs_cal["g_b"], obs_cal["g_r"]))),
                "g_osc_signed": float(obs_cal["g_osc_signed"]),
                "g_b_signed": float(obs_cal["g_b_signed"]),
                "g_r_signed": float(obs_cal["g_r_signed"]),
                "g_h1_signed": float(obs_cal.get("g_h1_signed", obs_cal["g_osc_signed"])),
                "g_h2_signed": float(obs_cal.get("g_h2_signed", obs_cal["g_osc_signed"] if self.ENABLE_HARMONIC2 else 0.0)),
                "g_aux_signed": float(obs_cal.get("g_aux_signed", max(obs_cal["g_b_signed"], obs_cal["g_r_signed"]))),
                "lag_sec": float(obs_cal.get("lag_sec", 0.0)),
                "fit_corr": float(obs_cal["fit_corr"]) if np.isfinite(obs_cal["fit_corr"]) else float("nan"),
                "fit_rmse": float(obs_cal["fit_rmse"]) if np.isfinite(obs_cal["fit_rmse"]) else float("nan"),
                "n_frames": int(obs_cal["n_frames"]),
                "fallback_mode": fallback_mode,
            }

        if multichannel:
            meta_payload["family_confidence_policy"] = {
                "enabled": bool(any(bool(fc.get("enabled", False)) for fc in family_confidence)),
                "channels": [
                    {
                        "family": str(family),
                        "enabled": bool(fc.get("enabled", False)),
                        "pi_floor": float(fc.get("pi_floor", 0.0)),
                        "qdyn_scale": float(fc.get("qdyn_scale", 1.0)),
                        "r_scale": float(fc.get("r_scale", 1.0)),
                    }
                    for family, fc in zip(observation_families, family_confidence)
                ],
            }
        else:
            meta_payload["family_confidence_policy"] = {
                "enabled": bool(family_confidence.get("enabled", False)),
                "family": str(base_method),
                "pi_floor": float(family_confidence.get("pi_floor", 0.0)),
                "qdyn_scale": float(family_confidence.get("qdyn_scale", 1.0)),
                "r_scale": float(family_confidence.get("r_scale", 1.0)),
            }

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
            "q_dyn_raw_mean": float(np.mean(diag_q_dyn_raw)),
            "q_dyn_mean": float(np.mean(diag_q_dyn)),
            "q_osc_mean": float(np.mean(diag_q_osc)),
            "obs_osc_support_mean": float(np.mean(diag_obs_osc_support)),
            "obs_full_support_mean": float(np.mean(diag_obs_full_support)),
            "obs_nonosc_need_mean": float(np.mean(diag_obs_nonosc_need)),
            "obs_nonosc_need_eff_mean": float(np.mean(diag_obs_nonosc_need_eff)),
            "residual_prior_mean": float(np.mean(diag_residual_prior)),
            "aper_drive_mean": float(np.mean(diag_aper_drive)),
            "helper_support_mean": float(np.mean(diag_helper_support)),
            "helper_trust_mean": float(np.mean(diag_helper_trust)),
            "helper_bias_conf_mean": float(np.mean(diag_helper_bias_conf)),
            "helper_mismatch_mean": float(np.nanmean(diag_helper_mismatch)),
            "helper_freq_mean": float(np.mean(diag_helper_freq)),
            "helper_freq_std": float(np.std(diag_helper_freq)),
            "freq_rescue_active_frac": float(np.mean(diag_freq_rescue)),
            "output_rate_blend_active_frac": float(np.mean(diag_output_rate_blend)),
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
            "q_dyn_raw_t": diag_q_dyn_raw,
            "q_dyn_t": diag_q_dyn,
            "q_osc_t": diag_q_osc,
            "obs_osc_support_t": diag_obs_osc_support,
            "obs_full_support_t": diag_obs_full_support,
            "obs_nonosc_need_t": diag_obs_nonosc_need,
            "obs_nonosc_need_eff_t": diag_obs_nonosc_need_eff,
            "residual_prior_t": diag_residual_prior,
            "aper_drive_t": diag_aper_drive,
            "freq_t": diag_freq,
            "helper_support_t": diag_helper_support,
            "helper_trust_t": diag_helper_trust,
            "helper_bias_conf_t": diag_helper_bias_conf,
            "helper_freq_t": diag_helper_freq,
            "helper_mismatch_t": diag_helper_mismatch,
            "freq_rescue_t": diag_freq_rescue,
            "output_rate_blend_t": diag_output_rate_blend,
            "nis_empirical_t": diag_nis,
        }
        if multichannel:
            result["diagnostics"]["R_t_channels"] = diag_R_channels
            result["diagnostics"]["q_obs_t_channels"] = diag_q_obs_channels

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
                        closer_to_dom = abs(freq0 - f_dominant) < abs(freq0 - f_sub)
                        if closer_to_dom and acf[lag_sub] > acf[lag_dom] * self.HARMONIC_ACF_RATIO:
                            return f_sub
        except Exception:
            pass
        return freq0
