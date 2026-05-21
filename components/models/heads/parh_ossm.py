"""
PARH-OSSM — Physiology-Aligned Regime-Adaptive Harmonic Oscillatory SSM.

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

Quality score boundary:
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
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import signal as sps
from components.observations.semantics import get_observation_family_semantics
from ..core.base import _BaseOscillatorHead

MODEL_VERSION = "parh_ossm"


def _angle_wrap(x: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class oscillator_PARH_OSSM(_BaseOscillatorHead):
    """PARH-OSSM: Physiology-Aligned Regime-Adaptive Harmonic SSM."""

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
    OBS_CAL_ALLOWED_FAMILIES: str = "profile1d_quadratic,profile1d_cubic,profile1d_consensus"
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
    P1D_FIXED_FAMILY_PRIOR: bool = False

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
    FAMILY_CONFIDENCE_ALLOWED_FAMILIES: str = "profile1d_quadratic,profile1d_cubic,profile1d_quadratic_bridge,profile1d_cubic_bridge,profile1d_consensus"
    FAMILY_CONFIDENCE_MIN_FIT_CORR: float = 0.975
    FAMILY_CONFIDENCE_MAX_FIT_RMSE: float = 0.20
    FAMILY_CONFIDENCE_PI_FLOOR: float = 0.97
    FAMILY_CONFIDENCE_QDYN_SCALE: float = 0.55
    FAMILY_CONFIDENCE_R_SCALE: float = 0.85

    # ── Dynamic soft multi-observation law (experimental) ──
    ENABLE_DYNAMIC_MIXTURE: bool = False
    DYNAMIC_MIXTURE_TAU_SEC: float = 2.0
    DYNAMIC_MIXTURE_TEMPERATURE: float = 0.45
    DYNAMIC_MIXTURE_MIN_WEIGHT: float = 0.02
    DYNAMIC_MIXTURE_CONTEXT_FLOOR: float = 0.20
    DYNAMIC_MIXTURE_R_WEIGHT: float = 0.50
    DYNAMIC_MIXTURE_GLOBAL_QUALITY_FLOOR: float = 0.05
    ENABLE_RATE_OBSERVABILITY_MIXTURE: bool = True
    ENABLE_RATE_OBSERVABILITY_HELPER: bool = True
    RATE_OBS_WINDOW_SEC: float = 8.0
    RATE_OBS_STD_REF_HZ: float = 0.06
    RATE_OBS_AGREE_REF_HZ: float = 0.075
    RATE_OBS_HARMONIC_PENALTY: float = 0.58
    RATE_OBS_FLOOR: float = 0.18
    RATE_OBS_POWER: float = 1.20
    RATE_OBS_HELPER_BLEND: float = 0.65
    RATE_OBS_HELPER_MIN_SUPPORT: float = 0.28
    RATE_OBS_HELPER_MAX_STEP_HZ: float = 0.08

    # ── State-component reliability law ──
    # These are semantic exponents, not dataset sweep knobs.  Role priors should
    # regularize state updates, not become a brittle target-dataset selector.
    # Default floors keep weak target-side priors from completely suppressing
    # h1/z_osc updates under domain shift; stronger behavior can still be
    # enabled explicitly in diagnostic tests.
    STATE_ROLE_CONTEXT_POWER: float = 0.50
    STATE_ROLE_RATE_POWER: float = 0.50
    STATE_ROLE_CONTEXT_MULTIPLIER_FLOOR: float = 0.85
    STATE_ROLE_RATE_MULTIPLIER_FLOOR: float = 0.80
    STATE_ROLE_ABSTAIN_R_SCALE: float = 0.35

    # ── Target-side observability control ──
    # This promotes GT-free target observability from an audit artifact into the
    # update law.  It never chooses a final rate source directly; it only
    # controls observation trust before the Kalman/Student-t update.
    ENABLE_TARGET_OBSERVABILITY_CONTROL: bool = False
    TARGET_OBS_TRUST_FLOOR: float = 0.65
    TARGET_OBS_QDYN_FLOOR: float = 0.75
    TARGET_OBS_NUISANCE_R_SCALE: float = 0.55
    TARGET_OBS_ROLE_POWER: float = 0.50

    # ── Family-aware residual semantics (experimental) ──
    ENABLE_RESIDUAL_SEMANTICS: bool = False
    RESIDUAL_PRIOR_MIN: float = 0.10
    RESIDUAL_PRIOR_POWER: float = 1.0
    ENABLE_RESIDUAL_IDENTIFIABILITY_GUARD: bool = False
    RESIDUAL_GUARD_TRUST_FLOOR: float = 0.20
    RESIDUAL_GUARD_NUISANCE_SCALE: float = 0.80

    # ── Phase-anchored waveform morphology ──
    # This is a conservative readout regularizer, not an extra learned model.
    # It only blends a phase-binned morphology template into z_full when the
    # target-side morphology/reliability evidence is present.
    ENABLE_PHASE_ANCHORED_MORPHOLOGY: bool = False
    PHASE_MORPH_MAX_BLEND: float = 0.35
    PHASE_MORPH_BINS: float = 24.0

    # ── Group-balanced multi-observation fusion ──
    # Correlated variants inside the same semantic group must not count as
    # independent sensors in the Kalman update.
    ENABLE_GROUP_BALANCED_FUSION: bool = False

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
            "P1D_FIXED_FAMILY_PRIOR": self.P1D_FIXED_FAMILY_PRIOR,
            "ENABLE_FAMILY_CONFIDENCE": self.ENABLE_FAMILY_CONFIDENCE,
            "ENABLE_DYNAMIC_MIXTURE": self.ENABLE_DYNAMIC_MIXTURE,
            "ENABLE_RATE_OBSERVABILITY_MIXTURE": self.ENABLE_RATE_OBSERVABILITY_MIXTURE,
            "ENABLE_RATE_OBSERVABILITY_HELPER": self.ENABLE_RATE_OBSERVABILITY_HELPER,
            "ENABLE_TARGET_OBSERVABILITY_CONTROL": self.ENABLE_TARGET_OBSERVABILITY_CONTROL,
            "ENABLE_RESIDUAL_SEMANTICS": self.ENABLE_RESIDUAL_SEMANTICS,
            "ENABLE_RESIDUAL_IDENTIFIABILITY_GUARD": self.ENABLE_RESIDUAL_IDENTIFIABILITY_GUARD,
            "ENABLE_PHASE_ANCHORED_MORPHOLOGY": self.ENABLE_PHASE_ANCHORED_MORPHOLOGY,
            "ENABLE_GROUP_BALANCED_FUSION": self.ENABLE_GROUP_BALANCED_FUSION,
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
            "DYNAMIC_MIXTURE_TAU_SEC": self.DYNAMIC_MIXTURE_TAU_SEC,
            "DYNAMIC_MIXTURE_TEMPERATURE": self.DYNAMIC_MIXTURE_TEMPERATURE,
            "DYNAMIC_MIXTURE_MIN_WEIGHT": self.DYNAMIC_MIXTURE_MIN_WEIGHT,
            "DYNAMIC_MIXTURE_CONTEXT_FLOOR": self.DYNAMIC_MIXTURE_CONTEXT_FLOOR,
            "DYNAMIC_MIXTURE_R_WEIGHT": self.DYNAMIC_MIXTURE_R_WEIGHT,
            "DYNAMIC_MIXTURE_GLOBAL_QUALITY_FLOOR": self.DYNAMIC_MIXTURE_GLOBAL_QUALITY_FLOOR,
            "RATE_OBS_WINDOW_SEC": self.RATE_OBS_WINDOW_SEC,
            "RATE_OBS_STD_REF_HZ": self.RATE_OBS_STD_REF_HZ,
            "RATE_OBS_AGREE_REF_HZ": self.RATE_OBS_AGREE_REF_HZ,
            "RATE_OBS_HARMONIC_PENALTY": self.RATE_OBS_HARMONIC_PENALTY,
            "RATE_OBS_FLOOR": self.RATE_OBS_FLOOR,
            "RATE_OBS_POWER": self.RATE_OBS_POWER,
            "RATE_OBS_HELPER_BLEND": self.RATE_OBS_HELPER_BLEND,
            "RATE_OBS_HELPER_MIN_SUPPORT": self.RATE_OBS_HELPER_MIN_SUPPORT,
            "RATE_OBS_HELPER_MAX_STEP_HZ": self.RATE_OBS_HELPER_MAX_STEP_HZ,
            "STATE_ROLE_CONTEXT_POWER": self.STATE_ROLE_CONTEXT_POWER,
            "STATE_ROLE_RATE_POWER": self.STATE_ROLE_RATE_POWER,
            "STATE_ROLE_CONTEXT_MULTIPLIER_FLOOR": self.STATE_ROLE_CONTEXT_MULTIPLIER_FLOOR,
            "STATE_ROLE_RATE_MULTIPLIER_FLOOR": self.STATE_ROLE_RATE_MULTIPLIER_FLOOR,
            "STATE_ROLE_ABSTAIN_R_SCALE": self.STATE_ROLE_ABSTAIN_R_SCALE,
            "TARGET_OBS_TRUST_FLOOR": self.TARGET_OBS_TRUST_FLOOR,
            "TARGET_OBS_QDYN_FLOOR": self.TARGET_OBS_QDYN_FLOOR,
            "TARGET_OBS_NUISANCE_R_SCALE": self.TARGET_OBS_NUISANCE_R_SCALE,
            "TARGET_OBS_ROLE_POWER": self.TARGET_OBS_ROLE_POWER,
            "RESIDUAL_PRIOR_MIN": self.RESIDUAL_PRIOR_MIN,
            "RESIDUAL_PRIOR_POWER": self.RESIDUAL_PRIOR_POWER,
            "RESIDUAL_GUARD_TRUST_FLOOR": self.RESIDUAL_GUARD_TRUST_FLOOR,
            "RESIDUAL_GUARD_NUISANCE_SCALE": self.RESIDUAL_GUARD_NUISANCE_SCALE,
            "PHASE_MORPH_MAX_BLEND": self.PHASE_MORPH_MAX_BLEND,
            "PHASE_MORPH_BINS": self.PHASE_MORPH_BINS,
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

    def _channel_context_prior(
        self,
        base_method: str,
        obs_cal: Dict[str, float],
    ) -> float:
        """Slow target/context prior for dynamic multi-observation fusion.

        This is deliberately not a selector. It is a bounded prior that says how
        plausible a channel is before the fast innovation statistics arrive.
        The actual time-varying weight still depends on q_obs and R_t.
        """
        sem = self._observation_family_semantics(base_method)
        risk = str(sem.get("nuisance_risk", "medium")).strip().lower()
        risk_scale = {
            "low": 1.00,
            "medium": 0.86,
            "high": 0.72,
        }.get(risk, 0.82)
        role_scale = 1.0
        if bool(sem.get("waveform_primary", False)):
            role_scale *= 1.05
        if bool(sem.get("rate_primary", False)):
            role_scale *= 1.03
        if bool(sem.get("helper_heavy", False)):
            role_scale *= 0.92

        fit_corr = float(obs_cal.get("fit_corr", np.nan))
        fit_rmse = float(obs_cal.get("fit_rmse", np.nan))
        if bool(obs_cal.get("enabled", False)) and np.isfinite(fit_corr):
            corr_score = float(np.clip((fit_corr + 1.0) * 0.5, 0.0, 1.0))
        else:
            corr_score = 0.75
        if bool(obs_cal.get("enabled", False)) and np.isfinite(fit_rmse):
            rmse_score = float(np.exp(-0.5 * (max(fit_rmse, 0.0) / 1.10) ** 2))
        else:
            rmse_score = 0.80
        context = risk_scale * role_scale * (0.35 + 0.65 * corr_score * rmse_score)
        floor = float(np.clip(self.DYNAMIC_MIXTURE_CONTEXT_FLOOR, 0.01, 1.0))
        return float(np.clip(context, floor, 1.50))

    @staticmethod
    def _stable_softmax(logits: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, dtype=np.float64).reshape(-1)
        out = np.zeros_like(z)
        finite = np.isfinite(z)
        if not np.any(finite):
            out[:] = 1.0 / max(1, z.size)
            return out
        zf = z[finite]
        zf = zf - float(np.max(zf))
        ez = np.exp(np.clip(zf, -60.0, 60.0))
        denom = float(np.sum(ez))
        if denom <= 1e-12 or not np.isfinite(denom):
            out[:] = 1.0 / max(1, z.size)
            return out
        out[finite] = ez / denom
        if np.sum(out) <= 1e-12:
            out[:] = 1.0 / max(1, z.size)
        return out

    @staticmethod
    def _fit_track_length(track: np.ndarray, n: int) -> np.ndarray:
        arr = np.asarray(track, dtype=np.float64).reshape(-1)
        n = int(max(n, 0))
        if n == 0:
            return np.array([], dtype=np.float64)
        if arr.size == n:
            return arr.copy()
        if arr.size == 0:
            return np.full(n, np.nan, dtype=np.float64)
        if arr.size == 1:
            return np.full(n, float(arr[0]) if np.isfinite(arr[0]) else np.nan, dtype=np.float64)
        src = np.linspace(0.0, 1.0, arr.size)
        dst = np.linspace(0.0, 1.0, n)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.full(n, np.nan, dtype=np.float64)
        if np.count_nonzero(finite) == 1:
            return np.full(n, float(arr[finite][0]), dtype=np.float64)
        return np.interp(dst, src[finite], arr[finite]).astype(np.float64)

    def _coerce_state_role_prior_runtime(
        self,
        raw: object,
        *,
        n_obs: int,
        n: int,
    ) -> Dict[str, np.ndarray]:
        if not isinstance(raw, dict) or int(n_obs) <= 0 or int(n) <= 0:
            return {}
        out: Dict[str, np.ndarray] = {}
        for role in ("h1", "h2", "b", "r", "abstain", "z_osc", "z_full"):
            if role not in raw:
                continue
            try:
                arr = np.asarray(raw.get(role), dtype=np.float64)
            except Exception:
                continue
            if arr.ndim == 1 and arr.size >= int(n):
                arr = np.repeat(arr.reshape(1, -1), int(n_obs), axis=0)
            if arr.ndim != 2 or arr.shape[0] != int(n_obs) or arr.shape[1] < int(n):
                continue
            arr = arr[:, : int(n)]
            arr = np.where(np.isfinite(arr), arr, np.nan)
            arr = np.clip(arr, 0.0, 1.0)
            default = 0.0 if role == "abstain" else 1.0
            row_fill = np.nanmedian(arr, axis=1)
            row_fill = np.where(np.isfinite(row_fill), row_fill, default)
            bad = ~np.isfinite(arr)
            if np.any(bad):
                arr = arr.copy()
                rows, _cols = np.where(bad)
                arr[bad] = row_fill[rows]
            out[role] = np.clip(arr, 0.0, 1.0)
        return out

    def _state_role_visibility_from_H(self, H_row: np.ndarray) -> Dict[str, float]:
        row = np.asarray(H_row, dtype=np.float64).reshape(-1)
        if row.size < self.STATE_DIM:
            return {"h1": 1.0, "h2": 0.0, "b": 0.0, "r": 0.0}
        scores = {
            "h1": float(np.hypot(row[self.HC1], row[self.HS1])),
            "h2": float(np.hypot(row[self.HC2], row[self.HS2])) if self.ENABLE_HARMONIC2 else 0.0,
            "b": float(np.hypot(row[self.B], row[self.BDOT])) if self.ENABLE_BASELINE else 0.0,
            "r": float(np.hypot(row[self.R], row[self.RDOT])) if self.ENABLE_RESIDUAL else 0.0,
        }
        total = float(sum(max(v, 0.0) for v in scores.values()))
        if not np.isfinite(total) or total <= 1e-12:
            return {"h1": 1.0, "h2": 0.0, "b": 0.0, "r": 0.0}
        return {key: float(max(value, 0.0) / total) for key, value in scores.items()}

    def _state_role_context_from_H(
        self,
        H: np.ndarray,
        role_runtime: Dict[str, np.ndarray],
        t: int,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        H_arr = np.asarray(H, dtype=np.float64)
        if H_arr.ndim == 1:
            H_arr = H_arr.reshape(1, -1)
        n_obs = int(H_arr.shape[0])
        if not role_runtime or n_obs <= 0:
            ones = np.ones(n_obs, dtype=np.float64)
            return ones, {"h1": ones, "h2": ones, "b": ones, "r": ones, "abstain": np.zeros(n_obs)}

        role_values: Dict[str, np.ndarray] = {}
        for role in ("h1", "h2", "b", "r", "z_osc", "z_full"):
            mat = role_runtime.get(role)
            if mat is not None and mat.shape[0] == n_obs and int(t) < mat.shape[1]:
                vals = np.asarray(mat[:, int(t)], dtype=np.float64)
            else:
                vals = np.ones(n_obs, dtype=np.float64)
            role_values[role] = np.clip(np.nan_to_num(vals, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        abstain_mat = role_runtime.get("abstain")
        if abstain_mat is not None and abstain_mat.shape[0] == n_obs and int(t) < abstain_mat.shape[1]:
            abstain = np.asarray(abstain_mat[:, int(t)], dtype=np.float64)
        else:
            abstain = np.zeros(n_obs, dtype=np.float64)
        abstain = np.clip(np.nan_to_num(abstain, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        role_values["abstain"] = abstain

        context = np.ones(n_obs, dtype=np.float64)
        for idx in range(n_obs):
            visibility = self._state_role_visibility_from_H(H_arr[idx])
            weighted = 0.0
            for role in ("h1", "h2", "b", "r"):
                weighted += float(visibility.get(role, 0.0)) * float(role_values[role][idx])
            if not np.isfinite(weighted) or weighted <= 0.0:
                weighted = 1.0
            # z_full is morphology-level confidence, while h1/h2/b/r encode
            # component-specific state visibility. Combine them conservatively.
            weighted = float(np.sqrt(max(weighted, 1e-6) * max(float(role_values["z_full"][idx]), 1e-6)))
            weighted *= 1.0 / (1.0 + float(self.STATE_ROLE_ABSTAIN_R_SCALE) * float(abstain[idx]))
            context[idx] = float(np.clip(weighted, 1e-6, 1.0))
        return context, role_values

    def _family_group_key(self, family: str) -> str:
        sem = self._observation_family_semantics(str(family or ""))
        group = str(sem.get("family_group", "") or "").strip().lower()
        return group or str(family or "unknown").strip().lower()

    def _family_balanced_nanmedian(
        self,
        values: np.ndarray,
        families: Sequence[str],
        *,
        fallback: float,
    ) -> np.ndarray:
        """Median across semantic family groups, not raw channels.

        Several channels are deliberately related construction variants
        (e.g. P1D_quad/P1D_cub). Treating them as independent votes makes
        target-side rate evidence overconfident under domain shift.
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            return np.asarray(arr, dtype=np.float64)
        n_obs = int(arr.shape[0])
        if n_obs == 0:
            return np.asarray([], dtype=np.float64)
        if len(families) != n_obs:
            families = [f"ch{i}" for i in range(n_obs)]
        groups: Dict[str, List[int]] = {}
        for idx, family in enumerate(families):
            groups.setdefault(self._family_group_key(str(family)), []).append(idx)
        reduced = []
        for idxs in groups.values():
            block = arr[np.asarray(idxs, dtype=int)]
            with np.errstate(all="ignore"):
                reduced.append(np.nanmedian(block, axis=0))
        if not reduced:
            out = np.nanmedian(arr, axis=0)
        else:
            with np.errstate(all="ignore"):
                out = np.nanmedian(np.stack(reduced, axis=0), axis=0)
        out_arr = np.asarray(out, dtype=np.float64)
        if np.all(np.isfinite(out_arr)):
            return out_arr
        finite = np.isfinite(out_arr)
        fill = float(fallback) if np.isfinite(fallback) else 0.0
        if np.any(finite):
            fill = float(np.nanmedian(out_arr[finite]))
        return np.where(finite, out_arr, fill).astype(np.float64)

    def _family_group_indices(self, families: Sequence[str], n_obs: int) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        if len(families) != n_obs:
            families = [f"ch{i}" for i in range(n_obs)]
        for idx, family in enumerate(families):
            groups.setdefault(self._family_group_key(str(family)), []).append(idx)
        return groups

    @staticmethod
    def _weighted_nanmedian(values: np.ndarray, weights: np.ndarray, *, fallback: float) -> float:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if vals.size != w.size:
            return float(fallback)
        ok = np.isfinite(vals) & np.isfinite(w) & (w > 0.0)
        if not np.any(ok):
            return float(fallback)
        vals = vals[ok]
        w = w[ok]
        order = np.argsort(vals)
        vals = vals[order]
        w = w[order]
        cdf = np.cumsum(w)
        total = float(cdf[-1])
        if total <= 1e-12 or not np.isfinite(total):
            return float(fallback)
        idx = int(np.searchsorted(cdf, 0.5 * total, side="left"))
        idx = min(max(idx, 0), vals.size - 1)
        return float(vals[idx])

    def _harmonic_candidate_agreement(self, candidate_hz: float, observed_hz: float) -> float:
        if not np.isfinite(candidate_hz) or not np.isfinite(observed_hz):
            return 0.0
        f_min = float(self.params.f_min)
        f_max = float(self.params.f_max)
        observed_hz = float(np.clip(observed_hz, f_min, f_max))
        candidate_hz = float(np.clip(candidate_hz, f_min, f_max))
        ref = max(float(self.RATE_OBS_AGREE_REF_HZ), 1e-6)
        penalty = float(np.clip(self.RATE_OBS_HARMONIC_PENALTY, 0.0, 1.0))
        views = [(observed_hz, 1.0)]
        if observed_hz * 2.0 <= f_max:
            views.append((observed_hz * 2.0, penalty))
        if observed_hz * 0.5 >= f_min:
            views.append((observed_hz * 0.5, 0.75 * penalty))
        best = 0.0
        for view_hz, view_weight in views:
            score = float(view_weight) * float(np.exp(-0.5 * ((candidate_hz - view_hz) / ref) ** 2))
            best = max(best, score)
        return float(np.clip(best, 0.0, 1.0))

    def _rate_observability_scores(
        self,
        helper_freq_window: np.ndarray,
        families: Sequence[str],
        *,
        fallback_freq: float,
        anchor_freq: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """GT-free rate/phase observability from cross-family helper agreement.

        Innovation statistics can be low even when all channels agree on the
        wrong harmonic. This score asks a different question: do semantic
        family groups provide a stable and mutually compatible respiratory
        rate, allowing direct, double, and half-rate relationships with a
        penalty? It returns per-channel reliability plus a harmonic-aware
        consensus helper frequency for the current time.
        """
        arr = np.asarray(helper_freq_window, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return np.ones(0, dtype=np.float64), float(fallback_freq), 0.0
        n_obs, win = int(arr.shape[0]), int(arr.shape[1])
        floor = float(np.clip(self.RATE_OBS_FLOOR, 0.01, 1.0))
        scores = np.full(n_obs, 1.0, dtype=np.float64)
        if win < 3:
            return scores, float(fallback_freq), 0.0

        f_min = float(self.params.f_min)
        f_max = float(self.params.f_max)
        std_ref = max(float(self.RATE_OBS_STD_REF_HZ), 1e-6)
        min_valid = max(3, min(win, int(np.ceil(0.35 * win))))
        ch_freq = np.full(n_obs, np.nan, dtype=np.float64)
        ch_stability = np.full(n_obs, floor, dtype=np.float64)
        for idx in range(n_obs):
            vals = arr[idx]
            valid = vals[np.isfinite(vals) & (vals >= f_min) & (vals <= f_max)]
            if valid.size < min_valid:
                continue
            ch_freq[idx] = float(np.median(valid))
            ch_std = float(np.std(valid))
            ch_stability[idx] = float(np.exp(-0.5 * (ch_std / std_ref) ** 2))
            ch_stability[idx] = float(np.clip(ch_stability[idx], floor, 1.0))

        groups = self._family_group_indices(families, n_obs)
        group_freq: Dict[str, float] = {}
        group_stability: Dict[str, float] = {}
        for group, idxs in groups.items():
            idx_arr = np.asarray(idxs, dtype=int)
            vals = ch_freq[idx_arr]
            weights = ch_stability[idx_arr]
            ok = np.isfinite(vals)
            if not np.any(ok):
                continue
            group_freq[group] = self._weighted_nanmedian(vals, weights, fallback=float(fallback_freq))
            group_stability[group] = float(np.clip(np.nanmedian(weights[ok]), floor, 1.0))

        if not group_freq:
            return scores, float(fallback_freq), 0.0

        penalty = float(np.clip(self.RATE_OBS_HARMONIC_PENALTY, 0.0, 1.0))
        candidate_rates: List[Tuple[float, float]] = []
        for group, freq in group_freq.items():
            stab = float(group_stability.get(group, floor))
            if np.isfinite(freq) and f_min <= freq <= f_max:
                candidate_rates.append((float(freq), stab))
            if np.isfinite(freq) and f_min <= 2.0 * freq <= f_max:
                candidate_rates.append((float(2.0 * freq), stab * penalty))
            if np.isfinite(freq) and f_min <= 0.5 * freq <= f_max:
                candidate_rates.append((float(0.5 * freq), stab * 0.75 * penalty))

        if not candidate_rates:
            return scores, float(fallback_freq), 0.0

        weight_sum = max(float(np.sum(list(group_stability.values()))), 1e-12)
        best_rate = float(fallback_freq)
        best_score = -np.inf
        anchor = float(anchor_freq) if anchor_freq is not None and np.isfinite(anchor_freq) else np.nan
        anchor_ref = max(2.0 * float(self.RATE_OBS_AGREE_REF_HZ), 1e-6)
        for candidate, prior_weight in candidate_rates:
            support = 0.0
            for group, freq in group_freq.items():
                support += float(group_stability.get(group, floor)) * self._harmonic_candidate_agreement(candidate, freq)
            support = support / weight_sum
            if np.isfinite(anchor):
                anchor_score = float(np.exp(-0.5 * ((candidate - anchor) / anchor_ref) ** 2))
                support *= 0.70 + 0.30 * anchor_score
            support *= 0.75 + 0.25 * float(np.clip(prior_weight, 0.0, 1.0))
            if support > best_score:
                best_score = float(support)
                best_rate = float(candidate)

        best_score = float(np.clip(best_score, 0.0, 1.0)) if np.isfinite(best_score) else 0.0
        if best_score < float(np.clip(self.RATE_OBS_HELPER_MIN_SUPPORT, 0.0, 1.0)):
            best_rate = float(fallback_freq)

        group_support: Dict[str, float] = {
            group: self._harmonic_candidate_agreement(best_rate, freq)
            for group, freq in group_freq.items()
        }
        scores = np.full(n_obs, floor, dtype=np.float64)
        for idx in range(n_obs):
            group = self._family_group_key(str(families[idx])) if idx < len(families) else f"ch{idx}"
            support = float(group_support.get(group, floor))
            stability = float(ch_stability[idx]) if np.isfinite(ch_stability[idx]) else floor
            scores[idx] = floor + (1.0 - floor) * np.sqrt(max(support, 0.0) * max(stability, 0.0))
        scores = np.clip(np.nan_to_num(scores, nan=floor, posinf=1.0, neginf=floor), floor, 1.0)
        return scores.astype(np.float64), float(np.clip(best_rate, f_min, f_max)), best_score

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

    def _external_output_rate_postprocess(
        self,
        track_hz: np.ndarray,
        output_rate_hz: Optional[np.ndarray],
        output_rate_confidence: Optional[np.ndarray],
        alpha_used: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a decoupled z_osc timing readout without changing z_full.

        The external readout is the observation-law product for reported
        timing. When it forms a continuous target-side readout, it should
        arbitrate rate/BPM directly; otherwise it remains a conservative
        correction to the state-space phase estimate.
        """
        out = np.asarray(track_hz, dtype=np.float64).copy()
        active = np.zeros(out.size, dtype=np.float64)
        if output_rate_hz is None or output_rate_confidence is None or out.size == 0:
            return out, active
        candidate = np.asarray(output_rate_hz, dtype=np.float64).reshape(-1)
        confidence = np.asarray(output_rate_confidence, dtype=np.float64).reshape(-1)
        if candidate.size != out.size or confidence.size != out.size:
            return out, active

        candidate = np.where(
            np.isfinite(candidate)
            & (candidate >= float(self.params.f_min))
            & (candidate <= float(self.params.f_max)),
            candidate,
            np.nan,
        )
        confidence = np.clip(np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        valid = np.isfinite(candidate) & np.isfinite(out) & (confidence > 0.0)
        if not np.any(valid):
            return out, active

        candidate_filled = candidate.copy()
        candidate_filled[~np.isfinite(candidate_filled)] = out[~np.isfinite(candidate_filled)]
        candidate_filled = self._apply_post_smoothing(candidate_filled, alpha_override=alpha_used)
        candidate_filled = np.clip(candidate_filled, self.params.f_min, self.params.f_max)

        # The external readout is target-computable evidence, not permission to
        # discard the oscillator.  Earlier versions replaced the native track
        # whenever coverage was high; the COHFACE audit showed that this can
        # turn an excellent native z_osc track into a worse reported rate.
        #
        # Confidence alone is also not enough: hard-regime targets can produce
        # lower posterior confidence exactly when the native oscillator is
        # most aliased.  Use native/external disagreement as the primary
        # target-side signal, and use confidence to bound rather than veto the
        # correction.
        conflict_hz = np.abs(candidate_filled - out)
        conflict_strength = np.clip((conflict_hz - 0.015) / 0.055, 0.0, 1.0)
        confidence_strength = 0.45 + 0.55 * np.sqrt(np.clip(confidence, 0.0, 1.0))
        blend = confidence_strength * (0.04 + 0.76 * np.power(conflict_strength, 1.35))
        likely_harmonic_downshift = (candidate_filled < out) & (conflict_hz >= 0.020)
        blend = np.where(likely_harmonic_downshift, 1.12 * blend, blend)
        blend = np.where(valid, np.clip(blend, 0.0, 0.78), 0.0)
        out[valid] = (1.0 - blend[valid]) * out[valid] + blend[valid] * candidate_filled[valid]
        out = np.clip(out, self.params.f_min, self.params.f_max)
        active[valid] = blend[valid]
        return out, active

    def _rate_source_arbiter_v1_postprocess(
        self,
        current_track: np.ndarray,
        native_track: np.ndarray,
        state_freq: np.ndarray,
        output_rate_hz: Optional[np.ndarray],
        output_rate_confidence: Optional[np.ndarray],
        posterior: Dict[str, np.ndarray],
        alpha_used: float,
        guard_version: str = "v1",
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """GT-free conservative arbitration among available rate sources.

        This is not a supervised selector.  It keeps the current reported track
        as the anchor and only moves toward posterior/native/state candidates
        when target-computable confidence, ambiguity, alias, and agreement
        evidence supports the alternative.
        """
        current = np.asarray(current_track, dtype=np.float64).copy()
        n = current.size
        empty = {
            "blend": np.zeros(n, dtype=np.float64),
            "selected": np.zeros(n, dtype=np.float64),
            "current_score": np.zeros(n, dtype=np.float64),
            "posterior_mean_score": np.zeros(n, dtype=np.float64),
            "posterior_mode_score": np.zeros(n, dtype=np.float64),
            "native_score": np.zeros(n, dtype=np.float64),
            "state_score": np.zeros(n, dtype=np.float64),
            "output_score": np.zeros(n, dtype=np.float64),
        }
        if n == 0:
            return current, empty

        native = self._fit_track_length(np.asarray(native_track, dtype=np.float64), n)
        state = self._fit_track_length(np.asarray(state_freq, dtype=np.float64), n)
        state = self._apply_post_smoothing(state, alpha_override=alpha_used)
        state = np.clip(state, self.params.f_min, self.params.f_max)
        output = (
            self._fit_track_length(np.asarray(output_rate_hz, dtype=np.float64), n)
            if output_rate_hz is not None
            else current.copy()
        )
        output_conf = (
            self._fit_track_length(np.asarray(output_rate_confidence, dtype=np.float64), n)
            if output_rate_confidence is not None
            else np.zeros(n, dtype=np.float64)
        )
        output_conf = np.clip(np.nan_to_num(output_conf, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        post_mean = posterior.get("mean_hz")
        post_mode = posterior.get("mode_hz")
        post_conf = posterior.get("confidence")
        if post_mean is None or post_mode is None or post_conf is None:
            return current, empty
        post_mean = self._fit_track_length(np.asarray(post_mean, dtype=np.float64), n)
        post_mode = self._fit_track_length(np.asarray(post_mode, dtype=np.float64), n)
        post_conf = self._fit_track_length(np.asarray(post_conf, dtype=np.float64), n)
        entropy = self._fit_track_length(np.asarray(posterior.get("entropy", np.ones(n)), dtype=np.float64), n)
        top_gap = self._fit_track_length(np.asarray(posterior.get("top_gap", np.zeros(n)), dtype=np.float64), n)
        macro = self._fit_track_length(np.asarray(posterior.get("macro_support", np.zeros(n)), dtype=np.float64), n)
        direct_macro = self._fit_track_length(
            np.asarray(posterior.get("direct_macro_support", macro), dtype=np.float64), n
        )
        motion_direct = self._fit_track_length(
            np.asarray(posterior.get("motion_direct_support", np.zeros(n)), dtype=np.float64), n
        )
        alias = self._fit_track_length(np.asarray(posterior.get("alias_risk", np.zeros(n)), dtype=np.float64), n)
        h1_role = self._fit_track_length(np.asarray(posterior.get("h1_role_support", direct_macro), dtype=np.float64), n)
        abstain = self._fit_track_length(np.asarray(posterior.get("abstain_pressure", 1.0 - h1_role), dtype=np.float64), n)

        post_conf = np.clip(np.nan_to_num(post_conf, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        entropy = np.clip(np.nan_to_num(entropy, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        top_gap = np.clip(np.nan_to_num(top_gap, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        macro = np.clip(np.nan_to_num(macro, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        direct_macro = np.clip(np.nan_to_num(direct_macro, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        motion_direct = np.clip(np.nan_to_num(motion_direct, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        alias = np.clip(np.nan_to_num(alias, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        h1_role = np.clip(np.nan_to_num(h1_role, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        abstain = np.clip(np.nan_to_num(abstain, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

        valid_current = np.isfinite(current)
        for arr in (native, state, output, post_mean, post_mode):
            arr[~(np.isfinite(arr) & (arr >= self.params.f_min) & (arr <= self.params.f_max))] = np.nan

        agreement_ref = 0.075
        posterior_specificity = np.clip(
            post_conf
            * (0.35 + 0.65 * (1.0 - entropy))
            * (0.45 + 0.55 * top_gap)
            * (0.60 + 0.40 * macro)
            * (0.70 + 0.30 * direct_macro)
            * (1.0 - 0.55 * alias)
            * (0.70 + 0.30 * h1_role)
            * (1.0 - 0.35 * abstain),
            0.0,
            1.0,
        )
        output_state_agree = np.exp(-0.5 * ((output - state) / agreement_ref) ** 2)
        output_native_agree = np.exp(-0.5 * ((output - native) / agreement_ref) ** 2)
        output_posterior_agree = np.exp(-0.5 * ((output - post_mean) / agreement_ref) ** 2)
        post_state_agree = np.exp(-0.5 * ((post_mean - state) / agreement_ref) ** 2)
        native_state_agree = np.exp(-0.5 * ((native - state) / agreement_ref) ** 2)

        output_current_agree = np.exp(-0.5 * ((output - current) / agreement_ref) ** 2)
        native_current_agree = np.exp(-0.5 * ((native - current) / agreement_ref) ** 2)
        post_current_agree = np.exp(-0.5 * ((post_mean - current) / agreement_ref) ** 2)
        native_posterior_agree = np.exp(-0.5 * ((native - post_mean) / agreement_ref) ** 2)
        output_preservation = np.clip(
            output_conf
            * (0.55 + 0.45 * np.nan_to_num(output_current_agree, nan=0.0))
            * (0.65 + 0.35 * (1.0 - np.nan_to_num(output_native_agree, nan=0.0)))
            * (0.70 + 0.30 * entropy),
            0.0,
            1.0,
        )
        native_alias_safety = np.clip(
            0.38 * np.nan_to_num(native_current_agree, nan=0.0)
            + 0.30 * np.nan_to_num(native_posterior_agree, nan=0.0)
            + 0.20 * posterior_specificity
            + 0.12 * motion_direct,
            0.0,
            1.0,
        )

        current_score = np.clip(
            0.20
            + 0.42 * output_conf
            + 0.18 * np.nan_to_num(output_state_agree, nan=0.0)
            + 0.12 * np.nan_to_num(output_native_agree, nan=0.0)
            + 0.08 * (1.0 - abstain),
            0.0,
            1.0,
        )
        posterior_mean_score = np.clip(
            0.05
            + 0.70 * posterior_specificity
            + 0.18 * np.nan_to_num(post_state_agree, nan=0.0)
            + 0.07 * motion_direct,
            0.0,
            1.0,
        )
        posterior_mode_score = np.clip(
            posterior_mean_score
            * (0.50 + 0.50 * top_gap)
            * (0.70 + 0.30 * (1.0 - entropy)),
            0.0,
            1.0,
        )
        native_score = np.clip(
            0.12
            + 0.42 * np.nan_to_num(native_state_agree, nan=0.0)
            + 0.22 * abstain
            + 0.12 * (1.0 - output_conf)
            + 0.12 * (1.0 - alias),
            0.0,
            1.0,
        )
        state_score = np.clip(
            0.10
            + 0.36 * np.nan_to_num(native_state_agree, nan=0.0)
            + 0.26 * abstain
            + 0.16 * (1.0 - entropy)
            + 0.12 * h1_role,
            0.0,
            1.0,
        )
        output_score = np.zeros(n, dtype=np.float64)

        version = str(guard_version).strip().lower()
        if version == "v2":
            # Native/state can agree on the same alias.  v2 keeps the final
            # readout as a stronger anchor unless the alternative has either
            # posterior specificity or very low current-output confidence.
            low_output_confidence = output_conf <= 0.18
            native_guard_multiplier = np.where(
                low_output_confidence,
                1.0,
                0.42 + 0.58 * native_alias_safety,
            )
            state_guard_multiplier = np.where(
                low_output_confidence,
                1.0,
                0.50
                + 0.30 * np.nan_to_num(post_state_agree, nan=0.0)
                + 0.20 * posterior_specificity,
            )
            current_score = np.clip(current_score + 0.24 * output_preservation, 0.0, 1.0)
            native_score = np.clip(native_score * native_guard_multiplier, 0.0, 1.0)
            state_score = np.clip(state_score * state_guard_multiplier, 0.0, 1.0)
            posterior_mean_score = np.clip(
                posterior_mean_score
                + 0.08 * np.nan_to_num(post_current_agree, nan=0.0)
                - 0.08 * alias,
                0.0,
                1.0,
            )
            posterior_mode_score = np.clip(
                posterior_mode_score
                + 0.05 * np.nan_to_num(post_current_agree, nan=0.0)
                - 0.05 * alias,
                0.0,
                1.0,
            )

        if version == "v3":
            # v3 is a source-role arbiter, not another generic blend constant.
            # It treats the target-computable external readout as an explicit
            # candidate only when its disagreement with the native oscillator is
            # accompanied by posterior/state support.  This preserves easy
            # observable trials while allowing hard-regime alias correction.
            output_native_conflict = np.abs(output - native)
            hard_conflict = np.clip((output_native_conflict - 0.025) / 0.075, 0.0, 1.0)
            output_support = np.clip(
                0.40 * output_conf
                + 0.24 * np.nan_to_num(output_posterior_agree, nan=0.0)
                + 0.22 * np.nan_to_num(output_state_agree, nan=0.0)
                + 0.14 * posterior_specificity,
                0.0,
                1.0,
            )
            current_score = np.clip(
                0.18
                + 0.28 * np.nan_to_num(output_current_agree, nan=0.0)
                + 0.20 * np.nan_to_num(output_native_agree, nan=0.0)
                + 0.18 * (1.0 - hard_conflict)
                + 0.10 * output_conf
                + 0.06 * (1.0 - posterior_specificity)
                + 0.08 * (1.0 - abstain)
                - 0.36 * hard_conflict * output_support
                - 0.12 * hard_conflict * output_conf,
                0.0,
                1.0,
            )
            output_score = np.clip(
                0.08
                + 0.30 * output_conf
                + 0.24 * np.nan_to_num(output_posterior_agree, nan=0.0)
                + 0.24 * np.nan_to_num(output_state_agree, nan=0.0)
                + 0.36 * hard_conflict
                + 0.10 * (1.0 - abstain),
                0.0,
                1.0,
            )
            posterior_mean_score = np.clip(
                posterior_mean_score
                + 0.16 * np.nan_to_num(output_posterior_agree, nan=0.0)
                + 0.10 * hard_conflict
                + 0.06 * output_conf,
                0.0,
                1.0,
            )
            posterior_mode_score = np.clip(
                posterior_mode_score
                + 0.10 * np.nan_to_num(output_posterior_agree, nan=0.0)
                + 0.06 * hard_conflict,
                0.0,
                1.0,
            )
            native_score = np.clip(
                native_score * (1.0 - 0.72 * hard_conflict * (0.35 + 0.65 * output_support)),
                0.0,
                1.0,
            )
            state_score = np.clip(
                state_score
                * (
                    0.55
                    + 0.25 * np.nan_to_num(output_state_agree, nan=0.0)
                    + 0.20 * posterior_specificity
                ),
                0.0,
                1.0,
            )

        candidates = np.vstack([current, post_mean, post_mode, native, state])
        scores = np.vstack([current_score, posterior_mean_score, posterior_mode_score, native_score, state_score])
        if version == "v3":
            candidates = np.vstack([candidates, output])
            scores = np.vstack([scores, output_score])
        finite_candidates = np.isfinite(candidates) & (candidates >= self.params.f_min) & (candidates <= self.params.f_max)
        scores = np.where(finite_candidates, scores, -np.inf)
        scores[0, valid_current] = np.maximum(scores[0, valid_current], 0.0)
        best_idx = np.argmax(scores, axis=0)
        best_score = np.max(scores, axis=0)
        best_value = candidates[best_idx, np.arange(n)]
        score_margin = best_score - current_score
        delta = np.abs(best_value - current)
        switch_margin = np.full(n, 0.15, dtype=np.float64)
        if version == "v3":
            switch_margin = np.where(best_idx == 5, 0.02, 0.12)
        switch = (
            valid_current
            & np.isfinite(best_value)
            & (best_idx > 0)
            & (score_margin >= switch_margin)
            & (delta >= 0.018)
        )
        if version == "v2":
            source_safety = np.ones(n, dtype=bool)
            native_or_state = (best_idx == 3) | (best_idx == 4)
            source_safety[native_or_state] = (
                (output_conf[native_or_state] <= 0.18)
                | (delta[native_or_state] <= 0.040)
                | (posterior_specificity[native_or_state] >= 0.36)
                | (
                    (native_alias_safety[native_or_state] >= 0.62)
                    & (np.nan_to_num(output_preservation[native_or_state], nan=0.0) <= 0.28)
                )
            )
            switch &= source_safety
        if version == "v3":
            output_native_conflict = np.abs(output - native)
            hard_conflict = np.clip((output_native_conflict - 0.025) / 0.075, 0.0, 1.0)
            output_supported = (
                (output_conf >= 0.08)
                & (hard_conflict >= 0.16)
                & (
                    (np.nan_to_num(output_posterior_agree, nan=0.0) >= 0.50)
                    | (np.nan_to_num(output_state_agree, nan=0.0) >= 0.50)
                    | (posterior_specificity >= 0.20)
                )
            )
            posterior_supported = (
                (posterior_specificity >= 0.22)
                | (
                    (hard_conflict >= 0.35)
                    & (np.nan_to_num(output_posterior_agree, nan=0.0) >= 0.45)
                )
            )
            native_state_supported = (
                (output_conf <= 0.18)
                | (delta <= 0.040)
                | (posterior_specificity >= 0.38)
                | (hard_conflict <= 0.18)
            )
            source_safety = np.ones(n, dtype=bool)
            source_safety[best_idx == 5] = output_supported[best_idx == 5]
            source_safety[(best_idx == 1) | (best_idx == 2)] = posterior_supported[(best_idx == 1) | (best_idx == 2)]
            source_safety[(best_idx == 3) | (best_idx == 4)] = native_state_supported[(best_idx == 3) | (best_idx == 4)]
            switch &= source_safety
        blend = np.zeros(n, dtype=np.float64)
        if version == "v3":
            output_native_conflict = np.abs(output - native)
            hard_conflict = np.clip((output_native_conflict - 0.025) / 0.075, 0.0, 1.0)
            blend_raw = 0.14 + 0.58 * score_margin + 0.20 * hard_conflict
            output_blend_raw = 0.25 + 0.40 * score_margin + 1.05 * hard_conflict
            blend_raw = np.where(
                best_idx == 5,
                output_blend_raw * (0.85 + 0.15 * output_conf),
                blend_raw,
            )
            output_cap = np.where(hard_conflict >= 0.18, 0.88, 0.38)
            blend_cap = np.where(best_idx == 5, output_cap, 0.58)
            blend_cap = np.where((best_idx == 3) | (best_idx == 4), 0.42, blend_cap)
            blend[switch] = np.minimum(np.clip(blend_raw[switch], 0.0, 0.86), blend_cap[switch])
        else:
            blend[switch] = np.clip(0.18 + 0.55 * score_margin[switch], 0.0, 0.52)
        out = current.copy()
        out[switch] = (1.0 - blend[switch]) * current[switch] + blend[switch] * best_value[switch]
        out = np.clip(out, self.params.f_min, self.params.f_max)
        diag = {
            "blend": blend,
            "selected": best_idx.astype(np.float64),
            "current_score": current_score,
            "posterior_mean_score": posterior_mean_score,
            "posterior_mode_score": posterior_mode_score,
            "native_score": native_score,
            "state_score": state_score,
            "output_score": output_score,
            "output_preservation": output_preservation,
            "native_alias_safety": native_alias_safety,
            "posterior_specificity": posterior_specificity,
        }
        return out, diag

    def _coerce_rate_posterior_runtime(self, raw: object, n: int) -> Dict[str, np.ndarray]:
        """Parse candidate-rate posterior arrays from materialization metadata."""
        if not isinstance(raw, dict) or int(n) <= 0:
            return {}
        out: Dict[str, np.ndarray] = {}
        for key in (
            "mode_hz",
            "mean_hz",
            "confidence",
            "entropy",
            "top_gap",
            "support",
            "direct_support",
            "macro_support",
            "direct_macro_support",
            "motion_direct_support",
            "p1d_half_support",
            "alias_risk",
            "independent_timing_support",
            "motion_timing_support",
            "bridge_timing_preservation",
            "p1d_direct_timing_support",
            "morphology_role_support",
            "morphology_alias_pressure",
            "h1_role_support",
            "abstain_pressure",
        ):
            value = raw.get(key)
            if value is None:
                continue
            try:
                arr = self._fit_track_length(np.asarray(value, dtype=np.float64), int(n))
            except Exception:
                continue
            if key in {"mode_hz", "mean_hz"}:
                arr = np.where(
                    np.isfinite(arr)
                    & (arr >= float(self.params.f_min))
                    & (arr <= float(self.params.f_max)),
                    arr,
                    np.nan,
                )
            else:
                arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            out[key] = arr.astype(np.float64)
        if "mode_hz" not in out or "confidence" not in out:
            return {}
        out["confidence"] = np.where(np.isfinite(out["mode_hz"]), out["confidence"], 0.0)
        if not np.any(out["confidence"] > 0.0):
            return {}
        return out

    def _coerce_target_observability_runtime(self, raw: object, n: int) -> Dict[str, np.ndarray]:
        """Parse GT-free target observability control arrays.

        The arrays are intentionally scalar-in-time controls rather than
        per-output overrides. They gate observation trust before the SSM update.
        """
        if not isinstance(raw, dict) or int(n) <= 0:
            return {}
        keys_defaults = {
            "target_observability": 1.0,
            "h1_timing": 1.0,
            "h2_morphology": 1.0,
            "baseline": 1.0,
            "residual": 1.0,
            "nuisance": 0.0,
            "source_spread_hz": 0.0,
            "source_agreement": 1.0,
            "posterior_specificity": 0.0,
            "alias_safety": 1.0,
        }
        out: Dict[str, np.ndarray] = {}
        any_present = False
        for key, default in keys_defaults.items():
            value = raw.get(key)
            if value is None:
                arr = np.full(int(n), float(default), dtype=np.float64)
            else:
                any_present = True
                try:
                    arr = self._fit_track_length(np.asarray(value, dtype=np.float64), int(n))
                except Exception:
                    arr = np.full(int(n), float(default), dtype=np.float64)
                if key == "source_spread_hz":
                    arr = np.nan_to_num(arr, nan=float(default), posinf=float(default), neginf=0.0)
                    arr = np.clip(arr, 0.0, np.inf)
                else:
                    arr = np.clip(
                        np.nan_to_num(arr, nan=float(default), posinf=1.0, neginf=0.0),
                        0.0,
                        1.0,
                    )
            out[key] = arr.astype(np.float64)
        if not any_present:
            return {}
        return out

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
        elif base in {"profile1d_linear", "profile1d_linear_bridge"}:
            is_bridge = base.endswith("_bridge")
            prior.update({
                "enabled": True,
                "obs_domain": str(sem.get("observation_domain", "displacement")),
                "g_h1": 1.00,
                "g_h2": 0.45 if is_bridge else 0.35,
                "g_aux": 0.05 if is_bridge else 0.10,
                "g_b": 0.04 if is_bridge else 0.08,
                "g_r": 0.06 if is_bridge else 0.10,
                "max_g_aux": min(float(self.OBS_CAL_MAX_GAIN_AUX), 0.18 if is_bridge else 0.30),
                "max_g_b": min(float(self.OBS_CAL_MAX_GAIN_B), 0.12 if is_bridge else 0.20),
                "max_g_r": min(float(self.OBS_CAL_MAX_GAIN_R), 0.18 if is_bridge else 0.25),
                "prior_strength": max(float(self.OBS_CAL_PRIOR_STRENGTH), 1.25 if is_bridge else 1.0),
                "max_lag_sec": min(0.08 if is_bridge else 0.12, float(self.OBS_CAL_MAX_LAG_SEC)),
                "min_fit_corr": float(self.OBS_CAL_MIN_FIT_CORR),
                "max_fit_rmse_norm": float(self.OBS_CAL_MAX_FIT_RMSE_NORM),
            })
        elif base == "profile1d_consensus":
            prior.update({
                "enabled": True,
                "obs_domain": str(sem.get("observation_domain", "displacement")),
                "g_h1": 1.00,
                "g_h2": 1.10,
                "g_aux": 0.0 if self.QUADCUB_HARMONIC_ONLY else 0.10,
                "g_b": 0.0 if self.QUADCUB_HARMONIC_ONLY else 0.08,
                "g_r": 0.0 if self.QUADCUB_HARMONIC_ONLY else 0.20,
                "max_g_h1": min(float(self.QUADCUB_MAX_GAIN_H1), float(self.OBS_CAL_MAX_GAIN_H1), 1.15),
                "max_g_h2": min(float(self.QUADCUB_MAX_GAIN_H2 if self.ENABLE_HARMONIC2 else 0.0), float(self.OBS_CAL_MAX_GAIN_H2 if self.ENABLE_HARMONIC2 else 0.0), 1.45),
                "max_g_aux": 0.0 if self.QUADCUB_HARMONIC_ONLY else min(float(self.OBS_CAL_MAX_GAIN_AUX), 0.22),
                "max_g_b": 0.0 if self.QUADCUB_HARMONIC_ONLY else min(float(self.OBS_CAL_MAX_GAIN_B), 0.12),
                "max_g_r": 0.0 if self.QUADCUB_HARMONIC_ONLY else min(float(self.OBS_CAL_MAX_GAIN_R), 0.30),
                "prior_strength": max(float(self.QUADCUB_PRIOR_STRENGTH), 1.45),
                "max_lag_sec": min(0.06, float(self.OBS_CAL_MAX_LAG_SEC)),
                "min_fit_corr": float(self.OBS_CAL_MIN_FIT_CORR),
                "max_fit_rmse_norm": float(self.OBS_CAL_MAX_FIT_RMSE_NORM),
            })
        elif str(sem.get("family_group", "")) in {"profile1d_harmonic", "profile1d_harmonic_bridge"}:
            is_bridge = base.endswith("_bridge")
            harmonic_only = bool(self.QUADCUB_HARMONIC_ONLY)
            prior.update({
                "enabled": True,
                "obs_domain": str(sem.get("observation_domain", "displacement")),
                "g_h1": 1.00,
                "g_h2": 1.15 if is_bridge else 1.05,
                "g_aux": 0.0 if harmonic_only else 0.18,
                "g_b": 0.0 if harmonic_only else 0.12,
                "g_r": 0.0 if harmonic_only else 0.32,
                "max_g_h1": min(float(self.QUADCUB_MAX_GAIN_H1), float(self.OBS_CAL_MAX_GAIN_H1), 1.20 if is_bridge else 1.25),
                "max_g_h2": min(
                    float(self.QUADCUB_MAX_GAIN_H2),
                    float(self.OBS_CAL_MAX_GAIN_H2 if self.ENABLE_HARMONIC2 else 0.0),
                    1.45 if is_bridge else 1.60,
                ),
                "max_g_aux": 0.0 if harmonic_only else min(float(self.OBS_CAL_MAX_GAIN_AUX), 0.45),
                "max_g_b": 0.0 if harmonic_only else min(float(self.OBS_CAL_MAX_GAIN_B), 0.30),
                "max_g_r": 0.0 if harmonic_only else min(float(self.OBS_CAL_MAX_GAIN_R), 0.75),
                "prior_strength": max(
                    float(self.QUADCUB_PRIOR_STRENGTH) if harmonic_only else float(self.OBS_CAL_PRIOR_STRENGTH),
                    1.40 if is_bridge else 1.25,
                ),
                "max_lag_sec": min(
                    float(self.QUADCUB_MAX_LAG_SEC) if harmonic_only else (0.08 if is_bridge else 0.12),
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

        This is intentionally conservative. The fallback is only enabled for
        families where the generic fixed-sum observation row is known to be a
        poor inductive bias. Right now that includes raw OF velocity semantics
        and an experimental P1D family-prior fallback.
        """
        out = dict(default)
        out["family"] = str(base_method or "").lower().strip()
        out["fallback_mode"] = "fixed_sum"

        base = out["family"]

        def _estimate_obs_sign() -> float:
            obs_sign_local = 1.0
            try:
                if y_fit is not None and y_helper_fit is not None:
                    y_fit_arr = np.asarray(y_fit, dtype=np.float64).reshape(-1)
                    y_helper_arr = np.asarray(y_helper_fit, dtype=np.float64).reshape(-1)
                    valid = np.isfinite(y_fit_arr) & np.isfinite(y_helper_arr)
                    if int(np.count_nonzero(valid)) >= 8:
                        corr = float(np.corrcoef(y_fit_arr[valid], y_helper_arr[valid])[0, 1])
                        if np.isfinite(corr) and corr < 0.0:
                            obs_sign_local = -1.0
            except Exception:
                obs_sign_local = 1.0
            return float(obs_sign_local)

        if bool(self.P1D_FIXED_FAMILY_PRIOR) and base in {
            "profile1d_linear",
            "profile1d_quadratic",
            "profile1d_cubic",
            "profile1d_consensus",
        }:
            prior = self._obs_cal_family_prior(base, apply_allowed_filter=False)
            if bool(prior.get("enabled", False)):
                obs_sign = _estimate_obs_sign()
                g_h1 = float(prior.get("g_h1", 1.0))
                g_h2 = float(prior.get("g_h2", 0.0 if not self.ENABLE_HARMONIC2 else 1.0))
                g_aux = float(prior.get("g_aux", 0.0))
                g_b = float(prior.get("g_b", g_aux)) if self.ENABLE_BASELINE else 0.0
                g_r = float(prior.get("g_r", g_aux)) if self.ENABLE_RESIDUAL else 0.0
                g_osc = float(0.5 * (g_h1 + g_h2) if self.ENABLE_HARMONIC2 else g_h1)
                mode = "family_phase_split_aux" if (abs(g_b - g_r) > 1e-9 or abs(g_aux) > 1e-9) else "family_phase_aux"
                out.update({
                    "enabled": False,
                    "mode": mode,
                    "obs_domain": str(prior.get("obs_domain", "displacement")),
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
                    "lag_sec": float(prior.get("max_lag_sec", 0.0)) * 0.0,
                    "fallback_mode": "p1d_family_prior_v1",
                })
                return out

        if not bool(self.OF_FIXED_VELOCITY_PRIOR):
            return out
        if (not base.startswith("of_")) or base == "of_disp_bridge":
            return out

        prior = self._obs_cal_family_prior(base, apply_allowed_filter=False)
        if not bool(prior.get("enabled", False)):
            return out

        obs_sign = _estimate_obs_sign()
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
        qx = float(qx) if np.isfinite(qx) else 0.0
        dt = float(dt) if np.isfinite(dt) and dt > 0.0 else (1.0 / self._REF_FPS)
        q_dyn = float(np.clip(q_dyn, 0.0, 1.0)) if np.isfinite(q_dyn) else 0.0
        q_osc = float(np.clip(q_osc, 0.0, 1.0)) if np.isfinite(q_osc) else 1.0
        obs_nonosc_need = (
            float(np.clip(obs_nonosc_need, 0.0, 1.0))
            if np.isfinite(obs_nonosc_need)
            else 0.0
        )
        residual_prior_scale = (
            float(np.clip(residual_prior_scale, 0.0, 1.0))
            if np.isfinite(residual_prior_scale)
            else 1.0
        )

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
            aper_drive = residual_prior_scale * max(1.0 - q_osc, 0.0)
            aper_obs_drive = residual_prior_scale * obs_nonosc_need
            aper_scale = (
                1.0
                + self.Q_APER_GAMMA * aper_drive
                + self.Q_APER_OBS_GAMMA * aper_obs_drive
            )
            Q[6, 6] = self.Q_RESIDUAL_POS * qx * aper_scale
            Q[7, 7] = self.Q_RESIDUAL_VEL * qx * aper_scale

        return np.nan_to_num(Q, nan=0.0, posinf=0.0, neginf=0.0)

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
        if (not bool(obs_cal.get("enabled", False))) and fallback_mode in {"of_velocity_prior_v1", "p1d_family_prior_v1"}:
            mode = str(obs_cal.get("mode", "family_phase_aux")).strip().lower()
            if mode == "family_phase_split_aux":
                if obs_domain == "velocity":
                    return self._build_H_family_velocity_split(
                        obs_cal.get("g_h1_signed", obs_cal.get("g_osc_signed", 1.0)),
                        obs_cal.get("g_h2_signed", obs_cal.get("g_osc_signed", 1.0)),
                        obs_cal.get("g_b_signed", obs_cal.get("g_aux_signed", 0.0)),
                        obs_cal.get("g_r_signed", obs_cal.get("g_aux_signed", 0.0)),
                        obs_cal.get("lag_sec", 0.0),
                        freq_hz,
                    )
                return self._build_H_family_visibility_split(
                    obs_cal.get("g_h1_signed", obs_cal.get("g_osc_signed", 1.0)),
                    obs_cal.get("g_h2_signed", obs_cal.get("g_osc_signed", 1.0)),
                    obs_cal.get("g_b_signed", obs_cal.get("g_aux_signed", 0.0)),
                    obs_cal.get("g_r_signed", obs_cal.get("g_aux_signed", 0.0)),
                    obs_cal.get("lag_sec", 0.0),
                    freq_hz,
                )
            if obs_domain == "velocity":
                return self._build_H_family_velocity(
                    obs_cal.get("g_h1_signed", obs_cal.get("g_osc_signed", 1.0)),
                    obs_cal.get("g_h2_signed", obs_cal.get("g_osc_signed", 1.0)),
                    obs_cal.get("g_aux_signed", obs_cal.get("g_b_signed", 0.0)),
                    obs_cal.get("lag_sec", 0.0),
                    freq_hz,
                )
            return self._build_H_family_visibility(
                obs_cal.get("g_h1_signed", obs_cal.get("g_osc_signed", 1.0)),
                obs_cal.get("g_h2_signed", obs_cal.get("g_osc_signed", 1.0)),
                obs_cal.get("g_aux_signed", obs_cal.get("g_b_signed", 0.0)),
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

    def _phase_anchored_morphology_readout(
        self,
        x: np.ndarray,
        z_full_additive: np.ndarray,
        *,
        morphology_confidence: np.ndarray,
        residual_gate: np.ndarray,
        has_target_evidence: bool,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Conservative phase-anchored morphology readout.

        The additive readout is still the fallback.  When enough target-side
        morphology evidence exists, we estimate a circular residual template
        over the fundamental phase and blend it into z_full.  This prevents the
        residual branch from becoming an unconstrained waveform fitter while
        preserving the interpretable h1/h2/b/r decomposition.
        """
        n = int(np.asarray(x).shape[0])
        z_full_additive = np.asarray(z_full_additive, dtype=np.float64).reshape(-1)
        if n < 12 or z_full_additive.size != n:
            return z_full_additive, {"enabled": False, "reason": "too_short"}
        if not bool(self.ENABLE_PHASE_ANCHORED_MORPHOLOGY):
            return z_full_additive, {"enabled": False, "reason": "disabled"}
        if not bool(has_target_evidence):
            return z_full_additive, {
                "enabled": False,
                "reason": "missing_target_observability",
            }

        phase = np.arctan2(x[:, self.HS1], x[:, self.HC1])
        phase = np.mod(phase, 2.0 * np.pi)
        if not np.all(np.isfinite(phase)):
            return z_full_additive, {"enabled": False, "reason": "bad_phase"}

        h1 = np.asarray(x[:, self.HC1], dtype=np.float64)
        baseline = np.asarray(x[:, self.B], dtype=np.float64) if self.ENABLE_BASELINE else np.zeros(n)
        h2 = np.asarray(x[:, self.HC2], dtype=np.float64) if self.ENABLE_HARMONIC2 else np.zeros(n)
        residual = np.asarray(x[:, self.R], dtype=np.float64) if self.ENABLE_RESIDUAL else np.zeros(n)
        gate = np.clip(
            np.nan_to_num(np.asarray(residual_gate, dtype=np.float64).reshape(-1)[:n], nan=0.0),
            0.0,
            1.0,
        )
        if gate.size != n:
            gate = np.ones(n, dtype=np.float64)
        morph = h2 + gate * residual

        conf = np.clip(
            np.nan_to_num(np.asarray(morphology_confidence, dtype=np.float64).reshape(-1)[:n], nan=0.0),
            0.0,
            1.0,
        )
        if conf.size != n:
            return z_full_additive, {"enabled": False, "reason": "bad_confidence"}
        coverage = float(np.mean(conf > 0.20))
        if coverage < 0.25:
            return z_full_additive, {"enabled": False, "reason": "low_morphology_coverage", "coverage": coverage}

        n_bins = int(np.clip(round(float(self.PHASE_MORPH_BINS)), 8, 64))
        bin_idx = np.floor(phase / (2.0 * np.pi) * n_bins).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        sums = np.zeros(n_bins, dtype=np.float64)
        weights = np.zeros(n_bins, dtype=np.float64)
        for idx in range(n):
            w = float(conf[idx])
            if w <= 0.0 or not np.isfinite(morph[idx]):
                continue
            bidx = int(bin_idx[idx])
            sums[bidx] += w * float(morph[idx])
            weights[bidx] += w
        valid = weights > 1e-9
        if np.count_nonzero(valid) < max(4, n_bins // 3):
            return z_full_additive, {
                "enabled": False,
                "reason": "insufficient_phase_bins",
                "valid_bin_fraction": float(np.mean(valid)),
            }
        template = np.zeros(n_bins, dtype=np.float64)
        template[valid] = sums[valid] / weights[valid]
        if not np.all(valid):
            centers = np.arange(n_bins, dtype=np.float64)
            valid_centers = centers[valid]
            valid_values = template[valid]
            wrap_centers = np.r_[valid_centers - n_bins, valid_centers, valid_centers + n_bins]
            wrap_values = np.r_[valid_values, valid_values, valid_values]
            template = np.interp(centers, wrap_centers, wrap_values)

        # Light circular smoothing keeps the readout phase-anchored rather than
        # bin-noise anchored.
        template = (
            0.25 * np.roll(template, 1)
            + 0.50 * template
            + 0.25 * np.roll(template, -1)
        )
        phase_pos = phase / (2.0 * np.pi) * n_bins
        centers = np.arange(n_bins, dtype=np.float64)
        wrap_centers = np.r_[centers - n_bins, centers, centers + n_bins]
        wrap_template = np.r_[template, template, template]
        morph_template = np.interp(phase_pos, wrap_centers, wrap_template)
        phase_full = h1 + baseline + morph_template
        blend = float(np.clip(self.PHASE_MORPH_MAX_BLEND, 0.0, 1.0)) * float(np.nanmean(conf))
        blend = float(np.clip(blend, 0.0, float(np.clip(self.PHASE_MORPH_MAX_BLEND, 0.0, 1.0))))
        out = (1.0 - blend) * z_full_additive + blend * phase_full
        return out.astype(np.float64), {
            "enabled": True,
            "blend": blend,
            "coverage": coverage,
            "valid_bin_fraction": float(np.mean(valid)),
            "bins": n_bins,
        }

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
        innovation = np.nan_to_num(innovation, nan=0.0, posinf=0.0, neginf=0.0)
        R_diag = np.nan_to_num(R_diag, nan=1.0, posinf=1e6, neginf=1.0)
        R_diag = np.maximum(R_diag, 1e-9)
        lambda_t = 1.0
        for _ in range(self.VB_ITERS):
            R_vb = np.diag(R_diag / max(lambda_t, 1e-12))
            S_vb = H @ P @ H.T + R_vb
            S_vb = 0.5 * (S_vb + S_vb.T)
            if not np.all(np.isfinite(S_vb)):
                S_vb = np.nan_to_num(S_vb, nan=0.0, posinf=1e6, neginf=-1e6)
                S_vb = 0.5 * (S_vb + S_vb.T) + 1e-6 * np.eye(m_obs, dtype=np.float64)
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
        if not np.all(np.isfinite(S_vb)):
            S_vb = np.nan_to_num(S_vb, nan=0.0, posinf=1e6, neginf=-1e6)
            S_vb = 0.5 * (S_vb + S_vb.T) + 1e-6 * np.eye(m_obs, dtype=np.float64)
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
            e_t = float(y[t] - float((H0 @ x_pred).item()))
            S = float((H0 @ P_pred @ H0.T).item()) + max(R_init, 1e-9)
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
            token.strip().lower()
            for token in str(self.OBS_CAL_ALLOWED_FAMILIES or "").split(",")
            if token.strip()
        }
        if allowed_families and base_method not in allowed_families:
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
        observation_rate_map = run_meta.pop("observation_rate_tracks_runtime", {}) or {}
        assistant_rate_map = run_meta.pop("assistant_rate_tracks_runtime", {}) or {}
        external_rate_anchor_raw = run_meta.pop("external_rate_anchor_runtime", None)
        external_rate_anchor_conf_raw = run_meta.pop("external_rate_anchor_confidence_runtime", None)
        external_output_rate_raw = run_meta.pop("external_output_rate_runtime", None)
        external_output_rate_conf_raw = run_meta.pop("external_output_rate_confidence_runtime", None)
        external_rate_posterior_raw = run_meta.pop("external_rate_posterior_runtime", None)
        external_state_role_prior_raw = run_meta.pop("external_state_role_prior_runtime", None)
        external_target_observability_raw = run_meta.pop("external_target_observability_runtime", None)
        enable_target_observability_control = bool(
            run_meta.pop(
                "enable_target_observability_control_runtime",
                bool(self.ENABLE_TARGET_OBSERVABILITY_CONTROL),
            )
        )
        enable_rate_source_arbiter_v1 = bool(run_meta.pop("enable_rate_source_arbiter_v1_runtime", False))
        enable_rate_source_arbiter_v2 = bool(run_meta.pop("enable_rate_source_arbiter_v2_runtime", False))
        enable_rate_source_arbiter_v3 = bool(run_meta.pop("enable_rate_source_arbiter_v3_runtime", False))
        rate_source_arbiter_version = "v3" if enable_rate_source_arbiter_v3 else ("v2" if enable_rate_source_arbiter_v2 else "v1")
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

        external_rate_tracks = None
        if multichannel and isinstance(observation_rate_map, dict):
            tracks = []
            for family in observation_families:
                raw_track = observation_rate_map.get(family)
                if raw_track is None:
                    tracks.append(np.full(n, np.nan, dtype=np.float64))
                    continue
                track = self._fit_track_length(np.asarray(raw_track, dtype=np.float64), n)
                track = np.where(
                    np.isfinite(track) & (track >= float(p.f_min)) & (track <= float(p.f_max)),
                    track,
                    np.nan,
                )
                tracks.append(track)
            if tracks:
                external_rate_tracks = np.vstack(tracks)

        external_rate_anchor = None
        external_rate_anchor_confidence = None
        if external_rate_anchor_raw is not None:
            try:
                anchor = self._fit_track_length(np.asarray(external_rate_anchor_raw, dtype=np.float64), n)
            except Exception:
                anchor = np.full(n, np.nan, dtype=np.float64)
            anchor = np.where(
                np.isfinite(anchor) & (anchor >= float(p.f_min)) & (anchor <= float(p.f_max)),
                anchor,
                np.nan,
            )
            if external_rate_anchor_conf_raw is None:
                confidence = np.where(np.isfinite(anchor), 1.0, 0.0)
            else:
                try:
                    confidence = self._fit_track_length(np.asarray(external_rate_anchor_conf_raw, dtype=np.float64), n)
                except Exception:
                    confidence = np.zeros(n, dtype=np.float64)
                confidence = np.clip(np.nan_to_num(confidence, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            confidence = np.where(np.isfinite(anchor), confidence, 0.0)
            if np.any(confidence > 0.0):
                external_rate_anchor = anchor.astype(np.float64)
                external_rate_anchor_confidence = confidence.astype(np.float64)

        external_rate_posterior = self._coerce_rate_posterior_runtime(external_rate_posterior_raw, n)
        target_observability_runtime = self._coerce_target_observability_runtime(
            external_target_observability_raw,
            n,
        )
        if not target_observability_runtime:
            enable_target_observability_control = False

        external_output_rate = None
        external_output_rate_confidence = None
        if external_output_rate_raw is not None:
            try:
                output_rate = self._fit_track_length(np.asarray(external_output_rate_raw, dtype=np.float64), n)
            except Exception:
                output_rate = np.full(n, np.nan, dtype=np.float64)
            output_rate = np.where(
                np.isfinite(output_rate)
                & (output_rate >= float(p.f_min))
                & (output_rate <= float(p.f_max)),
                output_rate,
                np.nan,
            )
            if external_output_rate_conf_raw is None:
                output_confidence = np.where(np.isfinite(output_rate), 1.0, 0.0)
            else:
                try:
                    output_confidence = self._fit_track_length(
                        np.asarray(external_output_rate_conf_raw, dtype=np.float64),
                        n,
                    )
                except Exception:
                    output_confidence = np.zeros(n, dtype=np.float64)
                output_confidence = np.clip(
                    np.nan_to_num(output_confidence, nan=0.0, posinf=1.0, neginf=0.0),
                    0.0,
                    1.0,
                )
            output_confidence = np.where(np.isfinite(output_rate), output_confidence, 0.0)
            if external_rate_anchor is not None and external_rate_anchor_confidence is not None:
                anchor_ok = (
                    np.isfinite(external_rate_anchor)
                    & np.isfinite(external_rate_anchor_confidence)
                    & (external_rate_anchor_confidence > 0.15)
                    & np.isfinite(output_rate)
                )
                if np.any(anchor_ok):
                    conflict = np.abs(output_rate - external_rate_anchor)
                    weak_conflict = anchor_ok & (conflict > 0.08) & (output_confidence < 0.45)
                    output_confidence[weak_conflict] *= 0.20
            if external_rate_posterior:
                posterior_mode = external_rate_posterior.get("mode_hz")
                posterior_conf = external_rate_posterior.get("confidence")
                posterior_entropy = external_rate_posterior.get("entropy", np.zeros(n, dtype=np.float64))
                posterior_gap = external_rate_posterior.get("top_gap", np.ones(n, dtype=np.float64))
                if posterior_mode is not None and posterior_conf is not None:
                    posterior_ok = (
                        np.isfinite(posterior_mode)
                        & np.isfinite(posterior_conf)
                        & (posterior_conf > 0.05)
                        & np.isfinite(output_rate)
                    )
                    if np.any(posterior_ok):
                        posterior_conf_eff = np.clip(
                            np.asarray(posterior_conf, dtype=np.float64)
                            * (1.0 - 0.50 * np.asarray(posterior_entropy, dtype=np.float64))
                            * (0.55 + 0.45 * np.asarray(posterior_gap, dtype=np.float64)),
                            0.0,
                            1.0,
                        )
                        conflict = np.abs(output_rate - posterior_mode)
                        ambiguous_conflict = (
                            posterior_ok
                            & (conflict > 0.08)
                            & (posterior_conf_eff < 0.35)
                        )
                        output_confidence[ambiguous_conflict] *= 0.35
            if np.any(output_confidence > 0.0):
                external_output_rate = output_rate.astype(np.float64)
                external_output_rate_confidence = output_confidence.astype(np.float64)

        assistant_raw_channels = []
        assistant_family_list = []
        assistant_rate_tracks = []
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
                if isinstance(assistant_rate_map, dict):
                    raw_track = assistant_rate_map.get(family)
                    if raw_track is not None:
                        assistant_rate_tracks.append(self._fit_track_length(raw_track, n))

        dt = 1.0 / fs
        D = self.STATE_DIM

        # ── Per-sample EMA constants ──
        ALPHA_R = float(np.exp(-dt / max(self.TAU_R_SEC, 1e-6)))
        ALPHA_MAD = float(np.exp(-dt / max(self.TAU_MAD_SEC, 1e-6)))
        ALPHA_KAPPA = float(np.exp(-dt / max(self.TAU_KAPPA_SEC, 1e-6)))
        ALPHA_AMP = float(np.exp(-dt / max(self.TAU_AMP_SEC, 1e-6)))
        ALPHA_MIX = float(np.exp(-dt / max(self.DYNAMIC_MIXTURE_TAU_SEC, 1e-6)))
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
            helper_init_by_group: Dict[str, List[float]] = {}
            for raw_i, family in zip(raw_channels, observation_families):
                y_init_helper_i = self._helper_preprocess(raw_i[:init_len], fs)
                if y_init_helper_i.size == 0:
                    continue
                fi_raw = self._coarse_freq(y_init_helper_i, fs)
                fi = self._harmonic_refine(fi_raw, y_init_helper_i, fs)
                if np.isfinite(fi):
                    helper_init_by_group.setdefault(self._family_group_key(family), []).append(float(fi))
            helper_init_tracks = [
                float(np.median(vals))
                for vals in helper_init_by_group.values()
                if vals
            ]
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
                helper_amp_consensus = self._family_balanced_nanmedian(
                    helper_amp,
                    observation_families,
                    fallback=1.0,
                )
                helper_freq_consensus = self._family_balanced_nanmedian(
                    helper_freq,
                    observation_families,
                    fallback=freq0,
                )
                helper_freq_consensus = np.clip(helper_freq_consensus, p.f_min, p.f_max)
                rate_evidence_freq = helper_freq.copy()
                if external_rate_tracks is not None and external_rate_tracks.shape == helper_freq.shape:
                    ext = np.asarray(external_rate_tracks, dtype=np.float64)
                    valid_ext = np.isfinite(ext) & (ext >= p.f_min) & (ext <= p.f_max)
                    if np.any(valid_ext):
                        rate_evidence_freq = np.where(valid_ext, ext, helper_freq)
                        helper_freq_consensus = self._family_balanced_nanmedian(
                            rate_evidence_freq,
                            observation_families,
                            fallback=freq0,
                        )
                        helper_freq_consensus = np.clip(helper_freq_consensus, p.f_min, p.f_max)
            else:
                y_helper, helper_amp, helper_freq = self._helper_features(signal_arr, fs, freq0)
                if y_helper.size != n:
                    y_helper = y.copy()
                    helper_amp = np.abs(y_helper).astype(np.float64)
                    helper_freq = np.full(n, freq0, dtype=np.float64)
                helper_amp_consensus = helper_amp.copy()
                helper_freq_consensus = helper_freq.copy()
                rate_evidence_freq = helper_freq.copy()
        else:
            if multichannel:
                y_helper = y.copy()
                helper_amp = np.abs(y_helper).astype(np.float64)
                helper_freq = np.full((y.shape[0], n), freq0, dtype=np.float64)
                helper_amp_consensus = np.nanmedian(helper_amp, axis=0)
                helper_freq_consensus = np.full(n, freq0, dtype=np.float64)
                rate_evidence_freq = helper_freq.copy()
                if external_rate_tracks is not None and external_rate_tracks.shape == helper_freq.shape:
                    ext = np.asarray(external_rate_tracks, dtype=np.float64)
                    valid_ext = np.isfinite(ext) & (ext >= p.f_min) & (ext <= p.f_max)
                    if np.any(valid_ext):
                        rate_evidence_freq = np.where(valid_ext, ext, helper_freq)
                        helper_freq_consensus = self._family_balanced_nanmedian(
                            rate_evidence_freq,
                            observation_families,
                            fallback=freq0,
                        )
                        helper_freq_consensus = np.clip(helper_freq_consensus, p.f_min, p.f_max)
            else:
                y_helper = y.copy()
                analytic = sps.hilbert(y_helper) if y_helper.size else np.array([], dtype=np.complex128)
                helper_amp = np.abs(analytic).astype(np.float64) if y_helper.size else np.array([], dtype=np.float64)
                helper_freq = np.full(n, freq0, dtype=np.float64)
                helper_amp_consensus = helper_amp.copy()
                helper_freq_consensus = helper_freq.copy()
                rate_evidence_freq = helper_freq.copy()

        external_rate_anchor_applied = False
        if external_rate_anchor is not None and external_rate_anchor_confidence is not None:
            valid_anchor = (
                np.isfinite(external_rate_anchor)
                & np.isfinite(external_rate_anchor_confidence)
                & (external_rate_anchor_confidence > 0.0)
            )
            if np.any(valid_anchor):
                anchor_w = np.clip(external_rate_anchor_confidence, 0.0, 1.0)
                blended_helper = np.asarray(helper_freq_consensus, dtype=np.float64).copy()
                blended_helper[valid_anchor] = (
                    (1.0 - anchor_w[valid_anchor]) * blended_helper[valid_anchor]
                    + anchor_w[valid_anchor] * external_rate_anchor[valid_anchor]
                )
                helper_freq_consensus = np.clip(blended_helper, p.f_min, p.f_max)
                external_rate_anchor_applied = True

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

        external_observation_prior = run_meta.get("external_observation_prior") if isinstance(run_meta, dict) else None
        external_observation_prior_weight = float(run_meta.get("external_observation_prior_weight", 0.0) or 0.0) if isinstance(run_meta, dict) else 0.0
        if (not multichannel) and isinstance(external_observation_prior, dict) and external_observation_prior_weight > 0.0:
            blend_w = float(np.clip(external_observation_prior_weight, 0.0, 1.0))
            obs_cal = dict(obs_cal)
            for key in ("g_osc_signed", "g_b_signed", "g_r_signed", "g_h1_signed", "g_h2_signed", "g_aux_signed"):
                if key in external_observation_prior:
                    base_val = float(obs_cal.get(key, 0.0))
                    prior_val = float(external_observation_prior.get(key, base_val))
                    obs_cal[key] = (1.0 - blend_w) * base_val + blend_w * prior_val
            if "lag_sec" in external_observation_prior:
                obs_cal["lag_sec"] = (1.0 - blend_w) * float(obs_cal.get("lag_sec", 0.0)) + blend_w * float(external_observation_prior.get("lag_sec", 0.0))
            obs_cal["external_prior_blend_weight"] = blend_w

        if multichannel:
            family_confidence = [
                self._family_confidence_policy(family, oc)
                for family, oc in zip(observation_families, obs_cal)
            ]
            channel_context_prior = np.asarray([
                self._channel_context_prior(family, oc)
                for family, oc in zip(observation_families, obs_cal)
            ], dtype=np.float64)
            if not np.any(np.isfinite(channel_context_prior)) or float(np.sum(channel_context_prior)) <= 1e-12:
                channel_context_prior = np.ones(len(observation_families), dtype=np.float64)
            channel_context_prior = np.clip(channel_context_prior, 1e-6, np.inf)
            external_context_prior_raw = run_meta.get("external_channel_context_prior")
            external_context_prior = None
            if external_context_prior_raw is not None:
                try:
                    external_context_prior = np.asarray(external_context_prior_raw, dtype=np.float64).reshape(-1)
                except Exception:
                    external_context_prior = None
            if external_context_prior is not None and external_context_prior.size == channel_context_prior.size:
                external_context_prior = np.where(
                    np.isfinite(external_context_prior) & (external_context_prior > 0.0),
                    external_context_prior,
                    np.nan,
                )
                finite_ext = external_context_prior[np.isfinite(external_context_prior)]
                if finite_ext.size and float(np.sum(finite_ext)) > 1e-12:
                    fill = float(np.nanmedian(finite_ext))
                    external_context_prior = np.where(np.isfinite(external_context_prior), external_context_prior, fill)
                    ext_mean = max(float(np.mean(external_context_prior)), 1e-12)
                    # Trial-level reliability graph acts as a prior multiplier;
                    # q_obs/R/lambda still determine frame-level posterior trust.
                    channel_context_prior = channel_context_prior * np.clip(external_context_prior / ext_mean, 1e-3, 1e3)
                    channel_context_prior = np.clip(channel_context_prior, 1e-6, np.inf)
            external_context_prior_runtime = None
            external_context_prior_runtime_raw = run_meta.get("external_channel_context_prior_runtime")
            if external_context_prior_runtime_raw is not None:
                try:
                    arr_runtime = np.asarray(external_context_prior_runtime_raw, dtype=np.float64)
                except Exception:
                    arr_runtime = np.array([], dtype=np.float64)
                if arr_runtime.ndim == 2 and arr_runtime.shape[0] == channel_context_prior.size and arr_runtime.shape[1] >= n:
                    arr_runtime = arr_runtime[:, :n]
                    arr_runtime = np.where(
                        np.isfinite(arr_runtime) & (arr_runtime > 0.0),
                        arr_runtime,
                        np.nan,
                    )
                    row_fill = np.nanmedian(arr_runtime, axis=1)
                    row_fill = np.where(np.isfinite(row_fill) & (row_fill > 0.0), row_fill, 1.0)
                    bad = ~np.isfinite(arr_runtime)
                    if np.any(bad):
                        arr_runtime = arr_runtime.copy()
                        rows, _cols = np.where(bad)
                        arr_runtime[bad] = row_fill[rows]
                    external_context_prior_runtime = np.clip(arr_runtime, 1e-6, np.inf)
        else:
            family_confidence = self._family_confidence_policy(base_method, obs_cal)
            channel_context_prior = None

        # ── Matrices ──
        if multichannel:
            H = np.vstack([self._build_H_from_obs_cal(oc, freq0) for oc in obs_cal])
        else:
            H = self._build_H_from_obs_cal(obs_cal, freq0)
        I_D = np.eye(D, dtype=np.float64)
        state_role_prior_runtime = self._coerce_state_role_prior_runtime(
            external_state_role_prior_raw,
            n_obs=int(y.shape[0]) if multichannel else 1,
            n=n,
        )

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
        diag_residual_gate = np.ones(n, dtype=np.float64)
        diag_residual_failure_absorption = np.zeros(n, dtype=np.float64)
        diag_aper_drive = np.zeros(n, dtype=np.float64)
        diag_helper_support = np.ones(n, dtype=np.float64)
        diag_helper_trust = np.ones(n, dtype=np.float64)
        diag_helper_bias_conf = np.zeros(n, dtype=np.float64)
        diag_helper_freq = helper_freq_consensus.copy()
        diag_helper_mismatch = np.full(n, np.nan, dtype=np.float64)
        diag_external_rate_anchor = (
            np.asarray(external_rate_anchor, dtype=np.float64).copy()
            if external_rate_anchor is not None
            else np.full(n, np.nan, dtype=np.float64)
        )
        diag_external_rate_anchor_confidence = (
            np.asarray(external_rate_anchor_confidence, dtype=np.float64).copy()
            if external_rate_anchor_confidence is not None
            else np.zeros(n, dtype=np.float64)
        )
        diag_external_output_rate = (
            np.asarray(external_output_rate, dtype=np.float64).copy()
            if external_output_rate is not None
            else np.full(n, np.nan, dtype=np.float64)
        )
        diag_external_output_rate_confidence = (
            np.asarray(external_output_rate_confidence, dtype=np.float64).copy()
            if external_output_rate_confidence is not None
            else np.zeros(n, dtype=np.float64)
        )
        diag_external_rate_posterior_mode = (
            np.asarray(external_rate_posterior.get("mode_hz"), dtype=np.float64).copy()
            if external_rate_posterior and external_rate_posterior.get("mode_hz") is not None
            else np.full(n, np.nan, dtype=np.float64)
        )
        diag_external_rate_posterior_mean = (
            np.asarray(external_rate_posterior.get("mean_hz"), dtype=np.float64).copy()
            if external_rate_posterior and external_rate_posterior.get("mean_hz") is not None
            else np.full(n, np.nan, dtype=np.float64)
        )
        diag_external_rate_posterior_confidence = (
            np.asarray(external_rate_posterior.get("confidence"), dtype=np.float64).copy()
            if external_rate_posterior and external_rate_posterior.get("confidence") is not None
            else np.zeros(n, dtype=np.float64)
        )
        diag_external_rate_posterior_entropy = (
            np.asarray(external_rate_posterior.get("entropy"), dtype=np.float64).copy()
            if external_rate_posterior and external_rate_posterior.get("entropy") is not None
            else np.zeros(n, dtype=np.float64)
        )
        diag_external_rate_posterior_top_gap = (
            np.asarray(external_rate_posterior.get("top_gap"), dtype=np.float64).copy()
            if external_rate_posterior and external_rate_posterior.get("top_gap") is not None
            else np.zeros(n, dtype=np.float64)
        )
        diag_target_observability = (
            np.asarray(target_observability_runtime.get("target_observability"), dtype=np.float64).copy()
            if enable_target_observability_control
            else np.ones(n, dtype=np.float64)
        )
        diag_target_h1_timing = (
            np.asarray(target_observability_runtime.get("h1_timing"), dtype=np.float64).copy()
            if enable_target_observability_control
            else np.ones(n, dtype=np.float64)
        )
        diag_target_h2_morphology = (
            np.asarray(target_observability_runtime.get("h2_morphology"), dtype=np.float64).copy()
            if enable_target_observability_control
            else np.ones(n, dtype=np.float64)
        )
        diag_target_baseline = (
            np.asarray(target_observability_runtime.get("baseline"), dtype=np.float64).copy()
            if enable_target_observability_control
            else np.ones(n, dtype=np.float64)
        )
        diag_target_residual = (
            np.asarray(target_observability_runtime.get("residual"), dtype=np.float64).copy()
            if enable_target_observability_control
            else np.ones(n, dtype=np.float64)
        )
        diag_target_nuisance = (
            np.asarray(target_observability_runtime.get("nuisance"), dtype=np.float64).copy()
            if enable_target_observability_control
            else np.zeros(n, dtype=np.float64)
        )
        diag_target_source_spread_hz = (
            np.asarray(target_observability_runtime.get("source_spread_hz"), dtype=np.float64).copy()
            if enable_target_observability_control
            else np.zeros(n, dtype=np.float64)
        )
        diag_target_posterior_specificity = (
            np.asarray(target_observability_runtime.get("posterior_specificity"), dtype=np.float64).copy()
            if enable_target_observability_control
            else np.zeros(n, dtype=np.float64)
        )
        diag_target_alias_safety = (
            np.asarray(target_observability_runtime.get("alias_safety"), dtype=np.float64).copy()
            if enable_target_observability_control
            else np.ones(n, dtype=np.float64)
        )
        diag_external_rate_posterior_blend = np.zeros(n, dtype=np.float64)
        diag_freq_rescue = np.zeros(n, dtype=np.float64)
        diag_output_rate_blend = np.zeros(n, dtype=np.float64)
        diag_external_output_rate_blend = np.zeros(n, dtype=np.float64)
        diag_q_dyn_raw = np.zeros(n, dtype=np.float64)
        diag_prior_collapse = np.zeros(n, dtype=np.float64)
        diag_mixture_entropy = np.zeros(n, dtype=np.float64)
        if multichannel:
            n_obs = y.shape[0]
            diag_R_channels = np.zeros((n_obs, n), dtype=np.float64)
            diag_R_eff_channels = np.zeros((n_obs, n), dtype=np.float64)
            diag_q_obs_channels = np.ones((n_obs, n), dtype=np.float64)
            diag_pi_channels = np.ones((n_obs, n), dtype=np.float64)
            diag_mixture_channels = np.full((n_obs, n), 1.0 / max(1, n_obs), dtype=np.float64)
            diag_rate_observability_channels = np.ones((n_obs, n), dtype=np.float64)
            diag_rate_observability_score = np.zeros(n, dtype=np.float64)
            diag_rate_helper_freq = helper_freq_consensus.copy()
            diag_context_channels = np.repeat(
                np.asarray(channel_context_prior, dtype=np.float64).reshape(-1, 1),
                n,
                axis=1,
            )
            diag_state_role_context_channels = np.ones((n_obs, n), dtype=np.float64)
            diag_state_role_abstain_channels = np.zeros((n_obs, n), dtype=np.float64)
            diag_state_role_h1_channels = np.ones((n_obs, n), dtype=np.float64)
            diag_state_role_zosc_channels = np.ones((n_obs, n), dtype=np.float64)
            diag_group_balance_scale_channels = np.ones((n_obs, n), dtype=np.float64)
        else:
            n_obs = 1
            diag_R_channels = None
            diag_R_eff_channels = None
            diag_q_obs_channels = None
            diag_pi_channels = None
            diag_mixture_channels = None
            diag_rate_observability_channels = None
            diag_rate_observability_score = None
            diag_rate_helper_freq = None
            diag_context_channels = None
            diag_state_role_context_channels = None
            diag_state_role_abstain_channels = None
            diag_state_role_h1_channels = None
            diag_state_role_zosc_channels = None
            diag_group_balance_scale_channels = None

        group_balance_scale = np.ones(n_obs, dtype=np.float64)
        if multichannel and bool(self.ENABLE_GROUP_BALANCED_FUSION):
            groups = self._family_group_indices(observation_families, n_obs)
            for idxs in groups.values():
                scale = float(max(len(idxs), 1))
                for idx in idxs:
                    group_balance_scale[int(idx)] = scale

        # ── Init state ──
        x = np.zeros(D, dtype=np.float64)
        Q0 = self._build_Q_disentangled(qx, dt, q_dyn=0.0, q_osc=1.0)
        P = Q0.copy()
        external_initial_state = run_meta.get("external_initial_state") if isinstance(run_meta, dict) else None
        external_initial_state_weight = float(run_meta.get("external_initial_state_weight", 0.0) or 0.0) if isinstance(run_meta, dict) else 0.0
        if (not multichannel) and external_initial_state is not None and external_initial_state_weight > 0.0:
            try:
                x_ext = np.asarray(external_initial_state, dtype=np.float64).reshape(-1)
            except Exception:
                x_ext = np.array([], dtype=np.float64)
            if x_ext.size == D:
                init_w = float(np.clip(external_initial_state_weight, 0.0, 1.0))
                x = (1.0 - init_w) * x + init_w * x_ext
                P = P / max(1.0 + 2.0 * init_w, 1e-6)

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
            mixture_t = np.asarray(channel_context_prior, dtype=np.float64).copy()
            mixture_t = mixture_t / max(float(np.sum(mixture_t)), 1e-12)
        else:
            signal_scale = max(float(np.std(y[:min(n, int(3.0 * fs))])), 1e-6)
            y_prev = y[0] if n > 0 else 0.0
            mixture_t = None

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
            rate_obs_vec = np.ones(n_obs, dtype=np.float64) if multichannel else None
            rate_obs_score_t = 0.0
            if (
                multichannel
                and bool(self.ENABLE_DYNAMIC_MIXTURE)
                and bool(self.ENABLE_RATE_OBSERVABILITY_MIXTURE)
            ):
                rate_win = max(3, int(round(float(self.RATE_OBS_WINDOW_SEC) * fs)))
                rate_start = max(0, t - rate_win + 1)
                rate_obs_vec, rate_helper_freq_t, rate_obs_score_t = self._rate_observability_scores(
                    rate_evidence_freq[:, rate_start:t + 1],
                    observation_families,
                    fallback_freq=helper_freq_t,
                    anchor_freq=freq_t,
                )
                if bool(self.ENABLE_RATE_OBSERVABILITY_HELPER) and np.isfinite(rate_helper_freq_t):
                    prev_rate_helper = (
                        float(diag_rate_helper_freq[t - 1])
                        if diag_rate_helper_freq is not None and t > 0 and np.isfinite(diag_rate_helper_freq[t - 1])
                        else helper_freq_t
                    )
                    step_cap = max(float(self.RATE_OBS_HELPER_MAX_STEP_HZ), 1e-6)
                    bounded_target = prev_rate_helper + float(np.clip(
                        rate_helper_freq_t - prev_rate_helper,
                        -step_cap,
                        step_cap,
                    ))
                    blend = float(np.clip(self.RATE_OBS_HELPER_BLEND, 0.0, 1.0))
                    helper_freq_t = float(np.clip(
                        (1.0 - blend) * helper_freq_t + blend * bounded_target,
                        p.f_min,
                        p.f_max,
                    ))
            if external_rate_anchor is not None and external_rate_anchor_confidence is not None:
                anchor_t = float(external_rate_anchor[t]) if t < external_rate_anchor.size else float("nan")
                anchor_conf_t = float(external_rate_anchor_confidence[t]) if t < external_rate_anchor_confidence.size else 0.0
                if np.isfinite(anchor_t) and anchor_conf_t > 0.0:
                    anchor_conf_t = float(np.clip(anchor_conf_t, 0.0, 1.0))
                    helper_freq_t = float(np.clip(
                        (1.0 - anchor_conf_t) * helper_freq_t + anchor_conf_t * anchor_t,
                        p.f_min,
                        p.f_max,
                    ))
            if external_rate_posterior:
                posterior_mean_t = (
                    float(diag_external_rate_posterior_mean[t])
                    if t < diag_external_rate_posterior_mean.size
                    else float("nan")
                )
                posterior_mode_t = (
                    float(diag_external_rate_posterior_mode[t])
                    if t < diag_external_rate_posterior_mode.size
                    else float("nan")
                )
                posterior_conf_t = (
                    float(diag_external_rate_posterior_confidence[t])
                    if t < diag_external_rate_posterior_confidence.size
                    else 0.0
                )
                posterior_entropy_t = (
                    float(diag_external_rate_posterior_entropy[t])
                    if t < diag_external_rate_posterior_entropy.size
                    else 0.0
                )
                posterior_gap_t = (
                    float(diag_external_rate_posterior_top_gap[t])
                    if t < diag_external_rate_posterior_top_gap.size
                    else 1.0
                )
                posterior_target_t = posterior_mean_t if np.isfinite(posterior_mean_t) else posterior_mode_t
                if np.isfinite(posterior_target_t) and posterior_conf_t > 0.0:
                    posterior_conf_eff = float(np.clip(
                        posterior_conf_t
                        * (1.0 - 0.50 * float(np.clip(posterior_entropy_t, 0.0, 1.0)))
                        * (0.55 + 0.45 * float(np.clip(posterior_gap_t, 0.0, 1.0))),
                        0.0,
                        1.0,
                    ))
                    # The posterior mean is target-computable rate evidence.
                    # Use it as a bounded frequency cue for the oscillator, not
                    # merely as a final-output override. Confidence/entropy/gap
                    # keep this an adaptive observation-law input rather than a
                    # hard target-side shortcut.
                    posterior_blend = float(np.clip(0.65 * posterior_conf_eff, 0.0, 0.55))
                    if posterior_blend > 0.0:
                        helper_freq_t = float(np.clip(
                            (1.0 - posterior_blend) * helper_freq_t + posterior_blend * posterior_target_t,
                            p.f_min,
                            p.f_max,
                        ))
                        diag_external_rate_posterior_blend[t] = posterior_blend
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
                conf_qdyn_values = [
                    float(fc.get("qdyn_scale", 1.0))
                    for fc in family_confidence
                    if bool(fc.get("enabled", False))
                ]
                conf_qdyn_scale = float(np.mean(conf_qdyn_values)) if conf_qdyn_values else 1.0
            else:
                conf_qdyn_scale = float(family_confidence.get("qdyn_scale", 1.0)) if bool(family_confidence.get("enabled", False)) else 1.0
            q_dyn_t = float(np.clip(q_dyn_t * conf_qdyn_scale, 0.0, 1.0))
            if not np.isfinite(q_dyn_t):
                q_dyn_t = 0.0
            if self._helper_trust_scales_qdyn(base_method):
                qdyn_floor = float(np.clip(self.HELPER_TRUST_QDYN_FLOOR, 0.0, 1.0))
                qdyn_scale = qdyn_floor + (1.0 - qdyn_floor) * float(np.clip(helper_trust_t, 0.0, 1.0))
                q_dyn_t = float(np.clip(q_dyn_raw_t * qdyn_scale, 0.0, 1.0))
                if not np.isfinite(q_dyn_t):
                    q_dyn_t = 0.0
            target_obs_t = 1.0
            target_h1_t = 1.0
            target_h2_t = 1.0
            target_baseline_t = 1.0
            target_residual_t = 1.0
            target_nuisance_t = 0.0
            if enable_target_observability_control:
                target_obs_t = float(np.clip(diag_target_observability[t], 0.0, 1.0))
                target_h1_t = float(np.clip(diag_target_h1_timing[t], 0.0, 1.0))
                target_h2_t = float(np.clip(diag_target_h2_morphology[t], 0.0, 1.0))
                target_baseline_t = float(np.clip(diag_target_baseline[t], 0.0, 1.0))
                target_residual_t = float(np.clip(diag_target_residual[t], 0.0, 1.0))
                target_nuisance_t = float(np.clip(diag_target_nuisance[t], 0.0, 1.0))
                trust_floor = float(np.clip(self.TARGET_OBS_TRUST_FLOOR, 0.0, 1.0))
                qdyn_floor = float(np.clip(self.TARGET_OBS_QDYN_FLOOR, 0.0, 1.0))
                target_trust = trust_floor + (1.0 - trust_floor) * target_obs_t
                if multichannel:
                    q_obs_vec = np.clip(q_obs_vec * target_trust, float(self.Q_OBS_MIN), 1.0)
                    q_obs_t = float(np.mean(q_obs_vec))
                else:
                    q_obs_t = float(np.clip(q_obs_t * target_trust, float(self.Q_OBS_MIN), 1.0))
                q_dyn_t = float(np.clip(q_dyn_t * (qdyn_floor + (1.0 - qdyn_floor) * target_h1_t), 0.0, 1.0))
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
            residual_gate_t = 1.0
            if bool(self.ENABLE_RESIDUAL_SEMANTICS) and bool(self.ENABLE_RESIDUAL_IDENTIFIABILITY_GUARD):
                trust_floor = float(np.clip(self.RESIDUAL_GUARD_TRUST_FLOOR, 0.0, 1.0))
                qobs_gate = trust_floor + (1.0 - trust_floor) * float(np.clip(q_obs_t, 0.0, 1.0))
                if enable_target_observability_control:
                    residual_role_gate = trust_floor + (1.0 - trust_floor) * target_residual_t
                    nuisance_gate = 1.0 - float(np.clip(self.RESIDUAL_GUARD_NUISANCE_SCALE, 0.0, 1.0)) * target_nuisance_t
                    residual_gate_t = qobs_gate * residual_role_gate * float(np.clip(nuisance_gate, 0.0, 1.0))
                else:
                    residual_gate_t = qobs_gate
                residual_gate_t = float(np.clip(residual_gate_t, 0.0, 1.0))
            residual_prior_eff_t = float(np.clip(residual_prior_t * residual_gate_t, 0.0, 1.0))
            obs_nonosc_need_eff = float(np.clip(residual_prior_eff_t * obs_nonosc_need, 0.0, 1.0))
            aper_drive_t = float(np.clip(residual_prior_eff_t * max(1.0 - q_osc_t, 0.0), 0.0, 1.0))
            residual_failure_absorption_t = float(np.clip(
                max(1.0 - q_obs_t, 0.0) * max(1.0 - target_obs_t, 0.0) * max(obs_nonosc_need, 1.0 - q_osc_t),
                0.0,
                1.0,
            ))
            helper_amp_baseline = ALPHA_AMP * helper_amp_baseline + (1.0 - ALPHA_AMP) * helper_amp_t

            # ── Prior trust π_t = f(q_obs_t) ──
            # Conservative: π_t = q_obs_t. Low q_obs inflates R_eff,
            # reducing Kalman gain for unreliable observations.
            state_role_abstain_t = None
            if multichannel:
                if bool(self.ENABLE_DYNAMIC_MIXTURE):
                    state_role_context_t = np.ones(n_obs, dtype=np.float64)
                    state_role_values_t: Dict[str, np.ndarray] = {}
                    if state_role_prior_runtime:
                        state_role_context_t, state_role_values_t = self._state_role_context_from_H(
                            H,
                            state_role_prior_runtime,
                            t,
                        )
                    context = np.asarray(channel_context_prior, dtype=np.float64)
                    if (
                        "external_context_prior_runtime" in locals()
                        and external_context_prior_runtime is not None
                        and np.asarray(external_context_prior_runtime).shape == (n_obs, n)
                    ):
                        context_runtime_t = np.asarray(external_context_prior_runtime[:, t], dtype=np.float64)
                        runtime_mean = max(float(np.nanmean(context_runtime_t)), 1e-12)
                        context = context * np.clip(context_runtime_t / runtime_mean, 1e-3, 1e3)
                    if enable_target_observability_control:
                        role_scores = np.ones(n_obs, dtype=np.float64)
                        for idx in range(n_obs):
                            visibility = self._state_role_visibility_from_H(H[idx])
                            role_scores[idx] = (
                                float(visibility.get("h1", 0.0)) * target_h1_t
                                + float(visibility.get("h2", 0.0)) * target_h2_t
                                + float(visibility.get("b", 0.0)) * target_baseline_t
                                + float(visibility.get("r", 0.0)) * target_residual_t
                            )
                        role_scores = np.clip(
                            np.nan_to_num(role_scores, nan=target_obs_t, posinf=1.0, neginf=0.0),
                            0.0,
                            1.0,
                        )
                        role_floor = float(np.clip(self.TARGET_OBS_TRUST_FLOOR, 0.0, 1.0))
                        role_power = float(np.clip(self.TARGET_OBS_ROLE_POWER, 0.0, 2.0))
                        role_multiplier = role_floor + (1.0 - role_floor) * np.power(role_scores, role_power)
                        context = context * role_multiplier
                    context = np.clip(np.nan_to_num(context, nan=1.0, posinf=1.0, neginf=1e-6), 1e-6, np.inf)
                    if state_role_prior_runtime:
                        role_mean = max(float(np.nanmean(state_role_context_t)), 1e-12)
                        role_multiplier = np.power(
                            np.clip(state_role_context_t / role_mean, 1e-3, 1e3),
                            float(np.clip(self.STATE_ROLE_CONTEXT_POWER, 0.0, 2.0)),
                        )
                        role_multiplier = np.maximum(
                            role_multiplier,
                            float(np.clip(self.STATE_ROLE_CONTEXT_MULTIPLIER_FLOOR, 0.0, 1.0)),
                        )
                        context = context * role_multiplier
                    q_fast = np.clip(np.asarray(q_obs_vec, dtype=np.float64), float(self.Q_OBS_MIN), 1.0)
                    r_arr = np.maximum(np.asarray(R_t, dtype=np.float64), 1e-12)
                    r_med = float(np.nanmedian(r_arr[np.isfinite(r_arr)])) if np.any(np.isfinite(r_arr)) else float(R_init)
                    r_score = np.power(
                        np.clip(r_med / np.maximum(r_arr, 1e-12), 0.10, 10.0),
                        float(np.clip(self.DYNAMIC_MIXTURE_R_WEIGHT, 0.0, 2.0)),
                    )
                    rate_score = (
                        np.asarray(rate_obs_vec, dtype=np.float64)
                        if rate_obs_vec is not None and np.asarray(rate_obs_vec).size == n_obs
                        else np.ones(n_obs, dtype=np.float64)
                    )
                    rate_score = np.clip(np.nan_to_num(rate_score, nan=1.0, posinf=1.0, neginf=0.0), 1e-6, 1.0)
                    if enable_target_observability_control:
                        h1_floor = float(np.clip(self.TARGET_OBS_TRUST_FLOOR, 0.0, 1.0))
                        rate_score = rate_score * (h1_floor + (1.0 - h1_floor) * target_h1_t)
                    if state_role_prior_runtime:
                        rate_role = state_role_values_t.get("z_osc")
                        if rate_role is None:
                            rate_role = state_role_values_t.get("h1")
                        if rate_role is not None and np.asarray(rate_role).size == n_obs:
                            rate_multiplier = np.power(
                                np.clip(np.asarray(rate_role, dtype=np.float64), 1e-6, 1.0),
                                float(np.clip(self.STATE_ROLE_RATE_POWER, 0.0, 2.0)),
                            )
                            rate_multiplier = np.maximum(
                                rate_multiplier,
                                float(np.clip(self.STATE_ROLE_RATE_MULTIPLIER_FLOOR, 0.0, 1.0)),
                            )
                            rate_score = rate_score * rate_multiplier
                    rate_power = float(np.clip(self.RATE_OBS_POWER, 0.0, 4.0))
                    evidence = np.clip(context * q_fast * r_score * np.power(rate_score, rate_power), 1e-9, np.inf)
                    temp = max(float(self.DYNAMIC_MIXTURE_TEMPERATURE), 1e-3)
                    target_mix = self._stable_softmax(np.log(evidence) / temp)
                    mixture_t = ALPHA_MIX * np.asarray(mixture_t, dtype=np.float64) + (1.0 - ALPHA_MIX) * target_mix
                    min_w = float(np.clip(self.DYNAMIC_MIXTURE_MIN_WEIGHT, 0.0, 0.49 / max(1, n_obs)))
                    if min_w > 0.0:
                        mixture_t = (1.0 - min_w * n_obs) * mixture_t + min_w
                    mixture_sum = max(float(np.sum(mixture_t)), 1e-12)
                    mixture_t = mixture_t / mixture_sum
                    global_quality = float(np.clip(
                        np.nanmean(q_fast) * np.nanmean(np.clip(context, 0.0, 1.0)),
                        float(self.DYNAMIC_MIXTURE_GLOBAL_QUALITY_FLOOR),
                        1.0,
                    ))
                    if enable_target_observability_control:
                        target_quality_floor = float(np.clip(self.TARGET_OBS_TRUST_FLOOR, 0.0, 1.0))
                        global_quality *= target_quality_floor + (1.0 - target_quality_floor) * target_obs_t
                        global_quality = float(np.clip(
                            global_quality,
                            float(self.DYNAMIC_MIXTURE_GLOBAL_QUALITY_FLOOR),
                            1.0,
                        ))
                    # mixture_t is a normalized allocation over observation
                    # equations.  It is not itself an absolute trust
                    # probability: with eight equally plausible channels,
                    # mixture_t is 1/8 even when all channels are reliable.
                    # Rescale by n_obs before using it as the R_eff trust
                    # gate; otherwise multichannel PARH artificially inflates
                    # every observation variance and collapses toward the
                    # process prior.
                    mixture_trust = np.clip(mixture_t * float(n_obs), 1e-6, 1.0)
                    pi_t = np.clip(mixture_trust * global_quality, 1e-6, 1.0)
                    if diag_context_channels is not None:
                        diag_context_channels[:, t] = context
                    if diag_state_role_context_channels is not None:
                        diag_state_role_context_channels[:, t] = state_role_context_t
                    if diag_state_role_abstain_channels is not None:
                        state_role_abstain_t = state_role_values_t.get(
                            "abstain",
                            np.zeros(n_obs, dtype=np.float64),
                        )
                        diag_state_role_abstain_channels[:, t] = state_role_values_t.get(
                            "abstain",
                            np.zeros(n_obs, dtype=np.float64),
                        )
                    if diag_state_role_h1_channels is not None:
                        diag_state_role_h1_channels[:, t] = state_role_values_t.get(
                            "h1",
                            np.ones(n_obs, dtype=np.float64),
                        )
                    if diag_state_role_zosc_channels is not None:
                        diag_state_role_zosc_channels[:, t] = state_role_values_t.get(
                            "z_osc",
                            np.ones(n_obs, dtype=np.float64),
                        )
                else:
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
                        residual_prior_scale=residual_prior_eff_t,
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
                if bool(self.ENABLE_GROUP_BALANCED_FUSION):
                    R_eff = R_eff * group_balance_scale
                if state_role_abstain_t is not None and np.asarray(state_role_abstain_t).size == R_eff.size:
                    R_eff = R_eff * (
                        1.0
                        + float(np.clip(self.STATE_ROLE_ABSTAIN_R_SCALE, 0.0, 10.0))
                        * np.clip(np.asarray(state_role_abstain_t, dtype=np.float64), 0.0, 1.0)
                    )
                if family_confidence:
                    r_scales = np.array([
                        float(fc.get("r_scale", 1.0)) if bool(fc.get("enabled", False)) else 1.0
                        for fc in family_confidence
                    ], dtype=np.float64)
                    if r_scales.size == R_eff.size:
                        R_eff = R_eff * r_scales
                if enable_target_observability_control:
                    R_eff = R_eff * (
                        1.0
                        + float(np.clip(self.TARGET_OBS_NUISANCE_R_SCALE, 0.0, 10.0))
                        * target_nuisance_t
                    )
                S_eff = H @ P_pred @ H.T + np.diag(R_eff)
                S_eff = 0.5 * (S_eff + S_eff.T)
                if not np.all(np.isfinite(S_eff)):
                    finite_r = R_eff[np.isfinite(R_eff)]
                    diag_floor = float(np.median(finite_r)) if finite_r.size else float(R_init)
                    diag_floor = max(diag_floor, 1e-6)
                    S_eff = np.nan_to_num(S_eff, nan=0.0, posinf=1e6, neginf=-1e6)
                    S_eff = 0.5 * (S_eff + S_eff.T) + diag_floor * np.eye(n_obs, dtype=np.float64)
            else:
                R_eff = R_t / max(pi_t, 1e-6)
                if bool(family_confidence.get("enabled", False)):
                    R_eff = float(R_eff) * float(family_confidence.get("r_scale", 1.0))
                if enable_target_observability_control:
                    R_eff = float(R_eff) * (
                        1.0
                        + float(np.clip(self.TARGET_OBS_NUISANCE_R_SCALE, 0.0, 10.0))
                        * target_nuisance_t
                    )
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
                    if external_rate_anchor is not None and external_rate_anchor_confidence is not None:
                        anchor_win = np.asarray(external_rate_anchor[helper_start:t + 1], dtype=np.float64)
                        anchor_conf_win = np.asarray(external_rate_anchor_confidence[helper_start:t + 1], dtype=np.float64)
                        anchor_ok = (
                            np.isfinite(anchor_win)
                            & np.isfinite(anchor_conf_win)
                            & (anchor_conf_win > 0.0)
                        )
                        if np.any(anchor_ok):
                            anchor_med = self._weighted_nanmedian(
                                anchor_win[anchor_ok],
                                anchor_conf_win[anchor_ok],
                                fallback=helper_med,
                            )
                            anchor_conf_med = float(np.clip(np.nanmedian(anchor_conf_win[anchor_ok]), 0.0, 1.0))
                            if np.isfinite(anchor_med) and anchor_conf_med > 0.0:
                                helper_med = float(np.clip(
                                    (1.0 - anchor_conf_med) * helper_med + anchor_conf_med * anchor_med,
                                    p.f_min,
                                    p.f_max,
                                ))
                                f_spec = float(np.clip(
                                    (1.0 - anchor_conf_med) * f_spec + anchor_conf_med * anchor_med,
                                    p.f_min,
                                    p.f_max,
                                ))
                    if external_rate_posterior:
                        posterior_mean_win = np.asarray(
                            diag_external_rate_posterior_mean[helper_start:t + 1],
                            dtype=np.float64,
                        )
                        posterior_mode_win = np.asarray(
                            diag_external_rate_posterior_mode[helper_start:t + 1],
                            dtype=np.float64,
                        )
                        posterior_target_win = np.where(
                            np.isfinite(posterior_mean_win),
                            posterior_mean_win,
                            posterior_mode_win,
                        )
                        posterior_conf_win = np.asarray(
                            diag_external_rate_posterior_confidence[helper_start:t + 1],
                            dtype=np.float64,
                        )
                        posterior_entropy_win = np.asarray(
                            diag_external_rate_posterior_entropy[helper_start:t + 1],
                            dtype=np.float64,
                        )
                        posterior_gap_win = np.asarray(
                            diag_external_rate_posterior_top_gap[helper_start:t + 1],
                            dtype=np.float64,
                        )
                        posterior_conf_eff_win = np.clip(
                            posterior_conf_win
                            * (1.0 - 0.50 * np.clip(posterior_entropy_win, 0.0, 1.0))
                            * (0.55 + 0.45 * np.clip(posterior_gap_win, 0.0, 1.0)),
                            0.0,
                            1.0,
                        )
                        posterior_ok = (
                            np.isfinite(posterior_target_win)
                            & np.isfinite(posterior_conf_eff_win)
                            & (posterior_conf_eff_win > 0.0)
                        )
                        if np.any(posterior_ok):
                            posterior_med = self._weighted_nanmedian(
                                posterior_target_win[posterior_ok],
                                posterior_conf_eff_win[posterior_ok],
                                fallback=helper_med,
                            )
                            posterior_conf_med = float(np.clip(
                                np.nanmedian(posterior_conf_eff_win[posterior_ok]),
                                0.0,
                                1.0,
                            ))
                            if np.isfinite(posterior_med) and posterior_conf_med > 0.0:
                                posterior_update_blend = float(np.clip(
                                    0.75 * posterior_conf_med,
                                    0.0,
                                    0.65,
                                ))
                                helper_med = float(np.clip(
                                    (1.0 - posterior_update_blend) * helper_med
                                    + posterior_update_blend * posterior_med,
                                    p.f_min,
                                    p.f_max,
                                ))
                                f_spec = float(np.clip(
                                    (1.0 - posterior_update_blend) * f_spec
                                    + posterior_update_blend * posterior_med,
                                    p.f_min,
                                    p.f_max,
                                ))
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
            diag_prior_collapse[t] = float(diag_pi[t] < 0.20)
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
            diag_residual_gate[t] = residual_gate_t
            diag_residual_failure_absorption[t] = residual_failure_absorption_t
            diag_aper_drive[t] = aper_drive_t
            diag_helper_support[t] = helper_support
            diag_helper_trust[t] = helper_trust_t
            diag_helper_bias_conf[t] = helper_bias_conf_t
            diag_helper_freq[t] = helper_freq_t
            if multichannel and diag_rate_helper_freq is not None:
                diag_rate_helper_freq[t] = helper_freq_t
            if multichannel and diag_rate_observability_score is not None:
                diag_rate_observability_score[t] = float(rate_obs_score_t)
            if np.isfinite(helper_freq_t):
                diag_helper_mismatch[t] = abs(helper_freq_t - freq_t)
            if multichannel:
                diag_R_channels[:, t] = np.asarray(R_t, dtype=np.float64)
                if diag_R_eff_channels is not None:
                    diag_R_eff_channels[:, t] = np.asarray(R_eff, dtype=np.float64)
                if diag_group_balance_scale_channels is not None:
                    diag_group_balance_scale_channels[:, t] = group_balance_scale
                diag_q_obs_channels[:, t] = np.asarray(q_obs_vec, dtype=np.float64)
                diag_pi_channels[:, t] = np.asarray(pi_t, dtype=np.float64)
                if diag_rate_observability_channels is not None and rate_obs_vec is not None:
                    diag_rate_observability_channels[:, t] = np.asarray(rate_obs_vec, dtype=np.float64)
                if bool(self.ENABLE_DYNAMIC_MIXTURE) and mixture_t is not None:
                    mix_arr = np.asarray(mixture_t, dtype=np.float64)
                    diag_mixture_channels[:, t] = mix_arr
                    diag_mixture_entropy[t] = float(
                        -np.sum(np.clip(mix_arr, 1e-12, 1.0) * np.log(np.clip(mix_arr, 1e-12, 1.0)))
                        / max(np.log(max(n_obs, 2)), 1e-12)
                    )
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
        morphology_confidence = np.sqrt(
            np.clip(diag_target_h2_morphology, 0.0, 1.0)
            * np.clip(diag_target_residual, 0.0, 1.0)
        )
        z_full_phase_anchored, phase_morph_meta = self._phase_anchored_morphology_readout(
            x_smooth,
            z_full_smoothed,
            morphology_confidence=morphology_confidence,
            residual_gate=diag_residual_gate,
            has_target_evidence=bool(enable_target_observability_control),
        )
        if bool(phase_morph_meta.get("enabled", False)):
            z_full_smoothed = z_full_phase_anchored
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
        track_hz_native_smoothed = np.asarray(track_hz_smoothed, dtype=np.float64).copy()
        if (
            multichannel
            and bool(self.ENABLE_DYNAMIC_MIXTURE)
            and bool(self.ENABLE_RATE_OBSERVABILITY_HELPER)
            and diag_rate_helper_freq is not None
            and np.asarray(diag_rate_helper_freq).size == track_hz_smoothed.size
        ):
            output_helper_freq_consensus = np.asarray(diag_rate_helper_freq, dtype=np.float64).copy()
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
        track_hz_smoothed, diag_external_output_rate_blend = self._external_output_rate_postprocess(
            track_hz_smoothed,
            external_output_rate,
            external_output_rate_confidence,
            alpha_used,
        )
        diag_output_rate_blend = np.maximum(diag_output_rate_blend, diag_external_output_rate_blend)
        diag_rate_source_arbiter = {
            "blend": np.zeros(n, dtype=np.float64),
            "selected": np.zeros(n, dtype=np.float64),
            "current_score": np.zeros(n, dtype=np.float64),
            "posterior_mean_score": np.zeros(n, dtype=np.float64),
            "posterior_mode_score": np.zeros(n, dtype=np.float64),
            "native_score": np.zeros(n, dtype=np.float64),
            "state_score": np.zeros(n, dtype=np.float64),
            "output_score": np.zeros(n, dtype=np.float64),
            "output_preservation": np.zeros(n, dtype=np.float64),
            "native_alias_safety": np.zeros(n, dtype=np.float64),
            "posterior_specificity": np.zeros(n, dtype=np.float64),
        }
        if (enable_rate_source_arbiter_v1 or enable_rate_source_arbiter_v2 or enable_rate_source_arbiter_v3) and external_rate_posterior:
            track_hz_smoothed, diag_rate_source_arbiter = self._rate_source_arbiter_v1_postprocess(
                track_hz_smoothed,
                track_hz_native_smoothed,
                diag_freq,
                external_output_rate,
                external_output_rate_confidence,
                external_rate_posterior,
                alpha_used,
                guard_version=rate_source_arbiter_version,
            )
            diag_output_rate_blend = np.maximum(diag_output_rate_blend, diag_rate_source_arbiter["blend"])

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
        if self.ENABLE_RESIDUAL and self.ENABLE_RESIDUAL_IDENTIFIABILITY_GUARD:
            active_modules.append("residual_identifiability_guard")
        if self.ENABLE_PHASE_ANCHORED_MORPHOLOGY and bool(phase_morph_meta.get("enabled", False)):
            active_modules.append("phase_anchored_morphology_readout")
        if multichannel and self.ENABLE_GROUP_BALANCED_FUSION:
            active_modules.append("group_balanced_fusion")
        if self.USE_HELPER_PATH: active_modules.append("helper_path")
        if self.USE_LIGHT_OBS_PATH: active_modules.append("light_obs_path")
        if (any(bool(oc["enabled"]) for oc in obs_cal) if multichannel else obs_cal["enabled"]):
            active_modules.append("obs_cal")
        if multichannel:
            active_modules.append("multichannel_observation")
        if multichannel and bool(self.ENABLE_DYNAMIC_MIXTURE):
            active_modules.append("dynamic_soft_observation_mixture")
            if bool(self.ENABLE_RATE_OBSERVABILITY_MIXTURE):
                active_modules.append("rate_observability_mixture")
            if bool(self.ENABLE_RATE_OBSERVABILITY_HELPER):
                active_modules.append("rate_observability_helper")
            if external_rate_tracks is not None:
                active_modules.append("external_rate_evidence")
        observation_law_enabled = bool(
            run_meta.get("observation_law_enabled", run_meta.get("observation_law_v2_enabled", False))
        )
        if observation_law_enabled:
            active_modules.append("observation_law")
        if state_role_prior_runtime:
            active_modules.append("state_component_reliability_law")
        if enable_target_observability_control:
            active_modules.append("target_observability_control")
        if external_rate_anchor_applied:
            active_modules.append("external_rate_anchor")
        external_output_rate_applied = bool(np.any(diag_external_output_rate_blend > 0.0))
        if external_output_rate_applied:
            active_modules.append("external_output_rate_readout")
        external_rate_posterior_present = bool(np.any(diag_external_rate_posterior_confidence > 0.0))
        external_rate_posterior_applied = bool(np.any(diag_external_rate_posterior_blend > 0.0))
        if external_rate_posterior_present:
            active_modules.append("candidate_rate_posterior")
        rate_source_arbiter_applied = bool(np.any(diag_rate_source_arbiter["blend"] > 0.0))
        if enable_rate_source_arbiter_v1 or enable_rate_source_arbiter_v2 or enable_rate_source_arbiter_v3:
            active_modules.append("diagnostic_rate_source_arbiter")

        meta_payload = dict(run_meta)
        meta_payload["f0"] = freq0
        meta_payload["f0_raw"] = float(freq0_raw)
        meta_payload["freq_source"] = "parh_ossm_phase"
        meta_payload["post_smooth_alpha_used"] = alpha_used
        meta_payload["model_version"] = MODEL_VERSION
        meta_payload["output_semantics"] = "smoothed"
        meta_payload["track_hz_semantics"] = (
            "final_reported_rate_after_helper_assistant_and_external_readout_postprocess"
        )
        meta_payload["track_hz_native_smoothed_semantics"] = (
            "native_PARH_OSSM_z_osc_frequency_after_RTS_and_post_smoothing_before_output_readout_postprocess"
        )
        meta_payload["warmup_frames"] = WARMUP
        meta_payload["active_modules"] = active_modules
        meta_payload["external_rate_anchor_applied"] = bool(external_rate_anchor_applied)
        meta_payload["external_output_rate_applied"] = bool(external_output_rate_applied)
        meta_payload["external_rate_posterior_applied"] = bool(external_rate_posterior_applied)
        meta_payload["target_observability_control_enabled"] = bool(enable_target_observability_control)
        meta_payload["target_observability_control"] = {
            "enabled": bool(enable_target_observability_control),
            "semantics": "GT-free target-side observation-trust control before state update",
            "trust_floor": float(self.TARGET_OBS_TRUST_FLOOR),
            "qdyn_floor": float(self.TARGET_OBS_QDYN_FLOOR),
            "nuisance_r_scale": float(self.TARGET_OBS_NUISANCE_R_SCALE),
            "role_power": float(self.TARGET_OBS_ROLE_POWER),
        }
        meta_payload["rate_source_arbiter_enabled"] = bool(
            enable_rate_source_arbiter_v1 or enable_rate_source_arbiter_v2 or enable_rate_source_arbiter_v3
        )
        meta_payload["rate_source_arbiter_mode"] = (
            "diagnostic_legacy"
            if (enable_rate_source_arbiter_v1 or enable_rate_source_arbiter_v2 or enable_rate_source_arbiter_v3)
            else "off"
        )
        meta_payload["rate_source_arbiter_applied"] = bool(rate_source_arbiter_applied)
        meta_payload["rate_source_arbiter_blend_active_frac"] = float(
            np.mean(diag_rate_source_arbiter["blend"] > 0.0)
        )
        meta_payload["rate_source_arbiter_blend_mean"] = (
            float(np.mean(diag_rate_source_arbiter["blend"][diag_rate_source_arbiter["blend"] > 0.0]))
            if np.any(diag_rate_source_arbiter["blend"] > 0.0)
            else 0.0
        )
        if external_rate_anchor is not None and external_rate_anchor_confidence is not None:
            valid_anchor_meta = np.isfinite(external_rate_anchor) & (external_rate_anchor_confidence > 0.0)
            meta_payload["external_rate_anchor_coverage"] = float(np.mean(valid_anchor_meta))
            meta_payload["external_rate_anchor_confidence_mean"] = (
                float(np.mean(external_rate_anchor_confidence[valid_anchor_meta]))
                if np.any(valid_anchor_meta) else 0.0
            )
            meta_payload["external_rate_anchor_hz_median"] = (
                float(np.median(external_rate_anchor[valid_anchor_meta]))
                if np.any(valid_anchor_meta) else float("nan")
            )
        if external_output_rate is not None and external_output_rate_confidence is not None:
            valid_output_meta = np.isfinite(external_output_rate) & (external_output_rate_confidence > 0.0)
            meta_payload["external_output_rate_coverage"] = float(np.mean(valid_output_meta))
            meta_payload["external_output_rate_confidence_mean"] = (
                float(np.mean(external_output_rate_confidence[valid_output_meta]))
                if np.any(valid_output_meta) else 0.0
            )
            meta_payload["external_output_rate_hz_median"] = (
                float(np.median(external_output_rate[valid_output_meta]))
                if np.any(valid_output_meta) else float("nan")
            )
        if external_rate_posterior_present:
            valid_posterior_meta = (
                np.isfinite(diag_external_rate_posterior_mode)
                & (diag_external_rate_posterior_confidence > 0.0)
            )
            meta_payload["external_rate_posterior_coverage"] = float(np.mean(valid_posterior_meta))
            meta_payload["external_rate_posterior_confidence_mean"] = (
                float(np.mean(diag_external_rate_posterior_confidence[valid_posterior_meta]))
                if np.any(valid_posterior_meta) else 0.0
            )
            meta_payload["external_rate_posterior_entropy_median"] = (
                float(np.median(diag_external_rate_posterior_entropy[valid_posterior_meta]))
                if np.any(valid_posterior_meta) else float("nan")
            )
            meta_payload["external_rate_posterior_top_gap_median"] = (
                float(np.median(diag_external_rate_posterior_top_gap[valid_posterior_meta]))
                if np.any(valid_posterior_meta) else float("nan")
            )
            meta_payload["external_rate_posterior_hz_median"] = (
                float(np.median(diag_external_rate_posterior_mode[valid_posterior_meta]))
                if np.any(valid_posterior_meta) else float("nan")
            )
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
        meta_payload["residual_identifiability_guard"] = {
            "enabled": bool(self.ENABLE_RESIDUAL_IDENTIFIABILITY_GUARD),
            "semantics": "open r only when non-oscillatory evidence is reliable rather than nuisance-dominated",
            "trust_floor": float(self.RESIDUAL_GUARD_TRUST_FLOOR),
            "nuisance_scale": float(self.RESIDUAL_GUARD_NUISANCE_SCALE),
            "mean_gate": float(np.mean(diag_residual_gate)),
            "failure_absorption_mean": float(np.mean(diag_residual_failure_absorption)),
        }
        meta_payload["phase_anchored_morphology"] = {
            **dict(phase_morph_meta or {}),
            "semantics": "bounded z_full readout that assembles morphology on z_osc phase when target evidence supports it",
        }
        meta_payload["observation_law"] = {
            "enabled": bool(observation_law_enabled),
            "nested_comparator": "",
            "safety_mode": "adaptive_R_eff_and_pi_prior_trust",
            "prior_collapse_threshold": 0.20,
            "requires_multichannel": True,
        }
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

        if multichannel:
            channel_context_list = [float(x) for x in np.asarray(channel_context_prior, dtype=np.float64).reshape(-1)]
            meta_payload["dynamic_observation_mixture"] = {
                "enabled": bool(self.ENABLE_DYNAMIC_MIXTURE),
                "tau_sec": float(self.DYNAMIC_MIXTURE_TAU_SEC),
                "temperature": float(self.DYNAMIC_MIXTURE_TEMPERATURE),
                "min_weight": float(self.DYNAMIC_MIXTURE_MIN_WEIGHT),
                "context_floor": float(self.DYNAMIC_MIXTURE_CONTEXT_FLOOR),
                "r_weight": float(self.DYNAMIC_MIXTURE_R_WEIGHT),
                "global_quality_floor": float(self.DYNAMIC_MIXTURE_GLOBAL_QUALITY_FLOOR),
                "rate_observability_enabled": bool(self.ENABLE_RATE_OBSERVABILITY_MIXTURE),
                "rate_observability_helper_enabled": bool(self.ENABLE_RATE_OBSERVABILITY_HELPER),
                "rate_obs_window_sec": float(self.RATE_OBS_WINDOW_SEC),
                "rate_obs_std_ref_hz": float(self.RATE_OBS_STD_REF_HZ),
                "rate_obs_agree_ref_hz": float(self.RATE_OBS_AGREE_REF_HZ),
                "rate_obs_harmonic_penalty": float(self.RATE_OBS_HARMONIC_PENALTY),
                "rate_obs_floor": float(self.RATE_OBS_FLOOR),
                "rate_obs_power": float(self.RATE_OBS_POWER),
                "rate_obs_helper_blend": float(self.RATE_OBS_HELPER_BLEND),
                "rate_obs_helper_min_support": float(self.RATE_OBS_HELPER_MIN_SUPPORT),
                "rate_obs_helper_max_step_hz": float(self.RATE_OBS_HELPER_MAX_STEP_HZ),
                "pi_trust_mode": "normalized_mixture_rescaled_by_n_channels",
                "external_rate_evidence": bool(external_rate_tracks is not None),
                "external_rate_anchor": bool(external_rate_anchor_applied),
                "channel_context_prior": channel_context_list,
                "channel_context_prior_sum": float(np.sum(channel_context_prior)),
                "external_channel_context_prior_applied": bool(
                    "external_context_prior" in locals()
                    and external_context_prior is not None
                    and np.asarray(external_context_prior).size == n_obs
                ),
                "external_channel_context_prior": (
                    [float(x) for x in np.asarray(external_context_prior, dtype=np.float64).reshape(-1)]
                    if "external_context_prior" in locals()
                    and external_context_prior is not None
                    and np.asarray(external_context_prior).size == n_obs
                    else []
                ),
                "external_channel_context_prior_runtime_applied": bool(
                    "external_context_prior_runtime" in locals()
                    and external_context_prior_runtime is not None
                    and np.asarray(external_context_prior_runtime).shape == (n_obs, n)
                ),
                "state_role_prior_runtime_applied": bool(state_role_prior_runtime),
                "state_role_context_power": float(self.STATE_ROLE_CONTEXT_POWER),
                "state_role_rate_power": float(self.STATE_ROLE_RATE_POWER),
                "state_role_context_multiplier_floor": float(self.STATE_ROLE_CONTEXT_MULTIPLIER_FLOOR),
                "state_role_rate_multiplier_floor": float(self.STATE_ROLE_RATE_MULTIPLIER_FLOOR),
                "state_role_abstain_r_scale": float(self.STATE_ROLE_ABSTAIN_R_SCALE),
                "group_balanced_fusion": bool(self.ENABLE_GROUP_BALANCED_FUSION),
                "group_balance_scale": [float(x) for x in np.asarray(group_balance_scale, dtype=np.float64)],
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
            "residual_gate_mean": float(np.mean(diag_residual_gate)),
            "residual_failure_absorption_mean": float(np.mean(diag_residual_failure_absorption)),
            "aper_drive_mean": float(np.mean(diag_aper_drive)),
            "helper_support_mean": float(np.mean(diag_helper_support)),
            "helper_trust_mean": float(np.mean(diag_helper_trust)),
            "helper_bias_conf_mean": float(np.mean(diag_helper_bias_conf)),
            "helper_mismatch_mean": float(np.nanmean(diag_helper_mismatch)),
            "helper_freq_mean": float(np.mean(diag_helper_freq)),
            "helper_freq_std": float(np.std(diag_helper_freq)),
            "external_rate_anchor_coverage": float(np.mean(diag_external_rate_anchor_confidence > 0.0)),
            "external_rate_anchor_confidence_mean": (
                float(np.mean(diag_external_rate_anchor_confidence[diag_external_rate_anchor_confidence > 0.0]))
                if np.any(diag_external_rate_anchor_confidence > 0.0) else 0.0
            ),
            "external_output_rate_coverage": float(np.mean(diag_external_output_rate_confidence > 0.0)),
            "external_output_rate_confidence_mean": (
                float(np.mean(diag_external_output_rate_confidence[diag_external_output_rate_confidence > 0.0]))
                if np.any(diag_external_output_rate_confidence > 0.0) else 0.0
            ),
            "external_output_rate_blend_active_frac": float(np.mean(diag_external_output_rate_blend > 0.0)),
            "external_rate_posterior_coverage": float(np.mean(diag_external_rate_posterior_confidence > 0.0)),
            "external_rate_posterior_confidence_mean": (
                float(np.mean(diag_external_rate_posterior_confidence[diag_external_rate_posterior_confidence > 0.0]))
                if np.any(diag_external_rate_posterior_confidence > 0.0) else 0.0
            ),
            "external_rate_posterior_entropy_median": (
                float(np.median(diag_external_rate_posterior_entropy[diag_external_rate_posterior_confidence > 0.0]))
                if np.any(diag_external_rate_posterior_confidence > 0.0) else float("nan")
            ),
            "external_rate_posterior_top_gap_median": (
                float(np.median(diag_external_rate_posterior_top_gap[diag_external_rate_posterior_confidence > 0.0]))
                if np.any(diag_external_rate_posterior_confidence > 0.0) else float("nan")
            ),
            "external_rate_posterior_blend_active_frac": float(np.mean(diag_external_rate_posterior_blend > 0.0)),
            "target_observability_control_enabled": bool(enable_target_observability_control),
            "target_observability_score_mean": float(np.mean(diag_target_observability)),
            "target_h1_timing_score_mean": float(np.mean(diag_target_h1_timing)),
            "target_h2_morphology_score_mean": float(np.mean(diag_target_h2_morphology)),
            "target_nuisance_mean": float(np.mean(diag_target_nuisance)),
            "target_source_spread_hz_median": float(np.median(diag_target_source_spread_hz)),
            "target_posterior_specificity_median": float(np.median(diag_target_posterior_specificity)),
            "target_alias_safety_median": float(np.median(diag_target_alias_safety)),
            "rate_source_arbiter_enabled": bool(
                enable_rate_source_arbiter_v1 or enable_rate_source_arbiter_v2 or enable_rate_source_arbiter_v3
            ),
            "rate_source_arbiter_mode": (
                "diagnostic_legacy"
                if (enable_rate_source_arbiter_v1 or enable_rate_source_arbiter_v2 or enable_rate_source_arbiter_v3)
                else "off"
            ),
            "rate_source_arbiter_blend_active_frac": float(np.mean(diag_rate_source_arbiter["blend"] > 0.0)),
            "rate_source_arbiter_blend_mean": (
                float(np.mean(diag_rate_source_arbiter["blend"][diag_rate_source_arbiter["blend"] > 0.0]))
                if np.any(diag_rate_source_arbiter["blend"] > 0.0) else 0.0
            ),
            "rate_source_arbiter_selected_current_frac": float(np.mean(diag_rate_source_arbiter["selected"] == 0.0)),
            "rate_source_arbiter_selected_posterior_mean_frac": float(
                np.mean(diag_rate_source_arbiter["selected"] == 1.0)
            ),
            "rate_source_arbiter_selected_posterior_mode_frac": float(
                np.mean(diag_rate_source_arbiter["selected"] == 2.0)
            ),
            "rate_source_arbiter_selected_native_frac": float(np.mean(diag_rate_source_arbiter["selected"] == 3.0)),
            "rate_source_arbiter_selected_state_frac": float(np.mean(diag_rate_source_arbiter["selected"] == 4.0)),
            "rate_source_arbiter_selected_output_frac": float(np.mean(diag_rate_source_arbiter["selected"] == 5.0)),
            "prior_collapse_frac": float(np.mean(diag_prior_collapse > 0.0)),
            "prior_trust_mean": float(np.mean(diag_pi)),
            "mixture_entropy_mean": float(np.mean(diag_mixture_entropy)) if multichannel else 0.0,
            "freq_rescue_active_frac": float(np.mean(diag_freq_rescue)),
            "output_rate_blend_active_frac": float(np.mean(diag_output_rate_blend)),
            "energy_h1": float(np.mean(x_smooth[:, self.HC1] ** 2)),
            "energy_h2": float(np.mean(x_smooth[:, self.HC2] ** 2)) if self.ENABLE_HARMONIC2 else 0.0,
            "energy_baseline": float(np.mean(x_smooth[:, self.B] ** 2)) if self.ENABLE_BASELINE else 0.0,
            "energy_residual": float(np.mean(x_smooth[:, self.R] ** 2)) if self.ENABLE_RESIDUAL else 0.0,
        }
        if multichannel:
            meta_payload["parh_ossm_diagnostics"]["pi_channel_mean"] = [
                float(np.mean(diag_pi_channels[idx])) for idx in range(n_obs)
            ]
            meta_payload["parh_ossm_diagnostics"]["mixture_channel_mean"] = [
                float(np.mean(diag_mixture_channels[idx])) for idx in range(n_obs)
            ]
            meta_payload["parh_ossm_diagnostics"]["mixture_channel_entropy_mean"] = float(np.mean(
                -np.sum(
                    np.clip(diag_mixture_channels, 1e-12, 1.0)
                    * np.log(np.clip(diag_mixture_channels, 1e-12, 1.0)),
                    axis=0,
                ) / max(np.log(max(n_obs, 2)), 1e-12)
            ))
            if diag_rate_observability_channels is not None:
                meta_payload["parh_ossm_diagnostics"]["rate_observability_channel_mean"] = [
                    float(np.mean(diag_rate_observability_channels[idx])) for idx in range(n_obs)
                ]
            if diag_rate_observability_score is not None:
                meta_payload["parh_ossm_diagnostics"]["rate_observability_score_mean"] = float(
                    np.mean(diag_rate_observability_score)
                )
            if diag_state_role_context_channels is not None:
                meta_payload["parh_ossm_diagnostics"]["state_role_context_channel_mean"] = [
                    float(np.mean(diag_state_role_context_channels[idx])) for idx in range(n_obs)
                ]
                meta_payload["parh_ossm_diagnostics"]["state_role_abstain_channel_mean"] = [
                    float(np.mean(diag_state_role_abstain_channels[idx])) for idx in range(n_obs)
                ]
                meta_payload["parh_ossm_diagnostics"]["state_role_h1_channel_mean"] = [
                    float(np.mean(diag_state_role_h1_channels[idx])) for idx in range(n_obs)
                ]

        result = self._package(signal_hat, track_hz, meta_payload)

        # Numpy arrays (not JSON-serialisable — attached after _package)
        result["z_osc"] = z_osc_smoothed
        result["z_full"] = z_full_smoothed
        result["z_full_phase_anchored"] = z_full_phase_anchored
        result["z_osc_causal"] = z_osc_causal
        result["z_full_causal"] = z_full_causal
        result["z_osc_smoothed"] = z_osc_smoothed
        result["z_full_smoothed"] = z_full_smoothed
        result["track_hz_causal"] = track_hz_causal
        result["track_hz_native_smoothed"] = track_hz_native_smoothed
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
            "residual_gate_t": diag_residual_gate,
            "residual_failure_absorption_t": diag_residual_failure_absorption,
            "aper_drive_t": diag_aper_drive,
            "freq_t": diag_freq,
            "helper_support_t": diag_helper_support,
            "helper_trust_t": diag_helper_trust,
            "helper_bias_conf_t": diag_helper_bias_conf,
            "helper_freq_t": diag_helper_freq,
            "helper_mismatch_t": diag_helper_mismatch,
            "external_rate_anchor_t": diag_external_rate_anchor,
            "external_rate_anchor_confidence_t": diag_external_rate_anchor_confidence,
            "external_output_rate_t": diag_external_output_rate,
            "external_output_rate_confidence_t": diag_external_output_rate_confidence,
            "external_output_rate_blend_t": diag_external_output_rate_blend,
            "external_rate_posterior_mode_t": diag_external_rate_posterior_mode,
            "external_rate_posterior_mean_t": diag_external_rate_posterior_mean,
            "external_rate_posterior_confidence_t": diag_external_rate_posterior_confidence,
            "external_rate_posterior_entropy_t": diag_external_rate_posterior_entropy,
            "external_rate_posterior_top_gap_t": diag_external_rate_posterior_top_gap,
            "external_rate_posterior_blend_t": diag_external_rate_posterior_blend,
            "target_observability_score_t": diag_target_observability,
            "target_h1_timing_score_t": diag_target_h1_timing,
            "target_h2_morphology_score_t": diag_target_h2_morphology,
            "target_baseline_score_t": diag_target_baseline,
            "target_residual_score_t": diag_target_residual,
            "target_nuisance_t": diag_target_nuisance,
            "target_source_spread_hz_t": diag_target_source_spread_hz,
            "target_posterior_specificity_t": diag_target_posterior_specificity,
            "target_alias_safety_t": diag_target_alias_safety,
            "rate_source_arbiter_blend_t": diag_rate_source_arbiter["blend"],
            "rate_source_arbiter_selected_t": diag_rate_source_arbiter["selected"],
            "rate_source_arbiter_current_score_t": diag_rate_source_arbiter["current_score"],
            "rate_source_arbiter_posterior_mean_score_t": diag_rate_source_arbiter["posterior_mean_score"],
            "rate_source_arbiter_posterior_mode_score_t": diag_rate_source_arbiter["posterior_mode_score"],
            "rate_source_arbiter_native_score_t": diag_rate_source_arbiter["native_score"],
            "rate_source_arbiter_state_score_t": diag_rate_source_arbiter["state_score"],
            "rate_source_arbiter_output_score_t": diag_rate_source_arbiter["output_score"],
            "rate_source_arbiter_output_preservation_t": diag_rate_source_arbiter["output_preservation"],
            "rate_source_arbiter_native_alias_safety_t": diag_rate_source_arbiter["native_alias_safety"],
            "rate_source_arbiter_posterior_specificity_t": diag_rate_source_arbiter["posterior_specificity"],
            "prior_collapse_t": diag_prior_collapse,
            "mixture_entropy_t": diag_mixture_entropy,
            "freq_rescue_t": diag_freq_rescue,
            "output_rate_blend_t": diag_output_rate_blend,
            "nis_empirical_t": diag_nis,
        }
        if multichannel:
            result["diagnostics"]["R_t_channels"] = diag_R_channels
            result["diagnostics"]["R_eff_t_channels"] = diag_R_eff_channels
            result["diagnostics"]["q_obs_t_channels"] = diag_q_obs_channels
            result["diagnostics"]["pi_t_channels"] = diag_pi_channels
            result["diagnostics"]["mixture_t_channels"] = diag_mixture_channels
            result["diagnostics"]["context_prior_channels"] = diag_context_channels
            result["diagnostics"]["state_role_context_t_channels"] = diag_state_role_context_channels
            result["diagnostics"]["state_role_abstain_t_channels"] = diag_state_role_abstain_channels
            result["diagnostics"]["state_role_h1_t_channels"] = diag_state_role_h1_channels
            result["diagnostics"]["state_role_zosc_t_channels"] = diag_state_role_zosc_channels
            result["diagnostics"]["group_balance_scale_t_channels"] = diag_group_balance_scale_channels
            result["diagnostics"]["rate_observability_t_channels"] = diag_rate_observability_channels
            result["diagnostics"]["rate_observability_score_t"] = diag_rate_observability_score
            result["diagnostics"]["rate_helper_freq_t"] = diag_rate_helper_freq

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
