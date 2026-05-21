from typing import Optional
from .core.base import OscillatorParams, _BaseOscillatorHead, PLLAdaptiveController
from .heads.kf_std import oscillator_KFstd
from .heads.simple_bandpass import oscillator_SimpleBandpass
from .heads.narossm import oscillator_NAROSSM
from .heads.parh_ossm import oscillator_PARH_OSSM
from .heads.parh_resonator_adaptive import oscillator_PARH_ResonatorAdaptive

# Archived heads — import conditionally for backward compatibility
try:
    from .heads.ukf_freq import oscillator_UKF_freq
except ImportError:
    oscillator_UKF_freq = None
try:
    from .heads.robust_attention import oscillator_Robust_Attention
except ImportError:
    oscillator_Robust_Attention = None
try:
    from .heads.head_ensemble import OscillatorHeadEnsemble
except ImportError:
    OscillatorHeadEnsemble = None
try:
    from .heads.ag_akf import oscillator_AGAKF
except ImportError:
    oscillator_AGAKF = None
try:
    from .heads.robust_ossm import oscillator_RobustOSSM
except ImportError:
    oscillator_RobustOSSM = None

HEAD_REGISTRY = {
    "kfstd": oscillator_KFstd,
    "kf_std": oscillator_KFstd,
    "simple_bandpass": oscillator_SimpleBandpass,
    "narossm": oscillator_NAROSSM,
    "parh_ossm": oscillator_PARH_OSSM,
    "parhossm": oscillator_PARH_OSSM,
    "parh_resonator_adaptive": oscillator_PARH_ResonatorAdaptive,
    "parhresonatoradaptive": oscillator_PARH_ResonatorAdaptive,
    "adaptive_resonator": oscillator_PARH_ResonatorAdaptive,
}

# Register archived heads if available
if oscillator_UKF_freq is not None:
    HEAD_REGISTRY["ukffreq"] = oscillator_UKF_freq
    HEAD_REGISTRY["ukf_freq"] = oscillator_UKF_freq
if oscillator_Robust_Attention is not None:
    HEAD_REGISTRY["robust_attention"] = oscillator_Robust_Attention
    HEAD_REGISTRY["robust"] = oscillator_Robust_Attention
if OscillatorHeadEnsemble is not None:
    HEAD_REGISTRY["ensemble"] = OscillatorHeadEnsemble
if oscillator_AGAKF is not None:
    HEAD_REGISTRY["agakf"] = oscillator_AGAKF
    HEAD_REGISTRY["ag_akf"] = oscillator_AGAKF
if oscillator_RobustOSSM is not None:
    HEAD_REGISTRY["robust_ossm"] = oscillator_RobustOSSM
    HEAD_REGISTRY["robust_bayesian"] = oscillator_RobustOSSM
    HEAD_REGISTRY["robust_ossm_ekf"] = oscillator_RobustOSSM
    HEAD_REGISTRY["robust_ossm_ukf"] = oscillator_RobustOSSM


def build_head(head_key: str, params: Optional[OscillatorParams] = None):
    key = head_key.lower()
    if key not in HEAD_REGISTRY:
        raise ValueError(f"Unknown oscillator head '{head_key}'")
    cls = HEAD_REGISTRY[key]
    return cls(params=params)
