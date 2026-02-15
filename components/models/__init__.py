from typing import Optional
from .core.base import OscillatorParams, _BaseOscillatorHead, PLLAdaptiveController
from .heads.kf_std import oscillator_KFstd
from .heads.ukf_freq import oscillator_UKF_freq
from .heads.robust_attention import oscillator_Robust_Attention
from .heads.robust_attention import oscillator_Robust_Attention
from .heads.head_ensemble import OscillatorHeadEnsemble
from .heads.ag_akf import oscillator_AGAKF
from .heads.robust_ossm import oscillator_RobustOSSM

HEAD_REGISTRY = {
    "kfstd": oscillator_KFstd,
    "kf_std": oscillator_KFstd,
    "ukffreq": oscillator_UKF_freq,
    "ukf_freq": oscillator_UKF_freq,
    "robust_attention": oscillator_Robust_Attention,
    "robust": oscillator_Robust_Attention,
    "robust": oscillator_Robust_Attention,
    "ensemble": OscillatorHeadEnsemble,
    "agakf": oscillator_AGAKF,
    "ag_akf": oscillator_AGAKF,
    "robust_ossm": oscillator_RobustOSSM,
    "robust_bayesian": oscillator_RobustOSSM,
    "robust_ossm_ekf": oscillator_RobustOSSM,
    "robust_ossm_ukf": oscillator_RobustOSSM,
}

def build_head(head_key: str, params: Optional[OscillatorParams] = None):
    key = head_key.lower()
    if key not in HEAD_REGISTRY:
        if key == 'ensemble': # Special case if registry didn't catch it
             return OscillatorHeadEnsemble(params=params)
        raise ValueError(f"Unknown oscillator head '{head_key}'")
    cls = HEAD_REGISTRY[key]
    return cls(params=params)
