from __future__ import annotations

from typing import Dict, Iterable, Tuple


CANONICAL_OBSERVATION_FAMILY_ORDER: Tuple[str, ...] = (
    "of_farneback",
    "of_disp_bridge",
    "dof",
    "dof_disp_bridge",
    "profile1d_linear",
    "profile1d_quadratic",
    "profile1d_cubic",
    "profile1d_consensus",
)


_DEFAULTS: Dict[str, object] = {
    "canonical_family": "",
    "display_name": "",
    "family_group": "unknown",
    "construction": "unknown",
    "observation_domain": "displacement",
    "primary_information": "unknown",
    "secondary_information": "unknown",
    "nuisance_risk": "unknown",
    "current_parh_role": "unassigned",
    "residual_observability": 0.50,
    "baseline_observability": 0.50,
    "default_inference_mode": "legacy",
    "allow_freq_rescue": False,
    "allow_helper_trust": False,
    "allow_family_confidence": False,
    "allow_output_rate_refine": False,
    "allow_obs_calibration": False,
    "helper_heavy": False,
    "waveform_primary": False,
    "rate_primary": False,
}


_REGISTRY: Dict[str, Dict[str, object]] = {
    "of_farneback": {
        "display_name": "OF",
        "family_group": "optical_flow",
        "construction": "raw vertical optical-flow surrogate",
        "observation_domain": "velocity",
        "primary_information": "oscillatory_motion",
        "secondary_information": "local_phase_frequency",
        "nuisance_risk": "medium",
        "current_parh_role": "helper-heavy rate family",
        "residual_observability": 0.70,
        "baseline_observability": 0.35,
        "default_inference_mode": "light",
        "allow_freq_rescue": True,
        "allow_helper_trust": True,
        "allow_output_rate_refine": True,
        "allow_obs_calibration": True,
        "helper_heavy": True,
        "waveform_primary": False,
        "rate_primary": True,
    },
    "of_disp_bridge": {
        "display_name": "OF_bridge",
        "family_group": "optical_flow_bridge",
        "construction": "OF-derived displacement-compatible bridge",
        "observation_domain": "displacement",
        "primary_information": "bridged_oscillatory_displacement",
        "secondary_information": "rate_plus_waveform",
        "nuisance_risk": "medium",
        "current_parh_role": "rate-oriented constructed family",
        "residual_observability": 0.55,
        "baseline_observability": 0.20,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": True,
        "allow_helper_trust": True,
        "allow_output_rate_refine": True,
        "allow_obs_calibration": True,
        "helper_heavy": False,
        "waveform_primary": False,
        "rate_primary": True,
    },
    "dof": {
        "display_name": "DoF",
        "family_group": "motion_energy",
        "construction": "thresholded frame-difference count",
        "observation_domain": "energy",
        "primary_information": "thresholded_motion_energy",
        "secondary_information": "burst_sensitivity",
        "nuisance_risk": "high",
        "current_parh_role": "nuisance-limited auxiliary family",
        "residual_observability": 0.30,
        "baseline_observability": 0.10,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": False,
        "allow_helper_trust": False,
        "allow_family_confidence": False,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": False,
        "helper_heavy": False,
        "waveform_primary": False,
        "rate_primary": False,
    },
    "dof_disp_bridge": {
        "display_name": "DoF_bridge",
        "family_group": "motion_energy_bridge",
        "construction": "signed DoF motion-centroid trajectory bridge",
        "observation_domain": "signed_motion",
        "primary_information": "coarse_signed_motion_trajectory",
        "secondary_information": "nuisance_reduced_motion_energy",
        "nuisance_risk": "high",
        "current_parh_role": "experimental constructed DoF family",
        "residual_observability": 0.40,
        "baseline_observability": 0.15,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": False,
        "allow_helper_trust": False,
        "allow_family_confidence": False,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": False,
        "helper_heavy": False,
        "waveform_primary": False,
        "rate_primary": False,
    },
    "profile1d_linear": {
        "display_name": "P1D_lin",
        "family_group": "profile1d_linear",
        "construction": "linear 1D profile shift surrogate",
        "observation_domain": "displacement",
        "primary_information": "fundamental_displacement",
        "secondary_information": "smooth_shift_surrogate",
        "nuisance_risk": "medium",
        "current_parh_role": "conservative displacement family",
        "residual_observability": 0.45,
        "baseline_observability": 0.25,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": True,
        "allow_helper_trust": False,
        "allow_family_confidence": False,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": False,
        "helper_heavy": False,
        "waveform_primary": False,
        "rate_primary": False,
    },
    "profile1d_linear_bridge": {
        "display_name": "P1D_lin_bridge",
        "family_group": "profile1d_linear_bridge",
        "construction": "centered signed-lag bridge from linear 1D profile xcorr",
        "observation_domain": "displacement",
        "primary_information": "signed_fundamental_shift",
        "secondary_information": "centered_displacement_surrogate",
        "nuisance_risk": "medium",
        "current_parh_role": "experimental constructed linear family",
        "residual_observability": 0.40,
        "baseline_observability": 0.20,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": True,
        "allow_helper_trust": False,
        "allow_family_confidence": False,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": True,
        "helper_heavy": False,
        "waveform_primary": False,
        "rate_primary": False,
    },
    "profile1d_quadratic": {
        "display_name": "P1D_quad",
        "family_group": "profile1d_harmonic",
        "construction": "quadratic 1D profile shift surrogate",
        "observation_domain": "displacement",
        "primary_information": "harmonic_morphology",
        "secondary_information": "inhale_exhale_asymmetry",
        "nuisance_risk": "low",
        "current_parh_role": "waveform-primary harmonic family",
        "residual_observability": 0.10,
        "baseline_observability": 0.05,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": False,
        "allow_helper_trust": False,
        "allow_family_confidence": True,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": True,
        "helper_heavy": False,
        "waveform_primary": True,
        "rate_primary": False,
    },
    "profile1d_quadratic_bridge": {
        "display_name": "P1D_quad_bridge",
        "family_group": "profile1d_harmonic_bridge",
        "construction": "centered signed-lag bridge from quadratic 1D profile xcorr",
        "observation_domain": "displacement",
        "primary_information": "centered_harmonic_shift",
        "secondary_information": "signed_morphology_surrogate",
        "nuisance_risk": "low",
        "current_parh_role": "experimental harmonic bridge family",
        "residual_observability": 0.10,
        "baseline_observability": 0.05,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": False,
        "allow_helper_trust": False,
        "allow_family_confidence": True,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": True,
        "helper_heavy": False,
        "waveform_primary": True,
        "rate_primary": False,
    },
    "profile1d_cubic": {
        "display_name": "P1D_cub",
        "family_group": "profile1d_harmonic",
        "construction": "cubic 1D profile shift surrogate",
        "observation_domain": "displacement",
        "primary_information": "harmonic_morphology",
        "secondary_information": "inhale_exhale_asymmetry",
        "nuisance_risk": "low",
        "current_parh_role": "waveform-primary harmonic family",
        "residual_observability": 0.12,
        "baseline_observability": 0.05,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": False,
        "allow_helper_trust": False,
        "allow_family_confidence": True,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": True,
        "helper_heavy": False,
        "waveform_primary": True,
        "rate_primary": False,
    },
    "profile1d_cubic_bridge": {
        "display_name": "P1D_cub_bridge",
        "family_group": "profile1d_harmonic_bridge",
        "construction": "centered signed-lag bridge from cubic 1D profile xcorr",
        "observation_domain": "displacement",
        "primary_information": "centered_harmonic_shift",
        "secondary_information": "signed_morphology_surrogate",
        "nuisance_risk": "low",
        "current_parh_role": "experimental harmonic bridge family",
        "residual_observability": 0.12,
        "baseline_observability": 0.05,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": False,
        "allow_helper_trust": False,
        "allow_family_confidence": True,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": True,
        "helper_heavy": False,
        "waveform_primary": True,
        "rate_primary": False,
    },
    "profile1d_consensus": {
        "display_name": "P1D_cons",
        "family_group": "profile1d_consensus",
        "construction": "weighted consensus of linear/quadratic/cubic profile surrogates",
        "observation_domain": "displacement",
        "primary_information": "consensus_displacement_morphology",
        "secondary_information": "cross_interp_stability",
        "nuisance_risk": "low",
        "current_parh_role": "experimental constructed P1D family",
        "residual_observability": 0.14,
        "baseline_observability": 0.06,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": False,
        "allow_helper_trust": False,
        "allow_family_confidence": True,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": True,
        "helper_heavy": False,
        "waveform_primary": True,
        "rate_primary": False,
    },
    "canonical_waveform": {
        "display_name": "Canonical_wave",
        "family_group": "canonical",
        "construction": "adapted multi-family canonical waveform observation",
        "observation_domain": "displacement",
        "primary_information": "canonical_waveform_reconstruction",
        "secondary_information": "shared_latent_alignment",
        "nuisance_risk": "medium",
        "current_parh_role": "adaptive canonical observation",
        "residual_observability": 0.35,
        "baseline_observability": 0.20,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": False,
        "allow_helper_trust": False,
        "allow_family_confidence": False,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": False,
        "helper_heavy": False,
        "waveform_primary": True,
        "rate_primary": False,
    },
    "canonical_osc": {
        "display_name": "Canonical_osc",
        "family_group": "canonical",
        "construction": "adapted multi-family canonical oscillatory observation",
        "observation_domain": "displacement",
        "primary_information": "canonical_rate_support",
        "secondary_information": "shared_latent_alignment",
        "nuisance_risk": "medium",
        "current_parh_role": "adaptive canonical oscillatory cue",
        "residual_observability": 0.10,
        "baseline_observability": 0.05,
        "default_inference_mode": "legacy",
        "allow_freq_rescue": False,
        "allow_helper_trust": False,
        "allow_family_confidence": False,
        "allow_output_rate_refine": False,
        "allow_obs_calibration": False,
        "helper_heavy": False,
        "waveform_primary": False,
        "rate_primary": True,
    },
}


def canonicalize_observation_family(name: str) -> str:
    key = str(name or "").strip().lower()
    if key in {"of", "of_model", "of_farneback"}:
        return "of_farneback"
    if key in {"of_disp_bridge", "of_displacement_bridge", "of_bridge"}:
        return "of_disp_bridge"
    if key == "dof":
        return "dof"
    if key in {"dof_disp_bridge", "dof_bridge", "dof_displacement_bridge"}:
        return "dof_disp_bridge"
    if key in {"profile1d_linear", "profile1d linear", "profile1d-linear"}:
        return "profile1d_linear"
    if key in {"profile1d_linear_bridge", "profile1d_linear_displacement_bridge"}:
        return "profile1d_linear_bridge"
    if key in {"profile1d_quadratic", "profile1d quadratic", "profile1d-quadratic"}:
        return "profile1d_quadratic"
    if key in {"profile1d_quadratic_bridge", "profile1d_quadratic_displacement_bridge"}:
        return "profile1d_quadratic_bridge"
    if key in {"profile1d_cubic", "profile1d cubic", "profile1d-cubic"}:
        return "profile1d_cubic"
    if key in {"profile1d_cubic_bridge", "profile1d_cubic_displacement_bridge"}:
        return "profile1d_cubic_bridge"
    if key in {"profile1d_consensus", "profile1d_cons"}:
        return "profile1d_consensus"
    if key in {"canonical_waveform", "canonical_wave"}:
        return "canonical_waveform"
    if key in {"canonical_osc", "canonical_rate", "canonical_rate_support"}:
        return "canonical_osc"
    return key


def get_observation_family_semantics(name: str) -> Dict[str, object]:
    canonical = canonicalize_observation_family(name)
    payload: Dict[str, object] = dict(_DEFAULTS)
    payload["canonical_family"] = canonical
    payload.update(_REGISTRY.get(canonical, {}))
    payload["canonical_family"] = canonical
    return payload


def observation_family_order_key(name: str) -> Tuple[int, str]:
    canonical = canonicalize_observation_family(name)
    try:
        return CANONICAL_OBSERVATION_FAMILY_ORDER.index(canonical), canonical
    except ValueError:
        return len(CANONICAL_OBSERVATION_FAMILY_ORDER), canonical


def ordered_observation_families(names: Iterable[str]) -> list[str]:
    seen: dict[str, str] = {}
    for name in names:
        canonical = canonicalize_observation_family(name)
        if canonical not in seen:
            seen[canonical] = name
    return [seen[k] for k in sorted(seen, key=observation_family_order_key)]
