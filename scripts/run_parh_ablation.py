#!/usr/bin/env python3
"""Run named PARH-OSSM ablation profiles via environment overrides."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ABLATIONS = {
    "full": {},
    "obs_policy_quadcub_helper": {
        "RESPYRE_PARH_OBS_FAMILY_POLICY": "quadcub_helper",
    },
    "obs_policy_quadcub_dof_helper": {
        "RESPYRE_PARH_OBS_FAMILY_POLICY": "quadcub_dof_helper",
    },
    "obs_policy_all_p1d_helper": {
        "RESPYRE_PARH_OBS_FAMILY_POLICY": "all_p1d_helper",
    },
    "obs_policy_quadcub_blend": {
        "RESPYRE_PARH_OBS_FAMILY_POLICY": "quadcub_blend",
        "RESPYRE_PARH_OBS_BLEND_ALPHA": "0.35",
    },
    "obs_policy_quadcub_dof_blend": {
        "RESPYRE_PARH_OBS_FAMILY_POLICY": "quadcub_dof_blend",
        "RESPYRE_PARH_OBS_BLEND_ALPHA": "0.35",
    },
    "obs_cal_v2": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "1",
        "RESPYRE_PARH_OBS_CAL_MODE": "global_signed_gain",
    },
    "obs_cal_v3": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "1",
        "RESPYRE_PARH_OBS_CAL_MODE": "osc_aux_two_gain",
    },
    "obs_cal_v4": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "1",
        "RESPYRE_PARH_OBS_CAL_MODE": "family_phase_aux",
        "RESPYRE_PARH_OBS_CAL_PRIOR_STRENGTH": "0.75",
        "RESPYRE_PARH_OBS_CAL_MAX_LAG_SEC": "0.30",
    },
    "obs_cal_v5": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "1",
        "RESPYRE_PARH_OBS_CAL_MODE": "family_phase_aux",
        "RESPYRE_PARH_OBS_CAL_ALLOWED_FAMILIES": "profile1d_quadratic,profile1d_cubic",
        "RESPYRE_PARH_OBS_CAL_PRIOR_STRENGTH": "1.50",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_AUX": "0.45",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_H1": "1.25",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_H2": "1.50",
        "RESPYRE_PARH_OBS_CAL_MAX_LAG_SEC": "0.12",
    },
    "obs_cal_v6": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "1",
        "RESPYRE_PARH_OBS_CAL_MODE": "family_phase_split_aux",
        "RESPYRE_PARH_OBS_CAL_ALLOWED_FAMILIES": "profile1d_quadratic,profile1d_cubic",
        "RESPYRE_PARH_OBS_CAL_PRIOR_STRENGTH": "1.75",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_H1": "1.25",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_H2": "1.50",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_B": "0.30",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_R": "0.75",
        "RESPYRE_PARH_OBS_CAL_MAX_LAG_SEC": "0.10",
    },
    "obs_cal_of_velocity_v2": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "1",
        "RESPYRE_PARH_OBS_CAL_MODE": "family_phase_aux",
        "RESPYRE_PARH_OBS_CAL_ALLOWED_FAMILIES": "of_farneback,profile1d_quadratic,profile1d_cubic",
        "RESPYRE_PARH_OF_HARMONIC_ONLY": "1",
        "RESPYRE_PARH_OF_PRIOR_STRENGTH": "1.40",
        "RESPYRE_PARH_OF_MAX_GAIN_H1": "1.05",
        "RESPYRE_PARH_OF_MAX_GAIN_H2": "0.75",
        "RESPYRE_PARH_OF_MAX_LAG_SEC": "0.05",
        "RESPYRE_PARH_OBS_CAL_MAX_LAG_SEC": "0.10",
    },
    "of_fixed_velocity_prior_v1": {
        "RESPYRE_PARH_OF_FIXED_VELOCITY_PRIOR": "1",
    },
    "no_h2": {
        "RESPYRE_PARH_ENABLE_HARMONIC2": "0",
    },
    "no_baseline": {
        "RESPYRE_PARH_ENABLE_BASELINE": "0",
    },
    "no_residual": {
        "RESPYRE_PARH_ENABLE_RESIDUAL": "0",
    },
    "no_adapt_r": {
        "RESPYRE_PARH_ENABLE_ADAPT_R": "0",
    },
    "no_student_t": {
        "RESPYRE_PARH_ENABLE_STUDENT_T": "0",
    },
    "legacy_q": {
        "RESPYRE_PARH_ENABLE_DISENTANGLED_Q": "0",
        "RESPYRE_PARH_ENABLE_LEGACY_COUPLED_Q": "1",
    },
    "no_freq_adapt": {
        "RESPYRE_PARH_ENABLE_FREQ_ADAPT": "0",
    },
    "no_freq_rescue": {
        "RESPYRE_PARH_ENABLE_FREQ_RESCUE": "0",
    },
    "freq_rescue_of_v2": {
        "RESPYRE_PARH_FREQ_RESCUE_POLICY": "of_v2",
    },
    "output_rate_of_v1": {
        "RESPYRE_PARH_OUTPUT_RATE_POLICY": "of_helper_blend_v1",
    },
    "output_rate_of_strong": {
        "RESPYRE_PARH_OUTPUT_RATE_POLICY": "of_helper_blend_v1",
        "RESPYRE_PARH_OUTPUT_RATE_BLEND_ALPHA": "0.65",
    },
    "output_rate_of_strong_relaxed": {
        "RESPYRE_PARH_OUTPUT_RATE_POLICY": "of_helper_blend_v1",
        "RESPYRE_PARH_OUTPUT_RATE_BLEND_ALPHA": "0.65",
        "RESPYRE_PARH_OUTPUT_RATE_MIN_SUPPORT": "0.66",
        "RESPYRE_PARH_OUTPUT_RATE_MIN_QDYN": "0.35",
        "RESPYRE_PARH_OUTPUT_RATE_MIN_MISMATCH_HZ": "0.03",
    },
    "output_rate_of_bias_v1": {
        "RESPYRE_PARH_OUTPUT_RATE_POLICY": "of_helper_bias_v1",
        "RESPYRE_PARH_OUTPUT_RATE_BLEND_ALPHA": "0.55",
        "RESPYRE_PARH_OUTPUT_RATE_MIN_SUPPORT": "0.84",
        "RESPYRE_PARH_OUTPUT_RATE_MIN_QDYN": "0.40",
        "RESPYRE_PARH_OUTPUT_RATE_MIN_MISMATCH_HZ": "0.035",
        "RESPYRE_PARH_OUTPUT_RATE_BIAS_WIN_SEC": "5.0",
        "RESPYRE_PARH_OUTPUT_RATE_BIAS_MIN_SIGN_STABILITY": "0.72",
        "RESPYRE_PARH_OUTPUT_RATE_BIAS_MAX_HELPER_STD_HZ": "0.075",
        "RESPYRE_PARH_OUTPUT_RATE_BIAS_MAX_CORR_HZ": "0.045",
    },
    "helper_trust_of_v1": {
        "RESPYRE_PARH_HELPER_TRUST_POLICY": "of_v1",
    },
    "helper_trust_of_v1_rescue_v2": {
        "RESPYRE_PARH_HELPER_TRUST_POLICY": "of_v1",
        "RESPYRE_PARH_FREQ_RESCUE_POLICY": "of_v2",
    },
    "helper_trust_rescue_only_v1": {
        "RESPYRE_PARH_HELPER_TRUST_POLICY": "of_rescue_only_v1",
    },
    "helper_trust_rescue_only_v1_rescue_v2": {
        "RESPYRE_PARH_HELPER_TRUST_POLICY": "of_rescue_only_v1",
        "RESPYRE_PARH_FREQ_RESCUE_POLICY": "of_v2",
    },
    "residual_release_v1": {
        "RESPYRE_PARH_Q_OSC_OBS_WEIGHT": "0.25",
        "RESPYRE_PARH_Q_OSC_OBS_MODE": "blend_support",
    },
    "residual_release_v2": {
        "RESPYRE_PARH_Q_OSC_OBS_WEIGHT": "0.20",
        "RESPYRE_PARH_Q_OSC_OBS_MODE": "penalize_unexplained_v1",
        "RESPYRE_PARH_Q_OSC_OBS_REF": "0.97",
        "RESPYRE_PARH_Q_OSC_OBS_BAND": "0.08",
    },
    "residual_release_v3": {
        "RESPYRE_PARH_Q_OSC_OBS_WEIGHT": "0.20",
        "RESPYRE_PARH_Q_OSC_OBS_MODE": "penalize_unexplained_v1",
        "RESPYRE_PARH_Q_OSC_OBS_REF": "0.97",
        "RESPYRE_PARH_Q_OSC_OBS_BAND": "0.08",
        "RESPYRE_PARH_Q_APER_OBS_GAMMA": "1.50",
    },
    "residual_release_v4": {
        "RESPYRE_PARH_Q_OSC_OBS_WEIGHT": "0.22",
        "RESPYRE_PARH_Q_OSC_OBS_MODE": "penalize_nonosc_gap_v1",
        "RESPYRE_PARH_Q_APER_OBS_GAMMA": "0.50",
    },
    "no_qdyn": {
        "RESPYRE_PARH_Q_DYN_GAMMA": "0",
    },
    "no_qosc_release": {
        "RESPYRE_PARH_Q_APER_GAMMA": "0",
    },
    "no_helper": {
        "RESPYRE_PARH_USE_HELPER_PATH": "0",
    },
    "legacy_obs_path": {
        "RESPYRE_PARH_USE_LIGHT_OBS_PATH": "0",
    },
    "no_obs_cal": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "0",
    },
}


def parse_args():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run named PARH ablation profiles.")
    parser.add_argument("--config", required=True, help="Base config JSON")
    parser.add_argument(
        "--out-root",
        default=str(root / "results" / "parh_ablations"),
        help="Root directory for ablation results",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["full", "obs_policy_quadcub_helper", "obs_policy_quadcub_dof_helper", "obs_policy_all_p1d_helper", "obs_policy_quadcub_blend", "obs_policy_quadcub_dof_blend", "obs_cal_v2", "obs_cal_v3", "obs_cal_v4", "obs_cal_v5", "obs_cal_v6", "no_h2", "no_baseline", "no_residual", "no_adapt_r", "no_student_t", "legacy_q", "no_freq_adapt", "no_freq_rescue", "freq_rescue_of_v2", "output_rate_of_v1", "output_rate_of_strong", "output_rate_of_strong_relaxed", "no_qdyn", "no_qosc_release", "no_helper", "legacy_obs_path", "no_obs_cal"],
        help="Ablation profiles to run",
    )
    parser.add_argument("--debug", action="store_true", help="Pass --debug to main.py")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for profile in args.profiles:
        if profile not in ABLATIONS:
            raise ValueError(f"Unknown ablation profile: {profile}")
        env = os.environ.copy()
        env.update(ABLATIONS[profile])
        result_dir = out_root / profile
        cmd = [
            sys.executable,
            "main.py",
            "--config",
            args.config,
            "--results",
            str(result_dir),
        ]
        if args.debug:
            cmd.append("--debug")
        print(f"\n=== Running ablation: {profile} ===")
        print(" ".join(cmd))
        if ABLATIONS[profile]:
            for k, v in sorted(ABLATIONS[profile].items()):
                print(f"  {k}={v}")
        subprocess.run(cmd, check=True, env=env, cwd=Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    main()
