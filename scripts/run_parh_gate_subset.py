#!/usr/bin/env python3
"""Run a lightweight multi-trial COHFACE gate before expensive full reruns."""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.datasets.impl import COHFACE  # noqa: E402


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
    "obs_cal_v7_harmonic_only": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "1",
        "RESPYRE_PARH_OBS_CAL_MODE": "family_phase_aux",
        "RESPYRE_PARH_OBS_CAL_ALLOWED_FAMILIES": "profile1d_quadratic,profile1d_cubic",
        "RESPYRE_PARH_OBS_CAL_PRIOR_STRENGTH": "1.70",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_AUX": "0.00",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_H1": "1.20",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_H2": "1.60",
        "RESPYRE_PARH_OBS_CAL_MAX_LAG_SEC": "0.08",
    },
    "obs_cal_v7_lowaux": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "1",
        "RESPYRE_PARH_OBS_CAL_MODE": "family_phase_aux",
        "RESPYRE_PARH_OBS_CAL_ALLOWED_FAMILIES": "profile1d_quadratic,profile1d_cubic",
        "RESPYRE_PARH_OBS_CAL_PRIOR_STRENGTH": "1.65",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_AUX": "0.12",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_H1": "1.20",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_H2": "1.60",
        "RESPYRE_PARH_OBS_CAL_MAX_LAG_SEC": "0.08",
    },
    "obs_cal_of_velocity_v1": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "1",
        "RESPYRE_PARH_OBS_CAL_MODE": "family_phase_aux",
        "RESPYRE_PARH_OBS_CAL_ALLOWED_FAMILIES": "of_farneback,profile1d_quadratic,profile1d_cubic",
        "RESPYRE_PARH_OBS_CAL_PRIOR_STRENGTH": "1.35",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_H1": "1.25",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_H2": "1.50",
        "RESPYRE_PARH_OBS_CAL_MAX_GAIN_AUX": "0.45",
        "RESPYRE_PARH_OBS_CAL_MAX_LAG_SEC": "0.12",
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
    "family_confidence_v1": {
        "RESPYRE_PARH_ENABLE_FAMILY_CONFIDENCE": "1",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_ALLOWED_FAMILIES": "profile1d_quadratic,profile1d_cubic",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_MIN_FIT_CORR": "0.97",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_MAX_FIT_RMSE": "0.22",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_PI_FLOOR": "0.95",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_QDYN_SCALE": "0.75",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_R_SCALE": "1.00",
    },
    "family_confidence_v2": {
        "RESPYRE_PARH_ENABLE_FAMILY_CONFIDENCE": "1",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_ALLOWED_FAMILIES": "profile1d_quadratic,profile1d_cubic",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_MIN_FIT_CORR": "0.975",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_MAX_FIT_RMSE": "0.20",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_PI_FLOOR": "0.965",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_QDYN_SCALE": "0.65",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_R_SCALE": "0.92",
    },
    "family_confidence_v3": {
        "RESPYRE_PARH_ENABLE_FAMILY_CONFIDENCE": "1",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_ALLOWED_FAMILIES": "profile1d_quadratic,profile1d_cubic",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_MIN_FIT_CORR": "0.975",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_MAX_FIT_RMSE": "0.20",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_PI_FLOOR": "0.970",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_QDYN_SCALE": "0.55",
        "RESPYRE_PARH_FAMILY_CONFIDENCE_R_SCALE": "0.85",
    },
    "no_obs_cal": {
        "RESPYRE_PARH_ENABLE_OBS_CAL": "0",
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
    "profile_harmonic_rate_v1": {
        "RESPYRE_PARH_OUTPUT_RATE_POLICY": "hybrid_semantics_v1",
        "RESPYRE_PARH_PROFILE_RATE_BLEND_ALPHA": "0.18",
        "RESPYRE_PARH_PROFILE_RATE_MIN_SUPPORT": "0.95",
        "RESPYRE_PARH_PROFILE_RATE_MAX_QDYN": "0.40",
        "RESPYRE_PARH_PROFILE_RATE_MIN_MISMATCH_HZ": "0.025",
        "RESPYRE_PARH_PROFILE_RATE_MAX_MISMATCH_HZ": "0.10",
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
    "residual_semantics_v1": {
        "RESPYRE_PARH_ENABLE_RESIDUAL_SEMANTICS": "1",
        "RESPYRE_PARH_Q_OSC_OBS_WEIGHT": "0.22",
        "RESPYRE_PARH_Q_OSC_OBS_MODE": "penalize_nonosc_gap_v1",
        "RESPYRE_PARH_Q_APER_OBS_GAMMA": "0.50",
        "RESPYRE_PARH_RESIDUAL_PRIOR_MIN": "0.10",
        "RESPYRE_PARH_RESIDUAL_PRIOR_POWER": "1.00",
    },
    "assistant_of_v2": {
        "RESPYRE_PARH_ASSISTANT_POLICY": "of_rate_assistant_v2",
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
    "legacy_obs_path": {
        "RESPYRE_PARH_USE_LIGHT_OBS_PATH": "0",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PARH gate subset on COHFACE.")
    parser.add_argument("--config", required=True, help="Base config JSON")
    parser.add_argument(
        "--out-root",
        default=str(ROOT / "results" / "parh_gate_subset"),
        help="Output root for gate subset runs",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["full", "no_freq_rescue", "legacy_obs_path", "obs_policy_quadcub_helper"],
        help="Profiles to run",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=12,
        help="Number of COHFACE trials to include when --subset is omitted",
    )
    parser.add_argument(
        "--subset",
        nargs="+",
        default=None,
        help="Explicit subset keys like 1_0 5_1 12_3",
    )
    return parser.parse_args()


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_evenly_spaced(keys, num_trials: int):
    if num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if len(keys) <= num_trials:
        return keys
    if num_trials == 1:
        return [keys[len(keys) // 2]]
    idxs = sorted(
        {
            min(len(keys) - 1, int(round(i * (len(keys) - 1) / (num_trials - 1))))
            for i in range(num_trials)
        }
    )
    return [keys[i] for i in idxs]


def _default_subset(cfg: dict, num_trials: int):
    ds = COHFACE()
    cohface_cfg = None
    for d_cfg in cfg.get("datasets", []):
        if str(d_cfg.get("name", "")).lower() == "cohface":
            cohface_cfg = dict(d_cfg)
            break
    if cohface_cfg is None:
        raise ValueError("Base config does not contain a COHFACE dataset entry")
    ds.configure(cohface_cfg)
    ds.load_dataset()
    keys = []
    for item in ds.data:
        subject = str(item.get("subject", "")).strip()
        trial = str(item.get("trial", "")).strip()
        if subject and trial:
            keys.append(f"{subject}_{trial}")
    keys = sorted(dict.fromkeys(keys))
    return _select_evenly_spaced(keys, num_trials)


def _write_subset_cfg(base_cfg: dict, subset_keys):
    cfg = json.loads(json.dumps(base_cfg))
    updated = False
    for d_cfg in cfg.get("datasets", []):
        if str(d_cfg.get("name", "")).lower() == "cohface":
            d_cfg["subset"] = list(subset_keys)
            updated = True
    if not updated:
        raise ValueError("Could not inject COHFACE subset into config")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", prefix="parh_gate_subset_", delete=False)
    with tmp:
        json.dump(cfg, tmp, indent=2)
    return Path(tmp.name)


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = _load_cfg(cfg_path)
    subset_keys = list(args.subset) if args.subset else _default_subset(cfg, args.num_trials)

    for profile in args.profiles:
        if profile not in ABLATIONS:
            raise ValueError(f"Unknown profile: {profile}")

    subset_cfg = _write_subset_cfg(cfg, subset_keys)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Subset keys:", " ".join(subset_keys))
    print("Temporary config:", subset_cfg)

    try:
        for profile in args.profiles:
            env = os.environ.copy()
            env.update(ABLATIONS[profile])
            out_dir = out_root / profile
            cmd = [
                sys.executable,
                "main.py",
                "--config",
                str(subset_cfg),
                "--results",
                str(out_dir),
            ]
            print(f"\n=== Running gate subset: {profile} ===")
            print(" ".join(cmd))
            if ABLATIONS[profile]:
                for key, value in sorted(ABLATIONS[profile].items()):
                    print(f"  {key}={value}")
            subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    finally:
        try:
            subset_cfg.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
