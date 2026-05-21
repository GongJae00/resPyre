#!/usr/bin/env python3
"""Emit shell exports for the locked PARH paper-facing profile.

Use with:

    eval "$(python setup/locked_paper_profile.py)"

This script is intentionally conservative. It does not tune numeric
hyperparameters. It clears or pins high-risk experimental switches so a paper
run cannot inherit hidden environment overrides from an earlier probe.
"""

from __future__ import annotations

import argparse
import json
import shlex
from collections import OrderedDict


LOCKED_ENV = OrderedDict(
    [
        ("RESPYRE_PARH_ENABLE_HARMONIC2", "1"),
        ("RESPYRE_PARH_ENABLE_BASELINE", "1"),
        ("RESPYRE_PARH_ENABLE_RESIDUAL", "1"),
        ("RESPYRE_PARH_ENABLE_ADAPT_R", "1"),
        ("RESPYRE_PARH_ENABLE_DISENTANGLED_Q", "1"),
        ("RESPYRE_PARH_ENABLE_LEGACY_COUPLED_Q", "0"),
        ("RESPYRE_PARH_ENABLE_STUDENT_T", "1"),
        ("RESPYRE_PARH_ENABLE_FREQ_ADAPT", "1"),
        ("RESPYRE_PARH_USE_HELPER_PATH", "1"),
        ("RESPYRE_PARH_USE_LIGHT_OBS_PATH", "1"),
        ("RESPYRE_PARH_ENABLE_OBS_CAL", "1"),
        ("RESPYRE_PARH_ENABLE_FAMILY_CONFIDENCE", "1"),
        ("RESPYRE_PARH_ENABLE_DYNAMIC_MIXTURE", "0"),
        ("RESPYRE_PARH_ENABLE_RESIDUAL_SEMANTICS", "0"),
        ("RESPYRE_PARH_HELPER_TRUST_POLICY", "off"),
        ("RESPYRE_PARH_OBS_FAMILY_POLICY", "bridge_v1"),
        ("RESPYRE_PARH_OUTPUT_RATE_POLICY", "of_helper_blend_v1"),
        ("RESPYRE_PARH_FREQ_RESCUE_POLICY", "bridge_v1"),
    ]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="Print the locked profile as JSON instead of shell exports.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.json:
        print(json.dumps(LOCKED_ENV, indent=2))
        return 0
    print("# PARH locked paper-facing profile")
    for key, value in LOCKED_ENV.items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
