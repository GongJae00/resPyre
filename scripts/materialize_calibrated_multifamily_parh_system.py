#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.models.heads.parh_ossm import oscillator_PARH_OSSM
from core.pipeline.evaluation_step import run_evaluation


DEFAULT_METHODS: Tuple[Tuple[str, str, str], ...] = (
    ("of_farneback", "of_farneback", "of"),
    ("of_disp_bridge", "of_disp_bridge", "of_bridge"),
    ("DoF", "dof", "dof"),
    ("dof_disp_bridge", "dof_disp_bridge", "dof_bridge"),
    ("profile1D linear", "profile1d_linear", "profile1d"),
    ("profile1D quadratic", "profile1d_quadratic", "profile1d"),
    ("profile1D cubic", "profile1d_cubic", "profile1d"),
    ("profile1d_consensus", "profile1d_consensus", "profile1d_consensus"),
)


def _rate_variant_label(method_label: str, family_name: str, source: str) -> str:
    source = _normalize_external_rate_evidence_source(source)
    if source == "base":
        return method_label
    suffix = "__kfstd" if source == "kfstd" else "__parh_ossm"
    return f"{family_name}{suffix}"


def _normalize_external_rate_evidence_source(source: str) -> str:
    source = str(source or "base").strip().lower().replace("-", "_")
    aliases = {
        "base": "base",
        "raw": "base",
        "kfstd": "kfstd",
        "kf_std": "kfstd",
        "ossm_kfstd": "kfstd",
        "ossm+kfstd": "kfstd",
        "ossm_kf": "kfstd",
        "parh": "parh",
        "parh_ossm": "parh",
    }
    if source not in aliases:
        raise ValueError(
            f"unknown external rate evidence source {source!r}; "
            "use base, ossm_kfstd, or parh_ossm"
        )
    return aliases[source]


def _public_external_rate_evidence_source(source: str) -> str:
    normalized = _normalize_external_rate_evidence_source(source)
    if normalized == "kfstd":
        return "ossm_kfstd"
    if normalized == "parh":
        return "parh_ossm"
    return "base"

DISPLACEMENT_ANCHOR_FAMILIES = {
    "of_disp_bridge",
    "dof_disp_bridge",
    "profile1d_linear",
    "profile1d_quadratic",
    "profile1d_cubic",
    "profile1d_consensus",
}

FAMILY_GROUPS = {
    "of_farneback": "G_OF",
    "of_disp_bridge": "G_OF_bridge",
    "dof": "G_DoF",
    "dof_disp_bridge": "G_DoF_bridge",
    "profile1d_linear": "G_P1D_low",
    "profile1d_quadratic": "G_P1D_morph",
    "profile1d_cubic": "G_P1D_morph",
    "profile1d_consensus": "G_P1D_cons",
}

MOTION_TIMING_TRACK_GROUPS = {"G_DoF", "G_DoF_bridge"}
STABLE_TRACK_BLEND_STABILITY = 0.80
DERIVED_PARENT_GROUPS = {
    "G_OF_bridge": "G_OF",
    "G_DoF_bridge": "G_DoF",
}
RELEASE_RATE_POSTERIOR_OUTPUTS = {"final", "calibrated_mean"}
DIAGNOSTIC_RATE_POSTERIOR_OUTPUTS = {
    "mode",
    "mean",
    "specific_calibrated_mean",
    "macroguard_specific_calibrated_mean",
    "roleaware_macroguard_specific_calibrated_mean",
    "source_validity",
    "source_validity_guarded",
    "source_arbiter_v1",
    "source_arbiter_v2",
    "source_arbiter_v3",
}


def _normalize_rate_posterior_output_source(source: str) -> str:
    source = str(source or "off").strip().lower()
    if source in {"final", "bounded"}:
        return "calibrated_mean"
    return source


def _public_rate_posterior_output_source(source: str) -> str:
    normalized = _normalize_rate_posterior_output_source(source)
    if normalized == "calibrated_mean":
        return "final"
    return normalized


def _rate_posterior_output_role(source: str) -> str:
    source = str(source or "off").strip().lower()
    if source == "off":
        return "native_state_update_only"
    if source in RELEASE_RATE_POSTERIOR_OUTPUTS:
        return "release_bounded_readout"
    if source in DIAGNOSTIC_RATE_POSTERIOR_OUTPUTS:
        return "diagnostic_or_compatibility"
    return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Materialize a GT-free target-calibrated multi-family observation stack, "
            "run PARH-OSSM on it, and optionally evaluate the synthetic run."
        )
    )
    p.add_argument("--data-dir", required=True, help="Input run data directory containing PKL trials.")
    p.add_argument("--out-run", required=True, help="Output run directory. It will contain data/ and metrics/.")
    p.add_argument("--name", default="target_calibrated_multifamily_parh_ossm")
    p.add_argument("--source-key", default="signal_hat", choices=["signal_hat", "z_osc", "z_full"])
    p.add_argument(
        "--external-rate-evidence-source",
        dest="external_rate_evidence_source",
        choices=["base", "ossm_kfstd", "parh_ossm"],
        default="base",
        help=(
            "Saved method variant used only as external GT-free timing evidence. "
            "This is not a nested fallback model; waveform observation channels "
            "and PARH-OSSM state updates are unchanged."
        ),
    )
    p.add_argument(
        "--rate-track-source",
        dest="external_rate_evidence_source",
        choices=["base", "kfstd", "parh"],
        help=argparse.SUPPRESS,
    )
    p.add_argument("--max-files", type=int, default=0, help="Optional trial limit for smoke tests.")
    p.add_argument("--jobs", type=int, default=0, help="Process workers. Default uses PARALLEL_PROCS/RESPYRE_JOBS.")
    p.add_argument("--min-hz", type=float, default=0.08)
    p.add_argument("--max-hz", type=float, default=0.50)
    p.add_argument("--lag-max-sec", type=float, default=4.0)
    p.add_argument("--min-support-corr", type=float, default=0.08)
    p.add_argument(
        "--anchor-policy",
        choices=["displacement", "all"],
        default="displacement",
        help="Restrict coordinate anchors to displacement-compatible observations by default.",
    )
    p.add_argument(
        "--canonical-policy",
        choices=["displacement", "all"],
        default="displacement",
        help="Use only displacement-compatible observations in the canonical waveform mixture by default.",
    )
    p.add_argument(
        "--parh-input",
        choices=["canonical", "multichannel"],
        default="canonical",
        help=(
            "canonical: feed one support-weighted calibrated observation to PARH "
            "and keep source channels in metadata/assistant signals. multichannel: "
            "feed all calibrated channels directly; useful for debugging but currently less stable."
        ),
    )
    p.add_argument(
        "--parh-base-method",
        default="of_disp_bridge",
        help=(
            "PARH base-family semantics for the calibrated canonical input. "
            "of_disp_bridge keeps rate-rescue enabled while remaining displacement-compatible."
        ),
    )
    p.add_argument(
        "--reliability-group-csv",
        type=Path,
        help=(
            "Optional GT-free target reliability graph group CSV. When provided, "
            "per-trial group priors modulate canonical family weights and are "
            "attached to PARH metadata for multichannel context priors."
        ),
    )
    p.add_argument(
        "--state-reliability-group-csv",
        type=Path,
        help=(
            "Optional CSV used only for canonical/state/PARH update priors. "
            "Defaults to --reliability-group-csv when omitted."
        ),
    )
    p.add_argument(
        "--readout-reliability-group-csv",
        type=Path,
        help=(
            "Optional CSV used only for decoupled z_osc rate/BPM readout and "
            "candidate-posterior evidence. Defaults to --reliability-group-csv "
            "when omitted."
        ),
    )
    p.add_argument(
        "--reliability-prior-scope",
        choices=["all", "readout_only", "state_only"],
        default="all",
        help=(
            "Scope for --reliability-group-csv. all preserves the historical behavior: "
            "the CSV can affect canonical calibration, state context, rate anchors, "
            "and decoupled readout. readout_only uses the CSV only for the decoupled "
            "z_osc rate/BPM readout arbiter fields; native target reliability columns "
            "still support z_full/state construction. state_only uses the CSV for "
            "canonical/state/PARH update priors but withholds it from the decoupled "
            "rate readout, so state-update effects can be evaluated independently."
        ),
    )
    p.add_argument(
        "--enable-phase-anchor-validation",
        action="store_true",
        help=(
            "Experimental no-go guard: downweight local rate anchors when "
            "sinusoidal phase-fit/continuity is weak. Off by default because "
            "the current MAHNOB smoke shows this guard is not yet promotable."
        ),
    )
    p.add_argument(
        "--enable-regime-observation-law",
        action="store_true",
        help=(
            "Enable the GT-free regime-aware observation law. This treats OF, "
            "DoF, and P1D as different observation equations, estimates a "
            "window-local regime, and exports channel context / anchor "
            "confidence to PARH instead of using reliability scores as a "
            "plain clean-signal selector."
        ),
    )
    p.add_argument(
        "--enable-observation-law",
        dest="enable_observation_law",
        action="store_true",
        help=(
            "Enable the final PARH-OSSM observation-law path. This requires "
            "--parh-input multichannel, enables regime context, dynamic soft "
            "observation mixture, rate-observability evidence, residual semantics, "
            "residual identifiability guarding, phase-anchored morphology, "
            "group-balanced fusion, and stores activation diagnostics. It is "
            "not an OSSM-KF fallback."
        ),
    )
    p.add_argument(
        "--enable-observation-law-v2",
        dest="enable_observation_law",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--regime-anchor-policy",
        choices=["context_only", "replace"],
        default="context_only",
        help=(
            "How the regime law affects the local z_osc anchor. context_only "
            "is the safe default: use the regime law for channel precision but "
            "keep the spectral+harmonic anchor. replace is a no-go diagnostic "
            "that lets the regime law replace the anchor track."
        ),
    )
    p.add_argument(
        "--enable-decoupled-rate-readout",
        action="store_true",
        help=(
            "Export a GT-free z_osc timing readout track from the observation "
            "law. This does not change z_full waveform reconstruction; it only "
            "lets PARH report rate/BPM from the best supported observation "
            "equation when target-side evidence is strong."
        ),
    )
    p.add_argument("--enable-rate-hypothesis-graph", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--enable-rate-hypothesis-graph-v4",
        dest="enable_rate_hypothesis_graph",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--enable-derived-consistency-scaling",
        action="store_true",
        help=(
            "Diagnostic only: downweight derived bridge observations when they "
            "disagree with parent/peer groups. Off by default because the "
            "2026-05-02 MAHNOB smoke regressed median MAE."
        ),
    )
    p.add_argument(
        "--enable-rate-posterior",
        action="store_true",
        help=(
            "Experimental: export a GT-free candidate-frequency posterior "
            "over observation-class hypotheses. PARH uses it as a weak h1 "
            "uncertainty/anchor signal, not as a hard class selector."
        ),
    )
    p.add_argument(
        "--enable-target-observability-control",
        action="store_true",
        help=(
            "Experimental structural mode: build GT-free target observability "
            "controls from source spread, posterior specificity, alias safety, "
            "and readout confidence, then pass them into PARH before the state "
            "update. This is not a final-output source override."
        ),
    )
    p.add_argument(
        "--enable-signal-sqi-observability",
        action="store_true",
        help=(
            "Experimental structural mode: estimate target-computable signal "
            "quality indices from candidate observations (spectral peakiness, "
            "autocorrelation periodicity, sinusoidal fit, and phase coherence) "
            "and feed them into the posterior/observability law. This adds "
            "evidence quality, not a GT-tuned source selector."
        ),
    )
    p.add_argument(
        "--rate-posterior-output-source",
        default="final",
        help=(
            "Select how the candidate-rate posterior affects the decoupled "
            "z_osc rate readout. Use 'final' for the release bounded "
            "readout, or 'off' to keep posterior evidence as state-update/"
            "diagnostic context without overriding the readout. Legacy "
            "diagnostic names are accepted for old analysis scripts but are "
            "not release."
        ),
    )
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--eval-use-track",
        action="store_true",
        help=(
            "Evaluate rate accuracy from track_hz instead of the spectrum of "
            "signal_hat. Use this for decoupled z_osc timing-readout probes."
        ),
    )
    p.add_argument("--win-size", type=float, default=30.0)
    p.add_argument("--stride", type=float, default=1.0)
    p.add_argument("--artifact-policy", choices=["full", "lean", "smoke"], default="full")
    args = p.parse_args()
    args.rate_track_source = _normalize_external_rate_evidence_source(args.external_rate_evidence_source)
    args.external_rate_evidence_source = _public_external_rate_evidence_source(args.rate_track_source)
    return args


def _artifact_policy_settings(policy: str) -> Dict[str, object]:
    table = {
        "full": {"keep_data": True, "save_metric_pickles": True},
        "lean": {"keep_data": False, "save_metric_pickles": False},
        "smoke": {"keep_data": False, "save_metric_pickles": False},
    }
    return dict(table[str(policy).strip().lower()])


def _prune_run_artifacts(run_dir: Path, *, keep_data: bool, save_metric_pickles: bool) -> None:
    if not keep_data:
        shutil.rmtree(run_dir / "data", ignore_errors=True)
    if not save_metric_pickles:
        metrics_dir = run_dir / "metrics"
        if metrics_dir.exists():
            for p in metrics_dir.glob("metrics_*.pkl"):
                p.unlink(missing_ok=True)


def _load_pkl(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_reliability_priors(path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"reliability group CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"video", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"reliability group CSV missing columns: {sorted(missing)}")
    score_col = "soft_group_weight" if "soft_group_weight" in df.columns else "reliability_score"
    if score_col not in df.columns:
        raise ValueError("reliability group CSV must contain soft_group_weight or reliability_score")
    out: Dict[str, Dict[str, float]] = {}
    for video, sub in df.groupby("video"):
        rows: Dict[str, float] = {}
        for _, row in sub.iterrows():
            group = str(row.get("group", "")).strip()
            try:
                score = float(row.get(score_col, np.nan))
            except Exception:
                score = float("nan")
            if group and np.isfinite(score) and score > 0.0:
                rows[group] = max(rows.get(group, 0.0), float(score))
        if rows:
            total = float(sum(rows.values()))
            if total > 1e-12 and score_col != "soft_group_weight":
                rows = {k: float(v / total) for k, v in rows.items()}
            out[str(video)] = rows
    return out


def _load_windowed_reliability_priors(path: Optional[Path]) -> Dict[str, List[Dict[str, float]]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"reliability group CSV not found: {path}")
    df = pd.read_csv(path)
    if not {"video", "group"}.issubset(df.columns):
        return {}
    if "window_start_sec" not in df.columns and "start_sec" not in df.columns:
        return {}
    if "window_end_sec" not in df.columns and "end_sec" not in df.columns:
        return {}
    score_col = "soft_group_weight" if "soft_group_weight" in df.columns else "reliability_score"
    if score_col not in df.columns:
        return {}
    out: Dict[str, List[Dict[str, float]]] = {}
    for _, row in df.iterrows():
        video = str(row.get("video", "")).strip()
        group = str(row.get("group", "")).strip()
        try:
            start = float(row.get("window_start_sec", row.get("start_sec", np.nan)))
            end = float(row.get("window_end_sec", row.get("end_sec", np.nan)))
            score = float(row.get(score_col, np.nan))
            reliability_score = float(row.get("reliability_score", score))
            timing_reliability_score_v3 = float(row.get("timing_reliability_score_v3", np.nan))
            timing_reliability_score = float(row.get("timing_reliability_score", np.nan))
            morphology_reliability_score = float(row.get("morphology_reliability_score", np.nan))
            arbiter_timing_score = float(row.get("arbiter_timing_score", np.nan))
            arbiter_timing_weight = float(row.get("arbiter_timing_weight", np.nan))
            arbiter_morphology_score = float(row.get("arbiter_morphology_score", np.nan))
            arbiter_morphology_weight = float(row.get("arbiter_morphology_weight", np.nan))
            arbiter_readout_conf = float(row.get("arbiter_readout_conf", np.nan))
            arbiter_abstain_score = float(row.get("arbiter_abstain_score", np.nan))
            event_timing_score = float(row.get("event_timing_score", np.nan))
            macro_timing_support_score = float(row.get("macro_timing_support_score", np.nan))
            bridge_timing_score = float(row.get("bridge_timing_score", np.nan))
            abstain_score = float(row.get("abstain_score", np.nan))
            rate_phase_score = float(row.get("rate_phase_support_score", np.nan))
            support_group_count = float(row.get("support_group_count", np.nan))
            possible_support_group_count = float(row.get("possible_support_group_count", np.nan))
            h1_timing_weight = float(row.get("h1_timing_weight", np.nan))
            h2_harmonic_weight = float(row.get("h2_harmonic_weight", np.nan))
            b_baseline_weight = float(row.get("b_baseline_weight", np.nan))
            r_residual_weight = float(row.get("r_residual_weight", np.nan))
            component_abstain_score = float(row.get("component_abstain_score", np.nan))
            z_osc_readout_weight = float(row.get("z_osc_readout_weight", np.nan))
            z_full_readout_weight = float(row.get("z_full_readout_weight", np.nan))
        except Exception:
            continue
        if not video or not group or not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        if not np.isfinite(score) or score <= 0.0:
            continue
        out.setdefault(video, []).append(
            {
                "group": group,
                "start_sec": start,
                "end_sec": end,
                "score": score,
                "reliability_score": reliability_score if np.isfinite(reliability_score) else score,
                "timing_reliability_score_v3": (
                    timing_reliability_score_v3
                    if np.isfinite(timing_reliability_score_v3)
                    else (
                        timing_reliability_score
                        if np.isfinite(timing_reliability_score)
                        else (reliability_score if np.isfinite(reliability_score) else score)
                    )
                ),
                "timing_reliability_score": (
                    timing_reliability_score
                    if np.isfinite(timing_reliability_score)
                    else (reliability_score if np.isfinite(reliability_score) else score)
                ),
                "morphology_reliability_score": (
                    morphology_reliability_score
                    if np.isfinite(morphology_reliability_score)
                    else (reliability_score if np.isfinite(reliability_score) else score)
                ),
                "arbiter_timing_score": arbiter_timing_score if np.isfinite(arbiter_timing_score) else float("nan"),
                "arbiter_timing_weight": arbiter_timing_weight if np.isfinite(arbiter_timing_weight) else float("nan"),
                "arbiter_morphology_score": (
                    arbiter_morphology_score if np.isfinite(arbiter_morphology_score) else float("nan")
                ),
                "arbiter_morphology_weight": (
                    arbiter_morphology_weight if np.isfinite(arbiter_morphology_weight) else float("nan")
                ),
                "arbiter_readout_conf": arbiter_readout_conf if np.isfinite(arbiter_readout_conf) else float("nan"),
                "arbiter_abstain_score": arbiter_abstain_score if np.isfinite(arbiter_abstain_score) else float("nan"),
                "event_timing_score": event_timing_score if np.isfinite(event_timing_score) else float("nan"),
                "macro_timing_support_score": (
                    macro_timing_support_score if np.isfinite(macro_timing_support_score) else float("nan")
                ),
                "bridge_timing_score": bridge_timing_score if np.isfinite(bridge_timing_score) else float("nan"),
                "abstain_score": abstain_score if np.isfinite(abstain_score) else float("nan"),
                "rate_phase_score": rate_phase_score if np.isfinite(rate_phase_score) else 1.0,
                "support_group_count": support_group_count if np.isfinite(support_group_count) else 0.0,
                "possible_support_group_count": possible_support_group_count if np.isfinite(possible_support_group_count) else 0.0,
                "h1_timing_weight": h1_timing_weight if np.isfinite(h1_timing_weight) else float("nan"),
                "h2_harmonic_weight": h2_harmonic_weight if np.isfinite(h2_harmonic_weight) else float("nan"),
                "b_baseline_weight": b_baseline_weight if np.isfinite(b_baseline_weight) else float("nan"),
                "r_residual_weight": r_residual_weight if np.isfinite(r_residual_weight) else float("nan"),
                "component_abstain_score": (
                    component_abstain_score if np.isfinite(component_abstain_score) else float("nan")
                ),
                "z_osc_readout_weight": z_osc_readout_weight if np.isfinite(z_osc_readout_weight) else float("nan"),
                "z_full_readout_weight": z_full_readout_weight if np.isfinite(z_full_readout_weight) else float("nan"),
            }
        )
    return out


def _finite_row_float(row: Dict[str, float], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _timing_reliability_from_row(row: Dict[str, float], fallback: float = float("nan")) -> float:
    for key in (
        "h1_timing_weight",
        "z_osc_readout_weight",
        "timing_reliability_score_v3",
        "timing_reliability_score",
        "reliability_score",
        "score",
    ):
        value = _finite_row_float(row, key)
        if np.isfinite(value) and value > 0.0:
            return float(value)
    return float(fallback)


def _readout_timing_reliability_from_row(row: Dict[str, float], fallback: float = float("nan")) -> float:
    for key in (
        "z_osc_readout_weight",
        "h1_timing_weight",
        "arbiter_timing_score",
        "timing_reliability_score_v3",
        "timing_reliability_score",
        "reliability_score",
        "score",
    ):
        value = _finite_row_float(row, key)
        if np.isfinite(value) and value > 0.0:
            return float(value)
    return float(fallback)


def _morphology_reliability_from_row(row: Dict[str, float], fallback: float = float("nan")) -> float:
    for key in (
        "z_full_readout_weight",
        "h2_harmonic_weight",
        "r_residual_weight",
        "morphology_reliability_score",
        "reliability_score",
        "score",
    ):
        value = _finite_row_float(row, key)
        if np.isfinite(value) and value > 0.0:
            return float(value)
    return float(fallback)


def _channel_reliability_priors(names: Sequence[str], group_priors: Dict[str, float]) -> Dict[str, float]:
    if not group_priors:
        return {}
    priors: Dict[str, float] = {}
    for name in names:
        group = FAMILY_GROUPS.get(str(name), str(name))
        val = float(group_priors.get(group, np.nan))
        if np.isfinite(val) and val > 0.0:
            priors[str(name)] = val
    return priors


def _split_reliability_scope(
    scope: str,
    trial_priors: Dict[str, float],
    window_rows: Sequence[Dict[str, float]],
) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, float], List[Dict[str, float]]]:
    """Separate state-building priors from decoupled readout priors.

    Source-supervised arbiter scores are valid readout evidence, but they must
    not silently change the canonical waveform mixture or PARH state update.
    Native target-side reliability columns remain available to the state path.
    """
    state_trial_priors = dict(trial_priors or {})
    state_window_rows = [dict(row) for row in (window_rows or [])]
    readout_trial_priors = dict(trial_priors or {})
    readout_window_rows = [dict(row) for row in (window_rows or [])]
    if str(scope).strip().lower() == "readout_only":
        state_window_rows = [
            {key: value for key, value in row.items() if not str(key).startswith("arbiter_")}
            for row in state_window_rows
        ]
    elif str(scope).strip().lower() == "state_only":
        readout_trial_priors = {}
        readout_window_rows = []
    return state_trial_priors, state_window_rows, readout_trial_priors, readout_window_rows


def _channel_context_prior_runtime(
    names: Sequence[str],
    window_rows: Sequence[Dict[str, float]],
    n_frames: int,
    fps: float,
    fallback_priors: Dict[str, float],
) -> Optional[List[List[float]]]:
    n_channels = len(names)
    n = int(n_frames)
    if n_channels <= 0 or n <= 0 or not window_rows:
        return None
    score_sum = np.zeros((n_channels, n), dtype=np.float64)
    count = np.zeros((n_channels, n), dtype=np.float64)
    groups = [FAMILY_GROUPS.get(str(name), str(name)) for name in names]
    for row in window_rows:
        group = str(row.get("group", ""))
        try:
            score = _morphology_reliability_from_row(row, fallback=float(row.get("score", np.nan)))
            start = int(max(0, round(float(row.get("start_sec", 0.0)) * float(fps))))
            end = int(min(n, round(float(row.get("end_sec", 0.0)) * float(fps))))
        except Exception:
            continue
        if not np.isfinite(score) or score <= 0.0 or end <= start:
            continue
        for idx, channel_group in enumerate(groups):
            if channel_group == group:
                score_sum[idx, start:end] += score
                count[idx, start:end] += 1.0
    mat = np.full((n_channels, n), np.nan, dtype=np.float64)
    valid = count > 0.0
    mat[valid] = score_sum[valid] / count[valid]
    for idx, name in enumerate(names):
        fallback = float(fallback_priors.get(str(name), np.nan))
        if not np.isfinite(fallback) or fallback <= 0.0:
            fallback = 1.0
        row = mat[idx]
        row[~np.isfinite(row)] = fallback
        mat[idx] = row
    return mat.tolist()


def _state_role_prior_runtime(
    names: Sequence[str],
    window_rows: Sequence[Dict[str, float]],
    n_frames: int,
    fps: float,
) -> Dict[str, List[List[float]]]:
    """Build per-channel, per-time state-role reliability matrices.

    Unlike the older channel context prior, these matrices keep the reliability
    of the fundamental oscillator, harmonic scaffold, baseline, and residual as
    separate signals. PARH can then combine them using the actual observation
    row `H` instead of collapsing a family into one scalar winner.
    """
    role_cols = {
        "h1": "h1_timing_weight",
        "h2": "h2_harmonic_weight",
        "b": "b_baseline_weight",
        "r": "r_residual_weight",
        "abstain": "component_abstain_score",
        "z_osc": "z_osc_readout_weight",
        "z_full": "z_full_readout_weight",
    }
    n_channels = len(names)
    n = int(n_frames)
    if n_channels <= 0 or n <= 0 or not window_rows:
        return {}
    groups = [FAMILY_GROUPS.get(str(name), str(name)) for name in names]
    accum = {
        role: np.full((n_channels, n), np.nan, dtype=np.float64)
        for role in role_cols
    }
    counts = {
        role: np.zeros((n_channels, n), dtype=np.float64)
        for role in role_cols
    }
    sums = {
        role: np.zeros((n_channels, n), dtype=np.float64)
        for role in role_cols
    }
    for row in window_rows:
        group = str(row.get("group", ""))
        try:
            start = int(max(0, round(float(row.get("start_sec", 0.0)) * float(fps))))
            end = int(min(n, round(float(row.get("end_sec", 0.0)) * float(fps))))
        except Exception:
            continue
        if end <= start:
            continue
        idxs = [idx for idx, channel_group in enumerate(groups) if channel_group == group]
        if not idxs:
            continue
        for role, col in role_cols.items():
            try:
                value = float(row.get(col, np.nan))
            except Exception:
                value = float("nan")
            if not np.isfinite(value):
                continue
            value = float(np.clip(value, 0.0, 1.0))
            for idx in idxs:
                sums[role][idx, start:end] += value
                counts[role][idx, start:end] += 1.0
    out: Dict[str, List[List[float]]] = {}
    for role in role_cols:
        mat = accum[role]
        valid = counts[role] > 0.0
        mat[valid] = sums[role][valid] / counts[role][valid]
        if not np.any(np.isfinite(mat)):
            continue
        default = 0.0 if role == "abstain" else 1.0
        row_fill = np.nanmedian(mat, axis=1)
        row_fill = np.where(np.isfinite(row_fill), row_fill, default)
        bad = ~np.isfinite(mat)
        if np.any(bad):
            rows, _cols = np.where(bad)
            mat[bad] = row_fill[rows]
        out[role] = np.clip(mat, 0.0, 1.0).tolist()
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray, *, fallback: float = np.nan) -> float:
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


def _harmonic_candidate_support(candidate_hz: float, observed_hz: float, rate_ref: float) -> float:
    if not np.isfinite(candidate_hz) or not np.isfinite(observed_hz):
        return 0.0
    ref = max(float(rate_ref), 1e-6)
    support = 0.0
    # A true respiratory fundamental often appears as stronger second/third
    # harmonic in motion-derived observations. Keep that relation explicit
    # instead of treating the largest peak as the rate by default.
    for multiplier, penalty in ((1.0, 1.00), (2.0, 0.92), (3.0, 0.78), (0.5, 0.55)):
        support = max(
            support,
            float(penalty) * float(np.exp(-0.5 * ((float(observed_hz) - multiplier * float(candidate_hz)) / ref) ** 2)),
        )
    return float(np.clip(support, 0.0, 1.0))


def _harmonic_rate_anchor(
    rates: np.ndarray,
    weights: np.ndarray,
    *,
    min_hz: float,
    max_hz: float,
    rate_ref: float,
) -> Tuple[float, float]:
    rate_arr = np.asarray(rates, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    ok = (
        np.isfinite(rate_arr)
        & np.isfinite(weight_arr)
        & (weight_arr > 0.0)
        & (rate_arr >= float(min_hz))
        & (rate_arr <= float(max_hz))
    )
    if not np.any(ok):
        return float("nan"), 0.0
    rate_arr = rate_arr[ok]
    weight_arr = weight_arr[ok]
    candidate_rows: List[Tuple[float, float]] = []
    for observed in rate_arr:
        for divisor, prior in ((1.0, 1.00), (2.0, 0.92), (3.0, 0.78)):
            candidate = float(observed) / divisor
            if float(min_hz) <= candidate <= float(max_hz):
                candidate_rows.append((candidate, prior))
        candidate = 2.0 * float(observed)
        if float(min_hz) <= candidate <= float(max_hz):
            candidate_rows.append((candidate, 0.55))
    if not candidate_rows:
        return _weighted_median(rate_arr, weight_arr, fallback=float(np.nanmedian(rate_arr))), 0.0

    total_weight = max(float(np.sum(weight_arr)), 1e-12)
    scored: List[Tuple[float, float]] = []
    for candidate, alias_prior in candidate_rows:
        support = 0.0
        for observed, weight in zip(rate_arr, weight_arr):
            support += float(weight) * _harmonic_candidate_support(candidate, observed, rate_ref)
        support = support / total_weight
        support *= 0.85 + 0.15 * float(alias_prior)
        scored.append((float(candidate), float(support)))
    best_support = max(score for _, score in scored)
    if not np.isfinite(best_support) or best_support <= 0.0:
        return _weighted_median(rate_arr, weight_arr, fallback=float(np.nanmedian(rate_arr))), 0.0
    # Fundamental-first tie break: if a lower harmonic explains the window
    # almost as well as a high-rate peak, use the lower candidate as z_osc.
    near_best = [candidate for candidate, score in scored if score >= 0.97 * best_support]
    anchor = min(near_best) if near_best else max(scored, key=lambda row: row[1])[0]
    support = 0.0
    for observed, weight in zip(rate_arr, weight_arr):
        support += float(weight) * _harmonic_candidate_support(anchor, observed, rate_ref)
    support = support / total_weight
    return float(np.clip(anchor, float(min_hz), float(max_hz))), float(np.clip(support, 0.0, 1.0))


def _direct_rate_anchor(
    rates: np.ndarray,
    weights: np.ndarray,
    *,
    min_hz: float,
    max_hz: float,
    rate_ref: float,
) -> Tuple[float, float]:
    rate_arr = np.asarray(rates, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    ok = (
        np.isfinite(rate_arr)
        & np.isfinite(weight_arr)
        & (weight_arr > 0.0)
        & (rate_arr >= float(min_hz))
        & (rate_arr <= float(max_hz))
    )
    if not np.any(ok):
        return float("nan"), 0.0
    rate_arr = rate_arr[ok]
    weight_arr = weight_arr[ok]
    anchor = _weighted_median(rate_arr, weight_arr, fallback=float(np.nanmedian(rate_arr)))
    if not np.isfinite(anchor):
        return float("nan"), 0.0
    total_weight = max(float(np.sum(weight_arr)), 1e-12)
    support = 0.0
    ref = max(float(rate_ref), 1e-6)
    for observed, weight in zip(rate_arr, weight_arr):
        support += float(weight) * float(np.exp(-0.5 * ((float(observed) - float(anchor)) / ref) ** 2))
    support /= total_weight
    return float(np.clip(anchor, float(min_hz), float(max_hz))), float(np.clip(support, 0.0, 1.0))


def _track_first_rate_anchor(
    rates: np.ndarray,
    weights: np.ndarray,
    *,
    min_hz: float,
    max_hz: float,
    rate_ref: float,
) -> Tuple[float, float, str, float, float]:
    """Select a local rate anchor without letting harmonics overrule tracks.

    Direct track consensus is the default because `track_hz` is already the
    declared timing readout. Harmonic demotion is allowed only when direct
    agreement is weak and the harmonic explanation is substantially stronger.
    """
    direct, direct_support = _direct_rate_anchor(
        rates,
        weights,
        min_hz=float(min_hz),
        max_hz=float(max_hz),
        rate_ref=float(rate_ref),
    )
    harmonic, harmonic_support = _harmonic_rate_anchor(
        rates,
        weights,
        min_hz=float(min_hz),
        max_hz=float(max_hz),
        rate_ref=float(rate_ref),
    )
    if not np.isfinite(direct):
        if np.isfinite(harmonic):
            return float(harmonic), float(harmonic_support), "harmonic_fallback", float(direct_support), float(harmonic_support)
        return float("nan"), 0.0, "empty", float(direct_support), float(harmonic_support)
    use_harmonic = (
        np.isfinite(harmonic)
        and harmonic < 0.72 * direct
        and direct >= 0.18
        and direct_support <= 0.45
        and harmonic_support >= max(0.80, direct_support + 0.35)
    )
    if use_harmonic:
        return float(harmonic), float(harmonic_support), "harmonic_demote", float(direct_support), float(harmonic_support)
    return float(direct), float(direct_support), "direct_track_consensus", float(direct_support), float(harmonic_support)


def _direct_candidate_support(candidate: float, observed: float, rate_ref: float) -> float:
    if not (np.isfinite(candidate) and np.isfinite(observed)):
        return 0.0
    return float(np.exp(-0.5 * ((float(observed) - float(candidate)) / max(float(rate_ref), 1e-6)) ** 2))


def _group_aware_half_rate_rescue(
    anchor: float,
    rates: np.ndarray,
    weights: np.ndarray,
    groups: Sequence[str],
    *,
    min_hz: float,
    max_hz: float,
    rate_ref: float,
) -> Tuple[float, bool, float]:
    """Promote a P1D-dominated half-rate anchor when motion groups support 2x.

    P1D morphology can lock onto every other respiratory cycle under weak
    target observability. The rescue is deliberately group-aware: it only fires
    when P1D supports the low candidate while OF/DoF motion evidence supports
    the doubled candidate directly. This avoids using a blind harmonic tweak as
    a dataset-specific hyperparameter.
    """
    if not np.isfinite(anchor) or anchor <= 0.0:
        return float(anchor), False, 0.0
    doubled = 2.0 * float(anchor)
    if doubled > float(max_hz) or doubled < float(min_hz):
        return float(anchor), False, 0.0
    rate_arr = np.asarray(rates, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if rate_arr.size != weight_arr.size or rate_arr.size != len(groups):
        return float(anchor), False, 0.0

    motion_groups = {"G_OF", "G_OF_bridge", "G_DoF", "G_DoF_bridge"}
    p1d_groups = {"G_P1D_low", "G_P1D_morph", "G_P1D_cons"}
    motion_low = motion_high = motion_weight = 0.0
    p1d_low = p1d_high = p1d_weight = 0.0
    total_low = total_high = total_weight = 0.0
    for group, rate, weight in zip(groups, rate_arr, weight_arr):
        if not (np.isfinite(rate) and np.isfinite(weight) and weight > 0.0):
            continue
        low = _direct_candidate_support(float(anchor), float(rate), float(rate_ref))
        high = _direct_candidate_support(float(doubled), float(rate), float(rate_ref))
        total_low += float(weight) * low
        total_high += float(weight) * high
        total_weight += float(weight)
        if group in motion_groups:
            motion_low += float(weight) * low
            motion_high += float(weight) * high
            motion_weight += float(weight)
        elif group in p1d_groups:
            p1d_low += float(weight) * low
            p1d_high += float(weight) * high
            p1d_weight += float(weight)
    if total_weight <= 1e-12 or motion_weight <= 1e-12 or p1d_weight <= 1e-12:
        return float(anchor), False, 0.0

    motion_low /= motion_weight
    motion_high /= motion_weight
    p1d_low /= p1d_weight
    p1d_high /= p1d_weight
    total_low /= total_weight
    total_high /= total_weight
    rescue_strength = float(np.clip((motion_high - motion_low) * (p1d_low - p1d_high), 0.0, 1.0))
    should_rescue = (
        motion_high >= 0.44
        and motion_high >= motion_low + 0.12
        and p1d_low >= 0.42
        and p1d_low >= p1d_high + 0.15
        and total_high >= 0.35 * max(total_low, 1e-6)
    )
    return (float(doubled), True, rescue_strength) if should_rescue else (float(anchor), False, rescue_strength)


def _macro_timing_family(group: str) -> str:
    if group in {"G_OF", "G_OF_bridge"}:
        return "OF"
    if group in {"G_DoF", "G_DoF_bridge"}:
        return "DoF"
    if group in {"G_P1D_low", "G_P1D_morph", "G_P1D_cons"}:
        return "P1D"
    return str(group)


def _independent_timing_witness_score(macro_direct_support: Dict[str, float]) -> float:
    """Score independent h1 timing witnesses without letting one family saturate.

    This is intentionally stricter than ``macro_count``.  Correlated variants
    inside one macro family are first collapsed, then the score requires at
    least two independent macro families to support the same candidate.  A lone
    very strong family is useful evidence, but it should not masquerade as a
    robust cross-environment timing witness.
    """
    vals = [
        float(np.clip(value, 0.0, 1.0))
        for key, value in macro_direct_support.items()
        if key in {"OF", "DoF", "P1D"} and np.isfinite(value)
    ]
    if not vals:
        return 0.0
    vals = sorted(vals, reverse=True)
    if len(vals) == 1:
        return float(0.35 * vals[0])
    top, second = vals[0], vals[1]
    balance = second / max(top, 1e-12)
    return float(np.clip(math.sqrt(max(top * second, 0.0)) * (0.55 + 0.45 * balance), 0.0, 1.0))


def _motion_timing_witness_score(macro_direct_support: Dict[str, float]) -> float:
    vals = [
        float(np.clip(macro_direct_support.get(key, 0.0), 0.0, 1.0))
        for key in ("OF", "DoF")
    ]
    if not vals:
        return 0.0
    return float(np.clip(max(vals), 0.0, 1.0))


def _bridge_timing_preservation_score(
    candidate: float,
    group_rates: Dict[str, float],
    group_weights: Dict[str, float],
    *,
    rate_ref: float,
) -> float:
    """Measure whether derived bridge observations preserve parent timing.

    A bridge channel is helpful only when its timing remains compatible with
    the direct parent.  This target-computable score makes that explicit rather
    than treating a smoothed bridge as an independent vote.
    """
    scores: List[float] = []
    for bridge, parent in DERIVED_PARENT_GROUPS.items():
        if bridge not in group_rates or parent not in group_rates:
            continue
        bridge_rate = float(group_rates[bridge])
        parent_rate = float(group_rates[parent])
        bridge_w = float(group_weights.get(bridge, 0.0))
        parent_w = float(group_weights.get(parent, 0.0))
        if not (
            np.isfinite(bridge_rate)
            and np.isfinite(parent_rate)
            and np.isfinite(bridge_w)
            and np.isfinite(parent_w)
            and bridge_w > 0.0
            and parent_w > 0.0
        ):
            continue
        pair_agree = _direct_candidate_support(bridge_rate, parent_rate, float(rate_ref))
        bridge_direct = _direct_candidate_support(float(candidate), bridge_rate, float(rate_ref))
        parent_direct = _direct_candidate_support(float(candidate), parent_rate, float(rate_ref))
        weight_balance = min(bridge_w, parent_w) / max(max(bridge_w, parent_w), 1e-12)
        scores.append(float(pair_agree * math.sqrt(max(bridge_direct * parent_direct, 0.0)) * (0.50 + 0.50 * weight_balance)))
    return float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.0


def _preserve_trustworthy_track_rate(
    group: str,
    track_rate: float,
    stability: float,
    anchored_rate: float,
    *,
    min_hz: float,
    max_hz: float,
) -> Tuple[float, str]:
    """Keep target-computable motion timing from being reinterpreted away.

    DoF-like observations are not smooth morphology proxies; they are bursty
    motion-timing evidence. When their saved rate track is locally available,
    the track itself is the h1 timing observation. Reliability is handled by
    the separate group score; windowed standard deviation must not veto genuine
    breathing-rate changes or filter adaptation. Other groups retain the more
    conservative harmonic anchor with the old high-stability blend.
    """
    if not np.isfinite(track_rate):
        return float(anchored_rate), "harmonic_anchor"
    track = float(np.clip(track_rate, float(min_hz), float(max_hz)))
    if group in MOTION_TIMING_TRACK_GROUPS:
        return track, "motion_timing_track_preserved"
    if float(stability) >= STABLE_TRACK_BLEND_STABILITY and np.isfinite(anchored_rate):
        blended = 0.65 * track + 0.35 * float(anchored_rate)
        return float(np.clip(blended, float(min_hz), float(max_hz))), "stable_track_blend"
    return float(anchored_rate), "harmonic_anchor"


def _derived_consistency_score_scale(
    group: str,
    group_rates: Dict[str, float],
    group_scores: Dict[str, float],
    *,
    rate_ref: float,
) -> Tuple[float, str]:
    """Discount derived bridge observations when their source family disagrees.

    OF_bridge and DoF_bridge are useful only as parent-compatible observation
    enrichments. They are not independent sensors. A bridge may still dominate
    when other families support its timing, but an isolated bridge should not
    override its parent raw family.
    """
    parent = DERIVED_PARENT_GROUPS.get(str(group))
    if not parent or group not in group_rates:
        return 1.0, "native_or_untracked"
    child_rate = float(group_rates[group])
    if not np.isfinite(child_rate):
        return 0.35, "derived_invalid"

    support_num = 0.0
    support_den = 0.0
    parent_support = float("nan")
    for peer, rate in group_rates.items():
        if peer == group:
            continue
        weight = float(group_scores.get(peer, 0.0))
        if not (np.isfinite(rate) and np.isfinite(weight) and weight > 0.0):
            continue
        support = float(np.exp(-0.5 * ((child_rate - float(rate)) / max(float(rate_ref), 1e-6)) ** 2))
        support_num += weight * support
        support_den += weight
        if peer == parent:
            parent_support = support

    peer_support = support_num / support_den if support_den > 1e-12 else 0.0
    if not np.isfinite(parent_support):
        parent_support = peer_support

    if parent_support >= 0.50 or peer_support >= 0.45:
        return 1.0, "derived_supported"
    if parent_support >= 0.30 or peer_support >= 0.30:
        return 0.65, "derived_weakly_supported"
    return 0.35, "derived_isolated"


def _apply_derived_consistency_scaling(
    group_rates: Dict[str, float],
    group_scores: Dict[str, float],
    *,
    rate_ref: float,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for group in list(group_rates):
        scale, mode = _derived_consistency_score_scale(
            group,
            group_rates,
            group_scores,
            rate_ref=rate_ref,
        )
        if scale < 1.0 and group in group_scores:
            group_scores[group] = float(group_scores[group]) * float(scale)
        counts[mode] = int(counts.get(mode, 0)) + 1
    return counts


def _source_validity_peer_support(
    group: str,
    group_rate: float,
    group_rates: Dict[str, float],
    group_scores: Dict[str, float],
    *,
    rate_ref: float,
) -> float:
    if not np.isfinite(group_rate):
        return 0.0
    own_macro = _macro_timing_family(group)
    num = 0.0
    den = 0.0
    for peer, rate in group_rates.items():
        if peer == group or _macro_timing_family(peer) == own_macro:
            continue
        weight = float(group_scores.get(peer, 0.0))
        if not (np.isfinite(rate) and np.isfinite(weight) and weight > 0.0):
            continue
        support = _direct_candidate_support(float(group_rate), float(rate), float(rate_ref))
        num += weight * support
        den += weight
    return float(np.clip(num / den, 0.0, 1.0)) if den > 1e-12 else 0.0


def _source_validity_rate_readout_runtime(
    names: Sequence[str],
    signals: Sequence[np.ndarray],
    rate_tracks: Sequence[np.ndarray],
    window_rows: Sequence[Dict[str, float]],
    n_frames: int,
    fps: float,
    fallback_priors: Dict[str, float],
    min_hz: float,
    max_hz: float,
) -> Tuple[Optional[List[float]], Optional[List[float]], Dict[str, object]]:
    """GT-free source-validity posterior over observation groups.

    This is intentionally not a hard selector. It asks which observation
    source is locally valid, then reports a conservative source-posterior
    readout. The posterior uses only target-computable features already
    exported by the reliability graph plus intra-window source agreement.
    """
    n = int(max(n_frames, 0))
    if n <= 0 or fps <= 0.0 or not names or not window_rows:
        return None, None, {"enabled": False, "reason": "missing_inputs"}

    sig_rows = [_fit_length(np.asarray(sig, dtype=np.float64), n, fill_value=0.0) for sig in signals]
    rate_rows = []
    for track in rate_tracks:
        arr = _fit_length(np.asarray(track, dtype=np.float64), n)
        arr = np.where(
            np.isfinite(arr) & (arr >= float(min_hz)) & (arr <= float(max_hz)),
            arr,
            np.nan,
        )
        rate_rows.append(arr)
    if not sig_rows or not rate_rows:
        return None, None, {"enabled": False, "reason": "empty_inputs"}
    signal_mat = np.vstack(sig_rows)
    rate_mat = np.vstack(rate_rows)

    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        groups.setdefault(FAMILY_GROUPS.get(str(name), str(name)), []).append(idx)

    by_window: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
    for row in window_rows:
        try:
            start = int(max(0, round(float(row.get("start_sec", 0.0)) * float(fps))))
            end = int(min(n, round(float(row.get("end_sec", 0.0)) * float(fps))))
        except Exception:
            continue
        if end <= start:
            continue
        by_window.setdefault((start, end), []).append(row)
    if not by_window:
        return None, None, {"enabled": False, "reason": "no_valid_windows"}

    readout_sum = np.zeros(n, dtype=np.float64)
    conf_sum = np.zeros(n, dtype=np.float64)
    conf_count = np.zeros(n, dtype=np.float64)
    rate_ref = max(0.03, 0.10 * (float(max_hz) - float(min_hz)))
    accepted = 0
    confidence_values: List[float] = []
    entropy_values: List[float] = []
    top_gap_values: List[float] = []
    mode_values: List[float] = []
    selected_counts: Dict[str, int] = {}
    group_rate_mode_counts: Dict[str, int] = {}

    for (start, end), rows in sorted(by_window.items()):
        row_by_group = {str(row.get("group", "")).strip(): row for row in rows}
        group_rates: Dict[str, float] = {}
        base_scores: Dict[str, float] = {}
        row_cache: Dict[str, Dict[str, float]] = {}
        track_present: Dict[str, float] = {}

        for group, idxs in groups.items():
            row = row_by_group.get(group)
            if row is None:
                continue
            idx_arr = np.asarray(idxs, dtype=int)
            valid_rates = rate_mat[idx_arr, start:end].reshape(-1)
            valid_rates = valid_rates[
                np.isfinite(valid_rates)
                & (valid_rates >= float(min_hz))
                & (valid_rates <= float(max_hz))
            ]
            local_rates: List[float] = []
            local_weights: List[float] = []
            track_rate = float("nan")
            stability = 0.0
            if valid_rates.size >= max(3, int(0.08 * max(1, end - start))):
                track_rate = float(np.median(valid_rates))
                std = float(np.std(valid_rates))
                stability = float(np.exp(-0.5 * (std / rate_ref) ** 2))
                local_rates.append(track_rate)
                local_weights.append(max(stability, 0.05))

            sig_window = np.nanmedian(signal_mat[idx_arr, start:end], axis=0)
            spectral_candidates = _spectral_rate_candidates(
                sig_window,
                fps,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                max_candidates=4,
            )
            spectral_score = max((float(rel_power) for _, rel_power in spectral_candidates), default=0.0)
            if not np.isfinite(track_rate):
                for peak_hz, rel_power in spectral_candidates:
                    local_rates.append(float(peak_hz))
                    local_weights.append(float(rel_power))
            if not local_rates:
                continue
            group_rate, internal_support = _harmonic_rate_anchor(
                np.asarray(local_rates, dtype=np.float64),
                np.asarray(local_weights, dtype=np.float64),
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                rate_ref=rate_ref,
            )
            if not np.isfinite(group_rate):
                continue
            group_rate, group_rate_mode = _preserve_trustworthy_track_rate(
                group,
                float(track_rate),
                float(stability),
                float(group_rate),
                min_hz=float(min_hz),
                max_hz=float(max_hz),
            )
            soft_score = _finite_row_float(row, "score")
            base = _readout_timing_reliability_from_row(row, fallback=soft_score)
            if not np.isfinite(base) or base <= 0.0:
                base = float(fallback_priors.get(group, np.nan))
            if not np.isfinite(base) or base <= 0.0:
                base = 0.20
            support_score = _finite_row_float(row, "group_support_score", base)
            macro_score = _finite_row_float(row, "macro_timing_support_score", base)
            event_score = _finite_row_float(row, "event_timing_score", 0.0)
            rate_phase = _finite_row_float(row, "rate_phase_score", 1.0)
            harmonic_risk = _finite_row_float(row, "harmonic_risk_score", 0.50)
            abstain = _finite_row_float(row, "abstain_score", 0.50)
            timing_specificity = max(float(internal_support), float(stability), float(spectral_score), 0.05)
            score = (
                float(np.clip(base, 0.0, 1.0))
                * float(np.clip(rate_phase, 0.05, 1.0))
                * float(RATE_TIMING_GROUP_PRIOR.get(group, 0.80))
                * (0.45 + 0.55 * float(np.clip(support_score, 0.0, 1.0)))
                * (0.55 + 0.45 * float(np.clip(max(macro_score, event_score), 0.0, 1.0)))
                * (0.60 + 0.40 * float(np.clip(timing_specificity, 0.0, 1.0)))
                * (1.0 - 0.22 * float(np.clip(harmonic_risk, 0.0, 1.0)))
                * (1.0 - 0.18 * float(np.clip(abstain, 0.0, 1.0)))
            )
            if np.isfinite(track_rate):
                score *= 1.05
            if score <= 0.02:
                continue
            group_rates[group] = float(group_rate)
            base_scores[group] = float(np.clip(score, 0.0, 2.0))
            row_cache[group] = row
            track_present[group] = 1.0 if np.isfinite(track_rate) else 0.0
            group_rate_mode_counts[str(group_rate_mode)] = int(group_rate_mode_counts.get(str(group_rate_mode), 0)) + 1

        if not group_rates:
            continue

        source_scores: Dict[str, float] = {}
        for group, rate in group_rates.items():
            peer_support = _source_validity_peer_support(
                group,
                float(rate),
                group_rates,
                base_scores,
                rate_ref=rate_ref,
            )
            row = row_cache.get(group, {})
            macro_score = _finite_row_float(row, "macro_timing_support_score", peer_support)
            parent_scale, _ = _derived_consistency_score_scale(
                group,
                group_rates,
                base_scores,
                rate_ref=rate_ref,
            )
            source_scores[group] = float(base_scores[group]) * (
                0.50 + 0.50 * max(peer_support, float(np.clip(macro_score, 0.0, 1.0)))
            ) * (0.70 + 0.30 * float(parent_scale))

        group_names = list(group_rates)
        rates = np.asarray([group_rates[g] for g in group_names], dtype=np.float64)
        scores = np.asarray([max(source_scores[g], 1e-9) for g in group_names], dtype=np.float64)
        probs = scores ** 1.35
        probs = probs / max(float(np.sum(probs)), 1e-12)
        source_median = _weighted_median(rates, probs, fallback=float(np.nanmedian(rates)))
        source_mean = float(np.sum(probs * rates))
        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        top_prob = float(probs[top_idx])
        second_prob = float(probs[order[1]]) if probs.size > 1 else 0.0
        top_gap = float(np.clip((top_prob - second_prob) / max(top_prob, 1e-12), 0.0, 1.0))
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        entropy_norm = (
            float(np.clip(entropy / max(math.log(max(probs.size, 2)), 1e-12), 0.0, 1.0))
            if probs.size > 1 else 0.0
        )
        if not np.isfinite(source_median):
            continue
        top_rate = float(rates[top_idx])
        # Avoid hard mode selection: source validity can sharpen a posterior,
        # but source topology is still uncertain under target shift.
        sharpness = float(np.clip((top_prob - 0.35) / 0.45, 0.0, 1.0))
        readout = float(np.clip(
            (1.0 - 0.35 * sharpness) * source_median
            + 0.20 * sharpness * source_mean
            + 0.15 * sharpness * top_rate,
            float(min_hz),
            float(max_hz),
        ))
        support = 0.0
        for prob, rate in zip(probs, rates):
            support += float(prob) * _direct_candidate_support(readout, float(rate), rate_ref)
        confidence = float(np.clip(
            (0.25 + 0.75 * support)
            * (0.35 + 0.65 * (1.0 - entropy_norm))
            * (0.45 + 0.55 * top_gap),
            0.0,
            1.0,
        ))
        if confidence <= 0.02:
            continue
        readout_sum[start:end] += confidence * readout
        conf_sum[start:end] += confidence
        conf_count[start:end] += 1.0
        accepted += 1
        confidence_values.append(confidence)
        entropy_values.append(entropy_norm)
        top_gap_values.append(top_gap)
        mode_values.append(top_rate)
        selected = str(group_names[top_idx])
        selected_counts[selected] = int(selected_counts.get(selected, 0)) + 1

    valid = conf_sum > 1e-12
    if not np.any(valid):
        return None, None, {"enabled": False, "reason": "no_supported_source_validity_windows"}
    readout_track = np.full(n, np.nan, dtype=np.float64)
    confidence_track = np.zeros(n, dtype=np.float64)
    readout_track[valid] = readout_sum[valid] / conf_sum[valid]
    confidence_track[valid] = np.clip(conf_sum[valid] / np.maximum(conf_count[valid], 1.0), 0.0, 1.0)
    fill_fraction = 0.0
    coverage = float(np.mean(valid))
    if 0.25 <= coverage < 1.0 and np.count_nonzero(valid) >= 2:
        idx = np.arange(n, dtype=np.float64)
        valid_idx = idx[valid]
        fill_mask = ~valid
        filled_rate = np.interp(idx, valid_idx, readout_track[valid])
        filled_conf = np.interp(idx, valid_idx, confidence_track[valid])
        readout_track[fill_mask] = filled_rate[fill_mask]
        confidence_track[fill_mask] = 0.50 * filled_conf[fill_mask]
        fill_fraction = float(np.mean(fill_mask))
    meta: Dict[str, object] = {
        "enabled": True,
        "source": "source_validity_posterior",
        "coverage": coverage,
        "persistence_fill_fraction": fill_fraction,
        "accepted_windows": int(accepted),
        "confidence_mean": float(np.mean(confidence_track[valid])),
        "source_validity_entropy_median": float(np.median(entropy_values)) if entropy_values else float("nan"),
        "source_validity_top_gap_median": float(np.median(top_gap_values)) if top_gap_values else float("nan"),
        "source_validity_mode_hz_median": float(np.median(mode_values)) if mode_values else float("nan"),
        "selected_group_counts": dict(selected_counts),
        "group_rate_mode_counts": dict(group_rate_mode_counts),
    }
    return readout_track.tolist(), confidence_track.tolist(), meta


def _rate_hypothesis_graph_anchor(
    rates: np.ndarray,
    weights: np.ndarray,
    groups: Sequence[str],
    *,
    min_hz: float,
    max_hz: float,
    rate_ref: float,
) -> Tuple[float, float, str, Dict[str, object]]:
    """Resolve timing as a frequency-hypothesis graph rather than a row selector.

    Each observation group contributes support to candidate frequency nodes.
    Motion-like observations (OF/DoF) primarily support their direct frequency;
    P1D morphology can support a doubled frequency only when a motion macro
    family also directly supports that doubled node. This keeps all families in
    play while preventing smooth P1D half-rate morphology from dominating
    z_osc timing under target shift.
    """
    rate_arr = np.asarray(rates, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if rate_arr.size != weight_arr.size or rate_arr.size != len(groups):
        return float("nan"), 0.0, "invalid", {}
    ok = (
        np.isfinite(rate_arr)
        & np.isfinite(weight_arr)
        & (weight_arr > 0.0)
        & (rate_arr >= float(min_hz))
        & (rate_arr <= float(max_hz))
    )
    if not np.any(ok):
        return float("nan"), 0.0, "empty", {}
    rate_arr = rate_arr[ok]
    weight_arr = weight_arr[ok]
    group_arr = [str(g) for g, keep in zip(groups, ok) if bool(keep)]
    if rate_arr.size == 0:
        return float("nan"), 0.0, "empty", {}

    candidates: List[float] = []
    for rate in rate_arr:
        for cand in (float(rate), 0.5 * float(rate), 2.0 * float(rate)):
            if float(min_hz) <= cand <= float(max_hz):
                candidates.append(cand)
    direct, direct_support = _direct_rate_anchor(
        rate_arr,
        weight_arr,
        min_hz=float(min_hz),
        max_hz=float(max_hz),
        rate_ref=float(rate_ref),
    )
    harmonic, harmonic_support = _harmonic_rate_anchor(
        rate_arr,
        weight_arr,
        min_hz=float(min_hz),
        max_hz=float(max_hz),
        rate_ref=float(rate_ref),
    )
    for cand in (direct, harmonic):
        if np.isfinite(cand) and float(min_hz) <= float(cand) <= float(max_hz):
            candidates.append(float(cand))
    if not candidates:
        return float("nan"), 0.0, "empty_candidates", {}

    # Merge near-duplicate frequency nodes without assuming a fixed dataset grid.
    merged: List[float] = []
    for cand in sorted(candidates):
        if not merged or abs(float(cand) - float(merged[-1])) > max(0.5 * float(rate_ref), 1e-6):
            merged.append(float(cand))
        else:
            merged[-1] = float(0.5 * (float(merged[-1]) + float(cand)))

    macro_weight: Dict[str, float] = {}
    for group, weight in zip(group_arr, weight_arr):
        macro = _macro_timing_family(group)
        macro_weight[macro] = max(float(macro_weight.get(macro, 0.0)), float(weight))
    macro_weight_total = max(sum(macro_weight.values()), 1e-12)
    group_rate_map = {str(group): float(rate) for group, rate in zip(group_arr, rate_arr)}
    group_weight_map = {str(group): float(weight) for group, weight in zip(group_arr, weight_arr)}

    rows: List[Dict[str, object]] = []
    for cand in merged:
        motion_direct = 0.0
        motion_total = 0.0
        p1d_half = 0.0
        p1d_total = 0.0
        for group, rate, weight in zip(group_arr, rate_arr, weight_arr):
            macro = _macro_timing_family(group)
            direct_sup = _direct_candidate_support(cand, float(rate), float(rate_ref))
            if macro in {"OF", "DoF"}:
                motion_direct += float(weight) * direct_sup
                motion_total += float(weight)
            elif macro == "P1D":
                p1d_half += float(weight) * _direct_candidate_support(cand, 2.0 * float(rate), float(rate_ref))
                p1d_total += float(weight)
        motion_direct_norm = motion_direct / max(motion_total, 1e-12)
        p1d_half_norm = p1d_half / max(p1d_total, 1e-12)
        p1d_double_gate = float(np.clip(motion_direct_norm, 0.0, 1.0))

        macro_support: Dict[str, float] = {}
        direct_macro_support: Dict[str, float] = {}
        direct_macro_norm: Dict[str, float] = {}
        for group, rate, weight in zip(group_arr, rate_arr, weight_arr):
            macro = _macro_timing_family(group)
            direct_sup = _direct_candidate_support(cand, float(rate), float(rate_ref))
            half_sup = _direct_candidate_support(cand, 0.5 * float(rate), float(rate_ref))
            double_sup = _direct_candidate_support(cand, 2.0 * float(rate), float(rate_ref))
            if macro in {"OF", "DoF"}:
                # Motion observations are strongest as direct timing evidence;
                # their doubled bursts can weakly explain a lower fundamental.
                support = max(direct_sup, 0.42 * half_sup, 0.15 * double_sup)
            elif macro == "P1D":
                # P1D morphology may be half-rate. Count doubled support only
                # when motion macros independently support this candidate.
                support = max(direct_sup * 0.88, (0.20 + 0.55 * p1d_double_gate) * double_sup, 0.25 * half_sup)
            else:
                support = direct_sup
            macro_support[macro] = max(float(macro_support.get(macro, 0.0)), float(weight) * float(support))
            direct_macro_support[macro] = max(float(direct_macro_support.get(macro, 0.0)), float(weight) * float(direct_sup))
            direct_macro_norm[macro] = max(
                float(direct_macro_norm.get(macro, 0.0)),
                float(direct_sup),
            )

        support_score = sum(macro_support.values()) / macro_weight_total
        direct_score = sum(direct_macro_support.values()) / macro_weight_total
        macro_count = sum(1 for value in macro_support.values() if value >= 0.10)
        direct_macro_count = sum(1 for value in direct_macro_support.values() if value >= 0.10)
        diversity = float(np.clip(macro_count / max(len(macro_weight), 1), 0.0, 1.0))
        independent_timing = _independent_timing_witness_score(direct_macro_norm)
        motion_timing = _motion_timing_witness_score(direct_macro_norm)
        p1d_direct_timing = float(np.clip(direct_macro_norm.get("P1D", 0.0), 0.0, 1.0))
        bridge_preservation = _bridge_timing_preservation_score(
            float(cand),
            group_rate_map,
            group_weight_map,
            rate_ref=float(rate_ref),
        )
        morphology_alias_pressure = float(np.clip(p1d_half_norm * (1.0 - motion_direct_norm), 0.0, 1.0))
        h1_role_support = float(np.clip(
            0.48 * independent_timing
            + 0.30 * motion_timing
            + 0.17 * bridge_preservation
            + 0.05 * p1d_direct_timing
            - 0.30 * morphology_alias_pressure,
            0.0,
            1.0,
        ))
        morphology_role_support = float(np.clip(max(p1d_direct_timing, p1d_half_norm, support_score), 0.0, 1.0))
        abstain_pressure = float(np.clip(
            0.55 * morphology_alias_pressure
            + 0.30 * (1.0 - h1_role_support)
            + 0.15 * (1.0 - diversity),
            0.0,
            1.0,
        ))
        score = float(np.clip(support_score * (0.55 + 0.45 * diversity) * (0.70 + 0.30 * direct_score), 0.0, 1.5))
        rows.append(
            {
                "candidate_hz": float(cand),
                "score": score,
                "support_score": float(support_score),
                "direct_score": float(direct_score),
                "macro_count": int(macro_count),
                "direct_macro_count": int(direct_macro_count),
                "motion_direct": float(motion_direct_norm),
                "p1d_half_support": float(p1d_half_norm),
                "independent_timing_support": float(independent_timing),
                "motion_timing_support": float(motion_timing),
                "bridge_timing_preservation": float(bridge_preservation),
                "p1d_direct_timing_support": float(p1d_direct_timing),
                "morphology_role_support": float(morphology_role_support),
                "morphology_alias_pressure": float(morphology_alias_pressure),
                "h1_role_support": float(h1_role_support),
                "abstain_pressure": float(abstain_pressure),
                "macro_support": macro_support,
            }
        )

    if not rows:
        return float("nan"), 0.0, "empty_scored", {}
    rows.sort(key=lambda row: (float(row["score"]), float(row["direct_score"]), int(row["macro_count"])), reverse=True)
    best = rows[0]
    anchor = float(best["candidate_hz"])
    mode = "rate_hypothesis_graph"
    if float(best.get("p1d_half_support", 0.0)) >= 0.35 and float(best.get("motion_direct", 0.0)) >= 0.35:
        mode = "rate_hypothesis_graph_p1d_half_resolved"
    detail = {
        "top_candidates": [
            {
                "hz": float(row["candidate_hz"]),
                "score": float(row["score"]),
                "support": float(row["support_score"]),
                "direct": float(row["direct_score"]),
                "macro_count": int(row["macro_count"]),
                "direct_macro_count": int(row["direct_macro_count"]),
                "motion_direct": float(row["motion_direct"]),
                "p1d_half_support": float(row["p1d_half_support"]),
                "independent_timing_support": float(row["independent_timing_support"]),
                "motion_timing_support": float(row["motion_timing_support"]),
                "bridge_timing_preservation": float(row["bridge_timing_preservation"]),
                "p1d_direct_timing_support": float(row["p1d_direct_timing_support"]),
                "morphology_role_support": float(row["morphology_role_support"]),
                "morphology_alias_pressure": float(row["morphology_alias_pressure"]),
                "h1_role_support": float(row["h1_role_support"]),
                "abstain_pressure": float(row["abstain_pressure"]),
            }
            for row in rows[:5]
        ],
        "direct_anchor_hz": float(direct) if np.isfinite(direct) else float("nan"),
        "direct_support": float(direct_support),
        "harmonic_anchor_hz": float(harmonic) if np.isfinite(harmonic) else float("nan"),
        "harmonic_support": float(harmonic_support),
        "best_p1d_half_support": float(best.get("p1d_half_support", 0.0)),
        "best_motion_direct": float(best.get("motion_direct", 0.0)),
        "best_macro_count": int(best.get("macro_count", 0)),
        "best_direct_macro_count": int(best.get("direct_macro_count", 0)),
        "best_direct_score": float(best.get("direct_score", 0.0)),
        "best_independent_timing_support": float(best.get("independent_timing_support", 0.0)),
        "best_motion_timing_support": float(best.get("motion_timing_support", 0.0)),
        "best_bridge_timing_preservation": float(best.get("bridge_timing_preservation", 0.0)),
        "best_p1d_direct_timing_support": float(best.get("p1d_direct_timing_support", 0.0)),
        "best_morphology_role_support": float(best.get("morphology_role_support", 0.0)),
        "best_morphology_alias_pressure": float(best.get("morphology_alias_pressure", 0.0)),
        "best_h1_role_support": float(best.get("h1_role_support", 0.0)),
        "best_abstain_pressure": float(best.get("abstain_pressure", 0.0)),
    }
    return float(np.clip(anchor, float(min_hz), float(max_hz))), float(np.clip(float(best["score"]), 0.0, 1.0)), mode, detail


def _wrap_phase(x: float) -> float:
    return float((float(x) + np.pi) % (2.0 * np.pi) - np.pi)


def _sinusoid_phase_fit(signal: np.ndarray, fps: float, freq_hz: float) -> Tuple[float, float]:
    arr = _finite_fill(np.asarray(signal, dtype=np.float64).reshape(-1))
    if arr.size < 8 or fps <= 0.0 or not np.isfinite(freq_hz) or freq_hz <= 0.0:
        return 0.0, float("nan")
    arr = sps.detrend(arr, type="linear")
    arr = arr - float(np.mean(arr))
    sst = float(np.dot(arr, arr))
    if not np.isfinite(sst) or sst <= 1e-12:
        return 0.0, float("nan")
    t = (np.arange(arr.size, dtype=np.float64) / float(fps))
    t = t - float(np.mean(t))
    omega = 2.0 * np.pi * float(freq_hz)
    design = np.column_stack([
        np.cos(omega * t),
        np.sin(omega * t),
        np.ones_like(t),
    ])
    try:
        beta, *_ = np.linalg.lstsq(design, arr, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, float("nan")
    pred = design @ beta
    sse = float(np.sum((arr - pred) ** 2))
    r2 = float(np.clip(1.0 - sse / max(sst, 1e-12), 0.0, 1.0))
    amp = float(np.hypot(beta[0], beta[1]))
    amp_score = float(np.clip(amp / max(float(np.std(arr)), 1e-12), 0.0, 1.0))
    phase = _wrap_phase(float(np.arctan2(beta[1], beta[0])))
    return float(np.sqrt(max(r2, 0.0)) * amp_score), phase


def _weighted_circular_mean(phases: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    ph = np.asarray(phases, dtype=np.float64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if ph.size != w.size:
        return float("nan"), 0.0
    ok = np.isfinite(ph) & np.isfinite(w) & (w > 0.0)
    if not np.any(ok):
        return float("nan"), 0.0
    ph = ph[ok]
    w = w[ok]
    vec = np.sum(w * np.exp(1j * ph))
    total = float(np.sum(w))
    if total <= 1e-12 or not np.isfinite(total):
        return float("nan"), 0.0
    concentration = float(np.clip(np.abs(vec) / total, 0.0, 1.0))
    return _wrap_phase(float(np.angle(vec))), concentration


def _spectral_rate_candidates(
    signal: np.ndarray,
    fps: float,
    *,
    min_hz: float,
    max_hz: float,
    max_candidates: int = 3,
) -> List[Tuple[float, float]]:
    arr = _finite_fill(np.asarray(signal, dtype=np.float64).reshape(-1))
    if arr.size < max(16, int(4.0 * max(float(fps), 1.0))) or fps <= 0.0:
        return []
    arr = sps.detrend(arr, type="linear")
    arr = arr - float(np.mean(arr))
    if float(np.std(arr)) <= 1e-9:
        return []
    nperseg = min(arr.size, max(16, int(round(float(fps) * 30.0))))
    try:
        freqs, power = sps.welch(arr, fs=float(fps), nperseg=nperseg)
    except Exception:
        return []
    keep = (
        np.isfinite(freqs)
        & np.isfinite(power)
        & (freqs >= float(min_hz))
        & (freqs <= float(max_hz))
        & (power > 0.0)
    )
    if not np.any(keep):
        return []
    f = freqs[keep]
    pxx = power[keep]
    if f.size == 0:
        return []
    peaks, _ = sps.find_peaks(pxx)
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(pxx))], dtype=int)
    order = peaks[np.argsort(pxx[peaks])[::-1]]
    top_power = float(pxx[order[0]]) if order.size else 0.0
    if top_power <= 0.0 or not np.isfinite(top_power):
        return []
    out: List[Tuple[float, float]] = []
    for idx in order[: max(1, int(max_candidates))]:
        rel = float(pxx[idx] / top_power)
        if rel < 0.25:
            continue
        out.append((float(f[idx]), float(np.clip(rel, 0.0, 1.0))))
    return out


def _autocorr_periodicity_score(signal: np.ndarray, fps: float, *, min_hz: float, max_hz: float) -> float:
    arr = _finite_fill(np.asarray(signal, dtype=np.float64).reshape(-1))
    if arr.size < max(16, int(4.0 * max(float(fps), 1.0))) or fps <= 0.0:
        return 0.0
    arr = sps.detrend(arr, type="linear")
    arr = arr - float(np.mean(arr))
    scale = float(np.std(arr))
    if not np.isfinite(scale) or scale <= 1e-9:
        return 0.0
    arr = arr / scale
    corr = np.correlate(arr, arr, mode="full")[arr.size - 1 :]
    denom = float(corr[0]) if corr.size else 0.0
    if denom <= 1e-12 or not np.isfinite(denom):
        return 0.0
    corr = corr / denom
    lag_min = int(max(1, math.floor(float(fps) / max(float(max_hz), 1e-6))))
    lag_max = int(min(corr.size - 1, math.ceil(float(fps) / max(float(min_hz), 1e-6))))
    if lag_max <= lag_min:
        return 0.0
    peak = float(np.nanmax(corr[lag_min : lag_max + 1]))
    return float(np.clip((peak - 0.10) / 0.65, 0.0, 1.0))


def _signal_sqi_features(
    signal: np.ndarray,
    fps: float,
    *,
    min_hz: float,
    max_hz: float,
    freq_hint: float = float("nan"),
) -> Dict[str, float]:
    """Target-computable evidence quality for one candidate observation.

    The score is intentionally signal-internal: it uses spectral concentration,
    periodic autocorrelation, and sinusoidal fit at the locally supported rate.
    It never looks at GT and it does not pick a final source by itself.
    """
    arr = _finite_fill(np.asarray(signal, dtype=np.float64).reshape(-1))
    out = {
        "signal_sqi": 0.0,
        "spectral_peakiness": 0.0,
        "periodicity": 0.0,
        "phase_fit": 0.0,
        "rate_coherence": 0.0,
        "phase": float("nan"),
    }
    if arr.size < max(16, int(4.0 * max(float(fps), 1.0))) or fps <= 0.0:
        return out
    arr = sps.detrend(arr, type="linear")
    arr = arr - float(np.mean(arr))
    if float(np.std(arr)) <= 1e-9:
        return out
    nperseg = min(arr.size, max(16, int(round(float(fps) * 30.0))))
    try:
        freqs, power = sps.welch(arr, fs=float(fps), nperseg=nperseg)
    except Exception:
        freqs = np.asarray([], dtype=np.float64)
        power = np.asarray([], dtype=np.float64)
    keep = (
        np.isfinite(freqs)
        & np.isfinite(power)
        & (freqs >= float(min_hz))
        & (freqs <= float(max_hz))
        & (power > 0.0)
    )
    top_hz = float("nan")
    if np.any(keep):
        f = freqs[keep]
        pxx = power[keep]
        peak_idx = int(np.argmax(pxx))
        top_hz = float(f[peak_idx])
        top_power = float(pxx[peak_idx])
        total_power = float(np.sum(pxx))
        median_power = float(np.median(pxx))
        concentration = top_power / max(total_power, 1e-12)
        prominence = (top_power - median_power) / max(top_power + median_power, 1e-12)
        # Welch windows can spread a good respiratory peak over nearby bins, so
        # combine concentration and local prominence rather than requiring a
        # single-bin spike.
        out["spectral_peakiness"] = float(np.clip(0.45 * ((concentration - 0.06) / 0.24) + 0.55 * prominence, 0.0, 1.0))

    out["periodicity"] = _autocorr_periodicity_score(arr, fps, min_hz=float(min_hz), max_hz=float(max_hz))
    fit_freq = float(freq_hint) if np.isfinite(freq_hint) and freq_hint > 0.0 else top_hz
    if np.isfinite(fit_freq) and float(min_hz) <= fit_freq <= float(max_hz):
        phase_fit, phase = _sinusoid_phase_fit(arr, fps, fit_freq)
        out["phase_fit"] = float(np.clip(phase_fit, 0.0, 1.0))
        out["phase"] = float(phase) if np.isfinite(phase) else float("nan")
        if np.isfinite(top_hz):
            out["rate_coherence"] = _direct_candidate_support(float(fit_freq), float(top_hz), max(0.03, 0.10 * (float(max_hz) - float(min_hz))))
    out["signal_sqi"] = float(np.clip(
        0.34 * out["spectral_peakiness"]
        + 0.28 * out["periodicity"]
        + 0.24 * out["phase_fit"]
        + 0.14 * out["rate_coherence"],
        0.0,
        1.0,
    ))
    return out


def _windowed_signal_sqi_observability(
    names: Sequence[str],
    signals: Sequence[np.ndarray],
    rate_tracks: Sequence[np.ndarray],
    window_rows: Sequence[Dict[str, float]],
    n_frames: int,
    fps: float,
    *,
    min_hz: float,
    max_hz: float,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, object]]:
    n = int(max(n_frames, 0))
    if n <= 0 or fps <= 0.0 or not names or len(signals) == 0:
        return None, {"enabled": False, "reason": "missing_inputs"}
    signal_rows = [_fit_length(np.asarray(sig, dtype=np.float64), n, fill_value=0.0) for sig in signals]
    if not signal_rows:
        return None, {"enabled": False, "reason": "empty_signals"}
    signal_mat = np.vstack(signal_rows)
    rate_rows = []
    for track in rate_tracks:
        arr = _fit_length(np.asarray(track, dtype=np.float64), n)
        arr = np.where(np.isfinite(arr) & (arr >= float(min_hz)) & (arr <= float(max_hz)), arr, np.nan)
        rate_rows.append(arr)
    rate_mat = np.vstack(rate_rows) if rate_rows else np.full((len(names), n), np.nan, dtype=np.float64)
    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        groups.setdefault(FAMILY_GROUPS.get(str(name), str(name)), []).append(idx)

    windows: List[Tuple[int, int, List[str]]] = []
    if window_rows:
        by_window: Dict[Tuple[int, int], List[str]] = {}
        for row in window_rows:
            group = str(row.get("group", "")).strip()
            try:
                start = int(max(0, round(float(row.get("start_sec", 0.0)) * float(fps))))
                end = int(min(n, round(float(row.get("end_sec", 0.0)) * float(fps))))
            except Exception:
                continue
            if end <= start or group not in groups:
                continue
            by_window.setdefault((start, end), []).append(group)
        windows = [(start, end, sorted(set(win_groups))) for (start, end), win_groups in sorted(by_window.items())]
    if not windows:
        windows = [(0, n, sorted(groups))]

    keys = ("signal_sqi", "spectral_peakiness", "periodicity", "phase_fit", "phase_coherence")
    sums = {key: np.zeros(n, dtype=np.float64) for key in keys}
    counts = np.zeros(n, dtype=np.float64)
    accepted = 0
    sqi_values: List[float] = []
    phase_coherence_values: List[float] = []

    for start, end, win_groups in windows:
        group_features: List[Dict[str, float]] = []
        group_weights: List[float] = []
        phases: List[float] = []
        phase_weights: List[float] = []
        for group in win_groups:
            idxs = groups.get(group)
            if not idxs:
                continue
            idx_arr = np.asarray(idxs, dtype=int)
            sig_window = np.nanmedian(signal_mat[idx_arr, start:end], axis=0)
            rate_vals = rate_mat[idx_arr, start:end].reshape(-1)
            rate_vals = rate_vals[np.isfinite(rate_vals)]
            freq_hint = float(np.median(rate_vals)) if rate_vals.size else float("nan")
            feat = _signal_sqi_features(
                sig_window,
                fps,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                freq_hint=freq_hint,
            )
            weight = max(float(feat["signal_sqi"]), 1e-6)
            group_features.append(feat)
            group_weights.append(weight)
            phase = float(feat.get("phase", float("nan")))
            if np.isfinite(phase) and weight > 1e-6:
                phases.append(phase)
                phase_weights.append(weight)
        if not group_features:
            continue
        if phases:
            _, phase_coherence = _weighted_circular_mean(
                np.asarray(phases, dtype=np.float64),
                np.asarray(phase_weights, dtype=np.float64),
            )
        else:
            phase_coherence = 0.0
        weights = np.asarray(group_weights, dtype=np.float64)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        values = {}
        for key in ("signal_sqi", "spectral_peakiness", "periodicity", "phase_fit"):
            values[key] = float(np.sum(weights * np.asarray([float(feat[key]) for feat in group_features], dtype=np.float64)))
        values["phase_coherence"] = float(np.clip(phase_coherence, 0.0, 1.0))
        for key, value in values.items():
            sums[key][start:end] += float(value)
        counts[start:end] += 1.0
        accepted += 1
        sqi_values.append(float(values["signal_sqi"]))
        phase_coherence_values.append(float(values["phase_coherence"]))

    valid = counts > 0.0
    if not np.any(valid):
        return None, {"enabled": False, "reason": "no_valid_sqi_windows"}
    arrays: Dict[str, np.ndarray] = {}
    for key, arr_sum in sums.items():
        arr = np.zeros(n, dtype=np.float64)
        arr[valid] = arr_sum[valid] / counts[valid]
        if np.any(~valid) and np.count_nonzero(valid) >= 2:
            idx = np.arange(n, dtype=np.float64)
            arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
        arrays[key] = np.clip(arr, 0.0, 1.0)
    meta = {
        "enabled": True,
        "accepted_windows": int(accepted),
        "coverage": float(np.mean(valid)),
        "signal_sqi_median": float(np.median(sqi_values)) if sqi_values else float("nan"),
        "phase_coherence_median": float(np.median(phase_coherence_values)) if phase_coherence_values else float("nan"),
    }
    return arrays, meta


def _windowed_rate_anchor_runtime(
    names: Sequence[str],
    rate_tracks: Sequence[np.ndarray],
    window_rows: Sequence[Dict[str, float]],
    n_frames: int,
    fps: float,
    fallback_priors: Dict[str, float],
    min_hz: float,
    max_hz: float,
    signals: Optional[Sequence[np.ndarray]] = None,
    enable_phase_validation: bool = False,
    enable_rate_hypothesis_graph_v4: bool = False,
) -> Tuple[Optional[List[float]], Optional[List[float]], Dict[str, float]]:
    """Estimate a GT-free local respiratory-rate anchor from windowed group support.

    The anchor is deliberately group-level: P1D_quad/P1D_cub can refine the
    same morphology group, but they cannot outvote cross-family evidence as
    independent sensors.
    """
    n = int(max(n_frames, 0))
    if n <= 0 or fps <= 0.0 or not names or not rate_tracks or not window_rows:
        return None, None, {"enabled": False, "reason": "missing_inputs"}

    tracks = []
    for track in rate_tracks:
        arr = _fit_length(np.asarray(track, dtype=np.float64), n)
        arr = np.where(np.isfinite(arr) & (arr >= float(min_hz)) & (arr <= float(max_hz)), arr, np.nan)
        tracks.append(arr)
    if not tracks:
        return None, None, {"enabled": False, "reason": "empty_tracks"}
    track_mat = np.vstack(tracks)
    signal_mat = None
    if signals is not None:
        signal_rows = []
        for sig in signals:
            signal_rows.append(_fit_length(np.asarray(sig, dtype=np.float64), n, fill_value=0.0))
        if signal_rows:
            signal_mat = np.vstack(signal_rows)

    group_to_indices: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        group_to_indices.setdefault(FAMILY_GROUPS.get(str(name), str(name)), []).append(idx)

    by_window: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
    for row in window_rows:
        try:
            start = int(max(0, round(float(row.get("start_sec", 0.0)) * float(fps))))
            end = int(min(n, round(float(row.get("end_sec", 0.0)) * float(fps))))
        except Exception:
            continue
        if end <= start:
            continue
        by_window.setdefault((start, end), []).append(row)

    if not by_window:
        return None, None, {"enabled": False, "reason": "no_valid_windows"}

    anchor_sum = np.zeros(n, dtype=np.float64)
    conf_sum = np.zeros(n, dtype=np.float64)
    conf_count = np.zeros(n, dtype=np.float64)
    rate_ref = max(0.03, 0.10 * (float(max_hz) - float(min_hz)))
    accepted = 0
    candidate_counts: List[int] = []
    confidence_values: List[float] = []
    anchor_values: List[float] = []
    phase_support_values: List[float] = []
    temporal_support_values: List[float] = []
    anchor_mode_counts: Dict[str, int] = {}
    prev_center_sec: Optional[float] = None
    prev_phase: Optional[float] = None
    prev_anchor: Optional[float] = None

    for (start, end), rows in sorted(by_window.items()):
        rates: List[float] = []
        weights: List[float] = []
        groups: List[str] = []
        group_windows: List[Optional[np.ndarray]] = []
        for row in rows:
            group = str(row.get("group", "")).strip()
            idxs = group_to_indices.get(group)
            if not idxs:
                continue
            idx_arr = np.asarray(idxs, dtype=int)
            local_rates: List[float] = []
            local_weights: List[float] = []
            vals = track_mat[idx_arr, start:end].reshape(-1)
            vals = vals[np.isfinite(vals) & (vals >= float(min_hz)) & (vals <= float(max_hz))]
            track_stability = 0.0
            track_rate = float("nan")
            if vals.size >= max(3, int(0.10 * max(1, end - start))):
                track_rate = float(np.median(vals))
                group_std = float(np.std(vals))
                track_stability = float(np.exp(-0.5 * (group_std / rate_ref) ** 2))
                local_rates.append(track_rate)
                local_weights.append(max(track_stability, 0.05))
            spectral_score = 0.0
            if signal_mat is not None:
                sig_window = np.nanmedian(signal_mat[idx_arr, start:end], axis=0)
                spectral_candidates = _spectral_rate_candidates(
                    sig_window,
                    fps,
                    min_hz=float(min_hz),
                    max_hz=float(max_hz),
                )
                for _, rel_power in spectral_candidates:
                    spectral_score = max(spectral_score, float(rel_power))
                if not np.isfinite(track_rate):
                    for peak_hz, rel_power in spectral_candidates:
                        local_rates.append(float(peak_hz))
                        local_weights.append(float(rel_power))
            else:
                sig_window = None
            if len(local_rates) == 0:
                continue
            if np.isfinite(track_rate):
                group_rate = float(track_rate)
                group_internal_support = max(float(track_stability), 0.25 * float(spectral_score), 0.05)
            else:
                group_rate, group_internal_support, _, _, _ = _track_first_rate_anchor(
                    np.asarray(local_rates, dtype=np.float64),
                    np.asarray(local_weights, dtype=np.float64),
                    min_hz=float(min_hz),
                    max_hz=float(max_hz),
                    rate_ref=rate_ref,
                )
            if not np.isfinite(group_rate):
                continue
            stability = max(float(track_stability), float(group_internal_support), 0.05)
            try:
                soft_score = float(row.get("score", np.nan))
                base_score = _timing_reliability_from_row(row, fallback=soft_score)
                rate_phase_score = float(row.get("rate_phase_score", 1.0))
            except Exception:
                continue
            if not np.isfinite(base_score) or base_score <= 0.0:
                base_score = soft_score
            if not np.isfinite(base_score) or base_score <= 0.0:
                base_score = float(fallback_priors.get(group, np.nan))
            if not np.isfinite(base_score) or base_score <= 0.0:
                continue
            weight = (
                float(base_score)
                * float(np.clip(rate_phase_score, 0.05, 1.0))
                * float(np.clip(stability, 0.05, 1.0))
                * float(RATE_TIMING_GROUP_PRIOR.get(group, 0.80))
            )
            if not np.isfinite(weight) or weight <= 0.0:
                continue
            rates.append(group_rate)
            weights.append(weight)
            groups.append(group)
            group_windows.append(sig_window)

        if len(rates) < 2:
            continue
        rate_arr = np.asarray(rates, dtype=np.float64)
        weight_arr = np.asarray(weights, dtype=np.float64)
        if enable_rate_hypothesis_graph_v4:
            anchor, anchor_support, anchor_mode, _ = _rate_hypothesis_graph_anchor(
                rate_arr,
                weight_arr,
                groups,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                rate_ref=rate_ref,
            )
        else:
            anchor, anchor_support, anchor_mode, _, _ = _track_first_rate_anchor(
                rate_arr,
                weight_arr,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                rate_ref=rate_ref,
            )
        if not np.isfinite(anchor):
            continue
        if not enable_rate_hypothesis_graph_v4:
            anchor, half_rate_rescued, half_rate_rescue_strength = _group_aware_half_rate_rescue(
                anchor,
                rate_arr,
                weight_arr,
                groups,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                rate_ref=rate_ref,
            )
            if half_rate_rescued:
                anchor_mode = "group_aware_half_rate_rescue"
                anchor_support = max(float(anchor_support), float(half_rate_rescue_strength))
        agreement = float(anchor_support)
        norm_w = weight_arr / max(float(np.sum(weight_arr)), 1e-12)
        entropy = float(-np.sum(norm_w * np.log(np.clip(norm_w, 1e-12, 1.0))))
        diversity = float(np.clip(entropy / max(np.log(max(len(norm_w), 2)), 1e-12), 0.0, 1.0))
        support_mass = float(np.clip(np.sum(weight_arr), 0.0, 1.0))
        confidence = float(np.clip(np.sqrt(max(agreement, 0.0) * max(diversity, 0.0)) * (0.5 + 0.5 * support_mass), 0.0, 1.0))
        phase_support = 1.0
        anchor_phase = float("nan")
        if signal_mat is not None:
            phase_rows: List[float] = []
            phase_weights: List[float] = []
            fit_scores: List[float] = []
            for sig_window, group_weight in zip(group_windows, weight_arr):
                if sig_window is None:
                    continue
                fit_score, phase = _sinusoid_phase_fit(sig_window, fps, anchor)
                if not np.isfinite(phase) or fit_score <= 0.0:
                    continue
                phase_rows.append(float(phase))
                phase_weights.append(float(group_weight) * float(max(fit_score, 1e-6)))
                fit_scores.append(float(fit_score))
            if phase_rows:
                anchor_phase, phase_conc = _weighted_circular_mean(
                    np.asarray(phase_rows, dtype=np.float64),
                    np.asarray(phase_weights, dtype=np.float64),
                )
                fit_support = float(np.average(
                    np.asarray(fit_scores, dtype=np.float64),
                    weights=np.maximum(np.asarray(phase_weights, dtype=np.float64), 1e-12),
                ))
                phase_support = float(np.sqrt(max(phase_conc, 0.0) * max(fit_support, 0.0)))
            else:
                phase_support = 0.0

        temporal_support = 1.0
        center_sec = 0.5 * (float(start) + float(end)) / float(fps)
        if (
            prev_center_sec is not None
            and prev_phase is not None
            and prev_anchor is not None
            and np.isfinite(anchor_phase)
        ):
            dt_sec = max(0.0, center_sec - float(prev_center_sec))
            expected = _wrap_phase(float(prev_phase) + 2.0 * np.pi * 0.5 * (float(prev_anchor) + float(anchor)) * dt_sec)
            phase_err = abs(_wrap_phase(anchor_phase - expected))
            temporal_support = float(np.exp(-0.5 * (phase_err / 0.90) ** 2))

        phase_temporal_support = float(np.sqrt(max(phase_support, 0.0) * max(temporal_support, 0.0)))
        if bool(enable_phase_validation):
            confidence = float(np.clip(confidence * (0.50 + 0.50 * phase_temporal_support), 0.0, 1.0))
        if confidence <= 0.0:
            continue
        anchor = float(np.clip(anchor, float(min_hz), float(max_hz)))
        anchor_sum[start:end] += confidence * anchor
        conf_sum[start:end] += confidence
        conf_count[start:end] += 1.0
        accepted += 1
        anchor_mode_counts[str(anchor_mode)] = int(anchor_mode_counts.get(str(anchor_mode), 0)) + 1
        candidate_counts.append(len(set(groups)))
        confidence_values.append(confidence)
        anchor_values.append(anchor)
        phase_support_values.append(float(phase_support))
        temporal_support_values.append(float(temporal_support))
        if np.isfinite(anchor_phase) and confidence > 0.05:
            prev_center_sec = center_sec
            prev_phase = float(anchor_phase)
            prev_anchor = float(anchor)

    valid = conf_sum > 1e-12
    if not np.any(valid):
        return None, None, {"enabled": False, "reason": "no_supported_anchor_windows"}
    anchor_track = np.full(n, np.nan, dtype=np.float64)
    conf_track = np.zeros(n, dtype=np.float64)
    anchor_track[valid] = anchor_sum[valid] / conf_sum[valid]
    conf_track[valid] = np.clip(conf_sum[valid] / np.maximum(conf_count[valid], 1.0), 0.0, 1.0)
    anchor_track = np.where(np.isfinite(anchor_track), np.clip(anchor_track, float(min_hz), float(max_hz)), np.nan)
    meta = {
        "enabled": True,
        "accepted_windows": int(accepted),
        "coverage": float(np.mean(valid)),
        "confidence_mean": float(np.mean(conf_track[valid])),
        "window_confidence_median": float(np.median(confidence_values)) if confidence_values else float("nan"),
        "support_group_count_median": float(np.median(candidate_counts)) if candidate_counts else 0.0,
        "anchor_hz_median": float(np.median(anchor_values)) if anchor_values else float("nan"),
        "anchor_hz_std": float(np.std(anchor_values)) if anchor_values else float("nan"),
        "phase_support_median": float(np.median(phase_support_values)) if phase_support_values else float("nan"),
        "temporal_support_median": float(np.median(temporal_support_values)) if temporal_support_values else float("nan"),
        "phase_validation_enabled": bool(enable_phase_validation),
        "anchor_mode_counts": dict(anchor_mode_counts),
    }
    return anchor_track.tolist(), conf_track.tolist(), meta


REGIME_GROUP_PRIOR = {
    "G_OF": 0.72,
    "G_OF_bridge": 0.98,
    "G_DoF": 0.92,
    "G_DoF_bridge": 1.18,
    "G_P1D_low": 0.88,
    "G_P1D_morph": 1.00,
    "G_P1D_cons": 1.08,
}

RATE_TIMING_GROUP_PRIOR = {
    "G_OF": 1.00,
    "G_OF_bridge": 0.90,
    "G_DoF": 1.25,
    "G_DoF_bridge": 1.05,
    "G_P1D_low": 0.55,
    "G_P1D_morph": 0.55,
    "G_P1D_cons": 0.65,
}

REGIME_CHANNEL_FLOOR = 0.06


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0)) if np.isfinite(x) else 0.0


def _semantic_group_support(groups: Dict[str, float], names: Sequence[str]) -> np.ndarray:
    vals = []
    for name in names:
        group = FAMILY_GROUPS.get(str(name), str(name))
        vals.append(float(groups.get(group, REGIME_CHANNEL_FLOOR)))
    return np.asarray(vals, dtype=np.float64)


def _regime_name_from_scores(
    motion_score: float,
    morphology_score: float,
    velocity_score: float,
    weak_score: float,
) -> str:
    if weak_score >= max(motion_score, morphology_score, velocity_score):
        return "weak_observable"
    if motion_score >= morphology_score and motion_score >= velocity_score:
        return "motion_energy"
    if morphology_score >= velocity_score:
        return "profile_morphology"
    return "optical_velocity"


def _regime_observation_law_runtime(
    names: Sequence[str],
    signals: Sequence[np.ndarray],
    rate_tracks: Sequence[np.ndarray],
    window_rows: Sequence[Dict[str, float]],
    n_frames: int,
    fps: float,
    fallback_priors: Dict[str, float],
    min_hz: float,
    max_hz: float,
) -> Tuple[Optional[List[List[float]]], Optional[List[float]], Optional[List[float]], Dict[str, object]]:
    """GT-free observation-equation regime law.

    This is intentionally not a hard selector. It asks which semantic
    observation equation is locally capable of explaining the respiratory
    fundamental. Smoothness alone is not enough: a clean OF trace can be a
    non-respiratory oscillator, while a burst-like DoF trace can still carry
    strong timing information in MAHNOB-like regimes.
    """
    n = int(max(n_frames, 0))
    n_channels = len(names)
    if n <= 0 or n_channels <= 0 or fps <= 0.0 or not window_rows:
        return None, None, None, {"enabled": False, "reason": "missing_inputs"}

    sig_rows = []
    for sig in signals:
        sig_rows.append(_fit_length(np.asarray(sig, dtype=np.float64), n, fill_value=0.0))
    if not sig_rows:
        return None, None, None, {"enabled": False, "reason": "empty_signals"}
    signal_mat = np.vstack(sig_rows)

    rate_rows = []
    for track in rate_tracks:
        arr = _fit_length(np.asarray(track, dtype=np.float64), n)
        arr = np.where(np.isfinite(arr) & (arr >= float(min_hz)) & (arr <= float(max_hz)), arr, np.nan)
        rate_rows.append(arr)
    if not rate_rows:
        return None, None, None, {"enabled": False, "reason": "empty_rate_tracks"}
    rate_mat = np.vstack(rate_rows)

    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        groups.setdefault(FAMILY_GROUPS.get(str(name), str(name)), []).append(idx)

    by_window: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
    for row in window_rows:
        try:
            start = int(max(0, round(float(row.get("start_sec", 0.0)) * float(fps))))
            end = int(min(n, round(float(row.get("end_sec", 0.0)) * float(fps))))
        except Exception:
            continue
        if end <= start:
            continue
        by_window.setdefault((start, end), []).append(row)
    if not by_window:
        return None, None, None, {"enabled": False, "reason": "no_valid_windows"}

    context_sum = np.zeros((n_channels, n), dtype=np.float64)
    context_count = np.zeros((n_channels, n), dtype=np.float64)
    anchor_sum = np.zeros(n, dtype=np.float64)
    anchor_conf_sum = np.zeros(n, dtype=np.float64)
    anchor_conf_count = np.zeros(n, dtype=np.float64)
    rate_ref = max(0.03, 0.10 * (float(max_hz) - float(min_hz)))

    accepted = 0
    confidence_values: List[float] = []
    support_group_counts: List[int] = []
    regime_counts: Dict[str, int] = {}
    abstain_values: List[float] = []
    motion_values: List[float] = []
    morphology_values: List[float] = []
    velocity_values: List[float] = []

    for (start, end), rows in sorted(by_window.items()):
        row_by_group = {str(row.get("group", "")).strip(): row for row in rows}
        group_rates: Dict[str, float] = {}
        group_quality: Dict[str, float] = {}
        group_stability: Dict[str, float] = {}
        group_spectral: Dict[str, float] = {}

        for group, idxs in groups.items():
            if group not in row_by_group:
                continue
            idx_arr = np.asarray(idxs, dtype=int)
            sig_window = np.nanmedian(signal_mat[idx_arr, start:end], axis=0)
            rate_window = rate_mat[idx_arr, start:end].reshape(-1)
            valid_rates = rate_window[
                np.isfinite(rate_window)
                & (rate_window >= float(min_hz))
                & (rate_window <= float(max_hz))
            ]
            local_rates: List[float] = []
            local_weights: List[float] = []
            stability = 0.0
            track_rate = float("nan")
            if valid_rates.size >= max(3, int(0.08 * max(1, end - start))):
                med = float(np.median(valid_rates))
                std = float(np.std(valid_rates))
                stability = float(np.exp(-0.5 * (std / rate_ref) ** 2))
                track_rate = med
                local_rates.append(med)
                local_weights.append(max(stability, 0.05))
            spectral_score = 0.0
            spectral_candidates = _spectral_rate_candidates(
                sig_window,
                fps,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                max_candidates=4,
            )
            for _, rel_power in spectral_candidates:
                spectral_score = max(spectral_score, float(rel_power))
            if not np.isfinite(track_rate):
                for peak_hz, rel_power in spectral_candidates:
                    local_rates.append(float(peak_hz))
                    local_weights.append(float(rel_power))
            if not local_rates:
                continue
            if np.isfinite(track_rate):
                group_rate = float(track_rate)
                internal_support = max(float(stability), 0.25 * float(spectral_score), 0.05)
            else:
                group_rate, internal_support, _, _, _ = _track_first_rate_anchor(
                    np.asarray(local_rates, dtype=np.float64),
                    np.asarray(local_weights, dtype=np.float64),
                    min_hz=float(min_hz),
                    max_hz=float(max_hz),
                    rate_ref=rate_ref,
                )
            if not np.isfinite(group_rate):
                continue
            row = row_by_group[group]
            try:
                soft_score = float(row.get("score", np.nan))
                base = _timing_reliability_from_row(row, fallback=soft_score)
                rel = float(row.get("reliability_score", soft_score))
                rate_phase = float(row.get("rate_phase_score", 1.0))
            except Exception:
                base = float("nan")
                rel = float("nan")
                rate_phase = 1.0
            if not np.isfinite(base) or base <= 0.0:
                base = float(fallback_priors.get(group, rel))
            if not np.isfinite(base) or base <= 0.0:
                base = 0.25
            role_prior = float(REGIME_GROUP_PRIOR.get(group, 0.90))
            timing = float(np.sqrt(max(internal_support, 0.0) * max(stability, spectral_score, 0.05)))
            quality = (
                float(np.clip(base, 0.0, 1.0))
                * float(np.clip(rate_phase, 0.05, 1.0))
                * timing
                * role_prior
            )
            group_rates[group] = float(group_rate)
            group_quality[group] = float(np.clip(quality, 0.0, 1.50))
            group_stability[group] = float(np.clip(stability, 0.0, 1.0))
            group_spectral[group] = float(np.clip(spectral_score, 0.0, 1.0))

        if len(group_rates) < 2:
            continue

        rate_arr = np.asarray(list(group_rates.values()), dtype=np.float64)
        weight_arr = np.asarray([max(group_quality[g], 1e-6) for g in group_rates], dtype=np.float64)
        anchor, anchor_support = _harmonic_rate_anchor(
            rate_arr,
            weight_arr,
            min_hz=float(min_hz),
            max_hz=float(max_hz),
            rate_ref=rate_ref,
        )
        if not np.isfinite(anchor):
            continue
        group_names = list(group_rates)
        anchor, half_rate_rescued, half_rate_rescue_strength = _group_aware_half_rate_rescue(
            anchor,
            rate_arr,
            weight_arr,
            group_names,
            min_hz=float(min_hz),
            max_hz=float(max_hz),
            rate_ref=rate_ref,
        )
        if half_rate_rescued:
            anchor_support = max(float(anchor_support), float(half_rate_rescue_strength))

        support_to_anchor = {
            group: _harmonic_candidate_support(anchor, rate, rate_ref)
            for group, rate in group_rates.items()
        }
        supported_groups = [
            group for group, support in support_to_anchor.items()
            if float(support) * float(group_quality.get(group, 0.0)) >= 0.08
        ]
        support_group_count = len(set(supported_groups))
        norm_w = weight_arr / max(float(np.sum(weight_arr)), 1e-12)
        entropy = float(-np.sum(norm_w * np.log(np.clip(norm_w, 1e-12, 1.0))))
        diversity = float(np.clip(entropy / max(np.log(max(len(norm_w), 2)), 1e-12), 0.0, 1.0))

        def _group_score(*group_names: str) -> float:
            vals = [
                float(group_quality.get(g, 0.0)) * float(support_to_anchor.get(g, 0.0))
                for g in group_names
            ]
            return max(vals) if vals else 0.0

        motion_score = _group_score("G_DoF_bridge", "G_DoF")
        morphology_score = _group_score("G_P1D_cons", "G_P1D_morph", "G_P1D_low")
        velocity_score = _group_score("G_OF_bridge", "G_OF")
        support_mass = float(np.clip(sum(group_quality.values()) / max(len(group_quality), 1), 0.0, 1.0))
        weak_score = float(np.clip(1.0 - max(motion_score, morphology_score, velocity_score, anchor_support), 0.0, 1.0))
        regime = _regime_name_from_scores(motion_score, morphology_score, velocity_score, weak_score)
        regime_counts[regime] = int(regime_counts.get(regime, 0)) + 1

        group_context: Dict[str, float] = {}
        for group, quality in group_quality.items():
            support = float(support_to_anchor.get(group, 0.0))
            boost = 1.0
            if regime == "motion_energy":
                if group == "G_DoF_bridge":
                    boost = 1.45
                elif group == "G_DoF":
                    boost = 1.20
                elif group == "G_OF":
                    boost = 0.58
                elif group.startswith("G_P1D"):
                    boost = 0.82
            elif regime == "profile_morphology":
                if group in {"G_P1D_cons", "G_P1D_morph"}:
                    boost = 1.35
                elif group in {"G_DoF", "G_OF"}:
                    boost = 0.72
            elif regime == "optical_velocity":
                if group == "G_OF_bridge":
                    boost = 1.25
                elif group == "G_OF":
                    boost = 0.88
                elif group == "G_DoF":
                    boost = 0.78
            else:
                boost = 0.55
            if group in {"G_OF", "G_DoF"} and f"{group}_bridge" in group_quality:
                boost *= 0.88
            score = float(np.clip(quality * support * boost, 0.0, 1.35))
            group_context[group] = float(np.clip(REGIME_CHANNEL_FLOOR + 0.94 * score, REGIME_CHANNEL_FLOOR, 1.35))

        anchor_conf = float(np.clip(
            anchor_support
            * np.sqrt(max(diversity, 0.0))
            * min(1.0, support_group_count / 2.0)
            * (0.35 + 0.65 * support_mass),
            0.0,
            1.0,
        ))
        abstain = float(np.clip(1.0 - anchor_conf, 0.0, 1.0))
        if regime == "weak_observable":
            anchor_conf *= 0.35
        if anchor_conf <= 0.0:
            continue

        channel_context = _semantic_group_support(group_context, names)
        context_sum[:, start:end] += channel_context.reshape(-1, 1)
        context_count[:, start:end] += 1.0
        anchor_sum[start:end] += anchor_conf * float(anchor)
        anchor_conf_sum[start:end] += anchor_conf
        anchor_conf_count[start:end] += 1.0

        accepted += 1
        confidence_values.append(float(anchor_conf))
        support_group_counts.append(int(support_group_count))
        abstain_values.append(abstain)
        motion_values.append(float(motion_score))
        morphology_values.append(float(morphology_score))
        velocity_values.append(float(velocity_score))

    context_valid = context_count > 0.0
    if not np.any(context_valid):
        return None, None, None, {"enabled": False, "reason": "no_supported_windows"}

    context = np.full((n_channels, n), np.nan, dtype=np.float64)
    context[context_valid] = context_sum[context_valid] / np.maximum(context_count[context_valid], 1.0)
    for idx, name in enumerate(names):
        group = FAMILY_GROUPS.get(str(name), str(name))
        fallback = float(fallback_priors.get(group, np.nan))
        if not np.isfinite(fallback) or fallback <= 0.0:
            fallback = float(REGIME_GROUP_PRIOR.get(group, 0.80))
        row = context[idx]
        row[~np.isfinite(row)] = fallback
        context[idx] = np.clip(row, REGIME_CHANNEL_FLOOR, 1.35)

    valid_anchor = anchor_conf_sum > 1e-12
    anchor_track = np.full(n, np.nan, dtype=np.float64)
    anchor_conf_track = np.zeros(n, dtype=np.float64)
    anchor_track[valid_anchor] = anchor_sum[valid_anchor] / anchor_conf_sum[valid_anchor]
    anchor_conf_track[valid_anchor] = np.clip(
        anchor_conf_sum[valid_anchor] / np.maximum(anchor_conf_count[valid_anchor], 1.0),
        0.0,
        1.0,
    )

    meta: Dict[str, object] = {
        "enabled": True,
        "accepted_windows": int(accepted),
        "coverage": float(np.mean(context_valid.any(axis=0))),
        "anchor_coverage": float(np.mean(valid_anchor)),
        "anchor_confidence_mean": (
            float(np.mean(anchor_conf_track[valid_anchor])) if np.any(valid_anchor) else 0.0
        ),
        "window_confidence_median": float(np.median(confidence_values)) if confidence_values else float("nan"),
        "support_group_count_median": float(np.median(support_group_counts)) if support_group_counts else 0.0,
        "abstain_median": float(np.median(abstain_values)) if abstain_values else float("nan"),
        "motion_score_median": float(np.median(motion_values)) if motion_values else float("nan"),
        "morphology_score_median": float(np.median(morphology_values)) if morphology_values else float("nan"),
        "velocity_score_median": float(np.median(velocity_values)) if velocity_values else float("nan"),
        "regime_counts": dict(regime_counts),
    }
    return context.tolist(), anchor_track.tolist(), anchor_conf_track.tolist(), meta


def _decoupled_rate_readout_runtime(
    names: Sequence[str],
    signals: Sequence[np.ndarray],
    rate_tracks: Sequence[np.ndarray],
    window_rows: Sequence[Dict[str, float]],
    n_frames: int,
    fps: float,
    fallback_priors: Dict[str, float],
    min_hz: float,
    max_hz: float,
    enable_rate_hypothesis_graph_v4: bool = False,
    enable_derived_consistency_scaling: bool = False,
) -> Tuple[Optional[List[float]], Optional[List[float]], Dict[str, object]]:
    """GT-free output-rate readout for z_osc.

    The SSM still reconstructs z_full. This routine only asks which
    observation equation currently carries timing evidence. It can accept a
    single strong family when cross-family agreement is absent, but marks that
    case lower-confidence instead of forcing a hard latent-state correction.
    """
    n = int(max(n_frames, 0))
    if n <= 0 or fps <= 0.0 or not names or not window_rows:
        return None, None, {"enabled": False, "reason": "missing_inputs"}

    sig_rows = []
    for sig in signals:
        sig_rows.append(_fit_length(np.asarray(sig, dtype=np.float64), n, fill_value=0.0))
    rate_rows = []
    for track in rate_tracks:
        arr = _fit_length(np.asarray(track, dtype=np.float64), n)
        arr = np.where(
            np.isfinite(arr) & (arr >= float(min_hz)) & (arr <= float(max_hz)),
            arr,
            np.nan,
        )
        rate_rows.append(arr)
    if not sig_rows or not rate_rows:
        return None, None, {"enabled": False, "reason": "empty_inputs"}
    signal_mat = np.vstack(sig_rows)
    rate_mat = np.vstack(rate_rows)

    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        groups.setdefault(FAMILY_GROUPS.get(str(name), str(name)), []).append(idx)

    by_window: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
    for row in window_rows:
        try:
            start = int(max(0, round(float(row.get("start_sec", 0.0)) * float(fps))))
            end = int(min(n, round(float(row.get("end_sec", 0.0)) * float(fps))))
        except Exception:
            continue
        if end <= start:
            continue
        by_window.setdefault((start, end), []).append(row)
    if not by_window:
        return None, None, {"enabled": False, "reason": "no_valid_windows"}

    readout_sum = np.zeros(n, dtype=np.float64)
    conf_sum = np.zeros(n, dtype=np.float64)
    conf_count = np.zeros(n, dtype=np.float64)
    rate_ref = max(0.03, 0.10 * (float(max_hz) - float(min_hz)))

    accepted = 0
    mode_counts: Dict[str, int] = {}
    selected_counts: Dict[str, int] = {}
    group_rate_mode_counts: Dict[str, int] = {}
    derived_consistency_counts: Dict[str, int] = {}
    confidence_values: List[float] = []
    rate_values: List[float] = []
    support_counts: List[int] = []

    for (start, end), rows in sorted(by_window.items()):
        row_by_group = {str(row.get("group", "")).strip(): row for row in rows}
        group_rates: Dict[str, float] = {}
        group_scores: Dict[str, float] = {}
        group_spectral: Dict[str, float] = {}
        group_stability: Dict[str, float] = {}

        for group, idxs in groups.items():
            row = row_by_group.get(group)
            if row is None:
                continue
            idx_arr = np.asarray(idxs, dtype=int)
            rate_window = rate_mat[idx_arr, start:end].reshape(-1)
            valid_rates = rate_window[
                np.isfinite(rate_window)
                & (rate_window >= float(min_hz))
                & (rate_window <= float(max_hz))
            ]
            local_rates: List[float] = []
            local_weights: List[float] = []
            stability = 0.0
            track_rate = float("nan")
            if valid_rates.size >= max(3, int(0.08 * max(1, end - start))):
                med = float(np.median(valid_rates))
                std = float(np.std(valid_rates))
                stability = float(np.exp(-0.5 * (std / rate_ref) ** 2))
                track_rate = med
                local_rates.append(med)
                local_weights.append(max(stability, 0.05))

            sig_window = np.nanmedian(signal_mat[idx_arr, start:end], axis=0)
            spectral_score = 0.0
            spectral_candidates = _spectral_rate_candidates(
                sig_window,
                fps,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                max_candidates=4,
            )
            for _, rel_power in spectral_candidates:
                spectral_score = max(spectral_score, float(rel_power))
            if not np.isfinite(track_rate):
                for peak_hz, rel_power in spectral_candidates:
                    local_rates.append(float(peak_hz))
                    local_weights.append(float(rel_power))
            if not local_rates:
                continue

            group_rate, internal_support = _harmonic_rate_anchor(
                np.asarray(local_rates, dtype=np.float64),
                np.asarray(local_weights, dtype=np.float64),
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                rate_ref=rate_ref,
            )
            if not np.isfinite(group_rate):
                continue
            group_rate, group_rate_mode = _preserve_trustworthy_track_rate(
                group,
                float(track_rate),
                float(stability),
                float(group_rate),
                min_hz=float(min_hz),
                max_hz=float(max_hz),
            )
            try:
                soft_score = float(row.get("score", np.nan))
                base = _readout_timing_reliability_from_row(row, fallback=soft_score)
                rel = float(row.get("reliability_score", soft_score))
                rate_phase = float(row.get("rate_phase_score", 1.0))
            except Exception:
                base = float("nan")
                rel = float("nan")
                rate_phase = 1.0
            if not np.isfinite(base) or base <= 0.0:
                base = float(fallback_priors.get(group, rel))
            if not np.isfinite(base) or base <= 0.0:
                base = 0.20

            timing = float(np.sqrt(max(internal_support, 0.0) * max(stability, spectral_score, 0.05)))
            score = (
                float(np.clip(base, 0.0, 1.0))
                * float(np.clip(rate_phase, 0.05, 1.0))
                * timing
                * float(RATE_TIMING_GROUP_PRIOR.get(group, 0.80))
            )
            score = float(np.clip(score, 0.0, 1.50))
            if score <= 0.02:
                continue
            group_rates[group] = float(group_rate)
            group_scores[group] = score
            group_spectral[group] = float(np.clip(spectral_score, 0.0, 1.0))
            group_stability[group] = float(np.clip(stability, 0.0, 1.0))
            group_rate_mode_counts[str(group_rate_mode)] = int(group_rate_mode_counts.get(str(group_rate_mode), 0)) + 1

        if not group_rates:
            continue
        if enable_derived_consistency_scaling:
            for mode, count in _apply_derived_consistency_scaling(
                group_rates,
                group_scores,
                rate_ref=rate_ref,
            ).items():
                derived_consistency_counts[mode] = int(derived_consistency_counts.get(mode, 0)) + int(count)
        group_names = list(group_rates)
        rate_arr = np.asarray([group_rates[g] for g in group_names], dtype=np.float64)
        weight_arr = np.asarray([max(group_scores[g], 1e-6) for g in group_names], dtype=np.float64)
        direct_readout = _weighted_median(
            rate_arr,
            weight_arr,
            fallback=float(np.nanmedian(rate_arr)),
        )
        if enable_rate_hypothesis_graph_v4:
            graph_readout, graph_support, graph_mode, _ = _rate_hypothesis_graph_anchor(
                rate_arr,
                weight_arr,
                group_names,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                rate_ref=rate_ref,
            )
            readout = float(graph_readout if np.isfinite(graph_readout) else direct_readout)
            mode = str(graph_mode)
            agreement_support = float(graph_support)
        else:
            direct_support = 0.0
            if np.isfinite(direct_readout):
                total_weight = max(float(np.sum(weight_arr)), 1e-12)
                for rate, weight in zip(rate_arr, weight_arr):
                    direct_support += float(weight) * float(
                        np.exp(-0.5 * ((float(rate) - float(direct_readout)) / rate_ref) ** 2)
                    )
                direct_support /= total_weight

            harmonic_anchor, harmonic_support = _harmonic_rate_anchor(
                rate_arr,
                weight_arr,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                rate_ref=rate_ref,
            )
            use_harmonic_guard = (
                np.isfinite(harmonic_anchor)
                and np.isfinite(direct_readout)
                and harmonic_anchor < 0.72 * direct_readout
                and harmonic_support >= direct_support + 0.25
                and direct_readout >= 0.18
            )
            readout = float(harmonic_anchor if use_harmonic_guard else direct_readout)
            readout, half_rate_rescued, half_rate_rescue_strength = _group_aware_half_rate_rescue(
                readout,
                rate_arr,
                weight_arr,
                group_names,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                rate_ref=rate_ref,
            )
            mode = "multi_family_harmonic_guard" if use_harmonic_guard else "multi_family_direct"
            agreement_support = float(harmonic_support if use_harmonic_guard else direct_support)
            if half_rate_rescued:
                mode = "group_aware_half_rate_rescue"
                agreement_support = max(float(agreement_support), float(half_rate_rescue_strength))
        support_to_anchor = {
            group: float(np.exp(-0.5 * ((float(rate) - readout) / rate_ref) ** 2))
            for group, rate in group_rates.items()
        } if np.isfinite(readout) else {}
        support_group_count = sum(
            1 for group in group_names
            if float(group_scores[group]) * float(support_to_anchor.get(group, 0.0)) >= 0.06
        )
        strongest = max(group_names, key=lambda g: group_scores[g])
        strongest_score = float(group_scores[strongest])
        if (
            not np.isfinite(readout)
            or agreement_support < 0.30
            or support_group_count < 2
        ):
            if strongest_score < 0.12:
                continue
            readout = float(group_rates[strongest])
            agreement_support = max(float(agreement_support), 0.50 * strongest_score)
            support_group_count = max(support_group_count, 1)
            mode = "single_family_timing"

        norm_w = weight_arr / max(float(np.sum(weight_arr)), 1e-12)
        entropy = float(-np.sum(norm_w * np.log(np.clip(norm_w, 1e-12, 1.0))))
        diversity = (
            float(np.clip(entropy / max(np.log(max(norm_w.size, 2)), 1e-12), 0.0, 1.0))
            if norm_w.size > 1 else 0.55
        )
        support_mass = float(np.clip(np.sum(weight_arr) / max(len(weight_arr), 1), 0.0, 1.0))
        confidence = float(np.clip(
            (0.30 + 0.70 * max(agreement_support, 0.0))
            * (0.45 + 0.55 * diversity)
            * (0.35 + 0.65 * support_mass),
            0.0,
            1.0,
        ))
        if mode == "single_family_timing":
            confidence *= 0.70
        if strongest == "G_OF" and any(g.endswith("_bridge") for g in group_scores):
            confidence *= 0.85
        if mode == "rate_hypothesis_graph_p1d_half_resolved":
            confidence *= 1.05
        if confidence <= 0.02:
            continue

        readout = float(np.clip(readout, float(min_hz), float(max_hz)))
        readout_sum[start:end] += confidence * readout
        conf_sum[start:end] += confidence
        conf_count[start:end] += 1.0
        accepted += 1
        mode_counts[mode] = int(mode_counts.get(mode, 0)) + 1
        selected_counts[strongest] = int(selected_counts.get(strongest, 0)) + 1
        confidence_values.append(float(confidence))
        rate_values.append(readout)
        support_counts.append(int(support_group_count))

    valid = conf_sum > 1e-12
    if not np.any(valid):
        return None, None, {
            "enabled": False,
            "reason": "no_supported_readout_windows",
            "source": "abstained_no_supported_readout_windows",
            "calibration": "none",
            "abstention_guard": "no_supported_readout_windows",
            "coverage": 0.0,
            "release_hard_abstention": True,
        }
    readout_track = np.full(n, np.nan, dtype=np.float64)
    confidence_track = np.zeros(n, dtype=np.float64)
    readout_track[valid] = readout_sum[valid] / conf_sum[valid]
    confidence_track[valid] = np.clip(conf_sum[valid] / np.maximum(conf_count[valid], 1.0), 0.0, 1.0)
    fill_fraction = 0.0
    coverage = float(np.mean(valid))
    if 0.25 <= coverage < 1.0 and np.count_nonzero(valid) >= 2:
        idx = np.arange(n, dtype=np.float64)
        valid_idx = idx[valid]
        filled_rate = np.interp(idx, valid_idx, readout_track[valid])
        filled_conf = np.interp(idx, valid_idx, confidence_track[valid])
        fill_mask = ~valid
        readout_track[fill_mask] = filled_rate[fill_mask]
        confidence_track[fill_mask] = 0.55 * filled_conf[fill_mask]
        fill_fraction = float(np.mean(fill_mask))
    meta: Dict[str, object] = {
        "enabled": True,
        "accepted_windows": int(accepted),
        "coverage": coverage,
        "persistence_fill_fraction": fill_fraction,
        "confidence_mean": float(np.mean(confidence_track[valid])),
        "window_confidence_median": float(np.median(confidence_values)) if confidence_values else float("nan"),
        "readout_hz_median": float(np.median(rate_values)) if rate_values else float("nan"),
        "readout_hz_std": float(np.std(rate_values)) if rate_values else float("nan"),
        "support_group_count_median": float(np.median(support_counts)) if support_counts else 0.0,
        "mode_counts": dict(mode_counts),
        "selected_group_counts": dict(selected_counts),
        "group_rate_mode_counts": dict(group_rate_mode_counts),
        "derived_consistency_counts": dict(derived_consistency_counts),
    }
    return readout_track.tolist(), confidence_track.tolist(), meta


def _candidate_rate_posterior_runtime(
    names: Sequence[str],
    signals: Sequence[np.ndarray],
    rate_tracks: Sequence[np.ndarray],
    window_rows: Sequence[Dict[str, float]],
    n_frames: int,
    fps: float,
    fallback_priors: Dict[str, float],
    min_hz: float,
    max_hz: float,
    enable_derived_consistency_scaling: bool = False,
    enable_signal_sqi_observability: bool = False,
) -> Tuple[Optional[Dict[str, List[float]]], Dict[str, object]]:
    """Build a GT-free candidate-frequency posterior for the h1 oscillator.

    Unlike the decoupled output readout, this does not select a final BPM by
    itself. It exposes the posterior mode/mean/entropy/gap so PARH can treat
    target-side rate evidence as uncertainty-aware timing support.
    """
    n = int(max(n_frames, 0))
    if n <= 0 or fps <= 0.0 or not names or not window_rows:
        return None, {"enabled": False, "reason": "missing_inputs"}

    sig_rows = [_fit_length(np.asarray(sig, dtype=np.float64), n, fill_value=0.0) for sig in signals]
    rate_rows = []
    for track in rate_tracks:
        arr = _fit_length(np.asarray(track, dtype=np.float64), n)
        arr = np.where(
            np.isfinite(arr) & (arr >= float(min_hz)) & (arr <= float(max_hz)),
            arr,
            np.nan,
        )
        rate_rows.append(arr)
    if not sig_rows or not rate_rows:
        return None, {"enabled": False, "reason": "empty_inputs"}
    signal_mat = np.vstack(sig_rows)
    rate_mat = np.vstack(rate_rows)

    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        groups.setdefault(FAMILY_GROUPS.get(str(name), str(name)), []).append(idx)

    by_window: Dict[Tuple[int, int], List[Dict[str, float]]] = {}
    for row in window_rows:
        try:
            start = int(max(0, round(float(row.get("start_sec", 0.0)) * float(fps))))
            end = int(min(n, round(float(row.get("end_sec", 0.0)) * float(fps))))
        except Exception:
            continue
        if end <= start:
            continue
        by_window.setdefault((start, end), []).append(row)
    if not by_window:
        return None, {"enabled": False, "reason": "no_valid_windows"}

    keys = (
        "mode_hz",
        "mean_hz",
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
        "signal_sqi_support",
        "phase_coherence_support",
    )
    sums = {key: np.zeros(n, dtype=np.float64) for key in keys}
    conf_sum = np.zeros(n, dtype=np.float64)
    conf_count = np.zeros(n, dtype=np.float64)
    rate_ref = max(0.03, 0.10 * (float(max_hz) - float(min_hz)))

    accepted = 0
    confidence_values: List[float] = []
    entropy_values: List[float] = []
    top_gap_values: List[float] = []
    mode_values: List[float] = []
    macro_support_values: List[float] = []
    direct_macro_support_values: List[float] = []
    motion_direct_support_values: List[float] = []
    alias_risk_values: List[float] = []
    independent_timing_values: List[float] = []
    bridge_preservation_values: List[float] = []
    morphology_alias_values: List[float] = []
    h1_role_values: List[float] = []
    abstain_values: List[float] = []
    signal_sqi_values: List[float] = []
    phase_coherence_values: List[float] = []
    mode_counts: Dict[str, int] = {}
    group_rate_mode_counts: Dict[str, int] = {}
    derived_consistency_counts: Dict[str, int] = {}

    for (start, end), rows in sorted(by_window.items()):
        row_by_group = {str(row.get("group", "")).strip(): row for row in rows}
        group_rates: Dict[str, float] = {}
        group_scores: Dict[str, float] = {}
        group_signal_sqi: Dict[str, float] = {}
        group_phase: Dict[str, float] = {}

        for group, idxs in groups.items():
            row = row_by_group.get(group)
            if row is None:
                continue
            idx_arr = np.asarray(idxs, dtype=int)
            local_rates: List[float] = []
            local_weights: List[float] = []
            valid_rates = rate_mat[idx_arr, start:end].reshape(-1)
            valid_rates = valid_rates[
                np.isfinite(valid_rates)
                & (valid_rates >= float(min_hz))
                & (valid_rates <= float(max_hz))
            ]
            track_rate = float("nan")
            stability = 0.0
            if valid_rates.size >= max(3, int(0.08 * max(1, end - start))):
                track_rate = float(np.median(valid_rates))
                std = float(np.std(valid_rates))
                stability = float(np.exp(-0.5 * (std / rate_ref) ** 2))
                local_rates.append(track_rate)
                local_weights.append(max(stability, 0.05))

            sig_window = np.nanmedian(signal_mat[idx_arr, start:end], axis=0)
            spectral_candidates = _spectral_rate_candidates(
                sig_window,
                fps,
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                max_candidates=4,
            )
            spectral_score = max((float(rel_power) for _, rel_power in spectral_candidates), default=0.0)
            if not np.isfinite(track_rate):
                for peak_hz, rel_power in spectral_candidates:
                    local_rates.append(float(peak_hz))
                    local_weights.append(float(rel_power))
            if not local_rates:
                continue

            group_rate, internal_support = _harmonic_rate_anchor(
                np.asarray(local_rates, dtype=np.float64),
                np.asarray(local_weights, dtype=np.float64),
                min_hz=float(min_hz),
                max_hz=float(max_hz),
                rate_ref=rate_ref,
            )
            if not np.isfinite(group_rate):
                continue
            group_rate, group_rate_mode = _preserve_trustworthy_track_rate(
                group,
                float(track_rate),
                float(stability),
                float(group_rate),
                min_hz=float(min_hz),
                max_hz=float(max_hz),
            )
            sqi_features = (
                _signal_sqi_features(
                    sig_window,
                    fps,
                    min_hz=float(min_hz),
                    max_hz=float(max_hz),
                    freq_hint=float(group_rate),
                )
                if bool(enable_signal_sqi_observability)
                else {"signal_sqi": 1.0, "phase": float("nan")}
            )
            signal_sqi = float(np.clip(float(sqi_features.get("signal_sqi", 1.0)), 0.0, 1.0))

            soft_score = _finite_row_float(row, "score")
            base = _readout_timing_reliability_from_row(row, fallback=soft_score)
            if not np.isfinite(base) or base <= 0.0:
                base = float(fallback_priors.get(group, np.nan))
            if not np.isfinite(base) or base <= 0.0:
                base = 0.20
            rate_phase = _finite_row_float(row, "rate_phase_score", 1.0)
            timing = float(np.sqrt(max(internal_support, 0.0) * max(stability, spectral_score, 0.05)))
            score = (
                float(np.clip(base, 0.0, 1.0))
                * float(np.clip(rate_phase, 0.05, 1.0))
                * timing
                * float(RATE_TIMING_GROUP_PRIOR.get(group, 0.80))
            )
            if bool(enable_signal_sqi_observability):
                # SQI is supporting evidence, not a hidden source selector.  Do
                # not let low single-signal SQI erase cross-family agreement.
                score *= float(0.90 + 0.20 * signal_sqi)
            score = float(np.clip(score, 0.0, 1.50))
            if score <= 0.02:
                continue
            group_rates[group] = float(group_rate)
            group_scores[group] = score
            group_signal_sqi[group] = signal_sqi
            group_phase[group] = float(sqi_features.get("phase", float("nan")))
            group_rate_mode_counts[str(group_rate_mode)] = int(group_rate_mode_counts.get(str(group_rate_mode), 0)) + 1

        if not group_rates:
            continue
        if enable_derived_consistency_scaling:
            for mode, count in _apply_derived_consistency_scaling(
                group_rates,
                group_scores,
                rate_ref=rate_ref,
            ).items():
                derived_consistency_counts[mode] = int(derived_consistency_counts.get(mode, 0)) + int(count)

        group_names = list(group_rates)
        rate_arr = np.asarray([group_rates[g] for g in group_names], dtype=np.float64)
        weight_arr = np.asarray([max(group_scores[g], 1e-6) for g in group_names], dtype=np.float64)
        graph_mode_hz, graph_support, graph_mode, detail = _rate_hypothesis_graph_anchor(
            rate_arr,
            weight_arr,
            group_names,
            min_hz=float(min_hz),
            max_hz=float(max_hz),
            rate_ref=rate_ref,
        )
        top_candidates = []
        if isinstance(detail, dict):
            raw_candidates = detail.get("top_candidates", [])
            if isinstance(raw_candidates, list):
                top_candidates = [row for row in raw_candidates if isinstance(row, dict)]
        if not top_candidates:
            for rate, weight, group in zip(rate_arr, weight_arr, group_names):
                top_candidates.append(
                    {
                        "hz": float(rate),
                        "score": float(weight),
                        "support": float(weight),
                        "direct": float(weight),
                        "macro_count": 1,
                        "group": str(group),
                    }
                )
        candidates = []
        for row in top_candidates:
            hz = _finite_row_float(row, "hz")
            score = _finite_row_float(row, "score")
            if np.isfinite(hz) and float(min_hz) <= hz <= float(max_hz) and np.isfinite(score) and score > 0.0:
                candidates.append(
                    {
                        "hz": float(hz),
                        "score": float(score),
                        "support": _finite_row_float(row, "support", score),
                        "direct": _finite_row_float(row, "direct", 0.0),
                        "macro_count": _finite_row_float(row, "macro_count", 1.0),
                    }
                )
        if not candidates:
            continue
        candidates.sort(key=lambda row: (float(row["score"]), float(row["direct"])), reverse=True)
        scores = np.asarray([max(float(row["score"]), 1e-9) for row in candidates], dtype=np.float64)
        probs = scores / max(float(np.sum(scores)), 1e-12)
        hz_arr = np.asarray([float(row["hz"]) for row in candidates], dtype=np.float64)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
        entropy_norm = (
            float(np.clip(entropy / max(math.log(max(len(candidates), 2)), 1e-12), 0.0, 1.0))
            if len(candidates) > 1 else 0.0
        )
        top_score = float(scores[0])
        second_score = float(scores[1]) if scores.size > 1 else 0.0
        top_gap = float(np.clip((top_score - second_score) / max(top_score, 1e-9), 0.0, 1.0))
        mode_hz = float(candidates[0]["hz"])
        if np.isfinite(graph_mode_hz):
            mode_hz = float(graph_mode_hz)
        mean_hz = float(np.sum(probs * hz_arr))
        possible_macros = max(len({_macro_timing_family(group) for group in group_names}), 1)
        macro_support = float(np.clip(float(candidates[0].get("macro_count", 1.0)) / possible_macros, 0.0, 1.0))
        direct_macro_support = float(
            np.clip(float(candidates[0].get("direct_macro_count", candidates[0].get("macro_count", 1.0))) / possible_macros, 0.0, 1.0)
        )
        motion_direct_support = float(np.clip(float(candidates[0].get("motion_direct", 0.0)), 0.0, 1.0))
        p1d_half_support = (
            float(_finite_row_float(detail, "best_p1d_half_support", 0.0))
            if isinstance(detail, dict)
            else 0.0
        )
        if isinstance(detail, dict):
            motion_direct_support = float(
                np.clip(_finite_row_float(detail, "best_motion_direct", motion_direct_support), 0.0, 1.0)
            )
        direct_support = float(_finite_row_float(detail, "direct_support", float(candidates[0]["direct"]))) if isinstance(detail, dict) else float(candidates[0]["direct"])
        alias_risk = float(np.clip(p1d_half_support * (1.0 - motion_direct_support), 0.0, 1.0))
        support = float(max(graph_support, float(candidates[0]["support"])))
        signal_sqi_support = 1.0
        phase_coherence_support = 1.0
        if bool(enable_signal_sqi_observability):
            sqi_num = 0.0
            sqi_den = 0.0
            phases: List[float] = []
            phase_weights: List[float] = []
            for group in group_names:
                rate = float(group_rates.get(group, np.nan))
                score = float(group_scores.get(group, 0.0))
                if not (np.isfinite(rate) and np.isfinite(score) and score > 0.0):
                    continue
                candidate_support = _direct_candidate_support(float(mode_hz), rate, rate_ref)
                weight = score * candidate_support
                sqi = float(group_signal_sqi.get(group, 0.0))
                sqi_num += weight * sqi
                sqi_den += weight
                phase = float(group_phase.get(group, float("nan")))
                if np.isfinite(phase) and weight > 1e-9:
                    phases.append(phase)
                    phase_weights.append(weight)
            signal_sqi_support = float(np.clip(sqi_num / sqi_den, 0.0, 1.0)) if sqi_den > 1e-12 else 0.0
            if phases:
                _, phase_coherence_support = _weighted_circular_mean(
                    np.asarray(phases, dtype=np.float64),
                    np.asarray(phase_weights, dtype=np.float64),
                )
                phase_coherence_support = float(np.clip(phase_coherence_support, 0.0, 1.0))
            else:
                phase_coherence_support = 0.0
        independent_timing_support = (
            float(np.clip(_finite_row_float(detail, "best_independent_timing_support", 0.0), 0.0, 1.0))
            if isinstance(detail, dict)
            else 0.0
        )
        motion_timing_support = (
            float(np.clip(_finite_row_float(detail, "best_motion_timing_support", motion_direct_support), 0.0, 1.0))
            if isinstance(detail, dict)
            else motion_direct_support
        )
        bridge_timing_preservation = (
            float(np.clip(_finite_row_float(detail, "best_bridge_timing_preservation", 0.0), 0.0, 1.0))
            if isinstance(detail, dict)
            else 0.0
        )
        p1d_direct_timing_support = (
            float(np.clip(_finite_row_float(detail, "best_p1d_direct_timing_support", 0.0), 0.0, 1.0))
            if isinstance(detail, dict)
            else 0.0
        )
        morphology_role_support = (
            float(np.clip(_finite_row_float(detail, "best_morphology_role_support", support), 0.0, 1.0))
            if isinstance(detail, dict)
            else float(np.clip(support, 0.0, 1.0))
        )
        morphology_alias_pressure = (
            float(np.clip(_finite_row_float(detail, "best_morphology_alias_pressure", alias_risk), 0.0, 1.0))
            if isinstance(detail, dict)
            else alias_risk
        )
        h1_role_support = (
            float(np.clip(_finite_row_float(detail, "best_h1_role_support", independent_timing_support), 0.0, 1.0))
            if isinstance(detail, dict)
            else independent_timing_support
        )
        abstain_pressure = (
            float(np.clip(_finite_row_float(detail, "best_abstain_pressure", 1.0 - h1_role_support), 0.0, 1.0))
            if isinstance(detail, dict)
            else float(np.clip(1.0 - h1_role_support, 0.0, 1.0))
        )
        confidence = float(np.clip(
            (0.25 + 0.75 * support)
            * (0.35 + 0.65 * top_gap)
            * (0.45 + 0.55 * (1.0 - entropy_norm))
            * (0.55 + 0.45 * macro_support),
            0.0,
            1.0,
        ))
        confidence *= float(1.0 - 0.45 * alias_risk)
        if bool(enable_signal_sqi_observability):
            confidence *= float(0.92 + 0.06 * signal_sqi_support + 0.02 * phase_coherence_support)
        if len(group_names) < 2:
            confidence *= 0.70
        if confidence <= 0.02:
            continue

        values = {
            "mode_hz": mode_hz,
            "mean_hz": mean_hz,
            "entropy": entropy_norm,
            "top_gap": top_gap,
            "support": support,
            "direct_support": direct_support,
            "macro_support": macro_support,
            "direct_macro_support": direct_macro_support,
            "motion_direct_support": motion_direct_support,
            "p1d_half_support": p1d_half_support,
            "alias_risk": alias_risk,
            "independent_timing_support": independent_timing_support,
            "motion_timing_support": motion_timing_support,
            "bridge_timing_preservation": bridge_timing_preservation,
            "p1d_direct_timing_support": p1d_direct_timing_support,
            "morphology_role_support": morphology_role_support,
            "morphology_alias_pressure": morphology_alias_pressure,
            "h1_role_support": h1_role_support,
            "abstain_pressure": abstain_pressure,
            "signal_sqi_support": signal_sqi_support,
            "phase_coherence_support": phase_coherence_support,
        }
        for key, value in values.items():
            sums[key][start:end] += confidence * float(value)
        conf_sum[start:end] += confidence
        conf_count[start:end] += 1.0
        accepted += 1
        confidence_values.append(confidence)
        entropy_values.append(entropy_norm)
        top_gap_values.append(top_gap)
        mode_values.append(mode_hz)
        macro_support_values.append(macro_support)
        direct_macro_support_values.append(direct_macro_support)
        motion_direct_support_values.append(motion_direct_support)
        alias_risk_values.append(alias_risk)
        independent_timing_values.append(independent_timing_support)
        bridge_preservation_values.append(bridge_timing_preservation)
        morphology_alias_values.append(morphology_alias_pressure)
        h1_role_values.append(h1_role_support)
        abstain_values.append(abstain_pressure)
        signal_sqi_values.append(signal_sqi_support)
        phase_coherence_values.append(phase_coherence_support)
        mode_counts[str(graph_mode)] = int(mode_counts.get(str(graph_mode), 0)) + 1

    valid = conf_sum > 1e-12
    if not np.any(valid):
        return None, {"enabled": False, "reason": "no_supported_posterior_windows"}

    runtime_arrays: Dict[str, np.ndarray] = {}
    for key, arr_sum in sums.items():
        arr = np.full(n, np.nan, dtype=np.float64)
        arr[valid] = arr_sum[valid] / conf_sum[valid]
        runtime_arrays[key] = arr
    confidence_track = np.zeros(n, dtype=np.float64)
    confidence_track[valid] = np.clip(conf_sum[valid] / np.maximum(conf_count[valid], 1.0), 0.0, 1.0)

    fill_fraction = 0.0
    coverage = float(np.mean(valid))
    if 0.25 <= coverage < 1.0 and np.count_nonzero(valid) >= 2:
        idx = np.arange(n, dtype=np.float64)
        valid_idx = idx[valid]
        fill_mask = ~valid
        for key, arr in runtime_arrays.items():
            filled = np.interp(idx, valid_idx, arr[valid])
            arr[fill_mask] = filled[fill_mask]
            runtime_arrays[key] = arr
        filled_conf = np.interp(idx, valid_idx, confidence_track[valid])
        confidence_track[fill_mask] = 0.50 * filled_conf[fill_mask]
        fill_fraction = float(np.mean(fill_mask))

    runtime: Dict[str, List[float]] = {
        key: runtime_arrays[key].tolist()
        for key in keys
    }
    runtime["confidence"] = confidence_track.tolist()
    meta: Dict[str, object] = {
        "enabled": True,
        "accepted_windows": int(accepted),
        "coverage": coverage,
        "persistence_fill_fraction": fill_fraction,
        "confidence_mean": float(np.mean(confidence_track[confidence_track > 0.0])) if np.any(confidence_track > 0.0) else 0.0,
        "posterior_entropy_median": float(np.median(entropy_values)) if entropy_values else float("nan"),
        "posterior_top_gap_median": float(np.median(top_gap_values)) if top_gap_values else float("nan"),
        "mode_hz_median": float(np.median(mode_values)) if mode_values else float("nan"),
        "macro_support_median": float(np.median(macro_support_values)) if macro_support_values else float("nan"),
        "direct_macro_support_median": (
            float(np.median(direct_macro_support_values)) if direct_macro_support_values else float("nan")
        ),
        "motion_direct_support_median": (
            float(np.median(motion_direct_support_values)) if motion_direct_support_values else float("nan")
        ),
        "alias_risk_median": float(np.median(alias_risk_values)) if alias_risk_values else float("nan"),
        "independent_timing_support_median": (
            float(np.median(independent_timing_values)) if independent_timing_values else float("nan")
        ),
        "bridge_timing_preservation_median": (
            float(np.median(bridge_preservation_values)) if bridge_preservation_values else float("nan")
        ),
        "morphology_alias_pressure_median": (
            float(np.median(morphology_alias_values)) if morphology_alias_values else float("nan")
        ),
        "h1_role_support_median": float(np.median(h1_role_values)) if h1_role_values else float("nan"),
        "abstain_pressure_median": float(np.median(abstain_values)) if abstain_values else float("nan"),
        "signal_sqi_observability_enabled": bool(enable_signal_sqi_observability),
        "signal_sqi_support_median": float(np.median(signal_sqi_values)) if signal_sqi_values else float("nan"),
        "phase_coherence_support_median": (
            float(np.median(phase_coherence_values)) if phase_coherence_values else float("nan")
        ),
        "mode_counts": dict(mode_counts),
        "group_rate_mode_counts": dict(group_rate_mode_counts),
        "derived_consistency_counts": dict(derived_consistency_counts),
    }
    return runtime, meta


def _target_observability_control_runtime(
    names: Sequence[str],
    rate_tracks: Sequence[np.ndarray],
    output_rate: Optional[Sequence[float]],
    output_confidence: Optional[Sequence[float]],
    posterior_runtime: Optional[Dict[str, Sequence[float]]],
    n_frames: int,
    *,
    min_hz: float,
    max_hz: float,
    signals: Optional[Sequence[np.ndarray]] = None,
    window_rows: Sequence[Dict[str, float]] = (),
    fps: float = 0.0,
    enable_signal_sqi_observability: bool = False,
) -> Tuple[Optional[Dict[str, List[float]]], Dict[str, object]]:
    """Build GT-free observation-trust controls for PARH's update law.

    This is deliberately not a rate source selector. It exposes whether the
    current target window has enough reliable, non-aliased evidence to update
    the resonant state strongly.
    """
    n = int(max(n_frames, 0))
    if n <= 0:
        return None, {"enabled": False, "reason": "empty_trial"}
    rate_rows = []
    for track in rate_tracks:
        arr = _fit_length(np.asarray(track, dtype=np.float64), n)
        arr = np.where(
            np.isfinite(arr) & (arr >= float(min_hz)) & (arr <= float(max_hz)),
            arr,
            np.nan,
        )
        rate_rows.append(arr)
    if not rate_rows:
        return None, {"enabled": False, "reason": "missing_rate_tracks"}

    groups: Dict[str, List[int]] = {}
    for idx, name in enumerate(names):
        groups.setdefault(FAMILY_GROUPS.get(str(name), str(name)), []).append(idx)
    group_tracks: List[np.ndarray] = []
    rate_mat = np.vstack(rate_rows)
    for idxs in groups.values():
        idx_arr = np.asarray(idxs, dtype=int)
        with np.errstate(all="ignore"):
            group_tracks.append(np.nanmedian(rate_mat[idx_arr], axis=0))
    if not group_tracks:
        return None, {"enabled": False, "reason": "missing_group_tracks"}
    group_mat = np.vstack(group_tracks)
    finite_counts = np.sum(np.isfinite(group_mat), axis=0)
    with np.errstate(all="ignore"):
        source_spread_hz = np.nanmax(group_mat, axis=0) - np.nanmin(group_mat, axis=0)
    source_spread_hz = np.where(finite_counts >= 2, source_spread_hz, np.nan)
    spread_fill = float(np.nanmedian(source_spread_hz[np.isfinite(source_spread_hz)])) if np.any(np.isfinite(source_spread_hz)) else 0.0
    source_spread_hz = np.nan_to_num(source_spread_hz, nan=spread_fill, posinf=spread_fill, neginf=0.0)
    source_spread_hz = np.clip(source_spread_hz, 0.0, float(max_hz) - float(min_hz))
    source_agreement = np.exp(-source_spread_hz / 0.055)
    source_agreement = np.clip(np.nan_to_num(source_agreement, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    if output_rate is not None and output_confidence is not None:
        out_conf = np.clip(
            np.nan_to_num(_fit_length(np.asarray(output_confidence, dtype=np.float64), n), nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )
    else:
        out_conf = np.zeros(n, dtype=np.float64)

    posterior = dict(posterior_runtime or {})
    post_conf = np.clip(np.nan_to_num(_fit_length(np.asarray(posterior.get("confidence", []), dtype=np.float64), n), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    entropy = np.clip(np.nan_to_num(_fit_length(np.asarray(posterior.get("entropy", []), dtype=np.float64), n), nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    top_gap = np.clip(np.nan_to_num(_fit_length(np.asarray(posterior.get("top_gap", []), dtype=np.float64), n), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    alias_risk = np.clip(np.nan_to_num(_fit_length(np.asarray(posterior.get("alias_risk", []), dtype=np.float64), n), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    h1_role = np.clip(np.nan_to_num(_fit_length(np.asarray(posterior.get("h1_role_support", []), dtype=np.float64), n), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    morph_role = np.clip(np.nan_to_num(_fit_length(np.asarray(posterior.get("morphology_role_support", []), dtype=np.float64), n), nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
    morph_alias_raw = _fit_length(
        np.asarray(posterior.get("morphology_alias_pressure", []), dtype=np.float64),
        n,
    )
    morph_alias = np.where(np.isfinite(morph_alias_raw), morph_alias_raw, alias_risk)
    morph_alias = np.clip(np.nan_to_num(morph_alias, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    abstain = np.clip(np.nan_to_num(_fit_length(np.asarray(posterior.get("abstain_pressure", []), dtype=np.float64), n), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    posterior_signal_sqi = np.clip(
        np.nan_to_num(
            _fit_length(np.asarray(posterior.get("signal_sqi_support", []), dtype=np.float64), n),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )
    posterior_phase_coherence = np.clip(
        np.nan_to_num(
            _fit_length(np.asarray(posterior.get("phase_coherence_support", []), dtype=np.float64), n),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )

    sqi_runtime = None
    sqi_meta: Dict[str, object] = {"enabled": False}
    if bool(enable_signal_sqi_observability) and signals is not None and fps > 0.0:
        sqi_runtime, sqi_meta = _windowed_signal_sqi_observability(
            names,
            signals,
            rate_tracks,
            window_rows,
            n,
            float(fps),
            min_hz=float(min_hz),
            max_hz=float(max_hz),
        )
    if sqi_runtime is not None:
        signal_sqi = np.clip(
            0.60 * np.asarray(sqi_runtime["signal_sqi"], dtype=np.float64)
            + 0.25 * np.asarray(sqi_runtime["phase_fit"], dtype=np.float64)
            + 0.15 * np.asarray(sqi_runtime["phase_coherence"], dtype=np.float64),
            0.0,
            1.0,
        )
        phase_coherence = np.clip(np.asarray(sqi_runtime["phase_coherence"], dtype=np.float64), 0.0, 1.0)
    elif bool(enable_signal_sqi_observability):
        signal_sqi = posterior_signal_sqi
        phase_coherence = posterior_phase_coherence
    else:
        signal_sqi = np.zeros(n, dtype=np.float64)
        phase_coherence = np.zeros(n, dtype=np.float64)

    gap_score = np.clip((top_gap - 0.08) / 0.20, 0.0, 1.0)
    entropy_score = np.clip((0.98 - entropy) / 0.25, 0.0, 1.0)
    posterior_specificity = np.clip(0.55 * gap_score + 0.25 * entropy_score + 0.20 * post_conf, 0.0, 1.0)
    alias_safety = np.clip(1.0 - np.maximum(alias_risk, morph_alias), 0.0, 1.0)
    source_disagreement = np.clip(1.0 - source_agreement, 0.0, 1.0)
    if bool(enable_signal_sqi_observability):
        h1_timing = np.clip(
            0.22 * out_conf
            + 0.26 * posterior_specificity
            + 0.16 * h1_role
            + 0.10 * alias_safety
            + 0.08 * source_disagreement
            + 0.18 * signal_sqi,
            0.0,
            1.0,
        )
        h2_morphology = np.clip(
            0.34 * morph_role
            + 0.20 * alias_safety
            + 0.14 * out_conf
            + 0.14 * source_agreement
            + 0.12 * signal_sqi
            + 0.06 * phase_coherence,
            0.0,
            1.0,
        )
        target_observability = np.clip(
            0.24 * out_conf
            + 0.20 * posterior_specificity
            + 0.14 * alias_safety
            + 0.12 * h1_role
            + 0.10 * source_agreement
            + 0.20 * signal_sqi
            - 0.10 * abstain,
            0.0,
            1.0,
        )
    else:
        h1_timing = np.clip(
            0.28 * out_conf
            + 0.34 * posterior_specificity
            + 0.18 * h1_role
            + 0.12 * alias_safety
            + 0.08 * source_disagreement,
            0.0,
            1.0,
        )
        h2_morphology = np.clip(
            0.40 * morph_role
            + 0.24 * alias_safety
            + 0.18 * out_conf
            + 0.18 * source_agreement,
            0.0,
            1.0,
        )
        target_observability = np.clip(
            0.30 * out_conf
            + 0.24 * posterior_specificity
            + 0.18 * alias_safety
            + 0.16 * h1_role
            + 0.12 * source_agreement
            - 0.10 * abstain,
            0.0,
            1.0,
        )
    nuisance = np.clip(
        0.45 * np.maximum(alias_risk, morph_alias)
        + 0.30 * abstain
        + 0.25 * np.maximum(0.0, 0.50 - target_observability),
        0.0,
        1.0,
    )
    if bool(enable_signal_sqi_observability):
        nuisance = np.clip(0.75 * nuisance + 0.25 * (1.0 - signal_sqi), 0.0, 1.0)
    baseline = np.clip(0.70 * target_observability + 0.30 * source_agreement, 0.0, 1.0)
    residual = np.clip(0.55 * h2_morphology + 0.25 * target_observability + 0.20 * (1.0 - nuisance), 0.0, 1.0)

    runtime = {
        "target_observability": target_observability.tolist(),
        "h1_timing": h1_timing.tolist(),
        "h2_morphology": h2_morphology.tolist(),
        "baseline": baseline.tolist(),
        "residual": residual.tolist(),
        "nuisance": nuisance.tolist(),
        "source_spread_hz": source_spread_hz.tolist(),
        "source_agreement": source_agreement.tolist(),
        "posterior_specificity": posterior_specificity.tolist(),
        "alias_safety": alias_safety.tolist(),
        "signal_sqi": signal_sqi.tolist(),
        "phase_coherence": phase_coherence.tolist(),
    }
    meta = {
        "enabled": True,
        "semantics": "target_computable_observability_control_not_output_override",
        "signal_sqi_observability_enabled": bool(enable_signal_sqi_observability),
        "signal_sqi_meta": sqi_meta,
        "n_groups": int(len(group_tracks)),
        "source_spread_hz_median": float(np.median(source_spread_hz)),
        "source_spread_bpm_median": float(60.0 * np.median(source_spread_hz)),
        "target_observability_mean": float(np.mean(target_observability)),
        "h1_timing_mean": float(np.mean(h1_timing)),
        "h2_morphology_mean": float(np.mean(h2_morphology)),
        "nuisance_mean": float(np.mean(nuisance)),
        "signal_sqi_mean": float(np.mean(signal_sqi)),
        "phase_coherence_mean": float(np.mean(phase_coherence)),
    }
    return runtime, meta


def _calibrated_posterior_mean_readout(
    base_rate: Sequence[float],
    base_confidence: Sequence[float],
    posterior_runtime: Dict[str, List[float]],
    *,
    min_hz: float,
    max_hz: float,
    enable_specificity_boost: bool = False,
    enable_macro_guard: bool = False,
    enable_role_guard: bool = False,
) -> Tuple[Optional[List[float]], Optional[List[float]], Dict[str, object]]:
    """Use posterior mean as a bounded correction to the existing z_osc readout."""
    if not posterior_runtime:
        return None, None, {"enabled": False, "reason": "missing_posterior"}
    base = np.asarray(base_rate, dtype=np.float64).reshape(-1)
    base_conf = np.asarray(base_confidence, dtype=np.float64).reshape(-1)
    post = np.asarray(posterior_runtime.get("mean_hz", []), dtype=np.float64).reshape(-1)
    post_conf = np.asarray(posterior_runtime.get("confidence", []), dtype=np.float64).reshape(-1)
    entropy = np.asarray(posterior_runtime.get("entropy", []), dtype=np.float64).reshape(-1)
    top_gap = np.asarray(posterior_runtime.get("top_gap", []), dtype=np.float64).reshape(-1)
    support = np.asarray(posterior_runtime.get("support", []), dtype=np.float64).reshape(-1)
    direct_support = np.asarray(posterior_runtime.get("direct_support", []), dtype=np.float64).reshape(-1)
    macro = np.asarray(posterior_runtime.get("macro_support", []), dtype=np.float64).reshape(-1)
    direct_macro = np.asarray(posterior_runtime.get("direct_macro_support", []), dtype=np.float64).reshape(-1)
    motion_direct = np.asarray(posterior_runtime.get("motion_direct_support", []), dtype=np.float64).reshape(-1)
    p1d_half = np.asarray(posterior_runtime.get("p1d_half_support", []), dtype=np.float64).reshape(-1)
    alias_risk = np.asarray(posterior_runtime.get("alias_risk", []), dtype=np.float64).reshape(-1)
    h1_role = np.asarray(posterior_runtime.get("h1_role_support", []), dtype=np.float64).reshape(-1)
    morphology_alias = np.asarray(posterior_runtime.get("morphology_alias_pressure", []), dtype=np.float64).reshape(-1)
    abstain_pressure = np.asarray(posterior_runtime.get("abstain_pressure", []), dtype=np.float64).reshape(-1)
    n = min(
        base.size,
        base_conf.size,
        post.size,
        post_conf.size,
        entropy.size,
        top_gap.size,
        support.size,
        direct_support.size,
        macro.size,
        direct_macro.size if direct_macro.size else macro.size,
        motion_direct.size if motion_direct.size else macro.size,
        p1d_half.size,
        alias_risk.size if alias_risk.size else macro.size,
        h1_role.size if h1_role.size else macro.size,
        morphology_alias.size if morphology_alias.size else macro.size,
        abstain_pressure.size if abstain_pressure.size else macro.size,
    )
    if n <= 0:
        return None, None, {"enabled": False, "reason": "empty_arrays"}
    base = base[:n]
    base_conf = np.clip(np.nan_to_num(base_conf[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    post = post[:n]
    post_conf = np.clip(np.nan_to_num(post_conf[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    entropy = np.clip(np.nan_to_num(entropy[:n], nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    top_gap = np.clip(np.nan_to_num(top_gap[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    support = np.clip(np.nan_to_num(support[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    direct_support = np.clip(np.nan_to_num(direct_support[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    macro = np.clip(np.nan_to_num(macro[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if direct_macro.size:
        direct_macro = np.clip(np.nan_to_num(direct_macro[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    else:
        direct_macro = macro.copy()
    if motion_direct.size:
        motion_direct = np.clip(np.nan_to_num(motion_direct[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    else:
        motion_direct = np.zeros(n, dtype=np.float64)
    p1d_half = np.clip(np.nan_to_num(p1d_half[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if alias_risk.size:
        alias_risk = np.clip(np.nan_to_num(alias_risk[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    else:
        alias_risk = np.clip(p1d_half * (1.0 - motion_direct), 0.0, 1.0)
    if h1_role.size:
        h1_role = np.clip(np.nan_to_num(h1_role[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    else:
        h1_role = direct_macro.copy()
    if morphology_alias.size:
        morphology_alias = np.clip(np.nan_to_num(morphology_alias[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    else:
        morphology_alias = alias_risk.copy()
    if abstain_pressure.size:
        abstain_pressure = np.clip(np.nan_to_num(abstain_pressure[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    else:
        abstain_pressure = np.clip(1.0 - h1_role, 0.0, 1.0)
    valid = (
        np.isfinite(base)
        & np.isfinite(post)
        & (base >= float(min_hz))
        & (base <= float(max_hz))
        & (post >= float(min_hz))
        & (post <= float(max_hz))
        & (base_conf > 0.0)
        & (post_conf > 0.0)
    )
    if not np.any(valid):
        return None, None, {"enabled": False, "reason": "no_valid_overlap"}

    if bool(enable_macro_guard):
        certainty = np.clip(
            0.20
            + 0.25 * post_conf
            + 0.20 * (1.0 - entropy)
            + 0.20 * top_gap
            + 0.05 * macro
            + 0.05 * direct_macro
            + 0.05 * h1_role,
            0.0,
            1.0,
        )
    else:
        certainty = np.clip(
            0.20
            + 0.25 * post_conf
            + 0.20 * (1.0 - entropy)
            + 0.20 * top_gap
            + 0.15 * macro,
            0.0,
            1.0,
        )
    alpha = 0.45 * certainty
    if bool(enable_macro_guard):
        p1d_half_rescue = (p1d_half >= 0.35) & (macro >= 0.66) & (motion_direct >= 0.35)
        alpha[p1d_half_rescue] = np.maximum(
            alpha[p1d_half_rescue],
            0.45 * p1d_half[p1d_half_rescue] * (0.70 + 0.30 * motion_direct[p1d_half_rescue]),
        )
    else:
        p1d_half_rescue = (p1d_half >= 0.35) & (macro >= 0.66)
        alpha[p1d_half_rescue] = np.maximum(alpha[p1d_half_rescue], 0.45 * p1d_half[p1d_half_rescue])

    ambiguous_alias = (entropy >= 0.94) & (top_gap <= 0.12) & (p1d_half < 0.25)
    alpha[ambiguous_alias] *= 0.20
    unresolved_p1d_alias = (p1d_half >= 0.35) & (motion_direct < 0.25)
    if bool(enable_macro_guard):
        alpha[unresolved_p1d_alias] *= 0.25
    if bool(enable_role_guard):
        alpha *= (1.0 - 0.40 * morphology_alias)
        alpha *= (1.0 - 0.35 * abstain_pressure)
    weak_macro = macro < 0.34
    alpha[weak_macro] *= 0.50
    weak_direct_macro = direct_macro < 0.34
    if bool(enable_macro_guard):
        alpha[weak_direct_macro] *= 0.65
        alpha *= (1.0 - 0.55 * alias_risk)
        specific_posterior_correction = (
            (post_conf >= 0.18)
            & (entropy <= 0.90)
            & (top_gap >= 0.20)
            & (support >= 0.24)
            & (direct_support >= 0.08)
            & (macro >= 0.60)
            & (direct_macro >= 0.45)
            & (alias_risk <= 0.55)
        )
        if bool(enable_role_guard):
            specific_posterior_correction &= (h1_role >= 0.35) & (abstain_pressure <= 0.65)
    else:
        specific_posterior_correction = (
            (post_conf >= 0.18)
            & (entropy <= 0.90)
            & (top_gap >= 0.20)
            & (support >= 0.24)
            & (direct_support >= 0.08)
            & (macro >= 0.60)
        )
    if bool(enable_specificity_boost):
        alpha[specific_posterior_correction] = np.maximum(
            alpha[specific_posterior_correction],
            0.32 * (0.70 + 0.30 * support[specific_posterior_correction]),
        )

    # Target-computable downshift guard.
    #
    # The posterior sometimes correctly lowers a high-rate readout, but the
    # MAHNOB32 audit showed that blind downshifts also create regressions.  A
    # downshift is therefore accepted only when the target evidence is internally
    # specific: low entropy, clear top hypothesis gap, non-trivial support, and
    # at least one independent macro-family timing witness.  This keeps the
    # source-supervised/readout-only boundary intact because no target labels are
    # used and the state update is unchanged.
    delta_hz = post - base
    large_downshift = (delta_hz <= -0.035) | (post <= 0.82 * np.maximum(base, 1e-6))
    if bool(enable_macro_guard):
        well_supported_downshift = (
            (post_conf >= 0.22)
            & (entropy <= 0.88)
            & (top_gap >= 0.18)
            & (support >= 0.25)
            & (direct_support >= 0.10)
            & (macro >= 0.50)
            & (direct_macro >= 0.40)
            & (alias_risk <= 0.60)
        )
        if bool(enable_role_guard):
            well_supported_downshift &= (h1_role >= 0.30) & (abstain_pressure <= 0.70)
    else:
        well_supported_downshift = (
            (post_conf >= 0.22)
            & (entropy <= 0.88)
            & (top_gap >= 0.18)
            & (support >= 0.25)
            & (direct_support >= 0.10)
            & (macro >= 0.50)
        )
    weak_downshift = large_downshift & ~well_supported_downshift
    alpha[weak_downshift] *= 0.25

    high_base_conflict = large_downshift & (base_conf >= 0.70) & (post_conf < 0.38)
    alpha[high_base_conflict] *= 0.50

    alpha = np.where(valid, np.clip(alpha, 0.0, 0.55), 0.0)
    out = base.copy()
    out[valid] = np.clip(
        base[valid] + alpha[valid] * (post[valid] - base[valid]),
        float(min_hz),
        float(max_hz),
    )
    confidence = np.where(valid, np.maximum(base_conf, alpha), 0.0)
    meta = {
        "enabled": True,
        "source": "candidate_rate_posterior_final",
        "posterior_output_override": True,
        "calibration": "bounded_correction_to_existing_readout",
        "abstention_guard": "target_computable_downshift_specificity",
        "coverage": float(np.mean(valid)),
        "alpha_mean": float(np.mean(alpha[valid])) if np.any(valid) else 0.0,
        "alpha_median": float(np.median(alpha[valid])) if np.any(valid) else 0.0,
        "alpha_active_fraction": float(np.mean(alpha > 1e-6)),
        "ambiguous_alias_fraction": float(np.mean(ambiguous_alias & valid)),
        "unresolved_p1d_alias_fraction": float(np.mean(unresolved_p1d_alias & valid)),
        "p1d_half_rescue_fraction": float(np.mean(p1d_half_rescue & valid)),
        "weak_direct_macro_fraction": float(np.mean(weak_direct_macro & valid)),
        "alias_risk_mean": float(np.mean(alias_risk[valid])) if np.any(valid) else 0.0,
        "h1_role_support_mean": float(np.mean(h1_role[valid])) if np.any(valid) else 0.0,
        "morphology_alias_pressure_mean": float(np.mean(morphology_alias[valid])) if np.any(valid) else 0.0,
        "abstain_pressure_mean": float(np.mean(abstain_pressure[valid])) if np.any(valid) else 0.0,
        "large_downshift_fraction": float(np.mean(large_downshift & valid)),
        "well_supported_downshift_fraction": float(np.mean(well_supported_downshift & large_downshift & valid)),
        "guarded_downshift_fraction": float(np.mean(weak_downshift & valid)),
        "high_base_conflict_fraction": float(np.mean(high_base_conflict & valid)),
        "specificity_boost_enabled": bool(enable_specificity_boost),
        "macro_guard_enabled": bool(enable_macro_guard),
        "role_guard_enabled": bool(enable_role_guard),
        "specific_posterior_correction_fraction": float(np.mean(specific_posterior_correction & valid)),
    }
    return out.tolist(), confidence.tolist(), meta


def _source_validity_guarded_readout(
    base_rate: Sequence[float],
    base_confidence: Sequence[float],
    source_rate: Sequence[float],
    source_confidence: Sequence[float],
    source_meta: Dict[str, object],
    *,
    min_hz: float,
    max_hz: float,
) -> Tuple[Optional[List[float]], Optional[List[float]], Dict[str, object]]:
    """Use source-validity as a specificity-gated correction, not a selector."""
    base = np.asarray(base_rate, dtype=np.float64).reshape(-1)
    base_conf = np.asarray(base_confidence, dtype=np.float64).reshape(-1)
    source = np.asarray(source_rate, dtype=np.float64).reshape(-1)
    source_conf = np.asarray(source_confidence, dtype=np.float64).reshape(-1)
    n = min(base.size, base_conf.size, source.size, source_conf.size)
    if n <= 0:
        return None, None, {"enabled": False, "reason": "empty_arrays"}
    base = base[:n]
    base_conf = np.clip(np.nan_to_num(base_conf[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    source = source[:n]
    source_conf = np.clip(np.nan_to_num(source_conf[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    valid = (
        np.isfinite(base)
        & np.isfinite(source)
        & (base >= float(min_hz))
        & (base <= float(max_hz))
        & (source >= float(min_hz))
        & (source <= float(max_hz))
        & (base_conf > 0.0)
        & (source_conf > 0.0)
    )
    if not np.any(valid):
        return None, None, {"enabled": False, "reason": "no_valid_overlap"}

    entropy_median = _finite_row_float(source_meta, "source_validity_entropy_median", 1.0)
    top_gap_median = _finite_row_float(source_meta, "source_validity_top_gap_median", 0.0)
    confidence_mean = _finite_row_float(source_meta, "confidence_mean", 0.0)
    source_specificity = np.clip((source_conf - 0.24) / 0.30, 0.0, 1.0)
    global_specificity = float(np.clip(
        (0.35 + 0.65 * max(0.0, 1.0 - float(entropy_median)))
        * (0.45 + 0.55 * float(top_gap_median))
        * (0.50 + 0.50 * float(confidence_mean)),
        0.0,
        1.0,
    ))
    conflict = np.abs(source - base)
    conflict_ok = np.exp(-0.5 * (conflict / 0.08) ** 2)
    alpha = 0.32 * source_specificity * global_specificity * (0.35 + 0.65 * conflict_ok)

    # Large disagreements are allowed only if the source posterior itself is
    # strong.  Otherwise source-validity is treated as abstention evidence.
    unsafe_conflict = (conflict > 0.08) & (source_conf < 0.45)
    alpha[unsafe_conflict] *= 0.20
    alpha = np.where(valid, np.clip(alpha, 0.0, 0.35), 0.0)

    out = base.copy()
    out[valid] = np.clip(
        base[valid] + alpha[valid] * (source[valid] - base[valid]),
        float(min_hz),
        float(max_hz),
    )
    confidence = np.where(valid, np.maximum(base_conf, alpha), 0.0)
    meta = {
        "enabled": True,
        "source": "source_validity_guarded",
        "calibration": "preserve_final_source_validity_bounded_correction",
        "coverage": float(np.mean(valid)),
        "source_validity_guard_alpha_mean": float(np.mean(alpha[valid])) if np.any(valid) else 0.0,
        "source_validity_guard_alpha_median": float(np.median(alpha[valid])) if np.any(valid) else 0.0,
        "source_validity_guard_active_fraction": float(np.mean(alpha > 1e-6)),
        "source_validity_guard_unsafe_conflict_fraction": float(np.mean(unsafe_conflict & valid)),
        "source_validity_global_specificity": global_specificity,
    }
    return out.tolist(), confidence.tolist(), meta


def _estimate_lookup(payload: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    estimates = payload.get("estimates", [])
    if not isinstance(estimates, list):
        return out
    for row in estimates:
        if not isinstance(row, dict):
            continue
        method = row.get("method")
        if method is None:
            continue
        out[str(method)] = row.get("estimate", row)
    return out


def _extract_signal(est: object, source_key: str) -> Optional[np.ndarray]:
    if isinstance(est, dict):
        value = est.get(source_key)
        if value is None and source_key != "signal_hat":
            value = est.get("signal_hat")
    else:
        value = est
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    return arr


def _finite_fill(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.zeros_like(arr)
    if np.all(mask):
        return arr
    idx = np.arange(arr.size)
    arr[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])
    return arr


def _fit_length(x: np.ndarray, n: int, *, fill_value: float = np.nan) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(max(n, 0))
    if n == 0:
        return np.array([], dtype=np.float64)
    if arr.size == n:
        return arr.copy()
    if arr.size == 0:
        return np.full(n, fill_value, dtype=np.float64)
    if arr.size == 1:
        return np.full(n, float(arr[0]) if np.isfinite(arr[0]) else fill_value, dtype=np.float64)
    src_x = np.linspace(0.0, 1.0, arr.size)
    dst_x = np.linspace(0.0, 1.0, n)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.full(n, fill_value, dtype=np.float64)
    if np.count_nonzero(finite) == 1:
        return np.full(n, float(arr[finite][0]), dtype=np.float64)
    return np.interp(dst_x, src_x[finite], arr[finite]).astype(np.float64)


def _robust_z(x: np.ndarray) -> np.ndarray:
    arr = _finite_fill(x)
    if arr.size == 0:
        return arr
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.std(arr))
    if not np.isfinite(scale) or scale < 1e-9:
        return np.zeros_like(arr)
    out = (arr - med) / scale
    return np.clip(out, -8.0, 8.0)


def _bandpass_z(x: np.ndarray, fs: float, min_hz: float, max_hz: float) -> np.ndarray:
    arr = _finite_fill(x)
    if arr.size < 16 or fs <= 0:
        return _robust_z(arr)
    nyq = fs * 0.5
    lo = max(float(min_hz), 1e-4)
    hi = min(float(max_hz), nyq * 0.90)
    if not (0.0 < lo < hi < nyq):
        return _robust_z(sps.detrend(arr, type="linear"))
    try:
        sos = sps.butter(2, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
        padlen = min(max(9, 3 * (2 * sos.shape[0] + 1)), max(arr.size - 1, 1))
        filtered = sps.sosfiltfilt(sos, arr, padlen=padlen)
    except Exception:
        filtered = sps.detrend(arr, type="linear")
    return _robust_z(filtered)


def _overlap(a: np.ndarray, b: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        if lag >= min(a.size, b.size):
            return np.array([]), np.array([])
        return a[:-lag], b[lag:]
    if lag < 0:
        k = -lag
        if k >= min(a.size, b.size):
            return np.array([]), np.array([])
        return a[k:], b[:-k]
    return a, b


def _shift_with_edge(x: np.ndarray, lag: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0 or lag == 0:
        return arr.copy()
    out = np.empty_like(arr)
    if lag > 0:
        if lag >= arr.size:
            out.fill(arr[-1])
        else:
            out[:-lag] = arr[lag:]
            out[-lag:] = arr[-1]
    else:
        k = -lag
        if k >= arr.size:
            out.fill(arr[0])
        else:
            out[k:] = arr[:-k]
            out[:k] = arr[0]
    return out


def _fit_to_target(source_z: np.ndarray, target_z: np.ndarray, lag: int) -> Dict[str, float]:
    target, source = _overlap(_finite_fill(target_z), _finite_fill(source_z), lag)
    if target.size < 8 or source.size < 8:
        return {
            "lag_samples": int(lag),
            "corr": 0.0,
            "scale": 0.0,
            "offset": 0.0,
            "residual_mad": float("inf"),
            "support": 0.0,
        }
    denom = float(np.dot(source, source))
    scale = float(np.dot(source, target) / denom) if denom > 1e-12 else 0.0
    if not np.isfinite(scale):
        scale = 0.0
    scale = float(np.clip(scale, -3.0, 3.0))
    offset = float(np.median(target - scale * source))
    if not np.isfinite(offset):
        offset = 0.0
    pred = scale * source + offset
    corr = float(np.corrcoef(target, source)[0, 1]) if np.std(source) > 1e-9 and np.std(target) > 1e-9 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    residual = target - pred
    residual_mad = float(np.median(np.abs(residual - np.median(residual))))
    if not np.isfinite(residual_mad):
        residual_mad = float("inf")
    support = abs(corr) * math.exp(-0.35 * residual_mad)
    if not np.isfinite(support):
        support = 0.0
    return {
        "lag_samples": int(lag),
        "corr": corr,
        "scale": scale,
        "offset": offset,
        "residual_mad": residual_mad,
        "support": float(np.clip(support, 0.0, 1.0)),
    }


def _relative_equation(source_z: np.ndarray, target_z: np.ndarray, fs: float, lag_max_sec: float) -> Dict[str, float]:
    n = min(source_z.size, target_z.size)
    source = _finite_fill(np.asarray(source_z[:n], dtype=np.float64))
    target = _finite_fill(np.asarray(target_z[:n], dtype=np.float64))
    if n < 16:
        return _fit_to_target(source, target, 0)
    max_lag = int(max(0, round(float(lag_max_sec) * float(fs))))
    max_lag = min(max_lag, max(0, n // 3))
    corr = sps.correlate(source, target, mode="full", method="auto")
    lags = sps.correlation_lags(source.size, target.size, mode="full")
    keep = np.abs(lags) <= max_lag
    if not np.any(keep):
        best_lag = 0
    else:
        scores = np.nan_to_num(np.abs(corr[keep]), nan=0.0, posinf=0.0, neginf=0.0)
        best_lag = int(lags[keep][int(np.argmax(scores))])
    return _fit_to_target(source, target, best_lag)


def _anchor_score(anchor_idx: int, names: Sequence[str], z_channels: Sequence[np.ndarray], fs: float, lag_max_sec: float) -> Dict[str, object]:
    anchor_name = names[anchor_idx]
    anchor_family = FAMILY_GROUPS.get(anchor_name, anchor_name)
    family_scores: Dict[str, List[float]] = {}
    pair_rows: List[Dict[str, object]] = []
    for i, source_name in enumerate(names):
        if i == anchor_idx:
            continue
        source_family = FAMILY_GROUPS.get(source_name, source_name)
        if source_family == anchor_family:
            continue
        eq = _relative_equation(z_channels[i], z_channels[anchor_idx], fs, lag_max_sec)
        family_scores.setdefault(source_family, []).append(float(eq["support"]))
        pair_rows.append(
            {
                "source": source_name,
                "anchor": anchor_name,
                "source_family": source_family,
                "anchor_family": anchor_family,
                **eq,
            }
        )
    means = [float(np.mean(vals)) for vals in family_scores.values() if vals]
    if means:
        score = 0.70 * float(np.mean(means)) + 0.30 * float(np.min(means))
    else:
        score = 0.0
    return {
        "anchor": anchor_name,
        "anchor_family": anchor_family,
        "score": score,
        "n_support_families": len(means),
        "family_scores": {k: float(np.mean(v)) for k, v in family_scores.items() if v},
        "pair_rows": pair_rows,
    }


def _calibrate_stack(
    channels: Sequence[np.ndarray],
    names: Sequence[str],
    fs: float,
    min_hz: float,
    max_hz: float,
    lag_max_sec: float,
    min_support_corr: float,
    anchor_policy: str,
    canonical_policy: str,
    reliability_group_prior: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], List[Dict[str, object]]]:
    n = min((np.asarray(c).size for c in channels), default=0)
    if n <= 0:
        raise ValueError("empty channel stack")
    raw = [np.asarray(c, dtype=np.float64).reshape(-1)[:n] for c in channels]
    z_channels = [_bandpass_z(c, fs, min_hz, max_hz) for c in raw]

    anchor_allowed = set(names)
    if str(anchor_policy).strip().lower() == "displacement":
        anchor_allowed = {name for name in names if name in DISPLACEMENT_ANCHOR_FAMILIES}
        if not anchor_allowed:
            anchor_allowed = set(names)
    anchor_candidates = [
        _anchor_score(idx, names, z_channels, fs, lag_max_sec)
        for idx in range(len(names))
        if names[idx] in anchor_allowed
    ]
    anchor_candidates.sort(
        key=lambda row: (float(row["score"]), int(row["n_support_families"])),
        reverse=True,
    )
    anchor = anchor_candidates[0] if anchor_candidates else {"anchor": names[0], "score": 0.0}
    anchor_name = str(anchor["anchor"])
    anchor_idx = list(names).index(anchor_name)
    target = z_channels[anchor_idx]

    calibrated: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    for idx, name in enumerate(names):
        if idx == anchor_idx:
            eq = {
                "lag_samples": 0,
                "corr": 1.0,
                "scale": 1.0,
                "offset": 0.0,
                "residual_mad": 0.0,
                "support": float(max(anchor.get("score", 0.0), 0.0)),
            }
            transformed = target.copy()
        else:
            eq = _relative_equation(z_channels[idx], target, fs, lag_max_sec)
            shifted = _shift_with_edge(z_channels[idx], int(eq["lag_samples"]))
            transformed = float(eq["scale"]) * shifted + float(eq["offset"])
            if abs(float(eq["corr"])) < float(min_support_corr):
                transformed *= 0.5
        transformed = np.nan_to_num(_robust_z(transformed), nan=0.0, posinf=0.0, neginf=0.0)
        calibrated.append(transformed[:n])
        rows.append(
            {
                "channel": name,
                "channel_family": FAMILY_GROUPS.get(name, name),
                "anchor": anchor_name,
                "anchor_family": FAMILY_GROUPS.get(anchor_name, anchor_name),
                **eq,
            }
        )

    supports = np.asarray([max(float(row.get("support", 0.0)), 0.0) for row in rows], dtype=np.float64)
    canonical_allowed = np.ones_like(supports)
    if str(canonical_policy).strip().lower() == "displacement":
        canonical_allowed = np.asarray(
            [1.0 if str(row["channel"]) in DISPLACEMENT_ANCHOR_FAMILIES else 0.0 for row in rows],
            dtype=np.float64,
        )
        if float(np.sum(canonical_allowed)) <= 0.0:
            canonical_allowed = np.ones_like(supports)
    supports = supports * canonical_allowed
    row_families = [str(row["channel_family"]) for row in rows]
    unique_families = sorted(set(row_families))
    anchor_family_scores = dict(anchor.get("family_scores", {}) or {})
    family_raw_scores: Dict[str, float] = {}
    for family in unique_families:
        idxs = [i for i, fam in enumerate(row_families) if fam == family]
        vals = supports[idxs]
        if family == FAMILY_GROUPS.get(anchor_name, anchor_name):
            score = float(max(anchor.get("score", 0.0), 0.0))
        else:
            score = float(anchor_family_scores.get(family, np.nan))
            if not np.isfinite(score):
                score = float(np.mean(vals)) if vals.size else 0.0
        family_raw_scores[family] = max(score, 0.0)
    family_score_vec = np.asarray([family_raw_scores[fam] for fam in unique_families], dtype=np.float64)
    reliability_group_prior = dict(reliability_group_prior or {})
    reliability_prior_vec = np.ones_like(family_score_vec, dtype=np.float64)
    if reliability_group_prior:
        for idx, family in enumerate(unique_families):
            val = float(reliability_group_prior.get(family, np.nan))
            reliability_prior_vec[idx] = float(val) if np.isfinite(val) and val > 0.0 else 1e-6
        prior_sum = float(np.sum(reliability_prior_vec))
        if prior_sum > 1e-12:
            reliability_prior_vec = reliability_prior_vec / prior_sum
        else:
            reliability_prior_vec = np.ones_like(reliability_prior_vec) / max(1, reliability_prior_vec.size)
    family_weight_vec = family_score_vec ** 2
    if reliability_group_prior:
        family_weight_vec = family_weight_vec * reliability_prior_vec
    if float(np.sum(family_weight_vec)) <= 1e-12:
        family_weight_vec = np.ones_like(family_weight_vec) / max(1, family_weight_vec.size)
    else:
        family_weight_vec = family_weight_vec / float(np.sum(family_weight_vec))
    family_weights = {fam: float(w) for fam, w in zip(unique_families, family_weight_vec)}

    weights = np.zeros_like(supports)
    for family in unique_families:
        idxs = np.asarray([i for i, fam in enumerate(row_families) if fam == family], dtype=int)
        local = supports[idxs] ** 2
        if float(np.sum(local)) <= 1e-12:
            local = np.ones_like(local) / max(1, local.size)
        else:
            local = local / float(np.sum(local))
        weights[idxs] = family_weights[family] * local
    canonical = np.sum(np.vstack(calibrated) * weights[:, None], axis=0)
    canonical = np.nan_to_num(_robust_z(canonical), nan=0.0, posinf=0.0, neginf=0.0)
    for row, weight in zip(rows, weights):
        row["canonical_weight"] = float(weight)
        row["target_reliability_group"] = str(row["channel_family"])
        row["target_reliability_prior"] = float(reliability_group_prior.get(str(row["channel_family"]), np.nan))

    meta = {
        "anchor": anchor_name,
        "anchor_family": FAMILY_GROUPS.get(anchor_name, anchor_name),
        "anchor_score": float(anchor.get("score", 0.0)),
        "anchor_n_support_families": int(anchor.get("n_support_families", 0)),
        "anchor_family_scores": anchor.get("family_scores", {}),
        "family_weights": family_weights,
        "family_raw_scores": family_raw_scores,
        "target_reliability_graph_enabled": bool(reliability_group_prior),
        "target_reliability_group_prior": {
            str(k): float(v) for k, v in reliability_group_prior.items()
            if np.isfinite(float(v))
        },
        "anchor_policy": str(anchor_policy),
        "canonical_policy": str(canonical_policy),
        "anchor_candidates": [
            {
                "anchor": row["anchor"],
                "anchor_family": row["anchor_family"],
                "score": float(row["score"]),
                "n_support_families": int(row["n_support_families"]),
                "family_scores": row["family_scores"],
            }
            for row in anchor_candidates
        ],
        "canonical_weights": {
            str(row["channel"]): float(row["canonical_weight"])
            for row in rows
        },
    }
    return np.vstack(calibrated), canonical, meta, rows


def _copy_payload_skeleton(payload: dict, estimate: dict, method_name: str) -> dict:
    keep_keys = [
        "video_path",
        "fps",
        "fs",
        "gt",
        "fs_gt",
        "respiration",
        "gt_alignment_meta",
        "alignment_meta",
    ]
    out = {k: payload[k] for k in keep_keys if k in payload}
    out["estimates"] = [{"method": method_name, "estimate": estimate}]
    return out


def _is_strict_release_cfg(cfg: Dict[str, object]) -> bool:
    return bool(cfg.get("enable_observation_law", False)) and bool(cfg.get("enable_rate_posterior", False)) and (
        str(cfg.get("rate_posterior_output_public_source", cfg.get("rate_posterior_output_source", "off"))).strip().lower()
        == "final"
    )


def _require_locked_activation(
    *,
    video: str,
    cfg: Dict[str, object],
    state_window_rows: Sequence[Dict[str, float]],
    readout_window_rows: Sequence[Dict[str, float]],
    rate_posterior_meta: Dict[str, object],
    output_rate_meta: Dict[str, object],
    target_observability_runtime: Optional[Dict[str, object]],
    target_observability_meta: Dict[str, object],
) -> None:
    """Fail locked runs when required adaptive evidence is inactive.

    The final path must not silently degrade into a trial-level prior or a
    plain multichannel filter.  Diagnostic/ablation runs can still disable
    these modules by not requesting the locked contract.
    """
    if not _is_strict_release_cfg(cfg):
        return
    missing: List[str] = []
    if str(cfg.get("parh_input", "")) != "multichannel":
        missing.append("parh_input=multichannel")
    scope = str(cfg.get("reliability_prior_scope", "all")).strip().lower()
    if bool(cfg.get("separate_reliability_csvs", False)) and not state_window_rows:
        missing.append("windowed state-role reliability rows from --state-reliability-group-csv")
    elif scope != "readout_only" and not state_window_rows:
        missing.append("windowed state-role reliability rows")
    if not readout_window_rows:
        missing.append("windowed readout reliability rows")
    if not bool(rate_posterior_meta.get("enabled", False)):
        missing.append(f"candidate rate posterior inactive ({rate_posterior_meta.get('reason', 'unknown')})")
    if not bool(output_rate_meta.get("enabled", False)):
        reason = str(output_rate_meta.get("reason", "unknown"))
        # In true full-dataset runs, a small number of hard-observability
        # trials can have no GT-free readout windows.  That is a valid failure
        # mode to report, not a reason to silently drop the trial from the
        # full artifact package.  All other inactive-readout reasons still
        # violate the locked contract.
        if reason == "no_supported_readout_windows":
            output_rate_meta["release_hard_abstention"] = True
            output_rate_meta.setdefault("abstention_guard", reason)
            output_rate_meta.setdefault("source", "abstained_no_supported_readout_windows")
            output_rate_meta.setdefault("coverage", 0.0)
        else:
            missing.append(f"decoupled final rate readout inactive ({reason})")
    if not bool(cfg.get("enable_target_observability_control", False)):
        missing.append("--enable-target-observability-control")
    if target_observability_runtime is None:
        missing.append(f"target observability runtime inactive ({target_observability_meta.get('reason', 'unknown')})")
    if missing:
        raise RuntimeError(
            f"locked activation contract failed for {video}: "
            + "; ".join(str(x) for x in missing)
        )


def _process_file(args: Tuple[str, str, Dict[str, object]]) -> Tuple[str, List[Dict[str, object]], List[Dict[str, object]], Optional[str]]:
    path_str, out_data_str, cfg = args
    path = Path(path_str)
    out_data = Path(out_data_str)
    try:
        method_name = str(cfg["name"])
        source_key = str(cfg["source_key"])
        payload = _load_pkl(path)
        fps = float(payload.get("fps") or payload.get("fs") or 0.0)
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        lookup = _estimate_lookup(payload)

        channels: List[np.ndarray] = []
        rate_tracks: List[np.ndarray] = []
        names: List[str] = []
        missing: List[str] = []
        rate_track_source = str(cfg.get("rate_track_source", "base"))
        for method_label, family_name, _ in DEFAULT_METHODS:
            est_row = lookup.get(method_label)
            signal = _extract_signal(est_row, source_key)
            if signal is None:
                missing.append(method_label)
                continue
            channels.append(signal)
            rate_label = _rate_variant_label(method_label, family_name, rate_track_source)
            rate_row = lookup.get(rate_label)
            rate_track = _extract_signal(rate_row, "track_hz")
            if rate_track is None:
                rate_track = _extract_signal(est_row, "track_hz")
            if rate_track is None:
                rate_track = np.full(signal.size, np.nan, dtype=np.float64)
            rate_tracks.append(_fit_length(rate_track, signal.size))
            names.append(family_name)
        if len(channels) < 3:
            return path.name, [], [], f"not enough channels ({len(channels)}), missing={missing}"

        reliability_by_video = cfg.get("reliability_priors_by_video", {})
        if not isinstance(reliability_by_video, dict):
            reliability_by_video = {}
        reliability_group_prior = reliability_by_video.get(path.stem, {})
        if not isinstance(reliability_group_prior, dict):
            reliability_group_prior = {}
        windowed_by_video = cfg.get("windowed_reliability_priors_by_video", {})
        if not isinstance(windowed_by_video, dict):
            windowed_by_video = {}
        if bool(cfg.get("separate_reliability_csvs", False)):
            state_by_video = cfg.get("state_reliability_priors_by_video", {})
            readout_by_video = cfg.get("readout_reliability_priors_by_video", {})
            state_windowed_by_video = cfg.get("state_windowed_reliability_priors_by_video", {})
            readout_windowed_by_video = cfg.get("readout_windowed_reliability_priors_by_video", {})
            if not isinstance(state_by_video, dict):
                state_by_video = {}
            if not isinstance(readout_by_video, dict):
                readout_by_video = {}
            if not isinstance(state_windowed_by_video, dict):
                state_windowed_by_video = {}
            if not isinstance(readout_windowed_by_video, dict):
                readout_windowed_by_video = {}
            state_reliability_group_prior = dict(state_by_video.get(path.stem, {}) or {})
            state_window_rows = [dict(row) for row in state_windowed_by_video.get(path.stem, [])]
            readout_reliability_group_prior = dict(readout_by_video.get(path.stem, {}) or {})
            readout_window_rows = [dict(row) for row in readout_windowed_by_video.get(path.stem, [])]
            scope = str(cfg.get("reliability_prior_scope", "all")).strip().lower()
            if scope == "readout_only":
                state_window_rows = [
                    {key: value for key, value in row.items() if not str(key).startswith("arbiter_")}
                    for row in state_window_rows
                ]
            elif scope == "state_only":
                readout_reliability_group_prior = {}
                readout_window_rows = []
        else:
            (
                state_reliability_group_prior,
                state_window_rows,
                readout_reliability_group_prior,
                readout_window_rows,
            ) = _split_reliability_scope(
                str(cfg.get("reliability_prior_scope", "all")),
                reliability_group_prior,
                windowed_by_video.get(path.stem, []),
            )
        channel_reliability_priors = _channel_reliability_priors(names, state_reliability_group_prior)

        stack, canonical, cal_meta, cal_rows = _calibrate_stack(
            channels,
            names,
            fps,
            float(cfg["min_hz"]),
            float(cfg["max_hz"]),
            float(cfg["lag_max_sec"]),
            float(cfg["min_support_corr"]),
            str(cfg.get("anchor_policy", "displacement")),
            str(cfg.get("canonical_policy", "displacement")),
            reliability_group_prior=state_reliability_group_prior,
        )
        stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
        canonical = np.nan_to_num(canonical, nan=0.0, posinf=0.0, neginf=0.0)
        channel_context_runtime = _channel_context_prior_runtime(
            names,
            state_window_rows,
            stack.shape[1],
            fps,
            channel_reliability_priors,
        )
        state_role_runtime = {}
        if str(cfg.get("reliability_prior_scope", "all")).strip().lower() != "readout_only":
            state_role_runtime = _state_role_prior_runtime(
                names,
                state_window_rows,
                stack.shape[1],
                fps,
            )
        rate_anchor_runtime, rate_anchor_confidence_runtime, rate_anchor_meta = _windowed_rate_anchor_runtime(
            names,
            rate_tracks,
            state_window_rows,
            stack.shape[1],
            fps,
            state_reliability_group_prior,
            float(cfg["min_hz"]),
            float(cfg["max_hz"]),
            signals=stack,
            enable_phase_validation=bool(cfg.get("enable_phase_anchor_validation", False)),
            enable_rate_hypothesis_graph_v4=bool(cfg.get("enable_rate_hypothesis_graph", False)),
        )
        output_rate_runtime = None
        output_rate_confidence_runtime = None
        output_rate_meta: Dict[str, object] = {"enabled": False}
        if bool(cfg.get("enable_decoupled_rate_readout", False)):
            output_rate_runtime, output_rate_confidence_runtime, output_rate_meta = _decoupled_rate_readout_runtime(
                names,
                stack,
                rate_tracks,
                readout_window_rows,
                stack.shape[1],
                fps,
                readout_reliability_group_prior,
                float(cfg["min_hz"]),
                float(cfg["max_hz"]),
                enable_rate_hypothesis_graph_v4=bool(cfg.get("enable_rate_hypothesis_graph", False)),
                enable_derived_consistency_scaling=bool(cfg.get("enable_derived_consistency_scaling", False)),
            )
        rate_posterior_runtime = None
        rate_posterior_meta: Dict[str, object] = {"enabled": False}
        source_validity_runtime = None
        source_validity_confidence_runtime = None
        source_validity_meta: Dict[str, object] = {"enabled": False}
        if bool(cfg.get("enable_rate_posterior", False)):
            rate_posterior_runtime, rate_posterior_meta = _candidate_rate_posterior_runtime(
                names,
                stack,
                rate_tracks,
                readout_window_rows,
                stack.shape[1],
                fps,
                readout_reliability_group_prior,
                float(cfg["min_hz"]),
                float(cfg["max_hz"]),
                enable_derived_consistency_scaling=bool(cfg.get("enable_derived_consistency_scaling", False)),
                enable_signal_sqi_observability=bool(cfg.get("enable_signal_sqi_observability", False)),
            )
            posterior_output_source = str(cfg.get("rate_posterior_output_source", "off")).strip().lower()
            if (
                rate_posterior_runtime is not None
                and posterior_output_source in {
                    "calibrated_mean",
                    "specific_calibrated_mean",
                    "macroguard_specific_calibrated_mean",
                    "roleaware_macroguard_specific_calibrated_mean",
                    "source_validity_guarded",
                    "source_arbiter_v1",
                    "source_arbiter_v2",
                    "source_arbiter_v3",
                }
                and output_rate_runtime is not None
                and output_rate_confidence_runtime is not None
            ):
                (
                    calibrated_rate_runtime,
                    calibrated_confidence_runtime,
                    calibrated_meta,
                ) = _calibrated_posterior_mean_readout(
                    output_rate_runtime,
                    output_rate_confidence_runtime,
                    rate_posterior_runtime,
                    min_hz=float(cfg["min_hz"]),
                    max_hz=float(cfg["max_hz"]),
                    enable_specificity_boost=posterior_output_source
                    in {
                        "specific_calibrated_mean",
                        "macroguard_specific_calibrated_mean",
                        "roleaware_macroguard_specific_calibrated_mean",
                    },
                    enable_macro_guard=posterior_output_source
                    in {"macroguard_specific_calibrated_mean", "roleaware_macroguard_specific_calibrated_mean"},
                    enable_role_guard=posterior_output_source == "roleaware_macroguard_specific_calibrated_mean",
                )
                if calibrated_rate_runtime is not None and calibrated_confidence_runtime is not None:
                    output_rate_runtime = calibrated_rate_runtime
                    output_rate_confidence_runtime = calibrated_confidence_runtime
                    output_rate_meta = {
                        **dict(output_rate_meta or {}),
                        **dict(calibrated_meta or {}),
                        "posterior_meta": rate_posterior_meta,
                    }
                    if posterior_output_source in {"source_arbiter_v1", "source_arbiter_v2", "source_arbiter_v3"}:
                        output_rate_meta["source"] = (
                            f"candidate_rate_posterior_calibrated_mean_plus_{posterior_output_source}"
                        )
                        output_rate_meta[f"{posterior_output_source}_requested"] = True
                if (
                    posterior_output_source == "source_validity_guarded"
                    and output_rate_runtime is not None
                    and output_rate_confidence_runtime is not None
                ):
                    (
                        source_validity_runtime,
                        source_validity_confidence_runtime,
                        source_validity_meta,
                    ) = _source_validity_rate_readout_runtime(
                        names,
                        stack,
                        rate_tracks,
                        readout_window_rows,
                        stack.shape[1],
                        fps,
                        readout_reliability_group_prior,
                        float(cfg["min_hz"]),
                        float(cfg["max_hz"]),
                    )
                    if source_validity_runtime is not None and source_validity_confidence_runtime is not None:
                        guarded_rate_runtime, guarded_confidence_runtime, guarded_meta = _source_validity_guarded_readout(
                            output_rate_runtime,
                            output_rate_confidence_runtime,
                            source_validity_runtime,
                            source_validity_confidence_runtime,
                            source_validity_meta,
                            min_hz=float(cfg["min_hz"]),
                            max_hz=float(cfg["max_hz"]),
                        )
                        if guarded_rate_runtime is not None and guarded_confidence_runtime is not None:
                            output_rate_runtime = guarded_rate_runtime
                            output_rate_confidence_runtime = guarded_confidence_runtime
                            output_rate_meta = {
                                **dict(output_rate_meta or {}),
                                **dict(guarded_meta or {}),
                                **dict(source_validity_meta or {}),
                                "source": "source_validity_guarded",
                                "source_validity_guarded_output": True,
                                "posterior_meta": rate_posterior_meta,
                            }
            elif rate_posterior_runtime is not None and posterior_output_source in {"mode", "mean"}:
                output_key = "mode_hz" if posterior_output_source == "mode" else "mean_hz"
                posterior_rate = rate_posterior_runtime.get(output_key)
                posterior_conf = rate_posterior_runtime.get("confidence")
                if posterior_rate is not None and posterior_conf is not None:
                    output_rate_runtime = posterior_rate
                    output_rate_confidence_runtime = posterior_conf
                    output_rate_meta = {
                        "enabled": True,
                        "source": f"candidate_rate_posterior_{posterior_output_source}",
                        "posterior_output_override": True,
                        "posterior_meta": rate_posterior_meta,
                    }
            elif posterior_output_source == "source_validity":
                (
                    source_validity_runtime,
                    source_validity_confidence_runtime,
                    source_validity_meta,
                ) = _source_validity_rate_readout_runtime(
                    names,
                    stack,
                    rate_tracks,
                    readout_window_rows,
                    stack.shape[1],
                    fps,
                    readout_reliability_group_prior,
                    float(cfg["min_hz"]),
                    float(cfg["max_hz"]),
                )
                if source_validity_runtime is not None and source_validity_confidence_runtime is not None:
                    output_rate_runtime = source_validity_runtime
                    output_rate_confidence_runtime = source_validity_confidence_runtime
                    output_rate_meta = {
                        "enabled": True,
                        "source": "source_validity_posterior",
                        "source_validity_output_override": True,
                        **dict(source_validity_meta or {}),
                        "posterior_meta": rate_posterior_meta,
                    }
        regime_law_meta: Dict[str, object] = {"enabled": False}
        if bool(cfg.get("enable_regime_observation_law", False)):
            (
                regime_context_runtime,
                regime_rate_anchor_runtime,
                regime_rate_anchor_confidence_runtime,
                regime_law_meta,
            ) = _regime_observation_law_runtime(
                names,
                stack,
                rate_tracks,
                state_window_rows,
                stack.shape[1],
                fps,
                state_reliability_group_prior,
                float(cfg["min_hz"]),
                float(cfg["max_hz"]),
            )
            if regime_context_runtime is not None:
                channel_context_runtime = regime_context_runtime
            if (
                str(cfg.get("regime_anchor_policy", "context_only")) == "replace"
                and regime_rate_anchor_runtime is not None
                and regime_rate_anchor_confidence_runtime is not None
            ):
                rate_anchor_runtime = regime_rate_anchor_runtime
                rate_anchor_confidence_runtime = regime_rate_anchor_confidence_runtime
                rate_anchor_meta = {
                    "source": "regime_observation_law",
                    **dict(rate_anchor_meta or {}),
                    "regime_observation_law": regime_law_meta,
                }
        target_observability_runtime = None
        target_observability_meta: Dict[str, object] = {"enabled": False}
        if bool(cfg.get("enable_target_observability_control", False)):
            target_observability_runtime, target_observability_meta = _target_observability_control_runtime(
                names,
                rate_tracks,
                output_rate_runtime,
                output_rate_confidence_runtime,
                rate_posterior_runtime,
                stack.shape[1],
                min_hz=float(cfg["min_hz"]),
                max_hz=float(cfg["max_hz"]),
                signals=stack,
                window_rows=readout_window_rows,
                fps=fps,
                enable_signal_sqi_observability=bool(cfg.get("enable_signal_sqi_observability", False)),
            )
        _require_locked_activation(
            video=path.stem,
            cfg=cfg,
            state_window_rows=state_window_rows,
            readout_window_rows=readout_window_rows,
            rate_posterior_meta=rate_posterior_meta,
            output_rate_meta=output_rate_meta,
            target_observability_runtime=target_observability_runtime,
            target_observability_meta=target_observability_meta,
        )
        head = oscillator_PARH_OSSM()
        if bool(cfg.get("enable_observation_law", False)):
            # The final observation-law path is structural, not an
            # environment-dependent experiment switch. Force the required
            # PARH modules on after construction so locked runtime
            # profiles cannot silently disable the multichannel law.
            head.ENABLE_DYNAMIC_MIXTURE = True
            head.ENABLE_RATE_OBSERVABILITY_MIXTURE = True
            head.ENABLE_RATE_OBSERVABILITY_HELPER = True
            head.ENABLE_RESIDUAL_SEMANTICS = True
            head.ENABLE_RESIDUAL_IDENTIFIABILITY_GUARD = True
            head.ENABLE_PHASE_ANCHORED_MORPHOLOGY = True
            head.ENABLE_GROUP_BALANCED_FUSION = True
        run_meta = {
            "method_name": method_name,
            "base_method": str(cfg.get("parh_base_method", "of_disp_bridge")),
            "target_calibrated_multifamily": True,
            "observation_law_enabled": bool(cfg.get("enable_observation_law", False)),
            "observation_law_design_lock": "PARH-OSSM release configuration",
            "calibration_space": "bandpass_z",
            "source_key": source_key,
            "calibration": cal_meta,
            "calibrated_source_families": names,
            "rate_track_source": rate_track_source,
            "reliability_prior_scope": str(cfg.get("reliability_prior_scope", "all")),
            "input_file": path.name,
            "target_reliability_channel_prior": channel_reliability_priors,
            "target_reliability_windowed_prior": bool(channel_context_runtime is not None),
            "target_state_role_prior": bool(state_role_runtime),
            "separate_reliability_csvs": bool(cfg.get("separate_reliability_csvs", False)),
            "reliability_group_csv": str(cfg.get("reliability_group_csv", "")),
            "state_reliability_group_csv": str(cfg.get("state_reliability_group_csv", "")),
            "readout_reliability_group_csv": str(cfg.get("readout_reliability_group_csv", "")),
            "target_reliability_rate_anchor": bool(rate_anchor_runtime is not None),
            "target_reliability_rate_anchor_meta": rate_anchor_meta,
            "regime_observation_law_enabled": bool(cfg.get("enable_regime_observation_law", False)),
            "regime_anchor_policy": str(cfg.get("regime_anchor_policy", "context_only")),
            "regime_observation_law_meta": regime_law_meta,
            "decoupled_rate_readout_enabled": bool(cfg.get("enable_decoupled_rate_readout", False)),
            "rate_candidate_posterior_enabled": bool(cfg.get("enable_rate_posterior", False)),
            "rate_posterior_output_source": str(
                cfg.get("rate_posterior_output_public_source", cfg.get("rate_posterior_output_source", "off"))
            ),
            "rate_hypothesis_graph_enabled": bool(cfg.get("enable_rate_hypothesis_graph", False)),
            "derived_consistency_scaling_enabled": bool(cfg.get("enable_derived_consistency_scaling", False)),
            "signal_sqi_observability_enabled": bool(cfg.get("enable_signal_sqi_observability", False)),
            "decoupled_rate_readout_meta": output_rate_meta,
            "rate_candidate_posterior_meta": rate_posterior_meta,
            "source_validity_meta": source_validity_meta,
            "target_observability_control_enabled": bool(target_observability_runtime is not None),
            "target_observability_control_meta": target_observability_meta,
        }
        if rate_anchor_runtime is not None and rate_anchor_confidence_runtime is not None:
            run_meta["external_rate_anchor_runtime"] = rate_anchor_runtime
            run_meta["external_rate_anchor_confidence_runtime"] = rate_anchor_confidence_runtime
        if output_rate_runtime is not None and output_rate_confidence_runtime is not None:
            run_meta["external_output_rate_runtime"] = output_rate_runtime
            run_meta["external_output_rate_confidence_runtime"] = output_rate_confidence_runtime
        if rate_posterior_runtime is not None:
            run_meta["external_rate_posterior_runtime"] = rate_posterior_runtime
        if target_observability_runtime is not None:
            run_meta["external_target_observability_runtime"] = target_observability_runtime
            run_meta["enable_target_observability_control_runtime"] = True
        posterior_output_source_runtime = str(cfg.get("rate_posterior_output_source", "off")).strip().lower()
        if posterior_output_source_runtime == "source_arbiter_v1":
            run_meta["enable_rate_source_arbiter_v1_runtime"] = True
        if posterior_output_source_runtime == "source_arbiter_v2":
            run_meta["enable_rate_source_arbiter_v2_runtime"] = True
        if posterior_output_source_runtime == "source_arbiter_v3":
            run_meta["enable_rate_source_arbiter_v3_runtime"] = True
        if str(cfg.get("parh_input", "canonical")) == "multichannel":
            run_meta["observation_families"] = names
            run_meta["external_channel_context_prior"] = [
                float(channel_reliability_priors.get(name, np.nan))
                for name in names
            ]
            if channel_context_runtime is not None and not state_role_runtime:
                run_meta["external_channel_context_prior_runtime"] = channel_context_runtime
            if state_role_runtime:
                run_meta["external_state_role_prior_runtime"] = state_role_runtime
            run_meta["observation_rate_tracks_runtime"] = {
                name: _fit_length(rate_tracks[idx], stack.shape[1]).tolist()
                for idx, name in enumerate(names)
            }
            estimate = head.run(stack, fps, run_meta)
        else:
            run_meta["assistant_observation_families_runtime"] = names
            run_meta["assistant_signals_runtime"] = {
                name: stack[idx]
                for idx, name in enumerate(names)
            }
            run_meta["assistant_rate_tracks_runtime"] = {
                name: _fit_length(rate_tracks[idx], canonical.size).tolist()
                for idx, name in enumerate(names)
            }
            estimate = head.run(canonical, fps, run_meta)
        out_payload = _copy_payload_skeleton(payload, estimate, method_name)
        out_path = out_data / path.name
        with out_path.open("wb") as f:
            pickle.dump(out_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        rows: List[Dict[str, object]] = []
        for row in cal_rows:
            rows.append(
                {
                    "data_file": path.name,
                    "video": path.stem,
                    "method": method_name,
                    "fps": fps,
                    **row,
                    "anchor_score": float(cal_meta["anchor_score"]),
                    "anchor_n_support_families": int(cal_meta["anchor_n_support_families"]),
                }
            )
        readout_rows: List[Dict[str, object]] = []
        if output_rate_meta:
            posterior_meta = output_rate_meta.get("posterior_meta", {})
            posterior_meta = posterior_meta if isinstance(posterior_meta, dict) else {}
            readout_rows.append(
                {
                    "data_file": path.name,
                    "video": path.stem,
                    "method": method_name,
                    "enabled": bool(output_rate_meta.get("enabled", False)),
                    "readout_abstained": bool(output_rate_meta.get("release_hard_abstention", False)),
                    "readout_inactive_reason": str(output_rate_meta.get("reason", "")),
                    "release_hard_abstention": bool(
                        output_rate_meta.get("release_hard_abstention", False)
                    ),
                    "source": str(output_rate_meta.get("source", "")),
                    "calibration": str(output_rate_meta.get("calibration", "")),
                    "abstention_guard": str(output_rate_meta.get("abstention_guard", "")),
                    "coverage": float(output_rate_meta.get("coverage", np.nan)),
                    "alpha_mean": float(output_rate_meta.get("alpha_mean", np.nan)),
                    "alpha_median": float(output_rate_meta.get("alpha_median", np.nan)),
                    "alpha_active_fraction": float(output_rate_meta.get("alpha_active_fraction", np.nan)),
                    "ambiguous_alias_fraction": float(output_rate_meta.get("ambiguous_alias_fraction", np.nan)),
                    "unresolved_p1d_alias_fraction": float(
                        output_rate_meta.get("unresolved_p1d_alias_fraction", np.nan)
                    ),
                    "p1d_half_rescue_fraction": float(output_rate_meta.get("p1d_half_rescue_fraction", np.nan)),
                    "weak_direct_macro_fraction": float(output_rate_meta.get("weak_direct_macro_fraction", np.nan)),
                    "alias_risk_mean": float(output_rate_meta.get("alias_risk_mean", np.nan)),
                    "h1_role_support_mean": float(output_rate_meta.get("h1_role_support_mean", np.nan)),
                    "morphology_alias_pressure_mean": float(
                        output_rate_meta.get("morphology_alias_pressure_mean", np.nan)
                    ),
                    "abstain_pressure_mean": float(output_rate_meta.get("abstain_pressure_mean", np.nan)),
                    "large_downshift_fraction": float(output_rate_meta.get("large_downshift_fraction", np.nan)),
                    "well_supported_downshift_fraction": float(
                        output_rate_meta.get("well_supported_downshift_fraction", np.nan)
                    ),
                    "guarded_downshift_fraction": float(output_rate_meta.get("guarded_downshift_fraction", np.nan)),
                    "high_base_conflict_fraction": float(output_rate_meta.get("high_base_conflict_fraction", np.nan)),
                    "posterior_confidence_mean": float(posterior_meta.get("confidence_mean", np.nan)),
                    "posterior_entropy_median": float(posterior_meta.get("posterior_entropy_median", np.nan)),
                    "posterior_top_gap_median": float(posterior_meta.get("posterior_top_gap_median", np.nan)),
                    "posterior_macro_support_median": float(posterior_meta.get("macro_support_median", np.nan)),
                    "posterior_direct_macro_support_median": float(
                        posterior_meta.get("direct_macro_support_median", np.nan)
                    ),
                    "posterior_motion_direct_support_median": float(
                        posterior_meta.get("motion_direct_support_median", np.nan)
                    ),
                    "posterior_alias_risk_median": float(posterior_meta.get("alias_risk_median", np.nan)),
                    "posterior_independent_timing_support_median": float(
                        posterior_meta.get("independent_timing_support_median", np.nan)
                    ),
                    "posterior_bridge_timing_preservation_median": float(
                        posterior_meta.get("bridge_timing_preservation_median", np.nan)
                    ),
                    "posterior_morphology_alias_pressure_median": float(
                        posterior_meta.get("morphology_alias_pressure_median", np.nan)
                    ),
                    "posterior_h1_role_support_median": float(posterior_meta.get("h1_role_support_median", np.nan)),
                    "posterior_abstain_pressure_median": float(posterior_meta.get("abstain_pressure_median", np.nan)),
                    "posterior_signal_sqi_observability_enabled": bool(
                        posterior_meta.get("signal_sqi_observability_enabled", False)
                    ),
                    "posterior_signal_sqi_support_median": float(
                        posterior_meta.get("signal_sqi_support_median", np.nan)
                    ),
                    "posterior_phase_coherence_support_median": float(
                        posterior_meta.get("phase_coherence_support_median", np.nan)
                    ),
                    "source_validity_confidence_mean": float(
                        output_rate_meta.get("confidence_mean", np.nan)
                        if output_rate_meta.get("source") in {"source_validity_posterior", "source_validity_guarded"}
                        else np.nan
                    ),
                    "source_validity_entropy_median": float(
                        output_rate_meta.get("source_validity_entropy_median", np.nan)
                    ),
                    "source_validity_top_gap_median": float(
                        output_rate_meta.get("source_validity_top_gap_median", np.nan)
                    ),
                    "source_validity_mode_hz_median": float(
                        output_rate_meta.get("source_validity_mode_hz_median", np.nan)
                    ),
                    "source_validity_selected_group_counts": json.dumps(
                        output_rate_meta.get("selected_group_counts", {}),
                        sort_keys=True,
                    ),
                    "source_validity_guard_alpha_mean": float(
                        output_rate_meta.get("source_validity_guard_alpha_mean", np.nan)
                    ),
                    "source_validity_guard_alpha_median": float(
                        output_rate_meta.get("source_validity_guard_alpha_median", np.nan)
                    ),
                    "source_validity_guard_active_fraction": float(
                        output_rate_meta.get("source_validity_guard_active_fraction", np.nan)
                    ),
                    "source_validity_global_specificity": float(
                        output_rate_meta.get("source_validity_global_specificity", np.nan)
                    ),
                    "specificity_boost_enabled": bool(output_rate_meta.get("specificity_boost_enabled", False)),
                    "specific_posterior_correction_fraction": float(
                        output_rate_meta.get("specific_posterior_correction_fraction", np.nan)
                    ),
                    "readout_group_rate_mode_counts": json.dumps(
                        output_rate_meta.get("group_rate_mode_counts", {}),
                        sort_keys=True,
                    ),
                    "posterior_group_rate_mode_counts": json.dumps(
                        posterior_meta.get("group_rate_mode_counts", {}),
                        sort_keys=True,
                    ),
                    "readout_derived_consistency_counts": json.dumps(
                        output_rate_meta.get("derived_consistency_counts", {}),
                        sort_keys=True,
                    ),
                    "posterior_derived_consistency_counts": json.dumps(
                        posterior_meta.get("derived_consistency_counts", {}),
                        sort_keys=True,
                    ),
                }
            )
        return path.name, rows, readout_rows, None
    except Exception as exc:
        return path.name, [], [], f"{type(exc).__name__}: {exc}"


def _worker_count(requested: int) -> int:
    if requested and requested > 0:
        return int(requested)
    for key in ("RESPYRE_JOBS", "PARALLEL_PROCS"):
        raw = os.environ.get(key)
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    return 1


def _write_activation_audit(out_data: Path, metrics_dir: Path) -> None:
    rows: List[Dict[str, object]] = []
    for path in sorted(Path(out_data).glob("*.pkl")):
        try:
            payload = _load_pkl(path)
            estimates = payload.get("estimates") or []
            est = estimates[0].get("estimate", {}) if estimates and isinstance(estimates[0], dict) else {}
            meta_raw = est.get("meta", {})
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw or {})
            diag = meta.get("parh_ossm_diagnostics", {})
            obs_law = meta.get("observation_law", {})
            phase_morph = meta.get("phase_anchored_morphology", {})
            residual_guard = meta.get("residual_identifiability_guard", {})
            rows.append(
                {
                    "video": path.stem,
                    "observation_law_enabled": bool(obs_law.get("enabled", False)),
                    "reliability_prior_scope": str(meta.get("reliability_prior_scope", "")),
                    "windowed_reliability_prior_applied": bool(
                        meta.get("target_reliability_windowed_prior", False)
                    ),
                    "state_role_prior_applied": bool(meta.get("target_state_role_prior", False)),
                    "separate_reliability_csvs": bool(meta.get("separate_reliability_csvs", False)),
                    "target_observability_control_enabled": bool(
                        meta.get("target_observability_control_enabled", False)
                    ),
                    "external_rate_posterior_applied": bool(meta.get("external_rate_posterior_applied", False)),
                    "external_rate_posterior_coverage": float(
                        diag.get("external_rate_posterior_coverage", np.nan)
                    ),
                    "external_rate_posterior_blend_active_frac": float(
                        diag.get("external_rate_posterior_blend_active_frac", np.nan)
                    ),
                    "mixture_entropy_mean": float(diag.get("mixture_entropy_mean", np.nan)),
                    "residual_guard_enabled": bool(residual_guard.get("enabled", False)),
                    "residual_gate_mean": float(diag.get("residual_gate_mean", np.nan)),
                    "residual_failure_absorption_mean": float(
                        diag.get("residual_failure_absorption_mean", np.nan)
                    ),
                    "phase_anchored_morphology_enabled": bool(phase_morph.get("enabled", False)),
                    "phase_anchored_morphology_blend": float(phase_morph.get("blend", np.nan)),
                    "active_modules": ",".join(str(x) for x in meta.get("active_modules", [])),
                }
            )
        except Exception as exc:
            rows.append({"video": path.stem, "audit_error": f"{type(exc).__name__}: {exc}"})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("video")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "activation_audit_raw.csv", index=False)
    summary = {
        "n_trials": int(len(df)),
        "observation_law_enabled_frac": float(df.get("observation_law_enabled", pd.Series(dtype=float)).mean()),
        "target_observability_control_enabled_frac": float(
            df.get("target_observability_control_enabled", pd.Series(dtype=float)).mean()
        ),
        "windowed_reliability_prior_applied_frac": float(
            df.get("windowed_reliability_prior_applied", pd.Series(dtype=float)).mean()
        ),
        "state_role_prior_applied_frac": float(
            df.get("state_role_prior_applied", pd.Series(dtype=float)).mean()
        ),
        "external_rate_posterior_applied_frac": float(
            df.get("external_rate_posterior_applied", pd.Series(dtype=float)).mean()
        ),
        "phase_anchored_morphology_enabled_frac": float(
            df.get("phase_anchored_morphology_enabled", pd.Series(dtype=float)).mean()
        ),
        "median_mixture_entropy": float(
            pd.to_numeric(df.get("mixture_entropy_mean", pd.Series(dtype=float)), errors="coerce").median()
        ),
        "median_residual_gate": float(
            pd.to_numeric(df.get("residual_gate_mean", pd.Series(dtype=float)), errors="coerce").median()
        ),
    }
    (metrics_dir / "activation_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if bool(args.enable_observation_law) and str(args.parh_input) != "multichannel":
        raise SystemExit("--enable-observation-law requires --parh-input multichannel")
    rate_output_internal = _normalize_rate_posterior_output_source(args.rate_posterior_output_source)
    rate_output_public = _public_rate_posterior_output_source(args.rate_posterior_output_source)
    rate_output_role = _rate_posterior_output_role(rate_output_public)
    if rate_output_role == "unknown":
        raise SystemExit(
            f"unknown --rate-posterior-output-source={args.rate_posterior_output_source!r}; "
            "use 'final' for the release path or 'off' to disable the posterior readout"
        )
    data_dir = Path(args.data_dir).expanduser().absolute()
    out_run = Path(args.out_run).expanduser().absolute()
    out_data = out_run / "data"
    metrics_dir = out_run / "metrics"
    out_data.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.pkl"))
    if args.max_files and args.max_files > 0:
        files = files[: int(args.max_files)]
    if not files:
        raise SystemExit(f"no PKL files found in {data_dir}")

    cfg = {
        "name": args.name,
        "source_key": args.source_key,
        "rate_track_source": args.rate_track_source,
        "external_rate_evidence_source": args.external_rate_evidence_source,
        "min_hz": args.min_hz,
        "max_hz": args.max_hz,
        "lag_max_sec": args.lag_max_sec,
        "min_support_corr": args.min_support_corr,
        "parh_input": args.parh_input,
        "anchor_policy": args.anchor_policy,
        "parh_base_method": args.parh_base_method,
        "canonical_policy": args.canonical_policy,
        "reliability_group_csv": str(args.reliability_group_csv) if args.reliability_group_csv else "",
        "state_reliability_group_csv": (
            str(args.state_reliability_group_csv) if args.state_reliability_group_csv else ""
        ),
        "readout_reliability_group_csv": (
            str(args.readout_reliability_group_csv) if args.readout_reliability_group_csv else ""
        ),
        "reliability_prior_scope": str(args.reliability_prior_scope),
        "reliability_priors_by_video": _load_reliability_priors(args.reliability_group_csv),
        "windowed_reliability_priors_by_video": _load_windowed_reliability_priors(args.reliability_group_csv),
        "separate_reliability_csvs": bool(args.state_reliability_group_csv or args.readout_reliability_group_csv),
        "state_reliability_priors_by_video": _load_reliability_priors(
            args.state_reliability_group_csv or args.reliability_group_csv
        ),
        "state_windowed_reliability_priors_by_video": _load_windowed_reliability_priors(
            args.state_reliability_group_csv or args.reliability_group_csv
        ),
        "readout_reliability_priors_by_video": _load_reliability_priors(
            args.readout_reliability_group_csv or args.reliability_group_csv
        ),
        "readout_windowed_reliability_priors_by_video": _load_windowed_reliability_priors(
            args.readout_reliability_group_csv or args.reliability_group_csv
        ),
        "enable_phase_anchor_validation": bool(args.enable_phase_anchor_validation),
        "enable_rate_hypothesis_graph": bool(args.enable_rate_hypothesis_graph),
        "enable_rate_posterior": bool(args.enable_rate_posterior),
        "rate_posterior_output_source": rate_output_internal,
        "rate_posterior_output_public_source": rate_output_public,
        "rate_posterior_output_role": rate_output_role,
        "enable_target_observability_control": bool(args.enable_target_observability_control),
        "enable_signal_sqi_observability": bool(args.enable_signal_sqi_observability),
        "enable_derived_consistency_scaling": bool(args.enable_derived_consistency_scaling),
        "enable_observation_law": bool(args.enable_observation_law),
        "enable_regime_observation_law": bool(args.enable_regime_observation_law or args.enable_observation_law),
        "regime_anchor_policy": str(args.regime_anchor_policy),
        "enable_decoupled_rate_readout": bool(args.enable_decoupled_rate_readout or args.enable_observation_law),
        "eval_use_track": bool(args.eval_use_track),
    }
    manifest_cfg = dict(cfg)
    manifest_cfg["rate_posterior_output_source"] = rate_output_public
    manifest_cfg["external_rate_evidence_source"] = args.external_rate_evidence_source
    manifest_cfg.pop("rate_track_source", None)
    manifest_cfg.pop("rate_posterior_output_public_source", None)
    prior_keys = [
        "reliability_priors_by_video",
        "windowed_reliability_priors_by_video",
        "state_reliability_priors_by_video",
        "state_windowed_reliability_priors_by_video",
        "readout_reliability_priors_by_video",
        "readout_windowed_reliability_priors_by_video",
    ]
    prior_summary = {
        key: len(manifest_cfg.get(key) or {})
        for key in prior_keys
    }
    for key in prior_keys:
        manifest_cfg.pop(key, None)
    manifest_cfg["reliability_prior_entry_counts"] = prior_summary
    manifest = {
        "name": args.name,
        "source_key": args.source_key,
        "data_dir": str(data_dir),
        "out_run": str(out_run),
        "n_files": len(files),
        "release_status": (
            "release_candidate"
            if (
                bool(args.enable_observation_law)
                and bool(args.enable_rate_posterior)
                and rate_output_public == "final"
            )
            else "diagnostic_or_ablation"
        ),
        "nested_comparator": "",
        "parh_input": str(args.parh_input),
        "external_rate_evidence_source": str(args.external_rate_evidence_source),
        "rate_posterior_output_source": rate_output_public,
        "rate_posterior_output_role": cfg["rate_posterior_output_role"],
        "enable_observation_law": bool(args.enable_observation_law),
        "enable_rate_posterior": bool(args.enable_rate_posterior),
        "eval_use_track": bool(args.eval_use_track),
        "methods": [
            {"method_label": m, "family_name": f, "group": FAMILY_GROUPS.get(f, g)}
            for m, f, g in DEFAULT_METHODS
        ],
        "cfg": manifest_cfg,
    }
    (metrics_dir / "calibrated_multifamily_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if bool(args.enable_rate_posterior) and cfg["rate_posterior_output_role"] == "diagnostic_or_compatibility":
        print(
            "[materialize] WARNING: --rate-posterior-output-source="
            f"{rate_output_public} is marked diagnostic/compatibility, "
            "not release. Use final for the current bounded "
            "locked readout."
        )

    jobs = _worker_count(args.jobs)
    tasks = [(str(p), str(out_data), cfg) for p in files]
    rows: List[Dict[str, object]] = []
    readout_rows: List[Dict[str, object]] = []
    errors: List[str] = []
    print(f"> Materializing {len(tasks)} calibrated multi-family PARH trials with jobs={jobs}")
    if jobs == 1:
        for task in tasks:
            name, task_rows, task_readout_rows, err = _process_file(task)
            rows.extend(task_rows)
            readout_rows.extend(task_readout_rows)
            if err:
                errors.append(f"{name}: {err}")
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(_process_file, task) for task in tasks]
            for fut in as_completed(futures):
                name, task_rows, task_readout_rows, err = fut.result()
                rows.extend(task_rows)
                readout_rows.extend(task_readout_rows)
                if err:
                    errors.append(f"{name}: {err}")

    if rows:
        pd.DataFrame(rows).sort_values(["data_file", "channel"]).to_csv(
            metrics_dir / "calibration_raw.csv",
            index=False,
        )
        summary = (
            pd.DataFrame(rows)
            .groupby(["anchor", "channel"], dropna=False)
            .agg(
                n=("data_file", "count"),
                corr_median=("corr", "median"),
                support_median=("support", "median"),
                lag_samples_median=("lag_samples", "median"),
                residual_mad_median=("residual_mad", "median"),
            )
            .reset_index()
        )
        summary.to_csv(metrics_dir / "calibration_summary.csv", index=False)
    if readout_rows:
        readout_df = pd.DataFrame(readout_rows).sort_values(["data_file"])
        readout_df.to_csv(metrics_dir / "readout_guard_raw.csv", index=False)
        numeric_cols = [
            c
            for c in readout_df.columns
            if c not in {"data_file", "video", "method", "source", "calibration", "abstention_guard"}
            and pd.api.types.is_numeric_dtype(readout_df[c])
        ]
        if numeric_cols:
            summary_row = {
                "method": str(args.name),
                "n_trials": int(len(readout_df)),
            }
            for col in numeric_cols:
                vals = pd.to_numeric(readout_df[col], errors="coerce")
                summary_row[f"{col}_median"] = float(vals.median()) if vals.notna().any() else float("nan")
                summary_row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            pd.DataFrame([summary_row]).to_csv(metrics_dir / "readout_guard_summary.csv", index=False)
    _write_activation_audit(out_data, metrics_dir)
    errors_path = metrics_dir / "materialization_errors.txt"
    if errors:
        errors_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"> Materialization completed with {len(errors)} skipped/error trials")
        if _is_strict_release_cfg(cfg):
            print("> Locked activation contract failed; refusing to continue to evaluation.")
            return 1
    else:
        if errors_path.exists():
            errors_path.unlink()
        print("> Materialization completed without skipped trials")

    if not args.skip_eval:
        run_evaluation(
            str(out_run.parent),
            run_label=out_run.name,
            win_size=float(args.win_size),
            stride=float(args.stride),
            min_hz=float(args.min_hz),
            max_hz=float(args.max_hz),
            eval_cfg={"use_track": True} if bool(args.eval_use_track) else None,
            save_metric_pickles=bool(_artifact_policy_settings(args.artifact_policy)["save_metric_pickles"]),
        )

    settings = _artifact_policy_settings(args.artifact_policy)
    _prune_run_artifacts(
        out_run,
        keep_data=bool(settings["keep_data"]),
        save_metric_pickles=bool(settings["save_metric_pickles"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
