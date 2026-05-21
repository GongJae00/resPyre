#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_calibrated_multifamily_parh_system import DEFAULT_METHODS as MATERIALIZE_DEFAULT_METHODS


DEFAULT_METHODS = [method_label for method_label, _family_name, _group_name in MATERIALIZE_DEFAULT_METHODS]


GROUP_ORDER = [
    "G_OF",
    "G_OF_bridge",
    "G_DoF",
    "G_DoF_bridge",
    "G_P1D_low",
    "G_P1D_morph",
    "G_P1D_cons",
]

GROUP_LABELS = {
    "G_OF": "OF",
    "G_OF_bridge": "OF_bridge",
    "G_DoF": "DoF",
    "G_DoF_bridge": "DoF_bridge",
    "G_P1D_low": "P1D_lin",
    "G_P1D_morph": "P1D_quad/cub",
    "G_P1D_cons": "P1D_cons",
}

MACRO_FAMILY = {
    "G_OF": "OF",
    "G_OF_bridge": "OF",
    "G_DoF": "DoF",
    "G_DoF_bridge": "DoF",
    "G_P1D_low": "P1D",
    "G_P1D_morph": "P1D",
    "G_P1D_cons": "P1D",
}

BRIDGE_COUNTERPARTS = {
    "G_OF": {"G_OF_bridge"},
    "G_OF_bridge": {"G_OF"},
    "G_DoF": {"G_DoF_bridge"},
    "G_DoF_bridge": {"G_DoF"},
    "G_P1D_low": {"G_P1D_morph", "G_P1D_cons"},
    "G_P1D_morph": {"G_P1D_low", "G_P1D_cons"},
    "G_P1D_cons": {"G_P1D_low", "G_P1D_morph"},
}

COMPONENT_ROLE_PRIORS = {
    "G_OF": dict(h1=0.80, h2=0.30, b=0.45, r=0.35, z_osc=0.80, z_full=0.45),
    "G_OF_bridge": dict(h1=0.72, h2=0.40, b=0.60, r=0.45, z_osc=0.70, z_full=0.55),
    "G_DoF": dict(h1=0.90, h2=0.25, b=0.35, r=0.30, z_osc=0.90, z_full=0.35),
    "G_DoF_bridge": dict(h1=0.82, h2=0.45, b=0.55, r=0.50, z_osc=0.82, z_full=0.58),
    "G_P1D_low": dict(h1=0.45, h2=0.70, b=0.55, r=0.55, z_osc=0.42, z_full=0.75),
    "G_P1D_morph": dict(h1=0.42, h2=0.82, b=0.55, r=0.70, z_osc=0.40, z_full=0.88),
    "G_P1D_cons": dict(h1=0.55, h2=0.78, b=0.62, r=0.75, z_osc=0.52, z_full=0.92),
}


def _component_role_weights(group: str, timing: float, morphology: float, abstain: float) -> dict[str, float]:
    timing = _clip01(timing) if np.isfinite(float(timing)) else 0.0
    morphology = _clip01(morphology) if np.isfinite(float(morphology)) else 0.0
    evidence = math.sqrt(max(timing, 0.0) * max(morphology, 0.0))
    prior = COMPONENT_ROLE_PRIORS.get(str(group), {})
    weights = {
        "h1_timing_weight": _clip01(timing * float(prior.get("h1", 0.5))),
        "h2_harmonic_weight": _clip01(morphology * float(prior.get("h2", 0.5))),
        "b_baseline_weight": _clip01(evidence * float(prior.get("b", 0.5))),
        "r_residual_weight": _clip01(morphology * (1.0 - 0.5 * _clip01(abstain)) * float(prior.get("r", 0.5))),
        "z_osc_readout_weight": _clip01(timing * float(prior.get("z_osc", 0.5))),
        "z_full_readout_weight": _clip01(morphology * float(prior.get("z_full", 0.5))),
    }
    weights["component_abstain_score"] = _clip01(max(float(abstain), 1.0 - max(weights["z_osc_readout_weight"], weights["z_full_readout_weight"])))
    return weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract GT-free target-side reliability graph features. The script "
            "scores observation groups using bounded cross-family transform support, "
            "rate/phase agreement, morphology residuals, nuisance evidence, and "
            "harmonic ambiguity. Target respiration labels are not used."
        )
    )
    p.add_argument("--data-dir", type=Path, action="append", required=True)
    p.add_argument("--dataset-label", action="append", default=[])
    p.add_argument("--method", action="append", default=[])
    p.add_argument("--candidate-output", choices=["signal_hat", "z_full", "z_osc"], default="signal_hat")
    p.add_argument("--out-edge", type=Path, required=True)
    p.add_argument("--out-group", type=Path, required=True)
    p.add_argument("--out-summary", type=Path, required=True)
    p.add_argument("--report-out", type=Path)
    p.add_argument("--min-hz", type=float, default=0.08)
    p.add_argument("--max-hz", type=float, default=0.50)
    p.add_argument("--feature-fs", type=float, default=8.0)
    p.add_argument("--max-lag-sec", type=float, default=6.0)
    p.add_argument(
        "--window-sec",
        type=float,
        default=0.0,
        help="If >0, compute reliability on sliding windows instead of whole trials.",
    )
    p.add_argument(
        "--window-stride-sec",
        type=float,
        default=0.0,
        help="Sliding-window stride. Defaults to half of --window-sec when omitted.",
    )
    p.add_argument("--min-support-corr", type=float, default=0.25)
    p.add_argument("--max-support-residual", type=float, default=1.25)
    p.add_argument("--max-files", type=int)
    p.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("RESPYRE_JOBS", os.environ.get("PARALLEL_PROCS", "1"))),
    )
    return p.parse_args()


def _canonical_base(method: str) -> str:
    text = str(method or "").strip().lower().replace(" ", "_")
    for suffix in ("__parh_ossm", "__kfstd"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    if text == "profile1d_linear":
        return "profile1d_linear"
    if text == "profile1d_quadratic":
        return "profile1d_quadratic"
    if text == "profile1d_cubic":
        return "profile1d_cubic"
    if text in {"dof", "of_farneback", "of_disp_bridge", "dof_disp_bridge", "profile1d_consensus"}:
        return text
    return text


def method_group(method: str) -> str | None:
    base = _canonical_base(method)
    if base == "of_farneback":
        return "G_OF"
    if base == "of_disp_bridge":
        return "G_OF_bridge"
    if base == "dof":
        return "G_DoF"
    if base == "dof_disp_bridge":
        return "G_DoF_bridge"
    if base == "profile1d_linear":
        return "G_P1D_low"
    if base in {"profile1d_quadratic", "profile1d_cubic"}:
        return "G_P1D_morph"
    if base == "profile1d_consensus":
        return "G_P1D_cons"
    return None


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _safe_median(values: Iterable[float]) -> float:
    arr = _finite_array(values)
    return float(np.nanmedian(arr)) if arr.size else float("nan")


def _clip01(x: float) -> float:
    if not np.isfinite(float(x)):
        return float("nan")
    return float(max(0.0, min(1.0, float(x))))


def _estimate_lookup(payload: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    estimates = payload.get("estimates", [])
    if not isinstance(estimates, list):
        return out
    for row in estimates:
        if not isinstance(row, dict):
            continue
        method = row.get("method")
        if method is None:
            continue
        estimate = row.get("estimate", row)
        out[str(method)] = estimate if isinstance(estimate, dict) else {"signal_hat": estimate}
    return out


def _extract_signal(estimate: object, source_key: str) -> np.ndarray | None:
    if isinstance(estimate, dict):
        value = estimate.get(source_key)
        if value is None and source_key != "signal_hat":
            value = estimate.get("signal_hat")
    else:
        value = estimate
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0 or np.count_nonzero(np.isfinite(arr)) == 0:
        return None
    return arr


def _finite_fill(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1).copy()
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if np.all(finite):
        return arr
    if not np.any(finite):
        return np.zeros_like(arr)
    idx = np.arange(arr.size)
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


def _resample_signal(signal: np.ndarray, source_fs: float, target_fs: float) -> tuple[np.ndarray, float]:
    arr = _finite_fill(signal)
    source_fs = float(source_fs)
    target_fs = float(target_fs)
    if arr.size < 2 or not np.isfinite(source_fs) or not np.isfinite(target_fs) or source_fs <= 0.0 or target_fs <= 0.0:
        return arr, source_fs
    if abs(source_fs - target_fs) / max(source_fs, 1e-9) < 0.05:
        return arr, source_fs
    duration = arr.size / source_fs
    n_target = max(2, int(round(duration * target_fs)))
    src_t = np.linspace(0.0, duration, arr.size, endpoint=False)
    dst_t = np.linspace(0.0, duration, n_target, endpoint=False)
    return np.interp(dst_t, src_t, arr).astype(np.float64), target_fs


def _standardize(signal: np.ndarray) -> np.ndarray:
    arr = _finite_fill(signal)
    if arr.size == 0:
        return arr
    arr = arr - float(np.nanmedian(arr))
    scale = float(np.nanstd(arr))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(arr)
    return arr / scale


def _dominant_hz(signal: np.ndarray, fs: float, min_hz: float, max_hz: float) -> tuple[float, float]:
    arr = _standardize(signal)
    fs = float(fs)
    if arr.size < 8 or not np.isfinite(fs) or fs <= 0.0:
        return float("nan"), float("nan")
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / fs)
    power = np.abs(np.fft.rfft(arr)) ** 2
    band = (freqs >= float(min_hz)) & (freqs <= float(max_hz))
    if not np.any(band):
        return float("nan"), float("nan")
    band_freqs = freqs[band]
    band_power = power[band]
    if band_power.size == 0 or not np.isfinite(band_power).any():
        return float("nan"), float("nan")
    peak = int(np.nanargmax(band_power))
    total_power = float(np.nansum(power[freqs > 0.0]))
    rel_power = float(band_power[peak] / max(total_power, 1e-12))
    return float(band_freqs[peak]), _clip01(rel_power)


def _track_median_hz(estimate: object, fallback_hz: float) -> float:
    if not isinstance(estimate, dict):
        return float(fallback_hz)
    value = estimate.get("track_hz")
    if value is None:
        value = estimate.get("track_hz_smoothed")
    if value is None:
        value = estimate.get("final_track_hz")
    if value is None:
        return float(fallback_hz)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return float(fallback_hz)
    return float(np.nanmedian(arr))


def _method_allowed(method: str, methods: Sequence[str]) -> bool:
    if not methods:
        return True
    wanted = {str(m) for m in methods}
    if str(method) in wanted:
        return True
    base = _canonical_base(method)
    return base in {_canonical_base(m) for m in wanted}


def _load_trial_signals(
    path: Path,
    *,
    dataset: str,
    methods: Sequence[str],
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    payload = pickle.loads(Path(path).read_bytes())
    if not isinstance(payload, dict):
        return [], {}
    fps = float(payload.get("fps", float("nan")) or float("nan"))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = float(getattr(args, "feature_fs", 8.0) or 8.0)
    source_key = str(getattr(args, "candidate_output", "signal_hat"))
    rows: list[dict] = []
    for method, estimate in _estimate_lookup(payload).items():
        if not _method_allowed(method, methods):
            continue
        signal = _extract_signal(estimate, source_key)
        if signal is None:
            continue
        band, fs = _resample_signal(signal, fps, float(getattr(args, "feature_fs", fps) or fps))
        band = _standardize(band)
        dom_hz, spectral_score = _dominant_hz(
            band,
            fs,
            float(getattr(args, "min_hz", 0.08)),
            float(getattr(args, "max_hz", 0.50)),
        )
        rows.append(
            {
                "dataset": dataset,
                "video": Path(path).stem,
                "method": method,
                "family": _canonical_base(method),
                "stage": source_key,
                "band": band,
                "fs": float(fs),
                "n": int(band.size),
                "duration_sec": float(band.size) / max(float(fs), 1e-12),
                "dom_hz": dom_hz,
                "track_median_hz": _track_median_hz(estimate, dom_hz),
                "spectral_score": spectral_score,
            }
        )
    return rows, payload


def _best_lagged_pair(source: np.ndarray, target: np.ndarray, fs: float, max_lag_sec: float) -> tuple[np.ndarray, np.ndarray, int]:
    a = _standardize(source)
    b = _standardize(target)
    n = min(a.size, b.size)
    if n < 8:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0
    a = a[:n]
    b = b[:n]
    max_lag = min(int(round(float(max_lag_sec) * max(float(fs), 1e-9))), max(n // 3, 0))
    best_corr = -np.inf
    best_pair = (a, b, 0)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: aa.size]
        elif lag > 0:
            aa = a[: n - lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b
        if aa.size < 8 or bb.size < 8:
            continue
        corr = float(np.corrcoef(aa, bb)[0, 1])
        if np.isfinite(corr) and abs(corr) > best_corr:
            best_corr = abs(corr)
            best_pair = (aa, bb, lag)
    return best_pair


def _relative_observation_equation(
    *,
    source: np.ndarray,
    target: np.ndarray,
    fs: float,
    max_lag_sec: float,
) -> dict[str, float]:
    src, dst, lag = _best_lagged_pair(source, target, fs, max_lag_sec)
    if src.size < 8 or dst.size < 8:
        return {}
    corr = float(np.corrcoef(src, dst)[0, 1])
    if not np.isfinite(corr):
        return {}
    sign = 1.0 if corr >= 0.0 else -1.0
    aligned = sign * src
    gain = float(np.dot(aligned, dst) / max(float(np.dot(aligned, aligned)), 1e-12))
    fitted = gain * aligned
    residual = dst - fitted
    residual_norm = float(np.nanstd(residual) / max(float(np.nanstd(dst)), 1e-12))
    abs_corr = abs(corr)
    lag_sec = float(lag) / max(float(fs), 1e-12)
    residual_score = 1.0 / (1.0 + max(residual_norm, 0.0))
    lag_score = 1.0 / (1.0 + abs(lag_sec) / max(float(max_lag_sec), 1e-9))
    pair_score = _geometric_mean([abs_corr, residual_score, lag_score])
    return {
        "corr": corr,
        "abs_corr": abs_corr,
        "sign": sign,
        "gain": gain,
        "lag_sec": lag_sec,
        "abs_lag_sec": abs(lag_sec),
        "residual_norm": residual_norm,
        "pair_score": pair_score,
    }


def _decay_score(value: float, scale: float) -> float:
    if not np.isfinite(float(value)):
        return float("nan")
    return float(1.0 / (1.0 + max(0.0, float(value)) / max(float(scale), 1e-9)))


def _rate_score_from_bpm(diff_bpm: float) -> float:
    # Six BPM is intentionally a tolerance scale, not a tunable selector.
    return _decay_score(diff_bpm, 6.0)


def _harmonic_ambiguous(a_hz: float, b_hz: float) -> float:
    if not (np.isfinite(float(a_hz)) and np.isfinite(float(b_hz))):
        return float("nan")
    lo = min(abs(float(a_hz)), abs(float(b_hz)))
    hi = max(abs(float(a_hz)), abs(float(b_hz)))
    if lo <= 1e-9:
        return float("nan")
    ratio = hi / lo
    return 1.0 if abs(ratio - 2.0) <= 0.18 else 0.0


def _macro_family(group: str) -> str:
    return MACRO_FAMILY.get(str(group), str(group))


def _local_peak_indices(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return np.array([], dtype=int)
    mid = arr[1:-1]
    return np.flatnonzero((mid >= arr[:-2]) & (mid > arr[2:])) + 1


def _interval_regular_score(events: np.ndarray, fs: float, min_hz: float, max_hz: float) -> float:
    idx = np.asarray(events, dtype=np.float64).reshape(-1)
    if idx.size < 3 or fs <= 0.0:
        return float("nan")
    intervals = np.diff(idx) / float(fs)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    if intervals.size < 2:
        return float("nan")
    period_min = 1.0 / max(float(max_hz), 1e-9)
    period_max = 1.0 / max(float(min_hz), 1e-9)
    candidates = []
    for multiplier in (1.0, 2.0):
        periods = intervals * multiplier
        valid = periods[(periods >= period_min) & (periods <= period_max)]
        if valid.size < 2:
            continue
        med = float(np.median(valid))
        mad = float(np.median(np.abs(valid - med)))
        cv = float(mad / max(med, 1e-9))
        regularity = float(np.exp(-0.5 * (cv / 0.22) ** 2))
        coverage = float(np.clip(valid.size / max(intervals.size, 1), 0.0, 1.0))
        density = float(np.clip(idx.size / 4.0, 0.0, 1.0))
        candidates.append(regularity * np.sqrt(coverage * density))
    return float(max(candidates)) if candidates else float("nan")


def _event_timing_score(signal: np.ndarray, fs: float, min_hz: float, max_hz: float) -> float:
    """GT-free respiratory event regularity.

    This is intentionally event-based rather than smoothness-based. It gives
    DoF-like burst observations a way to express timing evidence without
    requiring a clean sinusoidal waveform.
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size < max(16, int(round(6.0 * max(float(fs), 1.0)))) or fs <= 0.0:
        return float("nan")
    finite = np.isfinite(x)
    if np.count_nonzero(finite) < max(16, x.size // 3):
        return float("nan")
    fill = float(np.nanmedian(x[finite]))
    x = np.where(finite, x, fill)
    scale = float(np.nanstd(x))
    if not np.isfinite(scale) or scale <= 1e-9:
        return float("nan")
    z = (x - float(np.nanmedian(x))) / scale

    scores = []
    min_gap = max(1, int(round(0.45 / max(float(max_hz), 1e-9) * fs)))
    for series, q in ((z, 65.0), (-z, 65.0), (np.abs(z), 75.0)):
        peaks = _local_peak_indices(series)
        if peaks.size == 0:
            continue
        thresh = float(np.nanpercentile(series, q))
        peaks = peaks[series[peaks] >= thresh]
        if peaks.size == 0:
            continue
        kept = []
        for idx in peaks:
            if not kept or int(idx) - int(kept[-1]) >= min_gap:
                kept.append(int(idx))
            elif series[int(idx)] > series[int(kept[-1])]:
                kept[-1] = int(idx)
        score = _interval_regular_score(np.asarray(kept, dtype=int), fs, min_hz, max_hz)
        if np.isfinite(score):
            periodic_support = 0.0
            intervals = np.diff(np.asarray(kept, dtype=np.float64)) / float(fs)
            period_min = 1.0 / max(float(max_hz), 1e-9)
            period_max = 1.0 / max(float(min_hz), 1e-9)
            for multiplier in (1.0, 2.0):
                periods = intervals * multiplier
                valid = periods[(periods >= period_min) & (periods <= period_max)]
                if valid.size < 2:
                    continue
                lag = int(round(float(np.median(valid)) * float(fs)))
                if lag <= 0 or lag >= z.size // 2:
                    continue
                a = z[:-lag]
                b = z[lag:]
                if a.size < 8 or b.size < 8:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])
                if np.isfinite(corr):
                    periodic_support = max(periodic_support, _clip01(corr))
            salience = float(np.clip((np.nanmedian(series[kept]) - thresh + 1e-6) / 1.5, 0.0, 1.0))
            scores.append(
                float(
                    np.sqrt(
                        max(score, 0.0)
                        * max(0.25 + 0.75 * salience, 0.0)
                        * max(periodic_support, 0.0)
                    )
                )
            )
    return float(max(scores)) if scores else float("nan")


def _geometric_mean(scores: Iterable[float]) -> float:
    vals = [_clip01(v) for v in scores]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    eps = 1e-4
    return float(math.exp(float(np.mean(np.log(np.maximum(vals, eps))))))


def _trial_observability(path: Path, signals: list[dict]) -> dict[str, float | str]:
    fps = float("nan")
    estimate_count = float("nan")
    try:
        payload = pickle.loads(path.read_bytes())
        fps = float(payload.get("fps", float("nan")) or float("nan"))
        estimate_count = float(len(payload.get("estimates", []) or []))
    except Exception:
        pass
    duration = _safe_median(float(row.get("duration_sec", float("nan"))) for row in signals)
    fps_score = _clip01(fps / 20.0) if np.isfinite(fps) else float("nan")
    duration_score = _clip01(duration / 45.0) if np.isfinite(duration) else float("nan")
    coverage_score = _clip01(float(len(signals)) / max(float(len(DEFAULT_METHODS)), 1.0))
    roi_proxy = _geometric_mean([fps_score, duration_score, coverage_score])
    return {
        "native_fps": fps,
        "estimate_count": estimate_count,
        "duration_sec": duration,
        "roi_observability_score": roi_proxy,
        "roi_observability_source": "fps_duration_method_coverage_proxy",
    }


def _signal_rows(signals: list[dict], *, min_hz: float = 0.08, max_hz: float = 0.50) -> list[dict]:
    rows: list[dict] = []
    for row in signals:
        group = method_group(str(row.get("method", "")))
        if group is None:
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "video": row["video"],
                "method": row["method"],
                "base_method": _canonical_base(str(row["method"])),
                "group": group,
                "group_label": GROUP_LABELS[group],
                "family": row.get("family", ""),
                "stage": row.get("stage", ""),
                "spectral_score": float(row.get("spectral_score", float("nan"))),
                "dom_hz": float(row.get("dom_hz", float("nan"))),
                "track_median_hz": float(row.get("track_median_hz", float("nan"))),
                "event_timing_score": _event_timing_score(
                    np.asarray(row.get("band", []), dtype=np.float64),
                    float(row.get("fs", float("nan"))),
                    float(min_hz),
                    float(max_hz),
                ),
                "duration_sec": float(row.get("duration_sec", float("nan"))),
                "n": int(row.get("n", 0)),
            }
        )
    return rows


def _slice_signals(signals: list[dict], start: int, end: int) -> list[dict]:
    sliced: list[dict] = []
    for row in signals:
        band = np.asarray(row.get("band", []), dtype=np.float64).reshape(-1)
        if end > band.size:
            continue
        new = dict(row)
        new["band"] = band[start:end].copy()
        new["n"] = int(end - start)
        new["duration_sec"] = float(end - start) / max(float(row.get("fs", 1.0)), 1e-12)
        sliced.append(new)
    return sliced


def _edge_rows(signals: list[dict], args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    usable = [row for row in signals if method_group(str(row.get("method", ""))) is not None]
    if not usable:
        return rows
    fs = _safe_median(float(row.get("fs", float("nan"))) for row in usable)
    if not np.isfinite(fs):
        return rows
    for candidate in usable:
        c_group = method_group(str(candidate["method"]))
        if c_group is None:
            continue
        for anchor in usable:
            if str(candidate["method"]) == str(anchor["method"]):
                continue
            a_group = method_group(str(anchor["method"]))
            if a_group is None:
                continue
            eq = _relative_observation_equation(
                source=np.asarray(candidate["band"], dtype=np.float64).reshape(-1),
                target=np.asarray(anchor["band"], dtype=np.float64).reshape(-1),
                fs=float(fs),
                max_lag_sec=float(args.max_lag_sec),
            )
            if not eq:
                continue
            dom_diff_bpm = (
                abs(float(candidate["dom_hz"]) - float(anchor["dom_hz"])) * 60.0
                if np.isfinite(candidate["dom_hz"]) and np.isfinite(anchor["dom_hz"])
                else float("nan")
            )
            track_diff_bpm = (
                abs(float(candidate["track_median_hz"]) - float(anchor["track_median_hz"])) * 60.0
                if np.isfinite(candidate["track_median_hz"]) and np.isfinite(anchor["track_median_hz"])
                else float("nan")
            )
            rate_score = _rate_score_from_bpm(dom_diff_bpm)
            harmonic = _harmonic_ambiguous(float(candidate["dom_hz"]), float(anchor["dom_hz"]))
            harmonic_guard = 1.0 - harmonic if np.isfinite(harmonic) else float("nan")
            bounded_support = (
                float(eq.get("abs_corr", float("nan"))) >= float(args.min_support_corr)
                and float(eq.get("residual_norm", float("nan"))) <= float(args.max_support_residual)
            )
            rows.append(
                {
                    "dataset": candidate["dataset"],
                    "video": candidate["video"],
                    "candidate_output": str(args.candidate_output),
                    "candidate_method": candidate["method"],
                    "candidate_base_method": _canonical_base(str(candidate["method"])),
                    "candidate_group": c_group,
                    "candidate_group_label": GROUP_LABELS[c_group],
                    "candidate_stage": candidate.get("stage", ""),
                    "candidate_spectral_score": float(candidate["spectral_score"]),
                    "candidate_dom_hz": float(candidate["dom_hz"]),
                    "candidate_track_median_hz": float(candidate["track_median_hz"]),
                    "anchor_method": anchor["method"],
                    "anchor_base_method": _canonical_base(str(anchor["method"])),
                    "anchor_group": a_group,
                    "candidate_macro_family": _macro_family(c_group),
                    "anchor_macro_family": _macro_family(a_group),
                    "anchor_group_label": GROUP_LABELS[a_group],
                    "anchor_stage": anchor.get("stage", ""),
                    "anchor_spectral_score": float(anchor["spectral_score"]),
                    "anchor_dom_hz": float(anchor["dom_hz"]),
                    "anchor_track_median_hz": float(anchor["track_median_hz"]),
                    "same_group": bool(c_group == a_group),
                    "dom_diff_bpm": dom_diff_bpm,
                    "track_diff_bpm": track_diff_bpm,
                    "rate_agreement_score": rate_score,
                    "harmonic_ambiguous": harmonic,
                    "harmonic_guard_score": harmonic_guard,
                    "bounded_support": bool(bounded_support),
                    **eq,
                }
            )
    return rows


def _group_rows_for_trial(
    *,
    dataset: str,
    video: str,
    signal_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    meta: dict[str, float | str],
) -> list[dict]:
    rows: list[dict] = []
    if signal_df.empty:
        return rows
    possible_groups = [g for g in GROUP_ORDER if g in set(signal_df["group"])]
    for group in possible_groups:
        sig_sub = signal_df[signal_df["group"] == group].copy()
        edge_sub = edge_df[edge_df["candidate_group"] == group].copy() if not edge_df.empty else pd.DataFrame()
        cross = edge_sub[edge_sub["anchor_group"] != group].copy() if not edge_sub.empty else pd.DataFrame()
        same = edge_sub[edge_sub["anchor_group"] == group].copy() if not edge_sub.empty else pd.DataFrame()

        if not cross.empty:
            by_anchor = (
                cross.groupby("anchor_group", as_index=False)
                .agg(
                    median_pair_score=("pair_score", "median"),
                    median_abs_corr=("abs_corr", "median"),
                    median_residual_norm=("residual_norm", "median"),
                    median_abs_lag_sec=("abs_lag_sec", "median"),
                    median_rate_agreement=("rate_agreement_score", "median"),
                    harmonic_fraction=("harmonic_ambiguous", "mean"),
                    bounded_support=("bounded_support", "max"),
                )
                .reset_index(drop=True)
            )
        else:
            by_anchor = pd.DataFrame()

        support_group_count = int(by_anchor["bounded_support"].sum()) if not by_anchor.empty else 0
        possible_support_groups = max(len([g for g in possible_groups if g != group]), 1)
        support_diversity = float(support_group_count / possible_support_groups)
        cross_pair_score = _safe_median(by_anchor["median_pair_score"]) if not by_anchor.empty else float("nan")
        cross_abs_corr = _safe_median(by_anchor["median_abs_corr"]) if not by_anchor.empty else float("nan")
        cross_residual = _safe_median(by_anchor["median_residual_norm"]) if not by_anchor.empty else float("nan")
        cross_abs_lag = _safe_median(by_anchor["median_abs_lag_sec"]) if not by_anchor.empty else float("nan")
        rate_phase = _safe_median(by_anchor["median_rate_agreement"]) if not by_anchor.empty else float("nan")
        harmonic_fraction = _safe_median(by_anchor["harmonic_fraction"]) if not by_anchor.empty else float("nan")
        harmonic_guard = 1.0 - harmonic_fraction if np.isfinite(harmonic_fraction) else float("nan")
        macro_cross = (
            edge_sub[edge_sub["anchor_macro_family"] != _macro_family(group)].copy()
            if not edge_sub.empty and "anchor_macro_family" in edge_sub.columns
            else pd.DataFrame()
        )
        if not macro_cross.empty:
            by_macro = (
                macro_cross.groupby("anchor_macro_family", as_index=False)
                .agg(
                    median_pair_score=("pair_score", "median"),
                    median_rate_agreement=("rate_agreement_score", "median"),
                    bounded_support=("bounded_support", "max"),
                )
                .reset_index(drop=True)
            )
            macro_support_count = int(by_macro["bounded_support"].sum())
            macro_possible = max(
                len({ _macro_family(g) for g in possible_groups if _macro_family(g) != _macro_family(group) }),
                1,
            )
            macro_support_diversity = float(macro_support_count / macro_possible)
            macro_pair = _safe_median(by_macro["median_pair_score"])
            macro_rate = _safe_median(by_macro["median_rate_agreement"])
        else:
            macro_support_count = 0
            macro_support_diversity = float("nan")
            macro_pair = float("nan")
            macro_rate = float("nan")
        macro_timing_support = _geometric_mean([macro_support_diversity, macro_pair, macro_rate])
        bridge_groups = BRIDGE_COUNTERPARTS.get(group, set())
        bridge_cross = edge_sub[edge_sub["anchor_group"].isin(bridge_groups)].copy() if not edge_sub.empty and bridge_groups else pd.DataFrame()
        if not bridge_cross.empty:
            bridge_pair = _safe_median(bridge_cross["pair_score"])
            bridge_rate = _safe_median(bridge_cross["rate_agreement_score"])
            bridge_lag = _decay_score(_safe_median(bridge_cross["abs_lag_sec"]), 2.0)
            bridge_harmonic = 1.0 - _safe_median(bridge_cross["harmonic_ambiguous"])
            bridge_timing = _geometric_mean([bridge_pair, bridge_rate, bridge_lag, bridge_harmonic])
        else:
            bridge_timing = float("nan")
        event_timing = _safe_median(sig_sub["event_timing_score"]) if "event_timing_score" in sig_sub.columns else float("nan")
        spectral = _safe_median(sig_sub["spectral_score"])
        stage_invariance = _safe_median(same["pair_score"]) if not same.empty else spectral
        lag_score = _decay_score(cross_abs_lag, 2.0)
        residual_score = _decay_score(cross_residual, 1.0)
        group_support = _geometric_mean([support_diversity, cross_pair_score])
        morphology_support = _geometric_mean([cross_abs_corr, residual_score])
        rate_phase_support = _geometric_mean([rate_phase, lag_score])
        preprocessing_invariance = _clip01(stage_invariance)
        nuisance_penalty = _geometric_mean([residual_score, lag_score])
        roi_observability = float(meta.get("roi_observability_score", float("nan")))
        reliability = _geometric_mean(
            [
                group_support,
                rate_phase_support,
                morphology_support,
                preprocessing_invariance,
                roi_observability,
                nuisance_penalty,
                harmonic_guard,
            ]
        )
        timing_reliability = _geometric_mean(
            [
                group_support,
                rate_phase_support,
                spectral,
                nuisance_penalty,
                harmonic_guard,
                roi_observability,
            ]
        )
        timing_reliability_v3 = _geometric_mean(
            [
                macro_timing_support,
                bridge_timing,
                event_timing,
                rate_phase_support,
                spectral,
                nuisance_penalty,
                harmonic_guard,
                roi_observability,
            ]
        )
        morphology_reliability = _geometric_mean(
            [
                group_support,
                morphology_support,
                preprocessing_invariance,
                nuisance_penalty,
                roi_observability,
            ]
        )
        evidence_strength = max(
            timing_reliability_v3 if np.isfinite(timing_reliability_v3) else 0.0,
            morphology_reliability if np.isfinite(morphology_reliability) else 0.0,
        )
        harmonic_risk = harmonic_fraction if np.isfinite(harmonic_fraction) else 0.0
        abstain_score = _clip01(max(1.0 - evidence_strength, 0.50 * harmonic_risk))
        role_weights = _component_role_weights(
            group,
            timing_reliability_v3 if np.isfinite(timing_reliability_v3) else timing_reliability,
            morphology_reliability,
            abstain_score,
        )
        rows.append(
            {
                "dataset": dataset,
                "video": video,
                "group": group,
                "group_label": GROUP_LABELS[group],
                "reliability_score": reliability,
                "timing_reliability_score": timing_reliability,
                "timing_reliability_score_v3": timing_reliability_v3,
                "morphology_reliability_score": morphology_reliability,
                "event_timing_score": event_timing,
                "macro_timing_support_score": macro_timing_support,
                "bridge_timing_score": bridge_timing,
                "harmonic_risk_score": harmonic_risk,
                "abstain_score": abstain_score,
                "group_support_score": group_support,
                "macro_support_group_count": macro_support_count,
                "rate_phase_support_score": rate_phase_support,
                "morphology_support_score": morphology_support,
                "preprocessing_invariance_score": preprocessing_invariance,
                "preprocessing_invariance_source": "method_stage_proxy",
                "roi_observability_score": roi_observability,
                "roi_observability_source": str(meta.get("roi_observability_source", "")),
                "nuisance_penalty_score": nuisance_penalty,
                "harmonic_guard_score": harmonic_guard,
                "harmonic_ambiguity_fraction": harmonic_fraction,
                "cross_pair_score_median": cross_pair_score,
                "cross_abs_corr_median": cross_abs_corr,
                "cross_residual_norm_median": cross_residual,
                "cross_abs_lag_sec_median": cross_abs_lag,
                "spectral_score_median": spectral,
                "same_group_stage_pair_score_median": _safe_median(same["pair_score"]) if not same.empty else float("nan"),
                "support_group_count": support_group_count,
                "possible_support_group_count": possible_support_groups,
                "method_count": int(sig_sub["method"].nunique()),
                "edge_count": int(len(edge_sub)),
                "cross_edge_count": int(len(cross)),
                "native_fps": float(meta.get("native_fps", float("nan"))),
                "duration_sec": float(meta.get("duration_sec", float("nan"))),
                **role_weights,
            }
        )
    total = sum(float(r["reliability_score"]) for r in rows if np.isfinite(float(r["reliability_score"])))
    for row in rows:
        score = float(row["reliability_score"])
        row["soft_group_weight"] = float(score / total) if total > 0.0 and np.isfinite(score) else float("nan")
    return rows


def _rows_for_trial(task: tuple[Path, str, list[str], argparse.Namespace]) -> tuple[list[dict], list[dict], list[dict]]:
    path, dataset, methods, args = task
    signals, _ = _load_trial_signals(path, dataset=dataset, methods=methods, args=args)
    signals = [row for row in signals if method_group(str(row.get("method", ""))) is not None]
    all_edge_rows: list[dict] = []
    all_group_rows: list[dict] = []
    all_signal_rows: list[dict] = []
    fs = _safe_median(float(row.get("fs", float("nan"))) for row in signals)
    n = min((np.asarray(row.get("band", [])).size for row in signals), default=0)
    window_sec = float(getattr(args, "window_sec", 0.0) or 0.0)
    if window_sec > 0.0 and np.isfinite(fs) and n > 0:
        win = max(16, int(round(window_sec * fs)))
        stride_sec = float(getattr(args, "window_stride_sec", 0.0) or 0.0)
        if stride_sec <= 0.0:
            stride_sec = max(window_sec * 0.5, 1.0 / max(fs, 1e-12))
        stride = max(1, int(round(stride_sec * fs)))
        starts = list(range(0, max(n - win + 1, 1), stride))
        if starts and starts[-1] != max(n - win, 0):
            starts.append(max(n - win, 0))
        for window_id, start in enumerate(starts):
            end = min(start + win, n)
            if end - start < 16:
                continue
            sub_signals = _slice_signals(signals, start, end)
            if not sub_signals:
                continue
            signal_rows = _signal_rows(
                sub_signals,
                min_hz=float(args.min_hz),
                max_hz=float(args.max_hz),
            )
            edge_rows = _edge_rows(sub_signals, args)
            meta = _trial_observability(path, sub_signals)
            signal_df = pd.DataFrame(signal_rows)
            edge_df = pd.DataFrame(edge_rows)
            group_rows = _group_rows_for_trial(
                dataset=dataset,
                video=path.stem,
                signal_df=signal_df,
                edge_df=edge_df,
                meta=meta,
            )
            window_meta = {
                "window_id": int(window_id),
                "window_start_sec": float(start) / float(fs),
                "window_end_sec": float(end) / float(fs),
                "window_duration_sec": float(end - start) / float(fs),
            }
            for rows in (edge_rows, group_rows, signal_rows):
                for row in rows:
                    row.update(window_meta)
            all_edge_rows.extend(edge_rows)
            all_group_rows.extend(group_rows)
            all_signal_rows.extend(signal_rows)
        return all_edge_rows, all_group_rows, all_signal_rows

    signal_rows = _signal_rows(
        signals,
        min_hz=float(args.min_hz),
        max_hz=float(args.max_hz),
    )
    edge_rows = _edge_rows(signals, args)
    signal_df = pd.DataFrame(signal_rows)
    edge_df = pd.DataFrame(edge_rows)
    meta = _trial_observability(path, signals)
    group_rows = _group_rows_for_trial(
        dataset=dataset,
        video=path.stem,
        signal_df=signal_df,
        edge_df=edge_df,
        meta=meta,
    )
    return edge_rows, group_rows, signal_rows


def _summarize_groups(group_df: pd.DataFrame) -> pd.DataFrame:
    if group_df.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for (dataset, group), sub in group_df.groupby(["dataset", "group"]):
        top = group_df.loc[group_df["dataset"] == dataset].copy()
        top = top.sort_values(["video", "reliability_score"], ascending=[True, False])
        top = top.groupby("video", as_index=False).head(1)
        top_share = float((top["group"] == group).mean()) if not top.empty else float("nan")
        rows.append(
            {
                "dataset": dataset,
                "group": group,
                "group_label": GROUP_LABELS.get(group, group),
                "median_reliability_score": _safe_median(sub["reliability_score"]),
                "median_timing_reliability_score": _safe_median(sub["timing_reliability_score"]),
                "median_timing_reliability_score_v3": _safe_median(sub["timing_reliability_score_v3"]),
                "median_morphology_reliability_score": _safe_median(sub["morphology_reliability_score"]),
                "median_event_timing_score": _safe_median(sub["event_timing_score"]),
                "median_macro_timing_support_score": _safe_median(sub["macro_timing_support_score"]),
                "median_bridge_timing_score": _safe_median(sub["bridge_timing_score"]),
                "median_abstain_score": _safe_median(sub["abstain_score"]),
                "median_soft_group_weight": _safe_median(sub["soft_group_weight"]),
                "top_group_trial_share": top_share,
                "median_group_support_score": _safe_median(sub["group_support_score"]),
                "median_rate_phase_support_score": _safe_median(sub["rate_phase_support_score"]),
                "median_morphology_support_score": _safe_median(sub["morphology_support_score"]),
                "median_preprocessing_invariance_score": _safe_median(sub["preprocessing_invariance_score"]),
                "median_roi_observability_score": _safe_median(sub["roi_observability_score"]),
                "median_nuisance_penalty_score": _safe_median(sub["nuisance_penalty_score"]),
                "median_harmonic_guard_score": _safe_median(sub["harmonic_guard_score"]),
                "median_cross_abs_corr": _safe_median(sub["cross_abs_corr_median"]),
                "median_cross_residual_norm": _safe_median(sub["cross_residual_norm_median"]),
                "median_cross_abs_lag_sec": _safe_median(sub["cross_abs_lag_sec_median"]),
                "median_support_group_count": _safe_median(sub["support_group_count"]),
                "n_trials": int(sub["video"].nunique()),
                "n_windows": int(sub["window_id"].nunique()) if "window_id" in sub.columns else int(sub["video"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["dataset", "median_reliability_score"], ascending=[True, False]).reset_index(drop=True)


def _write_report(path: Path, args: argparse.Namespace, group_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Target-Side Reliability Graph Features",
        "",
        "This is a GT-free reliability diagnostic for the final observation-state design. It does not select candidates using target respiration labels and it does not claim performance.",
        "",
        f"- candidate output: `{args.candidate_output}`",
        f"- respiratory band: `{args.min_hz}`-`{args.max_hz}` Hz",
        f"- bounded lag: `{args.max_lag_sec}` sec",
        f"- feature fs: `{args.feature_fs}` Hz",
        f"- window sec: `{args.window_sec}`",
        f"- window stride sec: `{args.window_stride_sec}`",
        f"- edge CSV: `{args.out_edge.resolve()}`",
        f"- group CSV: `{args.out_group.resolve()}`",
        f"- summary CSV: `{args.out_summary.resolve()}`",
        "",
        "## Reliability Factorization",
        "",
        "`reliability = geometric_mean(group_support, rate_phase_support, morphology_support, preprocessing_invariance, roi_observability, nuisance_penalty, harmonic_guard)`",
        "",
        "`timing_reliability_score_v3 = geometric_mean(macro_timing_support, bridge_timing, event_timing, rate_phase_support, spectral, nuisance_penalty, harmonic_guard, roi_observability)`",
        "",
        "`morphology_reliability_score = geometric_mean(group_support, morphology_support, preprocessing_invariance, nuisance_penalty, roi_observability)`",
        "",
        "Same-group variants are treated as stage/proxy invariance, not as independent cross-family support.",
        "For timing, P1D variants are one macro-family; P1D-P1D agreement alone is not enough to claim respiratory timing evidence.",
        "",
        "## Dataset Summary",
        "",
    ]
    if not summary_df.empty:
        keep = [
            "dataset",
            "group_label",
            "median_reliability_score",
            "median_timing_reliability_score_v3",
            "median_morphology_reliability_score",
            "median_event_timing_score",
            "median_abstain_score",
            "median_soft_group_weight",
            "top_group_trial_share",
            "median_group_support_score",
            "median_rate_phase_support_score",
            "median_morphology_support_score",
            "median_harmonic_guard_score",
            "n_trials",
        ]
        if "n_windows" in summary_df.columns:
            keep.append("n_windows")
        lines.extend(["```csv", summary_df[keep].to_csv(index=False).strip(), "```"])
    lines.extend(["", "## Per-Dataset Top-Group Mix", ""])
    if not group_df.empty:
        top = group_df.sort_values(["dataset", "video", "reliability_score"], ascending=[True, True, False])
        top = top.groupby(["dataset", "video"], as_index=False).head(1)
        mix = (
            top.groupby(["dataset", "group_label"], as_index=False)
            .agg(trials=("video", "nunique"), median_score=("reliability_score", "median"))
            .sort_values(["dataset", "trials", "median_score"], ascending=[True, False, False])
        )
        lines.extend(["```csv", mix.to_csv(index=False).strip(), "```"])
    lines.extend(
        [
            "",
            "## Interpretation Guard",
            "",
            "- A high score means bounded cross-family agreement plus clean residual structure, not truth.",
            "- A low score can mean weak ROI observability, harmonic ambiguity, inconsistent preprocessing/stage behavior, or true target mismatch.",
            "- This artifact is the required bridge before injecting `pi_m,t`, `R_eff_m,t`, and helper cues into PARH-OSSM.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    labels = list(args.dataset_label)
    if labels and len(labels) != len(args.data_dir):
        raise SystemExit("--dataset-label must be repeated exactly once per --data-dir")
    methods = args.method or DEFAULT_METHODS
    tasks: list[tuple[Path, str, list[str], argparse.Namespace]] = []
    for i, data_dir in enumerate(args.data_dir):
        dataset = labels[i] if labels else data_dir.parent.name
        files = sorted(Path(data_dir).glob("*.pkl"))
        if args.max_files is not None:
            files = files[: int(args.max_files)]
        for path in files:
            tasks.append((path, dataset, methods, args))

    edge_rows: list[dict] = []
    group_rows: list[dict] = []
    signal_rows: list[dict] = []
    jobs = max(1, int(args.jobs))
    if jobs == 1 or len(tasks) <= 1:
        for task in tasks:
            e, g, s = _rows_for_trial(task)
            edge_rows.extend(e)
            group_rows.extend(g)
            signal_rows.extend(s)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for e, g, s in pool.map(_rows_for_trial, tasks, chunksize=1):
                edge_rows.extend(e)
                group_rows.extend(g)
                signal_rows.extend(s)

    edge_df = pd.DataFrame(edge_rows)
    group_df = pd.DataFrame(group_rows)
    summary_df = _summarize_groups(group_df)

    for path, frame in [
        (args.out_edge, edge_df),
        (args.out_group, group_df),
        (args.out_summary, summary_df),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    if args.report_out:
        _write_report(args.report_out, args, group_df, summary_df)

    print(f"Wrote {args.out_edge} ({len(edge_df)} edges)")
    print(f"Wrote {args.out_group} ({len(group_df)} group rows)")
    print(f"Wrote {args.out_summary} ({len(summary_df)} summary rows)")
    if args.report_out:
        print(f"Wrote {args.report_out}")
    if not summary_df.empty:
        print(summary_df.groupby("dataset").head(7).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
