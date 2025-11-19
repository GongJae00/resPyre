#!/usr/bin/env python3
"""
Generate family-level respiration overlays for COHFACE results.

Each figure shows:
  - GT vs baseline method (top/mid/bottom trials) on the upper-left axes.
  - GT vs all methods in the family (best/top trial) on the upper-right axes.
  - GT vs each tracker head (KFstd, UKF-Freq, Spec-Ridge, PLL) across
    best/middle/worst trials on the bottom four axes.

Usage:
    python tools/plot_family_overlays.py \
        --results results/cohface_motion_oscillator
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


FAMILIES = {
    "of_farneback": [
        "of_farneback",
        "of_farneback__kfstd",
        "of_farneback__ukffreq",
        "of_farneback__spec_ridge",
        "of_farneback__pll",
    ],
    "dof": [
        "dof",
        "dof__kfstd",
        "dof__ukffreq",
        "dof__spec_ridge",
        "dof__pll",
    ],
    "profile1d_linear": [
        "profile1d_linear",
        "profile1d_linear__kfstd",
        "profile1d_linear__ukffreq",
        "profile1d_linear__spec_ridge",
        "profile1d_linear__pll",
    ],
    "profile1d_quadratic": [
        "profile1d_quadratic",
        "profile1d_quadratic__kfstd",
        "profile1d_quadratic__ukffreq",
        "profile1d_quadratic__spec_ridge",
        "profile1d_quadratic__pll",
    ],
    "profile1d_cubic": [
        "profile1d_cubic",
        "profile1d_cubic__kfstd",
        "profile1d_cubic__ukffreq",
        "profile1d_cubic__spec_ridge",
        "profile1d_cubic__pll",
    ],
}

TRACKER_SUFFIXES = ["__kfstd", "__ukffreq", "__spec_ridge", "__pll"]
CATEGORY_COLORS = {"top": "#1b9e77", "mid": "#d95f02"}
CATEGORY_COLORS = {"top": "#1b9e77"}
METHOD_COLORS = {
    "__kfstd": "#1b9e77",
    "__ukffreq": "#2c7bb6",
    "__spec_ridge": "#7570b3",
    "__pll": "#d95f02",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create family overlay plots from resPyre evaluation outputs."
    )
    parser.add_argument(
        "--results",
        default="results/cohface_motion_oscillator",
        help="Run directory containing data/aux/logs (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store figures (default: <results>/plots/family_overlays)",
    )
    parser.add_argument(
        "--families",
        nargs="*",
        choices=list(FAMILIES.keys()),
        help="Optional subset of families to render.",
    )
    parser.add_argument(
        "--metric",
        default="MAE",
        help="Metric column used to rank trials (default: %(default)s)",
    )
    return parser.parse_args()


def load_metric_entries(metrics_path: Path, metric: str) -> Dict[str, List[Dict]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics pkl missing: {metrics_path}")
    with metrics_path.open("rb") as fp:
        metric_names, method_metrics = pickle.load(fp)
    try:
        idx = metric_names.index(metric)
    except ValueError:
        raise ValueError(f"metric '{metric}' not found in metrics list {metric_names}")
    entries: Dict[str, List[Dict]] = {}
    for method, records in method_metrics.items():
        method_entries: List[Dict] = []
        for rec in records:
            data_file = rec.get("data_file")
            if not data_file:
                continue
            metrics_list = rec.get("metrics") or []
            if idx >= len(metrics_list):
                continue
            try:
                score = float(metrics_list[idx])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(score):
                continue
            method_entries.append(
                {
                    "score": score,
                    "data_file": data_file,
                    "trial": rec.get("trial_key") or "",
                }
            )
        if method_entries:
            entries[method] = method_entries
    return entries


def pick_top(records: List[Dict]) -> Dict[str, Dict]:
    if not records:
        return {}
    ordered = sorted(records, key=lambda item: item["score"])
    return {"top": ordered[0]}


_trial_cache: Dict[str, Dict] = {}


def load_trial(results_root: Path, rel_path: str) -> Dict:
    key = rel_path.replace("\\", "/")
    if key in _trial_cache:
        return _trial_cache[key]
    full = results_root / rel_path
    if not full.exists():
        raise FileNotFoundError(f"trial data not found: {full}")
    with full.open("rb") as fp:
        payload = pickle.load(fp)
    trial = {
        "fps": float(payload.get("fps") or 1.0),
        "gt": np.asarray(payload.get("gt"), dtype=np.float64).reshape(-1),
        "methods": {},
    }
    for est in payload.get("estimates", []):
        name = est.get("method")
        if name is None:
            continue
        trial["methods"][name] = np.asarray(est.get("estimate"), dtype=np.float64).reshape(-1)
    _trial_cache[key] = trial
    return trial


def normalize_to_unit(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-9:
        return np.zeros_like(arr)
    return 2.0 * (arr - mn) / (mx - mn) - 1.0


def extract_pair(trial: Dict, method: str):
    gt = trial["gt"]
    wave = trial["methods"].get(method)
    if wave is None or gt.size == 0:
        return None
    n = min(len(gt), len(wave))
    limit = int(round(30.0 * max(trial["fps"], 1e-6)))
    if limit > 1:
        n = min(n, limit)
    if n < 2:
        return None
    t = np.arange(n, dtype=np.float64) / max(trial["fps"], 1e-6)
    gt_norm = normalize_to_unit(gt[:n])
    wave_norm = normalize_to_unit(wave[:n])
    return t, gt_norm, wave_norm


def plot_family(
    family: str,
    methods: List[str],
    picks: Dict[str, Dict],
    results_root: Path,
    output_dir: Path,
):
    if not picks:
        print(f"[warn] family {family}: insufficient quality entries, skipped")
        return
    base_method = methods[0]
    best_entry = picks["top"]
    trial_ids = {k: (Path(v["data_file"]).name, v.get("trial", "")) for k, v in picks.items()}

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(
        3,
        2,
        height_ratios=[1.2, 1.0, 1.0],
        figure=fig,
        hspace=0.35,
        wspace=0.25,
    )

    ax_base = fig.add_subplot(gs[0, 0])
    ax_family = fig.add_subplot(gs[0, 1])
    tracker_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    # Upper-left: baseline vs GT (top only)
    for label, entry in picks.items():
        trial = load_trial(results_root, entry["data_file"])
        pair = extract_pair(trial, base_method)
        if pair is None:
            continue
        t, gt_norm, base_norm = pair
        ax_base.plot(t, gt_norm, color="#111111", linestyle="--", linewidth=2.0, alpha=0.85)
        ax_base.plot(
            t,
            base_norm,
            color=CATEGORY_COLORS[label],
            label=f"{label} ({trial_ids[label][0]})",
            linewidth=1.4,
        )
    ax_base.set_title(f"{base_method} vs GT (min/max normalized)")
    ax_base.set_xlabel("Time (s)")
    ax_base.set_ylabel("Normalized amplitude")
    ax_base.legend(loc="upper right", fontsize=9)

    # Upper-right: GT + all methods for best/top trial only
    trial = load_trial(results_root, best_entry["data_file"])
    pair_gt = extract_pair(trial, base_method)
    if pair_gt is not None:
        t_ref, gt_norm, _ = pair_gt
        ax_family.plot(
            t_ref,
            gt_norm,
            color="#000000",
            linestyle="--",
            linewidth=2.0,
            label="GT",
        )
    for method in methods:
        pair = extract_pair(trial, method)
        if pair is None:
            continue
        t, _, wave_norm = pair
        color = METHOD_COLORS.get(method.replace(base_method, ""), None)
        ax_family.plot(t, wave_norm, label=method, linewidth=1.2, color=color)
    ax_family.set_title(f"{family} methods (best trial: {trial_ids['top'][0]})")
    ax_family.set_xlabel("Time (s)")
    ax_family.set_ylabel("Normalized amplitude")
    ax_family.legend(fontsize=8, loc="upper right")

    # Tracker rows: KFSTD/UKFFREQ (row2), SPEC/PLL (row3)
    for ax, suffix in zip(tracker_axes, TRACKER_SUFFIXES):
        method = f"{base_method}{suffix}"
        if method not in trial["methods"]:
            ax.set_visible(False)
            continue
        for label, entry in picks.items():
            trial = load_trial(results_root, entry["data_file"])
            pair = extract_pair(trial, method)
            if pair is None:
                continue
            t, gt_norm, wave_norm = pair
            ax.plot(t, gt_norm, color="#111111", linestyle="--", linewidth=1.2, alpha=0.6)
            ax.plot(
                t,
                wave_norm,
                color=CATEGORY_COLORS[label],
                linewidth=1.3,
            )
        label_suffix = method.replace(base_method + "__", "").upper()
        ax.set_title(label_suffix)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Norm amp")
        if method == f"{base_method}{TRACKER_SUFFIXES[0]}":
            ax.legend([f"{k} ({trial_ids[k][0]})" for k in picks], fontsize=8, loc="upper right")

    trial_summary = ", ".join(f"{k}:{trial_ids[k][0]}" for k in trial_ids)
    fig.suptitle(f"{family} overlay (top trial: {trial_summary})", fontsize=14)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{family}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[info] saved {out_path}")


def main():
    args = parse_args()
    results_root = Path(args.results).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else results_root / "plots" / "family_overlays"
    )

    metrics_pkl = results_root / "metrics" / "metrics.pkl"
    metric_map = load_metric_entries(metrics_pkl, args.metric)

    families = args.families or list(FAMILIES.keys())
    for family in families:
        methods = FAMILIES[family]
        base = methods[0]
        records = metric_map.get(base)
        if not records:
            print(f"[warn] no quality records for {base}; skipping {family}")
            continue
        picks = pick_top(records)
        plot_family(family, methods, picks, results_root, output_dir)


if __name__ == "__main__":
    main()
