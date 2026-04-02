#!/usr/bin/env python3
"""Precompute obs_roi_stats_v1.npz for all MAHNOB-HCI trials.

Usage:
    python scripts/precompute_mahnob_roi.py [--workers N]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm

from core.utils.common import get_chest_ROI, sort_nicely
from core.pipeline.wrapped_method import (
    compute_roi_stats_time_series,
    save_roi_stats_cache,
    _roi_stats_cache_path,
)


def process_one(video_path: str, mp_complexity: int = 1, skip_rate: int = 10) -> bool:
    """Extract chest ROIs and save stats cache for a single trial."""
    cache_path = _roi_stats_cache_path(video_path)
    if cache_path and os.path.exists(cache_path):
        return True  # already cached

    try:
        frames, fps, _ = get_chest_ROI(
            video_path, "mahnob",
            mp_complexity=mp_complexity,
            skip_rate=skip_rate,
        )
        if not frames:
            print(f"  [WARN] No frames extracted: {video_path}")
            return False

        stats_t, _, _, _ = compute_roi_stats_time_series(frames)
        if not stats_t:
            print(f"  [WARN] No stats computed: {video_path}")
            return False

        saved = save_roi_stats_cache(video_path, fps, stats_t)
        if saved:
            return True
        else:
            print(f"  [WARN] Save failed: {video_path}")
            return False
    except Exception as e:
        print(f"  [ERROR] {video_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Precompute MAHNOB ROI stats cache")
    parser.add_argument("--mp-complexity", type=int, default=1)
    parser.add_argument("--skip-rate", type=int, default=10)
    args = parser.parse_args()

    # Locate MAHNOB dataset
    from components.datasets.impl import MAHNOB
    ds = MAHNOB()
    base_path = ds.path

    # Enumerate all subject directories
    subjects = sort_nicely([
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ])

    # Collect video paths
    trials = []
    for sub in subjects:
        sub_path = os.path.join(base_path, sub)
        for fn in os.listdir(sub_path):
            if fn.endswith(".avi"):
                trials.append(os.path.join(sub_path, fn))
                break

    # Check which already have cache
    cached = 0
    to_process = []
    for vp in trials:
        cp = _roi_stats_cache_path(vp)
        if cp and os.path.exists(cp):
            cached += 1
        else:
            to_process.append(vp)

    print(f"Total trials: {len(trials)}, Already cached: {cached}, To process: {len(to_process)}")

    if not to_process:
        print("All trials already cached!")
        return

    success = 0
    fail = 0
    for vp in tqdm(to_process, desc="ROI caching"):
        ok = process_one(vp, args.mp_complexity, args.skip_rate)
        if ok:
            success += 1
        else:
            fail += 1

    print(f"\nDone. Success: {success}, Failed: {fail}, Previously cached: {cached}")


if __name__ == "__main__":
    main()
