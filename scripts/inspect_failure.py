#!/usr/bin/env python3
"""
Inspect QROBF frame logs for one method/trial.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _find_log(run_dir: str, method: str | None, trial_key: str | None) -> str:
    if method and trial_key:
        path = os.path.join(run_dir, "aux", method.replace(" ", "_"), "frame_logs", f"{trial_key}.npz")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(path)

    candidates = sorted(glob.glob(os.path.join(run_dir, "aux", "*", "frame_logs", "*.npz")))
    if not candidates:
        raise FileNotFoundError(f"No frame logs under {run_dir}/aux/*/frame_logs")
    return candidates[0]


def _col(arr, idx, name, default=np.nan):
    if name not in idx:
        return np.full(arr.shape[0], default, dtype=np.float64)
    return np.asarray(arr[:, idx[name]], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/cohface_robust_ossm")
    ap.add_argument("--method", default=None)
    ap.add_argument("--trial-key", default=None)
    args = ap.parse_args()

    path = _find_log(args.run_dir, args.method, args.trial_key)
    z = np.load(path, allow_pickle=True)
    fields = list(z["fields"])
    arr = z["data"]
    idx = {f: i for i, f in enumerate(fields)}

    method = os.path.basename(os.path.dirname(os.path.dirname(path)))
    trial = os.path.splitext(os.path.basename(path))[0]
    print(f"[Log] {path}")
    print(f"[Method] {method} | [Trial] {trial} | [Frames] {arr.shape[0]}")

    lambda_t = _col(arr, idx, "lambda_t")
    nis = _col(arr, idx, "nis")
    g_t = _col(arr, idx, "g_t")
    alpha_R = _col(arr, idx, "alpha_R")
    q_vis = _col(arr, idx, "q_vis")
    q_drift = _col(arr, idx, "q_drift")
    fail_div = _col(arr, idx, "fail_diverge", 0.0)
    fail_slip = _col(arr, idx, "fail_slip", 0.0)
    fail_lock = _col(arr, idx, "fail_lock", 0.0)
    fail_double = _col(arr, idx, "fail_double", 0.0)
    fail_total = np.maximum.reduce([fail_div, fail_slip, fail_lock, fail_double])

    def stat(name, x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            print(f"{name:16s}: nan")
            return
        print(f"{name:16s}: mean={np.mean(x):.4f} std={np.std(x):.4f} min={np.min(x):.4f} max={np.max(x):.4f}")

    print("\n=== Core Stats ===")
    stat("lambda_t", lambda_t)
    stat("nis", nis)
    stat("g_t", g_t)
    stat("alpha_R", alpha_R)
    stat("q_vis", q_vis)
    stat("q_drift", q_drift)
    stat("fail_total", fail_total)

    print("\n=== Failure Rates ===")
    for name, col in [
        ("fail_diverge", fail_div),
        ("fail_slip", fail_slip),
        ("fail_lock", fail_lock),
        ("fail_double", fail_double),
    ]:
        print(f"{name:16s}: {float(np.nanmean(col)):.4f}")

    low_lambda_idx = np.where(lambda_t < 0.5)[0]
    if low_lambda_idx.size:
        print(f"\nFrames with lambda<0.5: {low_lambda_idx.size} (first 10: {low_lambda_idx[:10].tolist()})")
    else:
        print("\nFrames with lambda<0.5: 0")


if __name__ == "__main__":
    main()

