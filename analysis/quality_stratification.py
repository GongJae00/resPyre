"""Quality Stratification Analysis for COHFACE / QROBF paper.

Splits 160 trials into 4 quality tiers based on spectral SNR of the raw
1D motion signal (OF Farneback) at the GT respiratory frequency, then
compares QROBF vs kfstd per tier to quantify robustness gains.

SNR formula:
    SNR = 10 * log10( P_signal / P_noise )
    P_signal = power in [f_gt ± SNR_BW] Hz
    P_noise  = total respiratory-band power − P_signal
    (respiratory band = [0.08, 0.50] Hz)

Tier splits (on OF SNR):
    Very Poor : bottom 20%  (32 trials)
    Poor      : 20 – 40 %  (32 trials)
    Fair      : 40 – 70 %  (48 trials)
    Good      : top    30%  (48 trials)
"""

from __future__ import annotations

import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RESP_FMIN   = 0.08   # Hz  (4.8 BPM)
RESP_FMAX   = 0.50   # Hz  (30 BPM)
SNR_BW      = 0.05   # Hz  half-bandwidth for signal-power window
EVAL_WIN_S  = 30.0   # seconds – windowed FFT window (matches evaluation pipeline)
EVAL_STRIDE = 1.0    # seconds – stride

TIER_CUTS   = [0.00, 0.20, 0.40, 0.70, 1.00]  # cumulative percentile edges
TIER_LABELS = ["Very Poor", "Poor", "Fair", "Good"]

# (base_method, kfstd_method, qrobf_method, family_label)
METHOD_FAMILIES = [
    ("of_farneback",    "of_farneback__kfstd",    "of_farneback__robust_ossm_ekf",    "OF"),
    ("profile1D cubic", "profile1d_cubic__kfstd",  "profile1d_cubic__robust_ossm_ekf",  "P1D-Cubic"),
    ("profile1D linear","profile1d_linear__kfstd", "profile1d_linear__robust_ossm_ekf", "P1D-Linear"),
    ("profile1D quadratic","profile1d_quadratic__kfstd","profile1d_quadratic__robust_ossm_ekf","P1D-Quad"),
    ("DoF",             "dof__kfstd",              "dof__robust_ossm_ekf",              "DoF"),
]

# SNR is computed from OF Farneback signal (primary stratification signal)
SNR_BASE_METHOD = "of_farneback"


# ─────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────

def compute_gt_freq_hz(gt: np.ndarray, fs: float) -> float:
    """Return dominant frequency (Hz) of the GT respiratory signal via FFT."""
    freqs = np.fft.rfftfreq(len(gt), 1.0 / fs)
    power = np.abs(np.fft.rfft(gt)) ** 2
    mask  = (freqs >= RESP_FMIN) & (freqs <= RESP_FMAX)
    if mask.sum() == 0:
        return np.nan
    return float(freqs[mask][np.argmax(power[mask])])


def compute_snr_db(signal: np.ndarray, fs: float, gt_freq_hz: float,
                   bw: float = SNR_BW) -> float:
    """Spectral SNR of *signal* at *gt_freq_hz* (dB).

    SNR = 10 * log10(P_signal / P_noise)
    where P_signal is power within ±bw of gt_freq_hz,
    P_noise is remaining power in the respiratory band.
    """
    freqs = np.fft.rfftfreq(len(signal), 1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2

    band_mask   = (freqs >= RESP_FMIN) & (freqs <= RESP_FMAX)
    signal_mask = band_mask & (np.abs(freqs - gt_freq_hz) <= bw)

    P_total  = power[band_mask].sum()
    P_signal = power[signal_mask].sum()
    P_noise  = P_total - P_signal

    if P_total == 0 or P_noise <= 0:
        return np.nan
    return float(10.0 * np.log10(P_signal / P_noise))


def _fft_peak_hz(sig: np.ndarray, fs: float) -> float:
    """Dominant frequency in respiratory band via FFT."""
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    power = np.abs(np.fft.rfft(sig)) ** 2
    mask  = (freqs >= RESP_FMIN) & (freqs <= RESP_FMAX)
    if mask.sum() == 0:
        return np.nan
    return float(freqs[mask][np.argmax(power[mask])])


def compute_windowed_freq_mae(signal: np.ndarray, fs: float,
                               gt_signal: np.ndarray, fs_gt: float,
                               win_s: float = EVAL_WIN_S,
                               stride_s: float = EVAL_STRIDE) -> float:
    """Windowed-FFT freq_mae matching the evaluation pipeline.

    Both GT and estimated signal are windowed independently; per-window
    GT peak frequency is used as the reference (not a global mean GT freq).
    Windows are aligned by their start-time fraction of the recording.
    """
    win_n     = int(round(win_s * fs))
    stride_n  = int(round(stride_s * fs))
    win_gt_n  = int(round(win_s * fs_gt))
    stride_gt = int(round(stride_s * fs_gt))

    if len(signal) < win_n or len(gt_signal) < win_gt_n:
        return np.nan

    errors = []
    est_starts = range(0, len(signal)    - win_n    + 1, stride_n)
    gt_starts  = range(0, len(gt_signal) - win_gt_n + 1, stride_gt)
    for s_e, s_g in zip(est_starts, gt_starts):
        seg_e  = signal[s_e: s_e + win_n]
        seg_g  = gt_signal[s_g: s_g + win_gt_n]
        est_hz = _fft_peak_hz(seg_e, fs)
        gt_hz  = _fft_peak_hz(seg_g, fs_gt)
        if np.isnan(est_hz) or np.isnan(gt_hz):
            continue
        errors.append(abs(est_hz * 60.0 - gt_hz * 60.0))

    return float(np.mean(errors)) if errors else np.nan


def compute_windowed_track_hz_mae(track_hz: np.ndarray, fps: float,
                                   gt_rr_bpm: float,
                                   win_s: float = EVAL_WIN_S,
                                   stride_s: float = EVAL_STRIDE) -> float:
    """Windowed freq_mae from KF track_hz (matches evaluation pipeline).

    For each window, takes the median of track_hz values in that window as the
    estimated RR, then computes MAE against GT RR.
    """
    if track_hz is None or len(track_hz) == 0:
        return np.nan
    arr = np.asarray(track_hz, dtype=float)
    if len(arr) == 0:
        return np.nan

    win_n    = int(round(win_s * fps))
    stride_n = int(round(stride_s * fps))
    if len(arr) < win_n:
        # short signal: use full array median
        valid = arr[np.isfinite(arr) & (arr > 0)]
        if len(valid) == 0:
            return np.nan
        return float(abs(np.median(valid) * 60.0 - gt_rr_bpm))

    errors = []
    for start in range(0, len(arr) - win_n + 1, stride_n):
        seg   = arr[start: start + win_n]
        valid = seg[np.isfinite(seg) & (seg > 0)]
        if len(valid) == 0:
            continue
        est_bpm = np.median(valid) * 60.0
        errors.append(abs(est_bpm - gt_rr_bpm))

    return float(np.mean(errors)) if errors else np.nan


def assign_tiers(snr_series: pd.Series) -> pd.Series:
    """Assign tier labels based on SNR percentile cuts."""
    quantiles = [snr_series.quantile(q) for q in TIER_CUTS[1:]]
    def _label(v):
        if np.isnan(v):
            return "Unknown"
        for i, q in enumerate(quantiles):
            if v <= q:
                return TIER_LABELS[i]
        return TIER_LABELS[-1]
    return snr_series.map(_label)


# ─────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────

def load_trial(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def compute_trial_metrics(trial: dict) -> dict:
    """Return per-trial SNR and freq_mae for all method families."""
    gt     = np.asarray(trial["gt"], dtype=float)
    fs_gt  = float(trial.get("fs_gt", 32.0))
    fps    = float(trial.get("fps", 20.0))

    gt_freq_hz  = compute_gt_freq_hz(gt, fs_gt)
    gt_rr_bpm   = gt_freq_hz * 60.0

    # index estimates by method name
    est_map = {e["method"]: e["estimate"] for e in trial["estimates"]}

    # ── SNR from OF base signal ───────────────
    snr_of = np.nan
    if SNR_BASE_METHOD in est_map:
        sig = np.asarray(est_map[SNR_BASE_METHOD]["signal_hat"], dtype=float)
        snr_of = compute_snr_db(sig, fps, gt_freq_hz)

    row: dict = {
        "gt_freq_hz": gt_freq_hz,
        "gt_rr_bpm":  gt_rr_bpm,
        "snr_of_db":  snr_of,
    }

    # ── Per-family metrics ────────────────────
    for base_m, kfstd_m, qrobf_m, label in METHOD_FAMILIES:
        # SNR for this family's base signal
        if base_m in est_map:
            sig = np.asarray(est_map[base_m]["signal_hat"], dtype=float)
            row[f"snr_{label}_db"] = compute_snr_db(sig, fps, gt_freq_hz)
        else:
            row[f"snr_{label}_db"] = np.nan

        # base method freq_mae (windowed FFT, per-window GT)
        if base_m in est_map:
            sig = np.asarray(est_map[base_m]["signal_hat"], dtype=float)
            row[f"freq_mae_{label}_base"] = compute_windowed_freq_mae(
                sig, fps, gt, fs_gt)
        else:
            row[f"freq_mae_{label}_base"] = np.nan

        # kfstd freq_mae (windowed FFT on signal_hat, per-window GT)
        if kfstd_m in est_map:
            sig = np.asarray(est_map[kfstd_m]["signal_hat"], dtype=float)
            row[f"freq_mae_{label}_kfstd"] = compute_windowed_freq_mae(
                sig, fps, gt, fs_gt)
        else:
            row[f"freq_mae_{label}_kfstd"] = np.nan

        # QROBF freq_mae (windowed FFT on signal_hat, per-window GT)
        if qrobf_m in est_map:
            sig = np.asarray(est_map[qrobf_m]["signal_hat"], dtype=float)
            row[f"freq_mae_{label}_qrobf"] = compute_windowed_freq_mae(
                sig, fps, gt, fs_gt)
        else:
            row[f"freq_mae_{label}_qrobf"] = np.nan

    return row


# ─────────────────────────────────────────────
# Run stratification
# ─────────────────────────────────────────────

def run_stratification(results_dir: str,
                        run_label: str = "cohface_robust_ossm",
                        output_dir: Optional[str] = None) -> pd.DataFrame:
    """Compute quality stratification DataFrame and save CSV + JSON summary.

    Parameters
    ----------
    results_dir : str
        Root results directory (e.g. "results").
    run_label : str
        Sub-directory name (e.g. "cohface_robust_ossm").
    output_dir : str, optional
        Where to save outputs. Defaults to
        results_dir/run_label/plots/quality_stratification/.

    Returns
    -------
    pd.DataFrame with one row per trial.
    """
    run_dir  = Path(results_dir) / run_label
    data_dir = run_dir / "data"

    pkl_files = sorted(data_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found in {data_dir}")

    print(f"[qs] Loading {len(pkl_files)} trials from {data_dir} ...")
    rows = []
    for pf in pkl_files:
        trial   = load_trial(pf)
        metrics = compute_trial_metrics(trial)
        metrics["trial_id"] = pf.stem          # e.g. "cohface_14_1"
        metrics["pkl_path"] = str(pf)
        rows.append(metrics)

    df = pd.DataFrame(rows)

    # ── Tier assignment (based on OF SNR) ────
    df["tier"] = assign_tiers(df["snr_of_db"])
    tier_order = {t: i for i, t in enumerate(TIER_LABELS)}
    df["tier_order"] = df["tier"].map(tier_order).fillna(99)
    df = df.sort_values(["tier_order", "trial_id"]).reset_index(drop=True)

    # ── Save ─────────────────────────────────
    out = Path(output_dir) if output_dir else run_dir / "plots" / "quality_stratification"
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "trial_stratification.csv"
    df.to_csv(csv_path, index=False)
    print(f"[qs] Trial CSV saved → {csv_path}")

    # ── Per-tier summary JSON ─────────────────
    summary = {}
    for tier in TIER_LABELS:
        sub = df[df["tier"] == tier]
        n   = len(sub)
        tier_dict: dict = {"n_trials": n}
        for label in ["OF", "P1D-Cubic", "P1D-Linear", "P1D-Quad", "DoF"]:
            for role in ["base", "kfstd", "qrobf"]:
                col = f"freq_mae_{label}_{role}"
                if col in sub.columns:
                    vals = sub[col].dropna()
                    tier_dict[col] = {
                        "mean":  round(float(vals.mean()), 4) if len(vals) else None,
                        "std":   round(float(vals.std()),  4) if len(vals) else None,
                        "n":     len(vals),
                    }
        summary[tier] = tier_dict

    json_path = out / "tier_summary.json"
    with open(json_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[qs] Tier summary saved → {json_path}")

    # ── Console summary ───────────────────────
    print("\n[qs] ── Tier Distribution ──────────────────────")
    for tier in TIER_LABELS:
        sub = df[df["tier"] == tier]
        snr_vals = sub["snr_of_db"].dropna()
        print(f"  {tier:10s}: {len(sub):3d} trials | "
              f"SNR [{snr_vals.min():.1f}, {snr_vals.max():.1f}] dB "
              f"(mean {snr_vals.mean():.1f})")

    print("\n[qs] ── freq_mae (BPM): base / kfstd / QROBF per Tier [all families] ──")
    for base_m, kfstd_m, qrobf_m, label in METHOD_FAMILIES:
        base_col  = f"freq_mae_{label}_base"
        kfstd_col = f"freq_mae_{label}_kfstd"
        qrobf_col = f"freq_mae_{label}_qrobf"
        has_base  = base_col in df.columns
        print(f"\n  [{label}]")
        if has_base:
            print(f"  {'Tier':12s} {'base':>8s} {'kfstd':>8s} {'QROBF':>8s} {'Δ%':>8s}")
        else:
            print(f"  {'Tier':12s} {'kfstd':>8s} {'QROBF':>8s} {'Δ%':>8s}")
        for tier in TIER_LABELS:
            sub = df[df["tier"] == tier]
            base = sub[base_col].dropna().mean() if has_base else float("nan")
            kf   = sub[kfstd_col].dropna().mean() if kfstd_col in df.columns else float("nan")
            qr   = sub[qrobf_col].dropna().mean() if qrobf_col in df.columns else float("nan")
            imp  = (kf - qr) / kf * 100 if (kf > 0 and not np.isnan(kf + qr)) else np.nan
            if has_base:
                base_str = f"{base:8.3f}" if np.isfinite(base) else "     N/A"
                print(f"  {tier:12s} {base_str} {kf:8.3f} {qr:8.3f} {imp:+7.1f}%")
            else:
                print(f"  {tier:12s} {kf:8.3f} {qr:8.3f} {imp:+7.1f}%")

    return df


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def merge_eval_metrics(df: pd.DataFrame,
                        eval_raw_csv: str) -> pd.DataFrame:
    """Merge per-trial freq_mae from evaluation pipeline into stratification df.

    Replaces proxy freq_mae_* columns with evaluation pipeline results when
    available.  Columns added/replaced:
        freq_mae_{label}_{role}_eval   (from evaluation pipeline)

    Parameters
    ----------
    df : pd.DataFrame
        Output of run_stratification().
    eval_raw_csv : str
        Path to metrics_freq_domain_raw.csv (from evaluation pipeline).

    Returns
    -------
    pd.DataFrame – enriched with _eval columns.
    """
    eval_df = pd.read_csv(eval_raw_csv)

    # Build method → label + role mapping
    method_map: dict[str, tuple[str, str]] = {}
    for base_m, kfstd_m, qrobf_m, label in METHOD_FAMILIES:
        method_map[base_m]   = (label, "base")
        method_map[kfstd_m]  = (label, "kfstd")
        method_map[qrobf_m]  = (label, "qrobf")

    # Pivot eval_df: trial_id × method → MAE
    eval_pivot = eval_df.pivot_table(
        index="video", columns="method", values="MAE", aggfunc="first"
    ).reset_index()
    eval_pivot = eval_pivot.rename(columns={"video": "trial_id"})

    # Rename columns to match quality stratification naming
    rename = {}
    for m_name, (label, role) in method_map.items():
        if m_name in eval_pivot.columns:
            rename[m_name] = f"freq_mae_{label}_{role}_eval"
    eval_pivot = eval_pivot.rename(columns=rename)

    # Merge on trial_id
    df = df.merge(eval_pivot[["trial_id"] + list(rename.values())],
                  on="trial_id", how="left")

    n_merged = df["trial_id"].isin(eval_pivot["trial_id"]).sum()
    print(f"[qs] Merged eval metrics for {n_merged}/{len(df)} trials.")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quality Stratification Analysis")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--run-label",   default="cohface_robust_ossm")
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--merge-eval",  default=None,
                        help="Optional path to metrics_freq_domain_raw.csv "
                             "to merge evaluation pipeline per-trial metrics.")
    args = parser.parse_args()

    df = run_stratification(args.results_dir, args.run_label, args.output_dir)

    if args.merge_eval:
        out = Path(args.output_dir) if args.output_dir else \
              Path(args.results_dir) / args.run_label / "plots" / "quality_stratification"
        df = merge_eval_metrics(df, args.merge_eval)
        csv_path = out / "trial_stratification.csv"
        df.to_csv(csv_path, index=False)
        print(f"[qs] Updated CSV with eval metrics → {csv_path}")
