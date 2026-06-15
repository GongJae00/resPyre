#!/usr/bin/env python3
"""Generate dataset-level RR distribution EDA for the reproduction package.

This script deliberately separates four dataset roles:

- COHFACE and MAHNOB-HCI: real waveform + rate benchmarks.
- V4V: real respiratory-rate labels only.
- SCAMPS: synthetic breathing signal diagnostics.

The outputs are not headline performance metrics. They document label coverage,
rate-regime overlap, and claim boundaries so that weak external/synthetic
evidence is useful without being mixed into real waveform performance claims.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation.figure_style import save_figure, set_manuscript_style


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohface-eda", type=Path, default=ROOT / "analysis" / "cohface_observation_eda_trials.csv")
    p.add_argument("--mahnob-eda", type=Path, default=ROOT / "analysis" / "mahnob_observation_eda_trials.csv")
    p.add_argument("--v4v-manifest", type=Path, default=ROOT / "analysis" / "v4v_rr_rate_manifest.csv")
    p.add_argument("--scamps-manifest", type=Path, default=ROOT / "analysis" / "scamps_rr_synthetic_manifest.csv")
    p.add_argument("--scamps-fps", type=float, default=30.0)
    p.add_argument("--min-hz", type=float, default=0.08)
    p.add_argument("--max-hz", type=float, default=0.70)
    p.add_argument("--scamps-max", type=int, default=0, help="Debug limit; 0 means all SCAMPS rows.")
    p.add_argument("--out-rate-csv", type=Path, default=ROOT / "analysis" / "dataset_rate_distribution_eda.csv")
    p.add_argument("--out-scamp-csv", type=Path, default=ROOT / "analysis" / "scamps_rr_signal_eda.csv")
    p.add_argument("--out-summary-csv", type=Path, default=ROOT / "analysis" / "dataset_distribution_eda.csv")
    p.add_argument("--table-out", type=Path, default=ROOT / "paper" / "tables_ready" / "S_T_dataset_distribution_eda.csv")
    p.add_argument("--out-md", type=Path, default=ROOT / "analysis" / "dataset_distribution_eda.md")
    p.add_argument("--figure-out", type=Path, default=ROOT / "paper" / "figures" / "S_F_dataset_distribution_eda.pdf")
    return p.parse_args()


def _safe_read(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, usecols=usecols)


def _q(series: pd.Series, q: float) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.quantile(q)) if not vals.empty else math.nan


def _median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.median()) if not vals.empty else math.nan


def _iqr(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return math.nan
    return float(vals.quantile(0.75) - vals.quantile(0.25))


def _fft_peak_bpm(x: np.ndarray, fs: float, lo_hz: float, hi_hz: float) -> float:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 32 or not np.isfinite(fs) or fs <= 0:
        return math.nan
    arr = arr - float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 1e-12:
        return math.nan
    arr = arr / std
    win = np.hanning(arr.size)
    spec = np.abs(np.fft.rfft(arr * win)) ** 2
    freq = np.fft.rfftfreq(arr.size, d=1.0 / float(fs))
    mask = (freq >= float(lo_hz)) & (freq <= float(hi_hz))
    if not np.any(mask):
        return math.nan
    idx = np.flatnonzero(mask)[int(np.argmax(spec[mask]))]
    return float(freq[idx] * 60.0)


def _real_waveform_rate_rows(path: Path, dataset_label: str) -> pd.DataFrame:
    cols = ["dataset", "video", "subject", "trial", "duration_sec", "fs_gt", "gt_peak_hz"]
    df = _safe_read(path, usecols=cols)
    one = df.drop_duplicates("video").copy()
    one["dataset"] = dataset_label
    one["unit_id"] = one["video"].astype(str)
    one["rate_bpm"] = pd.to_numeric(one["gt_peak_hz"], errors="coerce") * 60.0
    one["rate_source"] = "respiratory waveform spectral peak"
    one["label_samples"] = np.nan
    one["waveform_gt_available"] = True
    one["real_or_synthetic"] = "real"
    one["claim_scope"] = "main real waveform/rate benchmark"
    return one[
        [
            "dataset",
            "unit_id",
            "subject",
            "trial",
            "rate_bpm",
            "rate_source",
            "duration_sec",
            "label_samples",
            "waveform_gt_available",
            "real_or_synthetic",
            "claim_scope",
        ]
    ]


def _v4v_rate_rows(path: Path) -> pd.DataFrame:
    df = _safe_read(path)
    out = pd.DataFrame(
        {
            "dataset": "V4V",
            "unit_id": df["trial_id"].astype(str),
            "subject": df.get("subject", pd.Series([""] * len(df))).astype(str),
            "trial": df.get("trial", pd.Series([""] * len(df))).astype(str),
            "rate_bpm": pd.to_numeric(df["rr_median_bpm"], errors="coerce"),
            "rate_source": "frame-aligned RR label median",
            "duration_sec": np.nan,
            "label_samples": pd.to_numeric(df.get("n_rr_values", pd.Series(dtype=float)), errors="coerce"),
            "waveform_gt_available": False,
            "real_or_synthetic": "real",
            "claim_scope": "external RR-rate only",
        }
    )
    return out


def _scamps_signal_rows(path: Path, *, fps: float, lo_hz: float, hi_hz: float, max_rows: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = _safe_read(path)
    if max_rows and max_rows > 0:
        manifest = manifest.head(int(max_rows)).copy()

    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        rows = []
        for _, row in manifest.iterrows():
            rows.append(
                {
                    "dataset": "SCAMPS",
                    "trial_id": row.get("trial_id", ""),
                    "mat_path": row.get("mat_path", ""),
                    "d_br_peak_bpm": math.nan,
                    "duration_sec_assumed": math.nan,
                    "read_error": f"h5py unavailable: {exc}",
                }
            )
        return pd.DataFrame(rows), pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        mat_path = Path(str(row.get("mat_path", "")))
        rec = {
            "dataset": "SCAMPS",
            "trial_id": str(row.get("trial_id", mat_path.stem)),
            "mat_path": str(mat_path),
            "n_frames": math.nan,
            "d_br_peak_bpm": math.nan,
            "duration_sec_assumed": math.nan,
            "fps_assumed": float(fps),
            "read_error": "",
        }
        try:
            with h5py.File(mat_path, "r") as f:
                if "d_br" not in f:
                    rec["read_error"] = "missing d_br"
                else:
                    sig = np.asarray(f["d_br"], dtype=np.float64).reshape(-1)
                    rec["n_frames"] = int(sig.size)
                    rec["duration_sec_assumed"] = float(sig.size) / float(fps)
                    rec["d_br_peak_bpm"] = _fft_peak_bpm(sig, fs=fps, lo_hz=lo_hz, hi_hz=hi_hz)
        except Exception as exc:  # pragma: no cover - raw-file dependent
            rec["read_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)

    signal_df = pd.DataFrame(rows)
    rate_df = pd.DataFrame(
        {
            "dataset": "SCAMPS",
            "unit_id": signal_df["trial_id"].astype(str),
            "subject": "",
            "trial": signal_df["trial_id"].astype(str),
            "rate_bpm": pd.to_numeric(signal_df["d_br_peak_bpm"], errors="coerce"),
            "rate_source": f"synthetic d_br spectral peak, assumed {fps:g} fps",
            "duration_sec": pd.to_numeric(signal_df["duration_sec_assumed"], errors="coerce"),
            "label_samples": pd.to_numeric(signal_df["n_frames"], errors="coerce"),
            "waveform_gt_available": True,
            "real_or_synthetic": "synthetic",
            "claim_scope": "synthetic mechanism diagnostic only",
        }
    )
    return signal_df, rate_df


def _summarize(rate_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    role_map = {
        "COHFACE": ("primary real waveform/rate", "main benchmark", "supports real waveform/rate claims"),
        "MAHNOB-HCI": ("hard real waveform/rate", "hard-regime benchmark", "supports observability-boundary analysis"),
        "V4V": ("external real rate-only", "supplementary rate-only scope", "no waveform or morphology claims"),
        "SCAMPS": ("synthetic controlled diagnostic", "supplementary synthetic control", "no real-data performance claims"),
    }
    for dataset, sub in rate_df.groupby("dataset", sort=False):
        role, use, boundary = role_map.get(str(dataset), ("", "", ""))
        subjects = sub["subject"].dropna().astype(str)
        subjects = subjects[subjects.str.len() > 0]
        n_subjects = int(subjects.nunique()) if not subjects.empty and subjects.nunique() > 1 else math.nan
        rows.append(
            {
                "dataset": dataset,
                "role": role,
                "release_use": use,
                "n_units": int(len(sub)),
                "n_subjects": n_subjects,
                "rate_source": "; ".join(sorted(set(sub["rate_source"].astype(str)))),
                "rate_median_bpm": _median(sub["rate_bpm"]),
                "rate_iqr_bpm": _iqr(sub["rate_bpm"]),
                "rate_p05_bpm": _q(sub["rate_bpm"], 0.05),
                "rate_p95_bpm": _q(sub["rate_bpm"], 0.95),
                "duration_median_sec": _median(sub["duration_sec"]),
                "duration_iqr_sec": _iqr(sub["duration_sec"]),
                "label_samples_median": _median(sub["label_samples"]),
                "waveform_gt_available": bool(sub["waveform_gt_available"].fillna(False).astype(bool).any()),
                "real_or_synthetic": str(sub["real_or_synthetic"].iloc[0]),
                "claim_boundary": boundary,
            }
        )
    return pd.DataFrame(rows)


def _write_md(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Dataset Distribution EDA",
        "",
        "This report complements the main COHFACE/MAHNOB-HCI metrics with label and",
        "rate-regime evidence from V4V and SCAMPS. It is not a hidden training or",
        "target-tuning step.",
        "",
        "## Summary",
        "",
        "| dataset | role | N | median RR bpm | IQR | duration median | release use | boundary |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in summary.iterrows():
        dur = row["duration_median_sec"]
        dur_s = "" if pd.isna(dur) else f"{float(dur):.1f}"
        lines.append(
            f"| {row['dataset']} | {row['role']} | {int(row['n_units'])} | "
            f"{float(row['rate_median_bpm']):.2f} | {float(row['rate_iqr_bpm']):.2f} | "
            f"{dur_s} | {row['release_use']} | {row['claim_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- COHFACE and MAHNOB-HCI remain the only real waveform/rate benchmarks.",
            "- V4V contributes real RR-rate label distribution and external timing scope, but no waveform morphology evidence.",
            "- SCAMPS contributes controlled synthetic breathing-signal coverage and a sanity-check rate distribution, not real-world robustness.",
            "- The distribution view is useful because it shows that label availability is not the same as observation observability.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _plot(rate_df: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_manuscript_style("paper")
    order = ["COHFACE", "MAHNOB-HCI", "V4V", "SCAMPS"]
    colors = {
        "COHFACE": "#256d85",
        "MAHNOB-HCI": "#b55a30",
        "V4V": "#4f8f45",
        "SCAMPS": "#7653a6",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    xs = np.arange(len(order))
    counts = [int(summary.loc[summary["dataset"].eq(d), "n_units"].iloc[0]) for d in order]
    ax.bar(xs, counts, color=[colors[d] for d in order], edgecolor="#222222", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Units (log scale)")
    ax.set_title("Dataset coverage")
    for i, n in enumerate(counts):
        ax.text(i, n * 1.08, f"N={n}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    ax = axes[0, 1]
    data = []
    labels = []
    for d in order:
        vals = pd.to_numeric(rate_df.loc[rate_df["dataset"].eq(d), "rate_bpm"], errors="coerce").dropna()
        if not vals.empty:
            data.append(vals.to_numpy())
            labels.append(d)
    box = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
    for patch, d in zip(box["boxes"], labels):
        patch.set_facecolor(colors.get(d, "#777777"))
        patch.set_alpha(0.72)
    ax.set_ylabel("Respiratory rate (bpm)")
    ax.set_title("Label / waveform-derived RR distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1, 0]
    duration = [summary.loc[summary["dataset"].eq(d), "duration_median_sec"].iloc[0] for d in order]
    label_samples = [summary.loc[summary["dataset"].eq(d), "label_samples_median"].iloc[0] for d in order]
    x = np.arange(len(order))
    dur_vals = np.asarray([0.0 if pd.isna(v) else float(v) for v in duration])
    sample_vals = np.asarray([0.0 if pd.isna(v) else float(v) for v in label_samples])
    ax.bar(x - 0.18, dur_vals, 0.36, color="#6b8fbf", label="duration sec")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, sample_vals, 0.36, color="#d49a3a", label="label samples")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("Median duration (s)")
    ax2.set_ylabel("Median label samples")
    ax.set_title("Temporal / label support")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    lines, labs = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, frameon=False, loc="upper left", fontsize=8)

    ax = axes[1, 1]
    capability_cols = [
        ("real", lambda r: r["real_or_synthetic"] == "real"),
        ("waveform GT", lambda r: bool(r["waveform_gt_available"])),
        ("headline", lambda r: "benchmark" in str(r["release_use"])),
        ("rate-only ext.", lambda r: "rate-only" in str(r["release_use"])),
        ("synthetic", lambda r: r["real_or_synthetic"] == "synthetic"),
    ]
    arr = np.zeros((len(order), len(capability_cols)), dtype=float)
    for i, d in enumerate(order):
        row = summary[summary["dataset"].eq(d)].iloc[0].to_dict()
        for j, (_, pred) in enumerate(capability_cols):
            arr[i, j] = 1.0 if pred(row) else 0.0
    cmap = matplotlib.colors.ListedColormap(["#f0f2f4", "#164f63"])
    ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xticks(range(len(capability_cols)))
    ax.set_xticklabels([c[0] for c in capability_cols], rotation=25, ha="right")
    ax.set_title("Claim-scope capability matrix")
    ax.tick_params(length=0)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, "yes" if arr[i, j] else "-", ha="center", va="center", fontsize=8, color="white" if arr[i, j] else "#555555")
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_figure(fig, out_path)


def main() -> int:
    args = _parse_args()
    rows = [
        _real_waveform_rate_rows(args.cohface_eda, "COHFACE"),
        _real_waveform_rate_rows(args.mahnob_eda, "MAHNOB-HCI"),
        _v4v_rate_rows(args.v4v_manifest),
    ]
    scamps_signal, scamps_rate = _scamps_signal_rows(
        args.scamps_manifest,
        fps=float(args.scamps_fps),
        lo_hz=float(args.min_hz),
        hi_hz=float(args.max_hz),
        max_rows=int(args.scamps_max or 0),
    )
    rows.append(scamps_rate)
    rate_df = pd.concat(rows, ignore_index=True)
    summary = _summarize(rate_df)

    for path in (args.out_rate_csv, args.out_scamp_csv, args.out_summary_csv, args.table_out, args.out_md, args.figure_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    rate_df.to_csv(args.out_rate_csv, index=False, float_format="%.6f")
    scamps_signal.to_csv(args.out_scamp_csv, index=False, float_format="%.6f")
    summary.to_csv(args.out_summary_csv, index=False, float_format="%.6f")
    summary.to_csv(args.table_out, index=False, float_format="%.6f")
    _write_md(summary, args.out_md)
    _plot(rate_df, summary, args.figure_out)

    print("Wrote:")
    for path in (args.out_rate_csv, args.out_scamp_csv, args.out_summary_csv, args.table_out, args.out_md, args.figure_out):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
