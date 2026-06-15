#!/usr/bin/env python3
"""Build RR-study dataset manifests, scope tables, and a scope figure.

This script is release-oriented. It does not copy raw data and it
does not force every dataset into the same evaluation protocol. Instead it
locks the role of each usable dataset:

- COHFACE / MAHNOB-HCI: real respiration waveform + rate.
- V4V: real respiratory-rate labels only.
- SCAMPS: synthetic controlled breathing signal.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
ANALYSIS_DIR = ROOT / "analysis"
TABLE_DIR = ROOT / "paper" / "tables_ready"
FIGURE_DIR = ROOT / "paper" / "figures"

RAW_DATA_ROOT = Path(os.environ.get("RESPYRE_RAW_DATA_ROOT", str(DATASET_DIR / "raw")))
RAW_TARGETS = {
    "COHFACE": RAW_DATA_ROOT / "cohface",
    "MAHNOB": RAW_DATA_ROOT / "MAHNOB_HCI",
    "V4V": RAW_DATA_ROOT / "V4V",
    "SCAMPS": RAW_DATA_ROOT / "SCAMPS",
}

RAW_TARGET_LABELS = {
    "COHFACE": "$RESPYRE_RAW_DATA_ROOT/cohface",
    "MAHNOB": "$RESPYRE_RAW_DATA_ROOT/MAHNOB_HCI",
    "V4V": "$RESPYRE_RAW_DATA_ROOT/V4V",
    "SCAMPS": "$RESPYRE_RAW_DATA_ROOT/SCAMPS",
}


@dataclass(frozen=True)
class DatasetRole:
    dataset: str
    role: str
    real_or_synthetic: str
    label_scope: str
    metric_scope: str
    release_use: str
    claim_boundary: str


DATASET_ROLES = (
    DatasetRole(
        dataset="COHFACE",
        role="primary_real_waveform_rate",
        real_or_synthetic="real",
        label_scope="respiration waveform + derived RR",
        metric_scope="rate + aligned waveform + strict waveform + cycle diagnostics",
        release_use="main real benchmark",
        claim_boundary="supports waveform reconstruction and rate estimation claims",
    ),
    DatasetRole(
        dataset="MAHNOB-HCI",
        role="hard_real_waveform_rate",
        real_or_synthetic="real",
        label_scope="respiration-belt waveform + derived RR",
        metric_scope="rate + aligned waveform + strict waveform + observability failure diagnostics",
        release_use="hard-regime / cross-environment benchmark",
        claim_boundary="supports robustness analysis; low observability must be diagnosed explicitly",
    ),
    DatasetRole(
        dataset="V4V",
        role="external_real_rate_only",
        real_or_synthetic="real",
        label_scope="frame-aligned HR/RR labels; no raw respiratory waveform",
        metric_scope="RR rate only",
        release_use="external rate-only validation",
        claim_boundary="does not support waveform CCC/DTW or morphology claims",
    ),
    DatasetRole(
        dataset="SCAMPS",
        role="synthetic_controlled_diagnostic",
        real_or_synthetic="synthetic",
        label_scope="synthetic breathing signal d_br plus video arrays",
        metric_scope="controlled rate/waveform sanity checks and ablations",
        release_use="synthetic mechanism diagnostic / optional pretraining",
        claim_boundary="must not be mixed with real-data performance claims",
    ),
)


ABLATION_ROWS = (
    {
        "block": "A0",
        "name": "fixed_observation_families",
        "datasets": "COHFACE;MAHNOB-HCI;SCAMPS",
        "question": "Which observation operators actually expose respiratory information?",
        "comparison": "OF, OF_bridge, DoF, DoF_bridge, P1D_lin, P1D_quad, P1D_cub, P1D_cons",
        "metrics": "observation EDA, rate MAE/r, waveform CCC/DTW where waveform GT exists",
        "release_location": "T2, F2, supplementary observation overlays",
    },
    {
        "block": "A1",
        "name": "constructed_observation_value",
        "datasets": "COHFACE;MAHNOB-HCI;SCAMPS",
        "question": "Do bridges/consensus add information beyond raw operators?",
        "comparison": "OF->OF_bridge, DoF->DoF_bridge, P1D_lin/quad/cub->P1D_cons",
        "metrics": "delta heatmaps, family summary, wrong-operator stress diagnostics",
        "release_location": "F2, F3, S_F13, S_F14",
    },
    {
        "block": "A2",
        "name": "osssm_kfstd_comparator",
        "datasets": "COHFACE;MAHNOB-HCI",
        "question": "What happens when the resonator and a standard Kalman filter are simply attached?",
        "comparison": "Base vs OSSM-KF vs PARH-OSSM",
        "metrics": "T3/T4/T4b/T4c/T6",
        "release_location": "main tables",
    },
    {
        "block": "A3",
        "name": "adaptive_observation_law",
        "datasets": "COHFACE;MAHNOB-HCI;V4V",
        "question": "Can target-side reliability choose/weight evidence without GT at deployment?",
        "comparison": "fixed family, reliability-weighted family, adaptive law, decoupled output",
        "metrics": "rate MAE/r, confidence, consistency, observability failure taxonomy",
        "release_location": "T5, T6, T7, S_F10, S_F11",
    },
    {
        "block": "A4",
        "name": "decoupled_rate_waveform_readouts",
        "datasets": "COHFACE;MAHNOB-HCI",
        "question": "Why should z_osc timing and z_full morphology not be collapsed into one objective?",
        "comparison": "single output, rate expert, waveform expert, decoupled readouts",
        "metrics": "rate MAE/r, waveform CCC/DTW, strict waveform, cycle timing",
        "release_location": "T3, T4, T4b, T4c, F4",
    },
    {
        "block": "A5",
        "name": "external_rate_generalization",
        "datasets": "V4V",
        "question": "Does the final timing readout remain meaningful on a new real RR-rate dataset?",
        "comparison": "final z_osc RR readout vs available baselines/adapters",
        "metrics": "RR MAE, RMSE, Pearson r; no waveform metrics",
        "release_location": "external validation supplement",
    },
    {
        "block": "A6",
        "name": "synthetic_controlled_sanity",
        "datasets": "SCAMPS",
        "question": "Under controlled video/respiration generation, do model components behave as designed?",
        "comparison": "known synthetic d_br against observation/state/readout variants",
        "metrics": "rate, waveform, component reliability, failure flags",
        "release_location": "synthetic diagnostic supplement",
    },
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build RR dataset manifests and experiment-scope assets.")
    ap.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    ap.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    ap.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    ap.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    ap.add_argument("--create-symlinks", action="store_true", help="Create missing dataset symlinks.")
    ap.add_argument("--scamps-max", type=int, default=0, help="Limit SCAMPS manifest rows; 0 means all files.")
    ap.add_argument("--skip-scamp-h5", action="store_true", help="List SCAMPS files without opening HDF5 metadata.")
    return ap.parse_args()


def _mkdirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _resolve_link(path: Path) -> str:
    try:
        if path.is_symlink():
            return os.readlink(path)
        if path.exists():
            return str(path.resolve())
    except OSError:
        pass
    return ""


def ensure_symlinks(dataset_dir: Path, *, create: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name, target in RAW_TARGETS.items():
        link = dataset_dir / name
        status = "ok"
        if link.exists() or link.is_symlink():
            if link.is_symlink():
                current = Path(os.readlink(link))
                if current != target:
                    status = "target_mismatch"
            else:
                status = "exists_not_symlink"
        elif create:
            link.symlink_to(target)
            status = "created"
        else:
            status = "missing"
        rows.append(
            {
                "dataset": name,
                "link_path": str(link),
                "target_path": str(target),
                "link_target": _resolve_link(link),
                "target_exists": bool(target.exists()),
                "link_exists": bool(link.exists() or link.is_symlink()),
                "status": status,
            }
        )
    return rows


def _count_files(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob(pattern))


def build_dataset_scope(dataset_dir: Path, link_rows: list[dict[str, object]]) -> pd.DataFrame:
    link_by_name = {str(r["dataset"]): r for r in link_rows}
    counts = {
        "COHFACE": {
            "n_trials": _count_files(dataset_dir / "COHFACE", "data.hdf5"),
            "n_videos": _count_files(dataset_dir / "COHFACE", "data.avi") + _count_files(dataset_dir / "COHFACE", "data.mkv"),
            "label_evidence": "HDF5 respiration dataset",
        },
        "MAHNOB-HCI": {
            "n_trials": _count_files(dataset_dir / "MAHNOB", "*.bdf"),
            "n_videos": _count_files(dataset_dir / "MAHNOB", "*.avi"),
            "label_evidence": "BDF Resp channel / respiration belt",
        },
        "V4V": {
            "n_trials": _count_files(dataset_dir / "V4V" / "Phase_1_Training_Validation_sets" / "Ground_truth" / "Physiology", "*.txt"),
            "n_videos": _count_files(dataset_dir / "V4V", "*.mkv"),
            "label_evidence": "Physiology text rows: HR and RR",
        },
        "SCAMPS": {
            "n_trials": _count_files(dataset_dir / "SCAMPS" / "scamps_videos", "*.mat"),
            "n_videos": _count_files(dataset_dir / "SCAMPS" / "scamps_videos", "*.mat"),
            "label_evidence": "MATLAB v7.3 d_br field",
        },
    }
    rows = []
    for role in DATASET_ROLES:
        link_name = "MAHNOB" if role.dataset == "MAHNOB-HCI" else role.dataset
        link = link_by_name.get(link_name, {})
        c = counts.get(role.dataset, {})
        rows.append(
            {
                "dataset": role.dataset,
                "role": role.role,
                "real_or_synthetic": role.real_or_synthetic,
                "label_scope": role.label_scope,
                "metric_scope": role.metric_scope,
                "release_use": role.release_use,
                "claim_boundary": role.claim_boundary,
                "n_trials_or_label_files": c.get("n_trials", 0),
                "n_video_files_or_mat_files": c.get("n_videos", 0),
                "label_evidence": c.get("label_evidence", ""),
                "symlink_status": link.get("status", ""),
                "symlink_path": f"dataset/{link_name}",
                "raw_target": RAW_TARGET_LABELS.get(link_name, ""),
            }
        )
    return pd.DataFrame(rows)


def _parse_v4v_line(line: str) -> tuple[str, str, np.ndarray]:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 3:
        raise ValueError("V4V physiology line must contain video, signal name, and values.")
    video = parts[0]
    signal = parts[1].upper()
    values = np.asarray([float(v) for v in parts[2:] if v.strip() != ""], dtype=np.float64)
    return video, signal, values


def _find_v4v_video(v4v_root: Path, stem: str) -> tuple[str, str]:
    for split in ("train", "valid"):
        p = v4v_root / "Phase_1_Training_Validation_sets" / "Videos" / split / f"{stem}.mkv"
        if p.exists():
            return str(p), split
    return "", "missing"


def build_v4v_manifest(v4v_root: Path) -> pd.DataFrame:
    phys_dir = v4v_root / "Phase_1_Training_Validation_sets" / "Ground_truth" / "Physiology"
    rows = []
    for label_path in sorted(phys_dir.glob("*.txt")):
        lines = [line for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        signals: dict[str, np.ndarray] = {}
        video_name = f"{label_path.stem}.mkv"
        for line in lines:
            video, signal, values = _parse_v4v_line(line)
            video_name = video or video_name
            signals[signal] = values
        rr = signals.get("RR", np.asarray([], dtype=np.float64))
        hr = signals.get("HR", np.asarray([], dtype=np.float64))
        video_path, split = _find_v4v_video(v4v_root, Path(video_name).stem)
        finite_rr = rr[np.isfinite(rr)]
        finite_hr = hr[np.isfinite(hr)]
        rows.append(
            {
                "dataset": "V4V",
                "trial_id": Path(video_name).stem,
                "subject": Path(video_name).stem.split("_T", 1)[0],
                "trial": "T" + Path(video_name).stem.split("_T", 1)[1] if "_T" in Path(video_name).stem else "",
                "split": split,
                "video": video_name,
                "video_path": video_path,
                "video_exists": bool(video_path),
                "label_path": str(label_path),
                "n_rr_values": int(finite_rr.size),
                "n_hr_values": int(finite_hr.size),
                "rr_median_bpm": float(np.median(finite_rr)) if finite_rr.size else math.nan,
                "rr_iqr_bpm": float(np.percentile(finite_rr, 75) - np.percentile(finite_rr, 25)) if finite_rr.size else math.nan,
                "rr_min_bpm": float(np.min(finite_rr)) if finite_rr.size else math.nan,
                "rr_max_bpm": float(np.max(finite_rr)) if finite_rr.size else math.nan,
                "rr_unique_count": int(np.unique(np.round(finite_rr, 4)).size) if finite_rr.size else 0,
                "hr_median_bpm": float(np.median(finite_hr)) if finite_hr.size else math.nan,
                "label_scope": "rate_only",
            }
        )
    return pd.DataFrame(rows)


def build_scamps_manifest(scamps_root: Path, *, max_files: int = 0, skip_h5: bool = False) -> pd.DataFrame:
    mat_files = sorted((scamps_root / "scamps_videos").glob("*.mat"))
    if max_files and max_files > 0:
        mat_files = mat_files[:max_files]
    rows = []
    h5py = None
    if not skip_h5:
        try:
            import h5py as _h5py

            h5py = _h5py
        except Exception:
            h5py = None
    for mat_path in mat_files:
        row = {
            "dataset": "SCAMPS",
            "trial_id": mat_path.stem,
            "mat_path": str(mat_path),
            "mat_exists": mat_path.exists(),
            "has_raw_frames": False,
            "has_xsub": False,
            "has_d_br": False,
            "has_d_ppg": False,
            "has_d_ekg": False,
            "n_frames": math.nan,
            "rawframes_shape": "",
            "d_br_shape": "",
            "label_scope": "synthetic_waveform_rate",
            "read_error": "",
        }
        if h5py is not None:
            try:
                with h5py.File(mat_path, "r") as f:
                    keys = set(f.keys())
                    row["has_raw_frames"] = "RawFrames" in keys
                    row["has_xsub"] = "Xsub" in keys
                    row["has_d_br"] = "d_br" in keys
                    row["has_d_ppg"] = "d_ppg" in keys
                    row["has_d_ekg"] = "d_ekg" in keys
                    if "RawFrames" in keys:
                        shape = tuple(int(v) for v in f["RawFrames"].shape)
                        row["rawframes_shape"] = "x".join(map(str, shape))
                    if "d_br" in keys:
                        shape = tuple(int(v) for v in f["d_br"].shape)
                        row["d_br_shape"] = "x".join(map(str, shape))
                        row["n_frames"] = int(shape[0])
            except Exception as exc:  # pragma: no cover - raw-file dependent
                row["read_error"] = f"{type(exc).__name__}: {exc}"
        elif not skip_h5:
            row["read_error"] = "h5py unavailable"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_external_manifests(v4v: pd.DataFrame, scamps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not v4v.empty:
        rows.append(
            {
                "dataset": "V4V",
                "scope": "real_rate_only",
                "n_rows": len(v4v),
                "n_valid_paths": int(v4v["video_exists"].sum()),
                "median_rr_bpm": float(v4v["rr_median_bpm"].median()),
                "median_rr_iqr_bpm": float(v4v["rr_iqr_bpm"].median()),
                "label_note": "RR labels are rate-only; no waveform metrics.",
            }
        )
    if not scamps.empty:
        rows.append(
            {
                "dataset": "SCAMPS",
                "scope": "synthetic_diagnostic",
                "n_rows": len(scamps),
                "n_valid_paths": int(scamps["mat_exists"].sum()),
                "median_rr_bpm": math.nan,
                "median_rr_iqr_bpm": math.nan,
                "label_note": "Synthetic d_br field; keep separate from real-data claims.",
            }
        )
    return pd.DataFrame(rows)


def write_blueprint(scope: pd.DataFrame, external_summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# RR Experiment Blueprint",
        "",
        "This blueprint fixes the dataset roles before full runs. The key rule is to match each dataset to the strongest label it actually provides.",
        "",
        "## Dataset Roles",
        "",
        "| dataset | role | label scope | metric scope | claim boundary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in scope.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['role']} | {row['label_scope']} | {row['metric_scope']} | {row['claim_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Experiment Blocks",
            "",
            "| block | purpose | datasets | outputs |",
            "| --- | --- | --- | --- |",
            "| real waveform/rate | Main PARH-OSSM validation and strict morphology diagnostics | COHFACE, MAHNOB-HCI | T3, T4, T4b, T4c, T6, F3, F4 |",
            "| external rate-only | Check whether respiratory timing transfers to a real RR-only dataset | V4V | external RR MAE/RMSE/Pearson table; no waveform plots |",
            "| synthetic controlled | Verify mechanism behavior where breathing signal is controlled | SCAMPS | synthetic diagnostic table/figure; no real-data claim |",
            "| ablation/diagnostic | Show why each modeling decision exists | all compatible subsets | T2, T5, T7, S_F* diagnostics |",
            "",
            "## External Manifest Summary",
            "",
        ]
    )
    if external_summary.empty:
        lines.append("No external manifests were generated.")
    else:
        lines.extend(
            [
                "| dataset | scope | rows | valid paths | median RR | note |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in external_summary.iterrows():
            median_rr = row.get("median_rr_bpm")
            median_rr_str = "" if pd.isna(median_rr) else f"{float(median_rr):.3f}"
            lines.append(
                f"| {row['dataset']} | {row['scope']} | {int(row['n_rows'])} | {int(row['n_valid_paths'])} | {median_rr_str} | {row['label_note']} |"
            )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Waveform reconstruction claims must use only datasets with respiratory waveform ground truth.",
            "- V4V can support external RR-rate generalization only.",
            "- SCAMPS can support controlled mechanism evidence only and should not be pooled with real datasets.",
            "- Datasets without respiration labels remain excluded unless a respiratory annotation is identified and audited.",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_scope_figure(scope: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    waveform_roles = {"primary_real_waveform_rate", "hard_real_waveform_rate"}
    columns = [
        ("real", lambda row: str(row["real_or_synthetic"]) == "real"),
        ("real waveform GT", lambda row: str(row["role"]) in waveform_roles),
        ("RR/rate label", lambda row: str(row["dataset"]) in {"COHFACE", "MAHNOB-HCI", "V4V", "SCAMPS"}),
        ("main metrics", lambda row: str(row["role"]) in waveform_roles),
        ("external rate", lambda row: "external" in str(row["role"])),
        ("synthetic diagnostic", lambda row: str(row["real_or_synthetic"]) == "synthetic"),
    ]
    arr = np.zeros((len(scope), len(columns)), dtype=float)
    for i, (_, row) in enumerate(scope.iterrows()):
        for j, (_, pred) in enumerate(columns):
            arr[i, j] = 1.0 if pred(row) else 0.0

    fig, ax = plt.subplots(figsize=(10.6, 3.4))
    cmap = matplotlib.colors.ListedColormap(["#f3f5f7", "#256d85"])
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c[0] for c in columns], rotation=25, ha="right")
    ax.set_yticks(range(len(scope)))
    ax.set_yticklabels(scope["dataset"].tolist())
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, "yes" if arr[i, j] else "-", ha="center", va="center", fontsize=9, color="white" if arr[i, j] else "#5d6670")
    ax.set_title("Dataset evidence scope for respiratory-rate study", fontsize=12, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    _mkdirs(dataset_dir, analysis_dir, table_dir, figure_dir)

    link_rows = ensure_symlinks(dataset_dir, create=bool(args.create_symlinks))
    link_df = pd.DataFrame(link_rows)
    link_df.to_csv(analysis_dir / "rr_dataset_symlink_audit.csv", index=False)

    scope = build_dataset_scope(dataset_dir, link_rows)
    scope.to_csv(table_dir / "T1_dataset_protocol_scope.csv", index=False)
    scope.to_csv(analysis_dir / "rr_dataset_scope.csv", index=False)

    v4v = build_v4v_manifest(dataset_dir / "V4V")
    v4v.to_csv(analysis_dir / "v4v_rr_rate_manifest.csv", index=False, float_format="%.6f")

    scamps = build_scamps_manifest(
        dataset_dir / "SCAMPS",
        max_files=int(args.scamps_max or 0),
        skip_h5=bool(args.skip_scamp_h5),
    )
    scamps.to_csv(analysis_dir / "scamps_rr_synthetic_manifest.csv", index=False)

    external_summary = summarize_external_manifests(v4v, scamps)
    external_summary.to_csv(table_dir / "S_T_external_rr_manifest_summary.csv", index=False, float_format="%.4f")

    ablation = pd.DataFrame(ABLATION_ROWS)
    ablation.to_csv(table_dir / "S_T_rr_ablation_design_contract.csv", index=False)
    ablation.to_csv(analysis_dir / "rr_ablation_design_contract.csv", index=False)

    write_blueprint(scope, external_summary, analysis_dir / "rr_experiment_blueprint.md")
    plot_scope_figure(scope, figure_dir / "S_F_rr_dataset_scope_map.pdf")

    print("Wrote:")
    for path in (
        analysis_dir / "rr_dataset_symlink_audit.csv",
        analysis_dir / "rr_dataset_scope.csv",
        analysis_dir / "v4v_rr_rate_manifest.csv",
        analysis_dir / "scamps_rr_synthetic_manifest.csv",
        analysis_dir / "rr_ablation_design_contract.csv",
        analysis_dir / "rr_experiment_blueprint.md",
        table_dir / "T1_dataset_protocol_scope.csv",
        table_dir / "S_T_rr_ablation_design_contract.csv",
        table_dir / "S_T_external_rr_manifest_summary.csv",
        figure_dir / "S_F_rr_dataset_scope_map.pdf",
    ):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
