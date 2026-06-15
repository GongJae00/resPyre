#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COHFACE_DATA = (
    ROOT
    / "results"
    / "20260409_cohface_prod_ofbridge_dofbridge_p1dcons_e2e_policy_narrow"
    / "cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons"
    / "data"
)
DEFAULT_MAHNOB_DATA = ROOT / "results" / "mahnob_gt_tailaligned_patch_v1" / "data"
DEFAULT_RESULTS_ROOT = ROOT / "results" / "final_operating_point_sensitivity"
DEFAULT_ANALYSIS_CSV = ROOT / "analysis" / "final_operating_point_sensitivity.csv"
DEFAULT_ANALYSIS_MD = ROOT / "analysis" / "final_operating_point_sensitivity.md"
DEFAULT_COMMANDS_OUT = ROOT / "analysis" / "final_operating_point_sensitivity_commands.sh"


@dataclass(frozen=True)
class OperatingPoint:
    slug: str
    description: str
    window_sec: float
    window_stride_sec: float
    min_support_corr: float
    max_support_residual: float


LOCKED_OPERATING_POINTS: tuple[OperatingPoint, ...] = (
    OperatingPoint(
        slug="locked_default",
        description="current locked release setting; balanced locality and stability",
        window_sec=30.0,
        window_stride_sec=10.0,
        min_support_corr=0.25,
        max_support_residual=1.25,
    ),
    OperatingPoint(
        slug="more_local_windows",
        description="more local target reliability; tests within-trial adaptation without new labels",
        window_sec=20.0,
        window_stride_sec=5.0,
        min_support_corr=0.25,
        max_support_residual=1.25,
    ),
    OperatingPoint(
        slug="more_stable_windows",
        description="longer reliability windows; tests whether MAHNOB needs more stable evidence aggregation",
        window_sec=45.0,
        window_stride_sec=15.0,
        min_support_corr=0.25,
        max_support_residual=1.25,
    ),
    OperatingPoint(
        slug="stricter_cross_family_support",
        description="higher agreement requirement; tests whether weak cross-family edges should abstain",
        window_sec=30.0,
        window_stride_sec=10.0,
        min_support_corr=0.30,
        max_support_residual=1.00,
    ),
    OperatingPoint(
        slug="looser_cross_family_support",
        description="lower agreement requirement; tests whether hard regimes are under-supported",
        window_sec=30.0,
        window_stride_sec=10.0,
        min_support_corr=0.20,
        max_support_residual=1.50,
    ),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run or plan a bounded final operating-point sensitivity study. "
            "This is not a target-GT hyperparameter search: the grid is fixed, "
            "small, semantically interpretable, and reported as sensitivity "
            "evidence before the final full run."
        )
    )
    p.add_argument("--cohface-data-dir", type=Path, default=DEFAULT_COHFACE_DATA)
    p.add_argument("--mahnob-data-dir", type=Path, default=DEFAULT_MAHNOB_DATA)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--analysis-csv", type=Path, default=DEFAULT_ANALYSIS_CSV)
    p.add_argument("--analysis-md", type=Path, default=DEFAULT_ANALYSIS_MD)
    p.add_argument("--commands-out", type=Path, default=DEFAULT_COMMANDS_OUT)
    p.add_argument("--max-files", type=int, default=32)
    p.add_argument("--jobs", type=int, default=int(os.environ.get("RESPYRE_JOBS", os.environ.get("PARALLEL_PROCS", "1"))))
    p.add_argument("--artifact-policy", choices=["lean", "full", "smoke"], default="lean")
    p.add_argument("--execute", action="store_true", help="Actually run the fixed sensitivity grid.")
    p.add_argument("--skip-existing", action="store_true", help="Skip an operating point when both summary metrics already exist.")
    p.add_argument(
        "--only",
        action="append",
        default=[],
        help="Optional operating-point slug to run. Repeatable. Defaults to all locked points.",
    )
    return p.parse_args()


def _shell(cmd: Sequence[object]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def _python() -> str:
    return os.environ.get("PY", sys.executable)


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd: Sequence[object]) -> None:
    subprocess.run([str(x) for x in cmd], check=True)


def _selected_points(slugs: Iterable[str]) -> list[OperatingPoint]:
    selected = {str(x).strip() for x in slugs if str(x).strip()}
    points = list(LOCKED_OPERATING_POINTS)
    if not selected:
        return points
    known = {p.slug for p in points}
    unknown = sorted(selected - known)
    if unknown:
        raise SystemExit(f"unknown operating point(s): {unknown}; known={sorted(known)}")
    return [p for p in points if p.slug in selected]


def _metric_medians(metrics_dir: Path) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    files = {
        "rate": "metrics_freq_domain_raw.csv",
        "wave": "metrics_waveform_raw.csv",
        "strict": "metrics_waveform_strict_raw.csv",
        "guard": "readout_guard_raw.csv",
    }
    for section, name in files.items():
        path = metrics_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        nums = df.select_dtypes(include="number")
        med = nums.median(numeric_only=True)
        for key, value in med.items():
            if pd.notna(value):
                out[f"{section}_{key}"] = float(value)
    return out


def _summary_exists(run_dir: Path) -> bool:
    metrics = run_dir / "metrics"
    return (
        (metrics / "metrics_freq_domain_raw.csv").exists()
        and (metrics / "metrics_waveform_raw.csv").exists()
        and (metrics / "metrics_waveform_strict_raw.csv").exists()
    )


def _maybe_run_strict_metric_command(cmd: Sequence[object]) -> None:
    parts = [str(x) for x in cmd]
    try:
        data_dir = Path(parts[parts.index("--data-dir") + 1])
        out_dir = Path(parts[parts.index("--out-dir") + 1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"malformed strict metric command: {_shell(cmd)}") from exc

    strict_raw = out_dir / "metrics_waveform_strict_raw.csv"
    if strict_raw.exists():
        return
    if any(data_dir.rglob("*.pkl")):
        _run(cmd)
        return
    raise FileNotFoundError(
        "Strict metrics are missing, but lean data artifacts are not available "
        f"under {data_dir}. Re-run materialization or use --artifact-policy full."
    )


def _build_commands(args: argparse.Namespace, point: OperatingPoint) -> dict[str, list[list[object]]]:
    py = _python()
    root = Path(args.results_root).resolve() / point.slug
    priors = root / "priors"
    coh_run = root / "cohface"
    mahnob_run = root / "mahnob_tailaligned"

    common_extract = [
        py,
        ROOT / "scripts" / "extract_target_reliability_graph_features.py",
        "--window-sec",
        f"{point.window_sec:g}",
        "--window-stride-sec",
        f"{point.window_stride_sec:g}",
        "--min-support-corr",
        f"{point.min_support_corr:g}",
        "--max-support-residual",
        f"{point.max_support_residual:g}",
        "--max-files",
        str(args.max_files),
        "--jobs",
        str(args.jobs),
    ]
    combined_prior = [
        *common_extract,
        "--data-dir",
        Path(args.cohface_data_dir).resolve(),
        "--dataset-label",
        "COHFACE",
        "--data-dir",
        Path(args.mahnob_data_dir).resolve(),
        "--dataset-label",
        "MAHNOB_tailaligned",
        "--out-edge",
        priors / "cohface_mahnob_target_reliability_windowed_edges.csv",
        "--out-group",
        priors / "cohface_mahnob_target_computable_reliability_windowed.csv",
        "--out-summary",
        priors / "cohface_mahnob_target_computable_reliability_windowed_summary.csv",
        "--report-out",
        priors / "cohface_mahnob_target_computable_reliability_windowed.md",
    ]
    mahnob_prior = [
        *common_extract,
        "--data-dir",
        Path(args.mahnob_data_dir).resolve(),
        "--dataset-label",
        "MAHNOB_tailaligned",
        "--out-edge",
        priors / "mahnob_target_reliability_windowed_edges.csv",
        "--out-group",
        priors / "mahnob_target_computable_reliability_windowed.csv",
        "--out-summary",
        priors / "mahnob_target_computable_reliability_windowed_summary.csv",
        "--report-out",
        priors / "mahnob_target_computable_reliability_windowed.md",
    ]

    materialize_common = [
        py,
        ROOT / "scripts" / "materialize_calibrated_multifamily_parh_system.py",
        "--name",
        "parh_ossm",
        "--parh-input",
        "multichannel",
        "--external-rate-evidence-source",
        "ossm_kfstd",
        "--enable-observation-law",
        "--enable-rate-posterior",
        "--enable-target-observability-control",
        "--enable-signal-sqi-observability",
        "--rate-posterior-output-source",
        "final",
        "--eval-use-track",
        "--max-files",
        str(args.max_files),
        "--jobs",
        str(args.jobs),
        "--artifact-policy",
        str(args.artifact_policy),
    ]
    cohface_materialize = [
        *materialize_common,
        "--data-dir",
        Path(args.cohface_data_dir).resolve(),
        "--out-run",
        coh_run,
        "--reliability-group-csv",
        priors / "cohface_mahnob_target_computable_reliability_windowed.csv",
        "--reliability-prior-scope",
        "readout_only",
    ]
    mahnob_materialize = [
        *materialize_common,
        "--data-dir",
        Path(args.mahnob_data_dir).resolve(),
        "--out-run",
        mahnob_run,
        "--state-reliability-group-csv",
        priors / "mahnob_state_component_reliability_windowed.csv",
        "--readout-reliability-group-csv",
        priors / "mahnob_target_computable_reliability_windowed.csv",
    ]
    strict_coh = [
        py,
        ROOT / "scripts" / "generate_waveform_strict_metrics.py",
        "--data-dir",
        coh_run / "data",
        "--out-dir",
        coh_run / "metrics",
    ]
    strict_mahnob = [
        py,
        ROOT / "scripts" / "generate_waveform_strict_metrics.py",
        "--data-dir",
        mahnob_run / "data",
        "--out-dir",
        mahnob_run / "metrics",
    ]
    return {
        "prior": [combined_prior, mahnob_prior],
        "materialize": [cohface_materialize, mahnob_materialize],
        "strict": [strict_coh, strict_mahnob],
    }


def _write_commands(path: Path, args: argparse.Namespace, points: Sequence[OperatingPoint]) -> None:
    _mkdir(path.parent)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd \"$(git rev-parse --show-toplevel 2>/dev/null || pwd)\"",
        "",
        "# Fixed, bounded operating-point sensitivity. Do not report best-of-target sweep claims.",
    ]
    for point in points:
        lines.append("")
        lines.append(f"# {point.slug}: {point.description}")
        commands = _build_commands(args, point)
        for group_name, group in commands.items():
            if group_name == "strict" and str(args.artifact_policy) != "full":
                lines.append(
                    "# Standalone strict regeneration omitted for lean policy: "
                    "materialization writes strict metrics, and lean may not retain PKL data."
                )
                continue
            for cmd in group:
                lines.append(_shell(cmd))
            if group_name == "prior":
                prior_root = Path(args.results_root).resolve() / point.slug / "priors"
                lines.append(
                    _shell(
                        [
                            "cp",
                            prior_root / "mahnob_target_computable_reliability_windowed.csv",
                            prior_root / "mahnob_state_component_reliability_windowed.csv",
                        ]
                    )
                )
        lines.append(
            _shell(
                [
                    _python(),
                    ROOT / "scripts" / "run_final_operating_point_sensitivity.py",
                    "--only",
                    point.slug,
                    "--max-files",
                    args.max_files,
                    "--jobs",
                    args.jobs,
                    "--artifact-policy",
                    args.artifact_policy,
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_rows(args: argparse.Namespace, points: Sequence[OperatingPoint]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for point in points:
        root = Path(args.results_root).resolve() / point.slug
        for dataset, run_name in (("COHFACE", "cohface"), ("MAHNOB_tailaligned", "mahnob_tailaligned")):
            run_dir = root / run_name
            row: dict[str, float | str] = {
                "operating_point": point.slug,
                "dataset": dataset,
                "description": point.description,
                "window_sec": point.window_sec,
                "window_stride_sec": point.window_stride_sec,
                "min_support_corr": point.min_support_corr,
                "max_support_residual": point.max_support_residual,
                "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
            }
            row.update(_metric_medians(run_dir / "metrics"))
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    _mkdir(path.parent)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(row: dict[str, float | str], key: str) -> str:
    value = row.get(key, "")
    if value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _write_md(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    _mkdir(path.parent)
    lines = [
        "# Final Operating-Point Sensitivity",
        "",
        "This is a bounded sensitivity report, not a best-of-sweep selector.",
        "The grid is fixed before looking at target labels and changes only",
        "semantically interpretable target-computable reliability settings:",
        "window length/stride and cross-family support strictness.",
        "",
        "Paper rule: do not claim a tuned MAHNOB optimum from this table.",
        "Use it to check whether the locked operating point is fragile and to",
        "justify one final full rerun only if the same setting is defensible",
        "before target-GT performance is inspected.",
        "",
        "| operating point | dataset | rate MAE | rate R | aligned CCC | strict CCC | NMAE span | guard alpha | abstain |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("operating_point", "")),
                    str(row.get("dataset", "")),
                    _fmt(row, "rate_MAE"),
                    _fmt(row, "rate_PearsonR"),
                    _fmt(row, "wave_waveform_CCC"),
                    _fmt(row, "strict_strict_CCC"),
                    _fmt(row, "strict_strict_NMAE_span"),
                    _fmt(row, "guard_alpha_median"),
                    _fmt(row, "guard_abstain_pressure_mean"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Locked Grid",
            "",
        ]
    )
    for point in LOCKED_OPERATING_POINTS:
        lines.append(
            f"- `{point.slug}`: window `{point.window_sec:g}s/{point.window_stride_sec:g}s`, "
            f"support corr `{point.min_support_corr:g}`, residual `{point.max_support_residual:g}`; "
            f"{point.description}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    points = _selected_points(args.only)
    _write_commands(Path(args.commands_out), args, points)

    if args.execute:
        for point in points:
            root = Path(args.results_root).resolve() / point.slug
            if args.skip_existing and _summary_exists(root / "cohface") and _summary_exists(root / "mahnob_tailaligned"):
                continue
            _mkdir(root / "priors")
            commands = _build_commands(args, point)
            for cmd in commands["prior"]:
                _run(cmd)
            shutil.copyfile(
                root / "priors" / "mahnob_target_computable_reliability_windowed.csv",
                root / "priors" / "mahnob_state_component_reliability_windowed.csv",
            )
            for cmd in commands["materialize"]:
                _run(cmd)
            for cmd in commands["strict"]:
                _maybe_run_strict_metric_command(cmd)

    rows = _collect_rows(args, points)
    _write_csv(Path(args.analysis_csv), rows)
    _write_md(Path(args.analysis_md), rows)
    print(json.dumps({"rows": len(rows), "csv": str(args.analysis_csv), "md": str(args.analysis_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
