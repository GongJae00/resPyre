#!/usr/bin/env python3
"""One-shot PARH end-to-end runner.

Runs the current estimation pipeline and then generates analysis/paper artifacts
from the produced results so the user can launch one command and leave it
running unattended.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline.metadata_step import bootstrap_run_dirs
from core.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PARH estimate + postprocess + paper artifacts in one command.")
    parser.add_argument("--config", "-c", type=Path, required=True, help="Config JSON passed to main.py")
    parser.add_argument("--results", "-r", type=Path, default=None, help="Results root override")
    parser.add_argument("--skip-estimate", action="store_true", help="Skip main.py and only run postprocessing")
    parser.add_argument("--debug", action="store_true", help="Forward --debug to main.py")
    parser.add_argument(
        "--peer-metrics",
        action="append",
        default=[],
        help="Optional dataset metrics mapping for combined table generation: DATASET=/path/to/metrics",
    )
    parser.add_argument(
        "--paper-suite",
        choices=["auto", "none", "cohface"],
        default="auto",
        help="Whether to generate paper-facing COHFACE figures/manifests/PDF.",
    )
    parser.add_argument(
        "--overlay-family",
        default="P1D_quad",
        choices=["OF", "OF_bridge", "DoF", "P1D_lin", "P1D_quad", "P1D_cub"],
        help="Family used for same-trial waveform overlay figure.",
    )
    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip observation EDA and preprocessing heatmap generation.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip latexmk rebuild even when paper suite is enabled.",
    )
    return parser.parse_args()


def parse_peer_metrics(specs: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for spec in specs:
        raw = str(spec).strip()
        if not raw:
            continue
        if "=" not in raw:
            raise ValueError(
                f"Invalid --peer-metrics value '{spec}'. Expected DATASET=/path/to/metrics."
            )
        name, path_str = raw.split("=", 1)
        dataset = str(name).strip().upper()
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        out[dataset] = path
    return out


def run_cmd(cmd: List[str], *, cwd: Path = ROOT) -> None:
    pretty = " ".join(str(c) for c in cmd)
    print(f"\n[run] {pretty}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def load_run_status(run_dir: Path) -> dict:
    path = run_dir / "run_status.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def assert_run_completed(run_dir: Path) -> None:
    status = load_run_status(run_dir)
    if status.get("status") != "completed":
        raise RuntimeError(f"Run directory is not completed: {run_dir} (status={status.get('status')!r})")


def dataset_run_dirs(cfg: dict, results_root: Path) -> Dict[str, Path]:
    dataset_names = [str(d.get("name", "")).strip().upper() for d in cfg.get("datasets", []) if str(d.get("name", "")).strip()]
    run_dirs = [
        Path(p)
        for p in bootstrap_run_dirs(
            str(results_root),
            cfg.get("name"),
            dataset_names=dataset_names,
        )
    ]
    out: Dict[str, Path] = {}
    for dataset, run_dir in zip(dataset_names, run_dirs):
        out[dataset] = run_dir
    return out


def manifest_prefix(dataset_name: str) -> str:
    return dataset_name.strip().lower()


def family_to_manifest_slug(family: str) -> str:
    return family.strip().lower().replace("+", "plus")


def maybe_run_paper_suite(mode: str, run_map: Dict[str, Path]) -> bool:
    if mode == "none":
        return False
    if mode == "cohface":
        return True
    return "COHFACE" in run_map


def main() -> None:
    args = parse_args()
    cfg = load_config(str(args.config))
    results_root = Path(args.results or cfg.get("results_dir", "results")).expanduser()
    if not results_root.is_absolute():
        results_root = (ROOT / results_root).resolve()

    if not args.skip_estimate:
        cmd = [sys.executable, str(ROOT / "main.py"), "--config", str(args.config)]
        cmd += ["--results", str(results_root)]
        if args.debug:
            cmd.append("--debug")
        run_cmd(cmd)

    run_map = dataset_run_dirs(cfg, results_root)
    if not run_map:
        raise RuntimeError(f"Could not resolve run directories from config={args.config} results={results_root}")
    for run_dir in run_map.values():
        assert_run_completed(run_dir)

    metrics_map: Dict[str, Path] = {}
    for dataset_name, run_dir in run_map.items():
        metrics_map[dataset_name] = run_dir / "metrics"

    peer_metrics = parse_peer_metrics(args.peer_metrics)
    for dataset_name, metrics_dir in peer_metrics.items():
        if dataset_name not in metrics_map:
            metrics_map[dataset_name] = metrics_dir

    # Per-dataset postprocessing
    for dataset_name, run_dir in run_map.items():
        dataset_slug = manifest_prefix(dataset_name)
        data_dir = run_dir / "data"
        metrics_dir = run_dir / "metrics"
        trial_manifest = ROOT / "paper" / "manifests" / f"{dataset_slug}_parh_mechanism_trials.csv"
        family_table = ROOT / "paper" / "tables_ready" / f"T6b_{dataset_slug}_mechanism_audit.csv"

        run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "collect_parh_mechanism_audit.py"),
                "--data-dir",
                str(data_dir),
                "--trial-out",
                str(trial_manifest),
                "--family-out",
                str(family_table),
            ]
        )

        if not args.skip_eda:
            eda_trials = ROOT / "analysis" / f"{dataset_slug}_observation_eda_trials.csv"
            eda_family_stage = ROOT / "analysis" / f"{dataset_slug}_observation_eda_family_stage_summary.csv"
            eda_summary = ROOT / "analysis" / f"{dataset_slug}_preproc_summary.csv"
            eda_delta = ROOT / "analysis" / f"{dataset_slug}_preproc_deltas.csv"
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_observation_eda.py"),
                    "--data-dir",
                    str(data_dir),
                    "--config",
                    str(args.config),
                    "--dataset-name",
                    dataset_name,
                    "--trial-out",
                    str(eda_trials),
                    "--family-out",
                    str(eda_family_stage),
                ]
            )
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_preproc_eda.py"),
                    "--trial-csv",
                    str(eda_trials),
                    "--summary-out",
                    str(eda_summary),
                    "--delta-out",
                    str(eda_delta),
                ]
            )

        run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_case_study_manifest.py"),
                "--waveform-csv",
                str(metrics_dir / "metrics_waveform_raw.csv"),
                "--freq-csv",
                str(metrics_dir / "metrics_freq_domain_raw.csv"),
                "--dataset-name",
                dataset_name,
                "--out",
                str(ROOT / "paper" / "manifests" / f"{dataset_slug}_case_study_manifest.csv"),
            ]
        )

        if dataset_name == "COHFACE":
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_residual_case_manifest.py"),
                    "--mechanism-trials",
                    str(trial_manifest),
                    "--waveform-csv",
                    str(metrics_dir / "metrics_waveform_raw.csv"),
                    "--freq-csv",
                    str(metrics_dir / "metrics_freq_domain_raw.csv"),
                    "--dataset-name",
                    dataset_name,
                    "--out",
                    str(ROOT / "paper" / "manifests" / "cohface_residual_case_manifest.csv"),
                ]
            )
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_observation_construction_manifest.py"),
                    "--waveform-csv",
                    str(metrics_dir / "metrics_waveform_raw.csv"),
                    "--freq-csv",
                    str(metrics_dir / "metrics_freq_domain_raw.csv"),
                    "--dataset-name",
                    dataset_name,
                    "--out",
                    str(ROOT / "paper" / "manifests" / "cohface_ofbridge_case_manifest.csv"),
                ]
            )
            overlay_slug = family_to_manifest_slug(args.overlay_family)
            overlay_manifest = ROOT / "paper" / "manifests" / f"cohface_{overlay_slug}_overlay_manifest.csv"
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_waveform_overlay_manifest.py"),
                    "--waveform-csv",
                    str(metrics_dir / "metrics_waveform_raw.csv"),
                    "--freq-csv",
                    str(metrics_dir / "metrics_freq_domain_raw.csv"),
                    "--family",
                    args.overlay_family,
                    "--dataset-name",
                    dataset_name,
                    "--out",
                    str(overlay_manifest),
                ]
            )
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_failure_case_manifest.py"),
                    "--residual-manifest",
                    str(ROOT / "paper" / "manifests" / "cohface_residual_case_manifest.csv"),
                    "--bridge-manifest",
                    str(ROOT / "paper" / "manifests" / "cohface_ofbridge_case_manifest.csv"),
                    "--dataset-name",
                    dataset_name,
                    "--out",
                    str(ROOT / "paper" / "manifests" / "cohface_failure_case_manifest.csv"),
                ]
            )

    # Shared tables from explicit dataset mapping only.
    table_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_table_ready.py"),
        "--out-dir",
        str(ROOT / "paper" / "tables_ready"),
    ]
    for dataset_name, metrics_dir in sorted(metrics_map.items()):
        table_cmd.extend(["--dataset-metrics", f"{dataset_name}={metrics_dir}"])
    run_cmd(table_cmd)

    build_paper_suite = maybe_run_paper_suite(args.paper_suite, run_map)
    if build_paper_suite and "COHFACE" in run_map:
        coh_run_dir = run_map["COHFACE"]
        if not args.skip_eda:
            run_cmd(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "plot_observation_eda.py"),
                    "--summary-csv",
                    str(ROOT / "analysis" / "cohface_preproc_summary.csv"),
                    "--delta-csv",
                    str(ROOT / "analysis" / "cohface_preproc_deltas.csv"),
                    "--out-main",
                    str(ROOT / "paper" / "figures" / "F2_dataset_and_observation_regime.pdf"),
                    "--out-supp",
                    str(ROOT / "paper" / "figures" / "S_F5_preproc_delta_heatmaps.pdf"),
                ]
            )
        run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_main_family_figures.py"),
                "--t3-csv",
                str(ROOT / "paper" / "tables_ready" / "T3_rate_main.csv"),
                "--t6-csv",
                str(ROOT / "paper" / "tables_ready" / "T6_diagnostics_main.csv"),
                "--mech-csv",
                str(ROOT / "paper" / "tables_ready" / "T6b_cohface_mechanism_audit.csv"),
                "--out-f3",
                str(ROOT / "paper" / "figures" / "F3_t3_family_summary.pdf"),
                "--out-f5",
                str(ROOT / "paper" / "figures" / "F5_mechanism_activation.pdf"),
            ]
        )
        overlay_slug = family_to_manifest_slug(args.overlay_family)
        run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_waveform_overlay_grid.py"),
                "--manifest-csv",
                str(ROOT / "paper" / "manifests" / f"cohface_{overlay_slug}_overlay_manifest.csv"),
                "--run-dir",
                str(coh_run_dir),
                "--out",
                str(ROOT / "paper" / "figures" / "F4_waveform_overlay_grid.pdf"),
            ]
        )
        run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_failure_cases.py"),
                "--manifest-csv",
                str(ROOT / "paper" / "manifests" / "cohface_failure_case_manifest.csv"),
                "--run-dir",
                str(coh_run_dir),
                "--out",
                str(ROOT / "paper" / "figures" / "F6_failure_cases.pdf"),
            ]
        )
        run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "plot_of_construction_comparison.py"),
                "--t3-csv",
                str(ROOT / "paper" / "tables_ready" / "T3_rate_main.csv"),
                "--t4-csv",
                str(ROOT / "paper" / "tables_ready" / "T4_waveform_main.csv"),
                "--out-pdf",
                str(ROOT / "paper" / "figures" / "S_F6_of_construction_comparison.pdf"),
            ]
        )
        if not args.skip_pdf:
            if shutil.which("latexmk") is None:
                raise RuntimeError("latexmk not found in PATH; rerun with --skip-pdf or install latexmk.")
            run_cmd(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                cwd=ROOT / "paper",
            )

    summary = {
        "config": str(Path(args.config).resolve()),
        "results_root": str(results_root),
        "skip_estimate": bool(args.skip_estimate),
        "datasets_current_run": {k: str(v) for k, v in run_map.items()},
        "datasets_for_tables": {k: str(v) for k, v in metrics_map.items()},
        "paper_suite": args.paper_suite,
        "overlay_family": args.overlay_family,
        "skip_eda": bool(args.skip_eda),
        "skip_pdf": bool(args.skip_pdf),
    }
    summary_path = results_root / "e2e_bundle_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved bundle summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
