#!/usr/bin/env python3
"""Prune obsolete result directories and bulky completed-run subfolders."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

KEEP_RUNS = {
    "final_inputs",
    "final_validation",
    "20260409_cohface_prod_ofbridge_dofbridge_p1dcons_e2e_policy_narrow",
    "20260409_mahnob_prod_ofbridge_dofbridge_p1dcons_e2e",
    "mahnob_gt_tailaligned_patch_v1",
    "full_decoupled_validation_suite_v2",
    "strict_robustness_probe_v1",
    "20260408_mahnob_subset_ofbridge_gate",
    "20260407_cohface_pairfusion_prod",
    "cohface_rate_supervised_routed_full_gate_v1",
    "cohface_decoupled_system_gate_v2",
    "cohface_decoupled_hybrid_reference_v3",
    "two_stage_decoupled_matrix_v2",
    "target_reliability_graph_parh_probe",
    "observation_law_v2_probe",
    "preservation_safe_integration_probe",
}

PRUNE_SUBDIRS = {"aux", "logs", "plots"}


def run_status_for(run_dir: Path) -> dict | None:
    status_files = list(run_dir.glob("*/run_status.json"))
    if not status_files:
        return None
    try:
        return json.loads(status_files[0].read_text())
    except Exception:
        return None


def _parse_iso8601(ts: str | None):
    if not ts:
        return None
    try:
        if str(ts).endswith('Z'):
            ts = str(ts)[:-1] + '+00:00'
        return datetime.fromisoformat(str(ts))
    except Exception:
        return None


def is_running(run_dir: Path, stale_minutes: float = 180.0, allow_stale_running: bool = False) -> bool:
    status = run_status_for(run_dir)
    if not status:
        return False
    if status.get('status') != 'running':
        return False
    if not allow_stale_running:
        return True
    hb = _parse_iso8601(status.get('heartbeat_at') or status.get('updated_at') or status.get('completed_at'))
    if hb is None:
        return True
    age_min = (datetime.now(timezone.utc) - hb.astimezone(timezone.utc)).total_seconds() / 60.0
    return age_min <= float(stale_minutes)


def first_payload_dir(run_dir: Path) -> Path | None:
    subs = [p for p in run_dir.iterdir() if p.is_dir()]
    return subs[0] if len(subs) == 1 else None


def bytes_human(num: int) -> str:
    units = ["B", "K", "M", "G", "T"]
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num}B"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune obsolete PARH result directories.")
    parser.add_argument("--apply", action="store_true", help="Actually delete files instead of dry-run.")
    parser.add_argument("--stale-minutes", type=float, default=180.0, help="Heartbeat age threshold used only with --allow-stale-running.")
    parser.add_argument("--allow-stale-running", action="store_true", help="Allow stale running runs to be treated as non-active for pruning.")
    args = parser.parse_args()

    deleted_runs: list[tuple[str, int]] = []
    pruned_subdirs: list[tuple[str, int]] = []

    for run_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        name = run_dir.name
        payload_dir = first_payload_dir(run_dir)
        running = is_running(
            run_dir,
            stale_minutes=float(args.stale_minutes),
            allow_stale_running=bool(args.allow_stale_running),
        )

        if name not in KEEP_RUNS and not running:
            size = dir_size(run_dir)
            deleted_runs.append((name, size))
            if args.apply:
                shutil.rmtree(run_dir)
            continue

        if name in KEEP_RUNS and not running and payload_dir is not None:
            for sub in PRUNE_SUBDIRS:
                target = payload_dir / sub
                if target.exists():
                    size = dir_size(target)
                    pruned_subdirs.append((f"{name}/{payload_dir.name}/{sub}", size))
                    if args.apply:
                        shutil.rmtree(target)

    if args.allow_stale_running:
        print(f"Stale-running override enabled: heartbeat threshold {float(args.stale_minutes):.1f} min")
    else:
        print("Running-status protection enabled: any run with status=running is preserved")
    print("Deleted whole runs:")
    if deleted_runs:
        for name, size in deleted_runs:
            print(f"  {name}: {bytes_human(size)}")
    else:
        print("  (none)")

    print("Pruned subdirectories:")
    if pruned_subdirs:
        for name, size in pruned_subdirs:
            print(f"  {name}: {bytes_human(size)}")
    else:
        print("  (none)")

    total_saved = sum(size for _, size in deleted_runs) + sum(size for _, size in pruned_subdirs)
    print(f"Estimated bytes removed: {bytes_human(total_saved)}")


if __name__ == "__main__":
    main()
