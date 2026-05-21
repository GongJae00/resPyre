#!/usr/bin/env python3
"""Central runtime environment profiler for ResPyre.

Usage:
    eval "$(python setup/auto_profile.py)"
    eval "$(python setup/auto_profile.py --mode gpu_safe)"
    eval "$(AUTO_PROFILE_MODE=cpu_batch python setup/auto_profile.py)"

stdout contains shell exports by default, so it is safe to wrap with ``eval``.
Human-readable diagnostics are printed to stderr. Use ``--json`` when you want
the full OS/hardware audit record instead of shell exports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from .resource_profile import (
        MODES,
        build_runtime_policy,
        collect_resource_profile,
        format_shell_exports,
        format_summary,
        shell_exports,
    )
except ImportError:  # pragma: no cover - allows `python setup/auto_profile.py`
    from resource_profile import (  # type: ignore
        MODES,
        build_runtime_policy,
        collect_resource_profile,
        format_shell_exports,
        format_summary,
        shell_exports,
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected positive integer, got {value!r}")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit shell exports for a hardware-aware ResPyre runtime profile. "
            "Use with: eval \"$(python setup/auto_profile.py)\""
        )
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=os.environ.get("AUTO_PROFILE_MODE", "auto"),
        help=(
            "Runtime profile. auto picks gpu_safe when CUDA is available and "
            "cpu_batch otherwise. debug/conservative/gpu_safe/balanced/"
            "cpu_batch/throughput are available for explicit runs."
        ),
    )
    parser.add_argument("--parallel-procs", type=_positive_int, default=None, help="Override PARALLEL_PROCS/RESPYRE_JOBS.")
    parser.add_argument("--threads-per-proc", type=_positive_int, default=None, help="Override OMP/BLAS/PyTorch threads.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override NUM_WORKERS. Use 0 to disable workers.")
    parser.add_argument("--device", default=os.environ.get("AUTO_PROFILE_DEVICE", "auto"), help="Device override.")
    parser.add_argument("--json", action="store_true", help="Print full OS/hardware profile and policy JSON.")
    parser.add_argument("--summary", action="store_true", help="Print a human-readable profile summary.")
    parser.add_argument("--write-json", default="", help="Write full profile and policy JSON to this path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    profile = collect_resource_profile(repo_root=repo_root)
    policy = build_runtime_policy(
        profile,
        mode=args.mode,
        device=args.device,
        parallel_procs=args.parallel_procs,
        threads_per_proc=args.threads_per_proc,
        num_workers=args.num_workers,
        parallel_env_names=("RESPYRE_JOBS",),
    )
    payload = {"profile": profile.to_dict(), "policy": policy.to_dict()}

    if args.write_json:
        path = Path(args.write_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if args.summary:
        print(format_summary(profile, policy))
        return 0

    exports = shell_exports(policy, bool_style="int", project_prefix="RESPYRE")
    exports["AUTO_PROFILE_MODE"] = policy.mode
    print(format_shell_exports(exports))
    sys.stderr.write(
        "[auto_profile] Applied:\n"
        f"  MODE={policy.mode}  CPU_COUNT={policy.cpu_count}  PHYSICAL_CORES={policy.physical_cores}\n"
        f"  DEVICE={policy.device}\n"
        f"  NUM_WORKERS={policy.num_workers}  PIN_MEMORY={int(policy.pin_memory)}\n"
        f"  PERSISTENT_WORKERS={int(policy.persistent_workers)}  PREFETCH_FACTOR={policy.prefetch_factor}\n"
        f"  USE_AMP={int(policy.use_amp)}  AMP_DTYPE={policy.amp_dtype}\n"
        f"  PARALLEL_PROCS={policy.parallel_procs}  THREADS_PER_PROC={policy.threads_per_proc}\n"
        "  OMP/BLAS/NumExpr/OpenCV/PyTorch threads capped for safe parallel runs.\n"
    )
    if not profile.torch.installed:
        sys.stderr.write("[auto_profile] torch is not available for this interpreter; defaulted to CPU heuristics.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
