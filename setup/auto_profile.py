#!/usr/bin/env python3
"""
setup/auto_profile.py

Usage:
    AUTO_PROFILE_PARALLEL=5 eval "$(python setup/auto_profile.py)"
    eval "$(python setup/auto_profile.py)"

What it does:
  - Detect GPU & choose DEVICE
  - Set sensible NUM_WORKERS/PIN_MEMORY/PERSISTENT_WORKERS/PREFETCH_FACTOR
  - Decide AMP + AMP_DTYPE (bf16 if supported)
What it NEVER does:
  - NO window/stride exports (handled inside src/* only)
  - Does not set SEED (utils.DEFAULT_SEED governs; user may export SEED manually)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys

def _print_export(k: str, v: str | int):
    print(f'export {k}="{v}"')

def _int_from_env(keys: list[str], default: int) -> int:
    for k in keys:
        v = os.environ.get(k, None)
        if v is None:
            continue
        try:
            n = int(v)
            if n >= 1:
                return n
        except Exception:
            pass
    return default

def main():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        torch = None  # type: ignore
        torch_missing = True
    else:
        torch_missing = False

    # --- CPU-based knobs ---
    cpu_cnt = max(1, mp.cpu_count())
    # worker: 남는 코어 2개 정도는 OS/로깅용으로 남김
    if cpu_cnt >= 16:
        num_workers = min(16, max(6, cpu_cnt - 4))
    elif cpu_cnt >= 8:
        num_workers = max(4, cpu_cnt - 2)
    else:
        num_workers = max(2, cpu_cnt - 1)

    # --- Parallel process awareness (limit inner threads safely) ---
    # 우선순위: AUTO_PROFILE_PARALLEL, NUM_SHARDS, N_SHARDS, PARALLEL_PROCS, JOBS, NPROC, NUM_PROCS
    parallel_procs = _int_from_env(
        ["AUTO_PROFILE_PARALLEL", "NUM_SHARDS", "N_SHARDS", "PARALLEL_PROCS", "JOBS", "NPROC", "NUM_PROCS"],
        default=1,
    )
    # OS 여유 코어 예약(10% 또는 최소 2코어)
    reserve = max(2, int(round(cpu_cnt * 0.10)))
    # 프로세스당 스레드: 과도한 오버구독 방지 (상한 8로 캡)
    threads_per_proc = max(1, (cpu_cnt - reserve) // max(1, parallel_procs))
    threads_per_proc = min(threads_per_proc, 8)
    # DataLoader 워커도 스레드 예산 내에서 조정
    num_workers = min(num_workers, max(2, threads_per_proc - 1))
    
    # --- GPU/AMP detection (best-effort, no hard failure) ---
    bf16_ok = False
    device = "cpu"
    try:
        if not torch_missing and torch.cuda.is_available():
            # honor user-provided CUDA_VISIBLE_DEVICES if any
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if cvd is None:
                # default to GPU 0
                _print_export("CUDA_VISIBLE_DEVICES", "0")
            device = "cuda:0"
            # bf16 지원 여부
            try:
                bf16_ok = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            except Exception:
                bf16_ok = False
        else:
            device = "cpu"
    except Exception:
        pass

    _print_export("DEVICE", device)

    # --- Dataloader perf knobs ---
    _print_export("NUM_WORKERS", num_workers)
    _print_export("PREFETCH_FACTOR", 2 if num_workers > 0 else 0)
    _print_export("PERSISTENT_WORKERS", 1 if num_workers > 0 else 0)
    _print_export("PIN_MEMORY", 1 if device.startswith("cuda") else 0)

    # --- Thread caps for BLAS/OMP/OpenCV/PyTorch (safe defaults for parallel procs) ---
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "PYTORCH_NUM_THREADS",
    ):
        _print_export(k, threads_per_proc)
    # 바인딩/플레이스 힌트(성능 안정화; 미지원 환경에서는 무해)
    _print_export("OMP_PROC_BIND", "TRUE")
    _print_export("OMP_PLACES", "cores")
    # OpenCV 내부스레드 억제
    _print_export("OPENCV_NUM_THREADS", 1)
    
    # --- AMP policy ---
    if device.startswith("cuda"):
        use_amp = 1
        amp_dtype = "bf16" if bf16_ok else "fp16"
    else:
        use_amp = 0
        amp_dtype = "fp32"
    _print_export("USE_AMP", use_amp)
    _print_export("AMP_DTYPE", amp_dtype)

    # FYI only (stdout 메시지) — 윈도우/스트라이드 언급 금지
    sys.stderr.write(
        "[auto_profile] Applied:\n"
        f"  DEVICE={device}\n"
        f"  NUM_WORKERS={num_workers}  PIN_MEMORY={1 if device.startswith('cuda') else 0}\n"
        f"  PERSISTENT_WORKERS={1 if num_workers > 0 else 0}  PREFETCH_FACTOR={2 if num_workers > 0 else 0}\n"
        f"  USE_AMP={use_amp}  AMP_DTYPE={amp_dtype}\n"
        f"  PARALLEL_PROCS={parallel_procs}  THREADS_PER_PROC={threads_per_proc}\n"
        "  OMP/BLAS/NumExpr/OpenCV/PyTorch threads capped for safe parallel runs.\n"
    )
    if torch_missing:
        sys.stderr.write("[auto_profile] torch not available for this interpreter; defaulted to CPU heuristics.\n")


if __name__ == "__main__":
    main()