#!/usr/bin/env python3
"""Reusable OS and hardware resource profiler.

The module is intentionally dependency-free. It can be copied into a project's
``setup/`` directory and used from shell launchers or Python entrypoints:

    python setup/resource_profile.py --json
    eval "$(python setup/resource_profile.py --shell)"

It separates immutable machine facts from runtime policy. Machine facts are
useful for audit records; policy values are the environment knobs that vary by
project or workload.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from contextlib import redirect_stderr, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MODES = ("auto", "debug", "conservative", "gpu_safe", "balanced", "cpu_batch", "throughput")


@dataclass(frozen=True)
class OSInfo:
    system: str
    release: str
    version: str
    machine: str
    processor: str
    distro_name: str
    distro_version: str
    distro_id: str
    libc: str


@dataclass(frozen=True)
class PythonInfo:
    version: str
    executable: str
    implementation: str
    prefix: str
    base_prefix: str
    virtual_env: str


@dataclass(frozen=True)
class CPUInfo:
    logical_cpus: int
    physical_cores: int
    affinity_cpus: int
    model_name: str
    architecture: str
    current_mhz: float | None
    min_mhz: float | None
    max_mhz: float | None
    governor: str
    flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MemoryInfo:
    total_bytes: int
    available_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int


@dataclass(frozen=True)
class DiskInfo:
    path: str
    filesystem: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    percent_used: float


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_total_mb: int | None
    memory_free_mb: int | None
    driver_version: str
    uuid: str
    bus_id: str


@dataclass(frozen=True)
class TorchInfo:
    installed: bool
    version: str
    cuda_build: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_names: tuple[str, ...]
    bf16_supported: bool


@dataclass(frozen=True)
class ResourceProfile:
    generated_at_utc: str
    hostname: str
    user: str
    cwd: str
    repo_root: str
    os: OSInfo
    python: PythonInfo
    cpu: CPUInfo
    memory: MemoryInfo
    disks: tuple[DiskInfo, ...]
    nvidia_gpus: tuple[GPUInfo, ...]
    torch: TorchInfo
    selected_env: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimePolicy:
    requested_mode: str
    mode: str
    device: str
    cuda_visible_devices: str | None
    cpu_count: int
    usable_cpus: int
    physical_cores: int
    parallel_procs: int
    threads_per_proc: int
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    use_amp: bool
    amp_dtype: str
    opencv_num_threads: int
    omp_proc_bind: str
    omp_places: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_text(path: str | Path) -> str:
    with suppress(OSError, UnicodeDecodeError):
        return Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def _run(args: list[str], *, timeout: float = 2.0) -> str:
    with suppress(Exception):
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=timeout).strip()
    return ""


def _parse_float(raw: str) -> float | None:
    with suppress(ValueError, TypeError):
        return float(str(raw).strip())
    return None


def _read_khz_as_mhz(path: str | Path) -> float | None:
    raw = _read_text(path)
    if not raw:
        return None
    parsed = _parse_float(raw)
    return None if parsed is None else parsed / 1000.0


def _bytes_from_kib(value: str) -> int:
    match = re.search(r"(\d+)", value)
    if not match:
        return 0
    return int(match.group(1)) * 1024


def _detect_os() -> OSInfo:
    uname = platform.uname()
    distro: dict[str, str] = {}
    with suppress(Exception):
        distro = dict(platform.freedesktop_os_release())
    libc_name, libc_version = platform.libc_ver()
    libc = " ".join(part for part in (libc_name, libc_version) if part)
    return OSInfo(
        system=uname.system,
        release=uname.release,
        version=uname.version,
        machine=uname.machine,
        processor=uname.processor,
        distro_name=distro.get("PRETTY_NAME", "") or distro.get("NAME", ""),
        distro_version=distro.get("VERSION", "") or distro.get("VERSION_ID", ""),
        distro_id=distro.get("ID", ""),
        libc=libc,
    )


def _physical_cores_from_cpuinfo() -> tuple[int, str, float | None, tuple[str, ...]]:
    cpuinfo = _read_text("/proc/cpuinfo")
    if not cpuinfo:
        return 0, "", None, ()

    physical_ids: set[tuple[str, str]] = set()
    current_physical = ""
    current_core = ""
    model_name = ""
    current_mhz: float | None = None
    flags: tuple[str, ...] = ()

    for raw_line in cpuinfo.splitlines():
        line = raw_line.strip()
        if not line:
            if current_physical and current_core:
                physical_ids.add((current_physical, current_core))
            current_physical = ""
            current_core = ""
            continue
        if ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        if key == "physical id":
            current_physical = value
        elif key == "core id":
            current_core = value
        elif key == "model name" and not model_name:
            model_name = value
        elif key == "cpu MHz" and current_mhz is None:
            current_mhz = _parse_float(value)
        elif key in {"flags", "Features"} and not flags:
            flags = tuple(value.split())

    if current_physical and current_core:
        physical_ids.add((current_physical, current_core))
    return len(physical_ids), model_name, current_mhz, flags


def _detect_cpu() -> CPUInfo:
    logical = os.cpu_count() or 1
    affinity = logical
    with suppress(Exception):
        affinity = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    physical, model_name, current_mhz, flags = _physical_cores_from_cpuinfo()
    if physical <= 0:
        physical = max(1, logical // 2) if logical > 1 else 1
    min_mhz = _read_khz_as_mhz("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq")
    max_mhz = _read_khz_as_mhz("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    governor = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return CPUInfo(
        logical_cpus=int(logical),
        physical_cores=int(physical),
        affinity_cpus=max(1, int(affinity)),
        model_name=model_name,
        architecture=platform.machine(),
        current_mhz=current_mhz,
        min_mhz=min_mhz,
        max_mhz=max_mhz,
        governor=governor,
        flags=flags,
    )


def _detect_memory() -> MemoryInfo:
    data: dict[str, str] = {}
    meminfo = _read_text("/proc/meminfo")
    for line in meminfo.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return MemoryInfo(
        total_bytes=_bytes_from_kib(data.get("MemTotal", "0")),
        available_bytes=_bytes_from_kib(data.get("MemAvailable", data.get("MemFree", "0"))),
        swap_total_bytes=_bytes_from_kib(data.get("SwapTotal", "0")),
        swap_free_bytes=_bytes_from_kib(data.get("SwapFree", "0")),
    )


def _filesystem_type(path: Path) -> str:
    output = _run(["df", "-T", "-P", str(path)], timeout=1.0)
    lines = [line.split() for line in output.splitlines() if line.strip()]
    if len(lines) >= 2 and len(lines[1]) >= 2:
        return lines[1][1]
    return ""


def _detect_disks(paths: Iterable[Path]) -> tuple[DiskInfo, ...]:
    disks: list[DiskInfo] = []
    seen: set[str] = set()
    for path in paths:
        target = path.expanduser().resolve()
        if not target.exists():
            target = target.parent
        key = str(target)
        if key in seen:
            continue
        seen.add(key)
        with suppress(OSError):
            usage = shutil.disk_usage(target)
            percent = (usage.used / usage.total * 100.0) if usage.total else 0.0
            disks.append(
                DiskInfo(
                    path=key,
                    filesystem=_filesystem_type(target),
                    total_bytes=int(usage.total),
                    used_bytes=int(usage.used),
                    free_bytes=int(usage.free),
                    percent_used=round(percent, 2),
                )
            )
    return tuple(disks)


def _detect_nvidia_gpus() -> tuple[GPUInfo, ...]:
    if not shutil.which("nvidia-smi"):
        return ()
    output = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,driver_version,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        timeout=3.0,
    )
    if not output:
        return ()
    rows = csv.reader(output.splitlines())
    gpus: list[GPUInfo] = []
    for row in rows:
        if len(row) < 7:
            continue
        values = [value.strip() for value in row]
        with suppress(ValueError):
            gpus.append(
                GPUInfo(
                    index=int(values[0]),
                    name=values[1],
                    memory_total_mb=int(float(values[2])),
                    memory_free_mb=int(float(values[3])),
                    driver_version=values[4],
                    uuid=values[5],
                    bus_id=values[6],
                )
            )
    return tuple(gpus)


def _detect_torch() -> TorchInfo:
    try:
        with redirect_stderr(io.StringIO()):
            import torch  # type: ignore
    except Exception:
        return TorchInfo(False, "", "", False, 0, (), False)

    with redirect_stderr(io.StringIO()):
        version = str(getattr(torch, "__version__", ""))
        cuda_build = str(getattr(getattr(torch, "version", object()), "cuda", "") or "")
        cuda_available = False
        device_count = 0
        names: list[str] = []
        bf16_supported = False
        with suppress(Exception):
            cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            with suppress(Exception):
                device_count = int(torch.cuda.device_count())
            for index in range(device_count):
                with suppress(Exception):
                    names.append(str(torch.cuda.get_device_name(index)))
            with suppress(Exception):
                bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    return TorchInfo(
        installed=True,
        version=version,
        cuda_build=cuda_build,
        cuda_available=bool(cuda_available),
        cuda_device_count=int(device_count),
        cuda_device_names=tuple(names),
        bf16_supported=bool(bf16_supported),
    )


def _selected_env() -> dict[str, str]:
    exact = {
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "PYTORCH_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "OPENCV_NUM_THREADS",
        "NUM_WORKERS",
        "PREFETCH_FACTOR",
        "PERSISTENT_WORKERS",
        "PIN_MEMORY",
        "DEVICE",
        "AUTO_PROFILE_MODE",
        "RESOURCE_PROFILE_MODE",
        "THREADS_PER_PROC",
        "PARALLEL_PROCS",
    }
    prefixes = ("RESPYRE_", "SPHERICAL_RPPG_", "VOIDTOONE_", "DTT_")
    result: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if key in exact or key.startswith(prefixes):
            result[key] = value
    return result


def collect_resource_profile(*, repo_root: str | Path | None = None) -> ResourceProfile:
    cwd = Path.cwd()
    root = Path(repo_root).expanduser().resolve() if repo_root else cwd
    home = Path.home()
    return ResourceProfile(
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        hostname=socket.gethostname(),
        user=os.environ.get("USER", "") or os.environ.get("USERNAME", ""),
        cwd=str(cwd),
        repo_root=str(root),
        os=_detect_os(),
        python=PythonInfo(
            version=sys.version.replace("\n", " "),
            executable=sys.executable,
            implementation=platform.python_implementation(),
            prefix=sys.prefix,
            base_prefix=getattr(sys, "base_prefix", sys.prefix),
            virtual_env=os.environ.get("VIRTUAL_ENV", ""),
        ),
        cpu=_detect_cpu(),
        memory=_detect_memory(),
        disks=_detect_disks((root, cwd, home)),
        nvidia_gpus=_detect_nvidia_gpus(),
        torch=_detect_torch(),
        selected_env=_selected_env(),
    )


def _env_int(names: Iterable[str], default: int | None = None, *, minimum: int = 1) -> int | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        with suppress(ValueError):
            value = int(raw)
            if value >= minimum:
                return value
    return default


def _normalize_mode(mode: str | None) -> str:
    raw = (mode or os.environ.get("RESOURCE_PROFILE_MODE") or os.environ.get("AUTO_PROFILE_MODE") or "auto").strip()
    if raw not in MODES:
        raise ValueError(f"unknown runtime mode {raw!r}; expected one of: {', '.join(MODES)}")
    return raw


def _resolve_device(profile: ResourceProfile, requested: str | None) -> tuple[str, str | None]:
    raw = (requested or os.environ.get("RESOURCE_PROFILE_DEVICE") or os.environ.get("AUTO_PROFILE_DEVICE") or "auto").strip()
    normalized = raw.lower()
    cuda_available = profile.torch.cuda_available
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if normalized in {"", "auto"}:
        if cuda_available:
            return "cuda:0", None if cuda_visible else "0"
        return "cpu", None
    if normalized in {"cpu", "none"}:
        return "cpu", None
    if normalized.startswith("cuda"):
        if cuda_available:
            return ("cuda:0" if normalized == "cuda" else raw), None if cuda_visible else "0"
        return "cpu", None
    return raw, None


def _base_workers(cpu_count: int) -> int:
    if cpu_count >= 16:
        return min(16, max(6, cpu_count - 4))
    if cpu_count >= 8:
        return max(4, cpu_count - 2)
    return max(2, cpu_count - 1)


def _default_parallel(mode: str, usable: int, cuda_available: bool) -> int:
    if mode in {"debug", "gpu_safe", "conservative", "throughput"}:
        return 1
    if mode == "balanced":
        return 2 if usable >= 12 else 1
    if mode == "cpu_batch":
        return max(1, min(8, usable // 2))
    return 1 if cuda_available else max(1, min(8, usable // 2))


def _default_threads(mode: str, usable: int, parallel_procs: int) -> int:
    if mode == "debug":
        return 1
    if mode == "conservative":
        return min(4, max(1, usable // 6))
    if mode == "throughput":
        return min(8, max(2, usable // 3))
    cap = 4 if mode == "cpu_batch" else 8
    return max(1, min(cap, usable // max(1, parallel_procs)))


def _default_workers(mode: str, usable: int, threads_per_proc: int) -> int:
    if mode == "debug":
        return 0
    if mode == "conservative":
        return min(4, max(1, usable // 4))
    if mode == "throughput":
        return min(12, max(2, usable // 2))
    base = _base_workers(usable)
    if mode == "cpu_batch":
        return max(0, min(2, threads_per_proc - 1))
    if mode == "balanced":
        return min(base, max(2, threads_per_proc // 2))
    return min(base, max(2, threads_per_proc - 1))


def build_runtime_policy(
    profile: ResourceProfile | None = None,
    *,
    mode: str | None = None,
    device: str | None = None,
    parallel_procs: int | None = None,
    threads_per_proc: int | None = None,
    num_workers: int | None = None,
    parallel_env_names: Iterable[str] = (),
    thread_env_names: Iterable[str] = (),
    worker_env_names: Iterable[str] = (),
) -> RuntimePolicy:
    profile = profile or collect_resource_profile()
    requested_mode = _normalize_mode(mode)
    resolved_device, cuda_visible = _resolve_device(profile, device)
    resolved_mode = requested_mode
    if resolved_mode == "auto":
        resolved_mode = "gpu_safe" if resolved_device.startswith("cuda") else "cpu_batch"

    raw_cpu_count = max(1, int(profile.cpu.affinity_cpus or profile.cpu.logical_cpus or 1))
    reserve = max(1, int(round(raw_cpu_count * 0.10)))
    if raw_cpu_count >= 8:
        reserve = max(2, reserve)
    usable = max(1, raw_cpu_count - reserve)

    env_parallel = _env_int(
        [
            "RESOURCE_PROFILE_PARALLEL",
            "AUTO_PROFILE_PARALLEL",
            "PARALLEL_PROCS",
            "JOBS",
            "NPROC",
            "NUM_PROCS",
            *parallel_env_names,
        ],
        default=None,
    )
    resolved_parallel = (
        max(1, int(parallel_procs))
        if parallel_procs is not None
        else int(env_parallel)
        if env_parallel is not None
        else _default_parallel(resolved_mode, usable, profile.torch.cuda_available)
    )

    env_threads = _env_int(
        [
            "RESOURCE_PROFILE_THREADS",
            "AUTO_PROFILE_THREADS",
            "THREADS_PER_PROC",
            "TORCH_NUM_THREADS",
            "PYTORCH_NUM_THREADS",
            *thread_env_names,
        ],
        default=None,
    )
    resolved_threads = (
        max(1, int(threads_per_proc))
        if threads_per_proc is not None
        else int(env_threads)
        if env_threads is not None
        else _default_threads(resolved_mode, usable, resolved_parallel)
    )

    env_workers = _env_int(
        ["RESOURCE_PROFILE_NUM_WORKERS", "AUTO_PROFILE_NUM_WORKERS", "NUM_WORKERS", *worker_env_names],
        default=None,
        minimum=0,
    )
    resolved_workers = (
        max(0, int(num_workers))
        if num_workers is not None
        else int(env_workers)
        if env_workers is not None
        else _default_workers(resolved_mode, usable, resolved_threads)
    )

    prefetch = 0 if resolved_workers == 0 else (4 if resolved_mode == "throughput" else 2)
    persistent = bool(resolved_workers > 0)
    pin_memory = bool(resolved_device.startswith("cuda"))
    use_amp = bool(resolved_device.startswith("cuda"))
    amp_dtype = "bf16" if use_amp and profile.torch.bf16_supported else "fp16" if use_amp else "fp32"

    return RuntimePolicy(
        requested_mode=requested_mode,
        mode=resolved_mode,
        device=resolved_device,
        cuda_visible_devices=cuda_visible,
        cpu_count=int(profile.cpu.logical_cpus),
        usable_cpus=int(usable),
        physical_cores=int(profile.cpu.physical_cores),
        parallel_procs=int(resolved_parallel),
        threads_per_proc=int(resolved_threads),
        num_workers=int(resolved_workers),
        prefetch_factor=int(prefetch),
        persistent_workers=persistent,
        pin_memory=pin_memory,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        opencv_num_threads=1,
        omp_proc_bind="TRUE",
        omp_places="cores",
    )


def _bool_value(value: bool, style: str) -> str:
    if style == "int":
        return "1" if value else "0"
    return "true" if value else "false"


def shell_exports(
    policy: RuntimePolicy,
    *,
    bool_style: str = "int",
    project_prefix: str = "",
    project_workers_var: str = "",
) -> dict[str, str]:
    exports: dict[str, str] = {
        "RESOURCE_PROFILE_MODE": policy.mode,
        "RESOURCE_PROFILE_REQUESTED_MODE": policy.requested_mode,
        "CPU_COUNT": str(policy.cpu_count),
        "PHYSICAL_CORES": str(policy.physical_cores),
        "USABLE_CPUS": str(policy.usable_cpus),
        "PARALLEL_PROCS": str(policy.parallel_procs),
        "THREADS_PER_PROC": str(policy.threads_per_proc),
        "DEVICE": policy.device,
        "NUM_WORKERS": str(policy.num_workers),
        "PREFETCH_FACTOR": str(policy.prefetch_factor),
        "PERSISTENT_WORKERS": _bool_value(policy.persistent_workers, bool_style),
        "PIN_MEMORY": _bool_value(policy.pin_memory, bool_style),
        "USE_AMP": _bool_value(policy.use_amp, bool_style),
        "AMP_DTYPE": policy.amp_dtype,
        "OMP_NUM_THREADS": str(policy.threads_per_proc),
        "OPENBLAS_NUM_THREADS": str(policy.threads_per_proc),
        "MKL_NUM_THREADS": str(policy.threads_per_proc),
        "NUMEXPR_NUM_THREADS": str(policy.threads_per_proc),
        "BLIS_NUM_THREADS": str(policy.threads_per_proc),
        "VECLIB_MAXIMUM_THREADS": str(policy.threads_per_proc),
        "PYTORCH_NUM_THREADS": str(policy.threads_per_proc),
        "TORCH_NUM_THREADS": str(policy.threads_per_proc),
        "OPENCV_NUM_THREADS": str(policy.opencv_num_threads),
        "OMP_PROC_BIND": policy.omp_proc_bind,
        "OMP_PLACES": policy.omp_places,
    }
    if policy.cuda_visible_devices is not None:
        exports["CUDA_VISIBLE_DEVICES"] = policy.cuda_visible_devices
    prefix = project_prefix.strip().upper().rstrip("_")
    if prefix:
        exports[f"{prefix}_PROFILE_MODE"] = policy.mode
        exports[f"{prefix}_JOBS"] = str(policy.parallel_procs)
    if project_workers_var:
        exports[project_workers_var] = str(policy.parallel_procs)
    return exports


def format_shell_exports(exports: dict[str, str]) -> str:
    return "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in exports.items())


def apply_runtime_policy(policy: RuntimePolicy, *, force_env: bool = False, bool_style: str = "int") -> None:
    for key, value in shell_exports(policy, bool_style=bool_style).items():
        if force_env or not os.environ.get(key):
            os.environ[key] = str(value)
    with suppress(Exception):
        import cv2  # type: ignore

        cv2.setNumThreads(policy.opencv_num_threads)
    with suppress(Exception):
        import torch  # type: ignore

        torch.set_num_threads(policy.threads_per_proc)
        with suppress(Exception):
            torch.set_num_interop_threads(max(1, min(policy.threads_per_proc, 2)))


def _human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{num_bytes} B"


def format_summary(profile: ResourceProfile, policy: RuntimePolicy) -> str:
    gpu_names = ", ".join(gpu.name for gpu in profile.nvidia_gpus) or "none"
    torch_cuda = "yes" if profile.torch.cuda_available else "no"
    disk_lines = [
        f"  - {disk.path}: {_human_size(disk.free_bytes)} free / {_human_size(disk.total_bytes)} ({disk.filesystem or 'fs unknown'})"
        for disk in profile.disks
    ]
    return "\n".join(
        [
            f"generated_at_utc: {profile.generated_at_utc}",
            f"host: {profile.user}@{profile.hostname}",
            f"os: {profile.os.distro_name or profile.os.system} | kernel {profile.os.release} | {profile.os.machine}",
            f"python: {profile.python.version.split()[0]} | {profile.python.executable}",
            f"cpu: {profile.cpu.logical_cpus} logical / {profile.cpu.physical_cores} physical / {profile.cpu.affinity_cpus} affinity",
            f"cpu_model: {profile.cpu.model_name or profile.cpu.architecture}",
            f"memory: {_human_size(profile.memory.available_bytes)} available / {_human_size(profile.memory.total_bytes)} total",
            f"nvidia_gpus: {gpu_names}",
            f"torch: installed={profile.torch.installed} version={profile.torch.version or 'n/a'} cuda_available={torch_cuda} cuda_build={profile.torch.cuda_build or 'n/a'}",
            "disks:",
            *(disk_lines or ["  - none detected"]),
            "policy:",
            f"  mode={policy.mode} requested={policy.requested_mode} device={policy.device}",
            f"  parallel_procs={policy.parallel_procs} threads_per_proc={policy.threads_per_proc}",
            f"  num_workers={policy.num_workers} prefetch_factor={policy.prefetch_factor} persistent_workers={policy.persistent_workers}",
            f"  pin_memory={policy.pin_memory} use_amp={policy.use_amp} amp_dtype={policy.amp_dtype}",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect OS/hardware facts and emit runtime policy exports.")
    parser.add_argument("--mode", choices=MODES, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--parallel-procs", type=int, default=None)
    parser.add_argument("--threads-per-proc", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--json", action="store_true", help="Print full profile and policy JSON.")
    parser.add_argument("--shell", action="store_true", help="Print shell exports for eval.")
    parser.add_argument("--write-json", default="", help="Write full profile and policy JSON to this path.")
    parser.add_argument("--project-prefix", default="", help="Also emit <PREFIX>_PROFILE_MODE and <PREFIX>_JOBS.")
    parser.add_argument("--project-workers-var", default="", help="Also emit this variable with parallel_procs.")
    parser.add_argument("--bool-style", choices=("int", "lower"), default="int")
    parser.add_argument("--quiet", action="store_true", help="Suppress shell-mode diagnostics on stderr.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    profile = collect_resource_profile(repo_root=args.repo_root)
    policy = build_runtime_policy(
        profile,
        mode=args.mode,
        device=args.device,
        parallel_procs=args.parallel_procs,
        threads_per_proc=args.threads_per_proc,
        num_workers=args.num_workers,
    )
    payload = {"profile": profile.to_dict(), "policy": policy.to_dict()}

    if args.write_json:
        path = Path(args.write_json).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.shell:
        exports = shell_exports(
            policy,
            bool_style=args.bool_style,
            project_prefix=args.project_prefix,
            project_workers_var=args.project_workers_var,
        )
        print(format_shell_exports(exports))
        if not args.quiet:
            print(
                f"[resource_profile] mode={policy.mode} device={policy.device} "
                f"parallel={policy.parallel_procs} threads={policy.threads_per_proc} workers={policy.num_workers}",
                file=sys.stderr,
            )
    else:
        print(format_summary(profile, policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
