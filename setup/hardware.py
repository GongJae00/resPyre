"""Small Python API around the shared OS/hardware resource profiler."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

try:
    from .resource_profile import (
        RuntimePolicy,
        apply_runtime_policy,
        build_runtime_policy,
        collect_resource_profile,
    )
except ImportError:  # pragma: no cover - allows `python setup/hardware.py`
    from resource_profile import (  # type: ignore
        RuntimePolicy,
        apply_runtime_policy,
        build_runtime_policy,
        collect_resource_profile,
    )


@dataclass(frozen=True, slots=True)
class HardwareSettings:
    os_name: str
    kernel: str
    distro: str
    logical_cpus: int
    physical_cores: int
    affinity_cpus: int
    memory_total_bytes: int
    memory_available_bytes: int
    cuda_available: bool
    gpu_names: tuple[str, ...]
    torch_version: str
    torch_cuda_build: str
    mode: str
    device: str
    parallel_procs: int
    threads_per_proc: int
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    use_amp: bool
    amp_dtype: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["gpu_names"] = list(self.gpu_names)
        return payload


def load_hardware_settings(
    *,
    mode: str | None = None,
    device: str | None = None,
    parallel_procs: int | None = None,
    threads_per_proc: int | None = None,
    num_workers: int | None = None,
) -> HardwareSettings:
    profile = collect_resource_profile()
    policy = build_runtime_policy(
        profile,
        mode=mode,
        device=device,
        parallel_procs=parallel_procs,
        threads_per_proc=threads_per_proc,
        num_workers=num_workers,
    )
    return HardwareSettings(
        os_name=profile.os.system,
        kernel=profile.os.release,
        distro=profile.os.distro_name,
        logical_cpus=profile.cpu.logical_cpus,
        physical_cores=profile.cpu.physical_cores,
        affinity_cpus=profile.cpu.affinity_cpus,
        memory_total_bytes=profile.memory.total_bytes,
        memory_available_bytes=profile.memory.available_bytes,
        cuda_available=profile.torch.cuda_available,
        gpu_names=tuple(gpu.name for gpu in profile.nvidia_gpus) or tuple(profile.torch.cuda_device_names),
        torch_version=profile.torch.version,
        torch_cuda_build=profile.torch.cuda_build,
        mode=policy.mode,
        device=policy.device,
        parallel_procs=policy.parallel_procs,
        threads_per_proc=policy.threads_per_proc,
        num_workers=policy.num_workers,
        prefetch_factor=policy.prefetch_factor,
        persistent_workers=policy.persistent_workers,
        pin_memory=policy.pin_memory,
        use_amp=policy.use_amp,
        amp_dtype=policy.amp_dtype,
    )


def configure_runtime_environment(*, force_env: bool = False, policy: RuntimePolicy | None = None) -> RuntimePolicy:
    resolved = policy or build_runtime_policy(collect_resource_profile())
    apply_runtime_policy(resolved, force_env=force_env)
    return resolved


def apply_process_thread_limits(thread_count: int, *, force_env: bool = False) -> None:
    policy = build_runtime_policy(collect_resource_profile(), mode="balanced", threads_per_proc=max(1, int(thread_count)), num_workers=0)
    apply_runtime_policy(policy, force_env=force_env)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print hardware settings derived from setup.resource_profile")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--full-json", action="store_true", help="Print full raw profile and runtime policy.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.full_json:
        profile = collect_resource_profile()
        policy = build_runtime_policy(profile)
        print(json.dumps({"profile": profile.to_dict(), "policy": policy.to_dict()}, indent=2, ensure_ascii=False))
        return 0
    settings = load_hardware_settings()
    if args.json:
        print(json.dumps(settings.to_dict(), indent=2, ensure_ascii=False))
    else:
        for key, value in settings.to_dict().items():
            print(f"{key}\t{value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
