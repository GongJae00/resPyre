## Setup notes

- `setup.sh` installs a minimal stack (numpy/scipy/pandas/h5py via conda, rest via pip) for the motion+oscillator pipeline.
- No deep/rPPG/vendor packages are installed.
- `auto_profile.py` is the single runtime-profile entry point. It emits shell
  exports for device, AMP, DataLoader workers, OpenMP/BLAS/OpenCV/PyTorch
  thread caps, and process-level job counts.
- `resource_profile.py` records detailed OS, Python, CPU, memory, disk, NVIDIA
  GPU, and PyTorch CUDA facts so experiment logs can include the exact machine
  context.
- `hardware.py` exposes the same information as a small Python API for new
  entrypoints.
- Use `--verify` to run a light import check after install:
  ```bash
  ./setup/setup.sh --verify
  ```

## Runtime profiles

Always apply one runtime profile before large runs:

```bash
eval "$(python setup/auto_profile.py)"
```

Available modes:

```bash
# Conservative default for CUDA training/inference.
eval "$(python setup/auto_profile.py --mode gpu_safe)"

# Mixed CPU/GPU work; modest process-level parallelism.
eval "$(python setup/auto_profile.py --mode balanced)"

# CPU-heavy batch work such as trial/video-level audits, metrics, and figures.
eval "$(python setup/auto_profile.py --mode cpu_batch)"

# Final paper-package runs are CPU-batch oriented; pin CPU to avoid GPU contention.
eval "$(python setup/auto_profile.py --mode cpu_batch --device cpu --write-json analysis/final_resource_profile.json)"
```

The default `--mode auto` selects `gpu_safe` when CUDA is available and
`cpu_batch` otherwise. On a 24-thread workstation this typically yields:

| mode | exported process count | exported threads per process | intent |
|---|---:|---:|---|
| `gpu_safe` | `PARALLEL_PROCS=1` | `THREADS_PER_PROC=8` | avoid CPU/GPU oversubscription |
| `balanced` | `PARALLEL_PROCS=2` | `THREADS_PER_PROC=8` | mixed workloads |
| `cpu_batch` | `PARALLEL_PROCS=8` | `THREADS_PER_PROC=2` | independent CPU jobs |

Manual overrides are centralized here too:

```bash
python setup/auto_profile.py --mode cpu_batch --parallel-procs 6 --threads-per-proc 3
eval "$(AUTO_PROFILE_NUM_WORKERS=0 python setup/auto_profile.py --mode cpu_batch)"
```

Hardware audit:

```bash
python setup/auto_profile.py --summary
python setup/auto_profile.py --json > analysis/resource_profile.json
python -m setup.hardware --json
```
