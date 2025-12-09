#!/usr/bin/env bash
set -euo pipefail

# Minimal setup for the paper build (motion + oscillator heads only).
# Creates a conda env and installs requirements via conda+pip.

ENV_NAME="resPyre"
PY_VER="3.10"
VERIFY=0

usage() {
  cat << EOF
Usage: ./setup/setup.sh [options]

Options:
  -n, --name NAME    Conda env name (default: ${ENV_NAME})
  -p, --python VER   Python version (default: ${PY_VER})
      --verify       Import-check a few key packages after install
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name) ENV_NAME="$2"; shift 2;;
    -p|--python) PY_VER="$2"; shift 2;;
    --verify) VERIFY=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "[setup] Unknown option: $1"; usage; exit 1;;
  esac
done

echo "[setup] Env: name=${ENV_NAME}, python=${PY_VER}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] conda not found. Please install Miniconda/Conda and re-run." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] Creating env '${ENV_NAME}' with Python ${PY_VER}..."
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
else
  echo "[setup] Using existing env '${ENV_NAME}'"
fi

conda activate "${ENV_NAME}"

echo "[setup] Installing core numeric stack via conda (numpy/scipy/pandas/matplotlib/h5py)..."
conda install -y -c conda-forge "numpy=1.24.*" "scipy=1.10.*" "pandas=1.5.*" "matplotlib=3.7.*" "h5py=3.9.*"

python -m pip install --upgrade "pip<24.1" setuptools wheel
python -m pip install -r setup/requirements.txt

python -m pip check || echo "[setup] pip check reported potential conflicts (above)."

if [[ "${VERIFY}" -eq 1 ]]; then
  echo "[setup] Running post-install import check..."
  python - <<'PY'
mods = ['numpy','scipy','cv2','mediapipe','skimage','tqdm','PIL']
ok = True
for m in mods:
    try:
        __import__(m)
        print('[ok]', m)
    except Exception as e:
        ok = False
        print('[fail]', m, e)
if not ok:
    raise SystemExit(1)
PY
fi

echo "[setup] Done. Activate with: conda activate ${ENV_NAME}"
