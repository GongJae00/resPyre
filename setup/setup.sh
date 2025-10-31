#!/usr/bin/env bash
set -euo pipefail

# setup/setup.sh
# Portable environment setup for resPyre with conda and pip.
# - Creates a conda env (default: resPyre)
# - Installs unified requirements (motion + deep + SOMATA deps)
# - Optional: SOMATA local install, CmdStan toolchain
# - CPU-only 또는 CUDA wheel 자동/수동 선택

# Defaults tuned to your environment (Ubuntu 24.04, CUDA 12.x)
ENV_NAME="resPyre"
# 기본 Python은 3.10 (사용자 선호 및 호환성 기준)
PY_VER="3.10"
WITH_SOMATA=1
INSTALL_CMDSTAN=0
CPU_ONLY=0
CUDA_VERSION=""
PT_TORCH_INDEX=""
TORCH_VERSION="1.13.1"
VERIFY=0

usage() {
  cat << EOF
Usage: ./setup/setup.sh [options]

Options:
  -n, --name NAME          Conda env name (default: ${ENV_NAME})
  -p, --python VER         Python version (default: ${PY_VER})
      --cpu-only           Force CPU-only installs (torch CPU wheels)
      --cuda VER           CUDA major+minor (e.g., 117, 118, 121)
      --no-somata          Skip installing local somata package
      --install-cmdstan    Install CmdStan via cmdstanpy (may take time)
      --verify             Import-check key packages after install
  -h, --help               Show this help

Examples:
  # 통합 설치(GPU 자동 감지, PyTorch 1.13.1 cu117 매핑)
  ./setup/setup.sh

  # CmdStan 포함
  ./setup/setup.sh --install-cmdstan
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name) ENV_NAME="$2"; shift 2;;
    -p|--python) PY_VER="$2"; shift 2;;
    --cpu-only) CPU_ONLY=1; shift;;
    --cuda) CUDA_VERSION="$2"; CPU_ONLY=0; shift 2;;
    --no-somata) WITH_SOMATA=0; shift;;
    --install-cmdstan) INSTALL_CMDSTAN=1; shift;;
    --verify) VERIFY=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "[setup] Unknown option: $1"; usage; exit 1;;
  esac
done

echo "[setup] Env: name=${ENV_NAME}, python=${PY_VER}"
echo "[setup] Flags: cpu_only=${CPU_ONLY}, cuda=${CUDA_VERSION:-auto}, somata=${WITH_SOMATA}, cmdstan=${INSTALL_CMDSTAN}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[setup] Non-Linux OS detected. Some wheels (mediapipe/tensorflow) may not be available." >&2
fi

# Ensure conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] conda not found. Please install Miniconda/Conda and re-run." >&2
  exit 1
fi

# Activate conda for this shell
eval "$(conda shell.bash hook)"

# Create env if missing
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] Creating env '${ENV_NAME}' with Python ${PY_VER}..."
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
else
  echo "[setup] Using existing env '${ENV_NAME}'"
fi

conda activate "${ENV_NAME}"

# Install core scientific stack via conda to avoid binary incompatibilities
echo "[setup] Installing core numeric stack via conda (numpy/scipy/pandas/matplotlib/h5py)..."
conda install -y -c conda-forge "numpy=1.24.*" "scipy=1.10.*" "pandas=1.5.*" "matplotlib=3.7.*" "h5py=3.9.*"

# Upgrade pip tooling (ptlflow==0.2.7 requires pip <24.1)
# 일부 환경에서 pip가 가상환경 강제를 요구(PIP_REQUIRE_VIRTUALENV)할 수 있으므로 일시 해제
PREV_PIP_REQUIRE_VIRTUALENV=${PIP_REQUIRE_VIRTUALENV-}
PREV_PIP_REQUIRE_VENV=${PIP_REQUIRE_VENV-}
unset PIP_REQUIRE_VIRTUALENV || true
unset PIP_REQUIRE_VENV || true

python -m pip install --upgrade "pip<24.1" setuptools wheel

pick_cuda_tag() {
  local tag="cpu"
  if [[ ${CPU_ONLY} -eq 1 ]]; then echo "cpu"; return; fi
  if [[ -n "${CUDA_VERSION}" ]]; then
    # normalize (e.g., 118 -> cu118, cu118 -> cu118)
    if [[ "${CUDA_VERSION}" =~ ^cu ]]; then
      echo "${CUDA_VERSION}"
    else
      echo "cu${CUDA_VERSION}"
    fi
    return
  fi
  # auto-detect
  if command -v nvcc >/dev/null 2>&1; then
    local ver
    ver=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1\2/p' | tail -n1)
    if [[ -n "${ver}" ]]; then
      echo "cu${ver}"
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local nver
    nver=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1\2/p' | head -n1)
    if [[ -n "${nver}" ]]; then
      echo "cu${nver}"
      return
    fi
  fi
  echo "cpu"
}

map_torch_index() {
  local torch_ver="$1"; shift
  local cu_tag="$1"; shift
  # defaults
  if [[ "${cu_tag}" == "cpu" ]]; then
    echo "https://download.pytorch.org/whl/cpu"; return
  fi
  # Torch 1.13.1 supports cu116/cu117. Map others to cu117 with warning.
  if [[ "${torch_ver}" == 1.13.* ]]; then
    case "${cu_tag}" in
      cu116|cu117) echo "https://download.pytorch.org/whl/${cu_tag}";;
      *) echo "[setup] Warning: torch ${torch_ver} does not publish ${cu_tag} wheels; using cu117 instead." >&2; echo "https://download.pytorch.org/whl/cu117";;
    esac
    return
  fi
  # Fallback for newer torch (not default here)
  case "${cu_tag}" in
    cu118|cu121|cu124) echo "https://download.pytorch.org/whl/${cu_tag}";;
    *) echo "https://download.pytorch.org/whl/cpu";;
  esac
}

CU_TAG=$(pick_cuda_tag)
PT_TORCH_INDEX=$(map_torch_index "${TORCH_VERSION}" "${CU_TAG}")
echo "[setup] CUDA tag: ${CU_TAG}  (pytorch index: ${PT_TORCH_INDEX})"

echo "[setup] Installing unified requirements..."
python -m pip install --extra-index-url "${PT_TORCH_INDEX}" -r setup/requirements.txt

python -m pip check || echo "[setup] pip check reported potential conflicts (above)."

if [[ "${WITH_SOMATA}" -eq 1 ]]; then
  if [[ -d external/somata-main ]]; then
    echo "[setup] Installing local somata package (editable)..."
    python -m pip install -e external/somata-main
  else
    echo "[setup] Skipping somata install (external/somata-main not found)"
  fi

  if [[ "${INSTALL_CMDSTAN}" -eq 1 ]]; then
    echo "[setup] Installing CmdStan via cmdstanpy (this can take a while)..."
    python - <<'PY'
import cmdstanpy
try:
    cmdstanpy.install_cmdstan()
    print('[setup] CmdStan installed successfully')
except Exception as e:
    print('[setup] CmdStan install failed:', e)
    raise
PY
  fi
fi

# 설치 중 해제했던 pip 가상환경 강제 설정 복구
if [[ -n "${PREV_PIP_REQUIRE_VIRTUALENV}" ]]; then export PIP_REQUIRE_VIRTUALENV="${PREV_PIP_REQUIRE_VIRTUALENV}"; fi
if [[ -n "${PREV_PIP_REQUIRE_VENV}" ]]; then export PIP_REQUIRE_VENV="${PREV_PIP_REQUIRE_VENV}"; fi

echo "[setup] Done. Activate the env with: conda activate ${ENV_NAME}"
echo "[setup] Optional runtime tuning: eval \"$(python setup/auto_profile.py)\""

echo "[setup] Notes:"
  echo "  - TensorFlow/TVM은 기본 미설치(필요 시 별도 설치 권장; TF 2.13.* CPU 권장)."
  echo "  - MTTS_CAN/딥 모델 사용 시 TF 버전 호환성 확인 필요."

if [[ "${VERIFY}" -eq 1 ]]; then
  echo "[setup] Running post-install import check..."
  python - <<'PY'
mods = [
    'numpy','scipy','cv2','mediapipe','skimage','pyVHR','pyts','emd',
    'statsmodels','mne','tqdm','PIL','ptlflow','torch','cmdstanpy'
]
ok = True
for m in mods:
    try:
        __import__(m)
        print('[ok]', m)
    except Exception as e:
        ok = False
        print('[fail]', m, e)
import torch
print('torch.cuda.is_available:', torch.cuda.is_available())
PY
fi
