resPyre 설치 가이드
===================
빠른 가이드
```bash
./setup/setup.sh --verify
conda activate resPyre
eval "$(python setup/auto_profile.py)"
```

---

목표: 머신 간 일관된 재현 환경을 간단히 구성합니다. 본 스크립트는 다음을 설치합니다.
- 모션 기반 방법(OF_Model, profile1D, DoF)
- ROI 추출(MediaPipe)
- rPPG/분해(pyVHR, EMD/SSA)
- 평가/리포팅
- SOMATA 의존성(+ 로컬 패키지 선택 설치)
- 딥 스택 중 PTLFlow 포함(Pytorch 기반), TensorFlow/TVM은 기본 미설치(옵션)

사전 준비
- Linux + bash
- Conda(Miniconda/Anaconda)
- 선택: NVIDIA GPU + 적절한 드라이버/CUDA (GPU 사용 시)

빠른 시작
- 0) 초기화(권장)
  - `conda deactivate` (여러 번 실행해서 어떤 env도 활성화돼 있지 않은 상태로 만듭니다)
  - 기존 env 삭제: `conda remove -n resPyre --all -y` (존재할 경우)
  - pip 캐시 정리: `conda run -n base python -m pip cache purge` 또는 `pip cache purge`

- 1) 통합 설치(권장, GPU 자동 감지)
  - `./setup/setup.sh --verify`

- 2) CUDA 11.7로 명시적 GPU 설치(PyTorch 1.13.1과 호환)
  - `./setup/setup.sh --cuda 117 --verify`

- 3) CPU 전용 강제(딥스택 포함, CPU로 설치)
  - `./setup/setup.sh --cpu-only --verify`

- 4) CmdStan까지 설치(Stan 모델 빌드 필요 시)
  - `./setup/setup.sh --install-cmdstan`

- 5) 설치 후 활성화 + 프로파일 적용
  - `conda activate resPyre`
  - `eval "$(python setup/auto_profile.py)"`

- 6) 동작 확인(예: 모션 기반 기본 파이프라인)
  - `python run_all.py -a 0 -d results/`

무엇이 설치되나요?
- conda(바이너리 호환, conda-forge 채널 사용): numpy 1.24.x, scipy 1.10.x, pandas 1.5.x, matplotlib 3.7.x, h5py 3.9.x
- pip(requirements):
  - tqdm, Pillow
  - OpenCV(opencv-python, opencv-contrib-python), mediapipe(0.10.10), scikit-image
  - torch(1.13.1, cu117 휠 자동 선택), torchvision, torchmetrics, lightning, einops
  - ptlflow(0.3.2), timm, tabulate, kaleido, pypng, tensorboard
  - pyVHR, pyts, emd, pyEDFlib
  - prettytable, plotly
  - SOMATA deps: cmdstanpy, codetiming, colorcet, joblib, kneed, mne, spectrum, statsmodels

환경 이름/파이썬 버전
- 기본 conda 환경명: `resPyre`
- 기본 Python: `3.10` (사용자 환경 기준, Ubuntu 24.04 호환)
- 변경: `-n/--name`, `-p/--python`

CUDA/torch 감지·선택 로직
- 기본: `nvcc` → `nvidia-smi` 순으로 CUDA 버전을 자동 감지, 미감지 시 CPU로 설치
- PyTorch 1.13.1은 공식적으로 `cu116/cu117` 휠만 제공
  - 로컬에서 CUDA 12.x가 감지되어도 안전하게 `cu117`로 매핑하여 설치
  - 특정 CUDA를 강제하려면 `--cuda 117` 사용
- CPU 전용 강제: `--cpu-only`

Conda 의존성 충돌 방지 전략(중요)
- 깨끗한 env 생성(필요 시 `conda remove -n resPyre --all`)
- setup.sh가 pip를 자동으로 `pip<24.1`로 맞춘 뒤 requirements 설치(PTLFlow 0.2.7 메타데이터 호환 위해 필요)
- 단일 requirements로 일괄 설치 + 설치 후 `pip check`
- TensorFlow/TVM은 기본 미설치(필요 시 별도 설치 + 코드 호환 확인)

SOMATA 통합
- 통합 요구사항에는 SOMATA 런타임 의존성이 포함됩니다.
- 로컬 패키지 설치(기본값: 설치):
  - `external/somata-main`이 있을 경우 편집 모드로 설치합니다.
  - 건너뛰려면 `--no-somata`.
- CmdStan 설치(선택): `--install-cmdstan` (시간이 다소 소요됩니다)

런타임 자동 프로파일링
- 환경 활성화 후 아래를 실행하면 합리적인 런타임 변수들이 export 됩니다.
  - `eval "$(python setup/auto_profile.py)"`
- 설정: `DEVICE`, `NUM_WORKERS`, `PIN_MEMORY`, `PERSISTENT_WORKERS`, `PREFETCH_FACTOR`, `USE_AMP`, `AMP_DTYPE`

주의 및 호환성 메모
- MAHNOB(BDF) 로더는 `pybdf` 사용(기본 미포함). 필요 시 별도 설치 후 로더 보완 권장.
- `dataset/`에 심볼릭 링크가 있을 수 있음. 로컬 경로 유효성(권한/마운트) 확인.
- Windows/비-Linux에서는 mediapipe 휠이 없을 수 있음.

딥스택(TensorFlow) 사용 시 권고
- CPU 기준 예: `pip install "tensorflow==2.13.*"`
- GPU 사용은 TF 공식 가이드를 따르세요. MTTS_CAN 등 구 코드에서는 TF 상향 시 API 호환 패치가 필요할 수 있습니다.
