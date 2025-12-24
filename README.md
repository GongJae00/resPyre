# ResPyre: 비접촉 호흡 추정 프레임워크 (V3)

**ResPyre**는 구조화된 연구를 위한 비접촉 호흡 추정 프레임워크입니다.
**Components(정의)**와 **Core(실행)**가 분리된 **V3 아키텍처**를 따릅니다.

## 1. 프로젝트 구조 (Directory Structure)

```text
/home/gongjae/Projects/resPyre/
├── components/          # [연구 작업 공간] 모델/데이터 추가
│   ├── datasets/        # 데이터셋 코드 (BP4D.py, COHFACE.py...)
│   ├── observations/    # 관측 기법 (OF.py, DoF.py...)
│   └── models/          # 호흡 추정 알고리즘
│       ├── heads/       # (Logic) KF, UKF, PLL 등 실제 구현체
│       ├── core/        # (Base) OscillatorParams, BaseClass
│       └── autotune/    # (Util) 파라미터 자동 튜닝 로직
├── core/                # [엔진 구동 공간] 파이프라인 엔진
│   ├── pipeline/        # Runner, Wrapper
│   ├── evaluation/      # Metrics, Logging
│   └── utils/           # Config loader 등
├── dataset/             # [Data] 원본 데이터셋 루트
├── setup/               # [Setup] 설치 및 보조 스크립트
│   ├── setup.sh         # 설치 스크립트
│   ├── run_optuna.py    # (New) Optuna 최적화 러너
│   └── generate_metadata.py # (New) 메타데이터 생성기
├── configs/             # [Config] 실험 파라미터 파일 (*.json)
└── main.py              # [Main] 통합 실행기
```

---

## 2. 실행 흐름 (Execution Flow)

```bash
python main.py --config configs cohface_motion_oscillator.json
```
실행 시, 아래 순서로 내부 동작이 진행됩니다.

### 1. 설정 로드 (Configuration)
`configs/cohface_motion_oscillator.json` 파일을 읽어 실험 이름(`name`), 데이터셋(`COHFACE`), 사용할 방법론(`OF_Farneback__KFstd`)을 파싱합니다.

**Tip: 빠르게 검증하고 싶다면?**
전체 데이터셋 대신 첫 번째 비디오만 처리하여 동작을 확인하려면 `--debug` 옵션을 사용하세요.
```bash
python main.py --config configs/cohface_motion_oscillator.json --debug
```

### 2. 추정 단계 (Estimation Phase)
`core.pipeline.runner`가 구동되며 데이터셋의 각 비디오에 대해 다음 처리를 수행합니다.
1.  **관측 (Observation)**:
    *   `components.observations.OF`: 비디오에서 Optical Flow를 추출하여 호흡 관련 수직 움직임 신호 $y(t)$ 생성.
2.  **전처리 (Preprocessing)**:
    *   대역 통과 필터 (0.08~0.5Hz), Detrending, Robust Z-score 정규화 수행.
3.  **공진자 모델링 (Oscillator Modeling)**:
    *   `components.models.heads.kf_std`: $y(t)$를 입력받아 공진자 상태공간(State-Space)에 매핑.
    *   **Kalman Filter Update**: 매 프레임마다 관측값과 예측값의 오차(Innovation)를 계산하고, Kalman Gain을 통해 공진자 상태(위상, 진폭)를 실시간 보정.
4.  **결과 저장**:
    *   추정된 호흡 파형($\hat{s}$), 호흡률($f$), 상태 변수 등을 `results/.../data/`에 `.pkl` 형태로 저장.

### 3. 평가 단계 (Evaluation Phase)
`core.pipeline.evaluation_step`이 저장된 `.pkl` 파일을 로드합니다.
1.  **Ground Truth 로딩**: 데이터셋 원본의 생체 신호(GT)를 로드하고 동기화(Synchronization).
2.  **지표 산출**:
    *   **Time Domain**: RMSE, MAE, Pearson Correlation (파형 유사도).
    *   **Freq Domain**: 호흡률(BPM) 오차.
3.  **결과 집계**: 모든 비디오의 성능 평균을 `results/.../metrics/`에 저장.

### 4. 리포트 (Reporting)
1.  **시각화 (Visualization)**: `results/.../plots/`에 파형 비교 HTML 그래프 생성.
2.  **메타데이터 (Metadata)**: 실험 환경(Git Commit, 명령어, 파일 경로 등)을 기록한 `metadata.json` 생성.
3.  최종 성능 요약(`summary.json`) 저장 후 종료.

### 자주 묻는 질문 (FAQ)

**Q. 설정 파일에는 `of_farneback__kfstd`만 있는데 다른 것도 실행되나요?**
A. 네, `methods` 리스트에 있는 **모든 조합이 순차적으로 실행**됩니다. 예시 설정 파일에는 `dof`, `profile1d` 계열 등 약 15개 알고리즘이 정의되어 있어 전부 수행됩니다.

**Q. "재측정(Re-measurement)"을 하나요?**
A. Kalman Filter의 **"Update (Correction)" 단계**를 의미합니다. 비디오를 다시 찍는 것이 아니라, 매 프레임 관측값($y_t$)이 들어올 때마다 내부 상태(위상, 주파수)를 보정하는 과정입니다. ResPyre의 모든 모델은 실시간 프레임 단위 Update를 수행합니다.

**Q. 결과 그래프(Plot)는 어디 있나요?**
A. `configs/*.json`의 `steps`에 `"visualize"`가 포함되어 있다면, 다음 경로에 **HTML 인터랙티브 그래프**가 자동 생성됩니다.
```text
results/experiment_name/<dataset>/plots/
├── <video_name>_<method>.html
...
```
이를 브라우저로 열어 파형과 호흡수 추이(BPM Trace)를 상세히 비교 분석할 수 있습니다.

**Q. 결과 폴더 구조는 어떻게 되나요?**
```text
results/experiment_name/
├── data/            # 비디오별 추정 결과 (.pkl) - 파형, 주파수 등 포함
├── metrics/         # 평가 결과 (.json, .csv) - RMSE, MAE 수치
└── aux/             # (선택) 디버깅용 보조 데이터 (.npz)
```

### Q. 새로운 **모델**을 만들고 싶어요.
`components/models/heads/` 폴더에서 작업하세요.

1.  파일 생성: `components/models/heads/my_model.py`
2.  작성:
    ```python
    from ..core.base import _BaseOscillatorHead

    class MyModel(_BaseOscillatorHead):
        head_key = "my_model"
        def run(self, signal, fs, meta=None):
            # 알고리즘 구현
            return self._package(signal_hat, track_hz, meta)
    ```
3.  등록: `components/models/__init__.py` 에 클래스 추가.

### Q. 새로운 **데이터셋**을 추가하고 싶어요.
`components/datasets/` 폴더를 사용합니다.

1.  파일 생성: `components/datasets/my_dataset.py`
2.  작성: `DatasetBase` 상속 후 `load_dataset`, `load_gt` 구현.
3.  등록: `components/datasets/__init__.py` (선택 사항이나 권장).

### Q. **Optuna 최적화**는 어떻게 하나요?
`setup/run_optuna.py`를 사용합니다. 루트 디렉토리에서 실행하세요.
```bash
python setup/run_optuna.py --study-name my_study --n-trials 100
```

---

## 4. 설치 (Installation)

```bash
# 환경 생성 및 패키지 설치
./setup/setup.sh -n respyre_env

# 활성화
conda activate respyre_env
```

## 5. 검증 (Verification)

```bash
python main.py --help
```
위 명령어가 에러 없이 옵션을 출력하면 설치가 완료된 것입니다.