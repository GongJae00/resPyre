# Nature Scientific Reports Research Plan (v2.0)

**목표**: Nature Scientific Reports 투고
**주제**: Oscillator State-Space Model(OSSM) 기반의 비접촉 호흡 추정을 위한 도메인 특화 강건한 칼만 필터(AG-AKF) 개발
**핵심 기여**: 비접촉 신호의 비선형/비정규 잡음 특성을 정량적으로 규명하고, 이를 해결하기 위한 Attention 기반의 적응형 정밀 추적 모델 제안

## 1. 연구 배경 및 진단 (Scientific Vision)
*   **핵심 철학**: "Frequency as State" - 호흡 주파수를 정적인 값이 아닌 동적인 상태변수로 취급하여 실시간 생리 변화를 추적함.
*   **문제 의식**: 기존 연구(UKF-freq 등)는 이론적 방향성은 제안했으나, 실제 비접촉 환경의 극심한 비정규 잡음(Impulsive Noise)을 처리하지 못해 완성도가 떨어졌음.
*   **차별점**: 단순한 성능 비교를 넘어, 칼만 필터 자체를 해당 도메인(Remote Sensing)의 물리적/통계적 특성에 맞게 "완벽하게" 최적화하고 완성도를 극대화함.

## 2. 세부 연구 단계 (4-Step Pipeline)

### Phase 1: 체계적 데이터 분석 (Systematic Diagnosis)
**목표**: 비접촉 호흡 신호가 "왜" 기존 유클리드/가우시안 방법론으로 안 되는지 정량적 증거 확보.
*   **Step 1 (Raw Level)**: 영상 자체의 조명 변화 및 픽셀 동역학 분석 (Pixel-level Kurtosis).
*   **Step 2 (Observation Level)**: 모션 추출 신호(OF, Profile1D)의 고조파 왜곡(THD) 및 비선형성 분석.
*   **Step 3 (Preprocessed Level)**: 필터링 후에도 존재하는 돌발 잡음(Impulse-rate) 검출.
*   **Step 4 (Residual Level)**: 모델링 오차(Innovation)의 비정규성 입증 (Q-Q Plot, Kurtosis > 3, Phase Portrait).
*   **결과물**: "The Proof of Non-Gaussianity" 인포그래픽 생성.

### Phase 2: 도메인 특화 모델 설계 (AG-AKF Design)
**목표**: 분석 결과에 기반한 Attention-Guided Adaptive Kalman Filter 설계.
*   **OSSM 고도화**: 최신 상태공간 모델 이론을 적용하여 호흡 동역학($F_t$)을 정교화.
*   **Attention Mechnism**: 
    - 신호 품질에 따라 관측 잡음 공분산($R_t$)을 동적으로 조절.
    - $R_t \propto 1 / \text{Attention}$ (신뢰도가 낮으면 관측은 무시하고 모델 예측을 신뢰).
*   **강건 통계 적용**: Heavy-tailed noise를 처리하기 위한 Huber Loss 또는 Student-t 분포 개념의 수치적 통합.

### Phase 3: 실험 및 검증 (Implementation & Evaluation)
*   **구현**: `components/models/heads/ag_akf.py` 개발.
*   **최적화**: `core/optimization/run_optuna.py`를 통한 파라미터 정밀 튜닝.
*   **검증 전략**:
    - **Ablation Study**: Attention 기여도 확인 (w/ vs w/o Attention).
    - **Baseline Comparison**: `KF-std` 대비 우위 입증.
    - **Dataset**: COHFACE (다양한 조명/움직임 환경) 주력.

### Phase 4: 논문 작성 (Narrative & Visualization)
*   **Introduction**: 비접촉 환경의 통계적 특성에 대한 철저한 분석과 기존 연구의 한계 서술.
*   **Methods**: OSSM과 Attention의 수학적 결합 과정을 엄밀하게 기술.
*   **Discussion**: 설계된 모델이 어떻게 비정규 잡음을 억제하는지 데이터로 설명.

## 3. 실행 로드맵 (Roadmap)
1.  **[진행 중]** 심층 분석 스크립트(`analysis/run_noise_analysis.py`) 실행 및 데이터 확보.
2.  **[다음 단계]** 확보된 데이터를 기반으로 AG-AKF의 수학적 수식 확정.
3.  **[개발 단계]** AG-AKF 모델 구현 및 파이프라인 통합.
4.  **[검증 단계]** COHFACE 벤치마크 및 결과 시각화.
