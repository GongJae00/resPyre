# Scientific Reports 투고를 위한 연구 논리 및 사고 과정

**프로젝트**: ResPyre (강건한 비접촉 호흡 추정 프레임워크)
**목표 저널**: Nature Scientific Reports
**핵심 철학**: "완벽한 데이터 이해가 완벽한 모델 설계를 만든다."

## 1. 사용자의 연구 비전 및 스토리텔링
*   **핵심 아이디어 (The "Kick")**: 호흡 주파수($f$)를 고정된 값이 아닌 시변하는 상태($x_t$)로 정의한다.
    *   *이유*: 실제 호흡은 생리적으로 시시각각 변합니다. 단순 FFT나 Peak Detection은 이러한 동적 특성을 포착하지 못합니다.
*   **기존 모델의 한계 및 진단**:
    *   **Baseline (KF-std)**: 안전하지만 선형성 가정으로 인해 급격한 주파수 변화를 추적하지 못합니다.
    *   **Previous (UKF-freq)**: 기존 연구에서 비선형성 및 비정규성을 고려하여 설계되었으나, 실제 구현상의 완성도 부족과 비접촉 환경의 과도한 잡음을 처리할 수 있는 강건성(Robustness) 결여로 인해 의도한 성능을 발휘하지 못했습니다. 즉, 이론적 방향성은 맞았으나 실전에서의 "완성도"가 떨어져 단순 선형 모델과 차별화되지 못하거나 발산하는 문제를 보였습니다.
    *   *최종 해결책*: 신호의 비선형/비정규 특성을 "제대로" 처리할 수 있는 **"강건한 어텐션 기반 적응형 칼만 필터 (AG-AKF)"**를 완성도 있게 구축합니다.

## 2. 설정 및 방법론 (Config & Method Check)
*   **관측 단계 (Observation Layer)**: `OF_Farneback` (표준), `Profile1D` (효율성).
*   **모델 단계 (Model Layer)**: `AG-AKF` (Proposed) vs `KF-std`.

## 3. 데이터 분석 계획 (The "Perfect Exploration")
데이터의 특성을 단계별로 해부하여, 왜 기존 유클리드/가우시안 기반 방법론들이 실패할 수밖에 없는지 **체계적으로 증명**합니다.

### Step 1: 원본 데이터 수준 (Raw Data Level)
*   **목표**: 비접촉 영상 데이터 자체가 가지는 조명 변화 및 픽셀 분포 특성 파악.
*   **분석 항목**:
    *   Global Intensity Histogram: 조명 간섭의 정규성 여부.
    *   Pixel Dynamics: 시간축에 따른 픽셀 변화의 급격함(Jerkiness) 분석.

### Step 2: 관측 신호 수준 (Observation Signal Level - OF, Profile1D)
*   **목표**: 모션 추출 알고리즘이 내뱉는 원시(Raw) 신호의 특성 분석.
*   **분석 항목**:
    *   **비선형성 (Non-linearity)**:
        *   호흡 운동(Sinusoidal)과 관측된 신호 간의 매핑 왜곡 확인.
        *   고조파 왜곡(Harmonic Distortion) 비율 분석 (Fundamental freq 대비 Harmonics 파워).
    *   **Trend & Drift**: 저주파수의 비선형적 이동(Movement Artifacts) 존재 여부 시각화.

### Step 3: 전처리 신호 수준 (Preprocessed Signal Level)
*   **목표**: Detrending 및 Bandpass Filtering을 거친 후에도 남아있는 "나쁜 특성" 규명.
*   **분석 항목**:
    *   필터링된 신호의 포락선(Envelope) 변동성.
    *   급격한 진폭 변화(Impulse like events) 탐지.

### Step 4: 잔차 및 잡음 수준 (Residuals & Noise Level - 핵심)
*   **목표**: 모델링 오차(Innovation)가 가우시안을 따르는가? (핵심 가설 검증)
*   **분석 항목**:
    *   **정규성/비정규성 (Estimator: Innovation)**:
        *   **Kurtosis (첨도)**: > 3.0 (Heavy Tails) 여부 확인. 이것이 높으면 기존 칼만 필터($L_2$ norm)는 실패합니다.
        *   **Q-Q Plot**: 이론적 정규분포 대비 꼬리 부분의 이탈 정도 시각화.
        *   **Shapiro-Wilk / Kolmogorov-Smirnov Test**: 통계적 유의성(p-value) 확보.
    *   **비선형 동역학 특성**:
        *   Phase Portrait (위상 공간 궤적): $x_t$ vs $x_{t+1}$ 플롯이 타원형(선형)인지 찌그러진 형태(비선형)인지 확인.

## 4. 모델 설계에 주는 시사점 (Design Implication)
1.  **Heavy Tails 발견 시**: $L_2$ Loss(MSE) 대신 **Huber Loss**나 **Student-t 분포** 기반의 Likelihood, 혹은 **Attention Mechanism**을 통해 이상치 가중치를 낮춰야 함을 수학적으로 정당화.
2.  **Harmonic Distortion 발견 시**: 단순 Sinusoidal 모델(단일 주파수) 대신, 고조파를 포함한 상태 공간 모델이나 이를 보정하는 비선형 관측 모델이 필요함을 시사.
