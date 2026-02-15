# Robust Bayesian Filter: 단계별 실행 가이드 (Step-by-Step Usage Guide)

이 문서는 Scientific Reports 논문 작성을 위한 실험 수행, 코드 검증, 결과 생성 과정을 순차적으로 설명합니다.

---

## 1. 환경 설정 및 검증 (Setup & Verification)
실험을 시작하기 전 환경과 데이터가 준비되었는지 확인합니다.

### 1.1. 가상환경 활성화
```bash
source .venv/bin/activate
```

### 1.2. 필수 라이브러리 확인
COHFACE Ground Truth 로딩을 위해 `h5py`가 필수적입니다.
```bash
pip install h5py pandas scipy matplotlib opencv-python
```

### 1.3. COHFACE 데이터셋 확인
데이터셋이 올바른 위치에 마운트되어 있는지 확인합니다.
```bash
ls -F /home/gongjae/Projects/resPyre/dataset/COHFACE/
# 1/, 2/ ... 또는 subject_1/ 등의 디렉토리가 보여야 합니다.
```

### 1.4. 단위 테스트 실행 (Sanity Check)
본격적인 실험 전 22개의 테스트가 모두 통과하는지 확인합니다.
```bash
pytest tests/
# 예상 결과: 22 passed in x.xxs
```

---

## 2. Phase 0c: 데이터 탐색적 분석 (EDA)
노이즈가 Gaussian이 아닌 Heavy-tailed(Student-t) 분포를 따른다는 통계적 근거를 확보합니다.

### 2.1. 노이즈 분석 실행
COHFACE 데이터셋 구성을 담은 Config 파일을 지정하여 분석을 실행합니다. (스크립트 내부에서 5가지 방식을 모두 분석함)
```bash
python analysis/run_noise_analysis.py --config configs/cohface_robust_ossm.json
```
*   **결과 폴더:** `analysis/noise_properties/`
*   **결과 파일:** `summary.json`, `global_analysis_*.png`
*   **확인할 핵심 지표:**
    *   `kurtosis`: 3.0 이상이어야 함 (Heavy-tail 입증).
    *   `t_fit_nu`: 자유도($\nu$), 보통 4 ~ 10 사이.
    *   `t_aic_delta`: 양수(+) 값이어야 Student-t가 Gaussian보다 더 적합함을 의미.

---

## 3. Phase 7: 전체 벤치마크 실험 (Full Benchmark)
Baseline(기본 KF)과 제안 모델(Robust OSSM)의 성능을 비교하는 End-to-End 파이프라인을 실행합니다.

### 3.1. 드라이 런 (Debug Mode)
파이프라인이 정상 작동하는지 1개 샘플로 빠르게 테스트합니다.
```bash
python main.py --config configs/cohface_robust_ossm.json --debug
```
*   **체크포인트:** 에러 없이 완료되는지, MAE가 비정상적으로 크지 않은지 (< 5 BPM) 확인.

### 3.2. 전체 데이터셋 실험
전체 데이터에 대해 실험을 수행합니다. (하드웨어 성능에 따라 1~2시간 소요 가능)
```bash
python main.py --config configs/cohface_robust_ossm.json
```
*   **생성되는 산출물:**
    *   `results/cohface_robust_ossm/metrics/metrics_raw.csv`: 개별 시행 결과.
    *   `results/cohface_robust_ossm/metrics/summary.json`: 전체 평균/표준편차 요약.
    *   `results/cohface_robust_ossm/plots/`: 시각화 결과물.

---

## 4. 시각화 및 논문용 Figure 생성
논문에 삽입할 구체적인 그래프를 확인합니다.

### 4.1. 방법별 성능 비교 (Boxplot)
`main.py` 실행 시 자동으로 생성됩니다.
```bash
ls results/cohface_robust_ossm/plots/summary_mae_boxplot.png
```

### 4.2. 트래킹 결과 분석 (Qualitative Trace)
노이즈 상황에서의 강건함을 보여주는 샘플을 확인합니다.
*   **파형 정렬 비교 (Best Alignment):** `results/cohface_robust_ossm/plots/overlay_best_waveform_aligned_*.png`
*   **패밀리 오버레이 (Overlap):** `results/cohface_robust_ossm/plots/family_overlays/`

---

## 5. Prism 문서 업데이트 (Traceability)
새로운 실험 결과를 Prism 문서에 반영하여 논문의 근거를 업데이트합니다.
*   `notes/prism/02_experiments_protocol.md`: 최종 샘플 수 및 실험 조건 업데이트.
*   `notes/prism/04_assumptions_and_verification.md`: 최종 EDA 통계 수치 반영.

---

## 빠른 참조 커맨드 (Quick Reference)

| 작업 | 명령어 |
| :--- | :--- |
| **테스트** | `pytest tests/` |
| **EDA 분석** | `python analysis/run_noise_analysis.py ...` |
| **실행 (Deubg)** | `python main.py --config ... --debug` |
| **실행 (Full)** | `python main.py --config ...` |
| **결과 확인** | `ls results/cohface_robust_ossm/plots/` |
