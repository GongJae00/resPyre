# NAROSSM Paper — Figure & Table Index

## Figures (7개)

| ID | 내용 | 파일명 | 상태 | 생성 방법 |
|----|------|--------|------|----------|
| Fig.1 | System Overview (아키텍처 다이어그램) | `figures/fig1_system_overview.pdf` | **PaperBanana에서 제작** | 수동 |
| Fig.2 | NIS Calibration (Gaussian vs NAROSSM) | `figures/fig2_nis_calibration.pdf` | 실험 후 생성 | `analysis/` 스크립트 |
| Fig.3 | Adaptive Parameter Trajectories (R_t, ν_t, λ_t) | `figures/fig3_adaptive_params.pdf` | 실험 후 생성 | `analysis/` 스크립트 |
| Fig.4 | Per-Frame Diagnostic Trace (4-panel) | `figures/fig4_diagnostic_trace.pdf` | 실험 후 생성 | `analysis/` 스크립트 |
| Fig.5 | Waveform Reconstruction Comparison | `figures/fig5_waveform_comparison.pdf` | 실험 후 생성 | `analysis/` 스크립트 |
| Fig.6 | Algorithm Block Diagram | `figures/fig6_algorithm_block.pdf` | **PaperBanana에서 제작** | 수동 |
| Fig.freq | COHFACE Frequency Stability | `figures/fig_cohface_freq_stability.pdf` | 실험 후 생성 | `analysis/` 스크립트 |

## Tables (9개)

| ID | 내용 | 위치 | 상태 |
|----|------|------|------|
| Tab.1 | Noise Characterisation (kurtosis, JB, ARCH-LM per family) | Results §1 | ✅ 수치 입력됨 |
| Tab.2 | COHFACE Frequency-Domain Performance (MAE) | Results §4 | ✅ 수치 입력됨 |
| Tab.3 | COHFACE Time-Domain Performance (CCC, MAE) | Results §5 | ✅ 수치 입력됨 |
| Tab.4 | Ablation Study | Results §6 | ✅ 수치 입력됨 |
| Tab.5 | MAHNOB Frequency-Domain Performance | MAHNOB §1 | ✅ 수치 입력됨 (2026-03-24) |
| Tab.6 | MAHNOB Time-Domain Performance | MAHNOB §2 | ✅ 수치 입력됨 (2026-03-24) |
| Tab.7 | Cross-Dataset Comparison | Cross-dataset §1 | ✅ 수치 입력됨 (2026-03-24) |
| Tab.8 | Cross-Dataset R Transfer | Cross-dataset §2 | ⏳ Placeholder, 실험 후 채움 |
| Tab.9 | Outlier Injection Robustness | Outlier §1 | ⏳ Placeholder, 실험 후 채움 |

## 새 실험 스크립트

| 스크립트 | 용도 | 상태 |
|----------|------|------|
| `analysis/cross_dataset_R.py` | Cross-dataset R transfer 실험 (Tab.8) | ✅ 생성됨, 실행 필요 |
| `analysis/outlier_injection.py` | Outlier injection 실험 (Tab.9) | ✅ 생성됨, 실행 필요 |

## 실행 순서

1. `python analysis/cross_dataset_R.py` → Tab.8 수치 생성
2. `python analysis/outlier_injection.py` → Tab.9 수치 생성
3. FrameLogger re-run → Fig.2-5 데이터 생성
4. PaperBanana: Fig.1, Fig.6 수동 제작
5. 논문 placeholder 자리에 실제 수치 채움
