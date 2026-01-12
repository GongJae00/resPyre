## 상체 모션 기반 공진자 상태공간 호흡 추정 (논문 빌드)

- 연구 의도 (notes/2025-87_2_1.pdf 요약)
  - 얼굴 rPPG 대신 흉곽 상체 움직임을 직접 관측해 생리 기전과 신호 대역을 일치시킴.
  - 1D 모션 신호를 감쇠·회전 공진자로 표현하고, 칼만/UKF 추적기로 위상·주파수 궤적을 복원.
  - COHFACE에서 5개 모션 신호 × 2개 추적기(KFstd, UKFfreq) 평가 시 시간 영역 MAE가 기존 대비 유의하게 감소.

- 남은 코드(논문 재현용)
  - 모션 베이스라인: `of_farneback`, `dof`, `profile1d_linear`, `profile1d_quadratic`, `profile1d_cubic`
  - 오실레이터 헤드: `kfstd`, `ukffreq` (`<base>__kfstd`, `<base>__ukffreq`)
  - 데이터셋 로더: COHFACE (사용), BP4D/MAHNOB(보존)
  - 설정: `configs/cohface_motion_oscillator.json`


## 순차 실행 가이드

1) **환경 준비**
```bash
./setup/setup.sh
eval "$(python setup/auto_profile.py)"
```

2) **Estimate → Evaluate → Metrics**
```bash
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s estimate evaluate metrics \
  --auto_discover_methods true
```

3) **샤딩 Estimate (옵션)**
```bash
for idx in 0 1 2 3; do
  python run_all.py -c configs/cohface_motion_oscillator.json \
    -s estimate --num_shards 5 --shard_index $idx
done
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s evaluate metrics --auto_discover_methods true
```

4) **Overlay/플롯 확인 (옵션)**
```bash
python tools/plot_family_overlays.py \
  --results results/cohface_motion_oscillator
```

5) **EM Kalman Gain + Autotune (옵션)**
```bash
METHODS=(
  of_farneback__kfstd of_farneback__ukffreq
  dof__kfstd dof__ukffreq
  profile1d_linear__kfstd profile1d_linear__ukffreq
  profile1d_quadratic__kfstd profile1d_quadratic__ukffreq
  profile1d_cubic__kfstd profile1d_cubic__ukffreq
)
for m in "${METHODS[@]}"; do
  python train_em.py \
    --results results/cohface_motion_oscillator \
    --dataset COHFACE \
    --method "$m" \
    --build_autotune \
    --autotune-top-k 5
done
```

6) **Optuna 탐색 (옵션)**
```bash
python optuna_runner.py \
  --config configs/cohface_motion_oscillator.json \
  --output runs/optuna_all \
  --n-trials 40 \
  --em-mode trial \
  --auto-discover-methods true

# 번들 적용 후 재평가
cd runs/optuna_all/_bundles/best10_*/ && ./apply_all.sh
```

7) **메타데이터 기록 (옵션)**
```bash
python tools/write_metadata.py \
  --run results/cohface_motion_oscillator \
  --command "python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics"
```

- 재평가 시 기존 결과 재사용 가능. 스테일 메트릭을 방지하려면 `results/cohface_motion_oscillator/metrics`만 삭제.
- `aux/<method>/<trial>.npz`에 추적 주파수/재구성 파형/메타 저장, 요약 메트릭은 `results/cohface_motion_oscillator/metrics/`.