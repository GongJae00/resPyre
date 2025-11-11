# 비접촉 호흡 추정 연구 노트: 상체 움직임 추적과 상태 공간 모델 통합 (resPyre 최신 구조 반영본)

본 문서는 현재 resPyre 저장소(사용자 포크)에서 구현/실험 중인 “모션 기반 상체 움직임 + 공진자 상태공간 모델” 파이프라인을 정확히 기술하고, 이후 Codex와의 구현 대화와 논문 집필(Scientific Reports 스타일의 main.tex)에 바로 쓸 수 있도록 정리한 최신화 노트다. 결과 수치는 아직 튜닝·정합 중이므로, 본 문서는 “의도·설계·실험 절차 틀”을 완전히 고정하는 데 초점을 둔다.

---

## 1. 문제 배경과 연구 철학

카메라 기반 비접촉 호흡 추정은 주로 rPPG(피부 광학)에서 호흡 저주파(0.1–0.5 Hz) 성분을 분리하는 방식이 널리 쓰였으나, 조명·자세·미세운동 노이즈와 대역이 겹쳐 근본적 한계가 있었다. 본 연구는 관측 양식-생리 기원 정합성에 착안해 “혈류” 대신 “상체의 기계적 운동”을 직관적으로 추적한다. 즉, 상체 변위로 얻은 1D 시계열을 호흡 공진자로 해석하고, 상태공간 추정(칼만/UKF/PLL/스펙트럼 능선 추적)로 주파수·위상·진폭을 안정적으로 복원한다.

핵심 설계 원칙

* 생리 대역 합치: 전 과정 평가·후처리 대역을 0.08–0.50 Hz로 통일
* 관측-모델 분리: “모션 추출(순수)”과 “주파수/위상 추정(모델링)”을 모듈화
* 최소한의 가정 + 안정성: 필터/추정기의 파라미터는 보수적·물리적 해석 가능한 범위로
* 비교 가능성: 기존 5개 모션 기법은 그대로 유지(알고리즘 변경 금지), 모델링 블록만 병렬 추가

---

## 2. 저장소 구조와 실행 흐름

핵심 파일/폴더 (현재 포크 기준)

* motion/method_*.py

  * of_farneback, dof, profile1d_linear, profile1d_quadratic, profile1d_cubic 등 “순수 모션 기반” 5개 기법
  * method_oscillator_wrapped.py: 상기 5개 출력 신호 위에 “오실레이터 헤드”를 씌우는 래퍼
* riv/estimators/oscillator_heads.py

  * kfstd, ukffreq, spec_ridge, pll 오실레이터 헤드 4종(총 20개 조합 = 5×4) 구현
* run_all.py

  * step=estimate → step=evaluate → step=metrics 파이프라인
  * aux/<method>/<trial>.npz에 track_hz, rr_bpm 등 중간 산출 저장
* config_loader.py, utils.py

  * 공통 설정, 평가 대역(0.08–0.50 Hz) 일원화, COHFACE GT 샘플링 보정
* configs/cohface_motion_oscillator.json

  * 기본 실행 설정; eval.min_hz/max_hz/use_track, oscillator defaults 등

명령 예시

```
rm -rf results/cohface_motion_oscillator/metrics results/cohface_motion_oscillator/plots results/cohface_motion_oscillator/replot

python run_all.py --config configs/cohface_motion_oscillator.json --step estimate
python run_all.py --config configs/cohface_motion_oscillator.json --step evaluate
python run_all.py --config configs/cohface_motion_oscillator.json --step metrics

# 병렬
AUTO_PROFILE_PARALLEL=5 eval "$(python setup/auto_profile.py)"
python run_all.py -c configs/cohface_motion_oscillator.json -s estimate --num_shards 5 --shard_index 0
python run_all.py -c configs/cohface_motion_oscillator.json -s estimate --num_shards 5 --shard_index 4

python run_all.py -c configs/cohface_motion_oscillator.json -s evaluate metrics --auto_discover_methods
python run_all.py -c configs/cohface_motion_oscillator.json -s evaluate metrics --auto_discover_methods --override gating.debug.disable_gating=true

python optuna_runner.py \
  --config configs/cohface_motion_oscillator.json \
  --output runs/optuna_shard0 \
  --num-shards 5 --shard-index 0 \
  --n-trials 20

cd runs/optuna/_bundles/best20_<ts>
./apply_all.sh
```

실행 결과

* results/<run_label>/methods/<method>/<trial>.pkl : 추정 파형·점수
* results/<run_label>/aux/<method>/<trial>.npz : 오실레이터 추적(예: track_hz)
* results/<run_label>/metrics/metrics.pkl, metrics_summary.txt, eval_settings.json

### 전처리 정책(robust 기본)

* 의도: 상체 모션 y(t)는 스파이크·드리프트가 공존하는 heavy-tailed 분포이므로 평균/표준편차 기반 정규화는 곧바로 혁신 분산을 왜곡한다. median + MAD 기반의 **robust z-score**는 진폭 스케일을 제거하면서도 주파수 추정 정보(f(t))를 그대로 보존하고, `rv_auto`가 사용하는 “bandpass 출력의 MAD”와도 동일 척도를 공유해 일관성이 생긴다.
* 순서: `sign_align → detrend → bandpass(0.08–0.50 Hz, 2차, zero-phase) → robust z-score`. MAD는 반드시 **bandpass 직후** 신호에서 구하고, `rv_auto`는 해당 MAD(σ̂=1.4826·MAD)로부터 R을 산정한다.
* 계산식(폴백 없이 eps 바닥만 적용):

```
med = median(x_bp)
mad = median(|x_bp − med|)
sigma_hat = 1.4826 * mad
z = (x_bp − med) / max(sigma_hat, eps)
if clip is not None:
    z = clip(z, -clip, +clip)
```

  * 기본 파라미터: `enabled=true`, `eps=1e-6`, `clip=3.5`. 트래커 계열(KFSTD/UKF/PLL)은 clip=3.5로 혁신 분포를 정상화하고, `spec_ridge` 헤드는 `clip=null`(또는 5.0)로 피크 모양을 보존한다.
  * CLI/JSON `--override preproc.robust_zscore.*` 경로는 동일하게 동작하며, per-head `params.preproc`와 전역 `preproc`는 항상 딥-머지한다.
* 메타 기록: 모든 헤드는 `meta.robust_z` 블록에 `enabled`, `med`, `mad`, `sigma_hat`, `clip`, `clipped_frac`(=|z|이 clip에 닿은 비율)을 저장한다. 동일 입력이라면 이 값들은 재현성 체크 포인트로 사용된다.

---

## 3. 모션 기반 5개 베이스라인(순수 관측 계층; 변경 금지, 파이프라인 정합만 보정)

이 계층의 목표는 RGB 비디오에서 상체 호흡 움직임을 대표하는 1D 관측 신호 y(t)을 안정적으로 뽑는 것이다. 알고리즘 자체는 비교 기준을 위해 “순수한 원형”을 유지한다. 다만 공통 전처리(대역·부호·스케일 정합)와 평가 파이프라인(0.08–0.50 Hz, COHFACE GT 샘플링 정합, 창 기반 평가)은 일괄 적용한다.

공통 I/O

* 입력: 프레임열 I(x, y, t), ROI(상부 가슴/쇄골 영역) 좌표, fps(=fs_video)
* 출력: 1D 관측 시계열 y(t) [정규화됨], 메타(meta: fs, roi, method 등)
* 공통 전처리

  1. ROI 잘라내기 → 그레이스케일 변환 → 약한 가우시안 스무딩(예: σ=1)
  2. 드리프트 완화용 미약한 고역통과(예: 0.03 Hz 1차 Butterworth) 또는 이동평균 제거
  3. 부호 정렬: 초기 10–15 s 구간에서 기준 신호와의 상관 부호를 +로 맞춤(없으면 y(t) 자체 기준)
  4. 스케일 정규화: z-score 또는 IQR 정규화(평가 표준화 목적)
* 공통 후처리

  1. 평가/표시용 밴드패스: 0.08–0.50 Hz(파이프라인 전역 일치)
  2. 필요 시 다운샘플(예: 32 Hz) 후 저장(메타에 fs 기록)

3.1 of_farneback (밀집 광학 흐름 기반)

* 핵심 아이디어: Farnebäck dense optical flow로 ROI 내 프레임 간 운동벡터 v=(u, v)를 구하고, 호흡에 대응하는 축으로 평균 투영해 1D 변위 시계열을 생성
* 표준 파라미터(권장 시작점; OpenCV)

  * pyr_scale=0.5, levels=3–4, winsize=15–25, iterations=3
  * poly_n=5, poly_sigma=1.2, flags=0
* 단계

  1. t, t+1 프레임의 ROI에서 flow_t = Farneback(I_t, I_{t+1}) 계산
  2. 선택 축(예: 수직 v)을 마스크 평균: y_raw(t) = mean(v_t(mask))
  3. 누적/적분 없이 시분할 평균만 사용(누적은 드리프트 유발)
  4. 공통 전·후처리 적용 후 y(t) 산출
* 장점/한계

  * 장점: 미세 변위에 민감, 균일 ROI에서 SNR 우수
  * 한계: 텍스처 빈약/글레어/모션 블러에 취약, 프레임 드롭에 영향
* 실패 모드와 징후

  * 강한 카메라 요(yaw)/피치(pitch) → y(t) 저주파 드리프트↑
  * 수평 어긋남이 크면 v축 평균이 약화 → 상관 하락
* 권장 점검 로그: flow 분포(중앙값/표준편차), reject rate, y(t) SNR(대역내/외 파워비)

3.2 dof (Difference of Frames)

* 핵심 아이디어: |I_{t+1} − I_t|의 프레임 차이를 ROI 내 합산/평균하여 호흡성 주기적 면적 변화 추적
* 단계

  1. D_t = |I_{t+1} − I_t|, ROI 평균 y_raw(t) = mean(D_t(mask))
  2. 고주파 잡음을 약화하려면 D_t에 약한 가우시안 σ=1–1.5
  3. 공통 전·후처리 적용
* 장점/한계

  * 장점: 계산량 최소, 질감 조건에 둔감
  * 한계: 임의 움직임/헤드 동요/조명 깜빡임을 동등 취급 → 호흡 특이성 낮음
* 실패 모드: 갑작스런 자세 전이 → 대역외 폭증, clipping
* 점검: y(t) 스펙트럼에서 0.08–0.50 Hz 이외 피크 비율, 프레임차의 극단치 비율

3.3 profile1d_linear (1D 프로파일 교차상관; 선형 보간)

* 핵심 아이디어: ROI를 행(또는 열) 방향으로 축소하여 1D 프로파일 p_t(i)를 만들고, p_{t+1}와 교차상관/서브픽셀 보간으로 상대 이동 Δ(t)을 추정
* 단계

  1. ROI를 선택 축(예: 세로)로 합/평균 → p_t(i)
  2. cc(τ) = ∑*i p_t(i)·p*{t+1}(i+τ), τ 정수 그리드에서 최대 찾기
  3. 서브픽셀 보정: 선형 보간으로 τ_max 근방 미세 보정 → Δ(t)
  4. y_raw(t) = Δ(t) 누적 없이 시분할 기록, 공통 전·후처리
* 장점/한계

  * 장점: 카메라 기울기/약한 ROI 이동에 비교적 강건, 텍스처 분포 변화에 덜 민감
  * 한계: 큰 시차/회전엔 취약, 프로파일 대비가 낮으면 상관 피크 불명확
* 점검: 상관 피크 대비(2nd/1st ratio), 실패 시 프레임 건너뛰기 비율

3.4 profile1d_quadratic (2차 보간)

* profile1d_linear의 ③에서 서브픽셀 보정만 2차(3점) 패러볼라 피팅으로 대체

  * τ̂ = τ0 + (c_{-1} − c_{+1}) / [2(c_{-1} − 2c_0 + c_{+1})]
* 장점: 선형보다 서브픽셀 정밀도 개선, 계산량 증가 미미
* 주의: 피크 주변 3점이 평탄하면 불안정 → 신뢰도 임계치 설정

3.5 profile1d_cubic (3차 보간)

* 4점 또는 cubic spline 근사로 서브픽셀 정밀도 극대화
* 장점: SNR 좋을 때 최고 정밀, 상관 피크가 뾰족한 경우 유리
* 한계: 피크가 넓거나 다봉(peaky multiple)일 때 과보정/진동 가능 → median filtering으로 완화

실행·평가 공통 체크

* 대역 정렬: 최종 y(t) 평가용 필터/웰치/후처리 모두 [0.08, 0.50] Hz
* COHFACE GT: fs_gt를 메타에서 파생, decimate 후 metrics에 반영
* 창 기반 평가: 30 s, 50% overlap 권장
* 저장: results/.../methods/<method>/<trial>.pkl, aux/.../<trial>.npz(후단 헤드용)

---

## 4.0 오실레이터 헤드 설계 철학(의도/관점/선택 이유)

이 섹션은 왜 “모션 5개(관측) + 오실레이터 4개(모델링)”을 이런 식으로 설계했는지, 그리고 각 헤드가 내 연구 의도를 어떻게 구현하는지 정리한다. 핵심은 “호흡은 좁은 주파수 대역에서 움직이는 준주기적 진동”이라는 생리적 사실을 최대한 존중하고, 관측 시계열의 불안정성(자세·카메라·조명 등)이 **시간영역 추정치 자체를 왜곡**시키지 않도록 **주파수-위상 중심**으로 신뢰도를 확보하는 것이다.

### 4.0.1 기본 관점: “시간영역 신호값”보다 “대역·스펙트럼 구조”가 먼저다

* 생체 호흡은 0.08–0.50 Hz 대역 내에서 느리게 변하는 준주기적 현상이다. 관측 노이즈는 진폭/에너지의 갑작스런 변동이나 베이스라인 드리프트처럼 **시간영역에서 보이는 왜곡**으로 나타나지만, 실제 유효 정보는 **해당 대역의 스펙트럼 밀도 분포**에 비교적 안정적으로 보존된다.
* 따라서 “관측 공간이 시계열”인 표준 칼만류만 고집하면, 잡음을 맞추려는 과정에서 **시간영역 신호를 ‘억지로’ 구부리는** 경향이 나타나고, 결과적으로 주파수·위상 추정이 생리 대역에서 벗어나거나 과도 평활이 될 수 있다.
* 해결책은 **진동자 상태공간 모델**로 위상/주파수 상태를 직접 추적하고, 필요한 경우 **스펙트럼 쪽에서 먼저 락(lock)하거나 제약을 걸어** 시간영역이 이 유도장(field)을 따르게 만드는 것이다.

### 4.0.2 왜 공진자 상태공간(oscillator SSM)인가

* 호흡을 2차 공진자(회전 행렬 R(ωΔt)에 감쇠 ρ)로 보고, 관측과 분리된 **은닉 위상·진폭·주파수**를 상태로 둔다. 이렇게 하면

  1. 생리 대역 제약([0.08, 0.50] Hz)을 **상태 전이**에 자연스럽게 부여할 수 있고,
  2. 모션 신호의 부호·스케일·드리프트가 바뀌어도 **상태(위상·주파수)는 연속성**을 유지하며,
  3. 노이즈 상황에서 **불확실성(P)**을 명시적으로 다룬다.
* 이 접근은 “관측을 시계열 값으로 맞춘다”가 아니라 “**관측이 알려주는 주파수·위상 단서**를 바탕으로 진짜 호흡 진동을 복원한다”에 가깝다.

### 4.0.3 4개 헤드 선택 철학(각각 다른 관점으로 ‘같은 진실’에 접근)

* kfstd(표준 칼만+RTS):
  최소 가정의 **선형 2D 공진자**. 고정 또는 완만 가변 f 가정.
  목적: “모델이 너무 똑똑해서 생긴 착시”를 배제한 **보수적 기준선**. 이게 잘 되면 복잡한 장치가 꼭 필요하진 않다. 반대로 이게 부족한 장면에서만 고급 기법의 필요성이 설득된다.
* ukffreq(로그-주파수 UKF):
  **주파수 자체를 상태로 추적**한다. log f로 양수 제약과 수치 안정성을 동시에 확보.
  목적: 시간영역 왜곡 없이 **느린 f(t) 변화**를 온전히 반영. 자세 변화나 SNR 요동에도 대역 내에서 락을 유지.
* spec_ridge(STFT 능선):
  “주파수는 스펙트럼에서 ‘보여야’ 한다”는 철학. **대역 제한 리지 추적**으로 track_hz를 만든 뒤 시간영역 복원을 보조한다.
  목적: 칼만류가 관측을 맞추느라 비틀릴 위험을 줄이고, **스펙트럼 밀도**를 우선 신뢰. 잡음이 커도 대역 내 에너지 흐름만 잡아도 충분히 안정적이다.
* pll(PI-PLL):
  **위상 오차**를 직접 줄이는 피드백 제어 관점. 계산량이 작고, 순간 주파수 락 능력이 탁월.
  목적: 실시간·온디바이스를 염두에 둔 **경량 동기화 엔진**. 대역·게인을 올바르게 설정하면 의도한 주파수 범위 안에서 매우 견고하게 잠긴다.

네 헤드는 “같은 생리적 구조”를 서로 다른 수학/공학적 관점에서 구현한다. 실험 단계에서는 서로를 **교차 검증**하는 용도로도 쓰인다(예: ukffreq가 흔들릴 때 spec_ridge가 안정해지는지, pll 락률이 떨어질 때 kfstd가 보수적으로 버텨주는지).

### 4.0.4 대역 일치와 use_track 철학

* 모든 단계(초기화·추정·후처리·평가)에 **0.08–0.50 Hz**를 완전히 일치시켰다. 모델이 관측을 ‘학습’하는 게 아니라, **생리 대역 안에서만** 의미 있게 움직이도록 가드레일을 친다.
* 각 헤드는 자신의 **track_hz(t)**를 산출하고, 평가에서는 use_track=true일 때 이를 1순위로 쓴다. 이유는 간단하다.

  * 우리가 궁극적으로 알고 싶은 것은 시계열 파형의 모양이 아니라 **호흡률(=주파수)**이며,
  * 진폭/베이스라인은 관측 노이즈에 크게 흔들리지만 **주파수 경로는 상대적으로 보존**된다.
* 이 철학은 “시간영역 파형을 잘 맞추다 보니 주파수는 틀리는” 전형적 실패를 예방한다.

### 4.0.5 어떤 상황에서 어떤 헤드가 더 적합한가(현장 선택 가이드)

* SNR 보통·움직임 작음·빠른 비교가 필요: **kfstd**

  * 파라미터가 단순하고 표준편차가 작다. 벤치마크 기준선으로 최적.
* 자세 변화가 종종 있고 호흡수 자체가 서서히 바뀜: **ukffreq**

  * log f 상태가 느린 드리프트를 자연스럽게 흡수. qf만 올바로 잡으면 **과평활/과추종의 균형**을 잡기 쉽다.
* 조명/프레임 드롭/미세 움직임이 섞여 **시간영역이 지저분**할 때: **spec_ridge**

  * 대역 제한 리지와 시간 스무딩으로 **스펙트럼의 진실**을 먼저 잡고 간다.
* 실시간/온디바이스·낮은 지연·낮은 계산량이 최우선: **pll**

  * BW, ζ, 반풍업만 잘 잡으면 **가볍고 빠르게 잠기는** 솔루션.

### 4.0.6 수치 안정성·해석 가능성·재현성의 삼각형

* 수치 안정성: UKF의 P 대칭화(+εI), R 하한, qf 범위 제한, PLL의 wrap/반풍업 등. “돌아가야 비교가 된다.”
* 해석 가능성: 모든 헤드는 **생리 대역**과 **상태 변수**(위상·주파수)에 의미가 달라붙는다. 파라미터를 보고 **왜 그렇게 나왔는지 설명할 수 있어야** 한다.
* 재현성: eval_settings.json에 **대역/윈도/use_track**을 기록하고, aux에 **track_hz/rr_bpm/coarse f0/락률**을 남긴다. 다른 컴퓨터·다른 데이터에서도 같은 선택을 재현할 수 있어야 한다.

### 4.0.7 실패 모드와 의도된 완화

* 시간영역 맞추기 집착 → 주파수 일탈: **spec_ridge** 또는 **PLL**로 주파수 앵커를 먼저 잡는다.
* 경계 포화(0.08/0.50 Hz에 자주 닿음): **PLL BW↓** 또는 **UKF qf↓**, ridge 스무딩↑, 대역 마진 축소.
* UKF 비SPD/발산: P 대칭화+εI, R 하한↑, qx/qf 재조정(주파수보다 진폭에 더 관대하게).
* 리지 점프: hop↓, median 창↑, f-경로에 단조/가속도 제약(소프트) 도입.
* PLL 과추종/잡음 락: BW↓, Kp↓, Ki↓, 초기 f0 웰치 재점검.

### 4.0.8 요약

* “호흡은 대역이 먼저다.” 시계열을 꾸미는 대신 **대역 안에서 위상·주파수를 정직하게** 추적한다.
* 오실레이터 SSM은 이 철학을 구현하는 기본 그릇이고, kfstd/ukffreq/spec_ridge/pll은 서로 다른 관점에서 같은 목표(안정·해석·재현)를 충족한다.
* 모션 5개는 “관측 채널 다양성”을 제공하고, 오실레이터 4개는 “생리적 의미의 복원”을 담당한다. 두 층을 분리해두면 **문제의 원인 분리가 쉬워지고**(관측 vs 모델링), 논문·코드 모두에서 설명 가능성이 높아진다.

---

## 4. 오실레이터 헤드 4종(모델링 계층; y(t) → 위상·주파수·진폭 복원)

이 계층은 관측 y(t)을 “공진자 상태공간” 관점에서 정제·해석한다. 헤드마다 주파수 추적 방식이 다르지만, 모든 단계에서 대역/평가 규칙을 통일한다(0.08–0.50 Hz, use_track=true면 track_hz 우선).

공통 입력/출력

* 입력: 관측 y(t), fs, 초기 주파수 f0(웰치 band-limited argmax; 실패 시 0.2 Hz)
* 출력: 정제 파형 ŝ(t), 주파수 추적 track_hz(t), 보조 rr_bpm(t)
* 공통 기본값

  * 감쇠(ρ)는 “진폭 기억 시간” τ_env로 재파라미터화: ρ = exp(-1/(fs·τ_env)). 기본 τ_env=30 s (fs=64 → ρ≈0.99948).
  * 상태잡음 Qx는 ρ 스케일과 연동: qx = qx_scale·(1 − ρ²), 기본 qx_scale=0.3 (z-score 입력 기준).
  * 관측잡음 R은 로버스트 자동화: rv = max((rv_mad_scale·MAD(y)/0.6745)², rv_floor), 기본 rv_mad_scale=1.2, rv_floor=0.08.
  * f는 항상 [0.08, 0.50] Hz로 클램프하고, NaN/Inf/음수 발생 시 이전 유효값 또는 대역 중앙으로 폴백한다.
  * Welch 기반 초기 f0는 창 20–30 s, 50% overlap, [0.08, 0.50] Hz 밴드 한정에서 피크(실패 시 0.22 Hz 근방 가중 평균)로 설정.

4.1 kfstd (표준 칼만 + RTS 스무딩; 고정 또는 완만 가변 주파수)

* 상태/관측

  * x_t = [x1, x2]^T, 2D 공진자
  * x_{t+1} = ρ·R(ωΔt)·x_t + w_t,  w_t ~ N(0, Q=diag(qx, qx))
  * y_t = [1, 0]·x_t + v_t,        v_t ~ N(0, R)
  * ω = 2π f, f는 창 구간에서 고정(f0) 또는 아주 느리게 갱신
* 알고리즘

  1. 초기화: x_0=0, P_0=diag(1,1), f=f0
  2. 칼만 필터 예측/갱신(선형) → RTS 스무더로 양방향 보정
  3. ŝ(t)=x1(t), track_hz(t)=f(고정 또는 구간 갱신값)
* 장점: 구조 단순, 수치 안정성 높음, 계산량 최소
* 주의: 실제 f(t)가 느리게 변하면 구간별 재초기화 또는 f 후보 앙상블이 필요
* 팁: 창 단위(예: 15–30 s)로 f0 재추정 → 창 경계에서 매끄럽게 보간

4.2 ukffreq (UKF; 로그-주파수 상태로 동시 추정)

* 상태/관측

  * x_t = [x1, x2, log f]^T,  f = exp(log f) ∈ [f_min, f_max]
  * x_{t+1} = [ρ·R(ωΔt)·(x1, x2), log f_t]^T + w_t,  Q = diag(qx, qx, qf)
  * y_t = [1, 0, 0]·x_t + v_t

* 시그마 포인트

  * α=1e-3, β=2, κ=0 → λ=α^2(n+κ)−n, c=n+λ
  * SPD 보장: P ← 0.5(P+P^T) + εI, ε≈1e-12
  * R 산정(로버스트): σ ≈ MAD(y)/0.6745,  R = max((1.2·σ)^2, 0.08)

* 추정 루프

  1. σ_i = sigma_points(x, P)
  2. 예측: σ_i' = f(σ_i), x̂^- = Σ Wm σ_i',  P^- = Q + Σ Wc(σ_i'−x̂^-)(·)^T
  3. 관측: z_i = h(σ_i'),  ẑ = Σ Wm z_i,  S = R + Σ Wc(z_i−ẑ)(·)^T
  4. K = Pxz S^{-1},  x̂ = x̂^- + K(y−ẑ),  P = P^- − KSK^T
  5. log f를 [ln 0.08, ln 0.50]로 클램프, track_hz = exp(log f)

* 하이퍼파라미터 가이드

  * qf: **기본 5e-5 (샘플 단위)**
    (작으면 추종 지연↑, 크면 잡음 추종/포화↑)
  * qx: qx = qx_scale·(1 − ρ²) 권장(기본 qx_scale=0.3, z-score 입력 기준)
  * P0 = diag([1.0, 1.0, 0.25^2])

* 장점: f(t) 변화를 자연스럽게 추종, 자세 변화에도 락 유지
* 실패 모드: 초기 f0 오프셋 + qf 과소 → 로컬 고정; qf 과대 → 대역내 진동/포화
* 점검: track_hz 히스토그램(경계 포화), 공분산 조건수, 혁신 잔차 분포

4.3 spec_ridge (STFT 능선 추적)

* 목표: y(t)의 시간-주파수 에너지 능선을 연속성 제약으로 추적하여 track_hz(t) 획득

* 설정

  * 창: Hann, **12 s**; hop: **1.0 s**; NFFT는 분해능 ≤0.02 Hz 되도록 충분히 크게
  * 대역 제한: **0.08–0.50 Hz**만 탐색 (마스크)
  * 연속성 패널티: ridge_penalty ≈ **250** (hop당 허용 Δf_max ≈ 0.02 Hz 기준)
  * 시간 방향 스무딩: **median 5-tap** (≈5 s)

* 알고리즘

  1. STFT → |Y(f, τ)| 스펙트럼, f∈[0.08,0.50]만 남김
  2. 동적 프로그래밍으로 릿지 최적 경로 추적  
     (목적함수: −|Y| + ridge_penalty·|Δf(τ)|^2 최소)
  3. 릿지 주파수열 f̂(τ)에 median 5-tap 스무딩 적용
  4. 선형 보간으로 샘플레이트로 확장 → track_hz(t)
  5. (선택) 공진자 위상 주입으로 ŝ(t) 재구성

* 장점: 관측 스펙트럼의 밀도를 충실히 따르며 전역 연속성으로 점프/노이즈 완화
* 한계: 계산량↑, SNR 낮으면 penalty 튜닝 민감 → ridge_penalty로 제어
* 튜닝 팁: 창↑ ⇒ 주파수 분해능↑(추종 지연↑), hop↓ ⇒ 시간 분해능↑

4.4 pll (디지털 위상 동기 루프; PI 타입)

* 목표: 내부 발진기(공진자)의 위상/주파수를 관측 y(t)에 동기화하여 track_hz(t) 추출
* 위상 검출

  * 공진자 캐리어 c(t)=cos(φ̂), s(t)=sin(φ̂) 생성
  * e_φ(t)=wrap(atan2( s*y_filt, c*y_filt )) 등 단순 위상검출기(힐버트 불필요)
* 루프 설계(Type-II, ζ≈0.707)

  * 락 대역 BW_hz ≈ 0.02–0.04
  * ω_n = 2π·BW_hz, K0 ≈ 2π
  * 연속영역: Kp = 2ζω_n/K0, Ki = ω_n^2/K0
  * 이산화: Kp_d = Kp·dt, Ki_d = Ki·dt
* 주파수/위상 업데이트

  * f̂_{t+1} = clip( f̂_t + Kp_d·e_φ + Ki_d·∑e_φ , 0.08, 0.50 )
  * φ̂_{t+1} = φ̂_t + 2π f̂_{t+1}·dt
* 반풍업(anti-windup)

  * 포화 시 적분기 동결 또는 누설(∑e_φ ← λ∑e_φ, λ≈0.98)
* 출력: track_hz(t)=f̂(t), ŝ(t)=cos(φ̂(t)) 또는 공진자 상태 x1(t)
* 장점: 순간 주파수·위상 추적에 빠르고 계산량 매우 낮음
* 한계: BW가 너무 넓으면 잡음 추종, 너무 좁으면 큰 호흡 변화 놓침
* 튜닝 팁: 초기 f0에 빠르게 수렴하려면 잠깐 BW↑ 후 서서히 BW↓(스케줄링), 락률/포화율 로깅

공통 수치 안정 트릭

* 모든 헤드에서 f 범위 포화 일관 적용
* 초기 f0는 Welch band-limited peak(실패 시 0.2 Hz)
* 공통 R 하한(1e-6)과 MAD 기반 설정
* UKF의 P는 매 스텝 대칭화(+εI)로 SPD 유지
* PLL은 wrap, 반풍업을 반드시 적용

평가/로깅 일체화

* aux/<method>/<trial>.npz에 track_hz, rr_bpm(=median(track_hz_window)×60), coarse f0, 락률/포화율 저장
* use_track=true면 평가에서 track_hz 기반 rr_bpm를 1순위 사용
* metrics_summary.txt 상단에 “Evaluation band: [0.08, 0.50] Hz | use_track=True” 명시

권장 시작 하이퍼파라미터 요약

* 공통: band=[0.08,0.50], Welch 10 s/90%, 창 평가 30 s/50%
* kfstd: ρ=0.995, qx=1e-4, R=MAD+1e-6, f는 창별 고정/완만 갱신
* ukffreq: qx=1e-4, qf=5e-5(탐색 5e-6–5e-4), α=1e-3, β=2, κ=0, P0=diag([1,1,0.1])
* spec_ridge: Hann 10 s, hop 0.5–1.0 s, ridge median 1–3 s
* pll: BW=0.03 Hz, ζ=0.707, K0=2π, 반풍업 on, f 포화 포함

실패 모드별 빠른 처방

* track_hz가 경계 0.08/0.50에 자주 닿음 → PLL BW↓ 또는 UKF qf↓, 릿지 스무딩↑
* UKF가 불안정(Cholesky 에러) → P 대칭화+εI, R 하한↑, qx/qf 균형 재조정
* 릿지 점프 많음 → hop↓, median 창↑, 대역 마진 축소(예: 0.10–0.45)
* PLL 느린 추종 → BW↑, Kp↑(단 과추종 주의), 초기 f0 보정

---

## 5. 전역 대역·GT·평가 규칙(불일치 방지 패치)

1. 평가·후처리 대역 통일

* eval.min_hz=0.08, eval.max_hz=0.50, eval.use_track=true
* 모든 필터링/웰치/능선탐색/PLL 락 범위/후처리에서 동일 대역 사용

2. COHFACE GT 정합

* 원시 HDF5 메타데이터로 GT 샘플링률 파악
* 결과 산출 시 fs_gt를 metric에 사용(단위·창 길이 오류 방지)

3. 시간 윈도잉

* per-video 고정 길이 창(예: 30 s, 50% 겹침) 평가
* 창 단위로 rr_bpm 산출: track_hz의 중앙값×60, 혹은 필터 출력 파형의 peak-to-peak로 보조

4. 결과 기록

* 각 trial 마다 aux/<method>/<trial>.npz: track_hz, rr_bpm, coarse_f0, pll_lock_ratio 등
* eval_settings.json: 대역/윈도/사용 플래그를 함께 저장(재현성)

---

## 6. Codex용 구현 지침(스켈레톤이 아닌 “현재 파이프라인 준수” 명령)

아래 프롬프트를 그대로 전달하면 된다(굵은 글씨 사용 안 함).

```
너는 내 resPyre 포크에서 ‘모션 기반 5개 방법 × 오실레이터 헤드 4개 = 20개’ 파이프라인을 기존 구조를 엄격히 지키며 구현/보완한다.

반드시 지킬 것:
1) 디렉토리/이름: src/models 같은 새 디렉토리 만들지 말고, 기존 구조만 쓴다.
   - motion/method_oscillator_wrapped.py 에서 5개 모션 기법을 감싸는 래퍼를 유지/보완
   - riv/estimators/oscillator_heads.py 에 오실레이터 4종(kfstd, ukffreq, spec_ridge, pll)을 완성/튜닝
   - run_all.py, config_loader.py, utils.py 의 훅을 바꾸지 말고 필요한 부분만 패치
   - 결과는 results/<run_label> 및 aux/<method>/<trial>.npz 에 저장

2) 대역과 평가: 0.08–0.50 Hz 를 전 구간(초기화/추정/후처리/평가)에 일치 적용.
   - config의 eval.min_hz/max_hz/use_track 읽어 공통 반영
   - print_metrics 는 사용 대역과 use_track=True 여부를 요약 표 상단에 표시

3) GT 정합: COHFACE GT는 메타에서 fs 도출 → 필요시 안전한 decimate → fs_gt 기록 후 metric 산출

4) 모션 5개는 ‘순수 관측’으로 유지(알고리즘 바꾸지 말 것). 단, 공통 대역/GT/윈도/평가 패치는 적용.

5) 오실레이터 4종 구현 세부:
   - kfstd: 2차 공진자(ρ≈0.995) + RTS. 구간별 f0(coarse Welch) 고정 또는 천천히 변동.
   - ukffreq: 상태 [x1,x2,log f]. 시그마(alpha=1e-3, beta=2, kappa=0). qf는 5e-6~5e-4 범위 튜닝.
     P 갱신 후 대칭화, eps*I 더해 SPD 유지. log f를 [log 0.08, log 0.50] 범위에 포화.
   - spec_ridge: STFT(Hann, 창≈10 s, 90% overlap, 분해능~0.01–0.02 Hz), 대역 제한 릿지 추적.
     시간 스무딩(중앙값 1–3 s) 후 track_hz 산출.
   - pll: PI-PLL. BW_hz≈0.02–0.04, ζ≈0.707, K0≈2π.
     Kp=2ζωn/K0, Ki=ωn^2/K0, 이산화(Kp*dt, Ki*dt). 위상오차 랩(wrap), 반풍업 구현.
     f는 [0.08,0.50]로 포화. 잠금률/포화율을 aux에 로깅.

6) 공통 안정화:
   - Welch coarse f0는 band-limited 최대값 사용, 없으면 0.2 Hz 기본값.
   - 관측 R은 robust(MAD) + 1e-6 하한. 과정 Q는 qx, qf를 분리 튜닝.
   - 모든 추정기의 track_hz/rr_bpm 을 aux에 저장(use_track=True면 평가에서 이것을 사용).

7) 출력과 기록:
   - metrics_summary.txt 에 각 방법별 RMSE/MAE/MAPE/CORR/PCC/CCC 표와 평가 대역/플래그 기록
   - eval_settings.json 에 대역/윈도/use_track 및 fs_gt 포함
   - 예외/경고(mpeg4 slice, cholesky fail 등)는 무해화 로깅만 하고 진행

목표:
- 순수 5개 베이스라인을 교란하지 않으면서, 4개 오실레이터 헤드를 플러그인처럼 얹어 총 20개 비교
- 대역/GT/평가 일치로 재현 가능한 비교표 생성
- 안정적 UKF/PLL 동작(수치 에러/락 실패 최소화)과 릿지 추적의 과적응 방지
```

---

## 7. 튜닝 가이드(“공통 공진자 튜닝” 기본값)

주파수 대역(전 과정 공통)
* [0.08, 0.50] Hz — 초기화/추정/후처리/평가/PLL 락/능선 탐색 모두 동일 적용

감쇠·상태/관측 잡음
* τ_env = 30 s  →  ρ = exp(−1/(fs·τ_env))  (fs=64 → ρ≈0.99948)
* qx = qx_scale·(1 − ρ²),  qx_scale = 0.3   (z-score 입력 기준)
* R = max((1.2·MAD(y)/0.6745)^2, 0.08)

UKF (ukffreq)
* α=1e-3, β=2, κ=0,  P0=diag([1.0, 1.0, 0.25^2])
* qf 기본 **1e-7** (샘플 단위), 필요 시 5e-8–5e-6 탐색
* log f ∈ [ln 0.08, ln 0.50],  P 대칭화(+εI), SPD 보정

Spec-ridge
* 창 **12 s**, hop **1 s**, ridge_penalty ≈ **250** (hop당 Δf_max≈0.02 Hz 기준)
* [0.08,0.50] Hz 마스크 → 릿지 DP 추적 → **median 5-tap** 스무딩

PLL
* 자동 이득(자연주파수/감쇠비): ζ=0.9, T_track=5 s
  - ω_n = 2π/T_track,  dt=1/fs
  - Kp_d = 2ζω_n·dt,  Ki_d = (ω_n·dt)^2
  - fs=64 → Kp_d≈0.055, Ki_d≈0.00096
* anti-windup on, f 클램프 [0.08,0.50] Hz

KF (kfstd)
* A = ρ·R(ω0·dt) (ω0는 coarse f0),  Q=diag(qx,qx),  R=위 로버스트 R
* RTS 스무딩, track_hz는 구간 고정 f0(또는 완만 갱신 정책)


---

## 8. 재현 절차와 로그 확인

실행

```
eval "$(python setup/auto_profile.py)"
python run_all.py --config configs/cohface_motion_oscillator.json --step estimate
python run_all.py --config configs/cohface_motion_oscillator.json --step evaluate
python run_all.py --config configs/cohface_motion_oscillator.json --step metrics
```

확인 포인트

* results/.../eval_settings.json 에 min_hz=0.08, max_hz=0.50, use_track=true 기록
* aux/<method>/<trial>.npz 의 track_hz 가 대역 밖으로 포화되지 않는지, PLL 락률/포화율 확인
* metrics_summary.txt 맨 위에 “Evaluation band: [0.08, 0.50] Hz | use_track=True” 표기

자주 보는 경고

* mpeg4 slice end not reached…: 디코더 경고. 프레임 드롭 거의 없으면 무시 가능
* LinAlgError: SPD 이슈는 P 대칭화+eps, R 하한, qf 완화로 해결

---

## 9. 논문 작성 틀(Scientific Reports 템플릿에 맞춘 매핑)

Introduction

* rPPG 한계(대역 중첩), 모션-생리 정합성, 상태공간 공진자 접근의 장점
* 본 연구 기여: ① 모션 기반 5종 × 오실레이터 4종의 체계적 결합, ② 생리 대역 일치 평가 프로토콜, ③ 재현 가능한 벤치마크 코드/아티팩트

Methods

* Datasets: COHFACE(주), PURE/UBFC/VIPL-HR(선택)
* Motion extraction: of_farneback, dof, profile1d_(linear/quadratic/cubic)
* Oscillator heads: kfstd, ukffreq(로그 f), spec_ridge(STFT ridge), pll(PI-PLL)
* Band/pipeline alignment: 0.08–0.50 Hz 일치, GT fs 정합, 창 기반 평가
* Metrics: RMSE/MAE/MAPE, CORR/PCC/CCC, 창 단위·비디오 단위

Results

* 표/그림은 이후 채운다. abl. study: use_track on/off, qf 스윕, PLL BW 스윕, ridge 스무딩 길이

Discussion

* 모션 관측의 생리 정합성, 상태공간 해석의 안정성, 실시간 가능성
* 실패 모드: 큰 자세 전이, ROI 이탈, 극저호흡/무호흡 근접 등과 완화 전략(PLL 락 대역, ridge backtracking, UKF qf 적응)

Code and Data Availability

* 저장소 링크, config, eval_settings.json, aux 트랙 공개 지침

---

## 10. 오픈 이슈와 다음 단계

* ROI 강건화: 포즈/세그멘테이션 기반 마스크로 모션 신호 SNR 향상
* 적응형 qf/BW: 신뢰도 지표(락률, 스펙트럼 피크 대비)로 UKF/PLL 가변
* 멀티-오실레이터: 호흡 기본파 + 2차 고조파 동시 모델링(진폭 모듈레이션 완화)
* 실시간 경량화: PLL 우선, 그 다음 UKF(저차 근사), ridge는 배치 분석용

---

## 11. 부록: 기본 하이퍼파라미터 표(권장 시작점)

* 공통 대역: [0.08, 0.50] Hz
* Welch: window 10 s, 90% overlap, band-limited peak
* kfstd: ρ=0.995, qx=1e-4, R=MAD 기반+1e-6
* ukffreq: alpha=1e-3, beta=2, kappa=0, P0=diag([1,1,0.1]), qx=1e-4, qf=5e-5, R≥1e-6
* spec_ridge: Hann창 10 s, hop 0.5–1 s, ridge median 1–3 s
* pll: BW=0.03 Hz(시작점), ζ=0.707, K0=2π, 반풍업 on, f∈[0.08,0.50] 포화

---

## 12. Optuna 튜닝/적용 요약

1. **튜닝 실행 (20개 파생 메소드, 폴백 비활성)**  
   ```bash
   python optuna_runner.py --config configs/cohface_motion_oscillator.json --output runs/optuna --n-trials 20
   ```  
   - 베이스 5개는 자동 제외, `__ukffreq/__pll/__kfstd/__spec_ridge`만 대상  
   - trial당 `gating.debug.disable_gating=true`, `use_track=true`로 폴백 없이 평가  
  - 목적함수 = **MAE + RMSE** (기본 0.85/0.15). PCC/CCC/edge/nan/jerk는 **분석용 로그**만  
   - 생리 기반으로 축소된 탐색 차원 덕분에 shard당 20 trials(TPE+MedianPruner)면 수렴

2. **리더보드/번들 확인**  
   - 완료 시 `runs/optuna/dashboards/leaderboard.csv`에 objective/지표와 best.json 경로 기록  
   - `_bundles/best20_<ts>/manifest.json` + `apply_all.sh` 자동 생성 (20개 메소드 스냅샷)

3. **최적 파라미터 일괄 적용 & 재평가**  
   ```bash
   cd runs/optuna/_bundles/best20_<ts>
   ./apply_all.sh
   ```  
   - `apply_all.sh`는 각 메소드별 best.json을 `run_all.py`에 `--override-from`으로 주입하고  
     `--override profile=paper` + `--methods <method>` 조합으로 논문용 정책을 적용한 뒤  
     `results/paper_best/<method>` 디렉터리마다 `estimate → evaluate → metrics`를 순차 실행한다.

---

## Optuna 사용 요약

1. `python optuna_runner.py -c configs/cohface_motion_oscillator.json --output runs/optuna --n-trials 20` → allowlist 20개 파생 메소드만 튜닝하며 trial마다 게이팅 폴백을 완전히 끈다.
2. 종료 시 `runs/optuna/dashboards/leaderboard.csv` 정렬과 `_bundles/best20_<ts>` 생성 여부만 확인하면 objective/MAE/PCC/CCC 집계와 best.json 경로를 한 번에 검증할 수 있다.
3. 번들 안의 `apply_all.sh`는 `python run_all.py -c <config> -s estimate evaluate metrics --auto_discover_methods=false --methods <method> --override-from <best.json> --override profile=paper -d results/paper_best/<method>` 명령을 20회 반복해 scoreboard 상위 파라미터를 재평가한다.
4. **병렬 샤딩 튜닝**  
   ```bash
   # 터미널 5개를 열고 shard index 0-4를 각각 실행 (trials=20)
   python optuna_runner.py -c configs/cohface_motion_oscillator.json \
     --output runs/optuna_shard0 \
     --num-shards 5 --shard-index 0 \
     --n-trials 20

   python optuna_runner.py -c configs/cohface_motion_oscillator.json \
     --output runs/optuna_shard1 \
     --num-shards 5 --shard-index 1 \
     --n-trials 20

   python optuna_runner.py -c configs/cohface_motion_oscillator.json \
     --output runs/optuna_shard2 \
     --num-shards 5 --shard-index 2 \
     --n-trials 20

   python optuna_runner.py -c configs/cohface_motion_oscillator.json \
     --output runs/optuna_shard3 \
     --num-shards 5 --shard-index 3 \
     --n-trials 20
     
   python optuna_runner.py -c configs/cohface_motion_oscillator.json \
     --output runs/optuna_shard4 \
     --num-shards 5 --shard-index 4 \
     --n-trials 20
   ```  
   - `--num-shards 5 --shard-index k` 조합을 터미널 5개에서 동시에 실행하면, 각 shard가 서로 다른 베이스 모션(of_farneback/dof/…)을 맡아 4개 메소드를 탐색한다.  
   - shard마다 runs/optuna_shard* 디렉터리를 둔 뒤, 필요 시 마지막에 한 번 더 `--n-trials 0` 실행으로 리더보드/번들을 합칠 수 있다.

---

## 13. 논문용 평가 절차 (게이팅 완전 제거 + 표준 메트릭 보고)

### 13.1 실행 프리셋

```bash
python run_all.py --config configs/cohface_motion_oscillator.json --step evaluate metrics \
  --override gating.debug.disable_gating=true \
  --override gating.common.constant_ptp_max_hz=0.0 \
  --override heads.ukffreq.qf=3e-4 \
  --override heads.ukffreq.qx=1e-4 \
  --override heads.ukffreq.rv_floor=0.03
```

* `disable_gating=true`로 모든 하드 게이팅·상수 승격이 비활성화돼 메트릭 왜곡을 차단한다.
* `constant_ptp_max_hz=0.0`은 paper 프로파일이라도 상수 승격이 재개되지 않음을 보장한다(추후 config 로드 시에도 안전).

### 13.2 UKF 파라미터 권장 범위

* `qf`: 1e-4-5e-4 (SNR이 극단적으로 낮으면 1e-3까지 허용) → 10-30 s 윈도에서 1-3 bpm 변동을 따라감.
* `qx`: 1e-4 기준값 → 진폭 상태가 굳지 않고 완만하게 응답.
* `rv_floor`: 0.02-0.05 → 관측잡음 하한이 과도하게 커서 필터가 둔해지는 것을 방지.
* `rv_auto=True`, `rv_mad_scale≈1.0-1.2` 유지.
* 위 범위에서 Optuna/수동 튜닝 시 `method_quality.csv`의 `track_dyn_range_hz`가 0.01 Hz 이상으로 회복되는지 확인한다.

### 13.3 NaN 처리와 kfstd 해석

* 분산이 0(또는 ε 이하)인 상수 예측은 PCC/CCC가 정의되지 않으므로 **NaN 그대로 기록**한다. 이를 0으로 대체하면 통계적 의미가 왜곡된다.
* 집계 테이블에는 항상 `nan_rate`/`len_valid`를 함께 표기해 “유효 샘플 기반 평균”임을 명시한다. 예: `PCC=0.82 (nan_rate=7%)`.
* `kfstd`는 설계상 상수 추정기라 PCC/CCC가 NaN이거나 미정의가 정상이다. 따라서 논문에서는 `__kfstd`의 MAE/RMSE만 주 지표로 보고, 상관·일치도는 NaN 그대로 둔다.

### 13.4 체크리스트

1. `method_quality.csv`에서 `constant_track_promoted`가 전부 0인지 확인.
2. `summary.json`의 `series_stats.est_std`가 0 또는 수치오차 0+에서 벗어나며, `__ukffreq`는 0.3-0.7+ 수준 PCC/CCC로 회복됐는지 확인.
3. `metrics_summary.txt` 상단에 평가 대역 `[0.08, 0.50] Hz`와 `use_track=true`가 정확히 표기돼 있는지 확인.
4. 표/도표에 `nan_rate` 열을 넣어 상수 케이스의 비율을 명시한다.
