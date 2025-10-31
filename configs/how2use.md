# configs/how2use.md

## 개요
`configs/` 디렉토리에는 실험을 재현하거나 배포용으로 제공하기 위한 JSON 설정 파일을 보관합니다. 하나의 JSON만 선택해서 `run_all.py`에 넘기면 데이터셋 선정 → ROI 추출 파라미터 → 사용 메소드 → 평가/시각화 옵션까지 한 번에 적용됩니다.

## 설정 파일 구조
```jsonc
{
  "name": "cohface_motion",            // (선택) 런 이름. 로그/리포트 정리에 활용
  "results_dir": "results",            // 결과 루트 (상대/절대 경로 모두 가능)
  "datasets": [
    {
      "name": "COHFACE",
      "roi": {                          // 선택: ROI 추출 파라미터 외부화
        "chest": {"mp_complexity": 1, "skip_rate": 10},
        "face": { }
      }
    }
  ],
  "methods": [                          // 사용할 메소드 목록
    "OF_Model",
    "DoF",
    "profile1D"                         // 문자열 하나면 linear/quadratic/cubic 세 가지가 모두 추가됨
  ],
  "eval": {                             // 평가 옵션
    "win_size": 30,                    // 초 단위 또는 "video"
    "stride": 1                        // 윈도우 stride (초)
  },
    "report": {                           // 리포트 합본 설정
    "runs": ["results/COHFACE_OF_DoF_profile1D"],
    "output": "reports/cohface_motion",   // 결과 저장 경로
    "unique_window": false                 // metrics_1w 우선 사용할지 여부
  },
  "steps": ["estimate", "evaluate", "metrics"] // 기본 실행 단계
}
```
- `datasets`/`methods`/`report.runs` 항목은 문자열 또는 상세 딕셔너리를 혼용할 수 있습니다.
- `roi.chest.skip_rate`는 **초 단위**가 아니라 “frame 스킵 주기”처럼 정의됩니다. (예: 10 → `fps * 10` 프레임마다 Mediapipe Pose 갱신)

## 명령 예시
### 1. 설정 파일 하나로 추정→평가→지표 출력까지
```bash
python3 run_all.py --config configs/cohface_motion.json
```
- `configs/cohface_motion.json` 안의 `steps`가 `estimate → evaluate → metrics`로 정의되어 있으므로 한 번 실행으로 전 단계 진행.
- 결과 디렉터리: `results/COHFACE_OF_DoF_profile1D/` (자동 생성)
- 평가 단계에서 `plots/`에 bar/scatter/time/PSD/Bland-Altman plot까지 자동 생성.

### 2. 단계별로 나눠 실행
```bash
# (1) 추정만 수행
python3 run_all.py --config configs/cohface_motion.json --step estimate

# (2) 기존 결과에 대해 다시 평가 (stride만 일시적으로 바꾸고 싶을 때)
python3 run_all.py --config configs/cohface_motion.json \
  --step evaluate --win 30 --stride 2

# (3) 저장된 metrics 출력만 하고 싶을 때
python3 run_all.py --config configs/cohface_motion.json --step metrics
```

### 3. 리포트 합본 생성
```bash
# 특정 run 디렉토리 집계 (상대/절대 경로 모두 지원)
python3 run_all.py --config configs/cohface_motion.json \
  --step report --runs results/COHFACE_OF_DoF_profile1D

# config.report.runs 를 활용하고 싶다면 --runs 생략
python3 run_all.py --config configs/cohface_motion.json --step report
```
- `reports/<이름>/combined_metrics.csv`, `combined_metrics.txt`에 종합표 저장.
- `--prefer-unique` 옵션을 주면 `metrics_1w.pkl`을 우선 사용합니다.

## CLI 단축 옵션
- `--step` 대신 기존 `-a {0,1,2,3}` 형식도 유지됩니다. (`0=estimate`, `1=evaluate`, `2=metrics`, `3=report`)
- `--results`로 config의 `results_dir`를 일시적으로 바꿀 수 있습니다.
- 평가 단계는 `--win`, `--stride`로 1회성 override 가능합니다. 플롯(bar/scatter/time/PSD/Bland-Altman)과 요약 로그는 항상 생성됩니다. 표는 기본적으로 중앙값±표준편차(median±std)로 집계됩니다.

## 새로운 config 만들기
1. `configs/` 아래에 JSON 파일 추가
2. `datasets`에 사용할 데이터셋 이름과 ROI 설정을 정의
3. `methods`에 모듈 이름을 나열 (`OF_Deep` 등 파라미터 필요 시 `name` 키 포함 딕셔너리 사용)
4. 필요 시 `eval`, `report`, `steps`로 기본 동작 지정

예시:
```json
{
  "name": "mahnob_rppg",
  "results_dir": "results",
  "datasets": [{"name": "MAHNOB"}],
  "methods": ["peak", "morph"],
  "eval": {"win_size": 45, "stride": 3},
  "steps": ["estimate", "evaluate"]
}
```

## 참고
- ROI 추출/플롯/평가에 SciPy, Matplotlib 등 Python 패키지가 필요합니다. `setup/` 절차로 제작한 conda 환경(`resPyre`)을 활성화한 뒤 실행하세요.
- 리포트 합본 시 각 run 디렉터리의 `metrics.pkl` 또는 `metrics_1w.pkl`이 존재해야 합니다.
- config 없이 실행하면 기존 기본값( COHFACE + OF/DoF/profile1D )으로 동작합니다.
- `profile1D`를 문자열로 넣으면 linear/quadratic/cubic 세 가지 보간법이 모두 포함되고, 결과 플롯에는 Bland-Altman, 시간 오버레이, PSD, scatter가 자동으로 생성됩니다. `metrics_summary.txt`에는 `CORR`(평균 Pearson 상관계수)도 추가됩니다.
