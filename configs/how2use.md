# configs/how2use.md

## 개요
`configs/` 디렉터리는 `run_all.py` 실행을 위한 **단일 진실 소스(single source of truth)** 입니다. 하나의 JSON을 선택해서 `--config`로 넘기면
데이터셋 로딩 → ROI 추출 파라미터 → 사용할 메소드/오실레이터 헤드 → 평가/게이팅/리포트 조건까지 모두 한 번에 적용됩니다.

## 설정 파일 구조(예시)
```jsonc
{
  "name": "cohface_motion_oscillator",        // run 라벨. results/<name>/ 로 저장
  "results_dir": "results",                   // 모든 산출물의 루트
  "datasets": [
    {
      "name": "COHFACE",
      "roi": {"chest": {"mp_complexity": 1, "skip_rate": 10}}
    }
  ],
  "methods": [
    "of_farneback",
    "dof",
    {"name": "profile1d_linear__spec_ridge",  // 문자열 or dict 혼용 가능
     "preproc": {"robust_zscore": {"clip": null}}}
  ],
  "eval": {
    "win_size": 30,
    "stride": 1,
    "min_hz": 0.08,
    "max_hz": 0.5,
    "use_track": true
  },
  "gating": {
    "profile": "diagnostic-relaxed"           // _BUILTIN_GATING_PROFILES 의 키 혹은 직접 정의
  },
  "oscillator": {
    "f_min": 0.08,
    "f_max": 0.5
  },
  "report": {
    "runs": ["cohface_motion_oscillator"],
    "output": "reports/cohface_motion_oscillator",
    "unique_window": false
  },
  "runtime": {
    "device": "cpu"
  },
  "steps": ["estimate", "evaluate", "metrics"]
}
```
- `datasets`/`methods`/`report.runs`는 문자열과 상세 dict를 섞어도 됩니다. 문자열 `profile1d` 하나로 linear/quadratic/cubic을 한 번에 추가할 수도 있습니다.
- `name`을 지정하면 run 디렉터리는 항상 `results/<name>`(단일 데이터셋 기준)로 생성되므로, 이후 `--runs <name>` 형식으로 바로 재사용할 수 있습니다.
- `gating.profile`은 `_DEFAULT_GATING_CFG` → 프로필 → 사용자가 입력한 override 순으로 깊은 병합됩니다.

## 실행 시나리오
### 1) 전체 파이프라인 한 번에 돌리기
```bash
python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics
```
- `steps`를 생략하면 config에 지정된 기본 단계(또는 `["estimate"]`)만 실행됩니다.
- 산출물은 `results/cohface_motion_oscillator/` 아래에 `data/`, `aux/`, `metrics/`, `plots/`, `logs/`로 자동 정리됩니다.

### 2) 단계별로 나눠 실행
```bash
# 추정만
python run_all.py -c configs/cohface_motion_oscillator.json -s estimate --num_shards 5 --shard_index 0

# 저장된 결과 재평가(윈도우/stride override)
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s evaluate \
  --win 30 --stride 1 \
  --auto_discover_methods true \
  --allow-missing-methods \
  --runs cohface_motion_oscillator

# 메트릭 요약만 다시 출력
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s metrics \
  --auto_discover_methods true \
  --runs cohface_motion_oscillator
```
- `--runs` 값은 **config의 `results_dir` 기준 상대경로**를 넘기면 됩니다. (예: `cohface_motion_oscillator` 또는 절대경로 `/.../results/cohface_motion_oscillator`).  
  `results/COHFACE_...` 처럼 `results/`를 한 번 더 붙이면 실제 경로가 `results/results/...`가 되어 평가가 실패하니 주의하세요.
- `--auto_discover_methods true`를 주면 `aux/<method>/` 폴더를 스캔해서 현재 run에 존재하는 메소드만 자동으로 평가해 줍니다.

### 3) 리포트 합본 생성
```bash
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s report \
  --runs cohface_motion_oscillator another_run \
  --prefer-unique
```
- `reports/<label>/combined_metrics.csv` + `.txt`가 생성되고, 각 run의 `metrics.pkl`/`metrics_1w.pkl`에서 중앙값·표준편차가 합쳐집니다.
- config에 이미 `report.runs`가 있다면 `--runs`를 생략해도 됩니다.

## CLI 팁
- `--override foo.bar=value` 또는 `--override-from overrides.json`으로 config의 일부를 1회성 패치할 수 있습니다. (예: `--override eval.win_size=45`)
- 평가 단계는 항상 `metrics/eval_settings.json`, `metrics/metrics.pkl`, `metrics_summary.txt`, `logs/method_quality*.{csv,json}`, `logs/methods_seen.txt`를 생성합니다. 문제 발생 시 이 로그들을 확인하세요.
- `--allow-missing-methods` 기본값은 `true`입니다. 특정 메소드가 빠져 있어도 전체 파이프라인이 멈추지 않게 하려면 그대로 두고, 엄격 검증이 필요할 때 `--no-allow-missing-methods`를 사용하세요.
- `--num_shards/--shard_index`로 메소드 집합을 N-way로 나눌 수 있습니다. `methods` 목록 전체를 기준으로 round-robin 됩니다.

## 새 config 작성 절차
1. `configs/`에 JSON 파일을 생성하고 `name`, `results_dir`, `datasets`, `methods` 블록을 채웁니다.
2. 필요에 따라 `eval`, `gating`, `oscillator`, `report`, `runtime`, `steps` 블록을 덧붙입니다.
3. 실험 기록을 위해 `notes/` 또는 `results/<run>/metadata.json`을 `tools/write_metadata.py`로 생성해 두면 재현성이 좋아집니다.

간단한 예:
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

## 참고 & 권장 워크플로
- ROI 추출·평가에는 SciPy/Matplotlib 등 추가 의존성이 필요합니다. `setup/` 문서대로 환경(`resPyre`)을 활성화한 뒤 실행하세요.
- `results/<run>/data/*.pkl` 하나당 trial이 1개 들어가며, `aux/<method>/<trial>.npz`에 오실레이터 출력이 저장됩니다. 평가가 제대로 진행됐다면 `logs/methods_seen.txt`에 메소드 명단이 기록되고, `metrics/metrics_summary.txt`에 25개 메소드 전부의 지표가 채워집니다.
- `tools/write_metadata.py --run results/<run> --command "python run_all.py ..."`를 실행하면 run 디렉터리 내에 `metadata.json`을 남길 수 있습니다. Optuna/EM 튜닝 후 산출물 정리에 적극 활용하세요.
