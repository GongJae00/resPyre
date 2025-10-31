# Dataset layout

기본 레포는 `resPyre/dataset/` 아래에 각 데이터셋 폴더가 있다고 가정합니다.
현재 내 환경은 아래 구조(숫자는 예시 subject/trial ID)로 정리돼 있으며, `run_all.py`도 여기에 맞춰 업데이트돼 있습니다.

```
resPyre/
├── dataset/
│   ├── COHFACE/
│   │   ├── 1/                # subject 1
│   │   │   ├── 0/            # trial 0
│   │   │   │   ├── data.avi
│   │   │   │   ├── data.hdf5
│   │   │   │   └── (optional) data.mkv 등
│   │   │   └── 1/
│   │   │       └── ...
│   │   ├── 2/
│   │   └── ...
│   ├── MAHNOB/
│   │   ├── 2/                # subject 2
│   │   │   ├── P1-Rec1-....avi (비디오)
│   │   │   ├── Part_1_S_Trial1_emotion.bdf (EEG/Bio)
│   │   │   └── ... (subject 폴더 내 trial별 파일)
│   │   ├── 4/
│   │   └── ...
│   └── BP4Ddef/              # (선택) BP4D 원본 구조
└── run_all.py
```

## 코드가 기대하는 것
- **COHFACE**: `dataset/COHFACE/<subject>/<trial>/data.avi` 및 `data.hdf5`
- **MAHNOB**: `dataset/MAHNOB/<subject>/<video>.avi` + 동 폴더에 BDF 등 보조 파일
  - BDF ground truth는 `pyEDFlib`을 통해 읽습니다 (`setup/requirements.txt`에 포함). 다른 라이브러리를 쓰려면 `run_all.py`의 `MAHNOB.load_gt`를 수정하세요.
- **BP4D**: 사용 시 `dataset/BP4Ddef/<subject>/<trial>/vid.avi`

`run_all.py`의 `DatasetBase`가 기본 루트를 `dataset/`으로 잡습니다. 다른 위치에 데이터를 두고 싶다면 실행 전에 환경 변수를 지정합니다.

```bash
export RESPIRE_DATA_DIR=/absolute/path/to/my_dataset_root
python run_all.py -a 0 -d results/
```

## 구조가 다른 경우 고치는 위치
1. **최상위 경로가 다르면** `RESPIRE_DATA_DIR` 환경 변수를 쓰세요.
2. **폴더 depth/파일명이 다르면** `run_all.py` 안의 각 Dataset 클래스에서 경로를 조정하세요.
   - COHFACE: `COHFACE.load_dataset()`와 `load_gt()`에서 `data.avi`, `data.hdf5` 경로를 수정
   - MAHNOB: `MAHNOB.load_dataset()`에서 비디오 파일 선택 로직을 원하는 규칙으로 변경
   - BP4D: `BP4D.load_dataset()`에서 `vid.avi` 대신 실제 파일명을 사용

모든 경로 조합은 `os.path.join`으로 처리하므로 OS에 관계없이 동일하게 동작합니다. 변경 후에는 `python run_all.py -a 0 -d results/`로 경로가 정상 인식되는지 확인하세요.
