<div align="center">
  <img width="300" alt="rePyre logo" src="https://github.com/user-attachments/assets/d1775944-ab12-426a-93cb-86c2ba78f1d0" />
  <br>
	<br>
  
  [![Paper](https://img.shields.io/badge/Paper-ACM_Computing_Surveys-1975AE?logo=acm)](https://dl.acm.org/doi/10.1145/3771763)

</div>

**resPyre** is a comprehensive framework for estimating respiratory rate from video, using different methods and datasets.

## Quick Start (COHFACE + Oscillator Heads)

```bash
# 0) (중요) 이전 결과를 재활용하지 않는다면, 기존 metrics/plots/replot만 지우고 data/aux는 그대로 두어 재평가 가능.
#     단, 2025-11 패치 이후에는 새 dataset 메타를 적용하기 위해 estimate를 최소 1회 다시 돌려야 함.
rm -rf results/cohface_motion_oscillator/metrics results/cohface_motion_oscillator/plots results/cohface_motion_oscillator/replot

# 1) Auto profile runtime threads / device
eval "$(python setup/auto_profile.py)"

# 2) Estimate (sharded example: 5 shards)
for idx in 0 1 2 3 4; do
  python run_all.py -c configs/cohface_motion_oscillator.json \
    -s estimate --num_shards 5 --shard_index $idx
done

# 3) Evaluate + Metrics (oscillator tracks only; spectral diagnostics logged)
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s evaluate metrics --auto_discover_methods true

# 3-b) Re-run evaluate/metrics later (specify the run label, not results/<label>)
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s evaluate metrics \
  --auto_discover_methods true \
  --runs cohface_motion_oscillator

# 4) (Optional) Fit EM-based Kalman gain for a tracker head (KFstd/UKFFreq on COHFACE)
python train_em.py \
  --results results/cohface_motion_oscillator \
  --dataset COHFACE \
  --method of_farneback__kfstd

#   - EM 결과(q, r)는 runs/em_params/cohface_of_farneback__kfstd.json에 저장되고,
#     이후 run_all.py를 다시 돌리면 oscillator_KFstd가 qx_override/rv_floor_override로
#     이 값을 자동으로 불러와 표준 공진자 칼만필터의 Q/R을 데이터 기반으로 설정합니다.
```

- 결과는 `results/cohface_motion_oscillator/...`에 저장됩니다. `metrics/metrics_track_summary.txt`/`metrics_spectral_summary.txt` 외에도 `logs/method_quality.csv`, `logs/method_quality_summary.json`, `logs/methods_seen.txt`를 확인하세요.
- `--runs`로 기존 산출물을 지정할 때는 run 라벨(예: `cohface_motion_oscillator`)이나 절대 경로를 사용하세요. `results/` 접두사는 내부적으로 자동으로 붙으므로 다시 추가하면 `results/results/...`처럼 잘못된 위치가 됩니다.
- 동일 run 디렉터리에 `metadata.json`을 남겨 명령어·git 커밋·autotune/EM 버전·주요 메트릭 경로를 기록하는 것을 권장합니다 (템플릿은 아래 참조).
- COHFACE 설정은 기본적으로 `gating.profile="paper"`와 `disable_gating=false`를 사용하여 tracker 품질을 자동으로 검증합니다. 논문용으로 gating을 끄고 싶다면 평가 명령어에 `--override gating.debug.disable_gating=true`를 추가하세요.
- 전역 `oscillator.qx`는 2e-4, `preproc.robust_zscore.clip`은 3.0으로 조정되어 모션 잡음이 많은 trial에서도 공진자 기반 칼만필터가 안정적으로 동작합니다.

## Pipeline Overview

The main script [run_all.py](run_all.py) drives a three-stage pipeline:

1. **estimate**: Extract motion-based respiration signals (5 baselines) and pass each through an oscillator head (KFstd, UKF; Spec-Ridge/PLL remain available but are disabled in the default config).
2. **evaluate**: Consume the oscillator-provided `track_hz` directly while logging full quality diagnostics per trial.
3. **metrics**: Aggregate medians/variability, emit tables/plots, and persist `method_quality` summaries.

## Supported Methods

The following methods are implemented:

1. Deep Learning Methods:
- [`MTTS_CAN`](deep/MTTS_CAN/train.py): Multi-Task Temporal Shift Attention Network 
- [`BigSmall`](deep/BigSmall/predict_vitals.py): BigSmall Network

2. Motion-based Methods:
- [`OF_Deep`](run_all.py): Deep Optical Flow estimation
- [`OF_Model`](run_all.py): Traditional Optical Flow (Farneback)
- [`DoF`](run_all.py): Difference of Frames
- [`profile1D`](run_all.py): 1D Motion Profile

3. rPPG-based Methods:
- [`peak`](run_all.py): Peak detection
- [`morph`](run_all.py): Morphological analysis
- [`bss_ssa`](run_all.py): Blind Source Separation with SSA
- [`bss_emd`](run_all.py): Blind Source Separation with EMD

### Oscillator Heads & SOMATA-style pipeline

- Baseline motion estimators (`of_farneback`, `dof`, `profile1d_linear`, `profile1d_quadratic`, `profile1d_cubic`) can be paired with two tracker heads (`kfstd`, `ukffreq`) for 10 additional method names such as `of_farneback__kfstd`. (Spec-Ridge/PLL remain implemented but are excluded from the current COHFACE config and Optuna search.)
- Use `configs/cohface_motion_oscillator.json` to run all 15 variants (5 baselines + 10 tracker wraps) on COHFACE with `python run_all.py --config configs/cohface_motion_oscillator.json --step estimate evaluate metrics`.
- Each wrapped method writes its smoothed waveform to the main pickle file and stores detailed diagnostics (signal, frequency track, RR summary, quality flags) under `results/<run_name>/aux/<method>/<trial>.npz`. Ensemble mode additionally stores each component under `aux/<method>/components/<trial>/<head>.npz`.
- Evaluation-wide frequency bands are controlled via `eval.min_hz`/`eval.max_hz` (default `0.08–0.5 Hz`). Every method now emits **two** metric sets per run: a track-domain table (`metrics_track*.pkl`) built from the time-varying `track_hz` (or sliding-window RR for the 5 base methods) and a spectral table (`metrics_spectral.pkl`) based on Welch peaks from the same waveform. Track heads therefore expose both their real-time tracking skill and their averaged spectral accuracy, while the base methods appear in both tables (identical values for track vs. spectral in their case).
- Oscillator heads automatically tighten/loosen their Q/R noise levels based on the spectral peak confidence and recent SNR (`spec_guidance_*` parameters in `OscillatorParams`); strong peaks damp the random walk while low-confidence trials can drift more freely.
- PLL heads also compute a Hilbert-envelope SNR to stabilise their adaptive gains, so they remain responsive even when `_last_snr` from preprocessing is unreliable.
- Quality columns now include `std_bpm`, `unique_fraction`, `edge_saturation_fraction`, `track_frac_saturated_eval`, and a synthetic `track_reliability` score.

### Parameter autotune & EM gain learning

- `_BaseOscillatorHead` logs per-trial MAD/SNR/frequency stats through `riv/estimators/params_autotune.py` → `runs/autotune/<dataset>/<method>_stats.json`.
- To override parameters globally, drop a JSON file under `runs/autotune/<dataset>/<method>.json`:

```json
{
  "params": {
    "qx_override": 5e-5,
    "rv_floor_override": 0.02,
    "qf_override": 2e-4
  }
}
```
- Additional coarse frequency initialisers (Welch/autocorr/Hilbert) are blended by confidence; their candidates are preserved in `meta["coarse_candidates"]`.
- Spectral guidance hooks are enabled by default (`spec_guidance_strength`, `spec_guidance_offset`, `spec_guidance_confidence_scale`, `spec_guidance_snr_scale`), so you can tune how aggressively the Kalman/PLL noise drops when the spectral peak is clean.
- **EM gain optimisation:** after running `estimate` (and storing tracks), execute

```bash
python train_em.py \
  --results results/cohface_motion_oscillator \
  --dataset COHFACE \
  --method profile1d_cubic__ukffreq \
  --build_autotune
```

This stacks every `aux/<method>/<trial>.npz` track, fits Q/R via EM, saves `runs/em_params/cohface_profile1d_cubic__ukffreq.json`, and appends a row to `runs/em_logs/em_training.csv`. The oscillator heads automatically load these Q/R overrides on the next run.

When `--build_autotune` is set, the script also inspects the per-trial metrics, finds the top-K (default 5) lowest-MAE trials, and writes an autotune override (`runs/autotune/cohface/<method>.json`) whose `rv_floor_override`/`qx_override` come from their median `sigma_hat`/SNR statistics. You can adjust this behaviour with `--autotune-top-k`, `--autotune-rv-scale`, and `--autotune-qx-base`.

### PLL / Spec-Ridge upgrades & Ensemble mode

- These heads remain available for manual experiments but are **not** part of the current COHFACE config or Optuna search.
- `oscillator_PLL` now uses an adaptive controller (`PLLAdaptiveController`) that scales KP/KI based on the observed SNR and adds a phase-noise shaping term; reliability is reported via track variance.
- `oscillator_Spec_ridge` performs multi-resolution STFT (0.75×/1×/1.5× window scales), computes ridge regularisation, and fuses tracks according to power-driven reliability. Meta includes `multi_resolution_used` and the final reliability score.
- Optional ensemble execution mixes the four heads:

```json
{
  "name": "profile1d_cubic__ensemble",
  "ensemble": {
    "enabled": true,
    "heads": [
      {"name": "kfstd", "weight": 1.0},
      {"name": "ukffreq", "weight": 1.2},
      {"name": "spec_ridge", "weight": 1.0},
      {"name": "pll", "weight": 0.8}
    ]
  }
}
```

The ensemble stores both the fused track and the component outputs so you can compare single-head vs. mixed performance from the same run.

### Quality logging & metadata

- `results/<run>/logs/method_quality.csv` and `method_quality_summary.json` now have extra columns summarising `track_frac_saturated_eval` and reliability percentiles. `logs/methods_seen.txt` and `logs/evaluate_appends.log` include every trial/method pair, even if metrics fail.
- After `metrics`, drop a `results/<run>/metadata.json` describing the exact command, config overrides, git commit, EM/autotune versions, and notable metric summaries. A minimal template:

```json
{
  "command": "python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics",
  "git_commit": "<abc1234>",
  "autotune": "runs/autotune/cohface/...",
  "em_params": "runs/em_params/cohface_profile1d_cubic__ukffreq.json",
  "metrics": "results/cohface_motion_oscillator/metrics/metrics_track_summary.txt"
}
```

Keeping this metadata alongside the metrics folder ensures full reproducibility for paper submissions.

## Supported Datasets 

The code works with the following datasets:

- [`BP4D`](run_all.py): BP4D Dataset
- [`COHFACE`](run_all.py): COHFACE Dataset  
- [`MAHNOB`](run_all.py): MAHNOB-HCI Dataset

## Metadata & Reproducibility

After every run, create a `metadata.json` alongside `metrics/` summarising the configuration (command line, git commit, gating overrides, autotune/EM versions, shards, etc.). Template:

```json
{
  "command": "python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics",
  "git_commit": "<commit>",
  "autotune": "runs/autotune/cohface/profile1d_cubic__ukffreq.json",
  "em_params": "runs/em_params/cohface_profile1d_cubic__ukffreq.json",
  "ensemble": false,
  "metrics_summary": "results/cohface_motion_oscillator/metrics/metrics_track_summary.txt",
  "quality_logs": "results/cohface_motion_oscillator/logs/method_quality.csv"
}
```

This, combined with `runs/em_logs/em_training.csv` and the per-trial meta stored in `aux/`, makes the entire SOMATA-style pipeline reproducible for paper submission or peer review.

- Scripted helper:

```bash
python tools/write_metadata.py \
  --run results/cohface_motion_oscillator \
  --command "python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics"
```

This reads `metrics/eval_settings.json`, fills in git commit/artifact paths, and writes `results/<run>/metadata.json`. Add `--notes "shard run 0-4"` if you need additional context.

## Optuna + EM/MLflow Integration

```bash
python optuna_runner.py \
  --config configs/cohface_motion_oscillator.json \
  --output runs/optuna_all \
  --n-trials 40 \
  --em-mode trial \
  --mlflow-uri file:mlruns/optuna \
  --mlflow-experiment respyre-optuna
```

- The allowlist now covers 10 tracker methods (KFstd/UKFFreq across 5 bases); Spec-Ridge/PLL combinations are skipped.
- Search spaces are tightened around the current COHFACE defaults (e.g., `ukffreq` sweeps `qx`≈5e-5–1.6e-4, `qf`≈5e-5–3.5e-4, `rv_floor`≈0.02–0.06) to reduce divergence and focus on physiologic drifts.
- `--em-mode off|trial|best`: run EM-based Kalman gain learning after each trial (`trial`) or only for the best configuration (`best`). Best trials automatically persist to `runs/em_params/<dataset>_<method>.json` and append a row to `runs/em_logs/em_training.csv`.
- `--mlflow-uri`, `--mlflow-experiment`: if [MLflow](https://mlflow.org) is installed, every trial logs params/metrics (including EM `q`, `r`, `ll`) to the selected tracking server. When MLflow is unavailable the script falls back to CSV logging (`trials.csv`) and prints a warning.

## Paper Evaluation Settings

For publication-ready reporting, disable gating-based promotions and keep the UKF head in a gently time-varying regime:

```bash
python run_all.py --config configs/cohface_motion_oscillator.json --step evaluate metrics \
  --override gating.debug.disable_gating=true \
  --override gating.common.constant_ptp_max_hz=0.0
```

This combination guarantees that constant-track promotion stays off (so the Pearson correlation `R` remains `NaN` whenever the prediction is truly constant) and that the UKF frequency tracker admits slow respiratory drifts instead of collapsing to 0 variance. Internally, the UKF random-walk variance `qf` is now bounded to stay within a reasonable range (≈0.1–10× the configured base value), which prevents the frequency track from becoming completely frozen or numerically unstable under aggressive spectral guidance. When aggregating metrics, always report the accompanying `nan_rate` to show how many windows produced valid correlation scores. The `kfstd` head now implements a standard linear Kalman smoother on a damped oscillator state-space model and derives a dynamic respiratory frequency track from the smoothed oscillator phase; interpret its performance using the same set of metrics (MAE/RMSE/R/SNR plus `nan_rate`) as the other tracker heads.

If you need the gentler UKF parameters shown above (`qf=3e-4`, `qx=1e-4`, `rv_floor=0.03`), edit your config (either `configs/cohface_motion_oscillator.json` or a copy) and set them inside the shared `oscillator` block or within the specific `profile1d_*__ukffreq` method entry. CLI `--override heads.*` flags are not parsed by `run_all.py`, so the values must live in the config.

## Extending the Code

### Adding New Datasets

To add a new dataset, create a class that inherits from `DatasetBase` and implement the required methods:

```python
class NewDataset(DatasetBase):
    def __init__(self):
        super().__init__()
        self.name = 'new_dataset'  # Unique dataset identifier
        self.path = self.data_dir + 'path/to/dataset/'
        self.fs_gt = 1000  # Ground truth sampling frequency
        self.data = []

    def load_dataset(self):
        # Load dataset metadata and populate self.data list
        # Each item should be a dict with:
        # - video_path: path to video file
        # - subject: subject ID
        # - chest_rois: [] (empty list, populated during processing)
        # - face_rois: [] (empty list, populated during processing) 
        # - rppg_obj: [] (empty list, populated during processing)
        # - gt: ground truth respiratory signal

    def load_gt(self, trial_path):
        # Load ground truth respiratory signal for a trial
        pass

    def extract_ROI(self, video_path, region='chest'):
        # Extract ROIs from video for given region ('chest' or 'face')
        pass

    def extract_rppg(self, video_path, method='cpu_CHROM'):
        # Extract rPPG signal from video
        pass
```

### Adding New Methods

To add a new respiratory rate estimation method, inherit from `MethodBase`:

```python
class NewMethod(MethodBase):
    def __init__(self):
        super().__init__()
        self.name = 'new_method'  # Unique method identifier
        self.data_type = 'chest'  # Input type: 'chest', 'face' or 'rppg'

    def process(self, data):
        # Implement respiratory signal extraction
        # data contains:
        # - chest_rois: list of chest ROI frames 
        # - face_rois: list of face ROI frames
        # - rppg_obj: rPPG signal object
        # - fps: video framerate
        # Return the extracted respiratory signal
        pass
```

After implementing the new classes, you can use them with the existing pipeline:

```python
methods = [NewMethod()]
datasets = [NewDataset()]
extract_respiration(datasets, methods, "results/")
```

## Paths & Requirements

- **Datasets**: By default the code looks under `<repo>/dataset/` (e.g., `dataset/COHFACE/<subject>/<trial>`). To point elsewhere, set `RESPIRE_DATA_DIR=/absolute/path/to/your/datasets` before running `run_all.py`.
- **Run directories**: If the JSON config declares `"name"`, outputs are organized under `results/<name>/...`; otherwise the legacy `results/<DATASET>_<methods>/...` folders are used.
- **Results**: Each dataset still produces pickled artifacts in its run directory (e.g., `results/cohface_motion_oscillator/data/cohface_1_0.pkl`). Passing `-d custom_dir` relocates the base directory while keeping the same structure.
- **MAHNOB dependency**: Reading MAHNOB ground-truth BDF files requires `pyEDFlib` (installed via `setup/requirements.txt`). If you use a different reader, edit `MAHNOB.load_gt` accordingly.
- **Environment**: Use `setup/setup.sh` (single `setup/requirements.txt`). Run `eval "$(python setup/auto_profile.py)"` beforehand to set `DEVICE`/`NUM_WORKERS` 등 런타임 변수; `run_all.py` 는 `DEVICE` 값(`cuda:0`, `cuda:1`, `cpu` 등)에 맞춰 OF_Deep 등의 GPU 사용을 자동 설정합니다.

Quickstart:

```bash
# Full stack (auto-detect GPU)
./setup/setup.sh

# Force CPU-only
./setup/setup.sh --cpu-only

# Explicit CUDA 11.7 for PyTorch 1.13.1
./setup/setup.sh --cuda 117
```

## Reference

If you use this code, please cite the paper:

```
@article{boccignone2025remote,
  title={Remote Respiration Measurement with RGB Cameras: A Review and Benchmark},
  author={Boccignone, Giuseppe and Cuculo, Vittorio and D'Amelio, Alessandro and Grossi, Giuliano and Lanzarotti, Raffaella and Patania, Sabrina},
  journal={ACM Computing Surveys},
  year={2025},
  publisher={ACM New York, NY}
}
```

## License

This project is licensed under the GNU General Public License - see the [LICENSE](LICENSE) file for details.


---

```mermaid
flowchart LR
    subgraph obs["Observation Layers"]
        video[Video Frames]
        roi[Chest ROI Extraction]
        motion1["1D Motion Signals\n(Optical Flow, DoF, 1D Profiles)"]
        y["Raw Motion y(t)"]
        subgraph pre["Preprocess (per-signal)"]
            det[Linear Detrend]
            bp["Bandpass 0.08–0.5 Hz"]
            sign[Sign Alignment]
            rz["Robust Z-Score"]
        end
        video --> roi --> motion1 --> y
        y --> det --> bp --> sign --> rz --> ytilde["ŷ(t)"]
    end

    ytilde --> osc

    subgraph osc["Oscillator Head"]
        band["Respiratory Band Config\n(f_min=0.08 Hz, f_max=0.5 Hz)"]
        coarse["Coarse RR (Welch PSD) → f₀"]
        ssm["State-Space Oscillator Setup\n(F(ρ, ω₀), Q, R from MAD)"]
        band --> coarse --> ssm
        ssm -->|ŷ(t), f₀, F, Q, R| trackers
    end

    subgraph trackers["Tracker Head (Parallel)"]
        direction TB
        subgraph kf["KFstd"]
            kfin["Oscillator Kalman smoother\nState [x1, x2] on F(ρ,ω₀)"]
            kfout["ŝ_KF(t)=x̂₁(t), track_hz≈(1/2π)dφ/dt,\nA=√(x̂1²+x̂2²), φ=atan2(x̂2,x̂1)"]
            kfin --> kfout
        end
        subgraph ukf["UKF"]
            ukfin["State [x1, x2, log f]\nUnscented update"]
            ukfout["ŝ_UKF(t), track_hz=exp(log f̂),\nA/φ from [x̂1,x̂2]"]
            ukfin --> ukfout
        end
        subgraph spec["Spec-Ridge"]
            stft["STFT (12 s win, 50% overlap)"]
            ridge["Dynamic-program ridge\n+ sub-bin refine"]
            smooth["Temporal smoothing & interpolation"]
            specout["track_hz^SR(t)\n(ŷ(t) reused as ŝ)"]
            stft --> ridge --> smooth --> specout
        end
        subgraph pll["PLL"]
            hilbert["Hilbert transform → φ_y(t)"]
            loop["Phase detector + PI loop\n(NCO init at f₀)"]
            pllout["ŝ_PLL(t)=cos φ_PLL,\ntrack_hz=ω/(2π), φ=φ_PLL"]
            hilbert --> loop --> pllout
        end
    end

    trackers --> payload["Result Payload\n{signal_hat, track_hz, rr_bpm, meta}"]

```
- **실행 체크리스트:**  
  1. `run_all.py -s estimate`를 최소 1회 실행해 `dataset_name=cohface`가 기록된 최신 aux를 생성한다(EM/autotune 로더는 이 문자열을 사용한다).  
  2. `runs/em_params/cohface_<method>.json`과 `runs/autotune/cohface/<method>.json`이 준비돼 있는지 확인한다.  
  3. `run_all.py -s evaluate metrics --auto_discover_methods true`를 실행하면 fallback 없이 모든 트랙이 그대로 평가되며, 품질 지표는 `results/<run>/logs/method_quality*.{csv,json}`에서 확인한다.  
  4. 필요 시 Optuna/EM 학습을 돌리면 새 JSON이 즉시 적용된다(추가 재시작 불필요).
