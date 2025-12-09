# CLAUDE.md - AI Assistant Guide for resPyre

This document provides AI assistants with essential context about the resPyre codebase structure, development workflows, and conventions.

## Project Overview

**resPyre** is a comprehensive framework for **respiratory rate estimation from video** using multiple methods and datasets. This is a research-focused project implementing methods from the paper "Remote Respiration Measurement with RGB Cameras: A Review and Benchmark" (ACM Computing Surveys, 2025).

### Key Capabilities:
- Motion-based respiratory rate extraction (optical flow, frame differencing, 1D profiles)
- Deep learning models (MTTS_CAN, BigSmall)
- rPPG-based methods
- Advanced state-space oscillator tracking (Kalman filters, UKF, PLL)
- Hyperparameter optimization (Optuna)
- Parameter learning via EM algorithms
- Multi-dataset benchmarking (COHFACE, MAHNOB, BP4D)

### Research Context:
This is production research code with ~10,000+ lines. It balances academic experimentation with reproducibility requirements. Code quality varies: newer oscillator components are well-documented, while legacy dataset loaders may be less polished.

## Repository Structure

```
resPyre/
├── riv/                    # Respiratory signal processing & oscillator heads
│   ├── estimators/         # Oscillator tracking algorithms (KFstd, UKF, PLL, Spec-Ridge)
│   │   ├── oscillator_heads.py       # Core tracking implementations
│   │   ├── head_ensemble.py          # Multi-head fusion
│   │   └── params_autotune.py        # Parameter auto-tuning
│   ├── optim/              # EM-based parameter optimization
│   │   └── em_kalman.py              # Q/R matrix learning
│   ├── IMS.py              # Pulse segmentation from rPPG
│   └── resp_from_rPPG.py   # rPPG-based respiratory extraction
│
├── deep/                   # Deep learning models
│   ├── MTTS_CAN/           # Multi-Task Temporal Shift Attention Network (TensorFlow)
│   │   ├── model.py, train.py, predict_vitals.py
│   └── BigSmall/           # BigSmall Network
│
├── motion/                 # Motion-based estimation methods
│   ├── motion.py           # DoF, Optical Flow, profile1D algorithms
│   └── method_oscillator_wrapped.py  # Wrapper combining base methods + oscillator heads
│
├── external/               # Third-party libraries (vendored)
│   ├── somata-main/        # State-space oscillator modeling (Stanford Purdon Lab)
│   ├── pykalman-main/      # Kalman filtering
│   ├── ssqueezepy-master/  # Synchrosqueezing wavelets
│   └── scikit-dsp-comm-master/
│
├── configs/                # JSON configuration files
│   ├── cohface_motion_oscillator.json  # Main config
│   └── cohface_motion.json
│
├── dataset/                # Dataset storage (set via RESPIRE_DATA_DIR)
├── results/                # Output directory (estimates, metrics, plots)
├── runs/                   # EM params, autotune overrides, Optuna trials
├── setup/                  # Installation scripts
│   ├── setup.sh, requirements.txt
│   └── auto_profile.py     # Hardware detection
│
├── tools/                  # Utility scripts
│   └── write_metadata.py   # Generate metadata.json for runs
│
├── run_all.py              # Main pipeline executor (4098 lines)
├── train_em.py             # EM-based Kalman parameter learning (258 lines)
├── optuna_runner.py        # Hyperparameter optimization (904 lines)
├── test.py                 # Post-hoc evaluation (274 lines)
├── config_loader.py        # Configuration management
├── errors.py               # Metric computations (MAE, RMSE, Pearson R, CCC)
├── utils.py                # Common utilities
└── FIRfilt.py              # FIR filter implementations
```

## Core Concepts

### 1. Three-Stage Pipeline

The main workflow (`run_all.py`) has three stages:

1. **estimate**: Extract respiratory signals from video
   - Load dataset videos
   - Extract ROIs (chest/face) using MediaPipe
   - Apply methods (motion/deep/rPPG)
   - Apply oscillator heads for tracking
   - Save pickled results to `results/<run>/data/`

2. **evaluate**: Compare estimates to ground truth
   - Load estimates and ground truth
   - Compute windowed metrics (MAE, RMSE, R, SNR)
   - Save trial-level metrics to `results/<run>/eval/`

3. **metrics**: Aggregate statistics and generate reports
   - Compute method-level statistics (median, IQR)
   - Generate plots and summary tables
   - Save to `results/<run>/metrics/`

### 2. Method Naming Convention

**Base methods** (lowercase with underscores):
- `of_farneback` - Farneback optical flow
- `of_deep` - Deep optical flow (RAFT)
- `dof` - Difference of frames
- `profile1d_linear`, `profile1d_quadratic`, `profile1d_cubic` - 1D motion profiles
- `mtts_can`, `bigsmall` - Deep learning methods
- `peak`, `morph`, `bss_ssa`, `bss_emd` - rPPG-based methods

**Composite methods** (base + oscillator head):
- `of_farneback__kfstd` - Optical flow + Kalman filter
- `profile1d_cubic__ukffreq` - Cubic profile + UKF with frequency tracking
- `dof__pll` - Difference of frames + Phase-Locked Loop
- `of_farneback__ensemble` - Optical flow + multi-head ensemble

Format: `<base_method>__<oscillator_head>`

**Oscillator heads**:
- `kfstd` - Standard Kalman smoother on damped oscillator
- `ukffreq` - Unscented Kalman Filter with frequency state
- `spec_ridge` - STFT-based ridge extraction
- `pll` - Phase-Locked Loop with adaptive control
- `ensemble` - Weighted fusion of multiple heads

### 3. Configuration System

Configurations are JSON files with deep merging support:

```json
{
  "name": "cohface_motion_oscillator",  // Run label
  "datasets": [{"name": "COHFACE", ...}],
  "methods": ["of_farneback", "of_farneback__kfstd", ...],
  "eval": {
    "win_size": 30,           // Window size in seconds
    "min_hz": 0.08,           // Min respiratory frequency
    "max_hz": 0.5             // Max respiratory frequency
  },
  "oscillator": {             // Global oscillator parameters
    "qx": 0.00012,            // State noise
    "qf": 0.00012,            // Frequency random-walk noise
    "rv_floor": 0.03          // Observation noise floor
  },
  "preproc": {                // Preprocessing
    "robust_zscore": {"enabled": true, "clip": 2.5}
  },
  "gating": {                 // Quality control thresholds
    "profile": "paper",       // Gating profile
    "debug": {"disable_gating": false}
  }
}
```

**Override mechanism**:
```bash
# Command-line overrides
python run_all.py -c config.json --override gating.debug.disable_gating=true
python run_all.py -c config.json --override oscillator.qx=0.0001
```

### 4. State-Space Oscillator Framework

The core innovation is treating respiration as a **damped oscillator** with Kalman tracking:

**Signal Flow**:
```
Video → ROI extraction → Base method (e.g., optical flow)
  → Raw signal y(t)
  → Preprocessing (detrend, bandpass 0.08-0.5 Hz, robust z-score)
  → Preprocessed signal ŷ(t)
  → Oscillator head (KF/UKF/PLL/Spec)
  → Tracked signal ŝ(t), frequency track f(t), RR estimate
```

**Key files**:
- `riv/estimators/oscillator_heads.py` - Main tracking implementations
- `motion/method_oscillator_wrapped.py` - Wrapper integrating base + head
- `riv/optim/em_kalman.py` - EM learning for Q/R matrices

### 5. Parameter Learning & Auto-Tuning

**EM-based learning** (Expectation-Maximization):
```bash
python train_em.py \
  --results results/cohface_motion_oscillator \
  --dataset COHFACE \
  --method profile1d_cubic__ukffreq \
  --build_autotune
```

Output: `runs/em_params/cohface_profile1d_cubic__ukffreq.json`
```json
{
  "q": 0.00015,              // Learned state noise
  "r": 0.025,                // Learned observation noise
  "log_likelihood": -1234.5,
  "iterations": 15
}
```

**Auto-tuning**: Automatically adjusts parameters based on historical performance:
- Stored in `runs/autotune/<dataset>/<method>.json`
- Parameters: `qx_override`, `rv_floor_override`, `qf_override`
- Loaded automatically by oscillator heads

### 6. Result Storage

```
results/<run_name>/
├── data/                   # Pickled estimates (one per trial)
│   └── cohface_1_0.pkl     # {method_name: {signal, rr_bpm, meta}, ...}
├── aux/                    # Detailed oscillator diagnostics (.npz files)
│   └── of_farneback__kfstd/
│       └── cohface_1_0.npz # {signal, track_hz, rr_summary, quality_flags, ...}
├── eval/                   # Trial-level metrics
│   └── cohface_1_0.pkl     # {method_name: {mae, rmse, r, snr, ...}, ...}
├── metrics/                # Aggregated statistics
│   ├── metrics_track_summary.txt
│   ├── metrics_spectral_summary.txt
│   ├── metrics_track.pkl   # Pandas DataFrame
│   └── plots/              # Generated visualizations
├── logs/                   # Quality logging
│   ├── method_quality.csv
│   ├── method_quality_summary.json
│   └── methods_seen.txt
└── metadata.json           # Run metadata (command, git commit, etc.)
```

## Common Development Tasks

### Task 1: Run Full Pipeline on COHFACE

```bash
# 1. Set up environment
eval "$(python setup/auto_profile.py)"

# 2. Clean previous results (optional)
rm -rf results/cohface_motion_oscillator/metrics \
       results/cohface_motion_oscillator/plots

# 3. Run pipeline (single command)
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s estimate evaluate metrics

# 4. Generate metadata
python tools/write_metadata.py \
  --run results/cohface_motion_oscillator \
  --command "python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics"
```

### Task 2: Run Sharded Estimation (Parallel)

```bash
# Estimate in 5 parallel shards
for idx in 0 1 2 3 4; do
  python run_all.py -c configs/cohface_motion_oscillator.json \
    -s estimate --num_shards 5 --shard_index $idx &
done
wait

# Then evaluate all at once
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s evaluate metrics --auto_discover_methods true
```

### Task 3: Re-evaluate Existing Results

```bash
# Re-run evaluation without re-estimating
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s evaluate metrics \
  --runs cohface_motion_oscillator \
  --auto_discover_methods true
```

**Important**: `--runs` takes the run label (e.g., `cohface_motion_oscillator`), NOT the full path. Don't use `results/cohface_motion_oscillator` as it will create `results/results/...`.

### Task 4: Learn EM Parameters for a Method

```bash
# After running estimate, learn Q/R parameters
python train_em.py \
  --results results/cohface_motion_oscillator \
  --dataset COHFACE \
  --method of_farneback__kfstd \
  --build_autotune

# Check output
cat runs/em_params/cohface_of_farneback__kfstd.json
cat runs/autotune/cohface/of_farneback__kfstd.json

# Re-run estimation to apply learned parameters
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s estimate evaluate metrics
```

The oscillator heads automatically load EM params and autotune overrides on initialization.

### Task 5: Hyperparameter Optimization with Optuna

```bash
python optuna_runner.py \
  --config configs/cohface_motion_oscillator.json \
  --output runs/optuna_all \
  --n-trials 40 \
  --em-mode trial \
  --mlflow-uri file:mlruns/optuna \
  --mlflow-experiment respyre-optuna
```

**Flags**:
- `--em-mode trial`: Run EM after each trial
- `--em-mode best`: Run EM only for best trial
- `--em-mode off`: No EM learning
- `--mlflow-*`: MLflow experiment tracking (optional)

Output: `runs/optuna_all/study.db` (Optuna database)

### Task 6: Add a New Method

**For base methods** (e.g., new motion estimator):

1. Implement method in appropriate module (`motion/motion.py`, `riv/resp_from_rPPG.py`, etc.)
2. Register in `run_all.py`:
   ```python
   elif method_name == 'my_new_method':
       result = my_new_method_function(trial_data)
   ```
3. Add to config JSON:
   ```json
   "methods": ["my_new_method", "my_new_method__kfstd", ...]
   ```

**For oscillator heads**:

1. Inherit from `_BaseOscillatorHead` in `riv/estimators/oscillator_heads.py`
2. Implement required methods: `_process_impl()`, `_get_params_for_autotune()`
3. Register in `method_oscillator_wrapped.py`

### Task 7: Add a New Dataset

Create a class inheriting from `DatasetBase` in `run_all.py`:

```python
class NewDataset(DatasetBase):
    def __init__(self):
        super().__init__()
        self.name = 'NEWDATASET'  # Uppercase convention
        self.path = self.data_dir + 'newdataset/'
        self.fs_gt = 1000  # Ground truth sampling rate
        self.data = []

    def load_dataset(self):
        # Populate self.data with trial metadata
        # Each item: {video_path, subject, chest_rois, face_rois, rppg_obj, gt}
        pass

    def load_gt(self, trial_path):
        # Load ground truth respiratory signal
        pass

    def extract_ROI(self, video_path, region='chest'):
        # Extract ROI using MediaPipe or custom method
        pass
```

Then add to config:
```json
"datasets": [{"name": "NEWDATASET", ...}]
```

## Key Files to Know

### Critical Files (Modify with Care)

1. **run_all.py** (4098 lines)
   - Main pipeline orchestrator
   - Dataset loaders (COHFACE, MAHNOB, BP4D)
   - Method dispatching logic
   - Parallel processing with sharding
   - **Lines 1500-2500**: Dataset classes (intricate trial loading logic)
   - **Lines 2800-3500**: Main extraction loop

2. **riv/estimators/oscillator_heads.py** (~1200 lines)
   - Core oscillator tracking implementations
   - Preprocessing pipeline (detrend, bandpass, robust z-score)
   - Quality gating logic
   - **Classes**: `_BaseOscillatorHead`, `oscillator_KFstd`, `oscillator_UKFFreq`, `oscillator_PLL`, `oscillator_Spec_ridge`

3. **motion/method_oscillator_wrapped.py** (~350 lines)
   - Wrapper combining base methods with oscillator heads
   - Ensemble mode implementation
   - EM params and autotune loading

4. **config_loader.py** (145 lines)
   - Deep config merging
   - Override parsing (`--override key.subkey=value`)
   - Method-specific parameter extraction

### Important Utility Files

5. **errors.py** (9586 lines - mostly due to embedded legacy code)
   - Metric computations: MAE, RMSE, MAPE, Pearson R, CCC, SNR
   - Windowed metric computation (30s windows)
   - **Main functions**: `errors_multiple_methods()`, `compute_errors_window()`

6. **utils.py** (638 lines)
   - Signal processing utilities (bandpass, detrending, z-score)
   - Video I/O helpers
   - ROI utilities

7. **riv/optim/em_kalman.py** (412 lines)
   - EM algorithm for Kalman smoother
   - Q/R matrix estimation from tracks
   - **Main class**: `KalmanEM`

8. **optuna_runner.py** (904 lines)
   - Optuna study configuration
   - Search space definitions for each oscillator head
   - MLflow integration
   - EM triggering logic

### Configuration Files

9. **configs/cohface_motion_oscillator.json**
   - Primary configuration for COHFACE + oscillator pipeline
   - 15 methods: 5 base + 10 wrapped (kfstd/ukffreq)
   - Gating enabled by default
   - Global oscillator parameters

10. **setup/requirements.txt**
    - All Python dependencies
    - Includes deep learning (PyTorch, TensorFlow), CV (OpenCV, MediaPipe), signal processing (scipy, pyVHR)

## Development Workflows

### Workflow 1: Iterative Method Development

```bash
# 1. Implement method in appropriate file
vim motion/motion.py  # or riv/estimators/oscillator_heads.py

# 2. Add minimal test to run_all.py method dispatcher

# 3. Test on single trial
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s estimate --num_shards 10 --shard_index 0  # First trial only

# 4. Check output
python -c "import pickle; print(pickle.load(open('results/cohface_motion_oscillator/data/cohface_1_0.pkl','rb')).keys())"

# 5. Evaluate single trial
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s evaluate --runs cohface_motion_oscillator

# 6. Once working, run full dataset
python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics
```

### Workflow 2: Parameter Tuning

```bash
# 1. Run baseline with default parameters
python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics

# 2. Examine results
cat results/cohface_motion_oscillator/metrics/metrics_track_summary.txt

# 3. Try EM learning
python train_em.py --results results/cohface_motion_oscillator \
  --dataset COHFACE --method of_farneback__kfstd --build_autotune

# 4. Re-run with learned parameters (automatically loaded)
rm -rf results/cohface_motion_oscillator/data results/cohface_motion_oscillator/aux
python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics

# 5. Compare metrics (old vs new)
diff results_backup/metrics/metrics_track_summary.txt results/cohface_motion_oscillator/metrics/metrics_track_summary.txt

# 6. If better, commit learned params
git add runs/em_params/cohface_of_farneback__kfstd.json
git commit -m "Add learned EM params for of_farneback__kfstd on COHFACE"
```

### Workflow 3: Hyperparameter Search

```bash
# 1. Define search space in optuna_runner.py (already done for kfstd/ukffreq)

# 2. Run Optuna study
python optuna_runner.py \
  --config configs/cohface_motion_oscillator.json \
  --output runs/optuna_$(date +%Y%m%d_%H%M%S) \
  --n-trials 50 \
  --em-mode best

# 3. Examine best trial
python -c "import optuna; study = optuna.load_study('respyre_study', 'sqlite:///runs/optuna_20250101_120000/study.db'); print(study.best_trial)"

# 4. Apply best parameters to config
# Edit configs/cohface_motion_oscillator.json with best params

# 5. Validate with full run
python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics
```

### Workflow 4: Paper Evaluation (Reproducibility)

```bash
# 1. Disable gating for fair comparison
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s estimate evaluate metrics \
  --override gating.debug.disable_gating=true

# 2. Generate metadata for reproducibility
python tools/write_metadata.py \
  --run results/cohface_motion_oscillator \
  --command "python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics --override gating.debug.disable_gating=true" \
  --notes "Paper evaluation run with gating disabled"

# 3. Archive results
tar -czf cohface_paper_results_$(date +%Y%m%d).tar.gz \
  results/cohface_motion_oscillator/metrics/ \
  results/cohface_motion_oscillator/logs/ \
  results/cohface_motion_oscillator/metadata.json \
  runs/em_params/cohface_*.json \
  runs/autotune/cohface/*.json

# 4. Document git commit
git log -1 --pretty=format:"%H" > results/cohface_motion_oscillator/git_commit.txt
```

## Code Conventions & Patterns

### Naming Conventions

- **Datasets**: UPPERCASE (`COHFACE`, `MAHNOB`, `BP4D`)
- **Methods**: lowercase_with_underscores (`of_farneback`, `profile1d_cubic`)
- **Composite methods**: `base__head` format (`of_farneback__kfstd`)
- **Files**: lowercase with underscores (`oscillator_heads.py`)
- **Classes**: PascalCase (`DatasetBase`, `_BaseOscillatorHead`)
- **Functions**: lowercase_with_underscores (`extract_respiration`, `compute_errors_window`)
- **Config keys**: lowercase with underscores or camelCase (mixed in legacy code)

### Code Style

- **Indentation**: 4 spaces
- **Line length**: No strict limit, but generally <120 chars
- **Imports**: Grouped by stdlib, third-party, local (not always consistently followed)
- **Type hints**: Present in newer code (oscillator_heads.py), absent in legacy code
- **Docstrings**: Present for major functions, often missing for utility functions
- **Comments**: Extensive in oscillator code, sparse in dataset loaders

### Data Flow Patterns

**Pickled dictionaries** are the primary data exchange format:

```python
# Estimate output: results/<run>/data/<trial>.pkl
{
  'method_name': {
    'signal': np.array,      # Estimated respiratory signal
    'rr_bpm': float,         # Mean respiratory rate
    'meta': dict             # Method-specific metadata
  },
  ...
}

# Evaluation output: results/<run>/eval/<trial>.pkl
{
  'method_name': {
    'mae': float,
    'rmse': float,
    'r': float,              # Pearson correlation
    'snr': float,
    'nan_rate': float        # Fraction of NaN windows
  },
  ...
}
```

**NPZ auxiliary files** for detailed oscillator diagnostics:

```python
# results/<run>/aux/<method>/<trial>.npz
{
  'signal': np.array,           # Tracked signal
  'track_hz': np.array,         # Time-varying frequency track
  'rr_summary': dict,           # RR statistics
  'quality_flags': dict,        # Quality metrics
  'spectral_peak_confidence': np.array,
  'coarse_candidates': list     # Initial frequency estimates
}
```

### Error Handling Patterns

- **NaN propagation**: Metrics gracefully handle NaN values (e.g., constant predictions)
- **Try-except with logging**: Failures are logged but don't stop pipeline
- **Fallback values**: Many functions have sensible defaults when computation fails
- **Quality gating**: Poor-quality signals are flagged but still processed

### Common Pitfalls

1. **Path confusion with `--runs`**:
   - CORRECT: `--runs cohface_motion_oscillator`
   - WRONG: `--runs results/cohface_motion_oscillator` (creates `results/results/...`)

2. **Forgetting to clear old results**:
   - `estimate` won't overwrite existing `.pkl` files by default
   - Solution: `rm -rf results/<run>/data results/<run>/aux` before re-running

3. **EM params not loading**:
   - Check filename format: `runs/em_params/<dataset>_<method>.json`
   - Dataset name must match (case-sensitive): `COHFACE`, not `cohface`
   - Method name must be exact: `of_farneback__kfstd`, not `of_farneback_kfstd`

4. **Config overrides not working**:
   - Some parameters are method-specific, not global
   - Use `--override oscillator.qx=0.0001` for global
   - Edit JSON directly for method-specific overrides

5. **Sharding issues**:
   - All shards must complete before running `evaluate`
   - Check for missing `.pkl` files: `ls results/<run>/data/ | wc -l`

6. **DEVICE environment variable**:
   - Set with `eval "$(python setup/auto_profile.py)"`
   - Affects GPU usage for `of_deep` and deep learning methods
   - Can override manually: `export DEVICE=cpu` or `export DEVICE=cuda:0`

7. **MediaPipe initialization**:
   - Requires camera permission on some systems
   - May fail silently; check for empty `chest_rois` list

## Testing and Validation

### No Traditional Unit Tests

This repository uses **integration testing** through the pipeline:

1. **Estimate phase** validates method output format
2. **Evaluate phase** validates metric computation
3. **Metrics phase** validates aggregation logic

### Manual Testing Checklist

When modifying code, test with:

```bash
# 1. Single trial (fast smoke test)
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s estimate --num_shards 10 --shard_index 0

# 2. Check output format
python -c "
import pickle
data = pickle.load(open('results/cohface_motion_oscillator/data/cohface_1_0.pkl', 'rb'))
for method, result in data.items():
    print(f'{method}: signal shape={result[\"signal\"].shape}, rr_bpm={result[\"rr_bpm\"]:.2f}')
"

# 3. Evaluate single trial
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s evaluate --runs cohface_motion_oscillator

# 4. Check metrics
python -c "
import pickle
data = pickle.load(open('results/cohface_motion_oscillator/eval/cohface_1_0.pkl', 'rb'))
for method, metrics in data.items():
    print(f'{method}: MAE={metrics[\"mae\"]:.2f}, RMSE={metrics[\"rmse\"]:.2f}, R={metrics[\"r\"]:.2f}')
"

# 5. Full run on small shard
python run_all.py -c configs/cohface_motion_oscillator.json \
  -s estimate evaluate metrics --num_shards 10 --shard_index 0
```

### Validation Metrics

**Primary metrics** (lower is better):
- **MAE**: Mean Absolute Error (breaths/min)
- **RMSE**: Root Mean Squared Error (breaths/min)

**Secondary metrics**:
- **Pearson R**: Correlation coefficient (-1 to 1, higher is better)
- **CCC**: Concordance Correlation Coefficient
- **SNR**: Signal-to-Noise Ratio (dB, higher is better)
- **nan_rate**: Fraction of windows with NaN correlation (lower is better)

**Quality metrics** (from oscillator heads):
- **track_frac_saturated_eval**: Fraction of time track hits band edges
- **std_bpm**: Standard deviation of tracked BPM
- **unique_fraction**: Diversity of tracked values
- **track_reliability**: Synthetic quality score (0-1)

### Regression Testing

Before committing major changes:

```bash
# 1. Save baseline results
cp -r results/cohface_motion_oscillator results_baseline

# 2. Make changes

# 3. Re-run pipeline
python run_all.py -c configs/cohface_motion_oscillator.json -s estimate evaluate metrics

# 4. Compare metrics
python -c "
import pandas as pd
old = pd.read_pickle('results_baseline/metrics/metrics_track.pkl')
new = pd.read_pickle('results/cohface_motion_oscillator/metrics/metrics_track.pkl')
diff = new['mae'] - old['mae']
print('MAE changes:')
print(diff.sort_values())
"

# 5. If acceptable, commit
git add .
git commit -m "Description of changes"
```

## Git Workflow

### Branch Naming

Current branch: `claude/claude-md-miya3dqj0eo0qpe4-01M6xKzsAqC2Dr1S1jpDnTDJ`

Pattern: `claude/<session_id>`

### Commit Guidelines

**Good commit messages**:
```
Add EM parameter learning for KFstd head

Implement Expectation-Maximization algorithm to learn Q/R matrices
for Kalman smoother. Saves parameters to runs/em_params/ and
automatically loads on subsequent runs.

Tested on COHFACE dataset with of_farneback__kfstd method.
```

**What to commit**:
- Source code changes
- Configuration files
- Learned parameters (if stable and validated)
- Documentation updates

**What NOT to commit**:
- `results/` directory (too large, experiment-specific)
- `dataset/` directory (data files)
- `__pycache__/`, `*.pyc`
- Temporary files (`.pkl.tmp`, `.swp`)
- Large model checkpoints (unless essential)

### Push Workflow

```bash
# 1. Stage changes
git add file1.py file2.py configs/new_config.json

# 2. Commit with descriptive message
git commit -m "Add support for new oscillator head: spec_ridge_v2"

# 3. Push to feature branch
git push -u origin claude/claude-md-miya3dqj0eo0qpe4-01M6xKzsAqC2Dr1S1jpDnTDJ

# If push fails due to network errors, retry with exponential backoff
# (handled automatically in typical workflows)
```

### Important Git Notes

- **NEVER** force push to main/master
- **ALWAYS** use feature branches for development
- Current branch name **must** start with `claude/` and end with session ID
- Push will fail with 403 if branch name is incorrect

## Environment Setup

### Hardware Detection

```bash
# Auto-detect GPU/CPU and set environment variables
eval "$(python setup/auto_profile.py)"

# This sets:
# - DEVICE: cuda:0, cuda:1, or cpu
# - NUM_WORKERS: optimal number for data loading
# - Other runtime parameters
```

### Manual Environment Variables

```bash
# Override dataset directory
export RESPIRE_DATA_DIR=/path/to/datasets

# Force CPU-only
export DEVICE=cpu

# Use specific GPU
export DEVICE=cuda:1

# Set number of workers
export NUM_WORKERS=4
```

### Installation

```bash
# Full setup (auto-detect CUDA)
./setup/setup.sh

# CPU-only setup
./setup/setup.sh --cpu-only

# Specific CUDA version (e.g., CUDA 11.7)
./setup/setup.sh --cuda 117
```

## Troubleshooting

### Issue: "Cannot find dataset"

**Symptoms**: `FileNotFoundError` when loading dataset

**Solutions**:
1. Check `RESPIRE_DATA_DIR` environment variable
2. Verify dataset structure: `dataset/COHFACE/<subject>/<trial>/`
3. Ensure videos are in expected format (`.avi`, `.mp4`)

### Issue: "Method not found"

**Symptoms**: Method name not recognized in pipeline

**Solutions**:
1. Check method name in config matches implementation
2. For composite methods, ensure both base and head exist
3. Verify method is registered in `run_all.py` dispatcher

### Issue: "EM params not loading"

**Symptoms**: Oscillator uses default parameters despite EM training

**Solutions**:
1. Check filename: `runs/em_params/<dataset>_<method>.json` (exact format)
2. Verify dataset name is uppercase: `COHFACE` not `cohface`
3. Check JSON syntax: `{"q": 0.00015, "r": 0.025, ...}`
4. Look for loading messages in console output

### Issue: "Evaluation returns all NaN"

**Symptoms**: Metrics are NaN or infinite

**Solutions**:
1. Check if estimates exist: `ls results/<run>/data/`
2. Verify estimate format: `signal` must be same length as ground truth
3. Check for empty signals: `signal.size == 0`
4. Inspect quality flags in aux files

### Issue: "Sharding produces incomplete results"

**Symptoms**: Missing trials after sharded estimation

**Solutions**:
1. Count expected trials: `<total_trials> / <num_shards>`
2. Check for failed shards: `results/<run>/logs/estimate_errors.log`
3. Re-run failed shard: `--shard_index <failed_idx>`
4. Ensure all shards use same `--num_shards` value

### Issue: "Out of memory during deep learning inference"

**Symptoms**: CUDA out of memory error with MTTS_CAN or BigSmall

**Solutions**:
1. Reduce batch size in method implementation
2. Use CPU: `export DEVICE=cpu`
3. Process fewer trials per run
4. Close other GPU-using processes

### Issue: "MediaPipe fails to detect face/chest"

**Symptoms**: Empty `chest_rois` or `face_rois` lists

**Solutions**:
1. Check video quality (lighting, resolution)
2. Verify video codec is supported
3. Try different detection confidence thresholds
4. Manually inspect video: `ffplay <video_path>`

## Performance Optimization

### Parallel Processing

```bash
# Use sharding for parallel estimation
for i in {0..9}; do
  python run_all.py -c config.json -s estimate \
    --num_shards 10 --shard_index $i &
done
wait

# Use all CPU cores (set NUM_WORKERS)
export NUM_WORKERS=$(nproc)
```

### GPU Acceleration

```bash
# Use GPU for deep optical flow
export DEVICE=cuda:0

# Multi-GPU (manual split by shard)
DEVICE=cuda:0 python run_all.py ... --shard_index 0 &
DEVICE=cuda:1 python run_all.py ... --shard_index 1 &
```

### Memory Management

- Clear intermediate results: `rm -rf results/<run>/aux/` (after analysis)
- Use smaller window sizes: `"win_size": 20` instead of 30
- Process subsets: `--num_shards 10 --shard_index 0`

## Important Reminders for AI Assistants

1. **Read before modifying**: Always read files before editing, especially `run_all.py` (4000+ lines)

2. **Understand the pipeline**: Three stages (estimate, evaluate, metrics) must run in order

3. **Method naming is strict**: `base__head` format with double underscore

4. **Config paths**: `--runs` takes labels, not full paths

5. **EM params are automatic**: Once trained, they load automatically - no manual intervention needed

6. **Validation is key**: Always test on single trial before full runs

7. **Git branch names matter**: Must match `claude/<session_id>` format

8. **Documentation is sparse**: Legacy code may lack comments; infer from context

9. **Performance vs. accuracy**: This is research code - prefer correctness over speed

10. **Reproducibility**: Always generate `metadata.json` for runs that will be published

## Additional Resources

- **Main README**: `/home/user/resPyre/README.md` - Usage examples and quick start
- **Config examples**: `/home/user/resPyre/configs/` - JSON configuration templates
- **EM algorithm**: `/home/user/resPyre/riv/optim/em_kalman.py` - Parameter learning implementation
- **Oscillator theory**: `/home/user/resPyre/external/somata-main/` - State-space oscillator framework
- **Paper reference**: ACM Computing Surveys (2025) - Original research publication

## Summary

resPyre is a sophisticated respiratory rate estimation framework balancing research flexibility with reproducibility. Key principles:

- **Modular design**: Base methods + oscillator heads = composite methods
- **Configuration-driven**: JSON configs control entire pipeline
- **Parameter learning**: EM and Optuna for automatic tuning
- **Reproducibility**: Metadata tracking for all experiments
- **Quality-aware**: Gating and quality metrics throughout

When in doubt, start with the quick start in README.md, test on single trials, and consult oscillator_heads.py for advanced signal processing logic.
