# ResPyre

ResPyre is a research-oriented framework for contactless respiration estimation from video.
The codebase is organized around two layers:

- `components/`: dataset adapters, observation extractors, oscillator heads
- `core/`: pipeline orchestration, evaluation, visualization, metadata/reporting

The current repository is centered on chest-motion respiration experiments and
the Base, OSSM-KF, and PARH-OSSM comparison workflow. Use `execute.md` for the
full reproduction ledger.

## What Is In Scope

- Datasets: `COHFACE`, `MAHNOB-HCI`, plus V4V/SCAMPS as weak external evidence in `execute.md`
- Base observation methods: `of_farneback`, `of_disp_bridge`, `dof`, `dof_disp_bridge`, `profile1d_linear`, `profile1d_quadratic`, `profile1d_cubic`, `profile1d_consensus`
- Oscillator-wrapped methods via `<base>__<head>` naming
- Publicly documented heads: `kfstd` (`OSSM-KF` comparator), `parh_ossm`, `simple_bandpass`
- Compatibility head: `narossm`
- End-to-end steps from one CLI: `estimate`, `evaluate`/`metrics`, `eda`, `visualize`, `metadata`

The stable config examples currently in the repo:

- `configs/cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json`
- `configs/mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json`

The full validation path is driven by `execute.md`, because it also wires
target-side priors, the adaptive observation law, and post-run diagnostics.

## Citation and Archival Release

Citation metadata is provided in `CITATION.cff`, and Zenodo release metadata is
provided in `.zenodo.json`. Cite the archived software release together with the
accompanying paper.

## Repository Layout

```text
resPyre/
├── components/
│   ├── datasets/          # Dataset loaders and ROI extraction entry points
│   ├── observations/      # Base motion / profile observation methods
│   └── models/
│       ├── core/          # Shared oscillator parameters and base helpers
│       └── heads/         # OSSM-KF, PARH-OSSM, and other oscillator heads
├── analysis/              # Generated analysis outputs, ignored except README files
├── core/
│   ├── evaluation/        # Metrics, plotting, frame-log utilities
│   ├── pipeline/          # Runner, wrapper, evaluation, visualization, metadata
│   └── utils/             # Config loader and shared utilities
├── configs/               # Experiment configs
├── dataset/               # Public dataset notes and small metadata files
├── paper/                 # Local manuscript workspace, ignored by Git
├── results/               # Generated run artifacts, ignored by Git
├── setup/                 # Environment bootstrap
├── tests/                 # Regression and pipeline tests
├── execute.md             # Reproduction ledger
└── main.py                # Primary CLI entry point
```

## Installation

`setup/setup.sh` creates a minimal environment for the motion and oscillator stack.

```bash
./setup/setup.sh -n resPyre --verify
conda activate resPyre
```

Installed dependencies include NumPy/SciPy/Pandas/Matplotlib/HDF5 plus the motion stack
(`opencv`, `mediapipe`, `scikit-image`, `plotly`, `optuna`).

Additional setup notes are in `setup/README_setup.md`.

## Dataset Root

By default, dataset loaders read from:

```text
<repo>/dataset
```

To use another dataset root, set:

```bash
export RESPIRE_DATA_DIR=/path/to/datasets
```

The expected local dataset subdirectories are:

- `dataset/BP4Ddef`
- `dataset/COHFACE`
- `dataset/MAHNOB`
- `dataset/V4V`
- `dataset/SCAMPS`

Raw datasets and local symlinks are intentionally ignored by Git. The tracked
`dataset/` directory contains only dataset instructions and small metadata files
needed by the loaders.

## Quick Start

Run a stable config:

```bash
python main.py --config configs/cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json
python main.py --config configs/mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json
```

Run only the first sample from each configured dataset:

```bash
python main.py --config configs/cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json --debug
```

Override the results root at runtime:

```bash
python main.py \
  --config configs/cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json \
  --results /tmp/respyre_runs
```

CLI options currently exposed by `main.py`:

- `--config, -c`: config JSON path
- `--results, -r`: results directory override
- `--debug`: limit each dataset to one sample

## Config Model

Configs are loaded through `core/utils/config.py`.
Relative `results_dir` values are resolved relative to the project root, not the config file directory.

A minimal config looks like this:

```json
{
  "name": "cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons",
  "results_dir": "results/cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons",
  "datasets": [
    {
      "name": "COHFACE",
      "roi": {
        "chest": {
          "mp_complexity": 1,
          "skip_rate": 10
        }
      }
    }
  ],
  "methods": [
    "of_farneback",
    {
      "name": "of_farneback__parh_ossm",
      "params": {
        "oscillator": {
          "no_autotune": true,
          "em_mode": "off",
          "rv_auto": false,
          "rv_floor": 0.05,
          "qx": 0.005,
          "tau_env": 30.0,
          "post_smooth_alpha": 0.88
        }
      }
    }
  ],
  "preproc": {
    "robust_zscore": {
      "enabled": true,
      "clip": 5.0
    }
  },
  "eval": {
    "win_size": 30,
    "stride": 1,
    "min_hz": 0.08,
    "max_hz": 0.5
  },
  "steps": ["estimate", "metrics", "metadata"]
}
```

Key conventions:

- `datasets` entries can be strings or dicts
- `methods` entries can be strings or dicts
- Wrapped methods use `<base>__<head>`
- Method-level oscillator overrides can be passed under `params.oscillator`
- Global oscillator defaults live in the top-level `oscillator` block
- Supported step labels are `estimate`, `evaluate`, `metrics`, `eda`, `visualize`, `metadata`

## Execution Flow

### 1. Config load

`main.py` loads the config, normalizes dataset/method entries, resolves `results_dir`,
builds dataset objects, and instantiates methods.

### 2. Estimation

`core/pipeline/runner.py` loops over each trial and each method:

- dataset loader provides video path and ground truth
- ROI extraction is triggered lazily per region
- base observation methods produce a 1D respiratory proxy
- wrapped methods forward that proxy into an oscillator head
- results are merged into one per-trial pickle under `data/`

For wrapped methods, `core/pipeline/wrapped_method.py` also attaches:

- ROI-derived quality metadata
- method/config metadata
- optional cache-based observation loading
- optional per-method `.npz` payloads under `aux/`

### 3. Evaluation

`core/pipeline/evaluation_step.py` computes four artifact groups:

- time-domain metrics: waveform fidelity on aligned normalized signals
- frequency-domain metrics: windowed RPM accuracy and spectral-shape statistics
- filter diagnostics: NIS/failure/calibration summaries from frame logs
- waveform comparison metrics: unified output comparison for Base, OSSM-KF, and PARH workflows

Evaluation settings are persisted to `metrics/eval_settings.json`.

### 4. Optional EDA / visualization

If requested in `steps`, the pipeline can also run:

- `eda`: innovation/frame-log exploratory summaries
- `visualize`: PNG plots for summary metrics, overlays, traces, and diagnostics

### 5. Metadata and run status

Run bookkeeping is handled in `core/pipeline/metadata_step.py`.
Each run directory receives `run_status.json`, and `metadata.json` is emitted even when
the pipeline fails part-way through.

## PARH-OSSM Validation Path

The final validation path is:

```text
fixed observation classes
-> adaptive observation law
-> PARH-OSSM state-space update
-> bounded z_osc rate readout
-> z_full waveform diagnostics
```

`OSSM-KF` is a comparator and external timing-evidence source in the final
materialization command. It is not a nested fallback that replaces PARH-OSSM.
Run the commands in `execute.md` to regenerate the current COHFACE and MAHNOB
tail-aligned validation bundles and post-run diagnostics. Manuscript source and
rendered paper files live in the local `paper/` workspace and are not tracked in
this repository.

## Results Layout

Single-dataset runs normally resolve to:

```text
results/<name>/
```

Multi-dataset runs resolve to:

```text
results/<name>_<DATASET>/
```

Typical contents:

```text
results/<run>/
├── data/                         # per-trial merged pickle payloads
├── aux/                          # optional method-specific npz/frame-log artifacts
├── metrics/
│   ├── eval_settings.json
│   ├── metrics_time_domain_raw.csv
│   ├── metrics_freq_domain_raw.csv
│   ├── metrics_filter_diagnostics_raw.csv
│   ├── metrics_waveform_raw.csv
│   └── *_summary.txt / *.pkl / *.csv
├── plots/                        # visualization outputs when enabled
├── logs/
│   ├── config_usage.json
│   ├── method_quality.csv
│   ├── method_quality_summary.json
│   └── frame_log_manifest.json
├── metadata.json
└── run_status.json
```

Each `data/*.pkl` file contains shared trial metadata plus an `estimates` list for all
methods run on that trial.

## PARH-OSSM Notes

`components/models/heads/parh_ossm.py` implements the current PARH-OSSM head.

Current behavior:

- the primary packaged `signal_hat` is the smoothed oscillatory reconstruction
- PARH additionally stores `z_osc`, `z_full`, causal/smoothed variants, decomposition terms, and diagnostic arrays
- waveform comparison code treats `z_full` as the main PARH waveform output

## Extending The Framework

### Add a dataset

1. Create a new dataset class under `components/datasets/`
2. Inherit from `DatasetBase`
3. Implement `load_dataset()`, `load_gt()`, and `extract_ROI()`
4. Register or import it where needed in `main.py`

### Add an oscillator head

1. Create a new head under `components/models/heads/`
2. Inherit from `_BaseOscillatorHead`
3. Implement `run(signal, fs, meta=None)`
4. Register it in `components/models/__init__.py`
5. Use it from config via `<base>__<head>`

## Verification

Basic CLI check:

```bash
python main.py --help
```

Run the regression suite:

```bash
pytest -q
```

If you only want to validate a config wiring path, use `--debug` for a quick wiring check.
