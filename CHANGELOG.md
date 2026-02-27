# Changelog

## Unreleased

- Optuna tuning runner modernization (`core/optimization/run_optuna.py`):
  - aligned to current QROBF pipeline contracts (`main.py` + `metrics/*_summary.csv` + `logs/method_quality.csv`).
  - robust-family-first method selection (defaults to `__robust_ossm_ekf` / `__robust_ossm_ukf` when present).
  - per-trial audit artifacts:
    - `trial_manifest.json`
    - `metrics_summary.json`
    - `metrics_detail.csv`
    - `quality_trust_summary.json`
    - `config_used.json`
  - global/per-method trial indices:
    - `<output>/trial_index.csv`
    - `<output>/<family>/<method>/trial_index.csv`
  - objective now uses paper-aligned terms from modern outputs:
    - time/freq MAE, failure rate, invalid-row rate, clip rate, CCC penalty, spectral SNR penalty, NIS true-fail.
  - optional best-preset export with diff artifact (`--export-best-preset`).
  - objective redesign for paper alignment (trajectory stability + calibration):
    - added terms: `time_dtw`, `time_rmse`, `freq_rmse`, `nis_overstrict`, `coverage_dev`, `constraint_penalty`.
    - added configurable constraints (`optuna.objective.constraints`) and normalized constraint excess penalty.
    - trial/leaderboard indices now include calibration fields (`coverage95`, `nis_truefail`, `nis_overstrict`).
  - split-aware + normalized objective + effective pruning:
    - new split-aware objective path (`optuna.split.mode=subject_kfold`) using per-trial raw metrics aggregation.
    - new robust z-score term normalization (`optuna.objective.normalization`) with history-based scaling.
    - new debug pre-pruning stage (`optuna.pruning.debug_stage=true`) running `main.py --debug` and pruning before full run when possible.
    - trial artifacts now include `objective_raw`, `objective_terms_raw`, `objective_terms_used`, and normalization/split metadata.
  - objective alignment pass (2026-02-21):
    - added explicit failure-mode terms to objective computation: `fail_div`, `fail_slip`, `fail_lock`, `fail_double`.
    - constraint engine now supports per-mode bounds: `fail_div_max`, `fail_slip_max`, `fail_lock_max`, `fail_double_max`.
    - updated default objective weights/constraints to prioritize frequency robustness and failure-mode suppression.
    - `collect_trial_metrics` now propagates per-mode failure medians from diagnostics summaries into Optuna objective inputs.
  - stability hardening + baseline-relative objective (2026-02-21):
    - quality outlier score `q_out` is normalized to `[0,1]` (prevents alpha-R blow-up from raw Hampel spikes).
    - trust wiring adds `w_h_min` and `g_z_floor` to avoid over-suppressing frequency updates.
    - robust update adds `lambda_floor` and `r_eff_max_scale` (`R_eff` cap) to prevent UKF-side noise explosion.
    - objective adds baseline-relative hinge terms:
      - `vs_kfstd_time_mae`
      - `vs_kfstd_freq_mae`
      and optional `optuna.objective.kfstd_reference` config mapping.
    - constraint penalty engine now accepts zero upper bounds (e.g., `vs_kfstd_*_max: 0.0`) instead of silently skipping them.

- Added Optuna unit tests (`tests/test_optuna_runner.py`) covering:
  - robust-first method discovery,
  - objective fallback penalties for missing metrics,
  - metric collection from summary/method_quality artifacts,
  - best-preset export merge behavior.

### Added
- Failure-safe run bookkeeping:
  - `run_status.json` (`run_status.v1`) emitted for planned run dirs.
  - `metadata.json` always emitted (`allow_incomplete=True`) even on partial/failed runs.
- Method-quality artifacts:
  - `logs/method_quality.csv` (`method_quality.v1`)
  - `logs/method_quality_summary.json` with schema/version and config fingerprint.
- Explicit config usage tracing in `metrics/eval_settings.json`:
  - `eval_used_keys`, `eval_unused_keys`, `gating_used_keys`, `gating_unused_keys`, `unused_config_keys`.

### Changed
- Duplicate method names are now hard errors in `main.py` (strict-unique mode) to prevent silent overwrite.
- Trial identity hardening:
  - deterministic `trial_key/trial_uid` propagation in runner/wrapper/head fallbacks.
  - collision-safe frame-log writes (`<trial_key>_<k>.npz` suffix) with recorded suffix.
- Gating-scope enforcement:
  - `gating_scope` allowed values are now explicit: `evaluation_only` (default) or `filter_time`.
  - gating config is audit-only unless `filter_time` is explicitly selected.
- Config-usage artifact upgraded:
  - `logs/config_usage.json` now uses `config_usage.v1` with generated timestamp and strict-mode flag.
  - evaluation step finalizes `used/unused` keys and can raise on unused keys when `strict_key_usage=true`.
- QROBF event summary semantics updated:
  - `plots/qrobf_diagnostics/qrobf_event_summary*.{csv,png}` now aggregate across all trials per method.
  - Per-method timeline PNG remains representative.
- `robust_ossm` trial meta now includes:
  - `gating_consumed_keys`, `gating_unused_keys`, merged `unused_config_keys`.
- Frame-log resolution contract hardened:
  - added `aux/<method>/frame_logs/frame_logs_manifest.json` (`frame_logs_manifest.v1`).
  - added run-scoped canonical resolver + manifest: `logs/frame_log_manifest.json`.
  - evaluation/EDA/diagnostics now consume canonical logs only; stale/suffix/orphan logs are excluded from primary aggregates.
  - `method_quality.csv` now records `frame_log_filename_used`, `frame_log_resolution_mode`, `frame_log_suffix_used`.
  - `strict=True` now enforces hard failures on ambiguity/extras/stale contamination (paper-grade default).
- EDA strictness upgraded:
  - `analysis/run_innovation_eda.py` now always writes `innovation_summary.csv/json` plus `innovation_eda_skipped_logs.json`.
  - `allow_missing=False` raises on missing/corrupt/empty canonical logs after diagnostics are written.
  - `allow_missing=True` continues with explicit `incomplete=true` summaries and skipped-log reasons.
- Visualization diagnostics now emit placeholder artifacts when canonical logs are empty:
  - `qrobf_event_summary.csv` (header-only), `qrobf_event_summary_rates.csv`, `qrobf_event_summary.json`, `qrobf_event_summary.png`.
- Trial isolation for post-smoothing:
  - removed per-trial in-place mutation of `post_smooth_alpha`.
  - now logs `post_smooth_alpha_base` and `post_smooth_alpha_used`.
- Config usage/metadata now promote method-level unused keys from estimate metadata (e.g., `method.ensemble`) to run-level artifacts.
- `welch_df_hz` is now surfaced in `method_quality.csv`/summary as diagnostic-only.
- ROI quality metadata reuse hardened:
  - added deterministic ROI-stats cache file (`obs_roi_stats_v1.npz`, `roi_stats_cache.v1`) at trial directory level.
  - wrapped methods now reuse `roi_stats_t` from in-memory sample cache first, then disk cache, then compute.
  - `meta` now records `roi_stats_source` (`memory_cache|disk_cache|computed`) and optional `roi_stats_cache_path`.
  - runner now cleans temporary ROI metadata fields between samples to prevent cross-sample residue.

### Removed / Cleaned
- Dead buffers removed from `QualityEstimator`.
- Unused private helper removed from `RobustKalmanUpdater`.
- Legacy-unused helper functions removed from `evaluation_step`.

### Compatibility Notes
- Public artifact locations are preserved.
- New fields are additive except duplicate-method handling, which intentionally changes behavior from warning to hard failure.
