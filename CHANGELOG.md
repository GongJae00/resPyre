# Changelog

## Unreleased

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
  - evaluation now resolves frame logs via manifest first (`resolve_frame_log_path`) and fails on ambiguous suffix candidates in strict mode.
  - `method_quality.csv` now records `frame_log_filename_used`, `frame_log_resolution_mode`, `frame_log_suffix_used`.
- EDA strictness upgraded:
  - `analysis/run_innovation_eda.py` is fail-fast by default on unreadable logs.
  - `--allow-missing` writes `eda/innovation_eda_skipped_logs.json` and records `skipped_count`.
- Trial isolation for post-smoothing:
  - removed per-trial in-place mutation of `post_smooth_alpha`.
  - now logs `post_smooth_alpha_base` and `post_smooth_alpha_used`.
- Config usage/metadata now promote method-level unused keys from estimate metadata (e.g., `method.ensemble`) to run-level artifacts.
- `welch_df_hz` is now surfaced in `method_quality.csv`/summary as diagnostic-only.

### Removed / Cleaned
- Dead buffers removed from `QualityEstimator`.
- Unused private helper removed from `RobustKalmanUpdater`.
- Legacy-unused helper functions removed from `evaluation_step`.

### Compatibility Notes
- Public artifact locations are preserved.
- New fields are additive except duplicate-method handling, which intentionally changes behavior from warning to hard failure.
