# PARH-OSSM v1 Design-Code Reconciliation
# Date: 2026-03-31

## What was patched in this step

### 1. Quality scores q_obs / q_dyn / q_osc (NEW)
- `_compute_q_obs()`: robust standardised innovation + signal jump → [Q_OBS_MIN, 1]
- `_compute_q_dyn()`: frequency deviation + amplitude slope → [0, 1]
- `_compute_q_osc()`: phase coherence + amplitude stability → [0, 1]
- All three stored as per-sample diagnostic arrays

### 2. Prior trust pi_t separated from Student-t lambda_t (REFACTORED)
- Old: `g_t` was a direct R inflation factor (gated Kalman gain)
- New: `pi_t = q_obs_t` is explicit prior trust. `R_eff = R_selfcal / pi_t`
- `lambda_t` remains the Student-t VB posterior weight (unchanged logic)
- Both stored as separate diagnostic arrays: `diag_pi`, `diag_lambda`

### 3. Disentangled Q replaces coupled legacy (NEW DEFAULT)
- Old default: `q_dyn_scale = 1 + gamma * (R_t/R_init - 1)` — R-coupled
- New default: `ENABLE_DISENTANGLED_Q = True`
  - Q_osc = Q_osc_0 * (1 + Q_DYN_GAMMA * q_dyn)
  - Q_aper = Q_aper_0 * (1 + Q_APER_GAMMA * (1 - q_osc))
- Old coupled Q available via `ENABLE_LEGACY_COUPLED_Q = True` (ablation only)

### 4. Causal outputs stored (NEW)
- `z_osc_causal`, `z_full_causal`, `track_hz_causal` from forward pass
- `z_osc_smoothed`, `z_full_smoothed` from RTS pass
- Both in result dict

### 5. EW-MAD now feeds q_obs (ACTIVATED)
- `mad_e` → `robust_scale = mad_e * 1.4826` → used in `_compute_q_obs()`
- Previously: EW-MAD was computed but discarded

### 6. Model version and output semantics in metadata (NEW)
- `model_version = "parh_ossm_v1"`
- `output_semantics = "smoothed"` (primary)
- `active_modules` list
- `warmup_frames`

### 7. Complete diagnostic arrays (EXPANDED)
Added: `pi_t`, `q_obs_t`, `q_dyn_t`, `q_osc_t`
Kept: `R_t`, `lambda_t`, `nu_t`, `freq_t`, `nis_empirical_t`

### 8. Waveform metrics added to metrics.py (NEW)
- `waveform_ccc()`, `waveform_mae()`, `waveform_dtw()`
- `compute_dual_output_metrics()` orchestrator

## Honest boundary statement

In v1, R_selfcal (Mehra-based adaptive R) remains the PRIMARY noise adaptation
mechanism. q_obs provides a mild prior trust modulation (pi_t) that reduces
Kalman gain for unreliable observations. This is a conservative design.

q_dyn and q_osc drive the disentangled Q pathways, which is a genuine
separation from the old R-coupled approach. However, the q_dyn/q_osc
hyperparameters (weights, scales) have not been tuned — they use reasonable
defaults that need validation on real data.

The paper narrative MUST reflect this honest boundary:
- Contribution #1 (decomposition): fully implemented
- Contribution #2 (disentangled adaptation): structurally implemented, needs tuning
- Contribution #3 (self-calibrating framework): inherited from NAROSSM, extended

## What is deferred

- Warm-up-only frequency init (still uses full signal)
- Per-trial NPZ saving of all diagnostic arrays
- Frame logger integration for PARH-OSSM
- Full evaluation_step.py integration (currently metrics functions exist
  but the pipeline doesn't auto-call them for z_full)
- DTW dependency check (fastdtw optional)
