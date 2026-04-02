# PARH-OSSM v1 Truth Lock
# Date: 2026-04-01
# Status: LOCKED — all items verified against actual code and saved outputs

## v1 Must-Have (all verified present)

| Feature | Code location | Runtime evidence |
|---------|--------------|------------------|
| 8D state [h_c1,h_s1,h_c2,h_s2,b,bdot,r,rdot] | `parh_ossm.py:62-63` STATE_DIM=8 | x_filt shape=(n,8) |
| K=2 harmonic | `:131` ENABLE_HARMONIC2=True, `:153-157` F matrix | h2 std=0.361 (non-trivial) |
| Baseline state | `:132` ENABLE_BASELINE=True, `:159-161` F matrix | baseline std=0.009 |
| Residual state | `:133` ENABLE_RESIDUAL=True, `:163-167` damped F | residual std=0.026 |
| z_osc | `:246-253` _extract_outputs | shape=(1227,) nan=0 |
| z_full | `:254-259` z_osc + b + r | max|z_full-z_osc|=0.089 (COHFACE), 0.268 (MAHNOB) |
| EW-MAD robust scale | `:339-341` _ew_mad_update, `:487-488` mad_e*1.4826 | feeds q_obs computation |
| q_obs | `:278-298` _compute_q_obs, `:430,525,610,732` | min=0.083 max=1.000 |
| q_dyn | `:300-316` _compute_q_dyn, `:431,526,611,733` | min=0.000 max=1.000 |
| q_osc | `:318-332` _compute_q_osc, `:432,527,612,734` | min=0.499 max=1.000 |
| pi_t = f(q_obs) | `:532` pi_t=q_obs_t, `:548` R_eff=R_t/pi_t | min=0.083 max=1.000 |
| lambda_t (Student-t) | `:343-361` VB update, `:427,563,607,730` | min=0.364 max=1.227 |
| Disentangled Q default | `:136` ENABLE_DISENTANGLED_Q=True, `:170-204,539` | active_modules includes 'disentangled_Q' |
| Causal outputs | `:618-623` from x_filt, `:716-720` stored | z_osc_causal, z_full_causal, track_hz_causal |
| Smoothed outputs | `:645` from x_smooth, `:718-719` stored | z_osc_smoothed, z_full_smoothed |
| Diagnostics arrays (9) | `:727-737` | R_t, pi_t, lambda_t, nu_t, q_obs_t, q_dyn_t, q_osc_t, freq_t, nis_empirical_t |
| Warm-up-only init | `:118` FREQ_INIT_SEC=10.0, `:400-403` y_init=y[:init_len] | 10.0s window on 61.4s/136.4s signals |
| Eval CSV z_osc + z_full | `evaluation_step.py:1127-1154,1361-1363` | metrics_waveform_raw.csv with output_type, causal_or_smoothed |

## v1 Excluded (NOT in code)

- Multi-family fusion
- IMM / frequency bank
- Learned calibrator
- K=3 harmonic
- Semi-Markov regime logic

## Quality Score Formulas (as implemented)

### q_obs(t) — `parh_ossm.py:278-298`
```
robust_z = |e_t| / max(robust_scale, 1e-12)
term_z = exp(-0.5 * (robust_z / Q_OBS_ROBUST_Z_SCALE)^2)
jump = |y_t - y_prev| / max(signal_scale, 1e-12)
term_jump = exp(-0.5 * (jump / Q_OBS_JUMP_SCALE)^2)
q_obs = clip(term_z * term_jump, Q_OBS_MIN, 1.0)
```
- Inputs: innovation e_t, EW-MAD robust_scale (`:487-488`), signal y_t/y_prev, signal_scale (initial std `:449`)
- Range: [0.05, 1.0]
- Saved: diagnostics.q_obs_t
- Fallback: robust_scale clamped to 1e-12

### q_dyn(t) — `parh_ossm.py:300-316`
```
freq_dev = |freq_candidate - freq_t| / max(Q_DYN_FREQ_REF_HZ, 1e-6)
amp_slope = |amp_t - amp_baseline| / max(amp_baseline, 1e-6)
raw = Q_DYN_FREQ_WEIGHT * freq_dev + Q_DYN_AMP_WEIGHT * amp_slope
q_dyn = clip(1 - exp(-raw), 0, 1)
```
- Inputs: freq_candidate (`:587-591`), current freq_t, amp_t (`:504`), amp_baseline (`:512-513`)
- Range: [0.0, 1.0]
- Saved: diagnostics.q_dyn_t
- Fallback: returns 0.0 during warmup (t < WARMUP)

### q_osc(t) — `parh_ossm.py:318-332`
```
q_osc = clip(Q_OSC_PHASE_WEIGHT * phase_coh + Q_OSC_AMP_WEIGHT * amp_stab, 0, 1)
```
- Inputs: phase_coh (Gaussian kernel on phase prediction error `:510`), amp_stab (Gaussian kernel on amplitude deviation `:514`)
- Range: [0.0, 1.0]
- Saved: diagnostics.q_osc_t
- Fallback: returns 1.0 during gate warmup (t < GATE_WARMUP)

### Disentangled Q — `parh_ossm.py:170-204`
```
Q_osc = Q_osc_0 * (1 + Q_DYN_GAMMA * q_dyn)
Q_aper = Q_aper_0 * (1 + Q_APER_GAMMA * (1 - q_osc))
Q_baseline: independent (always slow)
```
Legacy coupled: `:206-228` `scale = 1 + gamma * (R_t/R_init - 1)` — ablation only (`ENABLE_LEGACY_COUPLED_Q=False`)
