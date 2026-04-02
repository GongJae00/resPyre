# PARH-OSSM v1 Truth Reconciliation
# Date: 2026-04-01 (final verification)

## Method
Each claim verified by: (1) grep actual code line, (2) run model and inspect result dict, (3) run evaluation pipeline and inspect CSV output.

## Verification Table

| # | Claim | Evidence | Verdict |
|---|-------|----------|---------|
| 1 | q_obs array saved | `parh_ossm.py:430` init, `:525` compute, `:610` store per-sample, `:732` in result["diagnostics"]["q_obs_t"]. Inspected: shape=(1227,) min=0.083 max=1.000 nan=0 | **TRUE** |
| 2 | q_dyn array saved | `parh_ossm.py:431` init, `:526` compute, `:611` store, `:733` in result. Inspected: shape=(1227,) min=0.000 max=1.000 nan=0 | **TRUE** |
| 3 | q_osc array saved | `parh_ossm.py:432` init, `:527` compute, `:612` store, `:734` in result. Inspected: shape=(1227,) min=0.499 max=1.000 nan=0 | **TRUE** |
| 4 | pi_t separate from gate | `parh_ossm.py:532` `pi_t = q_obs_t`, `:548` `R_eff = R_t / max(pi_t, 1e-6)`, `:426` `diag_pi`, `:606` store, `:729` in result["diagnostics"]["pi_t"]. No `g_t` variable exists. Inspected: min=0.083 max=1.000 | **TRUE** |
| 5 | lambda_t separate | `parh_ossm.py:427` `diag_lambda`, `:563` from Student-t VB, `:607` store, `:730` in result["diagnostics"]["lambda_t"]. Inspected: min=0.364 max=1.227 (COHFACE), min=1.000 max=1.212 (MAHNOB) | **TRUE** |
| 6 | Default Q = disentangled | `parh_ossm.py:136` `ENABLE_DISENTANGLED_Q: bool = True`, `:137` `ENABLE_LEGACY_COUPLED_Q: bool = False`, `:539` default path calls `_build_Q_disentangled`. Meta confirms: `active_modules=['adapt_R','disentangled_Q',...]` | **TRUE** |
| 7 | Warm-up-only freq init | `parh_ossm.py:118` `FREQ_INIT_SEC=10.0`, `:400-403` `y_init = y[:init_len]`, `_coarse_freq(y_init, fs)`, `_harmonic_refine(freq0_raw, y_init, fs)`. Verified: signal=61.4s, init_window=10.0s (COHFACE); signal=136.4s, init_window=10.0s (MAHNOB) | **TRUE** |
| 8 | z_osc_causal / z_full_causal saved | `parh_ossm.py:618` extract from x_filt, `:716-717` store in result. Inspected: both shape=(1227,) nan=0 (COHFACE), shape=(8323,) nan=0 (MAHNOB) | **TRUE** |
| 9 | z_osc_smoothed / z_full_smoothed saved | `parh_ossm.py:645` extract from x_smooth, `:718-719` store. Inspected: present, differ from causal (max diff=0.599) | **TRUE** |
| 10 | Diagnostics arrays in result | `parh_ossm.py:727-737` stores dict with 9 keys: R_t, pi_t, lambda_t, nu_t, q_obs_t, q_dyn_t, q_osc_t, freq_t, nis_empirical_t. All inspected as ndarray with correct shape and zero NaN | **TRUE** |
| 11 | Eval pipeline consumes z_full and emits waveform rows | `evaluation_step.py:988` `all_waveform_records`, `:1127-1154` Block 1b computes waveform_ccc/mae/dtw from z_full, `:1361-1363` saves as metrics_waveform_raw.csv. End-to-end test produced CSV with 2 rows (smoothed+causal) | **TRUE** |
| 12 | metrics CSV includes output_type | `evaluation_step.py:1146` `'output_type': 'z_full'`. CSV column verified: `['video','method','output_type','causal_or_smoothed','waveform_CCC','waveform_MAE','waveform_DTW','latency_ms','data_file']` | **TRUE** |
| 13 | metrics CSV includes causal_or_smoothed | `evaluation_step.py:1147` `'causal_or_smoothed': _variant_label`. CSV verified: rows with 'smoothed' and 'causal' | **TRUE** |
| 14 | COHFACE smoke executed | `results/parh_smoke_manifest.csv`: 27 COHFACE rows (9 base + 9 kfstd + 9 parh_ossm), all status=OK | **TRUE** |
| 15 | MAHNOB smoke executed | `results/parh_smoke_manifest.csv`: 27 MAHNOB rows (mahnob_10/12/14 x 3 families x 3 methods), all status=OK | **TRUE** |
| 16 | Saved result contains all required keys | COHFACE: 14 top-level keys, decomposition{4}, diagnostics{9}. MAHNOB: identical structure. Both zero NaN | **TRUE** |
| 17 | Safe to proceed to full experiment | All 16 above = TRUE. No schema inconsistency. Waveform eval protocol locked | **TRUE** |

## Previously FALSE items (now patched and verified)

| Item | Was FALSE because | Patched in | Verified by |
|------|-------------------|------------|-------------|
| #7 Warm-up init | `_coarse_freq(y, fs)` used full signal | `parh_ossm.py:118,400-403` | Meta shows init_window=10.0s on 61.4s/136.4s signals |
| #11 Eval pipeline | No z_full reference in evaluation_step.py | `evaluation_step.py:988,1127-1154,1361-1363` | End-to-end test produced metrics_waveform_raw.csv |
| #15 MAHNOB smoke | Script only had COHFACE trials | `scripts/run_parh_smoke.py` MAHNOB trials added | Manifest has 27 MAHNOB rows, all OK |
