# PARH-OSSM Current Scaffold Gap Audit
# Date: 2026-03-31

## Verdict Table

| # | Item | Status | Detail |
|---|------|--------|--------|
| 1 | State vector ordering/dims | CORRECT | 8D [h_c1,h_s1,h_c2,h_s2,b,bdot,r,rdot] |
| 2 | Observation model H | CORRECT | [1,0,1,0,1,0,1,0] with ablation-aware masking |
| 3 | z_osc / z_full extraction | CORRECT | z_osc=h_c1+h_c2, z_full=z_osc+b+r |
| 4 | Baseline block dynamics | CORRECT | Constant-velocity [[1,dt],[0,1]] |
| 5 | Residual block dynamics | CORRECT | Damped RW with TAU_RESIDUAL_SEC=5s |
| 6 | Harmonic-2 usage | CORRECT | K=2, rotation at 2*omega, ablation flag |
| 7 | Adaptive R logic | CORRECT | Mehra-based C_e - HP, R_ANCHOR_FRAC floor |
| 8 | Adaptive Q logic | **STRUCTURALLY INCONSISTENT** | Uses R_t/R_init coupled ratio, NOT disentangled q_dyn |
| 9 | q_obs/q_dyn/q_osc existence | **MISSING** | None of the 3 quality scores are computed or stored |
| 10 | Prior trust pi vs gating vs Student-t lambda | **STRUCTURALLY INCONSISTENT** | g_t acts as direct R inflation, not explicit pi_t layer |
| 11 | Robust scale estimator | APPROX CORRECT (needs patch) | EW-MAD computed but not actually used for adaptive R |
| 12 | Warm-up initialization | APPROX CORRECT | Uses full-signal _coarse_freq (should be warm-up window) |
| 13 | Frequency adaptation | CORRECT | Confirmation-gated, rate-limited |
| 14 | Causal vs RTS separation | **MISSING** | Only smoothed outputs stored; no causal z_osc/z_full |
| 15 | Diagnostic arrays/logging | APPROX CORRECT (needs patch) | Has diag_R/nis/lambda/nu/freq/gate/q_dyn but missing pi_t, q_obs, q_dyn, q_osc arrays |
| 16 | Ablation flag semantics | APPROX CORRECT (needs patch) | Missing ENABLE_DISENTANGLED_Q flag; coupled Q is the default |
| 17 | Evaluation compatibility | **MISSING** | Pipeline consumes signal_hat only; z_osc/z_full not evaluated |
| 18 | Config compatibility | CORRECT | cohface/mahnob configs exist with 5 families x 3 methods |
| 19 | R-based coupled vs disentangled | **STRUCTURALLY INCONSISTENT** | Current default is old coupled-Q: q_dyn_scale = 1 + gamma*(R_t/R_init - 1) |

## Top 10 Mismatches (priority order)

1. **q_obs/q_dyn/q_osc completely absent** — The 3 disentangled quality scores are the paper's Contribution #2. They are not computed, not stored, not logged.

2. **Adaptive Q is coupled legacy** — `_build_Q` receives `q_dyn_scale = 1 + gamma*(R_t/R_init - 1)`. This is R-coupled, not driven by independent q_dyn. The paper claims disentangled pathways.

3. **g_t is direct R inflation, not pi_t** — Current code: `R_eff = R_t / g_t`. This makes g_t a gain-scaling factor, not a named "prior trust" with its own diagnostic identity. No pi_t variable exists.

4. **No causal outputs** — Only smoothed z_osc/z_full are stored. Causal (forward-pass) outputs are discarded. Paper must report both.

5. **EW-MAD computed but unused** — `mad_e` is tracked but never feeds into adaptive R or any other logic. It should replace or augment C_e for robust scale.

6. **Evaluation pipeline ignores dual output** — `evaluation_step.py` reads `signal_hat` only. z_osc is signal_hat (OK for rate), but z_full waveform metrics (CCC, DTW) are never computed against GT waveform.

7. **Warm-up uses full signal** — `_coarse_freq(y, fs)` uses the entire preprocessed signal. For causal deployment and paper correctness, should use only the first N seconds.

8. **Q_aper not driven by q_osc** — `_build_Q` uses `q_res_scale = max(1, 2 - q_dyn_scale)` which is an approximation of "inverse oscillatory support" but is actually just inverse of the R-coupled scale. Should be driven by explicit q_osc.

9. **No model_version / output_semantics in metadata** — Result dict lacks version tagging, causal/smoothed flag, enabled modules list.

10. **Diagnostic arrays incomplete** — Missing: pi_t, q_obs_t, q_dyn_t, q_osc_t per-sample arrays.

## Fix-Now vs Defer

### Must fix this step (MVP v1)
- Implement q_obs, q_dyn, q_osc computation and storage
- Rename g_t to pi_t, separate from lambda_t in code and diagnostics
- Replace coupled-Q with disentangled Q driven by q_dyn and q_osc
- Add causal output extraction (before RTS)
- Add per-sample diagnostic arrays for pi_t, q_obs, q_dyn, q_osc
- Add model_version and output_semantics to metadata
- Wire EW-MAD into robust scale (used by q_obs at minimum)
- Add waveform CCC metric to evaluation for z_full
- Add ablation flag for disentangled vs legacy Q

### Defer to next step
- Full warm-up-only frequency init (requires base.py change; current full-signal approach still works)
- DTW metric (dependency check needed)
- Full causal-mode pipeline run (smoke uses smoothed as primary)
- Per-trial NPZ saving of all diagnostic arrays
- Frame logger integration for PARH-OSSM
