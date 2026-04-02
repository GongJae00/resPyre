# PARH-OSSM v1 Final Readiness Gate (E2E Persistent Proof)
# Date: 2026-04-01
# Status: READY

## Execution Evidence

| Item | Command | Status |
|------|---------|--------|
| COHFACE E2E | `python main.py --config configs/cohface_parh_ossm_smoke_e2e.json` | PASS |
| MAHNOB E2E | `python main.py --config configs/mahnob_parh_ossm_smoke_e2e.json` | PASS |

## Readiness Checklist

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| 1 | main.py entrypoint used (NOT standalone script) | **TRUE** | `python main.py --config configs/*_smoke_e2e.json` |
| 2 | Persistent results dir (NOT temp) | **TRUE** | `results/cohface_parh_ossm_smoke_e2e/`, `results/mahnob_parh_ossm_smoke_e2e/` |
| 3 | metrics_waveform_raw.csv exists on disk | **TRUE** | Both datasets: 18 rows each, output_type=z_full, causal_or_smoothed column |
| 4 | metrics_freq_domain_raw.csv exists on disk | **TRUE** | Both datasets: 27 rows each |
| 5 | metrics_filter_diagnostics_raw.csv exists on disk | **TRUE** | Both datasets: 27 rows each |
| 6 | Saved PKL has 14 top-level keys | **TRUE** | COHFACE: 14 keys, MAHNOB: 14 keys |
| 7 | diagnostics dict has 9 keys (incl q_obs, q_dyn, q_osc) | **TRUE** | Both datasets verified |
| 8 | decomposition dict has 4 keys (h1, h2, baseline, residual) | **TRUE** | Both datasets verified |
| 9 | Zero NaN in all arrays | **TRUE** | All inspected arrays: nan=0 |
| 10 | Causal/smoothed distinction in PKL and CSV | **TRUE** | PKL: z_osc_causal ≠ z_osc_smoothed. CSV: causal_or_smoothed column |
| 11 | T3 source (rate accuracy) traceable | **TRUE** | metrics_freq_domain_raw.csv → T3 |
| 12 | T4 source (waveform fidelity) traceable | **TRUE** | metrics_waveform_raw.csv → T4 |
| 13 | T6 source (filter diagnostics) traceable | **TRUE** | metrics_filter_diagnostics_raw.csv → T6 |
| 14 | Table source policy locked | **TRUE** | `notes/PARH_TABLE_SOURCE_POLICY.md` |
| 15 | Waveform protocol locked | **TRUE** | `analysis/parh_waveform_protocol_lock.md` |
| 16 | Subset filtering works via main.py | **TRUE** | 160→3 (COHFACE), full→3 (MAHNOB) |

## Final Verdict

**READY FOR FULL EXPERIMENT**

All 16 criteria = TRUE. Evidence is persistent on disk. No self-reports without file proof.

## Persistent Artifact Inventory

```
results/cohface_parh_ossm_smoke_e2e/
  cohface_parh_ossm_smoke_e2e/
    data/cohface_10_0.pkl
    data/cohface_11_1.pkl
    data/cohface_12_2.pkl
    metrics/metrics_waveform_raw.csv        (18 rows)
    metrics/metrics_freq_domain_raw.csv     (27 rows)
    metrics/metrics_filter_diagnostics_raw.csv (27 rows)
    metrics/metrics_time_domain_raw.csv     (27 rows)
    metadata.json
    methods.json
    run_status.json

results/mahnob_parh_ossm_smoke_e2e/
  mahnob_parh_ossm_smoke_e2e/
    data/mahnob_10.pkl
    data/mahnob_12.pkl
    data/mahnob_14.pkl
    metrics/metrics_waveform_raw.csv        (18 rows)
    metrics/metrics_freq_domain_raw.csv     (27 rows)
    metrics/metrics_filter_diagnostics_raw.csv (27 rows)
    metrics/metrics_time_domain_raw.csv     (27 rows)
    metadata.json
    methods.json
    run_status.json
```

## COHFACE Waveform Summary (parh_ossm, smoothed)

| Method | CCC (median) | wMAE (median) | DTW (median) |
|--------|-------------|---------------|-------------|
| of_farneback__parh_ossm | 0.810 | 0.495 | 0.343 |
| profile1d_quadratic__parh_ossm | 0.881 | 0.396 | 0.303 |
| dof__parh_ossm | 0.577–0.789 | 0.525–0.765 | 0.392–0.459 |

## MAHNOB Waveform Summary (parh_ossm, smoothed)

| Method | CCC (median) | wMAE (median) | DTW (median) |
|--------|-------------|---------------|-------------|
| of_farneback__parh_ossm | 0.229 | 0.890 | 0.664 |
| profile1d_quadratic__parh_ossm | 0.280 | 0.905 | 0.672 |
| dof__parh_ossm | 0.338 | 0.816 | 0.636–1.498 |

## Known Limitations (non-blocking)

1. MAHNOB waveform CCC is lower than COHFACE (~0.22-0.33 vs ~0.58-0.88). This is expected: MAHNOB has more motion artifacts, longer recordings, and noisier GT.
2. Filter diagnostics columns (NIS, Lambda, Coverage) show NaN for base methods — expected since these methods don't produce diagnostics.
3. DTW requires fastdtw. If absent, returns NaN — non-blocking.
4. Q score hyperparameters are defaults, not tuned — documented in honest boundary.
