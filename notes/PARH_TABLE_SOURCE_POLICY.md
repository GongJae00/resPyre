# PARH-OSSM Paper Table Source Policy
# Date: 2026-04-01
# Status: LOCKED

## Policy

Every number in the paper must trace back to a **persistent CSV file** produced by `main.py → run_evaluation()`.
No standalone scripts, no temporary directories, no manual calculations.

## Table → CSV Mapping

| Paper Table | Content | Source CSV | Key Columns |
|-------------|---------|-----------|-------------|
| T3 | Rate accuracy (MAE, RMSE, PearsonR) | `metrics/metrics_freq_domain_raw.csv` | MAE, RMSE, MAPE, PearsonR, SNR_Spec, Bias, LoA_Width |
| T4 | Waveform fidelity (CCC, wMAE, DTW) | `metrics/metrics_waveform_raw.csv` | waveform_CCC, waveform_MAE, waveform_DTW, latency_ms |
| T6 | Filter diagnostics / calibration | `metrics/metrics_filter_diagnostics_raw.csv` | Fail_Total, NIS_Mean, NIS_Pass, Lambda_Mean, Coverage95, Stability_Sec |

## Signal → Table Routing

| Signal | Evaluation Path | Target Table |
|--------|----------------|--------------|
| z_osc (= signal_hat) | → track_hz → BPM | T3 (rate accuracy) |
| z_full (= z_osc + baseline + residual) | → bandpass → zscore → align → waveform metrics | T4 (waveform fidelity) |
| diagnostics arrays | → per-trial aggregation | T6 (filter diagnostics) |

## Column Requirements

### metrics_waveform_raw.csv (T4)
- `output_type` = 'z_full' on every row
- `causal_or_smoothed` = 'smoothed' or 'causal'
- Paper T4 reports **smoothed** variant as primary, causal as supplementary

### metrics_freq_domain_raw.csv (T3)
- Computed from `signal_hat` (= z_osc_smoothed)
- Paper T3 reports median ± std across trials

### metrics_filter_diagnostics_raw.csv (T6)
- NIS calibration, failure modes, lambda stats
- Paper T6 reports per-method aggregated

## Figure → CSV Mapping

| Paper Figure | Source |
|-------------|--------|
| F5 | `decomposition` dict from saved PKL (h1, h2, baseline, residual) |
| F7 | `diagnostics` dict from saved PKL (q_obs_t, q_dyn_t, q_osc_t, pi_t, lambda_t, R_t) |

## Execution Policy

- **Production entrypoint**: `python main.py --config <config>.json`
- **Results directory**: `results/<config_name>/<config_name>/metrics/`
- **No temp dirs**: All results must persist on disk
- **Reproducibility**: `metadata.json` records git hash, command, config path
