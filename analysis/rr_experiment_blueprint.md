# RR Experiment Blueprint

This blueprint fixes the dataset roles before full runs. The key rule is to match each dataset to the strongest label it actually provides.

## Dataset Roles

| dataset | role | label scope | metric scope | claim boundary |
| --- | --- | --- | --- | --- |
| COHFACE | primary_real_waveform_rate | respiration waveform + derived RR | rate + aligned waveform + strict waveform + cycle diagnostics | supports waveform reconstruction and rate estimation claims |
| MAHNOB-HCI | hard_real_waveform_rate | respiration-belt waveform + derived RR | rate + aligned waveform + strict waveform + observability failure diagnostics | supports robustness analysis; low observability must be diagnosed explicitly |
| V4V | external_real_rate_only | frame-aligned HR/RR labels; no raw respiratory waveform | RR rate only | does not support waveform CCC/DTW or morphology claims |
| SCAMPS | synthetic_controlled_diagnostic | synthetic breathing signal d_br plus video arrays | controlled rate/waveform sanity checks and ablations | must not be mixed with real-data performance claims |

## Experiment Blocks

| block | purpose | datasets | outputs |
| --- | --- | --- | --- |
| real waveform/rate | Main PARH-OSSM validation and strict morphology diagnostics | COHFACE, MAHNOB-HCI | T3, T4, T4b, T4c, T6, F3, F4 |
| external rate-only | Check whether respiratory timing transfers to a real RR-only dataset | V4V | external RR MAE/RMSE/Pearson table; no waveform plots |
| synthetic controlled | Verify mechanism behavior where breathing signal is controlled | SCAMPS | synthetic diagnostic table/figure; no real-data claim |
| ablation/diagnostic | Show why each modeling decision exists | all compatible subsets | T2, T5, T7, S_F* diagnostics |

## External Manifest Summary

| dataset | scope | rows | valid paths | median RR | note |
| --- | --- | ---: | ---: | ---: | --- |
| V4V | real_rate_only | 724 | 724 | 13.820 | RR labels are rate-only; no waveform metrics. |
| SCAMPS | synthetic_diagnostic | 2800 | 2800 |  | Synthetic d_br field; keep separate from real-data claims. |

## Claim Boundary

- Waveform reconstruction claims must use only datasets with respiratory waveform ground truth.
- V4V can support external RR-rate generalization only.
- SCAMPS can support controlled mechanism evidence only and should not be pooled with real datasets.
- Datasets without respiration labels remain excluded unless a respiratory annotation is identified and audited.
