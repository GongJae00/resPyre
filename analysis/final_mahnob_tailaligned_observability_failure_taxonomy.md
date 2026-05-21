# Observability Failure Taxonomy Table

- source: `analysis/final_mahnob_tailaligned_observability_failure_modes.csv`
- trials: `525`
- boundary: paper-facing compression of a diagnostic audit; oracle-best columns are used only to interpret current observation-bank limits.

## Table

| failure_mode | paper_label | n_trials | fraction | median_final_mae_bpm | median_best_source_mae_bpm | median_oracle_room_bpm | median_observability | median_source_spread_bpm | interpretation | next_need |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bounded_or_no_clear_room | Bounded by current observation bank | 400 | 0.762 | 2.150 | 1.810 | 0.290 | 0.591 | 1.221 | Current candidate sources have little oracle room; better source selection alone is unlikely to solve these trials. | Create richer respiratory observations or report low observability. |
| source_selection_room_posterior_available | Source-selection room with posterior evidence | 60 | 0.114 | 3.655 | 2.545 | 1.030 | 0.603 | 1.823 | A different source can help and the GT-free posterior already contains some usable evidence. | Use posterior evidence cautiously as an adaptive-law diagnostic. |
| source_selection_room_agreement_available | Source-selection room with agreement evidence | 35 | 0.067 | 3.960 | 2.790 | 0.990 | 0.481 | 1.951 | A different source can help and cross-source agreement is partly informative. | Improve agreement-to-state trust without hard source switching. |
| likely_video_or_reference_limited | Likely video/reference limited | 19 | 0.036 | 4.510 | 4.200 | 0.170 | 0.615 | 0.850 | Even the best current source remains poor, suggesting weak visual respiratory evidence or reference/scale/lag risk. | Separate reference-risk reporting from model-error claims. |
| oracle_room_but_gtfree_evidence_weak | Oracle room but weak GT-free evidence | 9 | 0.017 | 5.050 | 3.110 | 1.940 | 0.183 | 4.418 | A better source exists, but current target-computable diagnostics do not identify it reliably. | Develop stronger observability features before promotion. |
| low_target_observability | low_target_observability | 2 | 0.004 | 3.715 | 3.055 | 0.660 | 0.442 | 1.603 |  |  |

## Paper Use

- Use this table to frame MAHNOB as an irregular/low-observability stress regime.
- Do not claim that PARH-OSSM solves MAHNOB strict reconstruction.
- The main scientific claim is that the observation-state decomposition exposes when current camera observations are insufficient and what future respiratory observations must supply.
