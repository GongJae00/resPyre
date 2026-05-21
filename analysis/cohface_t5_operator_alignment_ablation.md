# T5 Operator-Alignment Ablation

This is the current intent-aligned ablation under the revised thesis.

Important provenance note:
- rows come from the strongest available completed reruns for each question
- most rows come from the decoupled-system gate table so they share one live provenance
- `shared_observation_law` is included when its live paired source run is available
- unavailable historical latent/resonator rows are omitted rather than silently copied from stale artifacts
- native-scale strict raw MAE/DTW and fully empty diagnostic fields are omitted from this paper-facing ablation table
- this table is reviewer-facing and mechanistic, not a single unified leaderboard

                  row_id                               intent            source_run                                                  method  rate_MAE  rate_RMSE  rate_PearsonR  waveform_CCC  waveform_MAE  waveform_DTW  strict_CCC
        base_raw_p1dquad          raw single-family reference decoupled_system_gate                                     profile1D quadratic     0.170       0.22          0.965      0.864664      0.407865      0.342791    0.002246
        parh_rate_expert               structured rate expert decoupled_system_gate                          profile1d_quadratic__parh_ossm     0.145       0.20          0.900      0.890472      0.375686      0.316018    0.144367
adaptive_observation_law local observation adaptation control decoupled_system_gate                         adaptive_observation_law__torch     0.340       0.48          0.790      0.844399      0.341755      0.331680    0.611020
 staged_routed_multihead          coupled routed multi-output decoupled_system_gate adaptive_observation_law_staged_routed_multihead__torch     0.340       0.48          0.790      0.919011      0.298026      0.274680    0.849715
    waveform_expert_only  waveform-specialized learned expert decoupled_system_gate                       temporal_fusion_comparator__torch     0.620       0.76          0.785      0.919011      0.298026      0.274680    0.849715
        decoupled_system   current best honest overall system decoupled_system_gate        p1dquad_rate_temporal_waveform__decoupled_system     0.145       0.20          0.900      0.919011      0.298026      0.274680    0.849715
