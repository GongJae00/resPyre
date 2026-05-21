# Target-Side Observability Failure-Mode Audit

- trials: `525`
- boundary: diagnostic-only; labels may use GT-derived oracle room, scores are target-computable.

## Failure-Mode Counts

| failure_mode | n |
| --- | --- |
| bounded_or_no_clear_room | 400 |
| source_selection_room_posterior_available | 60 |
| source_selection_room_agreement_available | 35 |
| likely_video_or_reference_limited | 19 |
| oracle_room_but_gtfree_evidence_weak | 9 |
| low_target_observability | 2 |

## Hardest Trials

| video | failure_mode | final_mae | oracle_best_mae | oracle_best_source | oracle_room_bpm | target_observability_score | posterior_specificity_score | alias_safety_score | source_agreement_score | source_spread_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mahnob_3154 | likely_video_or_reference_limited | 13.280 | 12.960 | external_rate_posterior_mean_t | 0.320 | 0.760 | 0.952 | 1.000 | 0.917 | 0.287 |
| mahnob_3140 | source_selection_room_agreement_available | 11.520 | 9.550 | external_rate_posterior_mean_t | 1.970 | 0.187 | 0.000 | 1.000 | 0.367 | 3.308 |
| mahnob_2770 | source_selection_room_posterior_available | 10.190 | 8.790 | external_rate_posterior_mean_t | 1.400 | 0.685 | 0.912 | 1.000 | 0.645 | 1.447 |
| mahnob_3148 | source_selection_room_posterior_available | 10.070 | 9.210 | external_rate_posterior_mean_t | 0.860 | 0.677 | 0.865 | 1.000 | 0.646 | 1.441 |
| mahnob_1306 | bounded_or_no_clear_room | 9.240 | 8.830 | external_rate_posterior_mean_t | 0.410 | 0.645 | 0.570 | 0.939 | 0.769 | 0.867 |
| mahnob_3146 | bounded_or_no_clear_room | 9.070 | 8.330 | external_rate_posterior_mean_t | 0.740 | 0.538 | 0.403 | 0.996 | 0.519 | 2.167 |
| mahnob_1316 | source_selection_room_posterior_available | 9.000 | 7.970 | state_freq_t | 1.030 | 0.673 | 0.678 | 1.000 | 0.672 | 1.312 |
| mahnob_2756 | source_selection_room_posterior_available | 8.520 | 4.650 | external_rate_posterior_mean_t | 3.870 | 0.574 | 0.693 | 1.000 | 0.236 | 4.761 |
| mahnob_3144 | source_selection_room_posterior_available | 8.490 | 7.290 | external_output_rate_t | 1.200 | 0.663 | 0.853 | 1.000 | 0.523 | 2.141 |
| mahnob_2742 | oracle_room_but_gtfree_evidence_weak | 8.130 | 3.700 | external_rate_posterior_mean_t | 4.430 | 0.176 | 0.000 | 1.000 | 0.261 | 4.437 |
| mahnob_2758 | source_selection_room_posterior_available | 7.670 | 6.390 | external_rate_posterior_mean_t | 1.280 | 0.639 | 0.830 | 1.000 | 0.505 | 2.254 |
| mahnob_1304 | bounded_or_no_clear_room | 7.400 | 6.980 | native_smoothed_track_hz | 0.420 | 0.463 | 0.182 | 0.853 | 0.684 | 1.252 |
| mahnob_2732 | oracle_room_but_gtfree_evidence_weak | 7.380 | 3.750 | external_rate_posterior_mean_t | 3.630 | 0.167 | 0.000 | 1.000 | 0.169 | 5.861 |
| mahnob_2214 | likely_video_or_reference_limited | 7.080 | 6.900 | native_smoothed_track_hz | 0.180 | 0.585 | 0.466 | 1.000 | 0.827 | 0.625 |
| mahnob_3800 | source_selection_room_posterior_available | 6.780 | 5.860 | external_rate_posterior_mean_t | 0.920 | 0.581 | 0.607 | 1.000 | 0.622 | 1.566 |
| mahnob_3646 | bounded_or_no_clear_room | 6.640 | 6.280 | external_rate_posterior_mean_t | 0.360 | 0.551 | 0.187 | 1.000 | 0.518 | 2.169 |

## Largest Oracle Room

| video | failure_mode | final_mae | oracle_best_mae | oracle_best_source | oracle_room_bpm | target_observability_score | posterior_specificity_score | alias_safety_score | source_agreement_score | source_spread_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mahnob_2742 | oracle_room_but_gtfree_evidence_weak | 8.130 | 3.700 | external_rate_posterior_mean_t | 4.430 | 0.176 | 0.000 | 1.000 | 0.261 | 4.437 |
| mahnob_2756 | source_selection_room_posterior_available | 8.520 | 4.650 | external_rate_posterior_mean_t | 3.870 | 0.574 | 0.693 | 1.000 | 0.236 | 4.761 |
| mahnob_2732 | oracle_room_but_gtfree_evidence_weak | 7.380 | 3.750 | external_rate_posterior_mean_t | 3.630 | 0.167 | 0.000 | 1.000 | 0.169 | 5.861 |
| mahnob_1700 | source_selection_room_agreement_available | 5.470 | 2.590 | external_rate_posterior_mean_t | 2.880 | 0.188 | 0.000 | 1.000 | 0.376 | 3.231 |
| mahnob_2766 | oracle_room_but_gtfree_evidence_weak | 5.370 | 2.820 | external_rate_posterior_mean_t | 2.550 | 0.183 | 0.000 | 1.000 | 0.331 | 3.649 |
| mahnob_2734 | oracle_room_but_gtfree_evidence_weak | 3.600 | 1.610 | external_rate_posterior_mean_t | 1.990 | 0.182 | 0.000 | 1.000 | 0.315 | 3.808 |
| mahnob_3140 | source_selection_room_agreement_available | 11.520 | 9.550 | external_rate_posterior_mean_t | 1.970 | 0.187 | 0.000 | 1.000 | 0.367 | 3.308 |
| mahnob_1208 | oracle_room_but_gtfree_evidence_weak | 5.050 | 3.110 | external_rate_posterior_mean_t | 1.940 | 0.182 | 0.000 | 1.000 | 0.319 | 3.770 |
| mahnob_3808 | source_selection_room_posterior_available | 5.640 | 3.700 | external_rate_posterior_mean_t | 1.940 | 0.549 | 0.650 | 0.926 | 0.290 | 4.086 |
| mahnob_1340 | source_selection_room_posterior_available | 4.770 | 2.860 | external_rate_posterior_mean_t | 1.910 | 0.610 | 0.684 | 1.000 | 0.590 | 1.740 |
| mahnob_2748 | source_selection_room_agreement_available | 4.230 | 2.440 | native_smoothed_track_hz | 1.790 | 0.514 | 0.340 | 1.000 | 0.545 | 2.001 |
| mahnob_1328 | source_selection_room_posterior_available | 3.730 | 2.050 | external_rate_posterior_mean_t | 1.680 | 0.547 | 0.509 | 1.000 | 0.576 | 1.821 |
| mahnob_1180 | source_selection_room_posterior_available | 4.420 | 2.760 | external_rate_posterior_mean_t | 1.660 | 0.544 | 0.420 | 1.000 | 0.548 | 1.983 |
| mahnob_2750 | source_selection_room_posterior_available | 5.850 | 4.310 | external_output_rate_t | 1.540 | 0.573 | 0.708 | 0.975 | 0.259 | 4.453 |
| mahnob_16 | source_selection_room_posterior_available | 5.940 | 4.410 | external_rate_posterior_mean_t | 1.530 | 0.604 | 0.675 | 1.000 | 0.564 | 1.892 |
| mahnob_1338 | source_selection_room_agreement_available | 3.350 | 1.820 | external_rate_posterior_mean_t | 1.530 | 0.190 | 0.000 | 1.000 | 0.395 | 3.065 |

## Target-Computable Feature Correlations

| feature | corr_with_oracle_room | corr_with_final_mae |
| --- | --- | --- |
| source_spread_bpm | 0.452 | 0.161 |
| abstain_pressure_score | 0.034 | 0.100 |
| alias_safety_score | -0.103 | -0.134 |
| posterior_specificity_score | -0.169 | -0.060 |
| support_score | -0.246 | -0.030 |
| readout_confidence_score | -0.256 | 0.010 |
| target_observability_score | -0.392 | -0.157 |
| source_agreement_score | -0.431 | -0.132 |
| h1_role_score | -0.566 | -0.401 |

## Mean Scores by Oracle Room >= 0.75 BPM

| room_ge_075 | target_observability_score | posterior_specificity_score | source_agreement_score | alias_safety_score | source_spread_bpm | readout_confidence_score | support_score | h1_role_score | abstain_pressure_score | final_mae | oracle_room_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False | 0.584 | 0.466 | 0.682 | 0.996 | 1.329 | 0.184 | 0.767 | 0.812 | 0.095 | 2.474 | 0.302 |
| True | 0.519 | 0.402 | 0.544 | 0.986 | 2.148 | 0.165 | 0.697 | 0.678 | 0.109 | 4.315 | 1.216 |

## Design Consequence

- If many rows are `likely_video_or_reference_limited`, forcing a different readout cannot solve the dataset; the model needs uncertainty/observability reporting or reference-lag handling.
- If many rows are `oracle_room_but_gtfree_evidence_weak`, the next patch should improve target-computable observability features before adding another source selector.
- If many rows are `source_selection_room_*`, the observation law can safely learn a shallow target-side arbiter using those evidence channels.
