# Figure And Table Index

This index lists the final manuscript and Supplementary Information display
items included in the public release. It is a reproducibility map, not a draft
planning note.

## Main Manuscript

| Item | File | Source |
|---|---|---|
| Figure 1 | `paper/figures/F1_architecture.pdf` | final manually reviewed topology schematic |
| Figure 2 | `paper/figures/F2_dataset_and_observation_regime.pdf` | `scripts/plot_observation_eda.py` |
| Figure 3 | `paper/figures/F4_waveform_overlay_grid.pdf` | `scripts/plot_same_trial_all_base_overlay.py` |
| Figure 4 | `paper/figures/F5_mechanism_activation.pdf` | `scripts/plot_main_family_figures.py` |
| Figure 5 | `paper/figures/F6_failure_cases.pdf` | `scripts/plot_failure_cases.py` |
| Table 1 | embedded in `paper/main.tex` | `paper/tables_ready/T3_rate_main.csv` |
| Table 2 | embedded in `paper/main.tex` | `paper/tables_ready/T4_waveform_main.csv` |
| Table 3 | embedded in `paper/main.tex` | `paper/tables_ready/T7_observability_failure_taxonomy.csv` |

## Supplementary Information

| Item | File | Source |
|---|---|---|
| Supplementary Table S1 | embedded in `paper/supplementary_information.tex` | `paper/tables_ready/T1_dataset_protocol_scope.csv` |
| Supplementary Table S2 | embedded in `paper/supplementary_information.tex` | `paper/tables_ready/T2_observation_class_map.csv` |
| Supplementary Table S3 | embedded in `paper/supplementary_information.tex` | release artifact inventory |
| Supplementary Fig. S1 | `paper/figures/F3_rate_observation_class_summary.pdf` | `scripts/plot_main_family_figures.py` |
| Supplementary Fig. S2 | `paper/figures/S_F5_preproc_delta_heatmaps.pdf` | `scripts/plot_observation_eda.py` |
| Supplementary Fig. S3 | `paper/figures/S_F6_of_construction_comparison.pdf` | `scripts/plot_of_construction_comparison.py` |
| Supplementary Fig. S4 | `paper/figures/S_F_component_ablation_evidence.pdf` | `scripts/generate_component_ablation_evidence.py` |
| Supplementary Fig. S5 | `paper/figures/S_F_external_weak_evidence_summary.pdf` | `scripts/audit_external_weak_evidence.py` |
| Supplementary Fig. S6 | `paper/figures/S_F9_state_bundle_diagnostics.pdf` | `scripts/generate_state_bundle_visualization.py` |
| Supplementary Fig. S7 | `paper/figures/S_F10_decoupled_system_diagnostics.pdf` | `scripts/generate_decoupled_system_diagnostics.py` |
| Supplementary Fig. S8 | `paper/figures/S_F11_robust_fallback_diagnostics.pdf` | `scripts/generate_robust_fallback_diagnostics.py` |
| Supplementary Fig. S9 | `paper/figures/S_F12_within_transfer_compact_comparison.pdf` | `scripts/generate_within_transfer_compact_figure.py` |
| Supplementary Fig. S10 | `paper/figures/S_F13_of_construction_overlay_grid.pdf` | `scripts/plot_waveform_overlay_grid.py` |
| Supplementary Fig. S11 | `paper/figures/S_F14_dof_construction_overlay_grid.pdf` | `scripts/plot_waveform_overlay_grid.py` |
| Supplementary Fig. S12 | `paper/figures/S_F15_cohface_observation_class_overlay_atlas.pdf` | `scripts/plot_waveform_overlay_grid.py` |
| Supplementary Fig. S13 | `paper/figures/S_F16_mahnob_observation_class_overlay_atlas.pdf` | `scripts/plot_waveform_overlay_grid.py` |

## Supporting CSVs

The `paper/tables_ready/` directory stores final table-ready CSVs and
supplementary audit tables. The `paper/manifests/` directory stores the case
selection manifests used to regenerate overlay and failure-case figures.

Additional supporting tables include
`paper/tables_ready/S_T_final_observation_class_comparison.csv`,
`paper/tables_ready/S_T_external_weak_evidence_audit.csv`,
`paper/tables_ready/S_T_external_rr_manifest_summary.csv`,
`paper/tables_ready/S_T_dataset_distribution_eda.csv`,
`paper/tables_ready/S_T7_unified_split_operator_alignment.csv`,
`paper/tables_ready/S_T8_strict_scale_safe_hard_regime.csv`, and
`paper/tables_ready/S_T9_decoupled_validation_v2_summary.csv`.
