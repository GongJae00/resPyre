# Unified split operator-alignment companion

> Scope note: this is a same-split diagnostic companion retained for
> provenance. It is not the active full-dataset headline layer used by
> `paper/main.tex`, F4, or F5.

- source run: `results/cohface_rate_supervised_routed_full_gate_v1`
- decoupled root: `results/cohface_decoupled_system_gate_v2`

This table is the same-split companion to the main T5 mechanistic ablation.
Unlike the main T5 table, every row here comes from the same 32-trial COHFACE split or from decoupled systems materialized and re-evaluated on that same split.

| row | scope | rate MAE | rate r | aligned CCC | strict CCC | strict MAE | T4c PPI (s) | consistency | system conf. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw base (P1D_quad) | same_split_source_run | 0.170 | 0.965 | 0.865 | 0.002 | 55.931 | 0.558 | nan | nan |
| PARH rate expert | same_split_source_run | 0.145 | 0.900 | 0.890 | 0.138 | 0.745 | 0.376 | nan | nan |
| Adaptive observation law | same_split_source_run | 0.340 | 0.790 | 0.844 | 0.611 | 0.172 | 0.523 | nan | nan |
| Staged routed multi-output | same_split_source_run | 0.340 | 0.790 | 0.919 | 0.850 | 0.138 | 0.292 | nan | nan |
| Waveform expert only | same_split_source_run | 0.620 | 0.785 | 0.919 | 0.850 | 0.138 | 0.292 | nan | nan |
| Adaptive-rate decoupled system | same_split_decoupled_system | 0.340 | 0.790 | 0.919 | 0.850 | 0.138 | 0.292 | 0.364 | 0.233 |
| Decoupled system | same_split_decoupled_system | 0.145 | 0.900 | 0.919 | 0.850 | 0.138 | 0.292 | 0.391 | 0.262 |

## Reading rule
- Use this table to check whether the operator-alignment ordering still holds when provenance is restricted to one split.
- Use the main T5 table to keep the fuller mechanistic ladder, including lines that were validated in separate completed gates.

## Skipped incomplete rows

Rows below were omitted because at least one core metric was missing in the live artifact.

| row | scope | reason |
| --- | --- | --- |
| Shared observation law | same_split_source_run | missing core rate/waveform/strict metric |
