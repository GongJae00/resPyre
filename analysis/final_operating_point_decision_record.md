# Final Operating-Point Decision Record

Decision date: 2026-05-09

Source files:

- `analysis/final_operating_point_sensitivity.csv`
- `analysis/final_operating_point_sensitivity.md`
- `analysis/final_operating_point_sensitivity_interpretation.md`
- `analysis/final_operating_point_research_feedback.md`

## Decision

Keep the existing `locked_default` operating point for the final full run.

No `execute.md` default setting change is justified by this sensitivity run.

## Operating-Point Judgments

| operating point | judgment | reason |
| --- | --- | --- |
| `locked_default` | adopt | Best global balance. Preserves COHFACE and honestly exposes MAHNOB hard-regime limits. |
| `more_local_windows` | diagnostic-only | Slight MAE movement, but MAHNOB PearsonR drops and COHFACE strict worsens. |
| `more_stable_windows` | reject | MAHNOB rate improves slightly, but COHFACE rate MAE worsens and waveform does not improve. |
| `stricter_cross_family_support` | diagnostic-only | No meaningful downstream change; not an active bottleneck. |
| `looser_cross_family_support` | diagnostic-only | No meaningful downstream change; not an active bottleneck. |

## Adoption Criteria Check

Criterion: COHFACE preservation does not collapse.

Result: `locked_default` passes. `more_stable_windows` weakens COHFACE rate MAE
from `0.280` to `0.345`, so it should not be promoted.

Criterion: MAHNOB improvement is not limited to one scalar.

Result: no alternative passes. `more_stable_windows` improves MAHNOB rate MAE
and PearsonR slightly but worsens aligned waveform and does not affect strict
waveform. `more_local_windows` slightly improves MAHNOB aligned CCC but lowers
PearsonR.

Criterion: change is justified by semantic reliability, not target-GT tuning.

Result: no change required. Keeping `locked_default` avoids target-GT setting
selection.

Criterion: the paper can explain why this setting is used.

Result: `locked_default` is explainable as a balanced locality/stability point.
The sensitivity run supports its stability rather than identifying a tuned
target optimum.

## Final-Full Instruction

Use the existing final full command path in `execute.md`.

Do not mix table rows from the sensitivity runs into final paper tables. They
belong in analysis/supplementary sensitivity evidence only unless the manuscript
explicitly describes them as sensitivity diagnostics.

## Claim Boundary

Allowed:

- The operating point is not target-tuned.
- MAHNOB strict waveform failure persists across reasonable target-computable
  reliability settings.
- Sensitivity reveals a real timing/morphology tradeoff.

Not allowed:

- Claiming `more_stable_windows` as a better MAHNOB setting.
- Claiming support-threshold tuning solves hard-regime behavior.
- Claiming MAHNOB strict waveform robustness.
