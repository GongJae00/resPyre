# Final Scientific Lessons From the Full-Trial Paper Package

Run log: `analysis/final_full_execution_20260510_023954.log`

This note records the scientific interpretation of the current submission
package. It is deliberately separate from `paper/main.tex` so that claims can be
checked before promotion into the manuscript.

## Execution Integrity

- The final full-trial execution path completed successfully.
- The final package audit currently tracks `95/95` required artifacts with `0`
  missing files.
- The paper-code consistency audit passes `145/145` checks. The check set now
  treats T4b/T4c/T5/T6/T6b and rate-source details as supplementary companions
  rather than required main-text numeric claims.
- `paper/main.pdf`, `paper/main.tex`, and the table-ready CSVs are numerically
  synchronized by audit.
- The current submission-sized manuscript uses `5` main figures and `3` main
  tables. The abstract is below the Scientific Reports guideline limit
  (`179` rough words in the current source). The PDF is currently `15` pages,
  so page length remains a manual editorial risk even though the display-item
  count is within the target.
- The operating-point decision remains unchanged: keep the locked default and
  do not promote target-specific tuning.

## Main Full-Trial Performance

| dataset | rate MAE | rate RMSE | rate Pearson r | aligned waveform CCC | aligned waveform MAE | strict CCC | strict NMAE/span | cycle PPI MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| COHFACE | 0.335 | 0.410 | 0.855 | 0.859 | 0.423 | 0.157 | 0.548 | 0.347 |
| MAHNOB-HCI | 2.410 | 2.920 | 0.230 | 0.342 | 0.910 | 0.000 | 0.258 | 1.480 |

Interpretation:

- COHFACE supports the clean-regime claim. PARH-OSSM remains accurate in rate,
  preserves aligned morphology, and improves strict/cycle timing relative to
  the raw direct profile observation.
- MAHNOB-HCI supports a hard-regime rate-improvement and observability-boundary
  claim, not a universal waveform-recovery claim.
- Strict waveform raw MAE is scale-sensitive and must not be interpreted alone.
  The paper-facing interpretation uses strict CCC, span-normalized errors, and
  cycle metrics as companions.

## Representative Baseline and Comparator Comparison

COHFACE:

- `P1D_quad direct` is a very strong rate baseline (MAE `0.195`, Pearson
  `0.950`), so the proposed method should not claim rate dominance on this
  clean dataset.
- PARH-OSSM improves aligned CCC over the direct baseline (`0.859` vs `0.811`)
  and improves strict/cycle timing substantially over direct P1D. It is close
  to but not uniformly better than `OSSM-KF (P1D quad)` on every aligned
  waveform metric.

MAHNOB-HCI:

- PARH-OSSM improves rate MAE over the representative direct baseline
  (`2.410` vs `6.600` bpm) and over `OSSM-KF (P1D quad)` (`2.410` vs
  `4.325` bpm), while Pearson remains modest (`0.230`).
- Aligned waveform metrics are not improved over the representative
  comparator. The paper must therefore state that PARH-OSSM improves MAHNOB
  rate under the current observation bank but does not solve MAHNOB morphology
  or strict waveform recovery.
- MAHNOB representative baseline/comparator rows are available for `488`
  computable trials, whereas PARH-OSSM is reported on the full `525` final
  tail-aligned trials. The common-488 paired deltas are supplementary evidence
  and should not be hidden.

## Rate-Source Decomposition Lesson

COHFACE:

- The native respiratory timing evidence is strong enough that the final track
  remains accurate without target-specific source switching.
- The success case supports the decoupled timing/morphology design: `z_osc`
  can be read as timing/rate evidence while `z_full` is evaluated as waveform
  morphology.

MAHNOB-HCI:

- The current observation bank sometimes contains weak timing information, but
  no target-computable diagnostic identifies a universally better source across
  trials.
- This is the central hard-regime lesson: source selection alone is not a
  complete solution unless the target-side reliability features are strong
  enough to identify the useful source without GT.

## Full-Trial Hard-Regime Failure Taxonomy

The MAHNOB-HCI target-side audit assigns:

- `400/525` trials to `Bounded by current observation bank`.
- `60/525` trials to `Source-selection room with posterior evidence`.
- `35/525` trials to `Source-selection room with agreement evidence`.
- `19/525` trials to `Likely video/reference limited`.
- `9/525` trials to `Oracle room but weak GT-free evidence`.
- `2/525` trials to `low_target_observability`.

Interpretation:

- Most MAHNOB-HCI failures are not simply wrong source selection. The current
  OF/DoF/P1D observation bank often lacks enough target-computable respiratory
  evidence for robust waveform recovery.
- A smaller subset does show source-selection room, but promoting a selector
  from this would risk target-tuned behavior unless the GT-free diagnostics are
  improved.
- The taxonomy is a paper strength because it separates observation limitation,
  source-selection room, and likely reference/video-limited cases instead of
  reporting a single failed score.

## What The Model Demonstrates

Supported claims:

- Camera/ROI-based respiration is not just a waveform-extraction problem. The
  important modeling problem is how incomplete, heterogeneous observations of
  respiratory motion should be trusted, weighted, and read out.
- The eight observation operators are not interchangeable. They expose
  velocity-like, energy-like, signed-motion, and profile-displacement evidence
  with different nuisance risks.
- Candidate views are evidence views, not competing final models.
- The adaptive observation law is a reliability and role-weighting law, not a
  hard selector.
- Separating timing (`z_osc`) from morphology (`z_full`) is justified by the
  different behavior of rate, aligned waveform, strict waveform, and cycle
  metrics.
- `OSSM-KF (P1D quad)` is a comparator and weak timing-evidence boundary, not
  the proposed method body.

Claims that must remain cautious:

- Do not claim universal MAHNOB-HCI waveform robustness.
- Do not claim PARH-OSSM dominates every fixed family on every metric.
- Do not report raw strict waveform MAE without the normalized companion
  metrics.
- Do not use V4V or SCAMPS as real waveform-performance evidence. They remain
  external rate-only/synthetic diagnostic context.
- Do not promote sensitivity or source-arbiter variants as final defaults unless
  a new no-leakage lock note justifies them.

## Model-Limit Lessons

1. Observation-bank limitation:
   Current fixed observations are not sufficiently informative on many MAHNOB-HCI
   trials. Better respiratory observations, not merely more hyperparameter
   tuning, are needed.

2. Target-local reliability limitation:
   The framework formalizes target-side reliability, but the current GT-free
   features are not strong enough to identify every useful source or within-trial
   observability change.

3. Readout ambiguity:
   Rate, aligned waveform, strict waveform, and cycle metrics can move in
   different directions. A useful rate source does not automatically give a
   useful waveform morphology source.

4. Reference/evaluation sensitivity:
   Zero-lag, unit-preserving strict waveform metrics are intentionally harsh and
   expose lag/scale/reference risks that aligned waveform metrics can hide.

## Paper Position

The final paper should be framed as a mechanistic and auditable framework, not
as a universal state-of-the-art performance claim.

The strongest defensible story is:

Camera/ROI-based respiration requires more than selecting one waveform
extractor. A robust model must construct multiple observation operators, estimate
their target-side reliability without target labels, inject them into an
interpretable respiratory state, decouple timing from morphology, and report
where the observation bank is insufficient. The current package demonstrates
this clearly on COHFACE and uses MAHNOB-HCI to expose the remaining
observability boundary honestly.
