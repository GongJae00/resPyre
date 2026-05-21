# Final Operating-Point Research Feedback

This document records scientific feedback from the bounded operating-point
sensitivity run. It is intended for Discussion, Limitations, and future model
design, not for target-specific tuning.

## 1. Stable Operating Point

Observation:
  `locked_default` is the most stable choice across COHFACE preservation and
  MAHNOB hard-regime behavior.

Evidence:
  COHFACE stays strong at rate MAE `0.280`, PearsonR `0.920`, aligned waveform
  CCC `0.870`. MAHNOB remains difficult at rate MAE `3.365`, PearsonR `0.280`,
  aligned waveform CCC `0.333`, and strict CCC near zero.

Interpretation:
  The default does not solve MAHNOB, but it also does not hide the failure by
  overfitting the target regime. This is closer to an honest operating point
  for a paper about observation-state modeling limits.

Decision:
  Adopt `locked_default`.

Paper value:
  Use this as evidence that the final method is not a MAHNOB-tuned setting.

Risk:
  The reader may ask why a setting with lower MAHNOB MAE was not adopted. The
  answer is that the improvement is small, not multi-metric, and comes with
  source-domain or waveform tradeoffs.

## 2. Shorter Window Behavior

Observation:
  `more_local_windows` slightly lowers rate MAE but worsens correlation and
  strict/waveform stability.

Evidence:
  MAHNOB rate MAE changes `3.365 -> 3.355`, but PearsonR drops `0.280 -> 0.140`
  and abstain pressure rises `0.117 -> 0.125`. COHFACE rate MAE changes
  `0.280 -> 0.275`, but strict CCC drops `0.182 -> 0.171`.

Interpretation:
  Higher time locality is not automatically better. In hard regimes, local
  windows can track unstable evidence and reduce global rate ranking even if
  median absolute error moves slightly.

Decision:
  Diagnostic-only. Do not adopt.

Paper value:
  This supports the claim that time-local target reliability is necessary but
  must be regularized by physiological/state consistency, not treated as a
  simple shorter-window optimization.

Risk:
  The MAHNOB aligned CCC improvement (`0.333 -> 0.344`) is tempting, but it is
  not enough to justify a default change because rate PearsonR degrades.

## 3. Longer Window Behavior

Observation:
  `more_stable_windows` slightly improves MAHNOB rate MAE/PearsonR but hurts
  COHFACE and does not fix waveform.

Evidence:
  MAHNOB rate MAE improves `3.365 -> 3.280` and PearsonR `0.280 -> 0.305`.
  COHFACE rate MAE worsens `0.280 -> 0.345`; COHFACE aligned CCC drops
  `0.870 -> 0.864`; MAHNOB aligned CCC drops `0.333 -> 0.329`; MAHNOB strict
  CCC remains near zero.

Interpretation:
  Longer aggregation can stabilize noisy target evidence, but it blurs useful
  timing information in easier regimes and does not address strict waveform
  alignment. This points to a stability/locality tradeoff, not a solved
  robustness problem.

Decision:
  Reject as paper default; retain as diagnostic evidence.

Paper value:
  Useful for explaining why the model should not be tuned with a single target
  scalar and why future work needs state-aware local reliability rather than
  coarse window length changes.

Risk:
  If only MAHNOB rate MAE is reported, this could look like a better setting.
  The table must be interpreted with COHFACE preservation and waveform metrics.

## 4. Support Threshold Behavior

Observation:
  `stricter_cross_family_support` and `looser_cross_family_support` are nearly
  identical to `locked_default` after downstream materialization.

Evidence:
  COHFACE and MAHNOB rate, aligned waveform, strict, guard alpha, and abstain
  values remain effectively unchanged across support thresholds.

Interpretation:
  Small changes to support-correlation and residual thresholds are not the
  active bottleneck. The downstream readout and target observability boundary
  dominate over this local threshold.

Decision:
  Diagnostic-only. Do not adopt or tune.

Paper value:
  Supports the claim that failures are not explained by one arbitrary
  cross-family threshold.

Risk:
  The support prior still changes internally; if future code uses these values
  more strongly, this conclusion must be rechecked.

## 5. Readout Separation

Observation:
  Rate and waveform respond differently to operating-point changes.

Evidence:
  `more_local_windows` improves MAHNOB aligned CCC but damages rate PearsonR.
  `more_stable_windows` improves MAHNOB rate MAE/PearsonR but damages aligned
  CCC. Strict waveform remains unchanged in all MAHNOB conditions.

Interpretation:
  This supports the decoupled readout design: `z_osc` timing/rate evidence and
  `z_full` morphology/waveform evidence should not be forced into a single
  optimization target. MAHNOB strict failure is not fixed by reliability window
  changes, implying the remaining bottleneck is observability/reference/readout
  alignment rather than a simple confidence calibration error.

Decision:
  Keep decoupled readouts and locked default.

Paper value:
  This is a strong Discussion point: sensitivity reveals why scalar tuning is
  misleading and why separate timing/morphology outputs are scientifically
  necessary.

Risk:
  The paper must avoid claiming strict waveform robustness on MAHNOB.

## 6. Diagnostic Variables

Observation:
  Guard alpha and abstain pressure move only weakly across the grid.

Evidence:
  MAHNOB guard alpha stays around `0.185-0.189`; abstain pressure stays around
  `0.117-0.125`. COHFACE guard alpha stays around `0.240-0.252`.

Interpretation:
  The current guard is not highly sensitive to the tested support threshold
  changes. The hard-regime limit is likely not solved by mild trust-weight
  rescaling alone.

Decision:
  Preserve guard design for final full; do not tune guard thresholds.

Paper value:
  Supports a conservative claim: the guard detects hard-regime ambiguity but is
  not a magic correction mechanism.

Risk:
  Future stronger time-local guard designs may behave differently; this
  sensitivity only covers the current bounded grid.

## Bottom Line

The sensitivity run did not find a better final default. It produced a useful
scientific lesson: MAHNOB hard-regime behavior is robust to reasonable
operating-point perturbations, so the remaining failure should be discussed as
an observation/readout/evaluation boundary rather than a tuning miss.
