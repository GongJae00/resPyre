# Final Manual Review Checklist

This checklist is intentionally manual. Automated audits can verify paths,
numbers, provenance, and build status, but not whether the final PDF reads
well to a reviewer.

- [ ] Confirm title, authors, affiliations, and correspondence details on `paper/main.pdf`.
- [ ] Confirm the current submission manuscript still has `5` main figures, `3` main tables, and an abstract under 200 words after any final edit.
- [ ] Decide whether the current page count is acceptable for first review or whether another compression pass is needed toward the Scientific Reports ideal page length.
- [ ] Inspect F1 architecture (`paper/figures/F1_architecture.pdf`): decide whether the current architecture rendering is clear enough or should be redrawn before submission.
- [ ] Inspect every main figure for clipping, tiny text, broken legends, and panel-label ambiguity.
- [ ] Confirm the three main tables T3/T4/T7 match `paper/tables_ready/*.csv`; confirm T4b/T4c/T5/T6/T6b are treated as supplementary/diagnostic companions.
- [ ] Check that `P1D_quad direct`, `OSSM-KF (P1D quad)`, and `PARH-OSSM` are not described as interchangeable methods.
- [ ] Check that the COHFACE claim is strong but not overstated: PARH improves waveform/strict/cycle, while direct P1D_quad remains a very strong rate baseline.
- [ ] Check that the MAHNOB claim is explicitly bounded: rate improves over representative baseline/comparator, but waveform/strict/cycle remains a hard-regime limitation.
- [ ] Check that strict raw MAE is always interpreted with strict CCC and span-normalized companion metrics.
- [ ] Check that `OSSM-KF` reads only as a comparator/weak evidence channel, never as the proposed model.
- [ ] Check that candidate views are described as evidence views, not final model candidates.
- [ ] Check that adaptive observation law is described as reliability/role weighting, not one-best selector.
- [ ] Check that MAHNOB failure discussion reads as evidence-backed boundary analysis, not a post-hoc excuse.
- [ ] Decide whether a short cover note to the corresponding author is needed before external review.
