# PARH-OSSM Reviewer Risk Register

## Purpose

This file lists the main reasons a Scientific Reports submission could be
rejected or pushed into major revision, together with the required mitigation.

## Risk register

| Risk | Why reviewers will care | Current status | Required mitigation before submission |
|------|-------------------------|----------------|---------------------------------------|
| Overclaim vs evidence | The current repo does not yet support universal superiority or MAHNOB claims | High | Keep claims limited to persistent artifacts only |
| MAHNOB missing | The final story requires an irregular regime, but current artifacts are incomplete | High | Complete MAHNOB rerun and regenerate T3/T4/T6 |
| COHFACE-only over-interpretation | Clean datasets can hide structural differences | High | Use COHFACE as do-no-harm regime, not headline proof of superiority |
| No intent-aligned ablation | Reviewers will ask whether baseline/residual/helper path actually matter | High | Run and report the locked T5 ablations |
| Calibration asymmetry | T6 currently favors PARH because KFstd lacks comparable diagnostics | Medium | Add comparator-side diagnostics or clearly mark T6 as PARH-specific |
| Figure cherry-picking | Overlay figures are vulnerable to selection bias accusations | High | Use manifest-based top/median/bottom case selection |
| Manuscript/story drift | Older NAROSSM language can re-enter the paper and break consistency | High | Keep `paper/MANUSCRIPT_BLUEPRINT.md` as the story lock |
| Model not fully causal | Helper preprocessing still uses offline filtering in the current implementation stack | Medium | State this honestly or add causal preprocessing alternative for deployment claims |
| Weak differentiation from KFstd | If PARH still loses T4 after rerun, reviewers will question added complexity | High | Use regime analysis, ablation, and diagnostics to show what the extra states are doing |
| Missing significance strategy | Too many metrics with ad hoc tests looks weak | Medium | Predefine confirmatory tests and move exploratory analysis to supplement |
| Reference framing mismatch | Over-indexing on robust filtering without respiration-specific positioning weakens novelty | Medium | Strengthen physiology and contactless respiration literature positioning |
| Reproducibility ambiguity | Reviewers may not understand which tables come from which artifacts | Medium | Keep figure/table index and artifact paths explicit in manuscript and supplement |

## Immediate blockers

- full COHFACE rerun under current code
- full MAHNOB rerun under current code
- ablation runs
- overlay manifest generation
- final figure asset generation

## Safe reviewer-facing wording

Use:

- `physiology-aligned decomposition`
- `dual-output analysis`
- `self-auditing state-space scaffold`
- `do-no-harm behavior on stationary data`
- `irregular-regime motivation`

Avoid:

- `first`
- `optimal`
- `universal`
- `best`
- `solves`
- `equivalent to deep learning`
- `superior across datasets`

## Final submission gate

Do not freeze the paper until all items below are true:

- every main figure/table maps to a persistent file path
- all main claims map to completed rerun artifacts
- ablations answer a mechanistic question rather than a cosmetic one
- MAHNOB interpretation is present and artifact-backed
- architecture and overlay figures are generated from locked sources
- manuscript language matches the actual implemented model
