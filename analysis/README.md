# Analysis Artifacts

This directory keeps the small, paper-facing audit artifacts used to verify the
final PARH-OSSM submission package.

Retained artifacts include:

- dataset-scope and weak external-evidence audits
- final design, learning-boundary, paper-contract, and consistency audits
- target-computable reliability priors required by `execute.md`
- post-run rate-source, observability, baseline, statistical, and claim-boundary
  reports

Large transient diagnostics are not part of the public release. In particular,
edge-level reliability graph CSVs under `analysis/final_priors/` are generated
by `execute.md` for auditing and are intentionally ignored by Git.
