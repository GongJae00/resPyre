# PARH-OSSM Learning Boundary Audit (2026-05-21)

This audit fixes what is rule-based, what is target-side self-calibration,
what is shallow source-supervised fitting, and what remains diagnostic/no-go.
It is a paper-facing guardrail: performance claims must be consistent with this boundary.

## Boundary Thesis

- Learn mappings only where the physics does not determine observability.
- Keep physiology, observation operators, and state-space inference fixed and interpretable.
- Use target-computable evidence for adaptation; never use target GT for target selection.
- Prefer abstention/conservative covariance inflation over hard replacement under ambiguity.
- Treat source-supervised components as priors or diagnostics until no-sweep transfer evidence promotes them.

## Component Decisions

| Component | Boundary | Target GT | Status | Evidence Present |
|---|---|---|---|---|
| Observation operators: OF, OF_bridge, DoF, DoF_bridge, P1D_lin, P1D_quad, P1D_cub, P1D_cons | Fixed rule-based observation equations | Forbidden | Keep | yes |
| Candidate views: Raw, Detr., Band, Sign, R-z, Current, Helper | Fixed signal-processing views used as evidence | Forbidden | Keep with locked definitions | yes |
| Resonant state x_t=[h1c,h1s,h2c,h2s,b,b_dot,r,r_dot] | Fixed physiology-aligned state structure | Forbidden | Keep | yes |
| KF/RTS inference and Student-t innovation reweighting | Probabilistic inference, not learned | Forbidden | Keep | yes |
| Target-side sign/scale/lag calibration | Online self-calibration from target observations | Forbidden | Allowed if diagnostics are logged | yes |
| Windowed target reliability graph | Target-computable time-local reliability prior for the observation law | Forbidden | Required in final paper path | yes |
| Candidate-rate posterior final bounded readout | Deterministic target-computable timing evidence readout with preservation guards | Forbidden | Final paper-candidate readout when `--rate-posterior-output-source final` is used | yes |
| Source-supervised observation/readout arbiter | Shallow learned/fitted reliability mapping from source labels | Forbidden for fitting or target selection | Diagnostic/no-go for state path; readout-only candidate | yes |
| Source-validity graph v4 / source_validity readout | Source-informed diagnostic posterior | Forbidden | No-go except guarded preservation | yes |
| Torch learned observation/waveform probes | Deep/black-box learned controls | Subject-split only; not target-GT selection | Not current promoted path | yes |

## Detailed Rationale

### Observation operators: OF, OF_bridge, DoF, DoF_bridge, P1D_lin, P1D_quad, P1D_cub, P1D_cons

- Boundary: Fixed rule-based observation equations
- Evidence: `components/observations/methods.py; scripts/materialize_calibrated_multifamily_parh_system.py`
- Rationale: These are measurement operators, not trained models. They expose heterogeneous projections of respiration: velocity, displacement bridge, burst timing, morphology, and consensus stability.
- Current gap: MAHNOB audit shows the bank often contains a better candidate, but target-side reliability is not sharp enough to use it safely.
- Next action: Do not replace with a learned waveform head; improve target-computable observability evidence.

### Candidate views: Raw, Detr., Band, Sign, R-z, Current, Helper

- Boundary: Fixed signal-processing views used as evidence
- Evidence: `components/models/heads/parh_ossm.py; paper/figures/F2_dataset_and_observation_regime.pdf`
- Rationale: Views expose different nuisances and supports. They should inform reliability, not act as separate target-GT-selected models.
- Current gap: Some views are visually/diagnostically weak on MAHNOB; N/A/zero entries must be explained as unavailable evidence, not model failure.
- Next action: Report view availability and reliability as diagnostics; keep view definitions fixed before full runs.

### Resonant state x_t=[h1c,h1s,h2c,h2s,b,b_dot,r,r_dot]

- Boundary: Fixed physiology-aligned state structure
- Evidence: `components/models/heads/parh_ossm.py; paper/main.tex`
- Rationale: The oscillator/harmonic/baseline/residual split is the model thesis: timing and morphology are coupled through physiology but not forced into one waveform objective.
- Current gap: State structure alone does not solve target observability; wrong observation trust still corrupts rate readout.
- Next action: Keep the state fixed; improve observation covariance/mixture evidence feeding the state.

### KF/RTS inference and Student-t innovation reweighting

- Boundary: Probabilistic inference, not learned
- Evidence: `components/models/heads/parh_ossm.py; components/models/core/smoother.py`
- Rationale: Kalman/RTS gives transparent state inference; Student-t downweights outlier innovations without training a black-box rejector.
- Current gap: It is protective, not a selector. It cannot recover information when all observations are weak or aliased.
- Next action: Use NIS/lambda/prior-collapse as diagnostics and guards, not as post-hoc performance knobs.

### Target-side sign/scale/lag calibration

- Boundary: Online self-calibration from target observations
- Evidence: `scripts/extract_target_reliability_graph_features.py; execute.md`
- Rationale: New people/environments change observation coordinate systems. Relative calibration is necessary, but it must be estimated from inter-family agreement rather than target labels.
- Current gap: Reference/nonstationary lag dominates many MAHNOB failures, so calibration must distinguish real phase lag from unphysical matching.
- Next action: Audit bounded-lag vs unbounded-lag cases; do not promote unphysical lag compensation.

### Windowed target reliability graph

- Boundary: Target-computable time-local reliability prior for the observation law
- Evidence: `scripts/extract_target_reliability_graph_features.py; tests/test_target_reliability_graph_features.py; execute.md`
- Rationale: The final path needs local, target-computable evidence about which observation family/view is trustworthy. It is not a target-label selector: it provides reliability, support, and state-role priors that are consumed by the observation law.
- Current gap: Full validation has not been rerun after the closure patch, so the final priors must be regenerated before the paper package is complete.
- Next action: Regenerate `analysis/final_priors/*_windowed.csv` from `execute.md` before full materialization; the paper-candidate activation audit must show the runtime prior was applied.

### Candidate-rate posterior final bounded readout

- Boundary: Deterministic target-computable timing evidence readout with preservation guards
- Evidence: `scripts/materialize_calibrated_multifamily_parh_system.py; tests/test_rate_posterior_calibrated_readout.py; tests/test_paper_candidate_activation_contract.py; execute.md`
- Rationale: The posterior may adjust z_osc only when candidate evidence is specific and target-observable; otherwise it must preserve the native state-space readout. This keeps OSSM-KF from becoming an unbounded hidden fallback.
- Current gap: The closure contract is implemented, but the no-sweep full run is still pending after the latest patch.
- Next action: Run the final full commands and require `activation_audit_summary.json` to pass for each real dataset.

### Source-supervised observation/readout arbiter

- Boundary: Shallow learned/fitted reliability mapping from source labels
- Evidence: `scripts/materialize_calibrated_multifamily_parh_system.py; tests/test_observation_law_contract.py`
- Rationale: A small auditable mapping is acceptable when the target has no labels, but it must not override state/morphology unless transfer-safe ablations prove it.
- Current gap: It improves some MAHNOB trials but regresses others; source validity is not yet robust enough.
- Next action: Use as prior/diagnostic only; absorb useful features into target-side reliability, not a hard selector.

### Source-validity graph v4 / source_validity readout

- Boundary: Source-informed diagnostic posterior
- Evidence: `tests/test_rate_posterior_calibrated_readout.py; tests/test_rate_posterior_output_role.py`
- Rationale: It tests whether source-learned validity transfers. Current evidence says raw replacement is unsafe.
- Current gap: Ungarded source-validity can over-resolve ambiguous target evidence.
- Next action: Keep guarded/abstention behavior; do not promote raw source-validity replacement.

### Torch learned observation/waveform probes

- Boundary: Deep/black-box learned controls
- Evidence: `paper/main.tex; paper/supplementary_information.tex`
- Rationale: They are useful controls showing what learned capacity can do, but they weaken interpretability and several branches failed transfer or rate/waveform decoupling.
- Current gap: Good COHFACE behavior did not establish universal target robustness.
- Next action: Keep as baselines/controls unless a split protocol proves transfer without sacrificing interpretability.

## Current Performance Interpretation

The latest MAHNOB bottleneck audit should be interpreted through this boundary:

- The observation bank is not empty: oracle candidate-bank median MAE is lower than current readout.
- The main failure is target-side observability and reliability specificity, not lack of deep capacity alone.
- Source-supervised fitting is useful evidence, but current transfer evidence is not strong enough for hard promotion.
- The next promotable patch must sharpen target-computable reliability while preserving the conservative fallback behavior.

## Required Next Validation

1. Regenerate this audit and the design-boundary audit before any full run.
2. Run no-sweep COHFACE, MAHNOB, COHFACE->MAHNOB, and MAHNOB->COHFACE validation.
3. Report ablation ladder: Base, OSSM-KF, PARH-fixed, PARH-R, PARH-R+pi, PARH-full, PARH-target.
4. Promote only changes that improve one bottleneck class without regressing ambiguous/preservation-safe cases.
