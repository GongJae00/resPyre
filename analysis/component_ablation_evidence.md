# Component Ablation Evidence

This file is generated from existing paper-ready metrics. Negative `delta_rate_MAE` and negative `delta_waveform_DTW` are improvements; positive `delta_waveform_CCC`, `delta_rate_R`, and `delta_strict_CCC` are improvements.

| component | detachable test | delta rate MAE | delta waveform CCC | delta strict CCC | interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| OF displacement bridge | OF raw vs OF_bridge under the same PARH readout |  |  |  | Construction-level evidence: the final package uses OF/OF_bridge inside the joint observation law, while direct family rows are not regenerated in the integrated final run. |
| DoF signed-motion bridge | DoF raw vs DoF_bridge under the same PARH readout |  |  |  | Construction-level evidence: DoF_bridge is retained as a hard-regime signed-motion probe, not as the main morphology path. |
| P1D consensus | best single P1D harmonic vs P1D_cons under the same PARH readout |  |  |  | Construction-level evidence: P1D_cons is retained as a stable consensus observation inside the joint law, not as a claim that consensus alone dominates. |
| standard resonator Kalman comparator | best single Base vs best single OSSM-KF |  |  |  | Comparator-boundary evidence: OSSM-KF is a reference timing channel; the integrated final package does not treat it as the proposed adaptive observation law. |
| integrated final PARH-OSSM package | COHFACE final package vs MAHNOB hard-regime final package | 3.085 | -0.548 |  | The final model is strong on COHFACE but still limited on MAHNOB; this row supports the paper's observability/target-shift claim boundary. |
| target-local adaptive observation law | shared observation law vs local adaptive observation law |  |  |  | Rate and strict morphology improve when the observation law becomes local; waveform CCC tradeoff motivates decoupled readouts. |
| decoupled timing and morphology readouts | rate expert vs decoupled z_osc/z_full system | 0.000 | 0.029 | 0.705 | The decoupled readout improves rate MAE, waveform CCC, and strict CCC together; this is the strongest evidence for the final architecture. |
| waveform-specialized expert boundary | waveform-only expert vs decoupled system | -0.475 | 0.000 | 0.000 | Waveform-only is competitive for morphology but much worse for rate; this defends not collapsing the paper to a waveform fitter. |
| robust fallback policy | decoupled system vs consistency-first robust fallback |  |  |  | No live final-package robust row is required for the promoted model; keep fallback as diagnostic/hard-regime policy, not the default claim. |
