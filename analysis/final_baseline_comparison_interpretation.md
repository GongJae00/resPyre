# Final Baseline Comparison Interpretation

Observation:
  The final PARH-OSSM full-dataset run now has an explicit apples-to-apples comparison layer.
  Headline Base and OSSM-KF columns use the pre-locked P1D_quad direct and
  OSSM-KF (P1D quad) representative, while all eight fixed observation classes are
  retained in the supplementary observation-class table.

Evidence:
  COHFACE rate MAE: Base=0.195, OSSM-KF=0.485, PARH=0.335; PearsonR: Base=0.950, OSSM-KF=0.800, PARH=0.855.
  MAHNOB rate MAE: Base=6.600, OSSM-KF=4.325, PARH=2.410; PearsonR: Base=-0.030, OSSM-KF=-0.020, PARH=0.230.
  COHFACE aligned waveform CCC: Base=0.811, OSSM-KF=0.852, PARH=0.859.
  MAHNOB aligned waveform CCC: Base=0.352, OSSM-KF=0.345, PARH=0.342.
  COHFACE strict NMAE/span: Base=53.402, OSSM-KF=0.867, PARH=0.548.
  MAHNOB strict NMAE/span: Base=0.236, OSSM-KF=0.187, PARH=0.258.

Interpretation:
  COHFACE should be read as the clean-regime success case. MAHNOB should
  be read as a hard-regime observability boundary: rate can improve against
  fixed direct observation baselines, but strict waveform/cycle robustness
  remains limited and must not be overclaimed.

Decision:
  Main tables may report the representative fixed baseline and OSSM-KF
  comparator because they are pre-locked and evaluated on the same trial IDs.
  Full OF/DoF/P1D observation-class results belong in supplementary/diagnostic tables.

Paper value:
  This closes the reviewer-facing question: PARH-OSSM is not compared only
  to itself, and OSSM-KF remains a comparator rather than a hidden part of
  the proposed method.

Risk:
  The baseline layer is extracted from retained observation-class metrics rather
  than recomputing video processing from scratch. This is acceptable only
  because the exact full-dataset trial IDs and metric definitions are recorded
  in the provenance audit; it must remain explicit in Methods/Artifacts.
