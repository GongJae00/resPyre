# Tests

The active test suite is split by release role:

- Paper/full-package guards: `test_design_boundary_audit.py`,
  `test_external_weak_evidence_audit.py`, `test_learning_boundary_audit.py`,
  `test_storage_config_runtime.py`, `test_runtime_status_metadata.py`.
- PARH-OSSM contracts: observation-class ordering, observation-law behavior,
  calibrated rate readout, rate-source decomposition, and target-observability
  audits.
- Pipeline regression tests: evaluation, plotting, metadata, status, and result
  integrity.

Historical pre-PARH smoke tests and obsolete Optuna/QROBF/ROI-era tests were
removed from the public test surface. Do not add generated `__pycache__` files
or legacy copies back into this directory.
