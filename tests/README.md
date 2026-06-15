# Tests

The active test suite covers:

- release-boundary checks: `test_design_boundary_audit.py`,
  `test_external_weak_evidence_audit.py`, `test_learning_boundary_audit.py`,
  `test_storage_config_runtime.py`, `test_runtime_status_metadata.py`
- PARH-OSSM contracts: observation-class ordering, observation-law behavior,
  calibrated rate readout, rate-source decomposition, and target-observability
  diagnostics
- pipeline regressions: evaluation, plotting, metadata, status, and result
  integrity

Do not commit generated `__pycache__` files or local experiment copies.
