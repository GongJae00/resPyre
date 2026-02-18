import os
import tempfile

import numpy as np

from components.models.core.base import OscillatorParams
from components.models.core.failure_monitor import FailureFlags
from components.models.heads.robust_ossm import oscillator_RobustOSSM


class _AlwaysDivergeMonitor:
    def __init__(self, *args, **kwargs):
        pass

    def update(self, state, nis, trace_P):
        return FailureFlags(diverge=True, phase_slip=False, locking=False, doubling=False)

    def reset(self, f_ref=None):
        return None


def test_diverge_frame_logs_pre_reset_state(monkeypatch):
    """When divergence triggers, current frame should log pre-reset state."""
    import components.models.heads.robust_ossm as mod

    monkeypatch.setattr(mod, "FailureMonitor", _AlwaysDivergeMonitor)

    original_init_state = mod.OscillatorPredictor.init_state
    call_counter = {"n": 0}
    sentinel_x = np.array([123.0, 456.0, np.log(0.25)], dtype=np.float64)

    def _patched_init_state(self, freq0):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            return original_init_state(self, freq0)
        P = np.diag([1.0, 1.0, 0.25 ** 2]).astype(np.float64)
        return sentinel_x.copy(), P

    monkeypatch.setattr(mod.OscillatorPredictor, "init_state", _patched_init_state)

    fs = 20.0
    sig = np.sin(2.0 * np.pi * 0.25 * np.arange(12) / fs)
    params = OscillatorParams(fs=fs, f_min=0.08, f_max=0.5, trace_cap=50.0)
    head = oscillator_RobustOSSM(params)
    tmp = tempfile.mkdtemp(prefix="diverge_log_")
    head.run(sig, fs, {"aux_save_dir": tmp, "trial_key": "d"})

    arr = np.load(os.path.join(tmp, "frame_logs", "d.npz"), allow_pickle=True)
    data = arr["data"]
    fields = list(arr["fields"])
    x1 = data[:, fields.index("x1")]
    fail = data[:, fields.index("fail_diverge")]

    first_fail = int(np.where(fail > 0.5)[0][0])
    # If reset happened before logging, x1 at failure would be sentinel 123.0.
    assert not np.isclose(x1[first_fail], sentinel_x[0]), "failure frame logged post-reset sentinel state"
