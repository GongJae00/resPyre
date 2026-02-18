import os
import tempfile

import numpy as np

from components.models.core.base import OscillatorParams
from components.models.heads.robust_ossm import oscillator_RobustOSSM
from core.pipeline.wrapped_method import OscillatorWrappedMethod


def test_roi_stats_generation_has_required_keys():
    rois = [
        np.ones((12, 12), dtype=np.float32) * 100.0,
        np.pad(np.ones((10, 10), dtype=np.float32) * 100.0, ((1, 1), (1, 1)), constant_values=np.nan),
        np.array([], dtype=np.float32),
    ]
    wrapper = OscillatorWrappedMethod("dof", "robust_ossm")
    stats_t, _, _, _ = wrapper._roi_stats_time_series(rois)

    assert len(stats_t) == 3
    for item in stats_t:
        for key in ("roi_mean", "roi_std", "roi_snr_db", "roi_cx", "roi_cy", "valid_ratio", "center_disp", "global_mean"):
            assert key in item


def test_roi_quality_wiring_changes_qvis_and_alpha_r():
    n = 40
    fs = 10.0
    sig = np.sin(2 * np.pi * 0.2 * np.arange(n) / fs)
    roi_stats_t = []
    for i in range(n):
        roi_stats_t.append({
            "roi_mean": 100.0,
            "global_mean": 120.0,
            "roi_std": 2.0,
            "roi_snr_db": 18.0 if i < 20 else 2.0,
            "valid_ratio": 1.0 if i < 20 else 0.2,
            "roi_cx": 0.5,
            "roi_cy": 0.5,
        })

    params = OscillatorParams(fs=fs, f_min=0.1, f_max=0.5)
    head = oscillator_RobustOSSM(params)
    tmp = tempfile.mkdtemp(prefix="roi_wiring_")
    head.run(sig, fs=fs, meta={"roi_stats_t": roi_stats_t, "aux_save_dir": tmp, "trial_key": "smoke"})

    log_path = os.path.join(tmp, "frame_logs", "smoke.npz")
    loaded = np.load(log_path, allow_pickle=True)
    arr = loaded["data"]
    fields = list(loaded["fields"])

    q_vis = arr[:, fields.index("q_vis")]
    alpha_R = arr[:, fields.index("alpha_R")]
    assert np.mean(q_vis[:20]) > np.mean(q_vis[20:])
    assert np.mean(alpha_R[:20]) < np.mean(alpha_R[20:])
