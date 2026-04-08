import os
import tempfile

import numpy as np

from components.observations.methods import OF_Model
from components.models.core.base import OscillatorParams
from components.models.heads.robust_ossm import oscillator_RobustOSSM
from core.pipeline.wrapped_method import OscillatorWrappedMethod, compute_roi_stats_time_series, save_roi_stats_cache


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


def test_roi_stats_disk_cache_reuse(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="roi_cache_")
    video_path = os.path.join(tmp, "sample.avi")
    with open(video_path, "wb"):
        pass

    rois = [
        np.ones((12, 12), dtype=np.float32) * 100.0,
        np.ones((12, 12), dtype=np.float32) * 98.0,
        np.ones((12, 12), dtype=np.float32) * 102.0,
    ]
    wrapper = OscillatorWrappedMethod("dof", "robust_ossm")

    monkeypatch.setattr(wrapper.base_method, "process", lambda _data: np.zeros(16, dtype=np.float64))

    def _fake_run(base_signal, fs, meta):
        n = int(base_signal.size)
        return {
            "signal_hat": np.asarray(base_signal, dtype=np.float32),
            "track_hz": np.full(n, 0.25, dtype=np.float32),
            "rr_hz": 0.25,
            "rr_bpm": 15.0,
            "meta": "{}",
        }

    monkeypatch.setattr(wrapper.osc_head, "run", _fake_run)

    data1 = {
        "video_path": video_path,
        "fps": 20.0,
        "dataset_name": "dummy",
        "chest_rois": rois,
    }
    wrapper.process(data1)
    assert data1.get("roi_stats_source") == "computed"
    cache_path = os.path.join(tmp, "obs_roi_stats_v1.npz")
    assert os.path.exists(cache_path)

    data2 = {
        "video_path": video_path,
        "fps": 20.0,
        "dataset_name": "dummy",
        "chest_rois": rois,
    }
    monkeypatch.setattr(
        wrapper,
        "_roi_stats_time_series",
        lambda _rois: (_ for _ in ()).throw(AssertionError("should not recompute roi stats")),
    )
    wrapper.process(data2)
    assert data2.get("roi_stats_source") == "disk_cache"
    assert isinstance(data2.get("roi_stats_t"), list) and len(data2["roi_stats_t"]) == len(rois)


def test_wrapped_method_requires_roi_source_when_no_rois(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="roi_missing_src_")
    video_path = os.path.join(tmp, "sample.avi")
    with open(video_path, "wb"):
        pass

    wrapper = OscillatorWrappedMethod("dof", "robust_ossm")
    monkeypatch.setattr(wrapper.base_method, "process", lambda _data: np.zeros(16, dtype=np.float64))

    data = {
        "video_path": video_path,
        "fps": 20.0,
        "dataset_name": "dummy",
        "chest_rois": [],
    }
    try:
        wrapper.process(data)
    except ValueError as exc:
        assert "cannot build roi_stats_t without chest ROIs" in str(exc)
    else:
        raise AssertionError("expected ValueError when no chest ROIs and no roi_stats cache")


def test_can_run_without_chest_rois_with_disk_caches():
    tmp = tempfile.mkdtemp(prefix="roi_cacheonly_ready_")
    video_path = os.path.join(tmp, "sample.avi")
    with open(video_path, "wb"):
        pass

    # Base observation cache required by cache-only path.
    np.save(os.path.join(tmp, "obs_dof.npy"), np.zeros(32, dtype=np.float32))
    wrapper = OscillatorWrappedMethod("dof", "robust_ossm")

    data = {"video_path": video_path, "chest_rois": []}
    assert wrapper.can_run_without_chest_rois(data) is False

    rois = [np.ones((8, 8), dtype=np.float32) * 100.0 for _ in range(5)]
    stats_t, _, _, _ = compute_roi_stats_time_series(rois)
    save_roi_stats_cache(video_path, 20.0, stats_t)

    assert wrapper.can_run_without_chest_rois(data) is True


def test_can_run_without_chest_rois_with_of_bridge_disk_caches():
    tmp = tempfile.mkdtemp(prefix="roi_cacheonly_ofbridge_")
    video_path = os.path.join(tmp, "sample.avi")
    with open(video_path, "wb"):
        pass

    np.save(os.path.join(tmp, "obs_of_bridge.npy"), np.zeros(32, dtype=np.float32))
    wrapper = OscillatorWrappedMethod("of_disp_bridge", "robust_ossm")

    data = {"video_path": video_path, "chest_rois": []}
    assert wrapper.can_run_without_chest_rois(data) is False

    rois = [np.ones((8, 8), dtype=np.float32) * 100.0 for _ in range(5)]
    stats_t, _, _, _ = compute_roi_stats_time_series(rois)
    save_roi_stats_cache(video_path, 20.0, stats_t)

    assert wrapper.can_run_without_chest_rois(data) is True


def test_base_observation_uses_in_memory_cache_after_first_compute(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="obs_mem_cache_")
    video_path = os.path.join(tmp, "sample.avi")
    with open(video_path, "wb"):
        pass

    calls = {"n": 0}

    def _fake_of(rois, fps):
        calls["n"] += 1
        return np.arange(6, dtype=np.float64), {}

    monkeypatch.setattr("components.observations.methods.OF", _fake_of)

    model = OF_Model()
    data = {
        "video_path": video_path,
        "fps": 20.0,
        "chest_rois": [np.ones((8, 8, 3), dtype=np.uint8) * 10 for _ in range(3)],
    }

    sig1 = model.process(data)
    assert calls["n"] == 1
    assert os.path.exists(os.path.join(tmp, "obs_of.npy"))

    os.remove(os.path.join(tmp, "obs_of.npy"))
    sig2 = model.process(data)

    assert calls["n"] == 1
    assert np.allclose(sig1, sig2)
