import os
import tempfile

import numpy as np

from core.pipeline.runner import extract_respiration


class _DummyDataset:
    name = "dummy"
    fs_gt = 32.0

    def __init__(self, video_path):
        self.data = [{
            "video_path": video_path,
            "subject": "s01",
            "trial": "t01",
            "chest_rois": [],
            "face_rois": [],
            "gt": np.zeros(10, dtype=np.float32),
        }]
        self.extract_calls = 0

    def extract_ROI(self, video_path, region):
        self.extract_calls += 1
        if region != "chest":
            return []
        # one valid ROI frame is enough for order validation
        return [np.ones((8, 8, 3), dtype=np.uint8) * 10]


class _WrappedLikeMethod:
    data_type = "chest"
    name = "base__robust_ossm"

    def process(self, data):
        # Wrapped path must guarantee ROI preparation before calling process.
        assert data.get("chest_rois"), "wrapped method called before ROI preparation"
        # Simulate wrapped method side-channel metadata for cleanup verification.
        data["roi_stats_t"] = [{"roi_mean": 1.0}]
        data["_gray_chest_rois"] = [np.ones((8, 8), dtype=np.uint8)]
        data["_obs_signal_cache"] = {"obs_of.npy": np.zeros(10, dtype=np.float32)}
        data["roi_intensity_mean"] = 1.0
        data["roi_intensity_std"] = 0.1
        data["roi_intensity_snr_db"] = 20.0
        data["roi_stats_source"] = "memory_cache"
        data["roi_stats_cache_path"] = "/tmp/fake"
        return {
            "signal_hat": np.zeros(10, dtype=np.float32),
            "track_hz": np.zeros(10, dtype=np.float32),
            "times_hz": np.arange(10, dtype=np.float32),
            "meta": "{}",
        }


class _WrappedCacheOnlyMethod:
    data_type = "chest"
    name = "base__robust_ossm_cache_only"

    def __init__(self):
        self.calls = 0
        self.fallback_calls = 0
        self.cache_only = True

    def can_run_without_chest_rois(self, data):
        return self.cache_only

    def process(self, data):
        self.calls += 1
        if not data.get("chest_rois"):
            return {
                "signal_hat": np.zeros(10, dtype=np.float32),
                "track_hz": np.zeros(10, dtype=np.float32),
                "times_hz": np.arange(10, dtype=np.float32),
                "meta": "{}",
            }
        self.fallback_calls += 1
        return {
            "signal_hat": np.zeros(10, dtype=np.float32),
            "track_hz": np.zeros(10, dtype=np.float32),
            "times_hz": np.arange(10, dtype=np.float32),
            "meta": "{}",
        }


def test_runner_prepares_roi_before_wrapped_method():
    tmp = tempfile.mkdtemp(prefix="runner_roi_")
    # file doesn't need real video for this unit test path
    video_path = os.path.join(tmp, "dummy.avi")
    open(video_path, "wb").close()

    ds = _DummyDataset(video_path)
    method = _WrappedLikeMethod()

    extract_respiration(
        datasets=[ds],
        methods=[method],
        results_dir=tmp,
        run_label="unit_run",
    )
    sample = ds.data[0]
    assert "roi_stats_t" not in sample
    assert "roi_intensity_mean" not in sample
    assert "roi_intensity_std" not in sample
    assert "roi_intensity_snr_db" not in sample
    assert "roi_stats_source" not in sample
    assert "roi_stats_cache_path" not in sample
    assert "_gray_chest_rois" not in sample
    assert "_obs_signal_cache" not in sample


def test_runner_skips_roi_extraction_for_cache_only_wrapped_method():
    tmp = tempfile.mkdtemp(prefix="runner_roi_cacheonly_")
    video_path = os.path.join(tmp, "dummy.avi")
    open(video_path, "wb").close()

    ds = _DummyDataset(video_path)
    method = _WrappedCacheOnlyMethod()

    extract_respiration(
        datasets=[ds],
        methods=[method],
        results_dir=tmp,
        run_label="unit_run_cache_only",
    )

    assert method.calls >= 1
    assert method.fallback_calls == 0
    assert ds.extract_calls == 0


def test_runner_falls_back_to_roi_extraction_when_cache_only_unavailable():
    tmp = tempfile.mkdtemp(prefix="runner_roi_fallback_")
    video_path = os.path.join(tmp, "dummy.avi")
    open(video_path, "wb").close()

    ds = _DummyDataset(video_path)
    method = _WrappedCacheOnlyMethod()
    method.cache_only = False

    extract_respiration(
        datasets=[ds],
        methods=[method],
        results_dir=tmp,
        run_label="unit_run_fallback",
    )

    assert method.calls >= 1
    assert method.fallback_calls >= 1
    assert ds.extract_calls >= 1
