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

    def extract_ROI(self, video_path, region):
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
