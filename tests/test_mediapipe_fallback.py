import os
import tempfile

import cv2
import numpy as np

from core.utils.common import detect_face, get_chest_ROI


def _make_dummy_video(path, width=96, height=72, fps=10, n_frames=6):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (15 + i, 20 + i, 25 + i)
        # add simple upper bright blob to make fallback deterministic enough
        cv2.circle(frame, (width // 2, int(height * 0.28)), 8, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()


def test_detect_face_fallback_bbox_is_valid():
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    bbox = detect_face(img)
    assert isinstance(bbox, list)
    assert len(bbox) == 4
    xmin, xmax, ymin, ymax = bbox
    assert 0 <= xmin < xmax <= 160
    assert 0 <= ymin < ymax <= 120


def test_get_chest_roi_works_without_mediapipe_solutions():
    tmp = tempfile.mkdtemp(prefix="mp_fallback_")
    video_path = os.path.join(tmp, "dummy.avi")
    _make_dummy_video(video_path, n_frames=8)

    rois, fps, elapsed = get_chest_ROI(video_path, dataset="cohface")
    assert len(rois) == 8
    assert fps > 0
    assert elapsed >= 0.0
    # ensure non-empty PIL crops are produced
    first = np.asarray(rois[0])
    assert first.ndim == 3
    assert first.shape[0] > 0 and first.shape[1] > 0
