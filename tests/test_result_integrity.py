import os
import pickle
import tempfile

import pytest

from core.pipeline.common import _merge_results_payload, resolve_target_run_dirs
from main import _warn_duplicate_methods


def test_merge_results_prunes_stale_methods_with_method_order():
    tmp = tempfile.mkdtemp(prefix="merge_prune_")
    out = os.path.join(tmp, "trial.pkl")

    _merge_results_payload(
        out,
        {
            "video_path": "v",
            "fps": 20,
            "gt": [],
            "fs_gt": 32,
            "estimates": [
                {"method": "A", "estimate": {"signal_hat": [1]}},
                {"method": "B", "estimate": {"signal_hat": [2]}},
            ],
        },
        method_order=["A", "B"],
    )
    _merge_results_payload(
        out,
        {
            "video_path": "v",
            "fps": 20,
            "gt": [],
            "fs_gt": 32,
            "estimates": [
                {"method": "A", "estimate": {"signal_hat": [10]}},
            ],
        },
        method_order=["A"],
    )

    with open(out, "rb") as fp:
        data = pickle.load(fp)
    methods = [entry.get("method") for entry in data.get("estimates", [])]
    assert methods == ["A"]


def test_resolve_target_run_dirs_prefers_exact_label():
    tmp = tempfile.mkdtemp(prefix="run_dirs_")
    for name in ("cohface_robust_ossm", "cohface_robust_ossm0", "cohface_robust_ossm_COHFACE"):
        os.makedirs(os.path.join(tmp, name, "data"), exist_ok=True)

    exact = resolve_target_run_dirs(tmp, "cohface_robust_ossm")
    assert exact == [os.path.join(tmp, "cohface_robust_ossm")]


def test_resolve_target_run_dirs_fallback_to_multidataset_pattern():
    tmp = tempfile.mkdtemp(prefix="run_dirs_multi_")
    os.makedirs(os.path.join(tmp, "exp_COHFACE", "data"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "exp_MAHNOB", "data"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "exp0", "data"), exist_ok=True)

    resolved = sorted(resolve_target_run_dirs(tmp, "exp"))
    assert resolved == sorted([
        os.path.join(tmp, "exp_COHFACE"),
        os.path.join(tmp, "exp_MAHNOB"),
    ])


def test_duplicate_method_names_raise_hard_error():
    class _M:
        def __init__(self, name):
            self.name = name

    methods = [_M("profile1d_quadratic"), _M("profile1d_quadratic")]
    with pytest.raises(ValueError, match="Duplicate method names are not allowed"):
        _warn_duplicate_methods(methods)
