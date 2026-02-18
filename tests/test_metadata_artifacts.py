import os
import tempfile
import json
import pickle
from pathlib import Path
import numpy as np

from core.pipeline.metadata_step import _default_artifacts, run_metadata_generation


def test_metadata_artifacts_detects_new_metrics_and_png_plots():
    tmp = tempfile.mkdtemp(prefix="meta_artifacts_")
    metrics_dir = os.path.join(tmp, "metrics")
    plots_dir = os.path.join(tmp, "plots", "summary")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    open(os.path.join(metrics_dir, "metrics_time_domain_summary.txt"), "w").close()
    open(os.path.join(metrics_dir, "metrics_freq_domain_summary.txt"), "w").close()
    open(os.path.join(metrics_dir, "metrics_time_domain.pkl"), "wb").close()
    open(os.path.join(metrics_dir, "metrics_freq_domain.pkl"), "wb").close()
    open(os.path.join(metrics_dir, "eval_settings.json"), "w").close()
    open(os.path.join(tmp, "run_status.json"), "w").close()
    open(os.path.join(plots_dir, "a.png"), "wb").close()
    open(os.path.join(plots_dir, "b.png"), "wb").close()

    artifacts = _default_artifacts(Path(tmp))
    assert "metrics_time_summary" in artifacts
    assert "metrics_freq_summary" in artifacts
    assert "metrics_time_pickle" in artifacts
    assert "metrics_freq_pickle" in artifacts
    assert artifacts.get("plots_count") == 2
    assert "run_status" in artifacts


def test_metadata_generation_includes_method_quality_artifacts():
    tmp = tempfile.mkdtemp(prefix="meta_method_quality_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    with open(os.path.join(run_dir, "logs", "method_quality.csv"), "w", encoding="utf-8") as fp:
        fp.write("method,trial\\n")
    with open(os.path.join(run_dir, "logs", "method_quality_summary.json"), "w", encoding="utf-8") as fp:
        json.dump({"schema_version": "method_quality.v1"}, fp)
    with open(os.path.join(run_dir, "logs", "config_usage.json"), "w", encoding="utf-8") as fp:
        json.dump({"top_level_unused_keys": ["runtime"]}, fp)

    run_metadata_generation(tmp, run_label="testrun", allow_incomplete=True)
    with open(os.path.join(run_dir, "metadata.json"), "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    artifacts = meta.get("artifacts", {})
    assert "method_quality_csv" in artifacts
    assert "method_quality_json" in artifacts
    assert "config_usage_json" in artifacts
    assert "top.runtime" in meta.get("unused_config_keys", [])


def test_metadata_promotes_method_level_unused_keys_from_estimates():
    tmp = tempfile.mkdtemp(prefix="meta_unused_method_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    payload = {
        "video_path": "/tmp/dummy.avi",
        "fps": 20.0,
        "gt": np.zeros(200, dtype=np.float32),
        "fs_gt": 20.0,
        "estimates": [{
            "method": "profile1d_linear__robust_ossm_ekf",
            "estimate": {
                "signal_hat": np.zeros(200, dtype=np.float32),
                "track_hz": np.full(200, 0.25, dtype=np.float32),
                "rr_hz": 0.25,
                "rr_bpm": 15.0,
                "meta": {"unused_config_keys": ["method.ensemble"]},
            },
        }],
    }
    with open(os.path.join(run_dir, "data", "dummy_1_0.pkl"), "wb") as fp:
        pickle.dump(payload, fp)
    with open(os.path.join(run_dir, "logs", "config_usage.json"), "w", encoding="utf-8") as fp:
        json.dump({"unused_config_keys": []}, fp)

    run_metadata_generation(tmp, run_label="testrun", allow_incomplete=True)
    with open(os.path.join(run_dir, "metadata.json"), "r", encoding="utf-8") as fp:
        meta = json.load(fp)
    assert "method.ensemble" in meta.get("unused_config_keys", [])
