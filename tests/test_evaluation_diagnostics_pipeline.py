import os
import pickle
import tempfile
import json

import numpy as np
import pandas as pd
import pytest

from core.pipeline.evaluation_step import run_evaluation
from core.pipeline.visualize_step import run_visualization
from core.pipeline.common import update_frame_log_manifest


def _make_frame_log(path: str, n: int):
    fields = [
        "t", "v_t", "S_t", "nis", "lambda_t",
        "freq_std_hz",
        "fail_diverge", "fail_slip", "fail_lock", "fail_double",
        "q_vis", "q_drift", "alpha_R", "g_t", "g_z", "w_h",
    ]
    data = np.full((n, len(fields)), np.nan, dtype=np.float64)
    idx = {f: i for i, f in enumerate(fields)}
    t = np.arange(n, dtype=np.float64)
    data[:, idx["t"]] = t
    data[:, idx["v_t"]] = 0.1 * np.sin(2 * np.pi * t / 40.0)
    data[:, idx["S_t"]] = 0.2
    data[:, idx["nis"]] = (data[:, idx["v_t"]] ** 2) / data[:, idx["S_t"]]
    data[:, idx["lambda_t"]] = 1.0
    data[:, idx["freq_std_hz"]] = 0.02
    data[:, idx["fail_diverge"]] = 0.0
    data[:, idx["fail_slip"]] = 0.0
    data[:, idx["fail_lock"]] = 0.0
    data[:, idx["fail_double"]] = 0.0
    # Inject sparse synthetic events to validate overlay markers/CSV export.
    if n > 150:
        data[80, idx["fail_slip"]] = 1.0
        data[120, idx["fail_double"]] = 1.0
    data[:, idx["q_vis"]] = 0.7
    data[:, idx["q_drift"]] = 0.1
    data[:, idx["alpha_R"]] = 1.2
    data[:, idx["g_t"]] = 0.9
    data[:, idx["g_z"]] = 0.8
    data[:, idx["w_h"]] = 1.0
    np.savez_compressed(path, data=data, fields=np.asarray(fields, dtype=object))


def test_evaluation_writes_filter_diagnostics_and_visual_scatter():
    tmp = tempfile.mkdtemp(prefix="eval_diag_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    fs = 20.0
    n = 700  # > 30s window
    t = np.arange(n, dtype=np.float64) / fs
    gt = np.sin(2 * np.pi * 0.25 * t)
    est_sig = gt + 0.03 * np.sin(2 * np.pi * 0.35 * t)
    est_track = np.full(n, 0.25, dtype=np.float64)

    pkl_path = os.path.join(run_dir, "data", "cohface_1_0.pkl")
    payload = {
        "video_path": "/tmp/cohface_1_0.avi",
        "fps": fs,
        "gt": gt.astype(np.float32),
        "fs_gt": fs,
        "estimates": [
            {
                "method": "profile1d_quadratic__robust_ossm_ekf",
                "estimate": {
                    "signal_hat": est_sig.astype(np.float32),
                    "track_hz": est_track.astype(np.float32),
                    "rr_hz": float(np.mean(est_track)),
                    "rr_bpm": float(np.mean(est_track) * 60.0),
                    "meta": "{}",
                },
            }
        ],
    }
    with open(pkl_path, "wb") as fp:
        pickle.dump(payload, fp)

    log_dir = os.path.join(run_dir, "aux", "profile1d_quadratic__robust_ossm_ekf", "frame_logs")
    os.makedirs(log_dir, exist_ok=True)
    _make_frame_log(os.path.join(log_dir, "1_0.npz"), n=n)
    _make_frame_log(os.path.join(log_dir, "1_1.npz"), n=n)
    update_frame_log_manifest(
        aux_dir=os.path.join(run_dir, "aux", "profile1d_quadratic__robust_ossm_ekf"),
        base_trial_key="1_0",
        actual_filename="1_1.npz",
        suffix=1,
    )

    run_evaluation(
        tmp,
        run_label="testrun",
        win_size=30.0,
        stride=1.0,
        min_hz=0.08,
        max_hz=0.5,
        gating={"spectral": {"peak_ratio_min": 1.2, "prominence_min_db": 3.0}},
        eval_cfg={"win_size": 30.0, "stride": 1.0, "min_hz": 0.08, "max_hz": 0.5, "track_std_min_bpm": 0.8},
    )

    diag_csv = os.path.join(run_dir, "metrics", "metrics_filter_diagnostics_raw.csv")
    assert os.path.exists(diag_csv)
    ddf = pd.read_csv(diag_csv)
    required_cols = {"Fail_Total", "NIS_Mean", "Lambda_Mean", "Coverage95", "Stability_Sec"}
    assert required_cols.issubset(ddf.columns)
    assert ddf["method"].nunique() == 1

    freq_raw = os.path.join(run_dir, "metrics", "metrics_freq_domain_raw.csv")
    fdf = pd.read_csv(freq_raw)
    assert {"gt_bpm_avg", "est_bpm_avg"}.issubset(fdf.columns)
    with open(os.path.join(run_dir, "metrics", "eval_settings.json"), "r", encoding="utf-8") as fp:
        eval_settings = json.load(fp)
    assert "unused_config_keys" in eval_settings
    assert "eval.track_std_min_bpm" in list(eval_settings["unused_config_keys"])
    assert "gating.spectral.prominence_min_db" in list(eval_settings["unused_config_keys"])

    method_quality_csv = os.path.join(run_dir, "logs", "method_quality.csv")
    method_quality_json = os.path.join(run_dir, "logs", "method_quality_summary.json")
    assert os.path.exists(method_quality_csv)
    assert os.path.exists(method_quality_json)
    qdf = pd.read_csv(method_quality_csv)
    assert not qdf.empty
    assert {"method", "trial", "q_vis_mean", "alpha_R_mean", "lambda_mean", "R_post_mean"}.issubset(qdf.columns)

    run_visualization(tmp, run_label="testrun", win_size=30.0, stride=1.0, min_hz=0.08, max_hz=0.5)
    assert os.path.exists(os.path.join(run_dir, "plots", "summary_scatter_gt.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "filter_diagnostics_overview.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "filter_diagnostics_heatmap.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "latency_distribution.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "alignment_guide.txt"))
    assert os.path.exists(os.path.join(run_dir, "plots", "qrobf_diagnostics", "profile1d_quadratic__robust_ossm_ekf.png"))
    event_counts = os.path.join(run_dir, "plots", "qrobf_diagnostics", "profile1d_quadratic__robust_ossm_ekf_event_counts.csv")
    events = os.path.join(run_dir, "plots", "qrobf_diagnostics", "profile1d_quadratic__robust_ossm_ekf_events.csv")
    assert os.path.exists(event_counts)
    assert os.path.exists(events)
    assert os.path.exists(os.path.join(run_dir, "plots", "qrobf_diagnostics", "qrobf_event_summary.csv"))
    assert os.path.exists(os.path.join(run_dir, "plots", "qrobf_diagnostics", "qrobf_event_summary_rates.csv"))
    assert os.path.exists(os.path.join(run_dir, "plots", "qrobf_diagnostics", "qrobf_event_summary.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "qrobf_diagnostics", "qrobf_event_legend.txt"))
    qrobf_summary = pd.read_csv(os.path.join(run_dir, "plots", "qrobf_diagnostics", "qrobf_event_summary.csv"))
    assert int(qrobf_summary["n_trials"].iloc[0]) == 1
    assert int(qrobf_summary["n_frames"].iloc[0]) == n
    ec = pd.read_csv(event_counts)
    assert {"slip", "double"}.issubset(ec.columns)
    assert int(ec["slip"].iloc[0]) >= 1
    assert int(ec["double"].iloc[0]) >= 1


def test_evaluation_raises_on_malformed_frame_log():
    tmp = tempfile.mkdtemp(prefix="eval_malformed_log_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    fs = 20.0
    n = 700
    t = np.arange(n, dtype=np.float64) / fs
    gt = np.sin(2 * np.pi * 0.25 * t)
    pkl_path = os.path.join(run_dir, "data", "cohface_1_0.pkl")
    payload = {
        "video_path": "/tmp/cohface_1_0.avi",
        "fps": fs,
        "gt": gt.astype(np.float32),
        "fs_gt": fs,
        "estimates": [{
            "method": "profile1d_quadratic__robust_ossm_ekf",
            "estimate": {
                "signal_hat": gt.astype(np.float32),
                "track_hz": np.full(n, 0.25, dtype=np.float32),
                "rr_hz": 0.25,
                "rr_bpm": 15.0,
                "meta": "{}",
            },
        }],
    }
    with open(pkl_path, "wb") as fp:
        pickle.dump(payload, fp)

    log_dir = os.path.join(run_dir, "aux", "profile1d_quadratic__robust_ossm_ekf", "frame_logs")
    os.makedirs(log_dir, exist_ok=True)
    # malformed: missing required keys ['fields', 'data']
    np.savez_compressed(os.path.join(log_dir, "1_0.npz"), bad=np.array([1, 2, 3]))

    with pytest.raises(ValueError, match="Malformed frame log"):
        run_evaluation(tmp, run_label="testrun", win_size=30.0, stride=1.0, min_hz=0.08, max_hz=0.5)
