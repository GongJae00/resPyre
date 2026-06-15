import glob
import json
import os
import pickle
import tempfile
from typing import Dict

import numpy as np
import pandas as pd

from core.pipeline.runner import extract_respiration
from core.pipeline.wrapped_method import create_wrapped_method
from core.pipeline.evaluation_step import run_evaluation
from core.pipeline.visualize_step import run_visualization
from core.pipeline.common import update_frame_log_manifest


def _make_signal_bundle(fs: float, n: int, f_hz: float = 0.25) -> Dict[str, np.ndarray]:
    t = np.arange(n, dtype=np.float64) / fs
    gt = np.sin(2.0 * np.pi * f_hz * t)
    est = gt + 0.02 * np.sin(2.0 * np.pi * 0.37 * t)
    track = np.full(n, f_hz, dtype=np.float64)
    return {"gt": gt.astype(np.float32), "est": est.astype(np.float32), "track": track.astype(np.float32)}


def _write_frame_log(path: str, n: int) -> None:
    fields = [
        "t", "y_t", "y_pred", "v_t", "nis", "lambda_t",
        "alpha_R", "alpha_Q", "g_t", "g_z", "w_h", "g_z_eff",
        "q_vis", "q_drift", "q_cons", "q_out", "q_harm", "q_burst",
        "z", "freq_hz", "R_eff", "R_post", "qx_base", "qf_base", "rv_base",
        "qx_used", "qf_used", "rv_scaled",
        "fail_diverge", "fail_slip", "fail_lock", "fail_double",
    ]
    idx = {k: i for i, k in enumerate(fields)}
    data = np.full((n, len(fields)), np.nan, dtype=np.float64)
    tt = np.arange(n, dtype=np.float64)
    data[:, idx["t"]] = tt
    data[:, idx["y_t"]] = np.sin(2.0 * np.pi * 0.25 * tt / 20.0)
    data[:, idx["y_pred"]] = 0.9 * data[:, idx["y_t"]]
    data[:, idx["v_t"]] = data[:, idx["y_t"]] - data[:, idx["y_pred"]]
    data[:, idx["nis"]] = 0.4
    data[:, idx["lambda_t"]] = 1.0
    data[:, idx["alpha_R"]] = 1.1
    data[:, idx["alpha_Q"]] = 1.0
    data[:, idx["g_t"]] = 0.9
    data[:, idx["g_z"]] = 0.8
    data[:, idx["w_h"]] = 1.0
    data[:, idx["g_z_eff"]] = 0.8
    data[:, idx["q_vis"]] = 0.7
    data[:, idx["q_drift"]] = 0.2
    data[:, idx["q_cons"]] = 1.0
    data[:, idx["q_out"]] = 0.0
    data[:, idx["q_harm"]] = 0.1
    data[:, idx["q_burst"]] = 0.0
    data[:, idx["z"]] = np.log(0.25)
    data[:, idx["freq_hz"]] = 0.25
    data[:, idx["R_eff"]] = 0.05
    data[:, idx["R_post"]] = 0.05
    data[:, idx["qx_base"]] = 0.005
    data[:, idx["qf_base"]] = 0.005
    data[:, idx["rv_base"]] = 0.05
    data[:, idx["qx_used"]] = 0.005
    data[:, idx["qf_used"]] = 0.005
    data[:, idx["rv_scaled"]] = 0.05
    data[:, idx["fail_diverge"]] = 0.0
    data[:, idx["fail_slip"]] = 0.0
    data[:, idx["fail_lock"]] = 0.0
    data[:, idx["fail_double"]] = 0.0
    np.savez_compressed(path, data=data, fields=np.asarray(fields, dtype=object))


def test_trial_key_propagation(monkeypatch):
    monkeypatch.setattr("core.pipeline.runner.get_vid_stats", lambda _: (240, 20.0))

    class DummyDataset:
        name = "dummy"
        fs_gt = 20.0

        def __init__(self):
            roi = np.ones((8, 8, 3), dtype=np.uint8)
            self.data = [
                {
                    "video_path": "/tmp/subj1_t1.avi",
                    "subject": "subj1",
                    "trial": "t1",
                    "gt": np.zeros(240, dtype=np.float32),
                    "chest_rois": [roi.copy() for _ in range(240)],
                    "face_rois": [],
                },
                {
                    "video_path": "/tmp/subj2_t1.avi",
                    "subject": "subj2",
                    "trial": "t1",
                    "gt": np.zeros(240, dtype=np.float32),
                    "chest_rois": [roi.copy() for _ in range(240)],
                    "face_rois": [],
                },
            ]

        def load_dataset(self):
            return None

        def extract_ROI(self, *_args, **_kwargs):
            return []

    method = create_wrapped_method(
        "profile1d_linear__narossm",
        params={"params": {"fs": 20.0, "f_min": 0.08, "f_max": 0.5, "trace_cap": 50.0}},
    )
    out_root = tempfile.mkdtemp(prefix="trial_key_prop_")
    extract_respiration(
        datasets=[DummyDataset()],
        methods=[method],
        results_dir=out_root,
        run_label="trial_key_run",
    )

    frame_dir = os.path.join(
        out_root,
        "trial_key_run",
        "aux",
        "profile1d_linear__narossm",
        "frame_logs",
    )
    logs = sorted(glob.glob(os.path.join(frame_dir, "*.npz")))
    stems = [os.path.splitext(os.path.basename(p))[0] for p in logs]
    assert len(logs) >= 2
    assert len(set(stems)) >= 2


def test_create_wrapped_method_applies_global_oscillator_defaults():
    method = create_wrapped_method(
        "profile1d_linear__narossm",
        params={"params": {"oscillator": {"qx": 1e-4}}},
        oscillator_defaults={"qx": 0.005, "qf": 2e-5, "no_autotune": True, "em_mode": "off"},
    )
    assert abs(float(method.osc_head.params.qx) - 1e-4) < 1e-12
    assert abs(float(method.osc_head.params.qf) - 2e-5) < 1e-12
    assert bool(method.osc_head.params.no_autotune) is True
    assert str(method.osc_head.params.em_mode) == "off"


def test_method_quality_artifacts_and_missing_rows():
    tmp = tempfile.mkdtemp(prefix="method_quality_missing_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    fs = 20.0
    n = 700
    bundle = _make_signal_bundle(fs, n, f_hz=0.25)
    for subj in ("1", "2"):
        pkl_path = os.path.join(run_dir, "data", f"dummy_{subj}_0.pkl")
        payload = {
            "video_path": f"/tmp/dummy_{subj}_0.avi",
            "fps": fs,
            "gt": bundle["gt"],
            "fs_gt": fs,
            "estimates": [{
                "method": "profile1d_quadratic__robust_ossm_ekf",
                "estimate": {
                    "signal_hat": bundle["est"],
                    "track_hz": bundle["track"],
                    "rr_hz": float(np.mean(bundle["track"])),
                    "rr_bpm": float(np.mean(bundle["track"]) * 60.0),
                    "meta": "{}",
                },
            }],
        }
        with open(pkl_path, "wb") as fp:
            pickle.dump(payload, fp)

    # Only one trial gets frame log -> the other must be explicit missing row.
    log_dir = os.path.join(run_dir, "aux", "profile1d_quadratic__robust_ossm_ekf", "frame_logs")
    os.makedirs(log_dir, exist_ok=True)
    _write_frame_log(os.path.join(log_dir, "1_0.npz"), n)

    run_evaluation(
        tmp,
        run_label="testrun",
        win_size=30.0,
        stride=1.0,
        min_hz=0.08,
        max_hz=0.5,
        gating={"spectral": {"peak_ratio_min": 1.2}},
        eval_cfg={"win_size": 30.0, "stride": 1.0, "min_hz": 0.08, "max_hz": 0.5, "extra_unused": 1},
        gating_scope="evaluation_only",
    )

    q_csv = os.path.join(run_dir, "logs", "method_quality.csv")
    q_json = os.path.join(run_dir, "logs", "method_quality_summary.json")
    cfg_usage = os.path.join(run_dir, "logs", "config_usage.json")
    assert os.path.exists(q_csv)
    assert os.path.exists(q_json)
    assert os.path.exists(cfg_usage)

    qdf = pd.read_csv(q_csv)
    assert "missing_frame_log" in qdf.columns
    assert int(qdf["missing_frame_log"].fillna(False).astype(bool).sum()) >= 1

    with open(q_json, "r", encoding="utf-8") as fp:
        summary = json.load(fp)
    assert summary["n_trials_total"] >= 2
    assert summary["n_trials_missing_logs"] >= 1
    assert summary["n_trials_with_logs"] >= 1


def test_config_fingerprint_stability():
    tmp = tempfile.mkdtemp(prefix="cfg_fp_stability_")
    fs = 20.0
    n = 700

    def _mk_run(run_name: str, f_hz: float):
        run_dir = os.path.join(tmp, run_name)
        os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
        bundle = _make_signal_bundle(fs, n, f_hz=f_hz)
        pkl_path = os.path.join(run_dir, "data", f"{run_name}_1_0.pkl")
        with open(pkl_path, "wb") as fp:
            pickle.dump(
                {
                    "video_path": f"/tmp/{run_name}_1_0.avi",
                    "fps": fs,
                    "gt": bundle["gt"],
                    "fs_gt": fs,
                    "estimates": [{
                        "method": "profile1d_quadratic__robust_ossm_ekf",
                        "estimate": {
                            "signal_hat": bundle["est"],
                            "track_hz": bundle["track"],
                            "rr_hz": float(np.mean(bundle["track"])),
                            "rr_bpm": float(np.mean(bundle["track"]) * 60.0),
                            "meta": "{}",
                        },
                    }],
                },
                fp,
            )
        log_dir = os.path.join(run_dir, "aux", "profile1d_quadratic__robust_ossm_ekf", "frame_logs")
        os.makedirs(log_dir, exist_ok=True)
        _write_frame_log(os.path.join(log_dir, "1_0.npz"), n)
        return run_dir

    run_a = _mk_run("runa", 0.25)
    run_b = _mk_run("runb", 0.29)

    common_kwargs = dict(
        win_size=30.0,
        stride=1.0,
        min_hz=0.08,
        max_hz=0.5,
        gating={"spectral": {"peak_ratio_min": 1.2}},
        eval_cfg={"win_size": 30.0, "stride": 1.0, "min_hz": 0.08, "max_hz": 0.5},
        gating_scope="evaluation_only",
    )
    run_evaluation(tmp, run_label="runa", **common_kwargs)
    run_evaluation(tmp, run_label="runb", **common_kwargs)

    with open(os.path.join(run_a, "logs", "method_quality_summary.json"), "r", encoding="utf-8") as fp:
        a = json.load(fp)
    with open(os.path.join(run_b, "logs", "method_quality_summary.json"), "r", encoding="utf-8") as fp:
        b = json.load(fp)
    assert a["config_fingerprint"] == b["config_fingerprint"]


def test_event_summary_aggregate():
    tmp = tempfile.mkdtemp(prefix="event_summary_agg_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    fs = 20.0
    n = 700
    bundle = _make_signal_bundle(fs, n, f_hz=0.25)
    pkl_path = os.path.join(run_dir, "data", "dummy_1_0.pkl")
    with open(pkl_path, "wb") as fp:
        pickle.dump(
            {
                "video_path": "/tmp/dummy_1_0.avi",
                "fps": fs,
                "gt": bundle["gt"],
                "fs_gt": fs,
                "estimates": [{
                    "method": "profile1d_quadratic__robust_ossm_ekf",
                    "estimate": {
                        "signal_hat": bundle["est"],
                        "track_hz": bundle["track"],
                        "rr_hz": float(np.mean(bundle["track"])),
                        "rr_bpm": float(np.mean(bundle["track"]) * 60.0),
                        "meta": "{}",
                    },
                }],
            },
            fp,
        )

    log_dir = os.path.join(run_dir, "aux", "profile1d_quadratic__robust_ossm_ekf", "frame_logs")
    os.makedirs(log_dir, exist_ok=True)
    _write_frame_log(os.path.join(log_dir, "1_0.npz"), n)
    _write_frame_log(os.path.join(log_dir, "1_0_1.npz"), n)
    update_frame_log_manifest(
        aux_dir=os.path.join(run_dir, "aux", "profile1d_quadratic__robust_ossm_ekf"),
        base_trial_key="1_0",
        actual_filename="1_0_1.npz",
        suffix=1,
    )

    run_evaluation(
        tmp,
        run_label="testrun",
        win_size=30.0,
        stride=1.0,
        min_hz=0.08,
        max_hz=0.5,
        frame_log_strict=False,
    )
    run_visualization(
        tmp,
        run_label="testrun",
        win_size=30.0,
        stride=1.0,
        min_hz=0.08,
        max_hz=0.5,
        frame_log_strict=False,
    )

    summary_csv = os.path.join(run_dir, "plots", "qrobf_diagnostics", "qrobf_event_summary.csv")
    assert os.path.exists(summary_csv)
    sdf = pd.read_csv(summary_csv)
    assert int(sdf["n_trials"].iloc[0]) == 1
