import os
import pickle
import tempfile
import json

import numpy as np
import pandas as pd
import pytest

from core.pipeline.evaluation_step import run_evaluation, _compute_filter_diag_record
from core.pipeline.visualize_step import run_visualization, _save_trust_failure_overlays
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
        frame_log_strict=False,
    )

    diag_csv = os.path.join(run_dir, "metrics", "metrics_filter_diagnostics_raw.csv")
    assert os.path.exists(diag_csv)
    assert os.path.exists(os.path.join(run_dir, "metrics", "metrics_filter_calibration_split.csv"))
    assert os.path.exists(os.path.join(run_dir, "metrics", "metrics_filter_calibration_split_all.csv"))
    assert os.path.exists(os.path.join(run_dir, "metrics", "metrics_filter_calibration_split.json"))
    assert os.path.exists(os.path.join(run_dir, "metrics", "metrics_filter_calibration_split.txt"))
    resolver_diag_json = os.path.join(run_dir, "metrics", "resolver_diag.json")
    assert os.path.exists(resolver_diag_json)
    with open(resolver_diag_json, "r", encoding="utf-8") as fp:
        resolver_diag = json.load(fp)
    assert resolver_diag.get("schema_version") == "frame_log_resolver_diag.v1"
    assert "resolver" in resolver_diag
    ddf = pd.read_csv(diag_csv)
    required_cols = {"Fail_Total", "NIS_Mean", "Lambda_Mean", "Coverage95", "Stability_Sec"}
    assert required_cols.issubset(ddf.columns)
    assert {"NIS_Pass_Relaxed", "NIS_OverStrict", "NIS_TrueFail", "NIS_Mean_DevAbs", "NIS_InBand_DevAbs"}.issubset(ddf.columns)
    assert ddf["method"].nunique() == 1
    split_df = pd.read_csv(os.path.join(run_dir, "metrics", "metrics_filter_calibration_split.csv"))
    assert not split_df.empty
    assert split_df["method"].nunique() == 1
    assert "calibration_applicable" in split_df.columns
    assert bool(split_df["calibration_applicable"].iloc[0]) is True

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

    run_visualization(
        tmp,
        run_label="testrun",
        win_size=30.0,
        stride=1.0,
        min_hz=0.08,
        max_hz=0.5,
        frame_log_strict=False,
    )
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
    with open(os.path.join(run_dir, "plots", "qrobf_diagnostics", "qrobf_event_summary.json"), "r", encoding="utf-8") as fp:
        qrobf_json = json.load(fp)
    assert int(qrobf_json["resolver_diag"]["counts"]["n_expected_trials"]) == 1
    assert int(qrobf_json["resolver_diag"]["counts"]["n_missing_trials"]) == 0
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


def test_qrobf_placeholder_artifacts_when_no_canonical_logs():
    tmp = tempfile.mkdtemp(prefix="qrobf_empty_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    _save_trust_failure_overlays(run_dir, frame_log_strict=False)

    out_dir = os.path.join(run_dir, "plots", "qrobf_diagnostics")
    assert os.path.exists(os.path.join(out_dir, "qrobf_orphan_logs.csv"))
    assert os.path.exists(os.path.join(out_dir, "qrobf_event_summary.csv"))
    assert os.path.exists(os.path.join(out_dir, "qrobf_event_summary_rates.csv"))
    assert os.path.exists(os.path.join(out_dir, "qrobf_event_summary.json"))
    assert os.path.exists(os.path.join(out_dir, "qrobf_event_summary.png"))
    sdf = pd.read_csv(os.path.join(out_dir, "qrobf_event_summary.csv"))
    assert sdf.empty
    with open(os.path.join(out_dir, "qrobf_event_summary.json"), "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    assert payload.get("status") == "empty"
    assert payload.get("reason") == "no_canonical_logs"


def test_calibration_split_overstrict_vs_truefail():
    n = 256
    fields = np.asarray(["nis", "lambda_t", "fail_diverge", "fail_slip", "fail_lock", "fail_double"], dtype=object)
    idx = {f: i for i, f in enumerate(fields)}

    # Case A: strict fail (mean != 1) but relaxed pass (small deviation, in-band high).
    data_a = np.zeros((n, len(fields)), dtype=np.float64)
    data_a[:, idx["nis"]] = 0.8
    data_a[:, idx["lambda_t"]] = 1.0
    log_a = {"data": data_a, "idx": idx}
    rec_a = _compute_filter_diag_record(
        log_obj=log_a,
        method_name="m",
        fname="x",
        data_file_rel="x.pkl",
        fps=20.0,
        est_track_hz=None,
        gt_inst_hz=None,
        calibration_policy={
            "dof": 1.0,
            "alpha_strict": 0.05,
            "inband_target": 0.95,
            "mean_tol_relaxed": 0.35,
            "inband_tol_relaxed": 0.08,
        },
    )
    assert rec_a["NIS_Pass"] == 0.0
    assert rec_a["NIS_Pass_Relaxed"] == 1.0
    assert rec_a["NIS_OverStrict"] == 1.0
    assert rec_a["NIS_TrueFail"] == 0.0

    # Case B: both strict and relaxed fail -> true miscalibration.
    data_b = np.zeros((n, len(fields)), dtype=np.float64)
    data_b[:, idx["nis"]] = 10.0
    data_b[:, idx["lambda_t"]] = 1.0
    log_b = {"data": data_b, "idx": idx}
    rec_b = _compute_filter_diag_record(
        log_obj=log_b,
        method_name="m",
        fname="x",
        data_file_rel="x.pkl",
        fps=20.0,
        est_track_hz=None,
        gt_inst_hz=None,
        calibration_policy={
            "dof": 1.0,
            "alpha_strict": 0.05,
            "inband_target": 0.95,
            "mean_tol_relaxed": 0.35,
            "inband_tol_relaxed": 0.08,
        },
    )
    assert rec_b["NIS_Pass"] == 0.0
    assert rec_b["NIS_Pass_Relaxed"] == 0.0
    assert rec_b["NIS_OverStrict"] == 0.0
    assert rec_b["NIS_TrueFail"] == 1.0


def test_calibration_split_separates_applicable_and_non_applicable():
    tmp = tempfile.mkdtemp(prefix="eval_calib_split_applicable_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    fs = 20.0
    n = 700
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
            },
            {
                "method": "profile1D quadratic",
                "estimate": {
                    "signal_hat": est_sig.astype(np.float32),
                    "track_hz": est_track.astype(np.float32),
                    "rr_hz": float(np.mean(est_track)),
                    "rr_bpm": float(np.mean(est_track) * 60.0),
                    "meta": "{}",
                },
            },
        ],
    }
    with open(pkl_path, "wb") as fp:
        pickle.dump(payload, fp)

    # Only robust method has frame-log diagnostics.
    log_dir = os.path.join(run_dir, "aux", "profile1d_quadratic__robust_ossm_ekf", "frame_logs")
    os.makedirs(log_dir, exist_ok=True)
    _make_frame_log(os.path.join(log_dir, "1_0.npz"), n=n)

    run_evaluation(
        tmp,
        run_label="testrun",
        win_size=30.0,
        stride=1.0,
        min_hz=0.08,
        max_hz=0.5,
        frame_log_strict=False,
    )

    split = pd.read_csv(os.path.join(run_dir, "metrics", "metrics_filter_calibration_split.csv"))
    split_all = pd.read_csv(os.path.join(run_dir, "metrics", "metrics_filter_calibration_split_all.csv"))
    assert "calibration_applicable" in split_all.columns
    assert split_all["method"].nunique() == 2
    non_rob = split_all.loc[split_all["method"] == "profile1D quadratic"].iloc[0]
    assert bool(non_rob["calibration_applicable"]) is False
    assert non_rob["applicability_reason"] == "no_finite_nis_metrics"
    assert split["method"].nunique() == 1
    assert set(split["method"]) == {"profile1d_quadratic__robust_ossm_ekf"}
