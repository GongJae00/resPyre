import json
import os
import pickle
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from analysis.run_innovation_eda import run_innovation_eda
from components.models.core.base import OscillatorParams, _BaseOscillatorHead
from core.pipeline.common import (
    resolve_frame_log_path,
    update_frame_log_manifest,
    resolve_frame_logs_for_run,
)
from core.pipeline.evaluation_step import run_evaluation


def _write_frame_log(path: str, q_vis_value: float, n: int = 700) -> None:
    fields = [
        "t", "z", "freq_hz",
        "q_vis", "q_drift", "q_cons", "q_out", "q_harm", "q_burst",
        "alpha_R", "alpha_Q", "g_t", "g_z", "w_h", "g_z_eff",
        "lambda_t", "qx_base", "qf_base", "rv_base",
        "qx_used", "qf_used", "rv_scaled", "R_post",
        "fail_diverge", "fail_slip", "fail_lock", "fail_double",
    ]
    idx = {k: i for i, k in enumerate(fields)}
    data = np.full((n, len(fields)), np.nan, dtype=np.float64)
    tt = np.arange(n, dtype=np.float64)
    data[:, idx["t"]] = tt
    data[:, idx["z"]] = np.log(0.25)
    data[:, idx["freq_hz"]] = 0.25
    data[:, idx["q_vis"]] = q_vis_value
    data[:, idx["q_drift"]] = 0.1
    data[:, idx["q_cons"]] = 1.0
    data[:, idx["q_out"]] = 0.0
    data[:, idx["q_harm"]] = 0.1
    data[:, idx["q_burst"]] = 0.0
    data[:, idx["alpha_R"]] = 1.1
    data[:, idx["alpha_Q"]] = 1.0
    data[:, idx["g_t"]] = 0.9
    data[:, idx["g_z"]] = 0.8
    data[:, idx["w_h"]] = 1.0
    data[:, idx["g_z_eff"]] = 0.8
    data[:, idx["lambda_t"]] = 1.0
    data[:, idx["qx_base"]] = 0.005
    data[:, idx["qf_base"]] = 0.005
    data[:, idx["rv_base"]] = 0.05
    data[:, idx["qx_used"]] = 0.005
    data[:, idx["qf_used"]] = 0.005
    data[:, idx["rv_scaled"]] = 0.05
    data[:, idx["R_post"]] = 0.05
    data[:, idx["fail_diverge"]] = 0.0
    data[:, idx["fail_slip"]] = 0.0
    data[:, idx["fail_lock"]] = 0.0
    data[:, idx["fail_double"]] = 0.0
    np.savez_compressed(path, data=data, fields=np.asarray(fields, dtype=object))


def test_resolve_frame_log_path_manifest_and_ambiguity_rules():
    tmp = tempfile.mkdtemp(prefix="manifest_rules_")
    aux_dir = os.path.join(tmp, "aux", "m")
    log_dir = os.path.join(aux_dir, "frame_logs")
    os.makedirs(log_dir, exist_ok=True)

    _write_frame_log(os.path.join(log_dir, "trialA.npz"), q_vis_value=0.1, n=20)
    _write_frame_log(os.path.join(log_dir, "trialA_1.npz"), q_vis_value=0.9, n=20)

    update_frame_log_manifest(aux_dir, "trialA", "trialA_1.npz", suffix=1)
    path, info = resolve_frame_log_path(aux_dir, "trialA", strict=True)
    assert path.endswith("trialA_1.npz")
    assert info["frame_log_resolution_mode"] == "manifest"
    assert int(info["frame_log_suffix_used"]) == 1

    os.remove(os.path.join(log_dir, "frame_logs_manifest.json"))
    with pytest.raises(ValueError, match="Ambiguous frame log candidates"):
        resolve_frame_log_path(aux_dir, "trialA", strict=True)

    path_ns, info_ns = resolve_frame_log_path(aux_dir, "trialA", strict=False)
    assert path_ns.endswith("trialA_1.npz")
    assert info_ns["frame_log_resolution_mode"] == "fallback_suffix"
    assert int(info_ns["frame_log_suffix_used"]) == 1


def test_evaluation_uses_manifest_latest_log_and_promotes_unused_keys():
    tmp = tempfile.mkdtemp(prefix="manifest_eval_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    fs = 20.0
    n = 700
    t = np.arange(n, dtype=np.float64) / fs
    gt = np.sin(2.0 * np.pi * 0.25 * t).astype(np.float32)

    pkl_path = os.path.join(run_dir, "data", "dummy_1_0.pkl")
    payload = {
        "video_path": "/tmp/dummy_1_0.avi",
        "fps": fs,
        "gt": gt,
        "fs_gt": fs,
        "estimates": [{
            "method": "profile1d_quadratic__robust_ossm_ekf",
            "estimate": {
                "signal_hat": gt,
                "track_hz": np.full(n, 0.25, dtype=np.float32),
                "rr_hz": 0.25,
                "rr_bpm": 15.0,
                "meta": {
                    "welch_df_hz": 0.0123,
                    "unused_config_keys": ["method.ensemble"],
                },
            },
        }],
    }
    with open(pkl_path, "wb") as fp:
        pickle.dump(payload, fp)

    aux_dir = os.path.join(run_dir, "aux", "profile1d_quadratic__robust_ossm_ekf")
    log_dir = os.path.join(aux_dir, "frame_logs")
    os.makedirs(log_dir, exist_ok=True)

    _write_frame_log(os.path.join(log_dir, "1_0.npz"), q_vis_value=0.1, n=n)
    status_path = os.path.join(run_dir, "run_status.json")
    with open(status_path, "w", encoding="utf-8") as fp:
        json.dump({
            "schema_version": "run_status.v1",
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_instance_started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }, fp)
    _write_frame_log(os.path.join(log_dir, "1_0_1.npz"), q_vis_value=0.9, n=n)

    run_evaluation(tmp, run_label="testrun", win_size=30.0, stride=1.0, min_hz=0.08, max_hz=0.5)

    qdf = pd.read_csv(os.path.join(run_dir, "logs", "method_quality.csv"))
    row = qdf[qdf["method"] == "profile1d_quadratic__robust_ossm_ekf"].iloc[0]
    assert row["frame_log_filename_used"] == "1_0_1.npz"
    assert row["frame_log_resolution_mode"] == "canonical_selected"
    assert abs(float(row["q_vis_mean"]) - 0.9) < 1e-6
    assert abs(float(row["welch_df_hz"]) - 0.0123) < 1e-9
    orphan_csv = os.path.join(run_dir, "logs", "method_quality_orphan_logs.csv")
    assert os.path.exists(orphan_csv)
    odf = pd.read_csv(orphan_csv)
    assert not odf.empty
    assert "pre_epoch_stale" in set(odf["reason"].astype(str))

    with open(os.path.join(run_dir, "logs", "config_usage.json"), "r", encoding="utf-8") as fp:
        cfg_usage = json.load(fp)
    assert "method_unused_config_keys" in cfg_usage
    assert "method.ensemble" in cfg_usage["method_unused_config_keys"]
    assert "method.ensemble" in cfg_usage["unused_config_keys"]


def test_resolve_frame_logs_for_run_epoch_selects_latest_and_marks_orphans():
    tmp = tempfile.mkdtemp(prefix="resolver_epoch_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "aux", "m", "frame_logs"), exist_ok=True)
    log_dir = os.path.join(run_dir, "aux", "m", "frame_logs")

    p0 = os.path.join(log_dir, "trial.npz")
    p1 = os.path.join(log_dir, "trial_1.npz")
    p2 = os.path.join(log_dir, "trial_2.npz")
    _write_frame_log(p0, q_vis_value=0.1, n=32)
    _write_frame_log(p1, q_vis_value=0.2, n=32)
    _write_frame_log(p2, q_vis_value=0.3, n=32)

    now = datetime.now(timezone.utc)
    run_epoch = now - timedelta(seconds=5)
    with open(os.path.join(run_dir, "run_status.json"), "w", encoding="utf-8") as fp:
        json.dump({
            "schema_version": "run_status.v1",
            "status": "running",
            "started_at": (now - timedelta(minutes=1)).isoformat().replace("+00:00", "Z"),
            "run_instance_started_at": run_epoch.isoformat().replace("+00:00", "Z"),
            "updated_at": now.isoformat().replace("+00:00", "Z"),
        }, fp)

    old_t = run_epoch.timestamp() - 30.0
    new_t1 = run_epoch.timestamp() + 1.0
    new_t2 = run_epoch.timestamp() + 2.0
    os.utime(p0, (old_t, old_t))
    os.utime(p1, (new_t1, new_t1))
    os.utime(p2, (new_t2, new_t2))

    selected, orphans, diag = resolve_frame_logs_for_run(
        run_dir,
        expected_trials=[{"method": "m", "trial": "trial"}],
        strict=True,
    )
    assert selected["m"]["trial"].endswith("trial_2.npz")
    reasons = {str(o.get("reason")) for o in orphans}
    assert "duplicate_suffix_ignored" in reasons
    assert "pre_epoch_stale" in reasons
    assert str(diag.get("selection_policy")) == "post_epoch_latest_mtime_suffix"


def test_innovation_eda_writes_summary_and_skips_when_all_logs_malformed():
    tmp = tempfile.mkdtemp(prefix="eda_missing_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)
    fs = 20.0
    n = 128
    pkl_path = os.path.join(run_dir, "data", "dummy_1_0.pkl")
    with open(pkl_path, "wb") as fp:
        pickle.dump({
            "video_path": "/tmp/dummy_1_0.avi",
            "fps": fs,
            "gt": np.zeros(n, dtype=np.float32),
            "fs_gt": fs,
            "estimates": [{
                "method": "bad_method",
                "estimate": {
                    "signal_hat": np.zeros(n, dtype=np.float32),
                    "track_hz": np.full(n, 0.25, dtype=np.float32),
                    "meta": {},
                },
            }],
        }, fp)
    with open(os.path.join(run_dir, "run_status.json"), "w", encoding="utf-8") as fp:
        json.dump({
            "schema_version": "run_status.v1",
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_instance_started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }, fp)

    bad_dir = os.path.join(run_dir, "aux", "bad_method", "frame_logs")
    os.makedirs(bad_dir, exist_ok=True)
    np.savez_compressed(os.path.join(bad_dir, "1_0.npz"), bad=np.array([1, 2, 3]))

    run_innovation_eda(tmp, run_label="testrun", allow_missing=False)
    skipped_path = os.path.join(run_dir, "eda", "innovation_eda_skipped_logs.json")
    summary_json = os.path.join(run_dir, "eda", "innovation_summary.json")
    summary_csv = os.path.join(run_dir, "eda", "innovation_summary.csv")
    assert os.path.exists(skipped_path)
    assert os.path.exists(summary_json)
    assert os.path.exists(summary_csv)
    with open(skipped_path, "r", encoding="utf-8") as fp:
        skipped = json.load(fp)
    assert int(skipped.get("skipped_count", -1)) >= 1
    with open(summary_json, "r", encoding="utf-8") as fp:
        summary = json.load(fp)
    assert summary.get("status") == "no_valid_logs"
    assert int(summary.get("n_valid", -1)) == 0


def test_post_smooth_alpha_isolation_no_inplace_leakage():
    head = _BaseOscillatorHead(OscillatorParams(post_smooth_alpha=0.3, qx=1e-4, qf=5e-5, rv_floor=0.03))
    alpha_before = float(head.params.post_smooth_alpha)

    eff_hi = head._effective_params(
        30.0,
        {
            "signal_std": 5.0,
            "signal_abs_mean": 0.5,
            "roi_intensity_std": 30.0,
            "roi_intensity_mean": 30.0,
        },
    )
    eff_lo = head._effective_params(
        30.0,
        {
            "signal_std": 0.1,
            "signal_abs_mean": 0.5,
            "roi_intensity_std": 0.1,
            "roi_intensity_mean": 30.0,
        },
    )

    assert float(head.params.post_smooth_alpha) == alpha_before
    assert float(eff_hi["post_smooth_alpha_used"]) != float(eff_lo["post_smooth_alpha_used"])

    track = np.array([0.1, 0.4, 0.2, 0.45, 0.25], dtype=np.float64)
    sm_hi = head._apply_post_smoothing(track, alpha_override=float(eff_hi["post_smooth_alpha_used"]))
    sm_lo = head._apply_post_smoothing(track, alpha_override=float(eff_lo["post_smooth_alpha_used"]))
    assert not np.allclose(sm_hi, sm_lo)
