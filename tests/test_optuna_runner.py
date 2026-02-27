import csv
import json
import tempfile
from pathlib import Path

import optuna

from core.optimization.run_optuna import (
    DEFAULT_PARAM_SPACE,
    MethodStudy,
    StudyArgs,
    discover_tunable_methods,
    compute_objective,
    collect_trial_metrics,
    export_best_preset,
    normalize_objective_terms,
)


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _spec_map(family: str):
    return {spec.path: spec for spec in DEFAULT_PARAM_SPACE[family]}


def test_discover_tunable_methods_prefers_robust_by_default():
    entries = {
        "profile1d_cubic__kfstd": {"name": "profile1d_cubic__kfstd"},
        "profile1d_cubic__robust_ossm_ekf": {"name": "profile1d_cubic__robust_ossm_ekf"},
        "profile1d_cubic__robust_ossm_ukf": {"name": "profile1d_cubic__robust_ossm_ukf"},
    }

    selected = discover_tunable_methods(entries)
    assert set(selected) == {
        "profile1d_cubic__robust_ossm_ekf",
        "profile1d_cubic__robust_ossm_ukf",
    }

    selected_kf = discover_tunable_methods(entries, families=["kfstd"])
    assert selected_kf == ["profile1d_cubic__kfstd"]


def test_robust_search_space_covers_effective_runtime_controls():
    required_common = {
        "oscillator.qx",
        "oscillator.qf",
        "oscillator.rv_floor",
        "oscillator.rv_mad_scale",
        "oscillator.tau_env",
        "oscillator.init_margin_hz",
        "oscillator.student_t_nu",
        "oscillator.vb_iters",
        "oscillator.trace_cap",
        "oscillator.lambda_floor",
        "oscillator.r_eff_max_scale",
        "oscillator.g_z_eff_floor_ratio",
        "oscillator.post_smooth_alpha",
        "oscillator.rv_auto",
        "oscillator.rv",
        "oscillator.detrend",
        "oscillator.bandpass",
        "oscillator.zscore",
        "oscillator.spec_guidance_strength",
        "oscillator.spec_guidance_offset",
        "oscillator.spec_guidance_confidence_scale",
        "oscillator.spec_guidance_snr_scale",
        "trust.beta_1",
        "trust.beta_2",
        "trust.gamma_1",
        "trust.w_gate_vis",
        "trust.w_gate_cons",
        "trust.w_gate_nis",
        "trust.gate_bias",
        "trust.freq_jitter_decay",
        "trust.thd_max",
        "trust.w_h_min",
        "trust.g_z_floor",
        "trust.nis_hard_gate",
        "trust.alpha_R_max",
        "trust.alpha_Q_max",
        "quality.vis_eps",
        "quality.vis_snr_low_db",
        "quality.vis_snr_high_db",
        "quality.vis_blend_contrast",
        "quality.vis_blend_snr",
        "quality.vis_blend_valid",
        "quality.drift_scale",
        "quality.cons_window",
        "quality.hampel_k",
        "quality.hampel_thresh",
        "quality.harm_window_sec",
        "quality.harm_harmonics",
        "quality.burst_sigma",
        "quality.burst_window",
        "preproc.robust_zscore.enabled",
        "preproc.robust_zscore.clip",
        "preproc.robust_zscore.eps",
        "preproc.sign_align.enabled",
        "preproc.sign_align.seconds",
    }
    required_ukf = {"oscillator.ukf_alpha", "oscillator.ukf_beta", "oscillator.ukf_kappa"}

    ekf_paths = set(_spec_map("robust_ossm_ekf").keys())
    ukf_paths = set(_spec_map("robust_ossm_ukf").keys())

    assert required_common.issubset(ekf_paths)
    assert required_common.issubset(ukf_paths)
    assert required_ukf.issubset(ukf_paths)


def test_robust_search_space_has_conservative_bounds():
    ekf_specs = _spec_map("robust_ossm_ekf")
    ukf_specs = _spec_map("robust_ossm_ukf")

    assert float(ekf_specs["oscillator.qf"].high) <= 1.5e-3
    assert float(ukf_specs["oscillator.qf"].high) <= 1.5e-3
    assert float(ekf_specs["quality.harm_window_sec"].high) <= 8.5
    assert float(ukf_specs["quality.harm_window_sec"].high) <= 8.5
    assert float(ekf_specs["trust.alpha_R_max"].high) <= 50.0
    assert float(ukf_specs["trust.alpha_R_max"].high) <= 50.0
    assert float(ekf_specs["oscillator.r_eff_max_scale"].high) <= 150.0
    assert float(ukf_specs["oscillator.r_eff_max_scale"].high) <= 150.0
    assert list(ekf_specs["preproc.robust_zscore.enabled"].choices) == [True]
    assert list(ukf_specs["preproc.robust_zscore.enabled"].choices) == [True]


def test_compute_objective_penalizes_missing_metrics():
    weights = {
        "time_mae": 0.3,
        "freq_mae": 0.3,
        "fail_rate": 0.2,
        "invalid_rate": 0.1,
        "clip_rate": 0.1,
        "ccc_penalty": 0.0,
        "snr_penalty": 0.0,
        "nis_truefail": 0.0,
    }

    obj_good, _ = compute_objective(
        {
            "time_mae": 0.2,
            "freq_mae": 0.3,
            "fail_rate": 0.1,
            "invalid_rate": 0.02,
            "clip_rate": 0.03,
            "time_ccc": 0.9,
            "freq_snr": 8.0,
            "nis_truefail": 0.0,
        },
        weights,
    )
    obj_bad, _ = compute_objective(
        {
            "time_mae": float("nan"),
            "freq_mae": float("nan"),
            "fail_rate": float("nan"),
            "invalid_rate": float("nan"),
            "clip_rate": float("nan"),
        },
        weights,
    )
    assert obj_bad > obj_good


def test_compute_objective_prefers_calibrated_trial_when_calibration_weighted():
    weights = {
        "time_mae": 0.10,
        "freq_mae": 0.10,
        "fail_rate": 0.15,
        "nis_truefail": 0.25,
        "coverage_dev": 0.25,
        "constraint_penalty": 0.15,
    }
    constraints = {
        "fail_rate_max": 0.10,
        "invalid_rate_max": 0.03,
        "clip_rate_max": 0.20,
        "nis_truefail_max": 0.10,
        "coverage95_min": 0.90,
        "coverage95_target": 0.95,
    }

    good_metrics = {
        "time_mae": 0.35,
        "freq_mae": 0.30,
        "fail_rate": 0.05,
        "invalid_rate": 0.01,
        "clip_rate": 0.02,
        "time_ccc": 0.85,
        "freq_snr": 8.0,
        "nis_truefail": 0.03,
        "nis_overstrict": 0.08,
        "coverage95": 0.94,
    }
    bad_metrics = {
        "time_mae": 0.34,   # slightly better MAE but much worse calibration
        "freq_mae": 0.29,
        "fail_rate": 0.16,
        "invalid_rate": 0.08,
        "clip_rate": 0.25,
        "time_ccc": 0.82,
        "freq_snr": 8.5,
        "nis_truefail": 0.40,
        "nis_overstrict": 0.10,
        "coverage95": 0.62,
    }

    obj_good, terms_good = compute_objective(good_metrics, weights, constraints)
    obj_bad, terms_bad = compute_objective(bad_metrics, weights, constraints)

    assert terms_good["constraint_penalty"] == 0.0
    assert terms_bad["constraint_penalty"] > 0.0
    assert obj_bad > obj_good


def test_compute_objective_penalizes_lock_and_double_when_weighted():
    weights = {
        "time_mae": 0.0,
        "freq_mae": 0.0,
        "fail_lock": 0.5,
        "fail_double": 0.5,
    }
    constraints = {
        "fail_lock_max": 0.55,
        "fail_double_max": 0.18,
    }
    good_metrics = {"fail_lock": 0.40, "fail_double": 0.10}
    bad_metrics = {"fail_lock": 0.70, "fail_double": 0.30}

    obj_good, terms_good = compute_objective(good_metrics, weights, constraints)
    obj_bad, terms_bad = compute_objective(bad_metrics, weights, constraints)

    assert terms_good["constraint_penalty"] == 0.0
    assert terms_bad["constraint_penalty"] > 0.0
    assert obj_bad > obj_good


def test_compute_objective_penalizes_worse_than_kfstd_reference():
    weights = {
        "time_mae": 0.0,
        "freq_mae": 0.0,
        "vs_kfstd_time_mae": 0.5,
        "vs_kfstd_freq_mae": 0.5,
    }
    constraints = {
        "vs_kfstd_time_mae_max": 0.0,
        "vs_kfstd_freq_mae_max": 0.0,
    }
    better = {
        "time_mae": 0.34,
        "freq_mae": 0.20,
        "kfstd_time_mae_ref": 0.36,
        "kfstd_freq_mae_ref": 0.205,
    }
    worse = {
        "time_mae": 0.42,
        "freq_mae": 0.29,
        "kfstd_time_mae_ref": 0.36,
        "kfstd_freq_mae_ref": 0.205,
    }
    obj_better, terms_better = compute_objective(better, weights, constraints)
    obj_worse, terms_worse = compute_objective(worse, weights, constraints)
    assert terms_better["vs_kfstd_time_mae"] == 0.0
    assert terms_better["vs_kfstd_freq_mae"] == 0.0
    assert terms_worse["vs_kfstd_time_mae"] > 0.0
    assert terms_worse["vs_kfstd_freq_mae"] > 0.0
    assert terms_worse["constraint_penalty"] > terms_better["constraint_penalty"]
    assert obj_worse > obj_better


def test_compute_objective_normalizes_coverage_percent_scale():
    weights = {"coverage_dev": 1.0}
    constraints = {"coverage95_target": 0.95}

    obj_ratio, terms_ratio = compute_objective(
        {"coverage95": 0.95},
        weights,
        constraints,
    )
    obj_pct, terms_pct = compute_objective(
        {"coverage95": 95.0},
        weights,
        constraints,
    )

    assert abs(terms_ratio["coverage_dev"] - terms_pct["coverage_dev"]) < 1e-12
    assert abs(obj_ratio - obj_pct) < 1e-12
    assert terms_pct["coverage_dev"] < 1e-12


def test_constraint_penalty_normalizes_coverage_percent_scale():
    weights = {"constraint_penalty": 1.0}
    constraints = {"coverage95_min": 0.90}

    obj_ratio, terms_ratio = compute_objective(
        {"coverage95": 0.85},
        weights,
        constraints,
    )
    obj_pct, terms_pct = compute_objective(
        {"coverage95": 85.0},
        weights,
        constraints,
    )

    assert terms_ratio["constraint_penalty"] > 0.0
    assert terms_pct["constraint_penalty"] > 0.0
    assert abs(terms_ratio["constraint_penalty"] - terms_pct["constraint_penalty"]) < 1e-12
    assert abs(obj_ratio - obj_pct) < 1e-12


def test_collect_trial_metrics_reads_current_artifacts():
    tmp = Path(tempfile.mkdtemp(prefix="optuna_metrics_"))
    run_dir = tmp / "run"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"

    _write_csv(
        metrics_dir / "metrics_time_domain_summary.csv",
        [
            {"Method": "profile1d_cubic__robust_ossm_ekf", "MAE_median": 0.45, "RMSE_median": 0.62, "CCC_median": 0.82, "DTW_Dist_median": 0.31}
        ],
        ["Method", "MAE_median", "RMSE_median", "CCC_median", "DTW_Dist_median"],
    )
    _write_csv(
        metrics_dir / "metrics_freq_domain_summary.csv",
        [
            {"method": "profile1d_cubic__robust_ossm_ekf", "MAE_median": 0.28, "RMSE_median": 0.35, "SNR_Spec_median": 7.1}
        ],
        ["method", "MAE_median", "RMSE_median", "SNR_Spec_median"],
    )
    _write_csv(
        metrics_dir / "metrics_filter_diagnostics_summary.csv",
        [
            {
                "method": "profile1d_cubic__robust_ossm_ekf",
                "Fail_Total_median": 0.04,
                "Fail_Div_median": 0.01,
                "Fail_Slip_median": 0.02,
                "Fail_Lock_median": 0.30,
                "Fail_Double_median": 0.08,
                "NIS_TrueFail_median": 0.01,
                "NIS_OverStrict_median": 0.07,
                "Coverage95_median": 0.93
            }
        ],
        [
            "method", "Fail_Total_median", "Fail_Div_median", "Fail_Slip_median",
            "Fail_Lock_median", "Fail_Double_median",
            "NIS_TrueFail_median", "NIS_OverStrict_median", "Coverage95_median"
        ],
    )
    _write_csv(
        logs_dir / "method_quality.csv",
        [
            {
                "method": "profile1d_cubic__robust_ossm_ekf",
                "trial": "1_0",
                "missing_frame_log": False,
                "invalid_row_rate": 0.01,
                "freq_low_clip_rate": 0.02,
                "freq_high_clip_rate": 0.03,
                "z_low_clip_rate": 0.01,
                "z_high_clip_rate": 0.00,
                "q_vis_mean": 0.84,
                "alpha_R_mean": 1.2,
                "lambda_mean": 0.93,
            }
        ],
        [
            "method", "trial", "missing_frame_log", "invalid_row_rate",
            "freq_low_clip_rate", "freq_high_clip_rate", "z_low_clip_rate", "z_high_clip_rate",
            "q_vis_mean", "alpha_R_mean", "lambda_mean",
        ],
    )

    metrics, mq_rows, quality = collect_trial_metrics(run_dir, "profile1d_cubic__robust_ossm_ekf")

    assert abs(metrics["time_mae"] - 0.45) < 1e-9
    assert abs(metrics["freq_mae"] - 0.28) < 1e-9
    assert abs(metrics["fail_rate"] - 0.04) < 1e-9
    assert abs(metrics["fail_div"] - 0.01) < 1e-9
    assert abs(metrics["fail_slip"] - 0.02) < 1e-9
    assert abs(metrics["fail_lock"] - 0.30) < 1e-9
    assert abs(metrics["fail_double"] - 0.08) < 1e-9
    assert abs(metrics["coverage95"] - 0.93) < 1e-9
    assert abs(metrics["invalid_rate"] - 0.01) < 1e-9
    assert len(mq_rows) == 1
    assert quality["n_valid_rows"] == 1


def test_export_best_preset_updates_method_params():
    tmp = Path(tempfile.mkdtemp(prefix="optuna_export_"))
    output_root = tmp / "runs"
    best_dir = output_root / "robust_ossm_ekf" / "profile1d_cubic__robust_ossm_ekf"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_payload = {
        "schema_version": "optuna_best.v1",
        "method": "profile1d_cubic__robust_ossm_ekf",
        "family": "robust_ossm_ekf",
        "trial_number": 7,
        "objective": 0.123,
        "params": {
            "oscillator.qx": 1.2e-4,
            "oscillator.qf": 8.0e-6,
            "trust.beta_1": 2.7,
        },
        "metrics": {"time_mae": 0.2},
    }
    with open(best_dir / "best.json", "w", encoding="utf-8") as fp:
        json.dump(best_payload, fp, ensure_ascii=False, indent=2)

    base_cfg = {
        "name": "cohface_robust_ossm",
        "methods": [
            {"name": "profile1d_cubic__robust_ossm_ekf", "params": {"oscillator": {"qx": 0.005}}},
            {"name": "profile1d_cubic__robust_ossm_ukf", "params": {}},
        ],
    }

    out_preset = export_best_preset(
        base_cfg=base_cfg,
        config_path=Path("configs/cohface_robust_ossm.json"),
        output_root=output_root,
        destination=str(tmp / "preset_out"),
    )

    assert out_preset is not None
    assert out_preset.exists()
    with open(out_preset, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)

    methods = {m["name"]: m for m in cfg["methods"]}
    tuned = methods["profile1d_cubic__robust_ossm_ekf"]
    assert abs(tuned["params"]["oscillator"]["qx"] - 1.2e-4) < 1e-12
    assert abs(tuned["params"]["oscillator"]["qf"] - 8.0e-6) < 1e-12
    assert abs(tuned["params"]["trust"]["beta_1"] - 2.7) < 1e-12


def test_collect_trial_metrics_subject_kfold_uses_selected_validation_fold():
    tmp = Path(tempfile.mkdtemp(prefix="optuna_split_"))
    run_dir = tmp / "run"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    method = "profile1d_cubic__robust_ossm_ekf"

    # Summary CSV intentionally differs from split-aware raw trial aggregation.
    _write_csv(
        metrics_dir / "metrics_time_domain_summary.csv",
        [{"method": method, "MAE_median": 0.90, "RMSE_median": 1.1, "CCC_median": 0.50, "DTW_Dist_median": 0.70}],
        ["method", "MAE_median", "RMSE_median", "CCC_median", "DTW_Dist_median"],
    )
    _write_csv(
        metrics_dir / "metrics_freq_domain_summary.csv",
        [{"method": method, "MAE_median": 0.90, "RMSE_median": 1.0, "SNR_Spec_median": 3.0}],
        ["method", "MAE_median", "RMSE_median", "SNR_Spec_median"],
    )
    _write_csv(
        metrics_dir / "metrics_filter_diagnostics_summary.csv",
        [{"method": method, "Fail_Total_median": 0.20, "NIS_TrueFail_median": 0.20, "NIS_OverStrict_median": 0.20, "Coverage95_median": 0.70}],
        ["method", "Fail_Total_median", "NIS_TrueFail_median", "NIS_OverStrict_median", "Coverage95_median"],
    )

    time_raw = [
        {"method": method, "data_file": "data/cohface_1_0.pkl", "MAE": 2.00, "RMSE": 2.10, "CCC": 0.30, "DTW_Dist": 1.10},
        {"method": method, "data_file": "data/cohface_2_0.pkl", "MAE": 0.20, "RMSE": 0.30, "CCC": 0.95, "DTW_Dist": 0.10},
        {"method": method, "data_file": "data/cohface_3_0.pkl", "MAE": 1.90, "RMSE": 2.00, "CCC": 0.35, "DTW_Dist": 1.00},
        {"method": method, "data_file": "data/cohface_4_0.pkl", "MAE": 0.30, "RMSE": 0.40, "CCC": 0.92, "DTW_Dist": 0.12},
    ]
    _write_csv(
        metrics_dir / "metrics_time_domain_raw.csv",
        time_raw,
        ["method", "data_file", "MAE", "RMSE", "CCC", "DTW_Dist"],
    )

    freq_raw = [
        {"method": method, "data_file": "data/cohface_1_0.pkl", "MAE": 1.5, "RMSE": 1.6, "SNR_Spec": 2.0},
        {"method": method, "data_file": "data/cohface_2_0.pkl", "MAE": 0.2, "RMSE": 0.3, "SNR_Spec": 10.0},
        {"method": method, "data_file": "data/cohface_3_0.pkl", "MAE": 1.3, "RMSE": 1.5, "SNR_Spec": 2.2},
        {"method": method, "data_file": "data/cohface_4_0.pkl", "MAE": 0.3, "RMSE": 0.35, "SNR_Spec": 9.5},
    ]
    _write_csv(
        metrics_dir / "metrics_freq_domain_raw.csv",
        freq_raw,
        ["method", "data_file", "MAE", "RMSE", "SNR_Spec"],
    )

    diag_raw = [
        {"method": method, "data_file": "data/cohface_1_0.pkl", "Fail_Total": 0.4, "NIS_TrueFail": 0.3, "NIS_OverStrict": 0.2, "Coverage95": 0.7, "Lambda_LowFrac": 0.4},
        {"method": method, "data_file": "data/cohface_2_0.pkl", "Fail_Total": 0.02, "NIS_TrueFail": 0.01, "NIS_OverStrict": 0.04, "Coverage95": 0.94, "Lambda_LowFrac": 0.05},
        {"method": method, "data_file": "data/cohface_3_0.pkl", "Fail_Total": 0.3, "NIS_TrueFail": 0.2, "NIS_OverStrict": 0.1, "Coverage95": 0.75, "Lambda_LowFrac": 0.35},
        {"method": method, "data_file": "data/cohface_4_0.pkl", "Fail_Total": 0.03, "NIS_TrueFail": 0.01, "NIS_OverStrict": 0.03, "Coverage95": 0.93, "Lambda_LowFrac": 0.06},
    ]
    _write_csv(
        metrics_dir / "metrics_filter_diagnostics_raw.csv",
        diag_raw,
        ["method", "data_file", "Fail_Total", "NIS_TrueFail", "NIS_OverStrict", "Coverage95", "Lambda_LowFrac"],
    )

    mq_rows = [
        {"method": method, "trial": "1_0", "missing_frame_log": False, "invalid_row_rate": 0.20, "freq_low_clip_rate": 0.20, "freq_high_clip_rate": 0.10, "z_low_clip_rate": 0.05, "z_high_clip_rate": 0.05, "q_vis_mean": 0.3, "alpha_R_mean": 2.0, "lambda_mean": 0.7, "lambda_lt1_frac": 0.7, "lambda_low_frac": 0.5},
        {"method": method, "trial": "2_0", "missing_frame_log": False, "invalid_row_rate": 0.01, "freq_low_clip_rate": 0.01, "freq_high_clip_rate": 0.01, "z_low_clip_rate": 0.00, "z_high_clip_rate": 0.00, "q_vis_mean": 0.9, "alpha_R_mean": 1.1, "lambda_mean": 0.95, "lambda_lt1_frac": 0.4, "lambda_low_frac": 0.05},
        {"method": method, "trial": "3_0", "missing_frame_log": False, "invalid_row_rate": 0.10, "freq_low_clip_rate": 0.15, "freq_high_clip_rate": 0.10, "z_low_clip_rate": 0.05, "z_high_clip_rate": 0.02, "q_vis_mean": 0.4, "alpha_R_mean": 1.8, "lambda_mean": 0.8, "lambda_lt1_frac": 0.6, "lambda_low_frac": 0.4},
        {"method": method, "trial": "4_0", "missing_frame_log": False, "invalid_row_rate": 0.02, "freq_low_clip_rate": 0.01, "freq_high_clip_rate": 0.02, "z_low_clip_rate": 0.00, "z_high_clip_rate": 0.00, "q_vis_mean": 0.88, "alpha_R_mean": 1.2, "lambda_mean": 0.94, "lambda_lt1_frac": 0.35, "lambda_low_frac": 0.07},
    ]
    _write_csv(
        logs_dir / "method_quality.csv",
        mq_rows,
        [
            "method", "trial", "missing_frame_log", "invalid_row_rate",
            "freq_low_clip_rate", "freq_high_clip_rate", "z_low_clip_rate", "z_high_clip_rate",
            "q_vis_mean", "alpha_R_mean", "lambda_mean", "lambda_lt1_frac", "lambda_low_frac",
        ],
    )

    metrics, _, quality = collect_trial_metrics(
        run_dir,
        method,
        split_cfg={"mode": "subject_kfold", "n_folds": 2},
        trial_number=1,  # selects fold 1 => subjects 2 and 4
    )
    assert quality["split_mode_used"] == "subject_kfold"
    assert quality["split_selection_used"] is True
    assert set(quality["selected_trials"]) == {"2_0", "4_0"}
    assert abs(metrics["time_mae"] - 0.25) < 1e-9
    assert abs(metrics["fail_rate"] - 0.025) < 1e-9
    assert metrics["time_mae"] < 0.9  # confirms split-aware result overrides summary CSV


def test_normalize_objective_terms_robust_zscore():
    terms_raw = {"time_mae": 0.30, "fail_rate": 0.04, "coverage_dev": 0.01}
    weights = {"time_mae": 0.5, "fail_rate": 0.3, "coverage_dev": 0.2}
    history = [
        {"time_mae": 0.8, "fail_rate": 0.10, "coverage_dev": 0.05},
        {"time_mae": 0.7, "fail_rate": 0.09, "coverage_dev": 0.04},
        {"time_mae": 0.75, "fail_rate": 0.11, "coverage_dev": 0.03},
        {"time_mae": 0.78, "fail_rate": 0.12, "coverage_dev": 0.06},
    ]
    used, meta = normalize_objective_terms(
        terms_raw,
        weights,
        history,
        {"enabled": True, "mode": "robust_zscore", "min_history": 3, "clip": 4.0, "fallback": "raw"},
    )
    assert meta["enabled"] is True
    assert meta["per_term"]["time_mae"]["mode"] == "robust_zscore"
    assert meta["per_term"]["fail_rate"]["mode"] == "robust_zscore"
    assert used["time_mae"] < 0.0
    assert used["fail_rate"] < 0.0


def test_normalize_objective_terms_fallback_raw_when_history_short():
    terms_raw = {"time_mae": 0.40}
    weights = {"time_mae": 1.0}
    used, meta = normalize_objective_terms(
        terms_raw,
        weights,
        history_terms=[{"time_mae": 0.5}],
        cfg={"enabled": True, "mode": "robust_zscore", "min_history": 5, "fallback": "raw"},
    )
    assert used["time_mae"] == 0.40
    assert meta["per_term"]["time_mae"]["mode"] == "fallback_raw"


def test_reconcile_trial_records_backfills_missing_failed_trial_index_and_manifest():
    tmp = Path(tempfile.mkdtemp(prefix="optuna_reconcile_"))
    out_root = tmp / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    method = "profile1d_cubic__robust_ossm_ekf"
    args = StudyArgs(
        base_cfg={"name": "test", "methods": [{"name": method}]},
        config_path=Path("configs/cohface_robust_ossm.json"),
        output_root=out_root,
        n_trials=1,
        timeout=None,
        sampler_seed=42,
        pruner_enabled=True,
        keep_artifacts=True,
        em_mode="off",
        failure_objective=1e6,
        objective_weights={},
        objective_constraints={},
        objective_kfstd_reference={},
        objective_normalization={"enabled": False},
        search_space={"robust_ossm_ekf": []},
        family_defaults={"robust_ossm_ekf": {}},
        split_cfg={"mode": "none"},
        pruning_cfg={"debug_stage": False},
        repro_guard={"enabled": False},
    )
    ms = MethodStudy(
        method=method,
        family="robust_ossm_ekf",
        method_entry={"name": method, "params": {}},
        args=args,
    )
    study = optuna.create_study(
        study_name="reconcile_test",
        direction="minimize",
        storage=f"sqlite:///{ms.study_db.as_posix()}",
        load_if_exists=True,
    )

    def _fail_obj(trial):
        raise RuntimeError("intentional failure")

    study.optimize(_fail_obj, n_trials=1, catch=(RuntimeError,))
    # Simulate stale "running" index row that never got finalized.
    trial_dir = ms.trials_root / "trial_00000"
    trial_dir.mkdir(parents=True, exist_ok=True)
    ms._upsert_trial_index(
        path=ms.trial_index_path,
        trial_number=0,
        objective=float("nan"),
        objective_raw=float("nan"),
        status="running",
        duration_s=0.0,
        run_dir=None,
        trial_dir=trial_dir,
        cfg_fingerprint="",
        manifest_path="",
        metrics_summary_path="",
        params={},
        metrics={},
        split_info={},
        norm_meta={"mode": "raw"},
        prune_stage="",
    )
    ms._upsert_trial_index(
        path=ms.global_index_path,
        trial_number=0,
        objective=float("nan"),
        objective_raw=float("nan"),
        status="running",
        duration_s=0.0,
        run_dir=None,
        trial_dir=trial_dir,
        cfg_fingerprint="",
        manifest_path="",
        metrics_summary_path="",
        params={},
        metrics={},
        split_info={},
        norm_meta={"mode": "raw"},
        prune_stage="",
    )

    ms._reconcile_trial_records(study)

    with open(ms.trial_index_path, "r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "failed"
    assert row["trial_number"] == "0"
    assert row["manifest_path"] and Path(row["manifest_path"]).exists()
    assert row["metrics_summary_path"] and Path(row["metrics_summary_path"]).exists()
