from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parent.parent
TABLE_DIR = ROOT / "paper" / "tables_ready"
ANALYSIS_DIR = ROOT / "analysis"

pytestmark = pytest.mark.skipif(
    not TABLE_DIR.exists(),
    reason="local paper table workspace is not part of the public code repository",
)


def _csv(path: Path) -> pd.DataFrame:
    assert path.exists(), f"missing artifact: {path}"
    return pd.read_csv(path)


def test_headline_tables_have_representative_baseline_comparator_and_parh():
    required = {
        "T3_rate_main.csv": ["Base_MAE", "Base_PearsonR", "OSSM_KF_MAE", "OSSM_KF_PearsonR", "PARH_MAE", "PARH_PearsonR"],
        "T4_waveform_main.csv": ["Base_CCC", "Base_MAE", "OSSM_KF_CCC", "OSSM_KF_MAE", "PARH_CCC", "PARH_MAE"],
        "T4b_waveform_strict.csv": ["Base_CCC", "Base_NMAE_span", "OSSM_KF_CCC", "OSSM_KF_NMAE_span", "PARH_CCC", "PARH_NMAE_span"],
        "T4c_cycle_main.csv": ["Base_cycle_ppi_mae_s", "OSSM_KF_cycle_ppi_mae_s", "PARH_cycle_ppi_mae_s"],
    }
    for filename, metric_cols in required.items():
        df = _csv(TABLE_DIR / filename)
        assert set(df["dataset"]) == {"COHFACE", "MAHNOB"}
        n_cols = []
        for variant in ["Base", "OSSM_KF", "PARH"]:
            n_col = f"{variant}_N"
            assert n_col in df.columns
            assert pd.to_numeric(df[n_col], errors="coerce").gt(0).all(), filename
            n_cols.append(n_col)
        n_values = df[n_cols].apply(pd.to_numeric, errors="coerce")
        mismatch = n_values.nunique(axis=1, dropna=False) > 1
        if mismatch.any():
            assert {"comparison_scope", "coverage_note"}.issubset(df.columns), filename
            assert df.loc[mismatch, "comparison_scope"].eq("full_dataset_with_baseline_coverage_limit").all(), filename
            assert df.loc[mismatch, "coverage_note"].astype(str).str.contains("PARH_N=", regex=False).all(), filename
        for col in metric_cols:
            assert col in df.columns, f"{filename} missing {col}"
            assert pd.to_numeric(df[col], errors="coerce").notna().all(), f"{filename} has empty {col}"


def test_diagnostics_table_does_not_invent_ossm_kf_nis_or_lambda():
    df = _csv(TABLE_DIR / "T6_diagnostics_main.csv")
    assert set(df["dataset"]) == {"COHFACE", "MAHNOB"}
    assert "OSSM_KF_Stability_Sec" in df.columns
    assert "OSSM_KF_NIS_Mean" not in df.columns
    assert "OSSM_KF_Lambda_Mean" not in df.columns
    assert pd.to_numeric(df["OSSM_KF_N"], errors="coerce").gt(0).all()
    n_values = df[["OSSM_KF_N", "PARH_N"]].apply(pd.to_numeric, errors="coerce")
    mismatch = n_values.nunique(axis=1, dropna=False) > 1
    if mismatch.any():
        assert {"comparison_scope", "coverage_note"}.issubset(df.columns)
        assert df.loc[mismatch, "comparison_scope"].eq("full_dataset_with_baseline_coverage_limit").all()
        assert df.loc[mismatch, "coverage_note"].astype(str).str.contains("PARH_N=", regex=False).all()
    for col in ["PARH_NIS_Mean", "PARH_NIS_InBand", "PARH_Lambda_Mean", "PARH_Lambda_LT1_Frac", "PARH_Stability_Sec"]:
        assert pd.to_numeric(df[col], errors="coerce").notna().all(), col


def test_supplementary_observation_class_comparison_is_complete_and_nonempty():
    df = _csv(TABLE_DIR / "S_T_final_observation_class_comparison.csv")
    expected_classes = {"OF", "OF_bridge", "DoF", "DoF_bridge", "P1D_lin", "P1D_quad", "P1D_cub", "P1D_cons"}
    for dataset in ["COHFACE", "MAHNOB"]:
        sub = df[df["dataset"].eq(dataset)]
        assert expected_classes <= set(sub["observation_class"])
        for observation_class in expected_classes:
            variants = set(sub[sub["observation_class"].eq(observation_class)]["variant"])
            assert {"Base", "OSSM-KF", "class-local PARH"} <= variants
        assert ((sub["observation_class"].eq("PARH-OSSM")) & (sub["variant"].eq("PARH-OSSM"))).any()
    ladder = _csv(TABLE_DIR / "T6b_fusion_ladder.csv")
    assert len(ladder) >= 12
    assert {"direct_OF", "direct_DoF", "direct_P1D_quad", "direct_P1D_cons", "PARH-OSSM"} <= set(ladder["rung"])


def test_operator_alignment_table_is_scale_safe_and_not_sparse():
    df = _csv(TABLE_DIR / "T5_operator_alignment_ablation.csv")
    assert {"strict_MAE", "strict_DTW"}.isdisjoint(df.columns)
    for col in df.columns:
        assert not df[col].isna().all(), f"{col} is an all-empty paper-facing column"
    for col in ["rate_MAE", "rate_PearsonR", "waveform_CCC", "strict_CCC"]:
        assert col in df.columns
        assert pd.to_numeric(df[col], errors="coerce").notna().all()


def test_metric_provenance_makes_mixed_layers_explicit():
    prov = _csv(ANALYSIS_DIR / "final_metric_provenance_audit.csv")
    headline = prov[prov["headline_or_supplementary"].eq("headline")]
    assert not headline.empty
    assert headline["trial_count"].gt(0).all()
    assert headline["gt_use"].eq("evaluation_only").all()
    assert headline["source_layer"].astype(str).str.len().gt(0).all()
    gap = _csv(ANALYSIS_DIR / "final_submission_gap_audit.csv")
    assert not gap["status"].eq("gap").any()
    assert gap["status"].isin({"ok", "transparent_mixed_layers"}).all()
