from pathlib import Path

import pandas as pd

from analysis.plot_quality_stratification import _FAMILIES_BASE, BASE_COLS_BASE
from analysis.run_paper_analysis import (
    FAMILY_SPECS,
    _build_comprehensive_performance_table,
    _build_quality_tier_table,
)


def test_build_comprehensive_performance_table_includes_base_and_no_delta(tmp_path):
    run_dir = tmp_path / "run"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True)

    freq_rows = []
    time_rows = []
    for family_idx, spec in enumerate(FAMILY_SPECS):
        for variant_idx, (variant, method_key) in enumerate(
            (
                ("Base", "base_method"),
                ("kfstd", "kfstd_method"),
                ("QROBF", "qrobf_method"),
            )
        ):
            method = spec[method_key]
            base = 0.1 * (family_idx + 1) + 0.01 * variant_idx
            freq_rows.append(
                {
                    "method": method,
                    "MAE_median": base,
                    "RMSE_median": base + 0.1,
                    "SNR_Spec_median": 10.0 + family_idx,
                }
            )
            time_rows.append(
                {
                    "method": method,
                    "CCC_median": 0.8 - 0.01 * variant_idx,
                    "MAE_median": base + 0.2,
                    "RMSE_median": base + 0.3,
                }
            )

    pd.DataFrame(freq_rows).to_csv(metrics_dir / "metrics_freq_domain_summary.csv", index=False)
    pd.DataFrame(time_rows).to_csv(metrics_dir / "metrics_time_domain_summary.csv", index=False)

    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    _build_comprehensive_performance_table(str(run_dir), tables_dir)

    out = pd.read_csv(tables_dir / "performance_comprehensive.csv")
    assert len(out) == len(FAMILY_SPECS) * 3
    assert set(out["variant"]) == {"Base", "kfstd", "QROBF"}
    assert "DoF" in set(out["observation_family"])
    assert not any("delta" in col.lower() for col in out.columns)


def test_build_quality_tier_table_adds_overall_and_drops_delta_columns(tmp_path):
    rows = []
    tiers = ["Very Poor", "Poor", "Fair", "Good"]
    for idx, tier in enumerate(tiers):
        row = {"tier": tier}
        for fam_idx, (label, kf_col, qr_col, *_rest) in enumerate(_FAMILIES_BASE):
            base_col = BASE_COLS_BASE[label]
            base = float(fam_idx + 1 + idx)
            row[base_col] = base
            row[kf_col] = base + 0.1
            row[qr_col] = base + 0.2
        rows.append(row)

    df = pd.DataFrame(rows)
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    _build_quality_tier_table(df, tables_dir)

    out = pd.read_csv(tables_dir / "quality_tier_freq_mae.csv")
    assert "Overall" in set(out["tier"])
    assert "statistic" in out.columns
    assert not any("pct" in col.lower() or "delta" in col.lower() for col in out.columns)

    overall = out.loc[out["tier"] == "Overall"].iloc[0]
    assert overall["statistic"] == "median"
    assert abs(float(overall["P1D-Cub_base_mae"]) - 5.5) < 1e-9
