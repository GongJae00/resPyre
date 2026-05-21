#!/usr/bin/env python3
"""Check that the final paper execution package is complete.

This is a reproducibility gate, not a performance evaluator. It verifies that
the final full path produced the required dataset-scope assets, real benchmark
runs, post-run diagnostics, external weak-evidence audits, table-ready CSVs,
figures, and manuscript PDF.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ExpectedArtifact:
    group: str
    path: Path
    required: bool
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-csv", type=Path, default=ROOT / "analysis" / "final_paper_full_package_audit.csv")
    p.add_argument("--out-md", type=Path, default=ROOT / "analysis" / "final_paper_full_package_audit.md")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write the audit without failing when required artifacts are missing.",
    )
    return p.parse_args()


def expected_artifacts() -> list[ExpectedArtifact]:
    a = ROOT / "analysis"
    p = ROOT / "paper"
    f = p / "figures"
    t = p / "tables_ready"
    r = ROOT / "results" / "final_full_validation"
    return [
        ExpectedArtifact("execution_contract", ROOT / "execute.md", True, "single paper-facing full command ledger"),
        ExpectedArtifact("dataset_scope", a / "rr_dataset_symlink_audit.csv", True, "raw dataset symlink audit"),
        ExpectedArtifact("dataset_scope", a / "rr_dataset_scope.csv", True, "dataset claim-scope audit"),
        ExpectedArtifact("dataset_scope", a / "rr_experiment_blueprint.md", True, "dataset role blueprint"),
        ExpectedArtifact("runtime_profile", a / "final_resource_profile.json", False, "optional local hardware/runtime profile captured by setup/auto_profile.py"),
        ExpectedArtifact("external_weak_evidence", a / "v4v_rr_rate_manifest.csv", True, "V4V rate-only manifest"),
        ExpectedArtifact("external_weak_evidence", a / "scamps_rr_synthetic_manifest.csv", True, "SCAMPS synthetic manifest"),
        ExpectedArtifact("external_weak_evidence", a / "dataset_rate_distribution_eda.csv", True, "four-dataset rate-distribution EDA source"),
        ExpectedArtifact("external_weak_evidence", a / "dataset_distribution_eda.csv", True, "four-dataset distribution EDA summary"),
        ExpectedArtifact("external_weak_evidence", a / "dataset_distribution_eda.md", True, "four-dataset distribution EDA report"),
        ExpectedArtifact("external_weak_evidence", a / "scamps_rr_signal_eda.csv", True, "SCAMPS synthetic breathing-signal EDA source"),
        ExpectedArtifact("external_weak_evidence", a / "external_weak_evidence_audit.csv", True, "external weak-evidence audit"),
        ExpectedArtifact("external_weak_evidence", a / "external_weak_evidence_audit.md", True, "external weak-evidence report"),
        ExpectedArtifact("guard", a / "pre_full_design_boundary_check.md", True, "design-boundary guard"),
        ExpectedArtifact("guard", a / "pre_full_design_boundary_check.json", True, "design-boundary machine-readable guard"),
        ExpectedArtifact("guard", a / "pre_full_learning_boundary_check.md", True, "learning-boundary guard"),
        ExpectedArtifact("guard", a / "pre_full_learning_boundary_check.csv", True, "learning-boundary machine-readable guard"),
        ExpectedArtifact("guard", a / "pre_full_paper_contract_audit.md", True, "paper/execute/index contract guard"),
        ExpectedArtifact("guard", a / "pre_full_paper_contract_audit.csv", True, "paper/execute/index contract machine-readable guard"),
        ExpectedArtifact("guard", a / "paper_code_consistency_audit.md", True, "paper/code/table-ready consistency guard"),
        ExpectedArtifact("guard", a / "paper_code_consistency_audit.csv", True, "paper/code/table-ready consistency machine-readable guard"),
        ExpectedArtifact("guard", a / "final_submission_gap_audit.md", True, "submission gap audit after baseline/comparator refresh"),
        ExpectedArtifact("guard", a / "final_submission_gap_audit.csv", True, "machine-readable submission gap audit"),
        ExpectedArtifact("guard", a / "final_metric_provenance_audit.md", True, "headline metric provenance audit"),
        ExpectedArtifact("guard", a / "final_metric_provenance_audit.csv", True, "machine-readable headline metric provenance audit"),
        ExpectedArtifact("guard", a / "final_split_and_leakage_audit.md", True, "split/leakage audit for final comparison package"),
        ExpectedArtifact("guard", a / "final_split_and_leakage_audit.csv", True, "machine-readable split/leakage audit"),
        ExpectedArtifact("guard", a / "final_reference_and_format_audit.md", True, "Scientific Reports reference/format audit"),
        ExpectedArtifact("guard", a / "final_reference_and_format_audit.csv", True, "machine-readable reference/format audit"),
        ExpectedArtifact("guard", a / "final_reproducibility_audit.md", True, "release/reproducibility audit"),
        ExpectedArtifact("guard", a / "final_reproducibility_audit.csv", True, "machine-readable release/reproducibility audit"),
        ExpectedArtifact("target_reliability", a / "final_priors" / "cohface_mahnob_target_computable_reliability_windowed.csv", True, "windowed GT-free COHFACE/MAHNOB reliability priors"),
        ExpectedArtifact("target_reliability", a / "final_priors" / "mahnob_target_computable_reliability_windowed.csv", True, "windowed MAHNOB readout reliability priors"),
        ExpectedArtifact("target_reliability", a / "final_priors" / "mahnob_state_component_reliability_windowed.csv", True, "windowed MAHNOB state-role reliability priors"),
        ExpectedArtifact("main_real_run", r / "cohface" / "data", True, "COHFACE full-dataset real waveform/rate run data"),
        ExpectedArtifact("main_real_run", r / "cohface" / "metrics", True, "COHFACE full-dataset real waveform/rate metrics"),
        ExpectedArtifact("main_real_run", r / "cohface" / "metrics" / "activation_audit_raw.csv", True, "COHFACE final-path activation audit"),
        ExpectedArtifact("main_real_run", r / "cohface" / "metrics" / "activation_audit_summary.json", True, "COHFACE final-path activation summary"),
        ExpectedArtifact("main_real_run", r / "mahnob_tailaligned" / "data", True, "MAHNOB full-dataset hard-regime run data"),
        ExpectedArtifact("main_real_run", r / "mahnob_tailaligned" / "metrics", True, "MAHNOB full-dataset hard-regime metrics"),
        ExpectedArtifact("main_real_run", r / "mahnob_tailaligned" / "metrics" / "activation_audit_raw.csv", True, "MAHNOB final-path activation audit"),
        ExpectedArtifact("main_real_run", r / "mahnob_tailaligned" / "metrics" / "activation_audit_summary.json", True, "MAHNOB final-path activation summary"),
        ExpectedArtifact("post_run_audit", a / "final_cohface_rate_source_decomposition.csv", True, "COHFACE rate-source audit"),
        ExpectedArtifact("post_run_audit", a / "final_cohface_rate_source_decomposition.md", True, "COHFACE rate-source report"),
        ExpectedArtifact("post_run_audit", a / "final_mahnob_tailaligned_rate_source_decomposition.csv", True, "MAHNOB rate-source audit"),
        ExpectedArtifact("post_run_audit", a / "final_mahnob_tailaligned_rate_source_decomposition.md", True, "MAHNOB rate-source report"),
        ExpectedArtifact("post_run_audit", a / "final_mahnob_tailaligned_observability_failure_modes.csv", True, "MAHNOB observability taxonomy source"),
        ExpectedArtifact("post_run_audit", a / "final_mahnob_tailaligned_observability_failure_modes.md", True, "MAHNOB observability taxonomy report"),
        ExpectedArtifact("post_run_audit", a / "final_baseline_comparator_refresh.csv", True, "same-trial baseline/comparator refresh source"),
        ExpectedArtifact("post_run_audit", a / "final_baseline_comparator_refresh.md", True, "same-trial baseline/comparator refresh report"),
        ExpectedArtifact("post_run_audit", a / "final_statistical_comparison.csv", True, "paired comparison/effect-size source"),
        ExpectedArtifact("post_run_audit", a / "final_statistical_comparison.md", True, "paired comparison/effect-size report"),
        ExpectedArtifact("post_run_audit", a / "final_baseline_comparison_interpretation.md", True, "baseline/comparator interpretation"),
        ExpectedArtifact("post_run_audit", a / "final_submission_claim_boundary.md", True, "claim boundary after final comparison refresh"),
        ExpectedArtifact("post_run_audit", a / "final_reviewer_risk_response.md", True, "reviewer-facing risk response"),
        ExpectedArtifact("post_run_audit", a / "final_scientific_lessons.md", True, "final scientific lessons for discussion/limitations"),
        ExpectedArtifact("tables", t / "T1_dataset_protocol_scope.csv", True, "main dataset scope table"),
        ExpectedArtifact("tables", t / "T2_observation_class_map.csv", True, "observation-class table"),
        ExpectedArtifact("tables", t / "T3_rate_main.csv", True, "main rate table"),
        ExpectedArtifact("tables", t / "T4_waveform_main.csv", True, "main waveform table"),
        ExpectedArtifact("tables", t / "T4b_waveform_strict.csv", True, "strict waveform companion"),
        ExpectedArtifact("tables", t / "T4c_cycle_main.csv", True, "cycle companion"),
        ExpectedArtifact("tables", t / "T5_component_ablation_evidence.csv", True, "component ablation evidence"),
        ExpectedArtifact("tables", t / "T5_operator_alignment_ablation.csv", True, "operator-alignment ablation"),
        ExpectedArtifact("tables", t / "T6_diagnostics_main.csv", True, "diagnostic table"),
        ExpectedArtifact("tables", t / "T7_observability_failure_taxonomy.csv", True, "observability taxonomy table"),
        ExpectedArtifact("tables", t / "S_T_dataset_distribution_eda.csv", True, "four-dataset distribution EDA table"),
        ExpectedArtifact("tables", t / "S_T_external_rr_manifest_summary.csv", True, "external manifest summary"),
        ExpectedArtifact("tables", t / "S_T_external_weak_evidence_audit.csv", True, "external weak-evidence audit table"),
        ExpectedArtifact("tables", t / "S_T_rr_ablation_design_contract.csv", True, "ablation design contract"),
        ExpectedArtifact("tables", t / "S_T_final_observation_class_comparison.csv", True, "supplementary observation-class baseline/comparator/PARH comparison"),
        ExpectedArtifact("figures", f / "F1_architecture.pdf", True, "architecture figure"),
        ExpectedArtifact("figures", f / "F2_dataset_and_observation_regime.pdf", True, "dataset/observation regime figure"),
        ExpectedArtifact("figures", f / "F3_rate_observation_class_summary.pdf", True, "rate observation-class summary"),
        ExpectedArtifact("figures", f / "F4_waveform_overlay_grid.pdf", True, "current waveform overlay asset"),
        ExpectedArtifact("figures", f / "F5_mechanism_activation.pdf", True, "all-base full-dataset comparison ladder"),
        ExpectedArtifact("figures", f / "F6_failure_cases.pdf", True, "failure case figure"),
        ExpectedArtifact("figures", f / "S_F15_cohface_observation_class_overlay_atlas.pdf", True, "COHFACE observation-class supplementary overlay atlas"),
        ExpectedArtifact("figures", f / "S_F16_mahnob_observation_class_overlay_atlas.pdf", True, "MAHNOB-HCI observation-class supplementary overlay atlas"),
        ExpectedArtifact("figures", f / "S_F_rr_dataset_scope_map.pdf", True, "dataset scope supplementary figure"),
        ExpectedArtifact("figures", f / "S_F_dataset_distribution_eda.pdf", True, "four-dataset distribution supplementary figure"),
        ExpectedArtifact("figures", f / "S_F_external_weak_evidence_summary.pdf", True, "external weak-evidence supplementary figure"),
        ExpectedArtifact("figures", f / "S_F_component_ablation_evidence.pdf", True, "component ablation supplementary figure"),
        ExpectedArtifact("manifests", p / "manifests" / "f4_allbase_same_trial_overlay_manifest.csv", True, "main same-trial all-base overlay manifest"),
        ExpectedArtifact("manifests", p / "manifests" / "s_f15_cohface_observation_class_overlay_manifest.csv", True, "COHFACE observation-class overlay manifest"),
        ExpectedArtifact("manifests", p / "manifests" / "s_f16_mahnob_observation_class_overlay_manifest.csv", True, "MAHNOB-HCI observation-class overlay manifest"),
        ExpectedArtifact("manuscript", p / "main.tex", True, "paper source"),
        ExpectedArtifact("manuscript", p / "main.pdf", True, "compiled paper PDF"),
        ExpectedArtifact("manuscript", p / "supplementary_information.tex", True, "supplementary information source"),
        ExpectedArtifact("manuscript", p / "supplementary_information.pdf", True, "compiled supplementary information PDF"),
        ExpectedArtifact("submission", a / "final_submission_package_audit.md", True, "final submission package audit"),
        ExpectedArtifact("submission", a / "final_submission_package_audit.csv", True, "machine-readable final submission package audit"),
        ExpectedArtifact("submission", a / "final_manual_review_checklist.md", True, "manual reviewer checklist before submission"),
    ]


def _artifact_size(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(1 for _ in path.iterdir())
        return path.stat().st_size
    except OSError:
        return 0


def build_audit() -> pd.DataFrame:
    rows = []
    final_run_roots = {
        "cohface": ROOT
        / "results"
        / "final_full_validation"
        / "cohface",
        "mahnob": ROOT
        / "results"
        / "final_full_validation"
        / "mahnob_tailaligned",
    }
    for item in expected_artifacts():
        path = item.path
        exists = path.exists()
        note = item.note
        if item.group == "post_run_audit":
            stem = path.name.lower()
            if "cohface" in stem:
                required_root = final_run_roots["cohface"]
            elif "mahnob" in stem:
                required_root = final_run_roots["mahnob"]
            else:
                required_root = None
            if required_root is not None and not (
                (required_root / "data").exists() and (required_root / "metrics").exists()
            ):
                exists = False
                note = f"{note}; stale until `{required_root.relative_to(ROOT)}` exists"
        rows.append(
            {
                "group": item.group,
                "path": str(path.relative_to(ROOT)),
                "required": bool(item.required),
                "exists": bool(exists),
                "size_or_child_count": int(_artifact_size(path)) if exists else 0,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    missing = df[(df["required"]) & (~df["exists"])]
    lines = [
        "# Final Paper Full Package Audit",
        "",
        "This report checks whether the full paper-facing execution path produced",
        "all required real-benchmark, external weak-evidence, ablation, figure,",
        "table, and manuscript artifacts.",
        "",
        f"- Required artifacts: {int(df['required'].sum())}",
        f"- Missing required artifacts: {len(missing)}",
        "",
        "| group | path | required | exists | size/children | note |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['group']} | `{row['path']}` | {int(row['required'])} | "
            f"{int(row['exists'])} | {int(row['size_or_child_count'])} | {row['note']} |"
        )
    if not missing.empty:
        lines.extend(["", "## Missing Required Artifacts", ""])
        for _, row in missing.iterrows():
            lines.append(f"- `{row['path']}`: {row['note']}")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    audit = build_audit()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.out_csv, index=False)
    write_markdown(audit, args.out_md)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    missing = audit[(audit["required"]) & (~audit["exists"])]
    if not missing.empty:
        print(f"Missing required artifacts: {len(missing)}")
        if not args.allow_missing:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
