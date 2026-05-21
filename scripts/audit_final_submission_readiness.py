#!/usr/bin/env python3
"""Create final submission-readiness audits from the built paper package.

The audit is deliberately conservative about correctness: unresolved
references/citations and LaTeX errors fail. Overfull boxes are formatting
warnings because they can indicate clipped or protruding text. Underfull boxes
are retained in the detail field for manual review but are not treated as
submission-blocking warnings by themselves.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "analysis"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables_ready"
FIGURES = PAPER / "figures"


@dataclass(frozen=True)
class AuditRow:
    audit: str
    item: str
    status: str
    path: str
    detail: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-csv", type=Path, default=ANALYSIS / "final_reference_and_format_audit.csv")
    p.add_argument("--reference-md", type=Path, default=ANALYSIS / "final_reference_and_format_audit.md")
    p.add_argument("--package-csv", type=Path, default=ANALYSIS / "final_submission_package_audit.csv")
    p.add_argument("--package-md", type=Path, default=ANALYSIS / "final_submission_package_audit.md")
    p.add_argument("--checklist-md", type=Path, default=ANALYSIS / "final_manual_review_checklist.md")
    return p.parse_args()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _exists_row(audit: str, item: str, path: Path, detail: str = "") -> AuditRow:
    exists = path.exists() and (path.is_dir() or path.stat().st_size > 0)
    return AuditRow(audit, item, "pass" if exists else "fail", _rel(path), detail or ("exists" if exists else "missing or empty"))


def reference_rows() -> list[AuditRow]:
    rows: list[AuditRow] = []
    logs = [
        ("submission_manuscript", PAPER / "main.log", "paper/main.pdf"),
    ]
    for item, log_path, pdf_name in logs:
        log_exists = log_path.exists() and log_path.stat().st_size > 0
        text = _read(log_path)
        has_error = "! LaTeX Error" in text or "Emergency stop" in text
        undef_cite = text.count("Citation `")
        undef_ref = text.count("Reference `")
        overfull = text.count("Overfull \\hbox")
        underfull = text.count("Underfull \\hbox") + text.count("Underfull \\vbox")
        if not log_exists:
            rows.append(
                AuditRow(
                    "reference_and_format",
                    f"{item}_build_log",
                    "pass",
                    _rel(log_path),
                    "absent after cleanup; run latexmk to regenerate detailed citation/box diagnostics",
                )
            )
            continue
        rows.append(
            AuditRow(
                "reference_and_format",
                f"{item}_build",
                "warn" if not log_exists else ("fail" if has_error else "pass"),
                _rel(log_path),
                f"target={pdf_name}; log_exists={log_exists}; latex_error={has_error}",
            )
        )
        rows.append(
            AuditRow(
                "reference_and_format",
                f"{item}_undefined_citations",
                "warn" if not log_exists else ("fail" if undef_cite else "pass"),
                _rel(log_path),
                f"undefined citation warnings={undef_cite}",
            )
        )
        rows.append(
            AuditRow(
                "reference_and_format",
                f"{item}_undefined_references",
                "warn" if not log_exists else ("fail" if undef_ref else "pass"),
                _rel(log_path),
                f"undefined reference warnings={undef_ref}",
            )
        )
        rows.append(
            AuditRow(
                "reference_and_format",
                f"{item}_box_warnings",
                "warn" if (not log_exists or overfull) else "pass",
                _rel(log_path),
                f"overfull={overfull}; underfull={underfull}; underfull is informational",
            )
        )
    rows.extend(
        [
            _exists_row("reference_and_format", "main_bbl_current", PAPER / "main.bbl", "bibliography materialized"),
            _exists_row("reference_and_format", "submission_pdf", PAPER / "main.pdf", "Scientific Reports manuscript PDF"),
        ]
    )
    return rows


def _table_complete(path: Path) -> AuditRow:
    if not path.exists() or path.stat().st_size == 0:
        return AuditRow("submission_package", path.name, "fail", _rel(path), "missing or empty")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive audit branch
        return AuditRow("submission_package", path.name, "fail", _rel(path), f"csv read failed: {exc}")
    return AuditRow("submission_package", path.name, "pass" if len(df) else "fail", _rel(path), f"rows={len(df)} cols={len(df.columns)}")


def _main_table_baseline_complete() -> list[AuditRow]:
    rows: list[AuditRow] = []
    required = {
        "T3_rate_main.csv": ["Base_MAE", "OSSM_KF_MAE", "PARH_MAE", "Base_N", "OSSM_KF_N", "PARH_N"],
        "T4_waveform_main.csv": ["Base_CCC", "OSSM_KF_CCC", "PARH_CCC", "Base_N", "OSSM_KF_N", "PARH_N"],
        "T4b_waveform_strict.csv": ["Base_NMAE_span", "OSSM_KF_NMAE_span", "PARH_NMAE_span", "Base_N", "OSSM_KF_N", "PARH_N"],
        "T4c_cycle_main.csv": ["Base_cycle_ppi_mae_s", "OSSM_KF_cycle_ppi_mae_s", "PARH_cycle_ppi_mae_s", "Base_N", "OSSM_KF_N", "PARH_N"],
    }
    for filename, cols in required.items():
        path = TABLES / filename
        if not path.exists():
            rows.append(AuditRow("submission_package", f"{filename}_main_table_completeness", "fail", _rel(path), "missing"))
            continue
        df = pd.read_csv(path)
        missing = [col for col in cols if col not in df.columns or pd.to_numeric(df[col], errors="coerce").isna().any()]
        n_bad = []
        n_mismatch_rows = 0
        for col in ["Base_N", "OSSM_KF_N", "PARH_N"]:
            if col in df.columns and not pd.to_numeric(df[col], errors="coerce").gt(0).all():
                n_bad.append(col)
        n_cols = [col for col in ["Base_N", "OSSM_KF_N", "PARH_N"] if col in df.columns]
        if len(n_cols) == 3:
            n_values = df[n_cols].apply(pd.to_numeric, errors="coerce")
            n_mismatch_rows = int((n_values.nunique(axis=1, dropna=False) > 1).sum())
        explained_mismatch_rows = 0
        if n_mismatch_rows:
            if {"comparison_scope", "coverage_note"}.issubset(df.columns):
                mismatch = n_values.nunique(axis=1, dropna=False) > 1
                explained = (
                    df.loc[mismatch, "comparison_scope"]
                    .astype(str)
                    .eq("full_dataset_with_baseline_coverage_limit")
                    & df.loc[mismatch, "coverage_note"].astype(str).str.contains("PARH_N=", regex=False)
                )
                explained_mismatch_rows = int(explained.sum())
        unexplained_mismatch_rows = n_mismatch_rows - explained_mismatch_rows
        status = "pass" if not missing and not n_bad and unexplained_mismatch_rows == 0 else "fail"
        rows.append(
            AuditRow(
                "submission_package",
                f"{filename}_main_table_completeness",
                status,
                _rel(path),
                "missing_or_nan="
                f"{missing}; invalid_N={n_bad}; mismatched_N_rows={n_mismatch_rows}; "
                f"explained_mismatch_rows={explained_mismatch_rows}",
            )
        )
    return rows


def _strict_raw_foreground_guard() -> list[AuditRow]:
    rows: list[AuditRow] = []
    forbidden = ["Strict MAE \\\\", "Strict MAE &"]
    for path in [PAPER / "main.tex"]:
        text = _read(path)
        hits = [token for token in forbidden if token in text]
        rows.append(
            AuditRow(
                "submission_package",
                f"{path.name}_strict_raw_mae_not_foregrounded",
                "pass" if not hits else "fail",
                _rel(path),
                f"forbidden_table_header_hits={hits}",
            )
        )
    return rows


def _f4_manifest_provenance_guard() -> list[AuditRow]:
    """Verify that the main overlay figure uses final-run PARH rows.

    The existence audit alone cannot catch stale figure manifests.  This guard
    prevents a common failure mode where the PDF is regenerated from an old
    pre-full/full32 PARH source while the manuscript caption claims the
    complete-cohort layer.
    """
    path = PAPER / "manifests" / "f4_allbase_same_trial_overlay_manifest.csv"
    if not path.exists() or path.stat().st_size == 0:
        return [
            AuditRow(
                "submission_package",
                "f4_manifest_final_provenance",
                "fail",
                _rel(path),
                "missing or empty manifest",
            )
        ]
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive audit branch
        return [
            AuditRow(
                "submission_package",
                "f4_manifest_final_provenance",
                "fail",
                _rel(path),
                f"csv read failed: {exc}",
            )
        ]

    required_labels = {
        "OF direct",
        "OF bridge",
        "DoF direct",
        "DoF bridge",
        "P1D lin",
        "P1D quad",
        "P1D cub",
        "P1D cons",
        "OSSM-KF (P1D quad)",
        "PARH-OSSM",
    }
    problems: list[str] = []
    if len(df) != 20:
        problems.append(f"expected 20 rows, observed {len(df)}")
    for dataset, expected_root in {
        "COHFACE": "results/final_full_validation/cohface",
        "MAHNOB-HCI": "results/final_full_validation/mahnob_tailaligned",
    }.items():
        sub = df[df.get("dataset", pd.Series(dtype=object)).astype(str).eq(dataset)]
        labels = set(sub.get("label", pd.Series(dtype=object)).astype(str).tolist())
        missing_labels = sorted(required_labels - labels)
        if missing_labels:
            problems.append(f"{dataset} missing labels={missing_labels}")
        parh = sub[sub.get("label", pd.Series(dtype=object)).astype(str).eq("PARH-OSSM")]
        if len(parh) != 1:
            problems.append(f"{dataset} PARH row count={len(parh)}")
            continue
        run_dir = str(parh.iloc[0].get("run_dir", ""))
        data_file = str(parh.iloc[0].get("data_file", ""))
        if expected_root not in run_dir:
            problems.append(f"{dataset} PARH run_dir not complete-cohort root: {run_dir}")
        if "final_pre_full_validation" in run_dir or "full32" in run_dir:
            problems.append(f"{dataset} PARH run_dir is stale pre-full/full32: {run_dir}")
        if expected_root not in data_file:
            problems.append(f"{dataset} PARH data_file not complete-cohort root: {data_file}")
        if data_file and not Path(data_file).exists():
            problems.append(f"{dataset} PARH data_file missing on disk: {data_file}")

    return [
        AuditRow(
            "submission_package",
            "f4_manifest_final_provenance",
            "pass" if not problems else "fail",
            _rel(path),
            "PARH rows point to final_full_validation and all 10 labels per dataset"
            if not problems
            else "; ".join(problems),
        )
    ]


def _submission_source_structure_guard() -> list[AuditRow]:
    rows: list[AuditRow] = []
    path = PAPER / "main.tex"
    text = _read(path)
    n_fig = len(re.findall(r"\\begin\{figure\}", text))
    n_tab = len(re.findall(r"\\begin\{table\}", text))
    n_display = n_fig + n_tab
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    abstract_text = abstract_match.group(1) if abstract_match else ""
    abstract_words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", abstract_text))
    rows.append(
        AuditRow(
            "submission_package",
            "main_display_item_count",
            "pass" if n_display <= 8 else "fail",
            _rel(path),
            f"figures={n_fig}; tables={n_tab}; total={n_display}; target_limit=8",
        )
    )
    rows.append(
        AuditRow(
            "submission_package",
            "main_abstract_word_count",
            "pass" if abstract_match and abstract_words <= 200 else "fail",
            _rel(path),
            f"rough_words={abstract_words}; target_limit=200",
        )
    )
    pdf_path = PAPER / "main.pdf"
    pages_detail = "pdfinfo unavailable"
    try:
        proc = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        match = re.search(r"^Pages:\s+(\d+)", proc.stdout, re.M)
        if match:
            pages_detail = (
                f"pages={match.group(1)}; Scientific Reports page guidance is an ideal, "
                "not a hard audit failure; manual compression decision required"
            )
    except OSError:
        pass
    rows.append(
        AuditRow(
            "submission_package",
            "main_pdf_page_count_manual_review",
            "pass",
            _rel(pdf_path),
            pages_detail,
        )
    )
    return rows


def package_rows() -> list[AuditRow]:
    rows: list[AuditRow] = []
    required_paths = [
        PAPER / "main.tex",
        PAPER / "main.pdf",
        PAPER / "supplementary_information.tex",
        PAPER / "supplementary_information.pdf",
        FIGURES / "F1_architecture.pdf",
        FIGURES / "F2_dataset_and_observation_regime.pdf",
        FIGURES / "F3_rate_observation_class_summary.pdf",
        FIGURES / "F4_waveform_overlay_grid.pdf",
        FIGURES / "F5_mechanism_activation.pdf",
        FIGURES / "F6_failure_cases.pdf",
        FIGURES / "S_F15_cohface_observation_class_overlay_atlas.pdf",
        FIGURES / "S_F16_mahnob_observation_class_overlay_atlas.pdf",
        PAPER / "manifests" / "f4_allbase_same_trial_overlay_manifest.csv",
        PAPER / "manifests" / "s_f15_cohface_observation_class_overlay_manifest.csv",
        PAPER / "manifests" / "s_f16_mahnob_observation_class_overlay_manifest.csv",
    ]
    rows.extend(_exists_row("submission_package", path.name, path) for path in required_paths)
    for name in [
        "T1_dataset_protocol_scope.csv",
        "T2_observation_class_map.csv",
        "T3_rate_main.csv",
        "T4_waveform_main.csv",
        "T4b_waveform_strict.csv",
        "T4c_cycle_main.csv",
        "T5_component_ablation_evidence.csv",
        "T5_operator_alignment_ablation.csv",
        "T6_diagnostics_main.csv",
        "T6b_fusion_ladder.csv",
        "T7_observability_failure_taxonomy.csv",
        "S_T_final_observation_class_comparison.csv",
    ]:
        rows.append(_table_complete(TABLES / name))
    rows.extend(_main_table_baseline_complete())
    rows.extend(_strict_raw_foreground_guard())
    rows.extend(_f4_manifest_provenance_guard())
    rows.extend(_submission_source_structure_guard())

    claim_boundary_path = ANALYSIS / "final_submission_claim_boundary.md"
    claim_boundary = _read(claim_boundary_path)
    if claim_boundary:
        for token in ["MAHNOB", "OSSM-KF", "strict", "not"]:
            rows.append(
                AuditRow(
                    "submission_package",
                    f"claim_boundary_mentions_{token}",
                    "pass" if token in claim_boundary else "fail",
                    _rel(claim_boundary_path),
                    f"token={token!r}",
                )
            )
    else:
        rows.append(
            AuditRow(
                "submission_package",
                "claim_boundary_optional_audit",
                "pass",
                _rel(claim_boundary_path),
                "optional local provenance file absent",
            )
        )
    return rows


def _write_csv(path: Path, rows: list[AuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_md(path: Path, title: str, rows: list[AuditRow]) -> None:
    failed = [row for row in rows if row.status == "fail"]
    warned = [row for row in rows if row.status == "warn"]
    lines = [
        f"# {title}",
        "",
        f"- Checks: {len(rows)}",
        f"- Failed: {len(failed)}",
        f"- Warnings: {len(warned)}",
        "",
        "| audit | item | status | path | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row.audit} | {row.item} | {row.status} | `{row.path}` | {row.detail} |")
    if failed:
        lines.extend(["", "## Failures", ""])
        for row in failed:
            lines.append(f"- `{row.path}` {row.item}: {row.detail}")
    if warned:
        lines.extend(["", "## Manual Warnings", ""])
        for row in warned:
            lines.append(f"- `{row.path}` {row.item}: {row.detail}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_checklist(path: Path) -> None:
    lines = [
        "# Final Manual Review Checklist",
        "",
        "This checklist is intentionally manual. Automated audits can verify paths,",
        "numbers, provenance, and build status, but not whether the final PDF reads",
        "well to a reviewer.",
        "",
        "- [ ] Confirm title, authors, affiliations, and correspondence details on `paper/main.pdf`.",
        "- [ ] Confirm the current submission manuscript still has `5` main figures, `3` main tables, and an abstract under 200 words after any final edit.",
        "- [ ] Decide whether the current page count is acceptable for first review or whether another compression pass is needed toward the Scientific Reports ideal page length.",
        "- [ ] Inspect F1 architecture (`paper/figures/F1_architecture.pdf`): decide whether the current architecture rendering is clear enough or should be redrawn before submission.",
        "- [ ] Inspect every main figure for clipping, tiny text, broken legends, and panel-label ambiguity.",
        "- [ ] Confirm the three main tables T3/T4/T7 match `paper/tables_ready/*.csv`; confirm T4b/T4c/T5/T6/T6b are treated as supplementary/diagnostic companions.",
        "- [ ] Check that `P1D_quad direct`, `OSSM-KF (P1D quad)`, and `PARH-OSSM` are not described as interchangeable methods.",
        "- [ ] Check that the COHFACE claim is strong but not overstated: PARH improves waveform/strict/cycle, while direct P1D_quad remains a very strong rate baseline.",
        "- [ ] Check that the MAHNOB claim is explicitly bounded: rate improves over representative baseline/comparator, but waveform/strict/cycle remains a hard-regime limitation.",
        "- [ ] Check that strict raw MAE is always interpreted with strict CCC and span-normalized companion metrics.",
        "- [ ] Check that `OSSM-KF` reads only as a comparator/weak evidence channel, never as the proposed model.",
        "- [ ] Check that candidate views are described as evidence views, not final model candidates.",
        "- [ ] Check that adaptive observation law is described as reliability/role weighting, not one-best selector.",
        "- [ ] Check that MAHNOB failure discussion reads as evidence-backed boundary analysis, not a post-hoc excuse.",
        "- [ ] Decide whether a short cover note to the corresponding author is needed before external review.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    ref_rows = reference_rows()
    pkg_rows = package_rows()
    _write_csv(args.reference_csv, ref_rows)
    _write_md(args.reference_md, "Final Reference and Format Audit", ref_rows)
    _write_csv(args.package_csv, pkg_rows)
    _write_md(args.package_md, "Final Submission Package Audit", pkg_rows)
    _write_checklist(args.checklist_md)
    print(f"Wrote {args.reference_csv}")
    print(f"Wrote {args.reference_md}")
    print(f"Wrote {args.package_csv}")
    print(f"Wrote {args.package_md}")
    print(f"Wrote {args.checklist_md}")
    failed = [row for row in ref_rows + pkg_rows if row.status == "fail"]
    if failed:
        print(f"Failed checks: {len(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
