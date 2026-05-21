#!/usr/bin/env python3
"""Audit the CPU-side paper contract before complete-cohort validation.

This is not a performance evaluator. It checks that the manuscript, execution
ledger, and figure/table index point to the same submission evidence layer and
artifact contract before a GPU-heavy run starts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

EXPECTED_OBSERVATION_CLASS_ORDER = [
    "OF",
    "OF_bridge",
    "DoF",
    "DoF_bridge",
    "P1D_lin",
    "P1D_quad",
    "P1D_cub",
    "P1D_cons",
]

REQUIRED_ARTIFACTS = [
    "execute.md",
    "paper/main.tex",
    "paper/FIGURE_TABLE_INDEX.md",
    "paper/tables_ready/T1_dataset_protocol_scope.csv",
    "paper/tables_ready/T2_observation_class_map.csv",
    "paper/tables_ready/T3_rate_main.csv",
    "paper/tables_ready/T4_waveform_main.csv",
    "paper/tables_ready/T4b_waveform_strict.csv",
    "paper/tables_ready/T4c_cycle_main.csv",
    "paper/tables_ready/T5_component_ablation_evidence.csv",
    "paper/tables_ready/T5_operator_alignment_ablation.csv",
    "paper/tables_ready/T6_diagnostics_main.csv",
    "paper/tables_ready/T7_observability_failure_taxonomy.csv",
    "paper/figures/F1_architecture.pdf",
    "paper/figures/F2_dataset_and_observation_regime.pdf",
    "paper/figures/F3_rate_observation_class_summary.pdf",
    "paper/figures/F4_waveform_overlay_grid.pdf",
    "paper/figures/F5_mechanism_activation.pdf",
    "paper/figures/F6_failure_cases.pdf",
]

STALE_TEXT_TOKENS = [
    "results/final_validation/",
    "results/final_inputs/",
    "analysis/parh_design_boundary_audit_20260429",
]

REQUIRED_TEXT_TOKENS = {
    "execute.md": [
        "full reproducible paper package",
        "results/final_full_validation/cohface",
        "results/final_full_validation/mahnob_tailaligned",
        "external timing-evidence comparator channel",
    ],
    "paper/main.tex": [
        "integrated PARH-OSSM run",
        "Table~\\ref{tab:cohface_t3}",
        "observation-to-state topology",
        "OSSM-KF comparator",
    ],
    "paper/FIGURE_TABLE_INDEX.md": [
        "S_T_final_observation_class_comparison.csv",
        "T2_observation_class_map.csv",
        "F1_architecture.pdf",
    ],
}


@dataclass(frozen=True)
class Check:
    name: str
    severity: str
    status: str
    path: str
    detail: str


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read(path_text: str) -> str:
    return (ROOT / path_text).read_text(encoding="utf-8")


def _artifact_checks() -> list[Check]:
    checks: list[Check] = []
    for path_text in REQUIRED_ARTIFACTS:
        path = ROOT / path_text
        if path.exists():
            size = sum(1 for _ in path.iterdir()) if path.is_dir() else path.stat().st_size
            status = "pass" if size > 0 else "fail"
            detail = f"exists, size_or_children={size}"
        else:
            status = "fail"
            detail = "missing"
        checks.append(Check("required_artifact", "error", status, path_text, detail))
    return checks


def _stale_token_checks() -> list[Check]:
    checks: list[Check] = []
    for path_text in ["execute.md", "paper/main.tex", "paper/FIGURE_TABLE_INDEX.md"]:
        text = _read(path_text)
        for token in STALE_TEXT_TOKENS:
            found = token in text
            checks.append(
                Check(
                    "stale_text_token",
                    "error",
                    "fail" if found else "pass",
                    path_text,
                    f"token={token!r}" + (" found" if found else " absent"),
                )
            )
    return checks


def _required_token_checks() -> list[Check]:
    checks: list[Check] = []
    for path_text, tokens in REQUIRED_TEXT_TOKENS.items():
        text = _read(path_text)
        for token in tokens:
            found = token in text
            checks.append(
                Check(
                    "required_text_token",
                    "error",
                    "pass" if found else "fail",
                    path_text,
                    f"token={token!r}" + (" present" if found else " missing"),
                )
            )
    return checks


def _observation_class_order_check() -> list[Check]:
    path = ROOT / "paper/tables_ready/T2_observation_class_map.csv"
    if not path.exists():
        return [Check("observation_class_order", "error", "fail", _rel(path), "T2 table missing")]
    with path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    classes = [row.get("observation_class", "") for row in rows]
    status = "pass" if classes == EXPECTED_OBSERVATION_CLASS_ORDER else "fail"
    return [
        Check(
            "observation_class_order",
            "error",
            status,
            _rel(path),
            f"observed={classes}; expected={EXPECTED_OBSERVATION_CLASS_ORDER}",
        )
    ]


def _table_nonempty_checks() -> list[Check]:
    checks: list[Check] = []
    for path_text in [p for p in REQUIRED_ARTIFACTS if p.startswith("paper/tables_ready/")]:
        path = ROOT / path_text
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            rows = list(reader)
        has_header_and_data = len(rows) >= 2 and len(rows[0]) >= 1
        checks.append(
            Check(
                "table_nonempty",
                "error",
                "pass" if has_header_and_data else "fail",
                path_text,
                f"rows={len(rows)} cols={len(rows[0]) if rows else 0}",
            )
        )
    return checks


def _write_csv(path: Path, checks: list[Check]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(asdict(checks[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(check) for check in checks)


def _write_markdown(path: Path, checks: list[Check]) -> None:
    failed = [c for c in checks if c.status != "pass"]
    lines = [
        f"# Pre-Full Paper Contract Audit ({date.today().isoformat()})",
        "",
        "This CPU-only audit checks whether the paper-facing sources agree before",
        "the GPU-heavy complete-cohort validation is launched.",
        "",
        f"- Total checks: {len(checks)}",
        f"- Failed checks: {len(failed)}",
        "",
        "| check | severity | status | path | detail |",
        "|---|---|---|---|---|",
    ]
    for check in checks:
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.name} | {check.severity} | {check.status} | `{check.path}` | {detail} |")
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        for check in failed:
            lines.append(f"- `{check.path}`: {check.name} -> {check.detail}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-csv", type=Path, default=ROOT / "analysis" / "pre_full_paper_contract_audit.csv")
    p.add_argument("--out-md", type=Path, default=ROOT / "analysis" / "pre_full_paper_contract_audit.md")
    p.add_argument("--allow-fail", action="store_true", help="Write reports without returning a failing exit code.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    checks: list[Check] = []
    checks.extend(_artifact_checks())
    checks.extend(_stale_token_checks())
    checks.extend(_required_token_checks())
    checks.extend(_observation_class_order_check())
    checks.extend(_table_nonempty_checks())
    _write_csv(args.out_csv, checks)
    _write_markdown(args.out_md, checks)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    failed = [c for c in checks if c.status != "pass" and c.severity == "error"]
    if failed:
        print(f"Failed checks: {len(failed)}")
        if not args.allow_fail:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
