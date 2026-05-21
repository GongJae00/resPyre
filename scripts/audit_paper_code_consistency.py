#!/usr/bin/env python3
"""Audit one-to-one consistency between paper text and generated artifacts.

This is intentionally narrower than a reproducibility audit: it checks that the
active manuscript sources describe the same final table-ready package that the
code generated. Historical notes are not treated as active claims, but active
paper-facing files must not point to superseded metrics or missing diagnostics.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

ACTIVE_TEXT_FILES = [
    ROOT / "paper" / "main.tex",
    ROOT / "paper" / "FIGURE_TABLE_INDEX.md",
    ROOT / "execute.md",
]

CANONICAL_OBSERVATION_CLASS_ORDER = [
    "OF",
    "OF_bridge",
    "DoF",
    "DoF_bridge",
    "P1D_lin",
    "P1D_quad",
    "P1D_cub",
    "P1D_cons",
]

STALE_ACTIVE_TOKENS = [
    "target_reliability_no_go_20260427",
    "robust_model_correction_plan_20260427",
    "source_supervised_reliability_probe_summary_20260427",
    "mahnob_tailaligned_cohort_penalty_fixedbank_bounded10_light_summary_20260427",
    "target_reliability_graph_report_20260429",
    "target_reliability_graph_parh_probe_20260429",
    "windowed_target_reliability_graph_report_20260429",
    "mahnob_failure_mode_decomposition_20260429",
    "analysis/*20260427*",
    "tab:target_reliability_probe",
    "0.145\\pm1.54",
    "0.911\\pm0.08",
]


@dataclass(frozen=True)
class Check:
    check: str
    severity: str
    status: str
    path: str
    detail: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-csv", type=Path, default=ROOT / "analysis" / "paper_code_consistency_audit.csv")
    p.add_argument("--out-md", type=Path, default=ROOT / "analysis" / "paper_code_consistency_audit.md")
    return p.parse_args()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fmt3(value: object) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _fmt4(value: object) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def _csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def _contains(text: str, token: str) -> bool:
    return token in text


def _check_token(path: Path, token: str, label: str, severity: str = "error") -> Check:
    text = _read(path)
    return Check(label, severity, "pass" if _contains(text, token) else "fail", _rel(path), f"token={token!r}")


def _check_absent(path: Path, token: str, label: str, severity: str = "error") -> Check:
    text = _read(path)
    return Check(label, severity, "fail" if _contains(text, token) else "pass", _rel(path), f"stale_token={token!r}")


def _main_numeric_checks() -> list[Check]:
    main = ROOT / "paper" / "main.tex"
    checks: list[Check] = []
    rate = _csv("paper/tables_ready/T3_rate_main.csv")
    wave = _csv("paper/tables_ready/T4_waveform_main.csv")
    strict = _csv("paper/tables_ready/T4b_waveform_strict.csv")
    cycle = _csv("paper/tables_ready/T4c_cycle_main.csv")
    taxonomy = _csv("paper/tables_ready/T7_observability_failure_taxonomy.csv")

    for _, row in rate.iterrows():
        dataset = row["dataset"]
        for col in ["PARH_MAE", "PARH_RMSE", "PARH_PearsonR"]:
            checks.append(_check_token(main, _fmt3(row[col]), f"main_T3_{dataset}_{col}"))
    for _, row in wave.iterrows():
        dataset = row["dataset"]
        for col in ["PARH_CCC", "PARH_MAE", "PARH_DTW"]:
            checks.append(_check_token(main, _fmt3(row[col]), f"main_T4_{dataset}_{col}"))
    for _, row in strict.iterrows():
        dataset = row["dataset"]
        for col in ["PARH_CCC", "PARH_NMAE_span", "PARH_NDTW_span"]:
            checks.append(_check_token(main, _fmt3(row[col]), f"main_T4b_{dataset}_{col}"))
    for _, row in cycle.iterrows():
        dataset = row["dataset"]
        for col in ["PARH_peak_time_mae_s", "PARH_trough_time_mae_s", "PARH_cycle_ppi_mae_s", "PARH_cycle_ie_abs_err"]:
            checks.append(_check_token(main, _fmt3(row[col]), f"main_T4c_{dataset}_{col}"))
    for _, row in taxonomy.iterrows():
        checks.append(_check_token(main, str(row["paper_label"]), "main_T7_failure_mode"))
        checks.append(_check_token(main, str(int(row["n_trials"])), f"main_T7_count_{row['failure_mode']}"))
    return checks


def _semantic_and_source_checks() -> list[Check]:
    checks: list[Check] = []
    t2 = _csv("paper/tables_ready/T2_observation_class_map.csv")
    observed = list(t2["observation_class"])
    checks.append(
        Check(
            "T2_observation_class_order",
            "error",
            "pass" if observed == CANONICAL_OBSERVATION_CLASS_ORDER else "fail",
            "paper/tables_ready/T2_observation_class_map.csv",
            f"observed={observed}; expected={CANONICAL_OBSERVATION_CLASS_ORDER}",
        )
    )

    index = ROOT / "paper" / "FIGURE_TABLE_INDEX.md"
    for token in [
        "F3_rate_observation_class_summary.pdf",
        "T2_observation_class_map.csv",
        "T7_observability_failure_taxonomy.csv",
        "S_T_final_observation_class_comparison.csv",
    ]:
        checks.append(_check_token(index, token, "index_current_source_policy"))

    for path in ACTIVE_TEXT_FILES:
        for token in STALE_ACTIVE_TOKENS:
            checks.append(_check_absent(path, token, "active_stale_token_absent"))

    for token in [
        "paper_code_consistency_audit",
        "audit_paper_code_consistency.py",
        "integrated PARH-OSSM complete-cohort",
    ]:
        checks.append(_check_token(ROOT / "execute.md", token, "execute_consistency_gate"))

    return checks


def build_checks() -> list[Check]:
    checks: list[Check] = []
    checks.extend(_main_numeric_checks())
    checks.extend(_semantic_and_source_checks())
    return checks


def _write_csv(path: Path, checks: list[Check]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(asdict(checks[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(c) for c in checks)


def _write_markdown(path: Path, checks: list[Check]) -> None:
    failed = [c for c in checks if c.status != "pass"]
    lines = [
        "# Paper-Code Consistency Audit",
        "",
        "This audit compares active manuscript/ledger files against the generated",
        "table-ready CSV package. It is meant to catch stale paper claims before",
        "final submission packaging.",
        "",
        f"- Total checks: {len(checks)}",
        f"- Failed checks: {len(failed)}",
        "",
        "| check | severity | status | path | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in checks:
        lines.append(f"| {c.check} | {c.severity} | {c.status} | `{c.path}` | {c.detail} |")
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        for c in failed:
            lines.append(f"- `{c.path}` {c.check}: {c.detail}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    checks = build_checks()
    _write_csv(args.out_csv, checks)
    _write_markdown(args.out_md, checks)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    failed = [c for c in checks if c.status != "pass"]
    if failed:
        print(f"Failed checks: {len(failed)}")
        return 1
    print(f"All checks passed: {len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
