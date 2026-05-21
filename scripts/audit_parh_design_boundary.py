#!/usr/bin/env python3
"""Audit PARH-OSSM design boundaries and hyperparameter risk.

This script does not decide performance. It records which constants and CLI
arguments are structural, online-estimated, experimental, or ablation-only so
paper-facing runs do not silently become dataset-specific sweeps.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
PARH_PATH = ROOT / "components" / "models" / "heads" / "parh_ossm.py"
DEFAULT_SCRIPT_GLOBS = (
    "scripts/materialize_calibrated_multifamily_parh_system.py",
    "scripts/extract_target_reliability_graph_features.py",
    "scripts/audit_external_weak_evidence.py",
    "scripts/audit_final_paper_full_package.py",
    "scripts/audit_final_submission_readiness.py",
    "scripts/run_final_operating_point_sensitivity.py",
    "scripts/run_final_baseline_comparator_refresh.py",
    "scripts/generate_table_ready.py",
)


@dataclass(frozen=True)
class AuditItem:
    name: str
    value: str
    category: str
    source: str
    note: str


def _literal(value_node: ast.AST) -> str:
    try:
        value = ast.literal_eval(value_node)
    except Exception:
        return ast.unparse(value_node) if hasattr(ast, "unparse") else "<expr>"
    return repr(value)


def _parse_parh_constants(path: Path) -> list[tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "oscillator_PARH_OSSM":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    out.append((stmt.target.id, _literal(stmt.value)))
                elif isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            out.append((target.id, _literal(stmt.value)))
            break
    return out


def _classify_constant(name: str) -> tuple[str, str]:
    if name in {"STATE_DIM", "_REF_FPS"} or name in {"HC1", "HS1", "HC2", "HS2", "B", "BDOT", "R", "RDOT"}:
        return "fixed_structure", "state-space structure or reference unit"
    if name.startswith("ENABLE_") or name in {"USE_HELPER_PATH", "USE_LIGHT_OBS_PATH"}:
        return "ablation_flag", "must be locked for paper-facing runs"
    if name in {"P1D_FIXED_FAMILY_PRIOR"}:
        return "ablation_flag", "legacy family-prior switch; off for paper-facing runs unless explicitly ablated"
    if name.startswith("HARMONIC_"):
        return "frequency_harmonic_policy", "locked harmonic disambiguation policy; no target-dataset tuning"
    if name.startswith(("OUTPUT_RATE_", "PROFILE_RATE_", "FREQ_RESCUE_", "HELPER_TRUST_", "FAMILY_CONFIDENCE_", "DYNAMIC_MIXTURE_", "RATE_OBS_")):
        return "high_risk_tuning_or_experimental", "do not tune per dataset; promote only after no-sweep validation"
    if name.startswith("TARGET_OBS_"):
        return "high_risk_tuning_or_experimental", "target-computable observability control; promote only after no-sweep validation"
    if name.startswith("PHASE_MORPH_"):
        return "readout_policy", "phase-anchored morphology readout; fixed structural sensitivity, not target tuning"
    if name.startswith(("OBS_CAL_", "QUADCUB_", "OF_")):
        return "observation_equation_policy", "allowed only as a locked observation-law policy"
    if name.startswith(("Q_OBS_", "Q_DYN_", "Q_OSC_", "GATE_", "Q_APER_", "QX_ADAPT_", "STATE_ROLE_")):
        return "reliability_mapping", "should become normalized or online-estimated where possible"
    if name.startswith(("TAU_", "NU_", "VB_", "R_ANCHOR_", "WARMUP_", "FREQ_UPDATE_", "FREQ_CONFIRM_", "FREQ_MAX_", "FREQ_INIT_")):
        return "estimation_timescale", "physiology/statistics timescale; sensitivity only, no target tuning"
    if name.startswith(("Q_HARMONIC", "Q_BASELINE", "Q_RESIDUAL")):
        return "state_noise_scale", "state flexibility prior; must be justified and locked"
    if name.startswith(("OBS_",)):
        return "preprocessing_policy", "preprocessing is part of the observation law"
    if name.startswith(("RESIDUAL_",)):
        return "experimental_residual_policy", "diagnostic unless promoted by four-regime validation"
    return "uncategorized", "requires manual classification"


def _parse_add_arguments(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    args: list[str] = []
    for match in re.finditer(r"add_argument\(\s*(['\"])(--[A-Za-z0-9_\-]+)\1", text):
        args.append(match.group(2))
    return sorted(set(args))


def _classify_arg(arg: str) -> tuple[str, str]:
    lowered = arg.lower()
    if any(token in lowered for token in ("device", "jobs", "threads", "workers", "batch-size", "artifact-policy", "max-files", "max-trials")):
        return "runtime_resource", "hardware/runtime control; not a model claim"
    if any(token in lowered for token in ("epoch", "lr", "train-frac", "val-frac", "seed")):
        return "training_protocol", "must be fixed before paper-facing training"
    if any(token in lowered for token in ("threshold", "weight", "alpha", "gamma", "penalty", "bias", "fallback", "support", "frac", "margin", "temperature", "scale")):
        return "high_risk_tuning_arg", "candidate for no-sweep lock or ablation-only status"
    if any(token in lowered for token in ("lag", "freq", "window", "min", "max", "band", "rate")):
        return "model_or_metric_boundary_arg", "semantic parameter; justify and lock"
    if any(token in lowered for token in ("out", "dir", "label", "name", "report", "csv", "json", "raw", "source", "target")):
        return "io_or_dataset", "I/O, dataset, or reporting path"
    return "other_arg", "manual review"


def _iter_script_args(paths: Iterable[Path]) -> list[AuditItem]:
    items: list[AuditItem] = []
    for path in paths:
        if not path.exists():
            continue
        for arg in _parse_add_arguments(path):
            category, note = _classify_arg(arg)
            items.append(AuditItem(arg, "", category, str(path.relative_to(ROOT)), note))
    return items


def _markdown(items: list[AuditItem]) -> str:
    groups: dict[str, list[AuditItem]] = defaultdict(list)
    for item in items:
        groups[item.category].append(item)

    lines: list[str] = []
    lines.append(f"# PARH Design Boundary Audit ({date.today().isoformat()})")
    lines.append("")
    lines.append("This report classifies constants and command-line arguments by design role.")
    lines.append("It is a guardrail against accidental dataset-specific hyperparameter search.")
    lines.append("")
    lines.append("## Boundary Rule")
    lines.append("")
    lines.append("- Fixed physiology/state definitions may be part of the method.")
    lines.append("- Online-estimated reliability may be part of the method.")
    lines.append("- Learned reliability must be trained without target GT selection.")
    lines.append("- High-risk knobs cannot be tuned per dataset for paper-facing results.")
    lines.append("- Ablation-only knobs can appear in diagnostics, not as promoted defaults.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---:|")
    for category in sorted(groups):
        lines.append(f"| `{category}` | {len(groups[category])} |")
    lines.append("")

    for category in sorted(groups):
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Name | Value | Source | Note |")
        lines.append("|---|---|---|---|")
        for item in sorted(groups[category], key=lambda x: (x.source, x.name)):
            value = item.value.replace("|", "\\|")
            note = item.note.replace("|", "\\|")
            lines.append(f"| `{item.name}` | `{value}` | `{item.source}` | {note} |")
        lines.append("")

    risk = len(groups.get("high_risk_tuning_or_experimental", [])) + len(groups.get("high_risk_tuning_arg", []))
    lines.append("## Promotion Warning")
    lines.append("")
    lines.append(f"High-risk tuning surfaces detected: `{risk}`.")
    lines.append("")
    lines.append("A paper-facing run should either:")
    lines.append("")
    lines.append("1. lock these values before looking at target performance; or")
    lines.append("2. move them into online estimation; or")
    lines.append("3. report them only as ablation/sensitivity diagnostics.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-md", type=Path, default=ROOT / "analysis" / f"parh_design_boundary_audit_{date.today().strftime('%Y%m%d')}.md")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--script", action="append", type=Path, default=None, help="Additional script to scan for add_argument calls.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    constants: list[AuditItem] = []
    for name, value in _parse_parh_constants(PARH_PATH):
        category, note = _classify_constant(name)
        constants.append(AuditItem(name, value, category, str(PARH_PATH.relative_to(ROOT)), note))

    script_paths = [ROOT / p for p in DEFAULT_SCRIPT_GLOBS]
    if args.script:
        script_paths.extend([p if p.is_absolute() else ROOT / p for p in args.script])
    cli_items = _iter_script_args(script_paths)

    items = constants + cli_items
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_markdown(items), encoding="utf-8")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps([asdict(item) for item in items], indent=2), encoding="utf-8")

    print(f"Wrote {args.out_md}")
    if args.json_out:
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
