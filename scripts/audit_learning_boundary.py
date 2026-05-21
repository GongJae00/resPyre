#!/usr/bin/env python3
"""Record the PARH-OSSM learning boundary and current evidence status.

The project intentionally mixes fixed physiological structure, target-side
self-calibration, and a small amount of source-supervised reliability fitting.
This audit makes that boundary explicit so paper-facing claims do not become
ambiguous hyperparameter search or hidden target-GT selection.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BoundaryItem:
    component: str
    boundary: str
    target_gt: str
    promoted_status: str
    evidence: str
    rationale: str
    current_gap: str
    next_action: str


ITEMS: tuple[BoundaryItem, ...] = (
    BoundaryItem(
        component="Observation operators: OF, OF_bridge, DoF, DoF_bridge, P1D_lin, P1D_quad, P1D_cub, P1D_cons",
        boundary="Fixed rule-based observation equations",
        target_gt="Forbidden",
        promoted_status="Keep",
        evidence="components/observations/methods.py; scripts/materialize_calibrated_multifamily_parh_system.py",
        rationale=(
            "These are measurement operators, not trained models. They expose heterogeneous projections of "
            "respiration: velocity, displacement bridge, burst timing, morphology, and consensus stability."
        ),
        current_gap=(
            "MAHNOB audit shows the bank often contains a better candidate, but target-side reliability is "
            "not sharp enough to use it safely."
        ),
        next_action="Do not replace with a learned waveform head; improve target-computable observability evidence.",
    ),
    BoundaryItem(
        component="Candidate views: Raw, Detr., Band, Sign, R-z, Current, Helper",
        boundary="Fixed signal-processing views used as evidence",
        target_gt="Forbidden",
        promoted_status="Keep with locked definitions",
        evidence="components/models/heads/parh_ossm.py; paper/figures/F2_dataset_and_observation_regime.pdf",
        rationale=(
            "Views expose different nuisances and supports. They should inform reliability, not act as "
            "separate target-GT-selected models."
        ),
        current_gap="Some views are visually/diagnostically weak on MAHNOB; N/A/zero entries must be explained as unavailable evidence, not model failure.",
        next_action="Report view availability and reliability as diagnostics; keep view definitions fixed before full runs.",
    ),
    BoundaryItem(
        component="Resonant state x_t=[h1c,h1s,h2c,h2s,b,b_dot,r,r_dot]",
        boundary="Fixed physiology-aligned state structure",
        target_gt="Forbidden",
        promoted_status="Keep",
        evidence="components/models/heads/parh_ossm.py; paper/main.tex",
        rationale=(
            "The oscillator/harmonic/baseline/residual split is the model thesis: timing and morphology are "
            "coupled through physiology but not forced into one waveform objective."
        ),
        current_gap="State structure alone does not solve target observability; wrong observation trust still corrupts rate readout.",
        next_action="Keep the state fixed; improve observation covariance/mixture evidence feeding the state.",
    ),
    BoundaryItem(
        component="KF/RTS inference and Student-t innovation reweighting",
        boundary="Probabilistic inference, not learned",
        target_gt="Forbidden",
        promoted_status="Keep",
        evidence="components/models/heads/parh_ossm.py; components/models/core/smoother.py",
        rationale=(
            "Kalman/RTS gives transparent state inference; Student-t downweights outlier innovations without "
            "training a black-box rejector."
        ),
        current_gap="It is protective, not a selector. It cannot recover information when all observations are weak or aliased.",
        next_action="Use NIS/lambda/prior-collapse as diagnostics and guards, not as post-hoc performance knobs.",
    ),
    BoundaryItem(
        component="Target-side sign/scale/lag calibration",
        boundary="Online self-calibration from target observations",
        target_gt="Forbidden",
        promoted_status="Allowed if diagnostics are logged",
        evidence="scripts/extract_target_reliability_graph_features.py; execute.md",
        rationale=(
            "New people/environments change observation coordinate systems. Relative calibration is necessary, "
            "but it must be estimated from inter-family agreement rather than target labels."
        ),
        current_gap="Reference/nonstationary lag dominates many MAHNOB failures, so calibration must distinguish real phase lag from unphysical matching.",
        next_action="Audit bounded-lag vs unbounded-lag cases; do not promote unphysical lag compensation.",
    ),
    BoundaryItem(
        component="Windowed target reliability graph",
        boundary="Target-computable time-local reliability prior for the observation law",
        target_gt="Forbidden",
        promoted_status="Required in final paper path",
        evidence="scripts/extract_target_reliability_graph_features.py; tests/test_target_reliability_graph_features.py; execute.md",
        rationale=(
            "The final path needs local, target-computable evidence about which observation family/view is "
            "trustworthy. It is not a target-label selector: it provides reliability, support, and state-role "
            "priors that are consumed by the observation law."
        ),
        current_gap=(
            "Full validation has not been rerun after the closure patch, so the final priors must be "
            "regenerated before the paper package is complete."
        ),
        next_action=(
            "Regenerate `analysis/final_priors/*_windowed.csv` from `execute.md` before full materialization; "
            "the paper-candidate activation audit must show the runtime prior was applied."
        ),
    ),
    BoundaryItem(
        component="Candidate-rate posterior final bounded readout",
        boundary="Deterministic target-computable timing evidence readout with preservation guards",
        target_gt="Forbidden",
        promoted_status="Final paper-candidate readout when `--rate-posterior-output-source final` is used",
        evidence="scripts/materialize_calibrated_multifamily_parh_system.py; tests/test_rate_posterior_calibrated_readout.py; tests/test_paper_candidate_activation_contract.py; execute.md",
        rationale=(
            "The posterior may adjust z_osc only when candidate evidence is specific and target-observable; "
            "otherwise it must preserve the native state-space readout. This keeps OSSM-KF from becoming "
            "an unbounded hidden fallback."
        ),
        current_gap="The closure contract is implemented, but the no-sweep full run is still pending after the latest patch.",
        next_action="Run the final full commands and require `activation_audit_summary.json` to pass for each real dataset.",
    ),
    BoundaryItem(
        component="Source-supervised observation/readout arbiter",
        boundary="Shallow learned/fitted reliability mapping from source labels",
        target_gt="Forbidden for fitting or target selection",
        promoted_status="Diagnostic/no-go for state path; readout-only candidate",
        evidence="scripts/materialize_calibrated_multifamily_parh_system.py; tests/test_observation_law_contract.py",
        rationale=(
            "A small auditable mapping is acceptable when the target has no labels, but it must not override "
            "state/morphology unless transfer-safe ablations prove it."
        ),
        current_gap="It improves some MAHNOB trials but regresses others; source validity is not yet robust enough.",
        next_action="Use as prior/diagnostic only; absorb useful features into target-side reliability, not a hard selector.",
    ),
    BoundaryItem(
        component="Source-validity graph v4 / source_validity readout",
        boundary="Source-informed diagnostic posterior",
        target_gt="Forbidden",
        promoted_status="No-go except guarded preservation",
        evidence="tests/test_rate_posterior_calibrated_readout.py; tests/test_rate_posterior_output_role.py",
        rationale=(
            "It tests whether source-learned validity transfers. Current evidence says raw replacement is unsafe."
        ),
        current_gap="Ungarded source-validity can over-resolve ambiguous target evidence.",
        next_action="Keep guarded/abstention behavior; do not promote raw source-validity replacement.",
    ),
    BoundaryItem(
        component="Torch learned observation/waveform probes",
        boundary="Deep/black-box learned controls",
        target_gt="Subject-split only; not target-GT selection",
        promoted_status="Not current promoted path",
        evidence="paper/main.tex; paper/supplementary_information.tex",
        rationale=(
            "They are useful controls showing what learned capacity can do, but they weaken interpretability "
            "and several branches failed transfer or rate/waveform decoupling."
        ),
        current_gap="Good COHFACE behavior did not establish universal target robustness.",
        next_action="Keep as baselines/controls unless a split protocol proves transfer without sacrificing interpretability.",
    ),
)


def _exists_marker(path_text: str) -> str:
    paths = [p.strip() for p in path_text.split(";")]
    states: list[str] = []
    for p in paths:
        rel = Path(p)
        if not rel.suffix and "/" not in p:
            continue
        states.append("yes" if (ROOT / rel).exists() else "missing")
    if not states:
        return "n/a"
    return "yes" if all(s == "yes" for s in states) else "partial"


def _write_csv(path: Path, items: Iterable[BoundaryItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) | {"evidence_files_present": _exists_marker(item.evidence)} for item in items]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(items: Iterable[BoundaryItem]) -> str:
    rows = list(items)
    today = date.today().isoformat()
    lines: list[str] = [
        f"# PARH-OSSM Learning Boundary Audit ({today})",
        "",
        "This audit fixes what is rule-based, what is target-side self-calibration,",
        "what is shallow source-supervised fitting, and what remains diagnostic/no-go.",
        "It is a paper-facing guardrail: performance claims must be consistent with this boundary.",
        "",
        "## Boundary Thesis",
        "",
        "- Learn mappings only where the physics does not determine observability.",
        "- Keep physiology, observation operators, and state-space inference fixed and interpretable.",
        "- Use target-computable evidence for adaptation; never use target GT for target selection.",
        "- Prefer abstention/conservative covariance inflation over hard replacement under ambiguity.",
        "- Treat source-supervised components as priors or diagnostics until no-sweep transfer evidence promotes them.",
        "",
        "## Component Decisions",
        "",
        "| Component | Boundary | Target GT | Status | Evidence Present |",
        "|---|---|---|---|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item.component} | {item.boundary} | {item.target_gt} | "
            f"{item.promoted_status} | {_exists_marker(item.evidence)} |"
        )
    lines.extend(
        [
            "",
            "## Detailed Rationale",
            "",
        ]
    )
    for item in rows:
        lines.extend(
            [
                f"### {item.component}",
                "",
                f"- Boundary: {item.boundary}",
                f"- Evidence: `{item.evidence}`",
                f"- Rationale: {item.rationale}",
                f"- Current gap: {item.current_gap}",
                f"- Next action: {item.next_action}",
                "",
            ]
        )
    lines.extend(
        [
            "## Current Performance Interpretation",
            "",
            "The latest MAHNOB bottleneck audit should be interpreted through this boundary:",
            "",
            "- The observation bank is not empty: oracle candidate-bank median MAE is lower than current readout.",
            "- The main failure is target-side observability and reliability specificity, not lack of deep capacity alone.",
            "- Source-supervised fitting is useful evidence, but current transfer evidence is not strong enough for hard promotion.",
            "- The next promotable patch must sharpen target-computable reliability while preserving the conservative fallback behavior.",
            "",
            "## Required Next Validation",
            "",
            "1. Regenerate this audit and the design-boundary audit before any full run.",
            "2. Run no-sweep COHFACE, MAHNOB, COHFACE->MAHNOB, and MAHNOB->COHFACE validation.",
            "3. Report ablation ladder: Base, OSSM-KF, PARH-fixed, PARH-R, PARH-R+pi, PARH-full, PARH-target.",
            "4. Promote only changes that improve one bottleneck class without regressing ambiguous/preservation-safe cases.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    stamp = date.today().strftime("%Y%m%d")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-md", type=Path, default=ROOT / "analysis" / f"learning_boundary_audit_{stamp}.md")
    p.add_argument("--out-csv", type=Path, default=ROOT / "analysis" / f"learning_boundary_audit_{stamp}.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_markdown(ITEMS), encoding="utf-8")
    _write_csv(args.out_csv, ITEMS)
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
