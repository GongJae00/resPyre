from pathlib import Path
import json
import subprocess
import sys


def test_design_boundary_audit_generates_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out_md = tmp_path / "audit.md"
    out_json = tmp_path / "audit.json"

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "audit_parh_design_boundary.py"),
            "--out-md",
            str(out_md),
            "--json-out",
            str(out_json),
        ],
        cwd=root,
        check=True,
    )

    report = out_md.read_text(encoding="utf-8")
    assert "PARH Design Boundary Audit" in report
    assert "high_risk_tuning_or_experimental" in report
    assert "preprocessing_policy" in report
    assert out_json.exists()

    items = json.loads(out_json.read_text(encoding="utf-8"))
    uncategorized = [item["name"] for item in items if item["category"] == "uncategorized"]
    assert uncategorized == []
