from pathlib import Path
import subprocess
import sys


def test_learning_boundary_audit_generates_release_contract(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out_md = tmp_path / "learning_boundary.md"
    out_csv = tmp_path / "learning_boundary.csv"

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "audit_learning_boundary.py"),
            "--out-md",
            str(out_md),
            "--out-csv",
            str(out_csv),
        ],
        cwd=root,
        check=True,
    )

    report = out_md.read_text(encoding="utf-8")
    table = out_csv.read_text(encoding="utf-8")

    assert "PARH-OSSM Learning Boundary Audit" in report
    assert "Learn mappings only where the physics does not determine observability" in report
    assert "Source-supervised observation/readout arbiter" in report
    assert "Torch learned observation/waveform probes" in report
    assert "component,boundary,target_gt,promoted_status" in table
