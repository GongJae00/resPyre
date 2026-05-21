from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_final_submission_readiness_audit_generates_nonfailing_reports(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    ref_csv = tmp_path / "reference.csv"
    ref_md = tmp_path / "reference.md"
    pkg_csv = tmp_path / "package.csv"
    pkg_md = tmp_path / "package.md"
    checklist = tmp_path / "checklist.md"

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "audit_final_submission_readiness.py"),
            "--reference-csv",
            str(ref_csv),
            "--reference-md",
            str(ref_md),
            "--package-csv",
            str(pkg_csv),
            "--package-md",
            str(pkg_md),
            "--checklist-md",
            str(checklist),
        ],
        cwd=root,
        check=True,
    )

    reference = pd.read_csv(ref_csv)
    package = pd.read_csv(pkg_csv)
    assert not reference["status"].eq("fail").any()
    assert not package["status"].eq("fail").any()
    assert "Final Reference and Format Audit" in ref_md.read_text(encoding="utf-8")
    assert "Final Submission Package Audit" in pkg_md.read_text(encoding="utf-8")
    assert "F1 architecture" in checklist.read_text(encoding="utf-8")
