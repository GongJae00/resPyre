from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_paper_code_consistency_audit_passes_against_table_ready_package(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out_csv = tmp_path / "paper_code_consistency.csv"
    out_md = tmp_path / "paper_code_consistency.md"

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "audit_paper_code_consistency.py"),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ],
        cwd=root,
        check=True,
    )

    audit = pd.read_csv(out_csv)
    report = out_md.read_text(encoding="utf-8")

    assert "Paper-Code Consistency Audit" in report
    assert "main_T3" in "".join(audit["check"].astype(str).tolist())
    assert "active_stale_token_absent" in audit["check"].tolist()
    assert audit["status"].eq("pass").all()
