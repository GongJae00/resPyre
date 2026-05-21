from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_pre_full_paper_contract_audit_generates_clean_contract(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out_csv = tmp_path / "paper_contract.csv"
    out_md = tmp_path / "paper_contract.md"

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "audit_pre_full_paper_contract.py"),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
        ],
        cwd=root,
        check=True,
    )

    report = out_md.read_text(encoding="utf-8")
    audit = pd.read_csv(out_csv)

    assert "Pre-Full Paper Contract Audit" in report
    assert "stale_text_token" in audit["name"].tolist()
    assert audit["status"].eq("pass").all()
