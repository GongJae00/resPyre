from pathlib import Path
import subprocess
import sys

import pandas as pd


def test_external_weak_evidence_audit_writes_scope_outputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    v4v = tmp_path / "v4v.csv"
    scamps = tmp_path / "scamps.csv"
    out_csv = tmp_path / "audit.csv"
    out_md = tmp_path / "audit.md"
    table_out = tmp_path / "table.csv"
    figure_out = tmp_path / "figure.pdf"

    pd.DataFrame(
        [
            {
                "dataset": "V4V",
                "trial_id": "F001_T1",
                "subject": "F001",
                "split": "train",
                "video_exists": True,
                "n_rr_values": 100,
                "rr_median_bpm": 15.0,
                "rr_iqr_bpm": 4.0,
            }
        ]
    ).to_csv(v4v, index=False)
    pd.DataFrame(
        [
            {
                "dataset": "SCAMPS",
                "trial_id": "P000001",
                "mat_exists": True,
                "has_d_br": True,
                "has_raw_frames": True,
                "n_frames": 600,
                "d_br_shape": "600x1",
                "read_error": "",
            }
        ]
    ).to_csv(scamps, index=False)

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "audit_external_weak_evidence.py"),
            "--v4v-manifest",
            str(v4v),
            "--scamps-manifest",
            str(scamps),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
            "--table-out",
            str(table_out),
            "--figure-out",
            str(figure_out),
        ],
        cwd=root,
        check=True,
    )

    audit = pd.read_csv(out_csv)
    assert audit["dataset"].tolist() == ["V4V", "SCAMPS"]
    assert "external_real_rate_only" in audit["evidence_role"].tolist()
    assert "synthetic_controlled_diagnostic" in audit["evidence_role"].tolist()
    assert "waveform CCC/DTW" in out_md.read_text(encoding="utf-8")
    assert table_out.exists()
    assert figure_out.exists()
