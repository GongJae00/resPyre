import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_parh_rate_source_decomposition import main as audit_main


def test_rate_source_decomposition_separates_native_and_final_tracks(tmp_path, monkeypatch):
    fs = 20.0
    n = 900
    t = np.arange(n, dtype=float) / fs
    gt = np.sin(2.0 * np.pi * 0.25 * t)
    final = np.full(n, 0.25, dtype=float)
    native = np.full(n, 0.18, dtype=float)
    payload = {
        "fps": fs,
        "fs_gt": fs,
        "gt": gt,
        "estimates": [
            {
                "method": "parh_probe",
                "estimate": {
                    "signal_hat": gt,
                    "track_hz": final,
                    "track_hz_native_smoothed": native,
                    "track_hz_causal": native,
                    "diagnostics": {
                        "external_output_rate_t": final,
                        "external_output_rate_blend_t": np.ones(n, dtype=float),
                    },
                },
            }
        ],
    }
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with (data_dir / "trial_0.pkl").open("wb") as f:
        pickle.dump(payload, f)

    out_csv = tmp_path / "rate_sources.csv"
    out_md = tmp_path / "rate_sources.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_parh_rate_source_decomposition.py",
            "--data-dir",
            str(data_dir),
            "--out-csv",
            str(out_csv),
            "--out-md",
            str(out_md),
            "--win-size",
            "20",
            "--stride",
            "2",
        ],
    )

    assert audit_main() == 0
    rows = pd.read_csv(out_csv)
    by_source = {row["rate_source"]: float(row["MAE"]) for _, row in rows.iterrows()}
    assert by_source["final_track_hz"] < 0.1
    assert by_source["external_output_rate_t"] < 0.1
    assert by_source["native_smoothed_track_hz"] > 3.0
    assert "readout-carried" in out_md.read_text(encoding="utf-8")
