from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.run_final_operating_point_sensitivity import (
    LOCKED_OPERATING_POINTS,
    _build_commands,
    _selected_points,
)


def test_locked_operating_points_are_small_and_semantic() -> None:
    assert 3 <= len(LOCKED_OPERATING_POINTS) <= 6
    slugs = [p.slug for p in LOCKED_OPERATING_POINTS]
    assert "locked_default" in slugs
    assert len(slugs) == len(set(slugs))
    for point in LOCKED_OPERATING_POINTS:
        assert point.window_sec > 0
        assert point.window_stride_sec > 0
        assert 0.0 < point.min_support_corr < 1.0
        assert point.max_support_residual > 0.0


def test_selected_points_rejects_unknown_slug() -> None:
    assert [p.slug for p in _selected_points(["locked_default"])] == ["locked_default"]
    try:
        _selected_points(["not_a_point"])
    except SystemExit as exc:
        assert "unknown operating point" in str(exc)
    else:
        raise AssertionError("unknown operating point should fail closed")


def test_commands_include_gt_free_priors_and_final_readout(tmp_path: Path) -> None:
    args = SimpleNamespace(
        cohface_data_dir=tmp_path / "cohface",
        mahnob_data_dir=tmp_path / "mahnob",
        results_root=tmp_path / "results",
        max_files=32,
        jobs=2,
        artifact_policy="lean",
    )
    point = _selected_points(["locked_default"])[0]
    commands = _build_commands(args, point)
    flat = [[str(x) for x in cmd] for group in commands.values() for cmd in group]
    joined = "\n".join(" ".join(cmd) for cmd in flat)
    assert "extract_target_reliability_graph_features.py" in joined
    assert "--window-sec 30" in joined
    assert "--min-support-corr 0.25" in joined
    assert "materialize_calibrated_multifamily_parh_system.py" in joined
    assert "--enable-observation-law" in joined
    assert "--rate-posterior-output-source final" in joined
    assert "--eval-use-track" in joined
    assert "generate_waveform_strict_metrics.py" in joined
