import numpy as np

from core.pipeline.evaluation_step import _format_scalar


def test_format_scalar_preserves_tiny_values_in_scientific_notation():
    tiny = 1.2e-10
    s = _format_scalar(tiny, decimals=3)
    assert "e" in s.lower()
    assert s != "0.000"


def test_format_scalar_keeps_regular_fixed_point_for_normal_values():
    assert _format_scalar(0.123456, decimals=3) == "0.123"
    assert _format_scalar(0.0, decimals=3) == "0.000"
    assert _format_scalar(np.nan, decimals=3) == "nan"
