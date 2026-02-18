import os
import tempfile

import pandas as pd

from core.evaluation.plotting_paper import plot_summary_scatter_gt


def test_scatter_plot_handles_degenerate_points_with_jitter():
    df = pd.DataFrame(
        {
            "gt_bpm": [16.5, 16.5, 16.5, 16.5],
            "est_bpm": [16.8, 16.8, 16.8, 16.8],
            "method": [
                "of_farneback",
                "of_farneback__kfstd",
                "profile1d_quadratic__robust_ossm_ekf",
                "profile1d_cubic__robust_ossm_ukf",
            ],
        }
    )
    out_dir = tempfile.mkdtemp(prefix="scatter_plot_")
    out_path = os.path.join(out_dir, "scatter.png")
    plot_summary_scatter_gt(df, out_path, title="degenerate")
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0


def test_scatter_plot_handles_all_nan_rows():
    df = pd.DataFrame(
        {
            "gt_bpm": [float("nan"), float("nan")],
            "est_bpm": [float("nan"), float("nan")],
            "method": ["a", "b"],
        }
    )
    out_dir = tempfile.mkdtemp(prefix="scatter_plot_nan_")
    out_path = os.path.join(out_dir, "scatter_nan.png")
    plot_summary_scatter_gt(df, out_path, title="nan-only")
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
