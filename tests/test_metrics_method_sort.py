from core.pipeline.evaluation_step import _method_sort_key as eval_sort_key
from core.pipeline.visualize_step import _method_sort_key as viz_sort_key


def test_evaluation_method_sort_matches_family_rule():
    names = [
        "profile1d_cubic__robust_ossm_ukf",
        "profile1D linear",
        "of_farneback__robust_ossm",
        "profile1d_quadratic__kfstd",
        "of_farneback",
        "of_farneback__kfstd",
        "profile1d_linear__robust_ossm_ekf",
    ]
    ordered = sorted(names, key=eval_sort_key)
    assert ordered == [
        "of_farneback",
        "of_farneback__kfstd",
        "of_farneback__robust_ossm",
        "profile1D linear",
        "profile1d_linear__robust_ossm_ekf",
        "profile1d_quadratic__kfstd",
        "profile1d_cubic__robust_ossm_ukf",
    ]


def test_visualize_method_sort_matches_family_rule():
    names = [
        "profile1d_cubic__robust_ossm_ukf",
        "profile1D linear",
        "of_farneback__robust_ossm",
        "profile1d_quadratic__kfstd",
        "of_farneback",
        "of_farneback__kfstd",
        "profile1d_linear__robust_ossm_ekf",
    ]
    ordered = sorted(names, key=viz_sort_key)
    assert ordered == [
        "of_farneback",
        "of_farneback__kfstd",
        "of_farneback__robust_ossm",
        "profile1D linear",
        "profile1d_linear__robust_ossm_ekf",
        "profile1d_quadratic__kfstd",
        "profile1d_cubic__robust_ossm_ukf",
    ]
