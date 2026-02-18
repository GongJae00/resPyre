from types import SimpleNamespace

from main import _sort_methods_for_execution


def test_method_execution_order_grouped_by_family_and_variant():
    methods = [
        SimpleNamespace(name="profile1D quadratic"),
        SimpleNamespace(name="of_farneback__robust_ossm_ukf"),
        SimpleNamespace(name="profile1d_cubic__robust_ossm_ekf"),
        SimpleNamespace(name="of_farneback"),
        SimpleNamespace(name="profile1d_linear__kfstd"),
        SimpleNamespace(name="profile1D linear"),
        SimpleNamespace(name="of_farneback__kfstd"),
        SimpleNamespace(name="of_farneback__robust_ossm_ekf"),
    ]

    ordered = _sort_methods_for_execution(methods)
    names = [m.name for m in ordered]

    assert names == [
        "of_farneback",
        "of_farneback__kfstd",
        "of_farneback__robust_ossm_ekf",
        "of_farneback__robust_ossm_ukf",
        "profile1D linear",
        "profile1d_linear__kfstd",
        "profile1D quadratic",
        "profile1d_cubic__robust_ossm_ekf",
    ]
