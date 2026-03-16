"""Shared paper asset specifications used across analysis and manuscript helpers."""

from __future__ import annotations

import numpy as np

FAMILY_SPECS = (
    {
        "family": "OF-Farneback",
        "short": "OF",
        "base_method": "of_farneback",
        "kfstd_method": "of_farneback__kfstd",
        "qrobf_method": "of_farneback__robust_ossm_ekf",
        "qs_label": "OF",
        "noise_label": "OF_Farneback",
    },
    {
        "family": "DoF",
        "short": "DoF",
        "base_method": "DoF",
        "kfstd_method": "dof__kfstd",
        "qrobf_method": "dof__robust_ossm_ekf",
        "qs_label": "DoF",
        "noise_label": "DoF",
    },
    {
        "family": "Profile1D-Linear",
        "short": "P1D-Lin",
        "base_method": "profile1D linear",
        "kfstd_method": "profile1d_linear__kfstd",
        "qrobf_method": "profile1d_linear__robust_ossm_ekf",
        "qs_label": "P1D-Lin",
        "noise_label": "Profile1D_Linear",
    },
    {
        "family": "Profile1D-Quadratic",
        "short": "P1D-Quad",
        "base_method": "profile1D quadratic",
        "kfstd_method": "profile1d_quadratic__kfstd",
        "qrobf_method": "profile1d_quadratic__robust_ossm_ekf",
        "qs_label": "P1D-Quad",
        "noise_label": "Profile1D_Quad",
    },
    {
        "family": "Profile1D-Cubic",
        "short": "P1D-Cub",
        "base_method": "profile1D cubic",
        "kfstd_method": "profile1d_cubic__kfstd",
        "qrobf_method": "profile1d_cubic__robust_ossm_ekf",
        "qs_label": "P1D-Cub",
        "noise_label": "Profile1D_Cubic",
    },
)

ABLATION_ROWS = (
    {
        "variant": "kfstd Gaussian oscillator baseline",
        "freq_mae_bpm": 0.220,
        "time_ccc": 0.890,
    },
    {
        "variant": "QROBF: forward EKF only, Student-t nu=12, no EKS",
        "freq_mae_bpm": 0.270,
        "time_ccc": np.nan,
    },
    {
        "variant": "QROBF: EKS + Gaussian (nu->inf) + rv_auto",
        "freq_mae_bpm": 0.210,
        "time_ccc": 0.880,
    },
    {
        "variant": "QROBF: EKS + Student-t nu=12 + rv_auto (full)",
        "freq_mae_bpm": 0.210,
        "time_ccc": 0.880,
    },
)
