import tempfile
from pathlib import Path

import pandas as pd

from scripts.plot_of_construction_comparison import plot_comparison


def test_plot_comparison_tolerates_missing_of_family():
    tmp = Path(tempfile.mkdtemp(prefix='plot_of_construction_'))
    t3 = tmp / 't3.csv'
    t4 = tmp / 't4.csv'
    out = tmp / 'out.pdf'
    pd.DataFrame([
        {'dataset': 'COHFACE', 'family': 'OF_bridge', 'Base_MAE': 1.0, 'Base_PearsonR': 0.5, 'KFstd_MAE': 1.0, 'KFstd_PearsonR': 0.5, 'PARH_MAE': 1.0, 'PARH_PearsonR': 0.5}
    ]).to_csv(t3, index=False)
    pd.DataFrame([
        {'dataset': 'COHFACE', 'family': 'OF_bridge', 'Base_CCC': 0.5, 'Base_DTW': 0.5, 'KFstd_CCC': 0.5, 'KFstd_DTW': 0.5, 'PARH_CCC': 0.5, 'PARH_DTW': 0.5}
    ]).to_csv(t4, index=False)
    data = plot_comparison(t3, t4, out)
    assert data.empty
    assert out.exists()
    assert out.stat().st_size > 0
