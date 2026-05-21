import tempfile
from pathlib import Path

import pandas as pd

from scripts.generate_failure_case_manifest import build_manifest


def test_build_manifest_tolerates_empty_bridge_manifest():
    tmp = Path(tempfile.mkdtemp(prefix='failure_manifest_'))
    residual = tmp / 'residual.csv'
    bridge = tmp / 'bridge.csv'
    out = tmp / 'out.csv'

    pd.DataFrame([
        {'family': 'OF', 'case_rank': 'high_residual', 'trial': 'a'}
    ]).to_csv(residual, index=False)
    bridge.write_text('', encoding='utf-8')

    manifest = build_manifest(residual, bridge, out, 'COHFACE')
    assert len(manifest) == 1
    assert manifest.iloc[0]['panel_kind'] == 'residual_heavy_parh'
