import tempfile
from pathlib import Path

from scripts.plot_failure_cases import plot_failure_cases


def test_plot_failure_cases_tolerates_empty_manifest():
    tmp = Path(tempfile.mkdtemp(prefix='plot_failure_cases_'))
    manifest = tmp / 'manifest.csv'
    run_dir = tmp / 'run'
    out = tmp / 'out.pdf'
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text('', encoding='utf-8')
    plot_failure_cases(manifest, run_dir, out)
    assert out.exists()
    assert out.stat().st_size > 0
