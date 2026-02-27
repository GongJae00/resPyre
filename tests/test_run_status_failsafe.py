import json
import os
import tempfile

from core.pipeline.metadata_step import (
    bootstrap_run_dirs,
    mark_run_status_bulk,
    run_metadata_generation,
)


def test_run_status_and_metadata_exist_after_failure_signal():
    tmp = tempfile.mkdtemp(prefix="run_status_fail_")
    run_dirs = bootstrap_run_dirs(tmp, "testrun", dataset_names=["cohface"])
    assert run_dirs, "bootstrap_run_dirs must return at least one run directory"
    run_dir = run_dirs[0]

    mark_run_status_bulk(
        run_dirs,
        status="running",
        command="python main.py --config x.json",
        config_path="x.json",
        steps=["estimate", "evaluate", "metadata"],
        completed_steps=[],
    )
    mark_run_status_bulk(
        run_dirs,
        status="failed",
        command="python main.py --config x.json",
        config_path="x.json",
        steps=["estimate", "evaluate", "metadata"],
        completed_steps=["estimate"],
        error_summary="forced failure",
        error_traceback="Traceback: forced",
    )
    run_metadata_generation(
        tmp,
        run_label="testrun",
        command="python main.py --config x.json",
        allow_incomplete=True,
    )

    status_path = os.path.join(run_dir, "run_status.json")
    metadata_path = os.path.join(run_dir, "metadata.json")
    assert os.path.exists(status_path)
    assert os.path.exists(metadata_path)

    with open(status_path, "r", encoding="utf-8") as fp:
        status = json.load(fp)
    with open(metadata_path, "r", encoding="utf-8") as fp:
        meta = json.load(fp)

    assert status.get("status") == "failed"
    assert isinstance(status.get("completed_at"), str) and status.get("completed_at")
    assert meta.get("status") == "failed"
    assert isinstance(meta.get("run_status"), dict)
