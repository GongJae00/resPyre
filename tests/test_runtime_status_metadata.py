import json
import os
import tempfile

from core.pipeline.metadata_step import mark_run_status_bulk, run_metadata_generation, touch_run_status_bulk


def test_run_status_includes_runtime_and_heartbeat_fields():
    tmp = tempfile.mkdtemp(prefix="runtime_status_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(run_dir, exist_ok=True)

    mark_run_status_bulk(
        [run_dir],
        status="running",
        command="python main.py --config x.json",
        config_path="x.json",
        steps=["estimate"],
        completed_steps=[],
    )
    touch_run_status_bulk([run_dir])

    with open(os.path.join(run_dir, "run_status.json"), "r", encoding="utf-8") as fp:
        status = json.load(fp)

    assert status["schema_version"] == "run_status.v2"
    assert isinstance(status.get("heartbeat_at"), str) and status["heartbeat_at"]
    assert int(status.get("heartbeat_seq", 0)) >= 2
    runtime = status.get("runtime")
    assert isinstance(runtime, dict)
    assert runtime.get("python_executable")
    assert runtime.get("host")


def test_metadata_promotes_runtime_fields_from_run_status():
    tmp = tempfile.mkdtemp(prefix="runtime_meta_")
    run_dir = os.path.join(tmp, "testrun")
    os.makedirs(os.path.join(run_dir, "data"), exist_ok=True)

    mark_run_status_bulk(
        [run_dir],
        status="completed",
        command="python main.py --config x.json",
        config_path="x.json",
        steps=["estimate"],
        completed_steps=["estimate"],
    )
    run_metadata_generation(tmp, run_label="testrun", allow_incomplete=True)

    with open(os.path.join(run_dir, "metadata.json"), "r", encoding="utf-8") as fp:
        meta = json.load(fp)

    assert isinstance(meta.get("runtime"), dict)
    assert meta.get("heartbeat_at")
    assert "duration_sec" in meta
