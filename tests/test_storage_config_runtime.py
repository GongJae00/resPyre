import json
import tempfile
from pathlib import Path

from core.utils.config import load_config
from scripts.inspect_run_health import _device_hint


def test_storage_defaults_are_injected():
    tmp = tempfile.mkdtemp(prefix="storage_cfg_")
    cfg_path = Path(tmp) / "cfg.json"
    cfg_path.write_text(json.dumps({
        "datasets": [{"name": "cohface"}],
        "methods": [{"name": "of_farneback"}]
    }), encoding="utf-8")
    cfg = load_config(str(cfg_path))
    storage = cfg.get("storage", {})
    assert storage.get("save_aux") is True
    assert storage.get("save_frame_logs") is True
    assert storage.get("save_component_aux") is True


def test_storage_partial_override_is_respected():
    tmp = tempfile.mkdtemp(prefix="storage_cfg_partial_")
    cfg_path = Path(tmp) / "cfg.json"
    cfg_path.write_text(json.dumps({
        "datasets": [{"name": "cohface"}],
        "methods": [{"name": "of_farneback"}],
        "storage": {"save_aux": False}
    }), encoding="utf-8")
    cfg = load_config(str(cfg_path))
    storage = cfg.get("storage", {})
    assert storage.get("save_aux") is False
    assert storage.get("save_frame_logs") is False
    assert storage.get("save_component_aux") is False


def test_device_hint_accepts_gpu_list_payload():
    hint = _device_hint({
        "gpu": [{"name": "NVIDIA GeForce RTX 3060 Ti"}],
        "python_executable": "/tmp/python3",
    })
    assert hint.startswith("gpu:")
