import os
from typing import Dict, Optional

import numpy as np

import run_all  # type: ignore
from riv.estimators.oscillator_heads import OscillatorParams, build_head


def _normalize_base(name: str) -> str:
    key = name.lower()
    if key in ("of_model", "of", "of_farneback", "of_farneback"):
        return "of_farneback"
    if key == "dof":
        return "dof"
    if key in ("profile1d_linear", "profile1d linear", "profile1d-linear"):
        return "profile1d_linear"
    if key in ("profile1d_quadratic", "profile1d quadratic", "profile1d-quadratic"):
        return "profile1d_quadratic"
    if key in ("profile1d_cubic", "profile1d cubic", "profile1d-cubic"):
        return "profile1d_cubic"
    raise ValueError(f"Unknown base method '{name}' for oscillator wrapper")


def _normalize_head(name: str) -> str:
    key = name.lower().replace("-", "")
    if key in ("kfstd", "kf_std"):
        return "kfstd"
    if key in ("ukffreq", "ukf_freq"):
        return "ukffreq"
    if key in ("specridge", "spec_ridge"):
        return "spec_ridge"
    if key == "pll":
        return "pll"
    raise ValueError(f"Unknown oscillator head '{name}'")


def _build_base(base_key: str):
    if base_key == "of_farneback":
        base = run_all.OF_Model()
        base.name = "of_farneback"
        return base
    if base_key == "dof":
        base = run_all.DoF()
        base.name = "dof"
        return base
    if base_key == "profile1d_linear":
        base = run_all.profile1D("linear")
        base.name = "profile1d_linear"
        return base
    if base_key == "profile1d_quadratic":
        base = run_all.profile1D("quadratic")
        base.name = "profile1d_quadratic"
        return base
    if base_key == "profile1d_cubic":
        base = run_all.profile1D("cubic")
        base.name = "profile1d_cubic"
        return base
    raise ValueError(f"Unsupported base key '{base_key}'")


class OscillatorWrappedMethod(run_all.MethodBase):  # type: ignore
    """Wraps an existing chest-based method with an oscillator head."""

    def __init__(
        self,
        base_key: str,
        head_key: str,
        osc_params: Optional[OscillatorParams] = None,
        save_payload: Optional[Dict[str, bool]] = None,
    ):
        super().__init__()
        self.base_key = base_key
        self.head_key = head_key
        self.name = f"{base_key}__{head_key}"
        self.data_type = "chest"
        self.base_method = _build_base(base_key)
        self.osc_head = build_head(head_key, params=osc_params)
        self.save_payload = save_payload or {"npz": True}
        self._base_meta = {"base_method": base_key}

    def _store_npz(self, data: Dict, result: Dict[str, np.ndarray]):
        aux_dir = data.get("aux_save_dir")
        trial_key = data.get("trial_key")
        if not aux_dir or not trial_key:
            return
        os.makedirs(aux_dir, exist_ok=True)
        payload = {
            "signal_hat": np.asarray(result["signal_hat"], dtype=np.float32),
            "track_hz": np.asarray(result["track_hz"], dtype=np.float32),
            "rr_hz": np.array([result["rr_hz"]], dtype=np.float32),
            "rr_bpm": np.array([result["rr_bpm"]], dtype=np.float32),
            "meta": np.array([result["meta"]], dtype=object),
        }
        np.savez_compressed(os.path.join(aux_dir, f"{trial_key}.npz"), **payload)

    def process(self, data: Dict) -> np.ndarray:
        # Execute base method first to obtain motion proxy y(t).
        base_signal = self.base_method.process(data)
        base_signal = np.asarray(base_signal, dtype=np.float64).reshape(-1)
        fs = float(data.get("fps", self.osc_head.params.fs))
        meta = dict(self._base_meta)
        meta.update({"head": self.head_key, "fs": fs})
        result = self.osc_head.run(base_signal, fs, meta)
        if self.save_payload.get("npz", True):
            self._store_npz(data, result)
        return np.asarray(result["signal_hat"], dtype=np.float64)


def create_wrapped_method(method_name: str, params: Optional[Dict] = None) -> OscillatorWrappedMethod:
    if "__" not in method_name:
        raise ValueError("Wrapped method names must use `<base>__<head>` convention")
    base_part, head_part = method_name.split("__", 1)
    base_key = _normalize_base(base_part)
    head_key = _normalize_head(head_part)

    params = params or {}
    # Flatten nested parameter dictionaries.
    merged_params: Dict[str, float] = {}
    for key in ("params", "head_params", "oscillator", "oscillator_params"):
        if isinstance(params.get(key), dict):
            merged_params.update(params[key])
    merged_params.update({k: v for k, v in params.items() if k not in ("name", "params", "head_params", "oscillator", "oscillator_params")})

    osc_kwargs = {}
    for field in OscillatorParams().__dict__.keys():
        if field in merged_params:
            osc_kwargs[field] = merged_params[field]
    osc_params = OscillatorParams(**osc_kwargs) if osc_kwargs else None
    save_payload = params.get("save_payload")
    return OscillatorWrappedMethod(base_key, head_key, osc_params=osc_params, save_payload=save_payload)
