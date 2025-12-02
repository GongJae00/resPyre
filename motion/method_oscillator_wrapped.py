import copy
import os
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import signal as sps

import run_all  # type: ignore
from riv.estimators.oscillator_heads import OscillatorParams, build_head
from riv.estimators.head_ensemble import OscillatorEnsemble


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
        preproc_cfg: Optional[Dict] = None,
        ensemble_cfg: Optional[Dict] = None
    ):
        super().__init__()
        self.base_key = base_key
        self.head_key = head_key
        self.name = f"{base_key}__{head_key}"
        self.data_type = "chest"
        self.base_method = _build_base(base_key)
        self.osc_head = build_head(head_key, params=osc_params)
        self.ensemble_cfg = ensemble_cfg or {}
        self.ensemble_runner = None
        if self.ensemble_cfg.get("enabled"):
            head_defs = self.ensemble_cfg.get("heads") or [{"name": "kfstd"}, {"name": "ukffreq"}, {"name": "spec_ridge"}, {"name": "pll"}]
            self.ensemble_runner = OscillatorEnsemble(head_defs, self.preproc_cfg)
        self.save_payload = save_payload or {"npz": True}
        self._base_meta = {"base_method": base_key}
        self.preproc_cfg = copy.deepcopy(preproc_cfg) if isinstance(preproc_cfg, dict) else {}
        setattr(self.osc_head, "preproc_cfg", copy.deepcopy(self.preproc_cfg))

    def _roi_intensity_stats(self, rois: Optional[list]) -> Tuple[float, float, float]:
        """Compute coarse ROI intensity stats over time to proxy motion energy."""
        if not rois:
            return float("nan"), float("nan"), float("nan")
        try:
            frame_means = []
            frame_stds = []
            for roi in rois:
                arr = np.asarray(roi, dtype=np.float32)
                if arr.size == 0:
                    continue
                frame_means.append(float(np.mean(arr)))
                frame_stds.append(float(np.std(arr)))
            if not frame_means:
                return float("nan"), float("nan"), float("nan")
            means = np.asarray(frame_means, dtype=np.float64)
            stds = np.asarray(frame_stds, dtype=np.float64)
            mean_intensity = float(np.nanmean(means))
            std_intensity = float(np.nanmean(stds))
            snr_db = float(20.0 * np.log10(mean_intensity / max(std_intensity, 1e-6))) if std_intensity > 0 else float("nan")
            return mean_intensity, std_intensity, snr_db
        except Exception:
            return float("nan"), float("nan"), float("nan")

    def _signal_spectral_meta(self, signal_arr: np.ndarray, fs: float, f_min: float, f_max: float) -> Dict[str, float]:
        meta: Dict[str, float] = {}
        try:
            nperseg = min(signal_arr.size, max(64, int(round(fs * 4.0))))
            if nperseg >= 8:
                freqs, psd = sps.welch(signal_arr, fs=fs, nperseg=nperseg)
                band_mask = (freqs >= max(f_min, 1e-3)) & (freqs <= max(f_max, f_min + 1e-3))
                if np.any(band_mask):
                    band_freqs = freqs[band_mask]
                    band_psd = psd[band_mask]
                    idx = int(np.argmax(band_psd))
                    peak_hz = float(band_freqs[idx])
                    peak_power = float(band_psd[idx])
                    median_power = float(np.median(band_psd) + 1e-9)
                    ratio = peak_power / median_power
                    meta["welch_peak_hz"] = peak_hz
                    meta["welch_peak_ratio"] = ratio
                    meta["welch_peak_db"] = float(10.0 * np.log10(ratio))
                    # crude prominence proxy
                    meta["welch_prom_db"] = float(10.0 * np.log10(max(peak_power - median_power, 1e-9) / median_power))
                    if band_freqs.size > 1:
                        df = float(np.mean(np.diff(band_freqs)))
                        meta["welch_df_hz"] = df
        except Exception:
            pass
        return meta

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
        components = result.get("components")
        if components:
            comp_dir = os.path.join(aux_dir, "components", trial_key)
            os.makedirs(comp_dir, exist_ok=True)
            for comp in components:
                comp_result = comp['result']
                head_name = comp['name']
                comp_payload = {
                    "signal_hat": np.asarray(comp_result["signal_hat"], dtype=np.float32),
                    "track_hz": np.asarray(comp_result["track_hz"], dtype=np.float32),
                    "rr_hz": np.array([comp_result["rr_hz"]], dtype=np.float32),
                    "rr_bpm": np.array([comp_result["rr_bpm"]], dtype=np.float32),
                    "meta": np.array([comp_result["meta"]], dtype=object),
                }
                np.savez_compressed(os.path.join(comp_dir, f"{head_name}.npz"), **comp_payload)

    def process(self, data: Dict) -> np.ndarray:
        # Execute base method first to obtain motion proxy y(t).
        base_signal = self.base_method.process(data)
        base_signal = np.asarray(base_signal, dtype=np.float64).reshape(-1)
        fs = float(data.get("fps", self.osc_head.params.fs))
        meta = dict(self._base_meta)
        dataset_label = data.get("dataset_name") or data.get("dataset") or data.get("dataset_slug") or "unknown"
        trial_key = data.get("trial_key")
        meta.update({
            "head": self.head_key,
            "fs": fs,
            "dataset": dataset_label,
            "dataset_slug": dataset_label,
            "trial_key": trial_key,
            "method_name": self.name,
            "data_file": data.get("video_path")
        })
        # Base-signal diagnostics passed to oscillator heads
        if base_signal.size:
            abs_sig = np.abs(base_signal)
            meta.update({
                "signal_mean": float(np.nanmean(base_signal)),
                "signal_std": float(np.nanstd(base_signal)),
                "signal_ptp": float(np.nanmax(base_signal) - np.nanmin(base_signal)),
                "signal_energy": float(np.nanmean(base_signal ** 2)),
                "signal_abs_mean": float(np.nanmean(abs_sig)),
                "signal_abs_std": float(np.nanstd(abs_sig)),
                "signal_pos_fraction": float(np.mean(base_signal >= 0.0)),
            })
            meta.update(self._signal_spectral_meta(base_signal, fs, getattr(self.osc_head.params, "f_min", 0.08), getattr(self.osc_head.params, "f_max", 0.5)))
        roi_mean, roi_std, roi_snr_db = self._roi_intensity_stats(data.get("chest_rois"))
        meta.update({
            "roi_intensity_mean": roi_mean,
            "roi_intensity_std": roi_std,
            "roi_intensity_snr_db": roi_snr_db
        })
        if self.ensemble_runner:
            result = self.ensemble_runner.run(base_signal, fs, meta)
        else:
            result = self.osc_head.run(base_signal, fs, meta)
        if self.save_payload.get("npz", True):
            self._store_npz(data, result)
        return np.asarray(result["signal_hat"], dtype=np.float64)


def create_wrapped_method(method_name: str, params: Optional[Dict] = None, preproc_defaults: Optional[Dict] = None) -> OscillatorWrappedMethod:
    if "__" not in method_name:
        raise ValueError("Wrapped method names must use `<base>__<head>` convention")
    base_part, head_part = method_name.split("__", 1)
    base_key = _normalize_base(base_part)
    head_key = _normalize_head(head_part)

    params = params or {}
    preproc_cfg = copy.deepcopy(preproc_defaults) if isinstance(preproc_defaults, dict) else {}
    if isinstance(params.get("preproc"), dict):
        if preproc_cfg:
            preproc_cfg = run_all._deep_merge_dict(preproc_cfg, params["preproc"])
        else:
            preproc_cfg = copy.deepcopy(params["preproc"])
    # Flatten nested parameter dictionaries.
    merged_params: Dict[str, float] = {}
    for key in ("params", "head_params", "oscillator", "oscillator_params"):
        if isinstance(params.get(key), dict):
            merged_params.update(params[key])
    merged_params.update({k: v for k, v in params.items() if k not in ("name", "params", "head_params", "oscillator", "oscillator_params", "preproc")})

    ensemble_cfg = params.get("ensemble")
    osc_kwargs = {}
    for field in OscillatorParams().__dict__.keys():
        if field in merged_params:
            osc_kwargs[field] = merged_params[field]
    osc_params = OscillatorParams(**osc_kwargs) if osc_kwargs else None
    save_payload = params.get("save_payload")
    return OscillatorWrappedMethod(base_key, head_key, osc_params=osc_params, save_payload=save_payload, preproc_cfg=preproc_cfg, ensemble_cfg=ensemble_cfg)
