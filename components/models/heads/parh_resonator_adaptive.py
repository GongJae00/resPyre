import json
from typing import Dict, Optional

import numpy as np

from components.models.observation_adaptor import FamilyObservationAdaptor
from components.models.shared_respiratory_latent import SharedRespiratoryLatentBootstrap
from components.observations.semantics import get_observation_family_semantics
from ..core.base import _BaseOscillatorHead
from .parh_ossm import oscillator_PARH_OSSM


class oscillator_PARH_ResonatorAdaptive(_BaseOscillatorHead):
    """Adaptive observation front-end over a shared resonator scaffold.

    Stage-1 implementation:
    - consume family multi-view observation hypotheses
    - build canonical waveform / oscillatory observations
    - delegate latent dynamics to the existing PARH resonator core

    This keeps the pipeline runnable while moving the architecture from
    monolithic fixed observation laws toward shared law + adaptive observation.
    """

    head_key = "parh_resonator_adaptive"

    def __init__(self, params=None):
        super().__init__(params=params)
        self.core = oscillator_PARH_OSSM(params=self.params)
        self.adaptor = FamilyObservationAdaptor(
            f_min=float(getattr(self.params, "f_min", 0.08) or 0.08),
            f_max=float(getattr(self.params, "f_max", 0.50) or 0.50),
        )
        self.bootstrap = SharedRespiratoryLatentBootstrap()

    def _merge_meta(self, result: Dict[str, np.ndarray], meta_updates: Dict[str, object]) -> None:
        meta_raw = result.get("meta", "{}")
        try:
            payload = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw or {})
        except Exception:
            payload = {}
        payload.update(meta_updates)
        result["meta"] = json.dumps(payload)

    def _family_group_semantics(self, families):
        grouped = {}
        for family in families:
            sem = get_observation_family_semantics(family)
            key = str(sem.get("family_group") or family)
            grouped.setdefault(key, []).append(dict(sem))
        return grouped

    @staticmethod
    def _bootstrap_initial_state(bootstrap, fs: float) -> np.ndarray:
        def _grad0(x: np.ndarray) -> float:
            arr = np.asarray(x, dtype=np.float64).reshape(-1)
            if arr.size <= 1:
                return 0.0
            return float((arr[1] - arr[0]) * float(fs))

        h1 = np.asarray(bootstrap.latent_components.get("h1_drive", np.array([], dtype=np.float64)), dtype=np.float64)
        h2 = np.asarray(bootstrap.latent_components.get("h2_drive", np.array([], dtype=np.float64)), dtype=np.float64)
        b = np.asarray(bootstrap.latent_components.get("b_drive", np.array([], dtype=np.float64)), dtype=np.float64)
        r = np.asarray(bootstrap.latent_components.get("r_drive", np.array([], dtype=np.float64)), dtype=np.float64)
        state = np.zeros(8, dtype=np.float64)
        if h1.size:
            state[0] = float(h1[0])
            state[1] = _grad0(h1)
        if h2.size:
            state[2] = 0.75 * float(h2[0])
            state[3] = 0.75 * _grad0(h2)
        if b.size:
            state[4] = 0.15 * float(b[0])
            state[5] = 0.15 * _grad0(b)
        if r.size:
            state[6] = 0.10 * float(r[0])
            state[7] = 0.10 * _grad0(r)
        return state

    @staticmethod
    def _bootstrap_observation_prior(bootstrap) -> Dict[str, float]:
        row = dict(bootstrap.global_observation_row or {})
        gain_d = float(row.get("gain_d", 0.6))
        gain_v = float(row.get("gain_v", 0.6))
        gain_m = float(row.get("gain_m", 0.6))
        gain_q = float(row.get("gain_q", 0.2))
        nuis = float(row.get("nuisance_weight", 0.25))
        g_h1 = float(np.clip(0.65 * gain_d + 0.35 * gain_v, 0.25, 1.60))
        g_h2 = float(np.clip(0.25 * gain_d + 0.75 * gain_m, 0.05, 1.60))
        g_b = float(np.clip(0.85 * gain_q, 0.0, 0.80))
        g_r = float(np.clip(0.35 * gain_q + 0.45 * nuis, 0.0, 0.90))
        return {
            "g_osc_signed": g_h1,
            "g_b_signed": g_b,
            "g_r_signed": g_r,
            "g_h1_signed": g_h1,
            "g_h2_signed": g_h2,
            "g_aux_signed": max(g_b, g_r),
            "lag_sec": 0.0,
            "reliability": float(row.get("reliability", 1.0)),
        }

    @staticmethod
    def _bootstrap_quality(adapted, bootstrap, osc_freq: float, wave_freq: float) -> Dict[str, float]:
        details = dict(adapted.diagnostics.get("family_details") or {})
        if details:
            mean_corr = float(np.mean([float(d.get("mean_corr", 0.0)) for d in details.values()]))
            peak_ratio = float(np.mean([float(d.get("peak_ratio_mean", 1.0)) for d in details.values()]))
        else:
            mean_corr = 0.0
            peak_ratio = 1.0
        row = dict(bootstrap.global_observation_row or {})
        reliability = float(row.get("reliability", 1.0))
        nuisance = float(row.get("nuisance_weight", 0.25))
        rel_score = float(np.clip(reliability / (1.0 + reliability), 0.0, 1.0))
        corr_score = float(np.clip(mean_corr, 0.0, 1.0))
        peak_score = float(np.clip((peak_ratio - 1.0) / 4.0, 0.0, 1.0))
        if np.isfinite(osc_freq) and np.isfinite(wave_freq):
            freq_gap = abs(float(osc_freq) - float(wave_freq))
            freq_score = float(np.exp(-0.5 * (freq_gap / 0.03) ** 2))
        else:
            freq_score = 0.0
        nuisance_score = float(np.clip(1.0 - nuisance, 0.0, 1.0))
        quality = float(np.clip(0.30 * rel_score + 0.30 * corr_score + 0.15 * peak_score + 0.15 * freq_score + 0.10 * nuisance_score, 0.0, 1.0))
        return {
            "quality": quality,
            "rel_score": rel_score,
            "corr_score": corr_score,
            "peak_score": peak_score,
            "freq_score": freq_score,
            "nuisance_score": nuisance_score,
        }

    def run(
        self,
        signal: np.ndarray,
        fs: float,
        meta: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        run_meta = dict(meta or {})
        signal_arr = np.asarray(signal, dtype=np.float64)

        # Single-view inputs can still use the adaptive head; it degenerates to
        # a pass-through canonical observation and records the design mode.
        if signal_arr.ndim == 1:
            core_meta = dict(run_meta)
            core_meta["base_method"] = str(core_meta.get("base_method") or "canonical_waveform")
            result = self.core.run(signal_arr, fs, core_meta)
            self._merge_meta(result, {
                "model_version": "parh_resonator_adaptive_v2",
                "adaptive_mode": "single_view_passthrough",
            })
            result["canonical_waveform"] = np.asarray(signal_arr, dtype=np.float64)
            result["canonical_osc"] = np.asarray(signal_arr, dtype=np.float64)
            result["canonical_bundle_d"] = np.asarray(signal_arr, dtype=np.float64)
            result["canonical_bundle_v"] = np.asarray(signal_arr, dtype=np.float64)
            result["canonical_bundle_m"] = np.asarray(signal_arr, dtype=np.float64)
            result["canonical_bundle_q"] = np.asarray(signal_arr, dtype=np.float64)
            return result

        if signal_arr.ndim != 2:
            raise ValueError("parh_resonator_adaptive expects a 1D or 2D input signal")

        families = list(run_meta.get("observation_families") or [])
        if len(families) != int(signal_arr.shape[0]):
            families = [f"channel_{idx}" for idx in range(int(signal_arr.shape[0]))]

        adapted = self.adaptor.adapt(signal_arr, fs, families)
        bootstrap = self.bootstrap.build(adapted.canonical_bundle, adapted.family_params)
        canonical_wave = np.asarray(bootstrap.waveform_drive, dtype=np.float64).reshape(-1)
        canonical_osc = np.asarray(bootstrap.osc_drive, dtype=np.float64).reshape(-1)
        n = int(min(canonical_wave.size, canonical_osc.size))
        canonical_wave = canonical_wave[:n]
        canonical_osc = canonical_osc[:n]

        osc_freq = self._coarse_freq(canonical_osc, fs, run_meta) if n > 0 else float("nan")
        wave_freq = self._coarse_freq(canonical_wave, fs, run_meta) if n > 0 else float("nan")

        core_meta = dict(run_meta)
        core_meta["base_method"] = "canonical_waveform"
        core_meta["observation_families"] = ["canonical_waveform"]
        core_meta["observation_family_semantics"] = [get_observation_family_semantics("canonical_waveform")]
        bootstrap_quality = self._bootstrap_quality(adapted, bootstrap, osc_freq, wave_freq)
        quality_score = float(bootstrap_quality.get("quality", 0.0))
        core_meta["external_initial_state"] = self._bootstrap_initial_state(bootstrap, fs).tolist()
        core_meta["external_initial_state_weight"] = float(np.clip(0.03 + 0.12 * quality_score, 0.03, 0.15))
        core_meta["external_observation_prior"] = self._bootstrap_observation_prior(bootstrap)
        core_meta["external_observation_prior_weight"] = float(np.clip(0.02 + 0.08 * quality_score, 0.02, 0.10))
        core_meta["external_bootstrap_quality"] = bootstrap_quality
        core_meta["external_coarse_candidates"] = []
        if np.isfinite(osc_freq) and osc_freq > 0.0:
            core_meta["external_coarse_candidates"].append({
                "label": "canonical_osc",
                "hz": float(osc_freq),
                "confidence": float(bootstrap.global_observation_row.get("reliability", 1.0)),
            })
        if np.isfinite(wave_freq) and wave_freq > 0.0:
            core_meta["external_coarse_candidates"].append({
                "label": "canonical_waveform",
                "hz": float(wave_freq),
                "confidence": 1.0,
            })

        result = self.core.run(canonical_wave, fs, core_meta)
        family_wave_matrix = []
        family_wave_names = []
        for key, value in adapted.family_waveforms.items():
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size == n:
                family_wave_matrix.append(arr)
                family_wave_names.append(str(key))
        family_osc_matrix = []
        family_osc_names = []
        for key, value in adapted.family_oscillatory.items():
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
            if arr.size == n:
                family_osc_matrix.append(arr)
                family_osc_names.append(str(key))

        self._merge_meta(result, {
            "model_version": "parh_resonator_adaptive_v2",
            "adaptive_mode": "family_multiview_to_canonical",
            "input_observation_families": list(families),
            "family_group_semantics": self._family_group_semantics(families),
            "canonical_observation_family": "canonical_waveform",
            "canonical_osc_freq_hz": float(osc_freq) if np.isfinite(osc_freq) else float("nan"),
            "canonical_wave_freq_hz": float(wave_freq) if np.isfinite(wave_freq) else float("nan"),
            "observation_adaptor": adapted.diagnostics,
            "observation_family_params": adapted.family_params,
            "shared_latent_bootstrap": bootstrap.diagnostics,
            "bootstrap_quality": bootstrap_quality,
        })
        result["canonical_waveform"] = canonical_wave
        result["canonical_osc"] = canonical_osc
        result["canonical_bundle_d"] = np.asarray(adapted.canonical_bundle.get("d", canonical_wave), dtype=np.float64)[:n]
        result["canonical_bundle_v"] = np.asarray(adapted.canonical_bundle.get("v", canonical_osc), dtype=np.float64)[:n]
        result["canonical_bundle_m"] = np.asarray(adapted.canonical_bundle.get("m", canonical_wave), dtype=np.float64)[:n]
        result["canonical_bundle_q"] = np.asarray(adapted.canonical_bundle.get("q", canonical_wave), dtype=np.float64)[:n]
        result["family_params_json"] = json.dumps(adapted.family_params)
        result["shared_latent_bootstrap_json"] = json.dumps(bootstrap.diagnostics)
        result["global_observation_row_json"] = json.dumps(bootstrap.global_observation_row)
        for comp_key, comp_value in bootstrap.latent_components.items():
            result[f"latent_{comp_key}"] = np.asarray(comp_value, dtype=np.float64)[:n]
        if family_wave_matrix:
            result["family_waveforms"] = np.vstack(family_wave_matrix)
            result["family_waveform_names"] = np.asarray(family_wave_names, dtype=object)
        if family_osc_matrix:
            result["family_oscillatory"] = np.vstack(family_osc_matrix)
            result["family_oscillatory_names"] = np.asarray(family_osc_names, dtype=object)
        return result
