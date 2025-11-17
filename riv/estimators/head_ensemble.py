import copy
import json
from typing import Dict, List, Optional

import numpy as np

from .oscillator_heads import OscillatorParams, build_head


class OscillatorEnsemble:
    """Runs multiple oscillator heads and combines their outputs."""

    def __init__(self, head_defs: List[Dict], preproc_cfg: Dict):
        if not head_defs:
            raise ValueError("Ensemble configuration requires at least one head")
        self.components = []
        for entry in head_defs:
            if isinstance(entry, str):
                head_name = entry
                params_override = {}
                weight = 1.0
            else:
                head_name = entry.get('name') or entry.get('head')
                if not head_name:
                    continue
                params_override = entry.get('params') or {}
                weight = float(entry.get('weight', 1.0))
            osc_params = OscillatorParams(**params_override) if params_override else OscillatorParams()
            head_instance = build_head(head_name, params=osc_params)
            setattr(head_instance, "preproc_cfg", copy.deepcopy(preproc_cfg))
            self.components.append({
                "name": head_name,
                "instance": head_instance,
                "weight": weight
            })
        if not self.components:
            raise ValueError("Failed to initialise ensemble components")

    def run(self, signal: np.ndarray, fs: float, meta: Dict) -> Dict:
        component_results = []
        for comp in self.components:
            head = comp['instance']
            head_meta = dict(meta)
            head_meta['ensemble_component'] = comp['name']
            result = head.run(signal, fs, head_meta)
            meta_parsed = {}
            try:
                meta_parsed = json.loads(result['meta'])
            except Exception:
                meta_parsed = {}
            reliability = float(meta_parsed.get('reliability_score', 0.5))
            if not np.isfinite(reliability) or reliability <= 0:
                reliability = 0.1
            component_results.append({
                "name": comp['name'],
                "result": result,
                "weight": max(comp['weight'], 1e-6),
                "reliability": reliability
            })

        weights = np.asarray([c['weight'] * c['reliability'] for c in component_results], dtype=np.float64)
        weights = weights / max(np.sum(weights), 1e-9)
        tracks = np.vstack([c['result']['track_hz'] for c in component_results])
        combined_track = np.sum(weights[:, None] * tracks, axis=0)

        best_idx = int(np.argmax(weights))
        combined_signal = component_results[best_idx]['result']['signal_hat']

        meta_payload = dict(meta)
        components_meta = []
        for comp, w in zip(component_results, weights):
            comp_meta = {}
            try:
                comp_meta = json.loads(comp['result']['meta'])
            except Exception:
                comp_meta = {}
            components_meta.append({
                "head": comp['name'],
                "weight": float(w),
                "reliability": float(comp['reliability']),
                "meta": comp_meta
            })
        meta_payload['ensemble_components'] = components_meta
        meta_payload.setdefault('reliability_score', float(np.max(weights)))
        return {
            "signal_hat": combined_signal,
            "track_hz": combined_track,
            "rr_hz": float(np.nanmedian(combined_track)),
            "rr_bpm": float(np.nanmedian(combined_track)) * 60.0,
            "meta": json.dumps(meta_payload),
            "components": component_results
        }
