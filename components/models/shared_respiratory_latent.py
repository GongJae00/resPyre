from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SharedLatentBootstrapResult:
    waveform_drive: np.ndarray
    osc_drive: np.ndarray
    latent_components: Dict[str, np.ndarray]
    observation_rows: Dict[str, Dict[str, float]]
    global_observation_row: Dict[str, float]
    diagnostics: Dict[str, object]


class SharedRespiratoryLatentBootstrap:
    """Convert canonical observation bundle into latent-aligned bootstrap drives.

    This remains a lightweight front-end, but it now follows the state design
    more closely:
    - build a global observation-row summary from family-conditioned laws
    - construct latent-aligned bootstrap drives for h1 / h2 / b / r
    - expose both latent components and the final drives used by the resonator
    """

    @staticmethod
    def _robust_scale(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return 1.0
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = float(np.std(x))
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = 1.0
        return sigma

    @classmethod
    def _normalize(cls, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return x / max(cls._robust_scale(x), 1e-6)

    @staticmethod
    def _weighted_mean(values, weights, default):
        vals = np.asarray(list(values), dtype=np.float64)
        w = np.asarray(list(weights), dtype=np.float64)
        if vals.size == 0 or w.size == 0 or float(np.sum(w)) <= 0.0:
            return float(default)
        return float(np.sum(vals * w) / max(float(np.sum(w)), 1e-6))

    def build(
        self,
        canonical_bundle: Dict[str, np.ndarray],
        family_params: Dict[str, Dict[str, float]],
    ) -> SharedLatentBootstrapResult:
        d = self._normalize(canonical_bundle.get('d', np.array([], dtype=np.float64)))
        v = self._normalize(canonical_bundle.get('v', np.array([], dtype=np.float64)))
        m = self._normalize(canonical_bundle.get('m', np.array([], dtype=np.float64)))
        q = self._normalize(canonical_bundle.get('q', np.array([], dtype=np.float64)))

        observation_rows: Dict[str, Dict[str, float]] = {}
        family_weights = {}
        for family_key, params in family_params.items():
            rel = float(params.get('reliability', 0.0))
            nuis = float(params.get('nuisance_weight', 0.0))
            safe_weight = rel / max(1.0 + nuis, 1e-6)
            family_weights[str(family_key)] = float(max(safe_weight, 1e-6))
            observation_rows[str(family_key)] = {
                'gain_d': float(params.get('gain_d', 0.0)),
                'gain_v': float(params.get('gain_v', 0.0)),
                'gain_m': float(params.get('gain_m', 0.0)),
                'gain_q': float(params.get('gain_q', 0.0)),
                'lag': float(params.get('lag', 0.0)),
                'reliability': rel,
                'nuisance_weight': nuis,
                'R_scale': float(params.get('R_scale', 1.0)),
                'row_weight': family_weights[str(family_key)],
            }

        row_weights = list(family_weights.values())
        global_observation_row = {
            'gain_d': self._weighted_mean((p['gain_d'] for p in observation_rows.values()), row_weights, 0.6),
            'gain_v': self._weighted_mean((p['gain_v'] for p in observation_rows.values()), row_weights, 0.6),
            'gain_m': self._weighted_mean((p['gain_m'] for p in observation_rows.values()), row_weights, 0.6),
            'gain_q': self._weighted_mean((p['gain_q'] for p in observation_rows.values()), row_weights, 0.2),
            'reliability': self._weighted_mean((p['reliability'] for p in observation_rows.values()), row_weights, 1.0),
            'nuisance_weight': self._weighted_mean((p['nuisance_weight'] for p in observation_rows.values()), row_weights, 0.25),
            'R_scale': self._weighted_mean((p['R_scale'] for p in observation_rows.values()), row_weights, 1.0),
        }

        gd = float(global_observation_row['gain_d'])
        gv = float(global_observation_row['gain_v'])
        gm = float(global_observation_row['gain_m'])
        gq = float(global_observation_row['gain_q'])
        nuis = float(global_observation_row['nuisance_weight'])

        h1_drive = self._normalize((0.30 + 0.20 * gd) * d + (0.40 + 0.25 * gv) * v + (0.15 + 0.10 * gm) * m)
        h2_drive = self._normalize((0.20 + 0.10 * gd) * d + (0.55 + 0.20 * gm) * m - 0.10 * v)
        b_drive = self._normalize((0.80 + 0.20 * gq) * q)
        residual_seed = d - (0.60 * h1_drive + 0.25 * h2_drive + 0.15 * b_drive)
        r_drive = self._normalize((1.0 + 0.25 * nuis) * residual_seed)

        osc_drive = self._normalize(h1_drive)
        waveform_drive = self._normalize(0.55 * h1_drive + 0.25 * h2_drive + 0.12 * b_drive + 0.08 * r_drive)

        latent_components = {
            'h1_drive': np.asarray(h1_drive, dtype=np.float64),
            'h2_drive': np.asarray(h2_drive, dtype=np.float64),
            'b_drive': np.asarray(b_drive, dtype=np.float64),
            'r_drive': np.asarray(r_drive, dtype=np.float64),
        }

        diagnostics = {
            'bundle_norms': {
                'd': float(self._robust_scale(d)),
                'v': float(self._robust_scale(v)),
                'm': float(self._robust_scale(m)),
                'q': float(self._robust_scale(q)),
            },
            'family_weight_summary': family_weights,
            'global_observation_row': global_observation_row,
            'latent_component_mix': {
                'h1_drive': {'d': 0.30 + 0.20 * gd, 'v': 0.40 + 0.25 * gv, 'm': 0.15 + 0.10 * gm},
                'h2_drive': {'d': 0.20 + 0.10 * gd, 'm': 0.55 + 0.20 * gm, 'v': -0.10},
                'b_drive': {'q': 0.80 + 0.20 * gq},
                'r_drive': {'seed_scale': 1.0 + 0.25 * nuis},
                'waveform_drive': {'h1': 0.55, 'h2': 0.25, 'b': 0.12, 'r': 0.08},
                'osc_drive': {'h1': 1.0},
            },
            'family_observation_rows': observation_rows,
        }
        return SharedLatentBootstrapResult(
            waveform_drive=np.asarray(waveform_drive, dtype=np.float64),
            osc_drive=np.asarray(osc_drive, dtype=np.float64),
            latent_components=latent_components,
            observation_rows=observation_rows,
            global_observation_row=global_observation_row,
            diagnostics=diagnostics,
        )
