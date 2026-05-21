from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import signal as sps

from components.observations.semantics import get_observation_family_semantics


@dataclass
class FamilyObservationLaw:
    family_key: str
    members: List[str]
    reference: str
    gain_d: float
    gain_v: float
    gain_m: float
    gain_q: float
    lag: int
    reliability: float
    nuisance_weight: float
    r_scale: float


@dataclass
class FamilyAdaptationResult:
    canonical_waveform: np.ndarray
    canonical_osc: np.ndarray
    canonical_bundle: Dict[str, np.ndarray]
    family_waveforms: Dict[str, np.ndarray]
    family_oscillatory: Dict[str, np.ndarray]
    family_slow: Dict[str, np.ndarray]
    family_params: Dict[str, Dict[str, float]]
    diagnostics: Dict[str, object]


class FamilyObservationAdaptor:
    """Collapse heterogeneous observation hypotheses into canonical views.

    The current implementation is still deterministic, but it now follows the
    dimension-level design lock more closely:
    - build family-level hypotheses first
    - expose a canonical observation bundle {d, v, m, q}
    - expose family-conditioned observation-law parameters
    - keep the downstream latent estimator interface stable
    """

    FAMILY_ORDER = ("of", "p1d", "dof")

    def __init__(self, f_min: float = 0.08, f_max: float = 0.50):
        self.f_min = float(f_min)
        self.f_max = float(f_max)

    @staticmethod
    def _family_key(name: str) -> str:
        key = str(name or "").strip().lower()
        if key.startswith("of_") or key == "of":
            return "of"
        if key.startswith("profile1d_"):
            return "p1d"
        if key.startswith("dof"):
            return "dof"
        return "other"

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

    def _bandpass(self, x: np.ndarray, fs: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return x
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = sps.detrend(x, type="linear")
        nyq = 0.5 * float(fs)
        lo = max(float(self.f_min), 0.05)
        hi = min(float(self.f_max), nyq - 1e-3)
        if hi > lo and hi < nyq and lo > 0.0:
            b, a = sps.butter(3, [lo / nyq, hi / nyq], btype="bandpass")
            x = sps.filtfilt(b, a, x, method="gust")
        return x

    def _lowpass(self, x: np.ndarray, fs: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return x
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = sps.detrend(x, type="linear")
        nyq = 0.5 * float(fs)
        hi = min(max(0.12, self.f_min), nyq - 1e-3)
        if hi > 0.0 and hi < nyq:
            b, a = sps.butter(2, hi / nyq, btype="lowpass")
            x = sps.filtfilt(b, a, x, method="gust")
        return x

    def _spectral_peak_ratio(self, x: np.ndarray, fs: float) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        min_len = max(16, int(round(2.0 * fs)))
        if x.size < min_len:
            return 1.0
        nperseg = min(x.size, max(64, int(round(fs * 8.0))))
        freqs, psd = sps.welch(x, fs=fs, nperseg=nperseg)
        mask = (freqs >= self.f_min) & (freqs <= self.f_max)
        if not np.any(mask):
            return 1.0
        band = np.asarray(psd[mask], dtype=np.float64)
        peak = float(np.max(band))
        med = float(np.median(band))
        ratio = peak / max(med, 1e-9)
        if not np.isfinite(ratio) or ratio <= 0.0:
            return 1.0
        return ratio

    @staticmethod
    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = int(min(x.size, y.size))
        if n < 8:
            return 0.0
        x = x[:n]
        y = y[:n]
        if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
            return 0.0
        corr = float(np.corrcoef(x, y)[0, 1])
        return corr if np.isfinite(corr) else 0.0

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        scale = FamilyObservationAdaptor._robust_scale(x)
        return np.asarray(x, dtype=np.float64) / max(scale, 1e-6)

    @staticmethod
    def _gradient(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size < 3:
            return np.zeros_like(x)
        return np.gradient(x)

    def _normalize_view(self, sig: np.ndarray, fs: float) -> np.ndarray:
        return self._normalize(self._bandpass(sig, fs))

    def _slow_view(self, sig: np.ndarray, fs: float) -> np.ndarray:
        return self._normalize(self._lowpass(sig, fs))

    def _weight_from_semantics(self, family: str, *, rate_mode: bool) -> float:
        sem = get_observation_family_semantics(family)
        if rate_mode:
            base = 1.0 if bool(sem.get("rate_primary")) else 0.55
            if bool(sem.get("helper_heavy")):
                base += 0.15
        else:
            base = 1.0 if bool(sem.get("waveform_primary")) else 0.55
            nuisance = str(sem.get("nuisance_risk", "medium")).lower()
            if nuisance == "high":
                base *= 0.60
            elif nuisance == "medium":
                base *= 0.85
        return float(max(base, 0.10))

    def _family_param_template(self, family_key: str, members: Sequence[str], reliability: float, disagreement: float) -> FamilyObservationLaw:
        rel = float(np.clip(reliability, 0.05, 1.50))
        dis = float(np.clip(disagreement, 0.0, 1.0))
        nuisance = 0.25 + 0.55 * dis
        lag = 0
        if family_key == 'of':
            gain_d, gain_v, gain_m, gain_q = 0.45, 1.00, 0.35, 0.10
            nuisance = 0.30 + 0.40 * dis
        elif family_key == 'p1d':
            quad_like = any(('quadratic' in m) or ('cubic' in m) or ('consensus' in m) for m in members)
            gain_d = 1.00
            gain_v = 0.20
            gain_m = 0.95 if quad_like else 0.40
            gain_q = 0.45
            nuisance = 0.15 + 0.30 * dis
        elif family_key == 'dof':
            gain_d, gain_v, gain_m, gain_q = 0.30, 0.25, 0.15, 0.20
            nuisance = 0.55 + 0.35 * dis
        else:
            gain_d, gain_v, gain_m, gain_q = 0.50, 0.50, 0.35, 0.25
            nuisance = 0.35 + 0.35 * dis
        r_scale = float(1.0 / max(rel, 1e-3))
        return FamilyObservationLaw(
            family_key=str(family_key),
            members=list(members),
            reference=str(members[0]) if members else str(family_key),
            gain_d=float(gain_d),
            gain_v=float(gain_v),
            gain_m=float(gain_m),
            gain_q=float(gain_q),
            lag=int(lag),
            reliability=float(rel),
            nuisance_weight=float(np.clip(nuisance, 0.05, 1.50)),
            r_scale=float(np.clip(r_scale, 0.5, 4.0)),
        )

    def adapt(
        self,
        signals: np.ndarray,
        fs: float,
        families: Sequence[str],
    ) -> FamilyAdaptationResult:
        signal_arr = np.asarray(signals, dtype=np.float64)
        if signal_arr.ndim != 2:
            raise ValueError("FamilyObservationAdaptor expects signals with shape [C, T]")
        n_channels, n = signal_arr.shape
        if len(families) != n_channels:
            raise ValueError("signals/families length mismatch")
        if n == 0:
            empty = np.array([], dtype=np.float64)
            return FamilyAdaptationResult(
                canonical_waveform=empty,
                canonical_osc=empty,
                canonical_bundle={"d": empty, "v": empty, "m": empty, "q": empty},
                family_waveforms={},
                family_oscillatory={},
                family_slow={},
                family_params={},
                diagnostics={"n_channels": int(n_channels), "n_frames": 0},
            )

        normalized: Dict[str, np.ndarray] = {}
        slow_views: Dict[str, np.ndarray] = {}
        peak_ratio: Dict[str, float] = {}
        family_views: Dict[str, List[str]] = {}
        for idx, family in enumerate(families):
            key = str(family)
            normalized[key] = self._normalize_view(signal_arr[idx], fs)
            slow_views[key] = self._slow_view(signal_arr[idx], fs)
            peak_ratio[key] = self._spectral_peak_ratio(normalized[key], fs)
            family_views.setdefault(self._family_key(key), []).append(key)

        family_waveforms: Dict[str, np.ndarray] = {}
        family_osc: Dict[str, np.ndarray] = {}
        family_slow: Dict[str, np.ndarray] = {}
        family_params: Dict[str, Dict[str, float]] = {}
        family_diag: Dict[str, object] = {}

        for family_key, members in family_views.items():
            member_order = sorted(members, key=lambda name: peak_ratio.get(name, 1.0), reverse=True)
            ref_name = member_order[0]
            ref = normalized[ref_name]
            aligned: Dict[str, np.ndarray] = {}
            waveform_scores: Dict[str, float] = {}
            rate_scores: Dict[str, float] = {}
            slow_scores: Dict[str, float] = {}
            member_corrs: Dict[str, float] = {}
            for name in member_order:
                sig = normalized[name]
                corr = self._safe_corr(sig, ref)
                if corr < 0.0:
                    sig = -sig
                    corr = -corr
                aligned[name] = sig
                member_corrs[name] = corr
                peak = float(np.clip(peak_ratio.get(name, 1.0), 0.25, 8.0))
                waveform_scores[name] = self._weight_from_semantics(name, rate_mode=False) * (0.60 + 0.40 * corr) * peak
                rate_scores[name] = self._weight_from_semantics(name, rate_mode=True) * (0.45 + 0.55 * peak)
                slow_scores[name] = max(0.25, 0.70 + 0.30 * corr)

            wave_sum = max(sum(waveform_scores.values()), 1e-6)
            rate_sum = max(sum(rate_scores.values()), 1e-6)
            slow_sum = max(sum(slow_scores.values()), 1e-6)
            family_wave = sum((waveform_scores[name] / wave_sum) * aligned[name] for name in member_order)
            family_rate = sum((rate_scores[name] / rate_sum) * aligned[name] for name in member_order)
            family_s = sum((slow_scores[name] / slow_sum) * slow_views[name] for name in member_order)
            family_waveforms[family_key] = np.asarray(family_wave, dtype=np.float64)
            family_osc[family_key] = np.asarray(family_rate, dtype=np.float64)
            family_slow[family_key] = np.asarray(family_s, dtype=np.float64)

            reliability = float(np.mean([peak_ratio[m] for m in member_order])) * (0.55 + 0.45 * float(np.mean(list(member_corrs.values()))))
            disagreement = 1.0 - float(np.mean(list(member_corrs.values())))
            law = self._family_param_template(family_key, member_order, reliability=reliability, disagreement=disagreement)
            law.reference = ref_name
            family_params[family_key] = {
                'gain_d': law.gain_d,
                'gain_v': law.gain_v,
                'gain_m': law.gain_m,
                'gain_q': law.gain_q,
                'lag': float(law.lag),
                'reliability': law.reliability,
                'nuisance_weight': law.nuisance_weight,
                'R_scale': law.r_scale,
            }
            family_diag[family_key] = {
                'members': member_order,
                'reference': ref_name,
                'mean_corr': float(np.mean(list(member_corrs.values()))),
                'peak_ratio_mean': float(np.mean([peak_ratio[m] for m in member_order])),
                'waveform_weights': {name: float(waveform_scores[name] / wave_sum) for name in member_order},
                'rate_weights': {name: float(rate_scores[name] / rate_sum) for name in member_order},
                'slow_weights': {name: float(slow_scores[name] / slow_sum) for name in member_order},
                'observation_law': family_params[family_key],
            }

        d_terms = []
        v_terms = []
        m_terms = []
        q_terms = []
        d_w = []
        v_w = []
        m_w = []
        q_w = []
        family_bundle_weights: Dict[str, Dict[str, float]] = {}
        for family_key in family_waveforms:
            params = family_params[family_key]
            rel = params['reliability']
            nuis = params['nuisance_weight']
            safe_rel = rel / max(1.0 + nuis, 1e-6)
            d_terms.append(family_waveforms[family_key])
            d_w.append(max(params['gain_d'] * safe_rel, 1e-6))
            v_terms.append(self._gradient(family_osc[family_key]))
            v_w.append(max(params['gain_v'] * safe_rel, 1e-6))
            m_terms.append(family_osc[family_key])
            m_w.append(max(params['gain_m'] * safe_rel, 1e-6))
            q_terms.append(family_slow[family_key])
            q_w.append(max(params['gain_q'] * safe_rel, 1e-6))
            family_bundle_weights[family_key] = {
                'd': float(d_w[-1]),
                'v': float(v_w[-1]),
                'm': float(m_w[-1]),
                'q': float(q_w[-1]),
            }

        def _mix(terms: List[np.ndarray], weights: List[float]) -> np.ndarray:
            w = np.asarray(weights, dtype=np.float64)
            w = w / max(float(np.sum(w)), 1e-6)
            out = np.zeros((n,), dtype=np.float64)
            for wi, term in zip(w, terms):
                out += float(wi) * np.asarray(term, dtype=np.float64).reshape(-1)[:n]
            return np.asarray(out, dtype=np.float64)

        d_t = self._normalize(_mix(d_terms, d_w))
        v_t = self._normalize(_mix(v_terms, v_w))
        m_t = self._normalize(_mix(m_terms, m_w))
        q_t = self._normalize(_mix(q_terms, q_w))

        canonical_waveform = self._normalize(0.70 * d_t + 0.20 * m_t + 0.10 * q_t)
        canonical_osc = self._normalize(0.60 * v_t + 0.40 * m_t)

        diagnostics: Dict[str, object] = {
            'n_channels': int(n_channels),
            'n_frames': int(n),
            'view_families': list(families),
            'family_groups': {k: list(v) for k, v in family_views.items()},
            'peak_ratio_by_view': {name: float(peak_ratio[name]) for name in families},
            'family_details': family_diag,
            'canonical_bundle_weights': family_bundle_weights,
            'canonical_output_mix': {'waveform': {'d': 0.70, 'm': 0.20, 'q': 0.10}, 'osc': {'v': 0.60, 'm': 0.40}},
        }
        return FamilyAdaptationResult(
            canonical_waveform=np.asarray(canonical_waveform, dtype=np.float64),
            canonical_osc=np.asarray(canonical_osc, dtype=np.float64),
            canonical_bundle={'d': d_t, 'v': v_t, 'm': m_t, 'q': q_t},
            family_waveforms={k: np.asarray(v, dtype=np.float64) for k, v in family_waveforms.items()},
            family_oscillatory={k: np.asarray(v, dtype=np.float64) for k, v in family_osc.items()},
            family_slow={k: np.asarray(v, dtype=np.float64) for k, v in family_slow.items()},
            family_params=family_params,
            diagnostics=diagnostics,
        )
