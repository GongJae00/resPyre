import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional


def _safe_mkdir(path: str) -> str:
	os.makedirs(path, exist_ok=True)
	return path


@dataclass
class AutoTuneStats:
	count: int = 0
	sum_mad: float = 0.0
	sum_snr: float = 0.0
	sum_freq: float = 0.0
	max_sigma: float = 0.0
	meta: Dict[str, float] = field(default_factory=dict)

	def update(self, mad: float, snr: float, freq: float):
		self.count += 1
		self.sum_mad += float(mad)
		self.sum_snr += float(snr)
		self.sum_freq += float(freq)
		self.max_sigma = max(self.max_sigma, float(mad))

	def to_dict(self) -> Dict:
		return {
			'count': self.count,
			'sum_mad': self.sum_mad,
			'sum_snr': self.sum_snr,
			'sum_freq': self.sum_freq,
			'max_sigma': self.max_sigma,
			'meta': self.meta
		}

	@classmethod
	def from_dict(cls, payload: Dict) -> "AutoTuneStats":
		stats = cls()
		if not isinstance(payload, dict):
			return stats
		stats.count = int(payload.get('count', 0))
		stats.sum_mad = float(payload.get('sum_mad', 0.0))
		stats.sum_snr = float(payload.get('sum_snr', 0.0))
		stats.sum_freq = float(payload.get('sum_freq', 0.0))
		stats.max_sigma = float(payload.get('max_sigma', 0.0))
		stats.meta = dict(payload.get('meta', {}))
		return stats


class AutoTuneRepository:
	"""
	Small helper that stores dataset/method-specific statistics and parameter overrides.
	The files live under runs/autotune/<dataset>/<method>.json and *_stats.json.
	"""

	def __init__(self, base_dir: Optional[str] = None):
		self.base_dir = base_dir or os.environ.get('RESPYRE_AUTOTUNE_DIR', 'runs/autotune')

	def _dataset_dir(self, dataset: str) -> str:
		ds = dataset or 'unknown'
		return _safe_mkdir(os.path.join(self.base_dir, ds.lower()))

	def _param_path(self, dataset: str, method: str) -> str:
		filename = f"{method}.json"
		return os.path.join(self._dataset_dir(dataset), filename)

	def _stats_path(self, dataset: str, method: str) -> str:
		filename = f"{method}_stats.json"
		return os.path.join(self._dataset_dir(dataset), filename)

	def load_params(self, dataset: str, method: str) -> Dict:
		path = self._param_path(dataset, method)
		if not os.path.exists(path):
			return {}
		try:
			with open(path, 'r', encoding='utf-8') as fp:
				payload = json.load(fp)
			if isinstance(payload, dict):
				return payload.get('params', payload)
		except Exception:
			return {}
		return {}

	def record_stats(self, dataset: str, method: str, mad: float, snr: float, freq: float, extra_meta: Optional[Dict] = None):
		path = self._stats_path(dataset, method)
		stats = AutoTuneStats()
		if os.path.exists(path):
			try:
				with open(path, 'r', encoding='utf-8') as fp:
					stats = AutoTuneStats.from_dict(json.load(fp))
			except Exception:
				stats = AutoTuneStats()
		stats.update(mad, snr, freq)
		if isinstance(extra_meta, dict):
			stats.meta.update({k: float(v) for k, v in extra_meta.items() if isinstance(v, (int, float))})
		with open(path, 'w', encoding='utf-8') as fp:
			json.dump(stats.to_dict(), fp, ensure_ascii=False, indent=2)
