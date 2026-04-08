
import os

import numpy as np
import cv2 as cv
from scipy import signal as sps
from components.observations.motion import OF, DoF, profile1D


def _trial_dir(data):
	video_path = data.get('video_path')
	if not video_path:
		return None
	trial_dir = os.path.dirname(video_path)
	return trial_dir or None


def _get_obs_signal_cache(data):
	cache = data.get('_obs_signal_cache')
	if isinstance(cache, dict):
		return cache
	cache = {}
	data['_obs_signal_cache'] = cache
	return cache


def _load_obs_cache(data, filename):
	mem_cache = _get_obs_signal_cache(data)
	cached = mem_cache.get(filename)
	if isinstance(cached, np.ndarray):
		return cached
	trial_dir = _trial_dir(data)
	if not trial_dir:
		return None
	cache_path = os.path.join(trial_dir, filename)
	if not os.path.exists(cache_path):
		return None
	try:
		loaded = np.load(cache_path, allow_pickle=False)
		mem_cache[filename] = loaded
		return loaded
	except Exception:
		return None


def _save_obs_cache(data, filename, arr):
	mem_cache = _get_obs_signal_cache(data)
	mem_cache[filename] = np.asarray(arr, dtype=np.float32)
	trial_dir = _trial_dir(data)
	if not trial_dir:
		return None
	try:
		cache_path = os.path.join(trial_dir, filename)
		np.save(cache_path, np.asarray(arr, dtype=np.float32))
		return cache_path
	except Exception:
		return None


def _get_gray_chest_rois(data):
	cached = data.get('_gray_chest_rois')
	if isinstance(cached, list) and cached:
		return cached
	rois = data.get('chest_rois') or []
	if not rois:
		return []
	g_rois = []
	for roi in rois:
		arr = np.asarray(roi)
		if arr.ndim == 2:
			g_rois.append(arr)
		elif arr.ndim == 3:
			g_rois.append(cv.cvtColor(arr, cv.COLOR_RGB2GRAY))
		else:
			g_rois.append(arr.reshape(-1))
	data['_gray_chest_rois'] = g_rois
	return g_rois

class MethodBase:
	def __init__(self):
		self.name = ''
		self.win_size = 30
		self.data_type = ''

	def process(self, data):
		# This class can be used to process either videos or ROIs
		raise NotImplementedError("Subclasses must implement process method")

	def get_runtime_meta(self):
		return {}

class OF_Model(MethodBase):

	def __init__(self):
		super().__init__()
		# Canonical base method name used across configs/results.
		self.name = 'of_farneback'
		self.data_type = 'chest'

	def process(self, data):
		cached = _load_obs_cache(data, "obs_of.npy")
		if cached is not None:
			return cached

		g_rois = _get_gray_chest_rois(data)

		# estimate OF
		of, _ = OF(g_rois, data['fps'])
		_save_obs_cache(data, "obs_of.npy", of)
		return of


class OFDisplacementBridge_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'of_disp_bridge'
		self.data_type = 'chest'
		self.of_model = OF_Model()

	def process(self, data):
		fs = float(data.get('fps', 20.0))
		cached = _load_obs_cache(data, "obs_of_bridge.npy")
		if cached is not None:
			return cached
		of = np.asarray(self.of_model.process(data), dtype=np.float64).reshape(-1)
		of_bridge = _of_to_displacement_bridge(of, fs)
		_save_obs_cache(data, "obs_of_bridge.npy", of_bridge)
		return of_bridge

class DoF_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'DoF'
		self.data_type = 'chest'

	def process(self, data):
		cached = _load_obs_cache(data, "obs_dof.npy")
		if cached is not None:
			return cached

		g_rois = _get_gray_chest_rois(data)

		# estimate DoF
		dof, _ = DoF(g_rois, data['fps'])
		_save_obs_cache(data, "obs_dof.npy", dof)
		return dof

class profile1D_Model(MethodBase):

	def __init__(self, interp_type='quadratic'):
		super().__init__()
		self.name = 'profile1D ' + interp_type 
		self.data_type = 'chest'
		self.interp_type = interp_type

	def process(self, data):
		suffix = self.interp_type
		if suffix == 'quadratic': suffix = 'quad'
		cache_name = f"obs_p1d_{suffix}.npy"
		cached = _load_obs_cache(data, cache_name)
		if cached is not None:
			return cached

		g_rois = _get_gray_chest_rois(data)

		# estimate profile1D
		profile, _ = profile1D(g_rois, data['fps'], self.interp_type)
		_save_obs_cache(data, cache_name, profile)
		return profile


def _of_to_displacement_bridge(of: np.ndarray, fs: float) -> np.ndarray:
	of = np.asarray(of, dtype=np.float64).reshape(-1)
	if of.size == 0:
		return of
	of = np.nan_to_num(of, nan=0.0, posinf=0.0, neginf=0.0)
	of_centered = of - float(np.median(of))
	of_disp = np.cumsum(of_centered) / max(float(fs), 1e-6)
	if of_disp.size >= 8:
		of_disp = sps.detrend(of_disp, type='linear')
	return of_disp.astype(np.float64, copy=False)


class OFP1DQuadraticPair_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'pair_of_p1d_quadratic'
		self.data_type = 'chest'
		self.component_names = ['of_farneback', 'profile1d_quadratic']
		self.of_model = OF_Model()
		self.p1d_model = profile1D_Model('quadratic')

	def process(self, data):
		of = np.asarray(self.of_model.process(data), dtype=np.float64).reshape(-1)
		p1d = np.asarray(self.p1d_model.process(data), dtype=np.float64).reshape(-1)
		n = int(min(of.size, p1d.size))
		if n <= 0:
			return np.zeros((2, 0), dtype=np.float64)
		return np.vstack([of[:n], p1d[:n]])


class AssistOFP1DQuadratic_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'assist_of_p1d_quadratic'
		self.data_type = 'chest'
		self.component_names = ['profile1d_quadratic']
		self.assistant_component_names = ['of_farneback']
		self.primary_family = 'profile1d_quadratic'
		self.of_model = OF_Model()
		self.p1d_model = profile1D_Model('quadratic')
		self._last_runtime_meta = {}

	def process(self, data):
		of = np.asarray(self.of_model.process(data), dtype=np.float64).reshape(-1)
		p1d = np.asarray(self.p1d_model.process(data), dtype=np.float64).reshape(-1)
		n = int(min(of.size, p1d.size))
		assistant_policy = str(
			os.environ.get('RESPYRE_PARH_ASSISTANT_POLICY', 'of_rate_assistant_v2')
		).strip().lower() or 'of_rate_assistant_v2'
		if n <= 0:
			self._last_runtime_meta = {
				'assistant_policy_runtime': assistant_policy,
				'assistant_observation_families_runtime': ['of_farneback'],
				'assistant_signals_runtime': {'of_farneback': np.array([], dtype=np.float64)},
				'primary_observation_family_runtime': 'profile1d_quadratic',
			}
			return np.array([], dtype=np.float64)
		self._last_runtime_meta = {
			'assistant_policy_runtime': assistant_policy,
			'assistant_observation_families_runtime': ['of_farneback'],
			'assistant_signals_runtime': {'of_farneback': of[:n].astype(np.float64, copy=False)},
			'primary_observation_family_runtime': 'profile1d_quadratic',
		}
		return p1d[:n]

	def get_runtime_meta(self):
		return dict(self._last_runtime_meta)


class AssistOFBridgeOF_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'assist_ofbridge_of'
		self.data_type = 'chest'
		self.component_names = ['of_farneback']
		self.assistant_component_names = ['of_disp_bridge']
		self.primary_family = 'of_farneback'
		self.of_model = OF_Model()
		self.of_bridge_model = OFDisplacementBridge_Model()
		self._last_runtime_meta = {}

	def process(self, data):
		of = np.asarray(self.of_model.process(data), dtype=np.float64).reshape(-1)
		of_bridge = np.asarray(self.of_bridge_model.process(data), dtype=np.float64).reshape(-1)
		n = int(min(of.size, of_bridge.size))
		assistant_policy = str(
			os.environ.get('RESPYRE_PARH_ASSISTANT_POLICY', 'of_rate_assistant_v2')
		).strip().lower() or 'of_rate_assistant_v2'
		if n <= 0:
			self._last_runtime_meta = {
				'assistant_policy_runtime': assistant_policy,
				'assistant_observation_families_runtime': ['of_disp_bridge'],
				'assistant_signals_runtime': {'of_disp_bridge': np.array([], dtype=np.float64)},
				'primary_observation_family_runtime': 'of_farneback',
			}
			return np.array([], dtype=np.float64)
		self._last_runtime_meta = {
			'assistant_policy_runtime': assistant_policy,
			'assistant_observation_families_runtime': ['of_disp_bridge'],
			'assistant_signals_runtime': {'of_disp_bridge': of_bridge[:n].astype(np.float64, copy=False)},
			'primary_observation_family_runtime': 'of_farneback',
		}
		return of[:n]

	def get_runtime_meta(self):
		return dict(self._last_runtime_meta)


class AssistOFBridgeP1DQuadratic_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'assist_ofbridge_p1d_quadratic'
		self.data_type = 'chest'
		self.component_names = ['profile1d_quadratic']
		self.assistant_component_names = ['of_disp_bridge']
		self.primary_family = 'profile1d_quadratic'
		self.of_bridge_model = OFDisplacementBridge_Model()
		self.p1d_model = profile1D_Model('quadratic')
		self._last_runtime_meta = {}

	def process(self, data):
		of_bridge = np.asarray(self.of_bridge_model.process(data), dtype=np.float64).reshape(-1)
		p1d = np.asarray(self.p1d_model.process(data), dtype=np.float64).reshape(-1)
		n = int(min(of_bridge.size, p1d.size))
		assistant_policy = str(
			os.environ.get('RESPYRE_PARH_ASSISTANT_POLICY', 'of_rate_assistant_v2')
		).strip().lower() or 'of_rate_assistant_v2'
		if n <= 0:
			self._last_runtime_meta = {
				'assistant_policy_runtime': assistant_policy,
				'assistant_observation_families_runtime': ['of_disp_bridge'],
				'assistant_signals_runtime': {'of_disp_bridge': np.array([], dtype=np.float64)},
				'primary_observation_family_runtime': 'profile1d_quadratic',
			}
			return np.array([], dtype=np.float64)
		self._last_runtime_meta = {
			'assistant_policy_runtime': assistant_policy,
			'assistant_observation_families_runtime': ['of_disp_bridge'],
			'assistant_signals_runtime': {'of_disp_bridge': of_bridge[:n].astype(np.float64, copy=False)},
			'primary_observation_family_runtime': 'profile1d_quadratic',
		}
		return p1d[:n]

	def get_runtime_meta(self):
		return dict(self._last_runtime_meta)


class FusionOFP1DQuadratic_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'fusion_of_p1d_quadratic'
		self.data_type = 'chest'
		self.component_names = ['of_farneback', 'profile1d_quadratic']
		self.of_model = OF_Model()
		self.p1d_model = profile1D_Model('quadratic')

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

	@staticmethod
	def _spectral_peak_ratio(x: np.ndarray, fs: float, f_min: float = 0.08, f_max: float = 0.5) -> float:
		x = np.asarray(x, dtype=np.float64).reshape(-1)
		if x.size < max(16, int(round(2.0 * fs))):
			return 1.0
		nperseg = min(x.size, max(64, int(round(fs * 8.0))))
		freqs, psd = sps.welch(x, fs=fs, nperseg=nperseg)
		mask = (freqs >= float(f_min)) & (freqs <= float(f_max))
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
	def _bandpass(x: np.ndarray, fs: float, f_min: float = 0.08, f_max: float = 0.5) -> np.ndarray:
		x = np.asarray(x, dtype=np.float64).reshape(-1)
		if x.size == 0:
			return x
		x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
		x = sps.detrend(x, type='linear')
		nyq = 0.5 * float(fs)
		lo = max(float(f_min), 0.05)
		hi = min(float(f_max), nyq - 1e-3)
		if hi > lo and hi < nyq and lo > 0.0:
			b, a = sps.butter(3, [lo / nyq, hi / nyq], btype='bandpass')
			x = sps.filtfilt(b, a, x, method='gust')
		return x

	def process(self, data):
		fs = float(data.get('fps', 20.0))
		of = np.asarray(self.of_model.process(data), dtype=np.float64).reshape(-1)
		p1d = np.asarray(self.p1d_model.process(data), dtype=np.float64).reshape(-1)
		n = int(min(of.size, p1d.size))
		if n <= 0:
			return np.array([], dtype=np.float64)
		of = of[:n]
		p1d = p1d[:n]

		# OF is closer to a motion-velocity proxy; integrate after demeaning to
		# obtain a displacement-like surrogate before simple scalar fusion.
		of_centered = of - float(np.median(of))
		of_disp = np.cumsum(of_centered) / max(fs, 1e-6)
		of_bp = self._bandpass(of_disp, fs)
		p1d_bp = self._bandpass(p1d, fs)

		of_scale = self._robust_scale(of_bp)
		p1d_scale = self._robust_scale(p1d_bp)
		of_z = of_bp / of_scale
		p1d_z = p1d_bp / p1d_scale

		if of_z.size > 8 and p1d_z.size == of_z.size:
			corr = float(np.corrcoef(of_z, p1d_z)[0, 1])
			if np.isfinite(corr) and corr < 0.0:
				of_z = -of_z

		w_of = self._spectral_peak_ratio(of_z, fs)
		w_p1d = self._spectral_peak_ratio(p1d_z, fs)
		w_sum = max(w_of + w_p1d, 1e-6)
		w_of /= w_sum
		w_p1d /= w_sum
		return (w_of * of_z + w_p1d * p1d_z).astype(np.float64)
