import os
import argparse
import json
import hashlib
import shutil
import copy
import csv
from pathlib import Path
import numpy as np
from scipy import signal
import pickle
import utils
import errors
import cv2 as cv
from tqdm import tqdm
from functools import wraps
import time
import warnings
import tempfile
import contextlib
from collections import defaultdict
from config_loader import load_config

# Profiling registry (populated by timed_step when enabled)
_STEP_PROF = defaultdict(lambda: {"calls": 0, "total": 0.0})
_PROFILE_STEPS = True

PRIMARY_METRICS = ['RMSE', 'MAE', 'MAPE', 'PCC', 'CCC']

# Datasets class definitions

class DatasetBase:
	def __init__(self):
		repo_root = Path(__file__).resolve().parent
		custom_root = os.environ.get('RESPIRE_DATA_DIR')
		if custom_root:
			base = Path(custom_root).expanduser()
		else:
			base = repo_root / 'dataset'
		self.data_dir = base.resolve()
		self.roi_params = {
			'chest': {},
			'face': {}
		}

	def resolve(self, *parts):
		return os.path.join(str(self.data_dir), *parts)

	def load_dataset(self):
		raise NotImplementedError("Subclasses must implement load_datset method")

	def configure(self, cfg=None):
		"""Apply dataset-specific configuration (ROI params, etc.)."""
		if not cfg:
			return
		roi_cfg = cfg.get('roi', {}) if isinstance(cfg, dict) else {}
		for region, params in roi_cfg.items():
			self.roi_params.setdefault(region, {}).update(params)

class BP4D(DatasetBase):
	def __init__(self):
		super().__init__()
		self.name = 'bp4d'
		self.path = self.resolve('BP4Ddef') + os.sep
		self.fs_gt = 1000
		self.data = [] 

	def load_dataset(self):

		print('\nLoading dataset ' + self.name + '...')
		for sub in utils.sort_nicely(os.listdir(self.path)):
			sub_path = os.path.join(self.path, sub)
			if not os.path.isdir(sub_path):
				continue

			for trial in utils.sort_nicely(os.listdir(sub_path)):
				trial_path = os.path.join(sub_path, trial)
				if not os.path.isdir(trial_path):
					continue
				video_path = os.path.join(trial_path, 'vid.avi')

				if os.path.exists(video_path):
					d = {}
					d['video_path'] = video_path
					d['subject'] = sub
					d['trial'] = trial
					d['chest_rois'] = []
					d['face_rois'] = []
					d['rppg_obj'] = []
					d['gt'] = self.load_gt(trial_path)
					self.data.append(d)

		print('%d items loaded!' % len(self.data))

	def load_gt(self, trial_path):
		#Load GT
		gt = np.loadtxt(trial_path + "/Resp_Volts.txt")
		return gt

	def extract_ROI(self, video_path, region='chest'):
		if region == 'chest':
			params = self.roi_params.get('chest', {})
			mp_complexity = params.get('mp_complexity', 1)
			skip_rate = params.get('skip_rate', 10)
			rois, _, _ = utils.get_chest_ROI(video_path, self.name, mp_complexity=mp_complexity, skip_rate=skip_rate)
		elif region == 'face':
			params = self.roi_params.get('face', {})
			rois = utils.get_face_ROI(video_path, **params) if params else utils.get_face_ROI(video_path)
		else:
			rois = []
		return rois

	def extract_rppg(self, video_path, method='cpu_CHROM'):
		from riv.resp_from_rPPG import RR_from_rPPG

		rppg_obj = RR_from_rPPG(video_path, method=method)
		rppg_obj.get_rPPG()
		return rppg_obj

class COHFACE(DatasetBase):
	def __init__(self):
		super().__init__()
		self.name = 'cohface'
		self.path = self.resolve('COHFACE') + os.sep
		self._target_fs_gt = 32.0
		self.fs_gt = None
		self._raw_fs_gt = None
		self.data = []

	def load_dataset(self):
		print('\nLoading dataset ' + self.name + '...')
		for sub in utils.sort_nicely(os.listdir(self.path)):
			sub_path = os.path.join(self.path, sub)
			if not os.path.isdir(sub_path):
				continue

			for trial in utils.sort_nicely(os.listdir(sub_path)):
				trial_path = os.path.join(sub_path, trial)
				if not os.path.isdir(trial_path):
					continue
				video_path = None
				for cand in ('data.cfr.avi', 'data.avi', 'data.mkv'):
					candidate_path = os.path.join(trial_path, cand)
					if os.path.exists(candidate_path):
						video_path = candidate_path
						break
				if video_path is None:
					continue
 
				if os.path.exists(video_path):
					d = {}
					d['video_path'] = video_path
					d['subject'] = sub
					d['trial'] = trial
					d['chest_rois'] = []
					d['face_rois'] = []
					d['rppg_obj'] = []
					d['gt'] = self.load_gt(trial_path)
					self.data.append(d)

		print('%d items loaded!' % len(self.data)) 

	def load_gt(self, trial_path):
		import h5py

		import os
		with h5py.File(os.path.join(trial_path, 'data.hdf5'), 'r') as f:
			resp = f['respiration']
			raw_gt = np.array(resp)
			raw_fs = None
			for key in ('fs', 'frequency', 'sampling_frequency', 'sample_rate', 'samplingrate'):
				if key in resp.attrs:
					raw_fs = float(resp.attrs[key])
					break
			if raw_fs is None:
				raw_fs = self._raw_fs_gt or 256.0
			self._raw_fs_gt = raw_fs

		target_fs = self._target_fs_gt
		decim = max(1, int(round(raw_fs / target_fs)))
		effective_fs = float(raw_fs / decim)
		if decim > 1:
			gt = raw_gt[::decim]
		else:
			gt = raw_gt

		if self.fs_gt is None or abs(self.fs_gt - effective_fs) > 1e-6:
			self.fs_gt = effective_fs
		return gt.astype(np.float32)

	def extract_ROI(self, video_path, region='chest'):
		if region == 'chest':
			params = self.roi_params.get('chest', {})
			mp_complexity = params.get('mp_complexity', 1)
			skip_rate = params.get('skip_rate', 10)
			rois, _, _ = utils.get_chest_ROI(video_path, self.name, mp_complexity=mp_complexity, skip_rate=skip_rate)
		elif region == 'face':
			params = self.roi_params.get('face', {})
			rois = utils.get_face_ROI(video_path, **params) if params else utils.get_face_ROI(video_path)
		else:
			rois = []
		return rois

	def extract_rppg(self, video_path, method='cpu_CHROM'):
		from riv.resp_from_rPPG import RR_from_rPPG

		rppg_obj =  RR_from_rPPG(video_path, method=method)
		rppg_obj.get_rPPG()
		return rppg_obj

class MAHNOB(DatasetBase):
	def __init__(self):
		super().__init__()
		self.name = 'mahnob'
		self.path = self.resolve('MAHNOB') + os.sep
		self.data = []

	def load_gt(self, sbj_path):
		from pyedflib import EdfReader
		bdf_file = None
		for fn in os.listdir(sbj_path):
			if fn.lower().endswith('.bdf'):
				bdf_file = os.path.join(sbj_path, fn)
				break
		if bdf_file is None:
			raise FileNotFoundError('No .bdf file found in %s' % sbj_path)
		channel = 44
		reader = EdfReader(bdf_file)
		self.fs_gt = reader.getSampleFrequency(channel)
		gt = reader.readSignal(channel)
		reader.close()
		return gt


	def load_dataset(self):
		print('\nLoading dataset ' + self.name + '...')
		for sub in utils.sort_nicely(os.listdir(self.path)):
			sub_path = os.path.join(self.path, sub)
			if not os.path.isdir(sub_path):
				continue
			for fn in os.listdir(sub_path):
				if fn.endswith('.avi'):
					break
			video_path = os.path.join(sub_path, fn)

			if os.path.exists(video_path):
				d = {}
				d['video_path'] = video_path
				d['subject'] = sub
				d['chest_rois'] = []
				d['face_rois'] = []
				d['rppg_obj'] = []
				d['gt'] = self.load_gt(sub_path)
				self.data.append(d)

		print('%d items loaded!' % len(self.data)) 

	def extract_ROI(self, video_path, region='chest'):
		if region == 'chest':
			params = self.roi_params.get('chest', {})
			mp_complexity = params.get('mp_complexity', 1)
			skip_rate = params.get('skip_rate', 10)
			rois, _, _ = utils.get_chest_ROI(video_path, self.name, mp_complexity=mp_complexity, skip_rate=skip_rate)
		elif region == 'face':
			params = self.roi_params.get('face', {})
			rois = utils.get_face_ROI(video_path, **params) if params else utils.get_face_ROI(video_path)
		else:
			rois = []
		return rois

	def extract_rppg(self, video_path, method='cpu_CHROM'):
		from riv.resp_from_rPPG import RR_from_rPPG

		rppg_obj =  RR_from_rPPG(video_path, method=method)
		rppg_obj.get_rPPG()
		return rppg_obj

# Methods class definitions

class MethodBase:
	def __init__(self):
		self.name = ''
		self.win_size = 30
		self.data_type = ''

	def process(self, data):
		# This class can be used to process either videos or ROIs
		raise NotImplementedError("Subclasses must implement process method")

# Deep models

class MTTS_CAN(MethodBase):
	def __init__(self):
		super().__init__()
		self.name = 'MTTS_CAN'
		self.batch_size = 100
		self.data_type = 'face'

	def process(self, data):
		from deep.MTTS_CAN.my_predict_vitals import predict_vitals

		resp = predict_vitals(frames=data['face_rois'], batch_size=self.batch_size)
		return resp

class BigSmall(MethodBase):
	def __init__(self):
		super().__init__()
		self.name = 'BigSmall'
		self.data_type = 'face'

	def process(self, data):
		from deep.BigSmall.predict_vitals import predict_vitals

		resp = predict_vitals(data['face_rois'])
		return resp

# Motion based

class OF_Deep(MethodBase):

	def __init__(self, model, batch_size=64):
		super().__init__()
		self.name = 'OF_Deep' + ' ' + model
		self.data_type = 'chest'
		self.model = model
		self.ckpt = 'things'
		self.batch_size = batch_size

	def forward(self, inputs):
		import torch
		predictions = self.OFmodel(inputs)
		predictions = self.io_adapter.unpad_and_unscale(predictions)
		flows = torch.squeeze(predictions['flows'])[:,1,:,:]
		vert = flows.reshape(flows.shape[0],-1).cpu().detach().numpy()
		return vert
	
	def process(self, data, cuda=None):
		import ptlflow
		import torch
		from PIL import Image
		from ptlflow.utils import flow_utils
		from ptlflow.utils.io_adapter import IOAdapter
		import warnings
		warnings.filterwarnings("ignore") 
		if cuda is None:
			cuda = _use_cuda()
		if not cuda:
			torch.cuda.is_available = lambda : False
			device = torch.device('cpu')
		else:
			device = torch.device(_torch_device())
		self.OFmodel = ptlflow.get_model(self.model, pretrained_ckpt=self.ckpt)
		self.OFmodel.to(device)
		s = []
		newsize = (224, 144)
		video = [np.array(r.resize(newsize)) for r in data['chest_rois']]
		nframes = len(video)
		print("\n> Computing Optical Flow...")

		while True:
			try:
				print("\n> Attempting with batch size: " + str(self.batch_size))
				for i in tqdm(range(0, nframes, self.batch_size)):
					if i == 0:
						start = i
					else:
						start = i-1
					end = min(i+self.batch_size, nframes-1)
					batch = video[start:end]
					if len(batch) <= 2:
						continue
					if i == 0:
						self.io_adapter = IOAdapter(self.OFmodel, batch[0].shape[:2], cuda=cuda)
					inputs = self.io_adapter.prepare_inputs(batch)
					input_images = inputs["images"][0]
					video1 = input_images[:-1]
					video2 = input_images[1:]
					input_images = torch.stack((video1, video2), dim=1)
					if cuda:
						input_images = input_images.cuda()
					inputs["images"] = input_images
					vert = self.forward(inputs)
					s.append(np.median(vert, axis=1))
				break
			except RuntimeError:
				self.batch_size = self.batch_size // 2
				if self.batch_size < 4:
					raise ValueError("Batch size is too tiny, maybe need more GPU memory.")
		del self.OFmodel
		torch.cuda.empty_cache()
		sig = np.concatenate(s)
		return sig

class OF_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'OF_Model'
		self.data_type = 'chest'

	def process(self, data):
		from motion.motion import OF
		import cv2 as cv

		# convert rois to grayscale
		g_rois = [cv.cvtColor(np.asarray(x), cv.COLOR_RGB2GRAY) for x in data['chest_rois']];

		# estimate OF
		of, _ = OF(g_rois, data['fps'])
		return of

class DoF(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'DoF'
		self.data_type = 'chest'

	def process(self, data):
		from motion.motion import DoF
		import cv2 as cv

		# convert rois to grayscale
		g_rois = [cv.cvtColor(np.asarray(x), cv.COLOR_RGB2GRAY) for x in data['chest_rois']];

		# estimate DoF
		dof, _ = DoF(g_rois, data['fps'])
		return dof

class profile1D(MethodBase):

	def __init__(self, interp_type='quadratic'):
		super().__init__()
		self.name = 'profile1D ' + interp_type 
		self.data_type = 'chest'
		self.interp_type = interp_type

	def process(self, data):
		from motion.motion import profile1D
		import cv2 as cv

		# convert rois to grayscale
		g_rois = [cv.cvtColor(np.asarray(x), cv.COLOR_RGB2GRAY) for x in data['chest_rois']];

		# estimate profile1D
		profile, _ = profile1D(g_rois, data['fps'], self.interp_type)
		return profile

# RIV based

class peak(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'fiedler'
		self.data_type = 'rppg'

	def process(self, data):
		return data['rppg_obj'].extract_RIVs_from_peaks()

class morph(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'ims'
		self.data_type = 'rppg'

	def process(self, data):
		return data['rppg_obj'].extract_RIVs_from_IMS()

class bss_ssa(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'bss_ssa'
		self.data_type = 'rppg'
		self.nGroups = None

	def process(self, data):
		return data['rppg_obj'].extract_RIVs_from_SSA(self.nGroups)

class bss_emd(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'bss_emd'
		self.data_type = 'rppg'
		self.nIMF = 4

	def process(self, data):
		return data['rppg_obj'].extract_RIVs_from_EMD(self.nIMF)

PROFILE1D_INTERPS = ('linear', 'quadratic', 'cubic')

def _dataset_results_dir(results_dir, dataset_name):
	path = os.path.join(results_dir, dataset_name)
	os.makedirs(path, exist_ok=True)
	# New structure for artifacts
	for sub in ("data", "metrics", "plots", "logs"):
		os.makedirs(os.path.join(path, sub), exist_ok=True)
	return path


def _method_token(method_name):
	name = method_name.strip()
	lname = name.lower()
	if lname in ('of_model', 'of', 'of_farneback'):
		return 'of_farneback'
	if lname == 'dof':
		return 'dof'
	if lname.startswith('profile1d'):
		return lname.replace(' ', '_')
	if '__' in name:
		return name.replace(' ', '_')
	return name.replace(' ', '_').replace('-', '_')


def _method_suffix(methods):
	tokens = []
	seen = set()
	for m in methods:
		token = _method_token(m.name)
		if token in seen:
			continue
		seen.add(token)
		tokens.append(token)
	suffix = '_'.join(tokens)
	if len(suffix) <= 64:
		return suffix
	digest = hashlib.sha1(suffix.encode('utf-8')).hexdigest()[:10]
	return f"{len(tokens)}m_{digest}"


def _sanitize_run_label(label):
	if not label:
		return None
	label = label.strip()
	if not label:
		return None
	sanitized = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in label)
	sanitized = sanitized.strip('_-')
	return sanitized or None


def _base_of_method(name):
	if not name:
		return ''
	return name.split('__', 1)[0].lower()


def _ensure_dir(path):
	os.makedirs(path, exist_ok=True)
	return path


def _json_float(value):
	try:
		val = float(value)
	except (TypeError, ValueError):
		return None
	if np.isnan(val) or np.isinf(val):
		return None
	return val


def _json_list(arr, max_len=None):
	try:
		iterable = np.asarray(arr).tolist()
	except Exception:
		return None
	if max_len is not None and isinstance(iterable, list) and len(iterable) > max_len:
		return iterable[:max_len]
	return iterable


def _record_step_profile(step_name, duration):
	if not _PROFILE_STEPS:
		return
	stats = _STEP_PROF[step_name]
	stats['calls'] += 1
	stats['total'] += float(duration)


def _reset_step_prof():
	_STEP_PROF.clear()


def _profiler_report(label=None, topn=30):
	if not _PROFILE_STEPS:
		return []
	rows = []
	for name, stats in _STEP_PROF.items():
		calls = stats.get('calls', 0) or 1
		total = float(stats.get('total', 0.0))
		avg = total / calls if calls else 0.0
		rows.append((name, int(calls), total, avg))
	rows.sort(key=lambda item: item[2], reverse=True)
	if not rows:
		print("\n[Profiler] No timing data collected.")
		return rows

	headers = ("function", "calls", "total_ms", "avg_ms")
	formatted = []
	for name, calls, total, avg in rows[:topn]:
		formatted.append((
			name,
			f"{calls:,}",
			f"{total * 1000:,.2f}",
			f"{avg * 1000:,.2f}",
		))

	widths = [len(h) for h in headers]
	for row in formatted:
		for idx, cell in enumerate(row):
			widths[idx] = max(widths[idx], len(cell))

	def _fmt_row(row_cells):
		pieces = []
		for idx, cell in enumerate(row_cells):
			width = widths[idx]
			pieces.append(cell.ljust(width) if idx == 0 else cell.rjust(width))
		return "| " + " | ".join(pieces) + " |"

	def _border(char="-"):
		return "+" + "+".join(char * (w + 2) for w in widths) + "+"

	border = _border("-")
	header_border = _border("=")
	header = f"[Profiler]{f' {label}' if label else ''}"
	print(f"\n{header}")
	print(border)
	print(_fmt_row(headers))
	print(header_border)
	for row in formatted:
		print(_fmt_row(row))
	print(border)
	return rows


def _profiler_stage_start():
	if _PROFILE_STEPS:
		_reset_step_prof()


def _profiler_stage_end(label):
	if not _PROFILE_STEPS:
		return
	rows = _profiler_report(label=label)
	if not rows:
		_reset_step_prof()
	else:
		_reset_step_prof()


@contextlib.contextmanager
def _file_lock(lock_path):
	os.makedirs(os.path.dirname(lock_path), exist_ok=True)
	fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
	try:
		if os.name == 'nt':
			import msvcrt  # type: ignore
			os.lseek(fd, 0, os.SEEK_SET)
			msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
		else:
			import fcntl  # type: ignore
			fcntl.flock(fd, fcntl.LOCK_EX)
		yield
	finally:
		try:
			if os.name == 'nt':
				import msvcrt  # type: ignore
				os.lseek(fd, 0, os.SEEK_SET)
				msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
			else:
				import fcntl  # type: ignore
				fcntl.flock(fd, fcntl.LOCK_UN)
		finally:
			os.close(fd)


def _atomic_pickle_dump(data, path):
	dir_name = os.path.dirname(path)
	os.makedirs(dir_name, exist_ok=True)
	fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
	try:
		with os.fdopen(fd, 'wb') as tmp_file:
			pickle.dump(data, tmp_file)
		os.replace(tmp_path, path)
	except Exception:
		try:
			os.unlink(tmp_path)
		except OSError:
			pass
		raise


def _atomic_json_dump(data, path, indent=2):
	dir_name = os.path.dirname(path)
	os.makedirs(dir_name, exist_ok=True)
	fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
			json.dump(data, tmp_file, indent=indent)
		os.replace(tmp_path, path)
	except Exception:
		try:
			os.unlink(tmp_path)
		except OSError:
			pass
		raise

def timed_step(label=None):
	def decorator(func):
		step_name = label or func.__name__
		@wraps(func)
		def wrapper(*args, **kwargs):
			start = time.time()
			result = func(*args, **kwargs)
			duration = time.time() - start
			_record_step_profile(step_name, duration)
			print(f"[timing] {step_name} completed in {duration:.2f}s")
			return result
		return wrapper
	return decorator


def _filter_valid_rois(rois):
	filtered = []
	for roi in rois or []:
		arr = np.asarray(roi)
		if arr.size > 0:
			filtered.append(roi)
	return filtered


def _merge_results_payload(out_path, partial_results, method_order=None):
	if partial_results is None:
		return
	metadata_keys = ['video_path', 'fps', 'gt', 'fs_gt']
	new_estimates = list(partial_results.get('estimates', [])) if isinstance(partial_results, dict) else []
	lock_path = f"{out_path}.lock"
	with _file_lock(lock_path):
		if os.path.exists(out_path):
			with open(out_path, 'rb') as fp:
				data = pickle.load(fp)
		else:
			data = {}
		for key in metadata_keys:
			if key in partial_results:
				data[key] = partial_results[key]
		existing_estimates = list(data.get('estimates', []))
		index_map = {}
		for idx, entry in enumerate(existing_estimates):
			if isinstance(entry, dict):
				name = entry.get('method')
				if name is not None and name not in index_map:
					index_map[name] = idx
		for entry in new_estimates:
			if not isinstance(entry, dict):
				continue
			name = entry.get('method')
			if name is None:
				continue
			if name in index_map:
				existing_estimates[index_map[name]] = entry
			else:
				index_map[name] = len(existing_estimates)
				existing_estimates.append(entry)
		if method_order:
			order_index = {name: idx for idx, name in enumerate(method_order)}
			def _order_key(item):
				if isinstance(item, dict):
					return order_index.get(item.get('method'), len(order_index))
				return len(order_index)
			existing_estimates.sort(key=_order_key)
		data['estimates'] = existing_estimates
		_atomic_pickle_dump(data, out_path)


def _discover_methods_from_aux(run_dir):
	aux_dir = os.path.join(run_dir, 'aux')
	if not os.path.isdir(aux_dir):
		return []
	try:
		entries = utils.sort_nicely(os.listdir(aux_dir))
	except Exception:
		try:
			entries = os.listdir(aux_dir)
		except Exception:
			return []
	methods = []
	for entry in entries:
		entry_path = os.path.join(aux_dir, entry)
		if os.path.isdir(entry_path):
			methods.append(entry)
	return methods


_TRACKER_SUFFIXES = {'pll', 'kfstd', 'ukffreq'}
_SPECTRAL_SUFFIXES = {'spec_ridge'}
_DEFAULT_GATING_CFG = {
	'common': {
		'use_track': None,
		# Hz guard at band edges; recommend margin 0.005 Hz + 1 s persist when trackers are primary.
		'saturation_margin_hz': 0.0,
		'saturation_persist_sec': 0.0,
		'constant_ptp_max_hz': 0.0
	},
	'tracker': {
		'std_min_bpm': 0.3,
		'unique_min': 0.05,
		'saturation_max': 0.15,
		'std_is_soft': False
	},
	'spectral': {
		'peak_ratio_min': 1.5,
		'prominence_min_db': 3.0,
		'fwhm_max_hz': 0.08,
		'fwhm_df_guard': 1.25
	},
	'debug': {
		'disable_gating': False
	}
}

_BUILTIN_GATING_PROFILES = {
	'diagnostic-relaxed': {
		'common': {
			'use_track': True,
			'saturation_margin_hz': 0.005,
			'saturation_persist_sec': 1.0,
			'constant_ptp_max_hz': 0.008
		},
		'tracker': {
			'std_min_bpm': 0.05,
			'unique_min': 0.01,
			'saturation_max': 0.4,
			'std_is_soft': False
		},
		'spectral': {}
	},
	'paper': {
		'common': {
			'use_track': True,
			'saturation_margin_hz': 0.005,
			'saturation_persist_sec': 1.0,
			'constant_ptp_max_hz': 0.0
		},
		'tracker': {
			'std_min_bpm': 0.0,
			'unique_min': 0.0,
			'saturation_max': 1.0,
			'std_is_soft': True
		},
		'spectral': {
			'peak_ratio_min': 1.2,
			'prominence_min_db': 1.5,
			'fwhm_max_hz': 0.12,
			'fwhm_df_guard': 1.25
		}
	}
}


def _deep_merge_dict(base, new_values):
	if not isinstance(new_values, dict):
		return base
	for key, value in new_values.items():
		if isinstance(value, dict) and isinstance(base.get(key), dict):
			base[key] = _deep_merge_dict(base[key], value)
		else:
			base[key] = copy.deepcopy(value)
	return base


def _infer_method_capability(name):
	if not name:
		return 'spectral'
	lname = name.lower()
	if '__' in lname:
		suffix = lname.split('__', 1)[1]
		if suffix in _SPECTRAL_SUFFIXES:
			return 'spectral'
		for tracker_suffix in _TRACKER_SUFFIXES:
			if suffix.startswith(tracker_suffix):
				return 'tracker'
		return 'tracker'
	return 'spectral'


def _tracker_gating_decision(track_stats, cfg, disable=False):
	flags = {'low_std': False, 'low_unique': False, 'high_saturation': False, 'low_std_soft': False}
	reason = None
	if disable or not track_stats:
		return False, flags, reason
	std_bpm = track_stats.get('std_bpm')
	unique_frac = track_stats.get('unique_fraction')
	sat_frac = track_stats.get('saturation_fraction')
	std_soft = bool(cfg.get('std_is_soft'))
	if std_bpm is not None and np.isfinite(std_bpm):
		low_std = std_bpm < float(cfg.get('std_min_bpm', 0.0))
		flags['low_std'] = low_std
		if low_std and std_soft:
			flags['low_std_soft'] = True
	if unique_frac is not None and np.isfinite(unique_frac):
		flags['low_unique'] = unique_frac < float(cfg.get('unique_min', 0.0))
	if sat_frac is not None and np.isfinite(sat_frac):
		flags['high_saturation'] = sat_frac > float(cfg.get('saturation_max', 1.0))
	trigger = (
		(flags['low_std'] and not std_soft)
		or flags['low_unique']
		or flags['high_saturation']
	)
	if trigger:
		reason = 'tracker_quality'
	return trigger, flags, reason


def _spectral_stats_from_signal(filtered_sig, fps, win_seconds, min_hz, max_hz):
	stats = {}
	try:
		signal_array = np.asarray(filtered_sig, dtype=np.float64)
		if signal_array.ndim == 1:
			signal_array = signal_array[np.newaxis, :]
		win_param = max(1, int(round(max(win_seconds, 1.0) / 1.5)))
		nperseg = 0
		try:
			fps_val = float(fps)
		except (TypeError, ValueError):
			fps_val = 0.0
		if fps_val > 0.0:
			nperseg = int(max(1, round(fps_val * win_param)))
		freqs_bpm, power = utils.Welch_rpm(signal_array, fps, win_param, min_hz, max_hz)
		if nperseg > 0 and np.isfinite(fps_val) and fps_val > 0.0:
			stats['welch_df_hz'] = float(fps_val / nperseg)
		else:
			stats['welch_df_hz'] = float('nan')
		if power.size == 0:
			return stats
		mean_power = np.mean(power, axis=0)
		if mean_power.size == 0:
			return stats
		peak_idx = int(np.argmax(mean_power))
		peak_power = float(mean_power[peak_idx])
		eps = 1e-12
		if mean_power.size > 1:
			second = float(np.partition(mean_power, -2)[-2])
		else:
			second = peak_power
		stats['peak_ratio'] = float(peak_power / max(second, eps))
		stats['prominence_db'] = float(10.0 * np.log10((peak_power + eps) / (np.mean(mean_power) + eps)))
		stats['peak_bpm'] = float(freqs_bpm[peak_idx]) if freqs_bpm.size else float('nan')
		freq_hz = (np.asarray(freqs_bpm, dtype=np.float64) / 60.0) if freqs_bpm.size else np.asarray([], dtype=np.float64)
		half_power = peak_power * 0.5
		left = peak_idx
		while left > 0 and mean_power[left] >= half_power:
			left -= 1
		right = peak_idx
		while right < mean_power.size - 1 and mean_power[right] >= half_power:
			right += 1
		if freq_hz.size:
			stats['fwhm_hz'] = float(max(0.0, freq_hz[min(right, freq_hz.size - 1)] - freq_hz[max(left, 0)]))
		else:
			stats['fwhm_hz'] = float('nan')
	except Exception:
		return {}
	return stats


def _spectral_quality_flags(stats, cfg, disable=False):
	flags = {'low_ratio': False, 'low_prom': False, 'wide_fwhm': False}
	if disable or not stats:
		return True, flags, None
	peak_ratio = stats.get('peak_ratio')
	prom_db = stats.get('prominence_db')
	fwhm_hz = stats.get('fwhm_hz')
	welch_df = stats.get('welch_df_hz')
	try:
		df_guard_factor = float(cfg.get('fwhm_df_guard', 0.0))
	except (TypeError, ValueError):
		df_guard_factor = 0.0
	if peak_ratio is not None and np.isfinite(peak_ratio):
		flags['low_ratio'] = peak_ratio < float(cfg.get('peak_ratio_min', 0.0))
	if prom_db is not None and np.isfinite(prom_db):
		flags['low_prom'] = prom_db < float(cfg.get('prominence_min_db', 0.0))
	if fwhm_hz is not None and np.isfinite(fwhm_hz):
		fwhm_limit = float(cfg.get('fwhm_max_hz', np.inf))
		df_guard = 0.0
		if welch_df is not None and np.isfinite(welch_df) and welch_df > 0.0 and df_guard_factor > 0.0:
			df_guard = df_guard_factor * float(welch_df)
		flags['wide_fwhm'] = fwhm_hz > max(fwhm_limit, df_guard)
	trustworthy = not any(flags.values())
	reason = None if trustworthy else 'spectral_quality'
	return trustworthy, flags, reason


def _coerce_override_value(raw_value):
	value = raw_value.strip()
	if value.lower() in ('true', 'false'):
		return value.lower() == 'true'
	try:
		if '.' in value:
			return float(value)
		return int(value)
	except ValueError:
		return value


def _apply_overrides(cfg, overrides):
	if not overrides:
		return
	for item in overrides:
		if '=' not in item:
			print(f"> Warning: override '{item}' ignored (missing '=')")
			continue
		path, raw_value = item.split('=', 1)
		keys = [key for key in path.strip().split('.') if key]
		if not keys:
			print(f"> Warning: override '{item}' ignored (empty key)")
			continue
		target = cfg
		for key in keys[:-1]:
			if key not in target or not isinstance(target[key], dict):
				target[key] = {}
			target = target[key]
		target[keys[-1]] = _coerce_override_value(raw_value)


def _parse_bool_flag(value):
	if isinstance(value, bool):
		return value
	if value is None:
		return None
	val = str(value).strip().lower()
	if val in ('1', 'true', 'yes', 'on'):
		return True
	if val in ('0', 'false', 'no', 'off'):
		return False
	raise argparse.ArgumentTypeError(f"Expected boolean value for flag, got '{value}'")


def _resolve_gating_config(cfg):
	user_cfg = cfg.get('gating') if isinstance(cfg, dict) else {}
	base = copy.deepcopy(_DEFAULT_GATING_CFG)
	if not isinstance(user_cfg, dict):
		return base
	profile_name = user_cfg.get('profile')
	profile_sources = {}
	if isinstance(user_cfg.get('profiles'), dict):
		profile_sources.update(user_cfg['profiles'])
	if profile_name:
		if profile_name in profile_sources:
			profile_cfg = copy.deepcopy(profile_sources[profile_name])
		else:
			profile_cfg = copy.deepcopy(_BUILTIN_GATING_PROFILES.get(profile_name, {}))
		if profile_cfg:
			base = _deep_merge_dict(base, profile_cfg)
	# Merge explicit overrides (excluding profile metadata)
	user_cfg_copy = copy.deepcopy(user_cfg)
	user_cfg_copy.pop('profile', None)
	user_cfg_copy.pop('profiles', None)
	resolved = _deep_merge_dict(base, user_cfg_copy)
	if profile_name:
		resolved.setdefault('meta', {})['profile'] = profile_name
	return resolved


def _percentile_summary(values):
	arr = np.asarray(values, dtype=np.float64)
	if arr.ndim == 0:
		arr = arr.reshape(1)
	if arr.size:
		arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return {'median': float('nan'), 'q1': float('nan'), 'q3': float('nan')}
	return {
		'median': float(np.percentile(arr, 50)),
		'q1': float(np.percentile(arr, 25)),
		'q3': float(np.percentile(arr, 75))
	}


def _quality_row_from_record(method, record):
	quality = record.get('quality') or {}
	track_stats = record.get('track_stats') or {}
	spectral_stats = record.get('spectral_stats') or {}
	flags = quality.get('gating_flags') or record.get('gating_flags') or {}
	trial_key = quality.get('trial_key') or record.get('trial_key')
	data_file = quality.get('data_file') or record.get('data_file')
	capability = record.get('capability')
	source = record.get('source') or quality.get('source_label')
	is_tracker_fallback = bool(capability == 'tracker' and source == 'fallback')
	row = {
		'method': method,
		'capability': capability,
		'trial': trial_key,
		'data_file': data_file,
		'source': source,
		'track_used': int(bool(quality.get('track_used'))),
		'track_candidate_present': int(bool(quality.get('track_candidate_present'))),
		'is_fallback': int(is_tracker_fallback),
		'degenerate_reason': record.get('degenerate_reason'),
		'fallback_from': record.get('fallback_from'),
		'std_bpm': track_stats.get('std_bpm'),
		'unique_fraction': track_stats.get('unique_fraction'),
		'track_dyn_range_hz_ptp': track_stats.get('track_dyn_range_hz_ptp'),
		'median_hz': track_stats.get('median_hz'),
		'edge_saturation_fraction': track_stats.get('edge_saturation_fraction'),
		'constant_track_promoted': int(bool(track_stats.get('constant_track_promoted'))),
		'spectral_peak_ratio': spectral_stats.get('peak_ratio'),
		'spectral_prominence_db': spectral_stats.get('prominence_db'),
		'spectral_fwhm_hz': spectral_stats.get('fwhm_hz'),
		'spectral_welch_df_hz': spectral_stats.get('welch_df_hz'),
		'flag_std': int(bool(flags.get('std'))),
		'flag_uniq': int(bool(flags.get('uniq'))),
		'flag_sat': int(bool(flags.get('sat'))),
		'flag_low_ratio': int(bool(flags.get('low_ratio'))),
		'flag_low_prom': int(bool(flags.get('low_prom'))),
		'flag_wide_fwhm': int(bool(flags.get('wide_fwhm'))),
		'std_is_soft': int(bool(quality.get('std_is_soft'))),
		'std_violation_soft': int(bool(quality.get('std_violation_soft')))
	}
	return row, is_tracker_fallback


def _persist_quality_reports(method_metrics, results_dir):
	if not method_metrics:
		return ""
	logs_dir = os.path.join(results_dir, 'logs')
	os.makedirs(logs_dir, exist_ok=True)
	csv_path = os.path.join(logs_dir, 'method_quality.csv')
	summary_path = os.path.join(logs_dir, 'method_quality_summary.json')
	headers = [
		'method', 'capability', 'trial', 'data_file', 'source', 'track_used',
		'track_candidate_present', 'is_fallback', 'degenerate_reason', 'fallback_from',
		'std_bpm', 'unique_fraction', 'track_dyn_range_hz_ptp', 'median_hz',
		'edge_saturation_fraction', 'constant_track_promoted', 'spectral_peak_ratio',
		'spectral_prominence_db', 'spectral_fwhm_hz', 'spectral_welch_df_hz',
		'flag_std', 'flag_uniq', 'flag_sat', 'flag_low_ratio', 'flag_low_prom',
		'flag_wide_fwhm', 'std_is_soft', 'std_violation_soft'
	]
	rows = []
	method_summaries = {}
	overall_peak = []
	overall_fwhm = []
	overall_total = 0
	overall_tracker_total = 0
	overall_fallback = 0
	overall_reasons = defaultdict(int)
	for method, records in (method_metrics or {}).items():
		if not records:
			continue
		fallback_reasons = defaultdict(int)
		peak_vals = []
		fwhm_vals = []
		total = 0
		fallbacks = 0
		track_sources = 0
		for record in records:
			total += 1
			row, is_fallback = _quality_row_from_record(method, record)
			rows.append(row)
			source = row['source']
			if record.get('capability') == 'tracker' and source == 'track':
				track_sources += 1
			if record.get('capability') == 'tracker':
				overall_tracker_total += 1
			if is_fallback:
				fallbacks += 1
				overall_fallback += 1
				reason = record.get('degenerate_reason') or 'unknown'
				fallback_reasons[reason] += 1
				overall_reasons[reason] += 1
			spectral_stats = record.get('spectral_stats') or {}
			peak = spectral_stats.get('peak_ratio')
			if peak is not None and np.isfinite(peak):
				peak_vals.append(float(peak))
				overall_peak.append(float(peak))
			fwhm = spectral_stats.get('fwhm_hz')
			if fwhm is not None and np.isfinite(fwhm):
				fwhm_vals.append(float(fwhm))
				overall_fwhm.append(float(fwhm))
		method_summaries[method] = {
			'total': total,
			'track_source': track_sources,
			'fallbacks': fallbacks,
			'fallback_rate': float(fallbacks / total) if total else 0.0,
			'fallback_reasons': dict(fallback_reasons),
			'spectral_stats': {
				'peak_ratio': _percentile_summary(peak_vals),
				'fwhm_hz': _percentile_summary(fwhm_vals)
			}
		}
		overall_total += total
	quality_payload = {
		'methods': method_summaries,
		'overall': {
			'total': overall_total,
			'tracker_total': overall_tracker_total,
			'fallbacks': overall_fallback,
			'fallback_reasons': dict(overall_reasons),
			'spectral_stats': {
				'peak_ratio': _percentile_summary(overall_peak),
				'fwhm_hz': _percentile_summary(overall_fwhm)
			}
		}
	}
	rate_denom = overall_tracker_total if overall_tracker_total else overall_total
	quality_payload['overall']['fallback_rate'] = float(overall_fallback / rate_denom) if rate_denom else 0.0
	try:
		with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
			writer = csv.DictWriter(fp, fieldnames=headers)
			writer.writeheader()
			for row in rows:
				writer.writerow(row)
	except Exception as exc:
		print(f"> Warning: failed to write quality CSV ({exc})")
	try:
		with open(summary_path, 'w', encoding='utf-8') as fp:
			json.dump(quality_payload, fp, ensure_ascii=False, indent=2)
	except Exception as exc:
		print(f"> Warning: failed to write quality summary JSON ({exc})")
	fallback_rate = quality_payload['overall']['fallback_rate']
	return f"> Quality reports saved ({csv_path}, {summary_path}) | tracker fallback rate={fallback_rate * 100:.2f}%"


@timed_step('evaluate')
def evaluate(
	results_dir,
	metrics,
	win_size=30,
	stride=1,
	visualize=True,
	min_hz=0.08,
	max_hz=0.5,
	use_track=False,
	track_std_min_bpm=0.3,
	track_unique_min=0.05,
	track_saturation_max=0.15,
	saturation_margin_hz=0.0,
	saturation_persist_sec=0.0,
	constant_ptp_max_hz=0.0,
	gating=None
):
	"""
	Evaluate respiration tracks against ground truth and persist per-method metrics.

	Key gating knobs (all disabled when 0.0):
	  * saturation_margin_hz: absolute edge guard in Hz; recommended 0.005 for production runs.
	  * saturation_persist_sec: minimum continuous edge duration in seconds before flagging saturation.
	  * constant_ptp_max_hz: robust frequency range (Hz, P95−P5) below which tracks are treated as constant.
	"""
	gating_cfg = gating or {}
	gating_common = gating_cfg.get('common', {})
	tracker_gating_cfg = gating_cfg.get('tracker', {})
	spectral_gating_cfg = gating_cfg.get('spectral', {})
	disable_gating = bool(gating_cfg.get('debug', {}).get('disable_gating', False))
	if gating_common.get('use_track') is not None:
		use_track = bool(gating_common.get('use_track'))
	if gating_common.get('saturation_margin_hz') is not None:
		saturation_margin_hz = gating_common.get('saturation_margin_hz')
	if gating_common.get('saturation_persist_sec') is not None:
		saturation_persist_sec = gating_common.get('saturation_persist_sec')
	if gating_common.get('constant_ptp_max_hz') is not None:
		constant_ptp_max_hz = gating_common.get('constant_ptp_max_hz')

	print('\n> Loading extracted data from ' + results_dir + '...')

	try:
		track_std_min_bpm = float(track_std_min_bpm)
	except (TypeError, ValueError):
		track_std_min_bpm = 0.0
	track_std_min_bpm = max(0.0, track_std_min_bpm)
	try:
		track_unique_min = float(track_unique_min)
	except (TypeError, ValueError):
		track_unique_min = 0.0
	track_unique_min = max(0.0, track_unique_min)
	try:
		track_saturation_max = float(track_saturation_max)
	except (TypeError, ValueError):
		track_saturation_max = 0.15
	if track_saturation_max < 0.0:
		track_saturation_max = 0.0
	if track_saturation_max > 1.0:
		track_saturation_max = 1.0
	try:
		if saturation_margin_hz is None:
			saturation_margin_hz = 0.0
		saturation_margin_hz = float(saturation_margin_hz)
	except (TypeError, ValueError):
		saturation_margin_hz = 0.0
	if not np.isfinite(saturation_margin_hz):
		saturation_margin_hz = 0.0
	if saturation_margin_hz < 0.0:
		saturation_margin_hz = 0.0
	try:
		if saturation_persist_sec is None:
			saturation_persist_sec = 0.0
		saturation_persist_sec = float(saturation_persist_sec)
	except (TypeError, ValueError):
		saturation_persist_sec = 0.0
	if not np.isfinite(saturation_persist_sec):
		saturation_persist_sec = 0.0
	if saturation_persist_sec < 0.0:
		saturation_persist_sec = 0.0
	try:
		if constant_ptp_max_hz is None:
			constant_ptp_max_hz = 0.0
		constant_ptp_max_hz = float(constant_ptp_max_hz)
	except (TypeError, ValueError):
		constant_ptp_max_hz = 0.0
	if not np.isfinite(constant_ptp_max_hz):
		constant_ptp_max_hz = 0.0
	if constant_ptp_max_hz < 0.0:
		constant_ptp_max_hz = 0.0
	tracker_overrides = gating_cfg.get('tracker', {})
	if tracker_overrides.get('std_min_bpm') is not None:
		try:
			track_std_min_bpm = float(tracker_overrides.get('std_min_bpm'))
		except (TypeError, ValueError):
			pass
	if tracker_overrides.get('unique_min') is not None:
		try:
			track_unique_min = float(tracker_overrides.get('unique_min'))
		except (TypeError, ValueError):
			pass
	if tracker_overrides.get('saturation_max') is not None:
		try:
			track_saturation_max = float(tracker_overrides.get('saturation_max'))
		except (TypeError, ValueError):
			pass
	tracker_std_is_soft = tracker_overrides.get('std_is_soft')
	if tracker_std_is_soft is None:
		tracker_std_is_soft = _DEFAULT_GATING_CFG['tracker']['std_is_soft']
	tracker_std_is_soft = bool(tracker_std_is_soft)

	resolved_tracker_cfg = {
		'std_min_bpm': track_std_min_bpm,
		'unique_min': track_unique_min,
		'saturation_max': track_saturation_max,
		'std_is_soft': tracker_std_is_soft
	}
	track_std_is_soft = bool(tracker_std_is_soft)
	resolved_spectral_cfg = {
		'peak_ratio_min': float(spectral_gating_cfg.get('peak_ratio_min', _DEFAULT_GATING_CFG['spectral']['peak_ratio_min'])),
		'prominence_min_db': float(spectral_gating_cfg.get('prominence_min_db', _DEFAULT_GATING_CFG['spectral']['prominence_min_db'])),
		'fwhm_max_hz': float(spectral_gating_cfg.get('fwhm_max_hz', _DEFAULT_GATING_CFG['spectral']['fwhm_max_hz'])),
		'fwhm_df_guard': float(spectral_gating_cfg.get('fwhm_df_guard', _DEFAULT_GATING_CFG['spectral']['fwhm_df_guard']))
	}

	method_metrics = {}

	# Prefer new structure: results_dir/data/*.pkl, fallback to flat files
	data_dir = os.path.join(results_dir, 'data')
	if os.path.isdir(data_dir):
		files = [os.path.join('data', f) for f in utils.sort_nicely(os.listdir(data_dir)) if f.endswith('.pkl')]
	else:
		# If results_dir appears to be root with multiple runs, try to resolve uniquely
		candidate_runs = []
		for entry in utils.sort_nicely(os.listdir(results_dir)):
			full = os.path.join(results_dir, entry)
			if os.path.isdir(os.path.join(full, 'data')):
				candidate_runs.append(full)
		if len(candidate_runs) == 1:
			# Re-root evaluation under the single discovered run directory
			results_dir = candidate_runs[0]
			data_dir = os.path.join(results_dir, 'data')
			files = [os.path.join('data', f) for f in utils.sort_nicely(os.listdir(data_dir)) if f.endswith('.pkl')]
		else:
			# Fallback: recursively search for .pkl under results_dir (could mix runs)
			files = []
			for root, dirs, fnames in os.walk(results_dir):
				for fn in utils.sort_nicely(fnames):
					if fn.endswith('.pkl') and not fn.startswith('metrics'):
						# store relative path from results_dir
						rel = os.path.relpath(os.path.join(root, fn), results_dir)
						files.append(rel)
	ofdeep_models = ['_raft', '_raft_small', '_gma', '_irr_pwc', '_lcv_raft', '_craft']

	# Ensure logs directory exists early for debugging
	try:
		logs_dir = os.path.join(results_dir, 'logs')
		os.makedirs(logs_dir, exist_ok=True)
		debug_log_path = os.path.join(logs_dir, 'evaluate_appends.log')
	except Exception:
		logs_dir = None
		debug_log_path = None

	for filepath in tqdm(files, desc="Processing files"):
		tqdm.write("> Processing file %s" % (filepath))

		if 'metrics' in filepath:
			continue

		# Open the file with pickled data
		file = open(os.path.join(results_dir, filepath), 'rb')
		data = pickle.load(file)
		file.close()

		# Extract ground truth data
		fs_gt = data['fs_gt']
		gt = data['gt']
		base_name = os.path.splitext(os.path.basename(filepath))[0]
		name_parts = base_name.split('_')
		dataset_slug = name_parts[0] if name_parts else ''
		trial_key = base_name[len(dataset_slug) + 1:] if dataset_slug and len(base_name) > len(dataset_slug) + 1 else base_name

		# Filter ground truth
		filt_gt = utils.filter_RW(gt, fs_gt, lo=min_hz, hi=max_hz)

		if win_size == 'video':
			ws = filt_gt.shape[1] / fs_gt
		else:
			ws = win_size

		tqdm.write("> Length: %.2f sec" % (len(gt) / int(fs_gt)))

		# Apply windowing to ground truth
		gt_win, t_gt = utils.sig_windowing(filt_gt, fs_gt, ws, stride=stride)
		t_gt = np.asarray(t_gt, dtype=np.float64).reshape(-1)

		# Extract ground truth RPM using Welch with (win_size/1.5)
		gt_rpm_raw = utils.sig_to_RPM(gt_win, fs_gt, int(ws/1.5), min_hz, max_hz)
		gt_rpm_arr = np.squeeze(np.asarray(gt_rpm_raw, dtype=np.float64))
		if gt_rpm_arr.ndim == 0:
			gt_rpm_arr = gt_rpm_arr.reshape(1)
		gt_rpm_series = gt_rpm_arr.reshape(-1)
		if t_gt.size != gt_rpm_series.size:
			min_len = min(t_gt.size, gt_rpm_series.size)
			if min_len > 0:
				t_gt = t_gt[:min_len]
				gt_rpm_series = gt_rpm_series[:min_len]
			else:
				t_gt = np.asarray([], dtype=np.float64)
				gt_rpm_series = np.asarray([], dtype=np.float64)
		target_len = gt_rpm_series.size

		try:
			stride_val = float(stride)
		except (TypeError, ValueError):
			stride_val = 0.0
		if stride_val < 0:
			stride_val = abs(stride_val)
		if stride_val == 0.0:
			if t_gt.size > 1:
				grid_tolerance = 0.5 * float(np.min(np.diff(t_gt)))
			else:
				grid_tolerance = 0.0
		else:
			grid_tolerance = 0.5 * stride_val
		grid_tolerance = max(float(grid_tolerance), 1e-6)
		corr_guard_eps = 1e-6

		gt_rpm_full = gt_rpm_series.copy()
		t_gt_full = t_gt.copy()

		def _align_to_gt_grid(pred_times, pred_values):
			pred_times = np.asarray(pred_times, dtype=np.float64).reshape(-1) if pred_times is not None else np.asarray([], dtype=np.float64)
			pred_values = np.asarray(pred_values, dtype=np.float64).reshape(-1) if pred_values is not None else np.asarray([], dtype=np.float64)
			if pred_times.size and pred_values.size and pred_times.size != pred_values.size:
				min_len_local = min(pred_times.size, pred_values.size)
				pred_times = pred_times[:min_len_local]
				pred_values = pred_values[:min_len_local]
			meta = {
				'aligned': False,
				'len_gt': int(gt_rpm_full.size),
				'len_pred': int(pred_values.size),
				'len_final': 0,
				'len_valid': 0
			}
			if pred_times.size == 0 or pred_values.size == 0 or gt_rpm_full.size == 0:
				return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), meta
			tol = grid_tolerance if grid_tolerance > 0 else 1e-6
			matched = {}
			for idx_pred, tp in enumerate(pred_times):
				diff = np.abs(t_gt_full - tp)
				if diff.size == 0:
					continue
				gt_idx = int(np.argmin(diff))
				if diff[gt_idx] > tol:
					continue
				prev = matched.get(gt_idx)
				if prev is None or diff[gt_idx] < prev[1]:
					matched[gt_idx] = (idx_pred, diff[gt_idx])
			if not matched:
				return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64), meta
			sorted_gt_idx = sorted(matched.keys())
			meta['aligned'] = True
			meta['len_final'] = len(sorted_gt_idx)
			pred_aligned = pred_values[[matched[idx][0] for idx in sorted_gt_idx]]
			gt_aligned = gt_rpm_full[sorted_gt_idx]
			times_aligned = t_gt_full[sorted_gt_idx]
			return pred_aligned, gt_aligned, times_aligned, meta

		def _spectral_estimate_on_grid(filtered_sig, fps_val, window_size, centers):
			if filtered_sig is None or filtered_sig.size == 0:
				return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
			try:
				fps_val = float(fps_val)
			except (TypeError, ValueError):
				return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
			if fps_val <= 0:
				return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
			try:
				win_seconds = float(window_size)
			except (TypeError, ValueError):
				return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
			if win_seconds <= 0:
				return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
			centers = np.asarray(centers, dtype=np.float64).reshape(-1)
			if centers.size == 0:
				return np.asarray([], dtype=np.float64), centers
			win_frames = int(round(win_seconds * fps_val))
			if win_frames <= 0:
				return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
			half_window = 0.5 * win_seconds
			total_frames = filtered_sig.shape[1]
			windows_per_dim = [[] for _ in range(filtered_sig.shape[0])]
			valid_centers = []
			for center in centers:
				start_time = center - half_window
				end_time = center + half_window
				start_idx = int(round(start_time * fps_val))
				end_idx = start_idx + win_frames
				if start_idx < 0 or end_idx > total_frames:
					continue
				segment = filtered_sig[:, start_idx:end_idx]
				if segment.shape[1] != win_frames:
					continue
				for dim in range(filtered_sig.shape[0]):
					windows_per_dim[dim].append(np.asarray(segment[dim], dtype=np.float64))
				valid_centers.append(center)
			if not valid_centers:
				return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
			rpm_stack = []
			welch_win = max(1, int(round(win_seconds / 1.5)))
			for dim_windows in windows_per_dim:
				if not dim_windows:
					continue
				rpm_segment = np.asarray(
					utils.sig_to_RPM(dim_windows, fps_val, welch_win, min_hz, max_hz),
					dtype=np.float64
				).reshape(-1)
				rpm_stack.append(rpm_segment)
			if not rpm_stack:
				return np.asarray([], dtype=np.float64), np.asarray(valid_centers, dtype=np.float64)
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', category=RuntimeWarning)
				with np.errstate(invalid='ignore'):
					rpm_mean = np.nanmean(np.vstack(rpm_stack), axis=0)
			return rpm_mean, np.asarray(valid_centers, dtype=np.float64)

		# Extract estimation data
		fps = data['fps']

		for i, est in enumerate(data['estimates']):
			# Freeze the dict key early to avoid any later mutations shadowing the method name
			cur_method = est['method']
			method_key = str(cur_method)
			method_storage_name = est['method']
			capability = _infer_method_capability(method_storage_name)
			enable_track_processing = bool(use_track and capability == 'tracker')
			gating_flags = {
				'std': False,
				'uniq': False,
				'sat': False,
				'low_ratio': False,
				'low_prom': False,
				'wide_fwhm': False
			}
			spectral_stats = {}
			fallback_from = None
			is_trustworthy = True
			if cur_method == 'OF_Deep':
				cur_method += ofdeep_models[i]
			elif cur_method == 'OF_Model':
				cur_method = 'OF_Farneback'

			# >>> BEGIN per-method record scope (must stay inside the for-loop)
			try:
				sig = np.squeeze(est['estimate'])

				if win_size == 'video':
					if sig.ndim == 1:
						ws = len(sig) / fps
					else:
						ws = sig.shape[1] / fps
				else:
					ws = win_size

				if (sig.ndim == 1):
					sig = sig[np.newaxis,:]

				# Filter estimated signal over all dimensions
				filt_sig = []
				for d in range(sig.shape[0]):
					filt_sig.append(utils.filter_RW(sig[d,:], fps, lo=min_hz, hi=max_hz))

				filt_sig = np.vstack(filt_sig)

				if cur_method in ['bss_emd', 'bss_ssa']:
					filt_sig = utils.select_component(filt_sig, fps, int(ws/1.5), min_hz, max_hz)

				sanitized_method = method_storage_name.replace(' ', '_')
				sig_rpm_values = None
				t_sig = None
				track_used = False
				degenerate_track = False
				degenerate_reason = None
				track_stats = {}
				track_candidate_present = False
				track_series = None
				track_meta = {}
				rr_vals = None

			except Exception as exc:
				# Early fail-safe to keep evaluation running; downstream blocks proceed
				pass

			if enable_track_processing:
				aux_dir = os.path.join(results_dir, 'aux', sanitized_method)
				aux_file = os.path.join(aux_dir, f"{trial_key}.npz")
				if os.path.exists(aux_file):
					try:
						with np.load(aux_file, allow_pickle=True) as aux_data:
							if 'meta' in aux_data:
								try:
									meta_raw = aux_data['meta']
									meta_item = None
									if isinstance(meta_raw, np.ndarray):
										if meta_raw.size:
											meta_item = meta_raw.flat[0]
									else:
										meta_item = meta_raw
									if isinstance(meta_item, bytes):
										meta_item = meta_item.decode('utf-8', errors='ignore')
									if isinstance(meta_item, str):
										track_meta = json.loads(meta_item)
									elif isinstance(meta_item, dict):
										track_meta = meta_item
								except Exception as exc:
									track_meta = {}
									tqdm.write(f"> Warning: failed to parse aux meta for {method_storage_name}: {exc}")
							if 'track_hz' in aux_data:
								track_candidate_present = True
								track_series = np.asarray(aux_data['track_hz'], dtype=np.float64).reshape(-1)
								if track_series.size:
									track_series = np.clip(track_series, min_hz, max_hz)
									track_win, t_sig = utils.sig_windowing(track_series, fps, ws, stride=stride)
									if track_win:
										win_array = np.vstack([np.squeeze(w) for w in track_win])
										sig_rpm_values = np.median(win_array, axis=1) * 60.0
										track_used = True
							if 'rr_bpm' in aux_data:
								track_candidate_present = True
								rr_vals = np.asarray(aux_data['rr_bpm'], dtype=np.float64).reshape(-1)
					except Exception as exc:
						tqdm.write(f"> Warning: failed to read aux track for {method_storage_name}: {exc}")

			if enable_track_processing and (not track_used) and (rr_vals is not None):
				if rr_vals.size:
					if win_size == 'video':
						rr_val = float(rr_vals[-1])
						sig_rpm_values = np.asarray([rr_val], dtype=np.float64)
						if t_gt_full.size:
							t_sig = np.asarray(t_gt_full[:1], dtype=np.float64)
						else:
							t_sig = np.asarray([], dtype=np.float64)
						track_used = True
					else:
						if rr_vals.size == target_len and target_len == t_gt_full.size and target_len > 0:
							sig_rpm_values = rr_vals.astype(np.float64)
							t_sig = np.asarray(t_gt_full[:rr_vals.size], dtype=np.float64)
							track_used = True
						else:
							degenerate_track = True
							if degenerate_reason is None:
								degenerate_reason = 'length_mismatch'
				else:
					degenerate_track = True
					if degenerate_reason is None:
						degenerate_reason = 'low_unique'
	
			# Absolute margin in Hz prevents penalising physiologic extremes near the band edges.
			band_hz = float(max_hz) - float(min_hz)
			edge_margin_hz = float(saturation_margin_hz)
			if not np.isfinite(edge_margin_hz):
				edge_margin_hz = 0.0
			if band_hz > 0.0:
				max_margin = max(0.0, 0.5 * band_hz - 1e-8)
				if edge_margin_hz > max_margin:
					edge_margin_hz = max_margin
			else:
				edge_margin_hz = 0.0
			lower_edge = float(min_hz) + edge_margin_hz
			upper_edge = float(max_hz) - edge_margin_hz
	
			finite_track = np.asarray([], dtype=np.float64)
			sat_frac = float('nan')
			median_track_hz = float('nan')
			persist_samples = 1
			if saturation_persist_sec > 0.0 and fps > 0:
				persist_samples = max(1, int(round(float(saturation_persist_sec) * float(fps))))
			if track_series is not None:
				finite_track = track_series[np.isfinite(track_series)]
				if finite_track.size:
					median_track_hz = float(np.nanmedian(finite_track))
					if lower_edge <= upper_edge:
						sat_mask_raw = (finite_track <= (lower_edge + 1e-12)) | (finite_track >= (upper_edge - 1e-12))
					else:
						sat_mask_raw = (finite_track <= float(min_hz)) | (finite_track >= float(max_hz))
					if sat_mask_raw.size:
						if persist_samples > 1:
							kernel = np.ones(persist_samples, dtype=np.int32)
							convolved = np.convolve(sat_mask_raw.astype(np.int32), kernel, mode='same')
							sat_mask = convolved >= persist_samples
						else:
							sat_mask = sat_mask_raw
						sat_frac = float(np.mean(sat_mask))
					else:
						sat_frac = float('nan')
				else:
					sat_frac = 1.0
	
			mean_rpm = float('nan')
			est_std = float('nan')
			nuniq_frac = 0.0
			n_windows_est = 0
			finite_rpm = np.asarray([], dtype=np.float64)
			allow_constant_track = bool(track_meta.get('is_constant_track')) if isinstance(track_meta, dict) else False
			meta_for_stats = dict(track_meta) if isinstance(track_meta, dict) else {}
			meta_for_stats['track_frac_saturated_eval'] = float(sat_frac) if np.isfinite(sat_frac) else float('nan')
			meta_for_stats['saturation_margin_hz_used'] = float(edge_margin_hz)
			meta_for_stats['saturation_persist_samples'] = int(persist_samples)
			meta_for_stats['saturation_persist_sec'] = float(saturation_persist_sec)
			meta_for_stats['median_track_hz_eval'] = float(median_track_hz) if np.isfinite(median_track_hz) else float('nan')
			meta_for_stats['edge_saturation_fraction'] = float(sat_frac) if np.isfinite(sat_frac) else float('nan')
			meta_for_stats['std_is_soft'] = bool(track_std_is_soft)
			meta_for_stats['constant_track_promoted'] = bool(meta_for_stats.get('constant_track_promoted', False))
			constant_promoted = False
	
			if track_used and sig_rpm_values is not None:
				finite_mask_rpm = np.isfinite(sig_rpm_values)
				finite_rpm = sig_rpm_values[finite_mask_rpm]
				n_windows_est = int(sig_rpm_values.size)
				if finite_rpm.size:
					with np.errstate(invalid='ignore'):
						est_std = float(np.nanstd(finite_rpm))
						mean_rpm = float(np.nanmean(finite_rpm))
					unique_vals = np.unique(finite_rpm)
					nuniq_frac = float(unique_vals.size / max(1, finite_rpm.size))
			std_below_threshold = bool(np.isfinite(est_std) and (est_std < float(track_std_min_bpm)))
			meta_for_stats['std_below_threshold'] = std_below_threshold
			meta_for_stats['unique_below_threshold'] = bool(nuniq_frac < float(track_unique_min))
			meta_for_stats['edge_saturation_breach'] = bool(np.isfinite(sat_frac) and (sat_frac > float(track_saturation_max)))
	
			if constant_ptp_max_hz > 0.0 and finite_track.size:
				# Robust range (P95−P5) guards against transient spikes while detecting quasi-constant tracks.
				perc5 = float(np.percentile(finite_track, 5))
				perc95 = float(np.percentile(finite_track, 95))
				robust_range = float(max(0.0, perc95 - perc5))
				meta_for_stats['track_range_hz_p5_p95'] = robust_range if np.isfinite(robust_range) else float('nan')
				meta_for_stats['track_range_hz_ptp'] = float(np.ptp(finite_track))
				if np.isfinite(robust_range) and robust_range <= float(constant_ptp_max_hz):
					interior_ok = False
					if np.isfinite(median_track_hz):
						if lower_edge <= upper_edge:
							interior_ok = (median_track_hz >= lower_edge) and (median_track_hz <= upper_edge)
						else:
							interior_ok = (median_track_hz >= float(min_hz)) and (median_track_hz <= float(max_hz))
					if interior_ok:
						if not allow_constant_track:
							constant_promoted = True
						allow_constant_track = True
						if meta_for_stats is not None:
							meta_for_stats['constant_track_promoted'] = True
	
			track_stats = {
				'mean_bpm': mean_rpm if np.isfinite(mean_rpm) else float('nan'),
				'std_bpm': est_std if np.isfinite(est_std) else float('nan'),
				'unique_fraction': nuniq_frac,
				'saturation_fraction': sat_frac,
				'n_windows': int(n_windows_est),
				'median_hz': meta_for_stats.get('median_track_hz_eval'),
				'track_dyn_range_hz_ptp': meta_for_stats.get('track_range_hz_ptp'),
				'edge_saturation_fraction': meta_for_stats.get('edge_saturation_fraction'),
				'constant_track_promoted': meta_for_stats.get('constant_track_promoted'),
				'meta': meta_for_stats
			}
			trigger_reason = None
			fallback_due_to_gating = False
			if enable_track_processing and track_used and sig_rpm_values is not None:
				if finite_rpm.size == 0:
					trigger_reason = 'empty_or_allnan'
				elif finite_rpm.size < 2 and win_size != 'video':
					trigger_reason = 'empty_or_allnan'
				elif win_size != 'video':
					std_invalid = not np.isfinite(est_std)
					if std_invalid:
						gating_flags['std'] = True
						if not allow_constant_track:
							trigger_reason = 'low_std'
							fallback_due_to_gating = True
					elif std_below_threshold:
						gating_flags['std'] = True
						if (not allow_constant_track) and (not track_std_is_soft):
							trigger_reason = 'low_std'
							fallback_due_to_gating = True
					if (trigger_reason is None) and (nuniq_frac < float(track_unique_min)) and (not allow_constant_track):
						trigger_reason = 'low_unique'
						gating_flags['uniq'] = True
						fallback_due_to_gating = True
					elif (trigger_reason is None) and np.isfinite(sat_frac) and (sat_frac > float(track_saturation_max)):
						sat_repr = f"{sat_frac:.3f}" if np.isfinite(sat_frac) else "nan"
						med_repr = f"{median_track_hz:.3f}" if np.isfinite(median_track_hz) else "nan"
						const_repr = 'promoted' if constant_promoted else ('true' if allow_constant_track else 'false')
						persist_repr = f"{persist_samples}x@{float(saturation_persist_sec):.2f}s"
						band_repr = f"{float(min_hz):.3f}-{float(max_hz):.3f}"
						reason_detail = (
							f"high_saturation@{edge_margin_hz:.3f}"
							f"[band={band_repr};sat={sat_repr};med={med_repr};const={const_repr};persist={persist_repr}]"
						)
						should_trigger = True
						if allow_constant_track and finite_track.size:
							at_edge = False
							if np.isfinite(median_track_hz):
								if lower_edge <= upper_edge:
									at_edge = (median_track_hz <= lower_edge) or (median_track_hz >= upper_edge)
								else:
									at_edge = (median_track_hz <= float(min_hz)) or (median_track_hz >= float(max_hz))
							if (not at_edge) or sat_frac < 0.98:
								should_trigger = False
						if should_trigger:
							trigger_reason = reason_detail
							gating_flags['sat'] = True
							fallback_due_to_gating = True
			if fallback_due_to_gating and disable_gating:
				trigger_reason = None
				fallback_due_to_gating = False
			if trigger_reason:
				track_used = False
				sig_rpm_values = None
				degenerate_track = True
				if degenerate_reason is None:
					degenerate_reason = trigger_reason
				if track_candidate_present and degenerate_reason is None:
					degenerate_track = True
					degenerate_reason = 'empty_or_allnan'
				if capability == 'tracker':
					fallback_from = 'track'
	
			if not track_used:
				if enable_track_processing:
					flag_snapshot = {k: v for k, v in gating_flags.items() if v}
					if track_candidate_present:
						reason = degenerate_reason or 'unknown'
						tqdm.write(f"WARNING [evaluate] method={method_storage_name} cap=tracker flags={flag_snapshot or gating_flags} reason={reason} -> fallback=spectral")
					elif '__' in sanitized_method:
						if degenerate_reason is None:
							degenerate_reason = 'missing_track'
						degenerate_track = True
						tqdm.write(f"WARNING [evaluate] method={method_storage_name} cap=tracker flags={flag_snapshot or gating_flags} reason={degenerate_reason} -> fallback=spectral")
					else:
						tqdm.write(f"> Info: no track for base method {method_storage_name}::{trial_key}; using spectral RPM")
				sig_rpm_values, t_sig = _spectral_estimate_on_grid(filt_sig, fps, ws, t_gt_full)
				spectral_stats = _spectral_stats_from_signal(filt_sig, fps, ws, min_hz, max_hz)
				if capability == 'tracker':
					source_label = 'fallback'
					if fallback_from is None:
						fallback_from = 'track'
				else:
					source_label = 'spectral'
			else:
				source_label = 'track' if capability == 'tracker' else 'spectral'
	
			recomputed_spectral = False
			while True:
				if sig_rpm_values is None:
					sig_array_raw = np.asarray([], dtype=np.float64)
				else:
					sig_array_raw = np.asarray(sig_rpm_values, dtype=np.float64).reshape(-1)
				pred_len_raw = int(sig_array_raw.size)
				if t_sig is None:
					t_sig_array = np.asarray([], dtype=np.float64)
				else:
					t_sig_array = np.asarray(t_sig, dtype=np.float64).reshape(-1)
				if pred_len_raw and t_sig_array.size and pred_len_raw != t_sig_array.size:
					min_len_local = min(pred_len_raw, t_sig_array.size)
					sig_array = sig_array_raw[:min_len_local]
					t_sig_local = t_sig_array[:min_len_local]
				else:
					sig_array = sig_array_raw
					t_sig_local = t_sig_array
				sig_aligned_tmp, gt_aligned_tmp, times_aligned_tmp, align_meta = _align_to_gt_grid(t_sig_local, sig_array)
				align_meta['len_pred'] = pred_len_raw
				align_meta['len_gt'] = int(gt_rpm_full.size)
				sig_aligned_tmp = np.asarray(sig_aligned_tmp, dtype=np.float64).reshape(-1)
				gt_aligned_tmp = np.asarray(gt_aligned_tmp, dtype=np.float64).reshape(-1)
				times_aligned_tmp = np.asarray(times_aligned_tmp, dtype=np.float64).reshape(-1)
				min_len_aligned = min(sig_aligned_tmp.size, gt_aligned_tmp.size, times_aligned_tmp.size)
				if min_len_aligned:
					sig_aligned_tmp = sig_aligned_tmp[:min_len_aligned]
					gt_aligned_tmp = gt_aligned_tmp[:min_len_aligned]
					times_aligned_tmp = times_aligned_tmp[:min_len_aligned]
				else:
					sig_aligned_tmp = np.asarray([], dtype=np.float64)
					gt_aligned_tmp = np.asarray([], dtype=np.float64)
					times_aligned_tmp = np.asarray([], dtype=np.float64)
				align_meta['len_final'] = int(min_len_aligned)
				finite_mask_pair = np.isfinite(sig_aligned_tmp) & np.isfinite(gt_aligned_tmp)
				valid_count = int(finite_mask_pair.sum())
				align_meta['len_valid'] = valid_count
				if valid_count:
					sig_valid_tmp = sig_aligned_tmp[finite_mask_pair]
					gt_valid_tmp = gt_aligned_tmp[finite_mask_pair]
					times_valid_tmp = times_aligned_tmp[finite_mask_pair]
				else:
					sig_valid_tmp = np.asarray([], dtype=np.float64)
					gt_valid_tmp = np.asarray([], dtype=np.float64)
					times_valid_tmp = np.asarray([], dtype=np.float64)
				if source_label == 'track' and (align_meta['len_final'] == 0 or valid_count < 2) and not recomputed_spectral:
					degenerate_track = True
					if degenerate_reason is None:
						degenerate_reason = 'empty_or_allnan'
					track_used = False
					source_label = 'fallback' if capability == 'tracker' else 'spectral'
					if enable_track_processing:
						if track_candidate_present:
							reason = degenerate_reason or 'unknown'
							tqdm.write(f"> Warning: track rejected for {method_storage_name}::{trial_key} [reason: {reason}], falling back to spectral RPM")
						elif '__' in sanitized_method:
							if degenerate_reason is None:
								degenerate_reason = 'missing_track'
							degenerate_track = True
							tqdm.write(f"> Warning: track missing for {method_storage_name}::{trial_key}; falling back to spectral RPM")
						else:
							tqdm.write(f"> Info: no track for base method {method_storage_name}::{trial_key}; using spectral RPM")
					sig_rpm_values, t_sig = _spectral_estimate_on_grid(filt_sig, fps, ws, t_gt_full)
					spectral_stats = _spectral_stats_from_signal(filt_sig, fps, ws, min_hz, max_hz)
					if capability == 'tracker' and fallback_from is None:
						fallback_from = 'track'
					recomputed_spectral = True
					continue
				sig_aligned = sig_aligned_tmp
				gt_aligned = gt_aligned_tmp
				times_aligned = times_aligned_tmp
				sig_valid = sig_valid_tmp
				gt_valid = gt_valid_tmp
				times_valid = times_valid_tmp
				alignment_meta = align_meta
				break
	
			sig_finite = sig_valid[np.isfinite(sig_valid)] if sig_valid.size else np.asarray([], dtype=np.float64)
			gt_finite = gt_valid[np.isfinite(gt_valid)] if gt_valid.size else np.asarray([], dtype=np.float64)
	
			if sig_finite.size:
				with np.errstate(invalid='ignore'):
					est_mean_val = float(np.mean(sig_finite, dtype=np.float64))
					est_std_val = float(np.std(sig_finite, dtype=np.float64))
			else:
				est_mean_val = float('nan')
				est_std_val = float('nan')
	
			if gt_finite.size:
				with np.errstate(invalid='ignore'):
					gt_mean_val = float(np.mean(gt_finite, dtype=np.float64))
					gt_std_val = float(np.std(gt_finite, dtype=np.float64))
			else:
				gt_mean_val = float('nan')
				gt_std_val = float('nan')
	
			if capability == 'spectral':
				if not spectral_stats:
					spectral_stats = _spectral_stats_from_signal(filt_sig, fps, ws, min_hz, max_hz)
				trustworthy, spectral_flag_values, spectral_reason = _spectral_quality_flags(spectral_stats, resolved_spectral_cfg, disable_gating)
				for key, value in spectral_flag_values.items():
					if key in gating_flags:
						gating_flags[key] = value
					else:
						gating_flags[key] = value
				is_trustworthy = trustworthy
				if (not trustworthy) and spectral_reason and degenerate_reason is None:
					degenerate_reason = spectral_reason
				if not trustworthy:
					flag_snapshot = {k: v for k, v in gating_flags.items() if v}
					tqdm.write(f"WARNING [evaluate] method={method_storage_name} cap=spectral flags={flag_snapshot or gating_flags} trust=False")
			else:
				is_trustworthy = True
	
			if sig_valid.size >= 2:
				with np.errstate(invalid='ignore'):
					std_pair_est = float(np.std(sig_valid, dtype=np.float64))
			else:
				std_pair_est = float('nan')
			if gt_valid.size >= 2:
				with np.errstate(invalid='ignore'):
					std_pair_gt = float(np.std(gt_valid, dtype=np.float64))
			else:
				std_pair_gt = float('nan')
	
			if sig_valid.size >= 2 and gt_valid.size >= 2:
				if (np.isfinite(std_pair_gt) and std_pair_gt < 1e-12) or (np.isfinite(std_pair_est) and std_pair_est < 1e-12):
					scatter_slope = scatter_intercept = float('nan')
				else:
					with np.errstate(invalid='ignore'):
						try:
							scatter_slope, scatter_intercept = np.polyfit(gt_valid, sig_valid, 1)
						except Exception:
							scatter_slope = scatter_intercept = float('nan')
			else:
				scatter_slope = scatter_intercept = float('nan')
	
			if sig_valid.size >= 2 and gt_valid.size >= 2:
				corr_degenerate = (
					(np.isfinite(std_pair_gt) and std_pair_gt < corr_guard_eps)
					or (np.isfinite(std_pair_est) and std_pair_est < corr_guard_eps)
				)
			else:
				corr_degenerate = True

			if sig_valid.size >= 2 and gt_valid.size >= 2:
				e = errors.getErrors(sig_valid, gt_valid, times_valid, times_valid, metrics)
				raw_metrics = e[:-1]
				pair = e[-1]
				if corr_degenerate:
					for idx, metric_name in enumerate(metrics):
						if metric_name in ('PCC', 'CCC'):
							raw_metrics[idx] = float('nan')
			else:
				raw_metrics = [float('nan')] * len(metrics)
				pair = [
					np.atleast_2d(sig_valid.astype(np.float64)),
					gt_valid.astype(np.float64)
				]
	
			metric_values = []
			for val in raw_metrics:
				try:
					metric_values.append(float(val))
				except Exception:
					try:
						val_arr = np.atleast_1d(val).astype(float)
						metric_values.append(float(val_arr.flat[0]))
					except Exception:
						metric_values.append(float('nan'))
	
			times_valid_for_record = np.asarray(times_valid, dtype=np.float64)
			record_alignment = {
				'aligned': bool(alignment_meta.get('aligned', False)),
				'len_gt': int(alignment_meta.get('len_gt', 0)),
				'len_pred': int(alignment_meta.get('len_pred', 0)),
				'len_final': int(alignment_meta.get('len_final', 0)),
				'len_valid': int(alignment_meta.get('len_valid', 0))
			}
			spectral_snapshot = spectral_stats if isinstance(spectral_stats, dict) else {}
			quality_snapshot = {
				'track_used': bool(track_used),
				'track_candidate_present': bool(track_candidate_present),
				'source_label': source_label,
				'degenerate_reason': degenerate_reason,
				'fallback_from': fallback_from,
				'std_bpm': track_stats.get('std_bpm'),
				'unique_fraction': track_stats.get('unique_fraction'),
				'track_dyn_range_hz_ptp': track_stats.get('track_dyn_range_hz_ptp'),
				'median_hz': track_stats.get('median_hz'),
				'edge_saturation_fraction': track_stats.get('edge_saturation_fraction'),
				'constant_track_promoted': track_stats.get('constant_track_promoted'),
				'spectral_peak_ratio': spectral_snapshot.get('peak_ratio'),
				'spectral_prominence_db': spectral_snapshot.get('prominence_db'),
				'spectral_fwhm_hz': spectral_snapshot.get('fwhm_hz'),
				'spectral_welch_df_hz': spectral_snapshot.get('welch_df_hz'),
				'gating_flags': {k: bool(v) for k, v in gating_flags.items()},
				'std_is_soft': bool(track_std_is_soft),
				'std_violation_soft': bool(std_below_threshold and track_std_is_soft),
				'std_invalid': bool(not np.isfinite(est_std)),
				'trial_key': trial_key,
				'data_file': filepath
			}
	
			try:
				record = {
					'metrics': metric_values,
					'pair': pair,
					'source': source_label,
					'source_label': source_label,
					'capability': capability,
					'stride': stride,
					'times_est': times_valid_for_record,
					'times_gt': times_valid_for_record,
					'degenerate_track': bool(degenerate_track),
					'degenerate_reason': degenerate_reason,
					'track_stats': track_stats,
					'spectral_stats': spectral_stats,
					'gating_flags': dict(gating_flags),
					'fallback_from': fallback_from,
					'is_trustworthy': bool(is_trustworthy),
					'est_mean': est_mean_val,
					'est_std': est_std_val,
					'gt_mean': gt_mean_val,
					'gt_std': gt_std_val,
					'scatter_slope': float(scatter_slope) if np.isfinite(scatter_slope) else float('nan'),
					'scatter_intercept': float(scatter_intercept) if np.isfinite(scatter_intercept) else float('nan'),
					'aligned': record_alignment['aligned'],
					'len_gt': record_alignment['len_gt'],
					'len_pred': record_alignment['len_pred'],
					'len_final': record_alignment['len_final'],
					'len_valid': record_alignment['len_valid'],
					'alignment': record_alignment,
					'quality': quality_snapshot,
					'track_used': bool(track_used),
					'trial_key': trial_key,
					'data_file': filepath
				}
				method_metrics.setdefault(method_key, []).append(record)
				if debug_log_path:
					with open(debug_log_path, 'a') as fp:
						fp.write(f"{os.path.basename(filepath)}\t{method_key}\tsource={source_label}\tlen_valid={record.get('len_valid', 0)}\n")
			except Exception as exc:
				placeholder = {
					'metrics': [float('nan')] * len(metrics),
					'pair': [np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)],
					'source': 'error',
					'source_label': 'error',
					'capability': capability,
					'stride': stride,
					'times_est': np.asarray([], dtype=np.float64),
					'times_gt': np.asarray([], dtype=np.float64),
					'degenerate_track': True,
					'degenerate_reason': f'exception:{type(exc).__name__}',
					'track_stats': {},
					'spectral_stats': {},
					'gating_flags': dict(gating_flags),
					'fallback_from': fallback_from,
					'is_trustworthy': False,
					'est_mean': float('nan'),
					'est_std': float('nan'),
					'gt_mean': float('nan'),
					'gt_std': float('nan'),
					'scatter_slope': float('nan'),
					'scatter_intercept': float('nan'),
					'aligned': False,
					'len_gt': 0,
					'len_pred': 0,
					'len_final': 0,
					'len_valid': 0,
					'alignment': {'aligned': False, 'len_gt': 0, 'len_pred': 0, 'len_final': 0, 'len_valid': 0},
					'quality': {
						'track_used': False,
						'track_candidate_present': False,
						'source_label': 'error',
						'degenerate_reason': f'exception:{type(exc).__name__}',
						'fallback_from': fallback_from,
						'std_bpm': float('nan'),
						'unique_fraction': float('nan'),
						'track_dyn_range_hz_ptp': float('nan'),
						'median_hz': float('nan'),
						'edge_saturation_fraction': float('nan'),
						'constant_track_promoted': False,
						'spectral_peak_ratio': float('nan'),
						'spectral_prominence_db': float('nan'),
						'spectral_fwhm_hz': float('nan'),
						'spectral_welch_df_hz': float('nan'),
						'gating_flags': {k: bool(v) for k, v in gating_flags.items()},
						'std_is_soft': bool(track_std_is_soft),
						'std_violation_soft': False,
						'std_invalid': True,
						'trial_key': trial_key,
						'data_file': filepath
					},
					'track_used': False,
					'trial_key': trial_key,
					'data_file': filepath
				}
				method_metrics.setdefault(method_key, []).append(placeholder)
				if debug_log_path:
					try:
						with open(debug_log_path, 'a') as fp:
							fp.write(f"{os.path.basename(filepath)}\t{method_key}\tlen_valid=0\tEXC={type(exc).__name__}\n")
					except Exception:
						pass
	
	# Before saving, report which methods produced metrics (helps debug filtering issues)
	try:
		seen_methods = sorted(list(method_metrics.keys()))
		print(f"\n> Methods evaluated: {len(seen_methods)} -> {', '.join(seen_methods) if seen_methods else '(none)'}")
		# Persist a small debug log to help users verify which methods were seen
		logs_dir = os.path.join(results_dir, 'logs')
		os.makedirs(logs_dir, exist_ok=True)
		with open(os.path.join(logs_dir, 'methods_seen.txt'), 'w') as fp:
			for m in seen_methods:
				fp.write(m + "\n")
	except Exception as _exc:
		# Non-fatal; continue to saving metrics
		pass

	# Choose metrics output directory (new structure prefers results_dir/metrics)
	metrics_dir = os.path.join(results_dir, 'metrics') if os.path.isdir(os.path.join(results_dir, 'metrics')) else results_dir
	if win_size == 'video':
		fn = 'metrics_1w.pkl'
	else:
		fn = 'metrics.pkl'
	settings_path = os.path.join(metrics_dir, 'eval_settings.json')
	try:
		with open(settings_path, 'w') as fp:
			json.dump({
				"win_size": win_size,
				"stride": stride,
				"min_hz": min_hz,
				"max_hz": max_hz,
				"use_track": use_track,
				"track_std_min_bpm": track_std_min_bpm,
				"track_std_is_soft": track_std_is_soft,
				"track_unique_min": track_unique_min,
				"track_saturation_max": track_saturation_max,
				"saturation_margin_hz": saturation_margin_hz,
				"saturation_persist_sec": saturation_persist_sec,
				"constant_ptp_max_hz": constant_ptp_max_hz,
				"gating": gating_cfg
			}, fp, indent=2)
	except Exception as exc:
		print(f"> Warning: failed to write evaluation settings ({exc})")
	# Save the results of the applied methods
	print(f"> Saving metrics to {os.path.join(metrics_dir, fn)}")
	with open(os.path.join(metrics_dir, fn), 'wb') as fp:
		pickle.dump([metrics, method_metrics] , fp)
		print('> Metrics saved!\n')

	# Save human-readable table and plots (best-effort)
	try:
		annotation = f"# eval_band [{min_hz:.2f}, {max_hz:.2f}] Hz | use_track={use_track}"
		summaries = _save_metrics_table(metrics_dir, method_metrics, metrics, win_size == 'video', annotation=annotation)
		if visualize:
			_generate_plots(results_dir, summaries, metrics, win_size == 'video', stride=stride)
	except Exception as e:
		print(f"> Plot/log generation skipped due to error: {e}")
	quality_msg = _persist_quality_reports(method_metrics, results_dir)
	if quality_msg:
		print(quality_msg)
	return method_metrics


def print_metrics(results_dir, unique_window=False):
	if unique_window:
		print("Considering one window per video\n")
		fn = 'metrics_1w.pkl'
	else:
		print("Considering time windowing per each video\n")
		fn = 'metrics.pkl'

	# Load the calculated metrics (prefer new structure metrics/)
	metrics_dir = os.path.join(results_dir, 'metrics') if os.path.isdir(os.path.join(results_dir, 'metrics')) else results_dir
	with open(os.path.join(metrics_dir, fn), 'rb') as f: 
		metrics, method_metrics = pickle.load(f)
		metrics, method_metrics = _normalize_metrics_payload(metrics, method_metrics)
	settings_path = os.path.join(metrics_dir, 'eval_settings.json')
	if os.path.exists(settings_path):
		try:
			with open(settings_path, 'r') as fp:
				eval_settings = json.load(fp)
			min_hz = eval_settings.get('min_hz')
			max_hz = eval_settings.get('max_hz')
			use_track = eval_settings.get('use_track')
			if min_hz is not None and max_hz is not None:
				print(f"Evaluation band: [{min_hz:.2f}, {max_hz:.2f}] Hz | use_track={bool(use_track)}\n")
		except Exception:
			pass

	summaries = _summaries_with_samples(method_metrics, metrics, unique_window)

	if unique_window:
		headers = ['Method'] + PRIMARY_METRICS
	else:
		headers = ['Method'] + metrics
	table_rows = []

	for method, summary in summaries.items():
		values = summary['values']
		if unique_window:
			row_vals = [method]
			for key in PRIMARY_METRICS:
				row_vals.append(_format_scalar(values.get(key, float('nan'))))
		else:
			row_vals = [method]
			for metric in metrics:
				med = values.get(metric, float('nan'))
				std = summary['variability'].get(f'{metric}_std', float('nan'))
				row_vals.append(f"{_format_scalar(med)} ({_format_scalar(std, decimals=2)})")
		table_rows.append(row_vals)

	table_str = _render_table(headers, table_rows)

	print(table_str)
	# Save alongside pkl for reporting
	try:
		with open(os.path.join(metrics_dir, 'metrics_summary.txt'), 'w') as fp:
			fp.write(table_str + "\n")
	except Exception:
		pass


def _normalize_metrics_payload(metric_names, method_metrics):
	"""Drop deprecated CORR metric from stored payloads while keeping entries aligned."""
	if not metric_names:
		return metric_names, method_metrics
	names = list(metric_names)
	if 'CORR' not in names:
		return names, method_metrics

	drop_indices = [idx for idx, name in enumerate(metric_names) if name == 'CORR']
	removed = 0
	for idx in drop_indices:
		target = idx - removed
		if 0 <= target < len(names):
			names.pop(target)
		if method_metrics:
			for records in method_metrics.values():
				for record in records or []:
					vals = record.get('metrics')
					if vals is None:
						continue
					if isinstance(vals, tuple):
						vals = list(vals)
						record['metrics'] = vals
					if isinstance(vals, list) and len(vals) > target:
						del vals[target]
		removed += 1

	return names, method_metrics


@timed_step('extract_respiration')
def extract_respiration(datasets, methods, results_dir, run_label=None, manifest_methods=None, method_order=None):
	os.makedirs(results_dir, exist_ok=True)
	all_methods = manifest_methods or methods
	method_order = method_order or [m.name if hasattr(m, 'name') else str(m) for m in all_methods]
	method_suffix = _method_suffix(all_methods)
	sanitized_label = _sanitize_run_label(run_label) if run_label else None
	single_dataset = len(datasets) == 1

	for dataset in datasets:
		if sanitized_label:
			if single_dataset:
				dir_name = sanitized_label
			else:
				dir_name = f"{sanitized_label}_{dataset.name.upper()}"
		else:
			dir_name = f"{dataset.name.upper()}_{method_suffix}"
		dataset_results_dir = _dataset_results_dir(results_dir, dir_name)
		data_dir = os.path.join(dataset_results_dir, 'data')
		manifest_path = os.path.join(dataset_results_dir, 'methods.json')
		os.makedirs(data_dir, exist_ok=True)
		try:
			manifest_payload = []
			for m in all_methods:
				if hasattr(m, 'name'):
					manifest_payload.append(m.name)
				elif isinstance(m, str):
					manifest_payload.append(m)
				else:
					manifest_payload.append(str(m))
			_atomic_json_dump(manifest_payload, manifest_path, indent=2)
		except Exception as exc:
			print(f"> Warning: failed to write methods manifest ({exc})")

		dataset.load_dataset()
		# Loop over the dataset
		for d in tqdm(dataset.data, desc="Processing files"):

			if 'trial' in d.keys(): 
				outfilename = os.path.join(data_dir, dataset.name + '_' + d['subject'] + '_' + d['trial'] + '.pkl')
				trial_key = f"{d['subject']}_{d['trial']}"
			else:
				outfilename = os.path.join(data_dir, dataset.name + '_' + d['subject'] + '.pkl')
				trial_key = d['subject']

			_, d['fps'] = utils.get_vid_stats(d['video_path'])
			d['trial_key'] = trial_key

			results_payload = {
				'video_path': d['video_path'],
				'fps': d['fps'],
				'gt': d['gt'],
				'fs_gt': float(dataset.fs_gt) if dataset.fs_gt is not None else None,
				'estimates': []
			}

			if 'trial' in d.keys(): 
				tqdm.write("> Processing video %s/%s\n> fps: %d" % (d['subject'], d['trial'], d['fps']))
			else:
				tqdm.write("> Processing video %s\n> fps: %d" % (d['subject'], d['fps']))

			# Apply every method to each video
			for m in methods:
				tqdm.write("> Applying method %s ..." % m.name)
				skip_method = False
				aux_dir = os.path.join(dataset_results_dir, 'aux', m.name.replace(' ', '_'))
				d['aux_save_dir'] = aux_dir

				if m.data_type == 'chest':
					if not d['chest_rois']:
						d['chest_rois'] = _filter_valid_rois(dataset.extract_ROI(d['video_path'], m.data_type))
					else:
						d['chest_rois'] = _filter_valid_rois(d['chest_rois'])
					if not d['chest_rois']:
						tqdm.write(f"> Skipping method {m.name} (no valid chest ROIs)")
						skip_method = True
				elif m.data_type == 'face':
					if not d['face_rois']:
						d['face_rois'] = _filter_valid_rois(dataset.extract_ROI(d['video_path'], m.data_type))
					else:
						d['face_rois'] = _filter_valid_rois(d['face_rois'])
					if not d['face_rois']:
						tqdm.write(f"> Skipping method {m.name} (no valid face ROIs)")
						skip_method = True
				elif m.data_type == 'rppg':
					if not d['rppg_obj']:
						d['rppg_obj'] = dataset.extract_rppg(d['video_path'])

				if skip_method:
					continue

				estimate = m.process(d)
				results_payload['estimates'].append({'method': m.name, 'estimate': estimate})

			# release some memory between videos
			d.pop('aux_save_dir', None)
			d.pop('trial_key', None)
			d['chest_rois'] = []
			d['face_rois'] = []
			d['rppg_obj'] = None

			_merge_results_payload(outfilename, results_payload, method_order=method_order)
			tqdm.write('> Results updated!\n')

def _summaries_with_samples(method_metrics, metrics, unique_window):
	"""Compute aggregated metrics and retain representative samples for plotting."""
	summaries = {}
	if not method_metrics:
		return summaries
	if unique_window:
		from errors import RMSEerror, MAEerror, MAPEerror, PearsonCorr, LinCorr
	for method, records in method_metrics.items():
		if not records:
			summaries[method] = {
				'values': {m: float('nan') for m in metrics},
				'variability': {f'{m}_std': float('nan') for m in metrics},
				'sample': {'est': None, 'gt': None},
				'source': 'unknown',
				'record': None
			}
			continue
		first_record = records[0]
		pair = first_record.get('pair')
		try:
			sample_est = np.atleast_1d(np.squeeze(pair[0])).astype(float) if pair is not None else None
			sample_gt = np.atleast_1d(np.squeeze(pair[1])).astype(float) if pair is not None else None
		except Exception:
			sample_est = sample_gt = None

		values = {}
		variability = {}
		if unique_window:
			est_vals = []
			gt_vals = []
			for record in records:
				pair = record.get('pair')
				if pair is None:
					continue
				try:
					est = np.atleast_1d(np.squeeze(pair[0])).astype(float)
					gt = np.atleast_1d(np.squeeze(pair[1])).astype(float)
				except Exception:
					continue
				n = min(est.shape[-1] if est.ndim else est.size, gt.shape[-1] if gt.ndim else gt.size)
				if n == 0:
					continue
				est_vals.append(est.reshape(-1)[:n])
				gt_vals.append(gt.reshape(-1)[:n])
			if est_vals and gt_vals:
				est_concat = np.concatenate(est_vals).reshape(1, -1)
				gt_concat = np.concatenate(gt_vals)
				values['RMSE'] = RMSEerror(est_concat, gt_concat)
				values['MAE'] = MAEerror(est_concat, gt_concat)
				values['MAPE'] = MAPEerror(est_concat, gt_concat)
				values['PCC'] = PearsonCorr(est_concat, gt_concat)
				values['CCC'] = LinCorr(est_concat, gt_concat)
			else:
				for key in PRIMARY_METRICS:
					values[key] = float('nan')
		else:
			for idx, metric_name in enumerate(metrics):
				data = []
				for record in records:
					vals = record.get('metrics', [])
					if idx < len(vals):
						data.append(vals[idx])
				finite_data = [float(d) for d in data if isinstance(d, (int, float, np.floating)) and np.isfinite(d)]
				if finite_data:
					arr = np.asarray(finite_data, dtype=np.float64)
					med = float(np.median(arr))
					std = float(np.std(arr, dtype=np.float64))
				else:
					med = float('nan')
					std = float('nan')
				values[metric_name] = float(med) if not np.isnan(med) else float('nan')
				variability[f'{metric_name}_std'] = float(std) if not np.isnan(std) else float('nan')

		summaries[method] = {
			'values': values,
			'variability': variability,
			'sample': {'est': sample_est, 'gt': sample_gt},
			'source': first_record.get('source', 'unknown'),
			'record': first_record
		}
	return summaries


def _format_scalar(value, decimals=3):
	if isinstance(value, (float, np.floating)):
		if np.isnan(value):
			return "nan"
		return f"{float(value):.{decimals}f}"
	return str(value)


def _render_table(headers, rows):
	try:
		from prettytable import PrettyTable  # Optional dependency
	except ModuleNotFoundError:
		PrettyTable = None

	if PrettyTable is not None:
		t = PrettyTable(headers)
		for row in rows:
			t.add_row(row)
		return t.get_string()

	col_widths = [len(h) for h in headers]
	for row in rows:
		for idx, cell in enumerate(row):
			col_widths[idx] = max(col_widths[idx], len(cell))

	def _fmt_row(cells):
		return "|" + "|".join(f" {cell.ljust(col_widths[i])} " for i, cell in enumerate(cells)) + "|"

	separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
	lines = [separator, _fmt_row(headers), separator]
	for row in rows:
		lines.append(_fmt_row(row))
	lines.append(separator)
	return "\n".join(lines)



def _save_metrics_table(metrics_dir, method_metrics, metrics, unique_window, annotation=None):
	"""Save a pretty metrics summary table to text for reporting."""
	summaries = _summaries_with_samples(method_metrics, metrics, unique_window)
	if unique_window:
		table_headers = ['Method'] + PRIMARY_METRICS
	else:
		# 원본 구현과 동일: 중앙값±표준편차
		table_headers = ['Method'] + [f"{m} (median±std)" for m in metrics]
	table_rows = []
	for method, summary in summaries.items():
		values = summary['values']
		if unique_window:
			row = [method]
			for metric in PRIMARY_METRICS:
				val = values.get(metric, float('nan'))
				row.append(_format_scalar(val))
		else:
			row = [method]
			for metric in metrics:
				median_val = values.get(metric, float('nan'))
				std = summary['variability'].get(f'{metric}_std', float('nan'))
				row.append(f"{_format_scalar(median_val)} (±{_format_scalar(std, decimals=2)})")
		table_rows.append(row)

	table_str = _render_table(table_headers, table_rows)

	try:
		with open(os.path.join(metrics_dir, 'metrics_summary.txt'), 'w') as fp:
			if annotation:
				fp.write(annotation.rstrip() + "\n")
			fp.write(table_str + "\n")
	except Exception:
		pass
	return summaries

def _generate_plots(results_dir, summaries, metrics, unique_window, stride=1):
	"""Generate full set of plots for reporting."""
	import matplotlib.pyplot as plt
	from scipy import signal

	plots_dir = _ensure_dir(os.path.join(results_dir, 'plots'))
	replot_root = _ensure_dir(os.path.join(results_dir, 'replot'))
	print(f"> replot root: {replot_root}")

	settings_path = os.path.join(results_dir, 'metrics', 'eval_settings.json')
	eval_meta = {}
	if os.path.exists(settings_path):
		try:
			with open(settings_path, 'r') as fp:
				eval_meta = json.load(fp)
		except Exception as exc:
			print(f"> Warning: failed to load eval settings ({exc})")
	min_hz = eval_meta.get('min_hz')
	max_hz = eval_meta.get('max_hz')
	use_track = eval_meta.get('use_track')

	methods = list(summaries.keys())
	if methods:
		if unique_window:
			ordered = list(PRIMARY_METRICS)
		else:
			ordered = metrics
		for met in ordered:
			xs = np.arange(len(methods))
			ys = [summaries[m]['values'].get(met, np.nan) for m in methods]
			plt.figure(figsize=(8, 4))
			plt.bar(xs, ys)
			plt.xticks(xs, methods, rotation=30, ha='right')
			plt.title(met)
			plt.tight_layout()
			plt.savefig(os.path.join(plots_dir, f'{met}.png'))
			plt.close()

	aggregate_payload = []

	for method, summary in summaries.items():
		sample = summary.get('sample', {})
		record = summary.get('record')
		est = sample.get('est')
		gt = sample.get('gt')
		if est is None or gt is None or record is None:
			continue
		try:
			est = np.atleast_1d(np.squeeze(est)).astype(float)
			gt = np.atleast_1d(np.squeeze(gt)).astype(float)
		except Exception:
			continue
		n = min(est.size, gt.size)
		if n < 2:
			continue
		est = est[:n]
		gt = gt[:n]
		valid_mask = ~(np.isnan(est) | np.isnan(gt))
		est_valid = est[valid_mask]
		gt_valid = gt[valid_mask]
		n_valid = est_valid.size
		if n_valid < 2:
			continue

		base = _base_of_method(method)
		base_dir = _ensure_dir(os.path.join(replot_root, base))
		method_dir = os.path.join(base_dir, method)
		if os.path.isdir(method_dir):
			shutil.rmtree(method_dir)
		_ensure_dir(method_dir)

		source = summary.get('source', 'unknown')

		# Scatter statistics with guards for degenerate inputs
		if n_valid:
			with np.errstate(invalid='ignore'):
				sx = float(np.nanstd(gt_valid))
				sy = float(np.nanstd(est_valid))
			unique_frac_est = float(np.unique(est_valid).size / n_valid) if n_valid else 0.0
			unique_frac_gt = float(np.unique(gt_valid).size / n_valid) if n_valid else 0.0
		else:
			sx = sy = float('nan')
			unique_frac_est = unique_frac_gt = 0.0
		degenerate_corr = (
			n_valid < 2
			or not np.isfinite(sx) or sx < 1e-12
			or not np.isfinite(sy) or sy < 1e-12
			or unique_frac_est < 0.05
			or unique_frac_gt < 0.05
		)
		if degenerate_corr:
			r = slope = intercept = r2 = float('nan')
		else:
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', category=RuntimeWarning)
				try:
					r = float(np.corrcoef(gt_valid, est_valid)[0, 1])
				except Exception:
					r = float('nan')
				try:
					slope, intercept = np.polyfit(gt_valid, est_valid, 1)
				except Exception:
					slope = intercept = float('nan')
			r2 = r * r if np.isfinite(r) else float('nan')
			if not np.isfinite(r2):
				r2 = float('nan')
		scatter_info = {
			'pearson_r': _json_float(r),
			'slope': _json_float(slope),
			'intercept': _json_float(intercept),
			'r2': _json_float(r2),
			'n': int(n_valid)
		}

		# Time-domain stats
		diff = est_valid - gt_valid
		rpm_rmse = np.sqrt(np.nanmean(diff ** 2)) if n_valid > 0 else float('nan')
		abs_err = np.abs(diff)
		p50_err = np.nanmedian(abs_err) if n_valid > 0 else float('nan')
		effective_stride = record.get('stride', stride)
		if effective_stride is None:
			effective_stride = stride
		fs_rpm = (1.0 / effective_stride) if effective_stride and effective_stride > 0 else None
		gt_mean = np.nanmean(gt_valid) if n_valid > 0 else float('nan')
		est_mean = np.nanmean(est_valid) if n_valid > 0 else float('nan')
		gt_std = np.nanstd(gt_valid) if n_valid > 0 else float('nan')
		est_std = np.nanstd(est_valid) if n_valid > 0 else float('nan')
		rpm_info = {
			'len': int(n_valid),
			'fs_rpm': _json_float(fs_rpm),
			'gt_mean': _json_float(gt_mean),
			'est_mean': _json_float(est_mean),
			'gt_std': _json_float(gt_std),
			'est_std': _json_float(est_std),
			'rmse_series': _json_float(rpm_rmse),
			'p50_abs_err': _json_float(p50_err)
		}

		# Welch spectra
		welch_info = None
		if effective_stride and effective_stride > 0 and n_valid >= 4:
			fs = 1.0 / effective_stride
			nperseg = min(n_valid, 256)
			try:
				f_est, psd_est = signal.welch(est_valid, fs=fs, nperseg=nperseg)
				f_gt, psd_gt = signal.welch(gt_valid, fs=fs, nperseg=nperseg)
				peak_hz_est = f_est[int(np.argmax(psd_est))] if f_est.size else float('nan')
				peak_hz_gt = f_gt[int(np.argmax(psd_gt))] if f_gt.size else float('nan')
				if min_hz is not None and max_hz is not None:
					bw_mask_est = (f_est >= min_hz) & (f_est <= max_hz)
					bw_mask_gt = (f_gt >= min_hz) & (f_gt <= max_hz)
				else:
					bw_mask_est = slice(None)
					bw_mask_gt = slice(None)
				f_est_band = f_est[bw_mask_est] if isinstance(bw_mask_est, np.ndarray) else f_est
				psd_est_band = psd_est[bw_mask_est] if isinstance(bw_mask_est, np.ndarray) else psd_est
				f_gt_band = f_gt[bw_mask_gt] if isinstance(bw_mask_gt, np.ndarray) else f_gt
				psd_gt_band = psd_gt[bw_mask_gt] if isinstance(bw_mask_gt, np.ndarray) else psd_gt
				bandpower_est = np.trapz(psd_est_band, f_est_band) if f_est_band.size else float('nan')
				bandpower_gt = np.trapz(psd_gt_band, f_gt_band) if f_gt_band.size else float('nan')
				welch_info = {
					'fs': _json_float(fs),
					'nperseg': int(nperseg),
					'peak_hz_est': _json_float(peak_hz_est),
					'peak_hz_gt': _json_float(peak_hz_gt),
					'bandpower_est': _json_float(bandpower_est),
					'bandpower_gt': _json_float(bandpower_gt)
				}
				# Welch plot
				plt.figure(figsize=(8, 4))
				plt.semilogy(f_gt, psd_gt + 1e-12, label='GT')
				plt.semilogy(f_est, psd_est + 1e-12, label='Estimate')
				plt.xlabel('Frequency (Hz)')
				plt.ylabel('PSD')
				plt.title(f'{method}: Welch PSD')
				plt.legend()
				plt.tight_layout()
				plt.savefig(os.path.join(method_dir, 'welch.png'))
				plt.close()
			except Exception as exc:
				print(f"> Warning: Welch computation failed for {method}: {exc}")
				welch_info = None
		else:
			welch_info = None

		# Scatter plot
		plt.figure(figsize=(5, 5))
		plt.scatter(gt_valid, est_valid, s=6, alpha=0.5)
		min_lim = np.nanmin([gt_valid.min(), est_valid.min()])
		max_lim = np.nanmax([gt_valid.max(), est_valid.max()])
		plt.plot([min_lim, max_lim], [min_lim, max_lim], 'r--', linewidth=1)
		plt.xlabel('GT (RPM)')
		plt.ylabel('EST (RPM)')
		plt.title(f'{method}: GT vs EST')
		plt.tight_layout()
		plt.savefig(os.path.join(method_dir, 'scatter.png'))
		plt.close()

		# Time overlay
		time_step = effective_stride if effective_stride and effective_stride > 0 else 1.0
		times = np.arange(n_valid) * time_step
		plt.figure(figsize=(8, 4))
		plt.plot(times, gt_valid, label='GT', linewidth=1.5)
		plt.plot(times, est_valid, label='Estimate', linewidth=1.2, alpha=0.8)
		plt.xlabel('Time (s)')
		plt.ylabel('RPM')
		plt.title(f'{method}: RPM over time')
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(method_dir, 'time.png'))
		plt.close()

		# Bland-Altman plot
		mean_vals = 0.5 * (gt_valid + est_valid)
		mean_diff = np.nanmean(diff)
		sd_diff = np.nanstd(diff)
		upper = mean_diff + 1.96 * sd_diff
		lower = mean_diff - 1.96 * sd_diff
		plt.figure(figsize=(8, 4))
		plt.scatter(mean_vals, diff, s=6, alpha=0.5)
		if not np.isnan(mean_diff):
			plt.axhline(mean_diff, color='r', linestyle='--', linewidth=1.2, label='Mean diff')
		if not np.isnan(upper):
			plt.axhline(upper, color='k', linestyle=':', linewidth=1.0, label='±1.96 SD')
		if not np.isnan(lower):
			plt.axhline(lower, color='k', linestyle=':', linewidth=1.0)
		plt.xlabel('Mean RPM')
		plt.ylabel('Difference (EST - GT)')
		plt.title(f'{method}: Bland-Altman')
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(method_dir, 'bland_altman.png'))
		plt.close()

		bland_info = {
			'mean_diff': _json_float(mean_diff),
			'sd_diff': _json_float(sd_diff),
			'loa_lower': _json_float(lower),
			'loa_upper': _json_float(upper),
			'n': int(n_valid)
		}

		record_track_stats = record.get('track_stats') or {}
		n_windows_track = record_track_stats.get('n_windows', n_valid) if record_track_stats else n_valid
		try:
			n_windows_track = int(n_windows_track)
		except (TypeError, ValueError):
			n_windows_track = int(n_valid)
		def _safe_int(val, default=0):
			try:
				return int(val)
			except (TypeError, ValueError):
				return int(default)

		track_info = {
			'mean_bpm': _json_float(record_track_stats.get('mean_bpm')),
			'std_bpm': _json_float(record_track_stats.get('std_bpm')),
			'unique_fraction': _json_float(record_track_stats.get('unique_fraction')),
			'saturation_fraction': _json_float(record_track_stats.get('saturation_fraction')),
			'n_windows': n_windows_track,
			'median_hz': _json_float(record_track_stats.get('median_hz')),
			'track_dyn_range_hz_ptp': _json_float(record_track_stats.get('track_dyn_range_hz_ptp')),
			'edge_saturation_fraction': _json_float(record_track_stats.get('edge_saturation_fraction')),
			'constant_track_promoted': bool(record_track_stats.get('constant_track_promoted'))
		}
		track_meta_payload = record_track_stats.get('meta') if isinstance(record_track_stats, dict) else None
		if isinstance(track_meta_payload, dict):
			track_info['meta'] = track_meta_payload
		alignment_info_raw = record.get('alignment') or {}
		alignment_info = {
			'aligned': bool(alignment_info_raw.get('aligned', record.get('aligned', False))),
			'len_gt': _safe_int(alignment_info_raw.get('len_gt', record.get('len_gt', n_valid)), default=n_valid),
			'len_pred': _safe_int(alignment_info_raw.get('len_pred', record.get('len_pred', n_valid)), default=n_valid),
			'len_final': _safe_int(alignment_info_raw.get('len_final', record.get('len_final', n_valid)), default=n_valid),
			'len_valid': _safe_int(alignment_info_raw.get('len_valid', record.get('len_valid', n_valid)), default=n_valid)
		}
		series_stats = {
			'est_mean': _json_float(record.get('est_mean')),
			'est_std': _json_float(record.get('est_std')),
			'gt_mean': _json_float(record.get('gt_mean')),
			'gt_std': _json_float(record.get('gt_std')),
			'slope': _json_float(record.get('scatter_slope')),
			'intercept': _json_float(record.get('scatter_intercept'))
		}

		metrics_json = {m: _json_float(summary['values'].get(m, float('nan'))) for m in metrics}
		var_json = {f'{m}_std': _json_float(summary['variability'].get(f'{m}_std', float('nan'))) for m in metrics}
		json_payload = {
			'method': method,
			'base': base,
			'n_windows': int(n_valid),
			'metrics': metrics_json,
			'metrics_std': var_json,
			'bland_altman': bland_info,
			'scatter': scatter_info,
			'rpm_time': rpm_info,
			'welch': welch_info,
			'eval': {
				'min_hz': _json_float(min_hz),
				'max_hz': _json_float(max_hz),
				'use_track': bool(use_track) if isinstance(use_track, bool) else use_track
			},
			'aligned': alignment_info['aligned'],
			'len_gt': alignment_info['len_gt'],
			'len_pred': alignment_info['len_pred'],
			'len_final': alignment_info['len_final'],
			'len_valid': alignment_info['len_valid'],
			'source': source,
			'degenerate_track': bool(record.get('degenerate_track', False)),
			'degenerate_reason': record.get('degenerate_reason'),
			'track': {
				'degenerate': bool(record.get('degenerate_track', False)),
				'degenerate_reason': record.get('degenerate_reason'),
				'stats': track_info
			},
			'track_stats': track_info,
			'series_stats': series_stats,
			'alignment': alignment_info
		}
		json_path = os.path.join(method_dir, 'summary.json')
		with open(json_path, 'w', encoding='utf-8') as fp:
			json.dump(json_payload, fp, ensure_ascii=False, indent=2)
		print(f"> saved {method}/summary.json")
		aggregate_payload.append({
			**json_payload,
			'plots': {
				'scatter': os.path.join(base, method, 'scatter.png'),
				'time': os.path.join(base, method, 'time.png'),
				'welch': os.path.join(base, method, 'welch.png'),
				'bland_altman': os.path.join(base, method, 'bland_altman.png')
			}
		})

	if aggregate_payload:
		root_summary = {
			'eval': {
				'min_hz': _json_float(min_hz),
				'max_hz': _json_float(max_hz),
				'use_track': bool(use_track) if isinstance(use_track, bool) else use_track
			},
			'methods': aggregate_payload
		}
		with open(os.path.join(replot_root, 'summary.json'), 'w', encoding='utf-8') as fp:
			json.dump(root_summary, fp, ensure_ascii=False, indent=2)
		print(f"> aggregated summary saved to {os.path.join(replot_root, 'summary.json')}")


@timed_step('aggregate_runs')
def aggregate_runs(run_dirs, output_dir, prefer_unique=False):
	"""Aggregate metrics across multiple run directories into a single report."""
	import csv

	os.makedirs(output_dir, exist_ok=True)
	records = []
	for run_dir in run_dirs:
		run_dir = os.path.abspath(run_dir)
		metrics_dir = os.path.join(run_dir, 'metrics') if os.path.isdir(os.path.join(run_dir, 'metrics')) else run_dir
		candidates = []
		if prefer_unique:
			candidates = ['metrics_1w.pkl', 'metrics.pkl']
		else:
			candidates = ['metrics.pkl', 'metrics_1w.pkl']
		metrics_path = None
		unique_window = False
		for cand in candidates:
			cand_path = os.path.join(metrics_dir, cand)
			if os.path.exists(cand_path):
				metrics_path = cand_path
				unique_window = cand == 'metrics_1w.pkl'
				break
		if metrics_path is None:
			print(f"> Skipping {run_dir}: no metrics file found")
			continue
		with open(metrics_path, 'rb') as fp:
			metric_names, method_metrics = pickle.load(fp)
			metric_names, method_metrics = _normalize_metrics_payload(metric_names, method_metrics)
		summaries = _summaries_with_samples(method_metrics, metric_names, unique_window)
		for method, summary in summaries.items():
			row = {
				'run': os.path.basename(os.path.normpath(run_dir)),
				'run_path': run_dir,
				'metrics_file': os.path.basename(metrics_path),
				'unique_window': unique_window,
				'method': method
			}
			for key, val in summary['values'].items():
				row[key] = val
			for key, val in summary['variability'].items():
				row[key] = val
			records.append(row)

	if not records:
		print('> No records aggregated. Skipping report generation.')
		return None

	# Prepare CSV
	fieldnames = ['run', 'run_path', 'method', 'unique_window', 'metrics_file']
	extra_fields = sorted({key for rec in records for key in rec.keys() if key not in fieldnames})
	fieldnames.extend(extra_fields)
	csv_path = os.path.join(output_dir, 'combined_metrics.csv')
	with open(csv_path, 'w', newline='') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for rec in records:
			writer.writerow({k: rec.get(k, '') for k in fieldnames})

	# Pretty summary table
	table_rows = []
	for rec in records:
		row = []
		for key in fieldnames:
			val = rec.get(key, '')
			if isinstance(val, (int, float, np.floating)):
				row.append(_format_scalar(float(val)))
			else:
				row.append(val)
		table_rows.append(row)

	table_str = _render_table(fieldnames, table_rows)
	with open(os.path.join(output_dir, 'combined_metrics.txt'), 'w') as fp:
		fp.write(table_str + "\n")

	print(f"> Aggregated report saved to {csv_path}")
	return csv_path


def _build_methods(cfg_list, global_cfg=None):
	osc_defaults = {}
	preproc_defaults = {}
	if isinstance(global_cfg, dict):
		if 'oscillator' in global_cfg and isinstance(global_cfg.get('oscillator'), dict):
			osc_defaults = copy.deepcopy(global_cfg.get('oscillator', {}))
		if 'osc' in global_cfg and isinstance(global_cfg.get('osc'), dict):
			if osc_defaults:
				osc_defaults = _deep_merge_dict(osc_defaults, global_cfg.get('osc'))
			else:
				osc_defaults = copy.deepcopy(global_cfg.get('osc'))
		if 'preproc' in global_cfg and isinstance(global_cfg.get('preproc'), dict):
			preproc_defaults = copy.deepcopy(global_cfg.get('preproc'))
	methods = []
	for item in cfg_list:
		if isinstance(item, str):
			name = item
			params = {}
		else:
			name = item.get('name')
			if not name:
				raise ValueError("Method entry requires a 'name'")
			params = item
		lname = name.lower()

		if lname in ('of_model', 'of', 'of_farneback'):
			method = OF_Model()
			method.name = 'of_farneback'
			methods.append(method)
		elif lname == 'dof':
			method = DoF()
			method.name = 'dof'
			methods.append(method)
		elif lname in ('profile1d', 'profile1d_all'):
			interps = params.get('interp')
			if interps is None:
				target_interps = PROFILE1D_INTERPS
			elif isinstance(interps, (list, tuple, set)):
				target_interps = tuple(interps)
			else:
				target_interps = (interps,)
			for interp in target_interps:
				if interp not in PROFILE1D_INTERPS:
					raise ValueError(f"Unsupported profile1D interpolation '{interp}'. Supported: {PROFILE1D_INTERPS}")
				method = profile1D(interp)
				method.name = f'profile1d_{interp}'
				methods.append(method)
		elif lname.startswith('profile1d_') and lname.split('_')[-1] in PROFILE1D_INTERPS:
			interp = lname.split('_')[-1]
			method = profile1D(interp)
			method.name = f'profile1d_{interp}'
			methods.append(method)
		elif '__' in lname:
			from motion.method_oscillator_wrapped import create_wrapped_method
			local_params = copy.deepcopy(params)
			if osc_defaults:
				local_params.setdefault('oscillator', {})
				for key, value in osc_defaults.items():
					local_params['oscillator'].setdefault(key, value)
			methods.append(create_wrapped_method(lname, local_params, preproc_defaults))
		elif name == 'OF_Deep':
			model = params.get('model', 'raft_small')
			bs = int(params.get('batch_size', 64))
			methods.append(OF_Deep(model=model, batch_size=bs))
		elif name == 'MTTS_CAN':
			methods.append(MTTS_CAN())
		elif name == 'BigSmall':
			methods.append(BigSmall())
		elif name == 'peak':
			methods.append(peak())
		elif name == 'morph':
			methods.append(morph())
		elif name == 'bss_ssa':
			methods.append(bss_ssa())
		elif name == 'bss_emd':
			methods.append(bss_emd())
		else:
			raise ValueError(f"Unknown method in config: {name}")
	return methods

def _build_datasets(cfg_list):
	ds = []
	for item in cfg_list:
		if isinstance(item, str):
			name = item
			params = {}
		else:
			name = item.get('name')
			if not name:
				raise ValueError('Dataset configuration requires "name" field')
			params = item
		lname = name.lower()
		if lname == 'cohface':
			dset = COHFACE()
		elif lname == 'bp4d':
			dset = BP4D()
		elif lname == 'mahnob':
			dset = MAHNOB()
		else:
			raise ValueError(f"Unknown dataset in config: {name}")
		dset.configure(params)
		ds.append(dset)
	return ds

def _derive_run_dirs(results_root, datasets, methods, run_label=None):
	method_suffix = _method_suffix(methods)
	sanitized_label = _sanitize_run_label(run_label) if run_label else None
	single_dataset = len(datasets) == 1
	run_dirs = []
	for dataset in datasets:
		if sanitized_label:
			if single_dataset:
				dir_name = sanitized_label
			else:
				dir_name = f"{sanitized_label}_{dataset.name.upper()}"
		else:
			dir_name = f"{dataset.name.upper()}_{method_suffix}"
		run_dirs.append(os.path.join(results_root, dir_name))
	return run_dirs


def _resolve_paths(base, paths):
	resolved = []
	for p in paths:
		if os.path.isabs(p):
			resolved.append(p)
		else:
			resolved.append(os.path.abspath(os.path.join(base, p)))
	return resolved


def main(argv=None):
	parser = argparse.ArgumentParser(description="Run resPyre pipelines using config-driven steps.")
	parser.add_argument('-c', '--config', help='JSON config file describing datasets/methods/parameters')
	parser.add_argument('-s', '--step', choices=['estimate','evaluate','metrics','report','all'], nargs='+', help='Pipeline steps to execute')
	parser.add_argument('-d', '--results', help='Override results root directory')
	parser.add_argument('-a', '--action', type=int, help='Legacy action flag (0=estimate,1=evaluate,2=metrics,3=report)')
	parser.add_argument('--win', help='Override evaluation window (seconds or "video")')
	parser.add_argument('--stride', type=float, help='Override evaluation stride (seconds)')
	parser.add_argument('--runs', nargs='+', help='Explicit run directories for evaluate/metrics/report steps')
	parser.add_argument('--methods', nargs='+', help='Override the configured methods with an explicit list')
	parser.add_argument('--prefer-unique', action='store_true', help='Prefer metrics_1w.pkl during aggregation')
	parser.add_argument('--num_shards', type=int, default=1, help='Number of shards for parallel estimate execution')
	parser.add_argument('--shard_index', type=int, default=0, help='Shard index (0-based) selecting which methods to estimate')
	parser.add_argument('--auto_discover_methods', nargs='?', const=True, type=_parse_bool_flag, default=False, help='Discover available methods from results directory during evaluate/metrics steps (accepts true/false)')
	parser.add_argument('--override', action='append', help='Override config values using dotted paths (e.g., gating.debug.disable_gating=true)')
	parser.add_argument('--override-from', action='append', help='Load overrides from JSON files containing a \"params\" object')
	parser.add_argument('--allow-missing-methods', dest='allow_missing_methods', action='store_true', help='Allow evaluation/metrics to proceed when some configured methods are missing')
	parser.add_argument('--no-allow-missing-methods', dest='allow_missing_methods', action='store_false', help='Fail if configured methods are missing in results')
	parser.add_argument('--profile-steps', dest='profile_steps', action='store_true', help='Collect and display timing summaries for timed pipeline steps')
	parser.add_argument('--no-profile-steps', dest='profile_steps', action='store_false', help='Disable step timing summaries')
	parser.set_defaults(allow_missing_methods=True, profile_steps=True)
	args = parser.parse_args(argv)

	global _PROFILE_STEPS
	_PROFILE_STEPS = bool(args.profile_steps)

	num_shards = int(args.num_shards if args.num_shards is not None else 1)
	if num_shards < 1:
		parser.error("--num_shards must be >= 1")
	shard_index = int(args.shard_index if args.shard_index is not None else 0)
	if shard_index < 0 or shard_index >= num_shards:
		parser.error("--shard_index must satisfy 0 <= shard_index < num_shards")
	allow_missing_methods = bool(args.allow_missing_methods)

	cfg = load_config(args.config)
	override_items = []
	if args.override_from:
		for override_path in args.override_from:
			try:
				with open(override_path, 'r', encoding='utf-8') as fp:
					payload = json.load(fp)
			except Exception as exc:
				print(f"> Warning: failed to load override file '{override_path}': {exc}")
				continue
			params_block = payload.get('params', payload) if isinstance(payload, dict) else None
			if not isinstance(params_block, dict):
				print(f"> Warning: override file '{override_path}' does not contain a 'params' map; skipped")
				continue
			for key, value in params_block.items():
				if isinstance(value, bool):
					value_str = 'true' if value else 'false'
				else:
					value_str = str(value)
				override_items.append(f"{key}={value_str}")
	if args.override:
		override_items.extend(args.override)
	if override_items:
		_apply_overrides(cfg, override_items)
	if 'profile' in cfg and not isinstance(cfg.get('profile'), dict):
		profile_override = cfg.pop('profile')
		cfg.setdefault('gating', {})['profile'] = profile_override
	gating_cfg = _resolve_gating_config(cfg)
	cfg['gating'] = gating_cfg
	run_label = cfg.get('name') if cfg else None

	# Determine steps
	if args.step:
		steps = ['report' if step == 'all' else step for step in args.step]
	elif args.action is not None:
		legacy_map = {0: ['estimate'], 1: ['evaluate'], 2: ['metrics'], 3: ['report']}
		steps = legacy_map.get(args.action, ['estimate'])
	elif cfg and cfg.get('steps'):
		steps = cfg['steps']
	else:
		steps = ['estimate']

	steps = [step.lower() for step in steps]
	if 'all' in steps:
		steps = ['estimate','evaluate','metrics']

	results_root = os.path.abspath(args.results or cfg['results_dir'])
	os.makedirs(results_root, exist_ok=True)

	if args.methods:
		cfg['methods'] = list(args.methods)
	methods_all = _build_methods(cfg.get('methods', []), cfg)
	method_order = [m.name for m in methods_all]
	datasets = _build_datasets(cfg.get('datasets', []))

	# Runtime device from config (optional)
	if cfg.get('runtime', {}).get('device'):
		os.environ.setdefault('DEVICE', cfg['runtime']['device'])

	eval_cfg = cfg.get('eval', {})
	report_cfg = cfg.get('report', {})
	gating_common = gating_cfg.get('common', {})

	metrics = list(PRIMARY_METRICS)
	min_hz = eval_cfg.get('min_hz', 0.08)
	max_hz = eval_cfg.get('max_hz', 0.5)
	use_track = eval_cfg.get('use_track', False)
	if gating_common.get('use_track') is not None:
		use_track = bool(gating_common.get('use_track'))
	track_std_min_bpm = eval_cfg.get('track_std_min_bpm', 0.3)
	track_unique_min = eval_cfg.get('track_unique_min', 0.05)
	track_saturation_max = eval_cfg.get('track_saturation_max', 0.15)
	saturation_margin_hz = eval_cfg.get('saturation_margin_hz', gating_common.get('saturation_margin_hz', 0.0))
	saturation_persist_sec = eval_cfg.get('saturation_persist_sec', gating_common.get('saturation_persist_sec', 0.0))
	constant_ptp_max_hz = eval_cfg.get('constant_ptp_max_hz', gating_common.get('constant_ptp_max_hz', 0.0))
	win_override = args.win or eval_cfg.get('win_size', 'video')
	if isinstance(win_override, str) and win_override.lower() == 'video':
		win_size = 'video'
	else:
		try:
			win_size = float(win_override)
		except Exception:
			win_size = 'video'
	stride = args.stride if args.stride is not None else eval_cfg.get('stride', 1)
	visualize = True

	# Determine run directories for steps other than estimate
	derived_run_dirs = _derive_run_dirs(results_root, datasets, methods_all, run_label=run_label)
	if args.runs:
		run_dirs = _resolve_paths(results_root, args.runs)
	else:
		run_dirs = derived_run_dirs

	shard_methods = [m for idx, m in enumerate(methods_all) if idx % num_shards == shard_index]
	if num_shards > 1 or shard_index != 0:
		method_names = ', '.join(m.name for m in shard_methods) if shard_methods else '(none)'
		print(f"> Shard {shard_index}/{num_shards}: {len(shard_methods)} methods -> {method_names}")

	if 'estimate' in steps:
		_profiler_stage_start()
		print(f"Executing estimate step -> output root: {results_root}")
		if not shard_methods:
			print("> No methods assigned to this shard; estimate step skipped.")
		else:
			extract_respiration(
				datasets,
				shard_methods,
				results_root,
				run_label=run_label,
				manifest_methods=methods_all,
				method_order=method_order
			)
		_profiler_stage_end('estimate')

	expected_method_names = method_order
	discovered_methods_map = {}

	if 'evaluate' in steps:
		_profiler_stage_start()
		print("Executing evaluate step")
		for run_dir in run_dirs:
			if not os.path.isdir(run_dir):
				print(f"> Warning: run directory {run_dir} not found, skipping evaluation")
				continue
			discovered = None
			if args.auto_discover_methods:
				discovered = _discover_methods_from_aux(run_dir)
				discovered_methods_map[run_dir] = discovered
				if discovered:
					print(f"> Auto-discovered methods ({len(discovered)}): {', '.join(discovered)}")
				else:
					print("> Auto-discovered methods: none found in aux/")
				if expected_method_names:
					missing = sorted({name for name in expected_method_names if name not in (discovered or [])})
					if missing:
						message = f"> Missing methods for {run_dir}: {', '.join(missing)}"
						if allow_missing_methods:
							print(message)
						else:
							raise RuntimeError(message)
			print(f"> Evaluating run: {run_dir}")
			evaluate(
				run_dir,
				metrics,
				win_size=win_size,
				stride=stride,
				visualize=visualize,
				min_hz=min_hz,
				max_hz=max_hz,
				use_track=use_track,
				track_std_min_bpm=track_std_min_bpm,
				track_unique_min=track_unique_min,
				track_saturation_max=track_saturation_max,
				saturation_margin_hz=saturation_margin_hz,
				saturation_persist_sec=saturation_persist_sec,
				constant_ptp_max_hz=constant_ptp_max_hz,
				gating=gating_cfg
			)
		_profiler_stage_end('evaluate')

	if 'metrics' in steps:
		_profiler_stage_start()
		print("Executing metrics display step")
		for run_dir in run_dirs:
			if args.auto_discover_methods and run_dir not in discovered_methods_map:
				discovered_methods_map[run_dir] = _discover_methods_from_aux(run_dir)
			discovered = discovered_methods_map.get(run_dir)
			if args.auto_discover_methods:
				if discovered:
					print(f"> Auto-discovered methods ({len(discovered)}): {', '.join(discovered)}")
				else:
					print("> Auto-discovered methods: none found in aux/")
				if expected_method_names:
					missing = sorted({name for name in expected_method_names if name not in (discovered or [])})
					if missing:
						message = f"> Missing methods for {run_dir}: {', '.join(missing)}"
						if allow_missing_methods:
							print(message)
						else:
							raise RuntimeError(message)
			metrics_subdir = os.path.join(run_dir, 'metrics')
			if os.path.isdir(metrics_subdir):
				metrics_dir = metrics_subdir
			else:
				metrics_dir = run_dir
			metrics_filename = 'metrics_1w.pkl' if win_size == 'video' else 'metrics.pkl'
			metrics_path = os.path.join(metrics_dir, metrics_filename)
			if not os.path.exists(metrics_path):
				print(f"> Metrics file missing for {run_dir}; running evaluate first.")
				evaluate(
					run_dir,
					metrics,
					win_size=win_size,
					stride=stride,
					visualize=True,
					min_hz=min_hz,
					max_hz=max_hz,
					use_track=use_track,
					track_std_min_bpm=track_std_min_bpm,
					track_unique_min=track_unique_min,
					track_saturation_max=track_saturation_max,
					saturation_margin_hz=saturation_margin_hz,
					saturation_persist_sec=saturation_persist_sec,
					constant_ptp_max_hz=constant_ptp_max_hz,
					gating=gating_cfg
				)
			unique_window = os.path.exists(os.path.join(metrics_dir, 'metrics_1w.pkl'))
			print(f"\n== Metrics for {run_dir} ==")
			if not os.path.exists(metrics_path):
				print(f"> Metrics still unavailable for {run_dir}. Please rerun evaluate step.")
				continue
			print_metrics(run_dir, unique_window=unique_window)
		_profiler_stage_end('metrics')

	if 'report' in steps:
		_profiler_stage_start()
		report_runs = report_cfg.get('runs', []) if cfg else []
		if args.runs:
			report_runs = args.runs
		elif report_runs:
			report_runs = report_runs
		else:
			report_runs = [os.path.relpath(rd, results_root) for rd in derived_run_dirs]
		report_runs_abs = _resolve_paths(results_root, report_runs)
		output_rel = report_cfg.get('output', 'reports/combined') if cfg else 'reports/combined'
		output_dir = output_rel
		if not os.path.isabs(output_dir):
			output_dir = os.path.abspath(os.path.join(results_root, output_dir))
		prefer_unique = args.prefer_unique or report_cfg.get('unique_window', False)
		print(f"Executing report aggregation -> {output_dir}")
		aggregate_runs(report_runs_abs, output_dir, prefer_unique=prefer_unique)
		_profiler_stage_end('report')

	print('\nDone.')


if __name__ == "__main__":
	main()
def _runtime_config():
	def _get_env(name, default=None):
		return os.environ.get(name, default)

	cfg = {}
	cfg['device'] = _get_env('DEVICE', 'cuda:0')
	return cfg


RUNTIME_CONFIG = _runtime_config()


def _use_cuda():
	device = RUNTIME_CONFIG.get('device', 'cuda:0')
	return isinstance(device, str) and device.lower().startswith('cuda')


def _torch_device():
	device = RUNTIME_CONFIG.get('device', 'cuda:0')
	return device if _use_cuda() else 'cpu'
