import os
import argparse
import json
import hashlib
import shutil
import copy
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
from config_loader import load_config

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
				video_path = os.path.join(trial_path, 'data.avi')

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

def timed_step(label=None):
	def decorator(func):
		step_name = label or func.__name__
		@wraps(func)
		def wrapper(*args, **kwargs):
			start = time.time()
			result = func(*args, **kwargs)
			duration = time.time() - start
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
	track_saturation_max=0.15
):
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

		# Extract ground truth RPM using Welch with (win_size/1.5)
		gt_rpm = utils.sig_to_RPM(gt_win, fs_gt, int(ws/1.5), min_hz, max_hz)

		# Extract estimation data
		fps = data['fps']

		gt_rpm = np.squeeze(np.asarray(gt_rpm, dtype=np.float64))
		if gt_rpm.ndim == 0:
			gt_rpm = gt_rpm.reshape(1)
		target_len = gt_rpm.shape[-1]

		for i, est in enumerate(data['estimates']):

			cur_method = est['method']
			if cur_method == 'OF_Deep':
				cur_method += ofdeep_models[i]
			elif cur_method == 'OF_Model':
				cur_method = 'OF_Farneback'

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

			method_storage_name = est['method']
			sanitized_method = method_storage_name.replace(' ', '_')
			sig_rpm_values = None
			t_sig = None
			track_used = False
			degenerate_track = False
			degenerate_reason = None
			track_stats = {}

			if use_track:
				aux_dir = os.path.join(results_dir, 'aux', sanitized_method)
				aux_file = os.path.join(aux_dir, f"{trial_key}.npz")
				if os.path.exists(aux_file):
					try:
						with np.load(aux_file) as aux_data:
							if 'track_hz' in aux_data:
								track_series = np.asarray(aux_data['track_hz'], dtype=np.float64).reshape(-1)
								if track_series.size:
									track_series = np.clip(track_series, min_hz, max_hz)
									track_win, t_sig = utils.sig_windowing(track_series, fps, ws, stride=stride)
									if track_win:
										win_array = np.vstack([np.squeeze(w) for w in track_win])
										sig_rpm_values = np.median(win_array, axis=1) * 60.0
										track_used = True
										finite_mask_rpm = np.isfinite(sig_rpm_values)
										finite_rpm = sig_rpm_values[finite_mask_rpm]
										if finite_rpm.size:
											est_std = float(np.std(finite_rpm, dtype=np.float64))
											unique_vals = np.unique(finite_rpm)
											nuniq_frac = float(unique_vals.size / max(1, finite_rpm.size))
											mean_rpm = float(np.mean(finite_rpm))
										else:
											est_std = float('nan')
											nuniq_frac = 0.0
											mean_rpm = float('nan')
										finite_track = track_series[np.isfinite(track_series)]
										if finite_track.size:
											sat_mask = (finite_track <= (min_hz + 1e-6)) | (finite_track >= (max_hz - 1e-6))
											sat_frac = float(np.mean(sat_mask))
										else:
											sat_frac = 1.0
										track_stats = {
											'mean_bpm': mean_rpm if np.isfinite(mean_rpm) else float('nan'),
											'std_bpm': est_std if np.isfinite(est_std) else float('nan'),
											'unique_fraction': nuniq_frac,
											'saturation_fraction': sat_frac,
											'n_windows': int(sig_rpm_values.size)
										}
										trigger_reason = None
										if not np.isfinite(est_std) or est_std < float(track_std_min_bpm):
											trigger_reason = 'low_std'
										elif nuniq_frac < float(track_unique_min):
											trigger_reason = 'low_unique'
										elif sat_frac > float(track_saturation_max):
											trigger_reason = 'high_saturation'
										if trigger_reason:
											track_used = False
											sig_rpm_values = None
											degenerate_track = True
											degenerate_reason = trigger_reason
							if (not track_used) and ('rr_bpm' in aux_data):
								rr_vals = np.asarray(aux_data['rr_bpm'], dtype=np.float64).reshape(-1)
								if rr_vals.size:
									sig_rpm_values = np.full(target_len, rr_vals[-1], dtype=np.float64)
									t_sig = t_gt
									track_used = True
					except Exception as exc:
						tqdm.write(f"> Warning: failed to read aux track for {method_storage_name}: {exc}")

			if track_used and sig_rpm_values is not None and not track_stats:
				finite_mask_rpm = np.isfinite(sig_rpm_values)
				finite_rpm = sig_rpm_values[finite_mask_rpm]
				if finite_rpm.size:
					est_std = float(np.std(finite_rpm, dtype=np.float64))
					nuniq_frac = float(np.unique(finite_rpm).size / finite_rpm.size)
					mean_rpm = float(np.mean(finite_rpm))
				else:
					est_std = float('nan')
					nuniq_frac = 0.0
					mean_rpm = float('nan')
				track_stats = {
					'mean_bpm': mean_rpm if np.isfinite(mean_rpm) else float('nan'),
					'std_bpm': est_std if np.isfinite(est_std) else float('nan'),
					'unique_fraction': nuniq_frac,
					'saturation_fraction': float('nan'),
					'n_windows': int(sig_rpm_values.size)
				}
				if not np.isfinite(est_std) or est_std < float(track_std_min_bpm):
					degenerate_track = True
					degenerate_reason = 'low_std'
					track_used = False
					sig_rpm_values = None
				elif nuniq_frac < float(track_unique_min):
					degenerate_track = True
					degenerate_reason = 'low_unique'
					track_used = False
					sig_rpm_values = None

			if not track_used:
				tqdm.write(f"> Warning: track unavailable for {method_storage_name}::{trial_key}, falling back to spectral RPM")
				sig_rpm_list = []
				for d in range(filt_sig.shape[0]):
					# Apply windowing to the estimation
					sig_win, t_sig = utils.sig_windowing(filt_sig[d,:], fps, ws, stride=stride)
					# Extract estimated RPM
					rpm_segment = np.asarray(utils.sig_to_RPM(sig_win, fps, int(ws/1.5), min_hz, max_hz), dtype=np.float64).reshape(-1)
					sig_rpm_list.append(rpm_segment)
				if sig_rpm_list:
					sig_rpm_values = np.mean(np.vstack(sig_rpm_list), axis=0)
				else:
					sig_rpm_values = np.zeros(target_len, dtype=np.float64)

			sig_rpm_values = np.asarray(sig_rpm_values, dtype=np.float64).reshape(-1)

			if sig_rpm_values.size and np.any(np.isfinite(sig_rpm_values)):
				est_mean_val = float(np.nanmean(sig_rpm_values))
				est_std_val = float(np.nanstd(sig_rpm_values))
			else:
				est_mean_val = float('nan')
				est_std_val = float('nan')
			if gt_rpm.size and np.any(np.isfinite(gt_rpm)):
				gt_mean_val = float(np.nanmean(gt_rpm))
				gt_std_val = float(np.nanstd(gt_rpm))
			else:
				gt_mean_val = float('nan')
				gt_std_val = float('nan')
			finite_mask_pair = np.isfinite(sig_rpm_values) & np.isfinite(gt_rpm)
			if finite_mask_pair.sum() >= 2:
				try:
					scatter_slope, scatter_intercept = np.polyfit(gt_rpm[finite_mask_pair], sig_rpm_values[finite_mask_pair], 1)
				except Exception:
					scatter_slope = scatter_intercept = float('nan')
			else:
				scatter_slope = scatter_intercept = float('nan')

			e = errors.getErrors(sig_rpm_values, gt_rpm, t_sig, t_gt, metrics)
			raw_metrics = e[:-1]
			pair = e[-1]
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
			record = {
				'metrics': metric_values,
				'pair': pair,
				'source': 'track' if track_used else 'spectral',
				'stride': stride,
				'times_est': np.asarray(t_sig).astype(float) if t_sig is not None else None,
				'times_gt': np.asarray(t_gt).astype(float) if t_gt is not None else None,
				'degenerate_track': bool(degenerate_track),
				'degenerate_reason': degenerate_reason,
				'track_stats': track_stats,
				'est_mean': est_mean_val,
				'est_std': est_std_val,
				'gt_mean': gt_mean_val,
				'gt_std': gt_std_val,
				'scatter_slope': float(scatter_slope) if np.isfinite(scatter_slope) else float('nan'),
				'scatter_intercept': float(scatter_intercept) if np.isfinite(scatter_intercept) else float('nan')
			}
			method_metrics.setdefault(cur_method, []).append(record)

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
				"use_track": use_track
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
		headers = ['Method', 'RMSE', 'MAE', 'MAPE', 'CORR', 'PCC', 'CCC']
	else:
		headers = ['Method'] + metrics
	table_rows = []

	for method, summary in summaries.items():
		values = summary['values']
		if unique_window:
			row_vals = [method]
			for key in ['RMSE','MAE','MAPE','CORR','PCC','CCC']:
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


@timed_step('extract_respiration')
def extract_respiration(datasets, methods, results_dir, run_label=None):
	os.makedirs(results_dir, exist_ok=True)
	method_suffix = _method_suffix(methods)
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
		try:
			with open(manifest_path, 'w') as fp:
				json.dump([m.name for m in methods], fp, indent=2)
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

			if os.path.exists(outfilename):
				tqdm.write("> File %s already exists! Skipping..." % outfilename)
				continue

			_, d['fps'] = utils.get_vid_stats(d['video_path'])
			d['trial_key'] = trial_key

			results = {'video_path': d['video_path'],
			           'fps': d['fps'],
			           'gt' : d['gt'],
			           'fs_gt': float(dataset.fs_gt) if dataset.fs_gt is not None else None,
			           'estimates': [] }

			if 'trial' in d.keys(): 
				tqdm.write("> Processing video %s/%s\n> fps: %d" % (d['subject'], d['trial'], d['fps']))
			else:
				tqdm.write("> Processing video %s\n> fps: %d" % (d['subject'], d['fps']))

			# Apply every method to each video
			for m in methods:
				# Apply individual method
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
				results['estimates'].append({'method': m.name, 'estimate': estimate})

			# release some memory between videos
			d.pop('aux_save_dir', None)
			d.pop('trial_key', None)
			d['chest_rois'] = []
			d['face_rois'] = []
			d['rppg_obj'] = None

			# Save the results of the applied methods
			with open(outfilename, 'wb') as fp:
				pickle.dump(results, fp)
				tqdm.write('> Results saved!\n')

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
				corr = PearsonCorr(est_concat, gt_concat)
				values['CORR'] = corr
				values['PCC'] = corr
				values['CCC'] = LinCorr(est_concat, gt_concat)
			else:
				for key in ['RMSE', 'MAE', 'MAPE', 'CORR', 'PCC', 'CCC']:
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
		table_headers = ['Method', 'RMSE', 'MAE', 'MAPE', 'CORR', 'PCC', 'CCC']
	else:
		# 원본 구현과 동일: 중앙값±표준편차
		table_headers = ['Method'] + [f"{m} (median±std)" for m in metrics]
	table_rows = []
	for method, summary in summaries.items():
		values = summary['values']
		if unique_window:
			row = [method]
			for metric in ['RMSE','MAE','MAPE','CORR','PCC','CCC']:
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
			ordered = ['RMSE', 'MAE', 'MAPE', 'CORR', 'PCC', 'CCC']
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

		# Scatter statistics
		try:
			r = float(np.corrcoef(gt_valid, est_valid)[0, 1])
		except Exception:
			r = float('nan')
		try:
			slope, intercept = np.polyfit(gt_valid, est_valid, 1)
		except Exception:
			slope = intercept = float('nan')
		r2 = r * r if not np.isnan(r) else float('nan')
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
		track_info = {
			'mean_bpm': _json_float(record_track_stats.get('mean_bpm')),
			'std_bpm': _json_float(record_track_stats.get('std_bpm')),
			'unique_fraction': _json_float(record_track_stats.get('unique_fraction')),
			'saturation_fraction': _json_float(record_track_stats.get('saturation_fraction')),
			'n_windows': n_windows_track
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
			'source': source,
			'track': {
				'degenerate': bool(record.get('degenerate_track', False)),
				'degenerate_reason': record.get('degenerate_reason'),
				'stats': track_info
			},
			'series_stats': series_stats
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
	if isinstance(global_cfg, dict):
		osc_defaults = copy.deepcopy(global_cfg.get('oscillator', {})) if 'oscillator' in global_cfg else {}
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
			methods.append(create_wrapped_method(lname, local_params))
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
	parser.add_argument('--prefer-unique', action='store_true', help='Prefer metrics_1w.pkl during aggregation')
	args = parser.parse_args(argv)

	cfg = load_config(args.config)
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

	methods = _build_methods(cfg.get('methods', []), cfg)
	datasets = _build_datasets(cfg.get('datasets', []))

	# Runtime device from config (optional)
	if cfg.get('runtime', {}).get('device'):
		os.environ.setdefault('DEVICE', cfg['runtime']['device'])

	eval_cfg = cfg.get('eval', {})
	report_cfg = cfg.get('report', {})

	metrics = ['RMSE', 'MAE', 'MAPE', 'CORR', 'PCC', 'CCC']
	min_hz = eval_cfg.get('min_hz', 0.08)
	max_hz = eval_cfg.get('max_hz', 0.5)
	use_track = eval_cfg.get('use_track', False)
	track_std_min_bpm = eval_cfg.get('track_std_min_bpm', 0.3)
	track_unique_min = eval_cfg.get('track_unique_min', 0.05)
	track_saturation_max = eval_cfg.get('track_saturation_max', 0.15)
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
	derived_run_dirs = _derive_run_dirs(results_root, datasets, methods, run_label=run_label)
	if args.runs:
		run_dirs = _resolve_paths(results_root, args.runs)
	else:
		run_dirs = derived_run_dirs

	if 'estimate' in steps:
		print(f"Executing estimate step -> output root: {results_root}")
		extract_respiration(datasets, methods, results_root, run_label=run_label)

	if 'evaluate' in steps:
		print("Executing evaluate step")
		for run_dir in run_dirs:
			if not os.path.isdir(run_dir):
				print(f"> Warning: run directory {run_dir} not found, skipping evaluation")
				continue
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
				track_saturation_max=track_saturation_max
			)

	if 'metrics' in steps:
		print("Executing metrics display step")
		for run_dir in run_dirs:
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
					track_saturation_max=track_saturation_max
				)
			unique_window = os.path.exists(os.path.join(metrics_dir, 'metrics_1w.pkl'))
			print(f"\n== Metrics for {run_dir} ==")
			if not os.path.exists(metrics_path):
				print(f"> Metrics still unavailable for {run_dir}. Please rerun evaluate step.")
				continue
			print_metrics(run_dir, unique_window=unique_window)

	if 'report' in steps:
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
