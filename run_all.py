import os
import argparse
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
		self.fs_gt = 32
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

		#Load GT
		import os
		f = h5py.File(os.path.join(trial_path, 'data.hdf5'), 'r')
		gt = np.array(f['respiration'])
		gt = gt[np.arange(0, len(gt), 8)] # ???
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
	if method_name.startswith('OF_Model'):
		return 'OF'
	if method_name.startswith('profile1D '):
		return 'profile1D'
	return method_name.replace(' ', '_').replace('-', '_')


def _method_suffix(methods):
	tokens = []
	seen = set()
	for m in methods:
		token = _method_token(m.name)
		if token in seen:
			continue
		seen.add(token)
		tokens.append(token)
	return '_'.join(tokens)


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
def evaluate(results_dir, metrics, win_size=30, stride=1, visualize=True):
	print('\n> Loading extracted data from ' + results_dir + '...')

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

		# Filter ground truth
		filt_gt = utils.filter_RW(gt, fs_gt)

		if win_size == 'video':
			ws = filt_gt.shape[1] / fs_gt
		else:
			ws = win_size

		tqdm.write("> Length: %.2f sec" % (len(gt) / int(fs_gt)))

		# Apply windowing to ground truth
		gt_win, t_gt = utils.sig_windowing(filt_gt, fs_gt, ws, stride=stride)

		# Extract ground truth RPM using Welch with (win_size/1.5)
		gt_rpm = utils.sig_to_RPM(gt_win, fs_gt, int(ws/1.5), 0.2, 0.5)

		# Extract estimation data
		fps = data['fps']

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
				filt_sig.append(utils.filter_RW(sig[d,:], fps))

			filt_sig = np.vstack(filt_sig)

			if cur_method in ['bss_emd', 'bss_ssa']:
				filt_sig = utils.select_component(filt_sig, fps, int(ws/1.5), 0.2, 0.5)

			sig_rpm = []
			for d in range(filt_sig.shape[0]):
				# Apply windowing to the estimation
				sig_win, t_sig = utils.sig_windowing(filt_sig[d,:], fps, ws, stride=stride)
				# Extract estimated RPM
				sig_rpm.append(utils.sig_to_RPM(sig_win, fps, int(ws/1.5), 0.2, 0.5))

			sig_rpm = np.mean(sig_rpm, axis=0)

			e = errors.getErrors(sig_rpm, gt_rpm, t_sig, t_gt, metrics)

			method_metrics.setdefault(cur_method, []).append((e))

	# Choose metrics output directory (new structure prefers results_dir/metrics)
	metrics_dir = os.path.join(results_dir, 'metrics') if os.path.isdir(os.path.join(results_dir, 'metrics')) else results_dir
	if win_size == 'video':
		fn = 'metrics_1w.pkl'
	else:
		fn = 'metrics.pkl'
	# Save the results of the applied methods
	with open(os.path.join(metrics_dir, fn), 'wb') as fp:
		pickle.dump([metrics, method_metrics] , fp)
		print('> Metrics saved!\n')

	# Save human-readable table and plots (best-effort)
	try:
		summaries = _save_metrics_table(metrics_dir, method_metrics, metrics, win_size == 'video')
		if visualize:
			_generate_plots(results_dir, summaries, metrics, win_size == 'video', stride=stride)
	except Exception as e:
		print(f"> Plot/log generation skipped due to error: {e}")
	return method_metrics


def print_metrics(results_dir, unique_window=False):
	from prettytable import PrettyTable
	from errors import concordance_correlation_coefficient

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

	t = PrettyTable(['Method'] + metrics)

	for method, metrics_value in method_metrics.items():

		#import code; code.interact(local=locals())

		if unique_window:
			from errors import RMSEerror, MAEerror, MAPEerror, PearsonCorr, LinCorr
			bpmsEst = np.stack([np.squeeze(metric[-1][0]) for metric in metrics_value])[np.newaxis,:]
			bpmsGT = np.stack([np.squeeze(metric[-1][1]) for metric in metrics_value])		
			rmse = RMSEerror(bpmsEst, bpmsGT)
			mae = MAEerror(bpmsEst, bpmsGT)
			mape = MAPEerror(bpmsEst, bpmsGT)
			corr = PearsonCorr(bpmsEst, bpmsGT)
			pcc = corr
			ccc = LinCorr(bpmsEst, bpmsGT)		
			vals = [rmse, mae, mape, corr, pcc, ccc]
		else:
			vals = []
			for i, m in enumerate(metrics):
				avg = np.nanmedian([metric[i] for metric in metrics_value])
				std = np.nanstd([metric[i] for metric in metrics_value])
				vals.append(f"%.3f (%.2f)" % (float(avg), float(std)))

		t.add_row([method] + vals)

	print(t)
	# Save alongside pkl for reporting
	try:
		with open(os.path.join(metrics_dir, 'metrics_summary.txt'), 'w') as fp:
			fp.write(str(t) + "\n")
	except Exception:
		pass


@timed_step('extract_respiration')
def extract_respiration(datasets, methods, results_dir):
	os.makedirs(results_dir, exist_ok=True)
	method_suffix = _method_suffix(methods)

	for dataset in datasets:
		dir_name = f"{dataset.name.upper()}_{method_suffix}"
		dataset_results_dir = _dataset_results_dir(results_dir, dir_name)
		data_dir = os.path.join(dataset_results_dir, 'data')

		dataset.load_dataset()
		# Loop over the dataset
		for d in tqdm(dataset.data, desc="Processing files"):

			if 'trial' in d.keys(): 
				outfilename = os.path.join(data_dir, dataset.name + '_' + d['subject'] + '_' + d['trial'] + '.pkl')
			else:
				outfilename = os.path.join(data_dir, dataset.name + '_' + d['subject'] + '.pkl')

			if os.path.exists(outfilename):
				tqdm.write("> File %s already exists! Skipping..." % outfilename)
				continue

			_, d['fps'] = utils.get_vid_stats(d['video_path'])

			results = {'video_path': d['video_path'],
					   'fps': d['fps'],
					   'gt' : d['gt'],
					   'fs_gt': dataset.fs_gt,
					   'estimates': [] }

			if 'trial' in d.keys(): 
				tqdm.write("> Processing video %s/%s\n> fps: %d" % (d['subject'], d['trial'], d['fps']))
			else:
				tqdm.write("> Processing video %s\n> fps: %d" % (d['subject'], d['fps']))

		# Apply every method to each video
		for m in methods:
			tqdm.write("> Applying method %s ..." % m.name)
			skip_method = False
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
		sample_est = None
		sample_gt = None
		if records:
			try:
				sample_est = np.atleast_1d(np.squeeze(records[0][-1][0])).astype(float)
				sample_gt = np.atleast_1d(np.squeeze(records[0][-1][1])).astype(float)
			except Exception:
				sample_est = sample_gt = None
		values = {}
		variability = {}
		if unique_window:
			est_vals = []
			gt_vals = []
			for record in records:
				est = np.atleast_1d(np.squeeze(record[-1][0])).astype(float)
				gt = np.atleast_1d(np.squeeze(record[-1][1])).astype(float)
				n = min(est.shape[-1] if est.ndim else 1, gt.shape[-1] if gt.ndim else 1)
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
				values = {k: float('nan') for k in ['RMSE','MAE','MAPE','CORR','PCC','CCC']}
		else:
			for idx, metric_name in enumerate(metrics):
				data = [record[idx] for record in records]
				avg = np.nanmedian(data)
				std = np.nanstd(data)
				values[metric_name] = float(avg)
				variability[f'{metric_name}_std'] = float(std)
		summaries[method] = {
			'values': values,
			'variability': variability,
			'sample': {'est': sample_est, 'gt': sample_gt}
		}
	return summaries


def _save_metrics_table(metrics_dir, method_metrics, metrics, unique_window):
	"""Save a pretty metrics summary table to text for reporting."""
	from prettytable import PrettyTable
	summaries = _summaries_with_samples(method_metrics, metrics, unique_window)
	if unique_window:
		table_headers = ['Method', 'RMSE', 'MAE', 'MAPE', 'CORR', 'PCC', 'CCC']
	else:
		table_headers = ['Method'] + [f"{m} (median±std)" for m in metrics]
	t = PrettyTable(table_headers)
	for method, summary in summaries.items():
		values = summary['values']
		if unique_window:
			row = [method] + [values.get(metric, float('nan')) for metric in ['RMSE','MAE','MAPE','CORR','PCC','CCC']]
		else:
			row = [method]
			for metric in metrics:
				median = values.get(metric, float('nan'))
				std = summary['variability'].get(f'{metric}_std', float('nan'))
				row.append(f"{median:.3f} (±{std:.2f})")
			# ensure consistent columns
		t.add_row(row)

	try:
		with open(os.path.join(metrics_dir, 'metrics_summary.txt'), 'w') as fp:
			fp.write(str(t) + "\n")
	except Exception:
		pass
	return summaries

def _generate_plots(results_dir, summaries, metrics, unique_window, stride=1):
	"""Generate full set of plots for reporting."""
	import matplotlib.pyplot as plt
	from scipy import signal

	plots_dir = os.path.join(results_dir, 'plots') if os.path.isdir(os.path.join(results_dir, 'plots')) else results_dir
	os.makedirs(plots_dir, exist_ok=True)

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

	for method, summary in summaries.items():
		sample = summary.get('sample', {})
		est = sample.get('est')
		gt = sample.get('gt')
		if est is None or gt is None:
			continue
		est = np.atleast_1d(est).astype(float)
		gt = np.atleast_1d(gt).astype(float)
		n = min(est.shape[-1], gt.shape[-1]) if est.ndim and gt.ndim else 0
		if n < 2:
			continue
		# Scatter
		plt.figure(figsize=(5, 5))
		plt.scatter(gt[:n], est[:n], s=6, alpha=0.5)
		plt.xlabel('GT (RPM)')
		plt.ylabel('EST (RPM)')
		lims = [np.nanmin([gt[:n].min(), est[:n].min()]), np.nanmax([gt[:n].max(), est[:n].max()])]
		plt.plot(lims, lims, 'r--', linewidth=1)
		plt.title(f'{method}: GT vs EST')
		plt.tight_layout()
		plt.savefig(os.path.join(plots_dir, f'scatter_{method}.png'))
		plt.close()
		# Time overlay
		times = np.arange(n) * stride
		plt.figure(figsize=(8, 4))
		plt.plot(times, gt[:n], label='GT', linewidth=1.5)
		plt.plot(times, est[:n], label='Estimate', linewidth=1.2, alpha=0.8)
		plt.xlabel('Time (s)')
		plt.ylabel('RPM')
		plt.title(f'{method}: RPM over time')
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(plots_dir, f'time_{method}.png'))
		plt.close()
		# Welch PSD
		if stride > 0:
			fs = 1.0 / stride
			f_est, psd_est = signal.welch(est[:n], fs=fs, nperseg=min(n, 256))
			f_gt, psd_gt = signal.welch(gt[:n], fs=fs, nperseg=min(n, 256))
			plt.figure(figsize=(8, 4))
			plt.semilogy(f_gt, psd_gt + 1e-12, label='GT')
			plt.semilogy(f_est, psd_est + 1e-12, label='Estimate')
			plt.xlabel('Frequency (Hz)')
			plt.ylabel('PSD')
			plt.title(f'{method}: Welch PSD')
			plt.legend()
			plt.tight_layout()
			plt.savefig(os.path.join(plots_dir, f'welch_{method}.png'))
			plt.close()
		# Bland-Altman
		diff = est[:n] - gt[:n]
		mean_vals = 0.5 * (est[:n] + gt[:n])
		if np.all(np.isnan(diff)):
			continue
		mean_diff = float(np.nanmean(diff))
		sd_diff = float(np.nanstd(diff))
		upper = mean_diff + 1.96 * sd_diff
		lower = mean_diff - 1.96 * sd_diff
		plt.figure(figsize=(8, 4))
		plt.scatter(mean_vals, diff, s=6, alpha=0.5)
		plt.axhline(mean_diff, color='r', linestyle='--', linewidth=1.2, label='Mean diff')
		plt.axhline(upper, color='k', linestyle=':', linewidth=1.0, label='±1.96 SD')
		plt.axhline(lower, color='k', linestyle=':', linewidth=1.0)
		plt.xlabel('Mean RPM')
		plt.ylabel('Difference (EST - GT)')
		plt.title(f'{method}: Bland-Altman')
		plt.legend()
		plt.tight_layout()
		plt.savefig(os.path.join(plots_dir, f'bland_altman_{method}.png'))
		plt.close()


@timed_step('aggregate_runs')
def aggregate_runs(run_dirs, output_dir, prefer_unique=False):
	"""Aggregate metrics across multiple run directories into a single report."""
	import csv
	from prettytable import PrettyTable

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
	t = PrettyTable(fieldnames)
	for rec in records:
		row = []
		for key in fieldnames:
			val = rec.get(key, '')
			if isinstance(val, (int, float, np.floating)):
				if np.isnan(float(val)):
					row.append('nan')
				else:
					row.append(f"{float(val):.3f}")
			else:
				row.append(val)
		t.add_row(row)
	with open(os.path.join(output_dir, 'combined_metrics.txt'), 'w') as fp:
		fp.write(str(t) + "\n")

	print(f"> Aggregated report saved to {csv_path}")
	return csv_path


def _build_methods(cfg_list):
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
		if name == 'OF_Model':
			methods.append(OF_Model())
		elif name == 'DoF':
			methods.append(DoF())
		elif name == 'profile1D':
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
				methods.append(profile1D(interp))
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

def _derive_run_dirs(results_root, datasets, methods):
	method_suffix = _method_suffix(methods)
	run_dirs = []
	for dataset in datasets:
		run_dirs.append(os.path.join(results_root, f"{dataset.name.upper()}_{method_suffix}"))
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

	methods = _build_methods(cfg.get('methods', []))
	datasets = _build_datasets(cfg.get('datasets', []))

	# Runtime device from config (optional)
	if cfg.get('runtime', {}).get('device'):
		os.environ.setdefault('DEVICE', cfg['runtime']['device'])

	eval_cfg = cfg.get('eval', {})
	report_cfg = cfg.get('report', {})

	metrics = ['RMSE', 'MAE', 'MAPE', 'CORR', 'PCC', 'CCC']
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
	derived_run_dirs = _derive_run_dirs(results_root, datasets, methods)
	if args.runs:
		run_dirs = _resolve_paths(results_root, args.runs)
	else:
		run_dirs = derived_run_dirs

	if 'estimate' in steps:
		print(f"Executing estimate step -> output root: {results_root}")
		extract_respiration(datasets, methods, results_root)

	if 'evaluate' in steps:
		print("Executing evaluate step")
		for run_dir in run_dirs:
			if not os.path.isdir(run_dir):
				print(f"> Warning: run directory {run_dir} not found, skipping evaluation")
				continue
			print(f"> Evaluating run: {run_dir}")
			evaluate(run_dir, metrics, win_size=win_size, stride=stride, visualize=visualize)

	if 'metrics' in steps:
		print("Executing metrics display step")
		for run_dir in run_dirs:
			metrics_dir = os.path.join(run_dir, 'metrics') if os.path.isdir(os.path.join(run_dir, 'metrics')) else run_dir
			unique_window = os.path.exists(os.path.join(metrics_dir, 'metrics_1w.pkl'))
			print(f"\n== Metrics for {run_dir} ==")
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
