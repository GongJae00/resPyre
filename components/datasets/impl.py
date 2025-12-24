
import os
import numpy as np
from core.utils.common import sort_nicely, get_chest_ROI, get_face_ROI
from .base import DatasetBase

class BP4D(DatasetBase):
	def __init__(self):
		super().__init__()
		self.name = 'bp4d'
		self.path = self.resolve('BP4Ddef') + os.sep
		self.fs_gt = 1000
		self.data = [] 

	def load_dataset(self):

		print('\nLoading dataset ' + self.name + '...')
		for sub in sort_nicely(os.listdir(self.path)):
			sub_path = os.path.join(self.path, sub)
			if not os.path.isdir(sub_path):
				continue

			for trial in sort_nicely(os.listdir(sub_path)):
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
			rois, _, _ = get_chest_ROI(video_path, self.name, mp_complexity=mp_complexity, skip_rate=skip_rate)
		elif region == 'face':
			params = self.roi_params.get('face', {})
			rois = get_face_ROI(video_path, **params) if params else get_face_ROI(video_path)
		else:
			rois = []
		return rois

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
		for sub in sort_nicely(os.listdir(self.path)):
			sub_path = os.path.join(self.path, sub)
			if not os.path.isdir(sub_path):
				continue

			for trial in sort_nicely(os.listdir(sub_path)):
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
			rois, _, _ = get_chest_ROI(video_path, self.name, mp_complexity=mp_complexity, skip_rate=skip_rate)
		elif region == 'face':
			params = self.roi_params.get('face', {})
			rois = get_face_ROI(video_path, **params) if params else get_face_ROI(video_path)
		else:
			rois = []
		return rois

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
		for sub in sort_nicely(os.listdir(self.path)):
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
				d['gt'] = self.load_gt(sub_path)
				self.data.append(d)

		print('%d items loaded!' % len(self.data)) 

	def extract_ROI(self, video_path, region='chest'):
		if region == 'chest':
			params = self.roi_params.get('chest', {})
			mp_complexity = params.get('mp_complexity', 1)
			skip_rate = params.get('skip_rate', 10)
			rois, _, _ = get_chest_ROI(video_path, self.name, mp_complexity=mp_complexity, skip_rate=skip_rate)
		elif region == 'face':
			params = self.roi_params.get('face', {})
			rois = get_face_ROI(video_path, **params) if params else get_face_ROI(video_path)
		else:
			rois = []
		return rois
