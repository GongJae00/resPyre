
import numpy as np
import cv2 as cv
from components.observations.motion import OF, DoF, profile1D

class MethodBase:
	def __init__(self):
		self.name = ''
		self.win_size = 30
		self.data_type = ''

	def process(self, data):
		# This class can be used to process either videos or ROIs
		raise NotImplementedError("Subclasses must implement process method")

class OF_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'OF_Model'
		self.data_type = 'chest'

	def process(self, data):
		import cv2 as cv
		import os

		# Try loading cached observation
		if 'video_path' in data:
			trial_dir = os.path.dirname(data['video_path'])
			cache_path = os.path.join(trial_dir, "obs_of.npy")
			if os.path.exists(cache_path):
				# print(f"Loading cached OF: {cache_path}")
				return np.load(cache_path)

		# convert rois to grayscale
		g_rois = [cv.cvtColor(np.asarray(x), cv.COLOR_RGB2GRAY) for x in data['chest_rois']];

		# estimate OF
		of, _ = OF(g_rois, data['fps'])
		return of

class DoF_Model(MethodBase):

	def __init__(self):
		super().__init__()
		self.name = 'DoF'
		self.data_type = 'chest'

	def process(self, data):
		import cv2 as cv
		import os

		# Try loading cached observation
		if 'video_path' in data:
			trial_dir = os.path.dirname(data['video_path'])
			cache_path = os.path.join(trial_dir, "obs_dof.npy")
			if os.path.exists(cache_path):
				return np.load(cache_path)

		# convert rois to grayscale
		g_rois = [cv.cvtColor(np.asarray(x), cv.COLOR_RGB2GRAY) for x in data['chest_rois']];

		# estimate DoF
		dof, _ = DoF(g_rois, data['fps'])
		return dof

class profile1D_Model(MethodBase):

	def __init__(self, interp_type='quadratic'):
		super().__init__()
		self.name = 'profile1D ' + interp_type 
		self.data_type = 'chest'
		self.interp_type = interp_type

	def process(self, data):
		import cv2 as cv
		import os

		# Try loading cached observation
		if 'video_path' in data:
			trial_dir = os.path.dirname(data['video_path'])
			# Map interp_type to suffix: linear/quadratic/cubic -> p1d_linear/p1d_quad/p1d_cubic
			suffix = self.interp_type
			if suffix == 'quadratic': suffix = 'quad'
			cache_path = os.path.join(trial_dir, f"obs_p1d_{suffix}.npy")
			if os.path.exists(cache_path):
				return np.load(cache_path)

		# convert rois to grayscale
		g_rois = [cv.cvtColor(np.asarray(x), cv.COLOR_RGB2GRAY) for x in data['chest_rois']];

		# estimate profile1D
		profile, _ = profile1D(g_rois, data['fps'], self.interp_type)
		return profile
