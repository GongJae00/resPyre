
import os
import numpy as np
import copy
from pathlib import Path
from core.utils.common import sort_nicely, get_chest_ROI, get_face_ROI

class DatasetBase:
	def __init__(self):
		repo_root = Path(__file__).resolve().parent.parent.parent.parent # src/respyre/datasets -> ... -> root
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
