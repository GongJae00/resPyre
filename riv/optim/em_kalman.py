"""
Lightweight EM algorithm for refining Kalman gain (Q/R) parameters per dataset.
This module follows the conceptual outline from SOMATA but is adapted to resPyre signals.
"""
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EMConfig:
	max_iters: int = 25
	tol: float = 1e-4
	# Respiration tracks are z-scored, so allow slow but non-zero drift.
	min_q: float = 5e-5
	max_q: float = 0.05
	min_r: float = 5e-4
	max_r: float = 0.5
	init_q: float = 5e-4
	init_r: float = 0.05


class EMKalmanTrainer:
	def __init__(self, config: Optional[EMConfig] = None):
		self.cfg = config or EMConfig()

	def _forward_backward(self, observations: np.ndarray, q: float, r: float) -> Dict:
		n = observations.size
		if n == 0:
			return {}
		x_pred = np.zeros(n)
		p_pred = np.zeros(n)
		x_filt = np.zeros(n)
		p_filt = np.zeros(n)
		x_smooth = np.zeros(n)
		p_smooth = np.zeros(n)
		p_cross = np.zeros(n)

		x = 0.0
		P = 1.0
		for t in range(n):
			x_pred[t] = x
			p_pred[t] = P + q
			K = p_pred[t] / (p_pred[t] + r)
			x = x_pred[t] + K * (observations[t] - x_pred[t])
			P = (1 - K) * p_pred[t]
			x_filt[t] = x
			p_filt[t] = P

		x_smooth[-1] = x_filt[-1]
		p_smooth[-1] = p_filt[-1]
		for t in range(n - 2, -1, -1):
			denom = p_pred[t + 1] + 1e-9
			A = p_filt[t] / denom
			x_smooth[t] = x_filt[t] + A * (x_smooth[t + 1] - x_pred[t + 1])
			p_smooth[t] = p_filt[t] + A * (p_smooth[t + 1] - p_pred[t + 1]) * A
			p_cross[t + 1] = A * p_smooth[t + 1]

		return {
			"x_smooth": x_smooth,
			"p_smooth": p_smooth,
			"p_cross": p_cross,
			"x_filt": x_filt,
			"p_filt": p_filt,
			"x_pred": x_pred,
			"p_pred": p_pred
	}

	def _em_update(self, sequences: List[np.ndarray], q: float, r: float) -> Dict:
		total_samples = 0
		total_transitions = 0
		sum_q = 0.0
		sum_r = 0.0
		sum_ll = 0.0
		for obs in sequences:
			stats = self._forward_backward(obs, q, r)
			if not stats:
				continue
			n = obs.size
			if n == 0:
				continue
			total_samples += n
			x_smooth = stats["x_smooth"]
			p_smooth = stats["p_smooth"]
			p_cross = stats.get("p_cross")
			residual = obs - x_smooth
			sum_r += float(np.sum(residual ** 2 + p_smooth))
			sum_ll += float(-0.5 * np.sum(np.log(2 * np.pi * (p_smooth + r)) + (residual ** 2) / (p_smooth + r)))
			if n > 1:
				total_transitions += (n - 1)
				sq = 0.0
				for t in range(1, n):
					ex_t = x_smooth[t]
					ex_prev = x_smooth[t - 1]
					var_t = p_smooth[t]
					var_prev = p_smooth[t - 1]
					cross = 0.0 if p_cross is None else p_cross[t]
					sq += (ex_t ** 2 + var_t) + (ex_prev ** 2 + var_prev) - 2.0 * (ex_t * ex_prev + cross)
				sum_q += sq
		if total_samples == 0:
			return {"q": q, "r": r, "ll": float("-inf")}
		if total_transitions > 0:
			q_new = sum_q / total_transitions
		else:
			q_new = q
		r_new = sum_r / total_samples
		q_new = float(np.clip(q_new, self.cfg.min_q, self.cfg.max_q))
		r_new = float(np.clip(r_new, self.cfg.min_r, self.cfg.max_r))
		return {"q": q_new, "r": r_new, "ll": float(sum_ll)}

	def _normalize_sequences(self, observations) -> List[np.ndarray]:
		if isinstance(observations, np.ndarray):
			arr = np.asarray(observations, dtype=np.float64).reshape(-1)
			return [arr] if arr.size else []
		sequences: List[np.ndarray] = []
		for seq in observations or []:
			arr = np.asarray(seq, dtype=np.float64).reshape(-1)
			if arr.size:
				sequences.append(arr)
		return sequences

	def fit(self, observations, init_q: Optional[float] = None, init_r: Optional[float] = None) -> Dict:
		sequences = self._normalize_sequences(observations)
		if not sequences:
			return {"q": self.cfg.init_q, "r": self.cfg.init_r, "ll": float("-inf")}
		q = init_q if init_q is not None else self.cfg.init_q
		r = init_r if init_r is not None else self.cfg.init_r
		prev_ll = float("-inf")
		for _ in range(self.cfg.max_iters):
			update = self._em_update(sequences, q, r)
			q, r, ll = update["q"], update["r"], update["ll"]
			if abs(ll - prev_ll) < self.cfg.tol:
				break
			prev_ll = ll
		return {"q": q, "r": r, "ll": prev_ll}


def save_em_params(dataset: str, method: str, params: Dict, base_dir: str = "runs/em_params"):
	os.makedirs(base_dir, exist_ok=True)
	path = os.path.join(base_dir, f"{dataset.lower()}_{method}.json")
	payload = {"params": params}
	with open(path, 'w', encoding='utf-8') as fp:
		json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_em_params(dataset: str, method: str, base_dir: str = "runs/em_params") -> Dict:
	filename = f"{dataset.lower()}_{method}.json"
	path = os.path.join(base_dir, filename)
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


def log_em_result(dataset: str, method: str, params: Dict, source: str = "manual", log_dir: str = "runs/em_logs"):
	os.makedirs(log_dir, exist_ok=True)
	path = os.path.join(log_dir, "em_training.csv")
	row = {
		'dataset': dataset,
		'method': method,
		'q': params.get('q'),
		'r': params.get('r'),
		'log_likelihood': params.get('ll'),
		'source': source
	}
	write_header = not os.path.exists(path)
	with open(path, 'a', newline='', encoding='utf-8') as fp:
		writer = csv.DictWriter(fp, fieldnames=row.keys())
		if write_header:
			writer.writeheader()
		writer.writerow(row)
