import argparse
import json
import os
import pickle

import numpy as np

from riv.optim.em_kalman import EMKalmanTrainer, save_em_params, log_em_result


def _load_signals(results_dir: str, method: str):
	sanitized = method.replace(' ', '_')
	aux_dir = os.path.join(results_dir, 'aux', sanitized)
	signals = []
	lmethod = method.lower()
	# Tracker-style oscillator heads (KFstd/UKFfreq) should prefer their frequency track.
	is_tracker_like = ('__kfstd' in lmethod) or ('__ukffreq' in lmethod)
	if not os.path.isdir(aux_dir):
		return signals
	for fname in os.listdir(aux_dir):
		if not fname.endswith('.npz'):
			continue
		path = os.path.join(aux_dir, fname)
		try:
			with np.load(path, allow_pickle=True) as data:
				series = None
				if is_tracker_like and 'track_hz' in data:
					series = np.asarray(data['track_hz'], dtype=np.float64).reshape(-1)
				elif 'signal_hat' in data:
					series = np.asarray(data['signal_hat'], dtype=np.float64).reshape(-1)
				elif 'track_hz' in data:
					series = np.asarray(data['track_hz'], dtype=np.float64).reshape(-1)
				if series is None or not series.size:
					continue
				series = np.asarray(series, dtype=np.float64)
				series = series[np.isfinite(series)]
				if series.size < 16:
					continue
				median = float(np.median(series))
				mad = float(np.median(np.abs(series - median)))
				sigma = 1.4826 * mad if mad > 0 else float(np.std(series))
				scale = max(sigma, 1e-6)
				series = (series - median) / scale
				series = np.clip(series, -5.0, 5.0)
				signals.append(series)
		except Exception:
			continue
	return signals


def _load_metrics_for_method(results_dir: str, method: str):
	metrics_path = os.path.join(results_dir, 'metrics', 'metrics.pkl')
	if not os.path.exists(metrics_path):
		return None, []
	try:
		with open(metrics_path, 'rb') as fp:
			metric_names, method_metrics = pickle.load(fp)
	except Exception:
		return None, []
	records = {}
	if isinstance(method_metrics, dict):
		records = method_metrics
	return metric_names, records.get(method, [])


def _select_top_trials(metric_names, records, top_k):
	if not metric_names:
		return []
	try:
		idx_mae = metric_names.index('MAE')
	except ValueError:
		return []
	candidates = []
	for record in records:
		metrics = record.get('metrics') or []
		if idx_mae >= len(metrics):
			continue
		try:
			mae_val = float(metrics[idx_mae])
		except (TypeError, ValueError):
			continue
		if not np.isfinite(mae_val):
			continue
		trial_key = record.get('trial_key') or (record.get('quality') or {}).get('trial_key')
		if not trial_key:
			continue
		rel = None
		track_stats = record.get('track_stats') or {}
		try:
			rel = float(track_stats.get('reliability_meta'))
		except (TypeError, ValueError):
			rel = None
		if rel is None or not np.isfinite(rel):
			rel = 0.5
		rel = float(np.clip(rel, 0.05, 1.0))
		effective_mae = mae_val / max(0.5, rel)
		candidates.append((effective_mae, trial_key, record))
	candidates.sort(key=lambda item: item[0])
	return candidates[:top_k]


def _load_aux_meta(results_dir: str, method: str, trial_key: str):
	sanitized = method.replace(' ', '_')
	aux_path = os.path.join(results_dir, 'aux', sanitized, f"{trial_key}.npz")
	if not os.path.exists(aux_path):
		return {}
	try:
		with np.load(aux_path, allow_pickle=True) as data:
			if 'meta' not in data:
				return {}
			meta_raw = data['meta']
			meta = None
			if isinstance(meta_raw, np.ndarray):
				if meta_raw.size:
					meta = meta_raw.flat[0]
			else:
				meta = meta_raw
			if isinstance(meta, bytes):
				meta = meta.decode('utf-8', errors='ignore')
			if isinstance(meta, str):
				meta = json.loads(meta)
			if isinstance(meta, dict):
				return meta
	except Exception:
		return {}
	return {}


def _build_autotune_override(results_dir: str, dataset: str, method: str, top_k: int, rv_scale: float, qx_base: float):
	metric_names, records = _load_metrics_for_method(results_dir, method)
	top_trials = _select_top_trials(metric_names, records, top_k)
	if not top_trials:
		print(f"> Warning: unable to derive top trials for {method}; autotune override skipped")
		return None
	sigma_vals = []
	snr_vals = []
	weights = []
	for _, trial_key, record in top_trials:
		meta = _load_aux_meta(results_dir, method, trial_key)
		if not meta:
			continue
		rel = meta.get('reliability_meta')
		if rel is None:
			track_stats = record.get('track_stats') or {}
			rel = track_stats.get('reliability_meta')
		try:
			rel = float(rel)
		except (TypeError, ValueError):
			rel = None
		if rel is None or not np.isfinite(rel):
			rel = 0.5
		rel = float(np.clip(rel, 0.1, 1.0))
		robust = meta.get('robust_z', {}) if isinstance(meta, dict) else {}
		sigma = robust.get('sigma_hat') if isinstance(robust, dict) else None
		if sigma is None:
			sigma = meta.get('sigma_hat')
		try:
			sigma = float(sigma)
		except (TypeError, ValueError):
			sigma = None
		if sigma is not None and np.isfinite(sigma):
			sigma_vals.append(sigma)
			weights.append(rel)
		else:
			weights.append(rel)
		snr_val = meta.get('snr_estimate')
		if snr_val is None:
			track_stats = record.get('track_stats') or {}
			snr_val = track_stats.get('snr_estimate')
		try:
			snr_val = float(snr_val)
		except (TypeError, ValueError):
			snr_val = None
		if snr_val is not None and np.isfinite(snr_val):
			snr_vals.append(snr_val)
	if not sigma_vals:
		print(f"> Warning: insufficient sigma_hat stats for {method}; autotune override skipped")
		return None
	if weights and len(weights) == len(sigma_vals):
		sigma_med = float(np.average(np.asarray(sigma_vals, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64)))
	else:
		sigma_med = float(np.median(np.asarray(sigma_vals, dtype=np.float64)))
	if snr_vals:
		if weights and len(weights) >= len(snr_vals):
			snr_med = float(np.average(np.asarray(snr_vals, dtype=np.float64), weights=np.asarray(weights[:len(snr_vals)], dtype=np.float64)))
		else:
			snr_med = float(np.median(np.asarray(snr_vals, dtype=np.float64)))
	else:
		snr_med = float('nan')
	rv_override = max((rv_scale * sigma_med) ** 2, 1e-8)
	if np.isfinite(snr_med) and snr_med > 0.0:
		qx_override = max(qx_base * (1.0 / max(snr_med, 1e-6)), 1e-9)
	else:
		qx_override = float(qx_base)
	payload = {
		"source": "train_em/top_trials",
		"top_k": len(sigma_vals),
		"stats": {
			"sigma_hat_median": sigma_med,
			"snr_median": snr_med
		},
		"params": {
			"rv_floor_override": rv_override,
			"qx_override": qx_override
		}
	}
	ds_dir = os.path.join('runs', 'autotune', dataset.lower())
	os.makedirs(ds_dir, exist_ok=True)
	out_path = os.path.join(ds_dir, f"{method}.json")
	with open(out_path, 'w', encoding='utf-8') as fp:
		json.dump(payload, fp, ensure_ascii=False, indent=2)
	return out_path


def main():
	parser = argparse.ArgumentParser(description="EM training for Kalman gain parameters")
	parser.add_argument('--results', required=True, help='results/<run_label>')
	parser.add_argument('--dataset', required=True, help='Dataset label (e.g., COHFACE)')
	parser.add_argument('--method', required=True, help='Method name (e.g., profile1d_cubic__pll)')
	parser.add_argument('--max-iters', type=int, default=20)
	parser.add_argument('--build-autotune', action='store_true', help='Generate runs/autotune/<dataset>/<method>.json from top-K trials')
	parser.add_argument('--autotune-top-k', type=int, default=5, help='Number of best trials to aggregate for autotune overrides')
	parser.add_argument('--autotune-rv-scale', type=float, default=1.2, help='Scaling factor applied to sigma_hat when estimating rv_floor_override')
	parser.add_argument('--autotune-qx-base', type=float, default=1e-4, help='Base qx override when SNR is unavailable')
	args = parser.parse_args()

	signals = _load_signals(args.results, args.method)
	if not signals:
		print(f"No usable signals found under {args.results}/aux/{args.method}")
		return
	trainer = EMKalmanTrainer()
	trainer.cfg.max_iters = args.max_iters
	result = trainer.fit(signals)
	save_em_params(args.dataset, args.method, result)
	log_em_result(args.dataset, args.method, result, source="train_em")
	if args.build_autotune:
		path = _build_autotune_override(
			args.results,
			args.dataset,
			args.method,
			max(1, args.autotune_top_k),
			max(args.autotune_rv_scale, 1e-6),
			max(args.autotune_qx_base, 1e-9)
		)
		if path:
			print(f"Autotune override saved to {path}")
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
