import argparse
import json
import os

import numpy as np

from riv.optim.em_kalman import EMKalmanTrainer, save_em_params, log_em_result


def _load_tracks(results_dir: str, method: str):
	aux_dir = os.path.join(results_dir, 'aux', method)
	tracks = []
	if not os.path.isdir(aux_dir):
		return tracks
	for fname in os.listdir(aux_dir):
		if not fname.endswith('.npz'):
			continue
		path = os.path.join(aux_dir, fname)
		try:
			with np.load(path, allow_pickle=True) as data:
				if 'track_hz' in data:
					track = np.asarray(data['track_hz'], dtype=np.float64)
					if track.size:
						tracks.append(track)
		except Exception:
			continue
	return tracks


def main():
	parser = argparse.ArgumentParser(description="EM training for Kalman gain parameters")
	parser.add_argument('--results', required=True, help='results/<run_label>')
	parser.add_argument('--dataset', required=True, help='Dataset label (e.g., COHFACE)')
	parser.add_argument('--method', required=True, help='Method name (e.g., profile1d_cubic__pll)')
	parser.add_argument('--max-iters', type=int, default=20)
	args = parser.parse_args()

	tracks = _load_tracks(args.results, args.method)
	if not tracks:
		print(f"No tracks found under {args.results}/aux/{args.method}")
		return
	stacked = np.concatenate(tracks)
	trainer = EMKalmanTrainer()
	trainer.cfg.max_iters = args.max_iters
	result = trainer.fit(stacked)
	save_em_params(args.dataset, args.method, result)
	log_em_result(args.dataset, args.method, result, source="train_em")
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
