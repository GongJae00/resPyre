
import os
import time
import numpy as np
from functools import wraps

from core.utils.common import tqdm
from core.pipeline.common import _dataset_results_dir, _atomic_json_dump, _filter_valid_rois, _merge_results_payload, _method_suffix, _sanitize_run_label
# from core.utils.config import load_config # Config loader will be fixed separately

def get_vid_stats(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count, fps

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
            dataset_label = str(getattr(dataset, 'name', '') or '').strip().lower() or 'unknown'
            dataset_slug = dataset_label.upper()
            d['dataset_name'] = dataset_label
            d.setdefault('dataset', dataset_label)
            d['dataset_slug'] = dataset_slug

            if 'trial' in d.keys(): 
                outfilename = os.path.join(data_dir, dataset.name + '_' + d['subject'] + '_' + d['trial'] + '.pkl')
                trial_key = f"{d['subject']}_{d['trial']}"
            else:
                outfilename = os.path.join(data_dir, dataset.name + '_' + d['subject'] + '.pkl')
                trial_key = d['subject']

            _, d['fps'] = get_vid_stats(d['video_path'])
            d['trial_key'] = trial_key

            results_payload = {
                'dataset': dataset_label,
                'dataset_name': dataset_label,
                'dataset_slug': dataset_slug,
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
                if skip_method:
                    continue

                estimate = m.process(d)
                results_payload['estimates'].append({'method': m.name, 'estimate': estimate})

            # release some memory between videos
            d.pop('aux_save_dir', None)
            d.pop('trial_key', None)
            d['chest_rois'] = []
            d['face_rois'] = []

            _merge_results_payload(outfilename, results_payload, method_order=method_order)
            tqdm.write('> Results updated!\n')
