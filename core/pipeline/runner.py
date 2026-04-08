
import os
import time
import numpy as np
from functools import wraps

from core.utils.common import tqdm
from core.pipeline.common import (
    _dataset_results_dir,
    _atomic_json_dump,
    _filter_valid_rois,
    _merge_results_payload,
    _method_suffix,
    _sanitize_run_label,
    derive_trial_identifiers,
)
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

        if not getattr(dataset, 'data', None):
            dataset.load_dataset()
        # Loop over the dataset
        for sample_idx, d in enumerate(tqdm(dataset.data, desc="Processing files")):
            dataset_label = str(getattr(dataset, 'name', '') or '').strip().lower() or 'unknown'
            dataset_slug = dataset_label.upper()
            d['dataset_name'] = dataset_label
            d.setdefault('dataset', dataset_label)
            d['dataset_slug'] = dataset_slug

            if 'trial' in d.keys():
                outfilename = os.path.join(data_dir, dataset.name + '_' + d['subject'] + '_' + d['trial'] + '.pkl')
            else:
                outfilename = os.path.join(data_dir, dataset.name + '_' + d['subject'] + '.pkl')

            _, d['fps'] = get_vid_stats(d['video_path'])
            # Deterministic per-sample identifiers shared by all methods.
            trial_key, trial_key_full = derive_trial_identifiers(d, dataset_name=dataset_label, sample_index=sample_idx)
            d['trial_key'] = trial_key
            d['trial_key_full'] = trial_key_full
            d['trial_uid'] = trial_key_full

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
                tqdm.write("> Processing video subject=%s trial=%s\n> fps: %d" % (d['subject'], d['trial'], d['fps']))
            else:
                tqdm.write("> Processing video %s\n> fps: %d" % (d['subject'], d['fps']))

            chest_roi_notice_shown = False

            def _ensure_chest_rois(reason: str):
                nonlocal chest_roi_notice_shown
                if not d['chest_rois']:
                    if not chest_roi_notice_shown:
                        tqdm.write(f"> Preparing chest ROIs ({reason}); this can take a while...")
                        chest_roi_notice_shown = True
                    d['chest_rois'] = _filter_valid_rois(dataset.extract_ROI(d['video_path'], 'chest'))
                else:
                    d['chest_rois'] = _filter_valid_rois(d['chest_rois'])

            # Apply every method to each video
            for m in methods:
                tqdm.write("> Applying method %s ..." % m.name)
                skip_method = False
                aux_dir = os.path.join(dataset_results_dir, 'aux', m.name.replace(' ', '_'))
                d['aux_save_dir'] = aux_dir
                needs_roi_meta = hasattr(m, 'osc_head') or ('__' in getattr(m, 'name', ''))

                if m.data_type == 'chest':
                    # Wrapped oscillator heads always require ROI-derived per-frame
                    # metadata (roi_stats_t), so ROIs must be prepared first.
                    if needs_roi_meta:
                        success_cache_only = False
                        if not d['chest_rois'] and hasattr(m, "can_run_without_chest_rois"):
                            can_cache_only = False
                            try:
                                can_cache_only = bool(m.can_run_without_chest_rois(d))
                            except Exception:
                                can_cache_only = False
                            if can_cache_only:
                                try:
                                    estimate = m.process(d)
                                    success_cache_only = True
                                    tqdm.write(f"> Using cache-only path for {m.name} (ROI extraction skipped)")
                                except Exception as exc:
                                    # Fall back to ROI extraction path for resilience.
                                    tqdm.write(
                                        f"> Cache-only path failed for {m.name} ({type(exc).__name__}: {exc}); "
                                        "falling back to ROI extraction."
                                    )

                        if not success_cache_only:
                            _ensure_chest_rois("for wrapped quality metadata")
                            if not d['chest_rois']:
                                tqdm.write(f"> Skipping method {m.name} (no valid chest ROIs)")
                                skip_method = True
                            else:
                                estimate = m.process(d)
                    else:
                        # Base chest methods can use lazy cache-first processing.
                        success_lazy = False
                        if not d['chest_rois']:
                            try:
                                estimate = m.process(d)
                                # Verify non-empty result (empty signal = cache miss)
                                if isinstance(estimate, np.ndarray) and estimate.size == 0:
                                    success_lazy = False
                                elif isinstance(estimate, dict) and estimate.get('signal_hat', np.array([])).size == 0:
                                    success_lazy = False
                                else:
                                    success_lazy = True
                            except Exception:
                                pass

                        if not success_lazy:
                            _ensure_chest_rois(f"cache miss at {m.name}")

                            if not d['chest_rois']:
                                tqdm.write(f"> Skipping method {m.name} (no valid chest ROIs)")
                                skip_method = True
                            else:
                                estimate = m.process(d)

                elif m.data_type == 'face':
                     if not d['face_rois']:
                        d['face_rois'] = _filter_valid_rois(dataset.extract_ROI(d['video_path'], m.data_type))
                     else:
                        d['face_rois'] = _filter_valid_rois(d['face_rois'])
                     if not d['face_rois']:
                        tqdm.write(f"> Skipping method {m.name} (no valid face ROIs)")
                        skip_method = True
                     else:
                        estimate = m.process(d)

                if skip_method:
                    continue
                
                # If we successfully lazy-loaded (success_lazy=True), 'estimate' is already set.
                # If we re-ran (else block), 'estimate' is set there.
                
                # Normalize estimate to dictionary format if it's a raw numpy array (Base Models)
                if isinstance(estimate, np.ndarray):
                    # For base models, we don't have time-tracking, so we wrap it simply
                    estimate = {
                        "signal_hat": estimate.reshape(-1),
                        "track_hz": np.array([]), # Base models don't provide frequency tracking
                        "times_hz": np.array([]),
                        "meta": "{}"
                    }

                results_payload['estimates'].append({'method': m.name, 'estimate': estimate})

            # release some memory between videos
            d.pop('aux_save_dir', None)
            d.pop('trial_key', None)
            d.pop('trial_key_full', None)
            d.pop('trial_uid', None)
            d.pop('roi_stats_t', None)
            d.pop('roi_intensity_mean', None)
            d.pop('roi_intensity_std', None)
            d.pop('roi_intensity_snr_db', None)
            d.pop('roi_stats_source', None)
            d.pop('roi_stats_cache_path', None)
            d.pop('_gray_chest_rois', None)
            d.pop('_obs_signal_cache', None)
            d['chest_rois'] = []
            d['face_rois'] = []

            _merge_results_payload(outfilename, results_payload, method_order=method_order)
            tqdm.write('> Results updated!\n')
