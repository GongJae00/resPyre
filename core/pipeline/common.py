
import os
import glob
import hashlib
import numpy as np
import copy
import re
from collections import defaultdict
import time
from functools import wraps
import contextlib
import tempfile
import pickle
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

def _dataset_results_dir(results_dir, dataset_name):
	path = os.path.join(results_dir, dataset_name)
	os.makedirs(path, exist_ok=True)
	# New structure for artifacts
	for sub in ("data", "metrics", "plots", "logs"):
		os.makedirs(os.path.join(path, sub), exist_ok=True)
	return path


def sanitize_trial_key(value: str, fallback: str = "trial") -> str:
	"""Normalize a trial key to filename-safe `[A-Za-z0-9._-]`."""
	raw = str(value or "").strip()
	if not raw:
		raw = fallback
	safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
	safe = safe.strip("._-")
	if not safe:
		safe = fallback
	return safe


FRAME_LOG_MANIFEST_SCHEMA = "frame_logs_manifest.v1"


def derive_trial_identifiers(sample: dict, dataset_name: str = "", sample_index: int = 0):
	"""Derive deterministic short/full trial identifiers.

	Returns:
		(short_key, full_key)
	"""
	dataset = str(
		(sample or {}).get("dataset_name")
		or (sample or {}).get("dataset")
		or (sample or {}).get("dataset_slug")
		or dataset_name
		or "unknown"
	).strip().lower()
	subject = str((sample or {}).get("subject", "")).strip()
	trial = str((sample or {}).get("trial", "")).strip()

	if subject and trial:
		short_key = sanitize_trial_key(f"{subject}_{trial}", fallback=f"idx_{sample_index}")
		full_key = sanitize_trial_key(f"{dataset}__{subject}__{trial}", fallback=short_key)
		return short_key, full_key
	if subject:
		short_key = sanitize_trial_key(subject, fallback=f"idx_{sample_index}")
		full_key = sanitize_trial_key(f"{dataset}__{subject}", fallback=short_key)
		return short_key, full_key

	video_path = str((sample or {}).get("video_path", "")).strip()
	if video_path:
		stem = os.path.splitext(os.path.basename(video_path))[0]
		short_key = sanitize_trial_key(stem, fallback=f"idx_{sample_index}")
		full_key = sanitize_trial_key(f"{dataset}__{stem}", fallback=short_key)
		return short_key, full_key

	short_key = sanitize_trial_key(f"idx_{sample_index}", fallback="idx_0")
	full_key = sanitize_trial_key(f"{dataset}__{short_key}", fallback=short_key)
	return short_key, full_key


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


def resolve_target_run_dirs(results_dir, run_label=None):
	"""Resolve result run directories deterministically.

	Rules:
	  1) no run_label -> all result dirs with `data/`
	  2) run_label exact dir exists -> use exact dir only
	  3) otherwise fallback to multi-dataset pattern `<label>_*` only
	"""
	if not run_label:
		candidates = glob.glob(os.path.join(results_dir, "*"))
	else:
		label = _sanitize_run_label(run_label)
		exact = os.path.join(results_dir, label)
		if os.path.isdir(exact) and os.path.exists(os.path.join(exact, "data")):
			return [exact]
		candidates = glob.glob(os.path.join(results_dir, f"{label}_*"))

	return [
		d for d in candidates
		if os.path.isdir(d) and os.path.exists(os.path.join(d, "data"))
	]

def _deep_merge_dict(base, new_values):
	if not isinstance(new_values, dict):
		return base
	for key, value in new_values.items():
		if isinstance(value, dict) and isinstance(base.get(key), dict):
			base[key] = _deep_merge_dict(base[key], value)
		else:
			base[key] = copy.deepcopy(value)
	return base

@contextlib.contextmanager
def _file_lock(lock_path):
	os.makedirs(os.path.dirname(lock_path), exist_ok=True)
	fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
	try:
		if os.name == 'nt':
			import msvcrt  # type: ignore
			os.lseek(fd, 0, os.SEEK_SET)
			msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
		else:
			import fcntl  # type: ignore
			fcntl.flock(fd, fcntl.LOCK_EX)
		yield
	finally:
		try:
			if os.name == 'nt':
				import msvcrt  # type: ignore
				os.lseek(fd, 0, os.SEEK_SET)
				msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
			else:
				import fcntl  # type: ignore
				fcntl.flock(fd, fcntl.LOCK_UN)
		finally:
			os.close(fd)


def _atomic_pickle_dump(data, path):
	dir_name = os.path.dirname(path)
	os.makedirs(dir_name, exist_ok=True)
	fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
	try:
		with os.fdopen(fd, 'wb') as tmp_file:
			pickle.dump(data, tmp_file)
		os.replace(tmp_path, path)
	except Exception:
		try:
			os.unlink(tmp_path)
		except OSError:
			pass
		raise


def _atomic_json_dump(data, path, indent=2):
	dir_name = os.path.dirname(path)
	os.makedirs(dir_name, exist_ok=True)
	fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
	try:
		with os.fdopen(fd, 'w', encoding='utf-8') as tmp_file:
			json.dump(data, tmp_file, indent=indent)
		os.replace(tmp_path, path)
	except Exception:
		try:
			os.unlink(tmp_path)
		except OSError:
			pass
		raise


def _utc_iso_now() -> str:
	return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _frame_logs_dir(aux_dir: str) -> str:
	return os.path.join(str(aux_dir), "frame_logs")


def _frame_log_manifest_path(aux_dir: str) -> str:
	return os.path.join(_frame_logs_dir(aux_dir), "frame_logs_manifest.json")


def _read_frame_log_manifest(aux_dir: str):
	path = _frame_log_manifest_path(aux_dir)
	if not os.path.exists(path):
		return None
	try:
		with open(path, "r", encoding="utf-8") as fp:
			obj = json.load(fp)
	except Exception:
		return None
	if not isinstance(obj, dict):
		return None
	if obj.get("schema") != FRAME_LOG_MANIFEST_SCHEMA:
		return None
	entries = obj.get("entries", {})
	if not isinstance(entries, dict):
		obj["entries"] = {}
	return obj


def _sha256_file(path: str) -> str:
	h = hashlib.sha256()
	with open(path, "rb") as fp:
		for chunk in iter(lambda: fp.read(1024 * 1024), b""):
			h.update(chunk)
	return h.hexdigest()


def update_frame_log_manifest(
	aux_dir: str,
	base_trial_key: str,
	actual_filename: str,
	suffix: int,
	sha256: str = "",
	notes: str = "",
):
	"""Update per-method frame-log manifest so latest save resolves deterministically."""
	log_dir = _frame_logs_dir(aux_dir)
	os.makedirs(log_dir, exist_ok=True)
	manifest = _read_frame_log_manifest(aux_dir) or {
		"schema": FRAME_LOG_MANIFEST_SCHEMA,
		"updated_at": _utc_iso_now(),
		"entries": {},
	}
	entries = manifest.get("entries", {})
	if not isinstance(entries, dict):
		entries = {}
	manifest["entries"] = entries
	entries[str(base_trial_key)] = {
		"filename": str(actual_filename),
		"saved_at": _utc_iso_now(),
		"suffix": int(max(suffix, 0)),
		"sha256": str(sha256 or ""),
		"notes": str(notes or ""),
	}
	manifest["updated_at"] = _utc_iso_now()
	_atomic_json_dump(manifest, _frame_log_manifest_path(aux_dir), indent=2)
	return manifest


def _frame_log_candidates(aux_dir: str, base_trial_key: str):
	log_dir = _frame_logs_dir(aux_dir)
	if not os.path.isdir(log_dir):
		return []
	base = str(base_trial_key)
	pattern = re.compile(rf"^{re.escape(base)}(?:_(\d+))?\.npz$")
	candidates = []
	for path in glob.glob(os.path.join(log_dir, f"{base}*.npz")):
		fname = os.path.basename(path)
		m = pattern.match(fname)
		if not m:
			continue
		sfx = int(m.group(1)) if m.group(1) is not None else 0
		candidates.append((sfx, fname, path))
	candidates.sort(key=lambda x: x[0])
	return candidates


def resolve_frame_log_path(aux_dir: str, base_trial_key: str, strict: bool = True):
	"""Resolve frame log path deterministically.

	Resolution order:
	  1) manifest entry (if present)
	  2) fallback base/suffix scan
	"""
	base = str(base_trial_key or "").strip()
	if not base:
		raise ValueError("base_trial_key must be a non-empty string")
	info = {
		"base_trial_key": base,
		"frame_log_filename_used": "",
		"frame_log_resolution_mode": "missing",
		"frame_log_suffix_used": -1,
		"frame_log_ambiguity": False,
	}
	manifest = _read_frame_log_manifest(aux_dir)
	if isinstance(manifest, dict):
		entry = manifest.get("entries", {}).get(base)
		if isinstance(entry, dict):
			fname = str(entry.get("filename", "")).strip()
			if not fname:
				raise ValueError(
					f"Frame log manifest entry for '{base}' is malformed: missing filename."
				)
			path = os.path.join(_frame_logs_dir(aux_dir), fname)
			if not os.path.exists(path):
				raise ValueError(
					f"Frame log manifest entry for '{base}' points to missing file '{path}'."
				)
			sfx = entry.get("suffix", 0)
			try:
				sfx = int(sfx)
			except Exception:
				sfx = 0
			info.update({
				"frame_log_filename_used": fname,
				"frame_log_resolution_mode": "manifest",
				"frame_log_suffix_used": sfx,
				"frame_log_ambiguity": False,
			})
			return path, info

	candidates = _frame_log_candidates(aux_dir, base)
	if not candidates:
		return None, info
	if len(candidates) > 1 and strict:
		choices = ", ".join(c[1] for c in candidates)
		raise ValueError(
			f"Ambiguous frame log candidates for trial '{base}' without manifest: {choices}. "
			"Set strict=False for suffix fallback or regenerate logs with manifest."
		)
	# non-strict or unique candidate
	if any(sfx == 0 for sfx, _, _ in candidates):
		if len(candidates) == 1:
			sfx, fname, path = candidates[0]
			info.update({
				"frame_log_filename_used": fname,
				"frame_log_resolution_mode": "fallback_base",
				"frame_log_suffix_used": int(sfx),
				"frame_log_ambiguity": False,
			})
			return path, info
		# choose highest suffix when ambiguity is allowed
		sfx, fname, path = candidates[-1]
		info.update({
			"frame_log_filename_used": fname,
			"frame_log_resolution_mode": "fallback_suffix",
			"frame_log_suffix_used": int(sfx),
			"frame_log_ambiguity": True,
		})
		return path, info
	# suffix-only candidates
	sfx, fname, path = candidates[-1]
	info.update({
		"frame_log_filename_used": fname,
		"frame_log_resolution_mode": "fallback_suffix",
		"frame_log_suffix_used": int(sfx),
		"frame_log_ambiguity": len(candidates) > 1,
	})
	return path, info


def _parse_iso8601_to_epoch(value: str) -> Optional[float]:
	txt = str(value or "").strip()
	if not txt:
		return None
	try:
		if txt.endswith("Z"):
			txt = txt[:-1] + "+00:00"
		return float(datetime.fromisoformat(txt).timestamp())
	except Exception:
		return None


def infer_trial_key_from_data_stem(fname: str, dataset_token: Optional[str] = None) -> str:
	stem = str(fname or "").strip()
	if not stem:
		return ""
	if dataset_token:
		prefix = f"{str(dataset_token).lower()}_"
		if stem.lower().startswith(prefix):
			return stem[len(prefix):]
	if "_" in stem:
		return stem.split("_", 1)[1]
	return stem


def collect_expected_method_trials(run_dir: str) -> List[Dict[str, str]]:
	"""Collect expected (method, trial) pairs from run data/*.pkl."""
	out: List[Dict[str, str]] = []
	seen = set()
	data_dir = os.path.join(run_dir, "data")
	for pkl_path in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
		try:
			with open(pkl_path, "rb") as fp:
				obj = pickle.load(fp)
		except Exception:
			continue
		stem = os.path.splitext(os.path.basename(pkl_path))[0]
		dataset_token = stem.split("_", 1)[0] if "_" in stem else None
		trial = infer_trial_key_from_data_stem(stem, dataset_token=dataset_token)
		estimates = obj.get("estimates", []) if isinstance(obj, dict) else []
		for entry in estimates:
			if not isinstance(entry, dict):
				continue
			method = str(entry.get("method", "")).strip()
			if not method or not trial:
				continue
			key = (method, trial)
			if key in seen:
				continue
			seen.add(key)
			out.append({"method": method, "trial": trial})
	return out


def _frame_log_info_for_method(run_dir: str, method_slug: str):
	log_dir = os.path.join(run_dir, "aux", method_slug, "frame_logs")
	rows = []
	if not os.path.isdir(log_dir):
		return rows
	for path in sorted(glob.glob(os.path.join(log_dir, "*.npz"))):
		fname = os.path.basename(path)
		stem = os.path.splitext(fname)[0]
		suffix = 0
		base_guess = stem
		if "_" in stem:
			maybe_sfx = stem.rsplit("_", 1)[1]
			if maybe_sfx.isdigit():
				suffix = int(maybe_sfx)
				base_guess = stem.rsplit("_", 1)[0]
		try:
			mtime = float(os.path.getmtime(path))
		except OSError:
			mtime = 0.0
		rows.append({
			"path": path,
			"filename": fname,
			"stem": stem,
			"base_guess": base_guess,
			"suffix": int(suffix),
			"mtime": float(mtime),
			"method": method_slug,
		})
	return rows


def _select_latest_candidate(candidates: List[Dict]) -> Optional[Dict]:
	if not candidates:
		return None
	# Deterministic tie-break: mtime -> suffix -> stem -> path
	ordered = sorted(
		candidates,
		key=lambda c: (
			float(c.get("mtime", 0.0)),
			int(c.get("trial_suffix", c.get("suffix", 0))),
			str(c.get("stem", "")),
			str(c.get("path", "")),
		)
	)
	return ordered[-1]


def _trial_relative_suffix(stem: str, trial_base: str) -> int:
	"""Return collision suffix relative to expected trial key.

	Examples:
	  stem='10_3', trial_base='10_3'     -> 0
	  stem='10_3_1', trial_base='10_3'   -> 1
	  stem='10_3_x', trial_base='10_3'   -> -1 (not a valid collision suffix)
	"""
	stem_s = str(stem or "")
	base = str(trial_base or "")
	if not base:
		return -1
	if stem_s == base:
		return 0
	prefix = f"{base}_"
	if not stem_s.startswith(prefix):
		return -1
	tail = stem_s[len(prefix):]
	if tail.isdigit():
		return int(tail)
	return -1


def _match_trial_candidates(method_rows: List[Dict], trial_base: str) -> List[Dict]:
	base = str(trial_base)
	out = []
	for row in method_rows:
		stem = str(row.get("stem", ""))
		rel_suffix = _trial_relative_suffix(stem, base)
		if rel_suffix >= 0:
			row_copy = dict(row)
			row_copy["trial_suffix"] = int(rel_suffix)
			out.append(row_copy)
	return out


def resolve_frame_logs_for_run(
	run_dir: str,
	expected_trials: Optional[List[Dict[str, str]]] = None,
	method_filter: Optional[List[str]] = None,
	strict: bool = True,
	allow_empty: bool = False,
):
	"""Resolve canonical frame logs for this run execution.

	Returns a structured object:
	  {
	    "canonical": {method: {trial_key: absolute_npz_path}},
	    "extras": [{method, trial_key, path, reason, filename, suffix, mtime}],
	    "missing": [{method, trial_key, reason}],
	    "diag": {...}
	  }

	Strict mode (`strict=True`) raises ValueError when:
	  - ambiguous candidates exist without a unique manifest-based selection
	  - extra/stale/unindexed logs are present
	  - no canonical logs are selected while expected trials exist and allow_empty=False
	"""
	run_dir = os.path.abspath(str(run_dir))
	expected = expected_trials if isinstance(expected_trials, list) else collect_expected_method_trials(run_dir)
	method_filter_set = set(str(m) for m in method_filter) if method_filter else None
	method_expected: Dict[str, List[str]] = {}
	for item in expected:
		method = str(item.get("method", "")).strip()
		trial = str(item.get("trial", "")).strip()
		if not method or not trial:
			continue
		if method_filter_set is not None and method not in method_filter_set:
			continue
		method_expected.setdefault(method, [])
		if trial not in method_expected[method]:
			method_expected[method].append(trial)

	status_path = os.path.join(run_dir, "run_status.json")
	run_instance_started_at = None
	run_epoch = None
	selection_policy = "latest_mtime_suffix_no_epoch"
	warnings = []
	if os.path.exists(status_path):
		try:
			with open(status_path, "r", encoding="utf-8") as fp:
				status_obj = json.load(fp)
			run_instance_started_at = status_obj.get("run_instance_started_at")
			run_epoch = _parse_iso8601_to_epoch(run_instance_started_at)
		except Exception as exc:
			warnings.append(f"Failed to parse run_status.json: {exc}")
	if run_epoch is not None:
		selection_policy = "post_epoch_latest_mtime_suffix"
	else:
		warnings.append(
			"run_instance_started_at missing or invalid; using latest-mtime fallback across all frame logs."
		)
	slack_sec = 2.0
	cutoff = (run_epoch - slack_sec) if run_epoch is not None else None

	aux_dir = os.path.join(run_dir, "aux")
	method_slugs = []
	if os.path.isdir(aux_dir):
		for d in sorted(glob.glob(os.path.join(aux_dir, "*"))):
			if os.path.isdir(d):
				method_slugs.append(os.path.basename(d))

	for method in sorted(method_expected.keys()):
		slug = method.replace(" ", "_")
		if slug not in method_slugs:
			method_slugs.append(slug)

	if method_filter_set is not None:
		filter_slugs = set(m.replace(" ", "_") for m in method_filter_set)
		method_slugs = [m for m in method_slugs if m in filter_slugs]

	canonical: Dict[str, Dict[str, str]] = {}
	selected_details: Dict[str, Dict[str, Dict[str, object]]] = {}
	extras: List[Dict[str, object]] = []
	missing: List[Dict[str, object]] = []
	ambiguities: List[Dict[str, object]] = []

	for method_slug in sorted(method_slugs):
		method_rows = _frame_log_info_for_method(run_dir, method_slug)
		aux_method_dir = os.path.join(run_dir, "aux", method_slug)
		method_manifest = _read_frame_log_manifest(aux_method_dir)
		manifest_entries = {}
		if isinstance(method_manifest, dict):
			entries = method_manifest.get("entries", {})
			if isinstance(entries, dict):
				manifest_entries = entries

		method_name = method_slug
		if method_slug not in method_expected:
			for expected_name in method_expected.keys():
				if expected_name.replace(" ", "_") == method_slug:
					method_name = expected_name
					break

		expected_trials_for_method = list(method_expected.get(method_name, []))
		canonical[method_name] = {}
		selected_details[method_name] = {}
		consumed_paths = set()

		for trial_base in sorted(expected_trials_for_method):
			cands = _match_trial_candidates(method_rows, trial_base)
			if not cands:
				missing.append({
					"method": method_name,
					"trial_key": trial_base,
					"reason": "expected_not_found",
				})
				continue

			eligible = cands
			if cutoff is not None:
				eligible = [c for c in cands if float(c.get("mtime", 0.0)) >= cutoff]
				if not eligible:
					for c in cands:
						extras.append({
							"method": method_name,
							"trial_key": trial_base,
							"path": c["path"],
							"filename": c["filename"],
							"suffix": int(c.get("trial_suffix", c.get("suffix", -1))),
							"mtime": float(c["mtime"]),
							"reason": "pre_epoch_stale",
						})
						consumed_paths.add(c["path"])
					missing.append({
						"method": method_name,
						"trial_key": trial_base,
						"reason": "all_candidates_pre_epoch",
					})
					continue

			chosen = None
			resolution_mode = "canonical_selected"
			manifest_entry = manifest_entries.get(trial_base)
			if isinstance(manifest_entry, dict):
				fname = str(manifest_entry.get("filename", "")).strip()
				if fname:
					matches = [c for c in eligible if str(c.get("filename", "")) == fname]
					if len(matches) == 1:
						chosen = matches[0]
						resolution_mode = "manifest"
					elif len(matches) > 1:
						ambiguities.append({
							"method": method_name,
							"trial_key": trial_base,
							"reason": "manifest_nonunique_match",
							"candidates": [str(c.get("filename", "")) for c in matches],
						})
					else:
						ambiguities.append({
							"method": method_name,
							"trial_key": trial_base,
							"reason": "manifest_target_not_eligible",
							"manifest_filename": fname,
							"eligible": [str(c.get("filename", "")) for c in eligible],
						})

			if chosen is None:
				if len(eligible) > 1:
					ambiguities.append({
						"method": method_name,
						"trial_key": trial_base,
						"reason": "ambiguous_candidates_no_manifest",
						"candidates": [str(c.get("filename", "")) for c in eligible],
					})
				chosen = _select_latest_candidate(eligible)
				resolution_mode = "canonical_selected"

			if chosen is None:
				missing.append({
					"method": method_name,
					"trial_key": trial_base,
					"reason": "selection_failed",
				})
				continue

			canonical[method_name][trial_base] = chosen["path"]
			selected_details[method_name][trial_base] = {
				"frame_log_filename_used": str(chosen["filename"]),
				"frame_log_suffix_used": int(chosen.get("trial_suffix", chosen.get("suffix", -1))),
				"frame_log_resolution_mode": resolution_mode,
				"mtime": float(chosen["mtime"]),
			}
			consumed_paths.add(chosen["path"])

			for c in cands:
				if c["path"] == chosen["path"]:
					continue
				reason = "duplicate_suffix_ignored"
				if cutoff is not None and float(c.get("mtime", 0.0)) < cutoff:
					reason = "pre_epoch_stale"
				extras.append({
					"method": method_name,
					"trial_key": trial_base,
					"path": c["path"],
					"filename": c["filename"],
					"suffix": int(c.get("trial_suffix", c.get("suffix", -1))),
					"mtime": float(c["mtime"]),
					"reason": reason,
				})
				consumed_paths.add(c["path"])

		for c in method_rows:
			if c["path"] in consumed_paths:
				continue
			reason = "unindexed_extra_log"
			if cutoff is not None and float(c.get("mtime", 0.0)) < cutoff:
				reason = "pre_epoch_stale"
			extras.append({
				"method": method_name,
				"trial_key": str(c.get("stem", "")),
				"path": c["path"],
				"filename": c["filename"],
				"suffix": int(c["suffix"]),
				"mtime": float(c["mtime"]),
				"reason": reason,
			})

	canonical = {m: t for m, t in canonical.items() if t}
	selected_details = {m: t for m, t in selected_details.items() if t}
	extras = sorted(
		extras,
		key=lambda r: (str(r.get("method", "")), str(r.get("trial_key", "")), str(r.get("filename", "")))
	)
	missing = sorted(
		missing,
		key=lambda r: (str(r.get("method", "")), str(r.get("trial_key", "")), str(r.get("reason", "")))
	)
	ambiguities = sorted(
		ambiguities,
		key=lambda r: (str(r.get("method", "")), str(r.get("trial_key", "")), str(r.get("reason", "")))
	)
	n_selected = int(sum(len(v) for v in canonical.values()))
	n_expected = int(sum(len(v) for v in method_expected.values()))
	diag = {
		"schema_version": FRAME_LOG_MANIFEST_SCHEMA,
		"generated_at": _utc_iso_now(),
		"run_dir": run_dir,
		"selection_policy": selection_policy,
		"run_instance_started_at_used": run_instance_started_at,
		"run_epoch_cutoff_unix": cutoff,
		"warnings": list(warnings),
		"counts": {
			"n_expected_trials": n_expected,
			"n_selected_trials": n_selected,
			"n_missing_trials": int(len(missing)),
			"n_extra_logs": int(len(extras)),
			"n_ambiguities": int(len(ambiguities)),
			"n_methods": int(len(set(list(canonical.keys()) + list(method_expected.keys())))),
		},
		"selected_details": selected_details,
		"ambiguities": ambiguities,
	}

	logs_dir = os.path.join(run_dir, "logs")
	os.makedirs(logs_dir, exist_ok=True)
	manifest_payload = {
		"schema_version": FRAME_LOG_MANIFEST_SCHEMA,
		"generated_at": diag["generated_at"],
		"run_dir": run_dir,
		"selection_policy": selection_policy,
		"run_instance_started_at_used": run_instance_started_at,
		"run_epoch_cutoff_unix": cutoff,
		"warnings": list(warnings),
		"selected": selected_details,
		"missing": missing,
		"orphans": [
			{
				**o,
				"path": os.path.relpath(str(o.get("path", "")), run_dir) if o.get("path") else "",
			}
			for o in extras
		],
		"ambiguities": ambiguities,
		"counts": diag["counts"],
	}
	_atomic_json_dump(manifest_payload, os.path.join(logs_dir, "frame_log_manifest.json"), indent=2)

	result = {
		"canonical": canonical,
		"extras": extras,
		"missing": missing,
		"diag": diag,
	}

	if strict:
		issues = []
		if ambiguities:
			issues.append(f"ambiguities={len(ambiguities)}")
		if extras:
			issues.append(f"extras_or_stale={len(extras)}")
		if (not allow_empty) and (n_expected > 0) and (n_selected == 0):
			issues.append("no_canonical_logs")
		if issues:
			raise ValueError(
				"Frame log resolution strict failure: "
				+ ", ".join(issues)
				+ ". Use a fresh run_dir or cleanup stale/suffix logs, or set strict=False for exploratory mode."
			)

	return result


def _filter_valid_rois(rois):
	filtered = []
	for roi in rois or []:
		arr = np.asarray(roi)
		if arr.size > 0:
			filtered.append(roi)
	return filtered


def _merge_results_payload(out_path, partial_results, method_order=None):
    if partial_results is None:
        return
    metadata_keys = ['video_path', 'fps', 'gt', 'fs_gt']
    new_estimates = list(partial_results.get('estimates', [])) if isinstance(partial_results, dict) else []
    lock_path = f"{out_path}.lock"
    with _file_lock(lock_path):
        if os.path.exists(out_path):
            with open(out_path, 'rb') as fp:
                data = pickle.load(fp)
        else:
            data = {}
        for key in metadata_keys:
            if key in partial_results:
                data[key] = partial_results[key]

        existing_estimates = list(data.get('estimates', []))
        # Keep only methods present in this run manifest to prevent stale
        # methods from previous runs leaking into current evaluation.
        if method_order:
            allowed = set(method_order)
            existing_estimates = [
                entry for entry in existing_estimates
                if isinstance(entry, dict) and entry.get('method') in allowed
            ]

        index_map = {}
        for idx, entry in enumerate(existing_estimates):
            if isinstance(entry, dict):
                name = entry.get('method')
                if name is not None and name not in index_map:
                    index_map[name] = idx

        for entry in new_estimates:
            if not isinstance(entry, dict):
                continue
            name = entry.get('method')
            if name is None:
                continue
            if name in index_map:
                existing_estimates[index_map[name]] = entry
            else:
                index_map[name] = len(existing_estimates)
                existing_estimates.append(entry)

        if method_order:
            order_index = {name: idx for idx, name in enumerate(method_order)}

            def _order_key(item):
                if isinstance(item, dict):
                    return order_index.get(item.get('method'), len(order_index))
                return len(order_index)

            existing_estimates.sort(key=_order_key)
        data['estimates'] = existing_estimates
        _atomic_pickle_dump(data, out_path)
    # Best-effort cleanup: keep lock file lifecycle short to avoid stale
    # *.pkl.lock artifacts after successful single-process runs.
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except OSError:
        pass
