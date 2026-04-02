
import os
import argparse
import sys
import traceback
import json
from collections import Counter
from datetime import datetime

# Ensure src is in python path
# Ensure root is in python path if needed (usually defaults to cwd)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from core.utils.config import load_config
from components.datasets.impl import BP4D, COHFACE, MAHNOB
from components.observations.methods import OF_Model, DoF_Model, profile1D_Model
from core.pipeline.wrapped_method import create_wrapped_method
from core.pipeline.runner import extract_respiration


def _normalize_method_name(name: str) -> str:
    return str(name).strip().lower().replace(' ', '_')


def _method_family_key(name: str):
    n = _normalize_method_name(name)
    base = n.split('__', 1)[0] if '__' in n else n

    if base in ('of_model', 'of', 'of_farneback'):
        return 10
    if base == 'dof':
        return 20
    if base.startswith('profile1d_linear'):
        return 30
    if base.startswith('profile1d_quadratic'):
        return 40
    if base.startswith('profile1d_cubic'):
        return 50
    return 99


def _method_variant_key(name: str):
    n = _normalize_method_name(name)
    if '__' not in n:
        return 0  # base signal first
    head = n.split('__', 1)[1]

    if 'kfstd' in head:
        return 10
    if 'robust_ossm_ekf' in head or head.endswith('_ekf'):
        return 20
    if 'robust_ossm_ukf' in head or 'ukffreq' in head or head.endswith('_ukf'):
        return 30
    if 'agakf' in head:
        return 40
    return 90


def _sort_methods_for_execution(methods):
    return sorted(
        methods,
        key=lambda m: (
            _method_family_key(getattr(m, 'name', str(m))),
            _method_variant_key(getattr(m, 'name', str(m))),
            _normalize_method_name(getattr(m, 'name', str(m))),
        ),
    )


def _warn_duplicate_methods(methods):
    names = [getattr(m, 'name', str(m)) for m in methods]
    dup = {k: v for k, v in Counter(names).items() if v > 1}
    if dup:
        items = ", ".join(f"{k} (x{v})" for k, v in sorted(dup.items()))
        raise ValueError(
            "Duplicate method names are not allowed (strict-unique mode). "
            f"Found: {items}. "
            "Rename duplicated methods in config to avoid result overwrite."
        )

def _build_datasets(dataset_configs):
    datasets = []
    for d_cfg in dataset_configs:
        name = d_cfg['name'].lower()
        if name == 'bp4d':
            ds = BP4D()
        elif name == 'cohface':
            ds = COHFACE()
        elif name == 'mahnob':
            ds = MAHNOB()
        else:
            raise ValueError(f"Unknown dataset: {name}")
        ds.configure(d_cfg)
        # Subset filtering: load all then keep only matching trials
        subset = d_cfg.get('subset')
        if subset:
            ds.load_dataset()
            subset_set = set(str(s) for s in subset)
            filtered = []
            for d in ds.data:
                if 'trial' in d:
                    key = f"{d['subject']}_{d['trial']}"
                else:
                    key = str(d['subject'])
                if key in subset_set:
                    filtered.append(d)
            print(f"  Subset filter: {len(ds.data)} → {len(filtered)} trials (requested: {subset})")
            ds.data = filtered
        datasets.append(ds)
    return datasets

def _build_methods(method_configs, global_cfg=None):
    methods = []
    for entry in method_configs:
        if isinstance(entry, str):
            name = entry
            params = {}
        else:
            name = entry['name']
            params = entry

        if '__' in name:
            preproc = global_cfg.get('preproc', {}) if global_cfg else {}
            # Merge method-specific overrides
            methods.append(
                create_wrapped_method(
                    name,
                    params=params,
                    oscillator_defaults=(global_cfg.get('oscillator', {}) if global_cfg else None),
                    preproc_defaults=preproc,
                    gating_defaults=(global_cfg.get('gating', {}) if global_cfg else None),
                    quality_defaults=(global_cfg.get('quality', {}) if global_cfg else None),
                    trust_defaults=(global_cfg.get('trust', {}) if global_cfg else None),
                    gating_scope_default=((global_cfg or {}).get('gating_scope', 'evaluation_only')),
                )
            )
        # Base methods
        elif name.lower() in ('of_model', 'of_farneback', 'of'):
            methods.append(OF_Model())
        elif name.lower() == 'dof':
            methods.append(DoF_Model())
        elif name.lower().startswith('profile1d'):
            if ' ' in name:
                interp = name.split(' ')[1]
            elif '_' in name:
                interp = name.split('_')[1]
            else:
                interp = 'quadratic'
            methods.append(profile1D_Model(interp))
        else:
            print(f"Warning: Unknown method {name}")
    return methods


def _top_level_config_usage(cfg: dict) -> dict:
    used = {
        "name", "results_dir", "datasets", "methods",
        "eval", "oscillator", "preproc", "quality", "trust",
        "gating", "gating_scope", "steps",
        "config",
    }
    seen = set(cfg.keys())
    return {
        "schema_version": "config_usage.v1",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "strict_key_usage": bool((cfg.get('config') or {}).get('strict_key_usage', False)),
        "top_level_used_keys": sorted(used),
        "top_level_unused_keys": sorted(k for k in seen if k not in used),
    }

def main():
    parser = argparse.ArgumentParser(description="ResPyre Pipeline")
    parser.add_argument('--config', '-c', required=True, help='Path to config JSON')
    parser.add_argument('--results', '-r', help='Results directory override')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (max 1 sample)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = args.results or cfg.get('results_dir', 'results')

    datasets = _build_datasets(cfg['datasets'])
    methods = _build_methods(cfg['methods'], cfg)
    methods = _sort_methods_for_execution(methods)
    _warn_duplicate_methods(methods)

    if args.debug:
        print("> DEBUG mode: Limiting to 1 sample per dataset")
        for ds in datasets:
             # Force load if not loaded
             if not hasattr(ds, 'data') or not ds.data:
                 ds.load_dataset()
             ds.data = ds.data[:1]

    print(f"Loaded {len(datasets)} datasets and {len(methods)} methods.")
    
    steps = [s.lower() for s in cfg.get('steps', [])]
    run_estimate = (not steps) or any(s in steps for s in ('estimate', 'extract'))
    run_evaluate = (not steps) or any(s in steps for s in ('evaluate', 'metrics'))
    run_eda_step = 'eda' in steps
    run_visualize_step = 'visualize' in steps
    run_metadata_step = 'metadata' in steps

    eval_cfg = cfg.get('eval', {})
    common_eval_params = {
        'win_size': eval_cfg.get('win_size', 30.0),
        'stride': eval_cfg.get('stride', 1.0),
        'min_hz': eval_cfg.get('min_hz', 0.08),
        'max_hz': eval_cfg.get('max_hz', 0.5),
    }
    eval_params = {
        **common_eval_params,
        # Gating config is persisted to evaluation metadata and also forwarded
        # to wrapped methods for optional deterministic filter-time overrides.
        'gating': cfg.get('gating', {}),
        'gating_scope': cfg.get('gating_scope', 'evaluation_only'),
        'eval_cfg': eval_cfg,
        'strict_key_usage': bool((cfg.get('config') or {}).get('strict_key_usage', False)),
        'frame_log_strict': bool(eval_cfg.get('frame_log_strict', True)),
    }

    cmd_str = f"python main.py --config {args.config}"
    planned_steps = []
    if run_estimate:
        planned_steps.append("estimate")
    if run_evaluate:
        planned_steps.append("evaluate")
    if run_eda_step:
        planned_steps.append("eda")
    if run_visualize_step:
        planned_steps.append("visualize")
    if run_metadata_step:
        planned_steps.append("metadata")

    from core.pipeline.metadata_step import (
        bootstrap_run_dirs,
        mark_run_status_bulk,
        run_metadata_generation,
    )
    planned_run_dirs = bootstrap_run_dirs(
        results_dir,
        cfg.get('name'),
        dataset_names=[getattr(ds, 'name', '') for ds in datasets],
    )
    cfg_usage = _top_level_config_usage(cfg)
    for run_dir in planned_run_dirs:
        logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, "config_usage.json"), "w", encoding="utf-8") as fp:
            json.dump(cfg_usage, fp, ensure_ascii=False, indent=2)
    mark_run_status_bulk(
        planned_run_dirs,
        status="running",
        command=cmd_str,
        config_path=args.config,
        steps=planned_steps,
        completed_steps=[],
    )

    completed_steps = []
    pipeline_error = None
    error_trace = ""
    try:
        if run_estimate:
            extract_respiration(
                datasets=datasets,
                methods=methods,
                results_dir=results_dir,
                run_label=cfg.get('name')
            )
            completed_steps.append("estimate")

        if run_evaluate:
            from core.pipeline.evaluation_step import run_evaluation
            run_evaluation(results_dir, cfg.get('name'), **eval_params)
            completed_steps.append("evaluate")

        if run_eda_step:
            from analysis.run_innovation_eda import run_innovation_eda
            run_innovation_eda(
                results_dir,
                cfg.get('name'),
                allow_missing=bool(eval_cfg.get('allow_missing', False)),
                strict=bool(eval_cfg.get('frame_log_strict', True)),
            )
            completed_steps.append("eda")

        if run_visualize_step:
            from core.pipeline.visualize_step import run_visualization
            run_visualization(
                results_dir,
                cfg.get('name'),
                **common_eval_params,
                frame_log_strict=bool(eval_cfg.get('frame_log_strict', True)),
            )
            completed_steps.append("visualize")
    except Exception as exc:
        pipeline_error = exc
        error_trace = traceback.format_exc()
    finally:
        if "metadata" not in completed_steps:
            completed_steps.append("metadata")
        status = "failed" if pipeline_error else "completed"
        err_summary = str(pipeline_error) if pipeline_error else ""
        mark_run_status_bulk(
            planned_run_dirs,
            status=status,
            command=cmd_str,
            config_path=args.config,
            steps=planned_steps,
            completed_steps=completed_steps,
            error_summary=err_summary,
            error_traceback=error_trace,
        )
        # Always emit metadata (full or minimal) for reproducibility.
        run_metadata_generation(
            results_dir,
            cfg.get('name'),
            command=cmd_str,
            notes="auto-generated",
            allow_incomplete=True,
        )

    if pipeline_error is not None:
        raise pipeline_error

if __name__ == "__main__":
    main()
