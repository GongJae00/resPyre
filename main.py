
import os
import argparse
import sys

# Ensure src is in python path
# Ensure root is in python path if needed (usually defaults to cwd)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from core.utils.config import load_config
from components.datasets.impl import BP4D, COHFACE, MAHNOB
from components.observations.methods import OF_Model, DoF_Model, profile1D_Model
from core.pipeline.wrapped_method import create_wrapped_method
from core.pipeline.runner import extract_respiration

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

        # Base methods
        if name.lower() in ('of_model', 'of_farneback'):
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
        # Oscillator wrapped methods
        elif '__' in name:
            preproc = global_cfg.get('oscillator', {}) if global_cfg else {}
            # Merge method-specific overrides
            methods.append(create_wrapped_method(name, params=params, preproc_defaults=preproc))
        else:
            print(f"Warning: Unknown method {name}")
    return methods

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

    if args.debug:
        print("> DEBUG mode: Limiting to 1 sample per dataset")
        for ds in datasets:
             # Force load if not loaded
             if not hasattr(ds, 'data') or not ds.data:
                 ds.load_dataset()
             ds.data = ds.data[:1]

    print(f"Loaded {len(datasets)} datasets and {len(methods)} methods.")
    
    extract_respiration(
        datasets=datasets,
        methods=methods,
        results_dir=results_dir,
        run_label=cfg.get('name')
    )
    
    from core.pipeline.evaluation_step import run_evaluation
    run_evaluation(results_dir, cfg.get('name'))

    if 'visualize' in cfg.get('steps', []):
        from core.pipeline.visualize_step import run_visualization
        run_visualization(results_dir, cfg.get('name'))

    if 'metadata' in cfg.get('steps', []):
        from core.pipeline.metadata_step import run_metadata_generation
        # Construct a pseudo command string
        cmd_str = f"python main.py --config {args.config}"
        run_metadata_generation(results_dir, cfg.get('name'), command=cmd_str)

if __name__ == "__main__":
    main()
