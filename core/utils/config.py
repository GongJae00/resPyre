import json
import os
import copy

_DEFAULT_CFG = {
    "results_dir": "results",
    "datasets": ["COHFACE"],
    "methods": [
        "OF_Model",
        "DoF",
        "profile1D"
    ],
    "eval": {
        "win_size": "video",
        "stride": 1,
        "min_hz": 0.08,
        "max_hz": 0.5,
        "use_track": False,
        "track_std_min_bpm": 0.3,
        "track_unique_min": 0.05,
        "track_saturation_max": 0.15,
        "saturation_margin_hz": 0.0,
        "saturation_persist_sec": 0.0,
        "constant_ptp_max_hz": 0.0
    },
    "oscillator": {},
    "report": {
        "runs": [],
        "output": "reports/combined",
        "unique_window": False
    },
    "runtime": {},
    "steps": [],
    "gating": {}
}


def _deep_merge(base, update):
    if not isinstance(base, dict) or not isinstance(update, dict):
        return copy.deepcopy(update)
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_dataset(entry):
    if isinstance(entry, str):
        return {"name": entry, "roi": {}}
    norm = copy.deepcopy(entry)
    if "name" not in norm:
        raise ValueError("Dataset entry requires a 'name'")
    norm.setdefault("roi", {})
    return norm


def _normalize_method(entry):
    if isinstance(entry, str):
        return {"name": entry}
    if "name" not in entry:
        raise ValueError("Method entry requires a 'name'")
    return copy.deepcopy(entry)


def _resolve_path(base_dir, path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def load_config(path):
    if not path:
        cfg = copy.deepcopy(_DEFAULT_CFG)
        cfg['base_dir'] = os.getcwd()
        cfg['project_root'] = cfg['base_dir']
        cfg['results_dir'] = _resolve_path(cfg['project_root'], cfg['results_dir'])
        cfg['datasets'] = [_normalize_dataset(d) for d in cfg['datasets']]
        cfg['methods'] = [_normalize_method(m) for m in cfg['methods']]
        return cfg

    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path, 'r') as fp:
        raw = json.load(fp)

    cfg = _deep_merge(_DEFAULT_CFG, raw)
    cfg['base_dir'] = base_dir
    project_root = os.path.abspath(os.path.join(base_dir, os.pardir))
    cfg['project_root'] = project_root
    cfg['results_dir'] = _resolve_path(project_root, cfg.get('results_dir', 'results'))
    cfg['datasets'] = [_normalize_dataset(d) for d in cfg.get('datasets', [])]
    cfg['methods'] = [_normalize_method(m) for m in cfg.get('methods', [])]

    eval_cfg = cfg.setdefault('eval', {})
    eval_cfg.setdefault('win_size', 'video')
    eval_cfg.setdefault('stride', 1)
    eval_cfg.setdefault('min_hz', 0.08)
    eval_cfg.setdefault('max_hz', 0.5)
    eval_cfg.setdefault('use_track', False)
    eval_cfg.setdefault('track_std_min_bpm', 0.3)
    eval_cfg.setdefault('track_unique_min', 0.05)
    eval_cfg.setdefault('track_saturation_max', 0.15)
    eval_cfg.setdefault('saturation_margin_hz', 0.0)
    eval_cfg.setdefault('saturation_persist_sec', 0.0)
    eval_cfg.setdefault('constant_ptp_max_hz', 0.0)

    osc_cfg = cfg.get('oscillator', {})
    if not isinstance(osc_cfg, dict):
        osc_cfg = {}
    else:
        osc_cfg = copy.deepcopy(osc_cfg)
    osc_cfg.setdefault('fs', 64.0)
    osc_cfg.setdefault('f_min', eval_cfg['min_hz'])
    osc_cfg.setdefault('f_max', eval_cfg['max_hz'])
    osc_cfg.setdefault('tau_env', 30.0)
    osc_cfg.setdefault('qx_scale', 0.3)
    osc_cfg.setdefault('rv_auto', True)
    osc_cfg.setdefault('rv_mad_scale', 1.2)
    osc_cfg.setdefault('rv_floor', 0.08)
    osc_cfg.setdefault('qf', 5e-5)
    osc_cfg.setdefault('stft_win', 12.0)
    osc_cfg.setdefault('stft_hop', 1.0)
    osc_cfg.setdefault('ridge_penalty', 250.0)
    osc_cfg.setdefault('pll_autogain', True)
    osc_cfg.setdefault('pll_zeta', 0.9)
    osc_cfg.setdefault('pll_ttrack', 7.0)
    cfg['oscillator'] = osc_cfg

    report_cfg = cfg.get('report', {})
    runs = report_cfg.get('runs', [])
    if runs:
        report_cfg['runs'] = [
            _resolve_path(cfg['results_dir'], run) if not os.path.isabs(run) else run
            for run in runs
        ]
    cfg['report'] = report_cfg

    cfg['steps'] = [step.lower() for step in cfg.get('steps', [])]

    return cfg
