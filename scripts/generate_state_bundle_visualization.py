#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _z(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    s = float(np.std(x))
    if s < 1e-8:
        return x - float(np.mean(x))
    return (x - float(np.mean(x))) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--method', default='family_multiview_hypothesis_set__parh_resonator_adaptive')
    ap.add_argument('--out-pdf', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args()

    obj = pickle.load(open(args.pkl, 'rb'))
    fps = float(obj.get('fps', 1.0))
    gt = _z(obj.get('gt', np.array([], dtype=np.float64)))
    est = None
    for item in obj['estimates']:
        if item.get('method') == args.method:
            est = item['estimate']
            break
    if est is None:
        raise KeyError(f'method not found: {args.method}')

    sig = _z(est.get('signal_hat', np.array([], dtype=np.float64)))
    zfull = _z(est.get('z_full', np.array([], dtype=np.float64)))
    d = _z(est.get('canonical_bundle_d', np.array([], dtype=np.float64)))
    v = _z(est.get('canonical_bundle_v', np.array([], dtype=np.float64)))
    m = _z(est.get('canonical_bundle_m', np.array([], dtype=np.float64)))
    q = _z(est.get('canonical_bundle_q', np.array([], dtype=np.float64)))
    h1 = _z(est.get('latent_h1_drive', np.array([], dtype=np.float64)))
    h2 = _z(est.get('latent_h2_drive', np.array([], dtype=np.float64)))
    b = _z(est.get('latent_b_drive', np.array([], dtype=np.float64)))
    r = _z(est.get('latent_r_drive', np.array([], dtype=np.float64)))
    n = min(len(sig), len(zfull), len(d), len(v), len(m), len(q), len(h1), len(h2), len(b), len(r), len(gt) if gt.size else 10**9)
    if n <= 0:
        raise RuntimeError('empty series')
    t = np.arange(n, dtype=np.float64) / fps
    fam = json.loads(est.get('family_params_json', '{}'))
    boot = json.loads(est.get('shared_latent_bootstrap_json', '{}'))
    grow = json.loads(est.get('global_observation_row_json', '{}'))

    fig, axs = plt.subplots(3, 2, figsize=(14, 9), constrained_layout=True)
    axs = axs.reshape(3,2)

    axs[0,0].plot(t, gt[:n], label='GT', color='black', lw=1.5)
    axs[0,0].plot(t, sig[:n], label='signal_hat', color='#4c78a8', lw=1.1)
    axs[0,0].plot(t, zfull[:n], label='z_full', color='#54a24b', lw=1.1)
    axs[0,0].set_title('Observed vs reconstructed waveform')
    axs[0,0].legend(frameon=False, fontsize=8)

    axs[0,1].plot(t, d[:n], label='d', lw=1.0)
    axs[0,1].plot(t, v[:n], label='v', lw=1.0)
    axs[0,1].plot(t, m[:n], label='m', lw=1.0)
    axs[0,1].plot(t, q[:n], label='q', lw=1.0)
    axs[0,1].set_title('Canonical observation bundle')
    axs[0,1].legend(frameon=False, fontsize=8, ncol=2)

    axs[1,0].plot(t, h1[:n], label='h1_drive', lw=1.0)
    axs[1,0].plot(t, h2[:n], label='h2_drive', lw=1.0)
    axs[1,0].plot(t, b[:n], label='b_drive', lw=1.0)
    axs[1,0].plot(t, r[:n], label='r_drive', lw=1.0)
    axs[1,0].set_title('Latent bootstrap drives')
    axs[1,0].legend(frameon=False, fontsize=8, ncol=2)

    fam_names = list(fam.keys())
    rel = [float(fam[k].get('reliability', np.nan)) for k in fam_names]
    nuis = [float(fam[k].get('nuisance_weight', np.nan)) for k in fam_names]
    x = np.arange(len(fam_names))
    axs[1,1].bar(x - 0.18, rel, width=0.36, label='reliability')
    axs[1,1].bar(x + 0.18, nuis, width=0.36, label='nuisance')
    axs[1,1].set_xticks(x)
    axs[1,1].set_xticklabels(fam_names)
    axs[1,1].set_title('Class-local adaptor parameters')
    axs[1,1].legend(frameon=False, fontsize=8)

    row_names = ['gain_d','gain_v','gain_m','gain_q','reliability','nuisance_weight','R_scale']
    row_vals = [float(grow.get(k, np.nan)) for k in row_names]
    axs[2,0].bar(np.arange(len(row_names)), row_vals, color='#f58518')
    axs[2,0].set_xticks(np.arange(len(row_names)))
    axs[2,0].set_xticklabels(row_names, rotation=20, ha='right')
    axs[2,0].set_title('Global observation row')

    mix = boot.get('latent_component_mix', {})
    comp_names = ['h1_drive','h2_drive','b_drive','r_drive']
    fam_order = ['d','v','m','q']
    mat = np.array([[float(mix.get(c, {}).get(f, np.nan)) for f in fam_order] for c in comp_names], dtype=float)
    im = axs[2,1].imshow(mat, cmap='viridis', aspect='auto')
    axs[2,1].set_xticks(np.arange(len(fam_order)))
    axs[2,1].set_xticklabels(fam_order)
    axs[2,1].set_yticks(np.arange(len(comp_names)))
    axs[2,1].set_yticklabels(comp_names)
    axs[2,1].set_title('Bootstrap latent mix matrix')
    fig.colorbar(im, ax=axs[2,1], fraction=0.046, pad=0.04)

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200)
    plt.close(fig)

    lines = [
        '# State / Canonical Bundle Visualization',
        '',
        f'- Source PKL: `{args.pkl}`',
        f'- Method: `{args.method}`',
        '',
        'Panels:',
        '- observed/reconstructed waveform',
        '- canonical observation bundle `{d,v,m,q}`',
        '- latent bootstrap drives `{h1,h2,b,r}`',
        '- observation-class-local reliability/nuisance weights',
        '- global observation row',
        '- bootstrap latent mix matrix',
        '',
        'This figure is for mechanistic debugging and release explanatory follow-up; it is not a performance figure.',
        '',
    ]
    Path(args.out_md).write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
