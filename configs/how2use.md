# Configs

`configs/` now keeps only the paper-facing examples that are still useful for
the public CLI. The final manuscript execution path is not driven by these
JSON files alone; use [`execute.md`](../execute.md) for the full paper package.

## Active Configs

- `cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json`
- `mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json`

These preserve the canonical observation-class order used by the paper:

```text
OF, OF_bridge, DoF, DoF_bridge, P1D_lin, P1D_quad, P1D_cub, P1D_cons
```

Each config groups methods as:

```text
Base observation classes -> OSSM-KF comparator -> PARH-OSSM
```

## Usage

For the current paper-facing run, prefer:

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
bash -lc "$(sed -n '/^## Environment/,/^## Boundary/p' execute.md)"
```

For a lightweight CLI wiring check:

```bash
python main.py --config configs/cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json --debug
python main.py --config configs/mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json --debug
```

## Policy

- Do not add dated sweep configs to this directory.
- Ablation and exploratory commands belong in scripts or analysis reports, not
  as promoted config files.
- If a new config is added, it must be referenced by a test, `execute.md`, or a
  paper-facing reproduction command.
