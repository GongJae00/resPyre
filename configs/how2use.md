# Configs

`configs/` keeps the stable CLI examples. Use [`execute.md`](../execute.md) for
the full reproduction workflow.

## Active Configs

- `cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json`
- `mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json`

These preserve the canonical observation-class order used by the PARH-OSSM runs:

```text
OF, OF_bridge, DoF, DoF_bridge, P1D_lin, P1D_quad, P1D_cub, P1D_cons
```

Each config groups methods as:

```text
Base observation classes -> OSSM-KF comparator -> PARH-OSSM
```

## Usage

For the full workflow, prefer:

```bash
less execute.md
```

For a lightweight CLI wiring check:

```bash
python main.py --config configs/cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json --debug
python main.py --config configs/mahnob_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json --debug
```

## Policy

- Do not add dated sweep configs to this directory.
- Ablation and exploratory commands belong in scripts or generated analysis
  outputs, not as promoted config files.
- If a new config is added, it must be referenced by a test, `execute.md`, or a
  documented reproduction command.
