## Setup notes

- `setup.sh` installs a minimal stack (numpy/scipy/pandas/h5py via conda, rest via pip) for the motion+oscillator pipeline.
- No deep/rPPG/vendor packages are installed.
- Use `--verify` to run a light import check after install:
  ```bash
  ./setup/setup.sh --verify
  ```
