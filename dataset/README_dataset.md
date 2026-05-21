# Dataset Layout

This repository does not ship raw datasets. The code expects each dataset to be
available under `dataset/` by default, usually as local symlinks, or under the
directory pointed to by `RESPIRE_DATA_DIR`.

Expected local structure:

```text
resPyre/
├── dataset/
│   ├── COHFACE/
│   │   └── <subject>/<trial>/data.avi + data.hdf5
│   ├── MAHNOB/
│   │   └── <subject>/<trial video>.avi + matching BDF/auxiliary files
│   ├── V4V/
│   │   └── Phase_1_Training_Validation_sets/
│   │       ├── Videos/{train,valid}/<trial>.mkv
│   │       └── Ground_truth/Physiology/<trial>.txt
│   ├── SCAMPS/
│   │   └── scamps_videos/P*.mat
│   └── BP4Ddef/              # optional legacy BP4D layout
└── main.py
```

## Dataset Roles

- `COHFACE`: real waveform and rate benchmark.
- `MAHNOB`: real hard-regime waveform and rate benchmark; BDF labels are read
  with `pyEDFlib`.
- `V4V`: external real RR-rate-only evidence. Do not use it for waveform,
  morphology, CCC, DTW, strict waveform, or cycle-shape claims.
- `SCAMPS`: synthetic diagnostic/control evidence only. Do not mix it with
  real-data headline performance claims.
- `BP4Ddef`: optional legacy layout used by older code paths.

## Local Symlinks

Create symlinks from your local raw-data storage into `dataset/`. Example:

```bash
ln -s /path/to/cohface dataset/COHFACE
ln -s /path/to/MAHNOB_HCI dataset/MAHNOB
ln -s /path/to/V4V dataset/V4V
ln -s /path/to/SCAMPS dataset/SCAMPS
```

The paper-facing helper can also create the expected local links when its
default paths match your machine:

```bash
python scripts/build_rr_experiment_assets.py --create-symlinks
```

By default, that helper assumes `/mnt/hdd18t/rppg_dataset/raw`. Override it with:

```bash
export RESPYRE_RAW_DATA_ROOT=/absolute/path/to/raw_dataset_root
python scripts/build_rr_experiment_assets.py --create-symlinks
```

For a completely different dataset root, set:

```bash
export RESPIRE_DATA_DIR=/absolute/path/to/dataset_root
python main.py --config configs/cohface_parh_ossm_prod_ofbridge_dofbridge_p1dcons.json
```

## Path Customization

Dataset discovery lives in `components/datasets/impl.py`, with the shared root
logic in `components/datasets/base.py`. If a dataset has a different folder
depth or filename convention, update the corresponding dataset implementation
rather than committing local raw-data paths or symlinks.
