#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "analysis" / "publication_figure_manifest.json"


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing figure manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_png(path: Path, width_class: str) -> list[str]:
    warnings: list[str] = []
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=float)
        w, h = im.size
    min_width = 1700 if width_class == "single" else 3200
    if w < min_width:
        warnings.append(f"tiny width {w}px")
    if h < 600:
        warnings.append(f"tiny height {h}px")
    if float(np.nanstd(arr)) < 2.0:
        warnings.append("near-blank image")
    white_frac = float(np.mean(arr > 248))
    if white_frac > 0.985:
        warnings.append(f"mostly blank ({white_frac:.3f} white)")
    return warnings


def verify(manifest_path: Path) -> int:
    entries = _load_manifest(manifest_path)
    missing: list[str] = []
    warnings: list[str] = []
    for entry in entries:
        fig_id = entry["figure_id"]
        width_class = entry.get("width_class", "supp")
        for kind, out in entry["outputs"].items():
            path = Path(out)
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists():
                missing.append(f"{fig_id}: missing {kind} {path}")
                continue
            if path.stat().st_size < 1024:
                warnings.append(f"{fig_id}: suspiciously small {kind} {path} ({path.stat().st_size} bytes)")
        png = Path(entry["outputs"]["png"])
        if not png.is_absolute():
            png = ROOT / png
        if png.exists():
            for warning in _check_png(png, width_class):
                warnings.append(f"{fig_id}: {warning} in {png}")
    print(f"Checked {len(entries)} figure entries from {manifest_path}")
    if missing:
        print("MISSING")
        for item in missing:
            print(f"  - {item}")
    if warnings:
        print("WARNINGS")
        for item in warnings:
            print(f"  - {item}")
    if not missing and not warnings:
        print("All expected figure files exist and passed basic raster checks.")
    return 1 if missing else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify regenerated publication figure outputs.")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    args = ap.parse_args()
    raise SystemExit(verify(args.manifest))


if __name__ == "__main__":
    main()
