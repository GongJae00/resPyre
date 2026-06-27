from __future__ import annotations

import io
import json
import os
import pickle
import zipfile
from pathlib import Path

import pandas as pd


DEFAULT_RESULTS_ZIP = Path(
    os.environ.get("RESPYRE_RESULTS_ZIP", "results_20260613.zip")
)


class ArtifactStore:
    def __init__(self, zip_path: str | Path = DEFAULT_RESULTS_ZIP):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Missing results artifact zip: {self.zip_path}")
        self._zip = zipfile.ZipFile(self.zip_path)
        self._names = set(self._zip.namelist())

    def close(self) -> None:
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def exists(self, member: str) -> bool:
        return member in self._names

    def require(self, member: str) -> str:
        if member not in self._names:
            raise FileNotFoundError(f"Missing required artifact in {self.zip_path}: {member}")
        return member

    def read_bytes(self, member: str) -> bytes:
        return self._zip.read(self.require(member))

    def read_csv(self, member: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read_bytes(member)))

    def read_json(self, member: str) -> dict:
        return json.loads(self.read_bytes(member).decode("utf-8"))

    def read_pickle(self, member: str):
        return pickle.loads(self.read_bytes(member))

    def list_matching(self, text: str) -> list[str]:
        return sorted(name for name in self._names if text in name)


def member_join(prefix: str, *parts: str) -> str:
    return "/".join([prefix.rstrip("/"), *[p.strip("/") for p in parts]])
