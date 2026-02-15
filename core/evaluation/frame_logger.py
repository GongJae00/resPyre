"""
Per-frame audit logger for robust Bayesian filter.
Spec §11.5 — records every filter step for offline diagnostics.

Usage:
    logger = FrameLogger(n_frames=T)
    for t in range(T):
        logger.log_frame(t, ...)
    df = logger.to_dataframe()
    logger.save("path/to/log.parquet")
"""

import os
import numpy as np
from typing import Dict, Optional

# Schema: ordered list of field names logged per frame.
FRAME_SCHEMA = [
    # --- Time index ---
    't',
    # --- Raw state ---
    'x1', 'x2', 'z',
    # --- Decoded state ---
    'amp', 'phase_rad', 'freq_hz',
    # --- Observation ---
    'y_t', 'y_pred', 'v_t',
    # --- Robust filter outputs ---
    'nis', 'lambda_t',
    # --- Trust allocation ---
    'alpha_R', 'alpha_Q', 'g_t', 'g_z', 'w_h',
    # --- Covariance health ---
    'trace_P', 'det_P',
    # --- Failure flags ---
    'fail_diverge', 'fail_slip', 'fail_lock', 'fail_double',
]


class FrameLogger:
    """Accumulates per-frame filter diagnostics into a structured array.

    Designed for deterministic replay:
        Given the same (y, fs, config), the log must be identical.
    """

    def __init__(self, n_frames: int, extra_fields: Optional[list] = None):
        self.fields = list(FRAME_SCHEMA)
        if extra_fields:
            for f in extra_fields:
                if f not in self.fields:
                    self.fields.append(f)
        self.n_frames = n_frames
        self.n_fields = len(self.fields)
        self._data = np.full((n_frames, self.n_fields), np.nan, dtype=np.float64)
        self._field_idx = {name: i for i, name in enumerate(self.fields)}

    def log_frame(self, t: int, **kwargs):
        """Log a single frame. Keys must match schema fields."""
        if t < 0 or t >= self.n_frames:
            return
        self._data[t, self._field_idx['t']] = float(t)
        for key, val in kwargs.items():
            idx = self._field_idx.get(key)
            if idx is not None:
                self._data[t, idx] = float(val) if np.isfinite(float(val)) else np.nan

    def log_state(self, t: int, x: np.ndarray, P: np.ndarray,
                  y_t: float, y_pred: float, v_t: float,
                  nis: float, lambda_t: float = 1.0):
        """Convenience: log core filter state at once."""
        amp = float(np.sqrt(x[0]**2 + x[1]**2)) if x.size >= 2 else np.nan
        phase = float(np.arctan2(x[1], x[0])) if x.size >= 2 else np.nan
        freq = float(np.exp(x[2])) if x.size >= 3 else np.nan
        trace_P = float(np.trace(P))
        det_P = float(np.linalg.det(P)) if P.shape[0] <= 4 else np.nan

        self.log_frame(t,
                       x1=x[0], x2=x[1],
                       z=x[2] if x.size >= 3 else np.nan,
                       amp=amp, phase_rad=phase, freq_hz=freq,
                       y_t=y_t, y_pred=y_pred, v_t=v_t,
                       nis=nis, lambda_t=lambda_t,
                       trace_P=trace_P, det_P=det_P)

    def log_trust(self, t: int, alpha_R: float = 1.0, alpha_Q: float = 1.0,
                  g_t: float = 1.0, g_z: float = 1.0, w_h: float = 1.0):
        """Log trust allocation parameters."""
        self.log_frame(t, alpha_R=alpha_R, alpha_Q=alpha_Q,
                       g_t=g_t, g_z=g_z, w_h=w_h)

    def log_failure(self, t: int, diverge: bool = False, slip: bool = False,
                    lock: bool = False, double: bool = False):
        """Log failure mode flags."""
        self.log_frame(t,
                       fail_diverge=float(diverge),
                       fail_slip=float(slip),
                       fail_lock=float(lock),
                       fail_double=float(double))

    def get_array(self) -> np.ndarray:
        """Return raw (T, n_fields) array."""
        return self._data

    def get_column(self, name: str) -> np.ndarray:
        """Return a single column by name."""
        idx = self._field_idx.get(name)
        if idx is None:
            raise KeyError(f"Unknown field: {name}")
        return self._data[:, idx]

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Return dict of {field_name: 1d array}."""
        return {name: self._data[:, i] for name, i in self._field_idx.items()}

    def to_dataframe(self):
        """Return pandas DataFrame (import-on-demand)."""
        import pandas as pd
        return pd.DataFrame(self._data, columns=self.fields)

    def save(self, path: str, fmt: str = "npz"):
        """Save to disk. Supports 'npz' and 'parquet'."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if fmt == "parquet":
            df = self.to_dataframe()
            df.to_parquet(path, index=False)
        else:
            np.savez_compressed(path, data=self._data, fields=self.fields)

    @classmethod
    def load(cls, path: str) -> "FrameLogger":
        """Load from .npz file."""
        loaded = np.load(path, allow_pickle=True)
        data = loaded['data']
        fields = list(loaded['fields'])
        n_frames = data.shape[0]
        logger = cls(n_frames)
        logger.fields = fields
        logger._field_idx = {name: i for i, name in enumerate(fields)}
        logger.n_fields = len(fields)
        logger._data = data
        return logger
