"""Helpers shared by the bronze, silver and gold tiers.

Nothing in here decides anything about the measurement.  It is the small
machinery every tier needs - logging, provenance, time-base alignment, grid
arithmetic, table writing - collected in one place so that the tier modules
contain only their own step of the analysis.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eis.sync.resample import (
    advance_affine, fractional_delay_fft, integer_and_fractional,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def make_logger(verbose: bool = True, prefix: str = "") -> Callable[..., None]:
    """A ``print``-alike that can be silenced without ``if verbose`` everywhere."""
    if not verbose:
        return lambda *a, **k: None
    if not prefix:
        return print

    def log(*args: Any, **kwargs: Any) -> None:
        print(prefix, *args, **kwargs)

    return log


def banner(title: str, width: int = 72) -> str:
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def git_sha(default: str = "unknown") -> str:
    """Short SHA of the working tree, or ``default`` outside a checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except Exception:
        return default


def provenance(cfg, version: str) -> dict[str, str]:
    """The columns that let a stored row be traced back to its computation."""
    return {
        "pipeline_version": version,
        "param_hash": cfg.param_hash,
        "git_sha": git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "channel_delay_table": cfg.acquisition.channel_delay_table_version,
    }


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def ac_rms(x: np.ndarray) -> float:
    """RMS of the AC part - the DC operating point carries no timing."""
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean((x - x.mean()) ** 2)))


def align_channel(
    x: np.ndarray, fs: float, tau_s: float, delay_slope_ppm: float = 0.0
) -> np.ndarray:
    """Bring ``x`` onto the reference time base.

    ``tau_s`` is the measured delay of this channel (positive = it lags), so
    the channel has to be advanced by that amount.  A constant offset is
    applied as an exact integer shift plus an FFT phase ramp; a drifting one
    goes through the polyphase resampler, which handles offset and rate in a
    single band-limited interpolation.
    """
    if abs(delay_slope_ppm) > 1e-6:
        return advance_affine(x, fs, tau_s, delay_slope_ppm)
    whole, frac = integer_and_fractional(tau_s, fs)
    out = np.asarray(x, float)
    if whole:                                   # advance == take later samples
        out = np.roll(out, -whole)
        if whole > 0:
            out[-whole:] = out[-whole - 1]
        else:
            out[:-whole] = out[-whole]
    if abs(frac) > 1e-6:
        out = fractional_delay_fft(out, -frac / fs, fs)
    return out


# ---------------------------------------------------------------------------
# Frequency-band arithmetic
# ---------------------------------------------------------------------------

def nearest_index(values: np.ndarray, target: float) -> int:
    """Index of the entry closest to ``target``; ``-1`` for an empty array."""
    values = np.asarray(values, float)
    if values.size == 0:
        return -1
    return int(np.argmin(np.abs(values - target)))


def common_grid(grids: Iterable[np.ndarray], decimals: int = 6) -> np.ndarray:
    """Frequencies present in every one of ``grids``, sorted ascending."""
    shared: set[float] | None = None
    for g in grids:
        keys = set(np.round(np.asarray(g, float), decimals).tolist())
        shared = keys if shared is None else (shared & keys)
    if not shared:
        return np.array([], float)
    return np.array(sorted(shared), float)


# ---------------------------------------------------------------------------
# Small numeric utilities
# ---------------------------------------------------------------------------

def clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def safe_median(values: Iterable[float], default: float = float("nan")) -> float:
    array = np.asarray([v for v in values if np.isfinite(v)], float)
    return float(np.median(array)) if array.size else default


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write Parquet, falling back to CSV when ``pyarrow`` is unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception:
        csv = path.with_suffix(".csv")
        frame.to_csv(csv, index=False)
        return csv
