"""Physical calibration: shunt voltage -> local current density.

Each segment carries a via-shunt whose transfer function depends on the local
temperature through the resistivity of the copper vias:

    H(T) = c0 + 1e-3 * c1 * T          [V * cm^2 / A]
    j    = V_shunt / H(T)              [A / cm^2]
    I    = j * A_segment               [A]

The temperature dependence is not cosmetic: copper has a temperature
coefficient near 0.39 %/K, so evaluating H at a fixed nominal temperature when
the plate is 10 K away puts a ~4 % error straight onto every resistance - the
same order as the segment-to-segment variation the measurement is meant to
resolve.  This module therefore evaluates H at the temperature measured for
each segment, and records explicitly when it had to fall back to a constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ShuntCalibration:
    """Per-segment via-shunt coefficients ``H(T) = c0 + 1e-3*c1*T``."""

    c0: np.ndarray
    c1: np.ndarray
    source: str = ""

    @property
    def n_segments(self) -> int:
        return len(self.c0)

    def H(self, segment: int, temperature_c: float) -> float:
        """Transfer function [V*cm^2/A] of ``segment`` (1-based) at ``T``."""
        idx = int(segment) - 1
        if idx < 0 or idx >= self.n_segments:
            raise KeyError(
                f"segment {segment} has no shunt calibration "
                f"(file covers 1..{self.n_segments})"
            )
        return float(self.c0[idx] + 1e-3 * self.c1[idx] * temperature_c)

    def has(self, segment: int) -> bool:
        idx = int(segment) - 1
        return 0 <= idx < self.n_segments and np.isfinite(self.c0[idx])


def load_shunt_calibration(path: str | Path) -> ShuntCalibration:
    """Read the ``;``-separated, header-less ``c0;c1`` calibration file."""
    df = pd.read_csv(path, sep=";", header=None, names=["c0", "c1"])
    return ShuntCalibration(
        c0=df["c0"].to_numpy(float),
        c1=df["c1"].to_numpy(float),
        source=str(path),
    )


@dataclass
class TemperatureCalibration:
    """Linear sensor calibration ``T = (V - c0) / c1``."""

    c0: np.ndarray
    c1: np.ndarray
    source: str = ""

    def to_celsius(self, index: int, volts: float) -> float:
        return float((volts - self.c0[index]) / self.c1[index])


def load_temperature_calibration(path: str | Path) -> TemperatureCalibration:
    df = pd.read_csv(path, sep=";", header=None, names=["c0", "c1"])
    return TemperatureCalibration(
        c0=df["c0"].to_numpy(float), c1=df["c1"].to_numpy(float), source=str(path)
    )


# ---------------------------------------------------------------------------
# Temperature field
# ---------------------------------------------------------------------------

@dataclass
class TemperatureField:
    """Temperature at each segment, interpolated from the sparse sensors."""

    per_segment_c: dict[int, float] = field(default_factory=dict)
    sensor_values_c: dict[str, float] = field(default_factory=dict)
    fallback_c: float = 58.4
    degraded: bool = False
    note: str = ""

    def at(self, segment: int) -> float:
        return self.per_segment_c.get(int(segment), self.fallback_c)

    @property
    def mean_c(self) -> float:
        if self.per_segment_c:
            return float(np.mean(list(self.per_segment_c.values())))
        return self.fallback_c


def build_temperature_field(
    sensor_readings_c: dict[str, float],
    sensor_coords: dict[str, tuple[float, float]] | None,
    segment_coords: dict[int, tuple[float, float, float, float]],
    segments: list[int],
    fallback_c: float = 58.4,
    valid_range_c: tuple[float, float] = (0.0, 200.0),
) -> TemperatureField:
    """Interpolate sensor temperatures onto the segments.

    With coordinates for the sensors an RBF field is built; without them the
    sensors are averaged.  Either way, readings outside the plausible range are
    discarded and the fallback is used *and flagged*, never silently applied.
    """
    usable = {
        name: value for name, value in sensor_readings_c.items()
        if np.isfinite(value) and valid_range_c[0] <= value <= valid_range_c[1]
    }
    if not usable:
        return TemperatureField(
            per_segment_c={s: fallback_c for s in segments},
            sensor_values_c=sensor_readings_c, fallback_c=fallback_c,
            degraded=True,
            note=(
                "no temperature sensor gave a plausible reading "
                f"(range {valid_range_c} degC); H(T) evaluated at the "
                f"{fallback_c} degC fallback for every segment"
            ),
        )

    if sensor_coords and len(usable) >= 3 and segment_coords:
        known = [(sensor_coords[n], v) for n, v in usable.items() if n in sensor_coords]
        if len(known) >= 3:
            from scipy.interpolate import RBFInterpolator

            pts = np.array([p for p, _ in known], float)
            vals = np.array([v for _, v in known], float)
            targets = [s for s in segments if s in segment_coords]
            grid = np.array([segment_coords[s][:2] for s in targets], float)
            kernel = "thin_plate_spline" if len(known) >= 4 else "linear"
            rbf = RBFInterpolator(pts, vals, kernel=kernel, smoothing=1e-3)
            interp = rbf(grid)
            lo, hi = vals.min() - 15.0, vals.max() + 15.0
            interp = np.clip(interp, lo, hi)     # never extrapolate wildly
            per_segment = {s: float(t) for s, t in zip(targets, interp)}
            for s in segments:
                per_segment.setdefault(s, float(vals.mean()))
            return TemperatureField(
                per_segment_c=per_segment, sensor_values_c=usable,
                fallback_c=fallback_c, degraded=False,
                note=f"RBF field from {len(known)} located sensors",
            )

    mean = float(np.mean(list(usable.values())))
    return TemperatureField(
        per_segment_c={s: mean for s in segments},
        sensor_values_c=usable, fallback_c=fallback_c, degraded=False,
        note=(
            f"uniform temperature {mean:.1f} degC from {len(usable)} sensor(s); "
            f"no sensor coordinates configured, so no spatial field"
        ),
    )


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def shunt_voltage_to_current_density(
    v_shunt: np.ndarray, segment: int, shunt: ShuntCalibration, temperature_c: float
) -> np.ndarray:
    """Convert a shunt voltage trace [V] to local current density [A/cm^2]."""
    H = shunt.H(segment, temperature_c)
    if not np.isfinite(H) or H == 0.0:
        raise ValueError(f"segment {segment}: non-physical shunt factor H={H}")
    return np.asarray(v_shunt, float) / H


def sensor_voltage_to_celsius(
    v: np.ndarray, index: int, cal: TemperatureCalibration
) -> float:
    """DC temperature from a sensor trace, using the record mean."""
    return cal.to_celsius(index, float(np.mean(v)))
