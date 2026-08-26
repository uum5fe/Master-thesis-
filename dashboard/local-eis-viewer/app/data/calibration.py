r"""Reading and evaluating the plate calibration ("Abgleich") campaigns.

A calibration campaign is a folder of ``Step<n>_<T>Grad.csv`` files plus a
``coefficients/`` folder holding ``curr.csv`` and ``temp.csv``. Each step is one
plate temperature; each line of a step is one segment::

    s1:<TAB>temp1=3.000357V;temp2=2.464455V;temp3=2.484830V;temp4=3.215963V;
            <TAB>i_s=-0.000004A;u_s=0.006492V;<TAB>i_s=0.250458A;u_s=0.316003V;...

so a line carries the four plate temperature-sensor voltages and a current
sweep - here five points from 0 to 1 A - of injected current against the
measured shunt voltage.

WHAT THE NUMBERS MEAN, ESTABLISHED FROM THE DATA
------------------------------------------------
``Grad`` in the filename is the plate temperature in degrees Celsius, not an
angle: putting the sensor voltages through ``temp.csv`` reproduces the label.
On one campaign here the sensors read the label back exactly; on two others
they sit about 1.8 degrees above it, which is a real offset worth seeing rather
than a rounding artefact.

The shipped ``curr.csv`` is reproduced from the raw sweeps by fitting each
segment's sweep **through the origin** - ``u_s = H * i_s``, no intercept -
then fitting that sensitivity against temperature. Doing it that way makes the
ratio between the reconstruction and the shipped coefficient identical for all
72 segments to within 1e-4; fitting with an intercept instead leaves a scatter
of 1e-2, so the convention is not in doubt.

What that constant ratio *is* - roughly 2.95 on one campaign and 3.06 on two
others - is not derivable from these files. It is a property of the
measurement chain, and it changed between campaigns. This module therefore
reports it rather than assuming it: a constant ratio across segments is itself
the check that the coefficients belong to the sweeps, and a segment that
departs from it is a channel worth investigating.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: ``s12:`` at the start of a line, then the payload.
_SEGMENT = re.compile(r"^s(\d+)\s*:(.*)$")
_TEMP = re.compile(r"temp(\d+)\s*=\s*(-?[\d.]+)\s*V", re.IGNORECASE)
_PAIR = re.compile(r"i_s\s*=\s*(-?[\d.]+)\s*A\s*;\s*u_s\s*=\s*(-?[\d.]+)\s*V",
                   re.IGNORECASE)
#: ``Step4_-40Grad.csv`` -> index 4, nominal -40 degrees.
_STEP_NAME = re.compile(r"Step(\d+)_(-?\d+)\s*Grad", re.IGNORECASE)


@dataclass
class CalibrationStep:
    """One temperature point of a campaign."""

    index: int
    nominal_c: float
    path: str
    #: segment -> (i [A], u [V])
    sweeps: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    #: segment -> the four sensor voltages recorded on that line
    sensor_v: dict[str, list[float]] = field(default_factory=dict)
    #: measured temperature per sensor, once temp.csv has been applied
    sensor_c: list[float] = field(default_factory=list)

    @property
    def measured_c(self) -> float:
        return float(np.mean(self.sensor_c)) if self.sensor_c else float("nan")

    @property
    def label(self) -> str:
        return f"{self.nominal_c:g} °C"


@dataclass
class CalibrationRun:
    """One campaign: several temperature steps and the coefficients shipped with them."""

    name: str
    source: str
    steps: list[CalibrationStep] = field(default_factory=list)
    #: per-segment (c0, c1) from coefficients/curr.csv, in file order
    curr_coeff: np.ndarray | None = None
    #: per-sensor (c0, c1) from coefficients/temp.csv
    temp_coeff: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def segment_names(self) -> list[str]:
        names: set[str] = set()
        for step in self.steps:
            names |= set(step.sweeps)
        return sorted(names, key=lambda n: (0, int(n)) if n.isdigit() else (1, n))

    @property
    def n_segments(self) -> int:
        return len(self.segment_names)

    @property
    def temperatures(self) -> list[float]:
        return [s.measured_c for s in self.steps]

    def label(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def parse_step(path: str | Path) -> CalibrationStep:
    """Read one ``Step<n>_<T>Grad.csv``."""
    path = Path(path)
    m = _STEP_NAME.search(path.name)
    step = CalibrationStep(
        index=int(m.group(1)) if m else 0,
        nominal_c=float(m.group(2)) if m else float("nan"),
        path=str(path),
    )
    # latin-1: the files carry CRLF and occasionally a degree sign.
    for line in path.read_text(encoding="latin-1", errors="replace").splitlines():
        m = _SEGMENT.match(line.strip())
        if not m:
            continue
        segment, payload = m.group(1), m.group(2)
        pairs = _PAIR.findall(payload)
        if not pairs:
            continue
        step.sweeps[segment] = (
            np.array([float(a) for a, _ in pairs], float),
            np.array([float(b) for _, b in pairs], float),
        )
        temps = _TEMP.findall(payload)
        if temps:
            step.sensor_v[segment] = [float(v) for _n, v in
                                      sorted(temps, key=lambda t: int(t[0]))]
    return step


def read_coefficients(path: str | Path) -> np.ndarray | None:
    """Read a ``c0;c1`` per-line coefficient file."""
    path = Path(path)
    if not path.is_file():
        return None
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        line = line.strip().replace(",", ".")
        if not line:
            continue
        parts = re.split(r"[;\t]", line)
        if len(parts) < 2:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1])])
        except ValueError:
            continue
    return np.array(rows, float) if rows else None


def is_calibration_folder(directory: str | Path) -> bool:
    d = Path(directory)
    return d.is_dir() and any(d.glob("Step*.csv"))


def load_campaign(directory: str | Path, name: str = "") -> CalibrationRun:
    """Read a whole campaign folder."""
    d = Path(directory)
    run = CalibrationRun(name=name or d.name, source=str(d))

    files = sorted(d.glob("Step*.csv"),
                   key=lambda p: (int(_STEP_NAME.search(p.name).group(1))
                                  if _STEP_NAME.search(p.name) else 0, p.name))
    if not files:
        run.warnings.append(f"no Step*.csv under {d}")
        return run

    for path in files:
        run.steps.append(parse_step(path))

    coeff_dir = d / "coefficients"
    run.curr_coeff = read_coefficients(coeff_dir / "curr.csv")
    run.temp_coeff = read_coefficients(coeff_dir / "temp.csv")
    if run.curr_coeff is None:
        run.warnings.append(
            "no coefficients/curr.csv - the reconstruction cannot be compared "
            "against the coefficients actually in use")
    if run.temp_coeff is None:
        run.warnings.append(
            "no coefficients/temp.csv - sensor voltages cannot be converted to "
            "temperatures, so the nominal step labels are used instead")

    # Sensor voltages -> temperatures. T = (V - c0) / c1 per sensor.
    for step in run.steps:
        if not step.sensor_v:
            step.sensor_c = [step.nominal_c]
            continue
        volts = np.array(list(step.sensor_v.values()), float).mean(axis=0)
        if run.temp_coeff is not None and len(run.temp_coeff) >= len(volts):
            step.sensor_c = [float((v - run.temp_coeff[i, 0]) / run.temp_coeff[i, 1])
                             for i, v in enumerate(volts)]
        else:
            step.sensor_c = [step.nominal_c]

    if run.curr_coeff is not None and len(run.curr_coeff) != run.n_segments:
        run.warnings.append(
            f"coefficients/curr.csv has {len(run.curr_coeff)} rows but the "
            f"sweeps cover {run.n_segments} segments")
    return run


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def sensitivity(i: np.ndarray, u: np.ndarray) -> float:
    """Shunt sensitivity ``H = u / i`` fitted through the origin.

    Through the origin, not a free straight line: that is the convention the
    shipped coefficients were produced with, and using the other one puts a
    scatter into the comparison that is a hundred times the real one.
    """
    denominator = float(np.sum(i * i))
    return float(np.sum(u * i) / denominator) if denominator > 0 else float("nan")


def linearity_error_pct(i: np.ndarray, u: np.ndarray) -> float:
    """Largest deviation from the through-origin line, per cent of full scale."""
    H = sensitivity(i, u)
    if not np.isfinite(H):
        return float("nan")
    residual = u - H * i
    span = float(np.max(np.abs(u))) or float("nan")
    return float(100.0 * np.max(np.abs(residual)) / span)


def step_table(run: CalibrationRun) -> pd.DataFrame:
    """One row per temperature step: what it claims, and what the sensors read."""
    rows = []
    for step in run.steps:
        row = {
            "step": step.index,
            "file": Path(step.path).name,
            "nominal_C": step.nominal_c,
            "measured_C": step.measured_c,
            "offset_C": step.measured_c - step.nominal_c,
            "segments": len(step.sweeps),
            "points_per_sweep": (len(next(iter(step.sweeps.values()))[0])
                                 if step.sweeps else 0),
        }
        for index, value in enumerate(step.sensor_c, start=1):
            row[f"temp{index}_C"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def segment_table(run: CalibrationRun) -> pd.DataFrame:
    """One row per segment: the reconstruction, and how it compares.

    ``ratio_c0`` and ``ratio_c1`` are the reconstruction divided by the shipped
    coefficient. Their *value* depends on the measurement chain and is not
    knowable from these files; their *constancy* across segments is the check
    that matters, so the deviation of each segment from the plate median is
    reported as well.
    """
    temperatures = np.array(run.temperatures, float)
    rows = []

    for segment in run.segment_names:
        sensitivities, linearities, used = [], [], []
        for step, temperature in zip(run.steps, temperatures):
            sweep = step.sweeps.get(segment)
            if sweep is None:
                continue
            i, u = sweep
            sensitivities.append(sensitivity(i, u))
            linearities.append(linearity_error_pct(i, u))
            used.append(temperature)

        row: dict[str, object] = {"segment": segment, "n_steps": len(used)}
        if len(used) >= 2 and np.all(np.isfinite(sensitivities)):
            slope, intercept = np.polyfit(used, sensitivities, 1)
            row["H_c0"] = float(intercept)             # sensitivity at 0 °C
            row["H_c1"] = float(slope * 1000.0)        # per °C, x1000 as shipped
            row["H_at_60C"] = float(intercept + slope * 60.0)
            # How much of the sensitivity is temperature dependence, per cent
            # over the measured span - a segment far from the others here has a
            # different thermal path, not a different gain.
            span = float(np.ptp(used))
            row["drift_pct_over_span"] = (
                float(100.0 * slope * span / intercept) if intercept else np.nan)
            residual = np.asarray(sensitivities) - (slope * np.asarray(used) + intercept)
            row["fit_residual_pct"] = float(
                100.0 * np.max(np.abs(residual)) / abs(intercept)) if intercept else np.nan
        else:
            row |= {"H_c0": np.nan, "H_c1": np.nan, "H_at_60C": np.nan,
                    "drift_pct_over_span": np.nan, "fit_residual_pct": np.nan}

        row["linearity_worst_pct"] = (float(np.nanmax(linearities))
                                      if linearities else np.nan)

        index = int(segment) - 1 if segment.isdigit() else -1
        if (run.curr_coeff is not None and 0 <= index < len(run.curr_coeff)):
            c0, c1 = run.curr_coeff[index]
            row["file_c0"], row["file_c1"] = float(c0), float(c1)
            row["ratio_c0"] = float(row["H_c0"] / c0) if c0 else np.nan
            row["ratio_c1"] = float(row["H_c1"] / c1) if c1 else np.nan
        rows.append(row)

    frame = pd.DataFrame(rows)
    for name in ("ratio_c0", "ratio_c1"):
        if name in frame.columns and frame[name].notna().any():
            median = float(frame[name].median())
            frame[f"{name}_dev_pct"] = 100.0 * (frame[name] / median - 1.0)
    return frame


def repeat_drift(run: CalibrationRun) -> pd.DataFrame:
    """Change in sensitivity between repeats of the same nominal temperature.

    A campaign that starts and ends at 20 °C is asking exactly one question:
    did the plate come back to where it started? A segment that did not has
    drifted, or was disturbed during the campaign, and its coefficient is worth
    less than the others.
    """
    by_nominal: dict[float, list[CalibrationStep]] = {}
    for step in run.steps:
        by_nominal.setdefault(step.nominal_c, []).append(step)
    repeated = {t: s for t, s in by_nominal.items() if len(s) > 1}
    if not repeated:
        return pd.DataFrame()

    rows = []
    for nominal, steps in sorted(repeated.items()):
        first, last = steps[0], steps[-1]
        for segment in run.segment_names:
            a, b = first.sweeps.get(segment), last.sweeps.get(segment)
            if a is None or b is None:
                continue
            h0, h1 = sensitivity(*a), sensitivity(*b)
            rows.append({
                "segment": segment,
                "nominal_C": nominal,
                "first_step": first.index,
                "last_step": last.index,
                "H_first": h0,
                "H_last": h1,
                "drift_pct": 100.0 * (h1 / h0 - 1.0) if h0 else np.nan,
            })
    return pd.DataFrame(rows)


def summary(run: CalibrationRun) -> dict[str, object]:
    """The handful of numbers worth putting at the top of the page."""
    segments = segment_table(run)
    drift = repeat_drift(run)
    out: dict[str, object] = {
        "campaign": run.name,
        "n_steps": len(run.steps),
        "n_segments": run.n_segments,
        "temperature_range_C": (
            f"{min(run.temperatures):.0f} … {max(run.temperatures):.0f}"
            if run.temperatures else "—"),
        "worst_linearity_pct": float(segments["linearity_worst_pct"].max())
        if "linearity_worst_pct" in segments else np.nan,
    }
    if "ratio_c0" in segments and segments["ratio_c0"].notna().any():
        ratios = segments["ratio_c0"].dropna()
        out["chain_constant"] = float(ratios.median())
        out["chain_constant_spread"] = float(ratios.max() - ratios.min())
        out["coefficients_consistent"] = bool(
            (ratios.max() - ratios.min()) / ratios.median() < 1e-3)
    if not drift.empty:
        out["worst_repeat_drift_pct"] = float(drift["drift_pct"].abs().max())
    if run.steps:
        offsets = [s.measured_c - s.nominal_c for s in run.steps]
        out["sensor_offset_C"] = float(np.median(offsets))
    return out
