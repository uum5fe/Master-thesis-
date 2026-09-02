#!/usr/bin/env python3
"""
gamry_compare.py  --  the whole-cell reference, against the local aggregate
===========================================================================

The local EIS measures 72 segments.  Summed in parallel they must reproduce
what an ordinary whole-cell EIS of the same cell measured at the same moment.
That is the only end-to-end check the method has: it tests the calibration,
the geometry, the synchronisation and the chain response all at once, against
an instrument that shares none of them.

WHAT IS BEING COMPARED
----------------------
Gamry writes a whole-cell impedance in OHMS.  The pipeline writes an
area-specific impedance in OHM.CM2.  They are the same quantity in different
units, and the bridge is the cell area:

    Z_gamry_asr(f)  =  Z_gamry(f) * A_cell          [ohm.cm2]

    Z_local_asr(f)  =  A_cell / sum_s ( A_s / Z_s(f) )

The second line is a harmonic, area-weighted mean -- segments sit in parallel
across one cell voltage, so it is admittances that add.  Being harmonic it is
dominated by the LOW-impedance segments, and that is exactly why an integral
measurement hides local faults: a flooded segment has high Z, contributes
little admittance, and barely moves the cell curve.  So agreement here does
NOT mean the local map is uninformative -- it means the local map is correctly
scaled.  Disagreement is the interesting case, and the shape of it says where
to look:

    a constant real offset          -> a resistance in one path and not the
                                       other (lead, contact, sense point)
    a scale factor on all of Z      -> an area error, or a shunt calibration
                                       error common to every segment
    a growing phase error with f    -> uncorrected chain response, or
                                       acquisition skew
    agreement at HF, drift at LF    -> the operating point moved between the
                                       two recordings

WHAT MUST BE TRUE BEFORE A COMPARISON MEANS ANYTHING
----------------------------------------------------
1. SAME CELL.  Compared by order number, not by folder.
2. SAME OPERATING POINT.  A fuel cell's impedance is a strong function of
   current, temperature and humidity.  Where the bench log is available the
   state at the moment of each sweep is read out and reported next to the
   comparison, so "the curves disagree" can be separated from "the cell was
   not in the same condition".
3. OVERLAPPING FREQUENCIES.  Nothing is extrapolated.  The comparison is
   restricted to the band both instruments actually covered, and the number of
   points in it is reported.
4. THE CHAIN RESPONSE APPLIED TO THE LOCAL SIDE.  The segment current path
   has its own transfer function (see gamry_dta.py).  At the top of the
   pipeline band it is about eleven degrees, which is the same order as the
   discrepancies this comparison is looking for.  If it has not been applied,
   this module says so rather than quietly reporting the difference as
   physics.

The Gamry sweep is galvanostatic with IDCREQ = 0: the potentiostat supplies
only the AC perturbation and the bench holds the DC current, so both
instruments see the same excitation on the same cell.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import gamry_dta
import utils


# ---------------------------------------------------------------------------
# 1. Finding the reference sweeps
# ---------------------------------------------------------------------------

#: Current setpoint in the file name, e.g. "..._HFR_102_CurrVal_60.dta".
_CURRENT_RE = re.compile(r"CurrVal[_-]?(\d+(?:[.,]\d+)?)", re.I)
#: A bare setpoint used as the whole name or the TITLE label: "45A.DTA",
#: "TITLE LABEL 450A". Not anchored to the start, so "RO2612025_150A.DTA"
#: is read too, but the A must not be followed by more letters or digits or
#: "45Amps_2" would read as 45 and "1A5" as 1.
_BARE_CURRENT_RE = re.compile(r"(?:^|[^0-9A-Za-z])(\d+(?:[.,]\d+)?)\s*A"
                              r"(?![0-9A-Za-z])", re.I)
#: Absolute start time in the Gamry header.
_START_RE = re.compile(r"STARTTIME\s+LABEL\s+([\d.]+\s+[\d:]+)")
#: The order number as it appears in a bench file name, e.g. "RO2611976-01".
_ORDER_RE = re.compile(r"(R[OA]\d{6,}|FC\d{6,}|\d{7,})", re.I)


@dataclass(frozen=True)
class CellSweep:
    """One whole-cell reference spectrum."""

    path: Path
    freq: np.ndarray                  # Hz, ascending
    Z_ohm: np.ndarray                 # complex ohm, whole cell
    current_a: float | None           # DC setpoint, from the file name
    started: datetime | None
    meta: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def condition(self) -> str:
        """The pipeline's own condition label, e.g. "60A"."""
        if self.current_a is None:
            return "?"
        i = int(round(self.current_a))
        return f"{i}A" if abs(self.current_a - i) < 1e-9 else f"{self.current_a:g}A"

    def asr(self, area_cm2: float) -> np.ndarray:
        return self.Z_ohm * area_cm2

    def hfr_ohm(self) -> float:
        """The high-frequency intercept: Zimag's first zero coming down in f.

        Taken from the HIGH-frequency end deliberately.  The lowest |Zimag| in
        the sweep is often at the low-frequency end, where the curve turns back
        up, and reporting that as HFR silently substitutes the polarisation
        resistance for the ohmic one.
        """
        return _hf_intercept(self.freq, self.Z_ohm)


def _hf_intercept(freq: np.ndarray, Z: np.ndarray) -> float:
    """Z' where Z'' crosses zero on the way DOWN from the top of the band.

    Returns NaN when the crossing is not inside the band, and that case is not
    rare: the intercept moves up in frequency as the cell dries, and on this
    hardware it sits at 6.6 kHz at 45 A -- above the pipeline's 4500 Hz f_max.
    A scan that simply took the first sign change would then find the LOW
    frequency crossing, where the curve turns back up, and report the
    polarisation resistance as if it were the ohmic one.  On the delivered
    data that substitution is a factor of eight, so it has to be impossible
    rather than merely unlikely: if the response at the top of the band is
    still capacitive, the intercept is above the band and is not reported.
    """
    if freq.size < 2:
        return float("nan")
    o = np.argsort(freq)[::-1]
    f, z = freq[o], Z[o]
    if z.imag[0] < 0.0:
        return float("nan")            # still capacitive at f_max: HFR is higher
    for i in range(f.size - 1):
        a, b = z.imag[i], z.imag[i + 1]
        if a == 0.0:
            return float(z.real[i])
        if a * b < 0.0:
            w = a / (a - b)
            return float(z.real[i] + w * (z.real[i + 1] - z.real[i]))
    return float("nan")


def _hf_extrapolated(freq: np.ndarray, Z: np.ndarray, n_top: int = 5) -> float:
    """Z' extrapolated to Z'' = 0 from the top of the band.

    `_hf_intercept` refuses when the sweep stops below the inductive crossover,
    and on this campaign BOTH sides stop there: the Gamry whole-cell sweeps run
    out at 1.2-3.0 kHz while this cell is still capacitive.  Refusing is right
    -- there is no measured intercept -- but it leaves the cross-check with
    nothing to compare, so this offers the next best thing.

    Near the top of the band the Nyquist curve is close to a straight line, so
    a least-squares fit of Z' against Z'' over the top `n_top` points, read at
    Z'' = 0, lands on the intercept the sweep did not reach.  It is an
    EXTRAPOLATION, biased by whatever remains of the arc, and it is reported
    under its own name -- never as the intercept, never merged into `hfr`.
    What makes it useful is that both instruments get the identical treatment
    over the identical band, so the DIFFERENCE between the two extrapolations
    is a fair comparison even though each absolute value carries the bias.

    On the synthetic cell of the test suite the bias is measurable and it is
    what you would expect -- it grows as the band is cut shorter, and it is
    LOW, because the arc still closing above the band has not yet given back
    its real part: -3.8 % from a band ending at 3.0 kHz, -17 % from one ending
    at 1.2 kHz.  So it is a cross-check, not a measurement of R_ohm.
    """
    if freq.size < 3:
        return float("nan")
    o = np.argsort(freq)[::-1]
    zi = Z[o].imag[:max(3, n_top)]
    zr = Z[o].real[:max(3, n_top)]
    if zi.size < 3 or np.ptp(zi) <= 0.0:
        return float("nan")            # a flat imaginary part fixes no line
    b, a = np.polyfit(zi, zr, 1)       # Z' = a + b * Z''
    return float(a)


def _setpoint(name: str, meta: dict) -> float | None:
    """The DC current this sweep was taken at, in amperes.

    Three sources, most specific first, because no single one is always
    present:

      1. ``CurrVal_150`` in the file name -- the bench's own convention.
      2. A bare setpoint in the name, which is how a sweep saved by hand
         gets called: "45A.DTA", "RO2612025_150A.DTA".
      3. The Gamry ``TITLE`` label, which the operator types into the Test
         Identifier box and which the framework stores in the header.

    ``IDCREQ`` is deliberately NOT used even though it looks like the right
    field. On a galvanostatic sweep driven through a booster it reads
    0,00000E+000 -- the DC comes from the load bank, not the potentiostat,
    so the potentiostat records that it requested none. Trusting it would
    label every one of these sweeps 0 A.
    """
    for text, pattern in ((name, _CURRENT_RE),
                          (name, _BARE_CURRENT_RE),
                          (str(meta.get("TITLE", "")), _CURRENT_RE),
                          (str(meta.get("TITLE", "")), _BARE_CURRENT_RE)):
        m = pattern.search(text)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                continue
    return None


def read_cell_sweep(path) -> CellSweep:
    path = Path(path)
    sweep = gamry_dta.read_dta(path).sorted()
    text = path.read_text(encoding="latin-1", errors="ignore")

    current = _setpoint(path.name, sweep.meta)

    started = None
    m = _START_RE.search(text)
    if m:
        for fmt in ("%d.%m.%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
                    "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                started = datetime.strptime(m.group(1).strip(), fmt)
                break
            except ValueError:
                continue

    return CellSweep(path=path, freq=sweep.freq, Z_ohm=sweep.Z,
                     current_a=current, started=started, meta=dict(sweep.meta))


def find_cell_sweeps(root, pattern: str = "*.dta",
                     order_id: str | None = None) -> list[CellSweep]:
    """Every whole-cell sweep under `root`, newest naming first.

    Per-segment chain-response files carry a "#<n>" in the name and live in a
    `bode/` folder; they are a different measurement entirely and are skipped
    here rather than silently averaged into a cell reference.
    """
    root = Path(root)
    out: list[CellSweep] = []
    for path in sorted(root.rglob(pattern)) + sorted(root.rglob(pattern.upper())):
        if "bode" in {p.name.lower() for p in path.parents}:
            continue
        if re.search(r"#\d+", path.name) or path.name.lower().endswith("_raw.dta"):
            continue
        if any(s.path == path for s in out):
            continue
        if names_order(path.name, order_id) is False:
            continue                     # names a different cell
        try:
            out.append(read_cell_sweep(path))
        except Exception:                                   # noqa: BLE001
            continue
    if order_id:
        named = [s for s in out if names_order(s.path.name, order_id) is True]
        if named:
            return named                 # prefer the ones that say so
    return out


# ---------------------------------------------------------------------------
# 2. The bench log, when there is one
# ---------------------------------------------------------------------------

#: Channels worth reporting next to a comparison, and the label to print.
BENCH_CHANNELS: dict[str, str] = {
    "I_S": "I [A]",
    "U_S": "U_cell [V]",
    "n_Cells": "cells",
    # cathode gas, in and out -- both ends, because the plate sits between them
    "T_Si_C": "T_cath_in [C]",
    "T_So_C": "T_cath_out [C]",
    "p_Si_C": "p_cath_in [bara]",
    "p_So_C": "p_cath_out [bara]",
    "RH_Si_C_gas": "RH_cath_in [%]",
    "DPT_Si_C": "dew_cath_in [C]",
    "FN_Si_Air_C": "air_cath [Nl/min]",
    "FN_Si_Air_C_dry": "air_cath_dry [Nl/min]",
    # anode and coolant, for context rather than for the fields
    "T_Si_A": "T_an_in [C]",
    "T_So_A": "T_an_out [C]",
    "RH_Si_A_gas": "RH_an_in [%]",
    "T_Si_CL": "T_cool_in [C]",
    "T_So_CL": "T_cool_out [C]",
    "R_S_HFR": "bench HFR [ohm]",
}

_MF4_TIME_RE = re.compile(r"(\d{8}_\d{4})")


@dataclass
class BenchLog:
    """An ASAM MDF4 bench recording, reduced to what the comparison needs."""

    path: Path
    t0: datetime | None
    series: dict[str, tuple[np.ndarray, np.ndarray]]
    #: How far outside the recording a request may fall before it is refused.
    #: Generous, because clocks drift and a sweep can start just after the log
    #: stops -- but far short of the days that a wrong-file match produces.
    tolerance_s: float = 900.0

    def state_at(self, when: datetime, window_s: float = 30.0) -> dict[str, float]:
        """Bench state at `when`.

        Two sampling models, because this log has both.  Fast channels are
        recorded continuously and are best read as the median over the window
        ending at `when` -- median, not mean, and NaN-aware, since several
        channels park at NaN while their instrument is idle.

        Setpoint-like channels are recorded ON CHANGE: `I_S` holds a value for
        as long as the current is held and has gaps of several minutes.  A
        window median of those returns NaN precisely when the operating point
        is stable, which is exactly when it is being asked for -- the current
        came back missing at every one of the four HFR points until this was
        fixed.  For those the correct read is a zero-order hold: the last value
        recorded at or before `when`.
        """
        if self.t0 is None:
            return {}
        t_rel = (when - self.t0).total_seconds()

        # A zero-order hold is right for a channel logged on change. It is NOT
        # right for a timestamp that falls outside the recording altogether:
        # that means this log and this sweep are not the same session, and
        # holding the last value would report a number from a different day as
        # though it were the operating point. The delivered case was a sweep
        # timestamped 13 days after the log it was matched to.
        span = max((float(t[-1]) for t, _v in self.series.values() if t.size),
                   default=0.0)
        if t_rel < -self.tolerance_s or t_rel > span + self.tolerance_s:
            return {"t_rel_s": t_rel, "out_of_record": 1.0,
                    "record_span_s": span}

        out: dict[str, float] = {}
        for key, (t, v) in self.series.items():
            good = np.isfinite(v)
            k = good & (t >= t_rel - window_s) & (t <= t_rel + 5.0)
            if k.any():
                out[key] = float(np.median(v[k]))
                continue
            before = good & (t <= t_rel + 5.0)
            if before.any():
                out[key] = float(v[before][np.argmax(t[before])])   # held value
            else:
                out[key] = float("nan")
        out["t_rel_s"] = t_rel
        return out


def _asammdf_advice(exc: ImportError) -> list[str]:
    """Why the import failed, and the command that fixes THIS interpreter.

    "pip install asammdf" followed by "asammdf not installed" almost always
    means the pip that ran belongs to a different interpreter -- a system
    Python, a Store alias, another venv.  So the advice has to name the
    interpreter that is actually running, on a line of its own: the previous
    message put the path mid-sentence, where an 80-column console cut it off
    after "C:\\Use" and told the reader nothing.

    A second case looks identical from the outside: asammdf IS installed but
    fails to import, usually because it was built against a different numpy.
    The exception names which, so pass it on rather than repeating "not
    installed" at someone who just installed it.
    """
    import sys

    missing = getattr(exc, "name", "") or ""
    top = missing.split(".")[0]

    if top == "zstandard":
        # _ensure_zstd_importable() reached for the vendored shim -- see
        # _vendor/zstd.py -- and even THAT has nothing to import from,
        # because zstandard itself was never installed.
        return ["asammdf cannot import: its zstd compatibility shim needs "
                "the zstandard package, which is not installed",
                f"  interpreter: {sys.executable}",
                f'  fix: "{sys.executable}" -m pip install zstandard']
    if top and top != "asammdf":
        return [f"asammdf is installed but cannot import: {exc}",
                f"  its dependency {missing!r} is missing or incompatible "
                f"(a numpy built for another version is the usual cause)",
                f"  interpreter: {sys.executable}",
                f'  try: "{sys.executable}" -m pip install --force-reinstall '
                f"asammdf"]
    return ["asammdf not installed in the interpreter running this pipeline",
            "  the operating point (T, p, RH, current) will not be reported "
            "next to each comparison; everything else is unaffected",
            f"  interpreter: {sys.executable}",
            f'  fix: "{sys.executable}" -m pip install --no-deps asammdf',
            f'       "{sys.executable}" -m pip install chardet '
            f'"canmatrix[arxml,dbc]" lxml lz4 numexpr python-dateutil '
            f"typing-extensions zstandard",
            "  (NOT plain `pip install asammdf`: it also pulls in a package "
            "named \"zstd\", which has no Windows wheel past Python 3.10 and "
            "fails to compile without Microsoft's C++ Build Tools -- the "
            "two commands above install everything asammdf needs except "
            "that one, and this pipeline's own zstd stand-in covers it)"]


def _ensure_zstd_importable() -> None:
    """Make `from zstd import decompress` succeed without a C compiler.

    asammdf imports the PyPI package "zstd" unconditionally, and that
    package has published no Windows wheel past Python 3.10 -- installing it
    on a newer Python means compiling it, which needs Microsoft's C++ Build
    Tools. `_vendor/zstd.py` stands in for it using `zstandard`, an
    unrelated, actively maintained project that does ship Windows wheels.
    If the real "zstd" is already importable (Linux/macOS still get wheels
    for it), it takes priority -- this only reaches for the stand-in when it
    is not, so it never shadows a working real installation.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("zstd") is not None:
        return
    vendor = str(Path(__file__).resolve().parent / "_vendor")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def read_bench_log(path) -> BenchLog:
    """Read an MF4 bench log.  Requires `asammdf`; absent, this is skipped."""
    _ensure_zstd_importable()
    from asammdf import MDF                                 # noqa: PLC0415

    path = Path(path)
    t0 = None
    m = _MF4_TIME_RE.search(path.name)
    if m:
        try:
            t0 = datetime.strptime(m.group(1), "%Y%m%d_%H%M")
        except ValueError:
            t0 = None

    mdf = MDF(path)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in BENCH_CHANNELS:
        try:
            sig = mdf.get(name)
        except Exception:                                   # noqa: BLE001
            continue
        series[name] = (np.asarray(sig.timestamps, float),
                        np.asarray(sig.samples, float))
    if t0 is None and getattr(mdf, "start_time", None) is not None:
        t0 = mdf.start_time.replace(tzinfo=None)
    return BenchLog(path=path, t0=t0, series=series)


def _order_core(text: str) -> str:
    """The digits of an order id, so RO2611976-01 and 2611976 compare equal."""
    return re.sub(r"\D", "", text or "")


def names_order(name: str, order_id: str | None) -> bool | None:
    """Does this filename name this cell?

    True  -- it names this order.
    False -- it names a DIFFERENT one, so it is the wrong cell.
    None  -- it names none, so the filename cannot settle it either way.

    The distinction matters: a file that is silent about the cell may still be
    the right one and is allowed through as a last resort, but a file that
    names another cell never is. Reading RO2611959's bench log for a run of
    RO2611976 is not a near miss -- it is a different cell at a different
    operating point, and every field derived from it is fiction.
    """
    want = _order_core(order_id or "")
    if not want:
        return None
    found = {_order_core(m.group(0)) for m in _ORDER_RE.finditer(name)}
    if not found:
        return None
    return want in found


def find_bench_log(root, order_id: str | None = None) -> Path | None:
    """The bench .mf4 for this cell, or None.

    A campaign share holds one folder per cell, and pointing at the parent used
    to return whichever file sorted first -- silently, and for a different
    cell.
    """
    files = sorted(Path(root).rglob("*.mf4")) + sorted(Path(root).rglob("*.MF4"))
    # The "_Anfang" file is the run-up; the main file is the one with the data.
    files = [f for f in files if "anfang" not in f.name.lower()] or files
    if not files:
        return None

    named = [f for f in files if names_order(f.name, order_id) is True]
    if named:
        return named[0]
    # Nothing names the order. A file that names ANOTHER one is excluded;
    # what is left is silent about the cell and may legitimately be it.
    silent = [f for f in files if names_order(f.name, order_id) is None]
    return silent[0] if silent else None


# ---------------------------------------------------------------------------
# 3. The comparison itself
# ---------------------------------------------------------------------------

@dataclass
class Comparison:
    """One local aggregate against one whole-cell sweep."""

    condition: str
    sweep_name: str
    area_cm2: float
    freq: np.ndarray                  # the overlap band, ascending
    Z_local: np.ndarray               # ohm.cm2
    Z_ref: np.ndarray                 # ohm.cm2
    hfr_local: float                  # ohm.cm2, measured intercept or NaN
    hfr_ref: float                    # ohm.cm2, measured intercept or NaN
    hfr_local_fit: float = float("nan")   # ohm.cm2, extrapolated
    hfr_ref_fit: float = float("nan")     # ohm.cm2, extrapolated
    bench: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- scalar metrics -----------------------------------------------------

    @property
    def n_points(self) -> int:
        return int(self.freq.size)

    @property
    def hfr_rel(self) -> float:
        """Relative HFR difference, local against reference."""
        if not np.isfinite(self.hfr_ref) or self.hfr_ref == 0:
            return float("nan")
        return float(self.hfr_local / self.hfr_ref - 1.0)

    @property
    def hfr_fit_rel(self) -> float:
        """Relative difference of the two EXTRAPOLATED intercepts.

        Reported only when the measured intercept is missing on at least one
        side; when both sides measured it, `hfr_rel` is the honest number and
        this one adds nothing.
        """
        if not np.isfinite(self.hfr_ref_fit) or self.hfr_ref_fit == 0:
            return float("nan")
        return float(self.hfr_local_fit / self.hfr_ref_fit - 1.0)

    @property
    def mag_rel_median(self) -> float:
        return float(np.median(np.abs(self.Z_local) / np.abs(self.Z_ref) - 1.0))

    @property
    def phase_diff_deg(self) -> np.ndarray:
        return np.degrees(np.angle(self.Z_local) - np.angle(self.Z_ref))

    @property
    def rms_rel(self) -> float:
        """RMS of |Z_local - Z_ref| / |Z_ref| over the overlap band."""
        d = np.abs(self.Z_local - self.Z_ref) / np.abs(self.Z_ref)
        return float(np.sqrt(np.mean(d ** 2)))

    def summary(self) -> dict:
        return {
            "condition": self.condition,
            "reference": self.sweep_name,
            "n_points": self.n_points,
            "f_lo_hz": float(self.freq.min()) if self.n_points else float("nan"),
            "f_hi_hz": float(self.freq.max()) if self.n_points else float("nan"),
            "hfr_local_mohm_cm2": 1e3 * self.hfr_local,
            "hfr_ref_mohm_cm2": 1e3 * self.hfr_ref,
            "hfr_rel_pct": 100 * self.hfr_rel,
            "hfr_local_fit_mohm_cm2": 1e3 * self.hfr_local_fit,
            "hfr_ref_fit_mohm_cm2": 1e3 * self.hfr_ref_fit,
            "hfr_fit_rel_pct": 100 * self.hfr_fit_rel,
            "mag_rel_median_pct": 100 * self.mag_rel_median,
            "rms_rel_pct": 100 * self.rms_rel,
            "phase_diff_max_deg": (float(np.max(np.abs(self.phase_diff_deg)))
                                   if self.n_points else float("nan")),
            "notes": "; ".join(self.notes),
        }


def compare(freq_local: np.ndarray, Z_local: np.ndarray, sweep: CellSweep,
            area_cm2: float, bench: dict | None = None,
            chain_applied: bool | None = None) -> Comparison:
    """Put a local aggregate and a whole-cell sweep on the same axis.

    The reference is interpolated onto the local frequencies -- not the other
    way round -- because the local grid is the one the pipeline actually
    evaluated, and because the Gamry sweep is the denser of the two.  Only the
    overlap is kept: extrapolating either curve past the band its instrument
    covered would invent the very disagreement this is measuring.
    """
    freq_local = np.asarray(freq_local, float)
    Z_local = np.asarray(Z_local, complex)
    o = np.argsort(freq_local)
    freq_local, Z_local = freq_local[o], Z_local[o]

    ref_f, ref_Z = sweep.freq, sweep.asr(area_cm2)
    lo = max(freq_local.min(), ref_f.min()) if freq_local.size and ref_f.size else np.nan
    hi = min(freq_local.max(), ref_f.max()) if freq_local.size and ref_f.size else np.nan

    notes: list[str] = []
    if not np.isfinite(lo) or lo > hi:
        notes.append("no overlapping frequencies")
        band = np.zeros(0, dtype=bool)
    else:
        band = (freq_local >= lo) & (freq_local <= hi)
    f = freq_local[band]
    Zl = Z_local[band]
    Zr = utils.interp_complex(f, ref_f, ref_Z) if f.size else np.zeros(0, complex)

    # Both HFRs are read off the same way: the real part where the imaginary
    # part crosses zero from below.  Which side failed to reach that crossing
    # decides which note the reader needs, so resolve both before writing any.
    local_hfr = _hf_intercept(f, Zl) if f.size else float("nan")
    ref_hfr = _hf_intercept(ref_f, ref_Z) if ref_f.size else float("nan")

    # When a side did not reach its own intercept, extrapolate the top of its
    # band to Z'' = 0 so the cross-check still has a like-for-like number.
    # Both sides get the same treatment over the same overlap band.
    local_fit = ref_fit = float("nan")
    if f.size and (not np.isfinite(local_hfr) or not np.isfinite(ref_hfr)):
        local_fit = _hf_extrapolated(f, Zl)
        ref_fit = _hf_extrapolated(f, Zr)

    if not np.isfinite(ref_hfr) and ref_f.size:
        notes.append(
            f"HFR not measurable on EITHER side: the whole-cell sweep is still "
            f"capacitive at its own top frequency, {ref_f.max():.4g} Hz, so "
            f"neither instrument reached the intercept. The comparison of |Z| "
            f"and phase over the {f.size} overlapping point(s) is unaffected; "
            f"the HFR columns fall back to hfr_*_fit, an extrapolation of the "
            f"top of the band to Z''=0, which is biased by the unclosed arc on "
            f"both sides equally and so is only meaningful as a difference")
    elif f.size and not np.isfinite(local_hfr):
        notes.append(
            f"HFR not measurable locally: still capacitive at {f.max():.4g} Hz, "
            f"so the intercept is above the evaluated band -- raising "
            f"cfg.f_max_hz would reach it, since the reference did. Until "
            f"then the HFR columns fall back to hfr_*_fit, which extrapolates "
            f"BOTH sides from the same overlap band; the reference's own "
            f"measured intercept is {1e3 * ref_hfr:.2f} mohm.cm2, but "
            f"comparing it against an extrapolation would charge the "
            f"extrapolation's bias to the local side")
    if chain_applied is False:
        notes.append("chain response NOT applied to the local side -- a phase "
                     "difference growing with frequency is expected and is an "
                     "artefact, not physics")
    if f.size and f.size < 5:
        notes.append(f"only {f.size} overlapping point(s)")

    return Comparison(
        condition=sweep.condition, sweep_name=sweep.name, area_cm2=area_cm2,
        freq=f, Z_local=Zl, Z_ref=Zr,
        hfr_local=local_hfr,
        hfr_ref=ref_hfr,
        hfr_local_fit=local_fit,
        hfr_ref_fit=ref_fit,
        bench=dict(bench or {}), notes=notes,
    )


def read_cell_aggregate(path) -> tuple[np.ndarray, np.ndarray]:
    """Read the pipeline's own `cell_aggregate.csv` (mohm.cm2) as ohm.cm2.

    Plain csv rather than pandas, to match `utils.write_table` that wrote it
    and to keep the core path free of a pandas import.
    """
    import csv                                              # noqa: PLC0415

    with Path(path).open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("freq_hz")]
    if not rows:
        return np.zeros(0), np.zeros(0, complex)
    f = np.array([float(r["freq_hz"]) for r in rows])
    z = np.array([float(r["z_re_mohm_cm2"]) + 1j * float(r["z_im_mohm_cm2"])
                  for r in rows]) / 1e3
    o = np.argsort(f)
    return f[o], z[o]


# ---------------------------------------------------------------------------
# 4. Outputs
# ---------------------------------------------------------------------------

def write_outputs(comparisons: list[Comparison], out_dir, log=None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not comparisons:
        return

    utils.write_table(out / "gamry_comparison.csv",
                      [c.summary() for c in comparisons])

    rows = []
    for c in comparisons:
        for i in range(c.n_points):
            rows.append({
                "condition": c.condition,
                "freq_hz": round(float(c.freq[i]), 6),
                "z_re_local_mohm_cm2": round(1e3 * float(c.Z_local[i].real), 5),
                "z_im_local_mohm_cm2": round(1e3 * float(c.Z_local[i].imag), 5),
                "z_re_ref_mohm_cm2": round(1e3 * float(c.Z_ref[i].real), 5),
                "z_im_ref_mohm_cm2": round(1e3 * float(c.Z_ref[i].imag), 5),
                "phase_diff_deg": round(float(c.phase_diff_deg[i]), 4),
            })
    if rows:
        utils.write_table(out / "gamry_comparison_curves.csv", rows)

    if log:
        for c in comparisons:
            s = c.summary()
            if np.isfinite(c.hfr_ref) and np.isfinite(c.hfr_local):
                hfr = (f"HFR {s['hfr_local_mohm_cm2']:.2f} vs "
                       f"{s['hfr_ref_mohm_cm2']:.2f} mohm.cm2 "
                       f"({s['hfr_rel_pct']:+.1f} %)")
            else:
                hfr = (f"HFR~ {s['hfr_local_fit_mohm_cm2']:.2f} vs "
                       f"{s['hfr_ref_fit_mohm_cm2']:.2f} mohm.cm2 "
                       f"({s['hfr_fit_rel_pct']:+.1f} %, extrapolated)")
            log.info(f"  {s['condition']:>6}  vs {s['reference']}: "
                     f"{s['n_points']} pts {s['f_lo_hz']:.3g}-{s['f_hi_hz']:.3g} Hz, "
                     f"{hfr}, |Z| {s['mag_rel_median_pct']:+.1f} %")
            for note in c.notes:
                log.warning(f"         {note}")


def plot(comparisons: list[Comparison], path="gamry_comparison.png"):
    """Nyquist and residual, one column per condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparisons = [c for c in comparisons if c.n_points]
    if not comparisons:
        return None
    n = len(comparisons)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.4), squeeze=False)
    for j, c in enumerate(comparisons):
        ax = axes[0][j]
        ax.plot(1e3 * c.Z_ref.real, -1e3 * c.Z_ref.imag, "o-", ms=3, lw=1,
                color="0.35", label="whole cell (Gamry)")
        ax.plot(1e3 * c.Z_local.real, -1e3 * c.Z_local.imag, "s-", ms=3, lw=1,
                color="tab:red", label="local, aggregated")
        ax.set(xlabel="Z'  [mΩ·cm²]", ylabel="-Z''  [mΩ·cm²]",
               title=f"{c.condition}   ({c.n_points} pts)")
        ax.set_aspect("equal")
        ax.grid(alpha=.3)
        if j == 0:
            ax.legend(fontsize=8)

        ax = axes[1][j]
        rel = 100 * (np.abs(c.Z_local) / np.abs(c.Z_ref) - 1.0)
        ax.semilogx(c.freq, rel, "o-", ms=3, lw=1, color="tab:red",
                    label="|Z| difference [%]")
        ax.semilogx(c.freq, c.phase_diff_deg, "s-", ms=3, lw=1,
                    color="tab:blue", label="phase difference [°]")
        ax.axhline(0, color="0.5", lw=.8)
        ax.set(xlabel="f [Hz]", ylabel="local − reference")
        ax.grid(alpha=.3)
        if j == 0:
            ax.legend(fontsize=8)
    fig.suptitle("Local EIS aggregated to the cell, against the whole-cell "
                 "reference", y=.99)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


# ---------------------------------------------------------------------------
# 5. Driving it
# ---------------------------------------------------------------------------

def run(results_root, gamry_root, area_cm2: float, out_dir=None,
        bench_path=None, chain_applied: bool | None = None,
        only: str | None = None, order_id: str | None = None,
        log=None) -> list[Comparison]:
    """Compare every condition that has BOTH a local result and a sweep.

    `results_root` is a pipeline output tree laid out
    ``<root>/<condition>/silver/cell_aggregate.csv`` or a single such folder.
    Conditions with only one of the two are reported and skipped: a comparison
    against a missing half is not a weak comparison, it is not one at all.

    `only` restricts the comparison to one condition. A run of 150A writes its
    results into 150A's folder alone, so without this the other three sweeps
    in the campaign each produce a "no local result" warning -- three lines of
    alarm about conditions that were never part of this run and are not
    missing anything.
    """
    log = log or utils.get_logger(True)
    results_root, gamry_root = Path(results_root), Path(gamry_root)

    sweeps = {s.condition: s
              for s in find_cell_sweeps(gamry_root, order_id=order_id)}
    if only and only.upper() != "ALL":
        sweeps = {k: v for k, v in sweeps.items()
                  if k.upper() == only.upper()}
        if not sweeps:
            log.warning(f"no whole-cell sweep at {only} under {gamry_root}")
            return []
    if not sweeps:
        log.warning(f"no whole-cell .dta sweeps under {gamry_root}")
        return []

    bench = None
    if bench_path is None:
        bench_path = find_bench_log(gamry_root)
    if bench_path is not None:
        try:
            bench = read_bench_log(bench_path)
            log.info(f"bench log: {Path(bench_path).name} "
                     f"(t0 {bench.t0}, {len(bench.series)} channels)")
        except ImportError as exc:
            for line in _asammdf_advice(exc):
                log.warning(line)
        except Exception as exc:                            # noqa: BLE001
            log.warning(f"bench log unreadable: {exc}")

    locals_: dict[str, Path] = {}
    for cand in sorted(results_root.rglob("cell_aggregate.csv")):
        # <...>/<condition>/silver/cell_aggregate.csv
        cond = cand.parent.parent.name
        locals_.setdefault(cond, cand)
    if not locals_ and (results_root / "cell_aggregate.csv").is_file():
        locals_[results_root.parent.name] = results_root / "cell_aggregate.csv"

    def by_current(cond: str) -> tuple[int, float, str]:
        """Order 45A, 60A, 150A, 450A -- not 150A, 450A, 45A, 60A."""
        m = re.match(r"([\d.]+)\s*A", cond, re.I)
        return (0, float(m.group(1)), "") if m else (1, 0.0, cond)

    # A sweep with no local counterpart is the normal case, not a fault: one
    # pipeline run evaluates one condition and writes one folder, while the
    # campaign folder holds the sweeps for all of them.  Three warnings about
    # conditions that were never part of this run read as three failures, so
    # they are one informational line instead.  A LOCAL result with no sweep
    # is the other way round -- that one really is a gap -- and stays a
    # warning.
    unmatched = [c for c in sorted(set(sweeps) - set(locals_), key=by_current)]
    if unmatched and locals_:
        log.info(f"not evaluated in this run, so not compared: "
                 f"{', '.join(unmatched)} "
                 f"(sweeps exist; re-run the pipeline at those conditions to "
                 f"compare them)")
    elif unmatched:
        log.warning(f"no local results found under {results_root} -- nothing "
                    f"to compare against the {len(unmatched)} sweep(s) found")

    out: list[Comparison] = []
    for cond in sorted(set(sweeps) | set(locals_), key=by_current):
        if cond not in sweeps:
            log.warning(f"{cond}: local result but no whole-cell sweep")
            continue
        if cond not in locals_:
            continue
        f, z = read_cell_aggregate(locals_[cond])
        state = {}
        sweep = sweeps[cond]
        if bench is not None and sweep.started is not None:
            state = bench.state_at(sweep.started)
        out.append(compare(f, z, sweep, area_cm2, bench=state,
                           chain_applied=chain_applied))

    write_outputs(out, out_dir or results_root, log=log)
    return out


# ---------------------------------------------------------------------------
# 6. Self-test
# ---------------------------------------------------------------------------

def _selftest(log=None) -> int:
    """Round-trip a known cell through the comparison and check the metrics.

    The point of a cross-check is that it FAILS when it should, so the test
    plants a defect of each kind it claims to distinguish and asserts that the
    right metric moves and the others do not.
    """
    log = log or utils.get_logger(True)
    fails = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal fails
        fails += not ok
        log.info(f"    {name:<34}: {'PASS' if ok else 'FAIL'}  {detail}")

    area = 304.92
    freq = np.logspace(np.log10(0.3), np.log10(30000), 61)
    w = 2 * np.pi * freq
    Z = 1j * w * 2.6e-7 + 60e-3 + 300e-3 / (1.0 + (1j * w * 2e-3) ** 0.85)

    sweep = CellSweep(path=Path("synthetic_CurrVal_60.dta"), freq=freq,
                      Z_ohm=Z / area, current_a=60.0, started=None)
    check("condition from the file name", sweep.condition == "60A",
          sweep.condition)

    same = compare(freq, Z, sweep, area)
    check("identical input -> no difference",
          abs(same.mag_rel_median) < 1e-6 and
          np.max(np.abs(same.phase_diff_deg)) < 1e-6,
          f"|Z| {100*same.mag_rel_median:+.2e} %")

    scaled = compare(freq, 1.05 * Z, sweep, area)
    check("5 % scale error seen in |Z| only",
          abs(scaled.mag_rel_median - 0.05) < 1e-6 and
          np.max(np.abs(scaled.phase_diff_deg)) < 1e-6,
          f"|Z| {100*scaled.mag_rel_median:+.2f} %")

    lagged = compare(freq, Z * np.exp(-1j * w * 7e-6), sweep, area,
                     chain_applied=False)
    top = lagged.phase_diff_deg[np.argmax(lagged.freq)]
    want = -np.degrees(2 * np.pi * lagged.freq.max() * 7e-6)
    check("7 us lag seen in phase only",
          abs(lagged.mag_rel_median) < 1e-6 and abs(top - want) < 1.0,
          f"{top:+.1f} deg at {lagged.freq.max():.3g} Hz")
    check("uncorrected chain response is flagged",
          any("chain response NOT applied" in n for n in lagged.notes))

    narrow = compare(freq[freq <= 200], Z[freq <= 200], sweep, area)
    check("HFR above the band is refused",
          not np.isfinite(narrow.hfr_local) and
          any("above the evaluated band" in n for n in narrow.notes))

    fine = np.logspace(np.log10(0.3), np.log10(30000), 20001)
    wf = 2 * np.pi * fine
    Zf = 1j * wf * 2.6e-7 + 60e-3 + 300e-3 / (1.0 + (1j * wf * 2e-3) ** 0.85)
    truth = _hf_intercept(fine, Zf)
    check("HFR inside the band is found",
          abs(same.hfr_local / truth - 1.0) < 5e-3,
          f"{1e3*same.hfr_local:.2f} vs {1e3*truth:.2f} mohm.cm2")

    return int(fails)
