#!/usr/bin/env python3
"""
csv_source.py  --  reader for the newer CSV measurement files
=============================================================

The FAMOS route and the CSV route are not two readers feeding one pipeline.
They are two pipelines, and this module is the front half of the CSV one.
The reason is in the hardware, not the file format:

    FAMOS   five Dewetron cards, armed separately, each with its own
            free-running clock.  Nothing in the data says which sample of
            card 3 is simultaneous with which sample of card 1 -- measured
            offsets on the delivered set run to 5.7 s -- so the first job of
            the pipeline is to *measure* the inter-card delay by
            cross-correlating the shared cell-voltage channel, and to keep
            re-measuring it because it drifts.

    CSV     one logger, one clock, one row per instant for the whole plate.
            There is no second clock to correlate against.  Running the
            FAMOS synchronisation stage here would not merely be wasted work:
            the quantity it estimates does not exist, and what the estimator
            would return is noise dressed as a delay.

So the CSV path drops synchronisation entirely -- and picks up three
problems that FAMOS does not have, which is what the rest of this module is
about.

1. THE SAMPLE INTERVAL IS NOT CONSTANT
   A FAMOS file declares one sample interval in its header and honours it; the
   DFT of such a record is exact.  A CSV logger writes a timestamp per row,
   and that timestamp jitters -- scheduler latency, buffered writes, a
   USB frame boundary.  Resampling onto a uniform grid to "fix" it inserts
   an interpolation error that is largest exactly where the signal moves
   fastest, i.e. at the top of the band where the phase matters most.
   `regularity()` measures the jitter, and when it is above tolerance the
   phasor fit uses the RECORDED timestamps directly (see
   `fit_phasor_nonuniform`): a least-squares sine fit does not require a
   uniform grid, only known sample times, and non-uniform sampling actually
   *suppresses* aliasing rather than causing it.

2. THE FILE IS TEXT, SO THE DATA IS QUANTISED TWICE
   Once by the ADC and again by the number of digits printed.  Six decimals
   on a 10 mV shunt signal is a quantisation step worth about 0.3 % of the
   signal -- comparable to the electronic noise, and unlike it, uniform
   rather than Gaussian.  `quantisation_step()` recovers the printed
   resolution from the data itself, and the resulting variance q^2/12 is
   added to the per-point uncertainty instead of being ignored.

3. THE CHANNELS MAY BE SCANNED, NOT SAMPLED
   If the logger walks the channel list within a row, the segment and the
   cell voltage in "the same row" are not simultaneous.  That is a real skew,
   but unlike the FAMOS case it is *known* from the scan order and the row
   period rather than having to be measured -- pass `scan_rate_hz` and it is
   removed analytically.  Left at None nothing is applied and the assumption
   (simultaneous sampling) is recorded in the manifest rather than made
   silently.

SUPPORTED LAYOUTS  (`detect_dialect`, or force with cfg.csv_dialect)
--------------------------------------------------------------------
"records"     the bench record format, one line per segment per instant:
                  [t=1.234s;] s12: temp1=4.3V;...; i_s=0.5A;u_s=1.15V;
              This is the same syntax the Abgleich Step files use, which is
              why it is supported: the rig that writes those writes these.

"wide_time"   one row per instant, one column per channel:
                  time_s, u_cell, s1, s2, ... s72 [, temp1..temp4]
              Column names are matched loosely: "seg 12", "s12", "u_s12",
              "segment_12" all resolve to segment 12.

"long_time"   tidy: time_s, segment, u_s [, u_cell]

"freq"        already a spectrum, one row per (segment, frequency):
                  segment, freq_hz, z_re, z_im      (or |Z| and phase, or
                  Zreal/Zimag/Zmod/Zphz as Gamry names them)
              No phasor estimation happens at all on this path.

"gamry"       a folder of Gamry .DTA files, one per segment -- delegated to
              gamry_dta.py.

Every reader returns a `CsvMeasurement`, and `csv_pipeline` consumes only
that.  Adding a sixth layout means adding a reader here and nothing else.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# patterns
# --------------------------------------------------------------------------

_REC_SEG = re.compile(r"(?:^|[\t;,\s])s(?:eg)?[_\s]*(\d+)\s*:", re.I)
_REC_TEMP = re.compile(r"(temp\d)\s*=\s*([-\d.eE+]+)\s*V", re.I)
_REC_PAIR = re.compile(
    r"i_s\s*=\s*([-\d.eE+]+)\s*A\s*;\s*u_s\s*=\s*([-\d.eE+]+)\s*V", re.I)
_REC_U = re.compile(r"u_s\s*=\s*([-\d.eE+]+)\s*V", re.I)
_REC_T = re.compile(r"\bt\s*=\s*([-\d.eE+]+)\s*s", re.I)
_REC_UC = re.compile(r"\bu_?c(?:ell)?\s*=\s*([-\d.eE+]+)\s*V", re.I)

# column-name matching
_SEG_COL = re.compile(r"^(?:u_?s|s|seg|segment|ch|kanal)[\s_\-#]*(\d{1,2})$", re.I)
_TIME_COL = re.compile(r"^(t|time|zeit)(_?s|_?sec|_?seconds)?$", re.I)
_UCELL_COL = re.compile(r"^(u_?c(ell)?|ucell|u_zelle|cell_?v(oltage)?|uc)$", re.I)
_TEMP_COL = re.compile(r"^(temp|t)[\s_\-]*([1-4])$", re.I)
_FREQ_COL = re.compile(r"^(f|freq|frequency|frequenz)(_?hz)?$", re.I)
_ZRE_COL = re.compile(r"^(z_?re(al)?|re_?z|z'|zr|z_re_mohm_cm2|z_re_ohm_cm2)$", re.I)
_ZIM_COL = re.compile(r"^(z_?im(ag)?|im_?z|z''|zi|z_im_mohm_cm2|z_im_ohm_cm2)$", re.I)
_ZMOD_COL = re.compile(r"^(z_?mod|z_?mag|abs_?z|\|z\|)$", re.I)
_ZPHZ_COL = re.compile(r"^(z_?ph[zs]?|phase|phi|phase_deg)$", re.I)
_SEGID_COL = re.compile(r"^(seg(ment)?(_?id|_?no|_?nr)?|channel|kanal)$", re.I)

DIALECTS = ("records", "wide_time", "long_time", "freq", "gamry")


# --------------------------------------------------------------------------
# the common container
# --------------------------------------------------------------------------


@dataclass
class CsvMeasurement:
    """Whatever the CSV held, in the one shape csv_pipeline understands."""

    kind: str                                   # "time" | "frequency"
    dialect: str
    source: str

    # --- time domain -------------------------------------------------------
    t: np.ndarray | None = None                 # seconds, as recorded
    u_cell: np.ndarray | None = None            # V
    u_seg: dict[str, np.ndarray] = field(default_factory=dict)   # V per segment
    temps: dict[str, np.ndarray] = field(default_factory=dict)   # sensor volts

    # --- frequency domain --------------------------------------------------
    # segment -> (freq_hz, Z).  Units are carried in `z_unit`, never guessed
    # again downstream.
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    z_unit: str = "ohm_cm2"

    meta: dict = field(default_factory=dict)

    # --- derived -----------------------------------------------------------

    @property
    def segments(self) -> list[str]:
        keys = self.u_seg or self.spectra
        return sorted(keys, key=lambda s: int(s))

    @property
    def n_samples(self) -> int:
        return 0 if self.t is None else int(self.t.size)

    @property
    def fs_nominal(self) -> float:
        """Median sampling rate.  Median, not mean: one long gap in the
        record must not move the number that every frequency is judged
        against."""
        if self.t is None or self.t.size < 2:
            return float("nan")
        dt = np.diff(self.t)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if dt.size else float("nan")

    @property
    def duration_s(self) -> float:
        if self.t is None or self.t.size < 2:
            return float("nan")
        return float(self.t[-1] - self.t[0])

    def summary(self) -> dict:
        d = {"kind": self.kind, "dialect": self.dialect, "source": self.source,
             "n_segments": len(self.segments)}
        if self.kind == "time":
            d.update(n_samples=self.n_samples,
                     fs_nominal_hz=round(self.fs_nominal, 4),
                     duration_s=round(self.duration_s, 4),
                     has_u_cell=self.u_cell is not None,
                     n_temp_sensors=len(self.temps))
            d.update(regularity(self.t))
        else:
            n = [len(v[0]) for v in self.spectra.values()]
            d.update(z_unit=self.z_unit,
                     n_freq_median=int(np.median(n)) if n else 0,
                     f_min_hz=round(min(float(v[0].min())
                                        for v in self.spectra.values()), 6)
                     if self.spectra else None,
                     f_max_hz=round(max(float(v[0].max())
                                        for v in self.spectra.values()), 4)
                     if self.spectra else None)
        d.update(self.meta)
        return d


# --------------------------------------------------------------------------
# quality of the time base
# --------------------------------------------------------------------------


def regularity(t: np.ndarray | None) -> dict:
    """How uniform is the recorded time base?

    Returns the median interval, the relative jitter (robust sd over median)
    and the largest gap in units of the median interval.  `uniform` is the
    flag the phasor estimator branches on: below 0.1 % jitter a uniform-grid
    DFT and a fit on the true timestamps agree to well under the noise, and
    the uniform path is ~20x faster.
    """
    if t is None or t.size < 3:
        return {"uniform": True, "dt_median_s": float("nan"),
                "jitter_rel": 0.0, "max_gap_ratio": 1.0}
    dt = np.diff(t)
    med = float(np.median(dt))
    if med <= 0:
        return {"uniform": False, "dt_median_s": med, "jitter_rel": float("inf"),
                "max_gap_ratio": float("inf")}
    mad = float(np.median(np.abs(dt - med))) * 1.4826
    jit = mad / med
    gap = float(np.max(dt) / med)
    return {"uniform": bool(jit < 1e-3 and gap < 1.5),
            "dt_median_s": med, "jitter_rel": jit, "max_gap_ratio": gap}


def quantisation_step(x: np.ndarray) -> float:
    """Recover the printed resolution of a column from the data itself.

    The values in a text file live on a lattice of the last printed digit.
    The greatest common divisor of the differences between neighbouring
    distinct values IS that lattice step -- computed here as the smallest
    positive difference after sorting, which is the robust version (a true
    gcd is destroyed by one mis-rounded value).

    Returned as a step q; the variance it contributes is q^2/12, and that is
    what the uncertainty model adds.  Returns 0.0 when the data is finer than
    the noise, i.e. when quantisation is not the limit.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 32:
        return 0.0
    u = np.unique(np.round(x, 12))
    if u.size < 4:
        return 0.0
    d = np.diff(u)
    d = d[d > 0]
    if d.size == 0:
        return 0.0
    q = float(np.min(d))
    # If the smallest step is far below the spread, the record is not
    # quantisation-limited and reporting a step would overstate the effect.
    return q if q > 1e-4 * float(np.std(x)) else 0.0


# --------------------------------------------------------------------------
# phasor estimation on a possibly non-uniform grid
# --------------------------------------------------------------------------


def fit_phasor_nonuniform(t: np.ndarray, y: np.ndarray, f: float,
                          detrend: bool = True
                          ) -> tuple[complex, float, float]:
    """Least-squares sine fit at known f on ARBITRARY sample times.

    y ~ a cos(2 pi f t) + b sin(2 pi f t) + c [+ d*(t - tbar)]

    Returns (phasor, residual_rms, snr_db) with the phasor under exp(+j w t),
    i.e. y = Re{A e^{jwt}} + c and A = a - j b -- the same convention as
    utils.fit3, so the two are interchangeable.

    This is the estimator that makes a jittering logger usable.  Nothing in
    the normal equations requires t to be equally spaced; all that is needed
    is that t is *known*.  The linear term absorbs the slow drift of the
    operating point that a fuel cell always has, which otherwise leaks into
    the lowest tones.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n = t.size
    if n < 8 or not np.isfinite(f) or f <= 0:
        return complex("nan"), float("nan"), float("nan")
    w = 2.0 * np.pi * f
    cols = [np.cos(w * t), np.sin(w * t), np.ones(n)]
    if detrend:
        cols.append(t - t.mean())
    D = np.column_stack(cols)
    p, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ p
    A = complex(p[0], -p[1])
    r_rms = float(np.sqrt(np.mean(resid ** 2)))
    sig = abs(A) / np.sqrt(2.0)
    snr = 20 * np.log10(sig / r_rms) if r_rms > 0 else float("inf")
    return A, r_rms, float(snr)


def fit_phasor(t: np.ndarray, y: np.ndarray, f: float, uniform: bool,
               fs: float, detrend: bool = True):
    """Dispatch to the uniform-grid fit when that is safe, else to the
    timestamp fit.  Same signature and convention either way."""
    if uniform:
        import utils
        return utils.fit3(y, fs, f, detrend=detrend)
    return fit_phasor_nonuniform(t, y, f, detrend=detrend)


def fit_phasors_multi(t: np.ndarray, y: np.ndarray, freqs,
                      detrend: bool = True
                      ) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit EVERY tone of a multisine in one least-squares solve.

    Returns (phasors, residual_rms, snr_db per tone), same phasor convention
    as `fit_phasor_nonuniform`.

    Fitting the tones one at a time is not merely slower, it corrupts the
    uncertainty.  A single-tone fit to a twelve-tone record charges the other
    eleven tones to the residual, so the "noise" it reports is the rest of
    the excitation -- a number that does not change when the actual noise
    does, and that is therefore useless as a weight.  Fitting them jointly
    leaves a residual that really is noise, which is what makes the per-point
    sigma mean something and what makes a quantisation floor visible.

    It is also the leakage-free estimator: with all tones in one design
    matrix, a tone that is not an exact multiple of 1/T does not spill into
    its neighbours the way a windowed DFT bin does.  That is the entire
    reason a designed multisine is worth using.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    freqs = np.asarray(list(freqs), float)
    n = t.size
    if n < 4 * freqs.size + 8 or freqs.size == 0:
        return (np.full(freqs.shape, complex("nan")), float("nan"),
                np.full(freqs.shape, float("nan")))
    w = 2 * np.pi * freqs[None, :] * t[:, None]
    cols = [np.cos(w), np.sin(w), np.ones((n, 1))]
    if detrend:
        cols.append((t - t.mean())[:, None])
    D = np.hstack(cols)
    p, *_ = np.linalg.lstsq(D, y, rcond=None)
    m = freqs.size
    A = p[:m] - 1j * p[m:2 * m]
    resid = y - D @ p
    r_rms = float(np.sqrt(np.mean(resid ** 2)))
    sig = np.abs(A) / np.sqrt(2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 20 * np.log10(np.where(r_rms > 0, sig / max(r_rms, 1e-300),
                                     np.inf))
    return A, r_rms, snr


# --------------------------------------------------------------------------
# dialect detection
# --------------------------------------------------------------------------


def _sniff(path: Path, n_lines: int = 60) -> list[str]:
    out = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if line.strip():
                out.append(line)
            if len(out) >= n_lines:
                break
    return out


def _split(line: str) -> list[str]:
    """Split a header/data row on the separator that yields the most fields."""
    best, n = [line], 1
    for sep in (";", ",", "\t", "|"):
        parts = line.split(sep)
        if len(parts) > n:
            best, n = parts, len(parts)
    return [p.strip().strip('"') for p in best]


def detect_dialect(path) -> str:
    """Decide which layout `path` is, from its first lines.

    Ordered most-specific first.  A wrong guess here is loud, not silent:
    every reader validates what it found and raises with the head of the file
    attached, because a reader that half-succeeds on the wrong layout is how
    a plot ends up showing the temperature channel as segment 3.
    """
    p = Path(path)
    if p.is_dir():
        if any(p.glob("*.DTA")) or (p / "bode").is_dir():
            return "gamry"
        cands = sorted(p.glob("*.csv"))
        if not cands:
            raise ValueError(f"{p}: no .csv and no .DTA files")
        return detect_dialect(cands[0])
    if p.suffix.lower() == ".dta":
        return "gamry"

    lines = _sniff(p)
    if not lines:
        raise ValueError(f"{p}: empty file")

    # records: "s12:" plus "u_s=...V"
    hits = sum(1 for ln in lines if _REC_SEG.search(ln) and _REC_U.search(ln))
    if hits >= max(2, len(lines) // 4):
        return "records"

    header = _split(lines[0])
    low = [h.lower() for h in header]
    has_freq = any(_FREQ_COL.match(h) for h in low)
    has_z = (any(_ZRE_COL.match(h) for h in low)
             or any(_ZMOD_COL.match(h) for h in low))
    has_segid = any(_SEGID_COL.match(h) for h in low)
    has_time = any(_TIME_COL.match(h) for h in low)
    seg_cols = [h for h in low if _SEG_COL.match(h)]

    if has_freq and has_z:
        return "freq"
    if has_time and len(seg_cols) >= 2:
        return "wide_time"
    if has_time and has_segid:
        return "long_time"
    if len(seg_cols) >= 2:
        return "wide_time"          # no time column: assume uniform, warn later
    raise ValueError(
        f"{p.name}: cannot recognise the layout. Header was:\n  {header[:12]}\n"
        f"Expected one of: a 'freq'+'z_re' spectrum table, a time column with "
        f"per-segment columns, a tidy time/segment/u_s table, or the "
        f"'s<n>: ... u_s=...V' record format. "
        f"Set cfg.csv_dialect explicitly to override the guess.")


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------


def read_records(path) -> CsvMeasurement:
    """The bench record format: one `s<n>: ...` record per segment per instant.

    A record may carry `t=<seconds>s` and `uc=<volts>V`.  When it does not,
    instants are counted in file order (every segment seen once per instant)
    and the sample interval is taken from `meta['fs_hz']` if the caller knows
    it -- reported as an assumption, never silently invented.
    """
    path = Path(path)
    times: list[float] = []
    ucell: list[float] = []
    per_seg: dict[str, list[float]] = {}
    per_temp: dict[str, list[float]] = {}
    seen_this_instant: set[str] = set()
    n_instants = 0
    have_t = have_uc = False

    for raw in path.read_text(encoding="utf-8-sig",
                              errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _REC_SEG.search(line)
        if not m:
            continue
        seg = str(int(m.group(1)))
        um = _REC_U.search(line)
        if not um:
            continue

        if seg in seen_this_instant:
            n_instants += 1
            seen_this_instant = set()
        seen_this_instant.add(seg)

        per_seg.setdefault(seg, []).append(float(um.group(1)))

        tm = _REC_T.search(line)
        if tm:
            have_t = True
            if len(times) <= n_instants:
                times.append(float(tm.group(1)))
        ucm = _REC_UC.search(line)
        if ucm:
            have_uc = True
            if len(ucell) <= n_instants:
                ucell.append(float(ucm.group(1)))
        for k, v in _REC_TEMP.findall(line):
            k = k.lower()
            if len(per_temp.setdefault(k, [])) <= n_instants:
                per_temp[k].append(float(v))

    if not per_seg:
        raise ValueError(f"{path.name}: no 's<n>: ... u_s=...V' records found")

    n = min(len(v) for v in per_seg.values())
    u_seg = {k: np.asarray(v[:n], float) for k, v in per_seg.items()}
    t = np.asarray(times[:n], float) if have_t and len(times) >= n else None
    uc = np.asarray(ucell[:n], float) if have_uc and len(ucell) >= n else None
    temps = {k: np.asarray(v[:n], float) for k, v in per_temp.items()
             if len(v) >= n}

    meta = {"records_per_segment": n,
            "time_column": "t=..s" if t is not None else "absent (index order)",
            "u_cell": "uc=..V" if uc is not None else "absent"}
    return CsvMeasurement("time", "records", str(path), t=t, u_cell=uc,
                          u_seg=u_seg, temps=temps, meta=meta)


def _read_table(path) -> tuple[list[str], np.ndarray, list[list[str]]]:
    """Header, numeric block and the raw string rows (for non-numeric ids)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path.name}: empty")
    header = _split(lines[0])
    rows = [_split(ln) for ln in lines[1:]]
    w = len(header)
    rows = [r + [""] * (w - len(r)) if len(r) < w else r[:w] for r in rows]

    num = np.full((len(rows), w), np.nan)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            v = v.strip()
            if not v:
                continue
            try:
                num[i, j] = float(v.replace(",", ".") if v.count(",") == 1
                                  and "." not in v else v)
            except ValueError:
                pass
    return header, num, rows


def read_wide_time(path) -> CsvMeasurement:
    """time_s, u_cell, s1..s72 [, temp1..temp4] -- one row per instant."""
    header, num, _rows = _read_table(path)
    low = [h.lower() for h in header]

    t_idx = next((i for i, h in enumerate(low) if _TIME_COL.match(h)), None)
    uc_idx = next((i for i, h in enumerate(low) if _UCELL_COL.match(h)), None)
    seg_idx = {}
    for i, h in enumerate(low):
        m = _SEG_COL.match(h)
        if m:
            seg_idx[str(int(m.group(1)))] = i
    temp_idx = {}
    for i, h in enumerate(low):
        m = _TEMP_COL.match(h)
        if m:
            temp_idx[f"temp{int(m.group(2))}"] = i

    if not seg_idx:
        raise ValueError(
            f"{Path(path).name}: no per-segment columns recognised. "
            f"Header: {header[:12]}  Names like 's12', 'seg 12', 'u_s12' or "
            f"'segment_12' are understood.")

    t = num[:, t_idx] if t_idx is not None else None
    uc = num[:, uc_idx] if uc_idx is not None else None
    u_seg = {k: num[:, i] for k, i in seg_idx.items()}
    temps = {k: num[:, i] for k, i in temp_idx.items()}

    meta = {"time_column": header[t_idx] if t_idx is not None else
            "absent (rows assumed uniformly spaced)",
            "u_cell_column": header[uc_idx] if uc_idx is not None else "absent",
            "n_segment_columns": len(seg_idx)}
    return CsvMeasurement("time", "wide_time", str(path), t=t, u_cell=uc,
                          u_seg=u_seg, temps=temps, meta=meta)


def read_long_time(path) -> CsvMeasurement:
    """time_s, segment, u_s [, u_cell] -- tidy, one row per sample per segment."""
    header, num, rows = _read_table(path)
    low = [h.lower() for h in header]
    t_idx = next((i for i, h in enumerate(low) if _TIME_COL.match(h)), None)
    s_idx = next((i for i, h in enumerate(low) if _SEGID_COL.match(h)), None)
    uc_idx = next((i for i, h in enumerate(low) if _UCELL_COL.match(h)), None)
    u_idx = next((i for i, h in enumerate(low)
                  if h in ("u_s", "us", "u", "voltage", "u_v", "shunt_v")), None)
    if t_idx is None or s_idx is None or u_idx is None:
        raise ValueError(
            f"{Path(path).name}: expected time, segment and u_s columns; "
            f"header was {header[:12]}")

    segs = np.array([r[s_idx] for r in rows])
    keys = sorted({str(int(float(s))) for s in segs if s.strip()},
                  key=int)
    out: dict[str, np.ndarray] = {}
    t_ref = None
    uc = None
    for k in keys:
        sel = np.array([bool(s.strip()) and str(int(float(s))) == k
                        for s in segs])
        tk = num[sel, t_idx]
        o = np.argsort(tk)
        out[k] = num[sel, u_idx][o]
        if t_ref is None:
            t_ref = tk[o]
            if uc_idx is not None:
                uc = num[sel, uc_idx][o]
    n = min(len(v) for v in out.values())
    out = {k: v[:n] for k, v in out.items()}
    meta = {"n_segments": len(keys), "rows": int(num.shape[0])}
    return CsvMeasurement("time", "long_time", str(path),
                          t=None if t_ref is None else t_ref[:n],
                          u_cell=None if uc is None else uc[:n],
                          u_seg=out, meta=meta)


def read_freq(path, z_unit: str = "auto") -> CsvMeasurement:
    """A ready-made spectrum table: segment, freq_hz, and Z in some form.

    Accepts (z_re, z_im), (|Z|, phase_deg) or the Gamry names
    Zreal/Zimag/Zmod/Zphz.  `z_unit` may be "ohm", "ohm_cm2", "mohm_cm2" or
    "auto"; auto reads the median |Z| and picks the only interpretation that
    is physically possible for a PEMFC segment, and records what it chose.
    Override it whenever you know -- the guess is a convenience, not evidence.
    """
    header, num, rows = _read_table(path)
    low = [h.lower() for h in header]

    f_idx = next((i for i, h in enumerate(low) if _FREQ_COL.match(h)), None)
    s_idx = next((i for i, h in enumerate(low) if _SEGID_COL.match(h)), None)
    re_idx = next((i for i, h in enumerate(low) if _ZRE_COL.match(h)), None)
    im_idx = next((i for i, h in enumerate(low) if _ZIM_COL.match(h)), None)
    mod_idx = next((i for i, h in enumerate(low) if _ZMOD_COL.match(h)), None)
    phz_idx = next((i for i, h in enumerate(low) if _ZPHZ_COL.match(h)), None)

    if f_idx is None:
        raise ValueError(f"{Path(path).name}: no frequency column; "
                         f"header {header[:12]}")
    if re_idx is None and (mod_idx is None or phz_idx is None):
        raise ValueError(
            f"{Path(path).name}: need either (z_re, z_im) or (|Z|, phase); "
            f"header {header[:12]}")

    if re_idx is not None:
        Z = num[:, re_idx] + 1j * (num[:, im_idx] if im_idx is not None else 0.0)
    else:
        Z = num[:, mod_idx] * np.exp(1j * np.radians(num[:, phz_idx]))

    if s_idx is None:
        seg_of = np.array(["1"] * num.shape[0])
    else:
        seg_of = np.array([str(int(float(r[s_idx]))) if r[s_idx].strip() else ""
                           for r in rows])

    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for k in sorted({s for s in seg_of if s}, key=int):
        sel = seg_of == k
        f = num[sel, f_idx]
        z = Z[sel]
        good = np.isfinite(f) & np.isfinite(z.real) & np.isfinite(z.imag)
        o = np.argsort(f[good])
        spectra[k] = (f[good][o], z[good][o])

    # unit resolution, from a column name if it says so, else from magnitude
    unit = z_unit
    named = next((h for h in low if "mohm_cm2" in h), None)
    if unit == "auto" and named:
        unit = "mohm_cm2"
    if unit == "auto":
        med = float(np.nanmedian([np.abs(v[1]).max() for v in spectra.values()]))
        # A segment ASR is 0.05..2 ohm*cm2; the same number in mohm*cm2 is
        # 50..2000, and the raw impedance of a 5 cm2 segment is 0.01..0.4 ohm.
        unit = ("mohm_cm2" if med > 20 else
                "ohm_cm2" if med > 0.02 else "ohm")

    meta = {"unit_source": "given" if z_unit != "auto" else "inferred",
            "columns": header[:12]}
    return CsvMeasurement("frequency", "freq", str(path), spectra=spectra,
                          z_unit=unit, meta=meta)


def read_gamry(path, area_cm2: float | None = None) -> CsvMeasurement:
    """A folder of per-segment Gamry .DTA sweeps, or one file."""
    import gamry_dta
    p = Path(path)
    if p.is_dir():
        folder = p / "bode" if (p / "bode").is_dir() else p
        sweeps = gamry_dta.read_bode_folder(folder)
    else:
        s = gamry_dta.read_dta(p).sorted()
        sweeps = {s.segment or "1": s}
    if not sweeps:
        raise ValueError(f"{p}: no readable .DTA spectra")
    spectra = {k: (v.freq, v.Z) for k, v in sweeps.items()}
    meta = {"n_files": len(sweeps),
            "note": "Gamry .DTA holds ohms; multiply by the area to compare "
                    "with the pipeline's ohm*cm2"}
    return CsvMeasurement("frequency", "gamry", str(p), spectra=spectra,
                          z_unit="ohm", meta=meta)


_READERS = {
    "records": read_records,
    "wide_time": read_wide_time,
    "long_time": read_long_time,
    "freq": read_freq,
    "gamry": read_gamry,
}


def read(path, dialect: str = "auto", **kw) -> CsvMeasurement:
    """Read a CSV measurement, detecting the layout unless told which."""
    d = detect_dialect(path) if dialect in ("auto", "", None) else dialect
    if d not in _READERS:
        raise ValueError(f"unknown csv dialect {d!r}; known: {DIALECTS}")
    fn = _READERS[d]
    import inspect
    ok = {k: v for k, v in kw.items()
          if k in inspect.signature(fn).parameters}
    return fn(path, **ok)


# --------------------------------------------------------------------------
# self-test -- synthetic data through every reader
# --------------------------------------------------------------------------


def _selftest(log=None) -> int:
    """Round-trip a known spectrum through each layout.  Returns #failures."""
    def say(m):
        (log.info if log else print)(m)

    fails = 0
    rng = np.random.default_rng(7)
    fs, dur, f_tone = 500.0, 8.0, 7.0
    n = int(fs * dur)
    t = np.arange(n) / fs
    Z_true = 0.12 + 0.05j                       # ohm*cm2
    j_amp = 0.02                                # A/cm2
    K = 0.60                                    # V/(A/cm2)
    u_cell = np.abs(Z_true) * j_amp * np.cos(2 * np.pi * f_tone * t
                                             + np.angle(Z_true))
    u_seg = K * j_amp * np.cos(2 * np.pi * f_tone * t)

    # 1. uniform-grid fit recovers Z
    Au, _, _ = fit_phasor_nonuniform(t, u_cell, f_tone)
    As, _, _ = fit_phasor_nonuniform(t, u_seg, f_tone)
    Z = K * Au / As
    err = abs(Z - Z_true) / abs(Z_true)
    ok = err < 1e-6
    say(f"    uniform grid      : |dZ|/|Z| = {err:.2e}  {'PASS' if ok else 'FAIL'}")
    fails += not ok

    # 2. the same with 5 % jitter on the timestamps -- the whole point of the
    #    timestamp fit.  A uniform-grid DFT on this data is wrong; the fit is
    #    not.
    tj = np.sort(t + rng.normal(0, 0.05 / fs, n))
    ucj = np.abs(Z_true) * j_amp * np.cos(2 * np.pi * f_tone * tj
                                          + np.angle(Z_true))
    usj = K * j_amp * np.cos(2 * np.pi * f_tone * tj)
    Au, _, _ = fit_phasor_nonuniform(tj, ucj, f_tone)
    As, _, _ = fit_phasor_nonuniform(tj, usj, f_tone)
    err = abs(K * Au / As - Z_true) / abs(Z_true)
    ok = err < 1e-6
    say(f"    jittered timestamps: |dZ|/|Z| = {err:.2e}  {'PASS' if ok else 'FAIL'}")
    fails += not ok

    # 3. quantisation step recovery
    q_true = 1e-6
    xq = np.round(u_seg / q_true) * q_true
    q = quantisation_step(xq)
    ok = abs(q - q_true) / q_true < 1e-6
    say(f"    quantisation step  : {q:.3e} (true {q_true:.0e})  "
        f"{'PASS' if ok else 'FAIL'}")
    fails += not ok

    # 4. every text layout parses back to the same segments
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        wide = td / "wide.csv"
        with wide.open("w") as fh:
            fh.write("time_s,u_cell,s1,s2,temp1\n")
            for i in range(200):
                fh.write(f"{t[i]:.6f},{u_cell[i]:.9f},{u_seg[i]:.9f},"
                         f"{0.5*u_seg[i]:.9f},3.45\n")
        m = read(wide)
        ok = (m.dialect == "wide_time" and m.segments == ["1", "2"]
              and m.u_cell is not None and "temp1" in m.temps)
        say(f"    wide_time reader   : {m.summary()['n_segments']} segments  "
            f"{'PASS' if ok else 'FAIL'}")
        fails += not ok

        rec = td / "rec.csv"
        with rec.open("w") as fh:
            for i in range(200):
                for s in (1, 2):
                    fh.write(f"t={t[i]:.6f}s\tuc={u_cell[i]:.9f}V\t"
                             f"s{s}:\ttemp1=3.45V;\ti_s=0.1A;"
                             f"u_s={u_seg[i]/s:.9f}V;\n")
        m = read(rec)
        ok = (m.dialect == "records" and m.segments == ["1", "2"]
              and m.t is not None and m.u_cell is not None)
        say(f"    records reader     : {m.summary()['n_segments']} segments  "
            f"{'PASS' if ok else 'FAIL'}")
        fails += not ok

        frq = td / "spec.csv"
        with frq.open("w") as fh:
            fh.write("segment,freq_hz,z_re,z_im\n")
            for s in (1, 2):
                for f_ in (0.1, 1.0, 10.0, 100.0):
                    fh.write(f"{s},{f_},{Z_true.real},{-Z_true.imag}\n")
        m = read(frq)
        ok = (m.dialect == "freq" and m.kind == "frequency"
              and m.z_unit == "ohm_cm2" and len(m.spectra) == 2)
        say(f"    freq reader        : unit={m.z_unit}  "
            f"{'PASS' if ok else 'FAIL'}")
        fails += not ok

    return int(fails)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        m = read(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "auto")
        import json
        print(json.dumps(m.summary(), indent=2, default=str))
    else:
        raise SystemExit(_selftest())
