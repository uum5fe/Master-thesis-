#!/usr/bin/env python3
"""
bronze.py  --  LAYER 1 of 3:  raw ingestion, no physics
=======================================================

CONTRACT
--------
Bronze reads FAMOS recordings and produces RAW PHASORS with full provenance.
It applies no correction that could ever need to be undone: no de-skew, no
inductance removal, no Kramers-Kronig projection, no smoothing.  Everything
bronze writes is either measured or a direct algebraic consequence of the
calibration file.

The reason for the discipline is that the legacy pipeline mixed acquisition
and correction in one pass, so a change to the skew model meant re-reading
40 GB of .DAT files.  Here silver can be re-run in seconds against a cached
bronze table, which is what makes the skew model falsifiable in practice.

WHAT BRONZE PRODUCES
--------------------
    BronzeRun
      .schedule       consensus list of excitation steps (frequency + window)
      .channels       every channel on every card, with its ACQUISITION SLOT
      .spectra        {segment: BronzeSpectrum}
      .cards          per-card metadata (fs, n_ch, duration, reference used)
      .calibration    the loaded Abgleich, echoed for provenance

    BronzeSpectrum
      freq, Z_raw, snr_ref_db, snr_seg_db, snr_comb_db, thd, drift,
      n_samples_per_step, channel_slot, card, K, T_degC, u_dc, on_grid

TWO THINGS BRONZE DOES THAT THE OLD CODE DID NOT
------------------------------------------------
1.  IT RECORDS THE ACQUISITION SLOT OF EVERY CHANNEL.
    The FAMOS header order is the order in which a multiplexed converter
    digitises the channels.  That index is the single most useful number for
    fixing the high-frequency phase, and the old pipeline read it only to
    print a parity "cross-check".  Silver turns it into a structural skew
    model; bronze's job is simply never to lose it.

2.  IT BUILDS ONE CONSENSUS SCHEDULE FOR THE WHOLE PLATE.
    Detecting the sweep independently on each card gave each card a slightly
    different frequency for the same physical step, and those differences do
    not cancel in a plate map.  Here every reference channel votes, the votes
    are collapsed onto the sweep's own geometric grid, and one frequency per
    step is used everywhere.

REFERENCES
----------
IEEE Std 1057-2017, clause 7      -- three/four-parameter sine fitting
P. M. Ramos, A. Cruz Serra, Measurement 41 (2008) 135  -- joint two-channel fit
J. J. Giner-Sanz et al., Electrochim. Acta 186 (2015) 598  -- THD linearity test
J. C. Brown, J. Acoust. Soc. Am. 89 (1991) 425  -- constant-Q ridge extraction
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:                      # package layout: core/ holds the science modules
    import core           # noqa: F401  (puts the core dir on sys.path)
except ImportError:       # flat layout (Databricks, notebooks): already there
    pass
import r2d2_geometry as geom
from eis_local import (open_famos, PlateCalibration, Step, detect_schedule,
                       detect_multisine_schedule, fit_multitone,
                       group_simultaneous, classify_excitation)

import utils
from config import Config, DEFAULT, KNOWN_BAD_SEGMENTS, T_FALLBACK_C


# ===========================================================================
# 1. Containers
# ===========================================================================


@dataclass
class ChannelInfo:
    """One physical channel on one card.

    `slot` is the position in the FAMOS header, which for a multiplexed
    converter is the conversion order within a sample period.  `slot` is the
    input to the structural skew model in silver.py -- keep it exact.
    """

    name: str
    card: str
    slot: int
    n_ch_on_card: int
    kind: str                 # "segment" | "reference" | "temperature" | "other"
    fs: float

    @property
    def slot_seconds(self) -> float:
        """Nominal time offset of this channel inside one sample period."""
        return self.slot / (self.n_ch_on_card * self.fs)


@dataclass
class BronzeSpectrum:
    """Raw spectrum of one segment: what the instrument actually reported."""

    segment: str
    card: str
    freq: np.ndarray
    Z_raw: np.ndarray            # ohm*cm^2, K * A_ref / A_seg, NO corrections
    snr_ref_db: np.ndarray
    snr_seg_db: np.ndarray
    snr_comb_db: np.ndarray
    thd: np.ndarray
    drift: np.ndarray
    n_per_step: np.ndarray       # samples in each dwell -> feeds the CRLB
    on_grid: np.ndarray          # step lies on the sweep's geometric grid
    channel_slot: int
    ref_slot: int
    n_ch_on_card: int
    fs: float
    K: float                     # transfer coefficient V/(A/cm^2) at T
    K_imputed: bool              # True if the calibration row was missing
    T_degC: float
    u_dc: float                  # DC level on the segment channel, V
    ref_name: str

    @property
    def n_points(self) -> int:
        return int(np.sum(np.isfinite(self.Z_raw)))

    @property
    def slot_delta_seconds(self) -> float:
        """Nominal reference-to-segment acquisition skew from the mux order.

        A multiplexed converter samples slot k at t = n/fs + k/(n_ch*fs).
        The impedance is built from A_ref / A_seg, so the ratio carries
        exp(-j w dt) with dt = (slot_seg - slot_ref)/(n_ch*fs).

        At 10 kHz with 16 channels one slot is 6.25 us; a 10-slot separation
        is 62.5 us, which is 67 degrees at 3 kHz.  This is not a small
        correction and it is DIFFERENT FOR EVERY SEGMENT -- which is exactly
        the high-frequency fan-out the old plots showed.
        """
        return (self.channel_slot - self.ref_slot) / (self.n_ch_on_card * self.fs)

    def j_dc(self) -> float:
        """DC current density from the operating point, A/cm^2."""
        return self.u_dc / self.K if self.K else float("nan")


@dataclass
class CardInfo:
    path: Path
    stem: str
    fs: float
    n_ch: int
    n_samples: int
    duration_s: float
    ref_name: str
    ref_slot: int
    n_segments: int
    sensor_T: dict[str, float] = field(default_factory=dict)
    #: |NT trigger stamp, or None if the dialect carries none.  Used to
    #: CHECK the measured alignment, never to perform it.
    start_time: "datetime | None" = None


@dataclass
class BronzeRun:
    """Everything bronze knows.  Serialisable, cacheable, re-runnable."""

    schedule: list[Step]
    channels: dict[str, ChannelInfo]
    spectra: dict[str, BronzeSpectrum]
    cards: dict[str, CardInfo]
    grid: dict
    config_digest: str
    input_digest: str
    n_files: int
    lags: dict = field(default_factory=dict)
    sensor_T: dict = field(default_factory=dict)
    #: TimebaseReport.summary() -- what the headers said before alignment.
    timebase: dict = field(default_factory=dict)
    #: CalibrationReport.summary() -- what was calibrated vs substituted.
    calibration: dict = field(default_factory=dict)

    def segments_measured(self) -> list[str]:
        return sorted(self.spectra, key=int)

    def segments_missing(self) -> list[str]:
        got = set(self.spectra)
        return sorted((set(geom.SEGMENTS) - got), key=int)

    def summary(self) -> dict:
        meas = self.segments_measured()
        return {
            "n_files": self.n_files,
            "n_steps": len(self.schedule),
            "f_min_hz": float(min(s.freq for s in self.schedule)) if self.schedule else None,
            "f_max_hz": float(max(s.freq for s in self.schedule)) if self.schedule else None,
            "n_segments_measured": len(meas),
            "n_segments_missing": len(self.segments_missing()),
            "segments_missing": self.segments_missing(),
            "cards": {k: v.stem for k, v in self.cards.items()},
            "grid": {k: v for k, v in self.grid.items()
                     if k in ("ok", "ppd", "f0", "n_on_grid", "n_total")},
            "config_digest": self.config_digest,
            "input_digest": self.input_digest,
            # The card's OWN sample rate, not a constant.  This divided by a
            # hard-coded 10000.0, so every lag reported for a plate recorded
            # at any other rate was wrong by exactly that ratio -- silently,
            # and in the one number a reader would use to sanity-check the
            # alignment against the header stamps.
            "card_lag_s": {k: round(v["lag"] / self.cards[k].fs, 6)
                           for k, v in self.lags.items() if k in self.cards},
            "card_lag_corr": {k: round(v["corr"], 4)
                              for k, v in self.lags.items()},
            "card_lag_applied": {k: bool(v.get("applied"))
                                 for k, v in self.lags.items()},
            "sensor_T_degC": {k: round(v, 3) for k, v in self.sensor_T.items()},
            "timebase": self.timebase,
            "calibration": self.calibration,
        }


# ===========================================================================
# 2. Discovery and inventory
# ===========================================================================


#: A current setpoint inside a filename: "45A", "450 A", "1.5A". Used only
#: when no known pattern matched, to find out whether a folder holds one
#: condition or several before deciding what to do.
_CURRENT_IN_NAME = re.compile(r"\d+(?:[.,]\d+)?\s*A(?![A-Za-z])")


def discover_files(cfg: Config) -> list[Path]:
    """FAMOS files for THIS RUN, sorted so card order is deterministic.

    Every filename convention is tried before any fallback, because the
    fallback is dangerous: a campaign folder holds every condition, and
    "any .DAT in the directory" silently turns a request for 45A into a run
    over 45A, 60A, 150A and 450A at once -- four times the data, re-read at
    every stage, over a network share. That is a run that never finishes, and
    it looks like a hang rather than a mistake.

    So the fallback still filters on the condition, and if it cannot -- if the
    files that remain describe more than one condition -- this refuses rather
    than guessing. Reading the wrong data slowly is worse than stopping.
    """
    d = Path(cfg.dat_dir)
    if not d.exists():
        raise SystemExit(f"bronze: --dat directory does not exist: {d}")

    patterns = cfg.famos_patterns()
    files: list[Path] = []
    for pattern in patterns:
        files = sorted(d.glob(pattern))
        if files:
            return files

    # Nothing matched a known convention. Take what is there, but keep the
    # condition: the run was asked for one.
    every = sorted(d.glob("*.DAT")) + sorted(d.glob("*.dat"))
    if not every:
        raise SystemExit(
            f"bronze: no FAMOS files in {d}\n"
            f"  tried: {', '.join(patterns)}")

    cond = cfg.condition
    if not cond or cond == "ALL":
        return every

    wanted = [f for f in every
              if f"_Current_{cond}_".lower() in f.name.lower()]
    if wanted:
        return wanted

    # Still no filter. Before widening, find out what is actually in the
    # folder: any "<number>A" in a name is a current setpoint, whatever the
    # convention around it.
    seen = sorted({m.group(0).upper().replace(" ", "")
                   for m in (_CURRENT_IN_NAME.search(f.name) for f in every)
                   if m})
    if seen == [cond.upper()]:
        return every                       # one condition, and it is the one
    raise SystemExit(
        f"bronze: none of the known filename patterns matched in {d}, and the "
        f"files there cannot be narrowed to the requested condition.\n"
        f"  asked for : {cond}\n"
        f"  folder has: {', '.join(seen) if seen else 'no recognisable condition'}"
        f"  ({len(every)} .DAT files)\n"
        f"  tried     : {', '.join(patterns)}\n"
        f"  Processing all of them would read every condition at once, which "
        f"is four times the data and does not finish. Point --dat at one "
        f"condition's files, or set EIS_FAMOS_REGEX for this naming scheme.")


def _digest(items) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(str(it).encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def inventory_channels(files: list[Path], cfg: Config, log=None,
                       max_samples: int | None = None
                       ) -> tuple[dict[str, ChannelInfo], dict[str, CardInfo]]:
    """Open every card once and record what is on it, including slot order.

    `max_samples` bounds the read used to pick the reference channel.  That
    choice is "which UC channel carries the most AC content", and a few
    seconds settles it as well as the whole recording does -- but reading
    the whole recording of every card to decide it is what makes a header
    survey as expensive as an evaluation over a network share. bronze leaves
    it unbounded; preflight sets it.
    """
    log = log or utils.get_logger(cfg.verbose)
    channels: dict[str, ChannelInfo] = {}
    cards: dict[str, CardInfo] = {}

    for fp in files:
        fam = open_famos(fp)
        stem = fp.stem
        if not fam.uc_names:
            log.warning(f"  {fp.name}: no UC reference channel - card skipped")
            continue

        # the reference is the UC channel carrying the most AC content
        n_look = (fam.n_samples if max_samples is None
                  else min(fam.n_samples, int(max_samples)))
        ref = max(fam.uc_names,
                  key=lambda c: float(np.std(
                      fam.channel(c, 0, n_look)[::cfg_stride(cfg)])))
        ref_slot = fam.position(ref)

        for name in fam.names:
            if re.fullmatch(r"\d+", name):
                kind = "segment"
            elif name.upper().startswith("UC"):
                kind = "reference"
            elif name.lower().startswith("temp"):
                kind = "temperature"
            else:
                kind = "other"
            key = f"{stem}::{name}"
            channels[key] = ChannelInfo(name=name, card=stem,
                                        slot=fam.position(name),
                                        n_ch_on_card=fam.n_ch,
                                        kind=kind, fs=fam.fs)

        cards[stem] = CardInfo(
            path=fp, stem=stem, fs=fam.fs, n_ch=fam.n_ch,
            n_samples=fam.n_samples, duration_s=fam.n_samples / fam.fs,
            ref_name=ref, ref_slot=ref_slot,
            n_segments=len(fam.segment_names),
            start_time=fam.start_time,
        )
        stamp = (fam.start_time.strftime("%Y-%m-%d %H:%M:%S")
                 if fam.start_time else "no |NT stamp")
        log.info(f"  {fp.name}: {fam.fs:.0f} Hz, {fam.n_ch} ch, "
                 f"{fam.n_samples/fam.fs:.1f} s, ref={ref} (slot {ref_slot}), "
                 f"{len(fam.segment_names)} segments, started {stamp}")
    if not cards:
        raise SystemExit("bronze: no card carried a usable reference channel")
    return channels, cards


def cfg_stride(cfg: Config) -> int:
    """Decimation used for cheap statistics (std, DC level)."""
    return 10


# ===========================================================================
# 2a. Time base  (ARE THESE FIVE CARDS EVEN THE SAME RUN?)
# ===========================================================================


@dataclass
class TimebaseReport:
    """What the headers say about when each card was recording.

    WHY THIS RUNS BEFORE ANYTHING IS EVALUATED
    ------------------------------------------
    The alignment in `estimate_card_lags` is a cross-correlation, and a
    cross-correlation only ever sees sample indices.  Hand it two cards that
    recorded DIFFERENT runs an hour apart and it does not fail: it returns
    the lag of best agreement between two unrelated records, with a
    prominence score computed against that same unrelated background.  The
    number looks like an answer.  Nothing downstream can tell it is not one,
    because by the time silver sees the result the only symptom is segments
    failing an SNR gate -- which is what a wrong lag always looks like.

    The `|NT` header stamps are the independent statement that catches this.
    They are coarse (see `header_offsets_s` below) and are never used to
    align anything.  They are used to answer three questions the correlation
    cannot answer about itself:

      1. Do the cards even share a sample rate?  A lag in SAMPLES is
         meaningless between two cards clocked differently, and the search
         window is sized from one card's fs for all of them.
      2. Is the stagger within reach of the search window?  Arming five
         cards by hand puts seconds between them; `align_max_lag_s` bounds
         what the correlation may return.  A true offset outside that bound
         cannot be found, and the peak inside it is noise.
      3. Were all five cards recording at the same time at all?  Started one
         by one, an early card can stop before a late one starts.  Their
         common window is where the excitation must live, and if it is
         empty there is nothing to align.

    HOW COARSE ARE THE STAMPS
    -------------------------
    Coarse enough that this must not gate on small disagreements.  On the
    45 A set the stamps read 07:45:46 / 46 / 49 / 48 / 48 while the measured
    offsets were 0 / 0.0002 / 5.7120 / 2.5358 / 2.5491 s: the stamp spacing
    understates the true one by up to 2.7 s.  So the stamps are checked for
    ORDER and for gross disagreement, never for agreement to the second.
    """

    fs: dict[str, float]
    stamps: dict[str, "datetime | None"]
    #: Start of each card relative to the earliest stamped card, in seconds.
    header_offsets_s: dict[str, float]
    #: Seconds during which every card was recording simultaneously, from the
    #: stamps and durations.  NaN when the stamps do not support the sum.
    overlap_s: float
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def stamped(self) -> dict[str, "datetime"]:
        return {k: v for k, v in self.stamps.items() if v is not None}

    @property
    def header_spread_s(self) -> float:
        """Seconds between the first and last card being armed."""
        off = list(self.header_offsets_s.values())
        return (max(off) - min(off)) if len(off) >= 2 else 0.0

    def summary(self) -> dict:
        return {
            "fs_hz": dict(self.fs),
            "start_times": {k: (v.isoformat() if v else None)
                            for k, v in self.stamps.items()},
            "header_offsets_s": dict(self.header_offsets_s),
            "header_spread_s": self.header_spread_s,
            "overlap_s": self.overlap_s,
            "ok": self.ok,
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


def timebase_report(cards: dict[str, CardInfo], cfg: Config,
                    log=None) -> TimebaseReport:
    """Check, from the headers alone, that these cards belong to one run."""
    log = log or utils.get_logger(cfg.verbose)
    utils.section("time base (header stamps, before any alignment)", log)

    fs = {stem: c.fs for stem, c in cards.items()}
    stamps = {stem: c.start_time for stem, c in cards.items()}
    problems: list[str] = []
    notes: list[str] = []

    # ---- 1. one sample rate -------------------------------------------
    rates = sorted(set(round(v, 6) for v in fs.values()))
    if len(rates) > 1:
        # Not a warning.  estimate_card_lags sizes its search window from
        # one card's fs and returns a lag in SAMPLES that process_card then
        # applies to a different card; if the rates differ, that lag means a
        # different amount of time on each card and every dwell window it
        # places is wrong by a factor.
        per_rate = {r: [s for s, v in fs.items() if round(v, 6) == r]
                    for r in rates}
        detail = "; ".join(f"{r:.0f} Hz: {', '.join(sorted(c))}"
                           for r, c in per_rate.items())
        problems.append(
            f"the cards do not share a sample rate ({detail}). A lag measured "
            f"in samples is not the same amount of time on two cards clocked "
            f"differently, so the alignment cannot be applied. Re-record with "
            f"one rate, or split the run per rate.")
    else:
        log.info(f"  sample rate: {rates[0]:.0f} Hz on all "
                 f"{len(fs)} card(s)  OK")

    # ---- 2. stamps present --------------------------------------------
    stamped = {k: v for k, v in stamps.items() if v is not None}
    if not stamped:
        notes.append(
            "no card carries a |NT trigger stamp, so the measured alignment "
            "cannot be cross-checked against anything. It will be applied on "
            "its own evidence.")
        log.warning("  no |NT stamps: alignment will not be cross-checked")
        return TimebaseReport(fs=fs, stamps=stamps, header_offsets_s={},
                              overlap_s=float("nan"), problems=problems,
                              notes=notes)
    if len(stamped) < len(stamps):
        missing = sorted(set(stamps) - set(stamped))
        notes.append(f"no |NT stamp on: {', '.join(missing)} — those cards "
                     f"are not covered by the time-base check")

    t0 = min(stamped.values())
    offsets = {k: (v - t0).total_seconds() for k, v in stamped.items()}
    for stem in sorted(offsets, key=lambda s: offsets[s]):
        log.info(f"  {stem[-8:]}: armed {stamps[stem]:%Y-%m-%d %H:%M:%S}  "
                 f"{offsets[stem]:+8.1f} s   "
                 f"{cards[stem].duration_s:7.1f} s long")

    # ---- 3. same run at all --------------------------------------------
    spread = max(offsets.values()) - min(offsets.values())
    longest = max(cards[s].duration_s for s in stamped)
    if spread > longest:
        problems.append(
            f"the cards were armed {spread:.1f} s apart but the longest "
            f"recording is only {longest:.1f} s. These files cannot be one "
            f"event — check that every card in this folder belongs to the "
            f"same measurement.")

    # ---- 4. within reach of the search window --------------------------
    if spread > cfg.align_max_lag_s:
        problems.append(
            f"the header stamps put {spread:.1f} s between the first and last "
            f"card, but align_max_lag_s is {cfg.align_max_lag_s:.1f} s. The "
            f"true offset lies outside the search window, so the correlation "
            f"peak found inside it is not it. Raise align_max_lag_s above "
            f"{spread + 5:.0f} and re-run.")
    elif spread > 0.5 * cfg.align_max_lag_s:
        notes.append(
            f"the stamps span {spread:.1f} s against a search window of "
            f"±{cfg.align_max_lag_s:.1f} s. The stamps understate the true "
            f"stagger (by up to 2.7 s on the 45 A set), so this is close "
            f"enough to the limit to be worth raising align_max_lag_s.")

    # ---- 5. the window they share --------------------------------------
    starts = {k: offsets[k] for k in stamped}
    ends = {k: offsets[k] + cards[k].duration_s for k in stamped}
    overlap = min(ends.values()) - max(starts.values())
    if overlap <= 0:
        first_out = min(ends, key=lambda k: ends[k])
        last_in = max(starts, key=lambda k: starts[k])
        problems.append(
            f"the cards were never all recording at once: {first_out} stopped "
            f"{-overlap:.1f} s before {last_in} started. Started one by one, "
            f"an early card can finish before a late one is armed — there is "
            f"no common window for the excitation to sit in.")
    else:
        log.info(f"  common window: {overlap:.1f} s of the "
                 f"{longest:.1f} s longest recording")
        if overlap < 0.5 * longest:
            notes.append(
                f"only {overlap:.1f} s of {longest:.1f} s is common to every "
                f"card. Any step of the sweep outside that window is measured "
                f"on some cards and not others, and will be carried by fewer "
                f"votes in the consensus schedule.")

    for problem in problems:
        log.error(f"  REFUSED: {problem}")
    for note in notes:
        log.warning(f"  note: {note}")
    if not problems:
        log.info("  time base consistent with one run  OK")

    return TimebaseReport(fs=fs, stamps=stamps, header_offsets_s=offsets,
                          overlap_s=float(overlap) if stamped else float("nan"),
                          problems=problems, notes=notes)

# ===========================================================================
# 2b. Card alignment  (THE CARDS ARE ARMED SEPARATELY)
# ===========================================================================


def _energy_band(traces: dict[str, np.ndarray], fs: float, cfg: Config,
                 log=None) -> tuple[float, float]:
    """The band the excitation actually occupies, pooled over every card.

    WHY THIS IS MEASURED RATHER THAN FIXED
    --------------------------------------
    This used to be the literal band 0.5 .. 300 Hz, which is the right answer
    for a sweep that ends at 250 Hz and the wrong one for anything else.  A
    campaign whose tones live above 300 Hz has its ENTIRE excitation removed
    by that filter, and what then goes into the cross-correlation is the
    noise that survived it.  The correlation does not fail: it returns the
    best agreement between two noise records, and its prominence is scored
    against that same noise, so a run can come back "aligned" with lags that
    are pure chance.  Downstream this is indistinguishable from a card whose
    dwell windows are simply on the wrong tone, which is how it hides.

    The band is derived from the data instead: pool the cards' reference
    spectra and keep the span that carries the bulk of the AC energy.

    ONE BAND FOR EVERY CARD, NOT ONE PER CARD
    -----------------------------------------
    A cross-correlation between two differently filtered signals has its peak
    displaced by the difference of the two filters' group delays.  The whole
    measurement is that peak's position, so both traces must go through the
    identical filter.  Pooling the spectra gives one band that every card is
    then filtered with.
    """
    lo_floor = float(cfg.align_band_lo_hz)
    hi_ceil = min(float(cfg.align_band_hi_hz), 0.4 * fs)
    if hi_ceil <= lo_floor:
        return lo_floor, max(lo_floor * 2.0, 0.4 * fs)

    # A Welch-style average over a bounded number of frames: the full-record
    # spectrum of a 100 kHz card is tens of millions of bins, and the band
    # edges do not need that resolution.
    nper = 1 << int(np.ceil(np.log2(max(1024.0, fs))))   # ~1 s frames
    acc = None
    n_frames = 0
    for x in traces.values():
        if len(x) < nper:
            continue
        step = max(nper, (len(x) - nper) // 24 or nper)
        win = np.hanning(nper)
        for a in range(0, len(x) - nper + 1, step):
            spec = np.abs(np.fft.rfft((x[a:a + nper] - x[a:a + nper].mean())
                                      * win)) ** 2
            acc = spec if acc is None else acc + spec
            n_frames += 1
    if acc is None or n_frames == 0:
        return lo_floor, hi_ceil

    freqs = np.fft.rfftfreq(nper, 1.0 / fs)
    inside = (freqs >= lo_floor) & (freqs <= hi_ceil)
    if not inside.any() or acc[inside].sum() <= 0:
        return lo_floor, hi_ceil

    power = acc[inside]
    f_in = freqs[inside]
    cum = np.cumsum(power) / power.sum()
    lo = float(f_in[int(np.searchsorted(cum, cfg.align_band_quantile))])
    hi = float(f_in[int(min(len(f_in) - 1,
                            np.searchsorted(cum, 1.0 - cfg.align_band_quantile)))])
    # A band narrower than a factor of two makes a broad, ambiguous
    # correlation peak; widen it around the energy it did find.
    if hi < 2.0 * lo:
        centre = np.sqrt(max(lo, 1e-9) * max(hi, 1e-9))
        lo, hi = centre / 2.0, centre * 2.0
    lo = max(lo_floor, lo)
    hi = min(hi_ceil, max(hi, lo * 2.0))
    if log is not None:
        log.info(f"  alignment band: {lo:.2f} .. {hi:.1f} Hz "
                 f"(measured from the reference channels, not assumed)")
    return lo, hi


def _decimation(fs: float, band_hi: float) -> int:
    """Search stride: enough rate to keep the band, no more.

    The coarse lag places a dwell window that is tenths of a second long, so
    it needs to be right to about a millisecond -- not to one sample of a
    100 kHz record.  Searching at the full rate costs an FFT over twice the
    record for every ORDERED PAIR of cards (the anchor vote alone is
    n*(n-1) of them), which at 100 kHz over two minutes is tens of millions
    of points, twenty times over.  Decimating to just above Nyquist for the
    band that survives the filter changes no answer and is what makes a
    high-rate run finish.  The lag is refined at the full rate afterwards.
    """
    usable = max(band_hi, 1.0)
    q = int(max(1, np.floor(fs / (4.0 * usable))))
    return max(1, min(q, 64))


def estimate_card_lags(files: list[Path], cards: dict[str, CardInfo],
                       cfg: Config, log=None,
                       timebase: "TimebaseReport | None" = None
                       ) -> dict[str, dict]:
    """Sample offset of every card relative to the first, from the reference.

    WHY THIS EXISTS
    ---------------
    Each Dewetron card is armed by its own trigger, so sample n on card 3 is
    not the same instant as sample n on card 1.  On the 45 A set the header
    timestamps differ by whole seconds (07:45:46 / 46 / 49 / 48 / 48) and the
    TRUE offsets, recovered here, are 0, +0.0002, +5.7120, +2.5358, +2.5491 s.
    The header stamps have 1 s resolution and understate them.

    That matters because the excitation schedule is stored as sample indices.
    A window that is a genuine 0.25 s dwell on card 1 lands on idle record --
    or on a different tone entirely -- on card 3.  Measured effect: card 3
    kept 7 of 50 steps with the windows as-is and 15 of 50 once shifted.

    HOW
    ---
    Every card carries a copy of the same cell-voltage reference, so a plain
    cross-correlation of the band-passed reference gives the offset directly.
    The correlation is dominated by the excitation window, which is where the
    signal is, so the estimate is well determined.  The band is measured from
    the data (`_energy_band`) rather than fixed, because a fixed band that
    misses the excitation turns this into a correlation of noise.

    WHICH CARD IS THE ANCHOR
    ------------------------
    Not simply the first one.  A card whose reference channel is degraded
    makes a bad anchor: every other card then correlates weakly against it,
    and a gate on peak HEIGHT refuses them all.  That is not hypothetical --
    on the 45 A set card 1's reference yields 2 detectable steps where the
    others yield 44-49, and anchoring on it scored every card at |r| ~ 0.27.
    So the anchor is the card the others agree with best, chosen by the
    median cross-correlation each candidate achieves against the rest.

    WHY THE PEAK IS JUDGED BY PROMINENCE, NOT HEIGHT
    ------------------------------------------------
    Absolute |r| carries almost no information about whether a lag is right.
    On the 45 A set the KNOWN-correct near-zero lag of card 2 scores 0.269 --
    indistinguishable from the 0.276-0.278 of cards 3/4/5, whose lags are
    also known to be correct because they reproduce the TRUE offsets recorded
    above to four decimals.  A fixed threshold of 0.30 therefore refused four
    correct answers.  What actually separates a real lag from noise is how
    far the peak stands above the rest of the correlation curve, which is
    scale-free: a sharp peak at 0.27 on a flat background is certain, a
    ragged 0.5 is not.  `align_min_corr` stays only as an absolute floor
    against pure garbage.

    THE HEADER CROSS-CHECK
    ----------------------
    Prominence says the peak is sharp; it does not say the peak is the right
    one.  When `timebase` carries |NT stamps, the ORDER the cards were armed
    in is compared against the order the measured lags imply.  Order rather
    than value, because the stamps understate the true stagger by seconds
    (above) -- but they cannot get the sequence wrong, and a measured lag
    that contradicts it is not a small error, it is a different peak.

    Returns {stem: {"lag", "corr", "prominence", "applied"}}.
    """
    log = log or utils.get_logger(cfg.verbose)
    utils.section("card alignment (shared reference cross-correlation)", log)

    stems = [fp.stem for fp in files if fp.stem in cards]
    if not stems:
        return {}

    # One rate for the whole plate. timebase_report has already refused a
    # mixed-rate run; this is the assertion that the refusal was honoured,
    # because every lag below is a sample count applied to another card.
    rates = sorted(set(round(cards[s].fs, 6) for s in stems))
    if len(rates) > 1:
        raise SystemExit(
            "bronze: the cards do not share a sample rate "
            f"({', '.join(f'{r:.0f} Hz' for r in rates)}); a lag in samples "
            "cannot be carried between them. See the time base section above.")
    fs = float(rates[0])

    raw = {}
    for stem in stems:
        c = cards[stem]
        x = np.asarray(open_famos(c.path).channel(c.ref_name), float)
        raw[stem] = x - x.mean()

    lo, hi = _energy_band(raw, fs, cfg, log)

    def _bandpass(x: np.ndarray) -> np.ndarray:
        n = len(x)
        X = np.fft.rfft(x)
        fr = np.fft.rfftfreq(n, 1.0 / fs)
        X[(fr < lo) | (fr > hi)] = 0.0
        return np.fft.irfft(X, n)

    traces = {stem: _bandpass(x) for stem, x in raw.items()}
    del raw

    max_lag = int(cfg.align_max_lag_s * fs)
    decim = _decimation(fs, hi)
    if decim > 1:
        log.info(f"  searching at 1/{decim} rate ({fs / decim:.0f} Hz); "
                 f"the lag is refined at the full rate")

    base = _pick_anchor(stems, traces, max_lag, log, decim=decim)
    ref = traces[base]
    out = {base: {"lag": 0, "corr": 1.0, "prominence": float("inf"),
                  "applied": True}}
    log.info(f"  reference card: {base} (ref channel {cards[base].ref_name})")

    for stem in stems:
        if stem == base:
            continue
        lag, corr, prom = _best_lag(traces[stem], ref, max_lag, decim=decim)
        lag, corr = _disambiguate(
            stem, base, lag, corr, traces[stem], ref, max_lag, decim,
            cards, fs, timebase, cfg, log)
        # lag < 0 means this card's record RUNS AHEAD of the reference's,
        # i.e. it was armed later; the schedule index must be shifted by
        # +lag to read the same instant.
        strong = prom >= cfg.align_min_prominence
        above_floor = abs(corr) >= cfg.align_min_corr
        applied = bool(cfg.align_cards and strong and above_floor)
        out[stem] = {"lag": lag, "corr": corr, "prominence": prom,
                     "applied": applied}
        why = ("" if applied else
               "   REFUSED (peak not prominent)" if not strong else
               "   REFUSED (below the absolute floor)")
        log.info(f"  {stem[-8:]}: lag {lag:+8d} samples = "
                 f"{lag / fs:+8.4f} s   peak corr {corr:+.3f}"
                 f"   prominence {prom:5.1f}" + why)
        if not applied:
            log.warning(f"    {stem[-8:]} stays on its own clock. If its true "
                        f"offset is not ~0 its dwell windows will land on the "
                        f"wrong tone and EVERY segment on this card will fail "
                        f"the SNR gate in silver.")
        elif abs(lag) >= max_lag - 1:
            log.warning("    lag is at the search limit - raise "
                        "align_max_lag_s and re-run")
    if not cfg.align_cards:
        log.warning("  align_cards is off: schedule windows will be applied "
                    "to every card unshifted, which is only correct if the "
                    "cards were hardware-triggered together")

    _check_against_headers(out, base, cards, fs, timebase, cfg, log)
    return out


def _disambiguate(stem: str, base: str, lag: int, corr: float,
                  x: np.ndarray, ref: np.ndarray, max_lag: int, decim: int,
                  cards: dict[str, CardInfo], fs: float,
                  timebase: "TimebaseReport | None", cfg: Config, log
                  ) -> tuple[int, float]:
    """Choose between correlation peaks of comparable height, using the stamps.

    Only does anything when there IS a choice: peaks within
    `align_peak_tie_ratio` of the winner.  For a stepped sine there is one
    peak and this returns it unchanged.  For a periodic multisine there are
    several, equally good, one base period apart -- see
    :func:`_lag_candidates`.

    WHEN THE STAMPS CAN SETTLE IT, AND WHEN THEY CANNOT
    ---------------------------------------------------
    The |NT stamps resolve to about a second.  So they can pick between
    candidates a base period apart only when that PERIOD IS LONGER THAN
    THEIR RESOLUTION.  A 0.5 Hz multisine repeats every 2 s and the stamps
    settle it; a 5 Hz multisine repeats every 0.2 s and they cannot -- ten
    candidates fit inside one tick of the stamp, and snapping to the nearest
    would be picking one of them by rounding error and calling it measured.

    In that second case the honest output is the residual ambiguity itself.
    A lag that is right to within +/- one base period is still enough to
    place a dwell window when the period is short compared with the dwell,
    and the caller is told the number so it can judge that rather than
    assume it.
    """
    cands = _lag_candidates(x, ref, max_lag, decim=decim,
                            ratio=cfg.align_peak_tie_ratio)
    if len(cands) < 2:
        return lag, corr

    positions = np.array(sorted(c[0] for c in cands), float)
    gaps = np.diff(positions)
    period = float(np.median(gaps)) if gaps.size else 0.0
    consistent = bool(gaps.size and np.all(np.abs(gaps - period)
                                           <= 0.25 * max(period, 1.0)))
    note = ""
    if consistent and period > 0:
        note = (f", peaks {period / fs:.3f} s apart (a base frequency of "
                f"{fs / period:.2f} Hz repeats at exactly that interval)")

    offsets = timebase.header_offsets_s if timebase else {}
    have_stamps = base in offsets and stem in offsets

    if not have_stamps:
        log.warning(
            f"    {stem[-8:]}: {len(cands)} correlation peaks within "
            f"{100 * cfg.align_peak_tie_ratio:.0f} % of each other{note}. A "
            f"periodic excitation makes the lag ambiguous by whole periods "
            f"and there is no |NT stamp to break the tie; taking the "
            f"largest, which may be wrong by a multiple of that interval.")
        return lag, corr

    if consistent and period / fs <= cfg.align_header_resolution_s:
        log.warning(
            f"    {stem[-8:]}: the lag is ambiguous by +/- {period / fs:.3f} s"
            f"{note}, and the |NT stamps resolve only to about "
            f"{cfg.align_header_resolution_s:.1f} s, so they cannot choose "
            f"between them. Taking the largest peak and carrying the "
            f"ambiguity: dwell windows are placed to within one base period. "
            f"To remove it, trigger the cards from one hardware signal, or "
            f"put a non-periodic marker at the start of the recording.")
        return lag, corr

    want = offsets[base] - offsets[stem]              # seconds, coarse
    if consistent and period > 0:
        # Snap along the period rather than only among the peaks that were
        # returned: the true lag can be many periods out, and enumerating
        # the strongest handful need not reach it.
        k = round((want * fs - lag) / period)
        snapped = int(lag + k * period)
        if abs(snapped) <= max_lag and snapped != lag:
            log.warning(
                f"    {stem[-8:]}: the largest correlation peak is "
                f"{lag / fs:+.3f} s{note}. The |NT stamps put this card "
                f"{want:+.1f} s from the anchor, so {snapped / fs:+.3f} s "
                f"is taken instead -- {k:+d} base period(s) along.")
            return snapped, corr
        return lag, corr

    best = min(cands, key=lambda c: abs(c[0] / fs - want))
    if best[0] != lag:
        log.warning(
            f"    {stem[-8:]}: {len(cands)} correlation peaks within "
            f"{100 * cfg.align_peak_tie_ratio:.0f} % of the largest"
            f"{note}; the |NT stamps ({want:+.1f} s) select "
            f"{best[0] / fs:+.3f} s rather than {lag / fs:+.3f} s.")
    return best[0], best[1]


def _check_against_headers(lags: dict[str, dict], base: str,
                           cards: dict[str, CardInfo], fs: float,
                           timebase: "TimebaseReport | None",
                           cfg: Config, log) -> None:
    """Does the arming order the headers record match the measured lags?

    A measured lag of `k` samples means this card's copy of the event sits
    `k` samples later in its own array than the anchor's does, which happens
    when this card was armed EARLIER by k/fs seconds.  So

        measured (t_anchor - t_card)  =  lag / fs

    and the headers say the same quantity to about a second.  Their VALUES
    are too coarse to compare -- on the 45 A set the stamps put card 3 three
    seconds from the anchor where the truth was 5.712 -- but their ORDER is
    not coarse at all, and a lag whose sign contradicts the stamps is not a
    small error.  It is the correlation having locked onto a different step
    of the sweep, which is precisely the failure prominence cannot see: a
    periodic-looking excitation gives a sharp, confident peak one dwell
    over.
    """
    if timebase is None or not timebase.header_offsets_s:
        return
    offsets = timebase.header_offsets_s
    if base not in offsets:
        return

    disagreements = []
    for stem, info in lags.items():
        if stem == base or stem not in offsets or not info.get("applied"):
            continue
        header = offsets[base] - offsets[stem]          # seconds, coarse
        measured = info["lag"] / fs
        info["header_offset_s"] = header
        info["measured_offset_s"] = measured
        # Only pairs the stamps actually separate can testify about order.
        if abs(header) <= cfg.align_header_resolution_s:
            continue
        if np.sign(header) != np.sign(measured) or \
                abs(measured - header) > cfg.align_header_tol_s:
            disagreements.append((stem, header, measured))

    if not disagreements:
        if any("header_offset_s" in i for i in lags.values()):
            log.info("  header cross-check: measured lags agree with the "
                     "arming order in the |NT stamps  OK")
        return

    for stem, header, measured in disagreements:
        log.error(
            f"  {stem[-8:]}: the |NT stamps put it {header:+.1f} s from the "
            f"anchor but the correlation measured {measured:+.1f} s. The "
            f"stamps are coarse, so a small difference would mean nothing — "
            f"this one is not small.")
        lags[stem]["applied"] = False
        lags[stem]["refused_reason"] = "contradicts the |NT arming order"
    log.error("  those cards stay on their own clock rather than being "
              "shifted onto a lag the headers contradict. Check that every "
              ".DAT in this folder belongs to this measurement, then raise "
              "align_max_lag_s if the true stagger is larger than the search "
              "window.")


def _best_lag(x: np.ndarray, ref: np.ndarray,
              max_lag: int, decim: int = 1) -> tuple[int, float, float]:
    """The lag of best agreement, its correlation, and its prominence.

    With ``decim > 1`` the search runs on every ``decim``-th sample and the
    winner is then refined at the full rate over the one decimated bin it
    fell in, so the returned lag is still a full-rate sample count.
    """
    if decim > 1:
        n_d = min(len(x), len(ref)) // decim
        if n_d >= 256:
            k, _corr, prom = _best_lag(x[:n_d * decim:decim],
                                       ref[:n_d * decim:decim],
                                       max(1, max_lag // decim))
            centre = k * decim
            span = 2 * decim
            lo = max(0, centre - span)
            # Refine on the full-rate records, restricted to the neighbourhood
            # the coarse pass chose. Prominence stays the coarse figure: it is
            # a statement about the peak against the WHOLE curve, and a window
            # a few samples wide has no background to measure it against.
            best_k, best_v = centre, -np.inf
            n = min(len(x), len(ref))
            a = x[:n]
            b = ref[:n]
            denom = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
            for cand in range(centre - span, centre + span + 1):
                if abs(cand) > max_lag:
                    continue
                if cand >= 0:
                    v = float(np.dot(a[cand:], b[:n - cand]))
                else:
                    v = float(np.dot(a[:n + cand], b[-cand:]))
                if abs(v) > abs(best_v):
                    best_k, best_v = cand, v
            corr = best_v / denom if denom > 0 else 0.0
            return int(best_k), float(corr), prom
    n = min(len(x), len(ref))
    a, b = x[:n], ref[:n]
    m = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(a, m) * np.conj(np.fft.rfft(b, m)), m)
    cc = np.concatenate([cc[-(n - 1):], cc[:n]])
    lags = np.arange(-(n - 1), n)
    sel = np.abs(lags) <= max_lag
    cc_sel, lag_sel = cc[sel], lags[sel]
    k = int(np.argmax(np.abs(cc_sel)))
    denom = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    corr = float(cc_sel[k] / denom) if denom > 0 else 0.0
    return int(lag_sel[k]), corr, _prominence(np.abs(cc_sel), k)


def _lag_candidates(x: np.ndarray, ref: np.ndarray, max_lag: int,
                    decim: int = 1, n_cand: int = 5,
                    ratio: float = 0.8) -> list[tuple[int, float]]:
    """The competing correlation peaks, best first, as (lag, corr).

    WHY THE WINNER ALONE IS NOT ENOUGH FOR A MULTISINE
    --------------------------------------------------
    A designed multisine is PERIODIC: every tone is an integer multiple of a
    base frequency f0, so the excitation repeats every 1/f0 seconds.  The
    cross-correlation of a periodic signal is periodic too, and every lag
    differing by a whole period scores identically -- not approximately,
    identically.  Picking the largest peak then picks among equals by noise,
    and the answer is wrong by an integer number of base periods.

    That is not a small error.  The cards are armed by hand seconds apart,
    and with f0 = 1 Hz the ambiguity is one second, so the wrong period is
    entirely capable of looking like the right stagger.  Nothing downstream
    can see it: prominence is high (the peak really is sharp), the
    correlation is high, and what silver eventually reports is segments
    failing an SNR gate on a card whose dwell windows sit one period off.

    So the competing peaks are returned and the caller disambiguates with
    the header stamps, which are coarse but have no periodicity at all.
    """
    n = min(len(x), len(ref))
    a, b = x[:n], ref[:n]
    q = max(1, int(decim))
    n_d = n // q
    if q > 1 and n_d < 256:
        q, n_d = 1, n
    aa, bb = a[:n_d * q:q], b[:n_d * q:q]
    lag_cap = max(1, max_lag // q)

    m = 1 << int(np.ceil(np.log2(2 * n_d)))
    cc = np.fft.irfft(np.fft.rfft(aa, m) * np.conj(np.fft.rfft(bb, m)), m)
    cc = np.concatenate([cc[-(n_d - 1):], cc[:n_d]])
    lags = np.arange(-(n_d - 1), n_d)
    sel = np.abs(lags) <= lag_cap
    mag, lag_sel = np.abs(cc[sel]), lags[sel]
    denom = np.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))

    best = float(np.max(mag)) if mag.size else 0.0
    if best <= 0:
        return []
    # One guard band per candidate, so the same peak is not returned twice.
    guard = max(8, int(0.005 * mag.size))
    taken: list[int] = []
    order = np.argsort(-mag)
    for k in order:
        if mag[k] < ratio * best:
            break
        if all(abs(int(k) - t) > guard for t in taken):
            taken.append(int(k))
        if len(taken) >= n_cand:
            break
    out = []
    for k in taken:
        lag_full = int(lag_sel[k]) * q
        corr = float(cc[sel][k] / denom) if denom > 0 else 0.0
        out.append((lag_full, corr))
    return out


def _prominence(mag: np.ndarray, k: int, guard: int | None = None) -> float:
    """How many robust sigma the peak stands above the rest of the curve.

    Median and MAD rather than mean and sd, because the correlation of a
    stepped sweep against itself is not flat -- it has broad structure that
    a mean-based score would read as signal.  The guard band excludes the
    peak's own shoulders, which are part of the peak, not of the background.

    The guard is a FRACTION OF THE CURVE, not a fixed sample count.  It was
    5000 samples, which is half a second of shoulder at 10 kHz and fifty
    milliseconds of it at 100 kHz -- so on a fast card most of the peak's own
    skirt stayed in the "background", inflating the MAD and pushing the
    prominence of a perfectly good lag below the gate.  A refused lag is not
    a warning downstream: the card keeps its own clock, its dwell windows sit
    on the wrong tone, and every segment on it fails silver's SNR gate.
    """
    if guard is None:
        guard = max(64, int(0.02 * mag.size))
    lo, hi = max(0, k - guard), min(mag.size, k + guard + 1)
    background = np.concatenate([mag[:lo], mag[hi:]])
    if background.size < 64:
        return float("nan")
    med = float(np.median(background))
    mad = float(np.median(np.abs(background - med)))
    if mad <= 0:
        return float("inf") if mag[k] > med else 0.0
    return float((float(mag[k]) - med) / (1.4826 * mad))


def _pick_anchor(stems: list[str], traces: dict[str, np.ndarray],
                 max_lag: int, log, decim: int = 1) -> str:
    """The card the others agree with best.

    Anchoring on whichever card happens to be first is a coin flip, and it
    loses when that card's reference channel is the degraded one: every
    other card then scores weakly against it and a height gate refuses the
    lot.  Scoring each candidate by the MEDIAN correlation it achieves
    against the rest picks an anchor that is good for the group, and the
    median keeps one bad card from deciding it.
    """
    if len(stems) < 3:
        return stems[0]
    scores: dict[str, float] = {}
    for cand in stems:
        others = [abs(_best_lag(traces[s], traces[cand], max_lag,
                                decim=decim)[1])
                  for s in stems if s != cand]
        scores[cand] = float(np.median(others))
    best = max(scores, key=lambda s: scores[s])
    if log is not None and best != stems[0]:
        log.info(f"  anchoring on {best[-8:]} (median |r| {scores[best]:.3f}) "
                 f"rather than {stems[0][-8:]} ({scores[stems[0]]:.3f}): "
                 f"the first card is not the one the others agree with best")
    return best


# ===========================================================================
# 3. Consensus schedule
# ===========================================================================


def excitation_above_ceiling(ref: np.ndarray, fs: float, f_hi: float,
                             min_fraction: float = 0.05) -> dict:
    """Is there excitation the detection ceiling is excluding?

    The blind search only ever looks below `f_hi`, so "no steps up there" and
    "never looked up there" produce the identical empty result.  That is the
    whole reason a ceiling frozen at 4500 Hz could survive a move to 100 kHz
    sampling without anybody seeing it: the operator raised the rate to buy
    bandwidth, the detector kept searching the same 4.5 kHz, and the output
    was a spectrum that simply stopped where it always had.

    This measures the AC energy between the ceiling and Nyquist and names the
    strongest peak found there, so the two cases can be told apart.
    """
    x = np.asarray(ref, float)
    x = x - x.mean()
    nper = 1 << int(np.ceil(np.log2(max(1024.0, fs))))
    if len(x) < nper:
        nper = 1 << int(np.floor(np.log2(max(256, len(x)))))
    if len(x) < nper or nper < 256:
        return {"ok": False}

    win = np.hanning(nper)
    step = max(nper, (len(x) - nper) // 16 or nper)
    acc = None
    for a in range(0, len(x) - nper + 1, step):
        seg = x[a:a + nper]
        spec = np.abs(np.fft.rfft((seg - seg.mean()) * win)) ** 2
        acc = spec if acc is None else acc + spec
    if acc is None:
        return {"ok": False}

    freqs = np.fft.rfftfreq(nper, 1.0 / fs)
    ac = freqs > 0.5
    above = ac & (freqs > f_hi)
    total = float(acc[ac].sum())
    if total <= 0:
        return {"ok": False}
    fraction = float(acc[above].sum() / total)
    peak_hz = float(freqs[above][int(np.argmax(acc[above]))]) if above.any() \
        else float("nan")
    return {"ok": True, "fraction": fraction, "peak_hz": peak_hz,
            "f_hi": float(f_hi), "nyquist_hz": float(fs / 2),
            "significant": bool(fraction >= min_fraction)}


def consensus_schedule(files: list[Path], cards: dict[str, CardInfo],
                       cfg: Config, log=None,
                       lags: dict[str, dict] | None = None
                       ) -> tuple[list[Step], dict]:
    """One excitation schedule for the whole plate.

    Each card's reference channel is run through the blind detector, then all
    the candidate steps from all cards are collapsed:

      1. cluster candidates whose frequencies agree to `grid_tol`
      2. keep a cluster if at least `min_ref_channels` cards saw it, OR if it
         sits on the geometric grid that the surviving clusters define
      3. report one frequency per cluster: the SNR-weighted mean

    Step (2) is the important one.  A stepped sweep is a geometric
    progression, and noise does not produce a geometric progression.  Once the
    grid is known, a weak step that only one card saw is still credible if it
    lands on the grid -- which is how the top-of-band points survive without
    lowering the SNR gate for everything else.
    """
    log = log or utils.get_logger(cfg.verbose)
    utils.section("schedule detection (blind, consensus across cards)", log)

    per_card: dict[str, list[Step]] = {}
    for fp in files:
        stem = fp.stem
        if stem not in cards:
            continue
        fam = open_famos(fp)
        ref = fam.channel(cards[stem].ref_name)
        f_hi = cfg.f_hi(fam.fs)
        headroom = excitation_above_ceiling(ref, fam.fs, f_hi)
        if headroom.get("significant"):
            peak = headroom["peak_hz"]
            nyquist = headroom["nyquist_hz"]
            log.warning(
                f"  {stem[-8:]}: {100 * headroom['fraction']:.0f} % of the AC "
                f"energy on the reference sits ABOVE the {f_hi:.0f} Hz search "
                f"ceiling, strongest at {peak:.0f} Hz (Nyquist "
                f"{nyquist:.0f} Hz).")
            # TWO READINGS, AND THEY NEED OPPOSITE RESPONSES.  Saying only
            # "raise --f-max" assumes the energy is excitation, and for a
            # PEM cell a tone in the tens of kHz usually is not: a load
            # bank's switching frequency lands there, and chasing it wastes
            # the run at best and fits noise at worst.
            log.warning(
                f"    If those are excitation tones, raise --f-max to reach "
                f"them. If they are not -- a load-bank switching frequency "
                f"or pickup, which is what energy in the tens of kHz usually "
                f"is on a fuel cell -- then raising the ceiling makes the "
                f"detector chase interference, and what the interference "
                f"needs is a notch.")
            if peak > 0.85 * nyquist:
                log.warning(
                    f"    {peak:.0f} Hz is within 15 % of this card's "
                    f"Nyquist, so it may itself be an alias of something "
                    f"higher that the anti-alias filter did not stop.")
        # WHICH DETECTOR.  Decided per card from the record, because a
        # stepped multisine handed to the single-tone detector does not fail
        # loudly -- it returns a few tones with windows a millisecond long.
        kind = cfg.excitation
        if kind == "auto":
            verdict = classify_excitation(ref, fam.fs, cfg.f_min_hz, f_hi)
            kind = verdict["kind"]
            log.info(f"  {stem[-8:]}: excitation looks {kind} "
                     f"({verdict['n_tones']} strong tone(s), median "
                     f"{verdict['tones_at_once']:.1f} on at once"
                     + (f", {verdict['n_dwells']} dwell(s)"
                        if verdict['n_dwells'] else "") + ")")
        if kind == "multisine":
            steps = detect_multisine_schedule(
                ref, fam.fs, f_lo=cfg.f_min_hz, f_hi=f_hi,
                peak_db=cfg.tone_peak_db, verbose=False)
        else:
            steps = detect_schedule(ref, fam.fs, ppd=cfg.ppd,
                                    f_lo=cfg.f_min_hz, f_hi=f_hi,
                                    min_snr_db=cfg.min_snr_db,
                                    verbose=False)
        # PUT THE WINDOWS ON THE COMMON TIME BASE BEFORE THEY ARE POOLED.
        # Detection runs in each card's own sample index; the consensus that
        # follows mixes windows from different cards, so they have to mean
        # the same instant first.  process_card shifts them back.
        #
        # The `applied` test is not optional and must MATCH the one in the
        # phasor pass, which reads `lag` only when `applied`.  Without it a
        # refused card is shifted ONE WAY: onto the common base here, never
        # back onto its own clock there, so every window on it is off by
        # exactly the lag that was judged untrustworthy.  On the 45 A set
        # that silently cost 43 of 68 segments -- all of cards 3, 4 and 5,
        # whose refused lags were 5.71 s, 2.54 s and 2.55 s -- and it looked
        # like an SNR problem, because what silver sees is a dwell window
        # sitting on the wrong tone.  Refusing a lag has to mean NO shift
        # anywhere, not half a shift.
        info = lags.get(stem, {}) if lags else {}
        d = int(info.get("lag", 0)) if info.get("applied") else 0
        if d:
            steps = [Step(freq=s.freq, start=s.start - d, stop=s.stop - d,
                          amp=s.amp, snr_db=s.snr_db, thd=s.thd,
                          stationarity=s.stationarity) for s in steps]
        per_card[stem] = steps
        log.info(f"  {stem}: {len(steps)} candidate steps "
                 f"({steps[0].freq:.3f}..{steps[-1].freq:.1f} Hz)"
                 if steps else f"  {stem}: nothing found")

    # ---- cluster across cards ---------------------------------------------
    allsteps: list[tuple[str, Step]] = [(c, s) for c, ss in per_card.items()
                                        for s in ss]
    if not allsteps:
        raise SystemExit("bronze: no excitation steps found on any card")
    allsteps.sort(key=lambda cs: cs[1].freq)

    clusters: list[list[tuple[str, Step]]] = []
    for card, st in allsteps:
        if clusters and abs(st.freq / clusters[-1][-1][1].freq - 1.0) <= cfg.grid_tol:
            clusters[-1].append((card, st))
        else:
            clusters.append([(card, st)])

    # ---- provisional frequencies, then the geometric grid -----------------
    prov, prov_snr = [], []
    for cl in clusters:
        w = np.array([max(s.snr_db, 0.1) for _, s in cl], float)
        f = np.array([s.freq for _, s in cl], float)
        prov.append(float(np.sum(w * f) / np.sum(w)))
        prov_snr.append(float(np.nanmax([s.snr_db for _, s in cl])))
    prov = np.array(prov)
    prov_snr = np.array(prov_snr)

    # THE GRID IS FITTED ON CONFIDENT STEPS ONLY.
    # Grid membership is used below as evidence that a WEAK step is real.
    # That argument only holds if the grid was established independently of
    # the weak steps -- otherwise the junk defines the grid it is then
    # validated against, which is circular and admits everything.
    #
    # This is not hypothetical.  Fitted on all candidates, a synthetic sweep
    # of 18 steps produced 55 detections, every one of them "on grid",
    # because the spurious detections between the true steps pulled the fit
    # onto a spacing fine enough to contain them all.  The true steps sat at
    # +23 dB and the spurious ones between -12 and -45 dB.
    strong = prov_snr >= cfg.min_snr_db
    if strong.sum() >= 4:
        grid = utils.geometric_grid_fit(prov[strong], tol=cfg.grid_tol)
        grid["n_fitted_on"] = int(strong.sum())
    else:
        grid = utils.geometric_grid_fit(prov, tol=cfg.grid_tol)
        grid["n_fitted_on"] = int(len(prov))
        grid["weak_basis"] = True

    # A grid far finer than the confident steps themselves is not a sweep,
    # it is an artefact of over-fitting; refuse to use it as evidence.
    if grid.get("ok") and strong.sum() >= 4:
        f_strong = np.sort(prov[strong])
        if len(f_strong) > 1:
            true_ppd = 1.0 / np.median(np.diff(np.log10(f_strong)))
            if grid["ppd"] > 1.8 * abs(true_ppd):
                # Refit forcing the spacing implied by the confident steps,
                # rather than abandoning the grid altogether.  Disabling it
                # outright also discards the weak TOP-OF-BAND steps the
                # rescue exists to save, which is the opposite of the
                # intent: it cost a whole decade of bandwidth in testing.
                log.warning(f"  grid fit returned {grid['ppd']:.1f} points/decade "
                            f"against {abs(true_ppd):.1f} implied by the "
                            f"confident steps - refitting at the coarser "
                            f"spacing")
                coarse = utils.geometric_grid_fit(f_strong,
                                                  tol=cfg.grid_tol)
                if coarse.get("ok") and coarse["ppd"] <= 1.8 * abs(true_ppd):
                    grid = coarse
                    grid["n_fitted_on"] = int(strong.sum())
                    grid["refitted_coarse"] = True
                else:
                    grid["ok"] = False

    # ---- accept ------------------------------------------------------------
    kept: list[Step] = []
    n_votes_kept, n_grid_rescued = 0, 0
    for cl, f_hat in zip(clusters, prov):
        cards_seen = {c for c, _ in cl}
        on_grid = bool(grid.get("ok")) and _on_grid(f_hat, grid, cfg.grid_tol)
        enough = len(cards_seen) >= cfg.min_ref_channels
        if not (enough or on_grid):
            continue
        # representative window: the longest dwell in the cluster, which is
        # the one least likely to have been truncated by a neighbour
        best = max(cl, key=lambda cs: cs[1].stop - cs[1].start)[1]
        snr = float(np.nanmax([s.snr_db for _, s in cl]))
        thd = float(np.nanmedian([s.thd for _, s in cl]))
        drift = float(np.nanmedian([s.stationarity for _, s in cl]))
        kept.append(Step(freq=float(f_hat), start=best.start, stop=best.stop,
                         amp=best.amp, snr_db=snr, thd=thd, stationarity=drift))
        n_votes_kept += enough
        n_grid_rescued += (on_grid and not enough)

    kept.sort(key=lambda s: s.freq)
    log.info(f"  consensus: {len(kept)} steps "
             f"({kept[0].freq:.3f}..{kept[-1].freq:.1f} Hz), "
             f"{n_votes_kept} by card agreement, "
             f"{n_grid_rescued} rescued by grid membership")
    if grid.get("ok"):
        log.info(f"  geometric grid fitted on {grid.get('n_fitted_on','?')} "
                 f"confident steps (SNR >= {cfg.min_snr_db:.0f} dB)")
        log.info(f"  geometric grid: {grid['ppd']:.2f} points/decade, "
                 f"f0={grid['f0']:.1f} Hz, "
                 f"{grid['n_on_grid']}/{grid['n_total']} on grid")
    else:
        log.warning("  geometric grid: NOT recovered - every step had to be "
                    "carried by card agreement alone")
    return kept, grid


def _on_grid(f: float, grid: dict, tol: float) -> bool:
    # A grid that the fitter REJECTED is not evidence of anything.  Without
    # this check a discarded fit still set on_grid, which then swapped the
    # SNR gate for the far looser snr_floor_db on those points.
    if not grid.get("ok"):
        return False
    f0, r = grid.get("f0"), grid.get("ratio")
    if not f0 or not r or r <= 1:
        return False
    k = round(np.log(f0 / f) / np.log(r))
    return abs(f / (f0 * r ** -k) - 1.0) <= tol



# ===========================================================================
# 3b. Calibration status  (WAS IT ACTUALLY APPLIED?)
# ===========================================================================


@dataclass
class CalibrationReport:
    """Whether the current and temperature calibrations really were used.

    "Is the calibration done?" is not answerable from the fact that a file
    was passed on the command line, and each of the ways it can be half-done
    is silent:

    * `--curr-cal` names a file that exists, but with fewer rows than the
      plate has segments.  The segments it does not cover are carried on the
      PLATE MEDIAN coefficient (`_K_for`), which keeps the shape of their
      spectrum right and puts the absolute level off by whatever the real
      coefficient was.  Every map is then a mix of measured and imputed
      levels with nothing on the figure to say which is which.
    * `--temp-cal` is omitted, or its rows do not match the channel names.
      The temperature falls back to a single constant for the whole plate
      (`T_FALLBACK_C`), which discards the inlet-to-outlet gradient -- the
      one thing the sensors exist to measure -- and biases K on every
      segment, since K = c0 + 1e-3*c1*T.
    * The temperature channels are read but their values are implausible, so
      the sensors are dropped one by one until the fallback takes over
      anyway.

    None of these stops a run.  All of them change the numbers.  So the
    answer is computed and written into the manifest, in the same terms the
    question is asked in.
    """

    curr_path: str | None
    temp_path: str | None
    n_seg_rows: int
    n_seg_expected: int
    n_temp_rows: int
    #: Sensors that produced a usable reading, name -> degrees C.
    sensors_used: dict[str, float] = field(default_factory=dict)
    #: Sensors present on a card but rejected, name -> why.
    sensors_rejected: dict[str, str] = field(default_factory=dict)
    temperature_source: str = "unknown"
    t_min_c: float = float("nan")
    t_max_c: float = float("nan")
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def current_ok(self) -> bool:
        return self.n_seg_rows >= self.n_seg_expected

    @property
    def temperature_ok(self) -> bool:
        return self.temperature_source == "measured"

    def summary(self) -> dict:
        return {
            "current": {
                "file": self.curr_path,
                "rows": self.n_seg_rows,
                "expected": self.n_seg_expected,
                "complete": self.current_ok,
                "segments_on_plate_median": max(
                    0, self.n_seg_expected - self.n_seg_rows),
            },
            "temperature": {
                "file": self.temp_path,
                "rows": self.n_temp_rows,
                "source": self.temperature_source,
                "sensors_used": {k: round(v, 3)
                                 for k, v in self.sensors_used.items()},
                "sensors_rejected": dict(self.sensors_rejected),
                "segment_t_min_c": self.t_min_c,
                "segment_t_max_c": self.t_max_c,
                "measured": self.temperature_ok,
            },
            "problems": list(self.problems),
            "notes": list(self.notes),
        }


def calibration_report(cal: PlateCalibration, cfg: Config,
                       sensor_T: dict[str, float],
                       T_seg: dict[str, float],
                       rejected: dict[str, str] | None = None,
                       log=None) -> CalibrationReport:
    """State plainly what was calibrated and what was substituted."""
    log = log or utils.get_logger(cfg.verbose)
    utils.section("calibration status", log)

    temps = [v for v in T_seg.values() if np.isfinite(v)]
    measured = bool(sensor_T)
    rep = CalibrationReport(
        curr_path=str(cfg.curr_cal) if cfg.curr_cal else None,
        temp_path=str(cfg.temp_cal) if cfg.temp_cal else None,
        n_seg_rows=len(cal.seg_c0), n_seg_expected=geom.N_SEGMENTS,
        n_temp_rows=len(cal.temp_c0),
        sensors_used=dict(sensor_T),
        sensors_rejected=dict(rejected or {}),
        temperature_source="measured" if measured else "fallback constant",
        t_min_c=float(min(temps)) if temps else float("nan"),
        t_max_c=float(max(temps)) if temps else float("nan"),
    )

    # ---- current -------------------------------------------------------
    if not cal.seg_c0:
        rep.problems.append(
            "NO current calibration. j_s = u_s / K is the only absolute "
            "scale left once the potentiostat is gone, so without it the "
            "impedance is in shunt volts per amp, not ohms.")
        log.error("  current (Abgleich): MISSING")
    elif not rep.current_ok:
        missing = geom.N_SEGMENTS - len(cal.seg_c0)
        rep.problems.append(
            f"current calibration covers {len(cal.seg_c0)} of "
            f"{geom.N_SEGMENTS} segments; the other {missing} are carried on "
            f"the plate-median coefficient. Their spectrum SHAPE is "
            f"unaffected -- K is a scalar -- but their absolute level is off "
            f"by however far their true coefficient sits from the median, "
            f"and they are marked K_imputed in the output.")
        log.warning(f"  current (Abgleich): {len(cal.seg_c0)}/"
                    f"{geom.N_SEGMENTS} segment rows from "
                    f"{rep.curr_path}  -- {missing} imputed")
    else:
        c0 = np.array(list(cal.seg_c0.values()), float)
        log.info(f"  current (Abgleich): APPLIED, {len(cal.seg_c0)}/"
                 f"{geom.N_SEGMENTS} segment rows from {rep.curr_path}")
        log.info(f"    c0 spans {c0.min():.4g} .. {c0.max():.4g} "
                 f"(median {np.median(c0):.4g}) V/(A/cm^2)")

    # ---- temperature ---------------------------------------------------
    if not cfg.temp_cal:
        rep.notes.append(
            "no temperature calibration file: the plate temperature is the "
            f"fallback constant {T_FALLBACK_C} C for every segment. K = c0 + "
            f"1e-3*c1*T, so this biases every segment's scale, and it "
            f"discards the inlet-to-outlet gradient the sensors exist to "
            f"measure.")
        log.warning(f"  temperature: NOT CALIBRATED - falling back to "
                    f"{T_FALLBACK_C} C everywhere")
    elif not cal.temp_c0:
        rep.problems.append(
            f"temperature calibration {rep.temp_path} produced no usable "
            f"rows; expected lines of 'c0;c1', one per sensor.")
        log.error(f"  temperature: {rep.temp_path} has no usable rows")
    elif not measured:
        rep.problems.append(
            f"temperature calibration was loaded ({len(cal.temp_c0)} sensor "
            f"rows) but NO sensor produced a plausible reading, so the "
            f"fallback constant {T_FALLBACK_C} C is in use anyway. Check "
            f"that the temp channel names match the calibration rows and "
            f"that the sensors were connected.")
        log.error("  temperature: calibration loaded but every sensor was "
                  "rejected - fallback constant in use")
    else:
        log.info(f"  temperature: APPLIED, {len(cal.temp_c0)} sensor rows "
                 f"from {rep.temp_path}")
        pretty = ", ".join(f"{k}={v:.2f} C"
                           for k, v in sorted(rep.sensors_used.items()))
        log.info(f"    {len(sensor_T)} sensor(s) used: {pretty}")
        log.info(f"    segments interpolated over "
                 f"{rep.t_min_c:.2f} .. {rep.t_max_c:.2f} C")
    for name, why in (rejected or {}).items():
        log.warning(f"    {name} rejected: {why}")

    if not rep.problems and rep.current_ok and rep.temperature_ok:
        log.info("  both calibrations applied in full  OK")
    return rep


# ===========================================================================
# 4. Per-card extraction
# ===========================================================================


def _sensor_key(channel_name: str) -> str:
    """FAMOS channel name -> calibration key.

    The plate writes 'Temp_1'..'Temp_4'; PlateCalibration builds its keys as
    'temp1'..'temp4'.  Lower-casing alone leaves the underscore in place, so
    the lookup never matched and EVERY run silently fell back to
    T_FALLBACK_C.  Strip to the digits and rebuild.
    """
    digits = re.sub(r"\D", "", channel_name)
    return f"temp{digits}" if digits else channel_name.lower()


def plate_temperatures(files: list[Path], cards: dict[str, CardInfo],
                       cal: PlateCalibration, cfg: Config, log=None,
                       rejected: dict[str, str] | None = None,
                       max_samples: int | None = None
                       ) -> tuple[dict[str, float], dict]:
    """One temperature field for the whole plate, from every sensor on every
    card.

    The four sensors are not on one card: on this plate Temp_1/Temp_2 sit on
    card 1 and Temp_3/Temp_4 on card 3, while cards 2, 4 and 5 carry none at
    all.  Reading them per card therefore left three cards with no sensor and
    the fallback constant -- which throws away the whole inlet-to-outlet
    gradient, the one thing the sensors are there to measure.
    """
    log = log or utils.get_logger(cfg.verbose)
    sensor_T: dict[str, float] = {}
    for fp in files:
        if fp.stem not in cards:
            continue
        fam = open_famos(fp)
        for tn in fam.temp_names:
            key = _sensor_key(tn)
            if key not in cal.temp_c0:
                log.warning(f"  {tn}: no calibration row for {key!r} - skipped")
                if rejected is not None:
                    rejected[tn] = f"no calibration row for {key!r}"
                continue
            n_look = (fam.n_samples if max_samples is None
                      else min(fam.n_samples, int(max_samples)))
            u = float(np.mean(fam.channel(tn, 0, n_look)[::cfg_stride(cfg)]))
            T = cal.temperature(key, u)
            if 0.0 < T < 120.0:
                sensor_T[key] = T
            else:
                log.warning(f"  {tn}: u={u:.4f} V -> T={T:.1f} C implausible "
                            f"- sensor ignored")
                if rejected is not None:
                    rejected[tn] = (f"u={u:.4f} V gives T={T:.1f} C, outside "
                                    f"0..120 C")
    if sensor_T:
        pretty = ", ".join(f"{k}={v:.2f} C" for k, v in sorted(sensor_T.items()))
        log.info(f"  plate temperature from {len(sensor_T)} sensor(s): {pretty}")
        seg_T = geom.segment_temperatures(sensor_T)
        log.info(f"  interpolated to segments: "
                 f"{min(seg_T.values()):.2f} .. {max(seg_T.values()):.2f} C")
        return seg_T, sensor_T
    log.warning(f"  NO usable temperature sensor - falling back to "
                f"{T_FALLBACK_C} C for every segment")
    return {s: T_FALLBACK_C for s in geom.SEGMENTS}, {}


def _K_for(seg: str, cal: PlateCalibration, T: float,
           median_c0: float, median_c1: float,
           cfg: Config) -> tuple[float, bool]:
    """Transfer coefficient, imputing from the plate median if necessary.

    A missing calibration row used to delete the segment.  It is better to
    carry the segment with a plate-median coefficient and mark it: the SHAPE
    of the spectrum, and therefore every diagnostic that depends on shape, is
    unaffected by an error in the scalar K.  Only the absolute level is, and
    that is exactly what the flag warns about.
    """
    if seg in cal.seg_c0:
        return cal.K(seg, T), False
    return median_c0 + 1e-3 * median_c1 * T, True


def _fix_polarity(Z: np.ndarray, freqs: np.ndarray, snr_db: np.ndarray,
                  cfg: Config) -> np.ndarray:
    """Return Z or -Z, deciding on the low-frequency, high-SNR points."""
    ok = np.isfinite(Z.real) & np.isfinite(freqs) & (freqs > 0)
    ok &= np.isfinite(snr_db) & (snr_db >= cfg.min_snr_db)
    if ok.sum() < 3:                      # fall back to anything finite
        ok = np.isfinite(Z.real) & np.isfinite(freqs) & (freqs > 0)
    if not ok.any():
        return Z
    f_ok = freqs[ok]
    low = f_ok <= 10.0 * np.nanmin(f_ok)          # lowest decade
    if low.sum() < 3:
        low = np.ones_like(f_ok, bool)
    re = Z.real[ok][low]
    w = np.clip(snr_db[ok][low], 0.1, None)
    order = np.argsort(re)
    cw = np.cumsum(w[order])
    med = float(re[order][int(np.searchsorted(cw, 0.5 * cw[-1]))])
    if med < 0:
        # decisive only if the low decade is predominantly negative
        frac_neg = float(np.sum(w[re < 0]) / np.sum(w))
        if frac_neg > 0.6:
            return -Z
    return Z


def process_card(fp: Path, cal: PlateCalibration, schedule: list[Step],
                 grid: dict, cfg: Config, log=None,
                 T_seg: dict[str, float] | None = None,
                 lag: int = 0) -> dict[str, BronzeSpectrum]:
    """Raw phasors for every segment on one card.

    `schedule` windows are indices on the COMMON time base; `lag` is this
    card's offset relative to it, so the window actually read is
    (start + lag, stop + lag).
    """
    log = log or utils.get_logger(cfg.verbose)
    fam = open_famos(fp)
    stem = fp.stem
    if not fam.uc_names:
        return {}

    ref_name = max(fam.uc_names,
                   key=lambda c: float(np.std(fam.channel(c)[::cfg_stride(cfg)])))
    ref = fam.channel(ref_name)
    ref_slot = fam.position(ref_name)

    if T_seg is None:
        T_seg = {s: T_FALLBACK_C for s in geom.SEGMENTS}
    if lag:
        log.info(f"    window shift {lag:+d} samples "
                 f"({lag / fam.fs:+.4f} s) onto this card's clock")
    med_c0 = float(np.median(list(cal.seg_c0.values()))) if cal.seg_c0 else np.nan
    med_c1 = float(np.median(list(cal.seg_c1.values()))) if cal.seg_c1 else 0.0

    freqs = np.array([s.freq for s in schedule], float)
    on_grid = np.array([_on_grid(f, grid, cfg.grid_tol) for f in freqs], bool)
    groups = group_simultaneous(schedule)
    n_multi = sum(1 for g in groups if len(g) > 1)
    if n_multi:
        log.info(f"    {len(groups)} dwell group(s), {n_multi} carrying "
                 f"several tones at once - fitted jointly")

    out: dict[str, BronzeSpectrum] = {}
    n_excluded = 0
    for seg in fam.segment_names:
        if seg in cfg.exclude_segments:
            n_excluded += 1
            continue
        x = fam.channel(seg)
        T = T_seg.get(seg, T_FALLBACK_C)
        K, imputed = _K_for(seg, cal, T, med_c0, med_c1, cfg)
        if not np.isfinite(K) or K == 0:
            continue

        n_st = len(schedule)
        Z = np.full(n_st, np.nan, complex)
        snr_r = np.full(n_st, np.nan)
        snr_s = np.full(n_st, np.nan)
        thd = np.full(n_st, np.nan)
        drift = np.full(n_st, np.nan)
        n_per = np.zeros(n_st, int)

        n_skip = 0
        # TONES THAT WERE ON TOGETHER ARE FITTED TOGETHER.  For a stepped
        # sine every group has one member and this is the loop it always
        # was; for a stepped multisine the group is the dwell's whole tone
        # set, and fitting them one at a time would put each tone's
        # neighbours into its own residual -- which is the SNR, and
        # therefore the weight. See eis_local.fit_multitone.
        for g in groups:
            head = schedule[g[0]]
            a = max(st.start for st in (schedule[i] for i in g)) + lag
            b = min(st.stop for st in (schedule[i] for i in g)) + lag
            if b <= a or a < 0 or b > len(x) or b > len(ref):
                n_skip += len(g)
                continue
            yr, ys = ref[a:b], x[a:b]
            for i in g:
                n_per[i] = b - a

            if len(g) > 1:
                fg = [schedule[i].freq for i in g]
                ph_r, _rr, snr_gr, info = fit_multitone(yr, fam.fs, fg)
                ph_s, _rs, snr_gs, _info2 = fit_multitone(ys, fam.fs, fg)
                if not info.get("ok"):
                    n_skip += len(g)
                    continue
                if not info.get("resolvable", True):
                    log.warning(
                        f"    {a / fam.fs:.3f}-{b / fam.fs:.3f} s: tones "
                        f"{info['min_gap_hz']:.3f} Hz apart in a "
                        f"{info['window_s']:.3f} s window are not resolvable "
                        f"(need > 1/T); their split between neighbours is "
                        f"arbitrary")
                tone_set = np.asarray(fg, float)
                for j, i in enumerate(g):
                    A_ref, A_seg = ph_r[j], ph_s[j]
                    snr_r[i], snr_s[i] = snr_gr[j], snr_gs[j]
                    Z[i] = (K * A_ref / A_seg if A_seg not in (0, np.nan)
                            and np.isfinite(A_seg) and A_seg != 0
                            else complex("nan"))
                    f_used = float(schedule[i].freq)
                    # THD IS NOT MEASURABLE WHEN THE HARMONIC IS ITSELF A
                    # TONE.  Designed multisines are routinely built on
                    # small-integer ratios (1,2,3,5,8,...), so 2f or 3f is
                    # very often another member of the same group. Reporting
                    # the energy there as distortion would charge the
                    # excitation to the cell's non-linearity and fail the
                    # linearity gate on a perfectly linear measurement.
                    clash = np.any(np.abs(tone_set / (2 * f_used) - 1) < 0.02) \
                        or np.any(np.abs(tone_set / (3 * f_used) - 1) < 0.02)
                    thd[i] = (np.nan if clash else
                              utils.harmonic_distortion(ys, fam.fs, f_used))
                    drift[i] = utils.stationarity(ys, fam.fs, f_used)
                continue

            i, st = g[0], head
            if cfg.phasor_method == "joint7":
                jf = utils.fit7_joint(yr, ys, fam.fs, st.freq,
                                      n_iter=cfg.joint7_max_iter,
                                      tol=cfg.joint7_tol)
                A_ref, A_seg = jf.A_ref, jf.A_sig
                snr_r[i], snr_s[i] = jf.snr_ref_db, jf.snr_sig_db
                f_used = jf.freq
            else:                                  # legacy, for A/B only
                A_ref, _, snr_r[i] = utils.fit3(yr, fam.fs, st.freq)
                A_seg, _, snr_s[i] = utils.fit3(ys, fam.fs, st.freq)
                f_used = st.freq

            #  j_s = u_s / K   ->   Z = U_cell / j_s = K * A_ref / A_seg
            Z[i] = K * A_ref / A_seg if A_seg != 0 else complex("nan")
            thd[i] = utils.harmonic_distortion(ys, fam.fs, f_used)
            drift[i] = utils.stationarity(ys, fam.fs, f_used)

        # ---- wiring polarity ---------------------------------------------
        # A reversed sense pair inverts the WHOLE spectrum, so this decides
        # one sign for the segment.  The previous version judged it on the
        # high-frequency HALF of the band, which is the worst possible place:
        # up there the SNR collapses to -40 dB and the per-channel
        # acquisition skew rotates the phase by more than 90 deg, so the sign
        # of Re Z is essentially random.  Measured consequence on this plate:
        # segment 1 was left at Re Z = -460 mOhm*cm2 over its entire
        # low-frequency range -- an inverted but otherwise perfectly good
        # spectrum -- and was then destroyed by the passivity gate.
        #
        # The test now runs where the measurement is strong and skew-free:
        # the lowest decade of the band, weighted by SNR.  A delay cannot
        # rotate a 1 Hz point (0.04 deg at 100 us), and |Z| is at its largest
        # there, so the sign of Re Z is unambiguous.
        #
        # The Schneider et al. caution that motivated the old choice -- a
        # genuinely negative Re Z at low frequency from down-the-channel
        # starvation, ECS Trans. 25(1) 937 (2009) -- is respected by using a
        # WEIGHTED MEDIAN over the low decade rather than a mean, and by
        # requiring the evidence to be decisive: a couple of negative points
        # in an otherwise positive low-frequency spectrum will not flip it.
        with np.errstate(invalid="ignore"):
            Z = _fix_polarity(Z, freqs, utils.combine_snr_db(snr_r, snr_s), cfg)

        out[seg] = BronzeSpectrum(
            segment=seg, card=stem, freq=freqs, Z_raw=Z,
            snr_ref_db=snr_r, snr_seg_db=snr_s,
            snr_comb_db=utils.combine_snr_db(snr_r, snr_s),
            thd=thd, drift=drift, n_per_step=n_per, on_grid=on_grid,
            channel_slot=fam.position(seg), ref_slot=ref_slot,
            n_ch_on_card=fam.n_ch, fs=fam.fs,
            K=float(K), K_imputed=bool(imputed), T_degC=float(T),
            u_dc=float(np.mean(x[::cfg_stride(cfg)])), ref_name=ref_name,
        )

    n_imp = sum(1 for s in out.values() if s.K_imputed)
    slots = [s.channel_slot for s in out.values()]
    log.info(f"    {len(out)} segments extracted"
             + (f", {n_excluded} hardware-excluded" if n_excluded else "")
             + (f", {n_imp} with imputed calibration" if n_imp else "")
             + (f", slots {min(slots)}..{max(slots)} (ref {ref_slot})"
                if slots else ""))
    return out


# ===========================================================================
# 5. Entry point
# ===========================================================================


def run(cfg: Config = DEFAULT, log=None) -> BronzeRun:
    """Ingest every card into one BronzeRun."""
    log = log or utils.get_logger(cfg.verbose)
    utils.banner("BRONZE  --  raw ingestion", log)

    files = discover_files(cfg)
    log.info(f"  {len(files)} file(s) matching {cfg.famos_pattern()!r}")

    utils.section("channel inventory", log)
    channels, cards = inventory_channels(files, cfg, log)

    cal = PlateCalibration.load(cfg.curr_cal, cfg.temp_cal)
    if not cal.has_current_cal:
        raise SystemExit(
            "bronze: --curr-cal is required.  It is the only absolute scale "
            "left in the chain once the potentiostat is gone; without it the "
            "impedance has arbitrary units."
        )
    log.info(f"  calibration: {len(cal.seg_c0)}/{geom.N_SEGMENTS} segment rows, "
             f"{len(cal.temp_c0)} temperature sensors")

    # BEFORE ANY ALIGNMENT.  The correlation that follows cannot tell that
    # two cards recorded different runs; the headers can, and it is far
    # cheaper to refuse here than to explain a plate of failed SNR gates.
    timebase = timebase_report(cards, cfg, log)
    if not timebase.ok and cfg.require_timebase:
        raise SystemExit(
            "bronze: the cards do not form one consistent measurement:\n  - "
            + "\n  - ".join(timebase.problems)
            + "\n\nFix the selection, or pass --no-require-timebase to "
              "evaluate anyway and read every result as provisional.")

    lags = estimate_card_lags(files, cards, cfg, log, timebase=timebase)

    utils.section("plate temperature", log)
    rejected_sensors: dict[str, str] = {}
    T_seg, sensor_T = plate_temperatures(files, cards, cal, cfg, log,
                                         rejected=rejected_sensors)

    cal_report = calibration_report(cal, cfg, sensor_T, T_seg,
                                    rejected_sensors, log)

    schedule, grid = consensus_schedule(files, cards, cfg, log, lags=lags)

    utils.section("per-segment raw phasors", log)
    spectra: dict[str, BronzeSpectrum] = {}
    for fp in files:
        if fp.stem not in cards:
            continue
        log.info(f"  {fp.name}")
        info = lags.get(fp.stem, {})
        shift = int(info.get("lag", 0)) if info.get("applied") else 0
        got = process_card(fp, cal, schedule, grid, cfg, log,
                           T_seg=T_seg, lag=shift)
        for seg, sp in got.items():
            if seg in spectra:
                # two cards claim the same segment: keep the better SNR
                old = np.nanmedian(spectra[seg].snr_comb_db)
                new = np.nanmedian(sp.snr_comb_db)
                if not (new > old):
                    continue
                log.warning(f"    segment {seg} also on {spectra[seg].card}; "
                            f"keeping {sp.card} (SNR {new:.1f} > {old:.1f} dB)")
            spectra[seg] = sp

    run_obj = BronzeRun(
        schedule=schedule, channels=channels, spectra=spectra, cards=cards,
        grid=grid,
        config_digest=_digest([json.dumps(cfg.to_dict(), sort_keys=True)]),
        input_digest=_digest([f"{p.name}:{p.stat().st_size}" for p in files]),
        n_files=len(files), lags=lags, sensor_T=sensor_T,
        timebase=timebase.summary(), calibration=cal_report.summary(),
    )

    miss = run_obj.segments_missing()
    log.info(f"\n  bronze complete: {len(spectra)}/{geom.N_SEGMENTS} segments "
             f"carry raw data, {len(miss)} do not")
    if miss:
        log.info(f"  not measured: {', '.join(miss)}")
        log.info("  (these are NOT dropped - gold.py infers them from the "
                 "spatial field and marks them as inferred)")
    return run_obj


# ===========================================================================
# 6. Persistence
# ===========================================================================


def save(run_obj: BronzeRun, cfg: Config, log=None) -> Path:
    """Write bronze tables so silver can be re-run without touching .DAT."""
    log = log or utils.get_logger(cfg.verbose)
    out = Path(cfg.out_dir) / "bronze"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for seg in run_obj.segments_measured():
        sp = run_obj.spectra[seg]
        for i, f in enumerate(sp.freq):
            rows.append({
                "segment": seg, "card": sp.card, "freq_hz": f,
                "z_re_ohm_cm2": float(np.real(sp.Z_raw[i])),
                "z_im_ohm_cm2": float(np.imag(sp.Z_raw[i])),
                "snr_ref_db": float(sp.snr_ref_db[i]),
                "snr_seg_db": float(sp.snr_seg_db[i]),
                "snr_comb_db": float(sp.snr_comb_db[i]),
                "thd": float(sp.thd[i]), "drift": float(sp.drift[i]),
                "n_samples": int(sp.n_per_step[i]),
                "on_grid": int(sp.on_grid[i]),
            })
    utils.write_table(out / "raw_spectra.csv", rows)

    meta = []
    for seg in run_obj.segments_measured():
        sp = run_obj.spectra[seg]
        g = geom.SEGMENTS[seg]
        meta.append({
            "segment": seg, "card": sp.card,
            "area_cm2": round(g.area_cm2, 5),
            "cx_mm": round(g.cx_mm, 2), "cy_mm": round(g.cy_mm, 2),
            "channel_slot": sp.channel_slot, "ref_slot": sp.ref_slot,
            "n_ch_on_card": sp.n_ch_on_card,
            "slot_delta_us": round(sp.slot_delta_seconds * 1e6, 3),
            "fs_hz": sp.fs, "K": sp.K, "K_imputed": int(sp.K_imputed),
            "T_degC": round(sp.T_degC, 2), "u_dc_V": sp.u_dc,
            "j_dc_A_cm2": round(sp.j_dc(), 6),
            "n_points": sp.n_points, "ref_name": sp.ref_name,
        })
    utils.write_table(out / "segment_meta.csv", meta)

    utils.write_table(out / "schedule.csv", [
        {"index": i, "freq_hz": round(s.freq, 6), "start": s.start,
         "stop": s.stop, "n_samples": s.stop - s.start,
         "amp_V": s.amp, "snr_db": round(s.snr_db, 2),
         "thd": round(s.thd, 5) if np.isfinite(s.thd) else "",
         "drift": round(s.stationarity, 5) if np.isfinite(s.stationarity) else ""}
        for i, s in enumerate(run_obj.schedule)])

    utils.write_table(out / "channels.csv", [
        {"card": c.card, "name": c.name, "slot": c.slot, "kind": c.kind,
         "n_ch": c.n_ch_on_card, "slot_us": round(c.slot_seconds * 1e6, 3)}
        for c in sorted(run_obj.channels.values(), key=lambda c: (c.card, c.slot))])

    utils.write_json(out / "bronze_manifest.json", run_obj.summary())
    log.info(f"  bronze written to {out}")
    return out


if __name__ == "__main__":
    import sys as _sys

    # Detect Databricks: sys.argv is [''] or empty when run via "Run File"
    _has_cli_args = len(_sys.argv) > 1

    if _has_cli_args:
        try:
            cfg = Config.from_cli()
        except SystemExit:
            cfg = DEFAULT
    else:
        # Running inside Databricks without CLI args -- use Volume defaults
        cfg = DEFAULT.replace(
            dat_dir=Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Famos'),
            out_dir=Path('/tmp/eis_results/2611976'),
            leepa='2611976',
            condition='ALL',
            curr_cal=Path('/Workspace/Users/uum5fe@bosch.com/curr.csv'),
            temp_cal=Path('/Workspace/Users/uum5fe@bosch.com/temp.csv'),
        )
    r = run(cfg)
    save(r, cfg)
