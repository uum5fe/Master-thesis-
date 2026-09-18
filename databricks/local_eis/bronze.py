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
from pathlib import Path

import numpy as np

try:                      # package layout: core/ holds the science modules
    import core           # noqa: F401  (puts the core dir on sys.path)
except ImportError:       # flat layout (Databricks, notebooks): already there
    pass
import r2d2_geometry as geom
from eis_local import (FamosFile, PlateCalibration, detect_schedule,
                       pick_reference_channel, Step)
import hf_schedule
import ladder_snap

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
    #: Segments left out on purpose, from cfg.exclude_segments.
    excluded: frozenset = field(default_factory=frozenset)

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
            # EACH CARD'S OWN fs, NOT A CONSTANT.  A lag is measured in
            # SAMPLES; dividing by a hardcoded 10 kHz reports the right
            # number of seconds only on a 10 kHz card.  This campaign mixes
            # 50 kHz and 100 kHz cards, where the same constant understates
            # a lag by 5x and 10x -- and the lag is the number a reader uses
            # to decide whether the cards were aligned at all.
            "card_lag_s": {k: round(v["lag"] / self.cards[k].fs, 6)
                           for k, v in self.lags.items() if k in self.cards},
            "card_lag_corr": {k: round(v["corr"], 4)
                              for k, v in self.lags.items()},
            # WHICH LAGS WERE ACTUALLY USED, AND ON WHAT EVIDENCE.
            # A manifest that reports a lag without reporting whether it was
            # applied describes a run that may never have happened.
            "card_lag_applied": {k: bool(v.get("applied"))
                                 for k, v in self.lags.items()},
            "card_lag_prominence": {
                k: (round(v["prominence"], 2)
                    if np.isfinite(v.get("prominence", np.nan)) else None)
                for k, v in self.lags.items()},
            "card_lag_rescued": [k for k, v in self.lags.items()
                                 if v.get("rescued")],
            "card_clock_ppm": {
                k: round(v["drift"]["clock_mismatch_ppm"], 2) + 0.0
                for k, v in self.lags.items()
                if isinstance(v.get("drift"), dict) and v["drift"].get("ok")},
            "alignment_closure": next(
                (v["closure"] for v in self.lags.values() if "closure" in v),
                None),
            "sensor_T_degC": {k: round(v, 3) for k, v in self.sensor_T.items()},
            "coverage": self.coverage_summary(),
            # A segment that is absent because it was EXCLUDED and one that is
            # absent because nothing was recorded on it look identical in a
            # list of missing segments. Only the manifest can tell them apart,
            # so it says which were asked for.
            "excluded_segments": sorted(self.excluded, key=lambda s: int(s)
                                        if str(s).isdigit() else 0),
        }

    def coverage_summary(self) -> dict:
        """Per-frequency plate coverage, area-weighted.  See
        `bronze_frequency_coverage`."""
        return coverage_summary(bronze_frequency_coverage(self))


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


def inventory_channels(files: list[Path], cfg: Config,
                       log=None) -> tuple[dict[str, ChannelInfo], dict[str, CardInfo]]:
    """Open every card once and record what is on it, including slot order."""
    log = log or utils.get_logger(cfg.verbose)
    channels: dict[str, ChannelInfo] = {}
    cards: dict[str, CardInfo] = {}

    for fp in files:
        fam = FamosFile(fp)
        stem = fp.stem
        if not fam.uc_names:
            log.warning(f"  {fp.name}: no UC reference channel - card skipped")
            continue

        # THE REFERENCE IS THE SAME NAMED CHANNEL ON EVERY CARD.
        # One cell voltage is fanned out to all the cards on cfg.ref_channel
        # (UC2 on this campaign), so the reference is read off the wiring, not
        # re-guessed per file. See eis_local.pick_reference_channel for what
        # the old per-card argmax on std could do to the card alignment.
        ref = pick_reference_channel(fam, cfg.ref_channel,
                                     cfg_stride(cfg), log)
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
        )
        log.info(f"  {fp.name}: {fam.fs:.0f} Hz, {fam.n_ch} ch, "
                 f"{fam.n_samples/fam.fs:.1f} s, ref={ref} (slot {ref_slot}), "
                 f"{len(fam.segment_names)} segments")
    if not cards:
        raise SystemExit("bronze: no card carried a usable reference channel")
    return channels, cards


def cfg_stride(cfg: Config) -> int:
    """Decimation used for cheap statistics (std, DC level)."""
    return 10


# ===========================================================================
# 2b. Card alignment  (THE CARDS ARE ARMED SEPARATELY)
# ===========================================================================


def estimate_card_lags(files: list[Path], cards: dict[str, CardInfo],
                       cfg: Config, log=None) -> dict[str, dict]:
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
    Every card carries a copy of the same cell-voltage reference -- the one
    named by `cfg.ref_channel`, UC2 on this campaign -- so a plain cross-
    correlation of the band-passed reference gives the offset directly.  The
    correlation is dominated by the excitation window, which is where the
    signal is, so the estimate is well determined.

    That the SAME NAMED CHANNEL is compared on every card is what makes the
    lag mean anything.  While the reference was chosen per card by largest
    standard deviation, this could correlate one card's UC2 against another's
    UC1: two unrelated signals, whose peak can still clear the prominence
    gate and the floor, and whose "lag" then shifts every dwell window on
    that card onto the wrong tone.

    WHICH CARD IS THE ANCHOR  (the card, not the channel)
    -----------------------------------------------------
    The reference CHANNEL is fixed by name on every card (above).  Which
    CARD's clock the others are shifted onto is a separate question, and the
    answer is not simply the first one.  A card whose reference channel is degraded
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

    Returns {stem: {"lag", "corr", "prominence", "applied"}}.
    """
    log = log or utils.get_logger(cfg.verbose)
    utils.section("card alignment (shared reference cross-correlation)", log)

    stems = [fp.stem for fp in files if fp.stem in cards]
    if not stems:
        return {}

    traces = {stem: _alignment_trace(stem, cards, cfg) for stem in stems}

    # THE ANCHOR COMES FROM THE LARGEST GROUP OF CARDS THAT SHARE A RATE.
    # A lag in samples only means something between two cards sampled at the
    # same rate, and this campaign mixes rates. Choosing the anchor from all
    # cards at once could put it on a lone 100 kHz card and refuse the four
    # 50 kHz ones below -- refusing the majority to keep the outlier.
    by_fs: dict[float, list[str]] = {}
    for st in stems:
        by_fs.setdefault(float(cards[st].fs), []).append(st)
    group = max(by_fs.values(), key=lambda g: (len(g), -stems.index(g[0])))
    if len(by_fs) > 1:
        log.warning(f"  cards report {len(by_fs)} different sample rates "
                    f"({', '.join(f'{k:.0f} Hz x{len(v)}' for k, v in sorted(by_fs.items()))}). "
                    f"Aligning within the largest group only.")

    # max_lag and the guard are both TIMES converted to samples, so they are
    # taken from the rate of the group being aligned -- not from whichever
    # card happened to sort first, which on a mixed-rate plate is a different
    # number of samples for the same number of seconds.
    fs_group = cards[group[0]].fs
    max_lag = int(cfg.align_max_lag_s * fs_group)
    guard = _guard_samples(cfg, fs_group)

    base = _pick_anchor(group, traces, max_lag, log, guard=guard)
    ref = traces[base]
    out = {base: {"lag": 0, "corr": 1.0, "prominence": float("inf"),
                  "applied": True, "corroborated_by": [], "rescued": False}}
    log.info(f"  reference card: {base} (ref channel {cards[base].ref_name})")

    # A CARD AT A DIFFERENT SAMPLE RATE CANNOT BE CORRELATED WITH THIS ONE.
    # `_best_lag` compares two arrays sample by sample and returns an offset
    # in samples; if the two cards did not sample at the same rate, those
    # samples are not the same unit and the answer is meaningless -- the more
    # so because `max_lag` is derived from the anchor's fs alone. The
    # campaign does mix rates (50 kHz and 100 kHz cards appear in the same
    # run), so this is not hypothetical. Refusing is the only honest
    # outcome; a resampling path would be a real change, not a guard.
    fs_base = cards[base].fs
    mismatched = [s for s in stems
                  if s != base and not np.isclose(cards[s].fs, fs_base,
                                                  rtol=0, atol=1e-6)]
    for stem in mismatched:
        out[stem] = {"lag": 0, "corr": float("nan"), "prominence": float("nan"),
                     "applied": False, "corroborated_by": [], "rescued": False,
                     "refused_reason": "sample rate differs from the anchor"}
        log.warning(f"  {stem[-8:]}: fs = {cards[stem].fs:.0f} Hz against the "
                    f"anchor's {fs_base:.0f} Hz - NOT aligned. A lag in "
                    f"samples means nothing between two rates; this card "
                    f"stays on its own clock.")

    # ALL CANDIDATES FIRST, THEN THE DECISION.
    # The acceptance rule below asks whether ANOTHER card independently
    # returned the same offset, so no card can be judged until every card has
    # been measured.
    candidates: dict[str, tuple[int, float, float]] = {}
    for stem in stems:
        if stem == base or stem in mismatched:
            continue
        candidates[stem] = _best_lag(traces[stem], ref, max_lag, guard)

    for stem, (lag, corr, prom) in candidates.items():
        # lag < 0 means this card's record RUNS AHEAD of the reference's,
        # i.e. it was armed later; the schedule index must be shifted by
        # +lag to read the same instant.
        partners = _agreeing_cards(stem, lag, candidates, cards, cfg)
        rescued = bool(partners
                       and prom < cfg.align_min_prominence
                       and prom >= cfg.align_corroborate_min_prominence)
        strong = prom >= cfg.align_min_prominence
        above_floor = abs(corr) >= cfg.align_min_corr
        applied = bool(cfg.align_cards and above_floor and (strong or rescued))
        out[stem] = {"lag": lag, "corr": corr, "prominence": prom,
                     "applied": applied, "corroborated_by": partners,
                     "rescued": rescued}
        why = ("" if applied else
               "   REFUSED (below the absolute floor)" if not above_floor else
               "   REFUSED (peak not prominent, and no card agrees)")
        note = (f"   CORROBORATED by {', '.join(p[-8:] for p in partners)}"
                if rescued else "")
        log.info(f"  {stem[-8:]}: lag {lag:+8d} samples = "
                 f"{lag / cards[stem].fs:+8.4f} s   peak corr {corr:+.3f}"
                 f"   prominence {prom:5.1f}" + why + note)
        if not applied:
            log.warning(f"    {stem[-8:]} stays on its own clock. If its true "
                        f"offset is not ~0 its dwell windows will land on the "
                        f"wrong tone and EVERY segment on this card will fail "
                        f"the SNR gate in silver.")
            continue
        if abs(lag) >= max_lag - 1:
            log.warning("    lag is at the search limit - raise "
                        "align_max_lag_s and re-run")

        # ---- is ONE constant lag enough for the whole record? -------------
        drift = _blockwise_lag_diagnostic(traces[stem], ref, lag,
                                          cards[stem].fs, cfg)
        out[stem]["drift"] = drift
        if drift.get("ok"):
            log.info(f"    clock: {drift['clock_mismatch_ppm']:+7.2f} ppm over "
                     f"{len(drift['blocks'])} blocks "
                     f"(residual {drift['residual_span_samples']:+d} samples "
                     f"end to end)")
            if not drift.get("within_limit", True):
                log.warning(
                    f"    residual clock mismatch "
                    f"{drift['clock_mismatch_ppm']:+.1f} ppm exceeds "
                    f"{cfg.align_max_clock_ppm:.0f} ppm: a single constant lag "
                    f"does not align the whole record. The dwell windows at "
                    f"one end of the sweep are the ones that will suffer.")

    # ---- does the whole set of lags form one consistent timing network? ---
    closure = _pairwise_closure(stems, traces, out, cards, cfg, guard)
    if closure.get("pairs"):
        # Plate-level facts live on the ANCHOR's entry, not under a key of
        # their own: `out` is keyed by card stem and several callers iterate
        # it expecting every value to be a card record (summary() reads
        # v["corr"] off each one). The anchor is the card the whole timing
        # network is expressed against, so it is where a statement about that
        # network belongs.
        out[base]["closure"] = closure
        out[base]["band_hz"] = [float(cfg.align_f_lo_hz),
                                float(cfg.align_f_hi_hz)]
        out[base]["guard_s"] = float(cfg.align_guard_s)
        if closure["max_abs_error_samples"] is not None:
            log.info(f"  pairwise closure: max |error| "
                     f"{closure['max_abs_error_samples']} samples "
                     f"({closure['max_abs_error_s']:+.4f} s), RMS "
                     f"{closure['rms_error_samples']:.1f} samples over "
                     f"{closure['n_pairs']} pairs")
            if not closure["consistent"]:
                log.warning(
                    f"  the card offsets do NOT close: measuring A against B "
                    f"directly disagrees with going through the anchor by up "
                    f"to {closure['max_abs_error_s']:+.4f} s. At least one "
                    f"accepted lag is wrong, and the closure table says which "
                    f"pair carries it.")

    if not cfg.align_cards:
        log.warning("  align_cards is off: schedule windows will be applied "
                    "to every card unshifted, which is only correct if the "
                    "cards were hardware-triggered together")
    return out


#: The guard the prominence score used before it was expressed as a time.
#: Kept only so a direct `_best_lag` call without a config still behaves as
#: it always did; production passes `_guard_samples(cfg, fs)`.
GUARD_SAMPLES_LEGACY = 5000


def _guard_samples(cfg: Config, fs: float) -> int:
    """The prominence guard for this card, in samples.

    `align_guard_s` is a duration because the peak it has to exclude is a
    duration -- roughly 1/align_f_lo_hz wide, whatever the sample rate.
    """
    g = float(getattr(cfg, "align_guard_s", 0.0) or 0.0)
    if g <= 0.0:
        return GUARD_SAMPLES_LEGACY
    return max(1, int(round(g * float(fs))))


def _alignment_trace(stem: str, cards: dict[str, CardInfo],
                     cfg: Config) -> np.ndarray:
    """The card's reference, band-passed to the band the lag is measured in.

    The band kills DC drift at the bottom and, at the top, the part of the
    spectrum where two cards genuinely differ in phase -- neither of which
    carries information about the OFFSET between the records.  It limits the
    alignment estimate only; the impedance spectrum is built from the full
    band in `process_card`.

    The upper limit is clamped to 0.45*fs of THIS card.  Without the clamp a
    band chosen for the fast cards asks a slow card for frequencies it never
    recorded, and `X[(fr < f_lo) | (fr > f_hi)] = 0` then zeroes its entire
    trace -- an all-zero correlation, which is not a refusal but a silent
    one.
    """
    c = cards[stem]
    fam = FamosFile(c.path)
    x = np.asarray(fam.channel(c.ref_name), float)
    x = x - x.mean()

    n = len(x)
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1.0 / c.fs)

    f_lo = max(0.0, float(cfg.align_f_lo_hz))
    f_hi = min(float(cfg.align_f_hi_hz), 0.45 * c.fs)
    if not 0.0 <= f_lo < f_hi:
        raise ValueError(
            f"bronze: invalid alignment band {f_lo:g}..{f_hi:g} Hz for "
            f"fs = {c.fs:g} Hz on {stem}. align_f_lo_hz must be below both "
            f"align_f_hi_hz and 0.45*fs.")

    X[(fr < f_lo) | (fr > f_hi)] = 0.0
    return np.fft.irfft(X, n)


def _best_lag(x: np.ndarray, ref: np.ndarray, max_lag: int,
              guard: int = GUARD_SAMPLES_LEGACY) -> tuple[int, float, float]:
    """The lag of best agreement, its correlation, and its prominence."""
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
    return int(lag_sel[k]), corr, _prominence(np.abs(cc_sel), k, guard)


def _prominence(mag: np.ndarray, k: int,
                guard: int = GUARD_SAMPLES_LEGACY) -> float:
    """How many robust sigma the peak stands above the rest of the curve.

    Median and MAD rather than mean and sd, because the correlation of a
    stepped sweep against itself is not flat -- it has broad structure that
    a mean-based score would read as signal.  The guard band excludes the
    peak's own shoulders, which are part of the peak, not of the background.
    """
    lo, hi = max(0, k - guard), min(mag.size, k + guard + 1)
    background = np.concatenate([mag[:lo], mag[hi:]])
    if background.size < 64:
        return float("nan")
    med = float(np.median(background))
    mad = float(np.median(np.abs(background - med)))
    if mad <= 0:
        return float("inf") if mag[k] > med else 0.0
    return float((float(mag[k]) - med) / (1.4826 * mad))


def _agreeing_cards(stem: str, lag: int,
                    candidates: dict[str, tuple[int, float, float]],
                    cards: dict[str, CardInfo], cfg: Config) -> list[str]:
    """Which other cards independently returned the same offset.

    A weak peak that a SECOND card reproduces is not the same claim as a weak
    peak on its own.  The cards are armed in groups, so two of them genuinely
    sharing a trigger will genuinely share an offset; noise will not put two
    independent correlations within 20 ms of each other at an 8.6 s lag.

    This is evidence ABOUT PROMINENCE ONLY.  `align_min_corr` is not relaxed
    by agreement, and that matters: on RO2612030 at 150 A cards 1 and 2 agreed
    to 53 samples on a 215634-sample offset while scoring |r| = 0.083 on a
    dead reference, and that pair is exactly what the 0.50 floor exists to
    refuse.  Under the shipped floor they stay refused, agreement or not --
    corroboration buys a card past a ragged peak, never past a dead one.

    Set `align_corroborate_min_prominence` above `align_min_prominence` to
    switch the mechanism off entirely.
    """
    partners: list[str] = []
    fs = cards[stem].fs
    tol = int(round(cfg.align_agree_tol_s * fs))
    for other, (other_lag, _corr, other_prom) in candidates.items():
        if other == stem:
            continue
        if not np.isclose(cards[other].fs, fs, rtol=0, atol=1e-6):
            continue
        # A partner must itself clear the corroboration floor: two peaks that
        # are both indistinguishable from noise cannot vouch for each other,
        # however well they agree.
        if other_prom < cfg.align_corroborate_min_prominence:
            continue
        if abs(int(other_lag) - int(lag)) <= tol:
            partners.append(other)
    return partners


def _blockwise_lag_diagnostic(x: np.ndarray, ref: np.ndarray, global_lag: int,
                              fs: float, cfg: Config) -> dict:
    """Does ONE constant lag hold for the whole record, or do the clocks drift?

    A single global lag corrects a different START time.  It says nothing
    about the two cards keeping the same sample RATE: if their oscillators
    differ by even a few ppm, the offset that is right at the start of a
    300 s record is wrong at the end.  At 20 ppm over 300 s the records slide
    by 6 ms, which is a quarter of a 25 ms dwell -- enough to walk a window
    off its tone at the end of the sweep while the beginning still looks
    perfect.

    Measured by re-estimating the lag in blocks.  The key point, and the one
    the naive version gets wrong: the two blocks compared must cover the SAME
    PHYSICAL INTERVAL.  x is delayed relative to ref by `global_lag`, so
    ref[a:b] corresponds to x[a+lag : b+lag], and only then is the local
    result a RESIDUAL that should sit near zero.  Slicing both at [a:b] and
    searching +-50 ms would be searching for an 8 s offset in a 50 ms window
    and finding nothing.

    Report-only: this never rejects a lag.  It says whether a constant lag
    was the right model, which is a different question from whether the lag
    that was found is the best constant one.
    """
    n_blocks = int(getattr(cfg, "align_drift_blocks", 0) or 0)
    if n_blocks < 3:
        return {"ok": False, "reason": "disabled (align_drift_blocks < 3)"}

    lag = int(global_lag)
    # the stretch of ref for which the matching stretch of x exists
    lo = max(0, -lag)
    hi = min(len(ref), len(x) - lag)
    if hi - lo < 1024:
        return {"ok": False, "reason": "records do not overlap after the lag"}

    block_len = (hi - lo) // n_blocks
    half = max(2, int(round(cfg.align_drift_half_window_s * fs)))
    # The guard cannot be the global one here: it excludes +-align_guard_s
    # around the peak, and the whole local search is only +-half.  A guard
    # wider than the search leaves no background at all, and `_prominence`
    # returns NaN.  Keep three quarters of the window as background.
    guard_local = max(1, (2 * half + 1) // 8)

    rows = []
    for b_i in range(n_blocks):
        a = lo + b_i * block_len
        b = hi if b_i == n_blocks - 1 else lo + (b_i + 1) * block_len
        if b - a < 256:
            continue
        local, corr, _prom = _best_lag(x[a + lag:b + lag], ref[a:b],
                                       half, guard_local)
        rows.append({
            "block": b_i,
            "time_s": float(0.5 * (a + b) / fs),
            "lag_samples": int(lag + local),
            "residual_samples": int(local),
            "corr": float(corr),
        })

    if len(rows) < 3:
        return {"ok": False, "reason": "fewer than three usable blocks",
                "blocks": rows}

    tt = np.array([r["time_s"] for r in rows], float)
    ll = np.array([r["lag_samples"] for r in rows], float)
    slope, intercept = np.polyfit(tt, ll, 1)
    ppm = 1e6 * slope / float(fs)
    residuals = [r["residual_samples"] for r in rows]
    at_limit = [r for r in rows if abs(r["residual_samples"]) >= half - 1]
    return {
        "ok": True,
        "lag_intercept_samples": float(intercept),
        "lag_slope_samples_per_s": float(slope),
        "clock_mismatch_ppm": float(ppm),
        "within_limit": bool(abs(ppm) <= cfg.align_max_clock_ppm),
        "residual_span_samples": int(max(residuals) - min(residuals)),
        # a residual pinned at the edge of the search window means the true
        # drift is larger than this window can see, so the ppm is a floor
        "n_blocks_at_search_limit": len(at_limit),
        "half_window_samples": int(half),
        "blocks": rows,
    }


def _pairwise_closure(stems: list[str], traces: dict[str, np.ndarray],
                      lags: dict[str, dict], cards: dict[str, CardInfo],
                      cfg: Config, guard: int) -> dict:
    """Do the anchor-relative lags agree with direct card-to-card lags?

    Every lag here is measured against one anchor, so nothing so far has
    tested the set of them for CONSISTENCY.  Measuring A against B directly
    must reproduce lag(A) - lag(B), because the card index of a feature is
    `anchor_index + lag` on each card by construction.  If it does not, one
    of the two accepted lags is wrong -- and the closure table names the pair,
    which is more than the correlation scores can do on their own.

    Only cards whose lag was APPLIED take part: an unapplied lag is not a
    claim about the timing network, so including it would manufacture errors
    that mean nothing.

    Report-only, like the drift diagnostic.  A closure failure says the set
    cannot all be right; it does not say which one to throw away, and
    guessing is exactly what got this pipeline into trouble before.
    """
    usable = [s for s in stems
              if lags.get(s, {}).get("applied") and s in traces]
    rows: list[dict] = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            fs_a, fs_b = cards[a].fs, cards[b].fs
            if not np.isclose(fs_a, fs_b, rtol=0, atol=1e-6):
                rows.append({"card_a": a, "card_b": b, "ok": False,
                             "reason": "different sample rates"})
                continue
            max_lag = int(cfg.align_max_lag_s * fs_a)
            direct, corr, prom = _best_lag(traces[a], traces[b], max_lag, guard)
            # a_index = anchor + lag_a and b_index = anchor + lag_b, so a
            # measured against b must come out at lag_a - lag_b
            implied = int(lags[a]["lag"]) - int(lags[b]["lag"])
            error = int(direct) - implied
            rows.append({
                "card_a": a, "card_b": b, "ok": True,
                "direct_lag": int(direct), "implied_lag": implied,
                "closure_error_samples": error,
                "closure_error_s": float(error / fs_a),
                "corr": float(corr), "prominence": float(prom),
            })

    errors = [abs(r["closure_error_samples"]) for r in rows if r.get("ok")]
    fs_ref = cards[usable[0]].fs if usable else 1.0
    tol = int(round(cfg.align_agree_tol_s * fs_ref))
    return {
        "n_pairs": len(errors),
        "max_abs_error_samples": max(errors) if errors else None,
        "max_abs_error_s": float(max(errors) / fs_ref) if errors else None,
        "rms_error_samples": (float(np.sqrt(np.mean(np.square(errors))))
                              if errors else None),
        "tolerance_samples": tol,
        "consistent": bool(errors and max(errors) <= tol) if errors else True,
        "pairs": rows,
    }


def _pick_anchor(stems: list[str], traces: dict[str, np.ndarray],
                 max_lag: int, log,
                 guard: int = GUARD_SAMPLES_LEGACY) -> str:
    """The card whose clock the others are shifted onto: the one they agree
    with best.

    This chooses a CARD, not a channel.  Every trace handed here is the same
    named reference channel (`cfg.ref_channel`, UC2), read off each card;
    what is being chosen is which card's t0 the plate is expressed in.

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
        others = [abs(_best_lag(traces[s], traces[cand], max_lag, guard)[1])
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
    hf_info: dict[str, dict] = {}
    fs_seen: dict[str, float] = {}
    for fp in files:
        stem = fp.stem
        if stem not in cards:
            continue
        fam = FamosFile(fp)
        f_hi = cfg.f_hi(fam.fs)
        # DETECT ON THE CURRENT-CARRYING ENSEMBLE, NOT ON THE CELL VOLTAGE.
        # The sweep is galvanostatic, so the amplitude arriving on the UC
        # reference is |i_ac| * |Z_cell(f)| and it falls with |Z_cell| by an
        # order of magnitude across the band; the segment channels measure
        # current density, which the sweep holds constant, so their tone is
        # flat in frequency.  hf_schedule stacks them and hands the result to
        # detect_schedule UNCHANGED -- the estimator is the same one, only
        # its input improved -- then fits the sweep's ladder on the confident
        # low-frequency steps and verifies each predicted high-frequency rung
        # on a frequency CFAR and a rank-1 test across the array, neither of
        # which uses the ladder.  Measured on RO2612025-01 card 4 at 45 A:
        # 11 -> 21 steps from the stack, 42 with the extension, taking the
        # recovered band from 0.478-7.47 Hz to 0.478-18.9 kHz.
        hf = None
        if getattr(cfg, "hf_use_ensemble", False) and fam.segment_names:
            hf = hf_schedule.recover_schedule(
                hf_schedule.LazyChannels(fam), fam.fs,
                f_lo=cfg.f_min_hz, f_hi=f_hi, ppd=cfg.ppd,
                min_snr_db=cfg.min_snr_db,
                sigma_rel_max=cfg.sigma_rel_max,
                extend=getattr(cfg, "hf_ladder_extend", True),
                snap_ppd=getattr(cfg, "hf_ladder_snap_ppd", True),
                ladder_tol=getattr(cfg, "hf_ladder_tol", 0.02),
                prune=getattr(cfg, "hf_ladder_prune", False),
                # the old path's trace is pooled in, not replaced: nothing
                # the shipped pipeline would have found can be lost
                uc_ref=fam.channel(cards[stem].ref_name),
                log=log)
            hf_info[stem] = hf.summary()
        if hf is not None and hf.as_steps():
            steps = hf.as_steps()
        else:
            # no segment channels, or the stack found nothing: the old path
            # is still the right fallback, not an error
            if hf is not None:
                log.warning(f"  {stem}: the stacked ensemble found nothing - "
                            f"falling back to the {cards[stem].ref_name} "
                            f"reference channel")
            ref = fam.channel(cards[stem].ref_name)
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
            steps = [ladder_snap._rebuild(Step, s, start=s.start - d,
                                          stop=s.stop - d) for s in steps]
        per_card[stem] = steps
        fs_seen[stem] = float(fam.fs)
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
        # _finite_median, not nanmedian: a cluster whose every card reported
        # a NaN THD is not an error, it is a step where nobody could measure
        # distortion, and nanmedian answers that with a RuntimeWarning and a
        # NaN. The NaN is the right answer; the warning is noise that trains
        # the reader to ignore warnings.
        thd = _finite_median(s.thd for _, s in cl)
        drift = _finite_median(s.stationarity for _, s in cl)
        kept.append(ladder_snap._rebuild(Step, best, freq=float(f_hat),
                                         snr_db=snr, thd=thd,
                                         stationarity=drift))
        n_votes_kept += enough
        n_grid_rescued += (on_grid and not enough)

    kept.sort(key=lambda s: s.freq)

    # A MISSING DIAGNOSTIC IS "NOT MEASURED", NEVER "PASSED".
    # THD and stationarity are quality evidence; a step that has neither is a
    # step nothing is known about, and silver must not read that silence as a
    # clean bill of health. The counts are recorded so a run that lost its
    # distortion diagnostics says so in the manifest instead of looking like
    # a run with no distortion.
    grid["diagnostics"] = {
        "n_steps": len(kept),
        "steps_without_thd": int(sum(not np.isfinite(s.thd) for s in kept)),
        "steps_without_stationarity":
            int(sum(not np.isfinite(s.stationarity) for s in kept)),
    }
    _d = grid["diagnostics"]
    if _d["steps_without_thd"] or _d["steps_without_stationarity"]:
        log.warning(f"  {_d['steps_without_thd']} of {len(kept)} steps carry "
                    f"no THD and {_d['steps_without_stationarity']} no "
                    f"stationarity: those diagnostics are UNAVAILABLE for "
                    f"them, not passed")

    # ---- snap the consensus onto the excitation ladder ---------------------
    # The grid above is used only to ACCEPT or REJECT a cluster.  Using it to
    # CORRECT the frequency is strictly stronger, because the sine fit is
    # evaluated at the reported frequency and a 1 % error over a 0.25 s dwell
    # at 600 Hz is 1.7 cycles of phase slip -- ~16 dB of phasor SNR.  That is
    # the whole of the 590.29 Hz failure: the rung is at 596.99 Hz, and the
    # cards that reported 590.29 lost the step to the SNR gate while the cards
    # that reported 597 kept it.  Snapping also merges the duplicate
    # detections that share one dwell window (71 -> 45 steps at 45 A,
    # 77 -> 45 at 450 A on RO2612030), which is what `grid_tol` was failing to
    # do.  Nothing is deleted here; the quality gates still decide.
    if getattr(cfg, "ladder_snap", True) and kept:
        fs_ref = float(np.median(list(fs_seen.values()))) if fs_seen else 25000.0
        snapped, snap_info = ladder_snap.snap_steps(
            kept, fs_ref,
            ppd=getattr(cfg, "ladder_snap_ppd", None), log=log)
        if snap_info.get("ok"):
            kept = snapped
            # THE SNAPPED LADDER *IS* THE GRID NOW.  Overwrite the fitted one.
            # `_on_grid` decides which of the two SNR gates a point faces in
            # silver (snr_floor_db when on grid, min_snr_db when not).  On
            # RO2612030 the free fit broke on a single 1.46 %-apart pair, came
            # back at 158 points/decade, and `on_grid` was then false for all
            # 4828 rows -- so snr_floor_db never applied to anything and the
            # whole top of the band was gated at min_snr_db.  After snapping,
            # every step lies on the ladder by construction, so the grid is
            # exact rather than fitted and that failure cannot recur.
            grid = dict(grid)
            grid.update(ok=True, ppd=float(snap_info["ppd"]),
                        ratio=10.0 ** (1.0 / snap_info["ppd"]),
                        f0=float(kept[-1].freq),
                        n_on_grid=len(kept), n_total=len(kept),
                        from_ladder_snap=True)
            grid["ladder_snap"] = snap_info

    # ---- dwell-window sanity ----------------------------------------------
    # The snap corrects FREQUENCIES; it does not touch windows.  A rung whose
    # only detection was spurious therefore keeps a window that cannot belong
    # to the sweep, and silver then fits a sine at the right frequency over
    # the wrong stretch of record and every segment rejects the point.  A
    # stepped sweep is monotonic in time, so those windows are identifiable
    # without any extra data and can be replaced by the one the sweep would
    # have placed there.
    if getattr(cfg, "window_sanity", True) and len(kept) >= 4:
        fs_ref = float(np.median(list(fs_seen.values()))) if fs_seen else 25000.0
        fixed, win_info = ladder_snap.repair_windows(
            kept, fs_ref,
            min_dwell_frac=getattr(cfg, "window_min_dwell_frac", 0.40),
            max_repair_frac=getattr(cfg, "window_max_repair_frac", 0.25),
            log=log)
        if win_info.get("ok"):
            kept = fixed
            grid["window_sanity"] = win_info

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
    if hf_info:
        grid["hf_schedule"] = hf_info
    return kept, grid


def bronze_frequency_coverage(run_obj) -> list[dict]:
    """How much of the plate actually carries each frequency.

    A schedule step is a statement that the plate was excited at a frequency.
    It is NOT a statement that the frequency survived on any given segment,
    and an aggregate built from a schedule alone can be a plate-wide number
    computed from four segments in one corner.

    Two fractions, because they answer different questions:

      segment_fraction  -- what share of the measured segments returned a
                           finite phasor here.  Answers "is this frequency
                           broadly measured, or did one card carry it?"
      area_coverage     -- the same, weighted by segment AREA.  This is the
                           one that matters for any aggregate, because the
                           plate's segments span 0.678 to 8.470 cm^2, a
                           factor of 12.5: two thirds of the segments can be
                           a third of the plate.

    Report-only.  Nothing is discarded for low coverage here -- the right
    response to a thinly covered frequency is to say so next to the number,
    not to delete it and leave the gap unexplained.
    """
    rows: list[dict] = []
    segs = list(run_obj.spectra.items())
    if not segs:
        return rows
    areas = {seg: float(geom.SEGMENTS[seg].area_cm2)
             for seg, _ in segs if seg in geom.SEGMENTS}
    total_area = float(sum(areas.values()))

    for i, st in enumerate(run_obj.schedule):
        n_valid = 0
        area_ok = 0.0
        for seg, sp in segs:
            z = sp.Z_raw
            ok = bool(i < len(z) and np.isfinite(z[i].real)
                      and np.isfinite(z[i].imag)
                      and i < len(sp.n_per_step) and sp.n_per_step[i] > 0)
            if ok:
                n_valid += 1
                area_ok += areas.get(seg, 0.0)
        rows.append({
            "index": i,
            "freq_hz": float(st.freq),
            "n_valid_segments": int(n_valid),
            "n_measured_segments": int(len(segs)),
            "segment_fraction": round(n_valid / len(segs), 4) if segs else 0.0,
            "area_coverage": (round(area_ok / total_area, 4)
                              if total_area > 0 else float("nan")),
            "window_source": getattr(st, "window_source", "detected"),
        })
    return rows


def coverage_summary(rows: list[dict], min_area_coverage: float = 0.80) -> dict:
    """What the coverage table says in one line, for the manifest.

    `min_area_coverage` is a REPORTING threshold, not a gate: it marks the
    band over which an aggregate rests on most of the plate, so that a
    comparison against a whole-cell Gamry sweep can state the band it is
    entitled to use instead of quietly averaging over whatever survived.
    """
    if not rows:
        return {"ok": False, "reason": "no spectra"}
    ac = np.array([r["area_coverage"] for r in rows], float)
    f = np.array([r["freq_hz"] for r in rows], float)
    good = np.isfinite(ac) & (ac >= min_area_coverage)
    rep = [r for r in rows if r.get("window_source") != "detected"]
    return {
        "ok": True,
        "min_area_coverage": float(min_area_coverage),
        "n_steps": len(rows),
        "n_steps_above_threshold": int(good.sum()),
        "f_lo_covered_hz": float(f[good].min()) if good.any() else None,
        "f_hi_covered_hz": float(f[good].max()) if good.any() else None,
        "median_area_coverage": float(np.nanmedian(ac)) if ac.size else None,
        "worst_area_coverage": float(np.nanmin(ac)) if ac.size else None,
        "n_repaired_windows": len(rep),
        "repaired_freqs_hz": [round(r["freq_hz"], 3) for r in rep],
    }


def _finite_median(values) -> float:
    """Median of the finite values, or NaN when there are none.

    `np.nanmedian` of an all-NaN slice is NaN with a RuntimeWarning, which is
    the right value announced the wrong way: a step whose cards all failed to
    measure THD is an ordinary outcome near the top of the band, not an
    exceptional condition, and a run that prints dozens of All-NaN warnings
    teaches its reader to skip warnings.
    """
    a = np.asarray(list(values), float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


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
# 4. Per-card extraction
# ===========================================================================


def pooled_reference_phasors(files: list[Path], cards: dict[str, CardInfo],
                             schedule: list[Step], cfg: Config,
                             lags: dict[str, dict] | None = None, log=None
                             ) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """One cell-voltage phasor per step, averaged over every card.

    WHY THIS AND NOT AVERAGING THE SEGMENTS
    ---------------------------------------
    Averaging segment impedances across cards is wrong: they are DIFFERENT
    SEGMENTS, and the whole point of the plate is that they differ.  The five
    UC channels are not different quantities -- they are five measurements of
    ONE cell voltage, made by five converters with uncorrelated front-end
    noise.  Averaging them is valid, and it pays exactly where it is needed,
    because once detection has moved onto the segment ensemble the reference
    is the weak phasor in Z = K * A_ref / A_seg.  Five cards buy about 7 dB.

    TWO THINGS HAVE TO BE TRUE FIRST
    --------------------------------
    1.  The cards must be on a common time base, or the sum is partially
        destructive at the top of the band.  Only cards whose lag was
        APPLIED are pooled; a card whose alignment was refused contributes
        nothing rather than contributing noise with a phase error.
    2.  The multiplexer slot of each UC channel is a real per-card delay of
        slot/(n_ch*fs).  Each card's phasor is rotated back to SLOT 0 before
        it is averaged, and `process_card` then records `ref_slot = 0`, so
        silver's structural skew model still sees a consistent geometry.

    Weights are inverse residual variance, w = N / r_rms^2, which is the
    Cramer-Rao weighting for a phasor and is what makes this an estimator
    rather than an average.  The pooled SNR adds in linear power, as
    independent noise on a coherent signal does.

    Returns (A_ref, snr_ref_db, info), or None when fewer than two cards
    qualify -- in which case `process_card` uses its own reference and
    nothing changes.
    """
    log = log or utils.get_logger(cfg.verbose)
    n_st = len(schedule)
    if n_st == 0:
        return None

    num = np.zeros(n_st, complex)
    den = np.zeros(n_st, float)
    gam = np.zeros(n_st, float)
    n_cards = 0
    used: list[str] = []
    for fp in files:
        stem = fp.stem
        if stem not in cards:
            continue
        info = lags.get(stem, {}) if lags else {}
        if info and not info.get("applied", True):
            continue
        shift = int(info.get("lag", 0)) if info.get("applied") else 0
        fam = FamosFile(fp)
        name = cards[stem].ref_name
        if name not in fam.names:
            continue
        ref = fam.channel(name)
        slot_dt = fam.position(name) / (fam.n_ch * fam.fs)
        n_cards += 1
        used.append(stem)
        for i, st in enumerate(schedule):
            a, b = st.start + shift, st.stop + shift
            if b <= a or a < 0 or b > len(ref):
                continue
            A, r_rms, snr = utils.fit3(ref[a:b], fam.fs, st.freq)
            if not np.isfinite(snr) or not np.isfinite(A) or r_rms <= 0:
                continue
            # rotate this card's UC phasor from its own mux slot to slot 0
            A = A * np.exp(2j * np.pi * st.freq * slot_dt)
            w = (b - a) / (r_rms ** 2)
            num[i] += w * A
            den[i] += w
            gam[i] += 10.0 ** (snr / 10.0)

    if n_cards < 2:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        A_ref = np.where(den > 0, num / np.where(den > 0, den, 1.0),
                         complex("nan"))
        snr_db = np.where(gam > 0, 10.0 * np.log10(np.where(gam > 0, gam, 1.0)),
                          np.nan)
    n_ok = int(np.sum(np.isfinite(A_ref)))
    log.info(f"  reference pooled over {n_cards} card(s) "
             f"({', '.join(c[-8:] for c in used)}): {n_ok}/{n_st} steps, "
             f"median SNR {np.nanmedian(snr_db):.1f} dB")
    return A_ref, snr_db, {"n_cards": n_cards, "cards": used,
                           "n_steps": n_ok}


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
                       cal: PlateCalibration, cfg: Config,
                       log=None) -> tuple[dict[str, float], dict]:
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
        fam = FamosFile(fp)
        for tn in fam.temp_names:
            key = _sensor_key(tn)
            if key not in cal.temp_c0:
                log.warning(f"  {tn}: no calibration row for {key!r} - skipped")
                continue
            u = float(np.mean(fam.channel(tn)[::cfg_stride(cfg)]))
            T = cal.temperature(key, u)
            if 0.0 < T < 120.0:
                sensor_T[key] = T
            else:
                log.warning(f"  {tn}: u={u:.4f} V -> T={T:.1f} C implausible "
                            f"- sensor ignored")
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
                 lag: int = 0,
                 ref_pool: tuple[np.ndarray, np.ndarray, dict] | None = None
                 ) -> dict[str, BronzeSpectrum]:
    """Raw phasors for every segment on one card.

    `schedule` windows are indices on the COMMON time base; `lag` is this
    card's offset relative to it, so the window actually read is
    (start + lag, stop + lag).

    `ref_pool` is the plate-wide cell-voltage phasor from
    `pooled_reference_phasors`, already rotated to mux slot 0.  When it is
    given it REPLACES this card's own UC phasor in Z = K*A_ref/A_seg -- the
    reference is the weak measurement now that detection runs on the segment
    ensemble, and five cards measuring one cell voltage is the one average
    across cards that is physically legitimate.  The segment phasor, the
    frequency and every gate stay exactly as they were; only A_ref and its
    SNR change, and `ref_slot` is then recorded as 0 so that silver's
    structural skew model still reads a consistent geometry.
    """
    log = log or utils.get_logger(cfg.verbose)
    fam = FamosFile(fp)
    stem = fp.stem
    if not fam.uc_names:
        return {}

    # the same named channel inventory_channels and the alignment used, so
    # A_ref here is measured on the signal the lags were measured on
    ref_name = pick_reference_channel(fam, cfg.ref_channel,
                                      cfg_stride(cfg), log)
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

    A_pool = snr_pool = None
    if ref_pool is not None:
        A_pool, snr_pool, _pool_info = ref_pool
        # the pooled phasor was rotated to slot 0, so that is the slot the
        # skew model must be told about
        ref_slot = 0
        ref_name = f"pooled({_pool_info.get('n_cards', 0)} cards)"
        log.info(f"    reference: plate-wide pool, "
                 f"{_pool_info.get('n_cards', 0)} card(s), slot 0")

    out: dict[str, BronzeSpectrum] = {}
    n_excluded = 0
    # A SUBSTITUTED SEGMENT IS DROPPED HERE TOO.
    # The point of substituting is that this segment's own measurement is not
    # trusted; reading it and then overwriting it downstream would leave the
    # untrusted number in the raw tables, where something would eventually
    # use it. It is rebuilt from its neighbours in silver instead.
    _skip = set(cfg.exclude_segments) | set(
        getattr(cfg, "substitute_segments", ()) or ())
    for seg in fam.segment_names:
        if seg in _skip:
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
        for i, st in enumerate(schedule):
            a, b = st.start + lag, st.stop + lag
            if b <= a or a < 0 or b > len(x) or b > len(ref):
                n_skip += 1
                continue
            yr, ys = ref[a:b], x[a:b]
            n_per[i] = b - a

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

            # The joint fit still estimates the FREQUENCY from both channels
            # together, which is where its factor-of-six CRLB advantage comes
            # from; what the pool replaces is only the reference amplitude
            # and phase, which is the noisiest term in the ratio.
            if A_pool is not None and np.isfinite(A_pool[i]):
                A_ref, snr_r[i] = A_pool[i], snr_pool[i]

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
             + (f", {n_excluded} excluded by config" if n_excluded else "")
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

    lags = estimate_card_lags(files, cards, cfg, log)

    utils.section("plate temperature", log)
    T_seg, sensor_T = plate_temperatures(files, cards, cal, cfg, log)

    schedule, grid = consensus_schedule(files, cards, cfg, log, lags=lags)

    utils.section("per-segment raw phasors", log)
    # One cell-voltage phasor per step, pooled over every aligned card.  This
    # is the one average across cards that is physically legitimate: the five
    # UC channels measure ONE cell voltage, while the segment channels
    # measure different segments and must never be averaged together.
    ref_pool = (pooled_reference_phasors(files, cards, schedule, cfg,
                                         lags=lags, log=log)
                if getattr(cfg, "hf_pool_reference", False) else None)
    spectra: dict[str, BronzeSpectrum] = {}
    for fp in files:
        if fp.stem not in cards:
            continue
        log.info(f"  {fp.name}")
        info = lags.get(fp.stem, {})
        shift = int(info.get("lag", 0)) if info.get("applied") else 0
        got = process_card(fp, cal, schedule, grid, cfg, log,
                           T_seg=T_seg, lag=shift, ref_pool=ref_pool)
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
        excluded=frozenset(str(x) for x in (cfg.exclude_segments or ())),
    )

    miss = run_obj.segments_missing()
    log.info(f"\n  bronze complete: {len(spectra)}/{geom.N_SEGMENTS} segments "
             f"carry raw data, {len(miss)} do not")
    if miss:
        # Separate the two kinds of absence. "Not measured" invites a hunt for
        # a wiring fault; "left out on purpose" does not, and a reader cannot
        # tell them apart from a list of numbers.
        left_out = sorted((set(miss) & run_obj.excluded), key=int)
        unmeasured = sorted((set(miss) - run_obj.excluded), key=int)
        if unmeasured:
            log.info(f"  not measured: {', '.join(unmeasured)}")
        if left_out:
            log.info(f"  excluded on purpose: {', '.join(left_out)} "
                     f"(exclude_segments) - these carry no data anywhere "
                     f"downstream and are not inferred")
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

    # `window_source` travels to the CSV because a reader deciding whether to
    # believe a point needs to know whether its dwell window was found or
    # predicted, and a log line that scrolled past two stages ago is not an
    # answer to that.
    cov = {r["index"]: r for r in bronze_frequency_coverage(run_obj)}
    utils.write_table(out / "schedule.csv", [
        {"index": i, "freq_hz": round(s.freq, 6), "start": s.start,
         "stop": s.stop, "n_samples": s.stop - s.start,
         "amp_V": s.amp, "snr_db": round(s.snr_db, 2),
         "thd": round(s.thd, 5) if np.isfinite(s.thd) else "",
         "drift": round(s.stationarity, 5) if np.isfinite(s.stationarity) else "",
         "window_source": getattr(s, "window_source", "detected"),
         "window_repaired": int(getattr(s, "window_repaired", False)),
         "n_valid_segments": cov.get(i, {}).get("n_valid_segments", ""),
         "segment_fraction": cov.get(i, {}).get("segment_fraction", ""),
         "area_coverage": cov.get(i, {}).get("area_coverage", "")}
        for i, s in enumerate(run_obj.schedule)])

    utils.write_table(out / "channels.csv", [
        {"card": c.card, "name": c.name, "slot": c.slot, "kind": c.kind,
         "n_ch": c.n_ch_on_card, "slot_us": round(c.slot_seconds * 1e6, 3)}
        for c in sorted(run_obj.channels.values(), key=lambda c: (c.card, c.slot))])

    utils.write_table(out / "frequency_coverage.csv",
                      bronze_frequency_coverage(run_obj))

    # The alignment evidence, one row per card: what was measured, what was
    # believed, and why. Previously only the log carried it, so a cached run
    # could not be audited after the fact.
    utils.write_table(out / "card_alignment.csv", [
        {"card": k,
         "lag_samples": v.get("lag"),
         "lag_s": (round(v["lag"] / run_obj.cards[k].fs, 6)
                   if k in run_obj.cards else ""),
         "corr": round(v["corr"], 4) if np.isfinite(v.get("corr", np.nan)) else "",
         "prominence": (round(v["prominence"], 2)
                        if np.isfinite(v.get("prominence", np.nan)) else ""),
         "applied": int(bool(v.get("applied"))),
         "rescued": int(bool(v.get("rescued"))),
         "corroborated_by": " ".join(v.get("corroborated_by", []) or []),
         "refused_reason": v.get("refused_reason", ""),
         "clock_ppm": (round(v["drift"]["clock_mismatch_ppm"], 2) + 0.0
                       if isinstance(v.get("drift"), dict)
                       and v["drift"].get("ok") else "")}
        for k, v in sorted(run_obj.lags.items())])

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