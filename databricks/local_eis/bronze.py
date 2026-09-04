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
from eis_local import FamosFile, PlateCalibration, detect_schedule, Step
import hf_schedule

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
            "sensor_T_degC": {k: round(v, 3) for k, v in self.sensor_T.items()},
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

        # the reference is the UC channel carrying the most AC content
        ref = max(fam.uc_names,
                  key=lambda c: float(np.std(fam.channel(c)[::cfg_stride(cfg)])))
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
    Every card carries a copy of the same cell-voltage reference, so a plain
    cross-correlation of the band-passed reference gives the offset directly.
    The correlation is dominated by the excitation window, which is where the
    signal is, so the estimate is well determined.

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

    Returns {stem: {"lag", "corr", "prominence", "applied"}}.
    """
    log = log or utils.get_logger(cfg.verbose)
    utils.section("card alignment (shared reference cross-correlation)", log)

    stems = [fp.stem for fp in files if fp.stem in cards]
    if not stems:
        return {}

    def _ac(stem):
        c = cards[stem]
        fam = FamosFile(c.path)
        x = np.asarray(fam.channel(c.ref_name), float)
        x = x - x.mean()
        # keep the band that carries the sweep; kills DC drift and the top
        # of the band where the two cards genuinely differ in phase
        n = len(x)
        X = np.fft.rfft(x)
        fr = np.fft.rfftfreq(n, 1.0 / c.fs)
        X[(fr < 0.5) | (fr > 300.0)] = 0.0
        return np.fft.irfft(X, n)

    traces = {stem: _ac(stem) for stem in stems}
    max_lag = int(cfg.align_max_lag_s * cards[stems[0]].fs)

    base = _pick_anchor(stems, traces, max_lag, log)
    ref = traces[base]
    out = {base: {"lag": 0, "corr": 1.0, "prominence": float("inf"),
                  "applied": True}}
    log.info(f"  reference card: {base} (ref channel {cards[base].ref_name})")

    for stem in stems:
        if stem == base:
            continue
        lag, corr, prom = _best_lag(traces[stem], ref, max_lag)
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
                 f"{lag / cards[stem].fs:+8.4f} s   peak corr {corr:+.3f}"
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
    return out


def _best_lag(x: np.ndarray, ref: np.ndarray,
              max_lag: int) -> tuple[int, float, float]:
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
    return int(lag_sel[k]), corr, _prominence(np.abs(cc_sel), k)


def _prominence(mag: np.ndarray, k: int, guard: int = 5000) -> float:
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


def _pick_anchor(stems: list[str], traces: dict[str, np.ndarray],
                 max_lag: int, log) -> str:
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
        others = [abs(_best_lag(traces[s], traces[cand], max_lag)[1])
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
    if hf_info:
        grid["hf_schedule"] = hf_info
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
