#!/usr/bin/env python3
"""
hf_schedule.py  --  recovering the top decade of the sweep
==========================================================

THE PROBLEM
-----------
`bronze.consensus_schedule` runs the blind detector on each card's REFERENCE
channel, chosen in `inventory_channels` as the UC* channel with the largest
standard deviation -- that is, the cell voltage.

The sweep is galvanostatic.  The ac current amplitude is set by the load and
is constant across the sweep, so the amplitude that arrives on the voltage
channel is

    |u_ref(f)| = |i_ac| * |Z_cell(f)|

and |Z_cell| falls by roughly an order of magnitude from the bottom of the
band to the ~45 mOhm*cm2 minimum near 8 kHz.  The detector is being asked to
find a tone exactly where the cell has removed it.  That is not a
threshold-tuning problem: no value of `min_snr_db` puts signal back.

The SEGMENT channels behave the other way round.  They measure current
density (j_s = u_s / K), and current is what the sweep imposes, so their tone
amplitude is flat in frequency.  There are ~14 of them per card, driven by
the same tone at the same instant, with independent front-end noise.

The information is already in the file, on the channels the detector never
reads.

WHAT THIS MODULE DOES
---------------------
Three layers, in decreasing order of how much they are trusted.  No new
ESTIMATOR is introduced anywhere: layer 2 hands the work to the pipeline's
own `eis_local.detect_schedule`, unchanged.  What changes is which trace it
is given.

  1. STACK THE CURRENT-CARRYING CHANNELS.  `polarity_aligned_reference()`
     standardises every segment channel, checks each one's sign against a
     provisional sum so that a reversed sense pair cannot subtract, and adds
     them.  Tones add coherently, independent noise adds in power:
     ~10*log10(M) dB of array gain on top of the ~20 dB gained by not using
     the voltage channel.

  2. RUN THE PIPELINE'S OWN DETECTOR ON IT.  Same log-grid scan, same
     demodulated-envelope dwell localisation, same IEEE-1057 refinement,
     same gap fill.

  3. FIT THE LADDER, PREDICT THE REST, VERIFY EACH PREDICTION.  A stepped
     sweep is f_k = f0 * r^-k with a dwell law max(d_min, n_cyc/f_k) -- five
     numbers, fitted on the CONFIDENT low-frequency steps only, so the weak
     steps cannot vote on the model that validates them.  Each predicted rung
     is then localised in time and accepted on two tests that are independent
     of the ladder: a frequency-domain CFAR with an exact binomial threshold,
     and a rank-1 (maximum-eigenvalue) test across the channel array.

WHY THE LADDER ARGUMENT IS NOT CIRCULAR
---------------------------------------
`consensus_schedule` already makes this argument and already guards it: the
grid is fitted on `prov_snr >= cfg.min_snr_db` steps only, with the note
about 18 true steps producing 55 "on-grid" detections when the guard was
absent.  This module keeps that discipline and adds the missing half.  Grid
membership in `consensus_schedule` only ever RELAXES A GATE for a candidate
that was already detected; it never goes looking for a rung that produced no
detection at all.  That is the step that recovers the top of the band, and
the two acceptance tests below are what keep it honest.

SNAP THE SPACING TO AN INTEGER
------------------------------
Extrapolating a geometric ladder compounds the ratio error.  On the 45 A
card-4 record a ladder fitted freely on the fifteen steps below 12 Hz
returned 10.059 points/decade.  That 0.13 % error in r compounds with the
rung index: checked blind against fifteen tones observed independently
between 946 Hz and 24 kHz, the prediction error grew monotonically from
+3.4 % to +5.8 % -- outside any sane acceptance window, so the extension
found noise.  Snapping the spacing to the nearest integer, 10 points/decade,
brings the same blind prediction to -0.9 .. +1.0 %, mean |error| 0.39 %, and
the extension then verifies 27 of 32 predicted rungs.  Generators use round
numbers; the fit does not know that and the snap tells it.

WHAT THIS DOES NOT DO
---------------------
Detection gain is not estimation gain.  The array recovers WHICH frequency
and WHEN, with the ensemble's SNR.  The per-segment impedance still rests on
that one segment's phasor.  What the array buys there is indirect: a known f
in a known window drops the Cramer-Rao phase penalty from 6/(N*gamma) to
1/(N*gamma) -- a factor of sqrt(6) -- and removes the runaway-fit mechanism
entirely.

REFERENCES
----------
D. C. Rife, R. R. Boorstyn, IEEE Trans. Inf. Theory 20 (1974) 591
    -- var(A)/A^2 >= 1/(N*gamma); the threshold effect that breaks blind
       frequency search at low SNR.  This is why `crlb_usable` below
       thresholds on N*gamma and not on gamma.
Y. Zeng, Y.-C. Liang, IEEE Trans. Commun. 57 (2009) 1784
    -- eigenvalue-based detection with unknown noise power; the rank-1 test.
IEEE Std 1057-2017, clause 7        -- the sine fits this module calls.
M. Schoenleber, D. Klotz, E. Ivers-Tiffee, Electrochim. Acta 131 (2014) 20
    -- Lin-KK residual, the complementary acceptance test in silver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ===========================================================================
# 0. The gate that matters:  N*gamma, not gamma
# ===========================================================================


def crlb_usable(snr_db: float, n: int, sigma_rel_max: float = 0.60,
                freq_known: bool = True) -> bool:
    """Is a phasor from `n` samples at `snr_db` precise enough to keep?

    `Step.valid` used to gate on `snr_db >= min_snr_db`, where `fit3` defines
    `snr_db` as the tone amplitude over the residual rms across the whole
    Nyquist band.  That is not what determines a phasor's precision.  Rife &
    Boorstyn give

        sigma_A / A  >=  sqrt(1 / (N * gamma))

    so a long dwell beats the noise down and a short one does not, at the
    same SNR.  Measured on RO2612025-01 card 4 at 45 A: of 23 rungs located
    above 100 Hz, 10 pass the 5 dB gate and all 23 pass sigma_rel_max = 0.60,
    with sigma_rel between 0.3 % and 23 %.  The 11.95 kHz step reports
    -1.5 dB and has N*gamma = 2459 -- a 2.0 % phasor, thrown away by a gate
    that was never meant to decide this.

    `config.py` already argues exactly this in its own comment above
    `sigma_rel_max`; the criterion was simply applied in silver, after bronze
    had already discarded the step in `detect_schedule`.
    """
    if not np.isfinite(snr_db) or n is None or n < 4:
        return False
    gamma = 10.0 ** (float(snr_db) / 10.0)
    denom = max(float(n) * gamma, 1e-12)
    k = 1.0 if freq_known else 6.0
    return bool(math.sqrt(k / denom) <= float(sigma_rel_max))


# ===========================================================================
# 1. Layer 1 -- stack the current-carrying channels
# ===========================================================================


class LazyChannels:
    """Read-through view of one card's segment channels.

    `FamosFile.channel` materialises a full float64 copy of the channel, and
    a 538 s record at 50 kHz is 215 MB of that -- times fourteen channels it
    does not fit anywhere sensible.  Everything below either streams the
    channels one at a time or wants a short WINDOW out of each, so this hands
    out slices from the memmap instead of whole arrays.

    A plain ``{name: ndarray}`` dict works everywhere this is accepted; this
    class only exists so that bronze does not have to hold the card in RAM.
    """

    def __init__(self, fam, names: list[str] | None = None):
        self.fam = fam
        self.names = list(names if names is not None else fam.segment_names)

    def keys(self):
        return list(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __contains__(self, name) -> bool:
        return name in self.names

    def __getitem__(self, name: str) -> np.ndarray:
        return self.fam.channel(name)

    def window(self, name: str, a: int, b: int) -> np.ndarray:
        return np.asarray(self.fam.channel(name)[a:b], dtype=np.float64)


def _names(chans) -> list[str]:
    return list(chans.keys())


def _window(chans, name: str, a: int, b: int) -> np.ndarray:
    """One window out of one channel, without materialising the channel."""
    fn = getattr(chans, "window", None)
    if fn is not None:
        return fn(name, a, b)
    return np.asarray(chans[name][a:b], dtype=np.float64)


def _standardise(x: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance, NaNs removed.

    Standardising before the sum is what makes the stack a MAXIMAL-RATIO
    combiner rather than a weighted-by-accident one: the segment channels
    differ by their Abgleich constant K, which is a per-segment gain of no
    interest here and would otherwise let one channel dominate the sum.
    """
    y = np.asarray(x, dtype=np.float64)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = y - y.mean()
    s = float(y.std())
    return y / s if s > 0 else y


def polarity_aligned_reference(chans, stride: int = 0,
                               log=None) -> tuple[np.ndarray, dict]:
    """Sum the segment channels into one virtual reference trace.

    A reversed sense pair inverts a whole channel.  Summed blind, such a
    channel SUBTRACTS the tone instead of adding it, and with a handful of
    them the array gain goes the wrong way.  So: build a provisional sum,
    then take each channel's sign from its correlation with the provisional
    sum MINUS ITSELF (a channel always correlates with a sum it is part of),
    and rebuild.

    Returns (trace, info).  `trace` is standardised again at the end so that
    downstream amplitude thresholds -- `demod_envelope`'s peak-over-median
    test in particular -- see the same scale they would on a raw channel.
    """
    names = _names(chans)
    if not names:
        return np.zeros(0), {"n_channels": 0, "n_flipped": 0}

    total = None
    kept: list[str] = []
    for n in names:
        y = _standardise(chans[n])
        if y.size == 0 or not np.any(y):
            continue
        total = y if total is None else total + y
        kept.append(n)
    if total is None:
        return np.zeros(0), {"n_channels": 0, "n_flipped": 0}

    # The sign only needs the sign of a correlation, so it can be read off a
    # decimated copy; at 50 kHz over 500 s a stride of 50 still leaves half a
    # million points per channel.
    if stride <= 0:
        stride = max(1, total.size // 500_000)
    ref_s = total[::stride]

    signs: dict[str, int] = {}
    out = None
    for n in kept:
        y = _standardise(chans[n])
        ys = y[::stride]
        # subtract this channel's own contribution before correlating
        r = float(np.dot(ys, ref_s - ys))
        s = -1 if r < 0 else 1
        signs[n] = s
        out = (s * y) if out is None else out + s * y

    n_flip = sum(1 for v in signs.values() if v < 0)
    if log is not None and n_flip:
        log.info(f"    stacked reference: {n_flip}/{len(kept)} channel(s) "
                 f"entered with reversed polarity")

    info = {"n_channels": len(kept), "n_flipped": n_flip,
            "channels": kept, "signs": signs,
            "array_gain_db": float(10.0 * np.log10(max(len(kept), 1)))}
    return _standardise(out), info


# ===========================================================================
# 2. Layer 3 -- the ladder and its dwell law
# ===========================================================================


@dataclass
class Ladder:
    """f_k = f0 * ratio^-k, with dwell_s(f) = max(d_min, n_cyc / f)."""
    ok: bool = False
    f0: float = float("nan")
    ratio: float = float("nan")
    ppd: float = float("nan")
    ppd_free: float = float("nan")     # before the integer snap
    snapped: bool = False
    n_cyc: float = float("nan")        # cycles per dwell
    d_min: float = float("nan")        # dwell floor, seconds
    d_fixed: float = float("nan")      # dwell length when it does not scale
    dwell_mode: str = "cycles"         # "cycles" | "fixed"
    dwell_resid: float = float("nan")  # median |predicted/observed - 1|
    settle_s: float = 0.0              # dead time between dwells
    descending: bool = True            # frequency falls with time
    n_fitted_on: int = 0
    resid: float = float("nan")        # median |df/f| of the fitted steps
    k_mean: float = 0.0                # centroid of the fitted rung indices
    sigma_lnr: float = 0.0             # standard error of the fitted spacing
    sigma_lnf0: float = 0.0            # standard error of the offset
    note: str = ""

    def freq_of(self, k: float) -> float:
        return float(self.f0 * self.ratio ** (-k))

    def index_of(self, f: float) -> float:
        return float(np.log(self.f0 / f) / np.log(self.ratio))

    def tol_at(self, f: float, base: float = 0.02) -> float:
        """Membership window at `f`, widened by the fit's own uncertainty.

        A FIXED window is wrong, and wrong in a way that costs real steps.
        The ladder is a straight-line fit in log f, so its prediction error
        grows with distance from the centroid of the steps it was fitted on:
        |k - k_mean| * sigma_lnr, plus the offset's own error.  On the
        synthetic card set a ladder fitted at 4.9013 points/decade against a
        true 4.8891 -- 0.25 %, a perfectly good fit -- ran 4 % out at the
        ends of the band and a flat 2 % window then threw away three genuine
        steps, including the top one at 3.07 kHz.  This is the same
        compounding that the integer snap addresses when the generator used
        a round number; when it did not, the error is still there and has to
        be carried rather than assumed away.
        """
        if not self.ok:
            return float(base)
        dk = abs(self.index_of(f) - self.k_mean)
        return float(np.hypot(base,
                              np.hypot(dk * self.sigma_lnr, self.sigma_lnf0)))

    def dwell_s(self, f: float) -> float:
        """How long the generator holds this rung.

        TWO LAWS ARE IN COMMON USE and they are not close to each other.  A
        generator asked for a fixed CYCLE COUNT gives dwell = n_cyc/f, which
        is milliseconds at the top of the band; one asked for a fixed TIME
        gives the same seconds at every rung.  Assuming the first when the
        record uses the second mispredicts a high-frequency step's start by
        the whole length of the sweep, so the search bracket lands on a
        different tone entirely and the rung is refused.  `fit_ladder` fits
        both and keeps whichever describes the observed dwells better.
        """
        if self.dwell_mode == "fixed" and np.isfinite(self.d_fixed):
            return float(self.d_fixed)
        return float(max(self.d_min, self.n_cyc / max(f, 1e-12)))


def _snap_ppd(ppd_free: float, rel_tol: float = 0.02) -> tuple[float, bool]:
    """Round the spacing to the nearest integer points/decade when it is
    close enough that a generator would have been asked for that number.

    `rel_tol` is deliberately loose relative to the ladder's own residual: a
    free fit on fifteen clean low-frequency steps recovered 10.059 against a
    true 10, an 0.6 % error in ppd, and refusing the snap at 0.6 % would keep
    exactly the drift the snap exists to remove.  It is tight enough that a
    genuine 7.5 points/decade sweep is left alone.
    """
    if not np.isfinite(ppd_free) or ppd_free <= 0:
        return ppd_free, False
    target = round(ppd_free)
    if target < 1:
        return ppd_free, False
    if abs(ppd_free - target) <= rel_tol * ppd_free:
        return float(target), True
    return float(ppd_free), False


def fit_ladder(freqs, starts=None, stops=None, fs: float = 1.0,
               tol: float = 0.01, snap: bool = True) -> Ladder:
    """Recover the sweep's geometric ladder and dwell law.

    Pass ONLY the steps that are trusted on their own evidence.  The whole
    point of the ladder is to validate the weak steps; if the weak steps are
    allowed to vote on it, the fit tracks the noise and validates it.
    """
    f = np.array(sorted({float(v) for v in freqs
                         if np.isfinite(v) and v > 0}), float)
    if len(f) < 5:
        return Ladder(note=f"only {len(f)} confident steps, need 5")

    ln = np.log(f)
    d = np.diff(ln)
    d = d[(d > 1e-6) & np.isfinite(d)]
    if len(d) < 3:
        return Ladder(note="no usable ratios")

    # Seed with the smallest observed gap: a missed step makes an observed
    # gap an integer MULTIPLE of the true one, never a fraction of it.
    seed = float(np.median(d[d <= np.percentile(d, 40)]))

    # THE TIE MUST BREAK TOWARDS THE COARSER LADDER.
    # Every step of a 10 points/decade sweep also lies on a 20 points/decade
    # ladder -- on every other rung of it -- so a subdivision can never fit
    # FEWER of the observed steps, and a score that rewards ties picks the
    # finest one offered.  The invented rungs in between then get predicted,
    # and at the top of the band a half-step is a fraction of a DFT bin away
    # from a real tone, so the acceptance tests see the neighbour's leakage
    # and pass.  Measured cost when this was not guarded: 20 spurious steps
    # accepted on a synthetic whose true schedule the coarse ladder gets
    # exactly right.  So: try the divisions in order, and take a finer one
    # only if it explains STRICTLY MORE steps than the coarsest that works.
    # `consensus_schedule` already makes the same argument about its own
    # geometric grid; this is that guard, in the ladder.
    best = None
    for div in (1, 2, 3):
        lr = seed / div
        if lr <= 1e-6:
            continue
        k = np.round((ln[-1] - ln) / lr)      # k = 0 at the highest frequency
        if len(np.unique(k)) < 4:
            continue
        A = np.column_stack([np.ones_like(k), -k])
        sol, *_ = np.linalg.lstsq(A, ln, rcond=None)
        ln_f0, lr_fit = sol
        if lr_fit <= 1e-6:
            continue
        rel = np.abs(np.expm1(ln - (ln_f0 - k * lr_fit)))
        n_on = int(np.sum(rel <= tol))
        if best is None or n_on > best[0]:
            best = (n_on, float(ln_f0), float(lr_fit), rel)

    if best is None:
        return Ladder(note="no consistent ladder")
    n_on, ln_f0, lr_fit, rel = best
    if n_on < max(5, int(0.7 * len(f))):
        return Ladder(note=f"only {n_on}/{len(f)} steps on the fitted ladder")

    # Second guard, exact rather than heuristic.  If the fitted spacing is
    # q times too fine, then every observed step lands on a rung whose index
    # is congruent mod q -- the interleaved rungs are empty BY CONSTRUCTION,
    # because they do not exist.  A genuinely finer ladder with steps merely
    # missing does not produce that regularity: its gaps are irregular.  So
    # take the gcd of the observed rung indices and coarsen by it.
    #
    # Comparing the fitted spacing against the MEDIAN GAP of the confident
    # steps, which is the obvious guard and the one `consensus_schedule`
    # uses, does not work here: the confident set is the subset that passed
    # an SNR gate, so it has real holes in it, and the median gap reads two
    # rungs where the sweep's spacing is one.  Tested: that version refused
    # the correct 10 points/decade ladder outright.
    k_obs = np.round((ln[-1] - ln) / lr_fit).astype(int)
    k_uni = np.unique(k_obs - k_obs.min())
    g = 0
    for v in k_uni:
        g = math.gcd(g, int(v))
    if g > 1:
        lr_fit *= g
        k_obs = np.round((ln[-1] - ln) / lr_fit)
        A = np.column_stack([np.ones_like(k_obs), -k_obs])
        sol, *_ = np.linalg.lstsq(A, ln, rcond=None)
        ln_f0, lr_fit = float(sol[0]), float(sol[1])
        rel = np.abs(np.expm1(ln - (ln_f0 - k_obs * lr_fit)))

    ppd_free = float(1.0 / (lr_fit / np.log(10.0)))

    def _refit(ppd_try):
        """Hold the spacing, let f0 absorb the change, and score it."""
        lr_ = np.log(10.0) / ppd_try
        k_ = np.round((ln[-1] - ln) / lr_)
        ln_f0_ = float(np.mean(ln + k_ * lr_))
        return lr_, k_, ln_f0_, np.abs(np.expm1(ln - (ln_f0_ - k_ * lr_)))

    lr, k, ln_f0, rel = _refit(ppd_free)
    ppd, snapped = _snap_ppd(ppd_free) if snap else (ppd_free, False)
    if snapped:
        # THE SNAP HAS TO EARN ITS PLACE.  It is a prior about generators,
        # not a measurement, so it may not be allowed to make the fit worse:
        # on one synthetic card a free 4.904 points/decade snapped to 5.000
        # against a true 4.889, and the ladder then explained its own
        # confident steps to only 2 % -- after which it pruned 21 of 26
        # detections and cost a decade of band.  Take the snap only when the
        # snapped ladder still describes those steps about as well.
        lr_s, k_s, ln_f0_s, rel_s = _refit(ppd)
        if np.median(rel_s) <= max(2.0 * np.median(rel), tol):
            lr, k, ln_f0, rel = lr_s, k_s, ln_f0_s, rel_s
        else:
            ppd, snapped = ppd_free, False

    # Standard errors of the straight-line fit ln f = ln f0 - k ln r, used
    # by `tol_at` to widen the membership window away from the centroid.
    dev = ln - (ln_f0 - k * lr)
    dof = max(len(ln) - 2, 1)
    s_res = float(np.sqrt(np.sum(dev ** 2) / dof))
    k_mean = float(np.mean(k))
    sxx = float(np.sum((k - k_mean) ** 2))
    lad = Ladder(ok=True, f0=float(np.exp(ln_f0)),
                 ratio=float(np.exp(lr)), ppd=ppd, ppd_free=ppd_free,
                 snapped=snapped, n_fitted_on=len(f),
                 resid=float(np.median(rel)), k_mean=k_mean,
                 sigma_lnr=float(s_res / np.sqrt(sxx)) if sxx > 0 else 0.0,
                 sigma_lnf0=float(s_res / np.sqrt(len(ln))))

    # A LADDER THAT FITS ITS OWN STEPS NO BETTER THAN THE MEMBERSHIP WINDOW
    # IS NOT EVIDENCE ABOUT ANY OTHER STEP.  Everything the ladder is used
    # for -- pruning a detection, predicting a rung, validating a weak one --
    # rests on it being sharper than the window it is compared against.  At
    # 2 % residual against a 1 % tolerance it is not, and using it anyway is
    # how a bad fit on one card silently removed a decade of band.
    if not np.isfinite(lad.resid) or lad.resid > tol:
        return Ladder(note=f"ladder residual {100 * lad.resid:.2f} % exceeds "
                           f"the {100 * tol:.2f} % tolerance - not trusted")

    # ---- dwell law ------------------------------------------------------
    # A stepped-sine generator holds a fixed CYCLE COUNT per step until that
    # would take less than some floor, then holds the floor.  Both numbers
    # come straight off the windows the detector already found.
    if starts is not None and stops is not None and fs > 0:
        st = np.asarray(starts, float)
        sp = np.asarray(stops, float)
        order = np.argsort([float(v) for v in freqs])
        dwell = (sp - st)[order] / fs
        fo = np.array(sorted(float(v) for v in freqs), float)
        good = np.isfinite(dwell) & (dwell > 0) & np.isfinite(fo) & (fo > 0)
        if good.sum() >= 3:
            d_obs, f_obs = dwell[good], fo[good]
            lad.n_cyc = float(np.median(d_obs * f_obs))
            lad.d_min = float(np.min(d_obs))
            lad.d_fixed = float(np.median(d_obs))
            # THE MEAN, NOT THE MEDIAN.  Under a fixed-time record the
            # cycles law is exactly right on every rung above n_cyc/d_min --
            # roughly half of them, since n_cyc is itself a median -- and
            # badly wrong below it.  A median error therefore reads zero for
            # a law that describes half the sweep and nothing else, and the
            # wrong law wins the comparison.
            res_cyc = float(np.mean(np.abs(
                np.maximum(lad.d_min, lad.n_cyc / f_obs) / d_obs - 1.0)))
            res_fix = float(np.mean(np.abs(lad.d_fixed / d_obs - 1.0)))
            lad.dwell_mode = "cycles" if res_cyc <= res_fix else "fixed"
            lad.dwell_resid = min(res_cyc, res_fix)
            # dead time between consecutive dwells, read off the record
            t0 = st[order][good] / fs
            if len(t0) >= 3:
                gaps = np.diff(np.sort(t0)) - dwell[good][np.argsort(t0)][:-1]
                gaps = gaps[np.isfinite(gaps) & (gaps > -lad.d_min)]
                if len(gaps):
                    lad.settle_s = float(max(np.median(gaps), 0.0))
            # a sweep that runs downward in frequency has its HIGH
            # frequencies EARLY in the record
            if len(t0) >= 3:
                lad.descending = bool(np.corrcoef(fo[good], t0)[0, 1] < 0)
    if not np.isfinite(lad.n_cyc):
        lad.n_cyc, lad.d_min = 8.0, 0.0
        lad.note = "dwell law not fitted; using 8 cycles"
    return lad


def prune_off_ladder(freqs, lad: Ladder, tol: float = 0.02) -> np.ndarray:
    """Boolean mask: which of `freqs` sit on the ladder.

    This is the direction the SNR gate provably cannot handle.  The card-4
    record carries a continuous interferer at 6.79 kHz holding 12.9 % of the
    stacked trace's ac power -- far stronger than several genuine rungs -- but
    it sits 9.9 % off the ladder, so membership rejects it without any
    amplitude test at all.
    """
    f = np.asarray(freqs, float)
    if not lad.ok:
        return np.ones(f.shape, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.round(np.log(lad.f0 / f) / np.log(lad.ratio))
        rel = np.abs(f / (lad.f0 * lad.ratio ** (-k)) - 1.0)
    tol_eff = np.array([lad.tol_at(v, tol) for v in np.atleast_1d(f)])
    return np.isfinite(rel) & (rel <= tol_eff.reshape(rel.shape))


# ===========================================================================
# 3. Layer 3 -- acceptance tests independent of the ladder
# ===========================================================================


def _binom_sf_threshold(m: int, p: float, alpha: float = 1e-3) -> int:
    """Smallest c with P(Binomial(m, p) >= c) <= alpha.

    Under H0 -- no tone, independent channels -- the number of channels whose
    in-band bin beats every bin of their own guard band is Binomial(m, p)
    with p = 1/(1 + n_bg).  That is an EXACT threshold: no noise-power
    calibration and no tuning constant.
    """
    m = int(max(m, 1))
    p = min(max(float(p), 1e-9), 1.0 - 1e-9)
    tail = 0.0
    for c in range(m, -1, -1):
        tail += math.comb(m, c) * p ** c * (1.0 - p) ** (m - c)
        if tail > alpha:
            return min(c + 1, m)
    return 1


@dataclass
class CfarResult:
    """Outcome of the frequency-domain CFAR at one candidate rung."""
    array_is_max: bool = False   # the ARRAY periodogram peaks in the tested bin
    count: int = 0               # channels whose own bin beats their own band
    threshold: int = 1           # exact binomial threshold for `count`
    m: int = 0                   # channels that contributed
    n_bg: int = 0                # background bins compared against
    ratio_db: float = float("nan")   # array bin over the background median

    @property
    def p_false_alarm(self) -> float:
        return 1.0 / (1.0 + max(self.n_bg, 1))

    def accept(self) -> bool:
        """Either route is sufficient, and both have an exact H0 rate.

        The array route is the sensitive one and the reason for stacking at
        all; the per-channel count route survives the case where one channel
        is far better than the rest, which the array sum dilutes.
        """
        return bool(self.m >= 3
                    and (self.array_is_max
                         or self.count >= self.threshold))


def cfar_channel_count(chans, fs: float, f: float, a: int, b: int,
                       n_bg: int = 48, guard: int = 2) -> CfarResult:
    """Frequency-domain CFAR at `f`, over the channel array, one snapshot.

    Compare the periodogram at `f` against a guard band of `n_bg` bins either
    side inside the SAME window, skipping `guard` bins next to the tone so
    that its own spectral leakage cannot arm the test.

    TWO STATISTICS, BOTH WITH AN EXACT THRESHOLD AND NO TUNING CONSTANT:

    * the ARRAY periodogram, sum_ch P_ch.  Under H0 every one of the
      (1 + n_bg) bins is identically distributed, so the tested bin is the
      largest with probability exactly 1/(1 + n_bg) -- a rank test, which
      needs no noise-power calibration.  This is the sensitive one, and it
      is where the array gain is actually spent: a tone worth +2.9 dB in a
      single channel's bin, which cannot beat that channel's own background
      maximum, is decisive once fourteen channels are summed.  Testing each
      channel separately and counting throws that gain away -- measured, it
      rejected a true rung at 0/14 channels.

    * the per-channel COUNT, Binomial(m, 1/(1 + n_bg)) under H0.  Kept
      because the array sum dilutes the case where one channel is far
      better than the other thirteen.

    One snapshot is all either needs, which is what keeps them usable on a
    3 ms dwell at the top of the band, where the covariance test below has
    run out of snapshots.
    """
    names = _names(chans)
    n = int(b - a)
    if n < 32 or not names:
        return CfarResult(m=0)
    win = np.hanning(n)
    df = fs / n
    # THE STRADDLE PAIR IS floor AND floor+1, NOT round AND round+1.
    # A tone almost never lands on a bin centre, so its energy is split
    # between the two bins either side of f/df.  Rounding first and then
    # taking (k, k+1) picks the right pair only when the fractional part is
    # below 0.5; above it the pair skips the bin holding most of the energy
    # and lands on a background bin instead.  Measured: a rung at 11 dB of
    # in-bin SNR reported -0.04 dB and was rejected as absent.
    k0 = int(np.floor(f / df))
    if k0 < guard + 2 or k0 + n_bg + guard + 1 >= n // 2:
        return CfarResult(m=0)

    lo = np.arange(k0 - guard - n_bg, k0 - guard)
    hi = np.arange(k0 + guard + 2, k0 + guard + 2 + n_bg)
    bg_idx = np.concatenate([lo[lo > 0], hi[hi < n // 2]])
    if len(bg_idx) < 8:
        return CfarResult(m=0)

    count, m, P_arr = 0, 0, None
    for nm in names:
        y = _window(chans, nm, a, b)
        if y.size != n or not np.all(np.isfinite(y)):
            continue
        P = np.abs(np.fft.rfft((y - y.mean()) * win)) ** 2
        P_arr = P if P_arr is None else P_arr + P
        m += 1
        # the tone may straddle two bins; take the better of the pair
        p_sig = max(P[k0], P[k0 + 1] if k0 + 1 < len(P) else 0.0)
        if p_sig > float(np.max(P[bg_idx])):
            count += 1
    if m == 0 or P_arr is None:
        return CfarResult(m=0)

    sig_arr = max(P_arr[k0], P_arr[k0 + 1] if k0 + 1 < len(P_arr) else 0.0)
    bg_arr = P_arr[bg_idx]
    med = float(np.median(bg_arr))
    return CfarResult(
        array_is_max=bool(sig_arr > float(np.max(bg_arr))),
        count=count,
        threshold=_binom_sf_threshold(m, 1.0 / (1.0 + len(bg_idx))),
        m=m, n_bg=int(len(bg_idx)),
        ratio_db=float(10.0 * np.log10(sig_arr / med)) if med > 0
        else float("nan"))


@dataclass
class Rank1Result:
    """Outcome of the maximum-eigenvalue test at one candidate rung."""
    statistic: float = float("nan")   # lambda_max over the mean eigenvalue
    threshold: float = float("nan")   # Marchenko-Pastur edge for noise only
    m: int = 0                        # channels that contributed
    n_snap: int = 0                   # snapshots the covariance was built on
    participation: float = float("nan")   # how many channels the mode spans

    def usable(self) -> bool:
        """False when the window was too short to give snapshots at all."""
        return bool(np.isfinite(self.statistic)
                    and np.isfinite(self.threshold) and self.m >= 3)

    def accept(self) -> bool:
        """Strong AND shared.  Both halves are load-bearing.

        lambda_max alone answers "is there a strong component", not "is it
        COMMON to the array", and those differ exactly where it matters: a
        ground-loop spur on one front end -- card 4 carries two continuous
        interferers, one holding 12.9 % of the stacked trace's ac power --
        gives a large lambda_max whose eigenvector is concentrated on that
        one channel.  The participation ratio, 1/sum|v_i|^4, counts how many
        channels the mode actually spans: about 1 for such a spur, about m
        for a tone the whole array sees.  Requiring a third of the array
        rejects the spur without any reference to its amplitude.
        """
        return bool(self.usable()
                    and self.statistic >= self.threshold
                    and self.participation >= max(3.0, self.m / 3.0))


def rank1_statistic(chans, fs: float, f: float, a: int, b: int,
                    n_snap: int = 8) -> Rank1Result:
    """Maximum-eigenvalue detection across the channel array.

    A real tone is COMMON to every channel, so the snapshot covariance of the
    per-channel phasors is rank-1 plus noise.  Interference local to one
    front end, and noise, are not.  Split the window into `n_snap`
    sub-windows, fit the phasor at the known `f` in each (three-parameter,
    frequency fixed -- there is nothing to search for here), whiten each
    channel by its own sample standard deviation across snapshots so that no
    noise-power calibration is needed, and take lambda_max of the sample
    covariance over its mean eigenvalue.

    The threshold is the Marchenko-Pastur edge (1 + sqrt(m/L))^2 for
    noise-only data, which is where lambda_max sits when there is no common
    component.  `Rank1Result.accept` also requires the mode to be SHARED --
    see there for why lambda_max on its own is not enough.

    Zeng & Liang, IEEE Trans. Commun. 57 (2009) 1784.
    """
    names = _names(chans)
    n = int(b - a)
    L = int(max(n_snap, 4))
    blk = n // L
    if blk < 16 or blk < int(2 * fs / max(f, 1e-9)) or len(names) < 3:
        return Rank1Result(m=len(names), n_snap=L)

    t = np.arange(blk) / fs
    w = 2 * np.pi * f
    D = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones(blk)])
    # the phasor of each sub-window must be rotated back to a common origin
    rot = np.exp(-1j * w * (np.arange(L) * blk) / fs)

    rows = []
    for nm in names:
        y = _window(chans, nm, a, a + L * blk)
        if y.size != L * blk or not np.all(np.isfinite(y)):
            continue
        Y = y.reshape(L, blk).T
        p, *_ = np.linalg.lstsq(D, Y, rcond=None)
        ph = (p[0] - 1j * p[1]) * rot
        s = float(np.std(ph))
        rows.append(ph / s if s > 0 else ph)
    m = len(rows)
    if m < 3:
        return Rank1Result(m=m, n_snap=L)

    # THE SECOND MOMENT, NOT THE CENTRED COVARIANCE.  The coherent tone is
    # precisely the part of each channel's phasor that does NOT vary from
    # snapshot to snapshot, so subtracting the per-channel mean deletes the
    # signal and leaves the test staring at noise.  Whitening by the spread
    # ACROSS snapshots is still right, and is what makes the statistic free
    # of any noise-power calibration -- that spread is the noise.
    X = np.asarray(rows)                       # m x L
    R = (X @ X.conj().T) / L
    ev, evec = np.linalg.eigh(R)
    ev = ev.real
    mean_ev = float(np.mean(ev))
    if mean_ev <= 0:
        return Rank1Result(m=m, n_snap=L)
    top = np.abs(evec[:, int(np.argmax(ev))]) ** 2
    top = top / max(float(np.sum(top)), 1e-30)
    return Rank1Result(statistic=float(np.max(ev) / mean_ev),
                       threshold=float((1.0 + math.sqrt(m / L)) ** 2),
                       m=m, n_snap=L,
                       participation=float(1.0 / np.sum(top ** 2)))


# ===========================================================================
# 4. The recovered schedule
# ===========================================================================


@dataclass
class HFSchedule:
    """What `recover_schedule` returns.

    `steps` are `eis_local.Step` objects, the same type `detect_schedule`
    produces, so every consumer downstream is unchanged.
    """
    steps: list = field(default_factory=list)
    detected: list = field(default_factory=list)
    predicted: list = field(default_factory=list)
    ladder: Ladder = field(default_factory=Ladder)
    stack: dict = field(default_factory=dict)
    n_rejected: int = 0
    n_off_ladder: int = 0
    pruned_hz: list = field(default_factory=list)
    rejected_hz: list = field(default_factory=list)

    def as_steps(self) -> list:
        return list(self.steps)

    def summary(self) -> dict:
        f = [s.freq for s in self.steps]
        return {
            "n_steps": len(self.steps),
            "n_detected": len(self.detected),
            "n_predicted_verified": len(self.predicted),
            "n_predicted_rejected": self.n_rejected,
            "n_off_ladder_pruned": self.n_off_ladder,
            # the frequencies themselves, so "why does the band stop here"
            # can be answered from the manifest instead of another run
            "off_ladder_hz": [round(v, 4) for v in self.pruned_hz],
            "predicted_verified_hz": sorted(round(s.freq, 4)
                                            for s in self.predicted),
            "predicted_rejected_hz": [round(v, 4) for v in self.rejected_hz],
            "f_min_hz": float(min(f)) if f else None,
            "f_max_hz": float(max(f)) if f else None,
            "ladder_ok": bool(self.ladder.ok),
            "ladder_ppd": float(self.ladder.ppd),
            "ladder_ppd_free": float(self.ladder.ppd_free),
            "ladder_ppd_snapped": bool(self.ladder.snapped),
            "ladder_ratio": float(self.ladder.ratio),
            "ladder_resid": float(self.ladder.resid),
            "n_channels_stacked": int(self.stack.get("n_channels", 0)),
            "array_gain_db": float(self.stack.get("array_gain_db", 0.0)),
        }


def _median_channel_snr_db(chans, fs: float, f: float,
                           a: int, b: int, max_ch: int = 24) -> float:
    """SNR a SINGLE channel sees at this step, not the stack's.

    `Step.snr_db` is read by `consensus_schedule` to decide which steps are
    confident enough to fit the geometric grid on.  Reporting the stacked
    SNR there would hand that guard a number inflated by the array gain --
    every step would look confident and the guard would stop guarding.  So
    report the median of the per-channel SNRs, which is what one segment
    will actually be measured with.
    """
    from eis_local import fit3
    names = _names(chans)
    if not names:
        return float("nan")
    step = max(1, len(names) // max_ch)
    vals = []
    for nm in names[::step]:
        y = _window(chans, nm, a, b)
        if y.size < 8 or not np.all(np.isfinite(y)):
            continue
        _A, _r, snr = fit3(y, fs, f)
        if np.isfinite(snr):
            vals.append(float(snr))
    return float(np.median(vals)) if vals else float("nan")


def recover_schedule(chans, fs: float, f_lo: float | None = None,
                     f_hi: float | None = None, ppd: int = 12,
                     min_snr_db: float = 5.0, sigma_rel_max: float = 0.60,
                     extend: bool = True, snap_ppd: bool = True,
                     ladder_tol: float = 0.02, prune: bool = False,
                     uc_ref: np.ndarray | None = None,
                     verbose: bool = False, log=None) -> HFSchedule:
    """Recover one card's excitation schedule from its segment channels.

    `chans` is anything dict-like from channel name to samples -- a plain
    dict, or `LazyChannels(fam)` to avoid holding the card in memory.

    `uc_ref`, when given, is the card's cell-voltage channel: the trace the
    shipped pipeline used.  IT IS NOT REPLACED, IT IS ADDED TO.  Neither
    trace dominates the other everywhere -- which one carries a given step
    better depends on where |Z_cell| has got to, and on a record where the
    reference amplitude happens to be flat the old path locates the top
    rungs more precisely than the stack does.  So both are detected on, the
    candidates are pooled, and the ladder and the array tests arbitrate
    between them, which is what those tests are for.  Nothing the shipped
    path would have found can be lost this way.
    """
    from eis_local import (Step, detect_schedule, demod_envelope,
                           dwell_window, polish_window, fit3, fit4,
                           harmonic_distortion, _stationarity, _dedupe)

    def _say(msg):
        if log is not None:
            log.info(msg)
        elif verbose:
            print(msg)

    ref, stack = polarity_aligned_reference(chans, log=log)
    if ref.size == 0:
        return HFSchedule(stack=stack)
    n = ref.size
    f_hi = float(f_hi if f_hi else 0.35 * fs)
    f_lo = float(f_lo if f_lo else max(4.0 * fs / n, 0.02))
    _say(f"    stacked {stack['n_channels']} segment channel(s), "
         f"array gain {stack['array_gain_db']:.1f} dB")

    # ---- layer 2: the pipeline's own detector, on a better trace --------
    detected = detect_schedule(ref, fs, ppd=ppd, f_lo=f_lo, f_hi=f_hi,
                               min_snr_db=min_snr_db, verbose=False)
    n_stack = len(detected)
    if uc_ref is not None and len(uc_ref) == n:
        from_uc = detect_schedule(uc_ref, fs, ppd=ppd, f_lo=f_lo, f_hi=f_hi,
                                  min_snr_db=min_snr_db, verbose=False)
        # `_dedupe` is the pipeline's own collapse of candidates that
        # converged on the same step; the longer dwell wins, which is the
        # window least likely to have been truncated by a neighbour.
        detected = _dedupe(sorted(detected + from_uc, key=lambda s: s.freq))
        _say(f"    detector on the stack: {n_stack} step(s), on the cell "
             f"voltage: {len(from_uc)}, pooled to {len(detected)}")
    for s in detected:
        s.snr_db = _median_channel_snr_db(chans, fs, s.freq, s.start, s.stop)
    if uc_ref is None or len(uc_ref) != n:
        _say(f"    detector on the stack: {len(detected)} step(s)"
             + (f" ({detected[0].freq:.3f}..{detected[-1].freq:.1f} Hz)"
                if detected else ""))

    out = HFSchedule(steps=list(detected), detected=list(detected),
                     stack=stack)
    if not extend or len(detected) < 5:
        out.steps.sort(key=lambda s: s.freq)
        return out

    # ---- fit the ladder on the CONFIDENT steps only ---------------------
    # "Confident" here is the pipeline's own off-grid gate, applied to the
    # per-channel SNR.  These steps are accepted without any help from the
    # ladder, so the ladder they define is independent of the weak steps it
    # is about to validate.
    conf = [s for s in detected
            if np.isfinite(s.snr_db) and s.snr_db >= min_snr_db]
    if len(conf) < 5:                      # fall back to the strongest half
        conf = sorted(detected, key=lambda s: -(s.snr_db if
                                                np.isfinite(s.snr_db) else -99)
                      )[:max(5, len(detected) // 2)]
    lad = fit_ladder([s.freq for s in conf], [s.start for s in conf],
                     [s.stop for s in conf], fs=fs, snap=snap_ppd)
    out.ladder = lad
    if not lad.ok:
        _say(f"    ladder: NOT recovered ({lad.note}) - no extension")
        out.steps.sort(key=lambda s: s.freq)
        return out
    _say(f"    ladder: {lad.ppd:.3f} points/decade"
         + (f" (snapped from {lad.ppd_free:.3f})" if lad.snapped else "")
         + f", f0={lad.f0:.4g} Hz, fitted on {lad.n_fitted_on} confident "
           f"step(s), median residual {100 * lad.resid:.3f} %")

    # ---- prune off-ladder detections, only if asked ----------------------
    # OFF BY DEFAULT, AND THAT IS DELIBERATE.
    # Pruning is the one thing in this module that can REMOVE a step the
    # shipped pipeline would have kept, so it is the one thing that can make
    # a run worse.  It did, on real 45 A data: a band that reached 550 Hz
    # came back reaching 375 Hz.  A detection is a measurement; the ladder is
    # a model fitted to a handful of low-frequency steps and extrapolated
    # upward, and where the two disagree at the top of the band the model is
    # usually the one that is wrong -- that is exactly where its extrapolation
    # error is largest.
    #
    # Left off, this module is purely additive: it can only find steps the
    # old path missed, never lose one it found.  Turn it on (config
    # `hf_ladder_prune`) when the record carries a continuous interferer that
    # the detector keeps latching onto -- that is the case ladder membership
    # handles and an SNR gate provably cannot -- and check
    # `pruned_hz` in the summary to see what it took.
    on = prune_off_ladder([s.freq for s in detected], lad, tol=ladder_tol)
    dropped = [s.freq for s, k in zip(detected, on) if not k]
    out.pruned_hz = [float(v) for v in dropped]
    if dropped and prune:
        out.n_off_ladder = len(dropped)
        out.steps = [s for s, k in zip(detected, on) if k]
        _say(f"    ladder pruned {len(dropped)} off-ladder detection(s): "
             + ", ".join(f"{v:.1f} Hz" for v in dropped[:6])
             + (" ..." if len(dropped) > 6 else ""))
    elif dropped:
        _say(f"    {len(dropped)} detection(s) sit off the fitted ladder and "
             f"were KEPT (hf_ladder_prune is off): "
             + ", ".join(f"{v:.1f} Hz" for v in dropped[:6])
             + (" ..." if len(dropped) > 6 else ""))

    # ---- refit the ladder on its own members, once ------------------------
    # The confident steps are the strong ones, which is to say the LOW ones,
    # so the first fit has almost no lever arm: on the synthetic card set it
    # was fitted over 1.6-69 Hz and had to predict a rung at 3 kHz.  Its
    # spacing came out 4.9013 points/decade against a true 4.8891, and that
    # 0.25 % compounds over the intervening rungs into a prediction the
    # acceptance tests then look for in the wrong place.
    #
    # The members accepted above span 1.6-1182 Hz.  Refitting on them pins
    # the spacing to 4.8931 with a SMALLER residual, and it is not circular:
    # every member was admitted by the previous ladder, so an off-ladder
    # candidate cannot enter the fit that judges it.  This is ordinary
    # iterative refinement, and it is taken only if it improves the fit.
    if len(out.steps) > lad.n_fitted_on:
        lad2 = fit_ladder([s.freq for s in out.steps],
                          [s.start for s in out.steps],
                          [s.stop for s in out.steps], fs=fs, snap=snap_ppd)
        if lad2.ok and lad2.resid <= lad.resid:
            _say(f"    ladder refitted on its {lad2.n_fitted_on} members: "
                 f"{lad2.ppd:.3f} points/decade, residual "
                 f"{100 * lad.resid:.3f} % -> {100 * lad2.resid:.3f} %")
            lad = lad2
            out.ladder = lad

    # ---- predict the missing rungs and verify each one -------------------
    have = np.array([s.freq for s in out.steps], float)
    k_lo = int(np.floor(lad.index_of(f_hi)))
    k_hi = int(np.ceil(lad.index_of(f_lo)))
    anchors = sorted(out.steps, key=lambda s: s.freq)

    n_try = 0
    for k in range(min(k_lo, k_hi), max(k_lo, k_hi) + 1):
        f_pred = lad.freq_of(k)
        if not (f_lo <= f_pred <= f_hi):
            continue
        if have.size and (np.min(np.abs(have / f_pred - 1.0))
                          <= lad.tol_at(f_pred, ladder_tol)):
            continue                       # already detected
        n_try += 1
        st = _verify_rung(chans, ref, fs, f_pred, lad, anchors, ladder_tol,
                          sigma_rel_max, Step, demod_envelope, dwell_window,
                          polish_window, fit3, fit4, harmonic_distortion,
                          _stationarity)
        if st is None:
            out.n_rejected += 1
            out.rejected_hz.append(float(f_pred))
            continue
        out.predicted.append(st)
        out.steps.append(st)

    out.steps.sort(key=lambda s: s.freq)
    _say(f"    ladder extension: {len(out.predicted)}/{n_try} predicted "
         f"rung(s) verified on CFAR + rank-1, {out.n_rejected} rejected")
    if out.steps:
        _say(f"    recovered band: {out.steps[0].freq:.3f} .. "
             f"{out.steps[-1].freq:.1f} Hz from {len(out.steps)} step(s)")
    return out


def _verify_rung(chans, ref, fs, f_pred, lad, anchors, ladder_tol,
                 sigma_rel_max, Step, demod_envelope, dwell_window,
                 polish_window, fit3, fit4, harmonic_distortion,
                 _stationarity):
    """Locate one predicted rung in time, then accept or reject it.

    TIMING IS ANCHORED TO THE NEAREST DETECTED STEP, never extrapolated from
    the start of the record.  Over 110 s the accumulated error of the dwell
    law otherwise exceeds a 3 ms dwell by two orders of magnitude, and the
    search bracket lands on the wrong tone.  From the nearest anchor only a
    few rungs' worth of dwell has to be summed, which the law does carry.
    """
    if not anchors:
        return None
    k_t = lad.index_of(f_pred)
    anc = min(anchors, key=lambda s: abs(lad.index_of(s.freq) - k_t))
    k_a = lad.index_of(anc.freq)

    # time from the anchor to the target, summed rung by rung
    lo, hi = sorted((k_a, k_t))
    span = 0.0
    for j in range(int(round(lo)), int(round(hi))):
        span += lad.dwell_s(lad.freq_of(j + 0.5)) + lad.settle_s
    forward = (k_t > k_a) == lad.descending   # later in the record?
    t_anc = anc.start / fs
    t_pred = t_anc + (span if forward else -span)

    dwell = lad.dwell_s(f_pred)
    # bracket generously: the dwell law is fitted, not measured, and the
    # settle time is a median over steps that may not all have had one
    pad = max(6.0 * dwell, 3.0 * lad.settle_s + 0.05 * max(span, 0.0), 0.02)
    a0 = int(max(0, (t_pred - pad) * fs))
    b0 = int(min(len(ref), (t_pred + dwell + pad) * fs))
    if b0 - a0 < max(32, int(2 * fs / f_pred)):
        return None

    # localise inside the bracket with the pipeline's own machinery
    env, win = demod_envelope(ref[a0:b0], fs, f_pred)
    if len(env) < 4:
        return None
    peak, floor = float(np.max(env)), float(np.median(env))
    if not np.isfinite(peak) or peak <= floor:
        return None
    centre = a0 + int(np.argmax(env)) + win // 2
    a, b = dwell_window(ref, fs, f_pred, centre)
    if b - a < 32:
        return None
    a, b = polish_window(ref, fs, f_pred, a, b)
    if b - a < max(32, int(2 * fs / f_pred)):
        return None

    # ---- test 0: is this rung RESOLVABLE from its ladder neighbours? ----
    # Both acceptance tests below read a periodogram of the located window,
    # whose bin width is fs/N.  If the neighbouring rungs sit less than a few
    # bins away, a strong real tone one rung over leaks straight into the bin
    # being tested and BOTH tests pass on it -- the array agrees, because
    # every channel sees the same leakage.  At the top of the band the dwell
    # is milliseconds and the ladder spacing is a fixed FRACTION of f, so
    # this is exactly where it bites.  A rung that cannot be separated from
    # its neighbour is not evidence of anything and must not be accepted;
    # detection, not inference, is the only thing that can carry it.
    n_win = b - a
    df_bin = fs / n_win
    if (lad.ratio - 1.0) * f_pred < 3.0 * df_bin:
        return None

    # the frequency is the LADDER's, not a blind search's; a four-parameter
    # refit is allowed only to polish, and only if it stays on the rung
    f_ref, _A, _r, _snr = fit4(ref[a:b], fs, f_pred)
    f_use = float(f_ref) if (np.isfinite(f_ref)
                             and abs(f_ref / f_pred - 1.0)
                             <= lad.tol_at(f_pred, ladder_tol)
                             ) else float(f_pred)

    # ---- test 1: frequency CFAR across the array, exact threshold -------
    if not cfar_channel_count(chans, fs, f_use, a, b).accept():
        return None

    # ---- test 2: rank-1 / maximum eigenvalue ----------------------------
    # A window too short to give snapshots comes back unusable, and the CFAR
    # test then stands alone -- that single-snapshot case is exactly why the
    # CFAR was chosen alongside this one.
    ev = rank1_statistic(chans, fs, f_use, a, b)
    if ev.usable() and not ev.accept():
        return None

    snr = _median_channel_snr_db(chans, fs, f_use, a, b)
    if not crlb_usable(snr, b - a, sigma_rel_max):
        return None

    _A2, _r2, snr_stack = fit3(ref[a:b], fs, f_use)
    return Step(freq=f_use, start=int(a), stop=int(b),
                amp=float(abs(_A2)), snr_db=float(snr),
                thd=harmonic_distortion(ref[a:b], fs, f_use),
                stationarity=_stationarity(ref, fs, f_use, a, b))
