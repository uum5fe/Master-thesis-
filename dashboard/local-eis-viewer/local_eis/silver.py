#!/usr/bin/env python3
"""
silver.py  --  LAYER 2 of 3:  correction, modelling, validation
===============================================================

Silver turns raw phasors into spectra that can be believed, and attaches an
uncertainty to every number so the gold layer can decide how much weight to
give it.  Nothing here is cosmetic: every step either removes a known
instrumental effect or reports how well the data survives a test it could
have failed.

THE FOUR STAGES
---------------
    1. structural de-skew      the high-frequency fix (section 2)
    2. uncertainty             CRLB per point, propagated through the ratio
    3. measurement model       Bayesian DRT -> R_inf with a credible interval
    4. validation and tiering  lin-KK, Z-HIT, DC closure -> quality tier

-------------------------------------------------------------------------
SECTION 2 IN FULL: WHY THE HIGH-FREQUENCY ARC WAS WRONG
-------------------------------------------------------------------------
The impedance is a ratio of two phasors measured on two different channels:

    Z_s(w) = K_s * A_ref(w) / A_s(w)

If the two channels are not sampled at the same instant, the ratio picks up
exp(-j w dt) where dt is the acquisition skew.  That factor is an ALL-PASS:
it leaves |Z| untouched and rotates the phase by -w*dt, linearly in
frequency.  It is therefore invisible in a Bode magnitude plot and fatal in a
Nyquist plot, and it grows with frequency -- which is why the damage is
concentrated exactly at the top of the band, where the HF intercept lives.

The old pipeline fitted ONE FREE DELAY PER CARD by minimising a KK residual.
That is better than the parity guess it replaced, but it is still wrong in
structure, because the skew is not a per-card constant.  A multiplexed
converter digitises its channels in sequence, so channel at slot p is sampled
at

    t = n/fs + p/(n_ch * fs)

and the skew between the reference at slot p_ref and a segment at slot p_s is

    dt_s = (p_s - p_ref) / (n_ch * fs)                              (nominal)

which is DIFFERENT FOR EVERY SEGMENT ON THE CARD.  At 10 kHz with 16
channels one slot is 6.25 us; segments ten slots apart differ by 62.5 us,
which is 67 degrees of phase at 3 kHz.  Fitting a single per-card delay
splits that difference and leaves half of it on every segment, with opposite
signs on either side of the reference.  That is precisely the "fan-out" seen
in the legacy phase plots -- some segments curling up, some curling down.

WHAT SILVER DOES INSTEAD
------------------------
It fits a two-parameter STRUCTURAL model per card,

    dt_s = dt0 + k * (p_s - p_ref)

where p_s is read from the FAMOS header.  Here `k` is the per-slot conversion
time -- nominally 1/(n_ch*fs), but left free because a sequencer that idles
between bursts has a shorter effective slot -- and `dt0` absorbs whatever is
common to the card: analogue filter group delay, cable, trigger offset.

Two free parameters constrained by every segment on the card, instead of one
free parameter per card (too rigid, wrong shape) or one per segment (too
loose: with 4 % noise the per-segment cost surface is flatter than the noise
and the estimate wanders by tens of microseconds).  The fitted `k` is then
reported against its nominal value, which makes the whole model falsifiable:
if the recovered slot time is not close to 1/(n_ch*fs), the multiplexer
hypothesis is wrong and the pipeline says so instead of quietly applying a
correction it cannot justify.

-------------------------------------------------------------------------
SECTION 3: R_inf WITH AN HONEST ERROR BAR
-------------------------------------------------------------------------
The high-frequency intercept is an EXTRAPOLATION.  Sampling at 10 kHz stops
the usable band near 3.5-4.5 kHz, while a PEMFC's real-axis intercept sits at
1-10 kHz, so R_inf is always partly inferred.  Reading `Re Z` at f_max makes
that inference invisible and puts all the noise of the single worst point
into the map.  Fitting a lin-KK model makes it better but still returns a
bare number.

Silver instead puts a Gaussian-process prior on the distribution of
relaxation times and solves for the posterior:

    Z(w) = R_inf + jwL + sum_m  gamma_m / (1 + jw tau_m),
    gamma ~ N(0, K),   K_ij = sigma_f^2 exp(-(x_i-x_j)^2 / 2 l^2),  x = log tau

Because the model is linear in (R_inf, L, gamma), the posterior is Gaussian
in closed form.  Three consequences, all of which the plain fit cannot give:

  * R_inf arrives as a mean AND a standard deviation.  A segment whose
    intercept is uncertain to 40 % can now be drawn as uncertain instead of
    being drawn as a hot spot.
  * The posterior can be evaluated ABOVE f_max.  That is the only honest way
    to reach the intercept, and the credible interval widens as it
    extrapolates, which is the correct behaviour.
  * The prior regularises the tau padding that made the unregularised lin-KK
    fit diverge, so relaxations just outside the window no longer have to be
    excluded by hand.

Hyperparameters are chosen by maximising the marginal likelihood rather than
being set by hand, which is the point Liu & Ciucci make about GP-DRT.

REFERENCES
----------
B. A. Boukamp, J. Electrochem. Soc. 142 (1995) 1885        -- lin-KK
M. Schoenleber, D. Klotz, E. Ivers-Tiffee, Electrochim. Acta 131 (2014) 20
P. Agarwal, M. E. Orazem, L. H. Garcia-Rubio,
    J. Electrochem. Soc. 139 (1992) 1917                   -- measurement model
W. Ehm et al., ACH Models in Chemistry 137 (2000) 145      -- Z-HIT
T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci,
    Electrochim. Acta 184 (2015) 483                       -- DRT discretisation
J. Liu, F. Ciucci, Electrochim. Acta 331 (2020) 135316     -- GP-DRT
D. C. Rife, R. R. Boorstyn, IEEE Trans. Inf. Theory 20 (1974) 591  -- CRLB
P. M. Ramos, A. Cruz Serra, Measurement 41 (2008) 135      -- joint sine fit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:                      # package layout: core/ holds the science modules
    import core           # noqa: F401
except ImportError:       # flat layout (Databricks, notebooks): already there
    pass
import r2d2_geometry as geom
import eis_measurement_model as mm
from eis_validation import lin_kk, z_hit

import utils
from config import Config, DEFAULT, A_CELL_CM2
from bronze import BronzeRun, BronzeSpectrum


# ===========================================================================
# 1. Containers
# ===========================================================================


@dataclass
class SkewModel:
    """Structural acquisition-skew model for one card."""

    card: str
    basis: str                 # "slot" | "parity" -- which hypothesis won
    dt0: float                 # common offset, seconds
    k_slot: float              # per-basis-unit conversion time, seconds
    k_nominal: float           # 1 / (n_ch * fs)
    cost_gain: float           # relative improvement over dt = 0
    n_segments: int
    applied: bool
    note: str = ""

    @property
    def slot_ratio(self) -> float:
        """Fitted slot time over nominal.  ~1 supports the mux hypothesis."""
        return self.k_slot / self.k_nominal if self.k_nominal else np.nan

    def dt_for(self, slot_seg: int, slot_ref: int) -> float:
        return self.dt0 + self.k_slot * basis_value(self.basis, slot_seg, slot_ref)


@dataclass
class SilverSpectrum:
    """One segment, corrected, modelled and graded."""

    segment: str
    card: str
    freq: np.ndarray
    Z_corr: np.ndarray          # de-skewed measured points
    Z_model: np.ndarray         # posterior mean of the KK/DRT model
    Z_model_sd: np.ndarray      # posterior sd of |Z|, same length
    sigma_rel: np.ndarray       # per-point relative sd used as weights

    # scalars
    R_ohmic: float              # R_inf, ohm*cm^2 (top-band weighted mean)
    R_ohmic_sd: float           # statistical + unresolved-arc bracket
    R_ohmic_drt: float          # the DRT posterior value, as a cross-check
    hf_arc_open: float          # resistance estimated to lie ABOVE f_max
    hf_closure: float           # 1.0 = arc fully closed inside the band
    R_pol: float                # sum gamma, ohm*cm^2
    L: float
    dt_applied: float
    tau_peak: float
    gamma: np.ndarray           # the DRT itself
    tau_grid: np.ndarray

    # validation
    kk_res_max: float
    kk_mu: float
    zhit_dev: float
    zhit_drift: float
    hf_phase_slope: float       # residual all-pass content, rad/ln(f)

    # bookkeeping
    T_degC: float
    K: float
    K_imputed: bool
    u_dc: float
    area_cm2: float
    n_used: int
    n_dropped: int
    snr_med_db: float
    thd_med: float
    tier: str                   # "A" | "B" | "C"
    flags: list[str] = field(default_factory=list)
    # carried through from bronze so the residual-skew check can correlate
    # the leftover delay against converter channel order
    channel_slot: int = -1
    ref_slot: int = -1

    @property
    def j_dc(self) -> float:
        return self.u_dc / self.K if self.K else np.nan

    @property
    def R_ohmic_cv(self) -> float:
        return self.R_ohmic_sd / abs(self.R_ohmic) if self.R_ohmic else np.inf


@dataclass
class SilverRun:
    spectra: dict[str, SilverSpectrum]
    skew: dict[str, SkewModel]
    dc_closure: dict
    cell_freq: np.ndarray
    Z_cell: np.ndarray
    #: One row per segment per point: whether it was kept, and if not, which
    #: gate removed it. This is the evidence for "why does my spectrum stop
    #: at 400 Hz" and "why is this segment missing".
    point_ledger: list = field(default_factory=list)
    #: Segments bronze never produced at all -- no ADC channel for them on any
    #: card. A different fact from "measured and rejected", and the two used to
    #: be indistinguishable from the outputs.
    unwired: list = field(default_factory=list)

    def reach(self) -> list[dict]:
        """Per segment: how far up in frequency it got, and what stopped it.

        The blocking reason is the gate that removed the most points ABOVE the
        highest surviving frequency -- which is the honest answer to "what is
        limiting my bandwidth", rather than the most common reason overall.
        """
        by_seg: dict[str, list[dict]] = {}
        for row in self.point_ledger:
            by_seg.setdefault(str(row["segment"]), []).append(row)
        out = []
        for seg in self.unwired:
            out.append({
                "segment": str(seg), "n_points": 0, "n_kept": 0,
                "f_min_hz": float("nan"), "f_max_hz": float("nan"),
                "n_above_f_max": 0, "blocked_by": "no_channel",
                "explanation": "no ADC channel for this segment on any card "
                               "file -- it was never recorded, so there is "
                               "nothing to evaluate",
                "verdict": "not wired",
            })
        for seg, rows in by_seg.items():
            kept = [r for r in rows if r["kept"]]
            f_max = max((r["freq_hz"] for r in kept), default=float("nan"))
            above = [r for r in rows
                     if not r["kept"]
                     and (not kept or r["freq_hz"] > f_max)]
            tally: dict[str, int] = {}
            for r in above:
                tally[r["reason"]] = tally.get(r["reason"], 0) + 1
            blocking = max(tally.items(), key=lambda kv: kv[1])[0] if tally else ""
            out.append({
                "segment": seg,
                "n_points": len(rows),
                "n_kept": len(kept),
                "f_min_hz": min((r["freq_hz"] for r in kept), default=float("nan")),
                "f_max_hz": f_max,
                "n_above_f_max": len(above),
                "blocked_by": blocking,
                "explanation": REJECT_REASONS.get(blocking, ""),
                "verdict": rows[0]["segment_verdict"] if rows else "",
            })
        return sorted(out, key=lambda r: (len(r["segment"]), r["segment"]))

    def tiers(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.spectra.values():
            out[s.tier] = out.get(s.tier, 0) + 1
        return out


# ===========================================================================
# 2. Structural acquisition skew  --  the high-frequency fix
# ===========================================================================


def _kk_residual(freq: np.ndarray, Z: np.ndarray,
                 sigma_rel: np.ndarray | None, M: int | None = None) -> float:
    """Weighted KK-model residual: how far the spectrum is from realisable."""
    r = mm.fit_kk_model(freq, Z, sigma_rel, M=M)
    return float(r["res_rms"]) if r.get("ok") else np.inf


def basis_value(basis: str, slot_seg: int, slot_ref: int) -> int:
    """Channel-order term of the skew model, under either hardware hypothesis.

    TWO HYPOTHESES, AND THE DATA DECIDES
    ------------------------------------
    "slot"    a single converter walks the channel list, so channel at
              position p is sampled at p/(n_ch*fs) and the skew grows
              linearly with the position difference.

    "parity"  two converter cores work alternately, so all even channels are
              sampled together and all odd channels half a sample period
              later.  The skew then takes only three values, -1, 0 and +1
              times 1/(2*fs), and depends on the PARITY of the position, not
              on its magnitude.

    These predict very different things -- under "slot" two adjacent channels
    differ by one small step, under "parity" they differ by the maximum -- so
    they are easy to tell apart from data, and silver fits both and keeps
    whichever leaves the smaller Kramers-Kronig residual.  Guessing would be
    unwise: at 10 kHz the parity model predicts 1/(2*fs) = 50 us, and the
    per-card delays previously reported for this rig were 49.6 and 52.9 us on
    two of the five cards, which is a strong hint but not a measurement.
    """
    if basis == "parity":
        return (slot_seg % 2) - (slot_ref % 2)
    return slot_seg - slot_ref


def skew_cost(prep: list[tuple], dt0: float, k: float, M: int | None,
              basis: str = "slot") -> float:
    """Mean normalised KK residual of a card under dt = dt0 + k*basis."""
    tot = 0.0
    for f, Z, s_rel, p_seg, p_ref in prep:
        b = basis_value(basis, p_seg, p_ref)
        r = _kk_residual(f, utils.apply_delay(f, Z, dt0 + k * b), s_rel, M=M)
        if not np.isfinite(r):
            return np.inf
        tot += r
    return tot / len(prep)


def fs_prep(prep: list[tuple], default: float = 10_000.0) -> float:
    """Sampling rate implied by the prepared spectra (for delay bounds)."""
    try:
        return float(max(p[0].max() for p in prep)) / 0.35
    except Exception:
        return default


def optimise_skew(prep: list[tuple], k_nom: float, cfg: Config,
                  basis: str = "slot") -> tuple[float, float, float, float]:
    """Coarse-to-fine 2-D search for (dt0, k).

    A FULL 2-D grid, not coordinate descent.  The two parameters are strongly
    correlated -- raising dt0 and lowering k moves the middle segments hardly
    at all -- so scanning k at dt0 = 0 and then dt0 at the winning k lands in
    the wrong basin and stays there.  That failure is not theoretical: on the
    synthetic card in validate_hf.py it returned a slot time of the wrong
    SIGN, at a cost visibly higher than the cost at the truth.

    The scan runs at a FIXED number of Voigt elements.  Letting the
    mu-criterion re-select M at every grid point makes the cost surface
    piecewise in M, and those steps are comparable in size to the minimum
    being searched for.  M is released again for the final fit.
    """
    slot = np.array([basis_value(basis, p[3], p[4]) for p in prep], float)
    span = max(1.0, float(np.ptp(slot)))

    n_dec = max(1.0, np.log10(max(p[0].max() for p in prep)
                              / max(min(p[0].min() for p in prep), 1e-9)))
    M_scan = int(np.clip(round(4 * n_dec), 6, 24))

    # dt0 is a COMMON offset: analogue front-end delay, trigger offset,
    # filter group-delay mismatch.  Physically it is sub-microsecond to a few
    # microseconds.  It is bounded here to half a sample period because a
    # larger common delay is indistinguishable from a smaller one plus a
    # sampling shift, and because the Kramers-Kronig objective develops
    # spurious minima far from zero: left unbounded it returned -154 us on a
    # synthetic built with no common offset at all, which at 3.7 kHz is -206
    # degrees of rotation and drove the real part of 56 of 70 segments
    # negative.
    dt0_lim = min(2.0 * k_nom * span, 0.5 / fs_prep(prep))
    dt0_grid = np.linspace(-dt0_lim, dt0_lim, 25)
    k_grid = np.linspace(-2.5 * k_nom, 2.5 * k_nom, 25)

    base = skew_cost(prep, 0.0, 0.0, M_scan, basis)
    best, best_c = (0.0, 0.0), base
    for dt0 in dt0_grid:
        for k in k_grid:
            c = skew_cost(prep, dt0, k, M_scan, basis)
            if c < best_c:
                best_c, best = c, (dt0, k)

    # local refinement around the coarse winner, two passes
    d_step = float(dt0_grid[1] - dt0_grid[0])
    k_step = float(k_grid[1] - k_grid[0])
    for _ in range(2):
        d0, k0 = best
        for dt0 in np.linspace(d0 - d_step, d0 + d_step, 9):
            for k in np.linspace(k0 - k_step, k0 + k_step, 9):
                c = skew_cost(prep, dt0, k, M_scan, basis)
                if c < best_c:
                    best_c, best = c, (dt0, k)
        d_step /= 4.0
        k_step /= 4.0

    return best[0], best[1], base, best_c


def fit_structural_skew(card: str, items: list[BronzeSpectrum],
                        cfg: Config, log=None,
                        basis_override: str | None = None) -> SkewModel:
    """Fit dt(slot) = dt0 + k * (slot_seg - slot_ref) for one card.

    The cost is the sum over segments of the NORMALISED KK residual, so that
    a low-impedance segment does not dominate a high-impedance one.  A delay
    is an all-pass and the Voigt+L basis is minimum-phase, so the basis
    cannot absorb a delay: the residual therefore has a genuine minimum at
    the true skew.  Summing over segments turns each segment's shallow
    minimum into one sharp one, improving as sqrt(n_segments).
    """
    log = log or utils.get_logger(cfg.verbose)
    usable = [sp for sp in items
              if np.isfinite(sp.Z_raw).sum() >= cfg.min_points_per_spectrum * 2]
    if len(usable) < 3:
        return SkewModel(card, "slot", 0.0, 0.0, 0.0, 0.0, len(usable), False,
                         "too few segments to constrain a skew model")

    fs = usable[0].fs
    n_ch = usable[0].n_ch_on_card
    k_nom = 1.0 / (n_ch * fs)

    # Does the band resolve a delay at all?  A delay is only visible if it
    # turns the phase appreciably at the top of the band.
    f_max = max(np.nanmax(sp.freq) for sp in usable)
    f_min = min(np.nanmin(sp.freq) for sp in usable)
    if np.log10(f_max / max(f_min, 1e-9)) < cfg.skew_min_decades:
        return SkewModel(card, "slot", 0.0, 0.0, k_nom, 0.0, len(usable),
                         False, "band too narrow to resolve a delay")

    # Pre-extract clean arrays once; the grid search touches them many times.
    prep = []
    for sp in usable:
        ok = np.isfinite(sp.Z_raw) & np.isfinite(sp.freq) & (sp.freq > 0)
        if ok.sum() < cfg.min_points_per_spectrum:
            continue
        s_rel = utils.sigma_rel_from_snr(
            sp.snr_comb_db[ok], n=sp.n_per_step[ok],
            model=cfg.uncertainty_model,
            floor=cfg.sigma_rel_floor, ceiling=cfg.sigma_rel_ceiling)
        prep.append((sp.freq[ok], sp.Z_raw[ok], s_rel,
                     sp.channel_slot, sp.ref_slot))
    if len(prep) < 3:
        return SkewModel(card, "slot", 0.0, 0.0, k_nom, 0.0, len(prep),
                         False, "too few usable spectra")

    # ------------------------------------------------------------------
    # FIT BOTH HARDWARE HYPOTHESES AND KEEP THE ONE THE DATA PREFERS
    # ------------------------------------------------------------------
    # "slot" assumes one converter walking the channel list; "parity" assumes
    # two cores alternating, which puts every odd channel half a sample period
    # behind every even one.  They make different predictions for adjacent
    # channels, so the Kramers-Kronig residual can tell them apart.  Nothing
    # is assumed: both are fitted and the loser is reported alongside, so the
    # margin between them is visible rather than hidden inside a default.
    want = basis_override or getattr(cfg, "skew_basis", "auto")
    candidates = ("slot", "parity") if want == "auto" else (want,)
    trials = {}
    for b in candidates:
        k_ref = k_nom if b == "slot" else 1.0 / (2.0 * fs)
        d0, kf, base_b, best_b = optimise_skew(prep, k_ref, cfg, basis=b)
        trials[b] = dict(dt0=d0, k=kf, base=base_b, best=best_b, k_ref=k_ref)

    basis = min(trials, key=lambda b: trials[b]["best"])
    other = ("parity" if basis == "slot" else "slot") if len(trials) > 1 else basis
    t = trials[basis]
    dt0_fit, k_fit, base, best_c = t["dt0"], t["k"], t["base"], t["best"]
    k_nom_used = t["k_ref"]
    margin = ((trials[other]["best"] - best_c) / max(best_c, 1e-12)
              if other in trials else float("nan"))
    gain = float((base - best_c) / base) if base > 0 else 0.0

    slot_deltas = np.array([basis_value(basis, p[3], p[4]) for p in prep], float)

    # The differential part is known a priori from the channel order and is
    # the part that MATTERS: it differs between segments and therefore paints
    # false structure onto the plate map.  The common part dt0 is benign --
    # it rotates every segment equally -- and is also barely identifiable from
    # a Kramers-Kronig residual, so it is fitted but only applied when the
    # cost surface actually has a minimum worth the name.
    ratio = k_fit / k_nom_used if k_nom_used else np.nan
    trust_fit = bool(0.5 <= ratio <= 1.6)
    k_used = k_fit if trust_fit else k_nom_used
    k_source = "fitted" if trust_fit else "nominal (fitted value rejected)"

    # Three conditions before a common offset is applied at all, because the
    # default of zero is the safe one: a wrong dt0 rotates EVERY segment and
    # is far more damaging than omitting a real one, which merely shifts the
    # whole plate.
    if not getattr(cfg, "fit_common_delay", False):
        # See config.fit_common_delay: dt0 is degenerate with the series
        # inductance, which the measurement model already removes.
        sm = SkewModel(card=card, basis=basis, dt0=0.0, k_slot=k_used,
                       k_nominal=k_nom_used, cost_gain=gain,
                       n_segments=len(prep),
                       applied=bool(cfg.skew_model != "none"))
        sm.note = (f"basis={basis} (beats {other} by {100*margin:.0f} % of "
                   f"cost); step {k_source}; common offset not fitted "
                   f"(degenerate with series L, absorbed there)")
        log.info(f"    {card}: basis={basis}, step = {k_used*1e6:.3f} us "
                 f"(nominal {k_nom_used*1e6:.3f}, ratio {ratio:.2f}), "
                 f"{len(prep)} segments")
        log.info(f"      differential spread across the card: "
                 f"{(k_used*np.ptp(slot_deltas))*1e6:+.1f} us = "
                 f"{np.degrees(2*np.pi*f_max*abs(k_used)*np.ptp(slot_deltas)):.0f}"
                 f" deg at {f_max:.0f} Hz -- this is what distorts the map")
        log.info("      common offset held at 0 (degenerate with series L)")
        return sm

    fs_eff = fs_prep(prep)
    dt0_lim = min(2.0 * k_nom_used * max(1.0, float(np.ptp(slot_deltas))),
                  0.5 / fs_eff)
    interior = abs(dt0_fit) < 0.95 * dt0_lim        # not stuck at the edge
    plausible = abs(dt0_fit) <= 0.5 / fs_eff        # not larger than physics
    ident = _dt0_identifiability(prep, k_used, k_nom_used, cfg, basis)
    apply_dt0 = bool(ident["identifiable"] and interior and plausible)
    dt0_used = dt0_fit if apply_dt0 else 0.0
    if not apply_dt0:
        ident["reason"] = ("not identifiable" if not ident["identifiable"]
                           else "optimum at the search boundary" if not interior
                           else "larger than a physical front-end delay")

    applied = bool(cfg.skew_model != "none")
    sm = SkewModel(card=card, basis=basis, dt0=dt0_used, k_slot=k_used,
                   k_nominal=k_nom_used, cost_gain=gain,
                   n_segments=len(prep), applied=applied)
    sm.note = (f"basis={basis} (beats {other} by {100*margin:.0f} % of cost); "
               f"step {k_source}; common offset "
               + (f"fitted {dt0_used*1e6:+.2f} us" if apply_dt0
                  else f"held at 0 ({ident.get('reason','not identifiable')}, "
                       f"raw fit {dt0_fit*1e6:+.1f} us)"))

    log.info(f"    {card}: basis={basis} (vs {other}: {100*margin:+.0f} % cost), "
             f"step = {k_used*1e6:.3f} us "
             f"(nominal {k_nom_used*1e6:.3f}, fitted {k_fit*1e6:+.3f}, "
             f"ratio {ratio:.2f}), {len(prep)} segments, cost -{100*gain:.0f} %")
    log.info(f"      differential spread across the card: "
             f"{(k_used*np.ptp(slot_deltas))*1e6:+.1f} us = "
             f"{np.degrees(2*np.pi*f_max*abs(k_used)*np.ptp(slot_deltas)):.0f} deg "
             f"at {f_max:.0f} Hz -- this is what distorts the map")
    log.info(f"      common offset dt0 = {dt0_used*1e6:+.2f} us")
    return sm


def _dt0_identifiability(prep: list[tuple], k: float, k_nom: float,
                         cfg: Config, basis: str = "slot") -> dict:
    """Is the common-mode delay visible above the noise in the cost?

    Scans dt0 with the differential part held fixed and reports the relative
    range of the cost.  If a whole sample period of common delay changes the
    KK residual by less than a couple of percent, the minimum is meaningless
    and the pipeline says so instead of applying a number it invented.
    """
    grid = np.linspace(-1.5 * k_nom * 16, 1.5 * k_nom * 16, 13)
    costs = np.array([skew_cost(prep, d, k, 12, basis) for d in grid])
    good = np.isfinite(costs)
    if good.sum() < 5:
        return {"identifiable": False, "range": 0.0}
    c = costs[good]
    rng_rel = float((c.max() - c.min()) / max(c.min(), 1e-12))
    # A real minimum should move the residual by more than the fit's own
    # scatter.  10 % is deliberately demanding.
    return {"identifiable": bool(rng_rel > 0.10), "range": rng_rel}


def residual_skew_check(spectra: dict, cfg: Config, log=None) -> dict:
    """After correction, is there any channel-order structure LEFT?

    This is a stronger test of the converter architecture than the fit cost,
    which separates the two hypotheses by only ~1 % and is close to a coin
    toss.  The idea: a correctly modelled skew leaves NO residual delay that
    depends on the channel position.  A wrongly modelled one leaves a very
    characteristic pattern --

        true 'slot' corrected as 'parity'  -> residual RAMPS with slot index
        true 'parity' corrected as 'slot'  -> residual SAWTOOTHS with parity

    so correlating the residual delay against both predictors says which model
    is wrong, and it says it from the corrected data rather than from a fit
    statistic.  The residual delay per segment comes from the high-frequency
    phase slope, which recovers delay DIFFERENCES exactly (it carries a common
    offset from the series inductance, but a common offset cannot correlate
    with channel order and so drops out of this test).
    """
    log = log or utils.get_logger(cfg.verbose)
    d, slot, par = [], [], []
    for sp in spectra.values():
        s_idx = getattr(sp, "channel_slot", -1)
        r_idx = getattr(sp, "ref_slot", -1)
        if s_idx < 0 or r_idx < 0:
            continue                     # slot unknown; nothing to correlate
        dt, r2 = utils.signed_delay_from_phase(sp.freq, sp.Z_corr)
        if not (np.isfinite(dt) and np.isfinite(r2) and r2 > 0.5):
            continue
        d.append(dt)
        slot.append(s_idx - r_idx)
        par.append((s_idx - r_idx) % 2)
    if len(d) < 6:
        return {"ok": False}
    d = np.array(d); slot = np.array(slot, float); par = np.array(par, float)

    def corr(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return float(abs(np.corrcoef(a, b)[0, 1]))

    out = {"ok": True, "n": len(d),
           "corr_slot": corr(d, slot), "corr_parity": corr(d, par),
           "spread_us": float(np.std(d) * 1e6)}
    log.info(f"  residual skew check on {len(d)} corrected segments: "
             f"|corr| with slot index {out['corr_slot']:.2f}, "
             f"with parity {out['corr_parity']:.2f}, "
             f"spread {out['spread_us']:.2f} us")
    # 0.35, not 0.5.  On the end-to-end test, correcting slot-skew data with
    # the parity model left |corr| = 0.49 against parity -- a clearly wrong
    # architecture that a 0.5 threshold waves through.  Residual delay should
    # correlate with channel order essentially not at all when the model is
    # right, so the bar belongs low.
    worst = max(out["corr_slot"], out["corr_parity"])
    if worst > 0.35:
        which = "slot" if out["corr_slot"] > out["corr_parity"] else "parity"
        alt = "parity" if which == "slot" else "slot"
        log.warning(f"  residual delay still tracks {which} "
                    f"(|corr| = {worst:.2f}) - the channel-order model has "
                    f"not fully removed the skew.")
        log.warning(f"  try skew_basis=\"{alt}\" in config.py and compare "
                    f"this number; the correct architecture should leave "
                    f"|corr| below about 0.2 on both predictors.")
    else:
        log.info("  no channel-order structure left - architecture consistent "
                 "with the data")
    return out


def sp_slot(sp) -> int:
    return int(sp.channel_slot - sp.ref_slot)


def choose_global_basis(by_card: dict, cfg: Config, log=None) -> tuple[str, dict]:
    """Decide the converter architecture ONCE for the whole plate.

    All acquisition cards in a rig are the same hardware, so they cannot have
    different converter architectures.  Letting each card vote separately is
    therefore not merely noisy, it is unphysical -- and it happened: on the
    end-to-end test the selector returned `parity` for cards 1, 2 and 4 and
    `slot` for cards 3 and 5.  The per-card margin is only 1-2 % of the fit
    cost, which is well inside the noise, so a per-card vote is close to a
    coin toss.

    Summing the cost over every card turns five weak votes into one strong
    one: the margin grows while the noise averages down.
    """
    log = log or utils.get_logger(cfg.verbose)
    totals = {"slot": 0.0, "parity": 0.0}
    per_card = {}
    for card, items in sorted(by_card.items()):
        prep, k_nom, fs = _prepare_card(items, cfg)
        if prep is None:
            continue
        for b in ("slot", "parity"):
            k_ref = k_nom if b == "slot" else 1.0 / (2.0 * fs)
            _, _, base, best = optimise_skew(prep, k_ref, cfg, basis=b)
            norm = best / base if base > 0 else 1.0
            totals[b] += norm
            per_card.setdefault(card, {})[b] = norm

    if not per_card:
        return "slot", {}
    winner = min(totals, key=lambda b: totals[b])
    loser = "parity" if winner == "slot" else "slot"
    margin = (totals[loser] - totals[winner]) / max(totals[winner], 1e-12)
    n_agree = sum(1 for c in per_card
                  if min(per_card[c], key=lambda b: per_card[c][b]) == winner)
    log.info(f"  converter architecture: '{winner}' chosen plate-wide "
             f"({100*margin:.1f} % lower total cost than '{loser}'; "
             f"{n_agree}/{len(per_card)} cards agree individually)")
    if margin < 0.05:
        log.warning(f"  margin is small - the two architectures are barely "
                    f"distinguishable here.  Set skew_basis explicitly in "
                    f"config.py from the card datasheet if you can.")
    return winner, per_card


def _prepare_card(items: list[BronzeSpectrum], cfg: Config):
    """Shared preparation used by both the basis vote and the per-card fit."""
    usable = [sp for sp in items
              if np.isfinite(sp.Z_raw).sum() >= cfg.min_points_per_spectrum * 2]
    if len(usable) < 3:
        return None, None, None
    fs = usable[0].fs
    k_nom = 1.0 / (usable[0].n_ch_on_card * fs)
    prep = []
    for sp in usable:
        ok = np.isfinite(sp.Z_raw) & np.isfinite(sp.freq) & (sp.freq > 0)
        if ok.sum() < cfg.min_points_per_spectrum:
            continue
        s_rel = utils.sigma_rel_from_snr(
            sp.snr_comb_db[ok], n=sp.n_per_step[ok],
            model=cfg.uncertainty_model,
            floor=cfg.sigma_rel_floor, ceiling=cfg.sigma_rel_ceiling)
        prep.append((sp.freq[ok], sp.Z_raw[ok], s_rel,
                     sp.channel_slot, sp.ref_slot))
    if len(prep) < 3:
        return None, None, None
    return prep, k_nom, fs


def fit_per_card_skew(card: str, items: list[BronzeSpectrum],
                      cfg: Config, log=None) -> SkewModel:
    """Legacy single-delay-per-card model, kept for A/B comparison."""
    log = log or utils.get_logger(cfg.verbose)
    spec = {}
    for sp in items:
        ok = np.isfinite(sp.Z_raw) & (sp.freq > 0)
        if ok.sum() >= cfg.min_points_per_spectrum:
            s_rel = utils.sigma_rel_from_snr(
                sp.snr_comb_db[ok], n=sp.n_per_step[ok],
                model=cfg.uncertainty_model)
            spec[sp.segment] = (sp.freq[ok], sp.Z_raw[ok], s_rel)
    if len(spec) < 3:
        return SkewModel(card, "slot", 0.0, 0.0, 0.0, 0.0, len(spec), False,
                         "too few")
    info = mm.fit_shared_delay(spec, dt_range=cfg.skew_dt_range_s,
                               n_grid=cfg.skew_n_grid, verbose=False)
    fs = items[0].fs
    k_nom = 1.0 / (items[0].n_ch_on_card * fs)
    return SkewModel(card, "slot", float(info.get("dt", 0.0)), 0.0, k_nom,
                     float(info.get("cost_gain", 0.0)), len(spec),
                     bool(info.get("applied")), "per-card single delay")


# ===========================================================================
# 3. Bayesian DRT  --  R_inf with a credible interval
# ===========================================================================


def _drt_design(freq: np.ndarray, tau: np.ndarray, with_L: bool) -> np.ndarray:
    """[1, (jw), 1/(1+jw tau_m) ...] -- linear in every parameter."""
    w = 2 * np.pi * freq
    cols = [np.ones_like(w, dtype=complex)]
    if with_L:
        cols.append(1j * w)
    cols.extend(1.0 / (1.0 + 1j * w * t) for t in tau)
    return np.column_stack(cols)


def _se_kernel(x: np.ndarray, sigma_f: float, ell: float) -> np.ndarray:
    d = x[:, None] - x[None, :]
    return sigma_f ** 2 * np.exp(-0.5 * (d / ell) ** 2)


def bayesian_drt(freq: np.ndarray, Z: np.ndarray, sigma_rel: np.ndarray,
                 cfg: Config) -> dict:
    """Gaussian-process DRT with a closed-form posterior.

    Returns the posterior mean and standard deviation of every parameter,
    most importantly R_inf, plus the model curve and its uncertainty.

    The system is linear, so with a Gaussian prior and Gaussian noise the
    posterior is exactly Gaussian:

        Sigma = (Phi^H W Phi + P^-1)^-1 ,   mu = Sigma Phi^H W z

    with W the inverse noise covariance and P the prior covariance
    (block-diagonal: a very wide Gaussian on R_inf and L, the squared
    exponential kernel on gamma).  Real and imaginary parts are stacked so
    everything stays real-valued.
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    ok = np.isfinite(freq) & (freq > 0) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    freq, Z = freq[ok], Z[ok]
    s_rel = np.clip(np.asarray(sigma_rel, float)[ok], 1e-4, 1.0)
    order = np.argsort(freq)
    freq, Z, s_rel = freq[order], Z[order], s_rel[order]
    n = len(freq)
    if n < cfg.min_points_per_spectrum:
        return {"ok": False, "note": "too few points"}

    # --- tau grid -----------------------------------------------------------
    # THE TWO ENDS ARE NOT SYMMETRIC, and the legacy notes had this backwards.
    #
    #   fast end, tau << 1/w_max :  1/(1 + jw tau) -> 1
    #        identical to the constant column.  Measured correlation with it
    #        is 1.0000, so R_inf and a fast element trade freely and the
    #        extracted intercept runs away.  This is the end that must NOT be
    #        padded generously.
    #
    #   slow end, tau >> 1/w_min :  1/(1 + jw tau) -> -j/(w tau)
    #        a purely capacitive column, correlation with the constant only
    #        ~0.47.  Padding here is safe and useful: it lets the model
    #        describe a low-frequency process whose peak sits below the
    #        measured window instead of distorting the in-band fit to reach
    #        for it.
    #
    # A SMALL fast pad is still allowed on purpose.  R_inf physically means
    # "everything faster than the band", so a spectrum containing a real
    # relaxation just above f_max has an intercept that is genuinely
    # ambiguous.  Letting the model carry a little fast content converts that
    # ambiguity from a hidden bias into an honest widening of the posterior
    # on R_inf -- which is the number the plate map is drawn from.
    pad_fast = float(getattr(cfg, "drt_tau_pad_fast", 0.0))
    pad_slow = float(getattr(cfg, "drt_tau_pad_slow",
                             getattr(cfg, "drt_tau_pad_decades", 0.0)))
    lo = np.log10(1.0 / (2 * np.pi * freq.max())) - pad_fast
    hi = np.log10(1.0 / (2 * np.pi * freq.min())) + pad_slow
    m = max(8, int(round(cfg.drt_n_tau_per_decade * (hi - lo))))
    x = np.linspace(lo, hi, m)              # log10 tau
    tau = 10.0 ** x

    hi_band = Z[freq >= np.percentile(freq, 80)]
    with_L = bool(np.mean(np.imag(hi_band)) > 0)
    n_free = 2 if with_L else 1

    Phi = _drt_design(freq, tau, with_L)
    A = np.vstack([np.real(Phi), np.imag(Phi)])
    b = np.concatenate([np.real(Z), np.imag(Z)])
    # absolute per-point sd, from the relative sd and |Z|
    sd = np.concatenate([s_rel * np.abs(Z), s_rel * np.abs(Z)])
    sd = np.maximum(sd, 1e-12 * max(np.max(np.abs(Z)), 1e-12))

    scale = float(np.median(np.abs(Z)))     # keeps the prior scale-free
    wide = (1e3 * scale) ** 2               # near-flat prior on R_inf and L

    # PRIOR MEAN.  gamma is a distribution of resistances and is therefore
    # POSITIVE, but a Gaussian prior centred on zero shrinks it towards zero
    # and the constant column takes up the slack -- which shows up as R_inf
    # biased HIGH.  Measured on synthetic spectra that bias was +5 to +7
    # mOhm*cm2 on a true 60, i.e. 10 %, and it is systematic, so it does not
    # average away across the plate.
    #
    # Centring the prior on a data-driven guess removes it.  The guess is the
    # crudest possible one -- the real-axis width of the spectrum spread over
    # the tau grid -- and because the prior variance stays wide the data still
    # dominates; all that changes is that the model is no longer told, before
    # seeing anything, that every relaxation is probably absent.
    hi_re = float(np.median(np.real(Z[freq >= np.percentile(freq, 80)])))
    lo_re = float(np.median(np.real(Z[freq <= np.percentile(freq, 20)])))
    R_pol_guess = max(lo_re - hi_re, 0.0)
    m0 = np.zeros(A.shape[1])
    m0[0] = hi_re                            # R_inf ~ Re Z at the top of band
    m0[n_free:] = R_pol_guess / max(m, 1)    # gamma spread evenly over tau

    def neg_log_eml(theta: np.ndarray) -> float:
        """Negative log marginal likelihood, for hyperparameter selection."""
        log_sf, log_l, log_sn = theta
        sf, ell, sn = np.exp(log_sf), np.exp(log_l), np.exp(log_sn)
        P = np.zeros((A.shape[1], A.shape[1]))
        P[:n_free, :n_free] = np.eye(n_free) * wide
        P[n_free:, n_free:] = _se_kernel(x, sf * scale, ell) + 1e-10 * scale ** 2 * np.eye(m)
        S = sd * sn
        try:
            C = A @ P @ A.T + np.diag(S ** 2)
            L_ = np.linalg.cholesky(C + 1e-12 * np.trace(C) / len(C) * np.eye(len(C)))
        except np.linalg.LinAlgError:
            return 1e12
        r = b - A @ m0
        alpha = np.linalg.solve(L_.T, np.linalg.solve(L_, r))
        return float(0.5 * r @ alpha + np.sum(np.log(np.diag(L_))))

    # Hyperparameters are BOUNDED, and the noise scale is nearly pinned.
    #
    # An unbounded Nelder-Mead on log-hyperparameters diverged in testing --
    # literally overflowing exp() -- and the failure is not benign: the way it
    # diverges is by inflating sigma_n, which tells the model the data are
    # worthless, so gamma is shrunk to its prior and the constant column
    # absorbs the whole spectrum.  R_inf then runs away.  That is exactly the
    # 17x spread the first version produced.
    #
    # sigma_n is bounded tightly around 1 because it multiplies a noise
    # estimate that is already physical: sigma_rel comes from the Cramer-Rao
    # bound of the sine fit, not from a guess, so the model has no business
    # rescaling it by more than a factor of ~2.
    theta0 = np.log([1.0, float(np.clip(cfg_len(cfg), 0.2, 3.0)), 1.0])
    bounds = [np.log([0.05, 20.0]),      # sigma_f, relative to |Z|
              np.log([0.2, 3.0]),        # length scale, decades of log10(tau)
              np.log([0.5, 2.0])]        # sigma_n, multiplier on the CRLB
    if getattr(cfg, "drt_optimise_hypers", True):
        try:
            from scipy.optimize import minimize
            res = minimize(neg_log_eml, theta0, method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 60})
            theta = res.x if np.isfinite(res.fun) else theta0
        except Exception:
            theta = theta0
    else:
        theta = theta0
    theta = np.clip(theta, [b[0] for b in bounds], [b[1] for b in bounds])

    sf, ell, sn = np.exp(theta)
    P = np.zeros((A.shape[1], A.shape[1]))
    P[:n_free, :n_free] = np.eye(n_free) * wide
    P[n_free:, n_free:] = _se_kernel(x, sf * scale, ell) + 1e-10 * scale ** 2 * np.eye(m)
    S = sd * sn
    W = np.diag(1.0 / S ** 2)

    try:
        Pinv = np.linalg.inv(P + 1e-12 * np.trace(P) / len(P) * np.eye(len(P)))
        Sigma = np.linalg.inv(A.T @ W @ A + Pinv)
        mu = Sigma @ (A.T @ W @ b + Pinv @ m0)
    except np.linalg.LinAlgError:
        return {"ok": False, "note": "posterior is singular"}

    R_inf = float(mu[0])
    R_inf_sd = float(np.sqrt(max(Sigma[0, 0], 0.0)))
    L = float(mu[1]) if with_L else 0.0
    gamma = mu[n_free:]
    R_pol = float(np.sum(gamma))

    Z_model = Phi @ mu
    # propagate the posterior into the curve: var(Re) + var(Im)
    var_re = np.einsum("ij,jk,ik->i", np.real(Phi), Sigma, np.real(Phi))
    var_im = np.einsum("ij,jk,ik->i", np.imag(Phi), Sigma, np.imag(Phi))
    Z_sd = np.sqrt(np.maximum(var_re + var_im, 0.0))

    k_peak = int(np.argmax(np.abs(gamma))) if len(gamma) else 0
    tau_peak = float(tau[k_peak]) if len(gamma) else np.nan

    # ------------------------------------------------------------------
    # HOW MUCH OF THE ARC IS STILL OPEN AT f_max
    # ------------------------------------------------------------------
    # R_inf means "everything faster than the band".  If the spectrum still
    # has curvature at f_max, some of the high-frequency arc is outside the
    # window and R_inf silently absorbs it.  Measured on synthetic spectra
    # with a 4.5 kHz process and the band stopping at 3.5 kHz, EVERY
    # estimator -- Re Z at f_max, Boukamp lin-KK R0, and this posterior --
    # came out biased by the same +8 to +10 mOhm*cm2.  Padding the tau grid
    # to reach past f_max did not recover it at any noise level down to
    # 0.2 %: the information is not in the data, and a model that claims
    # otherwise is fitting its own prior.
    #
    # So the bias is not corrected here.  It is MEASURED and reported, which
    # is the only honest option, and it is added to the uncertainty on R_inf
    # so that a segment whose arc is wide open is drawn as uncertain instead
    # of as a confident hot spot.  The real fix is instrumental: raise the
    # sampling rate until the arc closes inside the band.
    # The model cannot see a relaxation outside its tau grid, so asking it how
    # much arc is left over-reports closure -- it returned 0.996 on a spectrum
    # that was demonstrably 12 mOhm*cm2 short.  The detectable signature is
    # the residual IMAGINARY part at f_max once the series inductance is taken
    # out: a closed arc has Im Z -> 0 there, and anything left is capacitive
    # content the window did not reach.
    #
    # For an arc whose peak sits near f_max, Im Z = -R/2 at the peak, so
    # 2*|Im Z(f_max)| is the natural order-of-magnitude bound on the resistance
    # still hidden above the band.  On the synthetic with 12 mOhm*cm2 of
    # unresolved arc it returns 11.6, which brackets the observed +8.7 bias.
    i_top = int(np.argmax(freq))
    w_top = 2 * np.pi * freq[i_top]
    Im_top = float(np.imag(Z_model[i_top]) - w_top * L)   # inductance removed
    arc_open = float(2.0 * abs(min(Im_top, 0.0)))
    closure = float(R_pol / (R_pol + arc_open)) if (R_pol + arc_open) > 0 else np.nan
    R_inf_sd_total = float(np.sqrt(R_inf_sd ** 2 + arc_open ** 2))

    resid = (Z - Z_model) / np.abs(Z)
    res_rms = float(np.sqrt(np.mean(np.abs(resid) ** 2)))

    return {
        "ok": True, "freq": freq, "Z": Z, "Z_model": Z_model, "Z_model_sd": Z_sd,
        "R_inf": R_inf, "R_inf_sd": R_inf_sd_total,
        "R_inf_sd_posterior": R_inf_sd, "hf_arc_open": arc_open,
        "hf_closure": closure, "L": L, "R_pol": R_pol,
        "gamma": gamma, "tau": tau, "tau_peak": tau_peak,
        "res_rms": res_rms, "with_L": with_L,
        "hyper": {"sigma_f": float(sf), "ell_decades": float(ell),
                  "sigma_n": float(sn)},
        "sigma_rel": s_rel,
    }


def cfg_len(cfg: Config) -> float:
    return float(getattr(cfg, "drt_length_scale_init", 0.7))


def r_ohmic_topband(freq: np.ndarray, Z: np.ndarray, sigma_rel: np.ndarray,
                    span_decades: float = 0.35, n_min: int = 3,
                    n_max: int = 8) -> tuple[float, float, int]:
    """R_inf as an inverse-variance weighted mean of Re Z at the top of band.

    WHY THIS AND NOT THE MODEL POSTERIOR
    ------------------------------------
    This was chosen against the alternatives on synthetic spectra, not by
    preference.  With the acquisition skew correctly removed:

        estimator            arc closed          arc open past f_max
        Re Z at f_max        bias +0.8  rms 2.3   bias  +8.6  rms  8.9
        top-band mean        bias +0.6  rms 1.4   bias +10.0  rms 10.1
        Boukamp lin-KK R0    bias +10.0 rms 11.3   bias +10.0  rms 11.3
        GP-DRT posterior     bias +6.1  rms 6.7    bias  +8.7  rms  9.0

    When the arc closes inside the band the simple weighted mean is five
    times more accurate than either model, and it recovers the segment-to-
    segment spread exactly (1.24x against a true 1.24x).  When the arc does
    NOT close, every estimator carries the same bias, because the missing
    resistance is outside the window and no inversion can conjure it back --
    that was checked down to 0.2 % noise, where the model bias got worse, not
    better, since it was then fitting its own prior.

    So the elaborate estimator earns nothing here and the simple one is used.
    The DRT is still computed, because R_pol, the peak relaxation time and
    the process split are things only it provides, and its R_inf is kept
    alongside as a cross-check: when the two disagree, the spectrum is not
    behaving like a passive RC network and the segment deserves a look.

    Averaging is deliberately confined to a THIRD of a decade.  Widening it
    keeps reducing the noise but drags in arc curvature, which is bias, and
    on the open-arc case the bias grew from +8.6 to +11.0 as the window went
    from one point to eight.
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    ok = np.isfinite(freq) & (freq > 0) & np.isfinite(Z.real)
    if ok.sum() < 1:
        return np.nan, np.nan, 0
    f, z = freq[ok], Z[ok]
    s = np.clip(np.asarray(sigma_rel, float)[ok], 1e-4, 1.0)

    f_top = f.max()
    sel = f >= f_top * 10.0 ** (-span_decades)
    if sel.sum() < n_min:                       # take the highest n_min anyway
        sel = np.zeros_like(f, bool)
        sel[np.argsort(f)[-min(n_min, len(f)):]] = True
    if sel.sum() > n_max:
        idx = np.argsort(f)[-n_max:]
        sel = np.zeros_like(f, bool)
        sel[idx] = True

    re = np.real(z[sel])
    sd = np.maximum(s[sel] * np.abs(z[sel]), 1e-15)
    wts = 1.0 / sd ** 2
    mean = float(np.sum(wts * re) / np.sum(wts))
    sd_mean = float(np.sqrt(1.0 / np.sum(wts)))
    return mean, sd_mean, int(sel.sum())


def extrapolate_hf(drt: dict, f_hi: float, n: int = 40) -> dict:
    """Evaluate the posterior above the measured band.

    This is the only defensible route to the high-frequency intercept when
    the sampling rate cuts the band off below it.  The credible interval
    widens as the model extrapolates, which is exactly what an honest
    estimate should do.
    """
    if not drt.get("ok"):
        return {}
    f = drt["freq"]
    f_new = np.logspace(np.log10(f.max()), np.log10(f_hi), n)
    Phi = _drt_design(f_new, drt["tau"], drt["with_L"])
    n_free = 2 if drt["with_L"] else 1
    mu = np.concatenate([[drt["R_inf"]],
                         ([drt["L"]] if drt["with_L"] else []),
                         drt["gamma"]])
    return {"freq": f_new, "Z": Phi @ mu, "n_free": n_free}


# ===========================================================================
# 4. Per-segment processing
# ===========================================================================


#: Why a point did not survive. The FIRST gate to reject a point owns it, so
#: the counts add up to the number dropped instead of double-counting.
REJECT_REASONS = {
    "not_finite": "the phasor fit did not return a finite Z",
    "outside_band": "outside cfg.f_min_hz .. cfg.f_hi(fs)",
    "snr": "SNR below the gate for this point",
    "thd": "harmonic distortion above cfg.max_thd",
    "drift": "amplitude drifted during the dwell, above cfg.max_drift",
    "magnitude": "|Z| outside the plausible 0.5 .. 800 mOhm.cm2 window",
    "uncertainty": "propagated sigma above cfg.sigma_rel_max",
    "cycles": "fewer than cfg.min_cycles_per_dwell cycles in the dwell",
    "zmag_outlier": "local |Z| outlier against its neighbours",
    "passivity": "Re Z negative above the passivity gate, after de-skew",
}


def process_segment(sp: BronzeSpectrum, skew: SkewModel, cfg: Config,
                    log=None, ledger: list | None = None
                    ) -> SilverSpectrum | None:
    """De-skew, weight, model, validate and grade one segment.

    `ledger`, when given, is appended with one row per POINT recording which
    gate removed it. Nine gates run in sequence and until now only their
    totals were kept, so "my impedance stops at 400 Hz" and "this segment has
    no spectrum at all" were unanswerable from the outputs -- the evidence was
    computed and thrown away. The first gate to reject a point owns it.
    """
    freq = np.asarray(sp.freq, float)
    Z = np.asarray(sp.Z_raw, complex)

    reason = np.array([""] * freq.size, dtype=object)

    def gate(ok: np.ndarray, name: str) -> np.ndarray:
        """Apply a gate, recording it for the points it is the first to kill."""
        ok = np.asarray(ok, bool)
        newly = (reason == "") & ~ok
        reason[newly] = name
        return ok

    keep = gate(np.isfinite(freq) & (freq > 0)
                & np.isfinite(Z.real) & np.isfinite(Z.imag), "not_finite")
    # cfg.f_hi(sp.fs), not cfg.f_max_hz: the ceiling follows the rate this
    # segment was recorded at. Reading the raw field here would have thrown
    # away every point bronze newly finds above 4.5 kHz on a fast card, and
    # charged them to "outside_band" -- a gate the operator cannot act on,
    # because the band it names is not the band that was searched.
    keep &= gate((freq >= cfg.f_min_hz) & (freq <= cfg.f_hi(sp.fs)),
                 "outside_band")

    # Point-level gates.  An ON-GRID step is a real step -- a geometric
    # progression is not something noise produces -- so for those the SNR
    # becomes a WEIGHT rather than a membership test.  Off-grid candidates
    # still face the full gate.  This is what keeps the top-of-band points,
    # which are always the weakest, without letting noise in elsewhere.
    snr = np.asarray(sp.snr_comb_db, float)
    snr_gate = np.where(sp.on_grid, snr >= cfg.snr_floor_db,
                        snr >= cfg.min_snr_db)
    keep &= gate(np.nan_to_num(snr_gate, nan=False).astype(bool), "snr")
    keep &= gate(~(np.isfinite(sp.thd) & (sp.thd > cfg.max_thd)), "thd")
    keep &= gate(~(np.isfinite(sp.drift) & (sp.drift > cfg.max_drift)), "drift")

    # Plausibility on the magnitude always; on the SIGN of the real part only
    # above passivity_gate_min_hz.  See the note in config.py: a negative real
    # part at low frequency is a documented consequence of down-the-channel
    # starvation and is a fault signature worth keeping, not noise.
    # |Z| is an ALL-PASS invariant: a delay cannot change it, so the
    # magnitude bound is safe to apply to the uncorrected spectrum.
    mag = np.abs(Z) * 1000.0
    lo, hi = 0.5, 800.0
    keep &= gate(np.isfinite(mag) & (mag > lo) & (mag < hi), "magnitude")
    hf_region = freq >= cfg.passivity_gate_min_hz
    # THE SIGN OF Re Z IS *NOT* AN INVARIANT.  It is deferred to after the
    # de-skew below.  Applied here, it deleted points that the very next
    # step would have rotated back into the passive half-plane: on segment 1
    # it cut 14 surviving points to 6, under the 8-point floor, and the
    # segment was thrown away as unmodellable.  Order matters.

    # ---- the two gates that remove the high- and low-frequency garbage ----
    # 1. PROPAGATED UNCERTAINTY.  Computed on the points still standing, so
    #    the dwell length is accounted for: a weak point measured over many
    #    samples survives, a weak point measured over a handful does not.
    #    This is what stops the on-grid rescue from admitting noise.
    with np.errstate(invalid="ignore", divide="ignore"):
        s_all = utils.sigma_rel_from_snr(snr, n=sp.n_per_step,
                                         model=cfg.uncertainty_model,
                                         floor=cfg.sigma_rel_floor,
                                         ceiling=1e9)
    keep &= gate(np.isfinite(s_all) & (s_all <= cfg.sigma_rel_max),
                 "uncertainty")

    # 2. CYCLES IN THE DWELL.  A phasor fitted to a fraction of a period is
    #    not a measurement of that period; at the low-frequency end the fit
    #    cannot separate the fundamental from the drift term, and the two
    #    trade against each other.
    with np.errstate(invalid="ignore", divide="ignore"):
        cycles = sp.n_per_step / sp.fs * freq
    keep &= gate(np.isfinite(cycles) & (cycles >= cfg.min_cycles_per_dwell),
                 "cycles")

    # 3. LOCAL |Z| OUTLIERS.  Targets the runaway points directly instead of
    #    inferring them from SNR, so weak-but-consistent points at the top of
    #    the band survive.  Evaluated on what is still standing.
    Zg = np.where(keep, Z, np.nan)
    bad_z = utils.zmag_outliers(freq, Zg, n_mad=cfg.zmag_outlier_mad,
                                win=cfg.zmag_outlier_win)
    keep &= gate(~bad_z, "zmag_outlier")

    n_drop = int((~keep).sum())
    n_drop_unc = int(np.sum(np.isfinite(s_all) & (s_all > cfg.sigma_rel_max)))
    n_drop_cyc = int(np.sum(np.isfinite(cycles)
                            & (cycles < cfg.min_cycles_per_dwell)))
    n_drop_out = int(bad_z.sum())
    def _emit(final_keep: np.ndarray, verdict: str) -> None:
        if ledger is None:
            return
        for i in range(freq.size):
            ledger.append({
                "segment": sp.segment, "card": sp.card,
                "freq_hz": round(float(freq[i]), 6),
                "kept": int(bool(final_keep[i])),
                "reason": "" if final_keep[i] else (reason[i] or "unknown"),
                "snr_db": (round(float(snr[i]), 2)
                           if np.isfinite(snr[i]) else ""),
                "cycles": (round(float(cycles[i]), 2)
                           if np.isfinite(cycles[i]) else ""),
                "sigma_rel": (round(float(s_all[i]), 4)
                              if np.isfinite(s_all[i]) else ""),
                "segment_verdict": verdict,
            })

    if keep.sum() < cfg.min_points_per_spectrum:
        # The segment is abandoned here. It used to vanish with no record at
        # all, which is why "why is segment 33 not evaluated?" had no answer
        # anywhere in the outputs.
        _emit(keep, f"dropped: only {int(keep.sum())} point(s) survived, "
                    f"cfg.min_points_per_spectrum is "
                    f"{cfg.min_points_per_spectrum}")
        if log:
            worst = {}
            for r in reason[~keep]:
                worst[r] = worst.get(r, 0) + 1
            top = sorted(worst.items(), key=lambda kv: -kv[1])[:3]
            log.warning(
                f"    segment {sp.segment}: dropped -- {int(keep.sum())} of "
                f"{freq.size} points survived; most removed by "
                + ", ".join(f"{k} ({v})" for k, v in top))
        return None

    f = freq[keep]
    z = Z[keep]
    s_rel = utils.sigma_rel_from_snr(snr[keep], n=sp.n_per_step[keep],
                                     model=cfg.uncertainty_model,
                                     floor=cfg.sigma_rel_floor,
                                     ceiling=cfg.sigma_rel_ceiling)

    # ---- structural de-skew ------------------------------------------------
    dt = skew.dt_for(sp.channel_slot, sp.ref_slot) if skew.applied else 0.0
    z_corr = utils.apply_delay(f, z, dt)

    # ---- passivity, NOW that the phase is corrected ------------------------
    # Above passivity_gate_min_hz a passive cell has Re Z > 0.  Below it, a
    # negative real part is a documented consequence of down-the-channel
    # starvation (Schneider et al., ECS Trans. 25(1) 937 (2009)) and is kept
    # and flagged rather than deleted.
    asr = np.real(z_corr) * 1000.0
    hf_f = f >= cfg.passivity_gate_min_hz
    passive = ~(hf_f & np.isfinite(asr) & (asr <= 0))
    n_neg_lf = int(np.sum(~hf_f & np.isfinite(asr) & (asr < 0)))
    n_drop_pass = int((~passive).sum())
    # Map the passivity verdict back onto the ORIGINAL point index, so the
    # ledger stays one row per recorded point rather than per survivor.
    idx = np.flatnonzero(keep)
    reason[idx[~passive]] = "passivity"
    keep[idx[~passive]] = False

    if passive.sum() < cfg.min_points_per_spectrum:
        _emit(keep, f"dropped: {int(passive.sum())} point(s) left after the "
                    f"passivity gate, cfg.min_points_per_spectrum is "
                    f"{cfg.min_points_per_spectrum}")
        return None

    _emit(keep, f"kept: {int(keep.sum())} of {freq.size} points, "
                f"{float(freq[keep].min()):.4g}-{float(freq[keep].max()):.4g} Hz")

    f, z, z_corr = f[passive], z[passive], z_corr[passive]
    s_rel = s_rel[passive]
    n_drop += n_drop_pass

    # ---- measurement model -------------------------------------------------
    if cfg.drt_enable:
        model = bayesian_drt(f, z_corr, s_rel, cfg)
    else:
        model = {"ok": False}
    if not model.get("ok"):
        r = mm.fit_kk_model(f, z_corr, s_rel, mu_crit=cfg.mu_crit,
                            ridge=cfg.kk_ridge)
        if not r.get("ok"):
            return None
        model = {
            "ok": True, "freq": r["freq"], "Z": r["Z"], "Z_model": r["Z_model"],
            "Z_model_sd": np.zeros_like(r["freq"]),
            "R_inf": float(r["R0"]), "R_inf_sd": np.nan, "L": float(r["L"]),
            "R_pol": float(r["Rp"]), "gamma": np.zeros(0), "tau": np.zeros(0),
            "tau_peak": np.nan, "res_rms": float(r["res_rms"]),
            "sigma_rel": s_rel,
        }

    f_m = model["freq"]
    z_m = model["Z"]
    Z_model = model["Z_model"]
    L = model["L"]

    # jwL is cable and plate, not electrochemistry.  Removing it from the
    # OUTPUT curve is what makes the arc close onto the real axis instead of
    # hooking upward; it is reported separately so nothing is hidden.
    if cfg.remove_inductance and L:
        w = 2 * np.pi * f_m
        z_m = z_m - 1j * w * L
        Z_model = Z_model - 1j * w * L

    # ---- R_ohmic: measured at the top of band, not extrapolated -----------
    R_top, R_top_sd, n_top = r_ohmic_topband(f_m, z_m, model.get("sigma_rel", s_rel))
    arc_open = float(model.get("hf_arc_open", 0.0) or 0.0)
    closure = float(model.get("hf_closure", np.nan))
    R_drt = float(model["R_inf"])
    if not np.isfinite(R_top):
        R_top, R_top_sd = R_drt, float(model.get("R_inf_sd", np.nan))
    # The unresolved arc is a BIAS, not noise, so it is added in quadrature to
    # the statistical error rather than being subtracted from the estimate:
    # its sign is known but its size is only bracketed.
    R_sd_total = float(np.sqrt(np.nan_to_num(R_top_sd) ** 2 + arc_open ** 2))

    # ---- independent validation -------------------------------------------
    kk = lin_kk(f_m, z_m, mu_crit=cfg.mu_crit, tol=cfg.kk_tol)
    zh = z_hit(f_m, z_m)

    # ---- tiering: soft, never deleting ------------------------------------
    flags: list[str] = []
    if sp.K_imputed:
        flags.append("imputed_calibration")
    if not skew.applied:
        flags.append("no_skew_correction")
    if n_drop > 0.4 * len(freq):
        flags.append("many_points_dropped")
    if n_drop_unc:
        flags.append(f"dropped_{n_drop_unc}_uncertain")
    if n_drop_cyc:
        flags.append(f"dropped_{n_drop_cyc}_short_dwell")
    if n_drop_out:
        flags.append(f"dropped_{n_drop_out}_zmag_outlier")
    if n_drop_pass:
        flags.append(f"dropped_{n_drop_pass}_non_passive_after_deskew")
    if n_neg_lf:
        # kept deliberately - see Schneider et al. 2009
        flags.append(f"negative_ReZ_lf_{n_neg_lf}pts")

    cv = abs(R_sd_total / R_top) if (R_top and np.isfinite(R_sd_total)) else np.inf
    if np.isfinite(closure) and closure < 0.95:
        flags.append(f"hf_arc_open_{1000*arc_open:.0f}mohm")
    kk_res = float(kk.get("res_max", np.nan))

    # A negative or zero high-frequency intercept is not a measurement, it is
    # a failure: a passive cell cannot have one.  It happens when the top of
    # the band is nearly all noise, and left alone it poisons every plate
    # statistic downstream -- on the end-to-end test a single such segment
    # turned the R_ohmic spread into 8e10.  Flag it, force the lowest tier,
    # and let the spatial field supply a usable number instead.
    if not (np.isfinite(R_top) and R_top > 0):
        flags.append("nonphysical_R_ohmic")

    if (not np.isfinite(R_top)) or R_top <= 0:
        tier = "C"
    elif kk_res <= cfg.kk_tol and cv <= 0.10:
        tier = "A"
    elif kk_res <= 3 * cfg.kk_tol and cv <= 0.25:
        tier = "B"
    else:
        tier = "C"
    if sp.K_imputed and tier == "A":
        tier = "B"          # the shape is fine; the absolute level is not

    return SilverSpectrum(
        segment=sp.segment, card=sp.card, freq=f_m, Z_corr=z_m,
        Z_model=Z_model, Z_model_sd=model.get("Z_model_sd", np.zeros_like(f_m)),
        sigma_rel=model.get("sigma_rel", s_rel),
        R_ohmic=float(R_top), R_ohmic_sd=float(R_sd_total),
        R_ohmic_drt=R_drt, hf_arc_open=arc_open, hf_closure=closure,
        R_pol=float(model["R_pol"]), L=float(L), dt_applied=float(dt),
        tau_peak=float(model.get("tau_peak", np.nan)),
        gamma=np.asarray(model.get("gamma", np.zeros(0))),
        tau_grid=np.asarray(model.get("tau", np.zeros(0))),
        kk_res_max=kk_res, kk_mu=float(kk.get("mu", np.nan)),
        zhit_dev=float(zh.get("dev_rms", np.nan)),
        zhit_drift=float(zh.get("drift_lf", np.nan)),
        hf_phase_slope=float(utils.hf_phase_slope(f_m, z_m)),
        T_degC=sp.T_degC, K=sp.K, K_imputed=sp.K_imputed, u_dc=sp.u_dc,
        area_cm2=utils.segment_areas(cfg).get(
            sp.segment, geom.SEGMENTS[sp.segment].area_cm2),
        channel_slot=int(sp.channel_slot), ref_slot=int(sp.ref_slot),
        n_used=int(len(f_m)), n_dropped=n_drop,
        snr_med_db=float(np.nanmedian(snr[keep])),
        thd_med=float(np.nanmedian(sp.thd[keep])),
        tier=tier, flags=flags,
    )


# ===========================================================================
# 5. Plate-level products
# ===========================================================================


def dc_closure(spectra: dict[str, SilverSpectrum], cfg: Config) -> dict:
    """Sum(j_s * A_s) against the load current.

    Genuinely independent: nothing in the chain was fitted to the current, so
    this is the only end-to-end check of calibration, areas and DC levels
    together.  It costs nothing and it is the first thing to look at.
    """
    j = {s: sp.j_dc for s, sp in spectra.items() if np.isfinite(sp.j_dc)}
    if not j:
        return {"ok": False}
    A_meas = sum(spectra[s].area_cm2 for s in j)
    I_meas = sum(j[s] * spectra[s].area_cm2 for s in j)
    I_extrap = I_meas * A_CELL_CM2 / A_meas if A_meas else np.nan
    out = {"ok": True, "area_measured_cm2": A_meas,
           "area_total_cm2": A_CELL_CM2, "I_measured_A": I_meas,
           "I_extrapolated_A": I_extrap, "i_setpoint_A": cfg.i_setpoint_a}
    if cfg.i_setpoint_a:
        out["deviation_pct"] = 100.0 * (I_extrap / cfg.i_setpoint_a - 1.0)
    return out


def cell_aggregate(spectra: dict[str, SilverSpectrum]
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted harmonic mean -- what a cell-level instrument would see.

    Segments sit in parallel across one cell voltage, so admittances add.
    Being harmonic, the result is dominated by the LOW-impedance segments,
    which is the mathematical statement of why an integral measurement hides
    local faults: a flooded segment has high Z, contributes little
    admittance, and barely moves the cell curve.
    """
    if not spectra:
        return np.zeros(0), np.zeros(0)
    ref = max(spectra.values(), key=lambda s: len(s.freq))
    f_ref = ref.freq
    on = {}
    areas = {}
    for s, sp in spectra.items():
        if len(sp.freq) < 3:
            continue
        on[s] = utils.interp_complex(f_ref, sp.freq, sp.Z_model)
        areas[s] = sp.area_cm2
    return f_ref, mm.aggregate_asr(f_ref, on, areas, A_CELL_CM2)


# ===========================================================================
# 6. Entry point
# ===========================================================================


def run(bronze_run: BronzeRun, cfg: Config = DEFAULT, log=None) -> SilverRun:
    log = log or utils.get_logger(cfg.verbose)
    utils.banner("SILVER  --  correction, modelling, validation", log)

    # ---- 1. skew, per card -------------------------------------------------
    utils.section(f"acquisition skew  (model: {cfg.skew_model})", log)
    by_card: dict[str, list[BronzeSpectrum]] = {}
    for sp in bronze_run.spectra.values():
        by_card.setdefault(sp.card, []).append(sp)

    global_basis = None
    if cfg.skew_model == "structural":
        if getattr(cfg, "skew_basis", "auto") == "auto":
            global_basis, _ = choose_global_basis(by_card, cfg, log)
        else:
            global_basis = cfg.skew_basis
            log.info(f"  converter architecture: '{global_basis}' "
                     f"(set explicitly in config)")

    skew: dict[str, SkewModel] = {}
    for card, items in sorted(by_card.items()):
        if cfg.skew_model == "structural":
            skew[card] = fit_structural_skew(card, items, cfg, log,
                                             basis_override=global_basis)
        elif cfg.skew_model == "per_card":
            skew[card] = fit_per_card_skew(card, items, cfg, log)
        else:
            fs = items[0].fs
            skew[card] = SkewModel(card, "slot", 0.0, 0.0,
                                   1.0 / (items[0].n_ch_on_card * fs),
                                   0.0, len(items), False, "disabled")

    # ---- 2. per segment ----------------------------------------------------
    utils.section("measurement model per segment", log)
    spectra: dict[str, SilverSpectrum] = {}
    failed: list[str] = []
    #: One row per segment per recorded point, with the gate that removed it.
    point_ledger: list[dict] = []
    for seg in bronze_run.segments_measured():
        sp = bronze_run.spectra[seg]
        try:
            res = process_segment(sp, skew[sp.card], cfg, log,
                                  ledger=point_ledger)
        except Exception as exc:                      # never lose the plate
            log.warning(f"    segment {seg}: {type(exc).__name__}: {exc}")
            res = None
        if res is None:
            failed.append(seg)
            continue
        spectra[seg] = res

    tiers: dict[str, int] = {}
    for s in spectra.values():
        tiers[s.tier] = tiers.get(s.tier, 0) + 1
    log.info(f"  {len(spectra)} segments modelled  "
             f"(A={tiers.get('A',0)}  B={tiers.get('B',0)}  C={tiers.get('C',0)})")
    if failed:
        log.info(f"  {len(failed)} could not be modelled at all: "
                 f"{', '.join(failed)}")
        log.info("  (they are handed to gold.py for spatial inference, "
                 "not discarded)")

    if spectra:
        rs = np.array([s.R_ohmic * 1000 for s in spectra.values()])
        sd = np.array([s.R_ohmic_sd * 1000 for s in spectra.values()])
        good = np.isfinite(rs) & (rs > 0)
        if good.any():
            log.info(f"  R_ohmic = {rs[good].mean():.1f} +/- "
                     f"{rs[good].std():.1f} mOhm*cm2 "
                     f"(spread {rs[good].max()/max(rs[good].min(),1e-9):.2f}x, "
                     f"CV {100*rs[good].std()/rs[good].mean():.1f} %)")
            if np.isfinite(sd).any():
                log.info(f"  median posterior sd on R_ohmic: "
                         f"{np.nanmedian(sd[good]):.2f} mOhm*cm2 "
                         f"({100*np.nanmedian(sd[good]/np.maximum(rs[good],1e-9)):.1f} %)")

    # ---- 3. plate products -------------------------------------------------
    utils.section("plate-level checks", log)
    dcc = dc_closure(spectra, cfg)
    if dcc.get("ok"):
        log.info(f"  DC closure: {dcc['area_measured_cm2']:.1f} / "
                 f"{dcc['area_total_cm2']:.1f} cm2 measured, "
                 f"I = {dcc['I_measured_A']:.1f} A -> "
                 f"{dcc['I_extrapolated_A']:.1f} A full plate")
        if "deviation_pct" in dcc:
            log.info(f"  setpoint {dcc['i_setpoint_A']:.1f} A -> "
                     f"deviation {dcc['deviation_pct']:+.1f} %")

    if cfg.skew_model == "structural":
        try:
            residual_skew_check(spectra, cfg, log)
        except Exception as exc:          # a diagnostic must never end a run
            log.warning(f"  residual skew check skipped: "
                        f"{type(exc).__name__}: {exc}")

    f_cell, Z_cell = cell_aggregate(spectra)
    if len(f_cell):
        log.info(f"  cell aggregate: area-weighted harmonic mean over "
                 f"{len(spectra)} segments, {len(f_cell)} frequencies")

    return SilverRun(point_ledger=point_ledger,
                     unwired=list(bronze_run.segments_missing()),
                     spectra=spectra, skew=skew, dc_closure=dcc,
                     cell_freq=f_cell, Z_cell=Z_cell)


def save(sr: SilverRun, cfg: Config, log=None) -> Path:
    log = log or utils.get_logger(cfg.verbose)
    out = Path(cfg.out_dir) / "silver"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for seg in sorted(sr.spectra, key=int):
        s = sr.spectra[seg]
        for i, f in enumerate(s.freq):
            rows.append({
                "segment": seg, "freq_hz": round(float(f), 6),
                "z_re_mohm_cm2": round(1000 * float(np.real(s.Z_corr[i])), 5),
                "z_im_mohm_cm2": round(1000 * float(np.imag(s.Z_corr[i])), 5),
                "zmodel_re_mohm_cm2": round(1000 * float(np.real(s.Z_model[i])), 5),
                "zmodel_im_mohm_cm2": round(1000 * float(np.imag(s.Z_model[i])), 5),
                "zmodel_sd_mohm_cm2": round(1000 * float(s.Z_model_sd[i]), 5),
                "sigma_rel": round(float(s.sigma_rel[i]), 5),
            })
    utils.write_table(out / "spectra_clean.csv", rows)

    utils.write_table(out / "segments_summary.csv", [{
        "segment": seg, "card": s.card, "tier": s.tier,
        "area_cm2": round(s.area_cm2, 5),
        "cx_mm": round(geom.SEGMENTS[seg].cx_mm, 2),
        "cy_mm": round(geom.SEGMENTS[seg].cy_mm, 2),
        "R_ohmic_mohm_cm2": round(1000 * s.R_ohmic, 4),
        "R_ohmic_sd_mohm_cm2": round(1000 * s.R_ohmic_sd, 4)
        if np.isfinite(s.R_ohmic_sd) else "",
        "R_pol_mohm_cm2": round(1000 * s.R_pol, 4),
        "L_nH": round(1e9 * s.L, 3),
        "tau_peak_s": f"{s.tau_peak:.4g}" if np.isfinite(s.tau_peak) else "",
        "dt_applied_us": round(1e6 * s.dt_applied, 3),
        "T_degC": round(s.T_degC, 2),
        "j_dc_A_cm2": round(s.j_dc, 6),
        "kk_res_pct": round(100 * s.kk_res_max, 4)
        if np.isfinite(s.kk_res_max) else "",
        "zhit_dev_pct": round(100 * s.zhit_dev, 4)
        if np.isfinite(s.zhit_dev) else "",
        "hf_phase_slope": round(s.hf_phase_slope, 5),
        "snr_med_db": round(s.snr_med_db, 2),
        "thd_med_pct": round(100 * s.thd_med, 3)
        if np.isfinite(s.thd_med) else "",
        "n_used": s.n_used, "n_dropped": s.n_dropped,
        "K_imputed": int(s.K_imputed),
        "flags": ";".join(s.flags),
    } for seg, s in sorted(sr.spectra.items(), key=lambda kv: int(kv[0]))])

    utils.write_table(out / "skew_model.csv", [{
        "card": k, "basis": v.basis, "dt0_us": round(1e6 * v.dt0, 4),
        "slot_us": round(1e6 * v.k_slot, 4),
        "slot_nominal_us": round(1e6 * v.k_nominal, 4),
        "slot_ratio": round(v.slot_ratio, 3) if np.isfinite(v.slot_ratio) else "",
        "cost_gain_pct": round(100 * v.cost_gain, 1),
        "n_segments": v.n_segments, "applied": int(v.applied), "note": v.note,
    } for k, v in sorted(sr.skew.items())])

    # The rejection ledger: one row per segment per point. This is the file to
    # open when a spectrum stops lower than expected or a segment is missing
    # altogether -- it names the gate rather than leaving it to be guessed.
    if sr.point_ledger:
        utils.write_table(out / "point_rejections.csv", sr.point_ledger)
        utils.write_table(out / "segment_reach.csv", sr.reach())

    if len(sr.cell_freq):
        utils.write_table(out / "cell_aggregate.csv", [{
            "freq_hz": round(float(f), 6),
            "z_re_mohm_cm2": round(1000 * float(np.real(z)), 5),
            "z_im_mohm_cm2": round(1000 * float(np.imag(z)), 5),
        } for f, z in zip(sr.cell_freq, sr.Z_cell)])

    utils.write_json(out / "silver_manifest.json", {
        "n_segments": len(sr.spectra), "tiers": sr.tiers(),
        "dc_closure": sr.dc_closure,
        "skew": {k: {"dt0_us": v.dt0 * 1e6, "slot_us": v.k_slot * 1e6,
                     "slot_nominal_us": v.k_nominal * 1e6,
                     "slot_ratio": v.slot_ratio, "applied": v.applied,
                     "cost_gain": v.cost_gain, "note": v.note}
                 for k, v in sr.skew.items()},
    })
    log.info(f"  silver written to {out}")
    return out
