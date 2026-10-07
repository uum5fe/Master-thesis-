#!/usr/bin/env python3
"""
card_gain.py
============
Does every card measure the same cell voltage with the same gain?  And does
the current scale (K) close against the bench, at every operating point?

WHY
---
Z = K * A_UC / A_seg.  With the reference pool off (the default), every card
divides by ITS OWN UC phasor, so a card whose voltage chain reads 4 % high
puts +4 % on every segment it carries -- R_ohmic, R_ct, |Z| alike -- and the
map shows a step at the card boundary that is not in the cell.  On 2612030
the R_ohmic card medians at 60 A were 66.5 / 66.0 / 70.3 / 70.2 / 64.8.

All five cards record the same cell voltage on UC2 (it is what the card
alignment cross-correlates), so their UC2 phasors at each step MUST agree.
The ratio of each card's UC2 phasor to the median of the cards is therefore a
measurement of that card's voltage-chain gain, made on every run, with no
extra hardware.

THREE MEASUREMENTS, ONE QUESTION
--------------------------------
1. `measure_reference` (bronze, needs the .DAT files): per card and UC
   channel, |A_card / A_median| over the steps -- the voltage-chain gain --
   its flatness (low band vs high band), the residual timing (phase slope),
   and the DC level.
2. `impedance_factors` (silver, works on any finished or re-evaluated run):
   per card, the median of |Z_seg / Z_plate| in four frequency bands, and the
   card's DC current density against the plate.  A GAIN error scales every
   frequency alike and moves j_dc the other way by the same amount; a real
   regional difference does neither.
3. `dc_closure_table` (across conditions): the sum of the segment DC currents
   against the bench current at 45 / 60 / 150 / 450 A, fitted as
   I_meas = a * I_bench + b.  `a` is the scale error that also sits in every
   impedance; `b` is a zero offset, which does not touch the impedance.

cfg.card_gain = "off" | "report" | "correct"   (default "report")
"correct" divides each card's Z by its measured UC gain, only when the gain
is flat over the band and within `card_gain_max_pct`; otherwise it is
reported and left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

MODES = ("off", "report", "correct")

OK = "ok"
OFF = "off_by_gain"
NOT_FLAT = "not_flat"
TOO_LARGE = "too_large"
TOO_FEW = "too_few_steps"
TWO_CARDS = "only_two_cards"


def _opt(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def mode(cfg) -> str:
    m = str(_opt(cfg, "card_gain", "report") or "off").lower()
    return m if m in MODES else "report"


def short(card: str) -> str:
    m = re.search(r"Karte_?\d+", str(card))
    return m.group(0) if m else str(card)


# ---------------------------------------------------------------------------
# 1. the UC channels, card against card
# ---------------------------------------------------------------------------

@dataclass
class CardReference:
    card: str
    channel: str
    n_steps: int = 0
    gain: float = float("nan")         # |A_card / A_median|, median over steps
    gain_lo: float = float("nan")      # same, f < 10 Hz
    gain_hi: float = float("nan")      # same, f > 100 Hz
    phase_deg: float = float("nan")    # median arg(A_card / A_median)
    dt_us: float = float("nan")        # phase slope as a time offset
    dc_V: float = float("nan")         # median DC level over the windows
    dc_ratio: float = float("nan")     # dc_V / median of the cards
    used_for_Z: bool = False           # this is the channel Z divides by
    applied: float = 1.0               # what Z was divided by
    status: str = ""
    #: "ac": gain from the step phasors; "dc": from the DC level, used when
    #: too few steps carry an AC phasor clean enough (on 2612030 the UC2 AC
    #: response sits at 0-20 dB per step, while the DC level is ~0.8 V and
    #: is read to 1e-4 over the whole record)
    source: str = ""

    @property
    def gain_pct(self) -> float:
        return 100.0 * (self.gain - 1.0)

    @property
    def flat_pct(self) -> float:
        return 100.0 * (self.gain_hi - self.gain_lo)

    def row(self) -> dict:
        def r(x, k=4):
            return round(float(x), k) if np.isfinite(x) else ""
        return {"card": self.card, "channel": self.channel,
                "used_for_Z": int(self.used_for_Z), "n_steps": self.n_steps,
                "gain": r(self.gain, 5), "gain_pct": r(self.gain_pct, 2),
                "gain_lo": r(self.gain_lo, 5), "gain_hi": r(self.gain_hi, 5),
                "flat_pct": r(self.flat_pct, 2),
                "phase_deg": r(self.phase_deg, 3), "dt_us": r(self.dt_us, 2),
                "dc_V": r(self.dc_V, 5), "dc_ratio": r(self.dc_ratio, 5),
                "applied": r(self.applied, 5), "status": self.status,
                "source": self.source}


def _band(freq, vals, lo, hi):
    m = (freq >= lo) & (freq < hi) & np.isfinite(vals)
    return float(np.median(vals[m])) if m.sum() >= 2 else float("nan")


def compare_phasors(freq: np.ndarray, A: dict[str, np.ndarray],
                    snr: dict[str, np.ndarray], dc: dict[str, np.ndarray],
                    cfg=None, channel: str = "UC2",
                    used_for_Z: dict[str, bool] | None = None
                    ) -> dict[str, CardReference]:
    """Card against card for ONE channel name.

    `A[card]` are the step phasors on a common time base and mux slot,
    `snr[card]` their SNR, `dc[card]` the DC level per step.  The reference
    per step is the median magnitude and the mean direction of the cards.
    """
    freq = np.asarray(freq, float)
    cards = sorted(A)
    snr_min = float(_opt(cfg, "card_gain_min_snr_db", 20.0))
    n_min = int(_opt(cfg, "card_gain_min_steps", 5))
    tol = float(_opt(cfg, "card_gain_tol_pct", 1.0))
    flat_max = float(_opt(cfg, "card_gain_flat_pct", 2.0))
    big = float(_opt(cfg, "card_gain_max_pct", 10.0))

    M = np.vstack([np.asarray(A[c], complex) for c in cards])
    S = np.vstack([np.asarray(snr[c], float) for c in cards])
    good = np.isfinite(M) & np.isfinite(S) & (S >= snr_min) & (np.abs(M) > 0)
    Mg = np.where(good, M, np.nan)
    import warnings
    with np.errstate(invalid="ignore", divide="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mag_ref = np.nanmedian(np.abs(Mg), axis=0)
        unit = np.where(good, M / np.abs(np.where(good, M, 1)), 0)
        ph_ref = np.angle(unit.sum(axis=0))
        ratio = Mg / (mag_ref * np.exp(1j * ph_ref))
    Dc = np.vstack([np.asarray(dc[c], float) for c in cards])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dc_card = np.nanmedian(Dc, axis=1)
    dc_ref = float(np.nanmedian(dc_card)) if np.isfinite(dc_card).any() \
        else float("nan")

    out = {}
    w = 2 * np.pi * freq
    for k, c in enumerate(cards):
        r = ratio[k]
        m = np.isfinite(r)
        ref = CardReference(card=c, channel=channel, n_steps=int(m.sum()),
                            used_for_Z=bool((used_for_Z or {}).get(c, False)))
        ref.dc_V = float(dc_card[k])
        ref.dc_ratio = (ref.dc_V / dc_ref if np.isfinite(dc_ref) and dc_ref
                        else float("nan"))
        if len(cards) < 3:
            ref.status = TWO_CARDS
        if m.sum() < n_min:
            if not ref.status and np.isfinite(ref.dc_ratio):
                # the DC level: same cell voltage on every card, read over
                # the whole record -- a gain to 1e-4, no flatness test
                ref.gain, ref.source = ref.dc_ratio, "dc"
                ref.status = (TOO_LARGE if abs(ref.gain_pct) > big else
                              OFF if abs(ref.gain_pct) > tol else OK)
            else:
                ref.status = ref.status or TOO_FEW
            out[c] = ref
            continue
        ref.source = "ac"
        g = np.abs(r)
        ph = np.angle(r)
        ref.gain = float(np.median(g[m]))
        ref.gain_lo = _band(freq, np.where(m, g, np.nan), 0.0, 10.0)
        ref.gain_hi = _band(freq, np.where(m, g, np.nan), 100.0, np.inf)
        ref.phase_deg = float(np.degrees(np.median(ph[m])))
        # phase = -w dt for a card that samples late
        ref.dt_us = float(-1e6 * np.sum(w[m] * ph[m]) / np.sum(w[m] ** 2))
        if ref.status:
            out[c] = ref
            continue
        if abs(ref.gain_pct) > big:
            ref.status = TOO_LARGE
        elif np.isfinite(ref.flat_pct) and abs(ref.flat_pct) > flat_max:
            ref.status = NOT_FLAT
        elif abs(ref.gain_pct) > tol:
            ref.status = OFF
        else:
            ref.status = OK
        out[c] = ref
    return out


def correctable(ref: CardReference) -> bool:
    return ref.status in (OK, OFF) and np.isfinite(ref.gain) and ref.gain > 0


def measure_reference(files, cards: dict, schedule, cfg, lags=None,
                      log=None) -> dict[str, dict[str, CardReference]]:
    """{channel: {card: CardReference}} for every UC channel the cards share.

    Reads only the UC channels, on the dwell windows, with each card's
    alignment applied, rotated from its mux slot to slot 0.  A card whose
    alignment was refused is still compared in magnitude (a time offset does
    not change |A| on a stationary step) and its phase is not used.
    """
    import utils
    from eis_local import FamosFile

    if mode(cfg) == "off" or not schedule:
        return {}
    freq = np.array([s.freq for s in schedule], float)
    n = len(schedule)
    per: dict[str, dict[str, tuple]] = {}
    for fp in files:
        stem = fp.stem
        if stem not in cards:
            continue
        info = (lags or {}).get(stem, {})
        aligned = bool(info.get("applied", True)) if info else True
        shift = int(info.get("lag", 0)) if info.get("applied") else 0
        try:
            fam = FamosFile(fp)
        except Exception as exc:                            # noqa: BLE001
            if log is not None:
                log.warning(f"  card gain: {fp.name} unreadable ({exc})")
            continue
        for name in fam.uc_names:
            x = fam.channel(name)
            slot_dt = fam.position(name) / (fam.n_ch * fam.fs)
            A = np.full(n, np.nan, complex)
            S = np.full(n, np.nan)
            D = np.full(n, np.nan)
            for i, st in enumerate(schedule):
                a, b = st.start + shift, st.stop + shift
                if b <= a or a < 0 or b > len(x):
                    continue
                y = x[a:b]
                Ai, _r, si = utils.fit3(y, fam.fs, st.freq)
                if not np.isfinite(Ai):
                    continue
                Ai = Ai * np.exp(2j * np.pi * st.freq * slot_dt)
                if not aligned:
                    Ai = complex(abs(Ai), 0.0)
                A[i], S[i], D[i] = Ai, si, float(np.mean(y))
            per.setdefault(name, {})[stem] = (A, S, D, aligned)
            del x

    out: dict[str, dict[str, CardReference]] = {}
    for name, d in per.items():
        if len(d) < 2:
            continue
        used = {c: (cards[c].ref_name == name) for c in d if c in cards}
        A = {c: v[0] for c, v in d.items()}
        # unaligned cards: compare magnitudes only, against magnitudes
        if not all(v[3] for v in d.values()):
            A = {c: np.abs(a).astype(complex) for c, a in A.items()}
        out[name] = compare_phasors(
            freq, A, {c: v[1] for c, v in d.items()},
            {c: v[2] for c, v in d.items()}, cfg, channel=name,
            used_for_Z=used)

    m = mode(cfg)
    for name, refs in out.items():
        for r in refs.values():
            if r.used_for_Z and m == "correct" and correctable(r):
                r.applied = r.gain
    if log is not None:
        _log(out, m, log)
    return out


def load_reference(run_dir) -> dict[str, dict[str, CardReference]]:
    """Read bronze/card_reference.csv back ({} when the run has none)."""
    import csv
    from pathlib import Path
    d = Path(run_dir)
    p = d / "bronze" / "card_reference.csv"
    if not p.is_file():
        p = d / "card_reference.csv"
    if not p.is_file():
        return {}

    def num(x, default=float("nan")):
        try:
            return float(x)
        except (TypeError, ValueError):
            return default

    out: dict[str, dict[str, CardReference]] = {}
    with p.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cr = CardReference(
                card=r["card"], channel=r["channel"],
                n_steps=int(num(r.get("n_steps"), 0)), gain=num(r.get("gain")),
                gain_lo=num(r.get("gain_lo")), gain_hi=num(r.get("gain_hi")),
                phase_deg=num(r.get("phase_deg")), dt_us=num(r.get("dt_us")),
                dc_V=num(r.get("dc_V")), dc_ratio=num(r.get("dc_ratio")),
                used_for_Z=str(r.get("used_for_Z", "0")) == "1",
                applied=num(r.get("applied"), 1.0), status=r.get("status", ""),
                source=r.get("source", "") or "")
            out.setdefault(cr.channel, {})[cr.card] = cr
    return out


def applied_gains(refs: dict[str, dict[str, CardReference]]) -> dict[str, float]:
    """card -> what its Z is divided by (1.0 where nothing is applied)."""
    g = {}
    for d in refs.values():
        for c, r in d.items():
            if r.used_for_Z:
                g[c] = float(r.applied)
    return g


def _log(out, m, log) -> None:
    for name, refs in sorted(out.items()):
        parts = []
        for c, r in sorted(refs.items()):
            if np.isfinite(r.gain):
                parts.append(f"{short(c)} {r.gain_pct:+.2f} %"
                             + ("" if r.status == OK else f" ({r.status})"))
            else:
                parts.append(f"{short(c)} -- ({r.status})")
        tag = " [Z divides by this]" if any(r.used_for_Z
                                           for r in refs.values()) else ""
        srcs = {r.source for r in refs.values() if r.source}
        how = (" from the DC level" if srcs == {"dc"} else
               " from the step phasors" if srcs == {"ac"} else "")
        log.info(f"  {name} gain vs the median of {len(refs)} cards{how}{tag}: "
                 + ", ".join(parts))
        dcs = [f"{short(c)} {1e3 * r.dc_V:.1f}" for c, r in sorted(refs.items())
               if np.isfinite(r.dc_V)]
        if dcs:
            log.info(f"  {name} DC level [mV]: " + ", ".join(dcs))
    corr = [(short(c), r.applied) for d in out.values() for c, r in d.items()
            if r.used_for_Z and r.applied != 1.0]
    if corr:
        log.info("  card gain removed from Z: " + ", ".join(
            f"{c} /{g:.4f}" for c, g in corr))
    elif m == "report":
        log.info("  mode 'report': nothing removed (set card_gain='correct' "
                 "to divide each card's Z by its gain)")


# ---------------------------------------------------------------------------
# 2. the impedance, card against plate
# ---------------------------------------------------------------------------

BANDS = (("lt1Hz", 0.0, 1.0), ("1_10Hz", 1.0, 10.0),
         ("10_100Hz", 10.0, 100.0), ("gt100Hz", 100.0, np.inf))


def _fkey(f) -> float:
    return float(f"{float(f):.4g}")


@dataclass
class CardFactor:
    card: str
    n_segments: int = 0
    bands_pct: dict = field(default_factory=dict)   # band -> median % vs plate
    j_dc_pct: float = float("nan")
    voltage_gain_pct: float = float("nan")          # from the UC check
    verdict: str = ""

    @property
    def spread_pct(self) -> float:
        v = [x for x in self.bands_pct.values() if np.isfinite(x)]
        return float(max(v) - min(v)) if len(v) >= 2 else float("nan")

    @property
    def level_pct(self) -> float:
        v = [x for x in self.bands_pct.values() if np.isfinite(x)]
        return float(np.median(v)) if v else float("nan")

    def row(self) -> dict:
        def r(x):
            return round(float(x), 2) if np.isfinite(x) else ""
        return {"card": self.card, "n_segments": self.n_segments,
                **{f"Z_{b}_pct": r(self.bands_pct.get(b, np.nan))
                   for b, _lo, _hi in BANDS},
                "Z_level_pct": r(self.level_pct),
                "Z_band_spread_pct": r(self.spread_pct),
                "j_dc_pct": r(self.j_dc_pct),
                "uc_gain_pct": r(self.voltage_gain_pct),
                "verdict": self.verdict}


def impedance_factors(spectra: dict, card_of: dict[str, str] | None = None,
                      voltage: dict[str, float] | None = None,
                      cfg=None) -> dict[str, CardFactor]:
    """Per card: how far its segments' |Z| sits from the plate median, per band.

    `spectra`: segment -> object with .freq, .Z_corr (or .Z), .card, .j_dc.
    A card factor that is the same in every band AND matched by an opposite
    j_dc factor is what a gain error looks like; one that changes with
    frequency is the cell.
    """
    flat_max = float(_opt(cfg, "card_gain_flat_pct", 2.0))
    tol = float(_opt(cfg, "card_gain_tol_pct", 1.0))
    by_f: dict[float, list[float]] = {}
    mags: dict[str, dict[float, float]] = {}
    for s, sp in spectra.items():
        Z = np.asarray(getattr(sp, "Z_corr", getattr(sp, "Z", [])), complex)
        f = np.asarray(sp.freq, float)
        d = {}
        for fi, zi in zip(f, Z):
            if np.isfinite(zi) and abs(zi) > 0:
                d[_fkey(fi)] = abs(zi)
                by_f.setdefault(_fkey(fi), []).append(abs(zi))
        mags[s] = d
    need = max(5, int(np.ceil(0.3 * len(spectra))))
    ref = {f: float(np.median(v)) for f, v in by_f.items() if len(v) >= need}
    card_of = card_of or {s: getattr(sp, "card", "") for s, sp in spectra.items()}

    j = {s: float(getattr(sp, "j_dc", np.nan)) for s, sp in spectra.items()}
    jv = np.array([v for v in j.values() if np.isfinite(v)])
    j_med = float(np.median(jv)) if jv.size else float("nan")

    out: dict[str, CardFactor] = {}
    for c in sorted(set(card_of.values())):
        segs = [s for s in spectra if card_of.get(s) == c]
        cf = CardFactor(card=c, n_segments=len(segs))
        for b, lo, hi in BANDS:
            vals = [np.log(mags[s][f] / ref[f]) for s in segs
                    for f in mags[s] if f in ref and lo <= f < hi]
            cf.bands_pct[b] = (100 * (np.exp(np.median(vals)) - 1)
                               if len(vals) >= 3 else float("nan"))
        js = [j[s] for s in segs if np.isfinite(j[s])]
        if js and np.isfinite(j_med) and j_med:
            cf.j_dc_pct = 100 * (np.median(js) / j_med - 1)
        if voltage and c in voltage:
            cf.voltage_gain_pct = float(voltage[c])
        lvl, spr = cf.level_pct, cf.spread_pct
        worst = max((abs(v) for v in cf.bands_pct.values() if np.isfinite(v)),
                    default=float("nan"))
        if not np.isfinite(lvl):
            cf.verdict = "too_few_points"
        elif worst <= 2 * tol:
            cf.verdict = "consistent"
        elif np.isfinite(spr) and spr <= flat_max and np.isfinite(cf.j_dc_pct) \
                and np.sign(cf.j_dc_pct) == -np.sign(lvl) \
                and abs(cf.j_dc_pct) >= 0.5 * abs(lvl):
            cf.verdict = "gain_like"
        else:
            cf.verdict = "frequency_dependent"
        out[c] = cf
    return out


# ---------------------------------------------------------------------------
# 3. the DC current, against the bench, across conditions
# ---------------------------------------------------------------------------

def current_of(condition: str) -> float:
    """"450A" -> 450.0; nan when the name carries no current."""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*A\b", str(condition or ""), re.I)
    return float(m.group(1).replace(",", ".")) if m else float("nan")


def dc_closure_table(rows: list[dict]) -> dict:
    """Fit I_meas = a * I_ref + b over the conditions.

    `rows`: one dict per condition with keys condition, i_full_A (the plate
    current from the segments, area-scaled) and i_ref_A (bench or setpoint).
    """
    pts = [(float(r["i_ref_A"]), float(r["i_full_A"])) for r in rows
           if np.isfinite(r.get("i_ref_A", np.nan))
           and np.isfinite(r.get("i_full_A", np.nan))]
    out = {"n": len(pts), "rows": rows}
    for r in rows:
        if np.isfinite(r.get("i_ref_A", np.nan)) and r["i_ref_A"]:
            r["dev_pct"] = 100 * (r["i_full_A"] / r["i_ref_A"] - 1)
    if len(pts) >= 2:
        x, y = np.array(pts).T
        a, b = np.polyfit(x, y, 1)
        out.update(scale=float(a), offset_A=float(b),
                   scale_pct=float(100 * (a - 1)),
                   resid_A=float(np.sqrt(np.mean((y - (a * x + b)) ** 2))))
    elif len(pts) == 1:
        x, y = pts[0]
        out.update(scale=y / x, offset_A=0.0, scale_pct=100 * (y / x - 1),
                   resid_A=float("nan"))
    return out


def plate_current(j_dc: dict[str, float], areas: dict[str, float],
                  plate_key: str = "gen1") -> tuple[float, float, float]:
    """(I over the measured area, measured area, I scaled to the plate)."""
    import r2d2_geometry
    pairs = [(j_dc[s], areas[s]) for s in j_dc
             if s in areas and np.isfinite(j_dc[s]) and np.isfinite(areas[s])]
    plate = r2d2_geometry.plate(plate_key)
    total = sum(s.area_cm2 for s in plate.segments.values())
    a = sum(p[1] for p in pairs)
    i = sum(p[0] * p[1] for p in pairs)
    return i, a, (i * total / a if a > 0 else float("nan"))
