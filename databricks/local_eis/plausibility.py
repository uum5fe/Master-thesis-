#!/usr/bin/env python3
"""
plausibility.py  --  is this local EIS result believable?
=========================================================

The gates in silver.py answer "is this POINT usable".  This module answers a
different and larger question: taking the surviving points at face value, does
the resulting picture of the plate hold together as physics?

WHY THIS IS SEPARATE FROM THE GATES
-----------------------------------
Every gate in silver is local -- it looks at one segment at one frequency and
asks whether that phasor is trustworthy.  A run can pass every gate on every
surviving point and still be wrong in ways only visible from above: a channel
map off by one puts a real spectrum on the wrong segment (every point is fine,
the MAP is nonsense); a wrong area scales every impedance on that segment by a
constant (every point is fine, the number is wrong); a swapped inlet reverses
the flow gradient (every point is fine, the physics is backwards).  None of
those is detectable one point at a time, and all of them are detectable from
the shape of the finished plate.

THE PARTIAL-PLATE CASE
----------------------
This is written for a plate that is only partly instrumented -- specifically
36 of the 72 segments, which is the wiring available here.  Two facts about
that make most of the difficulty:

1.  36 OF 72 SEGMENTS IS NOT HALF THE PLATE.  The R2-D2 segments have areas
    from 1.355 to 8.470 cm2, and the numbering is not area-neutral.  Measured
    on the gen1 map:

        segments  1-36 : 209.040 cm2 = 68.6 % of the plate  (the INTERIOR,
                         mean area 5.807 cm2)
        segments 37-72 :  95.880 cm2 = 31.4 % of the plate  (the PERIMETER,
                         mean area 2.663 cm2)

    gen2 is the same story (69.3 / 30.7 %).  So "the first 36" and "the last
    36" are not two comparable halves; they are the middle of the plate and
    the ring around it, and anything that extrapolates from one to the whole
    plate has to do it BY AREA, never by segment count.  Counting would put
    the first-36 case 18.6 percentage points wrong.

2.  BOTH SETS SPAN THE WHOLE PLATE.  Their centroids have the same mean in x
    and y; only the spread differs (the perimeter set reaches further out).
    So neither subset is a spatial half whose unmeasured complement lies
    somewhere else -- the unmeasured segments are interleaved with the
    measured ones everywhere.  That is good news for interpolation and bad
    news for anyone hoping to treat the two runs as independent halves.

WHAT IS AND IS NOT VALID ON A PARTIAL PLATE
-------------------------------------------
Impedance here is AREA-SPECIFIC (ohm.cm2), so a partial-plate aggregate is
directly comparable to a whole-cell one WITHOUT rescaling -- but only under
the assumption that the unmeasured area behaves like the measured area.  That
assumption is exactly what is false when the plate is non-uniform, which is
the reason for measuring locally in the first place.  So the aggregate check
is reported together with the area fraction it rests on, and the perimeter-
only case (31 %) is flagged as the weak one: a third of the area, and the
third least like the rest.

Each check returns a verdict rather than a bare number, and "n/a" is a real
verdict -- a check that cannot be evaluated says so instead of silently
passing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import r2d2_geometry
import utils


# ===========================================================================
# 1. What a check reports
# ===========================================================================

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "n/a"

#: Order matters: the report sorts by severity, and a run's overall verdict is
#: the worst it contains.
_SEVERITY = {PASS: 0, NA: 1, WARN: 2, FAIL: 3}


@dataclass(frozen=True)
class Check:
    """One plausibility question, its answer, and why that is the answer."""

    name: str
    verdict: str
    detail: str
    value: float = float("nan")
    #: What would have to be true for this verdict to be wrong. Recorded
    #: because a check whose assumptions are invisible is not evidence.
    rests_on: str = ""

    def __str__(self) -> str:
        mark = {PASS: "PASS", WARN: "WARN", FAIL: "FAIL", NA: " -- "}[self.verdict]
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """The worst verdict present; a report is only as good as its weakest."""
        if not self.checks:
            return NA
        return max((c.verdict for c in self.checks), key=lambda v: _SEVERITY[v])

    def by_verdict(self, verdict: str) -> list[Check]:
        return [c for c in self.checks if c.verdict == verdict]

    def rows(self) -> list[dict]:
        return [{"check": c.name, "verdict": c.verdict,
                 "value": round(float(c.value), 6), "detail": c.detail,
                 "rests_on": c.rests_on} for c in self.checks]


# ===========================================================================
# 2. Coverage: what fraction of the PLATE, not of the segment list
# ===========================================================================

def coverage(measured: list[str], plate_key: str = "gen1") -> Check:
    """How much of the plate this run actually saw, by area.

    Segment COUNT is the misleading number and the tempting one. On gen1 the
    first 36 segments are 68.6 % of the area and the last 36 are 31.4 %, so
    "36 of 72" describes two very different measurements depending on which
    36 they are.
    """
    plate = r2d2_geometry.plate(plate_key)
    areas = {int(k): s.area_cm2 for k, s in plate.segments.items()}
    total = sum(areas.values())
    have = {int(s) for s in measured} & set(areas)
    if not have:
        return Check("coverage", FAIL, "no segments measured at all", 0.0)

    seen = sum(areas[n] for n in have)
    frac = seen / total
    by_count = len(have) / len(areas)

    detail = (f"{len(have)}/{len(areas)} segments = {100 * frac:.1f} % of the "
              f"{total:.1f} cm2 plate (counting segments would say "
              f"{100 * by_count:.1f} %)")
    if frac >= 0.60:
        verdict = PASS
    elif frac >= 0.25:
        verdict = WARN
        detail += " -- a minority of the area carries the extrapolation"
    else:
        verdict = FAIL
        detail += " -- too little area to speak for the plate"
    return Check("coverage", verdict, detail, frac,
                 rests_on="the geometry in r2d2_geometry.py being the "
                          "as-built pad map for this plate")


def describe_block(measured: list[str], plate_key: str = "gen1") -> Check:
    """Name the measured set: the interior block, the perimeter, or neither.

    Worth stating explicitly, because the two 36-segment blocks are not
    interchangeable and the difference does not show up in any per-point
    number.
    """
    plate = r2d2_geometry.plate(plate_key)
    areas = {int(k): s.area_cm2 for k, s in plate.segments.items()}
    have = {int(s) for s in measured} & set(areas)
    if not have:
        return Check("measured block", NA, "nothing measured")

    lo, hi = set(range(1, 37)), set(range(37, 73))
    frac = sum(areas[n] for n in have) / sum(areas.values())
    if have == lo:
        return Check("measured block", PASS,
                     f"segments 1-36, the plate INTERIOR "
                     f"({100 * frac:.1f} % of area, the larger segments)",
                     frac)
    if have == hi:
        return Check("measured block", WARN,
                     f"segments 37-72, the plate PERIMETER "
                     f"({100 * frac:.1f} % of area, the smaller segments) -- "
                     f"the weaker of the two blocks to extrapolate from, "
                     f"being both less area and the least typical part of it",
                     frac)

    runs = _contiguous_runs(sorted(have))
    shape = (f"one contiguous run {runs[0][0]}-{runs[0][1]}" if len(runs) == 1
             else f"{len(runs)} separate runs")
    return Check("measured block", WARN,
                 f"{len(have)} segments in {shape}, {100 * frac:.1f} % of "
                 f"area -- neither of the two wiring blocks, so the coverage "
                 f"is whatever survived rather than what was cabled", frac)


def _contiguous_runs(nums: list[int]) -> list[tuple[int, int]]:
    """[1,2,3,7,8] -> [(1,3),(7,8)]."""
    if not nums:
        return []
    runs, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n != prev + 1:
            runs.append((start, prev))
            start = n
        prev = n
    runs.append((start, prev))
    return runs


# ===========================================================================
# 3. Current closure -- the one check nothing was fitted to
# ===========================================================================

def current_closure(j_dc: dict[str, float], areas: dict[str, float],
                    setpoint_a: float | None,
                    plate_key: str = "gen1") -> Check:
    """Sum(j_s * A_s) over the measured area, scaled to the whole plate.

    Nothing upstream is fitted to the load current, so this is the only fully
    independent check of the calibration, the areas and the DC levels at once.

    On a partial plate the scaling is BY AREA and it assumes the unmeasured
    area carries the same mean current density as the measured area. State
    that plainly: a 30 % disagreement here is either a calibration error or a
    genuinely non-uniform plate, and this check cannot tell you which. What it
    can do is fail loudly when the number is not physical at all.
    """
    pairs = [(float(j_dc[s]), float(areas[s])) for s in j_dc
             if s in areas and np.isfinite(j_dc[s]) and np.isfinite(areas[s])]
    if not pairs:
        return Check("current closure", NA, "no finite DC current densities")

    plate = r2d2_geometry.plate(plate_key)
    total_area = sum(s.area_cm2 for s in plate.segments.values())
    a_meas = sum(a for _j, a in pairs)
    i_meas = sum(j * a for j, a in pairs)
    if a_meas <= 0:
        return Check("current closure", NA, "measured area is zero")
    i_full = i_meas * total_area / a_meas

    if not setpoint_a:
        return Check("current closure", NA,
                     f"{i_meas:.1f} A over {a_meas:.1f} cm2 -> {i_full:.1f} A "
                     f"full plate, but no setpoint given to compare against "
                     f"(pass --i-setpoint)", i_full)

    dev = i_full / setpoint_a - 1.0
    detail = (f"{i_meas:.1f} A measured over {a_meas:.1f} cm2 -> "
              f"{i_full:.1f} A full plate vs {setpoint_a:.1f} A setpoint "
              f"({100 * dev:+.1f} %)")
    if abs(dev) <= 0.10:
        verdict = PASS
    elif abs(dev) <= 0.30:
        verdict = WARN
        detail += " -- check the shunt calibration before trusting the map"
    else:
        verdict = FAIL
        detail += " -- too far out to be non-uniformity alone"
    return Check("current closure", verdict, detail, dev,
                 rests_on="the unmeasured area carrying the same mean current "
                          "density as the measured area")


# ===========================================================================
# 4. Shape checks: things a wrong channel map cannot fake
# ===========================================================================

def neighbour_smoothness(values: dict[str, float], plate_key: str = "gen1",
                         param: str = "R_ohmic") -> Check:
    """Is the map spatially smooth, or does it look like shuffled labels?

    R_ohmic is set by membrane thickness, compression and contact -- all of
    which vary over millimetres, not between adjacent segments. So a correct
    map is smooth and a map with the channel order scrambled is not, and the
    difference is large enough to see without any model of the cell.

    The statistic is the measured neighbour contrast against the contrast of
    the SAME values randomly permuted over the same positions. A correct map
    scores far below 1; a shuffled one scores about 1 by construction. This
    is a permutation test, so it needs no assumption about the distribution
    of R_ohmic itself.
    """
    plate = r2d2_geometry.plate(plate_key)
    pos = {int(k): (s.cx_mm, s.cy_mm) for k, s in plate.segments.items()}
    good = {int(s): float(v) for s, v in values.items()
            if int(s) in pos and np.isfinite(v)}
    if len(good) < 8:
        return Check(f"{param} smoothness", NA,
                     f"only {len(good)} finite values; needs 8")

    nums = sorted(good)
    xy = np.array([pos[n] for n in nums])
    vals = np.array([good[n] for n in nums])

    d = np.hypot(xy[:, None, 0] - xy[None, :, 0],
                 xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    nearest = np.argmin(d, axis=1)

    def contrast(v: np.ndarray) -> float:
        scale = np.median(np.abs(v - np.median(v)))
        if scale <= 0:
            return 0.0
        return float(np.median(np.abs(v - v[nearest])) / scale)

    measured = contrast(vals)
    rng = np.random.default_rng(0)
    null = np.median([contrast(rng.permutation(vals)) for _ in range(200)])
    if null <= 0:
        return Check(f"{param} smoothness", NA, "values are all identical")
    ratio = measured / null

    detail = (f"neighbour contrast {ratio:.2f} of what the same values give "
              f"when shuffled over the same positions")
    if ratio <= 0.6:
        verdict, extra = PASS, " -- the map is spatially organised"
    elif ratio <= 0.85:
        verdict, extra = WARN, (" -- weakly organised; real structure this "
                                "flat is possible, but so is a partly wrong "
                                "channel map")
    else:
        verdict, extra = FAIL, (" -- indistinguishable from random placement; "
                                "suspect the channel-to-segment mapping "
                                "before interpreting this plate")
    return Check(f"{param} smoothness", verdict, detail + extra, ratio,
                 rests_on=f"{param} being a spatially smooth property of the "
                          f"cell, which is true for ohmic resistance and "
                          f"weaker for transport terms")


def passivity(spectra: dict) -> Check:
    """Re(Z) > 0 everywhere: a passive cell cannot generate energy.

    Trivially true of good data and a clear giveaway of a sign convention
    flipped somewhere in the chain, which is otherwise easy to miss because
    a flipped spectrum still looks like a spectrum.
    """
    bad, total = [], 0
    for seg, sp in spectra.items():
        z = np.asarray(getattr(sp, "Z_corr", sp), complex)
        z = z[np.isfinite(z)]
        total += z.size
        if z.size and np.any(z.real <= 0):
            bad.append(seg)
    if not total:
        return Check("passivity", NA, "no finite impedance points")
    if not bad:
        return Check("passivity", PASS,
                     f"Re(Z) > 0 at all {total} points", 0.0)
    frac = len(bad) / max(1, len(spectra))
    return Check("passivity", FAIL,
                 f"{len(bad)} segment(s) have Re(Z) <= 0 somewhere "
                 f"({', '.join(sorted(bad, key=str)[:8])}"
                 f"{' ...' if len(bad) > 8 else ''}) -- a passive cell cannot "
                 f"do that, so suspect a sign or a reference channel",
                 frac)


def flow_trend(values: dict[str, float], plate_key: str = "gen1",
               param: str = "R_ohmic", axis: str = "x") -> Check:
    """Does the map vary along the flow axis at all, and in which direction?

    Reported rather than judged. Air entering dry and leaving humid makes the
    membrane wetter downstream, so R_ohmic usually FALLS from inlet to outlet
    -- but the sign depends on the operating point, and at high current the
    outlet can flood instead. What would be suspicious is no gradient at all
    across a plate that is known to have one, so a flat result is the warning
    and either direction is merely stated.
    """
    plate = r2d2_geometry.plate(plate_key)
    # The pipeline's Plate carries no flow axis -- only the viewer's geometry
    # does -- so it is named here rather than guessed silently. x is the long
    # axis of this plate (250 mm against 116 mm) and the one the channels run
    # along; pass axis="y" if a future plate is plumbed the other way.
    coord = {int(k): (s.cx_mm if axis == "x" else s.cy_mm)
             for k, s in plate.segments.items()}
    good = {int(s): float(v) for s, v in values.items()
            if int(s) in coord and np.isfinite(v)}
    if len(good) < 8:
        return Check(f"{param} along flow", NA,
                     f"only {len(good)} finite values; needs 8")

    nums = sorted(good)
    x = np.array([coord[n] for n in nums])
    y = np.array([good[n] for n in nums])
    if np.std(x) == 0 or np.std(y) == 0:
        return Check(f"{param} along flow", NA, "no spread to correlate")
    r = float(np.corrcoef(x, y)[0, 1])

    where = "inlet to outlet" if axis == "x" else "across the flow"
    direction = "falls" if r < 0 else "rises"
    detail = f"{param} {direction} {where} (r = {r:+.2f} against {axis})"
    if abs(r) < 0.15:
        return Check(f"{param} along flow", WARN,
                     detail + " -- essentially flat; a plate with a real "
                              "humidity gradient should show one, so check "
                              "the flow direction and the segment positions",
                     r)
    return Check(f"{param} along flow", PASS, detail, r,
                 rests_on="the plate's flow_axis and inlet end being set "
                          "correctly in the geometry")


# ===========================================================================
# 5. The aggregate, against an instrument that shares nothing
# ===========================================================================

def aggregate_vs_reference(freq: np.ndarray, Z_agg: np.ndarray,
                           ref_freq: np.ndarray, ref_Z_asr: np.ndarray,
                           area_fraction: float) -> Check:
    """The partial-plate aggregate against a whole-cell sweep, both in ohm.cm2.

    No rescaling is applied and none is needed: both sides are area-specific
    already. What IS assumed is that the unmeasured area looks like the
    measured area, and that assumption gets weaker exactly as the coverage
    falls -- so the coverage is carried into the verdict rather than left for
    the reader to remember.
    """
    f = np.asarray(freq, float)
    z = np.asarray(Z_agg, complex)
    rf = np.asarray(ref_freq, float)
    rz = np.asarray(ref_Z_asr, complex)
    if f.size == 0 or rf.size == 0:
        return Check("aggregate vs whole cell", NA, "one side has no points")

    lo, hi = max(f.min(), rf.min()), min(f.max(), rf.max())
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return Check("aggregate vs whole cell", NA,
                     f"no overlap: local {f.min():.3g}-{f.max():.3g} Hz vs "
                     f"reference {rf.min():.3g}-{rf.max():.3g} Hz")

    zi = utils.interp_complex(f[band], rf, rz)
    dev = float(np.median(np.abs(z[band]) / np.abs(zi) - 1.0))

    detail = (f"|Z| differs by {100 * dev:+.1f} % (median over "
              f"{int(band.sum())} overlapping points, {lo:.3g}-{hi:.3g} Hz) "
              f"from {100 * area_fraction:.0f} % of the plate")
    # A partial plate is allowed to disagree by roughly the amount the plate
    # is non-uniform, so the tolerance widens as the coverage falls rather
    # than pretending one number fits both blocks.
    tol = 0.10 + 0.40 * (1.0 - max(0.0, min(1.0, area_fraction)))
    if abs(dev) <= tol:
        verdict = PASS
    elif abs(dev) <= 2 * tol:
        verdict = WARN
        detail += f" -- outside the {100 * tol:.0f} % expected at this coverage"
    else:
        verdict = FAIL
        detail += (f" -- far outside the {100 * tol:.0f} % expected at this "
                   f"coverage; suspect area or calibration, not uniformity")
    return Check("aggregate vs whole cell", verdict, detail, dev,
                 rests_on="the unmeasured area having a similar area-specific "
                          "impedance to the measured area")


# ===========================================================================
# 6. Running the lot
# ===========================================================================

def check_run(sr, cfg=None, plate_key: str = "gen1",
              reference: tuple[np.ndarray, np.ndarray] | None = None
              ) -> Report:
    """Every check that the available data supports, on a finished SilverRun."""
    spectra = sr.spectra
    measured = list(spectra)
    setpoint = getattr(cfg, "i_setpoint_a", None) if cfg else None

    cov = coverage(measured, plate_key)
    checks = [cov, describe_block(measured, plate_key)]

    checks.append(current_closure(
        {s: sp.j_dc for s, sp in spectra.items()},
        {s: sp.area_cm2 for s, sp in spectra.items()},
        setpoint, plate_key))

    checks.append(passivity(spectra))

    r_ohmic = {s: sp.R_ohmic for s, sp in spectra.items()}
    checks.append(neighbour_smoothness(r_ohmic, plate_key))
    checks.append(flow_trend(r_ohmic, plate_key))

    if reference is not None and len(getattr(sr, "cell_freq", [])):
        checks.append(aggregate_vs_reference(
            sr.cell_freq, sr.Z_cell, reference[0], reference[1],
            float(cov.value)))

    return Report(checks)


def report(rep: Report, log=None) -> None:
    utils.section("plausibility", log)
    log = log or utils.get_logger(True)
    for c in rep.checks:
        line = f"  {c}"
        if c.verdict == FAIL:
            log.warning(line)
        elif c.verdict == WARN:
            log.warning(line)
        else:
            log.info(line)
        if c.rests_on and c.verdict in (PASS, WARN):
            log.info(f"        rests on: {c.rests_on}")
    log.info(f"  overall: {rep.verdict.upper()}")


def save(rep: Report, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "plausibility.csv"
    utils.write_table(path, rep.rows())
    return path


# ===========================================================================
# 7. Standalone use, on results already on disk
# ===========================================================================

def _load_summary(path: Path) -> dict[str, dict]:
    import csv
    with Path(path).open(newline="") as fh:
        return {r["segment"]: r for r in csv.DictReader(fh)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Plausibility checks on a finished local EIS result.")
    ap.add_argument("run_dir", type=Path,
                    help="a run directory holding gold/plate_summary.csv")
    ap.add_argument("--plate", default="gen1", choices=["gen1", "gen2"])
    ap.add_argument("--i-setpoint", type=float, default=None,
                    help="load current in A, for the closure check")
    args = ap.parse_args(argv)

    log = utils.get_logger(True)
    summary = args.run_dir / "gold" / "plate_summary.csv"
    if not summary.is_file():
        log.warning(f"no {summary}")
        return 2

    rows = _load_summary(summary)

    def column(name: str) -> dict[str, float]:
        out = {}
        for seg, row in rows.items():
            if row.get("measured", "1") in ("0", "False", "false"):
                continue          # inferred values would test the interpolator
            try:
                out[seg] = float(row[name])
            except (KeyError, TypeError, ValueError):
                continue
        return out

    measured = list(column("R_ohmic"))
    checks = [coverage(measured, args.plate),
              describe_block(measured, args.plate),
              neighbour_smoothness(column("R_ohmic"), args.plate),
              flow_trend(column("R_ohmic"), args.plate)]

    j = column("j_dc")
    if j:
        plate = r2d2_geometry.plate(args.plate)
        areas = {k: s.area_cm2 for k, s in plate.segments.items()}
        checks.append(current_closure(j, areas, args.i_setpoint, args.plate))

    rep = Report(checks)
    report(rep, log)
    log.info(f"  written to {save(rep, args.run_dir)}")
    return 0 if rep.verdict != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
