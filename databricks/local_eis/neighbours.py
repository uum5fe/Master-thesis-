#!/usr/bin/env python3
"""
neighbours.py  --  which segments do not behave like the ones around them
=========================================================================

    from neighbours import adjacency, scalar_outliers, spectrum_outliers

A plate has a real spatial gradient. Reactant is consumed along the channel,
so current density falls from inlet to outlet, and the membrane is wetter at
one end than the other. A segment with twice the plate-median R_p is
therefore not news -- it may simply be at the outlet, where every segment has
a high R_p.

THE COMPARISON THAT MEANS SOMETHING IS LOCAL. A segment that differs from the
segments touching it is anomalous whatever the gradient does, because its
neighbours share its position in the gradient. That is the whole idea here,
and it is why this module compares against an ADJACENCY SET rather than
against the plate mean or against a fitted spatial field.

    plate mean      -> flags the whole outlet, which is physics, not a fault
    fitted field    -> the fit absorbs the fault it is meant to reveal
    immediate ring  -> the gradient cancels, the fault does not

WHAT COUNTS AS A NEIGHBOUR
--------------------------
Two segments are neighbours when a pad of one shares an EDGE with a pad of
the other. This is computed from the pad sets, not from centroid distance,
because the gen1 segments are not rectangles -- the edge segments are
staircases, and a centroid-distance rule pairs a staircase with whatever
happens to be near its centre of mass rather than with what it touches.

WHAT IS TESTED
--------------
1.  SCALAR PARAMETERS -- R_ohmic, R_p, j, temperature, whatever the run
    carries. A robust z-score against the neighbour ring: the modified
    z-score of Iglewicz & Hoaglin, 0.6745*(x - median)/MAD, which does not
    let one bad neighbour define the neighbourhood the way a mean and a
    standard deviation would.

2.  THE SPECTRUM ITSELF, frequency by frequency, against the neighbour
    median spectrum. This is the one that says WHAT KIND of difference it
    is, and the band it sits in is the diagnosis:

        offset at every frequency        -> contact or lead resistance
        high-frequency only              -> ohmic / membrane hydration
        mid-band arc bigger              -> kinetics, catalyst, poisoning
        low-frequency tail only          -> transport, flooding, starvation

WHAT THIS DOES NOT DO
---------------------
It does not decide that a segment is broken. A segment can differ from its
neighbours because it IS different -- an edge segment against interior ones,
a segment under a channel land against one under a channel. The output is
ranked evidence with the neighbourhood shown alongside, so the difference can
be read rather than trusted.

REFERENCE
---------
B. Iglewicz, D. C. Hoaglin, "How to Detect and Handle Outliers", ASQC Basic
References in Quality Control vol. 16 (1993) -- the modified z-score and the
3.5 cut used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import core            # noqa: F401
except ImportError:
    pass
import r2d2_geometry as geom


#: Iglewicz & Hoaglin's cut for the modified z-score. Not tuned here: it is
#: the published value, and moving it is a decision to report more or fewer
#: segments rather than a decision about the plate.
Z_CUT = 3.5

#: 1/Phi^-1(0.75). Scales the MAD to a standard-deviation estimate for
#: normally distributed data, which is what makes the z-score comparable to
#: the familiar 3-sigma intuition.
MAD_TO_SIGMA = 0.6745

#: A ring smaller than this cannot support a median and a spread. Corner
#: segments have two neighbours; they are still reported, flagged.
MIN_RING = 3


# ===========================================================================
# 1. Who touches whom
# ===========================================================================


def adjacency(plate_name: str | None = None) -> dict[str, set[str]]:
    """Segment -> the set of segments sharing a pad edge with it.

    Edge adjacency, not corner: two pads that meet only at a corner are not
    neighbours, because nothing flows between them.
    """
    segs = (geom.plate(plate_name).segments if plate_name else geom.SEGMENTS)
    owner: dict[tuple[int, int], str] = {}
    for name, seg in segs.items():
        for pad in seg.pads:
            owner[tuple(pad)] = name

    out: dict[str, set[str]] = {name: set() for name in segs}
    for (col, row), name in owner.items():
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = owner.get((col + dc, row + dr))
            if other is not None and other != name:
                out[name].add(other)
                out[other].add(name)
    return out


# ===========================================================================
# 2. Robust deviation from the ring
# ===========================================================================


def _modified_z(x: float, ring: np.ndarray) -> tuple[float, float, float]:
    """(z, median, MAD-as-sigma) of `x` against `ring`.

    The MAD is used rather than a standard deviation because the ring is
    small -- four to six segments -- and one genuinely faulty neighbour would
    otherwise inflate the spread enough to hide the segment under test. That
    is not hypothetical on a plate where faults come in adjacent pairs.
    """
    ring = np.asarray([v for v in ring if np.isfinite(v)], float)
    if ring.size < 2:
        return np.nan, np.nan, np.nan
    med = float(np.median(ring))
    mad = float(np.median(np.abs(ring - med)))
    sigma = mad / MAD_TO_SIGMA
    if sigma <= 0:
        # every neighbour identical: any difference at all is infinite z,
        # which is true but useless. Fall back to the ring's full range.
        rng = float(np.ptp(ring))
        if rng <= 0:
            return (0.0 if x == med else np.inf), med, 0.0
        sigma = rng / 2.0
    return float((x - med) / sigma), med, sigma


@dataclass
class Finding:
    """One segment, one parameter, one verdict."""
    segment: str
    param: str
    value: float
    ring_median: float
    ring_sigma: float
    z: float
    n_ring: int
    ring: list = field(default_factory=list)
    note: str = ""

    @property
    def direction(self) -> str:
        return "high" if self.z > 0 else "low"

    def as_row(self) -> dict:
        return {"segment": self.segment, "param": self.param,
                "value": round(float(self.value), 6),
                "ring_median": round(float(self.ring_median), 6),
                "ring_sigma": round(float(self.ring_sigma), 6),
                "z": round(float(self.z), 3), "n_ring": self.n_ring,
                "direction": self.direction,
                "neighbours": " ".join(sorted(self.ring, key=_as_int)),
                "note": self.note}


def _as_int(name: str) -> int:
    try:
        return int(name)
    except (TypeError, ValueError):
        return 0


def scalar_outliers(values: dict[str, float], param: str = "value",
                    plate_name: str | None = None, z_cut: float = Z_CUT,
                    adj: dict[str, set[str]] | None = None) -> list[Finding]:
    """Segments whose value stands out from the ring that touches them.

    `values` is segment name -> number, exactly what `gold` and the viewer
    already pass around. Segments missing from it are simply not compared,
    and they are not counted as neighbours either -- an unmeasured segment
    must not shrink somebody else's ring silently.
    """
    adj = adj if adj is not None else adjacency(plate_name)
    have = {k: float(v) for k, v in values.items()
            if v is not None and np.isfinite(float(v))}
    out: list[Finding] = []
    for name, x in have.items():
        ring_names = sorted(n for n in adj.get(name, ()) if n in have)
        ring = np.array([have[n] for n in ring_names], float)
        z, med, sigma = _modified_z(x, ring)
        if not np.isfinite(z):
            continue
        note = ("" if len(ring_names) >= MIN_RING else
                f"only {len(ring_names)} measured neighbour(s); the spread is "
                f"barely estimated, read this as a hint")
        if abs(z) >= z_cut:
            out.append(Finding(segment=name, param=param, value=x,
                               ring_median=med, ring_sigma=sigma, z=z,
                               n_ring=len(ring_names), ring=ring_names,
                               note=note))
    return sorted(out, key=lambda f: -abs(f.z))


# ===========================================================================
# 3. Where in the spectrum the difference sits
# ===========================================================================


@dataclass
class SpectrumFinding:
    segment: str
    freq_hz: np.ndarray
    dev_db: np.ndarray            # 20*log10(|Z_seg| / |Z_ring median|)
    dev_phase_deg: np.ndarray
    n_ring: int
    ring: list = field(default_factory=list)

    def worst(self) -> tuple[float, float]:
        """(frequency, deviation in dB) where it departs most."""
        if not len(self.dev_db):
            return float("nan"), float("nan")
        k = int(np.nanargmax(np.abs(self.dev_db)))
        return float(self.freq_hz[k]), float(self.dev_db[k])

    def band_summary(self) -> str:
        """Which decade carries the difference -- the diagnostic half."""
        f, d = self.freq_hz, self.dev_db
        ok = np.isfinite(d)
        if not ok.any():
            return "no overlapping frequencies with the ring"
        bands = [("low  (< 1 Hz)", f < 1.0),
                 ("mid  (1-100 Hz)", (f >= 1.0) & (f < 100.0)),
                 ("high (>= 100 Hz)", f >= 100.0)]
        parts = []
        for label, sel in bands:
            m = sel & ok
            if m.sum():
                parts.append(f"{label}: {np.median(d[m]):+.1f} dB")
        return ";  ".join(parts)


def spectrum_outliers(spectra: dict, plate_name: str | None = None,
                      adj: dict[str, set[str]] | None = None,
                      min_points: int = 4) -> list[SpectrumFinding]:
    """Per-frequency departure of each segment from its neighbour ring.

    `spectra` is segment -> (freq_hz, Z) with Z complex, in whatever unit the
    caller uses; the comparison is a RATIO, so the unit cancels and only the
    shape is tested.

    Frequencies are matched exactly rather than interpolated. Two segments on
    different cards can carry different schedules, and interpolating one onto
    the other's grid would manufacture agreement at frequencies neither
    actually measured.
    """
    adj = adj if adj is not None else adjacency(plate_name)
    clean = {}
    for name, item in spectra.items():
        f, z = item
        f = np.asarray(f, float)
        z = np.asarray(z, complex)
        ok = np.isfinite(f) & np.isfinite(z.real) & np.isfinite(z.imag) & (np.abs(z) > 0)
        if ok.sum() >= min_points:
            order = np.argsort(f[ok])
            clean[name] = (f[ok][order], z[ok][order])

    out: list[SpectrumFinding] = []
    for name, (f, z) in clean.items():
        ring_names = sorted(n for n in adj.get(name, ()) if n in clean)
        if len(ring_names) < 2:
            continue
        mag, pha = [], []
        for fi, zi in zip(f, z):
            vals = []
            for other in ring_names:
                fo, zo = clean[other]
                hit = np.flatnonzero(np.abs(fo / fi - 1.0) < 1e-6)
                if hit.size:
                    vals.append(zo[hit[0]])
            if len(vals) < 2:
                mag.append(np.nan)
                pha.append(np.nan)
                continue
            v = np.asarray(vals)
            ref_mag = float(np.median(np.abs(v)))
            ref_pha = float(np.median(np.degrees(np.angle(v))))
            mag.append(20.0 * np.log10(abs(zi) / ref_mag) if ref_mag > 0 else np.nan)
            pha.append(float(np.degrees(np.angle(zi)) - ref_pha))
        out.append(SpectrumFinding(segment=name, freq_hz=f,
                                   dev_db=np.asarray(mag, float),
                                   dev_phase_deg=np.asarray(pha, float),
                                   n_ring=len(ring_names), ring=ring_names))
    return sorted(out, key=lambda s: -abs(np.nan_to_num(s.worst()[1])))


# ===========================================================================
# 4. One call for the whole plate
# ===========================================================================


def analyse(params: dict[str, dict[str, float]],
            spectra: dict | None = None,
            plate_name: str | None = None,
            z_cut: float = Z_CUT) -> dict:
    """Run every scalar parameter and, when given, the spectra.

    `params` is {parameter name: {segment: value}} -- the shape gold already
    produces. Returns findings ranked by |z|, plus the adjacency actually
    used, so a reader can check which ring a verdict came from.
    """
    adj = adjacency(plate_name)
    findings: list[Finding] = []
    for name, values in params.items():
        findings.extend(scalar_outliers(values, param=name, adj=adj,
                                        z_cut=z_cut))
    findings.sort(key=lambda f: -abs(f.z))

    flagged: dict[str, list[str]] = {}
    for f in findings:
        flagged.setdefault(f.segment, []).append(f.param)

    spec = spectrum_outliers(spectra, adj=adj) if spectra else []
    return {
        "findings": findings,
        "rows": [f.as_row() for f in findings],
        "by_segment": flagged,
        "spectra": spec,
        "adjacency": {k: sorted(v, key=_as_int) for k, v in adj.items()},
        "z_cut": z_cut,
        "n_segments_flagged": len(flagged),
    }


def _self_test() -> int:
    """Plant one fault and check it is the segment that comes back."""
    adj = adjacency()
    print(f"adjacency: {len(adj)} segments, ring sizes "
          f"{min(len(v) for v in adj.values())}..{max(len(v) for v in adj.values())}, "
          f"mean {np.mean([len(v) for v in adj.values()]):.1f}")

    cents = geom.centroids()
    # a smooth inlet-to-outlet gradient: every segment differs from the plate
    # mean, and none of them is a fault
    values = {k: 0.30 - 0.0008 * cx for k, (cx, _cy) in cents.items()}
    quiet = scalar_outliers(values, param="gradient only")
    print(f"pure gradient  -> {len(quiet)} flagged (want 0)")

    victim = "34"
    values[victim] *= 1.9
    loud = scalar_outliers(values, param="one planted fault")
    print(f"+ one fault    -> {len(loud)} flagged: "
          + ", ".join(f"{f.segment} (z={f.z:+.1f})" for f in loud[:5]))
    ok = bool(loud) and loud[0].segment == victim and not quiet
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
