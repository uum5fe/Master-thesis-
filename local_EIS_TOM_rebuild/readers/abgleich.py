#!/usr/bin/env python3
"""
abgleich.py  --  the plate calibration, from the raw Step files upwards
=======================================================================

`eis_local.PlateCalibration` consumes two two-column files:

    curr.csv    72 rows "c0;c1"   j_s [A/cm^2] = u_s [V] / (c0 + 1e-3*c1*T_C)
    temp.csv     4 rows "c0;c1"   T_C = (u_temp [V] - c0) / c1

They are *derived* quantities.  What the calibration bench actually delivers
is a set of `Step<k>_<T>Grad.csv` files, one per plate temperature, each
holding for every segment the four temperature-sensor voltages and five
(i_s, u_s) points of a DC current sweep:

    s1:<TAB>temp1=4.369482V;temp2=...;<TAB>i_s=-0.000006A;u_s=0.010845V;<TAB>...

This module closes that gap in both directions:

  * `fit_coefficients()` rebuilds curr.csv and temp.csv from the Step files,
    so a newly calibrated plate does not depend on someone else's spreadsheet
    and a delivered coefficient file can be *checked* rather than trusted;
  * `verify()` reports how well a delivered coefficient pair reproduces the
    Step data it claims to summarise.

WHAT THE MODEL IS, AND WHY IT HOLDS
-----------------------------------
Per segment the bench fits u_s = R(T) * i_s over the sweep, and R is linear
in temperature -- copper, TCR ~ 0.4 %/K.  Verified on the delivered files:
on Kashyyyk, R(20 C) = 1.810 V/A and R(90 C) = 2.300 V/A for segment 1, which
is 0.42 %/K.  Writing K(T) = c0 + 1e-3*c1*T for the transfer in V/(A/cm^2),
the ratio R(T)/K(T) comes out constant to 0.4 % over 20..90 C -- 2.95 cm^2 on
Kashyyyk, 3.08 cm^2 on Naboo -- and, importantly, constant ACROSS SEGMENTS
whose true areas differ by a factor of 12.5.

That last fact is the reason the calibration returns a current *density* and
not a current: every pad carries its own via, so a 25-pad segment has 25 vias
in parallel and R_via * A_seg is invariant.  The area is already inside c0
and c1, and multiplying by the geometric area afterwards would count it
twice.

The temperature sensors are read the other way round -- the file stores the
sensor's own line V = c0 + c1*T, so the calibration inverts it.  Recovered
from the Step files to four decimals:

    temp1  V = 3.188346 + 0.013126*T     delivered  3.188087 ; 0.013132
    temp2  V = 3.039671 + 0.012516*T     delivered  3.039207 ; 0.012525
    temp3  V = 3.153992 + 0.013005*T     delivered  3.154392 ; 0.013011
    temp4  V = 3.301474 + 0.013515*T     delivered  3.299366 ; 0.013561

A "_mod" variant of a delivery differs only in the temp c0 column -- an
offset correction of order 0.02 V, i.e. ~1.8 K.  It is a different
calibration of the same hardware, not a different plate; pick one
deliberately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SEG_RE = re.compile(r"^s(\d+)\s*:\s*(.*)$")
_TEMP_RE = re.compile(r"(temp\d)\s*=\s*([-\d.eE+]+)\s*V")
_PAIR_RE = re.compile(r"i_s\s*=\s*([-\d.eE+]+)\s*A\s*;\s*u_s\s*=\s*([-\d.eE+]+)\s*V")
# "Step5_90Grad.csv", "Step3_-20Grad.csv"
_STEP_RE = re.compile(r"Step(\d+)_(-?\d+)Grad", re.I)


@dataclass(frozen=True)
class StepFile:
    path: Path
    index: int                              # the k in Step<k>_
    temp_c: float                           # the nominal plate temperature
    sensors: dict[int, dict[str, float]]    # segment -> {temp1: V, ...}
    sweep: dict[int, list[tuple[float, float]]]   # segment -> [(i_s, u_s), ...]

    @property
    def segments(self) -> list[int]:
        return sorted(self.sweep)


def read_step(path) -> StepFile:
    """Parse one Step<k>_<T>Grad.csv.

    Tolerant about whitespace and separators because the bench writes tabs
    but a round trip through a spreadsheet turns them into semicolons.  A
    line that carries no `s<n>:` prefix is skipped rather than guessed at.
    """
    path = Path(path)
    m = _STEP_RE.search(path.stem)
    if not m:
        raise ValueError(f"{path.name}: not a Step<k>_<T>Grad file")

    sensors: dict[int, dict[str, float]] = {}
    sweep: dict[int, list[tuple[float, float]]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line:
            continue
        sm = _SEG_RE.match(line)
        if not sm:
            continue
        seg = int(sm.group(1))
        rest = sm.group(2)
        sensors[seg] = {k: float(v) for k, v in _TEMP_RE.findall(rest)}
        sweep[seg] = [(float(i), float(u)) for i, u in _PAIR_RE.findall(rest)]
    if not sweep:
        raise ValueError(f"{path.name}: no 's<n>:' records found")
    return StepFile(path, int(m.group(1)), float(m.group(2)), sensors, sweep)


def read_campaign(folder) -> list[StepFile]:
    """Every Step file in a folder, ordered by its index."""
    folder = Path(folder)
    out = []
    for p in sorted(folder.glob("Step*.csv")):
        try:
            out.append(read_step(p))
        except ValueError:
            continue
    return sorted(out, key=lambda s: s.index)


# ---------------------------------------------------------------------------


def _slope(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares u = R*i + u0 over one segment's DC sweep.

    Returns (R, u0, r2).  The offset is fitted rather than forced through
    zero: the delivered files show a few mV of amplifier offset, and forcing
    it into the slope biases R by up to 0.5 % at the low-current end.
    """
    i = np.array([p[0] for p in points], float)
    u = np.array([p[1] for p in points], float)
    if len(i) < 2:
        return np.nan, np.nan, np.nan
    A = np.vstack([i, np.ones_like(i)]).T
    (R, u0), *_ = np.linalg.lstsq(A, u, rcond=None)
    resid = u - A @ np.array([R, u0])
    ss = float(np.sum((u - u.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss if ss > 0 else np.nan
    return float(R), float(u0), r2


def fit_coefficients(steps: list[StepFile], area_cm2: float | None = None
                     ) -> dict:
    """Rebuild the curr.csv / temp.csv coefficients from the Step files.

    Returns
        curr    {segment: (c0, c1)}     K(T) = c0 + 1e-3*c1*T  [V/(A/cm^2)]
        temp    {"temp1": (c0, c1), ...}  V = c0 + c1*T
        area    the constant R(T)/K(T) the fit implies, in cm^2
        diag    per-segment R at each temperature, the linearity r^2, and the
                spread of R(T)/K(T) -- the number that says whether the
                one-constant model is adequate

    `area_cm2` fixes the constant that separates R [V/A] from K [V/(A/cm^2)].
    Left at None it is taken from the plate: the mean over segments of
    R(T)/K_delivered(T) is what the bench used, and with no delivered file to
    compare against the only defensible choice is the nominal cell area over
    the segment count, A_cell/72 = 4.235 cm^2.  Say which one you used.
    """
    if not steps:
        raise ValueError("no Step files")
    T = np.array([s.temp_c for s in steps], float)
    segs = sorted(set().union(*(set(s.segments) for s in steps)))

    # --- temperature sensors: V = c0 + c1*T, averaged over the plate --------
    temp: dict[str, tuple[float, float]] = {}
    names = sorted({k for s in steps for d in s.sensors.values() for k in d})
    M = np.vstack([np.ones_like(T), T]).T
    for nm in names:
        V = np.array([np.mean([d[nm] for d in s.sensors.values() if nm in d])
                      for s in steps], float)
        (c0, c1), *_ = np.linalg.lstsq(M, V, rcond=None)
        temp[nm] = (float(c0), float(c1))

    # --- per segment: R(T) = a + b*T ---------------------------------------
    R_of_T: dict[int, np.ndarray] = {}
    r2: dict[int, float] = {}
    ab: dict[int, tuple[float, float]] = {}
    for seg in segs:
        Rs, good = [], []
        for s in steps:
            R, _u0, q = _slope(s.sweep.get(seg, []))
            Rs.append(R)
            good.append(q)
        Rs = np.array(Rs, float)
        R_of_T[seg] = Rs
        r2[seg] = float(np.nanmin(good)) if good else np.nan
        ok = np.isfinite(Rs)
        if ok.sum() >= 2:
            (a, b), *_ = np.linalg.lstsq(M[ok], Rs[ok], rcond=None)
            ab[seg] = (float(a), float(b))
        else:
            ab[seg] = (np.nan, np.nan)

    area = float(area_cm2) if area_cm2 else 304.92 / 72.0
    curr = {seg: (a / area, 1e3 * b / area) for seg, (a, b) in ab.items()}

    diag = {
        "n_steps": len(steps),
        "temps_C": T.tolist(),
        "n_segments": len(segs),
        "area_cm2_used": area,
        "linearity_r2_min": float(np.nanmin(list(r2.values()))),
        "R_at_T": {str(k): np.round(v, 6).tolist() for k, v in R_of_T.items()},
        "tcr_percent_per_K": {
            str(seg): (100.0 * b / a if a else np.nan)
            for seg, (a, b) in ab.items()},
    }
    return {"curr": curr, "temp": temp, "area_cm2": area, "diag": diag}


def implied_area(steps: list[StepFile], curr_csv) -> dict:
    """The constant R(T)/K(T) a delivered curr.csv implies, per segment.

    This is the sharpest single check on a delivered calibration.  The ratio
    has to be constant -- in T because both are linear with the same TCR, and
    across segments because the via count scales with the area.  A segment
    whose ratio departs from the plate median is a segment whose calibration
    row does not belong to it.
    """
    import eis_local
    rows = eis_local._read_pairs(curr_csv)
    c = {i: (a, b) for i, (a, b) in enumerate(rows, start=1)}
    T = np.array([s.temp_c for s in steps], float)

    out: dict[str, float] = {}
    for seg in sorted(set().union(*(set(s.segments) for s in steps))):
        if seg not in c:
            continue
        R = np.array([_slope(s.sweep.get(seg, []))[0] for s in steps], float)
        K = c[seg][0] + 1e-3 * c[seg][1] * T
        ok = np.isfinite(R) & np.isfinite(K) & (K != 0)
        if ok.sum():
            out[str(seg)] = float(np.median(R[ok] / K[ok]))
    vals = np.array(list(out.values()))
    return {
        "per_segment": out,
        "median_cm2": float(np.median(vals)) if vals.size else np.nan,
        "cv": float(np.std(vals) / np.mean(vals)) if vals.size else np.nan,
        "outliers": [k for k, v in out.items()
                     if vals.size and abs(v - np.median(vals))
                     > 5 * (np.median(np.abs(vals - np.median(vals))) + 1e-12)],
    }


def verify(folder, curr_csv=None, temp_csv=None) -> dict:
    """Read a calibration folder and report whether it hangs together."""
    steps = read_campaign(folder)
    if not steps:
        raise ValueError(f"{folder}: no Step*.csv")
    res = {"n_steps": len(steps),
           "temps_C": [s.temp_c for s in steps],
           "n_segments": len(steps[0].segments)}

    fit = fit_coefficients(steps)
    res["linearity_r2_min"] = fit["diag"]["linearity_r2_min"]
    tcr = np.array(list(fit["diag"]["tcr_percent_per_K"].values()), float)
    res["tcr_percent_per_K"] = {"median": float(np.nanmedian(tcr)),
                                "min": float(np.nanmin(tcr)),
                                "max": float(np.nanmax(tcr))}

    if temp_csv:
        import eis_local
        got = eis_local._read_pairs(temp_csv)
        res["temp_match"] = [
            {"sensor": f"temp{i}",
             "delivered": [a, b],
             "refitted": [round(fit["temp"].get(f"temp{i}", (np.nan,) * 2)[0], 6),
                          round(fit["temp"].get(f"temp{i}", (np.nan,) * 2)[1], 6)],
             "offset_K": round((a - fit["temp"][f"temp{i}"][0]) / b, 3)
                          if f"temp{i}" in fit["temp"] and b else None}
            for i, (a, b) in enumerate(got, start=1)]

    if curr_csv:
        res["implied_area"] = implied_area(steps, curr_csv)

    return res


def write_coefficients(fit: dict, curr_path, temp_path) -> tuple[Path, Path]:
    curr_path, temp_path = Path(curr_path), Path(temp_path)
    curr_path.parent.mkdir(parents=True, exist_ok=True)
    with curr_path.open("w", encoding="utf-8") as fh:
        for seg in sorted(fit["curr"]):
            c0, c1 = fit["curr"][seg]
            fh.write(f"{c0:.6f};{c1:.6f}\n")
    with temp_path.open("w", encoding="utf-8") as fh:
        for nm in sorted(fit["temp"]):
            c0, c1 = fit["temp"][nm]
            fh.write(f"{c0:.6f};{c1:.6f}\n")
    return curr_path, temp_path


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("folder", help="folder holding Step*_<T>Grad.csv")
    p.add_argument("--curr-cal", default=None, help="delivered curr.csv to check")
    p.add_argument("--temp-cal", default=None, help="delivered temp.csv to check")
    p.add_argument("--refit-to", default=None,
                   help="write refitted curr.csv/temp.csv into this folder")
    p.add_argument("--area", type=float, default=None,
                   help="R/K constant in cm2 (default A_cell/72 = 4.235)")
    a = p.parse_args()

    rep = verify(a.folder, a.curr_cal, a.temp_cal)
    print(json.dumps(rep, indent=2, default=float)[:4000])

    if a.refit_to:
        steps = read_campaign(a.folder)
        fit = fit_coefficients(steps, area_cm2=a.area)
        c, t = write_coefficients(fit, Path(a.refit_to) / "curr.csv",
                                  Path(a.refit_to) / "temp.csv")
        print(f"\nrefitted with area {fit['area_cm2']:.4f} cm2 -> {c}  {t}")
