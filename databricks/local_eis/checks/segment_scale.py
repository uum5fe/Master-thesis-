#!/usr/bin/env python3
"""
segment_scale.py
================
Is a segment's impedance off by a constant FACTOR -- the same at every
frequency and every load -- rather than by anything the cell does?

WHY
---
After the chain lag and the card checks, 2612030 still showed neighbouring
segments 5-10 mOhm*cm2 apart in HFR. Looked at across the four conditions
(45 / 60 / 150 / 450 A) that difference is:

  * a RATIO, not an offset: a segment 6 % high at 1 kHz is 6 % high at 300,
    100 and 30 Hz too (its absolute excess grows with |Z|);
  * the same at every current (r = 0.80-0.95 between conditions);
  * spatially random, and unrelated to area, K, mux slot, edge or card.

A real local HFR difference is an offset (it does not grow with |Z|), and a
real kinetic or transport difference changes with frequency and with load.
A constant factor is what a scale error in the segment's current path looks
like: its K (curr.csv) in situ, or the area the segment really collects
current from. Divided out -- with the factor estimated on the OTHER three
conditions -- the plate spread of Re Z at 1 kHz fell from 6 % to 3 %.

WHAT IT DOES
------------
For each run (one per condition): |Z_seg / Z_plate-median| at the points
between `f_lo` and `f_hi`, log-averaged per segment, plus the same in the
lower and upper half of that band (flatness). Across runs: the median factor,
its spread between conditions, and a leave-one-condition-out check of how
much of each run's spread it explains. A segment is "scale_like" when its
factor is beyond `min_pct`, flat over the band and repeatable.

Writes segment_scale.csv and, on request, a gain file of the factors (a
constant G = k per segment, the pipeline divides Z by it): on its own for
re-evaluating finished runs, or multiplied into the chain file for new ones.
It is NOT applied by default: it would also remove any real structure that
happens to be a constant factor at every frequency and load, and the factor
should first be traced to its cause (swap two channels' cables: a factor that
moves with the channel is the electronics, one that stays is the segment).

    python segment_scale.py RUN_45A RUN_60A RUN_150A RUN_450A -o out/
        [--write-gain scale_gain.csv] [--chain chain_gain_gen1.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _runs(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "silver" / "spectra_clean.csv").is_file():
            out.append(p)
        else:
            out += sorted(q.parent.parent for q in
                          p.rglob("silver/spectra_clean.csv"))
    return out


def _label(run: Path) -> str:
    try:
        return json.loads((run / "config_used.json").read_text()
                          ).get("condition") or run.name
    except Exception:                                       # noqa: BLE001
        return run.name


def run_factors(run: Path, f_lo: float = 30.0, f_hi: float = 1000.0
                ) -> pd.DataFrame:
    """Per measured segment of one run: log factor over the band and its two
    halves (natural log of |Z_seg / plate median|)."""
    sc = pd.read_csv(run / "silver" / "spectra_clean.csv",
                     dtype={"segment": str})
    meta = run / "bronze" / "segment_meta.csv"
    if meta.is_file():
        keep = set(pd.read_csv(meta, dtype={"segment": str}).segment)
        sc = sc[sc.segment.isin(keep)]
    sc = sc[(sc.freq_hz >= f_lo) & (sc.freq_hz <= f_hi)].copy()
    sc["f"] = sc.freq_hz.round(2)
    sc["M"] = np.hypot(sc.z_re_mohm_cm2, sc.z_im_mohm_cm2)
    piv = sc.pivot_table(index="f", columns="segment", values="M",
                         aggfunc="first")
    need = max(5, int(np.ceil(0.3 * piv.shape[1])))
    piv = piv[piv.notna().sum(axis=1) >= need]
    lr = np.log(piv.div(piv.median(axis=1), axis=0))
    mid = np.sqrt(f_lo * f_hi)
    lo, hi = lr[lr.index < mid], lr[lr.index >= mid]
    return pd.DataFrame({"log_k": lr.median(), "log_k_lo": lo.median(),
                         "log_k_hi": hi.median(),
                         "n": lr.notna().sum()})


def analyse(runs, f_lo: float = 30.0, f_hi: float = 1000.0,
            min_pct: float = 3.0, flat_pct: float = 2.0,
            repeat_pct: float = 2.5) -> dict:
    runs = list(runs)
    per = {_label(r): run_factors(r, f_lo, f_hi) for r in runs}
    L = pd.DataFrame({c: d.log_k for c, d in per.items()})
    lo = pd.DataFrame({c: d.log_k_lo for c, d in per.items()}).median(axis=1)
    hi = pd.DataFrame({c: d.log_k_hi for c, d in per.items()}).median(axis=1)
    k = np.exp(L.median(axis=1))
    tab = pd.DataFrame({
        "scale": k.round(5),
        "scale_pct": (100 * (k - 1)).round(2),
        "flat_pct": (100 * (np.exp(hi) - np.exp(lo))).round(2),
        "between_conditions_sd_pct": (100 * L.std(axis=1)).round(2),
        "n_conditions": L.notna().sum(axis=1)})
    tab["verdict"] = np.where(
        (tab.scale_pct.abs() >= min_pct) & (tab.flat_pct.abs() <= flat_pct)
        & (tab.between_conditions_sd_pct.fillna(0) <= repeat_pct)
        & (tab.n_conditions >= min(2, L.shape[1])),
        "scale_like", np.where(tab.scale_pct.abs() < min_pct, "small",
                               "not_a_clean_scale"))
    tab.index.name = "segment"
    tab = tab.sort_index(key=lambda s: s.astype(int))

    # leave one condition out: the factor from the others, applied to this one
    loo = {}
    for c in L.columns:
        others = [o for o in L.columns if o != c]
        if not others:
            continue
        resid = L[c] - L[others].median(axis=1)
        loo[c] = {"spread_pct": float(100 * L[c].std()),
                  "after_pct": float(100 * resid.std())}
    rep = L.corr().round(2) if L.shape[1] > 1 else None
    return {"table": tab, "conditions": list(L.columns), "loo": loo,
            "repeatability": rep, "band": (f_lo, f_hi)}


def gain_rows(scale: pd.Series, chain: dict | None = None,
              f_min: float = 0.1, f_max: float = 1e4, n: int = 120):
    """Gain-file rows: G = k (times the chain response when given)."""
    import utils
    f = np.geomspace(f_min, f_max, n)
    rows = []
    for s, k in scale.items():
        if not np.isfinite(k):
            continue
        g = np.full(f.size, float(k), complex)
        if chain:
            g = g * utils.gain_at(chain, str(s), f)
        rows += [(str(s), fi, gi.real, gi.imag) for fi, gi in zip(f, g)]
    return rows


def write_gain(path, scale: pd.Series, chain_file=None, note: str = "") -> Path:
    import utils
    chain = utils.load_gain(chain_file) if chain_file else None
    rows = gain_rows(scale, chain)
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# segment scale factors from segment_scale.py"
                 + (f" x chain response {Path(chain_file).name}"
                    if chain_file else "") + (f"; {note}" if note else "")
                 + "\n")
        fh.write("segment,freq_hz,gain_real,gain_imag\n")
        for s, f, gr, gi in rows:
            fh.write(f"{s},{f:.6g},{gr:.8f},{gi:.8f}\n")
    return path


def report(res: dict, log=print) -> None:
    t = res["table"]
    f_lo, f_hi = res["band"]
    log(f"  per-segment |Z| factor over {f_lo:g}-{f_hi:g} Hz, "
        f"{len(res['conditions'])} condition(s): "
        f"sd {t.scale_pct.std():.1f} %, range "
        f"{t.scale_pct.min():+.1f} .. {t.scale_pct.max():+.1f} %")
    if res["repeatability"] is not None:
        r = res["repeatability"].values
        off = r[~np.eye(len(r), dtype=bool)]
        log(f"  same pattern at every current: r = {off.min():.2f} .. "
            f"{off.max():.2f} between conditions")
    for c, v in res["loo"].items():
        log(f"  {c}: plate spread {v['spread_pct']:.1f} % -> "
            f"{v['after_pct']:.1f} % with the factor from the other "
            f"conditions divided out")
    sl = t[t.verdict == "scale_like"]
    log(f"  {len(sl)} segment(s) scale-like (|k| >= 3 %, flat, repeatable): "
        + ", ".join(f"{s} {v:+.1f}%" for s, v in
                    sl.scale_pct.sort_values(key=abs, ascending=False)
                    .head(12).items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("segment_scale"))
    ap.add_argument("--f-lo", type=float, default=30.0)
    ap.add_argument("--f-hi", type=float, default=1000.0)
    ap.add_argument("--write-gain", type=Path,
                    help="write the factors as a gain file (not applied)")
    ap.add_argument("--chain", type=Path,
                    help="multiply the factors into this chain gain file")
    a = ap.parse_args(argv)
    runs = _runs(a.runs)
    if not runs:
        print("no run folders (silver/spectra_clean.csv) found")
        return 2
    res = analyse(runs, a.f_lo, a.f_hi)
    a.out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(a.out / "segment_scale.csv")
    report(res)
    if a.write_gain:
        p = write_gain(a.write_gain, res["table"].scale, a.chain,
                       note=", ".join(res["conditions"]))
        print(f"  gain file: {p} (NOT applied -- set it as GAIN_FILE, or "
              f"reevaluate(..., gain_file=...), to test it)")
    print(f"  written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
