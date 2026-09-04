#!/usr/bin/env python3
"""
diagnose_scatter.py
===================
Why does a spectrum scatter?  Reads the bronze and silver tables of a
finished run and reports, per frequency, what the data actually looked like
and which gate removed what.

    python diagnose_scatter.py /tmp/eis_results/2611976/450A

Use it before changing any threshold.  The three failure modes below need
opposite responses, and they are indistinguishable from a Nyquist plot:

  A. NOISE            SNR collapses, sigma_rel > 0.35, |Z| runs away.
                      -> a gate problem.  Tighten sigma_rel_max.  Note that
                         acquisition skew CANNOT cause this: a delay is
                         all-pass and leaves |Z| untouched, so a rising |Z|
                         is never a timing problem.

  B. TOO FEW CYCLES   the dwell holds a fraction of a period, usually at the
                      low-frequency end.  -> raise min_cycles_per_dwell, or
                      lengthen the dwells in the next campaign.

  C. GENUINE SPREAD   SNR fine, sigma_rel small, points still scattered.
                      -> not a processing problem.  Look at the cell.
"""
from __future__ import annotations
import sys, csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def load(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(r, k, d=np.nan):
    try:
        v = r.get(k, "")
        return float(v) if v not in ("", None) else d
    except (TypeError, ValueError):
        return d


def main(root: str) -> int:
    root = Path(root)
    braw = root / "bronze" / "raw_spectra.csv"
    smeta = root / "silver" / "segments_summary.csv"
    if not braw.exists():
        print(f"not found: {braw}")
        return 2
    rows = load(braw)

    by_f = defaultdict(list)
    for r in rows:
        by_f[round(num(r, "freq_hz"), 4)].append(r)

    print(f"\n  {len(rows)} raw points over {len(by_f)} frequencies\n")
    print(f"{'f [Hz]':>10} {'segs':>5} {'SNR med':>8} {'SNR min':>8} "
          f"{'N':>7} {'cycles':>7} {'|Z| med':>9} {'|Z| max':>9} {'grid':>5}")
    print("  " + "-" * 76)
    flagged = []
    for f in sorted(by_f):
        g = by_f[f]
        snr = np.array([num(r, "snr_comb_db") for r in g])
        n = np.array([num(r, "n_samples") for r in g])
        zr = np.array([num(r, "z_re_ohm_cm2") for r in g])
        zi = np.array([num(r, "z_im_ohm_cm2") for r in g])
        zm = np.abs(zr + 1j * zi) * 1000.0
        cyc = np.nanmedian(n) / 10000.0 * f      # fs assumed 10 kHz for display
        og = sum(int(num(r, "on_grid", 0)) for r in g)
        zmed = np.nanmedian(zm)
        zmax = np.nanmax(zm) if len(zm) else np.nan
        mark = ""
        if np.nanmedian(snr) < 3:
            mark += " LOW-SNR"
        if cyc < 3:
            mark += " SHORT-DWELL"
        if np.isfinite(zmax) and np.isfinite(zmed) and zmed > 0 and zmax > 5 * zmed:
            mark += " RUNAWAY-|Z|"
        if mark:
            flagged.append((f, mark))
        print(f"{f:10.3f} {len(g):5d} {np.nanmedian(snr):8.1f} "
              f"{np.nanmin(snr):8.1f} {np.nanmedian(n):7.0f} {cyc:7.1f} "
              f"{zmed:9.1f} {zmax:9.1f} {og:5d}{mark}")

    if flagged:
        print("\n  frequencies needing attention:")
        for f, m in flagged:
            print(f"    {f:9.3f} Hz {m}")
        print("\n  LOW-SNR + RUNAWAY-|Z| together  -> failure mode A (noise).")
        print("  SHORT-DWELL at the bottom of the band -> failure mode B.")
        print("  RUNAWAY-|Z| with good SNR -> mode C, look at the cell.")
    else:
        print("\n  no frequency flagged: scatter is not explained by SNR, "
              "dwell length or |Z| runaway.")

    if smeta.exists():
        srows = load(smeta)
        print(f"\n  per-segment gate activity ({len(srows)} segments)")
        cnt = defaultdict(int)
        for r in srows:
            for fl in (r.get("flags", "") or "").split(";"):
                if fl:
                    cnt[fl.split("_")[0] + "_" + fl.split("_")[-1]
                        if fl.startswith("dropped") else fl] += 1
        for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
            print(f"    {v:4d}  {k}")
        used = np.array([num(r, "n_used") for r in srows])
        drop = np.array([num(r, "n_dropped") for r in srows])
        print(f"    points kept per segment: median {np.nanmedian(used):.0f}, "
              f"dropped: median {np.nanmedian(drop):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "./results"))
