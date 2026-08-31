#!/usr/bin/env python3
r"""
compare_conditions.py
=====================
Put two operating points of the same plate side by side, along the gas
paths that actually run across it.

    python compare_conditions.py --results <root>/2612025 --a 45A --b 450A
    python compare_conditions.py --results <root>/2612025 --a 45A --b 450A \
                                 --json compare.json --csv compare.csv

WHAT THE COMPARISON IS FOR
--------------------------
At 45 A the plate is close to dry: the product water is a small perturbation
on the gas that was fed in.  At 450 A it is an order of magnitude larger, and
the cathode goes from "humidified gas with a little product water" to "gas
carrying more water than it can hold".  That transition is the physics the
map is there to show, and it is local: it happens at the OXYGEN OUTLET first
and works backwards up the channel.

Which is why this reports along the gas paths rather than along x.  The
ports on this plate are at the four corners --

    top-left  O2 out          H2 out  top-right
    bottom-left  H2 in         O2 in  bottom-right

-- so hydrogen crosses the plate bottom-left to top-right while oxygen
crosses it bottom-right to top-left.  The two run in opposite directions.
A profile plotted against x is therefore right for one gas and mirrored for
the other, and the corner where the flooding starts -- top-left, the oxygen
outlet -- ends up drawn at the dry end.

WHAT TO LOOK AT
---------------
`R_mt` (mass transport) is the quantity that should move.  Liquid water in
the pores blocks the gas, which raises the low-frequency arc; R_ohmic moves
much less, and mostly the other way, because a wetter membrane conducts
better.  So the signature of flooding is a LARGE R_mt ratio concentrated at
the oxygen outlet, with R_ohmic flat or slightly lower.  If instead R_mt
rises uniformly over the whole plate, that is not flooding -- that is the
whole cell being pushed further up its polarisation curve.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plate_conditions as pc                               # noqa: E402
import r2d2_geometry as geom                                # noqa: E402

#: Quantities worth comparing, and whether a RATIO or a DIFFERENCE reads
#: better.  A resistance is compared as a ratio because the interesting
#: statement is "twice as much"; a temperature as a difference because
#: "eight degrees hotter" is the statement there.
QUANTITIES = {
    "R_ohmic": "ratio",
    "R_ct": "ratio",
    "R_mt": "ratio",
    "R_pol": "ratio",
    "j_dc": "ratio",
    "T_degC": "difference",
}


def load_plate_summary(path: Path) -> dict[str, dict[str, float]]:
    """One condition's per-segment scalars, keyed by segment."""
    if path.is_dir():
        candidates = [path / "gold" / "plate_summary.csv",
                      path / "plate_summary.csv"]
        path = next((c for c in candidates if c.is_file()), candidates[0])
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- run the pipeline for this condition first")
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            seg = (row.get("segment") or "").strip()
            if not seg:
                continue
            vals: dict[str, float] = {}
            for k, v in row.items():
                try:
                    vals[k] = float(v)
                except (TypeError, ValueError):
                    vals[k] = float("nan")
            vals["_class"] = row.get("class", "")
            out[seg] = vals
    return out


def compare(a: dict[str, dict], b: dict[str, dict],
            label_a: str, label_b: str,
            centroids: dict[str, tuple[float, float]] | None = None,
            ports: tuple[pc.Port, ...] = pc.DEFAULT_PORTS) -> dict:
    """Per-segment comparison, with both gas coordinates attached."""
    centroids = centroids or geom.centroids()
    shared = sorted(set(a) & set(b), key=lambda s: int(s) if s.isdigit() else 0)
    if not shared:
        raise ValueError("the two conditions share no segments")

    xi = {gas: pc.flow_coordinate(centroids, gas, ports)
          for gas in sorted({p.gas for p in ports})}

    rows = []
    for seg in shared:
        row: dict[str, object] = {"segment": seg}
        for gas, coord in xi.items():
            row[f"xi_{gas}"] = float(coord.get(seg, float("nan")))
        cx, cy = centroids.get(seg, (float("nan"),) * 2)
        row["cx_mm"], row["cy_mm"] = float(cx), float(cy)
        for q, how in QUANTITIES.items():
            va, vb = a[seg].get(q, np.nan), b[seg].get(q, np.nan)
            row[f"{q}_{label_a}"] = float(va)
            row[f"{q}_{label_b}"] = float(vb)
            if how == "ratio":
                row[f"{q}_ratio"] = (float(vb / va)
                                     if np.isfinite(va) and np.isfinite(vb)
                                     and va != 0 else float("nan"))
            else:
                row[f"{q}_delta"] = (float(vb - va)
                                     if np.isfinite(va) and np.isfinite(vb)
                                     else float("nan"))
        rows.append(row)
    return {"label_a": label_a, "label_b": label_b,
            "n_segments": len(rows), "rows": rows,
            "ports": pc.port_layout(centroids, ports)}


def along_path(result: dict, quantity: str, gas: str = "O2",
               n_bins: int = 8) -> list[dict]:
    """Bin the comparison along one gas path, inlet to outlet."""
    key = (f"{quantity}_ratio" if QUANTITIES.get(quantity) == "ratio"
           else f"{quantity}_delta")
    xs = np.array([r[f"xi_{gas}"] for r in result["rows"]], float)
    ys = np.array([r.get(key, np.nan) for r in result["rows"]], float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = ok & (xs >= lo) & (xs <= hi if hi == 1.0 else xs < hi)
        if not sel.any():
            continue
        out.append({"xi_lo": float(lo), "xi_hi": float(hi),
                    "n": int(sel.sum()),
                    "median": float(np.median(ys[sel])),
                    "min": float(np.min(ys[sel])),
                    "max": float(np.max(ys[sel]))})
    return out


def flooding_verdict(result: dict, gas: str = "O2") -> dict:
    """Is the extra mass-transport resistance concentrated at the outlet?

    The distinction this draws is the one the comparison exists for.  Every
    resistance rises with current, so "R_mt went up" on its own says only
    that the cell is working harder.  Liquid water is a LOCAL effect: it
    accumulates where the gas has already collected the most vapour, which
    is the outlet end, so its signature is a gradient in the ratio along the
    oxygen path -- not a uniform lift.
    """
    xs = np.array([r[f"xi_{gas}"] for r in result["rows"]], float)
    ys = np.array([r.get("R_mt_ratio", np.nan) for r in result["rows"]], float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 6:
        return {"ok": False, "note": "too few segments carry R_mt"}
    xs, ys = xs[ok], ys[ok]
    inlet = ys[xs <= 0.33]
    outlet = ys[xs >= 0.67]
    if inlet.size < 2 or outlet.size < 2:
        return {"ok": False, "note": "no segments at one end of the path"}
    med_in, med_out = float(np.median(inlet)), float(np.median(outlet))
    # Spearman-style rank correlation: monotone, and immune to the handful
    # of very large ratios a nearly-zero denominator produces.
    rank_x = np.argsort(np.argsort(xs)).astype(float)
    rank_y = np.argsort(np.argsort(ys)).astype(float)
    r = float(np.corrcoef(rank_x, rank_y)[0, 1])
    gradient = med_out / med_in if med_in > 0 else float("nan")
    return {
        "ok": True, "gas": gas,
        "median_ratio_inlet_third": med_in,
        "median_ratio_outlet_third": med_out,
        "outlet_over_inlet": gradient,
        "rank_correlation_with_path": r,
        "reads_as": ("water accumulating towards the outlet"
                     if np.isfinite(gradient) and gradient > 1.2 and r > 0.3
                     else "a uniform rise -- the whole cell working harder, "
                          "not local flooding"),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True,
                   help="the order id folder holding one sub-folder per "
                        "condition")
    p.add_argument("--a", default="45A", help="the reference condition")
    p.add_argument("--b", default="450A", help="the condition to compare")
    p.add_argument("--gas", default="O2", choices=["O2", "H2"],
                   help="which path to profile along (default O2: water is "
                        "produced at the cathode)")
    p.add_argument("--plate", choices=["gen1", "gen2"], default=None)
    p.add_argument("--json", type=Path)
    p.add_argument("--csv", type=Path)
    a = p.parse_args(argv)

    if a.plate:
        geom.use_plate({"gen1": "gen1_r2d2_72",
                        "gen2": "gen2_r2d2_naboo_72"}[a.plate])

    left = load_plate_summary(a.results / a.a)
    right = load_plate_summary(a.results / a.b)
    result = compare(left, right, a.a, a.b)

    print(f"\n{a.a} vs {a.b}: {result['n_segments']} shared segments\n")
    print("port map:")
    for port in result["ports"]["ports"]:
        print(f"  {port['corner']:<13} {port['gas']} {port['role']}")

    for quantity in ("R_ohmic", "R_mt", "R_ct"):
        bins = along_path(result, quantity, a.gas)
        if not bins:
            continue
        print(f"\n{quantity} ratio ({a.b}/{a.a}) along the {a.gas} path, "
              f"inlet -> outlet:")
        for b_ in bins:
            bar = "#" * int(np.clip((b_["median"] - 1.0) * 20, 0, 40))
            print(f"  xi {b_['xi_lo']:.2f}-{b_['xi_hi']:.2f}  n={b_['n']:2d}  "
                  f"median {b_['median']:6.2f}  {bar}")

    verdict = flooding_verdict(result, a.gas)
    print("\nverdict:")
    if verdict.get("ok"):
        print(f"  R_mt ratio  inlet third {verdict['median_ratio_inlet_third']:.2f}"
              f"   outlet third {verdict['median_ratio_outlet_third']:.2f}"
              f"   ({verdict['outlet_over_inlet']:.2f}x)")
        print(f"  rank correlation with the {a.gas} path: "
              f"{verdict['rank_correlation_with_path']:+.2f}")
        print(f"  reads as: {verdict['reads_as']}")
    else:
        print(f"  {verdict.get('note')}")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({**result, "verdict": verdict}, indent=2),
                          encoding="utf-8")
        print(f"\n  written: {a.json}")
    if a.csv:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with a.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(result["rows"][0]))
            w.writeheader()
            w.writerows(result["rows"])
        print(f"  written: {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
