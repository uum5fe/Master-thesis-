#!/usr/bin/env python3
"""
dc_closure.py
=============
Do the segment currents add up to the bench current -- at every operating
point -- and is what is missing a SCALE (which sits in every impedance too)
or an OFFSET (which does not)?

    python dc_closure.py RUN_45A RUN_60A RUN_150A RUN_450A -o out/
        [--bench bench.mf4 --gamry GAMRY_DIR]

Per run folder (one per condition): the plate current from bronze's
segment_meta.csv (sum j_dc * area over the measured segments, scaled by area
to the whole plate), against the bench current at the Gamry sweep of that
condition -- from the run's own manifest, else from --bench/--gamry, else
the current in the condition name. Then I_meas = a * I_bench + b over the
conditions:

  a - 1   the current-scale error. The same K converts the AC segment
          signal, so every impedance is off by 1/a as well.
  b       a zero offset of the current chains (amplifier offset, shunt
          thermo-EMF). It moves j_dc but not the impedance.

One condition cannot separate the two; two can; four can also say whether
a straight line is the right model (the residual).

With --gamry it also compares the VOLTAGE: each card's UC2 DC level against
the Gamry's own Vdc at the same condition. Two sense points that differ by a
resistance R_x read  UC2 - Vdc = -I * R_x . On 2612030 that is a straight
line through 45..450 A with 0.1 mV residual and R_x = 44 uOhm = 13.4
mOhm*cm2 over the plate: every segment's Z carries that much more series
resistance than the Gamry's. cfg.uc_series_mohm_cm2 subtracts it.

Per card it also writes the card's mean current density against the plate's
at each condition. A K error on a card is the same percentage at 45 A and at
450 A; a real regional difference changes with load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import card_gain


def _rows(path: Path) -> list[dict]:
    import csv
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def find_runs(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if (p / "bronze" / "segment_meta.csv").is_file():
            out.append(p)
        else:
            out += sorted(q.parent.parent for q in
                          p.rglob("bronze/segment_meta.csv"))
    seen, uniq = set(), []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            uniq.append(p)
    return uniq


def bench_current(run: Path, bench_log=None, gamry_dir=None,
                  cfg: dict | None = None) -> tuple[float, str]:
    """(current, source) for this run's condition."""
    cond = (cfg or {}).get("condition", "") or run.name
    if (cfg or {}).get("i_setpoint_a"):
        return float(cfg["i_setpoint_a"]), "given setpoint"
    man = run / "run_manifest.json"
    if man.is_file():
        try:
            g = json.loads(man.read_text()).get("stages", {}).get("gamry") or []
            for c in g:
                for k, lab in (("bench_I_S", "bench I_S (measured)"),
                               ("bench_I_S_set", "bench I_S_set")):
                    v = _f(c.get(k))
                    if np.isfinite(v) and v > 0:
                        return v, lab
        except Exception:                                   # noqa: BLE001
            pass
    if bench_log and gamry_dir:
        try:
            import gamry_compare as gc
            b = gc.read_bench_log(bench_log)
            for s in gc.find_cell_sweeps(gamry_dir):
                if s.condition.upper() == str(cond).upper() and s.started:
                    st = b.state_at(s.started)
                    for k, lab in (("I_S", "bench I_S (measured)"),
                                   ("I_S_set", "bench I_S_set")):
                        v = _f(st.get(k))
                        if np.isfinite(v) and v > 0:
                            return v, lab
        except Exception:                                   # noqa: BLE001
            pass
    c = card_gain.current_of(cond)
    return (c, "condition name") if np.isfinite(c) else (float("nan"), "")


def uc2_dc(run: Path) -> float:
    """Median over the cards of the UC2 DC level (bronze/card_reference.csv)."""
    v = [_f(r.get("dc_V")) for r in _rows(run / "bronze" / "card_reference.csv")
         if r.get("channel") == "UC2"]
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def gamry_vdc(gamry_dir, condition: str) -> float:
    if not gamry_dir:
        return float("nan")
    try:
        import gamry_compare as gc
        import gamry_sync as gs
        for s in gc.find_cell_sweeps(gamry_dir):
            if s.condition.upper() == str(condition).upper():
                return gs.read_vdc(s.path)
    except Exception:                                       # noqa: BLE001
        pass
    return float("nan")


def sense_offset(rows: list[dict], plate_key: str = "gen1") -> dict:
    """Fit UC2 - Vdc = b - I * R_x over the conditions."""
    import r2d2_geometry
    pts = [(r["i_ref_A"], r["uc2_V"] - r["gamry_vdc_V"]) for r in rows
           if np.isfinite(r.get("uc2_V", np.nan))
           and np.isfinite(r.get("gamry_vdc_V", np.nan))
           and np.isfinite(r.get("i_ref_A", np.nan))]
    if len(pts) < 2:
        return {"n": len(pts)}
    x, y = np.array(pts).T
    a, b = np.polyfit(x, y, 1)
    area = sum(s.area_cm2 for s in r2d2_geometry.plate(plate_key).segments.values())
    return {"n": len(pts), "R_x_uohm": float(-1e6 * a),
            "R_x_mohm_cm2": float(-1e3 * a * area), "offset_mV": float(1e3 * b),
            "resid_mV": float(1e3 * np.std(y - (a * x + b)))}


def analyse(runs, bench_log=None, gamry_dir=None, plate_key=None) -> dict:
    rows, cards = [], []
    for run in runs:
        cj = run / "config_used.json"
        cfg = json.loads(cj.read_text()) if cj.is_file() else {}
        cond = cfg.get("condition") or run.name
        pk = plate_key or cfg.get("plate", "gen1")
        meta = _rows(run / "bronze" / "segment_meta.csv")
        j = {r["segment"]: _f(r.get("j_dc_A_cm2")) for r in meta}
        a = {r["segment"]: _f(r.get("area_cm2")) for r in meta}
        i_meas, a_meas, i_full = card_gain.plate_current(j, a, pk)
        i_ref, src = bench_current(run, bench_log, gamry_dir, cfg)
        rows.append({"run": str(run), "leepa": cfg.get("leepa", ""),
                     "condition": cond, "n_segments": len(meta),
                     "area_measured_cm2": round(a_meas, 2),
                     "i_measured_A": round(i_meas, 3),
                     "i_full_A": round(i_full, 3),
                     "i_ref_A": i_ref, "i_ref_source": src,
                     "uc2_V": uc2_dc(run),
                     "gamry_vdc_V": gamry_vdc(gamry_dir, cond)})
        jp = i_meas / a_meas if a_meas else float("nan")
        by: dict[str, list] = {}
        for r in meta:
            by.setdefault(card_gain.short(r.get("card", "")), []).append(r)
        for c, rs in sorted(by.items()):
            ic = sum(_f(r["j_dc_A_cm2"]) * _f(r["area_cm2"]) for r in rs)
            ac = sum(_f(r["area_cm2"]) for r in rs)
            cards.append({"condition": cond, "leepa": cfg.get("leepa", ""),
                          "card": c, "n_segments": len(rs),
                          "area_cm2": round(ac, 2),
                          "j_mean_A_cm2": round(ic / ac, 5) if ac else "",
                          "j_vs_plate_pct": (round(100 * (ic / ac / jp - 1), 2)
                                             if ac and jp else "")})
    fit = card_gain.dc_closure_table(rows)
    fit["cards"] = cards
    fit["sense"] = sense_offset(rows, plate_key or "gen1")
    return fit


def plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in res["rows"] if np.isfinite(r["i_ref_A"])]
    if not rows:
        return
    x = np.array([r["i_ref_A"] for r in rows])
    y = np.array([r["i_full_A"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(x, y, "o", color="#1f5fa8")
    lim = [0, 1.05 * max(x.max(), y.max())]
    ax[0].plot(lim, lim, "k--", lw=0.8, label="closure")
    if "scale" in res:
        xx = np.linspace(*lim, 50)
        ax[0].plot(xx, res["scale"] * xx + res["offset_A"], color="#c0392b",
                   lw=1, label=f"fit: {res['scale']:.4f} I {res['offset_A']:+.2f} A")
    for r in rows:
        ax[0].annotate(r["condition"], (r["i_ref_A"], r["i_full_A"]),
                       textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax[0].set_xlabel("bench current [A]")
    ax[0].set_ylabel("sum of segment currents, plate-scaled [A]")
    ax[0].legend(fontsize=8)
    ax[1].plot(x, [r["dev_pct"] for r in rows], "o-", color="#1f5fa8")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("bench current [A]")
    ax[1].set_ylabel("closure deviation [%]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("DC current closure across conditions")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def print_sense(res: dict) -> None:
    se = res.get("sense") or {}
    if "R_x_uohm" not in se:
        return
    print(f"\n  UC2 DC vs the Gamry's Vdc over {se['n']} condition(s): "
          f"UC2 - Vdc = {se['offset_mV']:+.1f} mV - I x {se['R_x_uohm']:.1f} "
          f"uOhm (residual {se['resid_mV']:.1f} mV)")
    print(f"  -> every segment's Z carries {se['R_x_mohm_cm2']:+.1f} mOhm*cm2 "
          f"of series resistance the Gamry does not see; set "
          f"uc_series_mohm_cm2 = {se['R_x_mohm_cm2']:.1f} to put the maps on "
          f"the Gamry's reference plane")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path,
                    help="run folders (or folders holding them)")
    ap.add_argument("-o", "--out", type=Path, default=Path("dc_closure"))
    ap.add_argument("--bench", type=Path, help="bench .mf4")
    ap.add_argument("--gamry", type=Path, help="folder with the HFR .dta sweeps")
    a = ap.parse_args(argv)
    runs = find_runs(a.runs)
    if not runs:
        print("no run folders (bronze/segment_meta.csv) found")
        return 2
    res = analyse(runs, a.bench, a.gamry)
    a.out.mkdir(parents=True, exist_ok=True)
    import utils
    utils.write_table(a.out / "dc_closure.csv", res["rows"])
    utils.write_table(a.out / "dc_closure_cards.csv", res["cards"])
    plot(res, a.out / "dc_closure.png")
    print(f"\n  {'condition':>9} {'I_ref [A]':>10} {'I_plate [A]':>12} "
          f"{'dev':>7}  source")
    for r in res["rows"]:
        print(f"  {r['condition']:>9} {r['i_ref_A']:10.2f} {r['i_full_A']:12.2f} "
              f"{r.get('dev_pct', float('nan')):+6.1f} %  {r['i_ref_source']}")
    if "scale" in res:
        print(f"\n  fit over {res['n']} condition(s): I_plate = "
              f"{res['scale']:.4f} * I_bench {res['offset_A']:+.2f} A  "
              f"(scale {res['scale_pct']:+.2f} %, residual "
              f"{res['resid_A']:.2f} A)")
        print(f"  -> every impedance carries {-res['scale_pct']:+.1f} % from "
              f"the current scale; the offset does not reach the impedance")
    print_sense(res)
    print(f"\n  written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
