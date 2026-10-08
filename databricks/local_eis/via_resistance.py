#!/usr/bin/env python3
"""
via_resistance.py
=================
The current of each segment is measured as the voltage drop over its vias.
How does that via resistance behave with temperature, card by card -- and
how much can it move the local current density?

WHERE THE NUMBERS COME FROM
---------------------------
The Abgleich delivery holds the DC calibration that curr.csv was fitted from:

    Step1_20Grad.csv ... Step5_90Grad.csv   plate at 20/40/60/80/90 degC
    Step6_20Grad.csv                         back at 20 degC (hysteresis)

Each line s<n> is one segment: the four plate sensors (temp1..temp4, in V,
turned into degC with coefficients/temp.csv) and five DC current steps
(i_s, 0..1 A) with the amplified via voltage u_s. The slope

    k_s(T) = d u_s / d i_s      [V/A]

is the via resistance times the amplifier gain of that segment's channel --
the gain is fixed, so k is proportional to R_via and its temperature
dependence is the via's. curr.csv carries the same thing as the transfer
coefficient K_s(T) = c0 + 1e-3 * c1 * T (k / K is one constant, 2.94, for
every segment on Kashyyyk), and the pipeline computes j = u / K_s(T_seg).

WHAT IT MEANS FOR THE CURRENT DENSITY
-------------------------------------
    j = u / K(T)   ->   dj / j = -alpha * dT,    alpha = (1/K) dK/dT

On Kashyyyk alpha = 0.386-0.393 %/K, i.e. copper (0.393 %/K). The pipeline
already evaluates K at each segment's own temperature (interpolated from
T1..T4), so a residual error needs the via to be at a different temperature
than that interpolation says: each kelvin is -0.39 % of j (and +0.39 % of
every impedance of that segment).

WHAT IT WRITES
--------------
    via_steps.csv          segment, card, step, T_degC, k_V_per_A, offset,
                           nonlinearity
    via_segments.csv       per segment: k at 20 degC, alpha measured and from
                           curr.csv, return drift after 90 degC, k at each
                           condition's operating temperature
    via_vs_temperature.png one panel per card: k(T) of each segment, the
                           operating temperatures of the runs marked
    via_alpha.png          alpha and the 20 degC level per segment, by card

    python via_resistance.py ABGLEICH_DIR [RUN_DIR ...] -o out/
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

COPPER_ALPHA_PCT = 0.393          # %/K at 20 degC


def _short(card) -> str:
    m = re.search(r"Karte_?(\d+)", str(card))
    return f"card {m.group(1)}" if m else str(card)


def _pairs(path) -> np.ndarray:
    rows = []
    for ln in Path(path).read_text(encoding="utf-8-sig").splitlines():
        p = [q for q in re.split(r"[;,\t ]+", ln.strip()) if q]
        if len(p) >= 2:
            try:
                rows.append((float(p[0]), float(p[1])))
            except ValueError:
                continue
    return np.asarray(rows, float)


def read_steps(abgleich_dir) -> pd.DataFrame:
    """One row per segment and calibration step: temperature and slope."""
    ab = Path(abgleich_dir)
    files = sorted(ab.glob("Step*_*Grad.csv"))
    tp = ab / "coefficients" / "temp.csv"
    if not files or not tp.is_file():
        return pd.DataFrame()
    tc = _pairs(tp)
    rows = []
    for f in files:
        m = re.search(r"Step(\d+)_(\d+)Grad", f.name)
        if not m:
            continue
        step, nominal = int(m.group(1)), float(m.group(2))
        for ln in f.read_text(encoding="utf-8-sig").splitlines():
            ms = re.match(r"s(\d+):", ln.strip())
            if not ms:
                continue
            tv = [float(x) for x in re.findall(r"temp\d=([-\d.]+)V", ln)]
            T = [(v - tc[i, 0]) / tc[i, 1] for i, v in enumerate(tv)
                 if i < len(tc)]
            I = np.array([float(x) for x in re.findall(r"i_s=([-\d.]+)A", ln)])
            U = np.array([float(x) for x in re.findall(r"u_s=([-\d.]+)V", ln)])
            if I.size < 2 or I.size != U.size:
                continue
            k, u0 = np.polyfit(I, U, 1)
            res = U - (k * I + u0)
            rows.append({"segment": ms.group(1), "step": step,
                         "T_nominal": nominal,
                         "T_degC": float(np.mean(T)) if T else nominal,
                         "k_V_per_A": float(k), "offset_V": float(u0),
                         "nonlin_mV": float(1e3 * np.abs(res).max())})
    return pd.DataFrame(rows)


def run_context(run_dirs) -> tuple[dict, dict]:
    """card per segment, and {condition: {segment: T_degC}} from finished
    runs (bronze/segment_meta.csv)."""
    import json
    card, temps = {}, {}
    for rd in map(Path, run_dirs or []):
        meta = rd / "bronze" / "segment_meta.csv"
        if not meta.is_file():
            continue
        m = pd.read_csv(meta, dtype={"segment": str})
        card.update(dict(zip(m.segment, m.card.map(_short))))
        cond = rd.name
        cj = rd / "config_used.json"
        if cj.is_file():
            try:
                cond = json.loads(cj.read_text()).get("condition") or cond
            except Exception:                               # noqa: BLE001
                pass
        if "T_degC" in m:
            temps[cond] = dict(zip(m.segment, m.T_degC.astype(float)))
    return card, temps


def analyse(steps: pd.DataFrame, curr_csv=None, card: dict | None = None,
            temps: dict | None = None) -> pd.DataFrame:
    """Per segment: level, temperature coefficient, hysteresis, and k at the
    operating temperature of every run given."""
    cc = _pairs(curr_csv) if curr_csv and Path(curr_csv).is_file() else None
    out = []
    for seg, g in steps.groupby("segment"):
        up = g[g.step <= 5].sort_values("T_degC")
        if len(up) < 2:
            continue
        a, b = np.polyfit(up.T_degC, up.k_V_per_A, 1)
        k20 = a * 20.0 + b
        r = {"segment": seg, "card": (card or {}).get(seg, ""),
             "k20_V_per_A": k20, "alpha_pct_per_K": 100 * a / k20,
             "nonlin_mV_max": float(g.nonlin_mV.max())}
        s1, s6 = g[g.step == 1], g[g.step == 6]
        if len(s1) and len(s6):
            r["return_drift_pct"] = float(
                100 * (s6.k_V_per_A.iloc[0] / s1.k_V_per_A.iloc[0] - 1))
        i = int(seg) - 1
        if cc is not None and 0 <= i < len(cc):
            c0, c1 = cc[i]
            r["alpha_curr_pct_per_K"] = 100 * 1e-3 * c1 / (c0 + 1e-3 * c1 * 20)
            r["k_over_K"] = k20 / (c0 + 1e-3 * c1 * 20)
        for cond, tm in (temps or {}).items():
            if seg in tm and np.isfinite(tm[seg]):
                r[f"T_{cond}"] = tm[seg]
                r[f"k_{cond}"] = a * tm[seg] + b
        out.append(r)
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("segment", key=lambda s: s.astype(int))
        df["k20_vs_median_pct"] = 100 * (df.k20_V_per_A
                                         / df.k20_V_per_A.median() - 1)
    return df.reset_index(drop=True)


def summary(seg: pd.DataFrame, temps: dict | None = None) -> dict:
    s = {"n_segments": int(len(seg)),
         "alpha_median": float(seg.alpha_pct_per_K.median()),
         "alpha_min": float(seg.alpha_pct_per_K.min()),
         "alpha_max": float(seg.alpha_pct_per_K.max()),
         "k20_cv_pct": float(100 * seg.k20_V_per_A.std()
                             / seg.k20_V_per_A.mean()),
         "nonlin_mV_max": float(seg.nonlin_mV_max.max())}
    if "return_drift_pct" in seg:
        s["return_drift_pct_median"] = float(seg.return_drift_pct.median())
        s["return_drift_pct_maxabs"] = float(seg.return_drift_pct.abs().max())
    s["operating"] = {}
    for cond, tm in (temps or {}).items():
        T = np.array([v for v in tm.values() if np.isfinite(v)])
        if T.size:
            s["operating"][cond] = {
                "T_min": float(T.min()), "T_max": float(T.max()),
                # what the plate's own temperature spread does to K, and so
                # to j, across the plate (already corrected per segment)
                "K_span_pct": float(s["alpha_median"] * (T.max() - T.min()))}
    return s


def report(s: dict, say=print) -> None:
    say(f"  via coefficient over {s['n_segments']} segments: alpha "
        f"{s['alpha_median']:.3f} %/K ({s['alpha_min']:.3f}..{s['alpha_max']:.3f};"
        f" copper {COPPER_ALPHA_PCT}), level spread at 20 C {s['k20_cv_pct']:.1f} %"
        f" (cv), linear within {s['nonlin_mV_max']:.2f} mV")
    if "return_drift_pct_median" in s:
        say(f"  back at 20 C after 90 C: {s['return_drift_pct_median']:+.3f} % "
            f"(max |{s['return_drift_pct_maxabs']:.3f}| %)")
    for cond, o in s.get("operating", {}).items():
        say(f"  {cond}: segments at {o['T_min']:.1f}..{o['T_max']:.1f} C -> "
            f"K differs by {o['K_span_pct']:.2f} % across the plate (corrected "
            f"per segment); each K of local temperature error = "
            f"{s['alpha_median']:.2f} % of j")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_vs_temperature(steps: pd.DataFrame, seg: pd.DataFrame, path,
                        temps: dict | None = None, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    card = dict(zip(seg.segment, seg.card))
    cards = sorted(set(card.values()), key=lambda c: (len(c), c)) or [""]
    n = len(cards)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    cm = plt.get_cmap("viridis")
    ylo, yhi = steps.k_V_per_A.min(), steps.k_V_per_A.max()
    tcol = plt.get_cmap("autumn")
    conds = list((temps or {}).keys())
    for ax, cd in zip(axes.flat, cards):
        segs = [s for s in seg.segment if card.get(s, "") == cd]
        for i, s in enumerate(segs):
            g = steps[steps.segment == s].sort_values("step")
            c = cm(i / max(1, len(segs) - 1))
            up = g[g.step <= 5].sort_values("T_degC")
            ax.plot(up.T_degC, up.k_V_per_A, "o-", ms=3, lw=1, color=c,
                    label=f"seg {s}")
            back = g[g.step == 6]
            ax.plot(back.T_degC, back.k_V_per_A, "s", ms=4, mfc="none",
                    color=c)
        for j, cond in enumerate(conds):
            T = [temps[cond][s] for s in segs if s in temps[cond]]
            if T:
                ax.axvspan(min(T), max(T), color=tcol(j / max(1, len(conds))),
                           alpha=0.18, lw=0)
                ax.text(np.mean(T), yhi, cond, ha="center", va="top",
                        fontsize=7, rotation=90, color="0.25")
        ax.set_title(f"{cd or 'no card in these runs'}  ({len(segs)} "
                     f"segments)",
                     fontsize=10)
        ax.grid(alpha=0.3)
        if len(segs) <= 16:
            ax.legend(fontsize=6, ncol=2, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("via temperature [°C]")
    for ax in axes[:, 0]:
        ax.set_ylabel("via coefficient k = du/di  [V/A]  (∝ R_via)")
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes[0, 0].set_ylim(ylo * 0.95, yhi * 1.05)
    fig.suptitle(title or "Via resistance vs temperature per card (Abgleich "
                 "DC calibration; ○ up to 90 °C, □ back at 20 °C; shaded = "
                 "operating temperatures of the runs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def plot_alpha(seg: pd.DataFrame, path, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cards = sorted(set(seg.card), key=lambda c: (len(c), c))
    col = dict(zip(cards, plt.get_cmap("tab10").colors))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    x = seg.segment.astype(int).to_numpy()
    c = [col.get(cd, "0.5") for cd in seg.card]
    a1.bar(x, seg.alpha_pct_per_K, color=c)
    a1.axhline(COPPER_ALPHA_PCT, color="k", ls="--", lw=0.9,
               label=f"copper {COPPER_ALPHA_PCT} %/K")
    if "alpha_curr_pct_per_K" in seg:
        a1.plot(x, seg.alpha_curr_pct_per_K, "k.", ms=4, label="from curr.csv")
    al = np.r_[seg.alpha_pct_per_K, seg.get("alpha_curr_pct_per_K",
                                              pd.Series(dtype=float)).dropna()]
    a1.set_ylim(al.min() - 0.004, max(al.max(), COPPER_ALPHA_PCT) + 0.004)
    a1.set_ylabel("α = (1/k) dk/dT  [%/K]")
    a1.legend(fontsize=8, loc="lower right")
    a2.bar(x, seg.k20_vs_median_pct, color=c)
    a2.axhline(0, color="0.4", lw=0.8)
    a2.set_ylabel("k at 20 °C vs plate median [%]")
    a2.set_xlabel("segment")
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color=col[cd], label=cd or "no card")
                       for cd in cards], fontsize=8, ncol=len(cards),
              loc="lower left")
    for a in (a1, a2):
        a.grid(alpha=0.3, axis="y")
    fig.suptitle(title or "Via temperature coefficient and level per segment, "
                 "coloured by card", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def run(abgleich_dir, run_dirs=(), out_dir=None, write_png: bool = True,
        log=None) -> dict:
    say = log.info if log else print
    ab = Path(abgleich_dir)
    out = Path(out_dir) if out_dir else ab / "via_resistance"
    out.mkdir(parents=True, exist_ok=True)
    steps = read_steps(ab)
    if steps.empty:
        say(f"  no Step*_*Grad.csv calibration files in {ab}")
        return {}
    card, temps = run_context(run_dirs)
    steps["card"] = steps.segment.map(card).fillna("")
    seg = analyse(steps, ab / "coefficients" / "curr.csv", card, temps)
    steps.round(6).to_csv(out / "via_steps.csv", index=False)
    seg.round(5).to_csv(out / "via_segments.csv", index=False)
    s = summary(seg, temps)
    report(s, say)
    if write_png:
        plot_vs_temperature(steps, seg, out / "via_vs_temperature.png", temps)
        plot_alpha(seg, out / "via_alpha.png")
    s["out_dir"] = str(out)
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("abgleich", type=Path)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args(argv)
    run(a.abgleich, a.runs, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
