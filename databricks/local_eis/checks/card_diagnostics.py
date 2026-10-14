#!/usr/bin/env python3
"""
card_diagnostics.py
===================
Four plots that separate a DAQ-card problem from a synchronisation problem
from a real spatial effect in the cell:

  1. median Nyquist per card            (+ the Gamry whole cell, black)
  2. median Bode |Z| per card
  3. median Bode phase per card
  4. card-to-Gamry transfer function    H_c(f) = Z_card-median(f) / Z_Gamry(f)

per operating point, and the transfer function of every card across all
operating points side by side (cross_current.png). How to read the last one:

  * a deviation that is the SAME at every current is the acquisition (the
    cell changes a lot between 45 and 450 A; the hardware does not);
  * a deviation SHARED by every card at a frequency is common to the whole
    measurement chain or to the comparison (sense point, Gamry's own chain);
  * a deviation that changes with current and differs between cards is the
    cell's own spatial behaviour (the cards sit over different regions).

A card median is the complex median (Re and Im separately) of its measured
segments at each step; the Gamry is its own point at that step (nearest
frequency within 2 %), every point as recorded, times 304.92 cm^2.

    python card_diagnostics.py RUN_45A RUN_60A RUN_150A RUN_450A -o out/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: categorical slots 1-5 of the reference palette (validated: CVD dE >= 9.1)
#: plus a marker per card -- identity never rests on colour alone
CARD_STYLE = {
    "Karte_1": ("#2a78d6", "o"), "Karte_2": ("#eb6834", "s"),
    "Karte_3": ("#1baf7a", "^"), "Karte_4": ("#eda100", "D"),
    "Karte_5": ("#e87ba4", "v"),
}
#: one-hue sequential ramp (light -> dark) for the operating points
CURRENT_RAMP = ["#9cc1ee", "#5d98e1", "#2a78d6", "#18487f"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def _card(name: str) -> str:
    import re
    m = re.search(r"Karte_?\d+", str(name))
    return m.group(0) if m else str(name)


def _amps(cond: str) -> float:
    try:
        return float(str(cond).upper().rstrip("A"))
    except ValueError:
        return float("nan")


def card_medians(sp: pd.DataFrame) -> pd.DataFrame:
    """card, fk, Z: complex median of the card's segments at each step;
    plus the plate median as card 'plate'."""
    g = sp.assign(card=sp["card"].map(_card)).groupby(["card", "fk"])
    cm = (g["z_re_mohm_cm2"].median() + 1j * g["z_im_mohm_cm2"].median()).rename("Z")
    gp = sp.groupby("fk")
    pm = (gp["z_re_mohm_cm2"].median() + 1j * gp["z_im_mohm_cm2"].median()).rename("Z")
    pm = pm.to_frame().assign(card="plate").reset_index()
    return pd.concat([cm.reset_index(), pm], ignore_index=True)


def transfer(cm: pd.DataFrame, gamry: pd.DataFrame) -> pd.DataFrame:
    """H = Z_card / Z_Gamry at the Gamry's own point of each step."""
    gf = gamry["freq_hz"].to_numpy(float)
    gz = (gamry["z_re_mohm_cm2"].to_numpy(float)
          + 1j * gamry["z_im_mohm_cm2"].to_numpy(float))
    lf = np.log(gf)
    k = np.abs(np.log(cm["fk"].to_numpy(float))[:, None] - lf[None, :]).argmin(axis=1)
    near = np.abs(np.log(cm["fk"].to_numpy(float)) - lf[k]) <= np.log(1.02)
    H = np.where(near, cm["Z"].to_numpy() / gz[k], np.nan)
    out = cm[["card", "fk"]].copy()
    out["H_mag_pct"] = 100 * (np.abs(H) - 1)
    out["H_phase_deg"] = np.degrees(np.angle(H))
    return out


def series_offset(cm: pd.DataFrame, gamry: pd.DataFrame,
                  band=(100.0, 3000.0)) -> dict:
    """Plate median minus Gamry over `band`: median dRe and dIm [mOhm.cm2].

    On 2612030 dRe is +8..+12 with dIm ~0 at every current above 100 Hz: a
    series resistance in the FAMOS voltage path the Gamry's sense leads do
    not see (config.uc_series_mohm_cm2 removes it), not a card and not the
    cell."""
    p = cm[cm.card == "plate"].sort_values("fk")
    gf = gamry["freq_hz"].to_numpy(float)
    gz = (gamry["z_re_mohm_cm2"].to_numpy(float)
          + 1j * gamry["z_im_mohm_cm2"].to_numpy(float))
    f = p["fk"].to_numpy(float)
    k = np.abs(np.log(f)[:, None] - np.log(gf)[None, :]).argmin(axis=1)
    dz = p["Z"].to_numpy() - gz[k]
    m = (f >= band[0]) & (f < band[1])
    return {"dRe_mohm_cm2": float(np.median(dz.real[m])) if m.any() else float("nan"),
            "dIm_mohm_cm2": float(np.median(dz.imag[m])) if m.any() else float("nan"),
            "band_hz": list(band)}


def _axes_style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def figure_condition(cond: str, cm: pd.DataFrame, gamry, path) -> Path:
    """The four plots for one operating point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))
    a_ny, a_mag, a_ph, a_h = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    tr = transfer(cm, gamry) if gamry is not None else None
    for c, (col, mk) in CARD_STYLE.items():
        d = cm[cm.card == c].sort_values("fk")
        if d.empty:
            continue
        f, Z = d["fk"].to_numpy(float), d["Z"].to_numpy()
        kw = dict(color=col, marker=mk, ms=4, lw=1.6, label=c.replace("_", " "))
        a_ny.plot(Z.real, -Z.imag, **kw)
        a_mag.loglog(f, np.abs(Z), **kw)
        a_ph.semilogx(f, np.degrees(np.angle(Z)), **kw)
        # direct label at the high-frequency end of the |Z| curve
        a_mag.annotate(c[-1], (f[-1], np.abs(Z)[-1]), xytext=(4, 0),
                       textcoords="offset points", color=INK, fontsize=8,
                       va="center")
        if tr is not None:
            t = tr[tr.card == c].sort_values("fk")
            a_h.semilogx(t["fk"], t["H_mag_pct"], **kw)
    if gamry is not None:
        gf = gamry["freq_hz"].to_numpy(float)
        gz = gamry["z_re_mohm_cm2"].to_numpy(float) + 1j * gamry["z_im_mohm_cm2"].to_numpy(float)
        kw = dict(color=INK, marker="D", ms=3, lw=2.2, label="Gamry whole cell")
        a_ny.plot(gz.real, -gz.imag, **kw)
        a_mag.loglog(gf, np.abs(gz), **kw)
        a_ph.semilogx(gf, np.degrees(np.angle(gz)), **kw)
        if tr is not None:
            t = tr[tr.card == "plate"].sort_values("fk")
            a_h.semilogx(t["fk"], t["H_mag_pct"], color=MUTED, ls="--", lw=1.4,
                         label="plate median")
    a_ny.axhline(0, color=MUTED, lw=0.6)
    a_ny.set(xlabel="Z' [mΩ·cm²]", ylabel="−Z'' [mΩ·cm²]", title="1  Median Nyquist per card")
    a_ny.set_aspect("equal", adjustable="datalim")
    a_mag.set(xlabel="f [Hz]", ylabel="|Z| [mΩ·cm²]", title="2  Median Bode |Z| per card")
    a_ph.set(xlabel="f [Hz]", ylabel="phase [°]", title="3  Median Bode phase per card")
    a_h.axhline(0, color=MUTED, lw=0.6)
    a_h.set(xlabel="f [Hz]", ylabel="|Z_card / Z_Gamry| − 1 [%]",
            title="4  Card-to-Gamry transfer, magnitude")
    for a in ax.flat:
        _axes_style(a)
    a_ny.legend(fontsize=8, frameon=False)
    a_h.legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle(f"{cond}: card medians against the Gamry whole cell", color=INK)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def figure_cross_current(trs: dict[str, pd.DataFrame], path) -> Path:
    """Card-to-Gamry |H| and phase per card (columns), one line per current."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cards = list(CARD_STYLE) + ["plate"]
    conds = sorted(trs, key=_amps)
    ramp = CURRENT_RAMP[-len(conds):] if len(conds) <= len(CURRENT_RAMP) else \
        [plt.get_cmap("Blues")(x) for x in np.linspace(0.35, 0.95, len(conds))]
    fig, ax = plt.subplots(2, len(cards), figsize=(3.1 * len(cards), 6.6),
                           sharex=True, sharey="row")
    for j, c in enumerate(cards):
        for i, cond in enumerate(conds):
            t = trs[cond][trs[cond].card == c].sort_values("fk")
            if t.empty:
                continue
            ax[0, j].semilogx(t["fk"], t["H_mag_pct"], color=ramp[i], lw=1.6,
                              label=cond)
            ax[1, j].semilogx(t["fk"], t["H_phase_deg"], color=ramp[i], lw=1.6,
                              label=cond)
        ax[0, j].set_title(c.replace("_", " "), color=INK, fontsize=10)
        for r in (0, 1):
            ax[r, j].axhline(0, color=MUTED, lw=0.6)
            _axes_style(ax[r, j])
        ax[1, j].set_xlabel("f [Hz]")
    ax[0, 0].set_ylabel("|Z_card / Z_Gamry| − 1 [%]")
    ax[1, 0].set_ylabel("arg(Z_card / Z_Gamry) [°]")
    ax[0, -1].legend(fontsize=8, frameon=False, title="current")
    fig.suptitle("Card-to-Gamry transfer at every operating point: the same at every "
                 "current = acquisition; changing with current = the cell", color=INK)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def build(run_dirs, out_dir, log=print) -> dict:
    """The four plots per run, the cross-current figure and the tables."""
    import hf_response
    import nyquist
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trs, meds, written, offs = {}, [], [], {}
    for d in run_dirs:
        cond, sp, _cfg = hf_response.load_run(d)
        cm = card_medians(sp)
        g = nyquist.gamry_frame(d)
        written.append(figure_condition(cond, cm, g, out / f"cards_{cond}.png"))
        meds.append(cm.assign(condition=cond, z_re=cm["Z"].to_numpy().real,
                              z_im=cm["Z"].to_numpy().imag).drop(columns="Z"))
        if g is not None:
            trs[cond] = transfer(cm, g)
            offs[cond] = series_offset(cm, g)
    pd.concat(meds).to_csv(out / "card_median_spectra.csv", index=False)
    if trs:
        pd.concat([t.assign(condition=c) for c, t in trs.items()]).round(3) \
            .to_csv(out / "card_gamry_transfer.csv", index=False)
        written.append(figure_cross_current(trs, out / "cross_current.png"))
    if offs:
        pd.DataFrame(offs).T.drop(columns="band_hz").round(2).to_csv(
            out / "plate_vs_gamry_offset.csv")
    if log:
        log(f"  card diagnostics for {len(run_dirs)} run(s) -> {out}")
        for c, o in sorted(offs.items(), key=lambda kv: _amps(kv[0])):
            log(f"    {c:>6}: plate - Gamry over 100-3000 Hz: dRe "
                f"{o['dRe_mohm_cm2']:+.1f}, dIm {o['dIm_mohm_cm2']:+.1f} mOhm.cm2")
        if offs:
            med = float(np.median([o["dRe_mohm_cm2"] for o in offs.values()]))
            log(f"    a real offset at every current = series resistance in the "
                f"voltage path; uc_series_mohm_cm2 = {med:.1f} would remove it")
    return {"figures": written, "transfer": trs, "offset": offs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("card_diagnostics"))
    a = ap.parse_args(argv)
    build(a.runs, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
