#!/usr/bin/env python3
"""
frequency_response.py
=====================
How does the measuring chain of every segment respond to frequency -- on the
bench (ex-situ) and during the measurement (in-situ)?

WHY
---
Z = K * U_cell / u_seg is a ratio of two measured signals, so whatever the
chain does to either signal lands in Z. It lands hardest at the top of the
band: a phase error theta rotates each point about the origin and moves its
real part by about -theta * Im Z, and theta grows with frequency. On
2612030 an uncorrected 30 us card delay lowers a segment's HFR by ~16
mOhm*cm2 while the low-frequency arc looks perfect -- which is why this has
to be looked at, not assumed.

TWO VIEWS, BECAUSE NEITHER SEES EVERYTHING
------------------------------------------
ex-situ  G_d(f) = j_sigma / j_r of each segment's current path, from the
         Abgleich bode sweeps (Gamry, 1 Hz - 100 kHz, no cell). Absolute,
         but it sees only the amplifier, not the acquisition cards.
         Read from the chain gain file the run applied (gain = 1/H), or
         from the bode/ folder itself.
in-situ  H_rel,s(f) = Z_s(f) / median over segments Z(f), from the run.
         Relative -- whatever all segments share cancels -- but it sees the
         whole chain as it was during the measurement: cards, multiplexer
         slots, scale factors. Drawn twice: from bronze/raw_spectra.csv
         (chain file divided out, no timing correction) and from
         silver/spectra_clean.csv (after de-skew and channel lag), on the
         same points, so the plot shows what the timing correction removed
         and what is left.

WHAT IT WRITES (RUN_DIR/frequency_response/)
---------------------------------------------
    exsitu_response.csv      segment, freq_hz, mag, phase_deg
    insitu_response.csv      segment, card, stage (raw|clean), freq_hz, mag, phase_deg
    insitu_card_residual.csv per card: median phase and |H| step above f_hf
    summary.json             the numbers the plausibility checks use
    freq_response_exsitu.png / freq_response_insitu.png

and two plausibility checks ("chain response (ex-situ)", "segment frequency
response (in-situ)"). Nothing here changes Z: it is a diagnostic.

    python frequency_response.py RUN_DIR [--gain chain_gain_gen1.csv]
                                         [--bode ABGLEICH_DIR/bode]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

#: in-situ: phase spread (sd over segments, degrees) at the top of the band
PHASE_SD_WARN_DEG = 5.0
PHASE_SD_FAIL_DEG = 10.0
#: in-situ: a card whose |H| above F_HF moves from its own 30-1000 Hz level
CARD_STEP_WARN_PCT = 5.0
#: ex-situ: a sweep that is not flat (|G| - 1) between FLAT_BAND
FLAT_WARN_PCT = 1.0
FLAT_BAND = (5.0, 500.0)
#: ex-situ: segment spread of the phase at the top of the analysis band
EXSITU_SD_WARN_DEG = 2.0
F_HF = 2000.0
LEVEL_BAND = (30.0, 1000.0)
OUT = "frequency_response"


def _short(card) -> str:
    try:
        import card_gain
        return card_gain.short(card)
    except Exception:                                       # noqa: BLE001
        return str(card)


# ---------------------------------------------------------------------------
# ex-situ
# ---------------------------------------------------------------------------

def exsitu_from_gain(path) -> pd.DataFrame:
    """G_d per segment from a pipeline gain file (gain = 1/H -> H = 1/gain)."""
    import utils
    rows = []
    for seg, (f, g) in utils.load_gain(path).items():
        H = 1.0 / np.asarray(g, complex)
        rows.append(pd.DataFrame({
            "segment": str(seg), "freq_hz": np.asarray(f, float),
            "mag": np.abs(H), "phase_deg": np.degrees(np.unwrap(np.angle(H)))}))
    return pd.concat(rows, ignore_index=True) if rows else _empty_exsitu()


def exsitu_from_bode(folder, f_ref_hz: float | None = None) -> pd.DataFrame:
    """G_d per segment straight from the Abgleich bode/ sweeps."""
    import gamry_dta
    rows = []
    for seg, sw in gamry_dta.read_bode_folder(folder).items():
        s = sw.sorted()
        H = gamry_dta._normalised(s, f_ref_hz)
        rows.append(pd.DataFrame({
            "segment": str(seg), "freq_hz": s.freq, "mag": np.abs(H),
            "phase_deg": np.degrees(np.unwrap(np.angle(H)))}))
    return pd.concat(rows, ignore_index=True) if rows else _empty_exsitu()


def _empty_exsitu() -> pd.DataFrame:
    return pd.DataFrame(columns=["segment", "freq_hz", "mag", "phase_deg"])


def _at(df: pd.DataFrame, f0: float) -> pd.DataFrame:
    """Each segment's point nearest to f0 (log distance)."""
    d = df.assign(_d=np.abs(np.log(df.freq_hz / f0)))
    return d.loc[d.groupby("segment")._d.idxmin()].drop(columns="_d")


def exsitu_summary(df: pd.DataFrame, f_top: float = 4000.0) -> dict:
    if df.empty:
        return {"available": False}
    out = {"available": True, "n_segments": int(df.segment.nunique()),
           "f_top_hz": f_top, "at": {}}
    for f0 in (100.0, 1000.0, f_top, 10000.0):
        p = _at(df, f0)
        out["at"][f"{f0:g}"] = {
            "mag_median": float(p.mag.median()),
            "phase_deg_median": float(p.phase_deg.median()),
            "phase_deg_sd": float(p.phase_deg.std(ddof=0)),
            "mag_p5": float(p.mag.quantile(.05)),
            "mag_p95": float(p.mag.quantile(.95))}
    lo, hi = FLAT_BAND
    band = df[(df.freq_hz >= lo) & (df.freq_hz <= hi)]
    dev = (100 * (band.groupby("segment").mag.median() - 1.0)).round(2)
    out["not_flat"] = {str(s): float(v) for s, v in dev.items()
                       if abs(v) > FLAT_WARN_PCT}
    return out


# ---------------------------------------------------------------------------
# in-situ
# ---------------------------------------------------------------------------

def _relative(df: pd.DataFrame, re: str, im: str, scale: float
              ) -> pd.DataFrame:
    df = df.copy()
    df["key"] = np.round(np.log10(df.freq_hz), 4)
    pr = df.pivot_table(index="key", columns="segment", values=re,
                        aggfunc="first")
    pi = df.pivot_table(index="key", columns="segment", values=im,
                        aggfunc="first").reindex_like(pr)
    Z = (pr.to_numpy() + 1j * pi.to_numpy()) * scale
    need = max(3, int(np.ceil(0.5 * Z.shape[1])))
    ok = np.sum(np.isfinite(Z), axis=1) >= need
    med = np.full(Z.shape[0], np.nan, complex)
    med[ok] = (np.nanmedian(Z[ok].real, axis=1)
               + 1j * np.nanmedian(Z[ok].imag, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        H = Z / med[:, None]
    out = pd.DataFrame(H, index=pr.index, columns=pr.columns)
    long = out.stack(future_stack=True).rename("H").reset_index()
    long = long[np.isfinite(long.H.to_numpy())]
    long["freq_hz"] = 10.0 ** long.key
    long["mag"] = np.abs(long.H.to_numpy())
    long["phase_deg"] = np.degrees(np.angle(long.H.to_numpy()))
    return long.drop(columns=["key", "H"])


def insitu(run_dir) -> pd.DataFrame:
    """Z_s / plate median per segment, raw (bronze) and clean (silver),
    on the points silver kept."""
    run = Path(run_dir)
    clean = pd.read_csv(run / "silver" / "spectra_clean.csv",
                        dtype={"segment": str})
    meta_p = run / "bronze" / "segment_meta.csv"
    card = {}
    if meta_p.is_file():
        meta = pd.read_csv(meta_p, dtype={"segment": str})
        card = dict(zip(meta.segment, meta.card.map(_short)))
        clean = clean[clean.segment.isin(card)]
    parts = [_relative(clean, "z_re_mohm_cm2", "z_im_mohm_cm2", 1.0)
             .assign(stage="clean")]
    raw_p = run / "bronze" / "raw_spectra.csv"
    if raw_p.is_file():
        raw = pd.read_csv(raw_p, dtype={"segment": str})
        keep = set(zip(clean.segment, np.round(np.log10(clean.freq_hz), 4)))
        raw = raw[[k in keep for k in zip(
            raw.segment, np.round(np.log10(raw.freq_hz), 4))]]
        raw = raw.drop_duplicates(["segment", "freq_hz"])
        if not raw.empty:
            parts.append(_relative(raw, "z_re_ohm_cm2", "z_im_ohm_cm2",
                                   1000.0).assign(stage="raw"))
    df = pd.concat(parts, ignore_index=True)
    df["card"] = df.segment.map(card).fillna("")
    return df[["segment", "card", "stage", "freq_hz", "mag", "phase_deg"]]


def insitu_summary(df: pd.DataFrame) -> dict:
    out = {"available": not df.empty, "stages": {}}
    if df.empty:
        return out
    f_top = float(df[df.stage == "clean"].freq_hz.max())
    out["f_top_hz"] = f_top
    for st, d in df.groupby("stage"):
        rows = {}
        for f0 in (100.0, 1000.0, F_HF, f_top):
            p = _at(d, f0)
            rows[f"{f0:g}"] = {
                "freq_hz": float(p.freq_hz.median()),
                "phase_deg_sd": float(p.phase_deg.std(ddof=0)),
                "phase_deg_p5": float(p.phase_deg.quantile(.05)),
                "phase_deg_p95": float(p.phase_deg.quantile(.95)),
                "mag_sd_pct": float(100 * p.mag.std(ddof=0))}
        out["stages"][st] = rows
    out["card_residual"] = card_residual(df).to_dict(orient="records")
    return out


def card_residual(df: pd.DataFrame) -> pd.DataFrame:
    """Per card, after correction: median phase above F_HF, and how far |H|
    above F_HF moves from the same segment's own 30-1000 Hz level (a flat
    scale factor does not count, a frequency-dependent card response does)."""
    d = df[(df.stage == "clean") & (df.card != "")]
    if d.empty:
        return pd.DataFrame(columns=["card", "n_segments", "phase_hf_deg",
                                     "mag_step_hf_pct"])
    lo, hi = LEVEL_BAND
    lvl = d[(d.freq_hz >= lo) & (d.freq_hz <= hi)].groupby("segment").mag.median()
    hf = d[d.freq_hz >= F_HF]
    seg = hf.groupby("segment").agg(card=("card", "first"),
                                    mag=("mag", "median"),
                                    ph=("phase_deg", "median"))
    seg["step"] = 100 * (seg.mag / lvl.reindex(seg.index) - 1.0)
    out = seg.groupby("card").agg(n_segments=("step", "size"),
                                  phase_hf_deg=("ph", "median"),
                                  mag_step_hf_pct=("step", "median"))
    return out.round(2).reset_index()


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def checks(summary: dict) -> list:
    import plausibility as pl
    out = []
    ex = summary.get("exsitu", {})
    if not ex.get("available"):
        out.append(pl.Check("chain response (ex-situ)", pl.NA,
                            "no chain gain file or bode folder to read"))
    else:
        top = ex["at"][f"{ex['f_top_hz']:g}"]
        txt = (f"{ex['n_segments']} segments; at {ex['f_top_hz']:g} Hz "
               f"|G_d| {top['mag_median']:.3f}, arg {top['phase_deg_median']:+.1f}"
               f" deg, segment sd {top['phase_deg_sd']:.2f} deg")
        rests = ("the gain file being the Abgleich sweep of THIS plate, "
                 "normalised where every sweep is flat")
        bad = ex.get("not_flat", {})
        if bad:
            out.append(pl.Check(
                "chain response (ex-situ)", pl.WARN,
                f"{txt}; not flat between {FLAT_BAND[0]:g} and "
                f"{FLAT_BAND[1]:g} Hz: " + ", ".join(
                    f"seg {s} {v:+.1f} %" for s, v in bad.items())
                + " -- a step at the normalisation point, carried into Z at "
                  "every frequency", top["phase_deg_sd"], rests))
        elif top["phase_deg_sd"] > EXSITU_SD_WARN_DEG:
            out.append(pl.Check("chain response (ex-situ)", pl.WARN,
                                f"{txt}: amplifiers differ more than "
                                f"{EXSITU_SD_WARN_DEG:g} deg", top["phase_deg_sd"],
                                rests))
        else:
            out.append(pl.Check("chain response (ex-situ)", pl.PASS, txt,
                                top["phase_deg_sd"], rests))

    ins = summary.get("insitu", {})
    if not ins.get("available") or "clean" not in ins.get("stages", {}):
        out.append(pl.Check("segment frequency response (in-situ)", pl.NA,
                            "no spectra to compare"))
        return out
    key = f"{ins['f_top_hz']:g}"
    cl = ins["stages"]["clean"][key]
    raw = ins["stages"].get("raw", {}).get(key)
    txt = (f"phase spread at {cl['freq_hz']:.0f} Hz sd {cl['phase_deg_sd']:.1f}"
           f" deg (p5..p95 {cl['phase_deg_p5']:+.1f}..{cl['phase_deg_p95']:+.1f})")
    if raw:
        txt += f", before timing correction sd {raw['phase_deg_sd']:.1f} deg"
    cards = [c for c in ins.get("card_residual", [])
             if np.isfinite(c.get("mag_step_hf_pct", np.nan))]
    steps = [c for c in cards if abs(c["mag_step_hf_pct"]) > CARD_STEP_WARN_PCT]
    if cards:
        txt += "; |H| step above {:g} Hz per card: ".format(F_HF) + ", ".join(
            f"{c['card']} {c['mag_step_hf_pct']:+.1f} %" for c in cards)
    rests = ("all segments sitting on one cell, so their true spectra share a "
             "shape; a real local difference in the HF arc would also show here")
    sd = cl["phase_deg_sd"]
    if sd > PHASE_SD_FAIL_DEG:
        v = pl.FAIL
        txt = "segments still rotated against each other at HF -- " + txt
    elif sd > PHASE_SD_WARN_DEG or steps:
        v = pl.WARN
        if steps:
            txt += (f" -- grouped by card: a card response above {F_HF:g} Hz "
                    "that is not a pure delay, or, for a card that covers "
                    "one region of the plate, a regional effect; a bench "
                    "sweep through the cards separates the two")
    else:
        v = pl.PASS
    out.append(pl.Check("segment frequency response (in-situ)", v, txt, sd,
                        rests))
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_exsitu(df: pd.DataFrame, path, title: str = "", f_top=4000.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)
    segs = sorted(df.segment.unique(), key=lambda s: (len(s), s))
    cm = plt.get_cmap("viridis")
    for i, s in enumerate(segs):
        d = df[df.segment == s].sort_values("freq_hz")
        c = cm(i / max(1, len(segs) - 1))
        a1.semilogx(d.freq_hz, d.mag, color=c, lw=0.8)
        a2.semilogx(d.freq_hz, d.phase_deg, color=c, lw=0.8)
    for ax in (a1, a2):
        ax.grid(True, which="both", alpha=0.3)
        ax.axvline(f_top, color="0.4", ls="--", lw=0.8)
    a1.set_ylim(0, 1.1)
    a1.set_ylabel(r"$|G_d| = |j_\sigma / j_r|$")
    a2.set_ylabel(r"arg$(G_d)$ in degree")
    a2.set_xlabel("frequency f in Hz")
    a1.set_title(title or f"Ex-situ current-chain response, {len(segs)} "
                 f"segments (dashed: top of the analysis band)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_insitu(df: pd.DataFrame, path, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stages = [s for s in ("raw", "clean") if s in set(df.stage)]
    names = {"raw": "before: chain file divided out, no timing correction",
             "clean": "after: mux de-skew + channel-lag correction"}
    cards = sorted(set(df.card) - {""}, key=lambda c: (len(c), c))
    col = dict(zip(cards, plt.get_cmap("tab10").colors))
    fig, ax = plt.subplots(2, len(stages), figsize=(6 * len(stages), 7.2),
                           sharex=True, squeeze=False)
    for j, st in enumerate(stages):
        d = df[df.stage == st]
        for s, g in d.groupby("segment"):
            g = g.sort_values("freq_hz")
            c = col.get(g.card.iloc[0], "0.5")
            ax[0, j].semilogx(g.freq_hz, g.mag, color=c, lw=0.7, alpha=0.8)
            ax[1, j].semilogx(g.freq_hz, g.phase_deg, color=c, lw=0.7,
                              alpha=0.8)
        ax[0, j].set_title(names[st], fontsize=10)
        ax[0, j].set_ylim(0.8, 1.2)
        ax[1, j].set_ylim(-60, 30)
        ax[1, j].set_xlabel("frequency f in Hz")
        for a in ax[:, j]:
            a.grid(True, which="both", alpha=0.3)
    ax[0, 0].set_ylabel(r"$|Z_s / Z_{plate\ median}|$")
    ax[1, 0].set_ylabel(r"arg$(Z_s / Z_{plate\ median})$ in degree")
    for c in cards:
        ax[0, -1].plot([], [], color=col[c], label=c)
    if cards:
        ax[0, -1].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle(title or "In-situ relative response per segment "
                 "(each segment against the plate median)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(run_dir, gain_file=None, bode_dir=None, out_dir=None,
        write_png: bool = True, f_top: float = 4000.0, title: str = "",
        log=None) -> dict:
    """Both views for one finished run; writes RUN_DIR/frequency_response/."""
    say = log.info if log else print
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir else run_dir / OUT
    out.mkdir(parents=True, exist_ok=True)

    ex, src = _empty_exsitu(), ""
    if bode_dir and Path(bode_dir).is_dir():
        ex, src = exsitu_from_bode(bode_dir), str(bode_dir)
    elif gain_file and Path(gain_file).is_file():
        ex, src = exsitu_from_gain(gain_file), str(gain_file)
    summary = {"exsitu": exsitu_summary(ex, f_top), "exsitu_source": src}
    if not ex.empty:
        ex.to_csv(out / "exsitu_response.csv", index=False)

    ins = pd.DataFrame()
    if (run_dir / "silver" / "spectra_clean.csv").is_file():
        ins = insitu(run_dir)
        ins.round(6).to_csv(out / "insitu_response.csv", index=False)
        card_residual(ins).to_csv(out / "insitu_card_residual.csv",
                                  index=False)
    summary["insitu"] = insitu_summary(ins)
    (out / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 default=float))
    if write_png:
        try:
            if not ex.empty:
                plot_exsitu(ex, out / "freq_response_exsitu.png", f_top=f_top)
            if not ins.empty:
                plot_insitu(ins, out / "freq_response_insitu.png", title)
        except Exception as exc:                            # noqa: BLE001
            say(f"  frequency response plots skipped: {exc}")
    report(summary, say)
    return summary


def report(summary: dict, say=print) -> None:
    ex = summary.get("exsitu", {})
    if ex.get("available"):
        top = ex["at"][f"{ex['f_top_hz']:g}"]
        say(f"  ex-situ chain ({ex['n_segments']} segments): at "
            f"{ex['f_top_hz']:g} Hz |G_d| {top['mag_median']:.3f}, "
            f"arg {top['phase_deg_median']:+.1f} deg, segment sd "
            f"{top['phase_deg_sd']:.2f} deg")
        for s, v in ex.get("not_flat", {}).items():
            say(f"    seg {s}: {v:+.1f} % off 1 between {FLAT_BAND[0]:g} and "
                f"{FLAT_BAND[1]:g} Hz (normalisation step)")
    else:
        say("  ex-situ chain: no gain file / bode folder available")
    ins = summary.get("insitu", {})
    if ins.get("available"):
        key = f"{ins['f_top_hz']:g}"
        for st in ("raw", "clean"):
            r = ins["stages"].get(st, {}).get(key)
            if r:
                say(f"  in-situ {st:5s}: phase at {r['freq_hz']:.0f} Hz "
                    f"p5..p95 {r['phase_deg_p5']:+.1f}..{r['phase_deg_p95']:+.1f}"
                    f" deg (sd {r['phase_deg_sd']:.1f})")
        for c in ins.get("card_residual", []):
            say(f"    {c['card']}: above {F_HF:g} Hz phase "
                f"{c['phase_hf_deg']:+.1f} deg, |H| step "
                f"{c['mag_step_hf_pct']:+.1f} %")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--gain", type=Path, help="chain gain file (gain = 1/H)")
    ap.add_argument("--bode", type=Path, help="Abgleich bode/ folder")
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--f-top", type=float, default=4000.0)
    a = ap.parse_args(argv)
    gain = a.gain
    if gain is None:
        cj = a.run / "config_used.json"
        if cj.is_file():
            g = json.loads(cj.read_text()).get("gain_file")
            gain = Path(g) if g else None
    s = run(a.run, gain_file=gain, bode_dir=a.bode, out_dir=a.out,
            f_top=a.f_top)
    for c in checks(s):
        print(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
