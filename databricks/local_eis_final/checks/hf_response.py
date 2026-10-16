#!/usr/bin/env python3
"""
hf_response.py
==============
The high-frequency response of every channel, measured across operating
points and FROZEN -- the card and segment correction of the "Local EIS
Complete Fix" plan (fixes 3 and 4), estimated the way that plan demands:
from repeated measurements, validated on a current it was not fitted on.

WHAT IS LEFT AFTER ALL OTHER CORRECTIONS (2612030, 45 / 60 / 150 / 450 A)
-----------------------------------------------------------------------
After the chain gain file, the ADC slot delay, the card UC2 gain and UC2
magnitude equalisation, the channel lag and the per-card phase, each
segment's spectrum against the plate median still shows, above ~1 kHz:

  * a PHASE that is a straight line in f over 2-12.5 kHz (offset sd 13 deg,
    slope sd 11-13 us; after one line per segment 0.9 deg remain), giving
    the 18.5-19.6 deg spread at 11.95 kHz of the in-situ check;
  * a MAGNITUDE step per card that grows with f (card 1: +5 % at 2-4 kHz,
    +15 % at 7-13 kHz; card 3: -5 % -> -13 %; card 2 changes sign).

Both are the SAME at every current: offsets and slopes correlate r = 1.00
between 45 A and 450 A, the card steps agree to 1-2 %. The cell changes a
lot between 45 and 450 A; these do not. They are the acquisition (card
inputs, cabling, pickup paths downstream of the plate amplifiers), and a
regional ohmic difference cannot produce them: it is flat in f.

WHAT THIS MODULE DOES
---------------------
For every segment and every step at or above f_lo (1 kHz), across the runs
given (one per operating point):

  h_s(f) = m_card(s)(f) * exp(j * phi_s(f))

  phi_s(f)   median over runs of arg(Z_s / Z_plate-median)      per SEGMENT
  m_c(f)     median over the card's segments and the runs of
             |Z_s / Z_plate-median|, divided by the same median over
             100-1000 Hz (the card's level there is the cell's)    per CARD

smoothed by a quadratic in f. Magnitude per card, not per segment,
on purpose: a per-segment magnitude would also take out real differences in
the tail of the charge-transfer arc; the card fingerprint is what repeats.

`validate` estimates h from all runs but one and applies it to the one left
out: the phase spread at the top step and the largest card step, before and
after. Only a correction that passes this blind test should be applied.

The file is written in the gain-file format (segment, freq_hz, gain_real,
gain_imag) with h = 1 below f_lo; silver divides each segment's corrected Z
by it (config.hf_response_file), AFTER the per-card phase -- the stage it
was measured at -- so no other correction is counted twice.

    python hf_response.py RUN_45A RUN_60A RUN_150A RUN_450A -o out/

LIMITS
------
It is RELATIVE: every channel is brought onto the plate median. What all
channels share at high frequency (the plate median itself) is set by the
per-card phase against the Gamry, not by this. An injected calibration
sweep through every channel would replace it with an absolute calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

F_LO_HZ = 1000.0
LEVEL_BAND_HZ = (100.0, 1000.0)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def load_run(run_dir) -> tuple[str, pd.DataFrame, dict]:
    """(condition, spectra with segment / card / fk / Z, config) of one run."""
    d = Path(run_dir)
    cfg = {}
    if (d / "config_used.json").is_file():
        cfg = json.loads((d / "config_used.json").read_text())
    sp = pd.read_csv(d / "silver" / "spectra_clean.csv")
    sp["segment"] = sp["segment"].astype(str)
    card = {}
    meta = d / "bronze" / "segment_meta.csv"
    if meta.is_file():
        m = pd.read_csv(meta)
        card = dict(zip(m["segment"].astype(str), m["card"].astype(str)))
    sp["card"] = sp["segment"].map(card).fillna("?")
    sp["fk"] = sp["freq_hz"].round(1)
    sp["Z"] = sp["z_re_mohm_cm2"].to_numpy() + 1j * sp["z_im_mohm_cm2"].to_numpy()
    return str(cfg.get("condition", d.name)), sp, cfg


def deviations(sp: pd.DataFrame) -> pd.DataFrame:
    """Each point against the plate median at its frequency (complex)."""
    g = sp.groupby("fk")
    med = g["z_re_mohm_cm2"].median() + 1j * g["z_im_mohm_cm2"].median()
    out = sp.copy()
    out["r"] = out["Z"].to_numpy() / out["fk"].map(med).to_numpy()
    return out


# ---------------------------------------------------------------------------
# estimate, validate, write
# ---------------------------------------------------------------------------

def _quad(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y smoothed by a quadratic in f (least squares): over 2-12.5 kHz the
    residual phase of every segment is a line in f to 0.9 deg and a
    quadratic to 0.4 deg. A running median would bend the steep top end."""
    m = np.isfinite(y)
    if m.sum() < 4:
        return _smooth3(y)
    c = np.polyfit(f[m] / 1e3, y[m], 2)
    return np.polyval(c, f / 1e3)


def _smooth3(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    for i in range(len(y)):
        w = y[max(0, i - 1):i + 2]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = float(np.median(w))
    return out


def estimate(runs: list[pd.DataFrame], f_lo: float = F_LO_HZ
             ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{segment: (freq, h)} from deviations() of several runs."""
    D = pd.concat(runs, ignore_index=True)
    D["ph"] = np.angle(D["r"].to_numpy())
    D["mag"] = np.abs(D["r"].to_numpy())
    lo, hi = LEVEL_BAND_HZ
    level = D[(D.freq_hz >= lo) & (D.freq_hz < hi)].groupby("card")["mag"].median()
    up = D[D.freq_hz >= f_lo]
    m_card = up.groupby(["card", "fk"])["mag"].median()
    ph_seg = up.groupby(["segment", "fk"])["ph"].median()
    card_of = dict(zip(D["segment"], D["card"]))
    out = {}
    for seg in sorted(ph_seg.index.get_level_values(0).unique(), key=_seg_key):
        p = ph_seg.loc[seg].sort_index()
        f = p.index.to_numpy(float)
        c = card_of.get(seg, "?")
        if c in m_card.index.get_level_values(0) and c in level.index:
            mc = m_card.loc[c].reindex(f).to_numpy(float) / float(level[c])
        else:
            mc = np.ones(f.size)
        mc = np.where(np.isfinite(mc) & (mc > 0), mc, 1.0)
        h = _quad(f, mc) * np.exp(1j * _quad(f, p.to_numpy(float)))
        out[seg] = (f, h)
    return out


def apply(sp: pd.DataFrame, h: dict) -> pd.DataFrame:
    """Divide each point at or above f_lo by its segment's h (nearest step)."""
    out = sp.copy()
    Z = out["Z"].to_numpy().copy()
    for i, (seg, fk) in enumerate(zip(out["segment"], out["fk"])):
        if seg in h:
            f, g = h[seg]
            k = np.flatnonzero(np.isclose(f, fk, rtol=1e-4))
            if k.size:
                Z[i] = Z[i] / g[k[0]]
    out["Z"] = Z
    out["z_re_mohm_cm2"] = Z.real
    out["z_im_mohm_cm2"] = Z.imag
    return out


def metrics(sp: pd.DataFrame, f_card: float = 4000.0) -> dict:
    """Phase spread at the top common step, largest card |Z| step above f_card."""
    d = deviations(sp)
    n = d.groupby("fk")["segment"].nunique()
    top = n[n >= 0.6 * d["segment"].nunique()].index.max()
    at = d[d.fk == top]
    ph_sd = float(np.degrees(np.std(np.angle(at["r"].to_numpy()))))
    hi = d[d.freq_hz >= f_card]
    step = hi.groupby("card")["r"].apply(lambda r: np.median(np.abs(r.to_numpy())) - 1)
    return {"f_top_hz": float(top), "phase_sd_deg": ph_sd,
            "card_step_max_pct": float(100 * np.max(np.abs(step))) if len(step) else float("nan")}


def validate(runs: dict[str, pd.DataFrame], f_lo: float = F_LO_HZ) -> list[dict]:
    """Leave one run out: estimate on the others, apply to it, compare."""
    rows = []
    names = list(runs)
    for name in names:
        others = [deviations(runs[o]) for o in names if o != name]
        if not others:
            continue
        h = estimate(others, f_lo)
        before, after = metrics(runs[name]), metrics(apply(runs[name], h))
        rows.append({"held_out": name, "f_top_hz": before["f_top_hz"],
                     "phase_sd_before_deg": round(before["phase_sd_deg"], 2),
                     "phase_sd_after_deg": round(after["phase_sd_deg"], 2),
                     "card_step_before_pct": round(before["card_step_max_pct"], 2),
                     "card_step_after_pct": round(after["card_step_max_pct"], 2)})
    return rows


def write(h: dict, path, f_lo: float = F_LO_HZ, note: str = "") -> Path:
    """The gain-file format utils.load_gain reads; h = 1 below f_lo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# high-frequency response per segment, frozen across "
                 "operating points (checks/hf_response.py); silver divides Z by it\n")
        if note:
            fh.write(f"# {note}\n")
        fh.write("segment,freq_hz,gain_real,gain_imag\n")
        for seg, (f, g) in h.items():
            # guard points: exactly 1 below f_lo, so nothing leaks downwards
            fh.write(f"{seg},0.01,1,0\n{seg},{0.999 * f_lo:.6g},1,0\n")
            for fi, gi in zip(f, g):
                fh.write(f"{seg},{fi:.6g},{gi.real:.8g},{gi.imag:.8g}\n")
    return path


def build(run_dirs, out_dir, f_lo: float = F_LO_HZ, log=print) -> dict:
    """Estimate from all runs, validate leave-one-out, write the file.

    Runs already evaluated WITH a response file are composed with it: the
    new estimate is the residual on top of that file, so the written file is
    old x residual and a second pass refines rather than undoes the first.
    """
    import utils
    runs, prior = {}, set()
    for d in run_dirs:
        name, sp, cfg = load_run(d)
        runs[name] = sp
        prior.add(str(cfg.get("hf_response_file") or ""))
    if len(runs) < 2:
        raise ValueError("needs at least two operating points")
    if len(prior) > 1:
        raise ValueError("the runs were evaluated with different response "
                         f"files ({sorted(prior)}): re-run them alike first")
    prior_file = next(iter(prior))
    val = validate(runs, f_lo)
    h = estimate([deviations(sp) for sp in runs.values()], f_lo)
    if prior_file and Path(prior_file).is_file():
        old = utils.load_gain(prior_file)
        for seg, (f, g) in h.items():
            h[seg] = (f, g * utils.gain_at(old, seg, f))
    out = Path(out_dir)
    path = write(h, out / "hf_response.csv", f_lo,
                 note=f"from {', '.join(runs)}"
                 + (f"; composed with {prior_file}" if prior_file else ""))
    pd.DataFrame(val).to_csv(out / "hf_response_validation.csv", index=False)
    if log:
        log(f"  high-frequency response from {len(runs)} operating points "
            f"({', '.join(runs)}), {len(h)} segments, from {f_lo:g} Hz up")
        log("  leave one out (estimated on the others, applied to it):")
        for r in val:
            log(f"    {r['held_out']:>6}: phase sd at {r['f_top_hz']:.0f} Hz "
                f"{r['phase_sd_before_deg']:.1f} -> {r['phase_sd_after_deg']:.1f} deg, "
                f"largest card step {r['card_step_before_pct']:.1f} -> "
                f"{r['card_step_after_pct']:.1f} %")
        log(f"  written: {path}")
    return {"path": path, "validation": val, "n_segments": len(h)}


def _seg_key(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return 10 ** 6


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("hf_response"))
    ap.add_argument("--f-lo", type=float, default=F_LO_HZ)
    a = ap.parse_args(argv)
    build(a.runs, a.out, a.f_lo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
