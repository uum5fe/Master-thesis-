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

ROLLED-OFF CHANNELS (mag="segment", the default)
------------------------------------------------
2612087 (150 / 450 A, after a first card-magnitude file) shows what the
card fingerprint misses: whole interface boxes of the plate whose channels
roll off like a low-pass, whatever card they land on. Box r5 (11-28, 52,
58, on cards 1, 2, 3 and 5) falls to |H| 0.42-0.55 by 6 kHz, box r6
(31-36, 60-72) to 0.47-0.74; r5's neighbours on the same cards stay at
1.0. The drop is the same at both currents (r = 0.995 at 3.8 kHz), it
grows with f, and at 7.5 kHz it puts Re Z at ~20 mOhm*cm2 against ~50 for
the plate -- below any ohmic resistance the cell can have. A card median
over half rolled-off and half healthy channels corrects neither half, and
the |Z| outlier gate then cuts the rolled-off segments' top steps, so their
band ends at 5.9-7.5 kHz and the extrapolated R_ohmic drops to ~24.

So a channel whose repeatable |H| at its top steps lies more than
`rolloff_min` (25 %) below its own 100-1000 Hz level gets its OWN
magnitude, everyone else keeps the card's:

  1. flag: per run, ln|Z_s / plate median| minus the segment's own level;
     median over runs; mean of the three highest steps < ln(1-rolloff_min),
     and, with two or more runs, an rms spread across runs <= rep_tol;
  2. the plate median is taken again WITHOUT the flagged channels (a third
     of the plate rolled off pulls the median down and makes the healthy
     boxes look +15-30 %), and step 1 is repeated once on it;
  3. card magnitude from the card's unflagged channels only;
  4. a flagged channel's magnitude: a first-order low-pass a / sqrt(1 +
     (f/fc)^2) fitted to its steps, if it fits (rms <= LOWPASS_TOL in
     ln|H|); then it is also written for the plate's steps ABOVE the
     channel's last kept one (up to 2x), so the next run can keep its top
     steps. Otherwise the quadratic of its own steps, nothing extrapolated.

The flagged channels are listed in hf_response_segments.csv beside the
file, and silver flags them ("hf_magnitude_per_segment"): their top-of-band
magnitude is set by the correction, not measured independently, so their
R_ohmic is a corrected value, not an independent one. Only a calibration
sweep through the boxes would make it one. mag="card" is the old file.

`validate` estimates h from all runs but one and applies it to the one left
out: the phase spread at the top step and the largest card step, before and
after. Only a correction that passes this blind test should be applied.

The file is written in the gain-file format (segment, freq_hz, gain_real,
gain_imag) with h = 1 below f_lo; silver divides each segment's corrected Z
by it (config.hf_response_file), AFTER the per-card phase -- the stage it
was measured at -- so no other correction is counted twice.

    python hf_response.py RUN_45A RUN_60A RUN_150A RUN_450A -o out/ [--mag card]

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
MAG_MODES = ("segment", "card")
#: a channel whose |H| at its top steps is this far below its own level
ROLLOFF_MIN = 0.25
#: ... and repeats across runs to within this (rms of ln|H|)
REP_TOL = 0.05
#: a first-order low-pass is used for a flagged channel if it fits this well
LOWPASS_TOL = 0.10
#: and is then extended at most this factor above the channel's top step
EXTEND_MAX = 2.0
SEGMENTS_CSV = "hf_response_segments.csv"


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


def deviations(sp: pd.DataFrame, exclude=()) -> pd.DataFrame:
    """Each point against the plate median at its frequency (complex),
    the median taken without the segments in `exclude`."""
    ref = sp[~sp["segment"].isin(set(exclude))] if len(exclude) else sp
    g = ref.groupby("fk")
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


def _quad_at(f: np.ndarray, y: np.ndarray, f_at: np.ndarray) -> np.ndarray:
    """The quadratic of _quad at f_at; above the last step it is continued
    along its tangent there, not along the parabola."""
    m = np.isfinite(y)
    if m.sum() < 4:
        return np.interp(f_at, f[m], y[m]) if m.any() else np.zeros(f_at.size)
    c = np.polyfit(f[m] / 1e3, y[m], 2)
    x, x1 = f_at / 1e3, float(f[m].max()) / 1e3
    tangent = np.polyval(c, x1) + np.polyval(np.polyder(c), x1) * (x - x1)
    return np.where(x <= x1, np.polyval(c, x), tangent)


def lowpass_fit(f: np.ndarray, lm: np.ndarray) -> tuple[float, float, float]:
    """(a, fc, rms) of ln|H| = ln a - 0.5 ln(1 + (f/fc)^2), least squares;
    fc on a log grid, ln a in closed form for each."""
    m = np.isfinite(lm)
    f, lm = f[m], lm[m]
    if f.size < 3:
        return 1.0, np.inf, np.inf
    best = (1.0, np.inf, np.inf)
    for fc in np.geomspace(200.0, 2e5, 241):
        shape = -0.5 * np.log1p((f / fc) ** 2)
        la = float(np.mean(lm - shape))
        rms = float(np.sqrt(np.mean((lm - la - shape) ** 2)))
        if rms < best[2]:
            best = (float(np.exp(la)), float(fc), rms)
    return best


def _log_mag(runs: list[pd.DataFrame]) -> pd.DataFrame:
    """ln|r| minus the segment's own 100-1000 Hz level, per run."""
    lo, hi = LEVEL_BAND_HZ
    parts = []
    for i, d in enumerate(runs):
        x = d[["segment", "card", "fk", "freq_hz"]].copy()
        x["run"] = i
        x["lm"] = np.log(np.abs(d["r"].to_numpy()))
        lvl = x[(x.freq_hz >= lo) & (x.freq_hz < hi)].groupby("segment")["lm"].median()
        x["lm"] = x["lm"] - x["segment"].map(lvl)
        parts.append(x[np.isfinite(x["lm"])])
    return pd.concat(parts, ignore_index=True)


def rolloff(runs: list[pd.DataFrame], f_lo: float = F_LO_HZ) -> pd.DataFrame:
    """Per segment: |H| at its three highest steps against its own level
    (median over runs) and the rms spread of ln|H| across runs above f_lo
    (NaN with one run)."""
    L = _log_mag(runs)
    up = L[L.freq_hz >= f_lo]
    if up.empty:
        return pd.DataFrame(columns=["top", "spread"])
    med = up.groupby(["segment", "fk"])["lm"].median().unstack().sort_index(axis=1)
    top = med.apply(lambda r: r.dropna().iloc[-3:].mean() if r.notna().any()
                    else np.nan, axis=1)
    sd = up.groupby(["segment", "fk"])["lm"].std(ddof=0)
    n = up.groupby(["segment", "fk"])["run"].nunique()
    sd = sd[n >= 2]
    spread = np.sqrt((sd ** 2).groupby("segment").mean()) if len(sd) else None
    out = pd.DataFrame({"top": np.exp(top)})
    out["spread"] = spread.reindex(out.index) if spread is not None else np.nan
    return out


def flag_rolled_off(runs: list[pd.DataFrame], f_lo: float = F_LO_HZ,
                    rolloff_min: float = ROLLOFF_MIN, rep_tol: float = REP_TOL
                    ) -> tuple[set, pd.DataFrame]:
    """Channels that roll off; the plate median re-taken without them once."""
    flagged: set = set()
    for _ in range(2):
        dev = [deviations(d, flagged) for d in runs]
        ro = rolloff(dev, f_lo)
        ok = ro["spread"].isna() | (ro["spread"] <= rep_tol)
        flagged = set(ro.index[(ro["top"] < 1.0 - rolloff_min) & ok])
    return flagged, ro


def estimate(runs: list[pd.DataFrame], f_lo: float = F_LO_HZ,
             mag: str = "segment", rolloff_min: float = ROLLOFF_MIN,
             rep_tol: float = REP_TOL, info: dict | None = None
             ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{segment: (freq, h)} from deviations() of several runs.

    mag="card": magnitude per card. mag="segment": the same, except for the
    channels flag_rolled_off finds, which get their own (module docstring).
    `info`, if given, receives the per-segment table (mag_source, ...)."""
    if mag not in MAG_MODES:
        raise ValueError(f"mag must be one of {MAG_MODES}, not {mag!r}")
    flagged, ro = set(), None
    if mag == "segment":
        flagged, ro = flag_rolled_off(runs, f_lo, rolloff_min, rep_tol)
        runs = [deviations(d, flagged) for d in runs]
    D = pd.concat(runs, ignore_index=True)
    D["ph"] = np.angle(D["r"].to_numpy())
    D["mag"] = np.abs(D["r"].to_numpy())
    H = D[~D["segment"].isin(flagged)]
    lo, hi = LEVEL_BAND_HZ
    level = H[(H.freq_hz >= lo) & (H.freq_hz < hi)].groupby("card")["mag"].median()
    up = D[D.freq_hz >= f_lo]
    m_card = up[~up["segment"].isin(flagged)].groupby(["card", "fk"])["mag"].median()
    ph_seg = up.groupby(["segment", "fk"])["ph"].median()
    card_of = dict(zip(D["segment"], D["card"]))
    f_plate = np.sort(up["fk"].unique().astype(float))
    lm_seg = None
    if flagged:
        L = _log_mag([d[d["segment"].isin(flagged)] for d in runs])
        lm_seg = L[L.freq_hz >= f_lo].groupby(["segment", "fk"])["lm"].median()
    out, rows = {}, []
    for seg in sorted(ph_seg.index.get_level_values(0).unique(), key=_seg_key):
        p = ph_seg.loc[seg].sort_index()
        f = p.index.to_numpy(float)
        c = card_of.get(seg, "?")
        row = {"segment": seg, "card": c, "mag_source": "card",
               "top_rel": np.nan, "rep_spread": np.nan, "fc_hz": np.nan,
               "lowpass_rms": np.nan, "f_max_hz": float(f.max()),
               "f_written_hz": float(f.max())}
        if ro is not None and seg in ro.index:
            row["top_rel"] = float(ro.at[seg, "top"])
            row["rep_spread"] = float(ro.at[seg, "spread"])
        if seg in flagged and lm_seg is not None and seg in lm_seg.index:
            lm = lm_seg.loc[seg].reindex(f).to_numpy(float)
            a, fc, rms = lowpass_fit(f, lm)
            row.update(mag_source="segment", fc_hz=fc, lowpass_rms=rms)
            ph = p.to_numpy(float)
            if rms <= LOWPASS_TOL:
                f_ext = f_plate[(f_plate > f.max()) & (f_plate <= EXTEND_MAX * f.max())]
                f_all = np.concatenate([f, f_ext])
                ms = a / np.sqrt(1.0 + (f_all / fc) ** 2)
                ph_all = _quad_at(f, ph, f_all)
                row["f_written_hz"] = float(f_all.max())
            else:
                f_all = f
                ms = np.exp(_quad(f, lm))
                ph_all = _quad(f, ph)
            out[seg] = (f_all, ms * np.exp(1j * ph_all))
            rows.append(row)
            continue
        if c in m_card.index.get_level_values(0) and c in level.index:
            mc = m_card.loc[c].reindex(f).to_numpy(float) / float(level[c])
        else:
            mc = np.ones(f.size)
        mc = np.where(np.isfinite(mc) & (mc > 0), mc, 1.0)
        h = _quad(f, mc) * np.exp(1j * _quad(f, p.to_numpy(float)))
        out[seg] = (f, h)
        rows.append(row)
    if info is not None:
        info["segments"] = pd.DataFrame(rows)
        info["flagged"] = sorted(flagged, key=_seg_key)
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
    """Phase spread at the top common step, largest card |Z| step above
    f_card, and the channels' |H| at their top steps against their own
    level: its spread and how many lie more than ROLLOFF_MIN below."""
    d = deviations(sp)
    n = d.groupby("fk")["segment"].nunique()
    top = n[n >= 0.6 * d["segment"].nunique()].index.max()
    at = d[d.fk == top]
    ph_sd = float(np.degrees(np.std(np.angle(at["r"].to_numpy()))))
    hi = d[d.freq_hz >= f_card]
    step = hi.groupby("card")["r"].apply(lambda r: np.median(np.abs(r.to_numpy())) - 1)
    rel = rolloff([d])["top"].dropna()
    return {"f_top_hz": float(top), "phase_sd_deg": ph_sd,
            "card_step_max_pct": float(100 * np.max(np.abs(step))) if len(step) else float("nan"),
            "mag_top_sd_pct": float(100 * np.std(np.log(rel))) if len(rel) else float("nan"),
            "n_rolled_off": int((rel < 1.0 - ROLLOFF_MIN).sum())}


def validate(runs: dict[str, pd.DataFrame], f_lo: float = F_LO_HZ,
             mag: str = "segment") -> list[dict]:
    """Leave one run out: estimate on the others, apply to it, compare."""
    rows = []
    names = list(runs)
    for name in names:
        others = [deviations(runs[o]) for o in names if o != name]
        if not others:
            continue
        h = estimate(others, f_lo, mag=mag)
        before, after = metrics(runs[name]), metrics(apply(runs[name], h))
        rows.append({"held_out": name, "f_top_hz": before["f_top_hz"],
                     "phase_sd_before_deg": round(before["phase_sd_deg"], 2),
                     "phase_sd_after_deg": round(after["phase_sd_deg"], 2),
                     "card_step_before_pct": round(before["card_step_max_pct"], 2),
                     "card_step_after_pct": round(after["card_step_max_pct"], 2),
                     "mag_top_sd_before_pct": round(before["mag_top_sd_pct"], 2),
                     "mag_top_sd_after_pct": round(after["mag_top_sd_pct"], 2),
                     "rolled_off_before": before["n_rolled_off"],
                     "rolled_off_after": after["n_rolled_off"]})
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


def build(run_dirs, out_dir, f_lo: float = F_LO_HZ, log=print,
          mag: str = "segment") -> dict:
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
    val = validate(runs, f_lo, mag)
    info: dict = {}
    h = estimate([deviations(sp) for sp in runs.values()], f_lo, mag=mag,
                 info=info)
    if prior_file and Path(prior_file).is_file():
        old = utils.load_gain(prior_file)
        for seg, (f, g) in h.items():
            h[seg] = (f, g * utils.gain_at(old, seg, f))
    out = Path(out_dir)
    path = write(h, out / "hf_response.csv", f_lo,
                 note=f"from {', '.join(runs)}"
                 + (f"; composed with {prior_file}" if prior_file else ""))
    pd.DataFrame(val).to_csv(out / "hf_response_validation.csv", index=False)
    seg_tab = info.get("segments", pd.DataFrame())
    seg_path = out / SEGMENTS_CSV
    seg_tab.round(4).to_csv(seg_path, index=False)
    flagged = info.get("flagged", [])
    if log:
        log(f"  high-frequency response from {len(runs)} operating points "
            f"({', '.join(runs)}), {len(h)} segments, from {f_lo:g} Hz up, "
            f"magnitude per {mag}")
        if mag == "segment":
            log(f"  rolled-off channels (|H| at the top > {100 * ROLLOFF_MIN:.0f} % "
                f"below their own level), own magnitude: {len(flagged)}"
                + (f" -- {', '.join(flagged)}" if flagged else ""))
            for r in seg_tab[seg_tab.mag_source == "segment"].itertuples():
                ext = (f", written up to {r.f_written_hz:.0f} Hz"
                       if r.f_written_hz > r.f_max_hz else "")
                log(f"    seg {r.segment:>3} ({str(r.card)[-7:]}): |H| top "
                    f"{r.top_rel:.2f}, low-pass fc {r.fc_hz:.0f} Hz "
                    f"(rms {r.lowpass_rms:.3f}){ext}")
        log("  leave one out (estimated on the others, applied to it):")
        for r in val:
            log(f"    {r['held_out']:>6}: phase sd at {r['f_top_hz']:.0f} Hz "
                f"{r['phase_sd_before_deg']:.1f} -> {r['phase_sd_after_deg']:.1f} deg, "
                f"largest card step {r['card_step_before_pct']:.1f} -> "
                f"{r['card_step_after_pct']:.1f} %, |H| spread at the top "
                f"{r['mag_top_sd_before_pct']:.1f} -> {r['mag_top_sd_after_pct']:.1f} %, "
                f"rolled off {r['rolled_off_before']} -> {r['rolled_off_after']}")
        log(f"  written: {path}")
    return {"path": path, "validation": val, "n_segments": len(h),
            "flagged": flagged, "segments_path": seg_path}


def segment_sources(hf_file) -> dict[str, str]:
    """{segment: 'segment' | 'card'} from the table build() writes beside
    the response file; {} if there is none (an older file)."""
    if not hf_file:
        return {}
    p = Path(hf_file).with_name(SEGMENTS_CSV)
    if not p.is_file():
        return {}
    t = pd.read_csv(p, dtype={"segment": str})
    return dict(zip(t["segment"], t["mag_source"].astype(str)))


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
    ap.add_argument("--mag", choices=MAG_MODES, default="segment",
                    help="magnitude per card, or per segment for rolled-off "
                         "channels (default)")
    a = ap.parse_args(argv)
    build(a.runs, a.out, a.f_lo, mag=a.mag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
