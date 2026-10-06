#!/usr/bin/env python3
"""
diagnose_channel_lag.py
=======================
Does every segment channel see the same high-frequency phase, or does each
current-measurement chain add its own lag?  And how much of the R_ohmic map
is that lag rather than the cell?

WHY THIS EXISTS
---------------
On 2612030 the HFR map showed neighbours 20 mOhm*cm2 apart (14: 66.5, 15:
53.3, 19: 44.0, 23: 45.8 at 60 A). Their spectra agree below ~100 Hz and part
above it: the phase of Z_segment / Z_plate-median grows LINEARLY with
frequency (a time constant, not electrochemistry) and |Z| drops by about the
cosine of that phase -- the signature of a first-order lag
H = 1 / (1 + j w tau) in the current path (shunt L/R, amplifier, wiring).
Silver reads R_ohmic as Re Z at the top of the kept band (~1.2 kHz), so a
channel lagging by 37 deg there reads ~20 mOhm*cm2 low and one leading
reads high. The fitted tau is the same at 45 A and 60 A to within a few us
(correlation 1.00 over all segments), i.e. a property of the channel.

WHAT IT DOES
------------
For each run folder (bronze/raw_spectra.csv + gold/plate_summary.csv):

  * tau per segment from channel_lag.estimate -- the same code the pipeline
    runs in silver -- on silver's kept points: arg(Z_seg / Z_ref) =
    -atan(w tau) over f_lo..f_hi, Z_ref the plate median;
  * a diagnostic R_s per segment from a Randles-CPE fit to 100 Hz..f_hi,
    once on the raw spectrum and once with the lag removed, Z * (1 + j w tau);
  * across runs: tau side by side and their correlation (repeatability).

  python diagnose_channel_lag.py RUN [RUN ...] -o out_dir [--write-gain g.csv]

`--write-gain` writes the median tau per segment as a pipeline gain file
(segment,freq_hz,gain_real,gain_imag with G = 1/(1 + j w tau)); set it as
cfg.gain_file and bronze divides each segment's Z by it.

WHAT IT IS NOT
--------------
tau is measured RELATIVE TO THE PLATE MEDIAN. It removes the channel-to-
channel differences, not a lag every channel shares, and it would also flatten
any real local high-frequency difference. Treat the gain file as an interim
until the chains are measured ex-situ (gamry_dta.py on the Abgleich bode/
sweeps, or one signal fed to every input in parallel) -- that is the
calibration; this is the evidence that it is needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _find_run(path: Path) -> Path:
    path = Path(path)
    if (path / "bronze" / "raw_spectra.csv").is_file():
        return path
    hits = sorted(path.glob("*/bronze/raw_spectra.csv"))
    if not hits:
        raise FileNotFoundError(f"no bronze/raw_spectra.csv under {path}")
    return hits[0].parent.parent


def _label(run: Path) -> str:
    try:
        c = json.loads((run / "config_used.json").read_text())
        return f"{c.get('leepa', '')} {c.get('condition', '')}".strip()
    except Exception:                                       # noqa: BLE001
        return run.name


def load_spectra(run: Path, f_max: float):
    """(freq, Z[f, seg] in mOhm*cm2, segments, usable[f, seg]).

    `usable` is silver's verdict per point (silver/point_rejections.csv) when
    the run has one, else every finite point: the lag must be fitted on the
    points silver models, or the rejected top-of-band noise decides it.
    """
    raw = pd.read_csv(run / "bronze" / "raw_spectra.csv")
    raw = raw[raw.freq_hz <= f_max]
    re_ = raw.pivot_table(index="freq_hz", columns="segment",
                          values="z_re_ohm_cm2")
    im_ = raw.pivot_table(index="freq_hz", columns="segment",
                          values="z_im_ohm_cm2")
    Z = (re_.values + 1j * im_.values) * 1e3
    usable = np.isfinite(Z)
    pr = run / "silver" / "point_rejections.csv"
    if pr.is_file():
        k = pd.read_csv(pr)
        k = k.assign(freq_hz=k.freq_hz.round(3)).pivot_table(
            index="freq_hz", columns="segment", values="kept")
        k = k.reindex(index=np.round(re_.index.values, 3), columns=re_.columns)
        usable &= (k.values == 1)
    return (re_.index.values.astype(float), Z, [str(s) for s in re_.columns],
            usable)


def plate_reference(Z: np.ndarray) -> np.ndarray:
    return np.nanmedian(Z.real, axis=1) + 1j * np.nanmedian(Z.imag, axis=1)


def fit_rs(freq, z, f_lo: float = 100.0) -> float:
    """R_s from R + 1/(1/R_ct + Q (j w)^n) over freq >= f_lo (diagnostic)."""
    from scipy.optimize import least_squares
    m = (freq >= f_lo) & np.isfinite(z)
    if m.sum() < 5:
        return float("nan")
    w, zz = 2 * np.pi * freq[m], z[m]

    def res(p):
        R, Rct, lq, n = p
        zm = R + 1.0 / (1.0 / Rct + np.exp(lq) * (1j * w) ** n)
        d = (zm - zz) / np.abs(zz)
        return np.r_[d.real, d.imag]

    r = least_squares(res, [50.0, 200.0, np.log(1e-3), 0.85],
                      bounds=([0, 1, -30, 0.4], [500, 5000, 5, 1.0]))
    return float(r.x[0])


def analyse_run(run, f_lo: float = 50.0, f_max: float = 1200.0) -> pd.DataFrame:
    """Per segment: the pipeline's own lag estimate (channel_lag.estimate, in
    "report" mode) and a diagnostic R_s with and without it."""
    import channel_lag
    from config import DEFAULT
    run = _find_run(run)
    freq, Z, segs, usable = load_spectra(run, f_max)
    cfg = DEFAULT.replace(channel_lag="report", channel_lag_f_lo_hz=f_lo,
                          f_max_hz=f_max)
    lags = channel_lag.estimate(
        {s: (freq, Z[:, j], usable[:, j]) for j, s in enumerate(segs)}, cfg)
    rows = []
    for j, s in enumerate(segs):
        g = lags.get(s)
        z = np.where(usable[:, j], Z[:, j], np.nan)
        tau = g.tau_s if g is not None else float("nan")
        zc = z * channel_lag.correction(freq, tau) if np.isfinite(tau) else z
        rows.append(dict(segment=s, tau_us=1e6 * tau,
                         status=g.status if g is not None else "",
                         resid_deg=g.resid_deg if g is not None else np.nan,
                         dphi_top_deg=(g.dphi_top_deg if g is not None
                                       else np.nan),
                         Rs_fit_raw=fit_rs(freq, z),
                         Rs_fit_lag_removed=fit_rs(freq, zc)))
    out = pd.DataFrame(rows).set_index("segment")
    ps = run / "gold" / "plate_summary.csv"
    if ps.is_file():
        g = pd.read_csv(ps, dtype={"segment": str}).set_index("segment")
        out = out.join(g[[c for c in ("class", "R_ohmic", "R_ohmic_sd")
                          if c in g.columns]])
    out.attrs["label"] = _label(run)
    out.attrs["f_top"] = float(freq.max())
    return out


def gain_rows(tau_s: dict[str, float], f_min=0.1, f_max=1e4, n=200):
    """Gain-file rows G = 1/(1 + j w tau) per segment, on a log grid."""
    f = np.geomspace(f_min, f_max, n)
    rows = []
    for s, t in sorted(tau_s.items(), key=lambda kv: int(kv[0])):
        if not np.isfinite(t):
            continue
        g = 1.0 / (1.0 + 1j * 2 * np.pi * f * t)
        rows += [(s, fi, gi.real, gi.imag) for fi, gi in zip(f, g)]
    return rows


def _stats(v) -> str:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return (f"mean {v.mean():.1f}  sd {v.std(ddof=1):.1f}  "
            f"CV {100 * v.std(ddof=1) / v.mean():.1f} %  "
            f"range {v.min():.1f}..{v.max():.1f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("channel_lag"))
    ap.add_argument("--f-lo", type=float, default=50.0)
    ap.add_argument("--f-max", type=float, default=1200.0,
                    help="top of the band used (Hz); above it this data is "
                         "SNR/THD-rejected anyway")
    ap.add_argument("--write-gain", type=Path,
                    help="write the median tau per segment as a gain file")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    taus = {}
    for r in a.runs:
        df = analyse_run(r, a.f_lo, a.f_max)
        lab = df.attrs["label"]
        taus[lab] = df.tau_us
        df.round(3).to_csv(a.out / f"channel_lag_{lab.replace(' ', '_')}.csv")
        meas = df[df.get("class", "measured") == "measured"] \
            if "class" in df else df
        print(f"\n== {lab}  (band {a.f_lo:g}..{df.attrs['f_top']:.0f} Hz)")
        print(f"  tau [us]           {_stats(meas.tau_us)}")
        if "R_ohmic" in meas:
            print(f"  R_ohmic pipeline   {_stats(meas.R_ohmic)}")
        print(f"  R_s fit, raw       {_stats(meas.Rs_fit_raw)}")
        print(f"  R_s fit, lag removed {_stats(meas.Rs_fit_lag_removed)}")
    T = pd.DataFrame(taus)
    T.round(2).to_csv(a.out / "channel_lag_all_runs.csv")
    if T.shape[1] > 1:
        print("\n  tau repeatability (correlation between runs):")
        print(T.corr().round(2).to_string())
    if a.write_gain:
        med = T.median(axis=1) * 1e-6
        rows = gain_rows(med.to_dict())
        with open(a.write_gain, "w", encoding="utf-8") as fh:
            fh.write("# interim chain response from diagnose_channel_lag.py: "
                     "G = 1/(1 + j w tau), tau relative to the plate median "
                     f"of {', '.join(T.columns)}\n")
            fh.write("segment,freq_hz,gain_real,gain_imag\n")
            for s, f, gr, gi in rows:
                fh.write(f"{s},{f:.6g},{gr:.8f},{gi:.8f}\n")
        print(f"\n  gain file: {a.write_gain} ({len(med)} segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
