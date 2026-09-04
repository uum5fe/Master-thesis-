#!/usr/bin/env python3
"""
gamry_dta.py  --  Gamry Framework .DTA reader, and the chain-response export
============================================================================

Two jobs:

  1. Read any Gamry `.DTA` file (`ZCURVE` table) into (freq, Z).  That covers
     the whole-cell reference spectra used for validation.

  2. Turn the per-segment `bode/` sweeps that come with an Abgleich campaign
     into the `gain_file` the pipeline has always had a slot for and nobody
     has ever supplied.

WHAT THE bode/ FILES ACTUALLY ARE
---------------------------------
An Abgleich delivery looks like

    <plate>/coefficients/curr.csv        72 rows "c0;c1"   (DC calibration)
    <plate>/coefficients/temp.csv         4 rows "c0;c1"
    <plate>/Step<k>_<T>Grad.csv          the DC sweeps those were fitted from
    <plate>/bode/<name>_100kHz_1Hz_500mA_#<n>.DTA        <- these
    <plate>/bode/<name>_..._#<n>_Raw.DTA                 (raw waveforms)

Each `#n` is one segment's **current-measurement chain**, swept
galvanostatically at 500 mA rms from 100 kHz down to 1 Hz with no DC bias.
It is not electrochemistry: it is the shunt plus its amplifier, measured
ex-situ.  On both plates the response is flat below ~1 kHz and then rolls
off -- measured on the delivered files:

    f          |H|      arg H
    1 kHz     1.000     -2.5 deg
    4.5 kHz   ~0.99    ~-11   deg          <- top of the pipeline's band
    10 kHz     0.94    -24    deg
    100 kHz    0.24   -115    deg

THIS MATTERS, AND IT IS THE OPPOSITE OF A SMALL CORRECTION.
`config.f_max_hz` is 4500 Hz.  Eleven degrees of uncorrected phase at the top
of the band is the same order as the acquisition skew the pipeline works hard
to measure and remove -- and unlike a skew it is NOT all-pass, so it moves
|Z| as well.  It biases exactly the decade that sets R_omega, which is the
number the whole measurement exists to produce.  With the sweep in hand the
correction is a division; without it, the top decade carries a systematic
error that no amount of synchronisation fixes.

SIGN CONVENTION -- read this before exporting
---------------------------------------------
Let H(f) be the chain response normalised to 1 at low frequency, so the
recorded shunt voltage is  u_s = K * H(f) * j.  The pipeline forms

    j_hat = u_s / K = H * j            (the DC Abgleich knows nothing of f)
    Z_meas = U_cell / j_hat = Z_true / H

and `eis_local.segment_spectrum` / `utils.gain_at` then apply

    Z <- Z / gain

so the file must carry **gain = 1/H**, not H.  `write_gain_csv` does that by
default (`convention="pipeline"`); pass `convention="chain"` to write H
itself, e.g. for plotting.  Getting this backwards doubles the error instead
of removing it, which is why it is a named argument and not an assumption.

CROSS-CHECK BEFORE YOU TRUST IT
-------------------------------
`cross_check_abgleich()` correlates each segment's low-frequency |Z| from the
bode sweep against `c0` from `curr.csv`.  Both are the DC resistance of the
same per-segment path, so on a consistent delivery they track each other
almost perfectly.  Measured on the files delivered with these plates:

    Kashyyyk (gen1)   r = +0.999, ratio 3.82 +/- 0.6 %      consistent
    Naboo    (gen2)   r = +0.41,  ratio 6.10 +/- 12 %       NOT consistent

So on gen1 the bode index and the curr.csv row index refer to the same
segment and the export is safe.  On gen2 they do not agree, and the gen2
`curr.csv` is also about 1.45x lower in absolute terms than the gen1 one
while the bode sweeps of the two plates are nearly identical.  That is a
delivery/ordering question about the Naboo Abgleich, not something this
module can resolve -- so `write_gain_csv` refuses a suspicious pairing unless
`force=True`, and says why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Gamry writes with the locale of the machine that recorded it; the delivered
# files use a decimal comma.  Detecting per-field is safer than assuming.
_NUM = re.compile(r"^[-+]?[\d.,]+([eE][-+]?\d+)?$")


def _f(s: str) -> float:
    s = s.strip()
    if "," in s and "." in s:
        # "1.234,5" -> German thousands; "1,234.5" -> English thousands
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") \
            else s.replace(",", "")
    else:
        s = s.replace(",", ".")
    return float(s)


@dataclass(frozen=True)
class GamrySweep:
    path: Path
    segment: str | None            # from "#<n>" in the file name, if present
    freq: np.ndarray               # Hz, as recorded (high -> low)
    Z: np.ndarray                  # complex ohm
    meta: dict

    def sorted(self) -> "GamrySweep":
        o = np.argsort(self.freq)
        return GamrySweep(self.path, self.segment, self.freq[o], self.Z[o],
                          self.meta)

    @property
    def z_lf(self) -> complex:
        """Impedance at the lowest recorded frequency."""
        return complex(self.Z[np.argmin(self.freq)])

    def normalised(self) -> np.ndarray:
        """H(f): the sweep divided by its own low-frequency value."""
        return self.Z / self.z_lf


def read_dta(path) -> GamrySweep:
    """Parse the ZCURVE table of a Gamry .DTA file.

    Columns are Pt, Time, Freq, Zreal, Zimag, Zsig, Zmod, Zphz, ... .  The
    two header rows (names, units) are skipped by requiring the second field
    of a data row to be an integer point index, which also skips any trailing
    notes block.
    """
    path = Path(path)
    lines = path.read_text(encoding="latin-1", errors="replace").splitlines()

    meta: dict = {}
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith("ZCURVE"):
            start = i
            break
        parts = ln.split("\t")
        if len(parts) >= 3 and parts[0].strip():
            meta[parts[0].strip()] = parts[2].strip()
    if start is None:
        raise ValueError(f"{path.name}: no ZCURVE table "
                         f"(a _Raw.DTA holds waveforms, not a spectrum)")

    freq, zre, zim = [], [], []
    for ln in lines[start + 1:]:
        c = ln.split("\t")
        if len(c) < 6 or not c[1].strip().lstrip("+-").isdigit():
            continue
        try:
            freq.append(_f(c[3]))
            zre.append(_f(c[4]))
            zim.append(_f(c[5]))
        except ValueError:
            continue
    if not freq:
        raise ValueError(f"{path.name}: ZCURVE table is empty")

    m = re.search(r"#(\d+)", path.stem)
    return GamrySweep(path, m.group(1) if m else None,
                      np.asarray(freq, float),
                      np.asarray(zre, float) + 1j * np.asarray(zim, float),
                      meta)


def read_bode_folder(folder, skip_raw: bool = True) -> dict[str, GamrySweep]:
    """Read every per-segment sweep in a `bode/` folder -> {segment: sweep}."""
    folder = Path(folder)
    out: dict[str, GamrySweep] = {}
    for p in sorted(folder.glob("*.DTA")):
        if skip_raw and p.stem.endswith("_Raw"):
            continue
        try:
            s = read_dta(p).sorted()
        except ValueError:
            continue
        if s.segment is None:
            continue
        out[s.segment] = s
    return out


# ---------------------------------------------------------------------------
# Chain response export
# ---------------------------------------------------------------------------


def cross_check_abgleich(sweeps: dict[str, GamrySweep], curr_csv) -> dict:
    """Do the bode sweeps and the DC Abgleich describe the same segments?

    Both measure the DC resistance of one segment's current path, so
    |Z(f_min)| from the sweep and c0 from curr.csv must be proportional with
    a per-segment scatter of a few tenths of a percent.  Returns the
    correlation, the ratio and its spread, plus a verdict.
    """
    import eis_local
    rows = eis_local._read_pairs(curr_csv)
    c0 = {str(i): a for i, (a, _b) in enumerate(rows, start=1)}

    keys = [k for k in sweeps if k in c0]
    if len(keys) < 8:
        return {"ok": False, "n": len(keys), "reason": "too few common segments"}

    lf = np.array([abs(sweeps[k].z_lf) for k in keys])
    dc = np.array([c0[k] for k in keys])
    r = float(np.corrcoef(lf, dc)[0, 1])
    ratio = lf / dc
    cv = float(np.std(ratio) / np.mean(ratio))
    ok = r > 0.95 and cv < 0.03
    return {
        "ok": ok, "n": len(keys), "corr": r,
        "ratio_mean": float(np.mean(ratio)), "ratio_cv": cv,
        "reason": "" if ok else (
            f"bode |Z(f_min)| and curr.csv c0 correlate only r={r:+.3f} with "
            f"{100*cv:.1f} % ratio scatter. They measure the same per-segment "
            f"path, so on a consistent delivery r > 0.99 and the scatter is "
            f"below 1 %. Either the bode '#n' index and the curr.csv row "
            f"order refer to different segments, or the two were recorded "
            f"with different amplifier settings. Resolve that before using "
            f"the gain file."),
    }


def _normalised(sw: GamrySweep, f_ref_hz: float | None) -> np.ndarray:
    sw = sw.sorted()
    if f_ref_hz is None:
        z0 = sw.z_lf
    else:
        z0 = complex(np.interp(f_ref_hz, sw.freq, sw.Z.real)
                     + 1j * np.interp(f_ref_hz, sw.freq, sw.Z.imag))
    return sw.Z / z0


def median_response(sweeps: dict[str, GamrySweep], f_ref_hz=None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """The plate's shared chain response: median |H| and median arg H.

    Every segment sees the same amplifier design, and the measured
    segment-to-segment spread is 1.9 deg at 4.5 kHz against a common
    -11.2 deg.  So the median carries almost all of the correction and none
    of the per-segment index risk -- which is what makes it the right export
    when the bode numbering has not been reconciled with the Abgleich.
    """
    ref = next(iter(sweeps.values())).sorted().freq
    mags, phs = [], []
    for sw in sweeps.values():
        s = sw.sorted()
        H = _normalised(s, f_ref_hz)
        mags.append(np.interp(ref, s.freq, np.abs(H)))
        phs.append(np.interp(ref, s.freq, np.unwrap(np.angle(H))))
    m = np.median(np.vstack(mags), axis=0)
    p = np.median(np.vstack(phs), axis=0)
    return ref, m * np.exp(1j * p)


def write_gain_csv(sweeps: dict[str, GamrySweep], path,
                   convention: str = "pipeline",
                   f_ref_hz: float | None = None,
                   curr_csv=None, force: bool = False,
                   shared: bool = False) -> Path:
    """Write `segment,freq_hz,gain_real,gain_imag` for utils.load_gain.

    convention="pipeline"  gain = 1/H, the factor the pipeline DIVIDES Z by
    convention="chain"     gain = H, the response itself (for plots)

    `f_ref_hz` picks the normalisation point; the default is the lowest
    frequency in each sweep, where the chain is flat.  Pass `curr_csv` to run
    cross_check_abgleich() first -- a failed check raises unless force=True
    or shared=True.

    `shared=True` writes ONE curve under segment "all", the median over the
    plate.  Use it when the per-segment bode index is not reconciled with the
    Abgleich row order: the median is index-free, and it still removes the
    -11 deg of common phase at 4.5 kHz that is the bulk of the error.
    """
    if convention not in ("pipeline", "chain"):
        raise ValueError(f"convention must be 'pipeline' or 'chain', "
                         f"not {convention!r}")
    if curr_csv is not None and not shared:
        chk = cross_check_abgleich(sweeps, curr_csv)
        if not chk["ok"] and not force:
            raise ValueError(chk.get("reason", "cross-check failed")
                             + "  Pass force=True to write it anyway, or "
                               "shared=True to write the index-free plate "
                               "median instead.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# ex-situ measuring-chain response, from Gamry bode sweeps\n")
        fh.write(f"# convention={convention} "
                 f"({'gain = 1/H, Z is divided by it' if convention == 'pipeline' else 'gain = H itself'})\n")
        fh.write(f"# scope={'plate median (index-free)' if shared else 'per segment'}\n")
        fh.write("segment,freq_hz,gain_real,gain_imag\n")
        if shared:
            f, H = median_response(sweeps, f_ref_hz)
            g = (1.0 / H) if convention == "pipeline" else H
            for fi, v in zip(f, g):
                fh.write(f"all,{fi:.6g},{v.real:.8g},{v.imag:.8g}\n")
            return path
        for seg in sorted(sweeps, key=lambda s: int(s)):
            sw = sweeps[seg].sorted()
            H = _normalised(sw, f_ref_hz)
            g = (1.0 / H) if convention == "pipeline" else H
            for f, v in zip(sw.freq, g):
                fh.write(f"{seg},{f:.6g},{v.real:.8g},{v.imag:.8g}\n")
    return path


def chain_summary(sweeps: dict[str, GamrySweep],
                  at_hz=(1e3, 4.5e3, 1e4, 1e5)) -> list[dict]:
    """Median |H| and phase across segments at a few frequencies.

    Use it to see, in one line each, how much of the band the chain response
    actually touches before deciding whether to apply it.
    """
    out = []
    for f in at_hz:
        mags, phs = [], []
        for sw in sweeps.values():
            sw = sw.sorted()
            H = sw.normalised()
            mags.append(float(np.interp(f, sw.freq, np.abs(H))))
            phs.append(float(np.degrees(
                np.interp(f, sw.freq, np.unwrap(np.angle(H))))))
        out.append({"freq_hz": f,
                    "mag_median": float(np.median(mags)),
                    "phase_deg_median": float(np.median(phs)),
                    "phase_deg_spread": float(np.percentile(phs, 95)
                                              - np.percentile(phs, 5))})
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("bode_folder")
    p.add_argument("-o", "--out", default="chain_gain.csv")
    p.add_argument("--curr-cal", default=None)
    p.add_argument("--convention", choices=["pipeline", "chain"],
                   default="pipeline")
    p.add_argument("--force", action="store_true")
    p.add_argument("--shared", action="store_true",
                   help="write the index-free plate median as segment 'all'")
    a = p.parse_args()

    sw = read_bode_folder(a.bode_folder)
    print(f"{len(sw)} segment sweeps, "
          f"{len(next(iter(sw.values())).freq)} frequencies")
    for row in chain_summary(sw):
        print(f"  {row['freq_hz']:9.0f} Hz  |H| = {row['mag_median']:.4f}  "
              f"arg H = {row['phase_deg_median']:+7.2f} deg  "
              f"(p5..p95 spread {row['phase_deg_spread']:.2f} deg)")
    if a.curr_cal:
        chk = cross_check_abgleich(sw, a.curr_cal)
        print(f"  cross-check vs {a.curr_cal}: "
              f"r={chk.get('corr', float('nan')):+.4f} "
              f"ratio {chk.get('ratio_mean', float('nan')):.3f} "
              f"+/- {100*chk.get('ratio_cv', float('nan')):.1f} %  "
              f"{'OK' if chk['ok'] else 'SUSPECT'}")
        if not chk["ok"]:
            print("  " + chk["reason"])
    try:
        out = write_gain_csv(sw, a.out, convention=a.convention,
                             curr_csv=a.curr_cal, force=a.force,
                             shared=a.shared)
    except ValueError as exc:
        print(f"\n  NOT WRITTEN: {exc}")
        raise SystemExit(1)
    print(f"  written: {out}")
