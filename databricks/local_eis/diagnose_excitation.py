#!/usr/bin/env python3
"""
diagnose_excitation.py
======================
Answer one question from a recording, with no assumptions:

    Is the excitation a STEPPED SINE (one frequency at a time) or a
    BROADBAND / MULTISINE (all frequencies present at once)?

This matters more than it sounds. The two require different estimators, and
using the wrong one does not fail loudly -- it returns a spectrum that looks
plausible at low frequency and is noise at high frequency.

    stepped sine  -> locate each dwell, sine-fit inside it
                     (Welch over the whole record dilutes the estimate,
                      because each frequency occupies a small slice of time)

    multisine     -> Welch / cross-spectral density over the whole record
                     (dwell detection finds nothing to detect, because every
                      frequency is present throughout)

HOW THE TEST WORKS
------------------
Compute a spectrogram of the reference channel. For each strong spectral
peak, measure its OCCUPANCY: the fraction of time frames in which that
frequency carries most of its own peak power.

    occupancy ~ 1.0            every tone is on the whole time -> MULTISINE
    occupancy ~ 1/n_tones      each tone is on for its own slice -> STEPPED
    occupancy in between       mixed, or a swept sine (chirp)

The test needs no metadata: not the frequency list, not the sweep settings,
not the instrument configuration.

USAGE
-----
    python diagnose_excitation.py /path/to/file.DAT
    python diagnose_excitation.py /path/to/file.DAT --channel UC1
    python diagnose_excitation.py --self-test       # synthetic demonstration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))


# ---------------------------------------------------------------------------


def occupancy_test(x: np.ndarray, fs: float, f_lo: float = 0.5,
                   f_hi: float | None = None, n_frames: int = 60,
                   peak_db: float = 12.0, on_db: float = 10.0) -> dict:
    """Fraction of the record over which each strong tone is present."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    f_hi = float(f_hi or 0.4 * fs)

    # frame the record; each frame long enough to resolve the lowest tone
    nper = int(max(256, min(n // n_frames, n // 4)))
    nper = 2 ** int(np.floor(np.log2(nper)))
    step = max(1, nper // 2)
    starts = list(range(0, n - nper + 1, step))
    if len(starts) < 8:
        return {"ok": False, "note": "record too short to frame"}

    win = np.hanning(nper)
    freqs = np.fft.rfftfreq(nper, 1.0 / fs)
    band = (freqs >= f_lo) & (freqs <= f_hi)

    S = np.empty((len(starts), int(band.sum())))
    for i, a in enumerate(starts):
        seg = x[a:a + nper] * win
        S[i] = np.abs(np.fft.rfft(seg))[band]
    fb = freqs[band]

    # global spectrum -> where are the tones?
    g = S.max(axis=0)
    if not np.isfinite(g).any() or g.max() <= 0:
        return {"ok": False, "note": "no signal"}
    g_db = 20 * np.log10(np.maximum(g, 1e-30) / g.max())
    noise_db = float(np.median(g_db))

    # a tone is a local maximum well above the median level
    peaks = []
    for k in range(1, len(fb) - 1):
        if g_db[k] < noise_db + peak_db:
            continue
        if g[k] >= g[k - 1] and g[k] >= g[k + 1]:
            peaks.append(k)
    # thin out peaks that are adjacent bins of the same tone
    thinned = []
    for k in peaks:
        if thinned and fb[k] / fb[thinned[-1]] < 1.02:
            if g[k] > g[thinned[-1]]:
                thinned[-1] = k
            continue
        thinned.append(k)
    peaks = thinned
    if len(peaks) < 3:
        return {"ok": False, "note": f"only {len(peaks)} tones found"}

    occ = []
    for k in peaks:
        col = S[:, k]
        thr = col.max() / (10 ** (on_db / 20.0))
        occ.append(float(np.mean(col >= thr)))
    occ = np.array(occ)

    med = float(np.median(occ))
    expected_stepped = 1.0 / len(peaks)

    # verdict
    if med > 0.75:
        verdict, conf = "MULTISINE / BROADBAND", "high"
    elif med < max(3.0 * expected_stepped, 0.20):
        verdict, conf = "STEPPED SINE", "high"
    else:
        verdict, conf = "AMBIGUOUS (swept sine, or few long steps?)", "low"

    return {
        "ok": True, "verdict": verdict, "confidence": conf,
        "n_tones": len(peaks), "occupancy_median": med,
        "occupancy_min": float(occ.min()), "occupancy_max": float(occ.max()),
        "expected_if_stepped": expected_stepped,
        "freqs": fb[peaks], "occ": occ,
        "fs": fs, "duration_s": n / fs, "n_frames": len(starts),
        "noise_floor_db": noise_db,
    }


def grid_check(freqs: np.ndarray) -> str:
    """Are the detected tones logarithmically or linearly spaced?"""
    f = np.sort(np.asarray(freqs, float))
    f = f[f > 0]
    if len(f) < 4:
        return "too few tones to judge spacing"
    log_r = np.diff(np.log(f))
    lin_d = np.diff(f)
    cv_log = np.std(log_r) / max(abs(np.mean(log_r)), 1e-12)
    cv_lin = np.std(lin_d) / max(abs(np.mean(lin_d)), 1e-12)
    if cv_log < cv_lin and cv_log < 0.25:
        ppd = 1.0 / (np.mean(log_r) / np.log(10))
        return (f"LOGARITHMIC, about {ppd:.1f} points/decade "
                f"(ratio {np.exp(np.mean(log_r)):.4f})")
    if cv_lin < 0.25:
        return f"LINEAR, spacing about {np.mean(lin_d):.3f} Hz"
    return "irregular -- neither cleanly logarithmic nor linear"


def report(r: dict) -> None:
    if not r.get("ok"):
        print(f"  inconclusive: {r.get('note')}")
        return
    print(f"  sampling rate      {r['fs']:.0f} Hz")
    print(f"  record length      {r['duration_s']:.1f} s  "
          f"({r['n_frames']} analysis frames)")
    print(f"  distinct tones     {r['n_tones']}")
    print(f"  band covered       {r['freqs'].min():.2f} .. {r['freqs'].max():.1f} Hz")
    print(f"  tone spacing       {grid_check(r['freqs'])}")
    print()
    print(f"  OCCUPANCY (fraction of the record each tone is present)")
    print(f"    median {r['occupancy_median']:.3f}   "
          f"range {r['occupancy_min']:.3f} .. {r['occupancy_max']:.3f}")
    print(f"    would be ~{r['expected_if_stepped']:.3f} if stepped "
          f"({r['n_tones']} tones sharing the record)")
    print(f"    would be ~1.000 if broadband")
    print()
    print(f"  >>> VERDICT: {r['verdict']}   (confidence: {r['confidence']})")
    print()
    if r["verdict"].startswith("MULTISINE"):
        print("  Use a cross-spectral estimator over the whole record")
        print("  (Welch / CSD with coherence).  Dwell detection does not apply:")
        print("  there are no dwells to find.")
    elif r["verdict"].startswith("STEPPED"):
        print("  Use dwell localisation plus a sine fit inside each dwell.")
        print("  Welch over the whole record would dilute every estimate.")
    else:
        print("  Inspect the spectrogram by hand before choosing an estimator.")
        print("  A linear sweep (chirp) lands here, and needs different handling")
        print("  again.")


# ---------------------------------------------------------------------------


def run_file(path: str, channel: str | None) -> int:
    from eis_local import open_famos
    fam = open_famos(Path(path))
    print(f"\n  file               {Path(path).name}")
    print(f"  channels           {fam.n_ch}  "
          f"(segments {len(fam.segment_names)}, refs {fam.uc_names})")
    if channel is None:
        if not fam.uc_names:
            print("  no UC reference channel; pass --channel explicitly")
            return 2
        channel = max(fam.uc_names,
                      key=lambda c: float(np.std(fam.channel(c)[::10])))
    print(f"  analysing channel  {channel}")
    x = fam.channel(channel)
    r = occupancy_test(x, fam.fs)
    report(r)

    # the reference channel may be quiet; the segments carry the response
    if not r.get("ok") and fam.segment_names:
        seg = fam.segment_names[0]
        print(f"\n  retrying on segment channel {seg}")
        report(occupancy_test(fam.channel(seg), fam.fs))
    return 0


def self_test() -> int:
    """Demonstrate the test on synthetic records of each kind."""
    fs = 10000.0
    rng = np.random.default_rng(0)
    freqs = np.logspace(np.log10(1), np.log10(3000), 18)
    dwell = int(2 * fs)
    n = len(freqs) * dwell
    t = np.arange(n) / fs

    stepped = np.zeros(n)
    for k, f in enumerate(freqs):
        a, b = k * dwell, (k + 1) * dwell
        stepped[a:b] = 0.004 * np.cos(2 * np.pi * f * t[a:b])
    stepped += 2e-4 * rng.normal(size=n)

    multi = np.zeros(n)
    for f in freqs:
        multi += (0.004 / np.sqrt(len(freqs))) * np.cos(
            2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    multi += 2e-4 * rng.normal(size=n)

    ok = True
    for lbl, sig, want in (("stepped sine", stepped, "STEPPED"),
                           ("multisine", multi, "MULTISINE")):
        print(f"\n{'='*66}\n  synthetic {lbl} (truth: {want})\n{'='*66}")
        r = occupancy_test(sig, fs)
        report(r)
        got = r.get("verdict", "")
        good = got.startswith(want)
        ok &= good
        print(f"  self-test: {'PASS' if good else 'FAIL'}")
    print(f"\n  {'ALL PASS' if ok else 'FAILURES'}")
    return int(not ok)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="FAMOS .DAT recording")
    p.add_argument("--channel", default=None,
                   help="channel to analyse (default: the loudest UC reference)")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not a.file:
        p.print_help()
        return 2
    print("=" * 66)
    print("  EXCITATION TYPE DIAGNOSTIC")
    print("=" * 66)
    return run_file(a.file, a.channel)


if __name__ == "__main__":
    raise SystemExit(main())
