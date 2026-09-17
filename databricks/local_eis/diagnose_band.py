#!/usr/bin/env python3
"""
diagnose_band.py
================
Answer three questions about a recording, from the recording, with nothing
assumed:

    1. WHAT UNIT are the raw channels in?
    2. WHAT TONES are physically in the file, and how high do they go?
    3. WHERE does the band die -- which stage drops the steps above X Hz?

Question 3 is the one that matters when a Nyquist plot stops at 375 Hz and
you know the sweep went to 24 kHz.  There are five places a step can be
lost, and they need completely different fixes:

    a. it is not in the file            -> nothing to recover; check the sweep
    b. f_hi = min(f_max_hz, 0.45*fs)    -> raise cfg.f_max_hz (a CONSTANT,
                                           not Nyquist -- at fs = 100 kHz the
                                           converter has 45 kHz of headroom)
    c. detect_schedule never found it   -> hf_schedule / a better trace
    d. the ladder pruned it             -> cfg.hf_ladder_prune (off by
                                           default for exactly this reason)
    e. silver gated the point out       -> silver/point_rejections.csv says
                                           which gate, per segment per point

This script reports a, b, c and d directly, and points at the file that
answers e.

ON UNITS -- READ THIS BEFORE TRUSTING AN ABSOLUTE IMPEDANCE
-----------------------------------------------------------
`eis_local.FamosFile` reads the raw float32 samples and applies NO scaling.
It does not read the FAMOS `|CR` calibration factor, offset, or unit string:
`m_cr` is used only to get the channel count.  So the numbers the pipeline
works with are in whatever unit DASYLab wrote, and the pipeline never finds
out which.

That is not automatically a bug, because the impedance is a RATIO:

    Z = K * A_ref / A_seg          [ohm*cm^2]

A_ref and A_seg come from the same file, so a common scale factor cancels
exactly.  The unit survives in ONE place -- K, the Abgleich constant from
--curr-cal, which carries units of V/(A/cm^2).  So:

    channels in V,  K in V/(A/cm^2)   ->  Z correct
    channels in mV, K in V/(A/cm^2)   ->  Z too small by 1000
    channels in mV, K in mV/(A/cm^2)  ->  Z correct

The Nyquist SHAPE is identical in all three cases; only the axis numbers
move, by exactly a factor of 1000.  Two independent checks below settle it
without needing the instrument documentation:

  * the DC operating point.  Summing j_dc = u_dc/K over the measured area
    must reproduce the load setpoint.  If it lands on 45 mA or 45 kA instead
    of 45 A, the channel unit and K disagree by that factor.
  * the plausibility window.  A PEM segment sits at roughly 30-800
    mOhm*cm^2.  Being off by 10^3 puts it somewhere absurd.

USAGE
-----
    python diagnose_band.py /path/to/famos_dir --curr-cal curr.csv
    python diagnose_band.py /path/to/one_card.DAT
    python diagnose_band.py /path/to/famos_dir --f-max 30000

On Databricks, in a notebook cell beside the other modules:

    import diagnose_band
    diagnose_band.report('/Volumes/.../Famos', curr_cal='/Volumes/.../curr.csv',
                         condition='45A', f_max_hz=30000.0)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ===========================================================================
# 1. What is in the file
# ===========================================================================


def _raw_stats(fam, names, stride: int = 97) -> list[dict]:
    """Per-channel DC level and ac amplitude, in RAW FILE UNITS.

    The stride is a prime so that a decimated read cannot lock onto a tone
    and report its amplitude as a DC offset.
    """
    rows = []
    for n in names:
        x = fam.channel(n)[::stride]
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        rows.append({
            "channel": n,
            "slot": fam.position(n),
            "dc": float(np.mean(x)),
            "ac_rms": float(np.std(x)),
            "p99_abs": float(np.percentile(np.abs(x), 99)),
        })
    return rows


def guess_unit(rows: list[dict]) -> str:
    """A statement about ORDER OF MAGNITUDE, not a measurement.

    There is no unit string in what the reader parses, so this cannot be
    resolved from the file alone -- it is a prompt to check the two real
    tests (DC closure and the plausibility window), not a substitute for
    them.  A segment channel on this plate sits at a DC operating point of
    order 0.1 V; the same signal in millivolts reads ~100.
    """
    if not rows:
        return "no channels read"
    lvl = float(np.median([abs(r["dc"]) for r in rows] or [0.0]))
    if lvl == 0:
        return "all channels sit at zero DC - cannot say"
    if 1e-3 <= lvl <= 5.0:
        return (f"median |DC| = {lvl:.4g} -> consistent with VOLTS "
                f"(a segment sits near 0.1 V on this plate)")
    if 5.0 < lvl <= 5e3:
        return (f"median |DC| = {lvl:.4g} -> consistent with MILLIVOLTS; if "
                f"so, K from --curr-cal must be in mV/(A/cm2) too, or every "
                f"impedance is 1000x too small")
    return (f"median |DC| = {lvl:.4g} -> neither V nor mV at face value; "
            f"the export may carry a scale factor the reader does not apply")


# ===========================================================================
# 2. What tones are actually present
# ===========================================================================


def tone_census(x: np.ndarray, fs: float, f_lo: float, f_hi: float,
                n_seg: int = 64, top: int = 40) -> list[dict]:
    """Every narrowband tone in the record, found without any schedule.

    This is the GROUND TRUTH the rest of the pipeline is measured against: a
    spectrogram, and for each frequency bin the best frame it ever reaches.
    A stepped sweep shows up as a set of bins that are each loud in a few
    frames and quiet everywhere else, which is exactly what makes them
    findable here but easy to miss in a whole-record periodogram.

    No detector, no ladder, no gate.  If a frequency is not in this list it
    is not in the file, and no amount of pipeline work will produce it.
    """
    n = x.size
    seg = int(2 ** np.floor(np.log2(max(n // n_seg, 256))))
    if seg < 256 or n < 2 * seg:
        return []
    hop = seg // 2
    win = np.hanning(seg)
    n_frames = 1 + (n - seg) // hop
    freqs = np.fft.rfftfreq(seg, 1 / fs)
    band = (freqs >= f_lo) & (freqs <= f_hi)

    best = np.zeros(freqs.size)
    best_frame = np.zeros(freqs.size, int)
    med_acc = np.zeros(freqs.size)
    for i in range(n_frames):
        a = i * hop
        y = x[a:a + seg]
        y = y - y.mean()
        P = np.abs(np.fft.rfft(y * win)) ** 2
        med_acc += P
        hit = P > best
        best_frame[hit] = i
        best = np.where(hit, P, best)
    med_acc /= max(n_frames, 1)

    # A tone stands out against the BROADBAND floor of the same frame, not
    # against its own time average -- a tone that is on for one frame in
    # fifty has a tiny time average and would flatter itself.
    floor = np.median(best[band]) if band.any() else 1.0
    snr = 10 * np.log10(np.maximum(best, 1e-300) / max(floor, 1e-300))

    idx = np.where(band)[0]
    # keep local maxima only, so one tone does not appear as five bins
    keep = [i for i in idx
            if best[i] == best[max(i - 2, 0):i + 3].max()]
    keep.sort(key=lambda i: -snr[i])
    out = []
    for i in keep[:top]:
        out.append({
            "f_hz": float(freqs[i]),
            "snr_db": float(snr[i]),
            "t_s": float(best_frame[i] * hop / fs),
            "duty": float(med_acc[i] / max(best[i], 1e-300)),
        })
    return sorted(out, key=lambda r: r["f_hz"])


# ===========================================================================
# 3. Where the band dies
# ===========================================================================


def _attrition(fam, cfg, log=None) -> dict:
    """Detected on each trace, and what each later stage would keep."""
    import hf_schedule as H
    from eis_local import detect_schedule

    f_hi_cfg = cfg.f_hi(fam.fs)
    uc_names = fam.uc_names
    ref_name = (max(uc_names,
                    key=lambda c: float(np.std(fam.channel(c)[::10])))
                if uc_names else None)

    out = {"fs": fam.fs, "f_hi_used": f_hi_cfg,
           "f_hi_nyquist": cfg.f_hi_frac_fs * fam.fs,
           "f_max_hz": cfg.f_max_hz, "ref_name": ref_name}

    if ref_name:
        uc = fam.channel(ref_name)
        old = detect_schedule(uc, fam.fs, ppd=cfg.ppd, f_lo=cfg.f_min_hz,
                              f_hi=f_hi_cfg, min_snr_db=cfg.min_snr_db,
                              verbose=False)
        out["uc_hz"] = sorted(round(s.freq, 3) for s in old)

    chans = H.LazyChannels(fam)
    if len(chans):
        hf = H.recover_schedule(
            chans, fam.fs, f_lo=cfg.f_min_hz, f_hi=f_hi_cfg, ppd=cfg.ppd,
            min_snr_db=cfg.min_snr_db, sigma_rel_max=cfg.sigma_rel_max,
            extend=getattr(cfg, "hf_ladder_extend", True),
            snap_ppd=getattr(cfg, "hf_ladder_snap_ppd", True),
            ladder_tol=getattr(cfg, "hf_ladder_tol", 0.02),
            prune=getattr(cfg, "hf_ladder_prune", False),
            uc_ref=fam.channel(ref_name) if ref_name else None, log=log)
        out["ensemble_hz"] = sorted(round(s.freq, 3) for s in hf.as_steps())
        out["hf"] = hf.summary()
    return out


# ===========================================================================
# 4. Does the absolute scale hold up
# ===========================================================================


def dc_closure_check(fam, cal, cfg, areas: dict) -> dict:
    """Sum j_dc over the measured segments and compare with the setpoint.

    This is the test that settles the unit question, because it is the only
    place an ABSOLUTE current appears.  It needs no high-frequency content
    and no schedule: just the DC level of each segment channel and K.
    """
    import bronze
    T = 60.0
    med_c0 = (float(np.median(list(cal.seg_c0.values())))
              if cal.seg_c0 else float("nan"))
    med_c1 = float(np.median(list(cal.seg_c1.values()))) if cal.seg_c1 else 0.0
    tot_i, tot_a, n = 0.0, 0.0, 0
    for seg in fam.segment_names:
        K, _imp = bronze._K_for(seg, cal, T, med_c0, med_c1, cfg)
        A = areas.get(seg)
        if A is None or not np.isfinite(K) or K == 0:
            continue
        u_dc = float(np.mean(fam.channel(seg)[::97]))
        tot_i += (u_dc / K) * A
        tot_a += A
        n += 1
    return {"n_segments": n, "area_cm2": tot_a, "i_total_a": tot_i,
            "j_mean": (tot_i / tot_a) if tot_a else float("nan")}


# ===========================================================================
# 5. Report
# ===========================================================================


def _fmt_hz(v):
    return f"{v:.3g}" if v < 10 else f"{v:.0f}"


def report(dat, curr_cal=None, condition="ALL", f_max_hz=None,
           f_min_hz=None, i_setpoint_a=None, top=40) -> dict:
    from config import DEFAULT
    from eis_local import FamosFile, PlateCalibration
    import utils

    kw = {"condition": condition}
    if f_max_hz is not None:
        kw["f_max_hz"] = float(f_max_hz)
    if f_min_hz is not None:
        kw["f_min_hz"] = float(f_min_hz)
    if curr_cal:
        kw["curr_cal"] = Path(curr_cal)
    cfg = DEFAULT.replace(**kw)
    log = utils.get_logger(True)

    p = Path(dat)
    files = ([p] if p.is_file()
             else sorted(list(p.glob("*.DAT")) + list(p.glob("*.dat"))))
    if condition and condition != "ALL":
        sel = [f for f in files
               if f"_current_{condition}_".lower() in f.name.lower()]
        files = sel or files
    if not files:
        raise SystemExit(f"diagnose_band: no FAMOS files under {p}")

    cal = None
    areas = {}
    if curr_cal:
        cal = PlateCalibration.load(Path(curr_cal), None)
        areas = utils.segment_areas(cfg)

    print("=" * 78)
    print(f"  BAND DIAGNOSIS   {len(files)} file(s)   condition={condition}")
    print(f"  cfg.f_max_hz = {cfg.f_max_hz:.0f} Hz    "
          f"f_hi_frac_fs = {cfg.f_hi_frac_fs}    min_snr_db = "
          f"{cfg.min_snr_db}")
    print("=" * 78)

    out = {"files": {}, "cfg_f_max_hz": cfg.f_max_hz}
    rates: dict[float, list[str]] = {}
    dc_total = {"i": 0.0, "a": 0.0, "n": 0}
    for fp in files:
        fam = FamosFile(fp)
        rates.setdefault(fam.fs, []).append(fp.name)
        f_hi = cfg.f_hi(fam.fs)

        print(f"\n─── {fp.name}")
        print(f"    fs = {fam.fs:,.0f} Hz   {fam.n_ch} channels   "
              f"{fam.n_samples:,} samples = {fam.n_samples / fam.fs:.1f} s")
        print(f"    channels: {', '.join(fam.names)}")

        # ---- 1. units --------------------------------------------------
        rows = _raw_stats(fam, fam.segment_names[:6] + fam.uc_names[:2])
        print("\n    RAW LEVELS (file units, no scaling is applied anywhere)")
        for r in rows:
            print(f"      slot {r['slot']:3d}  {r['channel']:>8s}   "
                  f"DC {r['dc']:+12.6g}   ac_rms {r['ac_rms']:12.6g}   "
                  f"p99|x| {r['p99_abs']:12.6g}")
        seg_rows = [r for r in rows if r["channel"] in fam.segment_names]
        print(f"      -> {guess_unit(seg_rows)}")

        # ---- 2. what tones exist --------------------------------------
        import hf_schedule as H
        ens, _info = H.polarity_aligned_reference(H.LazyChannels(fam))
        ref_name = (max(fam.uc_names,
                        key=lambda c: float(np.std(fam.channel(c)[::10])))
                    if fam.uc_names else None)
        print(f"\n    TONES PRESENT  {cfg.f_min_hz} .. {f_hi:.0f} Hz  "
              f"(spectrogram, no detector, no ladder)")
        cen_e = tone_census(ens, fam.fs, cfg.f_min_hz, f_hi, top=top)
        cen_u = (tone_census(fam.channel(ref_name), fam.fs, cfg.f_min_hz,
                             f_hi, top=top) if ref_name else [])
        print(f"      segment ensemble : {len(cen_e)} tone(s)"
              + (f", {_fmt_hz(cen_e[0]['f_hz'])} .. "
                 f"{_fmt_hz(cen_e[-1]['f_hz'])} Hz" if cen_e else ""))
        print(f"      {str(ref_name or '-'):16s} : {len(cen_u)} tone(s)"
              + (f", {_fmt_hz(cen_u[0]['f_hz'])} .. "
                 f"{_fmt_hz(cen_u[-1]['f_hz'])} Hz" if cen_u else ""))
        # A bin at the noise floor is not a tone.  6 dB over the median of
        # the per-bin maxima is a low bar deliberately -- the point here is
        # to show what EXISTS, not to gate it -- but listing the floor itself
        # would bury the four real rungs in forty noise bins.
        hi = [r for r in cen_e if r["f_hz"] > 300 and r["snr_db"] >= 6.0]
        n_floor = sum(1 for r in cen_e if r["f_hz"] > 300 and r["snr_db"] < 6)
        if hi:
            print(f"      above 300 Hz on the ensemble ({len(hi)} above the "
                  f"floor, {n_floor} at it):")
            for r in hi[:14]:
                print(f"        {r['f_hz']:10.2f} Hz   {r['snr_db']:6.1f} dB"
                      f"   first loud at t = {r['t_s']:7.2f} s"
                      f"   duty {r['duty']:.3f}")
        else:
            print("      NOTHING above 300 Hz on the ensemble -- if the sweep "
                  "went higher, it is not in THIS file")

        # ---- 3. attrition ----------------------------------------------
        att = _attrition(fam, cfg, log=None)
        print("\n    WHERE THE BAND DIES")
        print(f"      ceiling in force : {att['f_hi_used']:.0f} Hz  "
              f"= min(f_max_hz {att['f_max_hz']:.0f}, "
              f"0.45*fs {att['f_hi_nyquist']:.0f})"
              + ("   <-- f_max_hz BINDS, raise it"
                 if att["f_max_hz"] < att["f_hi_nyquist"] else
                 "   (Nyquist binds, f_max_hz is not the limit)"))
        for tag, key in (("detected on " + str(att.get("ref_name")), "uc_hz"),
                         ("detected on ensemble+ladder", "ensemble_hz")):
            v = att.get(key)
            if v:
                print(f"      {tag:34s}: {len(v):3d} step(s), "
                      f"{_fmt_hz(min(v))} .. {_fmt_hz(max(v))} Hz")
        s = att.get("hf", {})
        if s:
            print(f"      ladder           : "
                  f"{'ok' if s['ladder_ok'] else 'NOT RECOVERED'}"
                  + (f", {s['ladder_ppd']:.3f} points/decade"
                     if s["ladder_ok"] else ""))
            print(f"      predicted rungs  : {s['n_predicted_verified']} "
                  f"verified, {s['n_predicted_rejected']} rejected")
            if s.get("off_ladder_hz"):
                kept = "KEPT (prune off)" if not s["n_off_ladder_pruned"] \
                    else "PRUNED"
                print(f"      off the ladder   : {len(s['off_ladder_hz'])} "
                      f"step(s) {kept}: "
                      + ", ".join(_fmt_hz(v)
                                  for v in s["off_ladder_hz"][:10]))

        # ---- 4. absolute scale, ACCUMULATED ACROSS CARDS ----------------
        # The setpoint is a PLATE current, and each card carries a fraction
        # of the plate, so a per-card sum is meaningless against it -- it
        # would read 1/5 of the setpoint on a healthy five-card plate and
        # look like a units error.  Accumulate, compare once at the end.
        if cal is not None and cal.has_current_cal and areas:
            dc = dc_closure_check(fam, cal, cfg, areas)
            dc_total["i"] += dc["i_total_a"]
            dc_total["a"] += dc["area_cm2"]
            dc_total["n"] += dc["n_segments"]
            print(f"\n    DC on this card: {dc['n_segments']} segment(s), "
                  f"{dc['area_cm2']:.1f} cm2 -> {dc['i_total_a']:.3f} A")

        out["files"][fp.name] = {"fs": fam.fs, "raw": rows,
                                 "tones_ensemble": cen_e,
                                 "tones_reference": cen_u,
                                 "attrition": att}

    if dc_total["n"]:
        print("\n" + "=" * 78)
        print("  ABSOLUTE SCALE  (the unit test that counts, whole plate)")
        j = dc_total["i"] / dc_total["a"] if dc_total["a"] else float("nan")
        print(f"    {dc_total['n']} segment(s), {dc_total['a']:.1f} cm2 -> "
              f"I = {dc_total['i']:.2f} A, j = {j:.4f} A/cm2")
        if i_setpoint_a:
            ratio = dc_total["i"] / float(i_setpoint_a)
            print(f"    setpoint {float(i_setpoint_a):.1f} A -> "
                  f"ratio {ratio:.4g}")
            if 0.5 < ratio < 2.0:
                print("    -> the channel unit and K agree. Absolute "
                      "impedances can be trusted.")
            elif 2e-4 < ratio < 5e-3:
                print("    -> OFF BY ~1000 LOW: the channels are in a unit "
                      "1000x smaller than K assumes (mV against V).\n"
                      "       Every |Z| is then 1000x too SMALL. The Nyquist "
                      "SHAPE is unaffected; the axis is not.")
            elif 200 < ratio < 5000:
                print("    -> OFF BY ~1000 HIGH: K is in a unit 1000x "
                      "smaller than the channels.\n"
                      "       Every |Z| is then 1000x too LARGE.")
            else:
                print("    -> does not close, and not by a round factor. "
                      "That is an Abgleich or coverage problem,\n"
                      "       not a unit problem: check how many segments "
                      "were measured and whether K was imputed.")
        else:
            print("    pass --current to compare this against the setpoint; "
                  "that comparison is what settles the unit.")

    if len(rates) > 1:
        print("\n" + "=" * 78)
        print("  MIXED SAMPLING RATES")
        for fs_hz, names in sorted(rates.items(), reverse=True):
            print(f"    {fs_hz:,.0f} Hz : {len(names)} card(s)  "
                  f"ceiling {cfg.f_hi(fs_hz):.0f} Hz   "
                  f"({', '.join(n[-12:] for n in names)})")
        print("    Cards at different rates are aligned and scheduled in "
              "SEPARATE groups: a schedule is stored as sample indices, and\n"
              "    sample N at 100 kHz is not the instant sample N is at "
              "50 kHz. Each group gets its own consensus schedule, so the\n"
              "    two groups can legitimately reach different top "
              "frequencies -- compare them per group, not per plate.")
    print("\n" + "=" * 78)
    print("  If a step is in TONES PRESENT but not in the detected lists, it "
          "is a detection problem.\n"
          "  If it is detected but missing from the Nyquist, silver gated "
          "it: read silver/point_rejections.csv,\n"
          "  which carries one row per segment per point with the blocking "
          "gate named.")
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dat", help="FAMOS .DAT file or a directory of them")
    p.add_argument("--curr-cal", default=None,
                   help="per-segment Abgleich; enables the absolute-scale "
                        "check, which is what settles the unit question")
    p.add_argument("--condition", default="ALL")
    p.add_argument("--f-max", type=float, default=None)
    p.add_argument("--f-min", type=float, default=None)
    p.add_argument("--current", type=float, default=None,
                   help="load setpoint in A, for the DC closure check")
    p.add_argument("--top", type=int, default=40,
                   help="how many tones to list per trace")
    a = p.parse_args()
    report(a.dat, curr_cal=a.curr_cal, condition=a.condition,
           f_max_hz=a.f_max, f_min_hz=a.f_min, i_setpoint_a=a.current,
           top=a.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
