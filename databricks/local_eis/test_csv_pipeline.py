#!/usr/bin/env python3
"""
test_csv_pipeline.py  --  end-to-end checks for the CSV evaluation path
=======================================================================

Synthesises a plate whose per-segment impedance is known exactly, writes it
out in each supported CSV layout, runs the pipeline over it and asks whether
the numbers come back.  Run it before trusting a real file:

    python test_csv_pipeline.py

Ground truth is a two-arc spectrum per segment,

    Z_s(f) = Rs + jwL + R1/(1+(jw tau1)^n1) + R2/(1+(jw tau2)^n2)

with Rs varying across the plate so the heat map has something to show.  The
time-domain records are built the way the hardware builds them:

    U_cell(t)  a designed multisine
    u_s(t)     K * U_cell / Z_s, tone by tone       (the Abgleich inverted)

which is the actual measurement equation, so a pipeline that gets Z back has
inverted it correctly rather than merely reproduced its own convention.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import csv_pipeline
import csv_source
import r2d2_geometry as geom
from config import DEFAULT

TONES = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
                  200.0, 500.0, 1000.0, 2000.0])
K_CAL = 0.60            # V/(A/cm^2) at the test temperature
FS = 20000.0
DUR = 20.0


def truth(seg: int) -> dict:
    """A believable segment: Rs drifts along the plate, arcs stay put."""
    return {"Rs": 0.070 + 0.0006 * seg, "L": 2.0e-7,
            "R1": 0.045, "tau1": 2.0e-4, "n1": 0.90,
            "R2": 0.080, "tau2": 8.0e-3, "n2": 0.85}


def z_truth(f, p) -> np.ndarray:
    w = 2 * np.pi * np.asarray(f, float)
    return (p["Rs"] + 1j * w * p["L"]
            + p["R1"] / (1 + (1j * w * p["tau1"]) ** p["n1"])
            + p["R2"] / (1 + (1j * w * p["tau2"]) ** p["n2"]))


def _write_curr_cal(path: Path, n=72):
    # K(T) = c0 + 1e-3*c1*T; pick c1 so K(25 C) = K_CAL exactly
    c1 = 2.4
    c0 = K_CAL - 1e-3 * c1 * 25.0
    path.write_text("\n".join(f"{c0:.6f};{c1:.6f}" for _ in range(n)) + "\n")


def _write_temp_cal(path: Path):
    path.write_text("\n".join("3.188000;0.013130" for _ in range(4)) + "\n")


def make_wide_csv(path: Path, segments, jitter=0.0, decimals=9, seed=1,
                  noise=2e-6):
    """A multisine record in the wide_time layout."""
    rng = np.random.default_rng(seed)
    n = int(FS * DUR)
    t = np.arange(n) / FS
    if jitter:
        t = np.sort(t + rng.normal(0, jitter / FS, n))

    phase = rng.uniform(0, 2 * np.pi, TONES.size)
    amp = 0.004 * np.ones(TONES.size)              # 4 mV per tone on U_cell

    u_cell = np.zeros(n)
    for a, f, ph in zip(amp, TONES, phase):
        u_cell += a * np.cos(2 * np.pi * f * t + ph)

    u_seg = {}
    for s in segments:
        p = truth(s)
        Z = z_truth(TONES, p)
        y = np.zeros(n)
        for a, f, ph, z in zip(amp, TONES, phase, Z):
            # U = Z * j  and  u_s = K * j  =>  u_s = K * U / Z
            c = K_CAL * (a * np.exp(1j * ph)) / z
            y += np.real(c * np.exp(1j * 2 * np.pi * f * t))
        y += rng.normal(0, noise, n)                # electronic noise
        u_seg[s] = y

    fmt = f"{{:.{decimals}f}}"
    with path.open("w") as fh:
        fh.write("time_s,u_cell," + ",".join(f"s{s}" for s in segments)
                 + ",temp1,temp2,temp3,temp4\n")
        for i in range(n):
            fh.write(f"{t[i]:.9f},{fmt.format(u_cell[i])},"
                     + ",".join(fmt.format(u_seg[s][i]) for s in segments)
                     + ",3.51830,3.51830,3.51830,3.51830\n")
    return path


def make_sweep_csv(path: Path, segments, tones=None, seed=11):
    """A stepped sweep: one tone per dwell, dwells concatenated.

    This is how the FAMOS rig excites, so a CSV logger attached to the same
    load bank produces it too -- and it is the branch where the windows have
    to be *found* rather than assumed.
    """
    rng = np.random.default_rng(seed)
    tones = np.array(tones if tones is not None
                     else [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
                           500.0, 1000.0])
    t_parts, uc_parts = [], []
    seg_parts = {s: [] for s in segments}
    t0 = 0.0
    for f in tones:
        dwell = max(1.0, 12.0 / f)                  # >= 12 cycles
        n = int(FS * dwell)
        tt = t0 + np.arange(n) / FS
        ph = rng.uniform(0, 2 * np.pi)
        a = 0.004
        uc_parts.append(a * np.cos(2 * np.pi * f * tt + ph))
        for s in segments:
            z = z_truth(np.array([f]), truth(s))[0]
            c = K_CAL * (a * np.exp(1j * ph)) / z
            seg_parts[s].append(np.real(c * np.exp(1j * 2 * np.pi * f * tt))
                                + rng.normal(0, 2e-6, n))
        t_parts.append(tt)
        t0 = tt[-1] + 1.0 / FS

    t = np.concatenate(t_parts)
    uc = np.concatenate(uc_parts)
    seg = {s: np.concatenate(v) for s, v in seg_parts.items()}
    with path.open("w") as fh:
        fh.write("time_s,u_cell," + ",".join(f"s{s}" for s in segments)
                 + ",temp1,temp2,temp3,temp4\n")
        for i in range(t.size):
            fh.write(f"{t[i]:.9f},{uc[i]:.9f},"
                     + ",".join(f"{seg[s][i]:.9f}" for s in segments)
                     + ",3.51830,3.51830,3.51830,3.51830\n")
    return path, tones


def make_r2d2_sweep(folder: Path, segments, tones=None, fs=11001.1026,
                    scan_step_us=None, dur_s=0.9, seed=23):
    """A sweep in the R2-D2 logger's own format: metadata.csv + p<k>.csv.

    Built the way the hardware builds it, which is the point of the test:

      * one point file per frequency;
      * 80 channels sampled one after another inside each row, each carrying
        the phase the analogue waveform actually had at ITS instant;
      * s columns already a current density (the logger applies the Abgleich),
        temps already in degC;
      * uc1..uc4 as two differential pairs around a 0.61 V cell.

    Tones above fs/2 are written as the hardware would record them -- folded.
    Nothing special is done to make the alias: sampling each channel at its
    own instant and printing the value produces it, exactly as the rig does.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tones = list(tones if tones is not None
                 else [10.0, 40.0, 150.0, 600.0, 2500.0, 9000.0])
    cols = ([f"s{s}" for s in segments] + ["uc1", "uc2", "uc3", "uc4"]
            + [f"temp{i}" for i in range(1, 5)])
    # The real logger spreads its 80 channels across 96 % of the sample
    # period.  Keep that invariant rather than the 1.1 us step, because it is
    # the SPAN that gives the scan its leverage on the analogue frequency.
    if scan_step_us is None:
        scan_step_us = 0.96e6 / fs / max(len(cols) - 1, 1)
    shifts = np.arange(len(cols)) * scan_step_us          # microseconds

    (folder / "metadata.csv").write_text(
        "R2-D2 Local EIS Measurements\r\n"
        "Date: 20.04.2026 10:21\r\n"
        "Software Version: V2.5\r\n"
        "Coefficients: TestPlate\r\n"
        "Leepa: FC0000000-00\r\n"
        "Segment 1: Cathode inlet\r\n")

    j_dc = {s: 1.8 + 0.004 * s for s in segments}
    u_dc = 0.61
    for k, f in enumerate(tones, start=1):
        n = int(fs * dur_s)
        row_t = np.arange(n) / fs
        U_amp = 4e-4
        lines = ["\t".join(["timestamp"] + cols),
                 "\t".join(["timeshifts"] + [f"{v:.6f}" for v in shifts])]
        chan = np.empty((n, len(cols)))
        for c, name in enumerate(cols):
            tt = row_t + shifts[c] * 1e-6          # this channel's instants
            if name.startswith("temp"):
                chan[:, c] = 60.0 + 4.0 * int(name[-1]) + rng.normal(0, 0.03, n)
                continue
            if name.startswith("uc"):
                # uc1/uc3 sit at the anode reference, uc2/uc4 at the cathode;
                # the DIFFERENCE is the cell voltage, so each half carries
                # half the modulation.
                base = 0.0 if name in ("uc1", "uc3") else u_dc
                sign = -0.5 if name in ("uc1", "uc3") else 0.5
                chan[:, c] = base + sign * U_amp * np.cos(2 * np.pi * f * tt) \
                    + rng.normal(0, 2e-6, n)
                continue
            s = int(name[1:])
            Z = z_truth(np.array([f]), truth(s))[0]
            amp = U_amp / abs(Z)
            chan[:, c] = (j_dc[s]
                          + amp * np.cos(2 * np.pi * f * tt - np.angle(Z))
                          + rng.normal(0, 2e-5, n))
        for i in range(n):
            us = 429062 + int(round(i * 1e6 / fs))
            ss, us = 11 + us // 1_000_000, us % 1_000_000
            lines.append(f"2026.04.20 10:21:{ss:02d},{us:06d}\t"
                         + "\t".join(f"{v:.6f}" for v in chan[i]))
        (folder / f"p{k}.csv").write_text("\n".join(lines) + "\n")
    return folder, tones


def make_freq_csv(path: Path, segments, unit="mohm_cm2"):
    k = 1000.0 if unit == "mohm_cm2" else 1.0
    f = np.geomspace(0.1, 5000.0, 40)
    with path.open("w") as fh:
        fh.write("segment,freq_hz,z_re,z_im\n")
        for s in segments:
            Z = z_truth(f, truth(s))
            for fi, z in zip(f, Z):
                fh.write(f"{s},{fi:.6g},{k*z.real:.8g},{k*z.imag:.8g}\n")
    return path


def _cfg(td: Path, csv_path: Path, plate: str, out: str):
    curr = td / "curr.csv"
    temp = td / "temp.csv"
    _write_curr_cal(curr)
    _write_temp_cal(temp)
    return DEFAULT.replace(
        source_format="csv", csv_path=csv_path, plate=plate,
        curr_cal=curr, temp_cal=temp, out_dir=td / out,
        csv_tones=tuple(TONES.tolist()),
        f_min_hz=0.2, f_max_hz=5000.0,
        write_png=False, write_html=False, verbose=False)


class _Log:
    """Quiet stand-in for the pipeline logger, for the direct-call checks."""
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def debug(self, *a, **k): pass


def _report(name, got, want, tol, unit=""):
    err = abs(got - want) / abs(want) if want else abs(got)
    ok = err <= tol
    print(f"    {name:38s} {got:10.4f}{unit}  (true {want:.4f}{unit}, "
          f"{100*err:5.2f} %)  {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    fails = 0
    segments = [1, 5, 9, 20, 37, 49, 60, 72]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # --- 1. time domain, clean, uniform --------------------------------
        print("\n  time-domain multisine, uniform grid")
        p = make_wide_csv(td / "clean.csv", segments)
        man = csv_pipeline.run_csv(_cfg(td, p, "gen1", "out_clean"))
        rows = _read_ecm(td / "out_clean")
        for s in (1, 20, 72):
            fails += _report(f"segment {s} R_ohmic",
                             rows[s]["R_ohmic_mohm_cm2"],
                             1000 * truth(s)["Rs"], 0.05, " mΩ·cm²")
        fails += _report("segments fitted", man["n_ecm_ok"], len(segments), 0.0)

        # --- 2. same data with a jittering time base ------------------------
        # This is the case that breaks a uniform-grid FFT and is the reason
        # the estimator fits on the recorded timestamps.
        print("\n  same record, 20 % sample-interval jitter")
        p = make_wide_csv(td / "jitter.csv", segments, jitter=0.2, seed=3)
        csv_pipeline.run_csv(_cfg(td, p, "gen1", "out_jit"))
        rows = _read_ecm(td / "out_jit")
        for s in (1, 72):
            fails += _report(f"segment {s} R_ohmic",
                             rows[s]["R_ohmic_mohm_cm2"],
                             1000 * truth(s)["Rs"], 0.06, " mΩ·cm²")

        # --- 3. coarse text export ------------------------------------------
        # Same signal, same (negligible) electronic noise, printed to 9 and
        # to 4 decimals.  The only difference between the two records is the
        # printed resolution, so any rise in sigma_rel is the quantiser and
        # nothing else -- which is the claim being tested.
        print("\n  text resolution as an error source")
        p_fine = make_wide_csv(td / "fine.csv", segments, decimals=9,
                               seed=5, noise=1e-9)
        p_coarse = make_wide_csv(td / "coarse.csv", segments, decimals=4,
                                 seed=5, noise=1e-9)
        csv_pipeline.run_csv(_cfg(td, p_fine, "gen1", "out_fine"))
        csv_pipeline.run_csv(_cfg(td, p_coarse, "gen1", "out_q"))
        rows = _read_ecm(td / "out_q")
        fails += _report("segment 1 R_ohmic, 4 decimals",
                         rows[1]["R_ohmic_mohm_cm2"],
                         1000 * truth(1)["Rs"], 0.08, " mΩ·cm²")
        s_fine = _median_sigma(td / "out_fine")
        s_coarse = _median_sigma(td / "out_q")
        ok = s_coarse > 10 * s_fine
        print(f"    sigma_rel tracks the printed digits    "
              f"{s_coarse:.2e} vs {s_fine:.2e}   {'PASS' if ok else 'FAIL'}")
        fails += not ok

        # --- 4. a file that is already a spectrum ---------------------------
        print("\n  frequency-domain CSV (mΩ·cm²)")
        p = make_freq_csv(td / "spec.csv", segments)
        man = csv_pipeline.run_csv(_cfg(td, p, "gen2", "out_spec"))
        rows = _read_ecm(td / "out_spec")
        for s in (5, 49):
            fails += _report(f"segment {s} R_ohmic",
                             rows[s]["R_ohmic_mohm_cm2"],
                             1000 * truth(s)["Rs"], 0.02, " mΩ·cm²")
        ok = man["plate"] == "gen2"
        print(f"    plate carried into the manifest        {man['plate']:>10s}"
              f"      {'PASS' if ok else 'FAIL'}")
        fails += not ok

        # --- 5. excitation discovery, with no tone list supplied ------------
        # The operator will not always know the schedule.  Detection has to
        # find the same tones the generator put in, or the whole path is
        # unusable on a file that arrives without a note.
        print("\n  excitation discovery (no tone list given)")
        cfg = _cfg(td, td / "clean.csv", "gen1", "out_auto").replace(
            csv_tones=())
        man = csv_pipeline.run_csv(cfg)
        found = np.array(sorted(man["schedule"]["tones"]))
        mode = man["schedule"]["mode"]
        matched = sum(1 for f in TONES
                      if found.size and np.min(np.abs(found - f)) < 0.02 * f)
        ok = mode == "multisine" and matched >= TONES.size - 1
        print(f"    mode={mode}, {matched}/{TONES.size} tones recovered"
              f"            {'PASS' if ok else 'FAIL'}")
        fails += not ok

        # --- 6. a stepped sweep, windows found rather than assumed ----------
        print("\n  stepped sweep")
        segs2 = [1, 20, 72]
        p, sw_tones = make_sweep_csv(td / "sweep.csv", segs2)
        cfg = _cfg(td, p, "gen1", "out_sweep").replace(
            csv_tones=(), f_min_hz=0.5, f_max_hz=2000.0,
            min_points_per_spectrum=5)
        man = csv_pipeline.run_csv(cfg)
        mode = man["schedule"]["mode"]
        found = np.array(sorted(man["schedule"]["tones"]))
        matched = sum(1 for f in sw_tones
                      if found.size and np.min(np.abs(found - f)) < 0.05 * f)
        ok = mode == "stepped_sweep" and matched >= len(sw_tones) - 1
        print(f"    mode={mode}, {matched}/{len(sw_tones)} dwells found"
              f"       {'PASS' if ok else 'FAIL'}")
        fails += not ok
        rows = _read_ecm(td / "out_sweep")
        for s in (1, 72):
            if str(rows.get(s, {}).get("ok")) not in ("True", "1.0"):
                print(f"    segment {s} ECM did not converge: "
                      f"{rows.get(s, {}).get('reason', 'missing')}   FAIL")
                fails += 1
                continue
            fails += _report(f"segment {s} R_ohmic",
                             rows[s]["R_ohmic_mohm_cm2"],
                             1000 * truth(s)["Rs"], 0.25, " mΩ·cm²")

        # --- 7. the R2-D2 logger format -------------------------------------
        # The real thing: one file per frequency, a channel scan inside every
        # row, and one tone deliberately placed above fs/2 so the alias path
        # is exercised rather than assumed.
        print("\n  R2-D2 logger sweep")
        segs3 = [1, 9, 20, 37, 49, 72]
        fold, sw = make_r2d2_sweep(td / "r2d2", segs3)
        cfg = _cfg(td, fold, "gen1", "out_r2d2").replace(
            csv_tones=(), f_min_hz=1.0, f_max_hz=5000.0,
            min_points_per_spectrum=4)
        man = csv_pipeline.run_csv(cfg)
        rep = man.get("r2d2", {})
        ok = man["schedule"]["mode"] == "r2d2_sweep" and rep.get("n_failed") == 0
        print(f"    {len(rep.get('points', []))} point files read, "
              f"{rep.get('n_failed')} failed              "
              f"{'PASS' if ok else 'FAIL'}")
        fails += not ok

        got = {round(p["f_analogue_hz"], 1) for p in rep["points"] if p["ok"]}
        matched = sum(1 for f in sw
                      if any(abs(g - f) < 0.02 * f for g in got))
        ok = matched == len(sw)
        print(f"    {matched}/{len(sw)} analogue frequencies recovered "
              f"(one above fs/2)   {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"      wanted {sw}\n      got    {sorted(got)}")
        fails += not ok

        under = [p for p in rep["points"] if p.get("undersampled")]
        ok = len(under) == sum(1 for f in sw if f > 0.5 * 11001.1026)
        print(f"    undersampled points flagged: {len(under)}"
              f"                    {'PASS' if ok else 'FAIL'}")
        fails += not ok

        rows = _read_ecm(td / "out_r2d2")
        for s in (1, 72):
            if str(rows.get(s, {}).get("ok")) not in ("True", "1.0"):
                print(f"    segment {s} ECM did not converge: "
                      f"{rows.get(s, {}).get('reason', 'missing')}   FAIL")
                fails += 1
                continue
            fails += _report(f"segment {s} R_ohmic",
                             rows[s]["R_ohmic_mohm_cm2"],
                             1000 * truth(s)["Rs"], 0.20, " mΩ·cm²")

        # The scan skew is the whole point of the format: check that ignoring
        # it would be visibly wrong at the top of the band.
        import csv_source as _cs
        pts = _cs.read_r2d2_sweep(fold)
        span = max(pts[0].timeshifts.values()) - min(pts[0].timeshifts.values())
        worst = 360.0 * max(sw) * span
        print(f"    scan span {1e6*span:.1f} us = {worst:.0f}° at {max(sw):.0f} Hz"
              f"   (why the deskew is not optional)")

        # --- 8. the real fixture --------------------------------------------
        # Synthetic data proves the algebra; it does not prove the parser
        # against the hardware's own output, which is where dialects bite.
        fx = Path(__file__).with_name("fixtures") / "r2d2_sample"
        if (fx / "p1.csv").exists():
            print("\n  real fixture (2500 rows of a real p1.csv)")
            import csv_source as _cs
            ok = _cs.detect_dialect(fx) == "r2d2_sweep"
            pts = _cs.read_r2d2_sweep(fx)
            m0 = pts[0]
            s0 = m0.summary()
            checks = [
                ("dialect detected as a sweep", ok),
                ("80 channels", s0["n_channels"] == 80),
                ("72 segments", len(m0.segments) == 72),
                ("scan spans ~96 % of a sample",
                 0.9 < s0["scan_fraction_of_sample"] < 1.0),
                ("s columns read as A/cm2", m0.seg_unit == "A/cm2"),
                ("temps read as degC", m0.temp_unit == "degC"),
                ("coefficient set recorded",
                 m0.meta["metadata"].get("coefficients") == "Coruscant"),
                ("four uc taps", sorted(m0.aux) == ["uc1", "uc2", "uc3", "uc4"]),
            ]
            for label, good in checks:
                print(f"    {label:38s}"
                      f"                {'PASS' if good else 'FAIL'}")
                fails += not good

            res = csv_pipeline.r2d2_point(
                m0, _cfg(td, fx, "gen1", "out_fx").replace(
                    f_min_hz=50.0, f_max_hz=5000.0), _Log())
            z = res["zone"]
            good = (res["ok"] and abs(res["f_alias_hz"] - 923.1) < 2.0
                    and z["undersampled"] and z["zone"] == 1
                    and z["R"] > 0.7 and z["R_baseband"] < 0.3)
            print(f"    alias {res.get('f_alias_hz', float('nan')):.1f} Hz -> "
                  f"analogue {res.get('f_hz', float('nan')):.0f} Hz, zone "
                  f"{z['zone']}, R {z['R']:.2f} vs {z['R_baseband']:.2f}"
                  f"   {'PASS' if good else 'FAIL'}")
            fails += not good

            # A record with no excitation must be refused, not analysed
            # against the 999 Hz instrument artefact that lives in it.
            head = m0.u_seg["1"].size
            m_quiet = _cs.read_r2d2(fx / "p1.csv")
            for k in list(m_quiet.u_seg):
                m_quiet.u_seg[k] = m_quiet.u_seg[k] * 0 + 1.9 \
                    + np.random.default_rng(0).normal(0, 1e-3, head)
            q = csv_pipeline.r2d2_point(m_quiet, _cfg(td, fx, "gen1", "o2"), _Log())
            good = not q.get("ok") and "no excitation" in q.get("reason", "")
            print(f"    a record with no excitation is refused"
                  f"                {'PASS' if good else 'FAIL'}")
            fails += not good
        else:
            print("\n  real fixture: not present, skipped")

        # --- 9. the plate selection must change the areas -------------------
        print("\n  plate selection")
        a1 = geom.areas("gen1")
        a2 = geom.areas("gen2")
        same = [s for s in a1 if abs(a1[s] - a2[s]) < 1e-9]
        ok = len(same) < 72 and set(str(i) for i in range(1, 37)) - set(same)
        print(f"    gen1 and gen2 differ on {72-len(same):2d} segments"
              f"                    {'PASS' if ok else 'FAIL'}")
        fails += not ok
        for key in ("gen1", "gen2"):
            chk = geom.self_check(verbose=False, plate_name=key)
            ok = not chk["problems"]
            print(f"    {key} tiles the plate exactly"
                  f"                        {'PASS' if ok else 'FAIL'}")
            fails += not ok

    print(f"\n  {'ALL PASS' if not fails else str(fails) + ' FAILURES'}\n")
    return int(fails)


def _read_ecm(out_dir: Path) -> dict:
    import csv
    rows = {}
    with (out_dir / "csv" / "ecm_parameters.csv").open() as fh:
        for r in csv.DictReader(fh):
            out = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows[int(r["segment"])] = out
    return rows


def _median_sigma(out_dir: Path) -> float:
    import csv
    v = []
    with (out_dir / "csv" / "spectra_clean.csv").open() as fh:
        for r in csv.DictReader(fh):
            try:
                v.append(float(r["sigma_rel"]))
            except (ValueError, KeyError):
                pass
    return float(np.median(v)) if v else float("nan")


if __name__ == "__main__":
    sys.exit(main())


def test_the_campaign_parent_is_refused_by_naming_its_sweeps(tmp_path):
    """`csv_files/` is not a sweep -- it is the folder of sweeps.

    The reader globs one level, so a parent used to fail with "no .csv and
    no .DTA files", which is true of that directory and useless to the reader:
    there are dozens of point files one level down. The error now names the
    sweep folders so the message doubles as the menu.
    """
    import csv_source

    campaign = tmp_path / "csv_files"
    for cond in ("Spectrum10_65degC_450A", "Spectrum11_65degC_150A"):
        d = campaign / cond
        d.mkdir(parents=True)
        (d / "p1.csv").write_text(
            "timestamp\ts1\ts2\ntimeshifts\t0.0\t1.1\n"
            "2026.04.20 10:21:11,792662\t1.9\t2.0\n")

    with pytest.raises(ValueError) as exc:
        csv_source.detect_dialect(campaign)

    message = str(exc.value)
    assert "campaign folder" in message
    assert "Spectrum10_65degC_450A" in message
    assert "Spectrum11_65degC_150A" in message
    assert "2 sweep folder(s)" in message


def test_a_genuinely_empty_folder_still_says_so(tmp_path):
    """The new message must not swallow the real 'there is nothing here'."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .csv and no .DTA files"):
        csv_source.detect_dialect(empty)
