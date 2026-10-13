"""
std_pipeline.py
===============
The STANDARD local-EIS evaluation, end to end, from the raw FAMOS signals.

A separate pipeline. It shares only the file readers (FAMOS, Gamry .DTA,
calibration files), the plate geometry and the plotting with the main
bronze/silver/gold pipeline -- every evaluation step here is the textbook
one, written out in this file:

  1. READ      each FAMOS card (all channels) and the Gamry .DTA of the
               condition: its ZCURVE table is the list of excitation
               frequencies, in sweep order.
  2. LOCATE    each step in each card's own record with a LOCK-IN: for every
               Gamry frequency f the channels are demodulated at f and
               averaged over a few periods; every instant belongs to the
               frequency whose tone is strongest there (relative to its own
               maximum). Each card is located on its own, so the cards need
               no time alignment.
  3. PHASORS   per step and channel: least-squares sine fit at the known f
               (cos, sin, offset, linear trend) over an integer number of
               periods, the first period skipped (settling). SNR from the
               fit residual.
  4. IMPEDANCE Z = K(T) * A_U / A_seg  (U = the cell-voltage channel of the
               same card, K = curr.csv calibration at the segment
               temperature), then only KNOWN corrections:
                 * the Abgleich chain calibration (gain file), if given;
                 * the ADC sampling delay of each channel's slot,
                   (slot_seg - slot_ref) / (n_channels * fs) -- a fixed
                   property of the multiplexed converter.
               Nothing is fitted to the data to correct it.
  5. QUALITY   a point is kept if the window holds >= 3 periods and both
               channels have SNR >= 3 dB; Lin-KK residual reported per
               segment (not used to drop data).
  6. PARAMETERS per segment, the standard equivalent circuit
               (standard_eis.py): R_ohmic = high-frequency intercept, then
               R_ohmic + jwL + (R_ct||CPE) + (R_mt||CPE) by CNLS; R_pol =
               R_ct + R_mt. Also j = u_dc/K, T, Re Z(1 kHz), |Z| and phase
               at 100 Hz.
  7. AGGREGATE the MEASURED segment impedances in parallel,
               Z_cell = A_used / sum(A_s / Z_s), compared with the Gamry
               whole-cell spectrum (x 304.92 cm2).
  8. OUTPUT    CSV tables, the Nyquist/Bode/phase figure of the measured
               spectra, one 2D interpolated plate map per parameter.

Run:  run(StdConfig(...))  ->  dict with the output paths and the tables.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

import r2d2_geometry as geom
import standard_eis

A_CELL_CM2 = 304.92
T_FALLBACK_C = 58.4

#: parameters drawn as plate maps (display units: mOhm*cm2, A/cm2, deg, degC)
MAP_PARAMS = ("R_ohmic", "ReZ_1kHz", "R_ct", "R_mt", "R_pol", "j_dc",
              "Z_mag_100Hz", "phase_100Hz", "T_degC")


@dataclass
class StdConfig:
    leepa: str = ""
    condition: str = ""
    dat_dir: Path | None = None           # folder of the FAMOS .DAT files
    files: tuple = ()                     # or the card files themselves
    gamry_dta: Path | None = None         # the condition's Gamry sweep
    gamry_dir: Path | None = None         # or a folder to search it in
    gamry_version: str = ""
    curr_cal: Path | None = None
    temp_cal: Path | None = None
    gain_file: Path | None = None         # Abgleich chain calibration
    out_dir: Path = Path("std_results")
    f_min_hz: float = 0.15
    f_max_hz: float = 2400.0
    ref_channel: str = "UC2"
    min_cycles: float = 3.0
    min_snr_db: float = 3.0
    adc_skew: bool = True
    plate: str = "gen1"
    write_png: bool = True
    write_html: bool = True
    #: callable(path) -> FAMOS-like reader (default eis_local.FamosFile); the
    #: notebook passes the datago reader here when the data come from Delta
    reader: object = None
    #: lock-in locator: how many segment channels to use besides U
    n_locator_segments: int = 4

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("reader", None)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# 1. read
# ---------------------------------------------------------------------------

FILE_PATTERNS = (
    "Leepa_{leepa}_Current_{cond}_Test_*_Karte_*.DAT",
    "Leepa_RO{leepa}_Current_{cond}_Test_*_Karte_*.DAT",
    "RO{leepa}-*_Current_{cond}_Test_*_Karte_*.DAT",
    "{leepa}_Current_{cond}_Test_*_Karte_*.DAT",
)


def discover_files(dat_dir, leepa: str, condition: str) -> list[Path]:
    """The card files of one condition, card 1 first."""
    d = Path(dat_dir)
    for pat in FILE_PATTERNS:
        files = sorted(d.glob(pat.format(leepa=leepa, cond=condition)),
                       key=lambda p: int(re.sub(r"\D", "", p.stem.split("_")[-1]) or 0))
        if files:
            return files
    return []


def find_gamry(cfg: StdConfig) -> Path | None:
    """The Gamry .DTA of this condition."""
    if cfg.gamry_dta:
        return Path(cfg.gamry_dta)
    if not cfg.gamry_dir:
        return None
    import gamry_compare
    want = str(cfg.condition).upper()
    hits = [s for s in gamry_compare.find_cell_sweeps(
        cfg.gamry_dir, order_id=cfg.leepa or None,
        version=cfg.gamry_version or None) if s.condition.upper() == want]
    return hits[-1].path if hits else None


def gamry_steps(path, f_lo: float, f_hi: float) -> list[dict]:
    """The excitation frequencies in sweep order, with the Gamry's own step
    times (1 s resolution) for the expected dwell of each step."""
    import gamry_sync
    tl = gamry_sync.read_timeline(path)
    out = []
    prev_t = None
    for f, t in zip(tl.freq, tl.t_s):
        dwell = (t - prev_t) if prev_t is not None else np.nan
        prev_t = t
        if f_lo <= f <= f_hi:
            out.append({"freq_hz": float(f), "t_gamry_s": float(t),
                        "dwell_est_s": float(dwell) if np.isfinite(dwell) else np.nan})
    return out


def _sensor_key(name: str) -> str:
    d = re.sub(r"\D", "", name)
    return f"temp{d}" if d else name.lower()


# ---------------------------------------------------------------------------
# 2. locate the steps (lock-in)
# ---------------------------------------------------------------------------

def _lockin_track(x: np.ndarray, fs: float, f: float, avg_s: float,
                  grid: np.ndarray, e: np.ndarray | None = None) -> np.ndarray:
    """|tone at f| on the time grid (sample indices), averaged over avg_s."""
    n = x.size
    if e is None:
        e = np.exp(-2j * np.pi * f * np.arange(n) / fs)
    c = np.concatenate([[0.0 + 0j], np.cumsum(x * e)])
    half = max(int(round(avg_s * fs / 2)), 1)
    lo = np.clip(grid - half, 0, n)
    hi = np.clip(grid + half, 0, n)
    return 2.0 * np.abs(c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def locate_steps(chans: list[np.ndarray], fs: float, steps: list[dict],
                 grid_s: float = 0.005, min_peak_ratio: float = 3.0
                 ) -> list[dict]:
    """Window [start, stop) in samples for every step, found by lock-in.

    For each frequency the normalised tone strength s_f(t) (its lock-in
    magnitude divided by its own maximum, summed over the locator channels)
    is computed; every instant is given to the frequency with the largest
    s_f(t); the step's window is the run of instants it owns around its own
    maximum. A frequency whose tone never rises clearly above its own noise
    (max / median < min_peak_ratio) is reported as not found."""
    n = min(c.size for c in chans)
    grid = np.arange(0, n, max(int(round(grid_s * fs)), 1))
    S = np.zeros((len(steps), grid.size))
    found = np.zeros(len(steps), bool)
    t_idx = np.arange(n) / fs
    for i, st in enumerate(steps):
        f = st["freq_hz"]
        dwell = st.get("dwell_est_s", np.nan)
        dwell = 1.0 if not np.isfinite(dwell) else max(dwell, 0.5)
        avg = float(np.clip(10.0 / f, 1.0 / f, max(0.4 * dwell, 1.0 / f)))
        e = np.exp(-2j * np.pi * f * t_idx)
        tot = np.zeros(grid.size)
        ratio = 0.0
        for x in chans:
            m = _lockin_track(x[:n], fs, f, avg, grid, e)
            mx = float(m.max())
            med = float(np.percentile(m, 10)) or 1e-30   # the noise floor
            ratio = max(ratio, mx / med)
            tot += m / (mx or 1.0)
        S[i] = tot / len(chans)
        found[i] = ratio >= min_peak_ratio
    label = np.argmax(np.where(found[:, None], S, -1.0), axis=0)
    strong = S[label, np.arange(grid.size)] > 0.3
    out = []
    for i, st in enumerate(steps):
        row = dict(st)
        row.update(start=None, stop=None, found=bool(found[i]), note="")
        if not found[i]:
            row["note"] = "tone not found"
            out.append(row)
            continue
        own = (label == i) & strong
        k = int(np.argmax(S[i]))
        if not own[k]:
            row["note"] = "never the strongest tone"
            out.append(row)
            continue
        a = k
        while a > 0 and own[a - 1]:
            a -= 1
        b = k
        while b < grid.size - 1 and own[b + 1]:
            b += 1
        row["start"] = int(grid[a])
        row["stop"] = int(min(grid[b] + (grid[1] - grid[0] if grid.size > 1 else 1), n))
        out.append(row)
    # sweep order: a window that starts inside the previous one is cut
    prev_stop = None
    for row in out:
        if row["start"] is None:
            continue
        if prev_stop is not None and row["start"] < prev_stop:
            row["start"] = prev_stop
            row["note"] = (row["note"] + "; overlap cut").strip("; ")
        if row["stop"] <= row["start"]:
            row["start"] = row["stop"] = None
            row["note"] = "empty after overlap cut"
            continue
        prev_stop = row["stop"]
    return out


def refine_window(x: np.ndarray, fs: float, f: float, start: int, stop: int,
                  lo_lim: int, hi_lim: int, k_noise: float = 4.0
                  ) -> tuple[int, int]:
    """Exact step edges: fit the sine at f in the middle of the located
    window, then grow the window outward for as long as the signal still
    follows that sine (running rms of the residual over a quarter period
    below k_noise x the residual inside, or 5 % of the amplitude). The
    lock-in places a short low-frequency step only to ~0.6 period; this puts
    the edge where the next frequency begins."""
    per = fs / f
    a = int(start + 0.15 * (stop - start))
    b = int(stop - 0.15 * (stop - start))
    if b - a < per:
        a, b = start, stop
    lo, hi = max(lo_lim, 0), min(hi_lim, x.size)
    if b - a < 4 or hi - lo < 4:
        return start, stop
    t = np.arange(lo, hi) / fs
    w = 2 * np.pi * f
    M = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t), t - t[0]])
    core = slice(a - lo, b - lo)
    p, *_ = np.linalg.lstsq(M[core], x[a:b], rcond=None)
    r2 = (x[lo:hi] - M @ p) ** 2
    amp = float(np.hypot(p[0], p[1]))
    sigma2 = float(np.mean(r2[core]))
    thr = max(k_noise ** 2 * sigma2, (0.05 * amp) ** 2)
    q = max(int(round(per / 4)), 1)
    c = np.concatenate([[0.0], np.cumsum(r2)])
    run = (c[q:] - c[:-q]) / q                 # rms^2 over [i, i+q)
    bad = run > thr
    # walk outward from the core
    i = a - lo
    while i > 0 and not bad[i - 1]:
        i -= 1
    j = b - lo - q
    while j < bad.size - 1 and not bad[j + 1]:
        j += 1
    return lo + i, lo + j + q


# ---------------------------------------------------------------------------
# 3. phasors
# ---------------------------------------------------------------------------

def sine_fit(x: np.ndarray, fs: float, f: float) -> tuple[complex, float]:
    """Least-squares a*cos + b*sin + c + d*t at the known f.
    Returns (phasor A = a - jb, SNR in dB from the residual)."""
    t = np.arange(x.size) / fs
    w = 2 * np.pi * f
    M = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t), t])
    p, *_ = np.linalg.lstsq(M, x, rcond=None)
    r = x - M @ p
    amp2 = (p[0] ** 2 + p[1] ** 2) / 2.0
    noise = float(np.mean(r ** 2)) or 1e-30
    return complex(p[0], -p[1]), float(10 * np.log10(amp2 / noise))


def step_window(start: int, stop: int, fs: float, f: float,
                min_cycles: float) -> tuple[int, int, float] | None:
    """Integer number of periods inside [start, stop), the first skipped."""
    per = fs / f
    n_cyc = int(np.floor((stop - start) / per))
    if n_cyc >= min_cycles + 1:
        start = int(round(start + per))          # settling: drop one period
        n_cyc -= 1
    if n_cyc < min_cycles:
        return None
    return start, int(round(start + n_cyc * per)), float(n_cyc)


# ---------------------------------------------------------------------------
# 4.-7. one condition
# ---------------------------------------------------------------------------

def _default_reader(path):
    from eis_local import FamosFile
    return FamosFile(path)


def evaluate_card(fam, steps, cal, gain, cfg: StdConfig, T_seg: dict, log):
    """Locate the steps on this card and return its rows and schedule."""
    fs = float(fam.fs)
    ref = cfg.ref_channel if cfg.ref_channel in fam.names else (
        fam.uc_names[0] if fam.uc_names else None)
    if ref is None:
        log(f"    {Path(str(fam.path)).name}: no cell-voltage channel, skipped")
        return [], []
    segs = [s for s in fam.segment_names if s in geom.SEGMENTS]
    u = np.asarray(fam.channel(ref), float)
    # locator: U plus the strongest few segment channels
    loc = [u - u.mean()]
    var = []
    for s in segs:
        y = np.asarray(fam.channel(s), float)
        var.append((float(np.var(y)), s))
    for _v, s in sorted(var, reverse=True)[:cfg.n_locator_segments]:
        y = np.asarray(fam.channel(s), float)
        loc.append(y - y.mean())
    band = [st for st in steps if st["freq_hz"] < 0.45 * fs]
    located = locate_steps(loc, fs, band)
    # exact edges on the cell-voltage channel, bounded by the neighbours
    got = [r for r in located if r["start"] is not None]
    for k, r in enumerate(got):
        lo_lim = got[k - 1]["start"] if k > 0 else 0
        hi_lim = got[k + 1]["stop"] if k + 1 < len(got) else u.size
        a, b = refine_window(loc[0], fs, r["freq_hz"], r["start"], r["stop"],
                             lo_lim, hi_lim)
        r["start"], r["stop"] = a, b
    for k in range(1, len(got)):                 # no overlap after refining
        if got[k]["start"] < got[k - 1]["stop"]:
            mid = (got[k]["start"] + got[k - 1]["stop"]) // 2
            got[k - 1]["stop"], got[k]["start"] = mid, mid
    sched, windows = [], []
    for st in located:
        w = None
        if st["start"] is not None:
            w = step_window(st["start"], st["stop"], fs, st["freq_hz"], cfg.min_cycles)
        sched.append({"card": Path(str(fam.path)).stem, "freq_hz": st["freq_hz"],
                      "start": st["start"], "stop": st["stop"],
                      "fit_start": w[0] if w else None, "fit_stop": w[1] if w else None,
                      "cycles": w[2] if w else 0.0,
                      "note": st["note"] if w or st["note"] else "fewer than "
                              f"{cfg.min_cycles:g} periods"})
        if w:
            windows.append((st["freq_hz"], w))
    # U phasors once
    U = {f: sine_fit(u[a:b], fs, f) for f, (a, b, _c) in windows}
    n_ch = int(getattr(fam, "n_ch", len(fam.names)))
    slot_ref = fam.position(ref) if hasattr(fam, "position") else 0
    rows = []
    for s in segs:
        y = np.asarray(fam.channel(s), float)
        T = T_seg.get(s, T_FALLBACK_C)
        K = cal.K(s, T) if s in cal.seg_c0 else np.nan
        slot = fam.position(s) if hasattr(fam, "position") else 0
        dt = (slot - slot_ref) / (n_ch * fs) if cfg.adc_skew else 0.0
        j_dc = float(np.mean(y)) / K if np.isfinite(K) and K else np.nan
        for f, (a, b, cyc) in windows:
            A_seg, snr_s = sine_fit(y[a:b], fs, f)
            A_u, snr_u = U[f]
            Z = K * A_u / A_seg if A_seg != 0 else np.nan + 0j
            if gain:
                import utils
                Z = Z / complex(utils.gain_at(gain, s, np.array([f]))[0])
            Z = Z * np.exp(2j * np.pi * f * dt)
            rows.append({"segment": s, "card": Path(str(fam.path)).stem,
                         "freq_hz": f, "Z": Z, "snr_seg_db": snr_s,
                         "snr_u_db": snr_u, "cycles": cyc, "slot": slot,
                         "slot_ref": slot_ref, "K": K, "T_degC": T,
                         "j_dc": j_dc})
    log(f"    {Path(str(fam.path)).name}: {len(windows)}/{len(located)} steps "
        f"located, {len(segs)} segments")
    return rows, sched


def _plate_temperatures(fams, cal) -> dict[str, float]:
    sensor = {}
    for fam in fams:
        for nm in getattr(fam, "temp_names", []):
            key = _sensor_key(nm)
            if key in cal.temp_c0:
                T = cal.temperature(key, float(np.mean(fam.channel(nm))))
                if 0 < T < 120:
                    sensor.setdefault(key, T)
    if not sensor:
        return {}
    return geom.segment_temperatures(sensor)


def run(cfg: StdConfig, log=print) -> dict:
    """Evaluate one condition; returns the tables and the written paths."""
    import pandas as pd
    from eis_local import PlateCalibration
    import utils
    from eis_validation import lin_kk

    t0 = time.time()
    geom.use_plate(cfg.plate)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reader = cfg.reader or _default_reader
    files = list(cfg.files) or discover_files(cfg.dat_dir, cfg.leepa, cfg.condition)
    if not files:
        raise SystemExit(f"std_pipeline: no card files for {cfg.leepa} {cfg.condition}")
    dta = find_gamry(cfg)
    if dta is None:
        raise SystemExit(f"std_pipeline: no Gamry .DTA for {cfg.condition} -- the "
                         f"standard evaluation needs its list of excitation "
                         f"frequencies")
    steps = gamry_steps(dta, cfg.f_min_hz, cfg.f_max_hz)
    log(f"  {cfg.leepa} {cfg.condition}: {len(files)} cards, Gamry {Path(dta).name}: "
        f"{len(steps)} frequencies in {cfg.f_min_hz:g}–{cfg.f_max_hz:g} Hz")
    cal = PlateCalibration.load(cfg.curr_cal, cfg.temp_cal)
    gain = utils.load_gain(cfg.gain_file) if cfg.gain_file else {}

    fams = [reader(p) for p in files]
    T_seg = _plate_temperatures(fams, cal)
    rows, sched = [], []
    for fam in fams:
        r, s = evaluate_card(fam, steps, cal, gain, cfg, T_seg, log)
        rows += r
        sched += s
    if not rows:
        raise SystemExit("std_pipeline: no segment spectra")
    df = pd.DataFrame(rows)
    # a segment recorded on two cards: keep the card with the higher SNR
    best = (df.groupby(["segment", "card"]).snr_seg_db.median().reset_index()
              .sort_values("snr_seg_db").drop_duplicates("segment", keep="last"))
    df = df.merge(best[["segment", "card"]], on=["segment", "card"])
    # polarity: a segment wired the other way round reads Re Z < 0 at low f
    for s, d in df.groupby("segment"):
        lf = d.nsmallest(3, "freq_hz").Z.values
        if np.median(lf.real) < 0:
            df.loc[d.index, "Z"] = -d.Z.values
            df.loc[d.index, "polarity_flipped"] = True
    df["kept"] = ((df.cycles >= cfg.min_cycles) & (df.snr_seg_db >= cfg.min_snr_db)
                  & (df.snr_u_db >= cfg.min_snr_db) & np.isfinite(df.Z.values.real))
    df["z_re_mohm_cm2"] = 1000 * df.Z.values.real
    df["z_im_mohm_cm2"] = 1000 * df.Z.values.imag

    # ---- parameters per segment
    params = []
    for s, d in df.groupby("segment"):
        k = d[d.kept].sort_values("freq_hz")
        f, Z = k.freq_hz.values, k.Z.values
        seg = geom.SEGMENTS[s]
        row = {"segment": int(s), "class": "measured", "card": d.card.iloc[0],
               "area_cm2": seg.area_cm2, "cx_mm": seg.cx_mm, "cy_mm": seg.cy_mm,
               "n_points": int(f.size), "n_dropped": int((~d.kept).sum()),
               "j_dc": d.j_dc.iloc[0], "T_degC": d.T_degC.iloc[0]}
        r = standard_eis.fit_spectrum(f, Z) if f.size >= 8 else {"ok": False}
        if r.get("ok"):
            row.update(R_ohmic=1e3 * r["R_ohmic"], R_ohmic_how=r["R_ohmic_method"],
                       R_ct=1e3 * r["R_ct"], R_mt=1e3 * r["R_mt"],
                       R_pol=1e3 * r["R_pol"], R_pol_lf=1e3 * r["R_pol_lf"],
                       f_ct_hz=r["f_ct_hz"], f_mt_hz=r["f_mt_hz"],
                       n_ct=r["params"]["n1"], n_mt=r["params"]["n2"],
                       L_nH=1e9 * r["L"], fit_rms_pct=r["rms_pct"],
                       fit_verdict=r["verdict"])
        else:
            row.update(fit_verdict="no fit: " + str(r.get("reason", f"{f.size} points")))
        if f.size >= 6:
            kk = lin_kk(f, Z)
            row["kk_res_pct"] = 100 * float(kk.get("res_max", np.nan))
        lf = np.log(f) if f.size else f
        if f.size >= 2 and f.min() <= 1000 <= f.max():
            row["ReZ_1kHz"] = 1e3 * float(np.interp(np.log(1000), lf, Z.real))
        if f.size:
            i = int(np.argmin(np.abs(f - 100)))
            if abs(f[i] - 100) <= 50:
                row["Z_mag_100Hz"] = 1e3 * float(abs(Z[i]))
                row["phase_100Hz"] = float(np.degrees(np.angle(Z[i])))
        params.append(row)
    P = pd.DataFrame(params).sort_values("segment")
    measured = set(P.segment.astype(str))
    for s in geom.SEGMENTS:
        if s not in measured:
            sg = geom.SEGMENTS[s]
            P = pd.concat([P, pd.DataFrame([{"segment": int(s), "class": "not measured",
                                             "area_cm2": sg.area_cm2, "cx_mm": sg.cx_mm,
                                             "cy_mm": sg.cy_mm}])], ignore_index=True)
    P = P.sort_values("segment").reset_index(drop=True)

    # ---- aggregate of the MEASURED spectra, segments in parallel
    agg = []
    for f, d in df[df.kept].groupby("freq_hz"):
        A = np.array([geom.SEGMENTS[s].area_cm2 for s in d.segment])
        Y = np.sum(A / d.Z.values)
        agg.append({"freq_hz": f, "z_re_mohm_cm2": 1e3 * (A.sum() / Y).real,
                    "z_im_mohm_cm2": 1e3 * (A.sum() / Y).imag,
                    "n_segments": int(len(d)), "area_cm2": float(A.sum()),
                    "area_coverage": float(A.sum() / A_CELL_CM2)})
    AG = pd.DataFrame(agg).sort_values("freq_hz")

    # ---- Gamry whole cell on the same frequencies
    import gamry_compare
    sw = gamry_compare.read_cell_sweep(dta)
    zg = sw.Z_ohm * A_CELL_CM2 * 1e3
    GC = AG[["freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2", "area_coverage"]].copy()
    GC["gamry_re_mohm_cm2"] = np.interp(np.log(GC.freq_hz), np.log(sw.freq), zg.real)
    GC["gamry_im_mohm_cm2"] = np.interp(np.log(GC.freq_hz), np.log(sw.freq), zg.imag)
    zl = GC.z_re_mohm_cm2 + 1j * GC.z_im_mohm_cm2
    zr = GC.gamry_re_mohm_cm2 + 1j * GC.gamry_im_mohm_cm2
    GC["mag_diff_pct"] = 100 * (np.abs(zl) / np.abs(zr) - 1)
    GC["phase_diff_deg"] = np.degrees(np.angle(zl / zr))

    # ---- write
    sp = df[["segment", "card", "freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2",
             "snr_seg_db", "snr_u_db", "cycles", "kept"]].copy()
    sp.to_csv(out / "spectra_all.csv", index=False)
    sp[sp.kept].drop(columns="kept").to_csv(out / "spectra.csv", index=False)
    pd.DataFrame(sched).to_csv(out / "schedule.csv", index=False)
    P.to_csv(out / "parameters.csv", index=False)
    AG.to_csv(out / "aggregate.csv", index=False)
    GC.to_csv(out / "gamry_comparison.csv", index=False)
    import plate_stats
    plate_stats.write_statistics(out / "parameters.csv", out / "plate_statistics.csv")
    manifest = {"pipeline": "standard", "config": cfg.to_dict(),
                "gamry_dta": str(dta), "cards": [str(p) for p in files],
                "n_segments": int(len(measured)),
                "n_points_kept": int(df.kept.sum()), "n_points": int(len(df)),
                "elapsed_s": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    figs = write_figures(out, sp[sp.kept], P, AG, GC, cfg) if cfg.write_png else {}
    log(f"  done in {time.time() - t0:.0f} s: {len(measured)} segments, "
        f"{int(df.kept.sum())}/{len(df)} points kept -> {out}")
    return {"out_dir": out, "spectra": sp, "parameters": P, "aggregate": AG,
            "gamry": GC, "schedule": pd.DataFrame(sched), "figures": figs,
            "manifest": manifest}


# ---------------------------------------------------------------------------
# 8. figures
# ---------------------------------------------------------------------------

def write_figures(out: Path, spectra, P, AG, GC, cfg: StdConfig) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nyquist
    import plate_figure
    written = {}
    title = f"{cfg.leepa} / {cfg.condition} — standard evaluation"
    try:
        nyquist.save_png(spectra, out / "nyquist.png", title=title)
        written["nyquist_png"] = out / "nyquist.png"
        if cfg.write_html:
            fig = nyquist.figure(spectra, title=title)
            fig.write_html(str(out / "nyquist.html"), include_plotlyjs="cdn")
            written["nyquist_html"] = out / "nyquist.html"
    except Exception as exc:                                # noqa: BLE001
        print(f"  nyquist figure skipped: {exc}")
    cls = {str(int(r.segment)): ("measured" if r["class"] == "measured" else "inferred")
           for _, r in P.iterrows()}
    for p in MAP_PARAMS:
        if p not in P.columns:
            continue
        vals = {str(int(r.segment)): r[p] for _, r in P.iterrows()
                if r["class"] == "measured" and np.isfinite(r[p])}
        if len(vals) < 4:
            continue
        fig = plate_figure.draw_flow_plate(vals, p, classes=cls,
                                           render="interpolated", title=title)
        path = out / f"plate_{p}_interp.png"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        written[f"map_{p}"] = path
    # aggregate of the measured spectra vs the Gamry
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    ax[0].plot(GC.gamry_re_mohm_cm2, -GC.gamry_im_mohm_cm2, "k.-", lw=2, label="Gamry whole cell")
    ax[0].plot(AG.z_re_mohm_cm2, -AG.z_im_mohm_cm2, "r.-", lw=1.5,
               label="measured segments in parallel")
    ax[0].set_xlabel("Z' / mΩ·cm²"); ax[0].set_ylabel("-Z'' / mΩ·cm²")
    ax[0].axhline(0, color="grey", lw=.5); ax[0].legend(); ax[0].set_title("Nyquist")
    ax[1].semilogx(GC.freq_hz, GC.mag_diff_pct, "b.-", label="|Z| difference %")
    ax[1].semilogx(GC.freq_hz, GC.phase_diff_deg, "g.-", label="phase difference °")
    ax[1].axhline(0, color="grey", lw=.5); ax[1].legend(); ax[1].set_xlabel("f / Hz")
    ax[1].set_title("aggregate − Gamry")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "gamry_vs_aggregate.png", dpi=140)
    plt.close(fig)
    written["gamry_png"] = out / "gamry_vs_aggregate.png"
    return written
