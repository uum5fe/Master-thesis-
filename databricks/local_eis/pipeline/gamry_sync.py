#!/usr/bin/env python3
"""
gamry_sync.py
=============
Put the FAMOS recording on the Gamry's clock, and let the Gamry say which
frequency was applied when.

WHY
---
The local EIS is excited by the Gamry: the FAMOS cards record the cell's
response to the SAME galvanostatic sweep that the whole-cell .dta describes.
On 2612030 / 60 A every FAMOS dwell window between 0.3 Hz and 1.2 kHz ends
at  t_gamry + 99.2 s  (+/- 0.5 s, the .dta's one-second time stamps). The
Gamry therefore knows two things the blind schedule detector has to guess:

  * THE EXACT FREQUENCY of every step. The detector snaps its estimates onto
    a geometric ladder; the Gamry's synthesiser does not produce one, and the
    two differ by up to 1 % (474.77 vs 477.43 Hz, 1501.4 vs 1516.5 Hz).
    The phasor fit is evaluated at the stated frequency, and bronze's own
    note on the snap measures a 1 % error over a 0.25 s dwell at 600 Hz as
    ~16 dB of phasor SNR.
  * WHEN each step happened. Above 1.2 kHz the detector placed 1501, 1890
    and 2379 Hz at 85, 60 and 35 s -- by interpolation -- and 2995 Hz at
    11 s, while the Gamry ran them at ~109 s. Silver then fitted the right
    sine over the wrong stretch of record and every segment rejected the
    point, which is why the band stopped at 1.2 kHz although the cell's arc
    closes near 4 kHz.

WHAT IT DOES (cfg.gamry_sync)
-----------------------------
  "report"     match the detected steps to the Gamry points, measure the
               clock offset, and write the table (bronze/gamry_sync.csv):
               per Gamry point, the step that matched, its frequency error
               and its time residual -- "ok", "misplaced" or "missing".
  "frequency"  as report, and give every matched step the Gamry's exact
               frequency (no raw data needed; windows untouched).
  "guide"      as frequency, and re-locate every misplaced or missing step
               inside the time slot the Gamry predicts (bounded by its
               neighbours in the sweep), on the segment-channel array, and
               keep it only if the CFAR test of hf_schedule confirms the tone
               there -- a prediction is never accepted on its own.
  "off"        nothing.

The offset is measured, not assumed: only steps the detector itself found
(window_source == "detected", SNR >= 3 dB) vote, and the run is refused if
fewer than 5 agree to within 1.5 s.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

MODES = ("off", "report", "frequency", "guide")


def mode(cfg) -> str:
    m = str(getattr(cfg, "gamry_sync", "off") or "off").lower()
    return m if m in MODES else "off"


# ---------------------------------------------------------------------------
# the Gamry side
# ---------------------------------------------------------------------------

@dataclass
class Timeline:
    """One Gamry sweep in the order it was run."""
    path: Path
    freq: np.ndarray          # Hz, exact, in measurement order
    t_s: np.ndarray           # s since the sweep started (1 s resolution)
    started: datetime | None = None
    iac_rms: float = float("nan")


def _num(s: str) -> float:
    s = s.strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def read_timeline(path) -> Timeline:
    """Freq and Time columns of a Gamry .dta ZCURVE table."""
    path = Path(path)
    lines = path.read_text(encoding="latin-1").splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.startswith("ZCURVE"))
    head = [h.strip() for h in lines[k + 1].split("\t")]
    i_t, i_f = head.index("Time"), head.index("Freq")
    f, t = [], []
    for ln in lines[k + 3:]:
        p = ln.split("\t")
        if len(p) <= max(i_t, i_f) or not ln.strip():
            break
        f.append(_num(p[i_f]))
        t.append(_num(p[i_t]))
    started, iac = None, float("nan")
    for ln in lines[:k]:
        m = re.match(r"STARTTIME\s+LABEL\s+([\d.]+\s+[\d:]+)", ln)
        if m:
            try:
                started = datetime.strptime(m.group(1), "%d.%m.%Y %H:%M:%S")
            except ValueError:
                pass
        if ln.startswith("IACREQ"):
            iac = _num(ln.split("\t")[2]) if len(ln.split("\t")) > 2 else iac
    f, t = np.asarray(f, float), np.asarray(t, float)
    ok = np.isfinite(f) & np.isfinite(t)
    return Timeline(path, f[ok], t[ok], started, iac)


def read_vdc(path) -> float:
    """Median of the Vdc column of a Gamry .dta ZCURVE table (the cell's DC
    voltage at the Gamry's own sense leads), nan when there is none."""
    path = Path(path)
    lines = path.read_text(encoding="latin-1").splitlines()
    try:
        k = next(i for i, ln in enumerate(lines) if ln.startswith("ZCURVE"))
        head = [h.strip() for h in lines[k + 1].split("\t")]
        i_v = head.index("Vdc")
    except (StopIteration, ValueError):
        return float("nan")
    v = []
    for ln in lines[k + 3:]:
        p = ln.split("\t")
        if len(p) <= i_v or not ln.strip():
            break
        v.append(_num(p[i_v]))
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def find_timeline(cfg, log=None) -> Timeline | None:
    """The sweep for cfg.condition under cfg.gamry_dir (same rules as the
    whole-cell comparison: order and build token decide the campaign)."""
    root = getattr(cfg, "gamry_dir", None)
    if not root or not Path(root).exists():
        return None
    import gamry_compare
    found = gamry_compare.find_cell_sweeps(
        Path(root), order_id=getattr(cfg, "leepa", None),
        version=getattr(cfg, "gamry_version", "") or None)
    want = str(getattr(cfg, "condition", "") or "").upper()
    hits = [s for s in found if s.condition.upper() == want]
    if not hits:
        if log is not None:
            log.info(f"  gamry sync: no whole-cell sweep for {want or '?'} "
                     f"under {root}")
        return None
    if len(hits) > 1 and log is not None:
        log.warning(f"  gamry sync: {len(hits)} sweeps for {want}; using "
                    f"{hits[-1].path.name}")
    return read_timeline(hits[-1].path)


# ---------------------------------------------------------------------------
# matching and the clock offset
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    ok: bool
    offset_s: float = float("nan")      # FAMOS window end - Gamry time stamp
    spread_s: float = float("nan")      # robust spread of that, over voters
    n_voters: int = 0
    rows: list = field(default_factory=list)
    reason: str = ""

    def summary(self) -> dict:
        count = {}
        for r in self.rows:
            count[r["verdict"]] = count.get(r["verdict"], 0) + 1
        return {"ok": self.ok, "offset_s": round(self.offset_s, 3)
                if np.isfinite(self.offset_s) else None,
                "spread_s": round(self.spread_s, 3)
                if np.isfinite(self.spread_s) else None,
                "n_voters": self.n_voters, "verdicts": count,
                "reason": self.reason}


def _match(f: float, freqs: np.ndarray, tol: float) -> int:
    d = np.abs(np.log(freqs / f))
    j = int(np.argmin(d))
    return j if d[j] <= np.log1p(tol) else -1


def align(steps, tl: Timeline, fs: float, f_lo: float = 0.0,
          f_hi: float = np.inf, tol: float = 0.03, min_snr_db: float = 3.0,
          max_spread_s: float = 1.5, time_tol_s: float = 1.5) -> SyncResult:
    """Offset between the FAMOS common time base and the Gamry clock, and a
    verdict for every Gamry point inside the analysis band.

    A step is "ok" if its window ends within max(time_tol_s, 3 periods) of
    the Gamry's stamp: the stamps have one-second resolution, and at the
    bottom of the band a dwell lasts many seconds and the detector may pick
    any stretch of it."""
    steps = list(steps)
    if not steps or tl.freq.size == 0:
        return SyncResult(False, reason="nothing to match")
    sf = np.array([s.freq for s in steps], float)
    end = np.array([s.stop / fs for s in steps], float)

    # voters: steps the detector really found, matched to a Gamry point
    off = []
    for s, e in zip(steps, end):
        if getattr(s, "window_source", "detected") != "detected":
            continue
        if not (np.isfinite(s.snr_db) and s.snr_db >= min_snr_db):
            continue
        j = _match(s.freq, tl.freq, tol)
        if j >= 0:
            off.append(e - tl.t_s[j])
    off = np.asarray(off, float)
    if off.size < 5:
        return SyncResult(False, n_voters=int(off.size),
                          reason=f"only {off.size} confident steps matched a "
                                 f"Gamry point (need 5)")
    offset = float(np.median(off))
    spread = float(1.4826 * np.median(np.abs(off - offset)))
    agree = np.abs(off - offset) <= max_spread_s
    if agree.sum() < 5 or spread > max_spread_s:
        return SyncResult(False, offset, spread, int(agree.sum()),
                          reason=f"the matched steps do not share one clock "
                                 f"offset (spread {spread:.2f} s)")

    rows = []
    for j, (fg, tg) in enumerate(zip(tl.freq, tl.t_s)):
        if not (f_lo <= fg <= f_hi):
            continue
        i = _match(fg, sf, tol)
        row = {"gamry_index": j, "f_gamry_hz": float(fg), "t_gamry_s": float(tg),
               "pred_end_s": float(tg + offset)}
        if i < 0:
            row.update(verdict="missing")
        else:
            s = steps[i]
            resid = float(end[i] - (tg + offset))
            row.update(step_index=i, f_famos_hz=float(s.freq),
                       freq_err_pct=100.0 * (s.freq / fg - 1.0),
                       famos_start_s=float(s.start / fs),
                       famos_end_s=float(end[i]), time_resid_s=resid,
                       window_source=getattr(s, "window_source", "detected"),
                       snr_db=float(s.snr_db),
                       verdict="ok" if abs(resid) <= max(time_tol_s, 3.0 / fg)
                       else "misplaced")
        rows.append(row)
    return SyncResult(True, offset, spread, int(agree.sum()), rows)


def apply_frequencies(steps, sync: SyncResult, rebuild) -> tuple[list, int]:
    """Matched steps take the Gamry's exact frequency. Windows untouched."""
    steps = list(steps)
    n = 0
    for r in sync.rows:
        i = r.get("step_index")
        if i is None or r["verdict"] != "ok":
            continue
        if abs(steps[i].freq - r["f_gamry_hz"]) > 1e-9:
            steps[i] = rebuild(steps[i], freq=float(r["f_gamry_hz"]))
            n += 1
    return steps, n


# ---------------------------------------------------------------------------
# confining low-frequency windows to their Gamry step
# ---------------------------------------------------------------------------

def confine_windows(steps, sync: SyncResult, fs: float, rebuild,
                    min_interval_s: float = 3.0, guard_s: float = 0.5,
                    settle_periods: float = 0.5, min_cycles: float = 2.5,
                    tol: float = 0.03, log=None) -> tuple[list, list]:
    """Put every long (low-frequency) step's window inside its Gamry step.

    WHY. The blind detector finds a step by demodulating the cell voltage at
    its frequency over at most 6 s. Below ~1 Hz that cannot tell the tone from
    its neighbour on the sweep (0.19 and 0.24 Hz are 0.05 Hz apart, a 6 s
    window resolves ~0.17 Hz), so the window grows into the neighbouring step
    and is then extended by another demodulation length. align() only checks
    where a window ENDS, with a tolerance of 3 periods (16 s at 0.19 Hz), so
    the overlap passed as "ok". Measured on RO2612030 at 450 A: the 0.19 Hz
    window started 9.6 s before its Gamry step, the 0.24 Hz one ended 6.9 s
    early; in a simulation of that sweep the misplaced windows put up to 4-5 %
    error on single points (3-6 Hz) of an outlet spectrum, alternating in sign
    from step to step -- a zig-zag along the arc.

    WHAT. The Gamry stamps the end of every point; the step of frequency f_k
    is the interval between the previous point's stamp and its own, mapped to
    the FAMOS base with the fitted offset. For every step whose interval is at
    least `min_interval_s` long (shorter steps share a one-second stamp and
    cannot be bounded this way) the window becomes

        [t_prev + guard + settle, t_own - guard]

    with guard >= the offset spread (stamps have 1 s resolution) and settle =
    `settle_periods` periods of the new tone (the transient after a frequency
    change), capped at a quarter of the step. If fewer than `min_cycles` cycles
    remain, the settle is dropped; if still fewer, the step is left as it was.

    Returns (steps, rows); each row names the step, the old and the new window.
    """
    steps = list(steps)
    if not sync.ok or not steps:
        return steps, []
    g = max(float(guard_s), float(sync.spread_s) if np.isfinite(sync.spread_s)
            else 0.0)
    freqs = np.array([s.freq for s in steps], float)
    rows = sorted(sync.rows, key=lambda r: r["gamry_index"])
    out = []
    for prev, r in zip(rows[:-1], rows[1:]):
        if r["gamry_index"] != prev["gamry_index"] + 1:
            continue                       # predecessor outside the band
        f = float(r["f_gamry_hz"])
        t0, t1 = float(prev["pred_end_s"]), float(r["pred_end_s"])
        if t1 - t0 < min_interval_s:
            continue
        i = _match(f, freqs, tol)
        if i < 0:
            continue
        settle = min(settle_periods / f, 0.25 * (t1 - t0))
        a, b = t0 + g + settle, t1 - g
        if (b - a) * f < min_cycles:
            a = t0 + g
        if (b - a) * f < min_cycles:
            continue
        old = (steps[i].start / fs, steps[i].stop / fs)
        steps[i] = rebuild(steps[i], start=int(round(a * fs)),
                           stop=int(round(b * fs)), window_source="gamry_step")
        out.append({"f_hz": f, "old_start_s": round(old[0], 3),
                    "old_end_s": round(old[1], 3), "new_start_s": round(a, 3),
                    "new_end_s": round(b, 3),
                    "shift_start_s": round(a - old[0], 3),
                    "shift_end_s": round(b - old[1], 3)})
    if log is not None and out:
        big = [o for o in out if max(abs(o["shift_start_s"]),
                                     abs(o["shift_end_s"])) > 1.0]
        log.info(f"  {len(out)} low-frequency window(s) confined to their "
                 f"Gamry step ({len(big)} moved by more than 1 s)")
    return steps, out


def separate_windows(steps, fs: float, rebuild, min_cycles: float = 2.0,
                     log=None) -> tuple[list, int]:
    """No two windows may overlap: one stretch of record holds one tone.

    Steps are taken in time order; where a window runs into the next one the
    two are cut at the middle of the overlap. A cut that would leave fewer
    than `min_cycles` cycles in either window is not made (that step is left
    for the quality gates). Returns (steps, number of cuts).
    """
    steps = list(steps)
    order = sorted(range(len(steps)), key=lambda k: steps[k].start)
    n = 0
    for a_i, b_i in zip(order[:-1], order[1:]):
        A, B = steps[a_i], steps[b_i]
        if A.stop <= B.start:
            continue
        cut = (A.stop + B.start) // 2
        if ((cut - A.start) / fs * A.freq < min_cycles
                or (B.stop - cut) / fs * B.freq < min_cycles):
            continue
        steps[a_i] = rebuild(A, stop=int(cut))
        steps[b_i] = rebuild(B, start=int(cut))
        n += 1
    if log is not None and n:
        log.info(f"  {n} overlapping window pair(s) cut apart")
    return steps, n


# ---------------------------------------------------------------------------
# re-locating a step in the slot the Gamry predicts
# ---------------------------------------------------------------------------

def _lockin_power(chans, fs: float, f: float, a: int, b: int,
                  hop: int, win: int) -> tuple[np.ndarray, np.ndarray]:
    """Array power at f in sliding windows over [a, b): sum over channels of
    |phasor|^2 -- sign-free, so a reversed sense pair cannot cancel."""
    import hf_schedule
    names = hf_schedule._names(chans)
    starts = np.arange(a, max(a + 1, b - win), hop, dtype=int)
    if starts.size == 0:
        return starts, np.zeros(0)
    n = np.arange(win)
    c = np.exp(-2j * np.pi * f * n / fs) * np.hanning(win)
    p = np.zeros(starts.size)
    for nm in names:
        y = hf_schedule._window(chans, nm, a, b + win)
        if y.size < (b - a) // 2:
            continue
        y = y - np.mean(y)
        for k, s in enumerate(starts):
            seg = y[s - a: s - a + win]
            if seg.size == win:
                p[k] += abs(np.dot(seg, c)) ** 2
    return starts, p


def locate(chans, fs: float, f: float, lo: int, hi: int,
           win_cycles: float = 24.0, min_dwell_cycles: float = 30.0
           ) -> tuple[int, int] | None:
    """The dwell of a tone at f inside [lo, hi): the contiguous run around
    the strongest array response where the power stays above a quarter of
    its peak, trimmed by 10 % at each end.

    The lock-in window is 24 cycles long: the sweep's neighbouring
    frequencies are only 26 % apart (10 points per decade), and a shorter
    window lets the neighbour's dwell leak in -- measured, an 8-cycle window
    "found" 1884 Hz inside the 1516 Hz dwell after the real one was removed.
    A run shorter than `min_dwell_cycles` is not a dwell."""
    win = int(max(win_cycles * fs / f, 64))
    if hi - lo < 2 * win:
        return None
    hop = max(win // 4, 1)
    starts, p = _lockin_power(chans, fs, f, lo, hi, hop, win)
    if p.size == 0 or not np.isfinite(p).all() or p.max() <= 0:
        return None
    k = int(np.argmax(p))
    on = p >= 0.25 * p[k]
    i0 = k
    while i0 > 0 and on[i0 - 1]:
        i0 -= 1
    i1 = k
    while i1 < p.size - 1 and on[i1 + 1]:
        i1 += 1
    a, b = int(starts[i0]), int(starts[i1] + win)
    trim = int(0.1 * (b - a))
    a, b = a + trim, b - trim
    if b - a < max(win, min_dwell_cycles * fs / f):
        return None
    return a, b


def relocate(steps, sync: SyncResult, tl: Timeline, chans, fs: float,
             lag: int, rebuild, make_step, log=None,
             slot_before_s: float = 2.0, slot_after_s: float = 1.0,
             min_window_snr_db: float | None = None
             ) -> tuple[list, list]:
    """Re-locate misplaced and missing steps in their Gamry slot -- and every
    window that was only interpolated.

    An interpolated window (bronze's window sanity guessed it from its
    neighbours) can sit inside the Gamry's time tolerance and still miss the
    tone: on 2612030 at 45 A and 450 A the 1884 and 2391 Hz windows passed
    as "ok" and were then rejected for drift in all 67 segments, which cut
    the band at 1.5 kHz. Such a window is searched for like a misplaced one;
    if nothing verifiable is found it is kept as it was.

    `chans`: the segment channels of ONE card (hf_schedule.LazyChannels or a
    dict), in that card's own sample index; `lag` is that card's shift onto
    the common base (common = own - lag). Each new window is bounded by the
    windows of its sweep neighbours and verified by hf_schedule's CFAR test;
    a step that fails keeps what it had (a misplaced one is dropped, since
    its window is known to be wrong). Returns (steps, per-step log rows).

    `min_window_snr_db`: also search an "ok" window whose segment channels
    hold no tone at its frequency (median channel SNR below this). The time
    check alone cannot see it at the top of the sweep, where the Gamry runs
    several steps per second and the stamps have one-second resolution: on
    2612030 the 3797 Hz window passed as "ok" with every segment at -30 dB.
    """
    import hf_schedule
    steps = list(steps)
    by_index = {r["gamry_index"]: r for r in sync.rows}
    order = sorted(by_index)                  # Gamry measurement order
    # common-base windows of the points already trusted, for the bounds
    empty = set()
    if min_window_snr_db is not None:
        for g, r in by_index.items():
            if r["verdict"] != "ok" or r.get("window_source") == "interpolated":
                continue
            s = steps[r["step_index"]]
            try:
                snr = hf_schedule._median_channel_snr_db(
                    chans, fs, r["f_gamry_hz"], int(s.start + lag),
                    int(s.stop + lag))
            except Exception:                               # noqa: BLE001
                continue
            if np.isfinite(snr) and snr < min_window_snr_db:
                empty.add(g)

    def guessed(r):
        return r["verdict"] == "ok" and (
            r.get("window_source") == "interpolated"
            or r["gamry_index"] in empty)

    trusted = {g: (steps[r["step_index"]].start, steps[r["step_index"]].stop)
               for g, r in by_index.items()
               if r["verdict"] == "ok" and not guessed(r)}
    out_rows, drop = [], set()
    for g in order:
        r = by_index[g]
        if r["verdict"] == "ok" and not guessed(r):
            continue
        f = r["f_gamry_hz"]
        T = r["pred_end_s"] * fs                         # common base
        lo = T - slot_before_s * fs
        hi = T + slot_after_s * fs
        prev = [trusted[x][1] for x in trusted if x < g]
        nxt = [trusted[x][0] for x in trusted if x > g]
        if prev:
            lo = max(lo, max(prev))
        if nxt:
            hi = min(hi, min(nxt))
        lo, hi = int(lo + lag), int(hi + lag)            # card's own index
        was = ("empty" if g in empty else
               "interpolated" if guessed(r) else r["verdict"])
        row = {"gamry_index": g, "f_gamry_hz": f, "was": was,
               "search_s": (round((lo - lag) / fs, 3), round((hi - lag) / fs, 3))}
        win = locate(chans, fs, f, lo, hi) if hi > lo else None
        if win is None:
            row.update(result="not found in slot")
        else:
            a, b = win
            cf = hf_schedule.cfar_channel_count(chans, fs, f, a, b)
            r1 = hf_schedule.rank1_statistic(chans, fs, f, a, b)
            if not cf.accept():
                row.update(result="CFAR rejected", ratio_db=cf.ratio_db)
                win = None
            elif r1.usable() and not r1.accept():
                # a tone common to the array is rank-1; a spur is not
                row.update(result="rank-1 test rejected",
                           ratio_db=cf.ratio_db)
                win = None
            else:
                snr = hf_schedule._median_channel_snr_db(chans, fs, f, a, b)
                a_c, b_c = a - lag, b - lag
                if r["verdict"] in ("misplaced", "ok"):
                    i = r["step_index"]
                    steps[i] = rebuild(steps[i], freq=float(f), start=int(a_c),
                                       stop=int(b_c), snr_db=float(snr),
                                       window_source="gamry")
                else:
                    steps.append(make_step(float(f), int(a_c), int(b_c),
                                           float(snr)))
                trusted[g] = (a_c, b_c)
                row.update(result="relocated", start_s=round(a_c / fs, 4),
                           end_s=round(b_c / fs, 4), snr_db=round(snr, 2),
                           ratio_db=round(float(cf.ratio_db), 2))
        if win is None and r["verdict"] == "misplaced":
            drop.add(r["step_index"])
            row["result"] = row["result"] + "; wrong window dropped"
        elif win is None and guessed(r):
            row["result"] = row["result"] + f"; {was} window kept"
        out_rows.append(row)
        if log is not None:
            log.info(f"    {f:9.2f} Hz ({was}): {row['result']}")
    steps = [s for i, s in enumerate(steps) if i not in drop]
    steps.sort(key=lambda s: s.freq)
    return steps, out_rows


# ---------------------------------------------------------------------------
# card alignment, corroborated by the Gamry clock
# ---------------------------------------------------------------------------

def corroborate_card_lags(per_card: dict, lags: dict, fs: dict, tl: Timeline,
                          f_lo: float, f_hi: float, tol_s: float = 0.5,
                          fs_steps: float | None = None) -> list[dict]:
    """Check every card's own clock against the Gamry's.

    `per_card`: card -> its detected steps, ON THE COMMON BASE where its lag
    was applied and on its own base where it was refused. Each card that
    syncs gets an offset to the Gamry clock; on the common base these must
    agree. A refused card whose offset differs from the aligned cards' by
    its MEASURED (refused) lag, to within `tol_s`, has that lag corroborated
    by an independent instrument. The lag applied is still bronze's
    cross-correlation value, sample-precise; the Gamry only vouches for it.
    Returns one row per card; the caller decides whether to apply.

    `fs`: each card's own rate, for its lag in samples; `fs_steps`: the rate
    of the indices in `per_card` (the common base's), when cards differ.
    """
    rows, offs = [], {}
    for c, steps in per_card.items():
        r = align(steps, tl, fs_steps or fs.get(c, 25000.0), f_lo, f_hi)
        offs[c] = r
        rows.append({"card": c, "sync_ok": r.ok,
                     "offset_s": round(r.offset_s, 3)
                     if np.isfinite(r.offset_s) else None,
                     "n_voters": r.n_voters,
                     "lag_applied": bool((lags.get(c) or {}).get("applied")),
                     "measured_lag_s": None, "corroborated": False})
    ref = [r.offset_s for c, r in offs.items()
           if r.ok and (lags.get(c) or {}).get("applied")]
    if not ref:
        return rows
    off_ref = float(np.median(ref))
    for row in rows:
        c = row["card"]
        info = lags.get(c) or {}
        if info.get("applied") or not offs[c].ok:
            continue
        lag_s = float(info.get("lag", 0)) / fs.get(c, 25000.0)
        row["measured_lag_s"] = round(lag_s, 4)
        row["gamry_implied_lag_s"] = round(offs[c].offset_s - off_ref, 3)
        row["corroborated"] = bool(
            abs((offs[c].offset_s - off_ref) - lag_s) <= tol_s)
    return rows
