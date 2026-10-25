"""Snap a recovered excitation schedule onto its own geometric ladder.

WHY
---
A stepped sweep is configured in an INTEGER number of points per decade, so the
applied frequencies lie exactly on

        f_k = 10 ** ((k + offset) / ppd)

That structure is known independently of any single detection, which makes it
the right thing to enforce.  It matters because the sine fit is evaluated at the
*reported* frequency: a 1 % frequency error held over a 0.25 s dwell at 600 Hz
is 1.7 cycles of accumulated phase slip, and the correlation of a 1.7-cycle
mismatch is ~16 dB down.  That is exactly why 590.29 Hz (1.12 % below the true
rung at 596.99 Hz) fell under the SNR gate on cards 1, 4 and 5 while the same
physical step passed on cards 2 and 3.

HOW THE PHASE IS RECOVERED
--------------------------
Not by averaging every detection.  Two populations corrupt a plain mean:

  * low-frequency steps, dwelled for only 4-6 cycles, whose frequency estimate
    is good to ~1 % at best (the fit sees a fraction of a cycle at each window
    edge, so the Cramer-Rao bound is optimistic by orders of magnitude);
  * high-frequency detections whose dwell window was mis-cut and spans two
    rungs -- 1516.54 Hz at 45 A sits in a 0.416 s window where every clean HF
    step sits in 0.243 s, and it lands 1.1 % off-rung while carrying the
    largest formal weight of any step in the file.

Both are rejected by one physical criterion: keep only steps from the
FIXED-DWELL region (many cycles) whose window duration matches the median
duration of that region.  Those are the steps the analyser actually resolved
cleanly, and their phases agree to ~0.005 rungs (0.1 % in frequency).

Verified on RO2612030: 45 A 71 -> 45 steps, 450 A 77 -> 45 steps, both
recovering ppd = 10 and placing rung 27 at 597.5 Hz (true rung 596.99 Hz,
0.09 % error) from two independent files.
"""
import numpy as np

__all__ = ["fit_ladder", "ladder_phase", "snap", "snap_schedule", "snap_steps",
           "repair_windows"]


def fit_ladder(freq, ppd_candidates=(5, 10, 12, 20, 25)):
    """Points per decade from the MEDIAN log-spacing of the detections.

    The median is essential: a mean or a least-squares fit is dragged to an
    absurdly fine spacing by a single duplicate pair sitting 1 % apart.
    """
    f = np.asarray(sorted(float(x) for x in freq), float)
    f = f[np.isfinite(f) & (f > 0)]
    if len(f) < 4:
        return None
    d = np.diff(np.log10(f))
    d = d[d > 1e-6]
    if not len(d):
        return None
    ppd_raw = 1.0 / np.median(d)
    return int(min(ppd_candidates, key=lambda c: abs(c - ppd_raw)))


def ladder_phase(freq, n_samples, amp, ppd, fs=25000.0,
                 min_cycles=25.0, dur_tol=0.20, outlier_rungs=0.08):
    """Ladder offset in [0,1), from the cleanly resolved fixed-dwell steps.

    Returns (offset, anchor_mask).  Falls back to an amplitude-weighted
    circular mean over everything if too few anchors survive.
    """
    f = np.asarray(freq, float)
    n = np.asarray(n_samples, float)
    a = np.nan_to_num(np.asarray(amp, float), nan=0.0)
    x = ppd * np.log10(f)
    cyc = n / float(fs) * f

    def circ(mask, w):
        ang = 2 * np.pi * x[mask]
        return float(np.mod(np.angle(np.sum(w[mask] * np.exp(1j * ang)))
                            / (2 * np.pi), 1.0))

    strong = cyc >= min_cycles
    if strong.sum() >= 4:
        strong &= a >= np.median(a[strong])          # drop the noise-floor hits
    if strong.sum() < 4:
        strong = cyc >= min_cycles
    if strong.sum() < 4:
        return circ(np.isfinite(f), np.where(np.isfinite(f), a, 0.0)), strong

    med_dwell = np.median(n[strong])
    keep = strong & (np.abs(n / med_dwell - 1.0) <= dur_tol)
    if keep.sum() < 3:
        keep = strong

    off = circ(keep, a)
    for _ in range(3):                                # one robust re-weighting
        d = np.abs(np.mod(x - off + 0.5, 1.0) - 0.5)
        inl = keep & (d <= outlier_rungs)
        if inl.sum() < 3:
            break
        off = circ(inl, a)
        keep = inl
    return off, keep


def snap(freq, ppd, offset):
    """Nearest rung frequency, distance to it in rung units, and rung index."""
    x = ppd * np.log10(np.asarray(freq, float))
    k = np.round(x - offset)
    return 10.0 ** ((k + offset) / ppd), np.abs(x - offset - k), k.astype(int)


def snap_schedule(rows, ppd=None, offset=None, fs=25000.0, amp_key="amp_V"):
    """Collapse a detected schedule onto its ladder.

    rows : list of dicts with freq_hz, n_samples, amp_V (start/stop preserved).

    Detections landing on the same rung are MERGED -- the one with the largest
    amplitude keeps its dwell window, and its frequency is replaced by the exact
    rung.  Nothing is deleted: a lone spurious detection on an otherwise empty
    rung survives and is left to the existing quality gates to reject.

    Returns (rows_out, info).
    """
    rows = list(rows)
    if len(rows) < 4:
        return rows, {"ok": False, "reason": "too few steps"}
    f = np.array([float(r["freq_hz"]) for r in rows], float)
    n = np.array([float(r.get("n_samples") or 0) for r in rows], float)
    a = np.nan_to_num(np.array([float(r.get(amp_key) or 0.0) for r in rows], float))

    if ppd is None:
        # SPACING from the strong half only (immune to duplicate pairs)
        strong = a >= np.median(a)
        ppd = fit_ladder(f[strong]) or fit_ladder(f)
    if ppd is None:
        return rows, {"ok": False, "reason": "no ladder"}
    if offset is None:
        offset, anchors = ladder_phase(f, n, a, ppd, fs=fs)
    else:
        anchors = np.zeros(len(f), bool)

    f_snap, dist, k = snap(f, ppd, offset)
    out = []
    for rung in sorted(set(k.tolist())):
        m = np.flatnonzero(k == rung)
        best = m[int(np.argmax(a[m]))]
        row = dict(rows[best])
        # The largest-amplitude detection identifies the rung, but a duplicate
        # detection of the SAME dwell sometimes carries the better window (the
        # analyser cut the loser short).  Adopt the widest window that fully
        # CONTAINS the winner's -- containment guarantees it is the same dwell,
        # not a neighbouring step.
        try:
            s0, e0 = float(rows[best]["start"]), float(rows[best]["stop"])
            for j in m:
                s1, e1 = float(rows[j]["start"]), float(rows[j]["stop"])
                if s1 <= s0 and e1 >= e0 and (e1 - s1) > (e0 - s0):
                    s0, e0 = s1, e1
            if (e0 - s0) > float(row["stop"]) - float(row["start"]):
                row["start"], row["stop"] = s0, e0
                row["n_samples"] = e0 - s0
        except (KeyError, TypeError, ValueError):
            pass
        row["freq_hz_raw"] = float(f[best])
        row["freq_hz"] = float(f_snap[best])
        row["rung"] = int(rung)
        row["n_merged"] = int(len(m))
        row["snap_shift_pct"] = float(100.0 * (f_snap[best] / f[best] - 1.0))
        out.append(row)
    out.sort(key=lambda r: r["freq_hz"])
    for i, r in enumerate(out):
        r["index"] = i
    shift = 100.0 * np.abs(f_snap / f - 1.0)
    info = {"ok": True, "ppd": int(ppd), "offset": float(offset),
            "n_anchor": int(np.sum(anchors)),
            "n_in": len(rows), "n_out": len(out),
            "median_shift_pct": float(np.median(shift[a >= np.median(a)])),
            "max_shift_pct": float(np.max(shift))}
    return out, info


# ===========================================================================
# adapter for bronze.consensus_schedule (works on Step dataclasses)
# ===========================================================================
def snap_steps(steps, fs, ppd=None, offset=None, log=None):
    """Snap a list of `eis_local.Step` onto the excitation ladder.

    Returns (steps_out, info).  The Step type is taken from the input, so this
    does not need to import eis_local.  Windows, amplitudes, SNR, THD and
    stationarity are carried through untouched -- only `freq` changes, and only
    onto the rung the step already belongs to.
    """
    if not steps:
        return list(steps), {"ok": False, "reason": "no steps"}
    cls = type(steps[0])
    rows = [{"freq_hz": float(s.freq), "start": int(s.start), "stop": int(s.stop),
             "n_samples": int(s.stop - s.start), "amp_V": float(s.amp),
             "snr_db": float(s.snr_db), "_i": i} for i, s in enumerate(steps)]
    out, info = snap_schedule(rows, ppd=ppd, offset=offset, fs=fs)
    if not info.get("ok"):
        if log:
            log.warning(f"  ladder snap skipped: {info.get('reason')}")
        return list(steps), info
    new = []
    for r in out:
        src = steps[r["_i"]]
        new.append(_rebuild(cls, src, freq=float(r["freq_hz"]),
                            start=int(r["start"]), stop=int(r["stop"])))
    new.sort(key=lambda s: s.freq)
    if log:
        log.info(f"  ladder snap: {info['n_in']} -> {info['n_out']} steps on a "
                 f"{info['ppd']} points/decade ladder "
                 f"(phase from {info['n_anchor']} clean fixed-dwell steps); "
                 f"median correction {info['median_shift_pct']:.2f} %")
    return new, info


# ===========================================================================
# window sanity: a stepped sweep is monotonic in time
# ===========================================================================
def _rebuild(cls, src, **changes):
    """A copy of `src` with `changes` applied, keeping every other field.

    Rebuilding a Step field by field silently drops any field added later --
    which is how `window_source` would have been reset to "detected" by the
    very function that repairs windows.  Copying what is there and overriding
    only what changed cannot lose a field it does not know about.
    """
    fields = ("freq", "start", "stop", "amp", "snr_db", "thd", "stationarity",
              "window_source")
    kw = {f: getattr(src, f) for f in fields if hasattr(src, f)}
    kw.update(changes)
    try:
        return cls(**kw)
    except TypeError:                     # a Step-alike without the new field
        kw.pop("window_source", None)
        return cls(**kw)


def repair_windows(steps, fs, descending=None, min_dwell_frac=0.40,
                   t_tol_s=0.5, max_repair_frac=0.25, log=None):
    """Replace dwell windows that cannot belong to the sweep.

    A stepped sweep visits its rungs in order, so the window start time is
    MONOTONIC in frequency -- decreasing for a descending sweep, increasing
    for an ascending one.  A step whose window breaks that order, or whose
    dwell is a small fraction of the local dwell, did not come from the sweep:
    the detector latched onto something else and the ladder snap, which
    corrects frequencies and not windows, carried it through.

    Measured on RO2612030 at 450 A: the 1501.05 Hz rung claimed a window at
    t = 246.9 s, some 132 s after both of its neighbours and well outside the
    swept part of the record, with a dwell of 0.036 s against a local median
    of 0.240 s.  Fitting a 1501 Hz sine there returns noise, and all 68
    segments rejected the point -- which is the straight chord between 1192
    and 2379 Hz on the Nyquist.

    The repair is interpolation in log-frequency between the nearest steps
    that DO sit in order, giving the window the sweep would have placed
    there.  It is a prediction, not a measurement: the quality gates still
    decide whether a real tone is found in it.  A step with no trustworthy
    neighbour on both sides is left untouched.

    Returns (steps_out, info).
    """
    import numpy as _np
    if not steps or len(steps) < 4:
        return list(steps), {"ok": False, "reason": "too few steps"}
    cls = type(steps[0])
    st = sorted(steps, key=lambda s: float(s.freq))
    f = _np.array([float(s.freq) for s in st], float)
    t0 = _np.array([float(s.start) for s in st], float) / float(fs)
    n = _np.array([float(s.stop) - float(s.start) for s in st], float)

    if descending is None:                 # decide from the majority trend
        d = _np.diff(t0)
        descending = bool(_np.sum(d < 0) >= _np.sum(d > 0))
    sgn = -1.0 if descending else 1.0
    tol = float(t_tol_s)

    # ---- longest run of steps that are mutually consistent in time --------
    # A single misplaced window must not disqualify its neighbours, so the
    # trusted set is the longest increasing subsequence of sgn*t0 rather
    # than a forward scan.
    key = sgn * t0
    best_len = _np.ones(len(key), int)
    prev = _np.full(len(key), -1, int)
    for i in range(len(key)):
        for j in range(i):
            if key[j] <= key[i] + tol and best_len[j] + 1 > best_len[i]:
                best_len[i], prev[i] = best_len[j] + 1, j
    i = int(_np.argmax(best_len))
    chain = []
    while i >= 0:
        chain.append(i); i = prev[i]
    trusted = _np.zeros(len(f), bool)
    trusted[chain] = True

    # ---- a dwell far below the local median is not a dwell ----------------
    med = _np.median(n[trusted]) if trusted.any() else _np.median(n)
    hi = f >= 100.0                       # fixed-dwell region of the sweep
    if hi.sum() >= 4:
        med_hi = _np.median(n[trusted & hi]) if (trusted & hi).any() else _np.median(n[hi])
        trusted &= ~(hi & (n < min_dwell_frac * med_hi))
    trusted &= n >= min_dwell_frac * med * 0.25

    idx = _np.flatnonzero(trusted)
    if len(idx) < 2:
        return list(steps), {"ok": False, "reason": "no trustworthy windows"}

    # A FEW stray windows are a detector slip and are repairable.  HALF the
    # schedule out of order is not: it means the schedule itself was built
    # from two clocks, which happens when a card's alignment lag was accepted
    # on a correlation that should have refused it.  Rewriting those windows
    # would bury the fault instead of reporting it, so refuse and say why.
    frac_bad = 1.0 - len(idx) / float(len(f))
    if frac_bad > max_repair_frac:
        if log:
            log.warning(
                f"  window sanity: {len(f) - len(idx)} of {len(f)} windows are "
                f"out of sweep order ({100*frac_bad:.0f} %) - REFUSING to repair. "
                f"That is not a detector slip; the schedule is built from more "
                f"than one time base. Check the card alignment: a lag accepted "
                f"on a weak correlation shifts one card's windows relative to "
                f"the rest.")
        return list(steps), {"ok": False, "reason": "schedule spans two time bases",
                             "n_out_of_order": int(len(f) - len(idx)),
                             "frac": round(float(frac_bad), 3)}

    lf, lt = _np.log10(f[idx]), t0[idx]
    ln = n[idx]
    out, repaired = [], []
    for k, s in enumerate(st):
        if trusted[k]:
            out.append(s); continue
        x = _np.log10(f[k])
        if x < lf[0] or x > lf[-1]:       # nothing to interpolate between
            out.append(s); continue
        t_new = float(_np.interp(x, lf, lt))
        n_new = float(_np.interp(x, lf, ln))
        a = int(round(t_new * float(fs)))
        b = a + int(round(n_new))
        out.append(_rebuild(cls, s, start=a, stop=b,
                            window_source="interpolated"))
        repaired.append({"freq_hz": float(s.freq),
                         "t_old_s": round(t0[k], 3), "t_new_s": round(t_new, 3),
                         "dwell_old_s": round(n[k] / float(fs), 4),
                         "dwell_new_s": round(n_new / float(fs), 4)})
    info = {"ok": True, "descending": bool(descending), "n_trusted": int(trusted.sum()),
            "n_repaired": len(repaired), "repaired": repaired}
    if log:
        if repaired:
            log.warning(f"  window sanity: {len(repaired)} of {len(st)} dwell windows "
                        f"could not belong to a "
                        f"{'descending' if descending else 'ascending'} sweep "
                        f"and were re-placed by interpolation")
            for r in repaired:
                log.warning(f"    {r['freq_hz']:9.2f} Hz: t {r['t_old_s']:8.2f} -> "
                            f"{r['t_new_s']:8.2f} s, dwell {r['dwell_old_s']:.3f} -> "
                            f"{r['dwell_new_s']:.3f} s")
        else:
            log.info(f"  window sanity: all {len(st)} dwell windows consistent "
                     f"with a {'descending' if descending else 'ascending'} sweep")
    return out, info