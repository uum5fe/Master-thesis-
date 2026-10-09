"""Low-frequency windows must sit inside their own Gamry step.

On RO2612030 the blind detector let the 0.19-0.3 Hz windows run 4-10 s into
the neighbouring step; each phasor was then fitted over two tones, and outlet
spectra zig-zagged along the arc. These tests pin the repair: confine_windows
puts long steps inside their Gamry interval, separate_windows forbids overlap,
and on a simulated stepped sweep the repaired windows recover the impedance.
"""
from dataclasses import dataclass, replace

import numpy as np
import pytest

import gamry_sync
import tone_estimation as te


@dataclass
class _Step:
    freq: float
    start: int
    stop: int
    window_source: str = "detected"


def _rebuild(st, **kw):
    return replace(st, **kw)


FS = 2500.0
FREQS = [3.0, 2.372, 1.878, 1.512, 1.202, 0.957, 0.756, 0.601, 0.478,
         0.378, 0.300, 0.239, 0.190]
DUR = [2, 2, 3, 3, 5, 6, 7, 8, 10, 11, 13, 22, 26]          # s, like the Gamry


def _sweep():
    """Gamry step ends (s), continuous-phase stepped sine, known Z per step."""
    ends = np.cumsum(DUR) + 10.0
    starts = ends - np.asarray(DUR)
    Z = {f: 0.1 + 0.2 / (1 + 1j * 2 * np.pi * f * 0.5) for f in FREQS}
    t = np.arange(int(5 * FS), int((ends[-1] + 5) * FS)) / FS
    u = np.zeros_like(t)
    i = np.zeros_like(t)
    ph = 0.0
    for f, a, b in zip(FREQS, starts, ends):
        k = (t >= a) & (t < b)
        x = ph + 2 * np.pi * f * (t[k] - a)
        i[k] = np.cos(x)
        u[k] = abs(Z[f]) * np.cos(x + np.angle(Z[f]))
        ph += 2 * np.pi * f * (b - a)
    return t, u, i, starts, ends, Z


def _sync(ends):
    rows = [{"gamry_index": k, "f_gamry_hz": f, "pred_end_s": float(e),
             "verdict": "ok"} for k, (f, e) in enumerate(zip(FREQS, ends))]
    return gamry_sync.SyncResult(True, 0.0, 0.2, len(rows), rows)


def _z(t, u, i, a, b, f):
    k = (t >= a) & (t < b)
    A, _, _ = te.phasors_at(np.vstack([u[k], i[k]]), FS, f)
    return A[0] / A[1]


def test_misplaced_windows_bias_and_confinement_repairs_it():
    t, u, i, starts, ends, Z = _sweep()
    # windows like the blind detector's: shifted ~30 % of a step early
    steps = [_Step(f, int((a - 0.3 * (b - a)) * FS), int((b - 0.3 * (b - a)) * FS))
             for f, a, b in zip(FREQS, starts, ends)]
    err_bad = [abs(_z(t, u, i, s.start / FS, s.stop / FS, s.freq) / Z[s.freq] - 1)
               for s in steps]
    fixed, rows = gamry_sync.confine_windows(steps, _sync(ends), FS, _rebuild)
    assert rows, "long steps must be confined"
    for s in fixed:
        k = FREQS.index(s.freq)
        if s.window_source == "gamry_step":
            assert s.start / FS >= starts[k] and s.stop / FS <= ends[k]
    err_ok = [abs(_z(t, u, i, s.start / FS, s.stop / FS, s.freq) / Z[s.freq] - 1)
              for s in fixed if s.window_source == "gamry_step"]
    assert max(err_bad) > 0.02            # the misplacement really biases
    assert max(err_ok) < 0.005            # and the confined windows do not


def test_short_steps_are_left_alone():
    _t, _u, _i, starts, ends, _Z = _sweep()
    steps = [_Step(f, int(a * FS), int(b * FS)) for f, a, b in
             zip(FREQS, starts, ends)]
    fixed, _ = gamry_sync.confine_windows(steps, _sync(ends), FS, _rebuild,
                                          min_interval_s=3.0)
    for s, d in zip(fixed, DUR):
        if d < 3:
            assert s.window_source == "detected"


def test_separate_windows_removes_overlap():
    steps = [_Step(1.0, 0, int(12 * FS)), _Step(0.8, int(10 * FS), int(25 * FS)),
             _Step(0.6, int(25 * FS), int(40 * FS))]
    out, n = gamry_sync.separate_windows(steps, FS, _rebuild)
    assert n == 1
    assert out[0].stop <= out[1].start
    assert out[1].stop <= out[2].start


def test_separate_windows_keeps_a_window_that_would_get_too_short():
    steps = [_Step(1.0, 0, int(2.2 * FS)), _Step(0.8, int(0.2 * FS), int(20 * FS))]
    out, n = gamry_sync.separate_windows(steps, FS, _rebuild, min_cycles=2.0)
    assert n == 0
