"""Finding the tones at the top of the band.

Two things stopped this detector somewhere around a kilohertz, and neither
looked like what it was.  Both are cheap to pin, because a synthetic sweep
has a known answer.

A sweep generator dwells for a fixed number of CYCLES, so the dwell is n/f
seconds and shrinks as the frequency rises.  That is what makes the top of
the band a different regime from the bottom, and it is the property both of
these bugs keyed on.
"""

from __future__ import annotations

import numpy as np
import pytest

import eis_local as E


FS = 100_000.0
N_CYC = 20


def sweep(freqs, fs=FS, n_cyc=N_CYC, noise=0.05, gap_s=0.005, seed=0):
    """A stepped sine with a fixed number of cycles per step."""
    rng = np.random.default_rng(seed)
    chunks = [np.zeros(int(0.05 * fs))]
    for f in freqs:
        n = int(round(n_cyc / f * fs))
        t = np.arange(n) / fs
        chunks.append(np.sin(2 * np.pi * f * t))
        chunks.append(np.zeros(int(gap_s * fs)))
    x = np.concatenate(chunks)
    return x + noise * rng.standard_normal(len(x))


def found(steps, f, tol=0.03) -> bool:
    if not steps:
        return False
    got = np.array([s.freq for s in steps])
    return bool(np.any(np.abs(got / f - 1.0) < tol))


# ---------------------------------------------------------------------------
# the whole point
# ---------------------------------------------------------------------------

def test_every_tone_up_to_4_khz_is_found_at_100_khz() -> None:
    """The requirement: a 100 kHz recording must yield tones up to 4 kHz.

    Before the window floor and the spectral seed were fixed this scored
    11/16, and the five it missed were 915, 1170, 1913, 2446 and 4000 Hz --
    i.e. everything above a kilohertz, which is exactly the symptom that
    gets reported as "the high-frequency excitation is too weak to detect".
    """
    freqs = np.geomspace(100.0, 4000.0, 16)
    steps = E.detect_schedule(sweep(freqs), FS, ppd=12, f_lo=50.0,
                              f_hi=45_000.0, min_snr_db=-30.0, verbose=False)
    missed = [f for f in freqs if not found(steps, f)]
    assert not missed, (
        f"missed {len(missed)} of {len(freqs)} tones: "
        + ", ".join(f"{f:.0f} Hz" for f in missed))


def test_the_top_of_the_band_survives_real_noise() -> None:
    """Not a fair-weather result: the tones must survive a noisy reference."""
    freqs = np.geomspace(100.0, 4000.0, 16)
    steps = E.detect_schedule(sweep(freqs, noise=0.5), FS, ppd=12, f_lo=50.0,
                              f_hi=45_000.0, min_snr_db=-30.0, verbose=False)
    hits = sum(found(steps, f) for f in freqs)
    assert hits >= 15, f"only {hits}/16 tones survived at noise 0.5"


# ---------------------------------------------------------------------------
# bug 1: an analysis window wider than the dwell it measures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f", [1000.0, 2000.0, 4000.0])
def test_the_demodulation_window_never_outlasts_a_short_dwell(f: float) -> None:
    """The window is n_cyc periods, so it scales with the dwell.

    It used to be floored at 40 ms, which is longer than a twenty-cycle
    dwell above 500 Hz.  Past that the window averages the tone together
    with the silence either side of it and the amplitude falls off as
    dwell/window -- about -18 dB at 4 kHz.  Turning the SNR gate down does
    not recover it, because the dilution scales peak and background alike.
    """
    x = sweep(np.array([f]))
    _env, win = E.demod_envelope(x, FS, f)
    dwell_samples = N_CYC / f * FS
    assert win <= dwell_samples, (
        f"at {f:.0f} Hz the window is {win / FS * 1e3:.1f} ms against a "
        f"{dwell_samples / FS * 1e3:.1f} ms dwell")


def test_the_window_still_has_enough_samples_to_average() -> None:
    """Removing the time floor must not leave a window of four samples.

    The floor was protecting something real; it just said it in the wrong
    units.  The guarantee is a sample count, which is what it was for.
    """
    x = sweep(np.array([4000.0]), fs=10_000.0)
    _env, win = E.demod_envelope(x, 10_000.0, 4000.0)
    assert win >= 8


# ---------------------------------------------------------------------------
# bug 2: a Gauss-Newton started outside its own basin
# ---------------------------------------------------------------------------

def test_the_sine_fit_is_seeded_inside_its_basin() -> None:
    """fit4 converges to the NEAREST optimum, not the best one.

    Its basin is roughly the DFT main lobe, +/- 1/(2N) for N cycles, so
    +/- 2.6 % on a twenty-cycle dwell -- while the 12-points-per-decade grid
    is 21 % apart.  On an ISOLATED tone that is survivable, because there is
    nothing else for the iteration to settle on.  In a real sweep there is:
    the dwell finder leaves a slice of the neighbouring steps in the window,
    and started 6 % away fit4 settles on that instead.  It does not fail
    loudly -- it reports a confident frequency, wrong by several percent,
    with an SNR around -14 dB, which reads downstream as a weak tone rather
    than as a fitting failure.  So the sweep, not a lone sine, is the
    setting this has to be tested in.
    """
    freqs = np.geomspace(100.0, 4000.0, 16)
    x = sweep(freqs, noise=0.02)
    true_f, grid_point = 1169.6070952851458, 1241.7

    env, win = E.demod_envelope(x, FS, grid_point)
    k = int(np.argmax(env)) + win // 2
    a, b = E.dwell_window(x, FS, grid_point, k)

    from_grid, _A, _r, snr_grid = E.fit4(x[a:b], FS, grid_point)
    seed = E.dominant_frequency(x[a:b], FS, grid_point / 1.3, grid_point * 1.3)
    from_seed, _A2, _r2, snr_seed = E.fit4(x[a:b], FS, seed)

    assert abs(from_grid / true_f - 1) > 0.03, (
        "this test is only meaningful if the raw grid start really does "
        f"miss; it returned {from_grid:.1f} for {true_f:.1f}")
    assert snr_grid < 0, (
        "the failure mode is a confident-looking fit with a terrible "
        f"residual; got {snr_grid:.1f} dB")
    assert abs(from_seed / true_f - 1) < 0.005, (
        f"spectral seed gave {from_seed:.1f}, want {true_f:.1f}")
    assert snr_seed > 20, f"seeded fit should be clean; got {snr_seed:.1f} dB"


def test_dominant_frequency_ignores_a_neighbouring_step() -> None:
    """The bracket holds a leftover slice of the next dwell; ignore it."""
    fs = 100_000.0
    n = int(0.02 * fs)
    t = np.arange(n) / fs
    wanted, neighbour = 1200.0, 1500.0
    y = np.sin(2 * np.pi * wanted * t)
    y[int(0.85 * n):] += 0.8 * np.sin(2 * np.pi * neighbour * t[int(0.85 * n):])
    assert E.dominant_frequency(y, fs, 900.0, 1600.0) == pytest.approx(
        wanted, rel=0.01)
