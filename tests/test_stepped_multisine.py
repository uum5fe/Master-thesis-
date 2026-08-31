"""The estimator that replaced Welch as the default.

Welch averages over the whole record.  For an excitation that STEPS, each
tone is present for its own slice and absent for the rest, so both the
estimate and the coherence that gates it are diluted by the fraction of the
record the tone was off for -- worst for the shortest dwells, which on a
constant-cycles sweep are the high-frequency ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from eis.spectra import (detect_dwells, impedance_stepped_multisine,
                         impedance_welch, tones_in_window)


FS = 20_000.0


def stepped_pair(groups, durations, Z_of, fs=FS, noise=0.002, seed=0):
    """Current and voltage records of a stepped multisine through Z(f)."""
    rng = np.random.default_rng(seed)
    cur, vol = [], []
    lead = int(0.05 * fs)
    cur.append(np.zeros(lead))
    vol.append(np.zeros(lead))
    for g, T in zip(groups, durations):
        n = int(T * fs)
        t = np.arange(n) / fs
        i_seg = np.zeros(n)
        u_seg = np.zeros(n)
        for f, ph in zip(g, rng.uniform(0, 2 * np.pi, len(g))):
            z = Z_of(f)
            i_seg += np.cos(2 * np.pi * f * t + ph)
            u_seg += abs(z) * np.cos(2 * np.pi * f * t + ph + np.angle(z))
        cur.append(i_seg)
        vol.append(u_seg)
        cur.append(np.zeros(int(0.02 * fs)))
        vol.append(np.zeros(int(0.02 * fs)))
    i = np.concatenate(cur)
    u = np.concatenate(vol)
    return (i + noise * rng.standard_normal(len(i)),
            u + noise * rng.standard_normal(len(u)))


def randles(f):
    """A plain R-(RC) so the test has an exact expected answer."""
    w = 2 * np.pi * f
    return 0.05 + 0.20 / (1 + 1j * w * 0.01)


GROUPS = [np.array([1, 2, 3, 5, 8], float) * 5.0,
          np.array([1, 2, 3, 5, 8], float) * 50.0,
          np.array([1, 2, 3, 5, 8], float) * 400.0]
DURATIONS = [2.0, 0.6, 0.3]


def test_the_impedance_is_recovered_at_every_tone() -> None:
    i, u = stepped_pair(GROUPS, DURATIONS, randles)
    out = impedance_stepped_multisine(i, u, FS, f_min=1.0, f_max=4000.0)
    assert out.method == "stepped_multisine"
    truth = np.array([randles(f) for f in out.f])
    err = np.abs(out.Z - truth) / np.abs(truth)
    assert np.median(err) < 0.02, f"median error {np.median(err):.3%}"


def test_it_beats_welch_on_the_shortest_dwells() -> None:
    """The high-frequency group is on for 0.3 s of a 3 s record.

    Welch averages it with the 90 % of the record in which it is absent.
    """
    i, u = stepped_pair(GROUPS, DURATIONS, randles)
    ms = impedance_stepped_multisine(i, u, FS, f_min=1.0, f_max=4000.0)
    we = impedance_welch(i, u, FS, nperseg=4096, f_min=1.0, f_max=4000.0)

    def err_above(res, f_lo):
        m = res.f >= f_lo
        if m.sum() == 0:
            return np.inf
        truth = np.array([randles(f) for f in res.f[m]])
        return float(np.median(np.abs(res.Z[m] - truth) / np.abs(truth)))

    assert err_above(ms, 300.0) < err_above(we, 300.0), (
        "the joint per-dwell fit must beat Welch where the dwell is short")


def test_welch_reports_a_cloud_of_bins_where_five_tones_were_applied() -> None:
    """Counting surviving points is not the measure; being right is.

    Welch returns a value at every DFT bin in the band, and after a
    coherence gate most of them survive -- not because they are
    measurements, but because the leakage from the strong tones is common to
    both channels and so is coherent. Above 300 Hz, where exactly five tones
    were applied for 0.3 s of a 3 s record, it reports hundreds of points at
    a median error of about 17 %. The joint per-dwell fit reports the five
    that exist, correctly.

    A cloud of coherent, wrong points is worse than none: it is what a
    spectrum looks like when it "goes noisy at high frequency", and it
    invites fixing the cell rather than the estimator.
    """
    i, u = stepped_pair(GROUPS, DURATIONS, randles)
    ms = impedance_stepped_multisine(i, u, FS, f_min=1.0, f_max=4000.0)
    we = impedance_welch(i, u, FS, nperseg=4096, f_min=1.0, f_max=4000.0)

    def above(res, f_lo=300.0):
        g = res.gate(0.7)
        m = g.f >= f_lo
        if not m.any():
            return 0, np.inf
        truth = np.array([randles(f) for f in g.f[m]])
        return int(m.sum()), float(np.median(np.abs(g.Z[m] - truth)
                                             / np.abs(truth)))

    n_ms, err_ms = above(ms)
    n_we, err_we = above(we)

    assert n_ms == len(GROUPS[-1]), (
        f"the high-frequency group holds {len(GROUPS[-1])} tones; the joint "
        f"fit reported {n_ms}")
    assert err_ms < 0.02, f"joint fit median error {err_ms:.1%}"
    assert err_we > 5 * err_ms, (
        f"Welch median error {err_we:.1%} over {n_we} surviving bins, "
        f"against {err_ms:.1%} over {n_ms} real tones")


def test_the_dwells_are_found_without_a_tone_list() -> None:
    i, u = stepped_pair(GROUPS, DURATIONS, randles)
    assert len(detect_dwells(u, FS)) == len(GROUPS)


def test_the_tones_are_found_without_being_configured() -> None:
    """The schedule is a property of the recording, not of a config file."""
    i, u = stepped_pair(GROUPS, DURATIONS, randles)
    a, b = detect_dwells(u, FS)[-1]
    found = tones_in_window(u[a:b], FS, 1.0, 8000.0)
    for f in GROUPS[-1]:
        assert np.any(np.abs(found / f - 1) < 0.02), f"missed {f:.0f} Hz"


def test_a_configured_tone_list_is_still_honoured() -> None:
    i, u = stepped_pair(GROUPS, DURATIONS, randles)
    wanted = [10.0, 25.0, 400.0]
    out = impedance_stepped_multisine(i, u, FS, tones_hz=wanted,
                                      f_min=1.0, f_max=4000.0)
    assert set(np.round(np.unique(out.f), 3)) <= set(wanted)


def test_auto_no_longer_means_welch() -> None:
    """The default that quietly diluted every high-frequency point."""
    from eis.config import SpectralConfig
    cfg = SpectralConfig()
    assert cfg.method == "auto"
    assert cfg.base_frequency_hz is None and cfg.excitation_tones_hz is None
    use_synchronous = cfg.method == "synchronous" or (
        cfg.method == "auto" and cfg.base_frequency_hz
        and cfg.excitation_tones_hz)
    use_multisine = (not use_synchronous
                     and cfg.method in ("auto", "stepped_multisine"))
    assert use_multisine, "auto must resolve to the stepped-multisine path"
